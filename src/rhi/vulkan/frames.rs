//! Resources for frames in flight.

use anyhow::{Context, Result};
use ash::vk;

use super::device::LogicalDevice;

/// How many frames the CPU may record ahead of the GPU.
///
/// While the GPU works on one frame, the CPU can record the next. More frames
/// in flight increase latency without much gain in throughput.
const FRAMES_IN_FLIGHT: usize = 2;

/// The resources for each frame in flight, used in turn.
///
/// Destroyed when dropped, so the owner must drop this before the logical
/// device it was created from.
pub struct Frames {
    /// Kept to destroy the resources.
    device: ash::Device,
    /// One entry per frame in flight: [`FRAMES_IN_FLIGHT`] once created.
    frames: Vec<Frame>,
    /// Index into `frames` of the frame to record next.
    current: usize,
}

/// The resources used to record and submit one frame.
pub struct Frame {
    /// Reset at the start of each frame, freeing `command_buffer`'s commands.
    command_pool: vk::CommandPool,
    /// The frame's commands, allocated from `command_pool`.
    pub command_buffer: vk::CommandBuffer,
    /// Signalled when the swapchain image the frame renders to is acquired.
    pub image_acquired: vk::Semaphore,
    /// Signalled when the GPU has finished the frame's commands, so that its
    /// resources can be reused. Created signalled, so that the first wait on
    /// it does not block.
    pub in_flight: vk::Fence,
}

impl Frames {
    /// Creates the resources for [`FRAMES_IN_FLIGHT`] frames, recording
    /// commands for the device's graphics queue.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the resources cannot be created.
    pub fn new(device: &LogicalDevice) -> Result<Self> {
        let graphics_family = device.queue_families().graphics;
        // Construct `Self` first, so that the frames already created are
        // destroyed by `Drop` if a later one fails.
        let mut frames = Self {
            device: device.handle().clone(),
            frames: Vec::with_capacity(FRAMES_IN_FLIGHT),
            current: 0,
        };
        for _ in 0..FRAMES_IN_FLIGHT {
            let frame = Frame::new(&frames.device, graphics_family)?;
            frames.frames.push(frame);
        }
        Ok(frames)
    }

    /// The resources for the frame to record next.
    pub fn current(&self) -> &Frame {
        &self.frames[self.current]
    }

    /// Moves on to the next frame's resources, after submitting the current
    /// frame.
    pub fn advance(&mut self) {
        self.current = (self.current + 1) % self.frames.len();
    }
}

impl Frame {
    /// Creates one frame's resources on `device`. If a later step fails, the
    /// resources already created are destroyed.
    fn new(device: &ash::Device, graphics_family: u32) -> Result<Self> {
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(graphics_family);
        // SAFETY: `device` is a valid logical device, and `pool_create_info` is
        // valid for the duration of the call.
        let command_pool = unsafe { device.create_command_pool(&pool_create_info, None) }
            .context("failed to create Vulkan command pool")?;
        let mut frame = Self {
            command_pool,
            command_buffer: vk::CommandBuffer::null(),
            image_acquired: vk::Semaphore::null(),
            in_flight: vk::Fence::null(),
        };
        if let Err(err) = frame.create_rest(device) {
            // SAFETY: Nothing has used the resources yet.
            unsafe { frame.destroy(device) };
            return Err(err);
        }
        Ok(frame)
    }

    /// Creates the resources after the command pool.
    fn create_rest(&mut self, device: &ash::Device) -> Result<()> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `device` is a valid logical device, the command pool was
        // created on it, and `allocate_info` is valid for the call.
        self.command_buffer = unsafe { device.allocate_command_buffers(&allocate_info) }
            .context("failed to allocate Vulkan command buffer")?[0];
        // SAFETY: `device` is a valid logical device.
        self.image_acquired =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .context("failed to create Vulkan semaphore")?;
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: `device` is a valid logical device, and
        // `fence_create_info` is valid for the call.
        self.in_flight = unsafe { device.create_fence(&fence_create_info, None) }
            .context("failed to create Vulkan fence")?;
        Ok(())
    }

    /// Destroys the frame's resources. Null handles are ignored, as Vulkan
    /// allows.
    ///
    /// # Safety
    ///
    /// The resources must have been created on `device`, and must not be in
    /// use by the GPU.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: Guaranteed by the caller. Destroying the command pool also
        // frees its command buffer.
        unsafe {
            device.destroy_fence(self.in_flight, None);
            device.destroy_semaphore(self.image_acquired, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }

    /// Resets the frame's command pool, so that its command buffer can be
    /// recorded again.
    ///
    /// # Safety
    ///
    /// The GPU must have finished executing the command buffer, which waiting
    /// on `in_flight` guarantees.
    pub unsafe fn reset_commands(&self, device: &ash::Device) -> Result<()> {
        // SAFETY: Guaranteed by the caller.
        unsafe { device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty()) }
            .context("failed to reset Vulkan command pool")
    }
}

impl Drop for Frames {
    fn drop(&mut self) {
        // SAFETY: Every frame was created on `self.device`. Waiting for the
        // device to be idle ensures that none is still in use.
        unsafe {
            // Nothing useful can be done if this fails while dropping.
            let _ = self.device.device_wait_idle();
            for frame in &self.frames {
                frame.destroy(&self.device);
            }
        }
    }
}
