//! The Vulkan logical device and its queues.

use anyhow::{Context, Result};
use ash::{Instance, khr, vk};

use super::device_selection::QueueFamilies;
use super::physical_device::PhysicalDevice;

/// A logical device, created from the selected physical device, and the
/// queues retrieved from it.
///
/// The device is destroyed when this is dropped, so the owner must drop it
/// after every object created from it, and before the instance.
pub struct LogicalDevice {
    device: ash::Device,
    queue_families: QueueFamilies,
    graphics_queue: vk::Queue,
    /// `None` when headless.
    present_queue: Option<vk::Queue>,
}

impl LogicalDevice {
    /// Creates a logical device on `physical_device`, with one queue from each
    /// of `queue_families`.
    ///
    /// Enables the Vulkan 1.3 `dynamic_rendering` and `synchronization2`
    /// features, which device selection requires. If `queue_families` has a
    /// present family, `VK_KHR_swapchain` is also enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if device creation fails.
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        queue_families: QueueFamilies,
    ) -> Result<Self> {
        // Vulkan forbids requesting the same family twice, so request the
        // present family separately only if it differs from the graphics one.
        let mut families = vec![queue_families.graphics];
        if let Some(present) = queue_families.present
            && present != queue_families.graphics
        {
            families.push(present);
        }
        let priorities = [1.0];
        let queue_infos: Vec<_> = families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities)
            })
            .collect();

        let mut extensions = Vec::new();
        if queue_families.present.is_some() {
            extensions.push(khr::swapchain::NAME.as_ptr());
        }

        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .push_next(&mut vulkan13_features);
        // SAFETY: `physical_device` was enumerated from `instance`, and
        // `create_info` and the structures it points to are valid for the
        // duration of the call. Selection checked that the queue families and
        // extensions are supported.
        let device = unsafe { instance.create_device(physical_device.handle, &create_info, None) }
            .context("failed to create Vulkan logical device")?;

        // SAFETY: One queue was requested from each of these families.
        let graphics_queue = unsafe { device.get_device_queue(queue_families.graphics, 0) };
        let present_queue = queue_families
            .present
            .map(|family| unsafe { device.get_device_queue(family, 0) });

        Ok(Self {
            device,
            queue_families,
            graphics_queue,
            present_queue,
        })
    }

    /// The device's function pointers and handle.
    pub fn handle(&self) -> &ash::Device {
        &self.device
    }

    /// The queue families the device's queues were created from.
    pub fn queue_families(&self) -> QueueFamilies {
        self.queue_families
    }

    /// The queue to submit rendering work to.
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// The queue to present swapchain images on, or `None` when headless.
    /// May be the same queue as [`graphics_queue`](Self::graphics_queue).
    pub fn present_queue(&self) -> Option<vk::Queue> {
        self.present_queue
    }
}

impl Drop for LogicalDevice {
    fn drop(&mut self) {
        // SAFETY: The device was created by `new` and is destroyed exactly
        // once here. Waiting for it to be idle ensures no submitted work still
        // uses it, and the owner guarantees that every object created from it
        // has been destroyed.
        unsafe {
            // Nothing useful can be done if this fails while dropping.
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}
