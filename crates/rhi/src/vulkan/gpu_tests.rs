//! T1 tests of the Vulkan backend: that the validation layer's messages
//! reach the test that caused them (R5), that synchronization validation is
//! on (R6), that the device's Vulkan 1.4 and extension features are queried
//! (R12), and that its capabilities describe it (design 0002, R2).
//!
//! Each validation test makes a mistake on purpose, and declares the message it expects
//! by its ID, so the test fails if the message never arrives, which is what
//! would happen if validation were off or its messages were lost.

use ash::vk;
use clockwork_diagnostics::VALIDATION_TARGET;
use clockwork_diagnostics::capture::TestSession;
use tracing::Level;

use super::test_support::{Capability, VulkanTestBackend};
use crate::{QueueKind, RenderHardwareInterface};

/// A buffer and the memory bound to it, destroyed when dropped.
struct TestBuffer<'a> {
    /// The device the buffer was created on.
    device: &'a ash::Device,
    /// The buffer.
    buffer: vk::Buffer,
    /// The memory bound to `buffer`.
    memory: vk::DeviceMemory,
}

impl<'a> TestBuffer<'a> {
    /// Creates a buffer of `size` bytes on `backend`'s device, for use as a
    /// transfer destination, bound to memory of the first type that can
    /// hold it.
    ///
    /// # Panics
    ///
    /// Panics if any Vulkan call fails.
    fn new(backend: &'a VulkanTestBackend, size: vk::DeviceSize) -> Self {
        let device = backend.core.device.handle();
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `device` is a valid logical device, and
        // `buffer_create_info` is valid for the call.
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap();
        // Construct `Self` first, so that the buffer is destroyed if a later
        // step fails.
        let mut test_buffer = Self {
            device,
            buffer,
            memory: vk::DeviceMemory::null(),
        };
        // SAFETY: the buffer was created on `device`.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = requirements.memory_type_bits.trailing_zeros();
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: `device` is a valid logical device, and the memory type is
        // one the buffer can use, so it exists on the device.
        test_buffer.memory = unsafe { device.allocate_memory(&allocate_info, None) }.unwrap();
        // SAFETY: the buffer and memory were created on `device`, the memory
        // is large enough and of a type the buffer allows, and the buffer
        // isn't bound yet.
        unsafe { device.bind_buffer_memory(buffer, test_buffer.memory, 0) }.unwrap();
        test_buffer
    }
}

impl Drop for TestBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: the buffer and memory were created on `device`, and no
        // submitted work uses them, as nothing that uses them is submitted.
        // Null handles are ignored, as Vulkan allows.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// A command pool with one primary command buffer, destroyed when dropped.
struct TestCommands<'a> {
    /// The device the pool was created on.
    device: &'a ash::Device,
    /// The pool, which frees `command_buffer` when destroyed.
    pool: vk::CommandPool,
    /// A command buffer allocated from `pool`.
    command_buffer: vk::CommandBuffer,
}

impl<'a> TestCommands<'a> {
    /// Creates a command pool for `backend`'s graphics queue family, and
    /// allocates one primary command buffer from it.
    ///
    /// # Panics
    ///
    /// Panics if any Vulkan call fails.
    fn new(backend: &'a VulkanTestBackend) -> Self {
        let device = backend.core.device.handle();
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(backend.core.device.queue_families().graphics);
        // SAFETY: `device` is a valid logical device, and `pool_create_info`
        // is valid for the call.
        let pool = unsafe { device.create_command_pool(&pool_create_info, None) }.unwrap();
        let mut commands = Self {
            device,
            pool,
            command_buffer: vk::CommandBuffer::null(),
        };
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was created on `device`, and `allocate_info` is
        // valid for the call.
        commands.command_buffer =
            unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap()[0];
        commands
    }
}

impl Drop for TestCommands<'_> {
    fn drop(&mut self) {
        // SAFETY: the pool was created on `device`, and its command buffer
        // was never submitted. Destroying the pool frees the command buffer.
        unsafe { self.device.destroy_command_pool(self.pool, None) };
    }
}

/// A validation error caused on purpose reaches the test's session, by its
/// VUID, which shows that validation is on and that its messages are
/// attributed to the session whose backend caused them (R5).
#[test]
fn validation_errors_are_counted() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    session.expect(Level::ERROR, VALIDATION_TARGET, |event| {
        event.message_id.as_deref() == Some("VUID-VkBufferCreateInfo-size-00912")
    });

    let device = backend.core.device.handle();
    // A buffer's size must be greater than zero.
    let buffer_create_info = vk::BufferCreateInfo::default()
        .size(0)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: deliberately invalid: `size` is zero, which the validation
    // layer reports. The layer then skips the call rather than passing it to
    // the driver, returning an error, so the driver never sees the invalid
    // parameter.
    let result = unsafe { device.create_buffer(&buffer_create_info, None) };
    if let Ok(buffer) = result {
        // SAFETY: the buffer was created on `device`, and is unused.
        unsafe { device.destroy_buffer(buffer, None) };
    }
}

/// Two transfer writes to the same buffer, with no barrier between them,
/// are reported as a write-after-write hazard, which shows that
/// synchronization validation is on (R6).
#[test]
fn sync_hazards_are_reported() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    session.expect(Level::ERROR, VALIDATION_TARGET, |event| {
        event.message_id.as_deref() == Some("SYNC-HAZARD-WRITE-AFTER-WRITE")
    });

    let buffer = TestBuffer::new(&backend, 256);
    let commands = TestCommands::new(&backend);
    let device = backend.core.device.handle();
    let command_buffer = commands.command_buffer;
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: the command buffer was allocated on `device`, and isn't
    // recording or in use.
    unsafe { device.begin_command_buffer(command_buffer, &begin_info) }.unwrap();
    // SAFETY: the command buffer is recording, and the buffer was created on
    // `device` with transfer-destination usage, bound to memory, and is large
    // enough. The missing barrier between the fills is deliberate: it's the
    // hazard synchronization validation reports, not undefined behaviour,
    // since the command buffer is never submitted.
    unsafe {
        device.cmd_fill_buffer(command_buffer, buffer.buffer, 0, vk::WHOLE_SIZE, 0);
        device.cmd_fill_buffer(command_buffer, buffer.buffer, 0, vk::WHOLE_SIZE, 1);
    }
    // SAFETY: the command buffer is recording.
    unsafe { device.end_command_buffer(command_buffer) }.unwrap();
}

/// The capabilities describe the selected device, and include what device
/// selection requires: timestamps on the graphics queue, and enough push
/// data (design 0002, R2).
#[test]
fn capabilities_describe_the_selected_device() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let capabilities = backend.device().capabilities();
    assert_eq!(
        capabilities.device_name,
        backend.core.physical_device.name()
    );
    assert!(capabilities.timestamps.contains(QueueKind::Graphics));
    assert!(capabilities.limits.max_push_data_size >= 128);
    assert!(capabilities.limits.timestamp_period_ns > 0.0);
}

/// The device has every capability `CLOCKWORK_REQUIRED_CAPABILITIES`
/// requires, as `cargo xtask check --gpu` sets it for the machine.
///
/// Every capability is asked about, so a missing required one fails here,
/// even before any test needs it. On lavapipe, that checks that the
/// expected set `cargo xtask check --lavapipe` uses is right. The others are
/// reported as skipped.
#[test]
fn every_required_capability_is_present() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    for capability in Capability::ALL {
        backend.has(capability);
    }
}

/// The Vulkan 1.4 and extension structures were queried (R12 in design
/// 0001). Device selection requires the ADR 0002 extensions, so every
/// selected device has them.
#[test]
fn reference_device_reports_required_extensions() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);

    let device = &backend.core.physical_device;
    // Every Vulkan 1.4 device supports `maintenance5`, so it's only unset if
    // the 1.4 features weren't queried.
    assert_eq!(device.features.vulkan14.maintenance5, vk::TRUE);
    // Every descriptor heap device has descriptors of some size, so it's only
    // zero if the extension's properties weren't queried.
    assert!(device.properties.descriptor_heap.buffer_descriptor_size > 0);
}
