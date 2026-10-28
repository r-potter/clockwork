//! T1 tests of buffers on a real device (design 0002, R3, R4 and R19).

use std::collections::HashSet;

use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::test_support::VulkanTestBackend;
use crate::{
    Buffer, BufferDesc, BufferUsage, GpuResources, MemoryLocation, RenderHardwareInterface,
};

/// A storage buffer of `size` bytes in `location`.
fn desc(label: &str, size: u64, location: MemoryLocation) -> BufferDesc<'_> {
    BufferDesc {
        label,
        size,
        usage: BufferUsage::STORAGE,
        location,
    }
}

/// The memory type `buffer`'s memory is of.
fn memory_type(backend: &VulkanTestBackend, buffer: Buffer) -> u32 {
    backend
        .core
        .lock_buffers_for_reading()
        .get(buffer.handle())
        .allocation
        .memory_type()
}

/// A buffer in each location has a non-zero device address of its own, and
/// destroying them gives their memory back (R4, R5).
#[test]
fn buffers_in_every_location_have_addresses() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();

    let buffers: Vec<_> = [
        MemoryLocation::Device,
        MemoryLocation::Upload,
        MemoryLocation::Readback,
    ]
    .into_iter()
    .map(|location| {
        resources
            .create_buffer(&desc("buffer", 1_024, location))
            .unwrap()
    })
    .collect();
    let addresses: HashSet<_> = buffers
        .iter()
        .map(|&buffer| resources.buffer_address(buffer).0)
        .collect();
    assert_eq!(addresses.len(), buffers.len());
    assert!(!addresses.contains(&0));

    let memory_types: Vec<_> = buffers
        .iter()
        .map(|&buffer| memory_type(&backend, buffer))
        .collect();
    for buffer in buffers {
        resources.destroy_buffer(buffer);
    }
    for memory_type in memory_types {
        let statistics = backend.core.allocator.statistics(memory_type);
        assert_eq!(statistics.pool.allocated, 0, "memory type {memory_type}");
    }
}

/// More buffers than the device allows memory allocations can be created,
/// since they share blocks, and destroying them all leaves no bytes in use
/// and at most one empty block (R19's `many_small_buffers`).
#[test]
fn many_small_buffers() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let limit = backend
        .core
        .physical_device
        .limits()
        .max_memory_allocation_count as usize;
    // Lavapipe's limit is far beyond what's worth creating, so stop at
    // 10,000, which still shows buffers sharing blocks.
    let count = (limit + 1).min(10_000);

    let buffers: Vec<_> = (0..count)
        .map(|_| {
            resources
                .create_buffer(&desc("small", 256, MemoryLocation::Device))
                .unwrap()
        })
        .collect();
    let memory_type = memory_type(&backend, buffers[0]);
    for buffer in buffers {
        resources.destroy_buffer(buffer);
    }
    let statistics = backend.core.allocator.statistics(memory_type);
    assert_eq!(statistics.pool.allocated, 0);
    assert!(statistics.pool.blocks <= 1, "{statistics:?}");
}

/// Using a destroyed buffer's handle panics, naming the buffer (R3's
/// `stale_handle_panics`).
#[test]
#[should_panic(expected = "stale buffer handle: \"vertices\" was destroyed")]
fn stale_handle_panics() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let buffer = resources
        .create_buffer(&desc("vertices", 64, MemoryLocation::Device))
        .unwrap();
    resources.destroy_buffer(buffer);
    resources.buffer_address(buffer);
}

/// A buffer of no bytes is a bug in the caller.
#[test]
#[should_panic(expected = "buffer \"empty\" has a size of zero")]
fn empty_buffers_panic() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let _ = backend
        .resources()
        .create_buffer(&desc("empty", 0, MemoryLocation::Device));
}

/// Buffers left when the device is dropped are destroyed with it, before
/// their memory and the device, so validation reports nothing.
#[test]
fn buffers_left_at_drop_are_destroyed() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources: &dyn GpuResources = backend.resources().as_ref();
    for location in [MemoryLocation::Device, MemoryLocation::Upload] {
        resources
            .create_buffer(&desc("left behind", 1_024, location))
            .unwrap();
    }
    drop(backend);
}
