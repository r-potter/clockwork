//! T1 tests of buffers on a real device (design 0002, R3, R4 and R19).

use std::collections::HashSet;

use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::test_support::VulkanTestBackend;
use crate::{Buffer, BufferDesc, BufferUsage, MemoryLocation, RenderHardwareInterface};

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
    resources.collect_garbage();
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
    resources.collect_garbage();
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

/// A copy of `len` bytes of `buffer`'s mapped memory, read directly.
fn mapped_bytes(backend: &VulkanTestBackend, buffer: Buffer, len: usize) -> Vec<u8> {
    let buffers = backend.core.lock_buffers_for_reading();
    let mapped = buffers.get(buffer.handle()).allocation.mapped().unwrap();
    // SAFETY: The buffer is mapped and at least `len` bytes long, and
    // nothing else accesses it while the table is locked.
    unsafe { std::slice::from_raw_parts(mapped.as_ptr(), len) }.to_vec()
}

/// Writing an upload buffer puts the bytes at the offset in its mapped
/// memory, and leaves the rest alone (R4).
#[test]
fn writes_reach_upload_memory() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let buffer = resources
        .create_buffer(&desc("upload", 16, MemoryLocation::Upload))
        .unwrap();
    resources.write_buffer(buffer, 0, &[0; 16]);
    resources.write_buffer(buffer, 4, &[1, 2, 3, 4]);
    assert_eq!(
        mapped_bytes(&backend, buffer, 16),
        [0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    );
    resources.destroy_buffer(buffer);
}

/// Reading a readback buffer returns what its memory holds, from the
/// offset, after any invalidation it needs (R4). The host writes the
/// memory here, and flushes it if it isn't coherent, standing in for the
/// GPU copies `rhi-commands` adds.
#[test]
fn reads_come_from_readback_memory() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let buffer = resources
        .create_buffer(&desc("readback", 256, MemoryLocation::Readback))
        .unwrap();
    {
        let buffers = backend.core.lock_buffers_for_reading();
        let allocation = &buffers.get(buffer.handle()).allocation;
        let mapped = allocation.mapped().unwrap();
        // SAFETY: The buffer is mapped and 256 bytes long, and nothing else
        // accesses it.
        let bytes = unsafe { std::slice::from_raw_parts_mut(mapped.as_ptr(), 256) };
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = index as u8;
        }
        if !allocation.is_coherent() {
            let range = ash::vk::MappedMemoryRange::default()
                .memory(allocation.memory())
                .size(ash::vk::WHOLE_SIZE);
            // SAFETY: The memory is mapped whole.
            unsafe {
                backend
                    .core
                    .device
                    .handle()
                    .flush_mapped_memory_ranges(&[range])
            }
            .unwrap();
        }
    }
    let mut out = [0; 8];
    resources.read_buffer(buffer, 100, &mut out).unwrap();
    assert_eq!(out, [100, 101, 102, 103, 104, 105, 106, 107]);
    resources.destroy_buffer(buffer);
}

/// Reading a buffer that isn't in readback memory panics, naming it.
#[test]
#[should_panic(expected = "cannot read buffer \"upload\": it's in Upload memory, not Readback")]
fn reads_outside_readback_memory_panic() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let buffer = resources
        .create_buffer(&desc("upload", 16, MemoryLocation::Upload))
        .unwrap();
    let _ = resources.read_buffer(buffer, 0, &mut [0; 4]);
}
