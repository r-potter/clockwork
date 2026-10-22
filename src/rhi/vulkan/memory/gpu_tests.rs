//! T1 tests of the memory allocator on a real device, with raw Vulkan
//! buffers (design 0002, R19). The RHI's own buffers arrive in
//! `rhi-resources`.

use std::ptr;

use ash::vk;

use super::{
    DEVICE_BLOCK_SIZE, MemoryAllocation, MemoryAllocator, MemoryRequest, Resource,
    buffer_requirements,
};
use crate::diagnostics::capture::TestSession;
use crate::rhi::MemoryLocation;
use crate::rhi::vulkan::test_support::VulkanTestBackend;

/// A buffer of `size` bytes on `device`, for use through its device address
/// and as a transfer destination.
fn create_buffer(device: &ash::Device, size: vk::DeviceSize) -> vk::Buffer {
    let buffer_create_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `device` is a valid logical device, and `buffer_create_info`
    // is valid for the call.
    unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap()
}

/// Allocates memory for `buffer` in `location`, and binds the buffer to it.
fn allocate_and_bind(
    allocator: &MemoryAllocator,
    device: &ash::Device,
    buffer: vk::Buffer,
    location: MemoryLocation,
) -> MemoryAllocation {
    // SAFETY: `buffer` was just created on `device`.
    let (requirements, dedication) = unsafe { buffer_requirements(device, buffer) };
    let allocation = allocator
        .allocate(&MemoryRequest {
            label: "test buffer",
            resource: Resource::Buffer(buffer),
            requirements,
            dedication,
            location,
        })
        .unwrap();
    // SAFETY: The buffer isn't bound yet, and the allocation was made for
    // its requirements, from memory of a type it allows, at an offset of its
    // alignment.
    unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }.unwrap();
    allocation
}

/// Destroys `buffer`, then frees its memory.
fn destroy(
    allocator: &MemoryAllocator,
    device: &ash::Device,
    buffer: vk::Buffer,
    allocation: MemoryAllocation,
) {
    // SAFETY: The buffer was created on `device`, and no submitted work uses
    // it, since nothing is submitted.
    unsafe { device.destroy_buffer(buffer, None) };
    allocator.free(allocation);
}

/// More buffers than the device allows memory allocations can each have
/// memory, since they share blocks, and freeing them all returns the
/// memory, keeping at most one empty block (design 0002, R19's
/// `many_small_buffers`).
#[test]
fn more_buffers_than_allocations_share_blocks() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let device = backend.device.handle();
    let allocator = MemoryAllocator::new(device, backend.physical_device.memory_properties);
    let limit = backend.physical_device.limits().max_memory_allocation_count as usize;
    // Lavapipe's limit is far beyond what's worth creating, so stop at
    // 10,000, which still shows buffers sharing blocks.
    let count = (limit + 1).min(10_000);

    let buffers: Vec<_> = (0..count)
        .map(|_| {
            let buffer = create_buffer(device, 256);
            let allocation = allocate_and_bind(&allocator, device, buffer, MemoryLocation::Device);
            (buffer, allocation)
        })
        .collect();
    let memory_type = buffers[0].1.memory_type();
    let statistics = allocator.statistics(memory_type);
    assert!(statistics.pool.blocks < 10, "{statistics:?}");
    assert_eq!(statistics.dedicated_count, 0);

    for (buffer, allocation) in buffers {
        destroy(&allocator, device, buffer, allocation);
    }
    let statistics = allocator.statistics(memory_type);
    assert_eq!(statistics.pool.allocated, 0);
    assert!(statistics.pool.blocks <= 1, "{statistics:?}");
}

/// A buffer larger than half a block gets memory of its own, which freeing
/// gives back.
#[test]
fn large_buffers_get_dedicated_memory() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let device = backend.device.handle();
    let allocator = MemoryAllocator::new(device, backend.physical_device.memory_properties);

    // Larger than half of any block, since no block is larger than
    // `DEVICE_BLOCK_SIZE`.
    let buffer = create_buffer(device, DEVICE_BLOCK_SIZE / 2 + 1);
    let allocation = allocate_and_bind(&allocator, device, buffer, MemoryLocation::Device);
    let memory_type = allocation.memory_type();
    assert!(allocation.is_dedicated());
    let statistics = allocator.statistics(memory_type);
    assert_eq!((statistics.dedicated_count, statistics.pool.blocks), (1, 0));

    destroy(&allocator, device, buffer, allocation);
    assert_eq!(allocator.statistics(memory_type).dedicated_count, 0);
}

/// Upload memory is mapped, each allocation's pointer is its block's
/// mapping plus its offset, and the CPU can write and read through it.
#[test]
fn upload_memory_is_mapped() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let device = backend.device.handle();
    let allocator = MemoryAllocator::new(device, backend.physical_device.memory_properties);

    let first_buffer = create_buffer(device, 1_024);
    let first = allocate_and_bind(&allocator, device, first_buffer, MemoryLocation::Upload);
    let second_buffer = create_buffer(device, 1_024);
    let second = allocate_and_bind(&allocator, device, second_buffer, MemoryLocation::Upload);
    let (first_pointer, second_pointer) = (first.mapped().unwrap(), second.mapped().unwrap());
    assert_eq!(first.memory(), second.memory());
    assert!(first.size() >= 1_024);
    assert_eq!(
        second_pointer.as_ptr() as usize - first_pointer.as_ptr() as usize,
        (second.offset() - first.offset()) as usize,
    );

    // SAFETY: The pointer is to the first allocation's 1,024 bytes of mapped,
    // coherent memory, which nothing else accesses.
    let bytes = unsafe {
        ptr::write_bytes(first_pointer.as_ptr(), 0xA5, 1_024);
        std::slice::from_raw_parts(first_pointer.as_ptr(), 1_024)
    };
    assert!(bytes.iter().all(|&byte| byte == 0xA5));

    destroy(&allocator, device, first_buffer, first);
    destroy(&allocator, device, second_buffer, second);
}
