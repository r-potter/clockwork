//! GPU memory allocation (design 0002, "Memory allocator").
//!
//! Vulkan limits how many memory allocations a device may have at once, so
//! the [`MemoryAllocator`] takes large blocks of memory from Vulkan and
//! suballocates from them. Most of it is pure, and tested without a GPU:
//! - [`memory_types`] chooses the memory type for each location
//! - `clockwork-gpu-memory`'s [`Pool`] holds the blocks of one memory type,
//!   and decides which block an allocation goes in and when a block is given
//!   back, suballocating each block with a TLSF suballocator
//!
//! The allocator itself is the thin shell that makes the Vulkan calls:
//! allocating, mapping and freeing blocks, and dedicated allocations for
//! resources that need or suit them.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "resources are allocated from rhi-resources")
)]

#[cfg(test)]
mod gpu_tests;
mod memory_types;

use std::collections::HashSet;
use std::ptr::NonNull;
use std::sync::{Mutex, PoisonError};

use anyhow::{Context, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use clockwork_gpu_memory::{Pool, PoolAllocation, PoolStatistics, ResourceKind};
use tracing::{debug, warn};

use crate::MemoryLocation;

/// The block size for device-local memory types, in bytes.
const DEVICE_BLOCK_SIZE: u64 = 256 << 20;

/// The block size for host-visible memory types, in bytes, whether or not
/// they're also device-local.
const HOST_BLOCK_SIZE: u64 = 64 << 20;

/// A pointer into persistently mapped memory.
///
/// Raw pointers are neither `Send` nor `Sync`, but the allocator must be
/// both, since design 0002 shares the device between threads. This wrapper
/// is the one `unsafe impl` of either in the RHI (design 0002, "Risks").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MappedPointer(NonNull<u8>);

// SAFETY: The pointer is into memory that stays mapped, and at the same
// address, for as long as the block or dedicated allocation it's in exists,
// whichever thread it's used on. Mapped memory has no thread affinity in
// Vulkan. The pointer is only an address: sending or sharing it reads or
// writes nothing, and whoever writes through it must make sure, as for any
// raw pointer, that no one else accesses the same bytes at the same time,
// as the per-buffer mapping locks of design 0002 ("Memory") will.
unsafe impl Send for MappedPointer {}
// SAFETY: as above.
unsafe impl Sync for MappedPointer {}

impl MappedPointer {
    /// The pointer.
    pub(super) fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr()
    }

    /// This pointer advanced by `offset` bytes.
    ///
    /// # Safety
    ///
    /// The result must stay within the mapping this pointer is into.
    unsafe fn add(self, offset: u64) -> Self {
        let offset = usize::try_from(offset).expect("a mapping's offsets fit in usize");
        // SAFETY: The caller guarantees that the result is within the same
        // mapping, which doesn't wrap around the address space, so it's
        // non-null too.
        Self(unsafe { self.0.add(offset) })
    }
}

/// The memory of one block, or of one dedicated allocation.
#[derive(Debug, Clone, Copy)]
struct MemoryBlock {
    /// The memory.
    gpu_memory: vk::DeviceMemory,
    /// Where the memory is mapped, if its type is host-visible.
    mapped: Option<MappedPointer>,
}

/// The resource memory is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Resource {
    /// A buffer, which goes in buffer blocks.
    Buffer(vk::Buffer),
    /// An image, with optimal tiling, which goes in image blocks.
    Image(vk::Image),
}

impl Resource {
    /// The block list the resource goes in.
    fn kind(self) -> ResourceKind {
        match self {
            Self::Buffer(_) => ResourceKind::Buffer,
            Self::Image(_) => ResourceKind::Image,
        }
    }
}

/// Whether a resource's driver wants it to have memory of its own, from
/// `VkMemoryDedicatedRequirements`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct DedicatedAllocationProperties {
    /// The driver performs better with a dedicated allocation.
    pub preferred: bool,
    /// The resource must have a dedicated allocation.
    pub required: bool,
}

/// What [`MemoryAllocator::allocate`] is asked for.
#[derive(Debug, Clone, Copy)]
pub(super) struct MemoryRequest<'a> {
    /// The resource's name, for messages.
    pub label: &'a str,
    /// The resource the memory is for.
    pub resource: Resource,
    /// The resource's size, alignment and allowed memory types.
    pub requirements: vk::MemoryRequirements,
    /// Whether the driver wants the resource to have memory of its own.
    pub dedication: DedicatedAllocationProperties,
    /// Where the memory should be.
    pub location: MemoryLocation,
}

/// Memory allocated for a resource, to bind it to, and to give back to
/// [`MemoryAllocator::free`].
#[derive(Debug)]
pub(super) struct MemoryAllocation {
    /// The memory the resource is in.
    memory: vk::DeviceMemory,
    /// Where the resource starts in `memory`.
    offset: u64,
    /// The bytes allocated.
    size: u64,
    /// The memory type.
    memory_type: u32,
    /// Where the resource's memory is mapped, if it's host-visible.
    mapped: Option<MappedPointer>,
    /// Where the memory came from.
    source: Source,
}

/// Where a [`MemoryAllocation`] came from.
#[derive(Debug)]
enum Source {
    /// A range of a pool's block.
    Pool(PoolAllocation),
    /// Memory of its own.
    Dedicated,
}

impl MemoryAllocation {
    /// The memory to bind the resource to.
    pub(super) fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    /// The offset to bind the resource at.
    pub(super) fn offset(&self) -> u64 {
        self.offset
    }

    /// The bytes allocated, as the resource's requirements asked.
    pub(super) fn size(&self) -> u64 {
        self.size
    }

    /// The memory type the memory is of.
    pub(super) fn memory_type(&self) -> u32 {
        self.memory_type
    }

    /// Where the resource's memory is mapped, if its type is host-visible:
    /// the first of its [`size`](Self::size) bytes.
    pub(super) fn mapped(&self) -> Option<MappedPointer> {
        self.mapped
    }

    /// Whether the memory is a dedicated allocation, rather than a range of
    /// a block.
    pub(super) fn is_dedicated(&self) -> bool {
        matches!(self.source, Source::Dedicated)
    }
}

/// One memory type's pool, and its dedicated allocations.
struct MemoryTypePool {
    /// The memory type's blocks.
    pool: Pool<MemoryBlock>,
    /// The memory of each live dedicated allocation, so that any left at
    /// drop are freed.
    dedicated: HashSet<vk::DeviceMemory>,
    /// The bytes in live dedicated allocations.
    dedicated_bytes: u64,
}

/// What one memory type's memory holds, for tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct MemoryStatistics {
    /// The memory type's pool of blocks.
    pub pool: PoolStatistics,
    /// The live dedicated allocations.
    pub dedicated_count: usize,
    /// The bytes in them.
    pub dedicated_bytes: u64,
}

/// Allocates device memory for resources: from pools of blocks, one per
/// memory type, or as dedicated allocations. See the module documentation.
///
/// Every method takes `&self`, and may be called from any thread: each
/// memory type's pool has its own lock.
///
/// It holds a copy of the device's function pointers, and frees every block
/// when dropped, so it must be dropped before the device is destroyed.
pub(super) struct MemoryAllocator {
    /// The device memory is allocated on.
    device: ash::Device,
    /// The device's memory heaps and types.
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// The pool for each memory type, by index.
    pools: Vec<Mutex<MemoryTypePool>>,
}

// The allocator is shared between threads (design 0002, "Threading").
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MemoryAllocator>();
    assert_send_sync::<MemoryAllocation>();
};

impl MemoryAllocator {
    /// An allocator for `device`, whose memory heaps and types are
    /// `memory_properties`, holding no memory yet.
    pub(super) fn new(
        device: &ash::Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> Self {
        let pools = memory_properties
            .memory_types_as_slice()
            .iter()
            .map(|memory_type| {
                let heap = memory_properties.memory_heaps[memory_type.heap_index as usize];
                Mutex::new(MemoryTypePool {
                    pool: Pool::new(block_size(memory_type.property_flags, heap.size)),
                    dedicated: HashSet::new(),
                    dedicated_bytes: 0,
                })
            })
            .collect();
        Self {
            device: device.clone(),
            memory_properties,
            pools,
        }
    }

    /// Allocates memory for `request`'s resource, in its location: a range
    /// of a block, or a dedicated allocation if the driver wants one or the
    /// resource is larger than half a block.
    ///
    /// # Errors
    ///
    /// Returns an error if no memory type the resource can use suits the
    /// location, or if allocating or mapping memory fails.
    pub(super) fn allocate(&self, request: &MemoryRequest<'_>) -> Result<MemoryAllocation> {
        let requirements = request.requirements;
        let memory_type = memory_types::choose(
            &self.memory_properties,
            requirements.memory_type_bits,
            request.location,
        )
        .with_context(|| {
            format!(
                "no memory type for \"{}\" suits {:?} memory",
                request.label, request.location
            )
        })?;
        let mut type_pool = self.lock(memory_type);

        let block_size = type_pool.pool.block_size();
        if needs_dedicated(request.dedication, requirements.size, block_size) {
            let memory = self
                .allocate_memory(memory_type, requirements.size, Some(request.resource))
                .with_context(|| {
                    format!(
                        "failed to allocate dedicated memory for \"{}\"",
                        request.label
                    )
                })?;
            type_pool.dedicated.insert(memory.gpu_memory);
            type_pool.dedicated_bytes += requirements.size;
            debug!(
                "allocated {} bytes of dedicated memory of type {memory_type} for \"{}\"",
                requirements.size, request.label,
            );
            return Ok(MemoryAllocation {
                memory: memory.gpu_memory,
                offset: 0,
                size: requirements.size,
                memory_type,
                mapped: memory.mapped,
                source: Source::Dedicated,
            });
        }

        let (block, allocation) = type_pool
            .pool
            .allocate(
                request.resource.kind(),
                requirements.size,
                requirements.alignment,
                |size| {
                    let block = self.allocate_memory(memory_type, size, None)?;
                    debug!("allocated a {size}-byte block of memory type {memory_type}");
                    Ok::<_, anyhow::Error>(block)
                },
            )
            .with_context(|| format!("failed to allocate memory for \"{}\"", request.label))?;
        // SAFETY: The range is within the block, which is mapped whole.
        let mapped = block
            .mapped
            .map(|base| unsafe { base.add(allocation.offset()) });
        Ok(MemoryAllocation {
            memory: block.gpu_memory,
            offset: allocation.offset(),
            size: allocation.size(),
            memory_type,
            mapped,
            source: Source::Pool(allocation),
        })
    }

    /// Gives `allocation` back. A block it leaves empty is freed, unless
    /// it's the last of its kind in its pool.
    ///
    /// The resource bound to it must have been destroyed, or never bound, and
    /// no submitted work may still use it.
    pub(super) fn free(&self, allocation: MemoryAllocation) {
        let mut type_pool = self.lock(allocation.memory_type);
        match allocation.source {
            Source::Pool(range) => {
                if let Some(block) = type_pool.pool.free(range) {
                    // SAFETY: The pool gave the block back, so no allocation
                    // uses it, and it was allocated by this allocator.
                    unsafe { self.device.free_memory(block.gpu_memory, None) };
                    debug!("freed a block of memory type {}", allocation.memory_type);
                }
            }
            Source::Dedicated => {
                type_pool.dedicated.remove(&allocation.memory);
                type_pool.dedicated_bytes -= allocation.size;
                // SAFETY: The memory was allocated by this allocator for this
                // allocation alone, which the caller no longer uses.
                unsafe { self.device.free_memory(allocation.memory, None) };
            }
        }
    }

    /// What `memory_type`'s memory holds.
    pub(super) fn statistics(&self, memory_type: u32) -> MemoryStatistics {
        let type_pool = self.lock(memory_type);
        MemoryStatistics {
            pool: type_pool.pool.statistics(),
            dedicated_count: type_pool.dedicated.len(),
            dedicated_bytes: type_pool.dedicated_bytes,
        }
    }

    /// Locks `memory_type`'s pool, recovering it if a panic poisoned it: a
    /// panic while holding it is already a bug being reported, and the pool
    /// is consistent between calls.
    fn lock(&self, memory_type: u32) -> std::sync::MutexGuard<'_, MemoryTypePool> {
        self.pools[memory_type as usize]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Allocates `size` bytes of `memory_type`, with device addresses, as
    /// memory of its own for `dedicated` if that's given, and maps it whole
    /// if the type is host-visible.
    ///
    /// # Errors
    ///
    /// Returns an error if allocating or mapping the memory fails.
    fn allocate_memory(
        &self,
        memory_type: u32,
        size: u64,
        dedicated: Option<Resource>,
    ) -> Result<MemoryBlock> {
        // Buffers are used through their device addresses (design 0002, R5),
        // so every allocation can hold them.
        let mut flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut dedicated_info = match dedicated {
            Some(Resource::Buffer(buffer)) => {
                Some(vk::MemoryDedicatedAllocateInfo::default().buffer(buffer))
            }
            Some(Resource::Image(image)) => {
                Some(vk::MemoryDedicatedAllocateInfo::default().image(image))
            }
            None => None,
        };
        let mut allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type)
            .push(&mut flags_info);
        if let Some(dedicated_info) = &mut dedicated_info {
            allocate_info = allocate_info.push(dedicated_info);
        }
        // SAFETY: `allocate_info` and its chain are valid for the call, the
        // memory type exists on the device, `bufferDeviceAddress` is enabled
        // (device selection requires it), and a dedicated resource is a live
        // resource on the device, which the caller guarantees.
        let memory =
            unsafe { self.device.allocate_memory(&allocate_info, None) }.with_context(|| {
                format!("failed to allocate {size} bytes of memory type {memory_type}")
            })?;

        let flags = self.memory_properties.memory_types[memory_type as usize].property_flags;
        if !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            return Ok(MemoryBlock {
                gpu_memory: memory,
                mapped: None,
            });
        }
        // SAFETY: The memory was just allocated, of a host-visible type, and
        // isn't mapped yet.
        let mapped = unsafe {
            self.device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .context("failed to map memory")
        .and_then(|pointer| {
            NonNull::new(pointer.cast::<u8>()).context("mapping memory returned a null pointer")
        });
        match mapped {
            Ok(pointer) => Ok(MemoryBlock {
                gpu_memory: memory,
                mapped: Some(MappedPointer(pointer)),
            }),
            Err(err) => {
                // SAFETY: The memory was just allocated, and nothing uses it.
                unsafe { self.device.free_memory(memory, None) };
                Err(err)
            }
        }
    }
}

impl Drop for MemoryAllocator {
    fn drop(&mut self) {
        for (memory_type, type_pool) in self.pools.iter_mut().enumerate() {
            let type_pool = type_pool.get_mut().unwrap_or_else(PoisonError::into_inner);
            let statistics = type_pool.pool.statistics();
            if statistics.allocated > 0 || !type_pool.dedicated.is_empty() {
                // The RHI's handle tables free every resource, and report
                // leaks themselves, so this is a bug in the backend.
                warn!(
                    "memory type {memory_type} still holds {} bytes and {} dedicated allocations",
                    statistics.allocated,
                    type_pool.dedicated.len(),
                );
            }
            for memory in type_pool
                .pool
                .drain()
                .into_iter()
                .map(|block| block.gpu_memory)
                .chain(type_pool.dedicated.drain())
            {
                // SAFETY: The memory was allocated by this allocator, and is
                // freed exactly once here. The owner guarantees that the
                // device is idle, and outlives the allocator.
                unsafe { self.device.free_memory(memory, None) };
            }
        }
    }
}

/// The block size for a memory type with `flags`, in a heap of
/// `heap_size` bytes: 256 MiB for device-local types and 64 MiB for
/// host-visible ones, or an eighth of the heap if that's smaller.
fn block_size(flags: vk::MemoryPropertyFlags, heap_size: u64) -> u64 {
    let preferred = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
        HOST_BLOCK_SIZE
    } else {
        DEVICE_BLOCK_SIZE
    };
    preferred.min(heap_size / 8).max(1)
}

/// Whether a resource of `size` bytes gets memory of its own: when its
/// driver requires or prefers that, or it's larger than half a block.
fn needs_dedicated(dedication: DedicatedAllocationProperties, size: u64, block_size: u64) -> bool {
    dedication.required || dedication.preferred || size > block_size / 2
}

/// The memory requirements of `buffer`, and whether its driver wants it to
/// have memory of its own.
///
/// # Safety
///
/// `buffer` must be a live buffer on `device`.
pub(super) unsafe fn buffer_requirements(
    device: &ash::Device,
    buffer: vk::Buffer,
) -> (vk::MemoryRequirements, DedicatedAllocationProperties) {
    let info = vk::BufferMemoryRequirementsInfo2::default().buffer(buffer);
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push(&mut dedicated);
    // SAFETY: The caller guarantees that `buffer` is live on `device`, and
    // the structures are valid for the call.
    unsafe { device.get_buffer_memory_requirements2(&info, &mut requirements) };
    let memory_requirements = requirements.memory_requirements;
    (
        memory_requirements,
        DedicatedAllocationProperties {
            preferred: dedicated.prefers_dedicated_allocation == vk::TRUE,
            required: dedicated.requires_dedicated_allocation == vk::TRUE,
        },
    )
}

#[cfg(test)]
mod tests {
    //! T0 tests of the allocator's pure rules: block sizes and when memory
    //! is dedicated (design 0002, R19).

    use super::*;

    /// One mebibyte.
    const MIB: u64 = 1 << 20;

    /// Device-local memory gets 256 MiB blocks, and host-visible memory,
    /// device-local or not, 64 MiB blocks.
    #[test]
    fn block_sizes_follow_the_memory_type() {
        let heap = 24 << 30;
        assert_eq!(
            block_size(vk::MemoryPropertyFlags::DEVICE_LOCAL, heap),
            256 * MIB
        );
        assert_eq!(
            block_size(vk::MemoryPropertyFlags::HOST_VISIBLE, heap),
            64 * MIB
        );
        assert_eq!(
            block_size(
                vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                heap
            ),
            64 * MIB
        );
    }

    /// A small heap gets blocks of an eighth of its size, as a 256 MiB BAR
    /// window does, and never blocks of zero bytes.
    #[test]
    fn small_heaps_get_an_eighth_each() {
        let host_visible = vk::MemoryPropertyFlags::HOST_VISIBLE;
        assert_eq!(block_size(host_visible, 256 * MIB), 32 * MIB);
        assert_eq!(block_size(host_visible, 7), 1);
    }

    /// Buffers and images go in their own kinds of block.
    #[test]
    fn resources_go_in_blocks_of_their_kind() {
        assert_eq!(
            Resource::Buffer(vk::Buffer::null()).kind(),
            ResourceKind::Buffer
        );
        assert_eq!(
            Resource::Image(vk::Image::null()).kind(),
            ResourceKind::Image
        );
    }

    /// Memory is dedicated when the driver requires or prefers it, or the
    /// resource is larger than half a block, and only then.
    #[test]
    fn dedicated_memory_follows_the_driver_and_the_size() {
        let none = DedicatedAllocationProperties::default();
        let block = 256 * MIB;
        assert!(!needs_dedicated(none, block / 2, block));
        assert!(needs_dedicated(none, block / 2 + 1, block));
        assert!(needs_dedicated(
            DedicatedAllocationProperties {
                preferred: true,
                required: false
            },
            1,
            block
        ));
        assert!(needs_dedicated(
            DedicatedAllocationProperties {
                preferred: false,
                required: true
            },
            1,
            block
        ));
    }
}
