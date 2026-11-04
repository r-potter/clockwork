//! Buffers in the Vulkan backend (design 0002, "Memory" and "Binding
//! model").
//!
//! Each buffer is a `VkBuffer` bound to memory from the
//! [`MemoryAllocator`](super::memory::MemoryAllocator), in the location it
//! was created for, with its device address queried once, when it's
//! created. The [`VulkanDevice`]'s buffer table holds them.
//!
//! Upload and readback buffers are mapped persistently, through their
//! allocations. [`write_buffer`](crate::GpuResources::write_buffer) and
//! [`read_buffer`](crate::GpuResources::read_buffer) copy through the mapping
//! while holding the buffer's own lock, so overlapping copies from two
//! threads are a logic error, never undefined behaviour. Readback memory
//! that isn't coherent is invalidated before it's read.
//!
//! Lock order: the buffer table, then a buffer's mapping lock. The table is
//! never held while allocating or freeing memory, so it's never held with a
//! memory pool's lock.

#[cfg(test)]
mod gpu_tests;

use std::ptr;
use std::sync::{Mutex, PoisonError, RwLockReadGuard, RwLockWriteGuard};

use anyhow::anyhow;
use ash::vk;

use super::VulkanDevice;
use super::error::{error, error_from_chain};
use super::memory::{MemoryAllocation, MemoryRequest, Resource, buffer_memory_requirements};
use crate::handle::HandleTable;
use crate::{Buffer, BufferDesc, BufferUsage, GpuAddress, MemoryLocation, RhiError, RhiResult};

/// A buffer in the buffer table.
#[derive(Debug)]
pub(super) struct BufferEntry {
    /// The buffer.
    buffer: vk::Buffer,
    /// The memory it's bound to.
    allocation: MemoryAllocation,
    /// Its address on the GPU.
    address: GpuAddress,
    /// Its size, as created, in bytes.
    size: u64,
    /// Where its memory is.
    location: MemoryLocation,
    /// Held while copying through the buffer's mapping.
    mapping: Mutex<()>,
}

impl BufferEntry {
    /// The buffer itself, for commands that name it.
    pub(super) fn buffer(&self) -> vk::Buffer {
        self.buffer
    }
}

impl From<BufferUsage> for vk::BufferUsageFlags {
    /// The Vulkan usage flags for a buffer used as `usage`. Every buffer can
    /// be copied to and from, and has a device address.
    fn from(usage: BufferUsage) -> Self {
        let mut flags = Self::SHADER_DEVICE_ADDRESS | Self::TRANSFER_SRC | Self::TRANSFER_DST;
        for (part, part_flags) in [
            (BufferUsage::STORAGE, Self::STORAGE_BUFFER),
            (BufferUsage::INDEX, Self::INDEX_BUFFER),
            (BufferUsage::INDIRECT, Self::INDIRECT_BUFFER),
        ] {
            if usage.contains(part) {
                flags |= part_flags;
            }
        }
        flags
    }
}

impl VulkanDevice {
    /// Creates a buffer for `desc`, binds it to memory in its location, and
    /// adds it to the buffer table.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`]
    /// if device memory runs out, or [`RhiError::Other`]
    /// if creating, allocating or binding fails otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the description is invalid (see [`BufferDesc::validate`]).
    pub(super) fn create_buffer_entry(&self, desc: &BufferDesc<'_>) -> RhiResult<Buffer> {
        desc.validate();
        let device = self.device.handle();
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(vk::BufferUsageFlags::from(desc.usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `buffer_create_info` is valid for the call, its size is
        // non-zero, and `bufferDeviceAddress` is enabled (device selection
        // requires it).
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .map_err(|result| error(result, "create buffer", desc.label))?;
        self.device.set_object_name(buffer, desc.label);

        // SAFETY: `buffer` was just created on `device`.
        let (requirements, dedication) = unsafe { buffer_memory_requirements(device, buffer) };
        let allocation = match self.allocator.allocate(&MemoryRequest {
            label: desc.label,
            resource: Resource::Buffer(buffer),
            requirements,
            dedication,
            usage: desc.location.into(),
        }) {
            Ok(allocation) => allocation,
            Err(err) => {
                // SAFETY: The buffer was just created, and nothing uses it.
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(error_from_chain(err, desc.label));
            }
        };
        // SAFETY: The buffer isn't bound yet, and the allocation was made for
        // its requirements, from memory of a type it allows, at an offset of
        // its alignment.
        let bound =
            unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) };
        if let Err(result) = bound {
            // SAFETY: The buffer was just created, and nothing uses it or
            // its memory.
            unsafe { device.destroy_buffer(buffer, None) };
            self.allocator.free(allocation);
            return Err(error(result, "bind memory for buffer", desc.label));
        }
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // SAFETY: The buffer was created with `SHADER_DEVICE_ADDRESS` usage,
        // and is bound to memory allocated with `DEVICE_ADDRESS`.
        let address = GpuAddress(unsafe { device.get_buffer_device_address(&address_info) });

        let entry = BufferEntry {
            buffer,
            allocation,
            address,
            size: desc.size,
            location: desc.location,
            mapping: Mutex::new(()),
        };
        Ok(Buffer::from_handle(
            self.lock_buffers_for_writing().insert(desc.label, entry),
        ))
    }

    /// Removes `buffer` from the table, then destroys it and frees its
    /// memory.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    pub(super) fn destroy_buffer_entry(&self, buffer: Buffer) {
        let (_, entry) = self.lock_buffers_for_writing().remove(buffer.handle());
        self.wait_for_submissions();
        // SAFETY: Every submission so far has completed. A command buffer
        // translated but not yet submitted may still name the buffer, which
        // deferred destruction, later in `rhi-commands`, rules out.
        unsafe { self.free_buffer(entry) };
    }

    /// The device address of `buffer`.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    pub(super) fn buffer_entry_address(&self, buffer: Buffer) -> GpuAddress {
        self.lock_buffers_for_reading().get(buffer.handle()).address
    }

    /// Copies `data` into the upload buffer `buffer` at `offset`.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale, the buffer isn't in upload memory, or
    /// the range is out of bounds.
    pub(super) fn write_buffer_entry(&self, buffer: Buffer, offset: u64, data: &[u8]) {
        let buffers = self.lock_buffers_for_reading();
        let label = buffers.label(buffer.handle());
        let entry = buffers.get(buffer.handle());
        check_location("write", label, entry.location, MemoryLocation::Upload);
        check_range("write", label, entry.size, offset, data.len());
        let mapped = entry.allocation.mapped().expect("upload memory is mapped");
        let _mapping = entry.mapping.lock().unwrap_or_else(PoisonError::into_inner);
        // Upload memory is always coherent (see `memory_types`), so the
        // write needs no flush.
        // SAFETY: The range is inside the buffer, which is inside its
        // allocation's mapped bytes. The mapping stays valid while the entry
        // is in the table, which the read lock keeps it. The buffer's mapping
        // lock keeps any other copy through it out, and `data` is a Rust
        // slice, so it doesn't overlap mapped memory.
        unsafe {
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                mapped.as_ptr().add(offset as usize),
                data.len(),
            );
        }
    }

    /// Copies from the readback buffer `buffer` at `offset` into `out`,
    /// invalidating the range first if the memory isn't coherent.
    ///
    /// # Errors
    ///
    /// Returns an error if non-coherent memory can't be invalidated.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale, the buffer isn't in readback memory,
    /// or the range is out of bounds.
    pub(super) fn read_buffer_entry(
        &self,
        buffer: Buffer,
        offset: u64,
        out: &mut [u8],
    ) -> RhiResult<()> {
        let buffers = self.lock_buffers_for_reading();
        let label = buffers.label(buffer.handle());
        let entry = buffers.get(buffer.handle());
        check_location("read", label, entry.location, MemoryLocation::Readback);
        check_range("read", label, entry.size, offset, out.len());
        if out.is_empty() {
            return Ok(());
        }
        let allocation = &entry.allocation;
        let mapped = allocation.mapped().expect("readback memory is mapped");
        let _mapping = entry.mapping.lock().unwrap_or_else(PoisonError::into_inner);
        if !allocation.is_coherent() {
            let atom = self.physical_device.limits().non_coherent_atom_size;
            let (start, size) = invalidate_range(
                allocation.offset() + offset,
                out.len() as u64,
                atom,
                allocation.memory_size(),
            );
            let range = vk::MappedMemoryRange::default()
                .memory(allocation.memory())
                .offset(start)
                .size(size);
            // SAFETY: The memory is mapped whole, and the range is inside it,
            // rounded as `nonCoherentAtomSize` requires.
            unsafe {
                self.device
                    .handle()
                    .invalidate_mapped_memory_ranges(std::slice::from_ref(&range))
            }
            // Running out of memory here isn't while creating anything, so
            // it isn't `OutOfDeviceMemory`.
            .map_err(|result| {
                RhiError::Other(anyhow!(
                    "failed to invalidate memory for buffer \"{label}\": {result:?}"
                ))
            })?;
        }
        // SAFETY: As for `write_buffer_entry`, with `out` as the destination.
        unsafe {
            ptr::copy_nonoverlapping(
                mapped.as_ptr().add(offset as usize),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(())
    }

    /// Destroys a buffer taken out of the table, and frees its memory.
    ///
    /// # Safety
    ///
    /// No submitted work may still use the buffer.
    pub(super) unsafe fn free_buffer(&self, entry: BufferEntry) {
        // SAFETY: The buffer was created on this device, and the caller
        // guarantees that nothing uses it. It's destroyed before its memory
        // is freed.
        unsafe { self.device.handle().destroy_buffer(entry.buffer, None) };
        self.allocator.free(entry.allocation);
    }

    /// Locks the buffer table for reading, recovering it if a panic
    /// poisoned it: no table operation panics halfway through, so it's
    /// consistent.
    pub(super) fn lock_buffers_for_reading(&self) -> RwLockReadGuard<'_, HandleTable<BufferEntry>> {
        self.buffers.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the buffer table for writing, recovering it as
    /// [`lock_buffers_for_reading`](Self::lock_buffers_for_reading) does.
    fn lock_buffers_for_writing(&self) -> RwLockWriteGuard<'_, HandleTable<BufferEntry>> {
        self.buffers.write().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Checks that a buffer in `location` can be used for `action`, which
/// needs it to be in `expected` memory.
///
/// # Panics
///
/// Panics if it isn't, naming the buffer.
fn check_location(action: &str, label: &str, location: MemoryLocation, expected: MemoryLocation) {
    assert!(
        location == expected,
        "cannot {action} buffer \"{label}\": it's in {location:?} memory, not {expected:?}",
    );
}

/// Checks that `len` bytes at `offset` are inside a buffer of `size` bytes.
///
/// # Panics
///
/// Panics if they aren't, naming the buffer.
fn check_range(action: &str, label: &str, size: u64, offset: u64, len: usize) {
    let end = offset.checked_add(len as u64);
    assert!(
        end.is_some_and(|end| end <= size),
        "cannot {action} {len} bytes at offset {offset} of buffer \"{label}\", which has {size}",
    );
}

/// The range of memory to invalidate before reading `len` bytes at `offset`
/// in memory of `memory_size` bytes, as an offset and a size: rounded out to
/// multiples of `atom` (`nonCoherentAtomSize`), or to the end of the memory
/// with `VK_WHOLE_SIZE` if that's where rounding up would reach.
fn invalidate_range(offset: u64, len: u64, atom: u64, memory_size: u64) -> (u64, u64) {
    let start = offset / atom * atom;
    let end = (offset + len).div_ceil(atom) * atom;
    if end >= memory_size {
        (start, vk::WHOLE_SIZE)
    } else {
        (start, end - start)
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of buffer usage flags, range and location checks, and
    //! invalidation ranges.

    use super::*;

    /// Every buffer can be copied and has an address, and each usage adds
    /// its own flag.
    #[test]
    fn usage_adds_to_copies_and_addresses() {
        let base = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        assert_eq!(vk::BufferUsageFlags::from(BufferUsage::COPIES), base);
        assert_eq!(
            vk::BufferUsageFlags::from(BufferUsage::STORAGE | BufferUsage::INDEX),
            base | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        );
        assert_eq!(
            vk::BufferUsageFlags::from(BufferUsage::INDIRECT),
            base | vk::BufferUsageFlags::INDIRECT_BUFFER,
        );
    }

    /// Ranges inside the buffer pass, up to and including its end, and an
    /// empty range at the end.
    #[test]
    fn ranges_inside_the_buffer_pass() {
        check_range("write", "data", 16, 0, 16);
        check_range("write", "data", 16, 12, 4);
        check_range("write", "data", 16, 16, 0);
    }

    /// A range past the end panics, naming the buffer.
    #[test]
    #[should_panic(expected = "cannot read 4 bytes at offset 13 of buffer \"data\", which has 16")]
    fn ranges_past_the_end_panic() {
        check_range("read", "data", 16, 13, 4);
    }

    /// A range whose end overflows panics, rather than wrapping inside the
    /// buffer.
    #[test]
    #[should_panic(expected = "of buffer \"data\", which has 16")]
    fn overflowing_ranges_panic() {
        check_range("write", "data", 16, u64::MAX, 1);
    }

    /// Writing to a buffer outside upload memory panics, naming the buffer
    /// and both locations.
    #[test]
    #[should_panic(
        expected = "cannot write buffer \"vertices\": it's in Device memory, not Upload"
    )]
    fn writes_outside_upload_memory_panic() {
        check_location(
            "write",
            "vertices",
            MemoryLocation::Device,
            MemoryLocation::Upload,
        );
    }

    /// Invalidation rounds out to whole atoms, around the range.
    #[test]
    fn invalidation_rounds_out_to_atoms() {
        assert_eq!(invalidate_range(0, 64, 64, 4_096), (0, 64));
        assert_eq!(invalidate_range(70, 10, 64, 4_096), (64, 64));
        assert_eq!(invalidate_range(60, 10, 64, 4_096), (0, 128));
    }

    /// A range that rounds up to or past the end of the memory invalidates
    /// to its end, since a size that isn't a multiple of the atom is only
    /// allowed there.
    #[test]
    fn invalidation_reaches_the_end_with_whole_size() {
        assert_eq!(
            invalidate_range(4_000, 96, 64, 4_096),
            (3_968, vk::WHOLE_SIZE)
        );
        assert_eq!(
            invalidate_range(1_000, 10, 64, 1_020),
            (960, vk::WHOLE_SIZE)
        );
    }
}
