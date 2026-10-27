//! Buffers in the Vulkan backend (design 0002, "Memory" and "Binding
//! model").
//!
//! Each buffer is a `VkBuffer` bound to memory from the
//! [`MemoryAllocator`](super::memory::MemoryAllocator), in the location it
//! was created for, with its device address queried once, when it's
//! created. The [`VulkanDevice`]'s buffer table holds them.
//!
//! Lock order: the buffer table is never held while allocating or freeing
//! memory, so it's never held with a memory pool's lock.

#[cfg(test)]
mod gpu_tests;

use std::sync::{PoisonError, RwLockReadGuard, RwLockWriteGuard};

use ash::vk;

use super::VulkanDevice;
use super::error::{error, error_from_chain};
use super::memory::{MemoryAllocation, MemoryRequest, Resource, buffer_requirements};
use crate::handle::HandleTable;
use crate::{Buffer, BufferDesc, BufferUsage, GpuAddress, RhiResult};

/// A buffer in the buffer table.
#[derive(Debug)]
pub(super) struct BufferEntry {
    /// The buffer.
    buffer: vk::Buffer,
    /// The memory it's bound to.
    allocation: MemoryAllocation,
    /// Its address on the GPU.
    address: GpuAddress,
}

/// The Vulkan usage flags for a buffer used as `usage`. Every buffer can be
/// copied to and from, and has a device address.
pub(super) fn usage_flags(usage: BufferUsage) -> vk::BufferUsageFlags {
    let mut flags = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST;
    for (part, part_flags) in [
        (BufferUsage::STORAGE, vk::BufferUsageFlags::STORAGE_BUFFER),
        (BufferUsage::INDEX, vk::BufferUsageFlags::INDEX_BUFFER),
        (BufferUsage::INDIRECT, vk::BufferUsageFlags::INDIRECT_BUFFER),
    ] {
        if usage.contains(part) {
            flags |= part_flags;
        }
    }
    flags
}

impl VulkanDevice {
    /// Creates a buffer for `desc`, binds it to memory in its location, and
    /// adds it to the buffer table.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`](crate::RhiError::OutOfDeviceMemory)
    /// if device memory runs out, or [`RhiError::Other`](crate::RhiError::Other)
    /// if creating, allocating or binding fails otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `desc.size` is zero.
    pub(super) fn create_buffer_entry(&self, desc: &BufferDesc<'_>) -> RhiResult<Buffer> {
        assert!(
            desc.size > 0,
            "buffer \"{}\" has a size of zero",
            desc.label
        );
        let device = self.device.handle();
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(usage_flags(desc.usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `buffer_create_info` is valid for the call, its size is
        // non-zero, and `bufferDeviceAddress` is enabled (device selection
        // requires it).
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .map_err(|result| error(result, "create buffer", desc.label))?;

        // SAFETY: `buffer` was just created on `device`.
        let (requirements, dedication) = unsafe { buffer_requirements(device, buffer) };
        let allocation = match self.allocator.allocate(&MemoryRequest {
            label: desc.label,
            resource: Resource::Buffer(buffer),
            requirements,
            dedication,
            location: desc.location,
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
        };
        Ok(Buffer::from_handle(
            self.write_buffers().insert(desc.label, entry),
        ))
    }

    /// Removes `buffer` from the table, then destroys it and frees its
    /// memory.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    pub(super) fn destroy_buffer_entry(&self, buffer: Buffer) {
        let (_, entry) = self.write_buffers().remove(buffer.handle());
        // SAFETY: Nothing can submit work through the `Device` yet, so no
        // submitted work uses the buffer. Once it can, deferred destruction
        // (`rhi-commands`) calls this only once nothing can.
        unsafe { self.free_buffer(entry) };
    }

    /// The device address of `buffer`.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    pub(super) fn buffer_entry_address(&self, buffer: Buffer) -> GpuAddress {
        self.read_buffers().get(buffer.handle()).address
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
    pub(super) fn read_buffers(&self) -> RwLockReadGuard<'_, HandleTable<BufferEntry>> {
        self.buffers.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the buffer table for writing, recovering it as
    /// [`read_buffers`](Self::read_buffers) does.
    fn write_buffers(&self) -> RwLockWriteGuard<'_, HandleTable<BufferEntry>> {
        self.buffers.write().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of buffer usage flags.

    use super::*;

    /// Every buffer can be copied and has an address, and each usage adds
    /// its own flag.
    #[test]
    fn usage_adds_to_copies_and_addresses() {
        let base = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        assert_eq!(usage_flags(BufferUsage::COPIES), base);
        assert_eq!(
            usage_flags(BufferUsage::STORAGE.union(BufferUsage::INDEX)),
            base | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        );
        assert_eq!(
            usage_flags(BufferUsage::INDIRECT),
            base | vk::BufferUsageFlags::INDIRECT_BUFFER,
        );
    }
}
