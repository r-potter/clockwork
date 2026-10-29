//! The descriptor heaps: their buffers, their layout, and how their slots
//! are allocated (design 0002, "Binding model").
//!
//! [`DescriptorHeaps`] holds the core's two heaps, each a buffer in
//! device-local memory the CPU can write where the device has it, mapped
//! for as long as the heap lives. The rest is pure, so it's tested without
//! a device: [`HeapLayout`] places each slot's descriptor and the driver's
//! reserved range in a heap's buffer, and [`SlotAllocator`] hands out slots
//! from a free list.
//!
//! Each heap's slot allocator is behind a `Mutex`. A slot is written by the
//! one thread it was handed to, before its index is returned, so writes to
//! the mapped heap never overlap (design 0002, "Binding model").

#[cfg(test)]
mod gpu_tests;

use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, PoisonError};

use anyhow::{Context, Result};
use ash::vk;

use super::memory::{
    DedicatedAllocationProperties, MemoryAllocation, MemoryAllocator, MemoryRequest, MemoryUse,
    Resource, buffer_memory_requirements,
};

/// The descriptors the resource heap holds: one per texture view a shader
/// can use.
pub(super) const RESOURCE_HEAP_CAPACITY: u32 = 65_536;

/// The descriptors the sampler heap holds.
pub(super) const SAMPLER_HEAP_CAPACITY: u32 = 2_048;

/// The core's two descriptor heaps.
#[derive(Debug)]
pub(super) struct DescriptorHeaps {
    /// Texture views, for shaders to sample and write.
    pub resource: DescriptorHeap,
    /// Samplers.
    pub sampler: DescriptorHeap,
}

impl DescriptorHeaps {
    /// Creates both heaps on `device`, laid out for its descriptor heap
    /// `properties`, with memory from `allocator`.
    ///
    /// # Errors
    ///
    /// Returns an error if either heap's buffer can't be created, or its
    /// memory allocated or bound.
    pub(super) fn new(
        device: &ash::Device,
        allocator: &MemoryAllocator,
        properties: &vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'_>,
    ) -> Result<Self> {
        let mut resource = DescriptorHeap::new(
            device,
            allocator,
            "resource heap",
            HeapLayout::resource(properties),
        )?;
        let sampler = DescriptorHeap::new(
            device,
            allocator,
            "sampler heap",
            HeapLayout::sampler(properties),
        )
        .inspect_err(|_| {
            // SAFETY: The heap was just created on `device`, with memory from
            // `allocator`, and nothing has used it.
            unsafe { resource.destroy(device, allocator) };
        })?;
        Ok(Self { resource, sampler })
    }

    /// Destroys both heaps, and frees their memory.
    ///
    /// # Safety
    ///
    /// The heaps must have been created on `device`, with memory from
    /// `allocator`, and no submitted work may still use them.
    pub(super) unsafe fn destroy(&mut self, device: &ash::Device, allocator: &MemoryAllocator) {
        // SAFETY: The caller's guarantees are the heaps' own.
        unsafe {
            self.resource.destroy(device, allocator);
            self.sampler.destroy(device, allocator);
        }
    }
}

/// A descriptor heap: a buffer that the CPU writes descriptors into, at the
/// offsets its [`HeapLayout`] gives, and that shaders read through its
/// device address.
#[derive(Debug)]
pub(super) struct DescriptorHeap {
    /// The heap's name, in messages and debug tools.
    label: &'static str,
    /// Where each slot's descriptor is in `buffer`.
    layout: HeapLayout,
    /// The buffer that holds the descriptors.
    buffer: vk::Buffer,
    /// The buffer's memory: a dedicated allocation, mapped for as long as
    /// the heap lives. `None` once the heap is destroyed.
    allocation: Option<MemoryAllocation>,
    /// The buffer's device address, which command buffers bind the heap at.
    address: vk::DeviceAddress,
    /// Which slots are handed out.
    slots: Mutex<SlotAllocator>,
}

impl DescriptorHeap {
    /// Creates a heap laid out as `layout` on `device`, and binds it to a
    /// dedicated allocation from `allocator`, of memory chosen for a
    /// descriptor heap.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer can't be created, or its memory
    /// allocated, mapped or bound.
    fn new(
        device: &ash::Device,
        allocator: &MemoryAllocator,
        label: &'static str,
        layout: HeapLayout,
    ) -> Result<Self> {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(layout.size())
            .usage(
                vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `buffer_create_info` is valid for the call, its size is
        // non-zero, and `descriptorHeap` and `bufferDeviceAddress` are
        // enabled (device selection requires them).
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .with_context(|| format!("failed to create the {label}"))?;

        // SAFETY: `buffer` was just created on `device`.
        let (requirements, dedication) = unsafe { buffer_memory_requirements(device, buffer) };
        let allocation = allocator
            .allocate(&MemoryRequest {
                label,
                resource: Resource::Buffer(buffer),
                requirements,
                // Each heap has memory of its own, whatever the driver
                // prefers, so that it takes no more of a small BAR window
                // than its size (design 0002, "Binding model").
                dedication: DedicatedAllocationProperties {
                    required: true,
                    ..dedication
                },
                usage: MemoryUse::DescriptorHeap,
            })
            .inspect_err(|_| {
                // SAFETY: The buffer was just created, and nothing uses it.
                unsafe { device.destroy_buffer(buffer, None) };
            })?;
        // SAFETY: The buffer isn't bound yet, and the allocation was made for
        // its requirements, from memory of a type it allows. A dedicated
        // allocation starts at offset 0.
        let bound =
            unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) };
        if let Err(result) = bound {
            // SAFETY: The buffer was just created, and nothing uses it or its
            // memory.
            unsafe { device.destroy_buffer(buffer, None) };
            allocator.free(allocation);
            return Err(result).with_context(|| format!("failed to bind memory for the {label}"));
        }
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // SAFETY: The buffer was created with `SHADER_DEVICE_ADDRESS` usage,
        // and is bound to memory allocated with `DEVICE_ADDRESS`.
        let address = unsafe { device.get_buffer_device_address(&address_info) };
        Ok(Self {
            label,
            layout,
            buffer,
            allocation: Some(allocation),
            address,
            slots: Mutex::new(SlotAllocator::new(layout.capacity())),
        })
    }

    /// The heap's name, in messages and debug tools.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "debug names name the heaps in a later PR")
    )]
    pub(super) fn label(&self) -> &'static str {
        self.label
    }

    /// Where each slot's descriptor is in the heap.
    #[cfg(test)]
    pub(super) fn layout(&self) -> HeapLayout {
        self.layout
    }

    /// The heap's device address, which command buffers bind it at.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "command buffers bind the heaps in `rhi-commands`")
    )]
    pub(super) fn address(&self) -> vk::DeviceAddress {
        self.address
    }

    /// The heap's memory.
    ///
    /// # Panics
    ///
    /// Panics if the heap has been destroyed.
    pub(super) fn allocation(&self) -> &MemoryAllocation {
        self.allocation
            .as_ref()
            .unwrap_or_else(|| panic!("the {} is destroyed", self.label))
    }

    /// Hands out a free slot, or returns `None` if every slot is in use.
    pub(super) fn allocate_slot(&self) -> Option<u32> {
        self.lock_slots().allocate()
    }

    /// Returns `slot` to the heap, to be handed out again.
    ///
    /// # Panics
    ///
    /// Panics if `slot` isn't in use.
    pub(super) fn free_slot(&self, slot: u32) {
        self.lock_slots().free(slot);
    }

    /// The number of slots in use.
    #[cfg(test)]
    pub(super) fn slots_in_use(&self) -> u32 {
        self.lock_slots().in_use()
    }

    /// The address range of `slot` in the heap's mapped memory, which holds
    /// its descriptor, for writing it with `vkWrite*DescriptorsEXT`.
    ///
    /// The range is a raw pointer, so this reads and writes nothing. The
    /// caller writes through it only while it holds the slot, which no other
    /// thread writes.
    ///
    /// # Panics
    ///
    /// Panics if `slot` isn't below the capacity, or the heap has been
    /// destroyed.
    pub(super) fn slot_address_range(&self, slot: u32) -> vk::HostAddressRangeEXT<'_> {
        let offset = self.layout.offset(slot);
        let mapped = self
            .allocation()
            .mapped()
            .unwrap_or_else(|| panic!("the {} isn't mapped", self.label));
        // SAFETY: The slot is below the capacity, so its descriptor range is
        // inside the heap's buffer, which is inside its dedicated allocation's
        // mapped bytes, starting at offset 0.
        let address = unsafe { mapped.as_ptr().add(offset as usize) };
        vk::HostAddressRangeEXT {
            address: address.cast::<c_void>(),
            size: self.layout.stride() as usize,
            _marker: PhantomData,
        }
    }

    /// Locks the slot allocator, recovering it if a panic poisoned it: no
    /// allocator operation panics halfway through, so it's consistent.
    fn lock_slots(&self) -> MutexGuard<'_, SlotAllocator> {
        self.slots.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Destroys the heap's buffer, and frees its memory. Does nothing if
    /// the heap is already destroyed.
    ///
    /// # Safety
    ///
    /// The heap must have been created on `device`, with memory from
    /// `allocator`, and no submitted work may still use it.
    unsafe fn destroy(&mut self, device: &ash::Device, allocator: &MemoryAllocator) {
        let Some(allocation) = self.allocation.take() else {
            return;
        };
        // SAFETY: The caller guarantees that the buffer was created on
        // `device`, and that no submitted work uses it.
        unsafe { device.destroy_buffer(self.buffer, None) };
        allocator.free(allocation);
    }
}

/// Where a heap's descriptors and reserved range are in its buffer: slot `i`
/// at `i` strides, then the range the driver reserves after the last slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HeapLayout {
    /// The number of slots.
    capacity: u32,
    /// The distance between two slots: the descriptor size, rounded up to
    /// the descriptor alignment.
    stride: vk::DeviceSize,
    /// The size of the driver's reserved range.
    reserved: vk::DeviceSize,
}

impl HeapLayout {
    /// The layout of a heap of `capacity` descriptors of `descriptor_size`
    /// bytes, each at a multiple of `alignment`, with `reserved` bytes for
    /// the driver after them.
    ///
    /// A stride too large to represent saturates, so that [`size`] reports
    /// a heap no device can hold rather than wrapping around to a small one.
    ///
    /// [`size`]: Self::size
    pub(super) fn new(
        capacity: u32,
        descriptor_size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        reserved: vk::DeviceSize,
    ) -> Self {
        let stride = descriptor_size
            .checked_next_multiple_of(alignment.max(1))
            .unwrap_or(vk::DeviceSize::MAX);
        Self {
            capacity,
            stride,
            reserved,
        }
    }

    /// The resource heap's layout on a device with `properties`: one image
    /// descriptor per slot.
    pub(super) fn resource(properties: &vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'_>) -> Self {
        Self::new(
            RESOURCE_HEAP_CAPACITY,
            properties.image_descriptor_size,
            properties.image_descriptor_alignment,
            properties.min_resource_heap_reserved_range,
        )
    }

    /// The sampler heap's layout on a device with `properties`.
    pub(super) fn sampler(properties: &vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'_>) -> Self {
        Self::new(
            SAMPLER_HEAP_CAPACITY,
            properties.sampler_descriptor_size,
            properties.sampler_descriptor_alignment,
            properties.min_sampler_heap_reserved_range,
        )
    }

    /// The number of slots.
    pub(super) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The offset of `slot`'s descriptor in the heap's buffer.
    ///
    /// # Panics
    ///
    /// Panics if `slot` isn't below the capacity.
    pub(super) fn offset(&self, slot: u32) -> vk::DeviceSize {
        assert!(
            slot < self.capacity,
            "slot {slot} is outside a heap of {} slots",
            self.capacity,
        );
        u64::from(slot) * self.stride
    }

    /// The size of a slot's descriptor range: the stride, so that writing
    /// one descriptor never reaches the next slot.
    pub(super) fn stride(&self) -> vk::DeviceSize {
        self.stride
    }

    /// The offset of the driver's reserved range, just after the last slot.
    pub(super) fn reserved_offset(&self) -> vk::DeviceSize {
        u64::from(self.capacity).saturating_mul(self.stride)
    }

    /// The size of the driver's reserved range.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the heaps are created on the core in the next PR")
    )]
    pub(super) fn reserved(&self) -> vk::DeviceSize {
        self.reserved
    }

    /// The size of the heap's buffer: every slot, then the reserved range.
    /// Saturates rather than overflowing.
    pub(super) fn size(&self) -> vk::DeviceSize {
        self.reserved_offset().saturating_add(self.reserved)
    }
}

/// Hands out a heap's slots: a freed slot before one that's never been
/// used, most recently freed first, so that a heap that's churned stays
/// compact.
#[derive(Debug)]
pub(super) struct SlotAllocator {
    /// The number of slots.
    capacity: u32,
    /// The lowest slot that's never been handed out. Every slot from here
    /// to the capacity is free.
    next_unused: u32,
    /// Freed slots below `next_unused`, the most recently freed last.
    free: Vec<u32>,
    /// Whether each slot below `next_unused` is handed out, one bit per
    /// slot, so that freeing a slot twice panics rather than handing it to
    /// two objects. Slot `i` is bit `i % 64` of word `i / 64`.
    in_use: Vec<u64>,
}

impl SlotAllocator {
    /// An allocator for a heap of `capacity` slots, all free.
    pub(super) fn new(capacity: u32) -> Self {
        Self {
            capacity,
            next_unused: 0,
            free: Vec::new(),
            in_use: Vec::new(),
        }
    }

    /// Hands out a free slot, or returns `None` if every slot is in use.
    pub(super) fn allocate(&mut self) -> Option<u32> {
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None if self.next_unused < self.capacity => {
                let slot = self.next_unused;
                self.next_unused += 1;
                if slot.is_multiple_of(u64::BITS) {
                    self.in_use.push(0);
                }
                slot
            }
            None => return None,
        };
        let (word, bit) = Self::bit(slot);
        self.in_use[word] |= bit;
        Some(slot)
    }

    /// Returns `slot` to the free list, to be handed out before any other.
    ///
    /// # Panics
    ///
    /// Panics if `slot` isn't in use.
    pub(super) fn free(&mut self, slot: u32) {
        let (word, bit) = Self::bit(slot);
        let Some(in_use) = self
            .in_use
            .get_mut(word)
            .filter(|in_use| **in_use & bit != 0)
        else {
            panic!("descriptor heap slot {slot} is freed but isn't in use");
        };
        *in_use &= !bit;
        self.free.push(slot);
    }

    /// Where `slot`'s bit is in `in_use`: the word's index, and the bit as
    /// a mask.
    fn bit(slot: u32) -> (usize, u64) {
        ((slot / u64::BITS) as usize, 1 << (slot % u64::BITS))
    }

    /// The number of slots in use.
    #[cfg(test)]
    pub(super) fn in_use(&self) -> u32 {
        self.next_unused - self.free.len() as u32
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of heap layouts and slot allocation.

    use super::*;

    /// Slots are a stride apart, each descriptor rounded up to its
    /// alignment, and the reserved range follows the last one.
    #[test]
    fn slots_are_a_stride_apart() {
        let layout = HeapLayout::new(4, 24, 32, 512);
        assert_eq!(layout.capacity(), 4);
        assert_eq!(layout.stride(), 32);
        assert_eq!(layout.offset(0), 0);
        assert_eq!(layout.offset(3), 96);
        assert_eq!(layout.reserved_offset(), 128);
        assert_eq!(layout.reserved(), 512);
        assert_eq!(layout.size(), 640);
    }

    /// A descriptor that's already aligned, or has no alignment, keeps its
    /// size as the stride.
    #[test]
    fn aligned_descriptors_keep_their_size() {
        assert_eq!(HeapLayout::new(4, 32, 32, 0).size(), 128);
        assert_eq!(HeapLayout::new(4, 32, 0, 0).stride(), 32);
    }

    /// Absurd sizes saturate rather than wrapping around to a small heap.
    #[test]
    fn sizes_saturate() {
        assert_eq!(HeapLayout::new(2, u64::MAX, 1, 0).size(), u64::MAX);
        assert_eq!(HeapLayout::new(1, u64::MAX - 1, 4, 0).size(), u64::MAX);
        assert_eq!(HeapLayout::new(1, 1, 1, u64::MAX).size(), u64::MAX);
    }

    /// A slot past the last panics, naming the slot and the capacity.
    #[test]
    #[should_panic(expected = "slot 4 is outside a heap of 4 slots")]
    fn slots_past_the_capacity_panic() {
        HeapLayout::new(4, 32, 32, 0).offset(4);
    }

    /// Slots that have never been used are handed out in order.
    #[test]
    fn unused_slots_are_handed_out_in_order() {
        let mut slots = SlotAllocator::new(3);
        assert_eq!(slots.allocate(), Some(0));
        assert_eq!(slots.allocate(), Some(1));
        assert_eq!(slots.allocate(), Some(2));
        assert_eq!(slots.in_use(), 3);
    }

    /// Freed slots are reused before unused ones, most recently freed first.
    #[test]
    fn freed_slots_are_reused_most_recent_first() {
        let mut slots = SlotAllocator::new(8);
        for _ in 0..4 {
            slots.allocate();
        }
        slots.free(1);
        slots.free(3);
        assert_eq!(slots.in_use(), 2);
        assert_eq!(slots.allocate(), Some(3));
        assert_eq!(slots.allocate(), Some(1));
        assert_eq!(slots.allocate(), Some(4));
        assert_eq!(slots.in_use(), 5);
    }

    /// A full heap hands out nothing, until a slot is freed.
    #[test]
    fn full_heaps_have_no_slots() {
        let mut slots = SlotAllocator::new(2);
        slots.allocate();
        slots.allocate();
        assert_eq!(slots.allocate(), None);
        slots.free(0);
        assert_eq!(slots.allocate(), Some(0));
        assert_eq!(slots.allocate(), None);
    }

    /// Slots in every word of the in-use bits are tracked apart, including
    /// the first and last bits of a word, and the last slot of a heap whose
    /// capacity isn't a multiple of 64.
    #[test]
    fn slots_are_tracked_across_words() {
        let mut slots = SlotAllocator::new(130);
        for slot in 0..130 {
            assert_eq!(slots.allocate(), Some(slot));
        }
        assert_eq!(slots.allocate(), None);
        for slot in [63, 64, 127, 128, 129] {
            slots.free(slot);
        }
        assert_eq!(slots.in_use(), 125);
        for slot in [129, 128, 127, 64, 63] {
            assert_eq!(slots.allocate(), Some(slot));
        }
        assert_eq!(slots.allocate(), None);
    }

    /// A slot past the last one handed out isn't in use, even when its bit
    /// shares a word with slots that are.
    #[test]
    #[should_panic(expected = "descriptor heap slot 70 is freed but isn't in use")]
    fn freeing_slots_past_the_last_handed_out_panics() {
        let mut slots = SlotAllocator::new(130);
        for _ in 0..65 {
            slots.allocate();
        }
        slots.free(70);
    }

    /// Freeing a slot twice panics, rather than handing it to two objects.
    #[test]
    #[should_panic(expected = "descriptor heap slot 0 is freed but isn't in use")]
    fn freeing_twice_panics() {
        let mut slots = SlotAllocator::new(2);
        slots.allocate();
        slots.free(0);
        slots.free(0);
    }

    /// Freeing a slot that was never handed out panics.
    #[test]
    #[should_panic(expected = "descriptor heap slot 1 is freed but isn't in use")]
    fn freeing_unused_slots_panics() {
        let mut slots = SlotAllocator::new(2);
        slots.allocate();
        slots.free(1);
    }
}
