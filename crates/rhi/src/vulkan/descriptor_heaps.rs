//! The layout of the descriptor heaps, and how their slots are allocated
//! (design 0002, "Binding model").
//!
//! Both are pure, so they're tested without a device: [`HeapLayout`] places
//! each slot's descriptor and the driver's reserved range in a heap's
//! buffer, and [`SlotAllocator`] hands out slots from a free list.

use ash::vk;

/// The descriptors the resource heap holds: one per texture view a shader
/// can use.
pub(super) const RESOURCE_HEAP_CAPACITY: u32 = 65_536;

/// The descriptors the sampler heap holds.
pub(super) const SAMPLER_HEAP_CAPACITY: u32 = 2_048;

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
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the heaps are created on the core in the next PR")
    )]
    pub(super) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The offset of `slot`'s descriptor in the heap's buffer.
    ///
    /// # Panics
    ///
    /// Panics if `slot` isn't below the capacity.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the heaps are created on the core in the next PR")
    )]
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
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the heaps are created on the core in the next PR")
    )]
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
    /// Whether each slot below `next_unused` is handed out, so that freeing
    /// a slot twice panics rather than handing it to two objects.
    in_use: Vec<bool>,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the heaps are created on the core in the next PR")
)]
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
                self.next_unused += 1;
                self.in_use.push(false);
                self.next_unused - 1
            }
            None => return None,
        };
        self.in_use[slot as usize] = true;
        Some(slot)
    }

    /// Returns `slot` to the free list, to be handed out before any other.
    ///
    /// # Panics
    ///
    /// Panics if `slot` isn't in use.
    pub(super) fn free(&mut self, slot: u32) {
        let Some(in_use) = self.in_use.get_mut(slot as usize).filter(|in_use| **in_use) else {
            panic!("descriptor heap slot {slot} is freed but isn't in use");
        };
        *in_use = false;
        self.free.push(slot);
    }

    /// The number of slots in use.
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
