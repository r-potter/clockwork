//! T1 tests of the descriptor heaps on a real device (design 0002,
//! "Binding model").

use ash::vk;
use clockwork_diagnostics::capture::TestSession;

use super::{DescriptorHeap, HeapLayout};
use crate::vulkan::test_support::VulkanTestBackend;

/// Each heap is a mapped, dedicated allocation, large enough for its
/// layout, with a device address. Its memory is host-coherent, and
/// device-local wherever the device has device-local memory the CPU can
/// write.
#[test]
fn heaps_are_mapped_dedicated_allocations() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let core = &backend.core;
    let memory_properties = &core.physical_device.memory_properties;
    let host_coherent =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let writable_device_memory =
        memory_properties
            .memory_types_as_slice()
            .iter()
            .any(|memory_type| {
                memory_type
                    .property_flags
                    .contains(host_coherent | vk::MemoryPropertyFlags::DEVICE_LOCAL)
            });
    let descriptor_heap = &core.physical_device.properties.descriptor_heap;

    let heaps: [(&DescriptorHeap, HeapLayout); 2] = [
        (&core.heaps.resource, HeapLayout::resource(descriptor_heap)),
        (&core.heaps.sampler, HeapLayout::sampler(descriptor_heap)),
    ];
    for (heap, layout) in heaps {
        let label = heap.label();
        assert_eq!(heap.layout(), layout, "{label}");
        assert_ne!(heap.address(), 0, "{label}");
        let allocation = heap.allocation();
        assert!(allocation.is_dedicated(), "{label}");
        assert!(allocation.mapped().is_some(), "{label}");
        assert!(allocation.size() >= layout.size(), "{label}");
        let flags =
            memory_properties.memory_types[allocation.memory_type() as usize].property_flags;
        assert!(flags.contains(host_coherent), "{label}: {flags:?}");
        assert_eq!(
            flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
            writable_device_memory,
            "{label}: {flags:?}",
        );
    }
}
