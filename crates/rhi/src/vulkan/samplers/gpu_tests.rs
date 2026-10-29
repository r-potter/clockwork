//! T1 tests of samplers on a real device (design 0002, R3, part of R6 and
//! part of R24).

use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::descriptor_heaps::SAMPLER_HEAP_CAPACITY;
use crate::vulkan::test_support::VulkanTestBackend;
use crate::{AddressMode, Filter, HeapKind, RenderHardwareInterface, RhiError, SamplerDesc};

/// A trilinear sampler that repeats, labelled `label`.
fn desc(label: &str) -> SamplerDesc<'_> {
    SamplerDesc {
        label,
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        mip_filter: Filter::Linear,
        address_modes: [AddressMode::Repeat; 3],
    }
}

/// Each sampler, with each filter and address mode, takes a slot of its
/// own in the sampler heap, and destroying it gives the slot back.
#[test]
fn samplers_are_slots_in_the_sampler_heap() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let heap = &backend.core.heaps.sampler;

    let samplers: Vec<_> = Filter::ALL
        .into_iter()
        .flat_map(|filter| {
            AddressMode::ALL.into_iter().map(move |mode| SamplerDesc {
                mag_filter: filter,
                min_filter: filter,
                mip_filter: filter,
                address_modes: [mode; 3],
                ..desc("sampler")
            })
        })
        .map(|desc| resources.create_sampler(&desc).unwrap())
        .collect();
    let mut indices: Vec<_> = samplers
        .iter()
        .map(|&sampler| resources.sampler_index(sampler).0)
        .collect();
    indices.sort_unstable();
    indices.dedup();
    assert_eq!(indices.len(), samplers.len());
    assert_eq!(heap.slots_in_use(), samplers.len() as u32);

    for sampler in samplers {
        resources.destroy_sampler(sampler);
    }
    resources.collect_garbage();
    assert_eq!(heap.slots_in_use(), 0);
}

/// Destroying a sampler frees its slot for the next one, once it's
/// collected (R6).
#[test]
fn freed_sampler_slots_are_reused() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let first = resources.create_sampler(&desc("first")).unwrap();
    let second = resources.create_sampler(&desc("second")).unwrap();
    let freed = resources.sampler_index(first);
    resources.destroy_sampler(first);
    resources.collect_garbage();
    let third = resources.create_sampler(&desc("third")).unwrap();
    assert_eq!(resources.sampler_index(third), freed);
    assert_ne!(resources.sampler_index(second), freed);
    resources.destroy_sampler(second);
    resources.destroy_sampler(third);
}

/// Creating a sampler with the heap full returns `DescriptorHeapFull`,
/// naming the heap and the sampler, until a slot is freed (R24).
#[test]
fn full_sampler_heap_is_reported() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let samplers: Vec<_> = (0..SAMPLER_HEAP_CAPACITY)
        .map(|_| resources.create_sampler(&desc("filler")).unwrap())
        .collect();

    let err = resources.create_sampler(&desc("one too many")).unwrap_err();
    assert!(
        matches!(
            &err,
            RhiError::DescriptorHeapFull { heap: HeapKind::Sampler, label } if label == "one too many"
        ),
        "{err:?}",
    );

    resources.destroy_sampler(samplers[0]);
    resources.collect_garbage();
    let fits = resources.create_sampler(&desc("fits")).unwrap();
    for sampler in samplers.into_iter().skip(1).chain([fits]) {
        resources.destroy_sampler(sampler);
    }
}

/// Using a destroyed sampler's handle panics, naming the sampler (R3).
#[test]
#[should_panic(expected = "stale sampler handle: \"shadow\" was destroyed")]
fn stale_sampler_handles_panic() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let sampler = resources.create_sampler(&desc("shadow")).unwrap();
    resources.destroy_sampler(sampler);
    resources.sampler_index(sampler);
}
