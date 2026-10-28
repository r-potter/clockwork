//! Choosing a memory type for a resource in each [`MemoryLocation`], and
//! for the descriptor heaps (design 0002, "Memory allocator" and "Binding
//! model").
//!
//! The choice is a pure function of the device's memory properties, the
//! types the resource can use, and what the memory is for, so every kind of
//! device is tested with fake properties.

use std::fmt;

use ash::vk;

use crate::MemoryLocation;

/// Flags of memory types the allocator never uses: protected memory, which
/// needs the protected-memory feature; lazily allocated memory, which only
/// transient attachments can use; and AMD's device-coherent and uncached
/// memory, which need `VK_AMD_device_coherent_memory` and are slow for
/// anything else.
const EXCLUDED: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::PROTECTED.as_raw()
        | vk::MemoryPropertyFlags::LAZILY_ALLOCATED.as_raw()
        | vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD.as_raw()
        | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD.as_raw(),
);

/// What memory is for: a resource in one of the RHI's locations, or one of
/// the backend's descriptor heaps, which have a rule of their own rather
/// than another public location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::vulkan) enum MemoryUse {
    /// A buffer or texture in a [`MemoryLocation`].
    Location(MemoryLocation),
    /// A descriptor heap, which the CPU writes descriptors into and the GPU
    /// reads on every texture and sampler access.
    DescriptorHeap,
}

impl fmt::Display for MemoryUse {
    /// The location, as in `Upload`, or `descriptor heap`, for messages.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Location(location) => write!(f, "{location:?}"),
            Self::DescriptorHeap => f.write_str("descriptor heap"),
        }
    }
}

impl From<MemoryLocation> for MemoryUse {
    fn from(location: MemoryLocation) -> Self {
        Self::Location(location)
    }
}

/// The flags a memory type must have to hold memory for `usage`.
fn required_flags(usage: MemoryUse) -> vk::MemoryPropertyFlags {
    match usage {
        MemoryUse::Location(MemoryLocation::Device) => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        MemoryUse::Location(MemoryLocation::Upload) | MemoryUse::DescriptorHeap => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        MemoryUse::Location(MemoryLocation::Readback) => vk::MemoryPropertyFlags::HOST_VISIBLE,
    }
}

/// Chooses the memory type for `usage`, from the types in `type_bits` (the
/// resource's `memoryTypeBits`), or returns `None` if none of them has the
/// flags `usage` requires.
///
/// Among the types that qualify, the preferences in design 0002's table
/// decide, in order:
///
/// | Location | Preferred |
/// |---|---|
/// | `Device` | not host-visible, so that host-visible device memory is left for uploads |
/// | `Upload` | device-local if its heap holds at least half the device's local memory, as with resizable BAR or an integrated GPU; otherwise not device-local, since a small BAR window is too precious to spend on uploads |
/// | `Readback` | host-cached, then host-coherent |
/// | descriptor heaps | device-local, in any heap, even a small BAR window, since the heaps are a few MiB and the GPU reads them constantly; otherwise system memory |
///
/// Types that remain tied are taken in the driver's order, which Vulkan
/// arranges from best to worst for each set of flags.
pub(super) fn choose_memory_type_index(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    usage: MemoryUse,
) -> Option<u32> {
    let types = properties.memory_types_as_slice();
    let required = required_flags(usage);
    (0..types.len() as u32)
        .filter(|&index| type_bits & (1 << index) != 0)
        .filter(|&index| {
            let flags = types[index as usize].property_flags;
            flags.contains(required) && !flags.intersects(EXCLUDED)
        })
        // `max_by_key` keeps the last of equal keys, so search from the end
        // to keep the driver's first.
        .rev()
        .max_by_key(|&index| preference(properties, index, usage))
}

/// How much `usage` prefers the memory type `index`, as a key that sorts
/// higher for better types. See [`choose_memory_type_index`].
///
/// The key is two preferences, compared in order: the first decides, and
/// the second only breaks ties in the first. Each is `true` when the type
/// has what `usage` prefers:
///
/// | Usage | First | Second |
/// |---|---|---|
/// | `Device` | not host-visible | unused, always `false` |
/// | `Upload` | device-local, in a heap that holds most of the local memory | not device-local, so system memory beats a small BAR window |
/// | `Readback` | host-cached | host-coherent |
/// | descriptor heaps | device-local | unused, always `false` |
fn preference(
    properties: &vk::PhysicalDeviceMemoryProperties,
    index: u32,
    usage: MemoryUse,
) -> (bool, bool) {
    let memory_type = properties.memory_types[index as usize];
    let flags = memory_type.property_flags;
    let device_local = flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
    match usage {
        MemoryUse::Location(MemoryLocation::Device) => (
            !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
            false,
        ),
        MemoryUse::Location(MemoryLocation::Upload) => {
            let large_heap = holds_most_local_memory(properties, memory_type.heap_index);
            // Best: device-local in a large heap. Then system memory. Last,
            // a small BAR window.
            (device_local && large_heap, !device_local)
        }
        MemoryUse::Location(MemoryLocation::Readback) => (
            flags.contains(vk::MemoryPropertyFlags::HOST_CACHED),
            flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
        ),
        MemoryUse::DescriptorHeap => (device_local, false),
    }
}

/// Whether the heap `heap_index` holds at least half of the device's
/// device-local memory, as the heap the CPU can see does with resizable BAR
/// or on an integrated GPU, and a 256 MiB BAR window doesn't.
fn holds_most_local_memory(
    properties: &vk::PhysicalDeviceMemoryProperties,
    heap_index: u32,
) -> bool {
    let heaps = properties.memory_heaps_as_slice();
    let local: vk::DeviceSize = heaps
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .sum();
    let heap = heaps[heap_index as usize];
    heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) && heap.size.saturating_mul(2) >= local
}

#[cfg(test)]
mod tests {
    //! T0 tests of memory type selection against the memory properties of
    //! each kind of device (design 0002, R20).

    use super::*;

    /// One gibibyte.
    const GIB: vk::DeviceSize = 1 << 30;

    /// Shorthands for memory property flags.
    const DEVICE_LOCAL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    /// See [`DEVICE_LOCAL`].
    const HOST_VISIBLE: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_VISIBLE;
    /// See [`DEVICE_LOCAL`].
    const HOST_COHERENT: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_COHERENT;
    /// See [`DEVICE_LOCAL`].
    const HOST_CACHED: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::HOST_CACHED;

    /// Every memory type.
    const ALL: u32 = u32::MAX;

    /// Memory properties with `heaps`, as sizes and whether each is
    /// device-local, and `types`, as flags and heap indices.
    fn properties(
        heaps: &[(vk::DeviceSize, bool)],
        types: &[(vk::MemoryPropertyFlags, u32)],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: heaps.len() as u32,
            memory_type_count: types.len() as u32,
            ..Default::default()
        };
        for (heap, &(size, local)) in properties.memory_heaps.iter_mut().zip(heaps) {
            heap.size = size;
            if local {
                heap.flags = vk::MemoryHeapFlags::DEVICE_LOCAL;
            }
        }
        for (memory_type, &(flags, heap_index)) in properties.memory_types.iter_mut().zip(types) {
            *memory_type = vk::MemoryType {
                property_flags: flags,
                heap_index,
            };
        }
        properties
    }

    /// A discrete GPU without resizable BAR, laid out as NVIDIA's drivers
    /// lay it out: video memory, system memory, and a 256 MiB BAR window.
    fn discrete_without_rebar() -> vk::PhysicalDeviceMemoryProperties {
        properties(
            &[(24 * GIB, true), (64 * GIB, false), (256 << 20, true)],
            &[
                (vk::MemoryPropertyFlags::empty(), 1),
                (DEVICE_LOCAL, 0),
                (HOST_VISIBLE | HOST_COHERENT, 1),
                (HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, 1),
                (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT, 2),
            ],
        )
    }

    /// The same GPU with resizable BAR, where the CPU can see all of video
    /// memory.
    fn discrete_with_rebar() -> vk::PhysicalDeviceMemoryProperties {
        properties(
            &[(24 * GIB, true), (64 * GIB, false)],
            &[
                (vk::MemoryPropertyFlags::empty(), 1),
                (DEVICE_LOCAL, 0),
                (HOST_VISIBLE | HOST_COHERENT, 1),
                (HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, 1),
                (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT, 0),
            ],
        )
    }

    /// An integrated GPU, whose one heap is shared system memory.
    fn integrated() -> vk::PhysicalDeviceMemoryProperties {
        properties(
            &[(16 * GIB, true)],
            &[
                (DEVICE_LOCAL, 0),
                (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT, 0),
                (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, 0),
            ],
        )
    }

    /// Lavapipe, which has one heap and one memory type with every flag.
    fn lavapipe() -> vk::PhysicalDeviceMemoryProperties {
        properties(
            &[(8 * GIB, true)],
            &[(DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, 0)],
        )
    }

    /// The type chosen for each location, as `(device, upload, readback)`.
    fn choices(properties: &vk::PhysicalDeviceMemoryProperties) -> (u32, u32, u32) {
        let choose = |location: MemoryLocation| {
            choose_memory_type_index(properties, ALL, location.into()).unwrap()
        };
        (
            choose(MemoryLocation::Device),
            choose(MemoryLocation::Upload),
            choose(MemoryLocation::Readback),
        )
    }

    /// Without resizable BAR, uploads go to system memory, not the small BAR
    /// window, and readbacks to cached system memory.
    #[test]
    fn discrete_gpus_without_rebar_upload_from_system_memory() {
        assert_eq!(choices(&discrete_without_rebar()), (1, 2, 3));
    }

    /// With resizable BAR, uploads go straight to video memory.
    #[test]
    fn discrete_gpus_with_rebar_upload_to_video_memory() {
        assert_eq!(choices(&discrete_with_rebar()), (1, 4, 3));
    }

    /// On an integrated GPU, device memory is the type the CPU can't see,
    /// uploads use the first coherent type, and readbacks the cached one.
    #[test]
    fn integrated_gpus_use_shared_memory() {
        assert_eq!(choices(&integrated()), (0, 1, 2));
    }

    /// Lavapipe's one type serves every location.
    #[test]
    fn lavapipe_uses_its_only_type() {
        assert_eq!(choices(&lavapipe()), (0, 0, 0));
    }

    /// Only types in the resource's `memoryTypeBits` are considered: without
    /// the preferred type, the next best is chosen.
    #[test]
    fn only_the_resources_types_are_considered() {
        let properties = discrete_without_rebar();
        let without = |index: u32| ALL & !(1 << index);
        assert_eq!(
            choose_memory_type_index(&properties, without(1), MemoryLocation::Device.into()),
            Some(4)
        );
        assert_eq!(
            choose_memory_type_index(&properties, without(2), MemoryLocation::Upload.into()),
            Some(3)
        );
        assert_eq!(
            choose_memory_type_index(&properties, without(3), MemoryLocation::Readback.into()),
            Some(2)
        );
    }

    /// When the small BAR window is the only host-visible type a resource
    /// can use, uploads use it rather than failing.
    #[test]
    fn a_small_bar_window_is_the_last_resort_for_uploads() {
        let properties = discrete_without_rebar();
        assert_eq!(
            choose_memory_type_index(&properties, 1 << 4, MemoryLocation::Upload.into()),
            Some(4)
        );
    }

    /// A readback type that's cached but not coherent is preferred to one
    /// that's coherent but not cached.
    #[test]
    fn readbacks_prefer_cached_to_coherent() {
        let properties = properties(
            &[(GIB, false)],
            &[
                (HOST_VISIBLE | HOST_COHERENT, 0),
                (HOST_VISIBLE | HOST_CACHED, 0),
            ],
        );
        assert_eq!(
            choose_memory_type_index(&properties, ALL, MemoryLocation::Readback.into()),
            Some(1)
        );
    }

    /// Ties go to the driver's first type.
    #[test]
    fn ties_keep_the_drivers_order() {
        let properties = properties(&[(GIB, true)], &[(DEVICE_LOCAL, 0), (DEVICE_LOCAL, 0)]);
        assert_eq!(
            choose_memory_type_index(&properties, ALL, MemoryLocation::Device.into()),
            Some(0)
        );
    }

    /// Protected, lazily allocated and AMD device-coherent types are never
    /// chosen, even when they're first and otherwise qualify.
    #[test]
    fn types_needing_other_features_are_excluded() {
        for excluded in [
            vk::MemoryPropertyFlags::PROTECTED,
            vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD,
            vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD,
        ] {
            let properties = properties(
                &[(GIB, true)],
                &[(DEVICE_LOCAL | excluded, 0), (DEVICE_LOCAL, 0)],
            );
            assert_eq!(
                choose_memory_type_index(&properties, ALL, MemoryLocation::Device.into()),
                Some(1),
                "{excluded:?}"
            );
        }
    }

    /// A location no type can serve is `None`, for the caller to report.
    #[test]
    fn no_qualifying_type_is_none() {
        let properties = properties(&[(GIB, true)], &[(DEVICE_LOCAL, 0)]);
        assert_eq!(
            choose_memory_type_index(&properties, ALL, MemoryLocation::Upload.into()),
            None
        );
        assert_eq!(
            choose_memory_type_index(&properties, 0, MemoryLocation::Device.into()),
            None
        );
    }

    /// The descriptor heaps go in device-local memory the CPU can write on
    /// every kind of device, even a small BAR window, which uploads avoid.
    #[test]
    fn descriptor_heaps_are_device_local() {
        let choose =
            |properties| choose_memory_type_index(&properties, ALL, MemoryUse::DescriptorHeap);
        assert_eq!(choose(discrete_without_rebar()), Some(4));
        assert_eq!(choose(discrete_with_rebar()), Some(4));
        assert_eq!(choose(integrated()), Some(1));
        assert_eq!(choose(lavapipe()), Some(0));
    }

    /// A device with no device-local memory the CPU can write puts the
    /// descriptor heaps in coherent system memory instead.
    #[test]
    fn descriptor_heaps_fall_back_to_system_memory() {
        let properties = properties(
            &[(24 * GIB, true), (64 * GIB, false)],
            &[
                (DEVICE_LOCAL, 0),
                (HOST_VISIBLE | HOST_CACHED, 1),
                (HOST_VISIBLE | HOST_COHERENT, 1),
            ],
        );
        assert_eq!(
            choose_memory_type_index(&properties, ALL, MemoryUse::DescriptorHeap),
            Some(2)
        );
    }

    /// A heap counts as most of the local memory from half of it up.
    #[test]
    fn half_the_local_memory_counts_as_most() {
        let less_than_half = properties(&[(GIB, true), (GIB, true), (GIB / 2, true)], &[]);
        assert!(!holds_most_local_memory(&less_than_half, 0));
        let exactly_half = properties(&[(GIB, true), (GIB, true)], &[]);
        assert!(holds_most_local_memory(&exactly_half, 0));
        let not_local = properties(&[(GIB, false)], &[]);
        assert!(!holds_most_local_memory(&not_local, 0));
    }
}
