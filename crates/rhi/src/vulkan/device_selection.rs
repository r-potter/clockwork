//! Choosing the physical device to render with.
//!
//! Selection happens in three phases:
//!
//! 0. **Filter.** If a device filter is given, only devices whose names
//!    contain it, ignoring case, are considered (R10). See [`filter`].
//! 1. **Requirements.** Every device that does not meet all of
//!    [`REQUIREMENTS`] is rejected, and the reasons are logged.
//! 2. **Comparison.** The remaining devices are compared in pairs with
//!    [`compare`], keeping the better of each pair, until one is left.
//!
//! The queue families to use on the selected device are then chosen with
//! [`queue_families`].

use std::cmp::Ordering;

use anyhow::{Result, bail};
use ash::{ext, khr, vk};
use tracing::info;

use super::API_VERSION;
use super::capabilities;
use super::descriptor_heaps::HeapLayout;
use super::physical_device::PhysicalDevice;
use super::surface::SurfaceSupport;
use crate::{Capabilities, MAX_PUSH_DATA_SIZE, PresentCapabilities, QueueKind};

/// A device being considered for selection.
pub(super) struct Candidate {
    /// The device, with the properties and features selection compares.
    pub device: PhysicalDevice,
    /// How the device can present to the window's surface, or `None` when
    /// headless.
    pub surface_support: Option<SurfaceSupport>,
}

/// The selected device, the queue families to use on it, and its
/// capabilities.
pub(super) struct Selection {
    /// The device that met every requirement and compared best.
    pub device: PhysicalDevice,
    /// The queue families to create the logical device's queues from.
    pub queue_families: QueueFamilies,
    /// What the device can do beyond the requirements, using
    /// `queue_families`.
    pub capabilities: Capabilities,
    /// What presenting can do on the device.
    pub present_capabilities: PresentCapabilities,
}

/// Indices of the queue families to use on a device, for each queue kind
/// and for presenting (design 0002, "Queues and submission").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct QueueFamilies {
    /// Supports graphics operations.
    pub graphics: u32,
    /// Compute work: a dedicated compute family if the device has one, and
    /// `graphics` otherwise.
    pub compute: u32,
    /// Transfers: a dedicated transfer family if the device has one, then
    /// `compute`.
    pub transfer: u32,
    /// Can present to the window's surface. `None` when headless. May be the
    /// same family as `graphics`.
    pub present: Option<u32>,
}

impl QueueFamilies {
    /// The family `kind`'s work is submitted to.
    pub(super) fn family(self, kind: QueueKind) -> u32 {
        match kind {
            QueueKind::Graphics => self.graphics,
            QueueKind::Compute => self.compute,
            QueueKind::Transfer => self.transfer,
        }
    }

    /// The kind whose queue `kind`'s work is submitted to: `kind` itself if
    /// its family is dedicated, and otherwise the most capable kind it
    /// shares a family with, which is what `Queue::kind` reports.
    pub(super) fn serving_kind(self, kind: QueueKind) -> QueueKind {
        let family = self.family(kind);
        QueueKind::ALL
            .into_iter()
            .find(|&other| self.family(other) == family)
            .expect("`kind` shares its own family")
    }
}

/// The push data, in bytes, that each draw or dispatch can carry (design
/// 0002, "Binding model"): [`MAX_PUSH_DATA_SIZE`], which recording checks
/// against.
pub(super) const MIN_PUSH_DATA_SIZE: u64 = MAX_PUSH_DATA_SIZE as u64;

/// A minimum requirement a device must meet to be selected.
struct Requirement {
    /// Describes what is required, for logging when a device fails it.
    description: &'static str,
    /// Returns whether the candidate meets the requirement.
    is_met: fn(&Candidate) -> bool,
}

/// Every requirement a device must meet to be selected.
///
/// Requirements about presenting only apply when there is a window, and are
/// always met when headless.
const REQUIREMENTS: &[Requirement] = &[
    Requirement {
        // The loader version check does not guarantee that each device
        // supports the version.
        description: "supports the targeted Vulkan version",
        is_met: |candidate| candidate.device.api_version() >= API_VERSION,
    },
    Requirement {
        description: "supports VK_EXT_shader_object",
        is_met: |candidate| {
            candidate
                .device
                .supports_extension(ext::shader_object::NAME)
                && candidate.device.features.shader_object.shader_object == vk::TRUE
        },
    },
    Requirement {
        // Needed by shaders that use `SV_VertexID`, as Slang compiles it using
        // the base vertex.
        description: "supports shader draw parameters",
        is_met: |candidate| candidate.device.features.vulkan11.shader_draw_parameters == vk::TRUE,
    },
    Requirement {
        description: "supports dynamic rendering and synchronization2",
        is_met: |candidate| {
            let features = &candidate.device.features.vulkan13;
            features.dynamic_rendering == vk::TRUE && features.synchronization2 == vk::TRUE
        },
    },
    Requirement {
        description: "supports VK_EXT_descriptor_heap",
        is_met: |candidate| {
            candidate
                .device
                .supports_extension(ext::descriptor_heap::NAME)
                && candidate.device.features.descriptor_heap.descriptor_heap == vk::TRUE
        },
    },
    Requirement {
        description: "supports VK_KHR_device_address_commands",
        is_met: |candidate| {
            let features = &candidate.device.features.device_address_commands;
            candidate
                .device
                .supports_extension(khr::device_address_commands::NAME)
                && features.device_address_commands == vk::TRUE
        },
    },
    Requirement {
        description: "supports buffer device addresses, timeline semaphores and host query reset",
        is_met: |candidate| {
            let features = &candidate.device.features.vulkan12;
            features.buffer_device_address == vk::TRUE
                && features.timeline_semaphore == vk::TRUE
                && features.host_query_reset == vk::TRUE
        },
    },
    Requirement {
        description: "supports enough push data",
        is_met: |candidate| {
            let properties = &candidate.device.properties.descriptor_heap;
            properties.max_push_data_size >= MIN_PUSH_DATA_SIZE
        },
    },
    Requirement {
        description: "supports large enough descriptor heaps",
        is_met: |candidate| {
            let properties = &candidate.device.properties.descriptor_heap;
            HeapLayout::resource(properties).size() <= properties.max_resource_heap_size
                && HeapLayout::sampler(properties).size() <= properties.max_sampler_heap_size
        },
    },
    Requirement {
        // Both rendering and its timings need it (design 0002,
        // "Timestamps").
        description: "has a queue family that supports graphics and timestamps",
        is_met: |candidate| graphics_families(&candidate.device).next().is_some(),
    },
    Requirement {
        description: "supports VK_KHR_swapchain",
        is_met: |candidate| {
            candidate.surface_support.is_none()
                || candidate.device.supports_extension(khr::swapchain::NAME)
        },
    },
    Requirement {
        description: "has a queue family that can present to the window",
        is_met: |candidate| {
            candidate
                .surface_support
                .as_ref()
                .is_none_or(|support| support.present_families.contains(&true))
        },
    },
    Requirement {
        description: "supports a surface format and present mode for the window",
        is_met: |candidate| {
            candidate.surface_support.as_ref().is_none_or(|support| {
                !support.formats.is_empty() && !support.present_modes.is_empty()
            })
        },
    },
];

/// Chooses the best of `candidates` that matches `device_filter` (if any)
/// and meets every requirement, and the queue families to use on it.
///
/// # Errors
///
/// Returns an error if no device matches the filter, listing every device,
/// or if no matching device meets every requirement.
pub(super) fn select(candidates: Vec<Candidate>, device_filter: Option<&str>) -> Result<Selection> {
    let selected = filter(candidates, device_filter)?
        .into_iter()
        .filter(|candidate| {
            let unmet = unmet_requirements(candidate);
            if !unmet.is_empty() {
                info!(
                    "rejected {}: does not meet requirements: {}",
                    candidate.device.name(),
                    unmet.join(", "),
                );
            }
            unmet.is_empty()
        })
        // Keep the earlier device on ties, so the result does not depend on
        // how ties are broken by the standard library.
        .reduce(|best, candidate| {
            if compare(&candidate.device, &best.device) == Ordering::Greater {
                candidate
            } else {
                best
            }
        });

    let Some(selected) = selected else {
        bail!("no Vulkan device meets the minimum requirements; see the log for details");
    };
    let queue_families = queue_families(&selected);
    let capabilities = capabilities::derive(&selected.device, queue_families);
    let present_capabilities = capabilities::derive_present(&selected.device);
    info!(
        "selected {} ({}), with queue families {queue_families:?}",
        selected.device.name(),
        crate::capabilities::summary(&capabilities, &present_capabilities),
    );
    Ok(Selection {
        device: selected.device,
        queue_families,
        capabilities,
        present_capabilities,
    })
}

/// Keeps the candidates whose names contain `device_filter`, ignoring case,
/// or all of them if it's `None`.
///
/// # Errors
///
/// Returns an error listing every candidate's name if none matches.
fn filter(candidates: Vec<Candidate>, device_filter: Option<&str>) -> Result<Vec<Candidate>> {
    let Some(device_filter) = device_filter else {
        return Ok(candidates);
    };
    let wanted = device_filter.to_lowercase();
    let names: Vec<String> = candidates
        .iter()
        .map(|candidate| candidate.device.name().into_owned())
        .collect();
    let matching: Vec<_> = candidates
        .into_iter()
        .zip(&names)
        .filter(|(_, name)| name.to_lowercase().contains(&wanted))
        .map(|(candidate, _)| candidate)
        .collect();
    if matching.is_empty() {
        let available = if names.is_empty() {
            "there are none".to_owned()
        } else {
            format!("the devices are: {}", names.join(", "))
        };
        bail!("no Vulkan device's name contains \"{device_filter}\"; {available}");
    }
    for name in names
        .iter()
        .filter(|name| !name.to_lowercase().contains(&wanted))
    {
        info!("ignored {name}: its name doesn't contain \"{device_filter}\"");
    }
    Ok(matching)
}

/// Returns the descriptions of the requirements that `candidate` does not
/// meet.
fn unmet_requirements(candidate: &Candidate) -> Vec<&'static str> {
    REQUIREMENTS
        .iter()
        .filter(|requirement| !(requirement.is_met)(candidate))
        .map(|requirement| requirement.description)
        .collect()
}

/// Compares two devices that both meet the requirements; the greater device
/// is preferred.
///
/// Devices are compared by each criterion in turn, moving on to the next only
/// when they are equal:
///
/// 1. device type, preferring discrete GPUs, then integrated, virtual, CPU
///    and other devices
/// 2. total size of the device-local memory heaps
fn compare(a: &PhysicalDevice, b: &PhysicalDevice) -> Ordering {
    device_type_rank(a.device_type())
        .cmp(&device_type_rank(b.device_type()))
        .then_with(|| a.device_local_memory().cmp(&b.device_local_memory()))
}

/// Ranks device types by preference; higher is better.
fn device_type_rank(device_type: vk::PhysicalDeviceType) -> u8 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 1,
        _ => 0,
    }
}

/// Chooses the queue families to use on a candidate that meets every
/// requirement.
///
/// Prefers a single family that supports both graphics and presenting, so
/// that swapchain images do not need to be shared between queue families.
/// Otherwise uses the first family that supports each. Compute and transfer
/// work use dedicated families where the device has them (see
/// [`QueueFamilies`]).
///
/// # Panics
///
/// Panics if `candidate` does not meet the requirements.
fn queue_families(candidate: &Candidate) -> QueueFamilies {
    let (graphics, present) = match &candidate.surface_support {
        None => (graphics_families(&candidate.device).next(), None),
        Some(support) => {
            let can_present = |family: &u32| support.present_families[*family as usize];
            let (graphics, present) = match graphics_families(&candidate.device).find(can_present) {
                Some(family) => (Some(family), Some(family)),
                None => (
                    graphics_families(&candidate.device).next(),
                    (0..support.present_families.len() as u32).find(can_present),
                ),
            };
            let present = present.expect("requirements guarantee a present queue family");
            (graphics, Some(present))
        }
    };
    let graphics = graphics.expect("requirements guarantee a graphics queue family");
    let compute = dedicated_compute_family(&candidate.device).unwrap_or(graphics);
    QueueFamilies {
        graphics,
        compute,
        transfer: dedicated_transfer_family(&candidate.device).unwrap_or(compute),
        present,
    }
}

/// The first of `device`'s queue families that supports compute but not
/// graphics, or `None` if it has none, when compute work shares the graphics
/// family (design 0002, "Queues and submission").
fn dedicated_compute_family(device: &PhysicalDevice) -> Option<u32> {
    first_family(device, vk::QueueFlags::COMPUTE, vk::QueueFlags::GRAPHICS)
}

/// The first of `device`'s queue families that supports transfers but
/// neither graphics nor compute, or `None` if it has none, when transfers
/// share a more capable family (design 0002, "Queues and submission").
fn dedicated_transfer_family(device: &PhysicalDevice) -> Option<u32> {
    first_family(
        device,
        vk::QueueFlags::TRANSFER,
        vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
    )
}

/// The index of `device`'s first queue family that supports everything in
/// `wanted` and nothing in `unwanted`.
fn first_family(
    device: &PhysicalDevice,
    wanted: vk::QueueFlags,
    unwanted: vk::QueueFlags,
) -> Option<u32> {
    device
        .queue_families
        .iter()
        .position(|family| {
            family.queue_flags.contains(wanted) && !family.queue_flags.intersects(unwanted)
        })
        .map(|index| index as u32)
}

/// The indices of `device`'s queue families that support graphics and can
/// write timestamps, which are the families rendering can use.
fn graphics_families(device: &PhysicalDevice) -> impl Iterator<Item = u32> + '_ {
    device
        .queue_families
        .iter()
        .enumerate()
        .filter(|(_, family)| {
            family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && family.timestamp_valid_bits > 0
        })
        .map(|(index, _)| index as u32)
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    use crate::vulkan::descriptor_heaps::{RESOURCE_HEAP_CAPACITY, SAMPLER_HEAP_CAPACITY};
    use crate::vulkan::physical_device::{Features, Properties};

    /// One gibibyte.
    const GIB: vk::DeviceSize = 1 << 30;

    /// Creates a device that meets every requirement.
    fn device(
        name: &str,
        device_type: vk::PhysicalDeviceType,
        memory: vk::DeviceSize,
    ) -> PhysicalDevice {
        let mut properties = Properties::default();
        properties.core.device_type = device_type;
        properties.core.api_version = API_VERSION;
        for (dst, src) in properties.core.device_name.iter_mut().zip(name.bytes()) {
            *dst = src as _;
        }
        // The RTX 4090's descriptor heap properties, from driver 617.42.
        let heap = &mut properties.descriptor_heap;
        heap.max_push_data_size = 256;
        heap.max_resource_heap_size = 32 << 20;
        heap.max_sampler_heap_size = 128 << 10;
        heap.min_resource_heap_reserved_range = 0x17a00;
        heap.min_sampler_heap_reserved_range = 0x200;
        heap.image_descriptor_size = 32;
        heap.image_descriptor_alignment = 32;
        heap.sampler_descriptor_size = 32;
        heap.sampler_descriptor_alignment = 32;

        let mut memory_properties = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: 1,
            ..Default::default()
        };
        memory_properties.memory_heaps[0] = vk::MemoryHeap {
            size: memory,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };

        let mut features = Features::default();
        features.vulkan13.dynamic_rendering = vk::TRUE;
        features.vulkan13.synchronization2 = vk::TRUE;
        features.vulkan11.shader_draw_parameters = vk::TRUE;
        features.vulkan12.buffer_device_address = vk::TRUE;
        features.vulkan12.timeline_semaphore = vk::TRUE;
        features.vulkan12.host_query_reset = vk::TRUE;
        features.shader_object.shader_object = vk::TRUE;
        features.descriptor_heap.descriptor_heap = vk::TRUE;
        features.device_address_commands.device_address_commands = vk::TRUE;

        PhysicalDevice {
            handle: vk::PhysicalDevice::null(),
            properties,
            features,
            memory_properties,
            queue_families: vec![vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS,
                queue_count: 1,
                timestamp_valid_bits: 64,
                ..Default::default()
            }],
            extensions: [
                ext::shader_object::NAME,
                ext::descriptor_heap::NAME,
                khr::device_address_commands::NAME,
            ]
            .into_iter()
            .map(|name| {
                vk::ExtensionProperties::default()
                    .extension_name(name)
                    .unwrap()
            })
            .collect(),
        }
    }

    /// Asserts that a device changed by `modify` is rejected, by checking
    /// that a worse device that meets every requirement is selected instead.
    fn assert_rejected(modify: impl FnOnce(&mut PhysicalDevice)) {
        let mut rejected = device("rejected", vk::PhysicalDeviceType::DISCRETE_GPU, 16 * GIB);
        modify(&mut rejected);
        let devices = vec![
            rejected,
            device("accepted", vk::PhysicalDeviceType::INTEGRATED_GPU, 8 * GIB),
        ];
        assert_eq!(selected_name(devices), "accepted");
    }

    /// Removes the extension `name` from `device`.
    fn remove_extension(device: &mut PhysicalDevice, name: &CStr) {
        device
            .extensions
            .retain(|extension| extension.extension_name_as_c_str() != Ok(name));
    }

    /// Wraps `devices` as candidates for headless selection.
    fn headless(devices: Vec<PhysicalDevice>) -> Vec<Candidate> {
        devices
            .into_iter()
            .map(|device| Candidate {
                device,
                surface_support: None,
            })
            .collect()
    }

    /// Adds `VK_KHR_swapchain` to `device`'s extensions, and wraps it as a
    /// candidate whose queue families can present as given by
    /// `present_families`.
    fn windowed(mut device: PhysicalDevice, present_families: &[bool]) -> Candidate {
        device.extensions.push(
            vk::ExtensionProperties::default()
                .extension_name(khr::swapchain::NAME)
                .unwrap(),
        );
        Candidate {
            device,
            surface_support: Some(SurfaceSupport {
                present_families: present_families.to_vec(),
                formats: vec![vk::SurfaceFormatKHR::default()],
                present_modes: vec![vk::PresentModeKHR::FIFO],
            }),
        }
    }

    /// Adds a queue family with `flags`, which can write timestamps, to
    /// `device`.
    fn with_family(mut device: PhysicalDevice, flags: vk::QueueFlags) -> PhysicalDevice {
        device.queue_families.push(vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: 1,
            timestamp_valid_bits: 64,
            ..Default::default()
        });
        device
    }

    fn selected_name(devices: Vec<PhysicalDevice>) -> String {
        select(headless(devices), None)
            .unwrap()
            .device
            .name()
            .into_owned()
    }

    #[test]
    fn prefers_discrete_over_other_types() {
        let devices = vec![
            device("cpu", vk::PhysicalDeviceType::CPU, 64 * GIB),
            device(
                "integrated",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                16 * GIB,
            ),
            device("discrete", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
            device("virtual", vk::PhysicalDeviceType::VIRTUAL_GPU, 32 * GIB),
        ];
        assert_eq!(selected_name(devices), "discrete");
    }

    #[test]
    fn prefers_more_memory_for_the_same_type() {
        let devices = vec![
            device("small", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
            device("large", vk::PhysicalDeviceType::DISCRETE_GPU, 16 * GIB),
        ];
        assert_eq!(selected_name(devices), "large");
    }

    #[test]
    fn keeps_the_first_device_on_ties() {
        let devices = vec![
            device("first", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
            device("second", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
        ];
        assert_eq!(selected_name(devices), "first");
    }

    #[test]
    fn compares_the_total_not_the_largest_heap() {
        let mut split = device("split", vk::PhysicalDeviceType::DISCRETE_GPU, 6 * GIB);
        split.memory_properties.memory_heap_count = 2;
        split.memory_properties.memory_heaps[1] = vk::MemoryHeap {
            size: 6 * GIB,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        let devices = vec![
            device("single", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
            split,
        ];
        assert_eq!(selected_name(devices), "split");
    }

    #[test]
    fn ignores_memory_that_is_not_device_local() {
        let mut host_heavy = device("host heavy", vk::PhysicalDeviceType::DISCRETE_GPU, 4 * GIB);
        host_heavy.memory_properties.memory_heap_count = 2;
        host_heavy.memory_properties.memory_heaps[1] = vk::MemoryHeap {
            size: 64 * GIB,
            flags: vk::MemoryHeapFlags::empty(),
        };
        let devices = vec![
            host_heavy,
            device(
                "device heavy",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                8 * GIB,
            ),
        ];
        assert_eq!(selected_name(devices), "device heavy");
    }

    #[test]
    fn rejects_devices_below_the_targeted_version() {
        let mut old = device("old", vk::PhysicalDeviceType::DISCRETE_GPU, 16 * GIB);
        old.properties.core.api_version = vk::API_VERSION_1_3;
        let devices = vec![
            old,
            device("new", vk::PhysicalDeviceType::INTEGRATED_GPU, 8 * GIB),
        ];
        assert_eq!(selected_name(devices), "new");
    }

    #[test]
    fn rejects_devices_without_dynamic_rendering() {
        let mut no_dynamic_rendering = device(
            "no dynamic rendering",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            16 * GIB,
        );
        no_dynamic_rendering.features.vulkan13.dynamic_rendering = vk::FALSE;
        let devices = vec![
            no_dynamic_rendering,
            device(
                "dynamic rendering",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                8 * GIB,
            ),
        ];
        assert_eq!(selected_name(devices), "dynamic rendering");
    }

    #[test]
    fn rejects_devices_without_shader_objects() {
        let mut no_extension = device(
            "no extension",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            16 * GIB,
        );
        no_extension.extensions.clear();
        let mut no_feature = device("no feature", vk::PhysicalDeviceType::DISCRETE_GPU, 16 * GIB);
        no_feature.features.shader_object.shader_object = vk::FALSE;
        let devices = vec![
            no_extension,
            no_feature,
            device(
                "shader objects",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                8 * GIB,
            ),
        ];
        assert_eq!(selected_name(devices), "shader objects");
    }

    #[test]
    fn rejects_devices_without_shader_draw_parameters() {
        let mut no_draw_parameters = device(
            "no draw parameters",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            16 * GIB,
        );
        no_draw_parameters.features.vulkan11.shader_draw_parameters = vk::FALSE;
        let devices = vec![
            no_draw_parameters,
            device(
                "draw parameters",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                8 * GIB,
            ),
        ];
        assert_eq!(selected_name(devices), "draw parameters");
    }

    /// Descriptor heaps need both the extension and its feature (ADR 0002).
    #[test]
    fn rejects_devices_without_descriptor_heaps() {
        assert_rejected(|device| remove_extension(device, ext::descriptor_heap::NAME));
        assert_rejected(|device| device.features.descriptor_heap.descriptor_heap = vk::FALSE);
    }

    /// Device address commands need both the extension and its feature
    /// (ADR 0002).
    #[test]
    fn rejects_devices_without_device_address_commands() {
        assert_rejected(|device| remove_extension(device, khr::device_address_commands::NAME));
        assert_rejected(|device| {
            let features = &mut device.features.device_address_commands;
            features.device_address_commands = vk::FALSE;
        });
    }

    /// Buffer device addresses, timeline semaphores and host query reset are
    /// each required (design 0002, R1).
    #[test]
    fn rejects_devices_without_each_vulkan_1_2_feature() {
        assert_rejected(|device| device.features.vulkan12.buffer_device_address = vk::FALSE);
        assert_rejected(|device| device.features.vulkan12.timeline_semaphore = vk::FALSE);
        assert_rejected(|device| device.features.vulkan12.host_query_reset = vk::FALSE);
    }

    /// A device must carry at least [`MIN_PUSH_DATA_SIZE`] bytes of push
    /// data, and exactly that much is enough.
    #[test]
    fn requires_enough_push_data() {
        assert_rejected(|device| {
            device.properties.descriptor_heap.max_push_data_size = MIN_PUSH_DATA_SIZE - 1;
        });
        let mut device = device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB);
        device.properties.descriptor_heap.max_push_data_size = MIN_PUSH_DATA_SIZE;
        assert!(select(headless(vec![device]), None).is_ok());
    }

    /// Each heap must hold its capacity of descriptors as well as the
    /// driver's reserved range.
    #[test]
    fn requires_large_enough_descriptor_heaps() {
        assert_rejected(|device| {
            let heap = &mut device.properties.descriptor_heap;
            heap.max_resource_heap_size =
                u64::from(RESOURCE_HEAP_CAPACITY) * heap.image_descriptor_size - 1;
            heap.min_resource_heap_reserved_range = 0;
        });
        assert_rejected(|device| {
            let heap = &mut device.properties.descriptor_heap;
            heap.max_sampler_heap_size =
                u64::from(SAMPLER_HEAP_CAPACITY) * heap.sampler_descriptor_size;
            heap.min_sampler_heap_reserved_range = 1;
        });
    }

    /// Rendering's timings need timestamps on the graphics queue (design
    /// 0002, "Timestamps").
    #[test]
    fn rejects_devices_whose_graphics_family_lacks_timestamps() {
        assert_rejected(|device| device.queue_families[0].timestamp_valid_bits = 0);
    }

    /// A graphics family without timestamps is passed over for one with
    /// them.
    #[test]
    fn chooses_a_graphics_family_with_timestamps() {
        let mut device = with_family(
            device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
            vk::QueueFlags::GRAPHICS,
        );
        device.queue_families[0].timestamp_valid_bits = 0;
        device.queue_families[1].timestamp_valid_bits = 36;
        let selection = select(headless(vec![device]), None).unwrap();
        assert_eq!(selection.queue_families.graphics, 1);
    }

    #[test]
    fn rejects_devices_without_graphics() {
        let mut compute_only = device(
            "compute only",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            16 * GIB,
        );
        compute_only.queue_families[0].queue_flags = vk::QueueFlags::COMPUTE;
        let devices = vec![
            compute_only,
            device("graphics", vk::PhysicalDeviceType::INTEGRATED_GPU, 8 * GIB),
        ];
        assert_eq!(selected_name(devices), "graphics");
    }

    #[test]
    fn fails_when_no_device_meets_the_requirements() {
        let mut old = device("old", vk::PhysicalDeviceType::DISCRETE_GPU, 16 * GIB);
        old.properties.core.api_version = vk::API_VERSION_1_3;
        assert!(select(headless(vec![old]), None).is_err());
        assert!(select(Vec::new(), None).is_err());
    }

    #[test]
    fn headless_selection_does_not_need_presenting() {
        // No swapchain extension and no present support.
        let device = device("headless", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB);
        let selection = select(headless(vec![device]), None).unwrap();
        assert_eq!(
            selection.queue_families,
            QueueFamilies {
                graphics: 0,
                compute: 0,
                transfer: 0,
                present: None,
            },
        );
    }

    #[test]
    fn rejects_windowed_devices_without_the_swapchain_extension() {
        let mut no_swapchain = windowed(
            device(
                "no swapchain",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                16 * GIB,
            ),
            &[true],
        );
        no_swapchain
            .device
            .extensions
            .retain(|extension| extension.extension_name_as_c_str() != Ok(khr::swapchain::NAME));
        let candidates = vec![
            no_swapchain,
            windowed(
                device("swapchain", vk::PhysicalDeviceType::INTEGRATED_GPU, 8 * GIB),
                &[true],
            ),
        ];
        assert_eq!(select(candidates, None).unwrap().device.name(), "swapchain");
    }

    #[test]
    fn rejects_windowed_devices_that_cannot_present() {
        let candidates = vec![
            windowed(
                device(
                    "cannot present",
                    vk::PhysicalDeviceType::DISCRETE_GPU,
                    16 * GIB,
                ),
                &[false],
            ),
            windowed(
                device(
                    "can present",
                    vk::PhysicalDeviceType::INTEGRATED_GPU,
                    8 * GIB,
                ),
                &[true],
            ),
        ];
        assert_eq!(
            select(candidates, None).unwrap().device.name(),
            "can present"
        );
    }

    #[test]
    fn rejects_windowed_devices_without_surface_formats() {
        let mut no_formats = windowed(
            device("no formats", vk::PhysicalDeviceType::DISCRETE_GPU, 16 * GIB),
            &[true],
        );
        no_formats.surface_support.as_mut().unwrap().formats.clear();
        assert!(select(vec![no_formats], None).is_err());
    }

    #[test]
    fn prefers_one_family_for_graphics_and_presenting() {
        // Family 0 supports graphics but cannot present; family 1 does both.
        let device = with_family(
            device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
            vk::QueueFlags::GRAPHICS,
        );
        let selection = select(vec![windowed(device, &[false, true])], None).unwrap();
        assert_eq!(
            selection.queue_families,
            QueueFamilies {
                graphics: 1,
                compute: 1,
                transfer: 1,
                present: Some(1),
            },
        );
    }

    #[test]
    fn uses_separate_families_when_no_family_does_both() {
        // Family 0 supports graphics but cannot present; family 1 can present
        // but does not support graphics.
        let device = with_family(
            device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
            vk::QueueFlags::COMPUTE,
        );
        let selection = select(vec![windowed(device, &[false, true])], None).unwrap();
        assert_eq!(
            selection.queue_families,
            QueueFamilies {
                graphics: 0,
                compute: 1,
                transfer: 1,
                present: Some(1),
            },
        );
    }

    /// The filter keeps only devices whose names contain it, even when a
    /// device it excludes would compare better.
    #[test]
    fn the_device_filter_keeps_matching_devices() {
        let devices = vec![
            device(
                "NVIDIA GeForce RTX 4090",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                24 * GIB,
            ),
            device(
                "Intel(R) UHD Graphics 770",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                8 * GIB,
            ),
        ];
        let selection = select(headless(devices), Some("UHD")).unwrap();
        assert_eq!(selection.device.name(), "Intel(R) UHD Graphics 770");
    }

    /// The filter ignores case, in both the filter and the device's name.
    #[test]
    fn the_device_filter_ignores_case() {
        let devices = vec![
            device(
                "llvmpipe (LLVM 19.1.7, 256 bits)",
                vk::PhysicalDeviceType::CPU,
                8 * GIB,
            ),
            device(
                "NVIDIA GeForce RTX 3090",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                24 * GIB,
            ),
        ];
        let selection = select(headless(devices), Some("LLVMpipe")).unwrap();
        assert_eq!(selection.device.name(), "llvmpipe (LLVM 19.1.7, 256 bits)");
    }

    /// A filter that matches nothing is an error listing every device, so
    /// that the user can correct it.
    #[test]
    fn a_device_filter_matching_nothing_lists_the_devices() {
        let devices = vec![
            device(
                "NVIDIA GeForce RTX 4090",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                24 * GIB,
            ),
            device(
                "Intel(R) UHD Graphics 770",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                8 * GIB,
            ),
        ];
        let Err(err) = select(headless(devices), Some("radeon")) else {
            panic!("selected a device");
        };
        assert_eq!(
            err.to_string(),
            "no Vulkan device's name contains \"radeon\"; the devices are: \
             NVIDIA GeForce RTX 4090, Intel(R) UHD Graphics 770",
        );
    }

    /// The families chosen for headless selection of `device`.
    fn families(device: PhysicalDevice) -> QueueFamilies {
        select(headless(vec![device]), None).unwrap().queue_families
    }

    /// With one family, every kind shares it, and is served by the graphics
    /// queue.
    #[test]
    fn a_single_family_serves_every_kind() {
        let families = families(device(
            "device",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            8 * GIB,
        ));
        assert_eq!((families.compute, families.transfer), (0, 0));
        for kind in QueueKind::ALL {
            assert_eq!(families.serving_kind(kind), QueueKind::Graphics, "{kind:?}");
        }
    }

    /// Dedicated compute and transfer families are found by their flags,
    /// and each kind is served by its own queue. A family that also
    /// supports graphics or compute isn't dedicated.
    #[test]
    fn dedicated_families_serve_their_own_kinds() {
        let device = with_family(
            with_family(
                device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
                vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            ),
            vk::QueueFlags::TRANSFER,
        );
        let families = families(device);
        assert_eq!((families.compute, families.transfer), (1, 2));
        for kind in QueueKind::ALL {
            assert_eq!(families.serving_kind(kind), kind, "{kind:?}");
        }
    }

    /// Without a dedicated transfer family, transfers share the dedicated
    /// compute family, and are served by the compute queue.
    #[test]
    fn transfers_fall_back_to_the_compute_family() {
        let device = with_family(
            device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
            vk::QueueFlags::COMPUTE,
        );
        let families = families(device);
        assert_eq!((families.compute, families.transfer), (1, 1));
        assert_eq!(
            families.serving_kind(QueueKind::Transfer),
            QueueKind::Compute
        );
    }

    /// Without a dedicated compute family, a dedicated transfer family is
    /// still used, and compute work shares the graphics queue.
    #[test]
    fn a_transfer_family_is_used_without_a_compute_family() {
        let device = with_family(
            device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
            vk::QueueFlags::TRANSFER,
        );
        let families = families(device);
        assert_eq!((families.compute, families.transfer), (0, 1));
        assert_eq!(
            families.serving_kind(QueueKind::Compute),
            QueueKind::Graphics
        );
        assert_eq!(
            families.serving_kind(QueueKind::Transfer),
            QueueKind::Transfer
        );
    }

    /// The selection carries the device's capabilities, for its queue
    /// families.
    #[test]
    fn the_selection_carries_capabilities() {
        let mut device = device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB);
        device.features.core.texture_compression_bc = vk::TRUE;
        let selection = select(headless(vec![device]), None).unwrap();
        assert_eq!(selection.capabilities.device_name, "device");
        assert!(selection.capabilities.bc_compression);
    }

    /// A device that matches the filter must still meet the requirements:
    /// the filter never selects a device that would otherwise be rejected.
    #[test]
    fn the_device_filter_doesnt_bypass_requirements() {
        let mut old = device("old", vk::PhysicalDeviceType::DISCRETE_GPU, 16 * GIB);
        old.properties.core.api_version = vk::API_VERSION_1_3;
        let devices = vec![
            old,
            device("new", vk::PhysicalDeviceType::INTEGRATED_GPU, 8 * GIB),
        ];
        assert!(select(headless(devices), Some("old")).is_err());
    }
}
