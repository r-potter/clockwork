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
use super::physical_device::PhysicalDevice;
use super::surface::SurfaceSupport;

/// A device being considered for selection.
pub struct Candidate {
    /// The device, with the properties and features selection compares.
    pub device: PhysicalDevice,
    /// How the device can present to the window's surface, or `None` when
    /// headless.
    pub surface_support: Option<SurfaceSupport>,
}

/// The selected device, and the queue families to use on it.
pub struct Selection {
    /// The device that met every requirement and compared best.
    pub device: PhysicalDevice,
    /// The queue families to create the logical device's queues from.
    pub queue_families: QueueFamilies,
}

/// Indices of the queue families to use on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFamilies {
    /// Supports graphics operations.
    pub graphics: u32,
    /// Can present to the window's surface. `None` when headless. May be the
    /// same family as `graphics`.
    pub present: Option<u32>,
}

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
        description: "has a queue family that supports graphics",
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
pub fn select(candidates: Vec<Candidate>, device_filter: Option<&str>) -> Result<Selection> {
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
    info!(
        "selected {} ({}), with queue families {queue_families:?}",
        selected.device.name(),
        capability_summary(&selected.device),
    );
    Ok(Selection {
        device: selected.device,
        queue_families,
    })
}

/// An optional capability that [`select`] reports for the selected device.
struct ReportedCapability {
    /// The capability's name, for the log.
    name: &'static str,
    /// Returns whether the device has the capability.
    is_supported: fn(&PhysicalDevice) -> bool,
}

/// The optional capabilities that [`select`] reports for the selected device.
///
/// These are the ADR 0002 extensions that [`REQUIREMENTS`] doesn't require,
/// until the code that uses them makes them required (design `0002-rhi`). A
/// feature is only set when the device supports its extension.
const REPORTED_CAPABILITIES: &[ReportedCapability] = &[
    ReportedCapability {
        name: "descriptor heap",
        is_supported: |device| device.features.descriptor_heap.descriptor_heap == vk::TRUE,
    },
    ReportedCapability {
        name: "device address commands",
        is_supported: |device| {
            let features = &device.features.device_address_commands;
            features.device_address_commands == vk::TRUE
        },
    },
    ReportedCapability {
        name: "present timing",
        is_supported: |device| device.features.present_timing.present_timing == vk::TRUE,
    },
    ReportedCapability {
        name: "swapchain maintenance",
        is_supported: |device| {
            let features = &device.features.swapchain_maintenance1;
            features.swapchain_maintenance1 == vk::TRUE
        },
    },
];

/// Summarises which of [`REPORTED_CAPABILITIES`] `device` has and lacks, for
/// the log: for example, `has descriptor heap; lacks present timing`.
fn capability_summary(device: &PhysicalDevice) -> String {
    let mut has = Vec::new();
    let mut lacks = Vec::new();
    for capability in REPORTED_CAPABILITIES {
        if (capability.is_supported)(device) {
            has.push(capability.name);
        } else {
            lacks.push(capability.name);
        }
    }
    match (has.is_empty(), lacks.is_empty()) {
        (false, true) => format!("has {}", has.join(", ")),
        (true, false) => format!("lacks {}", lacks.join(", ")),
        _ => format!("has {}; lacks {}", has.join(", "), lacks.join(", ")),
    }
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
/// Otherwise uses the first family that supports each.
///
/// # Panics
///
/// Panics if `candidate` does not meet the requirements.
fn queue_families(candidate: &Candidate) -> QueueFamilies {
    let Some(support) = &candidate.surface_support else {
        let graphics = graphics_families(&candidate.device).next();
        return QueueFamilies {
            graphics: graphics.expect("requirements guarantee a graphics queue family"),
            present: None,
        };
    };

    let can_present = |family: &u32| support.present_families[*family as usize];
    let (graphics, present) = match graphics_families(&candidate.device).find(can_present) {
        Some(family) => (Some(family), Some(family)),
        None => (
            graphics_families(&candidate.device).next(),
            (0..support.present_families.len() as u32).find(can_present),
        ),
    };
    QueueFamilies {
        graphics: graphics.expect("requirements guarantee a graphics queue family"),
        present: Some(present.expect("requirements guarantee a present queue family")),
    }
}

/// The indices of `device`'s queue families that support graphics.
fn graphics_families(device: &PhysicalDevice) -> impl Iterator<Item = u32> + '_ {
    device
        .queue_families
        .iter()
        .enumerate()
        .filter(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(index, _)| index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::rhi::vulkan::physical_device::{Features, Properties};

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
        features.shader_object.shader_object = vk::TRUE;

        PhysicalDevice {
            handle: vk::PhysicalDevice::null(),
            properties,
            features,
            memory_properties,
            queue_families: vec![vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS,
                queue_count: 1,
                ..Default::default()
            }],
            extensions: vec![
                vk::ExtensionProperties::default()
                    .extension_name(ext::shader_object::NAME)
                    .unwrap(),
            ],
        }
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

    /// Adds a queue family with `flags` to `device`.
    fn with_family(mut device: PhysicalDevice, flags: vk::QueueFlags) -> PhysicalDevice {
        device.queue_families.push(vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: 1,
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

    /// The summary lists every reported capability a device has.
    #[test]
    fn the_summary_lists_the_capabilities_a_device_has() {
        let mut device = device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB);
        let features = &mut device.features;
        features.descriptor_heap.descriptor_heap = vk::TRUE;
        features.device_address_commands.device_address_commands = vk::TRUE;
        features.present_timing.present_timing = vk::TRUE;
        features.swapchain_maintenance1.swapchain_maintenance1 = vk::TRUE;
        assert_eq!(
            capability_summary(&device),
            "has descriptor heap, device address commands, present timing, swapchain maintenance",
        );
    }

    /// The summary lists the capabilities a device lacks too, as the log
    /// contract says an optional extension a device lacks is notable.
    #[test]
    fn the_summary_lists_the_capabilities_a_device_lacks() {
        let mut device = device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB);
        device.features.descriptor_heap.descriptor_heap = vk::TRUE;
        device
            .features
            .device_address_commands
            .device_address_commands = vk::TRUE;
        assert_eq!(
            capability_summary(&device),
            "has descriptor heap, device address commands; \
             lacks present timing, swapchain maintenance",
        );
    }

    /// A device with none of the capabilities lists only what it lacks.
    #[test]
    fn the_summary_of_a_device_without_capabilities() {
        let device = device("device", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB);
        assert_eq!(
            capability_summary(&device),
            "lacks descriptor heap, device address commands, present timing, \
             swapchain maintenance",
        );
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
