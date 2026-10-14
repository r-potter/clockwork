//! Choosing the physical device to render with.
//!
//! Selection happens in two phases:
//!
//! 1. **Requirements.** Every device that does not meet all of
//!    [`REQUIREMENTS`] is rejected, and the reasons are logged.
//! 2. **Comparison.** The remaining devices are compared in pairs with
//!    [`compare`], keeping the better of each pair, until one is left.

use std::cmp::Ordering;

use anyhow::{Result, bail};
use ash::vk;
use tracing::info;

use super::API_VERSION;
use super::physical_device::PhysicalDevice;

/// A minimum requirement a device must meet to be selected.
struct Requirement {
    /// Describes what is required, for logging when a device fails it.
    description: &'static str,
    is_met: fn(&PhysicalDevice) -> bool,
}

/// Every requirement a device must meet to be selected.
const REQUIREMENTS: &[Requirement] = &[
    Requirement {
        // The loader version check does not guarantee that each device
        // supports the version.
        description: "supports the targeted Vulkan version",
        is_met: |device| device.api_version() >= API_VERSION,
    },
    Requirement {
        description: "has a queue family that supports graphics",
        is_met: |device| {
            device
                .queue_families
                .iter()
                .any(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        },
    },
];

/// Chooses the best of `devices` that meets every requirement.
///
/// # Errors
///
/// Returns an error if no device meets every requirement.
pub fn select(devices: Vec<PhysicalDevice>) -> Result<PhysicalDevice> {
    let selected = devices
        .into_iter()
        .filter(|device| {
            let unmet = unmet_requirements(device);
            if !unmet.is_empty() {
                info!(
                    "rejected {}: does not meet requirements: {}",
                    device.name(),
                    unmet.join(", "),
                );
            }
            unmet.is_empty()
        })
        // Keep the earlier device on ties, so the result does not depend on
        // how ties are broken by the standard library.
        .reduce(|best, device| {
            if compare(&device, &best) == Ordering::Greater {
                device
            } else {
                best
            }
        });

    let Some(selected) = selected else {
        bail!("no Vulkan device meets the minimum requirements; see the log for details");
    };
    info!("selected {}", selected.name());
    Ok(selected)
}

/// Returns the descriptions of the requirements that `device` does not meet.
fn unmet_requirements(device: &PhysicalDevice) -> Vec<&'static str> {
    REQUIREMENTS
        .iter()
        .filter(|requirement| !(requirement.is_met)(device))
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
/// 2. size of the largest device-local memory heap
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

        PhysicalDevice {
            handle: vk::PhysicalDevice::null(),
            properties,
            features: Features::default(),
            memory_properties,
            queue_families: vec![vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS,
                queue_count: 1,
                ..Default::default()
            }],
            extensions: Vec::new(),
        }
    }

    fn selected_name(devices: Vec<PhysicalDevice>) -> String {
        select(devices).unwrap().name().into_owned()
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
    fn compares_the_largest_heap_not_the_total() {
        let mut split = device("split", vk::PhysicalDeviceType::DISCRETE_GPU, 6 * GIB);
        split.memory_properties.memory_heap_count = 2;
        split.memory_properties.memory_heaps[1] = vk::MemoryHeap {
            size: 6 * GIB,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        let devices = vec![
            split,
            device("single", vk::PhysicalDeviceType::DISCRETE_GPU, 8 * GIB),
        ];
        assert_eq!(selected_name(devices), "single");
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
        assert!(select(vec![old]).is_err());
        assert!(select(Vec::new()).is_err());
    }
}
