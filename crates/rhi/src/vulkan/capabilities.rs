//! Deriving the RHI's [`Capabilities`] and [`PresentCapabilities`] from the
//! selected Vulkan device.

use ash::{ext, khr, vk};

use super::physical_device::PhysicalDevice;
use crate::{Capabilities, Limits, PresentCapabilities};

/// The capabilities of `device`. Pure, so it's tested with fake devices.
pub(super) fn derive(device: &PhysicalDevice) -> Capabilities {
    let features = &device.features;
    let limits = device.limits();
    Capabilities {
        device_name: device.name().into_owned(),
        bc_compression: features.core.texture_compression_bc == vk::TRUE,
        limits: Limits {
            max_texture_dimension_2d: limits.max_image_dimension2_d,
            max_compute_workgroup_size: limits.max_compute_work_group_size,
        },
    }
}

/// What presenting can do on `device`. Pure, so it's tested with fake
/// devices.
///
/// An optional extension's capability needs both the extension and its
/// feature, since a driver may advertise one without the other.
pub(super) fn derive_present(device: &PhysicalDevice) -> PresentCapabilities {
    let features = &device.features;
    PresentCapabilities {
        present_timing: device.supports_extension(ext::present_timing::NAME)
            && features.present_timing.present_timing == vk::TRUE,
        swapchain_maintenance: device.supports_extension(khr::swapchain_maintenance1::NAME)
            && features.swapchain_maintenance1.swapchain_maintenance1 == vk::TRUE,
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of deriving capabilities from fake devices.

    use std::ffi::CStr;

    use super::*;
    use crate::vulkan::physical_device::{Features, Properties};

    /// A device named `device` with the extensions `extensions`, no
    /// features and no queue families.
    fn device(extensions: &[&CStr]) -> PhysicalDevice {
        let mut properties = Properties::default();
        for (dst, src) in properties.core.device_name.iter_mut().zip(b"device") {
            *dst = *src as _;
        }
        PhysicalDevice {
            handle: vk::PhysicalDevice::null(),
            properties,
            features: Features::default(),
            memory_properties: vk::PhysicalDeviceMemoryProperties::default(),
            queue_families: Vec::new(),
            extensions: extensions
                .iter()
                .map(|name| {
                    vk::ExtensionProperties::default()
                        .extension_name(name)
                        .unwrap()
                })
                .collect(),
        }
    }

    /// An optional extension's capability needs both the extension and its
    /// feature, so a driver that only advertises it doesn't count.
    #[test]
    fn extension_capabilities_need_the_extension_and_the_feature() {
        let names = [ext::present_timing::NAME, khr::swapchain_maintenance1::NAME];
        let mut with_both = device(&names);
        with_both.features.present_timing.present_timing = vk::TRUE;
        with_both
            .features
            .swapchain_maintenance1
            .swapchain_maintenance1 = vk::TRUE;
        assert_eq!(
            derive_present(&with_both),
            PresentCapabilities {
                present_timing: true,
                swapchain_maintenance: true,
            },
        );

        let none = PresentCapabilities {
            present_timing: false,
            swapchain_maintenance: false,
        };
        assert_eq!(derive_present(&device(&names)), none);

        let mut features_only = device(&[]);
        features_only.features = with_both.features;
        assert_eq!(derive_present(&features_only), none);
    }

    /// BC compression, the name and the limits come from the device's
    /// features and properties.
    #[test]
    fn features_names_and_limits_are_copied() {
        let mut device = device(&[]);
        device.features.core.texture_compression_bc = vk::TRUE;
        let limits = &mut device.properties.core.limits;
        limits.max_image_dimension2_d = 32_768;
        limits.max_compute_work_group_size = [1_024, 1_024, 64];
        assert_eq!(
            derive(&device),
            Capabilities {
                device_name: "device".to_owned(),
                bc_compression: true,
                limits: Limits {
                    max_texture_dimension_2d: 32_768,
                    max_compute_workgroup_size: [1_024, 1_024, 64],
                },
            },
        );
    }
}
