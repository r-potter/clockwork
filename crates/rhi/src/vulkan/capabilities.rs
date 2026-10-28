//! Deriving the RHI's [`Capabilities`] and [`PresentCapabilities`] from the
//! selected Vulkan device.

use ash::{ext, khr, vk};

use super::device_selection::{self, QueueFamilies};
use super::physical_device::PhysicalDevice;
use crate::{Capabilities, Limits, PresentCapabilities, QueueKind, QueueKindSet};

/// The capabilities of `device`, when it renders on the graphics family in
/// `queue_families`. Pure, so it's tested with fake devices.
///
/// Compute work uses a dedicated compute family if the device has one, and
/// the graphics family otherwise. Transfers use a dedicated transfer family,
/// then a dedicated compute family, then the graphics family. A queue kind
/// can write timestamps if the family it uses can.
pub(super) fn derive(device: &PhysicalDevice, queue_families: QueueFamilies) -> Capabilities {
    let dedicated_compute = device_selection::dedicated_compute_family(device);
    let dedicated_transfer = device_selection::dedicated_transfer_family(device);
    let graphics = queue_families.graphics;
    let compute = dedicated_compute.unwrap_or(graphics);
    let transfer = dedicated_transfer.or(dedicated_compute).unwrap_or(graphics);

    let has_timestamps =
        |family: u32| device.queue_families[family as usize].timestamp_valid_bits > 0;
    let mut timestamps = QueueKindSet::EMPTY;
    for (kind, family) in [
        (QueueKind::Graphics, graphics),
        (QueueKind::Compute, compute),
        (QueueKind::Transfer, transfer),
    ] {
        if has_timestamps(family) {
            timestamps = timestamps.with(kind);
        }
    }

    let features = &device.features;
    let limits = device.limits();
    Capabilities {
        device_name: device.name().into_owned(),
        dedicated_compute_queue: dedicated_compute.is_some(),
        dedicated_transfer_queue: dedicated_transfer.is_some(),
        timestamps,
        bc_compression: features.core.texture_compression_bc == vk::TRUE,
        limits: Limits {
            max_texture_dimension_2d: limits.max_image_dimension2_d,
            max_compute_workgroup_size: limits.max_compute_work_group_size,
            timestamp_period_ns: limits.timestamp_period,
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

    /// A device named `device` with one graphics family that can write
    /// timestamps, the extensions `extensions` and no features.
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
        .with_family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 64)
    }

    impl PhysicalDevice {
        /// Adds a queue family with `flags` and `timestamp_valid_bits`.
        fn with_family(mut self, flags: vk::QueueFlags, timestamp_valid_bits: u32) -> Self {
            self.queue_families.push(vk::QueueFamilyProperties {
                queue_flags: flags | vk::QueueFlags::TRANSFER,
                queue_count: 1,
                timestamp_valid_bits,
                ..Default::default()
            });
            self
        }
    }

    /// The families when rendering on family 0, headless.
    const FAMILY_0: QueueFamilies = QueueFamilies {
        graphics: 0,
        present: None,
    };

    /// A device with only a graphics family has no dedicated queues, and
    /// every kind shares the graphics family's timestamps.
    #[test]
    fn a_single_family_is_shared_by_every_kind() {
        let capabilities = derive(&device(&[]), FAMILY_0);
        assert!(!capabilities.dedicated_compute_queue);
        assert!(!capabilities.dedicated_transfer_queue);
        assert_eq!(
            capabilities.timestamps,
            QueueKindSet::EMPTY
                .with(QueueKind::Graphics)
                .with(QueueKind::Compute)
                .with(QueueKind::Transfer),
        );
    }

    /// Dedicated families are found, and a kind on a family without
    /// timestamps can't write them.
    #[test]
    fn dedicated_families_have_their_own_timestamps() {
        let device = device(&[])
            .with_family(vk::QueueFlags::COMPUTE, 64)
            .with_family(vk::QueueFlags::empty(), 0);
        let capabilities = derive(&device, FAMILY_0);
        assert!(capabilities.dedicated_compute_queue);
        assert!(capabilities.dedicated_transfer_queue);
        assert_eq!(
            capabilities.timestamps,
            QueueKindSet::EMPTY
                .with(QueueKind::Graphics)
                .with(QueueKind::Compute),
        );
    }

    /// Without a dedicated transfer family, transfers use the dedicated
    /// compute family, and share its timestamps.
    #[test]
    fn transfers_fall_back_to_the_compute_family() {
        let device = device(&[]).with_family(vk::QueueFlags::COMPUTE, 0);
        let capabilities = derive(&device, FAMILY_0);
        assert!(capabilities.dedicated_compute_queue);
        assert!(!capabilities.dedicated_transfer_queue);
        assert_eq!(
            capabilities.timestamps,
            QueueKindSet::EMPTY.with(QueueKind::Graphics)
        );
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
        limits.timestamp_period = 1.5;
        assert_eq!(
            derive(&device, FAMILY_0),
            Capabilities {
                device_name: "device".to_owned(),
                dedicated_compute_queue: false,
                dedicated_transfer_queue: false,
                timestamps: QueueKindSet::EMPTY
                    .with(QueueKind::Graphics)
                    .with(QueueKind::Compute)
                    .with(QueueKind::Transfer),
                bc_compression: true,
                limits: Limits {
                    max_texture_dimension_2d: 32_768,
                    max_compute_workgroup_size: [1_024, 1_024, 64],
                    timestamp_period_ns: 1.5,
                },
            },
        );
    }
}
