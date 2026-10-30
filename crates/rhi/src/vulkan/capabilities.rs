//! Deriving the RHI's [`Capabilities`] and [`PresentCapabilities`] from the
//! selected Vulkan device.

use ash::{ext, khr, vk};

use super::device_selection::QueueFamilies;
use super::physical_device::PhysicalDevice;
use crate::{Capabilities, Limits, PresentCapabilities, QueueKind, QueueKindSet};

/// The capabilities of `device`, when its queues are from
/// `queue_families`. Pure, so it's tested with fake devices.
///
/// A kind's queue is dedicated if it serves itself, and a kind can write
/// timestamps if the family it uses can.
pub(super) fn derive(device: &PhysicalDevice, queue_families: QueueFamilies) -> Capabilities {
    let mut timestamps = QueueKindSet::EMPTY;
    for kind in QueueKind::ALL {
        let family = queue_families.family(kind);
        if device.queue_families[family as usize].timestamp_valid_bits > 0 {
            timestamps = timestamps.with(kind);
        }
    }
    let dedicated = |kind| queue_families.serving_kind(kind) == kind;

    let features = &device.features;
    let limits = device.limits();
    Capabilities {
        device_name: device.name().into_owned(),
        dedicated_compute_queue: dedicated(QueueKind::Compute),
        dedicated_transfer_queue: dedicated(QueueKind::Transfer),
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

    /// The families when every kind uses family 0, headless.
    const FAMILY_0: QueueFamilies = QueueFamilies {
        graphics: 0,
        compute: 0,
        transfer: 0,
        present: None,
    };

    /// When every kind shares the graphics family, no queue is dedicated,
    /// and every kind shares the graphics family's timestamps.
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

    /// Each kind on a family of its own is dedicated, with that family's
    /// timestamps, so one on a family without them can't write them.
    #[test]
    fn dedicated_families_have_their_own_timestamps() {
        let device = device(&[])
            .with_family(vk::QueueFlags::COMPUTE, 64)
            .with_family(vk::QueueFlags::empty(), 0);
        let families = QueueFamilies {
            compute: 1,
            transfer: 2,
            ..FAMILY_0
        };
        let capabilities = derive(&device, families);
        assert!(capabilities.dedicated_compute_queue);
        assert!(capabilities.dedicated_transfer_queue);
        assert_eq!(
            capabilities.timestamps,
            QueueKindSet::EMPTY
                .with(QueueKind::Graphics)
                .with(QueueKind::Compute),
        );
    }

    /// Transfers sharing the dedicated compute family aren't dedicated, and
    /// share its timestamps.
    #[test]
    fn transfers_share_the_compute_family() {
        let device = device(&[]).with_family(vk::QueueFlags::COMPUTE, 0);
        let families = QueueFamilies {
            compute: 1,
            transfer: 1,
            ..FAMILY_0
        };
        let capabilities = derive(&device, families);
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
