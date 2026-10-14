//! Information about the GPUs (physical devices) available to a Vulkan
//! instance, used to compare them and choose one to render with.

use std::borrow::Cow;
use std::ffi::CStr;
use std::ptr;

use anyhow::{Context, Result};
use ash::{Instance, vk};

/// A Vulkan physical device and its immutable properties.
///
/// Everything is queried once, on creation. The version-specific structures
/// (`vulkan11`, `vulkan12`, `vulkan13`) are only queried if the device
/// supports that Vulkan version, and are left zeroed otherwise; check
/// [`PhysicalDevice::api_version`] before relying on them.
///
/// `ash` 0.38 has no Vulkan 1.4 structures, so 1.4 features and properties are
/// not included.
#[derive(Debug)]
pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub properties: Properties,
    pub features: Features,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Indexed by queue family index.
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    pub extensions: Vec<vk::ExtensionProperties>,
}

/// Device properties, including limits, grouped by the Vulkan version that
/// introduced them.
///
/// The `p_next` pointers are always null.
#[derive(Debug, Default)]
pub struct Properties {
    /// Vulkan 1.0 properties, including the device name, type and limits.
    pub core: vk::PhysicalDeviceProperties,
    pub vulkan11: vk::PhysicalDeviceVulkan11Properties<'static>,
    pub vulkan12: vk::PhysicalDeviceVulkan12Properties<'static>,
    pub vulkan13: vk::PhysicalDeviceVulkan13Properties<'static>,
}

/// Device features, grouped by the Vulkan version that introduced them.
///
/// The `p_next` pointers are always null.
#[derive(Debug, Default)]
#[expect(dead_code, reason = "fields will be read by device selection")]
pub struct Features {
    /// Vulkan 1.0 features.
    pub core: vk::PhysicalDeviceFeatures,
    pub vulkan11: vk::PhysicalDeviceVulkan11Features<'static>,
    pub vulkan12: vk::PhysicalDeviceVulkan12Features<'static>,
    pub vulkan13: vk::PhysicalDeviceVulkan13Features<'static>,
}

impl PhysicalDevice {
    /// Queries every physical device available to `instance`.
    ///
    /// # Errors
    ///
    /// Returns an error if the devices or their extensions cannot be
    /// enumerated.
    pub fn enumerate(instance: &Instance) -> Result<Vec<Self>> {
        // SAFETY: `instance` is a valid Vulkan instance.
        let handles = unsafe { instance.enumerate_physical_devices() }
            .context("failed to enumerate Vulkan physical devices")?;
        handles
            .into_iter()
            .map(|handle| Self::new(instance, handle))
            .collect()
    }

    /// Queries the properties of the physical device `handle`.
    ///
    /// # Errors
    ///
    /// Returns an error if the device's extensions cannot be enumerated.
    pub fn new(instance: &Instance, handle: vk::PhysicalDevice) -> Result<Self> {
        // SAFETY (all `unsafe` blocks below): `handle` was enumerated from
        // `instance`, and every structure passed in is valid for the call.
        let core_properties = unsafe { instance.get_physical_device_properties(handle) };
        let api_version = core_properties.api_version;

        let mut properties = Properties {
            core: core_properties,
            ..Default::default()
        };
        let mut features = Features {
            core: unsafe { instance.get_physical_device_features(handle) },
            ..Default::default()
        };

        // The `Vulkan11*` and `Vulkan12*` structures were added in Vulkan 1.2,
        // and may only be chained if the device supports it.
        if api_version >= vk::API_VERSION_1_2 {
            let include_vulkan13 = api_version >= vk::API_VERSION_1_3;

            let mut properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut properties.vulkan11)
                .push_next(&mut properties.vulkan12);
            if include_vulkan13 {
                properties2 = properties2.push_next(&mut properties.vulkan13);
            }
            unsafe { instance.get_physical_device_properties2(handle, &mut properties2) };

            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut features.vulkan11)
                .push_next(&mut features.vulkan12);
            if include_vulkan13 {
                features2 = features2.push_next(&mut features.vulkan13);
            }
            unsafe { instance.get_physical_device_features2(handle, &mut features2) };

            // The chained structures point at each other; clear the pointers
            // so that none dangle once they are moved into `self`.
            properties.vulkan11.p_next = ptr::null_mut();
            properties.vulkan12.p_next = ptr::null_mut();
            properties.vulkan13.p_next = ptr::null_mut();
            features.vulkan11.p_next = ptr::null_mut();
            features.vulkan12.p_next = ptr::null_mut();
            features.vulkan13.p_next = ptr::null_mut();
        }

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(handle) };
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(handle) };
        let extensions = unsafe { instance.enumerate_device_extension_properties(handle) }
            .context("failed to enumerate Vulkan device extensions")?;

        Ok(Self {
            handle,
            properties,
            features,
            memory_properties,
            queue_families,
            extensions,
        })
    }

    /// The device's name, as reported by the driver.
    pub fn name(&self) -> Cow<'_, str> {
        self.properties
            .core
            .device_name_as_c_str()
            .map_or("<invalid name>".into(), CStr::to_string_lossy)
    }

    /// The device type, such as discrete or integrated GPU.
    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.properties.core.device_type
    }

    /// The highest Vulkan version the device supports.
    pub fn api_version(&self) -> u32 {
        self.properties.core.api_version
    }

    /// The size, in bytes, of the device's largest device-local memory heap, or
    /// 0 if it has none.
    ///
    /// On integrated GPUs this is typically shared system memory.
    pub fn device_local_memory(&self) -> vk::DeviceSize {
        self.memory_properties
            .memory_heaps_as_slice()
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .max()
            .unwrap_or(0)
    }

    /// The device's limits, such as maximum image sizes.
    #[expect(dead_code, reason = "will be used by device selection")]
    pub fn limits(&self) -> &vk::PhysicalDeviceLimits {
        &self.properties.core.limits
    }

    /// Whether the device supports the extension `name`.
    pub fn supports_extension(&self, name: &CStr) -> bool {
        self.extensions
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
    }
}
