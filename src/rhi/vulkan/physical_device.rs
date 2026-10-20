//! Information about the GPUs (physical devices) available to a Vulkan
//! instance, used to compare them and choose one to render with.

use std::borrow::Cow;
use std::ffi::CStr;
use std::{mem, ptr};

use anyhow::{Context, Result};
use ash::vk::TaggedStructure;
use ash::{Instance, ext, khr, vk};

/// A Vulkan physical device and its immutable properties.
///
/// Everything is queried once, on creation. The version-specific structures
/// (`vulkan11` to `vulkan14`) are only queried if the device supports that
/// Vulkan version, and extension structures only if it supports the
/// extension. Otherwise they are left zeroed; check
/// [`PhysicalDevice::api_version`] or
/// [`PhysicalDevice::supports_extension`] before relying on them.
#[derive(Debug)]
pub struct PhysicalDevice {
    /// The device's handle, valid for as long as the instance it was
    /// enumerated from.
    pub handle: vk::PhysicalDevice,
    /// The device's properties and limits.
    pub properties: Properties,
    /// The features the device supports.
    pub features: Features,
    /// The device's memory heaps and types.
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Indexed by queue family index.
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    /// The device extensions the device supports.
    pub extensions: Vec<vk::ExtensionProperties>,
}

// `PhysicalDevice` must be `Send` and `Sync`, so that the device design 0002
// shares between threads can hold it ("Threading"). It gets both from its
// fields: `ash` declares its structures `Send` and `Sync`, despite their
// `p_next` pointers. This fails to compile if a change loses either.
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PhysicalDevice>();
};

/// Device properties, including limits, grouped by the Vulkan version or
/// extension that introduced them.
///
/// The `p_next` pointers are always null.
#[derive(Debug, Default)]
pub struct Properties {
    /// Vulkan 1.0 properties, including the device name, type and limits.
    pub core: vk::PhysicalDeviceProperties,
    /// Vulkan 1.1 properties. Zeroed on devices older than Vulkan 1.2.
    pub vulkan11: vk::PhysicalDeviceVulkan11Properties<'static>,
    /// Vulkan 1.2 properties. Zeroed on devices older than Vulkan 1.2.
    pub vulkan12: vk::PhysicalDeviceVulkan12Properties<'static>,
    /// Vulkan 1.3 properties. Zeroed on devices older than Vulkan 1.3.
    pub vulkan13: vk::PhysicalDeviceVulkan13Properties<'static>,
    /// Vulkan 1.4 properties. Zeroed on devices older than Vulkan 1.4.
    ///
    /// The image layout lists (`p_copy_src_layouts` and
    /// `p_copy_dst_layouts`) are not queried, so their pointers are null,
    /// but their counts are set.
    pub vulkan14: vk::PhysicalDeviceVulkan14Properties<'static>,
    /// `VK_EXT_descriptor_heap` properties (ADR 0002). Zeroed on devices
    /// without the extension.
    pub descriptor_heap: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
}

/// Device features, grouped by the Vulkan version or extension that
/// introduced them.
///
/// The `p_next` pointers are always null.
#[derive(Debug, Default)]
pub struct Features {
    /// Vulkan 1.0 features.
    pub core: vk::PhysicalDeviceFeatures,
    /// Vulkan 1.1 features. Zeroed on devices older than Vulkan 1.2.
    pub vulkan11: vk::PhysicalDeviceVulkan11Features<'static>,
    /// Vulkan 1.2 features. Zeroed on devices older than Vulkan 1.2.
    pub vulkan12: vk::PhysicalDeviceVulkan12Features<'static>,
    /// Vulkan 1.3 features. Zeroed on devices older than Vulkan 1.3.
    pub vulkan13: vk::PhysicalDeviceVulkan13Features<'static>,
    /// Vulkan 1.4 features. Zeroed on devices older than Vulkan 1.4.
    pub vulkan14: vk::PhysicalDeviceVulkan14Features<'static>,
    /// `VK_EXT_shader_object` features (ADR 0002).
    pub shader_object: vk::PhysicalDeviceShaderObjectFeaturesEXT<'static>,
    /// `VK_EXT_descriptor_heap` features (ADR 0002). Zeroed on devices
    /// without the extension.
    pub descriptor_heap: vk::PhysicalDeviceDescriptorHeapFeaturesEXT<'static>,
    /// `VK_KHR_device_address_commands` features (ADR 0002). Zeroed on
    /// devices without the extension.
    pub device_address_commands: vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR<'static>,
    /// `VK_EXT_present_timing` features, optional in ADR 0002. Zeroed on
    /// devices without the extension.
    pub present_timing: vk::PhysicalDevicePresentTimingFeaturesEXT<'static>,
    /// `VK_KHR_swapchain_maintenance1` features, optional in ADR 0002.
    /// Zeroed on devices without the extension.
    pub swapchain_maintenance1: vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR<'static>,
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
        // SAFETY: `handle` was enumerated from `instance`. The same holds for
        // every query on `handle` below, and every structure passed to them is
        // valid for the call.
        let core_properties = unsafe { instance.get_physical_device_properties(handle) };
        let api_version = core_properties.api_version;
        // Queried first, as extension feature structures may only be chained
        // if the device supports the extension.
        // SAFETY: as above.
        let extensions = unsafe { instance.enumerate_device_extension_properties(handle) }
            .context("failed to enumerate Vulkan device extensions")?;

        let mut properties = Properties {
            core: core_properties,
            ..Default::default()
        };
        let mut features = Features {
            // SAFETY: as above.
            core: unsafe { instance.get_physical_device_features(handle) },
            ..Default::default()
        };

        // The `Vulkan11*` and `Vulkan12*` structures were added in Vulkan 1.2,
        // and may only be chained if the device supports it.
        if api_version >= vk::API_VERSION_1_2 {
            let include_vulkan13 = api_version >= vk::API_VERSION_1_3;
            let include_vulkan14 = api_version >= vk::API_VERSION_1_4;
            let supports = |name| extensions_include(&extensions, name);

            let mut properties2 = vk::PhysicalDeviceProperties2::default()
                .push(&mut properties.vulkan11)
                .push(&mut properties.vulkan12);
            if include_vulkan13 {
                properties2 = properties2.push(&mut properties.vulkan13);
            }
            if include_vulkan14 {
                properties2 = properties2.push(&mut properties.vulkan14);
            }
            if supports(ext::descriptor_heap::NAME) {
                properties2 = properties2.push(&mut properties.descriptor_heap);
            }
            // SAFETY: as above. The chain only includes structures for
            // versions and extensions the device supports.
            unsafe { instance.get_physical_device_properties2(handle, &mut properties2) };
            // SAFETY: The chain was built with `push`, so every structure in
            // it is borrowed by `properties2`.
            unsafe { unchain(&mut properties2) };

            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push(&mut features.vulkan11)
                .push(&mut features.vulkan12);
            if include_vulkan13 {
                features2 = features2.push(&mut features.vulkan13);
            }
            if include_vulkan14 {
                features2 = features2.push(&mut features.vulkan14);
            }
            if supports(ext::shader_object::NAME) {
                features2 = features2.push(&mut features.shader_object);
            }
            if supports(ext::descriptor_heap::NAME) {
                features2 = features2.push(&mut features.descriptor_heap);
            }
            if supports(khr::device_address_commands::NAME) {
                features2 = features2.push(&mut features.device_address_commands);
            }
            if supports(ext::present_timing::NAME) {
                features2 = features2.push(&mut features.present_timing);
            }
            if supports(khr::swapchain_maintenance1::NAME) {
                features2 = features2.push(&mut features.swapchain_maintenance1);
            }
            // SAFETY: as above. The chain only includes structures for
            // versions and extensions the device supports.
            unsafe { instance.get_physical_device_features2(handle, &mut features2) };
            // SAFETY: as for `properties2`.
            unsafe { unchain(&mut features2) };
        }

        // SAFETY: as above.
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(handle) };
        // SAFETY: as above.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(handle) };

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

    /// The total size, in bytes, of the device's device-local memory heaps, or
    /// 0 if it has none.
    ///
    /// On integrated GPUs this is typically shared system memory.
    pub fn device_local_memory(&self) -> vk::DeviceSize {
        self.memory_properties
            .memory_heaps_as_slice()
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum()
    }

    /// The device's limits, such as maximum image sizes.
    pub fn limits(&self) -> &vk::PhysicalDeviceLimits {
        &self.properties.core.limits
    }

    /// Whether the device supports the extension `name`.
    pub fn supports_extension(&self, name: &CStr) -> bool {
        extensions_include(&self.extensions, name)
    }
}

/// Clears the `p_next` pointer of `root` and of every structure chained to
/// it.
///
/// Chained structures point at each other, so their pointers would dangle
/// once the structures are moved. Walking the chain, rather than clearing
/// each structure's pointer by name, means that none can be missed.
///
/// # Safety
///
/// Every structure in the chain must be valid and borrowed by `root`, as
/// [`TaggedStructure::push`] makes it.
unsafe fn unchain<'a, T: TaggedStructure<'a>>(root: &mut T) {
    let mut structure = ptr::from_mut(root).cast::<vk::BaseOutStructure<'_>>();
    while !structure.is_null() {
        // SAFETY: `structure` is either `root`, which `TaggedStructure`
        // guarantees has the layout of `BaseOutStructure`, or a structure in
        // its chain, which the caller guarantees is valid and not otherwise
        // borrowed.
        structure = unsafe { mem::replace(&mut (*structure).p_next, ptr::null_mut()) };
    }
}

/// Whether `extensions` includes the extension `name`.
fn extensions_include(extensions: &[vk::ExtensionProperties], name: &CStr) -> bool {
    extensions
        .iter()
        .any(|extension| extension.extension_name_as_c_str() == Ok(name))
}

#[cfg(test)]
mod tests {
    //! T0 tests of the structure-chain helpers.

    use super::*;

    /// `unchain` clears the pointer of the root and of every structure
    /// chained to it, so that none dangles once they're moved.
    #[test]
    fn unchain_clears_every_pointer_in_the_chain() {
        let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push(&mut vulkan11)
            .push(&mut vulkan12)
            .push(&mut vulkan13);
        // SAFETY: The chain was built with `push`.
        unsafe { unchain(&mut features2) };
        assert!(features2.p_next.is_null());
        assert!(vulkan11.p_next.is_null());
        assert!(vulkan12.p_next.is_null());
        assert!(vulkan13.p_next.is_null());
    }
}
