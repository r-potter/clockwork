//! The Vulkan logical device and its queues.

use anyhow::{Context, Result};
use ash::vk::TaggedStructure;
use ash::{Instance, ext, khr, vk};
use tracing::debug;

use super::device_selection::QueueFamilies;
use super::physical_device::PhysicalDevice;
use crate::ffi::truncate_to_cstring;

/// A logical device, created from the selected physical device, and the
/// queues retrieved from it.
///
/// The device is destroyed when this is dropped, so the owner must drop it
/// after every object created from it, and before the instance.
pub(super) struct LogicalDevice {
    /// The device's function pointers and handle.
    device: ash::Device,
    /// `VK_EXT_shader_object` functions, for creating, binding and drawing
    /// with shader objects.
    shader_object: ext::shader_object::Device,
    /// `VK_EXT_descriptor_heap` functions, for writing descriptors into the
    /// heaps.
    descriptor_heap: ext::descriptor_heap::Device,
    /// `VK_KHR_device_address_commands` functions, for copies and fills by
    /// address.
    device_address_commands: khr::device_address_commands::Device,
    /// `VK_EXT_debug_utils` functions, for naming objects after their
    /// labels, or `None` if the extension isn't enabled.
    debug_utils: Option<ext::debug_utils::Device>,
    /// The families the queues below were created from.
    queue_families: QueueFamilies,
    /// One queue from each distinct family in `queue_families`, with its
    /// family. They're used through [`Queues`](super::queues::Queues), which
    /// locks them.
    queues: Vec<(u32, vk::Queue)>,
}

impl LogicalDevice {
    /// Creates a logical device on `physical_device`, with one queue from each
    /// distinct family in `queue_families`.
    ///
    /// Enables the extensions and features device selection requires:
    /// - `VK_EXT_shader_object`, `VK_EXT_descriptor_heap` and
    ///   `VK_KHR_device_address_commands`, each with its main feature
    /// - Vulkan 1.1's `shader_draw_parameters`
    /// - Vulkan 1.2's `buffer_device_address`, `timeline_semaphore` and
    ///   `host_query_reset`
    /// - Vulkan 1.3's `dynamic_rendering` and `synchronization2`
    ///
    /// Objects on it are named if `debug_utils`, meaning `VK_EXT_debug_utils`
    /// is enabled on `instance` (see [`LogicalDevice::set_object_name`]).
    ///
    /// If `queue_families` has a present family, `VK_KHR_swapchain` is also
    /// enabled. `texture_compression_bc` is enabled where the device supports
    /// it, since it's optional (design 0002, "Data structures and layout").
    ///
    /// # Errors
    ///
    /// Returns an error if device creation fails.
    pub(super) fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        queue_families: QueueFamilies,
        debug_utils: bool,
    ) -> Result<Self> {
        // Vulkan forbids requesting the same family twice, so request each
        // family once, however many kinds use it.
        let mut families = vec![
            queue_families.graphics,
            queue_families.compute,
            queue_families.transfer,
        ];
        families.extend(queue_families.present);
        families.sort_unstable();
        families.dedup();
        let priorities = [1.0];
        let queue_create_infos: Vec<_> = families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities)
            })
            .collect();

        let mut extensions = vec![
            ext::shader_object::NAME.as_ptr(),
            ext::descriptor_heap::NAME.as_ptr(),
            khr::device_address_commands::NAME.as_ptr(),
        ];
        if queue_families.present.is_some() {
            extensions.push(khr::swapchain::NAME.as_ptr());
        }

        // Slang compiles `SV_VertexID` using the base vertex, which needs
        // `shader_draw_parameters`.
        let mut vulkan11_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        // Buffers are used through their device addresses, queues are
        // synchronized with timeline semaphores, and timestamp queries are
        // reset from the host (design 0002).
        let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .host_query_reset(true);
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut shader_object_features =
            vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(true);
        let mut descriptor_heap_features =
            vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default().descriptor_heap(true);
        let mut device_address_commands_features =
            vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR::default()
                .device_address_commands(true);
        let core_features = vk::PhysicalDeviceFeatures::default().texture_compression_bc(
            physical_device.features.core.texture_compression_bc == vk::TRUE,
        );
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&core_features)
            .push(&mut vulkan11_features)
            .push(&mut vulkan12_features)
            .push(&mut vulkan13_features)
            .push(&mut shader_object_features)
            .push(&mut descriptor_heap_features)
            .push(&mut device_address_commands_features);
        // SAFETY: `physical_device` was enumerated from `instance`, and
        // `create_info` and the structures it points to are valid for the
        // duration of the call. Selection checked that the queue families and
        // extensions are supported.
        let device = unsafe { instance.create_device(physical_device.handle, &create_info, None) }
            .context("failed to create Vulkan logical device")?;

        let queues = families
            .into_iter()
            // SAFETY: One queue was requested from each of these families.
            .map(|family| (family, unsafe { device.get_device_queue(family, 0) }))
            .collect();

        Ok(Self {
            shader_object: ext::shader_object::Device::load(instance, &device),
            descriptor_heap: ext::descriptor_heap::Device::load(instance, &device),
            device_address_commands: khr::device_address_commands::Device::load(instance, &device),
            debug_utils: debug_utils.then(|| ext::debug_utils::Device::load(instance, &device)),
            device,
            queue_families,
            queues,
        })
    }

    /// The device's function pointers and handle.
    pub(super) fn handle(&self) -> &ash::Device {
        &self.device
    }

    /// The `VK_EXT_shader_object` functions.
    pub(super) fn shader_object(&self) -> &ext::shader_object::Device {
        &self.shader_object
    }

    /// The `VK_EXT_descriptor_heap` functions.
    pub(super) fn descriptor_heap(&self) -> &ext::descriptor_heap::Device {
        &self.descriptor_heap
    }

    /// The `VK_KHR_device_address_commands` functions.
    pub(super) fn device_address_commands(&self) -> &khr::device_address_commands::Device {
        &self.device_address_commands
    }

    /// The `VK_EXT_debug_utils` functions, or `None` if the extension isn't
    /// enabled, for debug labels in command buffers.
    pub(super) fn debug_utils(&self) -> Option<&ext::debug_utils::Device> {
        self.debug_utils.as_ref()
    }

    /// Names `object`, a live object created on this device, after `label`,
    /// for capture tools such as RenderDoc and Nsight, and for validation
    /// messages (design 0002, "Handles"). Does nothing without
    /// `VK_EXT_debug_utils`, when labels are still used in panics and logs.
    ///
    /// Buffers, images and the descriptor heaps' buffers are named. Memory
    /// isn't: a pooled block holds many resources, and a dedicated
    /// allocation is found through the resource bound to it.
    ///
    /// Failing to set a name only loses the name, so it's logged at `DEBUG`
    /// and otherwise ignored.
    pub(super) fn set_object_name<T: vk::Handle>(&self, object: T, label: &str) {
        let Some(debug_utils) = &self.debug_utils else {
            return;
        };
        let name = truncate_to_cstring(label);
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(object)
            .object_name(&name);
        // SAFETY: `name_info` is valid for the call, `debug_utils` was loaded
        // for this device with `VK_EXT_debug_utils` enabled on its instance,
        // and the caller passes a live object created on this device.
        if let Err(result) = unsafe { debug_utils.set_debug_utils_object_name(&name_info) } {
            debug!("failed to name {:?} \"{label}\": {result:?}", T::TYPE);
        }
    }

    /// The queue families the device's queues were created from.
    pub(super) fn queue_families(&self) -> QueueFamilies {
        self.queue_families
    }

    /// The queue created from `family`.
    ///
    /// # Panics
    ///
    /// Panics if no queue was created from `family`.
    pub(super) fn queue(&self, family: u32) -> vk::Queue {
        self.queues
            .iter()
            .find(|&&(queue_family, _)| queue_family == family)
            .map(|&(_, queue)| queue)
            .unwrap_or_else(|| panic!("no queue was created from family {family}"))
    }
}

impl Drop for LogicalDevice {
    fn drop(&mut self) {
        // SAFETY: The device was created by `new` and is destroyed exactly
        // once here. Waiting for it to be idle ensures no submitted work still
        // uses it, and the owner guarantees that every object created from it
        // has been destroyed.
        unsafe {
            // Nothing useful can be done if this fails while dropping.
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}
