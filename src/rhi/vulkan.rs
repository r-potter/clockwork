//! Vulkan implementation of the render hardware interface.
//!
//! This backend is selected with [`GraphicsApi::Vulkan`] and is only
//! constructed through [`super::create`]; the rest of the application interacts
//! with it solely via the [`RenderHardwareInterface`] trait.
//!
//! The Vulkan loader is loaded at runtime rather than linked, so the
//! application still starts on systems without Vulkan; creating this backend
//! fails with an error instead.

use std::fmt;

use anyhow::{Context, Result, bail};
use ash::{Entry, Instance, vk};

use super::{GraphicsApi, RenderHardwareInterface};

/// Name reported to the Vulkan driver for both the application and the engine.
const NAME: &std::ffi::CStr = c"clockwork";

/// Vulkan version the backend targets.
///
/// `ash` 0.38 was generated from the Vulkan 1.3 headers and has no
/// `API_VERSION_1_4` constant, so the version is built by hand. Its bindings
/// also lack the functions and types added to core in 1.4; use the equivalent
/// extension bindings instead.
const API_VERSION: u32 = vk::make_api_version(0, 1, 4, 0);

/// Vulkan render hardware interface.
///
/// Owns the Vulkan instance. Will later also own the selected physical device,
/// the logical device and its queues.
pub struct VulkanRenderHardwareInterface {
    /// Function pointers for the Vulkan loader. Must outlive `instance`, as it
    /// keeps the loader library loaded.
    _entry: Entry,
    instance: Instance,
}

impl VulkanRenderHardwareInterface {
    /// Initialises the Vulkan backend by loading the Vulkan loader and creating
    /// an instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the Vulkan loader cannot be found, if it is older
    /// than [`API_VERSION`], or if instance creation fails (for example,
    /// because no Vulkan driver is installed).
    pub fn new() -> Result<Self> {
        // SAFETY: Loading the Vulkan loader runs its library initialisation
        // code, which we trust to be well behaved.
        let entry = unsafe { Entry::load() }.context("failed to load the Vulkan loader")?;

        // Requesting a newer version than the driver supports does not make
        // instance creation fail, so check the loader's version explicitly.
        // SAFETY: `entry` holds a valid, loaded Vulkan loader.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .context("failed to query the Vulkan loader version")?
            // Vulkan 1.0 loaders do not provide `vkEnumerateInstanceVersion`.
            .unwrap_or(vk::API_VERSION_1_0);
        if loader_version < API_VERSION {
            bail!(
                "Vulkan {} is required, but the Vulkan loader only supports {}",
                format_version(API_VERSION),
                format_version(loader_version),
            );
        }

        let application_info = vk::ApplicationInfo::default()
            .application_name(NAME)
            .engine_name(NAME)
            .api_version(API_VERSION);
        let create_info = vk::InstanceCreateInfo::default().application_info(&application_info);

        // SAFETY: `create_info` and the structures it points to are valid for
        // the duration of the call.
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .context("failed to create Vulkan instance")?;

        Ok(Self {
            _entry: entry,
            instance,
        })
    }
}

/// Formats a Vulkan version number as `major.minor.patch`.
fn format_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version),
    )
}

impl Drop for VulkanRenderHardwareInterface {
    fn drop(&mut self) {
        // SAFETY: The instance was created by `new`, is destroyed exactly once
        // here, and no objects created from it outlive it.
        unsafe { self.instance.destroy_instance(None) };
    }
}

impl fmt::Debug for VulkanRenderHardwareInterface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanRenderHardwareInterface")
            .field("instance", &self.instance.handle())
            .finish_non_exhaustive()
    }
}

impl RenderHardwareInterface for VulkanRenderHardwareInterface {
    /// Always [`GraphicsApi::Vulkan`].
    fn api(&self) -> GraphicsApi {
        GraphicsApi::Vulkan
    }
}
