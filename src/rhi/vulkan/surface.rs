//! Vulkan surfaces, which connect the backend to a window.

use std::sync::Arc;

use anyhow::{Context, Result};
use ash::{Entry, Instance, khr, vk};

use super::physical_device::PhysicalDevice;
use crate::rhi::Window;

/// A Vulkan surface for a window.
///
/// Keeps the window alive, as the surface must not outlive it. The surface is
/// destroyed when this is dropped, so the owner must drop it before the
/// instance it was created from, and after any swapchain created for it.
pub struct Surface {
    /// The `VK_KHR_surface` functions for the instance.
    surface_fn: khr::surface::Instance,
    /// The surface, created for `_window`.
    surface: vk::SurfaceKHR,
    /// The window the surface presents to, kept alive for as long as the
    /// surface.
    _window: Arc<dyn Window>,
}

/// How a physical device can present to a surface.
#[derive(Debug, Clone, Default)]
pub struct SurfaceSupport {
    /// Whether each queue family can present to the surface, indexed by queue
    /// family index.
    pub present_families: Vec<bool>,
    /// The formats the device supports for swapchains on the surface.
    pub formats: Vec<vk::SurfaceFormatKHR>,
    /// The present modes the device supports for swapchains on the surface.
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl Surface {
    /// Creates a surface for `window`.
    ///
    /// `instance` must have been created with the extensions returned by
    /// `ash_window::enumerate_required_extensions` for the window's display.
    ///
    /// # Errors
    ///
    /// Returns an error if the window's handles are unavailable, or if
    /// surface creation fails (for example, because the window system is not
    /// supported).
    pub fn new(entry: &Entry, instance: &Instance, window: Arc<dyn Window>) -> Result<Self> {
        let display_handle = window
            .display_handle()
            .context("failed to get the window's display handle")?
            .as_raw();
        let window_handle = window
            .window_handle()
            .context("failed to get the window's window handle")?
            .as_raw();

        let factory = ash_window::SurfaceFactory::new(entry, instance, display_handle)
            .context("failed to create Vulkan surface")?;
        // SAFETY: The handles are valid for as long as `window` exists, and
        // `self` keeps it alive for as long as the surface exists. The caller
        // guarantees that the required extensions are enabled on `instance`,
        // and `window_handle` belongs to the display `factory` was made for.
        let surface = unsafe { factory.create_surface(window_handle, None) }
            .context("failed to create Vulkan surface")?;

        Ok(Self {
            surface_fn: khr::surface::Instance::load(entry, instance),
            surface,
            _window: window,
        })
    }

    /// The surface handle.
    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface
    }

    /// Queries how `device` can present to this surface.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the queries fail.
    pub fn support(&self, device: &PhysicalDevice) -> Result<SurfaceSupport> {
        let present_families = (0..device.queue_families.len())
            .map(|family| self.can_present(device.handle, family as u32))
            .collect::<Result<_>>()?;
        Ok(SurfaceSupport {
            present_families,
            formats: self.formats(device.handle)?,
            present_modes: self.present_modes(device.handle)?,
        })
    }

    /// Returns whether queue family `family` of `device` can present to this
    /// surface.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn can_present(&self, device: vk::PhysicalDevice, family: u32) -> Result<bool> {
        // SAFETY: `device` and the surface were created from the same
        // instance, which is still alive.
        unsafe {
            self.surface_fn
                .get_physical_device_surface_support(device, family, self.surface)
        }
        .context("failed to query Vulkan surface present support")
    }

    /// The formats `device` supports for swapchains on this surface.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn formats(&self, device: vk::PhysicalDevice) -> Result<Vec<vk::SurfaceFormatKHR>> {
        // SAFETY: as in `can_present`.
        unsafe {
            self.surface_fn
                .get_physical_device_surface_formats(device, self.surface)
        }
        .context("failed to query Vulkan surface formats")
    }

    /// The present modes `device` supports for swapchains on this surface.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn present_modes(&self, device: vk::PhysicalDevice) -> Result<Vec<vk::PresentModeKHR>> {
        // SAFETY: as in `can_present`.
        unsafe {
            self.surface_fn
                .get_physical_device_surface_present_modes(device, self.surface)
        }
        .context("failed to query Vulkan surface present modes")
    }

    /// The surface's current capabilities on `device`, such as its size and
    /// supported image counts. These change as the window does, so query
    /// them whenever a swapchain is created.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn capabilities(&self, device: vk::PhysicalDevice) -> Result<vk::SurfaceCapabilitiesKHR> {
        // SAFETY: as in `can_present`.
        unsafe {
            self.surface_fn
                .get_physical_device_surface_capabilities(device, self.surface)
        }
        .context("failed to query Vulkan surface capabilities")
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: The surface was created by `new`, is destroyed exactly once
        // here, and the owner guarantees that its instance still exists and
        // that no swapchain still uses it.
        unsafe { self.surface_fn.destroy_surface(self.surface, None) };
    }
}
