//! Vulkan surfaces, which connect the backend to a window.

use std::sync::Arc;

use anyhow::{Context, Result};
use ash::{Entry, Instance, khr, vk};

use crate::rhi::Window;

/// A Vulkan surface for a window.
///
/// Keeps the window alive, as the surface must not outlive it. The surface is
/// destroyed when this is dropped, so the owner must drop it before the
/// instance it was created from.
pub struct Surface {
    surface_fn: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    _window: Arc<dyn Window>,
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

        // SAFETY: The handles are valid for as long as `window` exists, and
        // `self` keeps it alive for as long as the surface exists. The caller
        // guarantees that the required extensions are enabled on `instance`.
        let surface = unsafe {
            ash_window::create_surface(entry, instance, display_handle, window_handle, None)
        }
        .context("failed to create Vulkan surface")?;

        Ok(Self {
            surface_fn: khr::surface::Instance::new(entry, instance),
            surface,
            _window: window,
        })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: The surface was created by `new`, is destroyed exactly once
        // here, and the owner guarantees that its instance still exists.
        unsafe { self.surface_fn.destroy_surface(self.surface, None) };
    }
}
