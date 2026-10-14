//! Render hardware interface: an abstraction over the underlying graphics API.
//!
//! Vulkan is currently the only supported backend, but the interface is kept
//! API-agnostic so that others (e.g. Metal, DirectX) can be added later.
//!
//! The application obtains a backend by calling [`create`] with the desired
//! [`GraphicsApi`], and from then on uses it only through the
//! [`RenderHardwareInterface`] trait. Concrete backend types are private to
//! this module.
//!
//! A backend either renders to a [`Window`], or runs headless with no window
//! (for example, in tests and on CI machines without a display). A windowed
//! backend can be suspended, releasing its window, and later resumed with a
//! window again, without recreating the backend.
//!
//! To add a backend:
//! 1. add a variant to [`GraphicsApi`],
//! 2. add a submodule with a type implementing [`RenderHardwareInterface`],
//! 3. add a match arm for it in [`create`].

mod vulkan;

use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use vulkan::VulkanRenderHardwareInterface;

/// A window a backend can render to.
///
/// Implemented for any type that provides `raw-window-handle` display and
/// window handles, such as `winit::window::Window`, so this module does not
/// depend on a particular windowing library.
pub trait Window: HasDisplayHandle + HasWindowHandle {}

impl<T: HasDisplayHandle + HasWindowHandle + ?Sized> Window for T {}

/// Graphics API used by the render hardware interface.
///
/// Passed to [`create`] to choose a backend, and returned by
/// [`RenderHardwareInterface::api`] to identify one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsApi {
    /// The Vulkan API, implemented by [`VulkanRenderHardwareInterface`].
    Vulkan,
}

/// Operations every graphics API backend must provide.
///
/// Used as a trait object, so it must stay object-safe: no generic methods,
/// and any method returning `Self` (such as a constructor) needs a
/// `where Self: Sized` bound. GPU resources should therefore be exposed as
/// opaque handles rather than as backend-specific associated types.
pub trait RenderHardwareInterface: Debug {
    /// Returns the graphics API this backend implements.
    fn api(&self) -> GraphicsApi;

    /// Returns whether this backend was created without a window.
    ///
    /// A headless backend can never render to a window.
    fn is_headless(&self) -> bool;

    /// Returns whether this backend is suspended: created with a window, but
    /// not currently able to render to one. Always `false` when headless.
    fn is_suspended(&self) -> bool;

    /// Releases the window and everything that depends on it, such as the
    /// window surface, while keeping all other resources.
    ///
    /// Must be called when the window becomes unavailable, for example when
    /// the application is suspended on Android. Does nothing if already
    /// suspended, or if headless.
    fn suspend(&mut self);

    /// Starts rendering to `window` again after [`suspend`](Self::suspend).
    ///
    /// `window` must be on the same display as the window the backend was
    /// created with. If the backend is not suspended, its current window is
    /// replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend is headless, or if the window's
    /// resources (such as its surface) cannot be created.
    fn resume(&mut self, window: Arc<dyn Window>) -> Result<()>;
}

/// Creates the render hardware interface for the given graphics API.
///
/// If `window` is `Some`, the backend renders to it, and keeps it alive until
/// it is suspended or dropped. If it is `None`, the backend runs headless.
///
/// # Errors
///
/// Returns an error if the backend fails to initialise, for example because
/// the graphics API is not supported on this system.
pub fn create(
    api: GraphicsApi,
    window: Option<Arc<dyn Window>>,
) -> Result<Box<dyn RenderHardwareInterface>> {
    Ok(match api {
        GraphicsApi::Vulkan => Box::new(VulkanRenderHardwareInterface::new(window)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every supported graphics API.
    const ALL_GRAPHICS_APIS: &[GraphicsApi] = &[GraphicsApi::Vulkan];

    // Fails to compile when a variant is added to `GraphicsApi`, as a reminder
    // to add it to `ALL_GRAPHICS_APIS`.
    const _: () = match GraphicsApi::Vulkan {
        GraphicsApi::Vulkan => (),
    };

    /// Runs headless, so it does not need a display.
    #[test]
    fn create_succeeds_for_every_graphics_api() {
        for &api in ALL_GRAPHICS_APIS {
            let rhi = create(api, None)
                .unwrap_or_else(|err| panic!("failed to create {api:?} backend: {err:?}"));
            assert_eq!(rhi.api(), api);
            assert!(rhi.is_headless());
            assert!(!rhi.is_suspended());
        }
    }

    /// A window whose handles are never available. Enough to check that a
    /// headless backend rejects windows without using them.
    struct UnavailableWindow;

    impl HasDisplayHandle for UnavailableWindow {
        fn display_handle(
            &self,
        ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            Err(raw_window_handle::HandleError::Unavailable)
        }
    }

    impl HasWindowHandle for UnavailableWindow {
        fn window_handle(
            &self,
        ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            Err(raw_window_handle::HandleError::Unavailable)
        }
    }

    #[test]
    fn headless_backends_ignore_suspend_and_reject_resume() {
        for &api in ALL_GRAPHICS_APIS {
            let mut rhi = create(api, None)
                .unwrap_or_else(|err| panic!("failed to create {api:?} backend: {err:?}"));
            rhi.suspend();
            assert!(rhi.is_headless());
            assert!(!rhi.is_suspended());
            assert!(rhi.resume(Arc::new(UnavailableWindow)).is_err());
            assert!(rhi.is_headless());
        }
    }
}
