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
//! To add a backend:
//! 1. add a variant to [`GraphicsApi`],
//! 2. add a submodule with a type implementing [`RenderHardwareInterface`],
//! 3. add a match arm for it in [`create`].

mod vulkan;

use std::fmt::Debug;

use anyhow::Result;

use vulkan::VulkanRenderHardwareInterface;

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
}

/// Creates the render hardware interface for the given graphics API.
///
/// # Errors
///
/// Returns an error if the backend fails to initialise, for example because
/// the graphics API is not supported on this system.
pub fn create(api: GraphicsApi) -> Result<Box<dyn RenderHardwareInterface>> {
    Ok(match api {
        GraphicsApi::Vulkan => Box::new(VulkanRenderHardwareInterface::new()?),
    })
}
