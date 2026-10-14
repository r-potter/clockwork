//! Vulkan implementation of the render hardware interface.
//!
//! This backend is selected with [`GraphicsApi::Vulkan`] and is only
//! constructed through [`super::create`]; the rest of the application interacts
//! with it solely via the [`RenderHardwareInterface`] trait.
//!
//! It is currently a placeholder: no Vulkan objects are created yet.

use anyhow::Result;

use super::{GraphicsApi, RenderHardwareInterface};

/// Vulkan render hardware interface.
///
/// Will own the Vulkan state needed for rendering, such as the instance,
/// the selected physical device, the logical device and its queues. Holds
/// nothing for now.
#[derive(Debug)]
pub struct VulkanRenderHardwareInterface {}

impl VulkanRenderHardwareInterface {
    /// Initialises the Vulkan backend.
    ///
    /// Returns a `Result` so that failures during Vulkan setup (for example,
    /// no Vulkan driver or no suitable GPU) can be reported to the caller.
    /// This placeholder cannot fail.
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }
}

impl RenderHardwareInterface for VulkanRenderHardwareInterface {
    /// Always [`GraphicsApi::Vulkan`].
    fn api(&self) -> GraphicsApi {
        GraphicsApi::Vulkan
    }
}
