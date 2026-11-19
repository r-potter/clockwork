//! Presenting to a window: the image a frame renders to, and what presenting
//! it submits (design 0002, "Presentation").
//!
//! [`acquire_surface_image`] hands out a swapchain image as a [`Texture`],
//! and [`present`] submits the command buffers that render to it on the
//! graphics queue, then presents it. The swapchain's own synchronization
//! stays inside the backend: the application only starts the image at
//! [`Access::Acquired`](crate::Access::Acquired) and ends it at
//! [`Access::Present`](crate::Access::Present).
//!
//! [`acquire_surface_image`]: crate::RenderHardwareInterface::acquire_surface_image
//! [`present`]: crate::RenderHardwareInterface::present

use crate::{CommandBuffer, Extent2d, Format, GpuTimepoint, Texture};

/// A swapchain image acquired for a frame to render to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceImage {
    /// The image, which may only be used in the command buffers presented
    /// with it. Its handle may be stale after the next
    /// [`acquire_surface_image`](crate::RenderHardwareInterface::acquire_surface_image),
    /// which can recreate the swapchain, and destroying it is a bug, since
    /// the swapchain owns it.
    ///
    /// Submitting a command buffer that uses it after its swapchain is gone
    /// is undefined behaviour, which isn't checked yet, unlike other misuse
    /// of a handle.
    pub texture: Texture,
    /// Its size, in pixels, which may differ from the window's.
    pub extent: Extent2d,
    /// Its format, an 8-bit sRGB RGBA or BGRA format.
    pub format: Format,
}

/// What [`present`](crate::RenderHardwareInterface::present) submits and
/// presents.
#[derive(Debug)]
pub struct PresentDesc<'a> {
    /// The image to present, which the last
    /// [`acquire_surface_image`](crate::RenderHardwareInterface::acquire_surface_image)
    /// returned.
    pub image: SurfaceImage,
    /// Command buffers translated by the graphics queue, which render to the
    /// image, run in this order. Together they move it from
    /// [`Access::Acquired`](crate::Access::Acquired) to
    /// [`Access::Present`](crate::Access::Present).
    pub command_buffers: Vec<CommandBuffer>,
    /// Timepoints, on any queue, that must be reached before the command
    /// buffers run, such as an upload's.
    pub wait_for: &'a [GpuTimepoint],
}

// Surface images are plain data, and a description crosses threads with its
// command buffers (design 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    /// Compiles only for types that are `Send`.
    const fn assert_send<T: Send>() {}
    assert_send_sync::<SurfaceImage>();
    assert_send::<PresentDesc<'_>>();
};
