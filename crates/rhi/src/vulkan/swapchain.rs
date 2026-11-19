//! Swapchains, which hold the images presented to a window.
//!
//! Each image is a texture in the core's texture table (design 0002,
//! "Presentation"), inserted when the swapchain is created and removed
//! before it's destroyed.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::VkResult;
use ash::{khr, vk};
use tracing::info;

use super::VulkanDevice;
use super::surface::Surface;
use crate::{Extent3d, Format, Texture, WindowSize};

/// The formats swapchain images may have, in order of preference, all in
/// the sRGB colour space.
///
/// Only sRGB formats, so that shaders write linear colour and have it encoded
/// for display: a UNORM format would show it unencoded, so too dark. The
/// desktop drivers the engine targets are expected to offer one of these,
/// and device selection rejects a device whose surface doesn't. Each is a
/// [`Format`], so that the images can be textures.
const PREFERRED_FORMATS: [Format; 2] = [Format::Bgra8Srgb, Format::Rgba8Srgb];

/// Present mode for the swapchain.
///
/// FIFO waits for vertical blank, so it limits the frame rate to the
/// display's refresh rate, and every device that can present supports it.
const PRESENT_MODE: vk::PresentModeKHR = vk::PresentModeKHR::FIFO;

/// How the swapchain's images are composited with other windows, in order of
/// preference. Desktop platforms usually support `OPAQUE`; Android often only
/// supports `INHERIT`.
const PREFERRED_COMPOSITE_ALPHA: &[vk::CompositeAlphaFlagsKHR] = &[
    vk::CompositeAlphaFlagsKHR::OPAQUE,
    vk::CompositeAlphaFlagsKHR::INHERIT,
    vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
    vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
];

/// The first of [`PREFERRED_FORMATS`] that a surface supporting `formats`
/// supports, and the surface format to create a swapchain with for it, or
/// `None` if it supports none of them.
pub(super) fn choose_format(
    formats: &[vk::SurfaceFormatKHR],
) -> Option<(Format, vk::SurfaceFormatKHR)> {
    PREFERRED_FORMATS.into_iter().find_map(|format| {
        let surface_format = vk::SurfaceFormatKHR {
            format: vk::Format::from(format),
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        formats
            .contains(&surface_format)
            .then_some((format, surface_format))
    })
}

/// A swapchain for a surface, its images as textures, a view of each image,
/// and a semaphore for presenting each image.
///
/// Destroyed when dropped, so the owner must drop it before the surface it
/// was created from.
pub(super) struct Swapchain {
    /// The core, whose texture table holds the images, and whose device
    /// destroys the image views and semaphores.
    core: Arc<VulkanDevice>,
    /// The `VK_KHR_swapchain` functions for the device.
    swapchain_fn: khr::swapchain::Device,
    /// The swapchain, created with `swapchain_fn`.
    swapchain: vk::SwapchainKHR,
    /// The swapchain's images, owned by the swapchain.
    images: Vec<vk::Image>,
    /// Each image in `images`, at the same index, as a texture.
    textures: Vec<Texture>,
    /// A view of each image in `images`, at the same index.
    image_views: Vec<vk::ImageView>,
    /// Signalled when rendering to each image has finished, and waited on to
    /// present it. Indexed by image rather than by frame in flight: a present
    /// may still be waiting on its semaphore after the frame's fence has
    /// signalled, but not once the same image has been acquired again.
    render_finished: Vec<vk::Semaphore>,
    /// The images' format.
    format: Format,
    /// The images' size, in pixels.
    extent: vk::Extent2D,
    /// The window size the swapchain was created for. May differ from
    /// `extent`, as some surfaces choose their own size.
    window_size: WindowSize,
}

impl Swapchain {
    /// Creates a swapchain for `surface`, sized to fit `window_size` where the
    /// surface leaves the size to the application (as on Wayland).
    ///
    /// If `old` is given, its resources may be reused by the new swapchain.
    /// `old` must still be dropped afterwards.
    ///
    /// Returns `None` if the surface currently has no area (for example,
    /// because the window is minimised), as no swapchain can be created for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the surface cannot be queried, supports none of
    /// [`PREFERRED_FORMATS`], or if the swapchain or its image views cannot
    /// be created.
    pub(super) fn new(
        core: &Arc<VulkanDevice>,
        surface: &Surface,
        window_size: WindowSize,
        old: Option<&Swapchain>,
    ) -> Result<Option<Self>> {
        let device = &core.device;
        let physical_device = &core.physical_device;
        let capabilities = surface.capabilities(physical_device.handle)?;
        // A current extent of `u32::MAX` means the surface takes its size from
        // the swapchain.
        let extent = if capabilities.current_extent.width == u32::MAX {
            // Check before clamping, as the minimum extent is usually 1x1.
            if window_size.width == 0 || window_size.height == 0 {
                return Ok(None);
            }
            vk::Extent2D {
                width: window_size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window_size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };
        if extent.width == 0 || extent.height == 0 {
            return Ok(None);
        }

        let formats = surface.formats(physical_device.handle)?;
        let Some((format, surface_format)) = choose_format(&formats) else {
            bail!(
                "the window surface supports none of the formats swapchain images may have \
                 ({PREFERRED_FORMATS:?} in the sRGB colour space), only {formats:?}"
            );
        };

        // One more than the minimum, so that the application does not have to
        // wait for the driver to release an image before acquiring the next.
        // A maximum of 0 means there is no maximum.
        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }

        let composite_alpha = PREFERRED_COMPOSITE_ALPHA
            .iter()
            .copied()
            .find(|&alpha| capabilities.supported_composite_alpha.contains(alpha))
            .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);

        let mut create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(PRESENT_MODE)
            .clipped(true)
            .old_swapchain(old.map_or(vk::SwapchainKHR::null(), |old| old.swapchain));
        // Images used by two queue families must be shared between them.
        let queue_families = device.queue_families();
        let shared_families;
        if let Some(present) = queue_families.present
            && present != queue_families.graphics
        {
            shared_families = [queue_families.graphics, present];
            create_info = create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&shared_families);
        }

        let swapchain_fn = khr::swapchain::Device::load(&core.instance.instance, device.handle());
        // SAFETY: `create_info` and the structures it points to are valid for
        // the duration of the call, and its surface and old swapchain belong to
        // this device's instance and device.
        let swapchain = unsafe { swapchain_fn.create_swapchain(&create_info, None) }
            .context("failed to create Vulkan swapchain")?;

        // Construct `Self` before creating the image views, so that the
        // swapchain and any views already created are destroyed by `Drop` if a
        // later step fails.
        let mut this = Self {
            core: core.clone(),
            swapchain_fn,
            swapchain,
            images: Vec::new(),
            textures: Vec::new(),
            image_views: Vec::new(),
            render_finished: Vec::new(),
            format,
            extent,
            window_size,
        };
        let device = device.handle();
        // SAFETY: `swapchain` was just created from `swapchain_fn`.
        this.images = unsafe { this.swapchain_fn.get_swapchain_images(swapchain) }
            .context("failed to get Vulkan swapchain images")?;
        for &image in &this.images {
            let view_create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            // SAFETY: `image` belongs to the swapchain, which was created on
            // this device, and `view_create_info` is valid for the duration of
            // the call.
            let view = unsafe { device.create_image_view(&view_create_info, None) }
                .context("failed to create Vulkan swapchain image view")?;
            this.image_views.push(view);

            // SAFETY: The create info is valid for the duration of the call.
            let semaphore =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .context("failed to create Vulkan semaphore")?;
            this.render_finished.push(semaphore);
        }
        // SAFETY: The images belong to the swapchain, created on the core's
        // device, and `drop` removes them before destroying it.
        this.textures = unsafe {
            core.insert_surface_images(
                &this.images,
                format,
                Extent3d {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
            )
        };

        info!(
            "created swapchain: {}x{}, {} images, {:?} {:?}, {:?}",
            extent.width,
            extent.height,
            this.images.len(),
            surface_format.format,
            surface_format.color_space,
            PRESENT_MODE,
        );
        Ok(Some(this))
    }

    /// The size of the swapchain's images, in pixels.
    pub(super) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// The format of the swapchain's images.
    pub(super) fn format(&self) -> Format {
        self.format
    }

    /// The window size the swapchain was created for.
    ///
    /// Compare new window sizes with this rather than with
    /// [`extent`](Self::extent) to decide whether to recreate the swapchain:
    /// where the surface chooses its own size, the extent may never match the
    /// reported window size.
    pub(super) fn window_size(&self) -> WindowSize {
        self.window_size
    }

    /// Image `index` as a texture.
    pub(super) fn texture(&self, index: u32) -> Texture {
        self.textures[index as usize]
    }

    /// Image `index` and a view of it.
    pub(super) fn image(&self, index: u32) -> (vk::Image, vk::ImageView) {
        let index = index as usize;
        (self.images[index], self.image_views[index])
    }

    /// The semaphore to signal when rendering to image `index` has finished,
    /// which [`present`](Self::present) waits on.
    pub(super) fn render_finished(&self, index: u32) -> vk::Semaphore {
        self.render_finished[index as usize]
    }

    /// Acquires the next image to render to, signalling `image_acquired` when
    /// it is ready. Waits for as long as it takes for an image to become
    /// available.
    ///
    /// Returns the image's index, and whether the swapchain no longer matches
    /// the surface exactly (`SUBOPTIMAL_KHR`) and should be recreated. Returns
    /// `Err(ERROR_OUT_OF_DATE_KHR)` if it can no longer be used at all.
    ///
    /// # Safety
    ///
    /// `image_acquired` must be unsignalled, with no pending signal or wait.
    pub(super) unsafe fn acquire(&self, image_acquired: vk::Semaphore) -> VkResult<(u32, bool)> {
        // SAFETY: Guaranteed by the caller, and the swapchain is valid.
        unsafe {
            self.swapchain_fn.acquire_next_image(
                self.swapchain,
                u64::MAX,
                image_acquired,
                vk::Fence::null(),
            )
        }
    }

    /// Queues image `index` for presentation on `queue`, once its
    /// [`render_finished`](Self::render_finished) semaphore is signalled.
    ///
    /// Returns whether the swapchain should be recreated (`SUBOPTIMAL_KHR`),
    /// or `Err(ERROR_OUT_OF_DATE_KHR)` if it can no longer be used at all.
    ///
    /// # Safety
    ///
    /// Image `index` must have been acquired and not yet presented, work
    /// that signals its semaphore must have been submitted, and `queue` must
    /// support presenting to the swapchain's surface.
    pub(super) unsafe fn present(&self, queue: vk::Queue, index: u32) -> VkResult<bool> {
        let wait_semaphores = [self.render_finished(index)];
        let swapchains = [self.swapchain];
        let indices = [index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: Guaranteed by the caller, and `present_info` is valid for
        // the duration of the call.
        unsafe { self.swapchain_fn.queue_present(queue, &present_info) }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let device = self.core.device.handle();
        // SAFETY: The textures, semaphores, views and swapchain were created
        // by `new` and are removed or destroyed exactly once here. Waiting
        // for the device, which the queues were retrieved from, to be idle
        // ensures that no submitted work still uses them. That no command
        // buffer translated with a texture is submitted later rests on the
        // application using a surface image only in the command buffers it
        // presents with it, as design 0002's "Presentation" requires: nothing
        // checks it yet (see TODO.md).
        unsafe {
            // Nothing useful can be done if this fails while dropping.
            let _ = self.core.device.queue_locks().wait_idle(device);
            self.core.remove_surface_images(&self.textures);
            for &semaphore in &self.render_finished {
                device.destroy_semaphore(semaphore, None);
            }
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of choosing a swapchain's format.

    use super::*;

    /// `format` in the sRGB colour space.
    fn srgb(format: vk::Format) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }
    }

    /// The first preferred format the surface supports is chosen, whatever
    /// order the surface lists them in.
    #[test]
    fn the_most_preferred_format_is_chosen() {
        let formats = [
            srgb(vk::Format::R8G8B8A8_UNORM),
            srgb(vk::Format::R8G8B8A8_SRGB),
            srgb(vk::Format::B8G8R8A8_SRGB),
        ];
        assert_eq!(
            choose_format(&formats),
            Some((Format::Bgra8Srgb, srgb(vk::Format::B8G8R8A8_SRGB))),
        );
    }

    /// A surface with only UNORM formats gets none, rather than one that
    /// would show linear colour unencoded.
    #[test]
    fn unorm_formats_are_not_chosen() {
        let formats = [
            srgb(vk::Format::B8G8R8A8_UNORM),
            srgb(vk::Format::R8G8B8A8_UNORM),
        ];
        assert_eq!(choose_format(&formats), None);
    }

    /// Formats in other colour spaces, or that `Format` can't name, aren't
    /// chosen.
    #[test]
    fn other_formats_are_not_chosen() {
        let formats = [
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
            },
            srgb(vk::Format::A2B10G10R10_UNORM_PACK32),
        ];
        assert_eq!(choose_format(&formats), None);
    }
}
