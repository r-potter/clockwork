//! Swapchains, which hold the images presented to a window.

use anyhow::{Context, Result, bail};
use ash::{Instance, khr, vk};
use tracing::info;

use super::device::LogicalDevice;
use super::physical_device::PhysicalDevice;
use super::surface::Surface;
use crate::rhi::WindowSize;

/// Surface formats to use for swapchain images, in order of preference. If the
/// surface supports none of them, its first format is used.
///
/// 8-bit sRGB formats, so that shaders can write linear colour and have it
/// converted for display.
const PREFERRED_FORMATS: &[vk::SurfaceFormatKHR] = &[
    vk::SurfaceFormatKHR {
        format: vk::Format::B8G8R8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    },
    vk::SurfaceFormatKHR {
        format: vk::Format::R8G8B8A8_SRGB,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    },
];

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

/// A swapchain for a surface, and a view of each of its images.
///
/// Destroyed when dropped, so the owner must drop it before the surface and
/// the logical device it was created from.
pub struct Swapchain {
    /// Kept to destroy the image views.
    device: ash::Device,
    swapchain_fn: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::SurfaceFormatKHR,
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
    /// Returns an error if the surface cannot be queried, has no formats, or
    /// if the swapchain or its image views cannot be created.
    pub fn new(
        instance: &Instance,
        device: &LogicalDevice,
        physical_device: &PhysicalDevice,
        surface: &Surface,
        window_size: WindowSize,
        old: Option<&Swapchain>,
    ) -> Result<Option<Self>> {
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
        let Some(&first_format) = formats.first() else {
            bail!("the window surface reports no supported formats");
        };
        let format = PREFERRED_FORMATS
            .iter()
            .find(|preferred| formats.contains(preferred))
            .copied()
            .unwrap_or(first_format);

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
            .image_format(format.format)
            .image_color_space(format.color_space)
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

        let swapchain_fn = khr::swapchain::Device::new(instance, device.handle());
        // SAFETY: `create_info` and the structures it points to are valid for
        // the duration of the call, and its surface and old swapchain belong to
        // this device's instance and device.
        let swapchain = unsafe { swapchain_fn.create_swapchain(&create_info, None) }
            .context("failed to create Vulkan swapchain")?;

        // Construct `Self` before creating the image views, so that the
        // swapchain and any views already created are destroyed by `Drop` if a
        // later step fails.
        let mut this = Self {
            device: device.handle().clone(),
            swapchain_fn,
            swapchain,
            images: Vec::new(),
            image_views: Vec::new(),
            format,
            extent,
            window_size,
        };
        // SAFETY: `swapchain` was just created from `swapchain_fn`.
        this.images = unsafe { this.swapchain_fn.get_swapchain_images(swapchain) }
            .context("failed to get Vulkan swapchain images")?;
        for &image in &this.images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            // SAFETY: `image` belongs to the swapchain, which was created on
            // this device, and `view_info` is valid for the duration of the
            // call.
            let view = unsafe { this.device.create_image_view(&view_info, None) }
                .context("failed to create Vulkan swapchain image view")?;
            this.image_views.push(view);
        }

        info!(
            "created swapchain: {}x{}, {} images, {:?} {:?}, {:?}",
            extent.width,
            extent.height,
            this.images.len(),
            format.format,
            format.color_space,
            PRESENT_MODE,
        );
        Ok(Some(this))
    }

    /// The size of the swapchain's images, in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// The format of the swapchain's images.
    pub fn format(&self) -> vk::SurfaceFormatKHR {
        self.format
    }

    /// The window size the swapchain was created for.
    ///
    /// Compare new window sizes with this rather than with
    /// [`extent`](Self::extent) to decide whether to recreate the swapchain:
    /// where the surface chooses its own size, the extent may never match the
    /// reported window size.
    pub fn window_size(&self) -> WindowSize {
        self.window_size
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        // SAFETY: The views and swapchain were created by `new` and are
        // destroyed exactly once here. Waiting for the device to be idle
        // ensures that no submitted work still uses them.
        unsafe {
            // Nothing useful can be done if this fails while dropping.
            let _ = self.device.device_wait_idle();
            for &view in &self.image_views {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
        }
    }
}
