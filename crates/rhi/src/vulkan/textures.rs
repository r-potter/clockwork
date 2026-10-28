//! Textures in the Vulkan backend (design 0002, "Memory" and "Data
//! structures and layout").
//!
//! Each texture is a `VkImage` with optimal tiling and exclusive sharing,
//! bound to GPU-only memory from the
//! [`MemoryAllocator`](super::memory::MemoryAllocator), in image blocks.
//! The [`VulkanDevice`]'s texture table holds them. Views, and their slots
//! in the descriptor heaps, are created when first asked for, by later
//! parts of design 0002.
//!
//! Lock order: as for buffers, the texture table is never held while
//! allocating or freeing memory.

#[cfg(test)]
mod gpu_tests;

use std::sync::{PoisonError, RwLockWriteGuard};

use ash::vk;

use super::VulkanDevice;
use super::error::{error, error_from_chain};
use super::memory::{MemoryAllocation, MemoryRequest, Resource, image_memory_requirements};
use crate::handle::HandleTable;
use crate::{
    Format, MemoryLocation, RhiResult, Texture, TextureDesc, TextureDimension, TextureUsage,
};

/// A texture in the texture table.
#[derive(Debug)]
pub(super) struct TextureEntry {
    /// The image.
    image: vk::Image,
    /// The memory it's bound to.
    allocation: MemoryAllocation,
}

/// The Vulkan format for `format`.
pub(super) fn vk_format(format: Format) -> vk::Format {
    match format {
        Format::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        Format::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        Format::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        Format::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
        Format::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        Format::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        Format::R32Uint => vk::Format::R32_UINT,
        Format::Depth32Float => vk::Format::D32_SFLOAT,
    }
}

/// Each usage, with the Vulkan image usage and the optimal-tiling format
/// feature it needs.
const USAGES: [(TextureUsage, vk::ImageUsageFlags, vk::FormatFeatureFlags); 6] = [
    (
        TextureUsage::SAMPLED,
        vk::ImageUsageFlags::SAMPLED,
        vk::FormatFeatureFlags::SAMPLED_IMAGE,
    ),
    (
        TextureUsage::STORAGE,
        vk::ImageUsageFlags::STORAGE,
        vk::FormatFeatureFlags::STORAGE_IMAGE,
    ),
    (
        TextureUsage::COLOUR_ATTACHMENT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
        vk::FormatFeatureFlags::COLOR_ATTACHMENT,
    ),
    (
        TextureUsage::DEPTH_ATTACHMENT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    ),
    (
        TextureUsage::COPY_SOURCE,
        vk::ImageUsageFlags::TRANSFER_SRC,
        vk::FormatFeatureFlags::TRANSFER_SRC,
    ),
    (
        TextureUsage::COPY_DESTINATION,
        vk::ImageUsageFlags::TRANSFER_DST,
        vk::FormatFeatureFlags::TRANSFER_DST,
    ),
];

/// The Vulkan image usage for a texture used as `usage`, and the format
/// features its format needs with optimal tiling.
pub(super) fn usage_flags(usage: TextureUsage) -> (vk::ImageUsageFlags, vk::FormatFeatureFlags) {
    USAGES
        .iter()
        .filter(|(part, ..)| usage.contains(*part))
        .fold(
            (
                vk::ImageUsageFlags::empty(),
                vk::FormatFeatureFlags::empty(),
            ),
            |(usage, features), &(_, part_usage, part_features)| {
                (usage | part_usage, features | part_features)
            },
        )
}

/// The Vulkan image type and creation flags for `dimension`.
pub(super) fn image_type(dimension: TextureDimension) -> (vk::ImageType, vk::ImageCreateFlags) {
    match dimension {
        TextureDimension::D2 => (vk::ImageType::TYPE_2D, vk::ImageCreateFlags::empty()),
        TextureDimension::D3 => (vk::ImageType::TYPE_3D, vk::ImageCreateFlags::empty()),
        TextureDimension::Cube => (
            vk::ImageType::TYPE_2D,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
        ),
    }
}

impl VulkanDevice {
    /// Creates an image for `desc`, binds it to GPU-only memory, and adds it
    /// to the texture table.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`](crate::RhiError::OutOfDeviceMemory)
    /// if device memory runs out, or [`RhiError::Other`](crate::RhiError::Other)
    /// if creating, allocating or binding fails otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the description is invalid (see
    /// [`TextureDesc::validate`]), or the device doesn't support its format for
    /// its usage.
    pub(super) fn create_texture_entry(&self, desc: &TextureDesc<'_>) -> RhiResult<Texture> {
        desc.validate();
        let format = vk_format(desc.format);
        let (usage, features) = usage_flags(desc.usage);
        // SAFETY: The physical device was enumerated from this instance.
        let supported = unsafe {
            self.instance
                .instance
                .get_physical_device_format_properties(self.physical_device.handle, format)
        }
        .optimal_tiling_features;
        assert!(
            supported.contains(features),
            "{} doesn't support {:?} textures with usage {:?}, needed by \"{}\"",
            self.physical_device.name(),
            desc.format,
            desc.usage,
            desc.label,
        );

        let device = self.device.handle();
        let (image_type, flags) = image_type(desc.dimension);
        let extent = desc.extent;
        let image_create_info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: extent.depth,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `image_create_info` is valid for the call: the description
        // was checked, and the format supports the usage.
        let image = unsafe { device.create_image(&image_create_info, None) }
            .map_err(|result| error(result, "create texture", desc.label))?;

        // SAFETY: `image` was just created on `device`.
        let (requirements, dedication) = unsafe { image_memory_requirements(device, image) };
        let allocation = match self.allocator.allocate(&MemoryRequest {
            label: desc.label,
            resource: Resource::Image(image),
            requirements,
            dedication,
            location: MemoryLocation::Device,
        }) {
            Ok(allocation) => allocation,
            Err(err) => {
                // SAFETY: The image was just created, and nothing uses it.
                unsafe { device.destroy_image(image, None) };
                return Err(error_from_chain(err, desc.label));
            }
        };
        // SAFETY: The image isn't bound yet, and the allocation was made for
        // its requirements, from memory of a type it allows, at an offset of
        // its alignment, in a block that holds only images.
        let bound =
            unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) };
        if let Err(result) = bound {
            // SAFETY: The image was just created, and nothing uses it or its
            // memory.
            unsafe { device.destroy_image(image, None) };
            self.allocator.free(allocation);
            return Err(error(result, "bind memory for texture", desc.label));
        }

        let entry = TextureEntry { image, allocation };
        Ok(Texture::from_handle(
            self.lock_textures_for_writing().insert(desc.label, entry),
        ))
    }

    /// Removes `texture` from the table, then destroys it and frees its
    /// memory.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    pub(super) fn destroy_texture_entry(&self, texture: Texture) {
        let (_, entry) = self.lock_textures_for_writing().remove(texture.handle());
        // SAFETY: Nothing can submit work through a `Queue` yet, so no
        // submitted work uses the texture. Once it can, deferred destruction
        // (`rhi-commands`) calls this only once nothing can.
        unsafe { self.free_texture(entry) };
    }

    /// Destroys a texture taken out of the table, and frees its memory.
    ///
    /// # Safety
    ///
    /// No submitted work may still use the texture.
    pub(super) unsafe fn free_texture(&self, entry: TextureEntry) {
        // SAFETY: The image was created on this device, and the caller
        // guarantees that nothing uses it. It's destroyed before its memory
        // is freed.
        unsafe { self.device.handle().destroy_image(entry.image, None) };
        self.allocator.free(entry.allocation);
    }

    /// Locks the texture table for writing, recovering it if a panic
    /// poisoned it: no table operation panics halfway through, so it's
    /// consistent.
    fn lock_textures_for_writing(&self) -> RwLockWriteGuard<'_, HandleTable<TextureEntry>> {
        self.textures
            .write()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the texture translations to Vulkan.

    use std::collections::HashSet;

    use super::*;

    /// Every format has a Vulkan format of its own.
    #[test]
    fn formats_map_to_distinct_vulkan_formats() {
        let formats: HashSet<_> = Format::ALL.into_iter().map(vk_format).collect();
        assert_eq!(formats.len(), Format::ALL.len());
        assert_eq!(vk_format(Format::Depth32Float), vk::Format::D32_SFLOAT);
    }

    /// Each usage adds its image usage and the format feature it needs.
    #[test]
    fn usages_add_their_flags_and_features() {
        assert_eq!(
            usage_flags(TextureUsage::SAMPLED | TextureUsage::COPY_DESTINATION),
            (
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST,
            ),
        );
        let every_usage = USAGES
            .iter()
            .fold(TextureUsage::NONE, |all, &(usage, ..)| all | usage);
        let (usage, features) = usage_flags(every_usage);
        assert_eq!(usage.as_raw().count_ones(), 6);
        assert_eq!(features.as_raw().count_ones(), 6);
    }

    /// Cube maps are cube-compatible 2D images, and 3D textures 3D images.
    #[test]
    fn dimensions_map_to_image_types() {
        assert_eq!(
            image_type(TextureDimension::D2),
            (vk::ImageType::TYPE_2D, vk::ImageCreateFlags::empty())
        );
        assert_eq!(
            image_type(TextureDimension::D3),
            (vk::ImageType::TYPE_3D, vk::ImageCreateFlags::empty())
        );
        assert_eq!(
            image_type(TextureDimension::Cube),
            (
                vk::ImageType::TYPE_2D,
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            )
        );
    }
}
