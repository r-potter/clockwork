//! Textures in the Vulkan backend (design 0002, "Memory" and "Data
//! structures and layout").
//!
//! Each texture is a `VkImage` with optimal tiling and exclusive sharing,
//! bound to GPU-only memory from the
//! [`MemoryAllocator`](super::memory::MemoryAllocator), in image blocks.
//! The [`VulkanDevice`]'s texture table holds them.
//!
//! A view's descriptor is written into a slot in the resource heap the
//! first time [`texture_index`](crate::GpuResources::texture_index) asks
//! for it, and its slot cached in the texture's entry, behind a `Mutex` of
//! its own, so writing one never takes the table's write lock (design
//! 0002, "Subresources and views"). `VK_EXT_descriptor_heap` writes an
//! image descriptor from a `VkImageViewCreateInfo`, so no `VkImageView` is
//! created. The slots are freed with the texture.
//!
//! Rendering needs a `VkImageView` of each attachment, so those are
//! created the first time a translation renders to one, and cached in the
//! entry in the same way, behind a `Mutex` of their own. They use no heap
//! slot, and are destroyed with the texture.
//!
//! Lock order: as for buffers, the texture table is never held while
//! allocating or freeing memory. Writing a view takes the table's read
//! lock, then the entry's view lock, then the resource heap's slot
//! allocator. Freeing a texture's slots takes only the allocator, since
//! the entry is out of the table by then. Translation takes an entry's
//! attachment view lock while it holds the table's read lock, and takes
//! nothing else while holding it.

#[cfg(test)]
mod gpu_tests;

use std::collections::HashMap;
use std::ops::Range;
use std::slice;
use std::sync::{Mutex, PoisonError, RwLockReadGuard, RwLockWriteGuard};

use ash::vk;

use super::VulkanDevice;
use super::destruction::Retired;
use super::error::{error, error_from_chain};
use super::memory::{MemoryAllocation, MemoryRequest, Resource, image_memory_requirements};
use crate::copy::CopiedTexture;
use crate::handle::HandleTable;
use crate::rendering::AttachedTexture;
use crate::view::{ViewType, ViewedTexture};
use crate::{
    Access, DescriptorIndex, Extent3d, Format, FormatInfo, HeapKind, MemoryLocation, RhiError,
    RhiResult, Subresource, Texture, TextureDesc, TextureDimension, TextureUsage, TextureView,
};

/// A texture in the texture table.
#[derive(Debug)]
pub(super) struct TextureEntry {
    /// The image.
    image: vk::Image,
    /// The memory it's bound to.
    allocation: MemoryAllocation,
    /// Its format.
    format: Format,
    /// Its shape.
    dimension: TextureDimension,
    /// How it's used.
    usage: TextureUsage,
    /// The size of its largest mip.
    extent: Extent3d,
    /// How many mips it has.
    mip_count: u32,
    /// How many array layers it has.
    layer_count: u32,
    /// The slot in the resource heap of each view asked for, by the view
    /// with its ranges resolved.
    views: Mutex<HashMap<TextureView, u32>>,
    /// The image view of each attachment rendered to, by its mip and its
    /// layers, resolved.
    attachment_views: Mutex<HashMap<(u32, Range<u32>), vk::ImageView>>,
}

impl TextureEntry {
    /// The image, for commands that name it.
    pub(super) fn image(&self) -> vk::Image {
        self.image
    }

    /// The aspects its format has, for the commands that name them.
    pub(super) fn aspect(&self) -> vk::ImageAspectFlags {
        aspect(self.format())
    }

    /// Its format's block layout.
    pub(super) fn format(&self) -> FormatInfo {
        self.format.info()
    }

    /// Its format.
    pub(super) fn texel_format(&self) -> Format {
        self.format
    }

    /// What copy checks need to know about the texture labelled `label`.
    pub(super) fn copied<'a>(&self, label: &'a str) -> CopiedTexture<'a> {
        CopiedTexture {
            label,
            format: self.format(),
            extent: self.extent,
            mip_count: self.mip_count,
            layer_count: self.layer_count,
        }
    }

    /// What view checks need to know about the texture labelled `label`.
    fn viewed<'a>(&self, label: &'a str) -> ViewedTexture<'a> {
        ViewedTexture {
            label,
            dimension: self.dimension,
            mip_count: self.mip_count,
            layer_count: self.layer_count,
            usage: self.usage,
        }
    }

    /// What attachment checks need to know about the texture labelled
    /// `label`.
    pub(super) fn attached<'a>(&self, label: &'a str) -> AttachedTexture<'a> {
        AttachedTexture {
            label,
            dimension: self.dimension,
            extent: self.extent,
            mip_count: self.mip_count,
            layer_count: self.layer_count,
            usage: self.usage,
        }
    }

    /// The Vulkan range for `subresource` of the texture labelled `label`,
    /// in every aspect its format has.
    ///
    /// # Panics
    ///
    /// Panics if the subresource is out of range (see
    /// [`Subresource::resolve`]).
    pub(super) fn subresource_range(
        &self,
        label: &str,
        subresource: &Subresource,
    ) -> vk::ImageSubresourceRange {
        let resolved = subresource.resolve(label, self.mip_count, self.layer_count);
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect(),
            base_mip_level: resolved.mips.start,
            level_count: resolved.mips.end - resolved.mips.start,
            base_array_layer: resolved.layers.start,
            layer_count: resolved.layers.end - resolved.layers.start,
        }
    }
}

/// The aspects of an image of `format`: depth for a depth format, and
/// colour otherwise. There are no stencil formats yet.
fn aspect(format: FormatInfo) -> vk::ImageAspectFlags {
    if format.depth {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

impl From<Format> for vk::Format {
    /// The Vulkan format for `format`.
    fn from(format: Format) -> Self {
        match format {
            Format::Rgba8Unorm => Self::R8G8B8A8_UNORM,
            Format::Rgba8Srgb => Self::R8G8B8A8_SRGB,
            Format::Bgra8Unorm => Self::B8G8R8A8_UNORM,
            Format::Bgra8Srgb => Self::B8G8R8A8_SRGB,
            Format::Rgba16Float => Self::R16G16B16A16_SFLOAT,
            Format::Rgba32Float => Self::R32G32B32A32_SFLOAT,
            Format::R32Uint => Self::R32_UINT,
            Format::Depth32Float => Self::D32_SFLOAT,
        }
    }
}

impl From<ViewType> for vk::ImageViewType {
    /// The Vulkan image view type for a view of `view_type`.
    fn from(view_type: ViewType) -> Self {
        match view_type {
            ViewType::D2 => Self::TYPE_2D,
            ViewType::D2Array => Self::TYPE_2D_ARRAY,
            ViewType::D3 => Self::TYPE_3D,
            ViewType::Cube => Self::CUBE,
        }
    }
}

/// The image layout and descriptor type of `view`: the layout of the
/// accesses shaders make through it.
fn view_layout_and_type(view: &TextureView) -> (vk::ImageLayout, vk::DescriptorType) {
    let (access, descriptor_type) = match view {
        TextureView::Sampled(_) => (
            Access::ComputeShaderSampledRead,
            vk::DescriptorType::SAMPLED_IMAGE,
        ),
        TextureView::Storage(_) => (
            Access::ComputeShaderWrite,
            vk::DescriptorType::STORAGE_IMAGE,
        ),
    };
    let layout = access.layout().expect("shader accesses apply to textures");
    (vk::ImageLayout::from(layout), descriptor_type)
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
    /// Returns [`RhiError::OutOfDeviceMemory`] if device memory runs out, or
    /// [`RhiError::Other`] if creating, allocating or binding fails
    /// otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the description is invalid (see
    /// [`TextureDesc::validate`]), or the device doesn't support its format for
    /// its usage.
    pub(super) fn create_texture_entry(&self, desc: &TextureDesc<'_>) -> RhiResult<Texture> {
        desc.validate();
        let format = vk::Format::from(desc.format);
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
        self.device.set_object_name(image, desc.label);

        // SAFETY: `image` was just created on `device`.
        let (requirements, dedication) = unsafe { image_memory_requirements(device, image) };
        let allocation = match self.allocator.allocate(&MemoryRequest {
            label: desc.label,
            resource: Resource::Image(image),
            requirements,
            dedication,
            usage: MemoryLocation::Device.into(),
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

        let entry = TextureEntry {
            image,
            allocation,
            format: desc.format,
            dimension: desc.dimension,
            usage: desc.usage,
            extent: desc.extent,
            mip_count: desc.mip_levels,
            layer_count: desc.array_layers,
            views: Mutex::new(HashMap::new()),
            attachment_views: Mutex::new(HashMap::new()),
        };
        Ok(Texture::from_handle(
            self.lock_textures_for_writing().insert(desc.label, entry),
        ))
    }

    /// Removes `texture` from the table, and retires it, to be destroyed
    /// and its memory freed once nothing translated before now can use it.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    pub(super) fn destroy_texture_entry(&self, texture: Texture) {
        let (_, entry) = self.lock_textures_for_writing().remove(texture.handle());
        self.retire(Retired::Texture(entry));
    }

    /// Destroys a texture taken out of the table and its attachment views,
    /// frees its memory, and frees its views' slots.
    ///
    /// # Safety
    ///
    /// No submitted work may still use the texture or its views.
    pub(super) unsafe fn free_texture(&self, entry: TextureEntry) {
        let device = self.device.handle();
        let attachment_views = entry
            .attachment_views
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);
        for view in attachment_views.into_values() {
            // SAFETY: The view was created on this device, and the caller
            // guarantees that nothing uses it. It's destroyed before its
            // image.
            unsafe { device.destroy_image_view(view, None) };
        }
        // SAFETY: The image was created on this device, and the caller
        // guarantees that nothing uses it. It's destroyed before its memory
        // is freed.
        unsafe { device.destroy_image(entry.image, None) };
        self.allocator.free(entry.allocation);
        let views = entry
            .views
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);
        for slot in views.into_values() {
            self.heaps.resource.free_slot(slot);
        }
    }

    /// The slot in the resource heap of `view` of `texture`, writing its
    /// descriptor into a free slot the first time it's asked for.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::DescriptorHeapFull`] if the resource heap has no
    /// free slot, or [`RhiError::Other`] if the descriptor can't be
    /// written.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale, or the view is invalid (see
    /// [`TextureView::resolve`]).
    pub(super) fn texture_entry_index(
        &self,
        texture: Texture,
        view: &TextureView,
    ) -> RhiResult<DescriptorIndex> {
        let textures = self.lock_textures_for_reading();
        let (label, entry) = textures
            .lookup(texture.handle())
            .unwrap_or_else(|err| panic!("{err}"));
        let (view, view_type) = view.resolve(&entry.viewed(label));
        // Held while the descriptor is written, so two threads asking for a
        // new view at once write it once.
        let mut views = entry.views.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(&slot) = views.get(&view) {
            return Ok(DescriptorIndex(slot));
        }
        let heap = &self.heaps.resource;
        let slot = heap
            .allocate_slot()
            .ok_or_else(|| RhiError::DescriptorHeapFull {
                heap: HeapKind::Resource,
                label: label.to_owned(),
            })?;
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(entry.image)
            .view_type(vk::ImageViewType::from(view_type))
            .format(vk::Format::from(entry.format))
            .subresource_range(entry.subresource_range(label, view.subresource()));
        let (layout, descriptor_type) = view_layout_and_type(&view);
        let image_descriptor_info = vk::ImageDescriptorInfoEXT::default()
            .view(&image_view_create_info)
            .layout(layout);
        let resource_descriptor_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(descriptor_type)
            .data(vk::ResourceDescriptorDataEXT {
                p_image: &raw const image_descriptor_info,
            });
        let slot_range = heap.slot_address_range(slot);
        // SAFETY: The infos are valid for the call, and name a live image,
        // since the table is locked, with a subresource inside it, a view
        // type its dimension allows, and a usage it has. `slot_range` is the
        // slot's range in the heap's mapped memory, a stride long, which
        // holds an image descriptor. The slot was just handed to this
        // thread, so nothing else writes to the range, and no submitted work
        // reads it, since a slot is only freed once nothing can.
        let written = unsafe {
            self.device.descriptor_heap().write_resource_descriptors(
                slice::from_ref(&resource_descriptor_info),
                slice::from_ref(&slot_range),
            )
        };
        if let Err(result) = written {
            heap.free_slot(slot);
            return Err(error(result, "write a view descriptor for texture", label));
        }
        views.insert(view, slot);
        Ok(DescriptorIndex(slot))
    }

    /// The image view of mip `mip` of `layers`, resolved, of `entry`, the
    /// texture labelled `label`, for rendering to, creating it the first
    /// time it's asked for: a 2D view of one layer, or a 2D array view of
    /// several.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`] if device memory runs out, or
    /// [`RhiError::Other`] if the view can't be created otherwise.
    ///
    /// # Safety
    ///
    /// `entry` must be in this device's texture table, which must stay
    /// locked while the view is recorded, and `mip` and `layers` must be
    /// inside the texture, which must not be a 3D texture.
    pub(super) unsafe fn attachment_view(
        &self,
        entry: &TextureEntry,
        label: &str,
        mip: u32,
        layers: Range<u32>,
    ) -> RhiResult<vk::ImageView> {
        // Held while the view is created, so two threads rendering to a new
        // attachment at once create it once.
        let mut views = entry
            .attachment_views
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let key = (mip, layers);
        if let Some(&view) = views.get(&key) {
            return Ok(view);
        }
        let (mip, layers) = &key;
        let view_type = if layers.len() == 1 {
            vk::ImageViewType::TYPE_2D
        } else {
            vk::ImageViewType::TYPE_2D_ARRAY
        };
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(entry.image)
            .view_type(view_type)
            .format(vk::Format::from(entry.format))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: entry.aspect(),
                base_mip_level: *mip,
                level_count: 1,
                base_array_layer: layers.start,
                layer_count: layers.end - layers.start,
            });
        // SAFETY: `image_view_create_info` is valid for the call, and names
        // a live image, since the table is locked, with a subresource inside
        // it, as the caller guarantees, and a view type a 2D or cube image
        // allows.
        let view = unsafe {
            self.device
                .handle()
                .create_image_view(&image_view_create_info, None)
        }
        .map_err(|result| error(result, "create an attachment view of texture", label))?;
        self.device.set_object_name(view, label);
        views.insert(key, view);
        Ok(view)
    }

    /// Locks the texture table for reading, recovering it if a panic
    /// poisoned it: no table operation panics halfway through, so it's
    /// consistent.
    pub(super) fn lock_textures_for_reading(
        &self,
    ) -> RwLockReadGuard<'_, HandleTable<TextureEntry>> {
        self.textures.read().unwrap_or_else(PoisonError::into_inner)
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
        let formats: HashSet<_> = Format::ALL.into_iter().map(vk::Format::from).collect();
        assert_eq!(formats.len(), Format::ALL.len());
        assert_eq!(
            vk::Format::from(Format::Depth32Float),
            vk::Format::D32_SFLOAT
        );
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

    /// Depth formats have the depth aspect, and the others colour.
    #[test]
    fn formats_have_their_aspects() {
        for format in Format::ALL {
            let expected = if format == Format::Depth32Float {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };
            assert_eq!(aspect(format.info()), expected, "{format:?}");
        }
    }

    /// Each view type has a Vulkan view type of its own.
    #[test]
    fn view_types_map_to_vulkan_view_types() {
        for (view_type, expected) in [
            (ViewType::D2, vk::ImageViewType::TYPE_2D),
            (ViewType::D2Array, vk::ImageViewType::TYPE_2D_ARRAY),
            (ViewType::D3, vk::ImageViewType::TYPE_3D),
            (ViewType::Cube, vk::ImageViewType::CUBE),
        ] {
            assert_eq!(vk::ImageViewType::from(view_type), expected);
        }
    }

    /// Sampled views are sampled image descriptors in the read-only layout,
    /// and storage views storage image descriptors in the general layout:
    /// the layouts of the accesses shaders make through them.
    #[test]
    fn views_have_their_layouts_and_descriptor_types() {
        assert_eq!(
            view_layout_and_type(&TextureView::sampled()),
            (
                vk::ImageLayout::READ_ONLY_OPTIMAL,
                vk::DescriptorType::SAMPLED_IMAGE
            ),
        );
        assert_eq!(
            view_layout_and_type(&TextureView::Storage(Subresource::mip(0))),
            (vk::ImageLayout::GENERAL, vk::DescriptorType::STORAGE_IMAGE),
        );
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
