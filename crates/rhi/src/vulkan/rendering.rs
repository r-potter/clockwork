//! Beginning and ending rendering, with dynamic rendering, to attachment
//! views created and cached on first use (design 0002, "Subresources and
//! views").

use std::slice;

use ash::vk;

use super::VulkanDevice;
use super::textures::TextureEntry;
use crate::handle::HandleTable;
use crate::{Access, ColorAttachment, Format, LoadOp, RhiResult, StoreOp};

impl<T> From<LoadOp<T>> for vk::AttachmentLoadOp {
    fn from(load: LoadOp<T>) -> Self {
        match load {
            LoadOp::Load => Self::LOAD,
            LoadOp::Clear(_) => Self::CLEAR,
            LoadOp::DontCare => Self::DONT_CARE,
        }
    }
}

impl From<StoreOp> for vk::AttachmentStoreOp {
    fn from(store: StoreOp) -> Self {
        match store {
            StoreOp::Store => Self::STORE,
            StoreOp::DontCare => Self::DONT_CARE,
        }
    }
}

/// The clear value that sets a colour attachment of `format` to `colour`,
/// red, green, blue and alpha: the values themselves for a format that
/// holds floats or normalized values, and converted to integers, rounding
/// toward zero, for one that holds integers. Pure, so it's tested without a
/// GPU.
fn clear_colour(format: Format, colour: [f32; 4]) -> vk::ClearColorValue {
    match format {
        Format::R32Uint => vk::ClearColorValue {
            uint32: colour.map(|channel| channel as u32),
        },
        Format::Rgba8Unorm
        | Format::Rgba8Srgb
        | Format::Bgra8Unorm
        | Format::Bgra8Srgb
        | Format::Rgba16Float
        | Format::Rgba32Float
        | Format::Depth32Float => vk::ClearColorValue { float32: colour },
    }
}

impl VulkanDevice {
    /// Records a command into `buffer` that begins rendering to
    /// `attachment`, a colour attachment whose texture is in `textures`,
    /// over the whole of the mip it names.
    ///
    /// # Errors
    ///
    /// Returns an error if the attachment view can't be created.
    ///
    /// # Safety
    ///
    /// `buffer` must be recording, on this device, for a family that
    /// supports graphics, outside rendering, and no other thread may use
    /// it. `textures` must be this device's texture table, locked for as
    /// long as `buffer` is recording.
    ///
    /// # Panics
    ///
    /// Panics, naming the texture, if it's stale or can't be a colour
    /// attachment, or if the attachment is out of range.
    pub(super) unsafe fn begin_rendering(
        &self,
        buffer: vk::CommandBuffer,
        attachment: &ColorAttachment,
        textures: &HandleTable<TextureEntry>,
    ) -> RhiResult<()> {
        let (label, entry) = textures
            .lookup(attachment.texture.handle())
            .unwrap_or_else(|err| panic!("{err}"));
        let (layers, extent) = attachment.resolve(&entry.attached(label));
        let layer_count = layers.end - layers.start;
        // SAFETY: The entry is in `textures`, which the caller keeps locked
        // while the view is used, and the attachment's checks passed, so its
        // mip and layers are inside a texture that isn't 3D.
        let view = unsafe { self.attachment_view(entry, label, attachment.mip, layers) }?;
        let clear_value = match attachment.load {
            LoadOp::Clear(colour) => vk::ClearValue {
                color: clear_colour(entry.texel_format(), colour),
            },
            LoadOp::Load | LoadOp::DontCare => vk::ClearValue::default(),
        };
        // The attachment is in the layout writing it needs, as the barriers
        // before rendering put it.
        let layout = vk::ImageLayout::from(
            Access::ColorAttachmentWrite
                .layout()
                .expect("attachment accesses apply to textures"),
        );
        let colour_info = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(layout)
            .load_op(attachment.load.into())
            .store_op(attachment.store.into())
            .clear_value(clear_value);
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            })
            .layer_count(layer_count)
            .color_attachments(slice::from_ref(&colour_info));
        // SAFETY: Guaranteed by the caller, the view is of a live image,
        // since the table is locked, the render area and layer count are
        // the view's, and `rendering_info` is valid for the call.
        unsafe {
            self.device
                .handle()
                .cmd_begin_rendering(buffer, &rendering_info);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of translating attachments' operations and clear values.

    use super::*;

    /// Each load and store operation has a Vulkan operation of its own.
    #[test]
    fn operations_map_to_vulkan_operations() {
        assert_eq!(
            vk::AttachmentLoadOp::from(LoadOp::<f32>::Load),
            vk::AttachmentLoadOp::LOAD
        );
        assert_eq!(
            vk::AttachmentLoadOp::from(LoadOp::Clear(0.0)),
            vk::AttachmentLoadOp::CLEAR
        );
        assert_eq!(
            vk::AttachmentLoadOp::from(LoadOp::<f32>::DontCare),
            vk::AttachmentLoadOp::DONT_CARE
        );
        assert_eq!(
            vk::AttachmentStoreOp::from(StoreOp::Store),
            vk::AttachmentStoreOp::STORE
        );
        assert_eq!(
            vk::AttachmentStoreOp::from(StoreOp::DontCare),
            vk::AttachmentStoreOp::DONT_CARE
        );
    }

    /// A float or normalized format is cleared to the values given, and an
    /// integer format to them as integers.
    #[test]
    fn clear_colours_suit_their_format() {
        let colour = [0.25, 1.0, 7.9, 255.0];
        // SAFETY: Each clear value is read as the member `clear_colour`
        // writes for its format.
        unsafe {
            assert_eq!(clear_colour(Format::Rgba16Float, colour).float32, colour);
            assert_eq!(clear_colour(Format::R32Uint, colour).uint32, [0, 1, 7, 255]);
        }
    }
}
