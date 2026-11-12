//! Render passes: the attachments draws render to, between
//! [`CommandList::begin_render_pass`](crate::CommandList::begin_render_pass) and
//! [`end_render_pass`](crate::CommandList::end_render_pass) (design 0002,
//! "Subresources and views").
//!
//! Each attachment names one mip of a texture and a range of its layers.
//! Whether a texture can be one, and the region a render pass covers, are
//! pure functions of the textures, checked when the list is translated and
//! tested without a device: [`ColorAttachment::resolve`] checks an
//! attachment against its texture, and [`render_area`] checks that a render
//! pass's attachments agree.

use std::ops::Range;

use crate::{Extent2d, Extent3d, Subresource, Texture, TextureDimension, TextureUsage};

/// A render pass: the attachments draws render to, from
/// [`CommandList::begin_render_pass`](crate::CommandList::begin_render_pass)
/// until [`end_render_pass`](crate::CommandList::end_render_pass).
///
/// Every attachment is the same size, at the mip it names, and names the
/// same number of layers. Rendering to several layers at once is layered
/// rendering, with the layer chosen in the shader
/// (`SV_RenderTargetArrayIndex`).
#[derive(Debug, Clone, PartialEq)]
pub struct RenderPassDesc {
    /// The colour attachments, by number, which a fragment shader's
    /// outputs (`SV_Target0` to `SV_Target7`) and a
    /// [`RenderState`](crate::RenderState)'s colour writes refer to. At
    /// most [`MAX_COLOUR_ATTACHMENTS`](crate::MAX_COLOUR_ATTACHMENTS), and
    /// at most the device's limit.
    pub colour: Vec<ColorAttachment>,
}

impl RenderPassDesc {
    /// A render pass to mip 0 of layer 0 of `texture`, a single colour
    /// attachment, loaded as `load` says and stored.
    pub fn colour(texture: Texture, load: LoadOp<[f32; 4]>) -> Self {
        Self {
            colour: vec![ColorAttachment {
                texture,
                mip: 0,
                layers: 0..1,
                load,
                store: StoreOp::Store,
            }],
        }
    }
}

/// A colour attachment: one mip of a texture, and the layers to render to.
///
/// The texture must have [`TextureUsage::COLOUR_ATTACHMENT`], can't be a 3D
/// texture, and must be in the layout of
/// [`Access::ColorAttachmentWrite`](crate::Access::ColorAttachmentWrite)
/// during the render pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorAttachment {
    /// The texture.
    pub texture: Texture,
    /// The mip rendered to.
    pub mip: u32,
    /// The layers rendered to. A range ending at
    /// [`Subresource::REMAINING`] reaches the texture's last layer.
    pub layers: Range<u32>,
    /// What the attachment holds when the render pass begins. A clear value is
    /// red, green, blue and alpha, converted to an integer format's
    /// integers.
    pub load: LoadOp<[f32; 4]>,
    /// What happens to what's rendered when the render pass ends.
    pub store: StoreOp,
}

impl ColorAttachment {
    /// The attachment's layers, resolved for `texture`, and the size of the
    /// mip it names.
    ///
    /// # Panics
    ///
    /// Panics, naming the texture, if it can't be a colour attachment, or
    /// if the mip or layers are out of range.
    pub(crate) fn resolve(&self, texture: &AttachedTexture<'_>) -> (Range<u32>, Extent2d) {
        resolve_attachment(
            texture,
            (
                TextureUsage::COLOUR_ATTACHMENT,
                "COLOUR_ATTACHMENT",
                "a colour attachment",
            ),
            self.mip,
            &self.layers,
        )
    }
}

/// What an attachment holds when the render pass begins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp<T> {
    /// What it held before, so that the render pass adds to it.
    Load,
    /// Every texel set to the value, which is faster than loading.
    Clear(T),
    /// Anything: the render pass will cover every texel it reads. The fastest.
    DontCare,
}

/// What happens to what's rendered to an attachment when the render pass ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoreOp {
    /// It's kept, for later passes to read.
    Store,
    /// It may be discarded, as for a depth attachment only this render pass
    /// tests against, which saves writing it to memory on some GPUs.
    DontCare,
}

/// What attachment checks need to know about a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttachedTexture<'a> {
    /// The texture's label, for messages.
    pub label: &'a str,
    /// Its shape.
    pub dimension: TextureDimension,
    /// The size of its largest mip.
    pub extent: Extent3d,
    /// How many mips it has.
    pub mip_count: u32,
    /// How many layers it has.
    pub layer_count: u32,
    /// How it's used.
    pub usage: TextureUsage,
}

/// The layers of `texture` an attachment names, `layers` of mip `mip`,
/// resolved, and the size of that mip, for an attachment of the kind
/// `usage` describes: the usage it needs, the usage's name, and the kind of
/// attachment, for messages.
///
/// # Panics
///
/// Panics, naming the texture, if it lacks the usage, is a 3D texture, or
/// if the mip or layers are out of range.
fn resolve_attachment(
    texture: &AttachedTexture<'_>,
    (usage, usage_name, kind): (TextureUsage, &str, &str),
    mip: u32,
    layers: &Range<u32>,
) -> (Range<u32>, Extent2d) {
    let label = texture.label;
    assert!(
        texture.usage.contains(usage),
        "texture \"{label}\" has no {usage_name} usage, so it can't be {kind}",
    );
    // A 3D texture's slices can be rendered to only through a 2D array view
    // of it, which needs it to be created for one.
    assert!(
        texture.dimension != TextureDimension::D3,
        "3D texture \"{label}\" can't be {kind}",
    );
    let subresource = Subresource {
        mips: mip..mip.saturating_add(1),
        layers: layers.clone(),
    }
    .resolve(label, texture.mip_count, texture.layer_count);
    let extent = Extent2d {
        width: (texture.extent.width >> mip).max(1),
        height: (texture.extent.height >> mip).max(1),
    };
    (subresource.layers, extent)
}

/// One attachment of a render pass, resolved: its texture's label, the layers
/// it names, and the size of its mip.
pub(crate) type ResolvedAttachment<'a> = (&'a str, Range<u32>, Extent2d);

/// The size of the region a render pass in the list `list` covers, and the
/// number of layers it renders to, from its `attachments`.
///
/// # Panics
///
/// Panics, naming the list and two attachments' textures, if the
/// attachments aren't all the same size, at the mips they name, or don't
/// all name the same number of layers.
pub(crate) fn render_area<'a>(
    list: &str,
    attachments: impl IntoIterator<Item = ResolvedAttachment<'a>>,
) -> (Extent2d, u32) {
    let mut attachments = attachments.into_iter();
    let (first, first_layers, extent) = attachments
        .next()
        .expect("recording checked that a render pass has an attachment");
    let layer_count = first_layers.len() as u32;
    for (label, layers, other_extent) in attachments {
        assert!(
            other_extent == extent,
            "command list \"{list}\" renders to attachments of different sizes: \"{first}\" is \
             {}x{} and \"{label}\" is {}x{}, at the mips they name",
            extent.width,
            extent.height,
            other_extent.width,
            other_extent.height,
        );
        assert!(
            layers.len() as u32 == layer_count,
            "command list \"{list}\" renders to {layer_count} layers of \"{first}\" but {} of \
             \"{label}\", and every attachment must name as many",
            layers.len(),
        );
    }
    (extent, layer_count)
}

// A render pass is plain data, recorded on any thread (design 0002,
// "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RenderPassDesc>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of attachment checks and render areas (design 0002, R21).

    use super::*;
    use crate::handle::HandleTable;

    /// A 2D texture of 64×32 texels, with 4 mips and 3 layers, that can
    /// be a colour attachment.
    fn texture() -> AttachedTexture<'static> {
        AttachedTexture {
            label: "albedo",
            dimension: TextureDimension::D2,
            extent: Extent3d::new_2d(64, 32),
            mip_count: 4,
            layer_count: 3,
            usage: TextureUsage::COLOUR_ATTACHMENT,
        }
    }

    /// A colour attachment of `mip` and `layers`, cleared to black.
    fn attachment(mip: u32, layers: Range<u32>) -> ColorAttachment {
        ColorAttachment {
            texture: Texture::from_handle(HandleTable::new("texture").insert("albedo", ())),
            mip,
            layers,
            load: LoadOp::Clear([0.0; 4]),
            store: StoreOp::Store,
        }
    }

    /// The message `f` panics with.
    fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let panic = std::panic::catch_unwind(f).expect_err("it should panic");
        panic.downcast_ref::<String>().unwrap().clone()
    }

    /// An attachment resolves its layers, and is the size of its mip, which
    /// is at least one texel on each side.
    #[test]
    fn attachments_are_the_size_of_their_mip() {
        let texture = texture();
        assert_eq!(
            attachment(0, 0..Subresource::REMAINING).resolve(&texture),
            (
                0..3,
                Extent2d {
                    width: 64,
                    height: 32,
                },
            ),
        );
        assert_eq!(
            attachment(1, 2..3).resolve(&texture),
            (
                2..3,
                Extent2d {
                    width: 32,
                    height: 16,
                },
            ),
        );
        let tall = AttachedTexture {
            extent: Extent3d::new_2d(2, 32),
            ..texture
        };
        assert_eq!(
            attachment(3, 0..1).resolve(&tall).1,
            Extent2d {
                width: 1,
                height: 4,
            },
        );
    }

    /// The `RenderPassDesc::colour` shorthand renders to mip 0 of layer 0, and
    /// stores it.
    #[test]
    fn single_colour_render_passes_use_the_first_mip_and_layer() {
        let expected = attachment(0, 0..1);
        assert_eq!(
            RenderPassDesc::colour(expected.texture, LoadOp::Clear([0.0; 4])).colour,
            [expected],
        );
    }

    /// A texture that can't be a colour attachment panics, naming the
    /// texture: one without the usage, or a 3D texture.
    #[test]
    fn colour_attachments_need_their_usage_and_two_dimensions() {
        let unusable = AttachedTexture {
            usage: TextureUsage::SAMPLED,
            ..texture()
        };
        assert_eq!(
            panic_message(|| {
                attachment(0, 0..1).resolve(&unusable);
            }),
            "texture \"albedo\" has no COLOUR_ATTACHMENT usage, so it can't be a colour attachment",
        );
        let volume = AttachedTexture {
            dimension: TextureDimension::D3,
            layer_count: 1,
            ..texture()
        };
        assert_eq!(
            panic_message(|| {
                attachment(0, 0..1).resolve(&volume);
            }),
            "3D texture \"albedo\" can't be a colour attachment",
        );
    }

    /// An attachment past the texture's mips or layers panics, as any
    /// subresource does.
    #[test]
    fn attachments_out_of_range_panic() {
        assert_eq!(
            panic_message(|| {
                attachment(4, 0..1).resolve(&texture());
            }),
            "texture \"albedo\" has 4 mips, so a subresource can't name mips 4..5",
        );
        assert_eq!(
            panic_message(|| {
                attachment(0, 2..4).resolve(&texture());
            }),
            "texture \"albedo\" has 3 layers, so a subresource can't name layers 2..4",
        );
    }

    /// Attachments of one size and layer count render to that size and
    /// that many layers.
    #[test]
    fn matching_attachments_give_the_render_area() {
        let extent = Extent2d {
            width: 32,
            height: 16,
        };
        assert_eq!(
            render_area(
                "gbuffer",
                [("albedo", 0..2, extent), ("normals", 3..5, extent)]
            ),
            (extent, 2),
        );
    }

    /// Attachments of different sizes, or different numbers of layers,
    /// panic, naming the list and the textures.
    #[test]
    fn mismatched_attachments_panic() {
        let large = Extent2d {
            width: 32,
            height: 16,
        };
        let small = Extent2d {
            width: 16,
            height: 8,
        };
        assert_eq!(
            panic_message(|| {
                render_area(
                    "gbuffer",
                    [("albedo", 0..1, large), ("normals", 0..1, small)],
                );
            }),
            "command list \"gbuffer\" renders to attachments of different sizes: \"albedo\" is \
             32x16 and \"normals\" is 16x8, at the mips they name",
        );
        assert_eq!(
            panic_message(|| {
                render_area(
                    "gbuffer",
                    [("albedo", 0..2, large), ("normals", 0..1, large)],
                );
            }),
            "command list \"gbuffer\" renders to 2 layers of \"albedo\" but 1 of \"normals\", and \
             every attachment must name as many",
        );
    }
}
