//! Texture views: how a shader sees part of a texture, through its slot in
//! the resource heap (design 0002, "Subresources and views").
//!
//! A [`TextureView`] is sampled or storage, over a [`Subresource`]. Which
//! views a texture can have, and the kind of view each is, are pure
//! functions of the texture, tested without a device:
//! [`TextureView::resolve`] checks a view against the texture and resolves
//! its ranges, and [`ViewType`] follows from the subresource.

use crate::{Subresource, TextureDimension, TextureUsage};

/// How a shader sees a texture: sampled, or as a storage image it can
/// write, over part of the texture.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TextureView {
    /// Read with or without a sampler, in the layout of
    /// [`Access::FragmentShaderSampledRead`](crate::Access::FragmentShaderSampledRead)
    /// and [`Access::ComputeShaderSampledRead`](crate::Access::ComputeShaderSampledRead).
    /// The texture must have [`TextureUsage::SAMPLED`].
    Sampled(Subresource),
    /// Read and written texel by texel, in the layout of
    /// [`Access::ComputeShaderWrite`](crate::Access::ComputeShaderWrite).
    /// Covers exactly one mip, as Vulkan requires of storage images that
    /// are written. The texture must have [`TextureUsage::STORAGE`].
    Storage(Subresource),
}

impl TextureView {
    /// A sampled view of every mip of every layer.
    pub const fn sampled() -> Self {
        Self::Sampled(Subresource::ALL)
    }

    /// The part of the texture the view covers.
    pub fn subresource(&self) -> &Subresource {
        match self {
            Self::Sampled(subresource) | Self::Storage(subresource) => subresource,
        }
    }

    /// Whether it's a storage view, rather than a sampled one.
    pub fn is_storage(&self) -> bool {
        matches!(self, Self::Storage(_))
    }

    /// The view with its ranges resolved for `texture`, so that views of
    /// the same part are equal however they name it, and the kind of view
    /// it is.
    ///
    /// # Panics
    ///
    /// Panics, naming the texture, if the subresource is out of range, if
    /// a storage view covers more than one mip, or if the texture's usage
    /// doesn't allow the view.
    pub(crate) fn resolve(&self, texture: &ViewedTexture<'_>) -> (Self, ViewType) {
        let label = texture.label;
        let (usage, usage_name, kind) = match self {
            Self::Sampled(_) => (TextureUsage::SAMPLED, "SAMPLED", "sampled"),
            Self::Storage(_) => (TextureUsage::STORAGE, "STORAGE", "storage"),
        };
        assert!(
            texture.usage.contains(usage),
            "texture \"{label}\" has no {usage_name} usage, so it can't have a {kind} view",
        );
        let subresource = self
            .subresource()
            .resolve(label, texture.mip_count, texture.layer_count);
        let view_type = ViewType::of(texture.dimension, self.is_storage(), &subresource);
        let resolved = match self {
            Self::Sampled(_) => Self::Sampled(subresource),
            Self::Storage(_) => {
                assert!(
                    subresource.mips.len() == 1,
                    "texture \"{label}\" has a storage view of mips {:?}, but a storage view \
                     covers exactly one mip",
                    subresource.mips,
                );
                Self::Storage(subresource)
            }
        };
        (resolved, view_type)
    }
}

/// What view checks need to know about a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ViewedTexture<'a> {
    /// The texture's label, for messages.
    pub label: &'a str,
    /// Its shape.
    pub dimension: TextureDimension,
    /// How many mips it has.
    pub mip_count: u32,
    /// How many layers it has.
    pub layer_count: u32,
    /// How it's used.
    pub usage: TextureUsage,
}

/// The kind of view a shader sees, which its type in the shader must
/// match: a `Texture2D`, a `Texture2DArray`, a `Texture3D` or a
/// `TextureCube`, or their `RW` forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ViewType {
    /// One layer of a 2D or cube texture.
    D2,
    /// Several layers of a 2D or cube texture.
    D2Array,
    /// A 3D texture.
    D3,
    /// The six faces of one cube of a cube texture, sampled together.
    Cube,
}

impl ViewType {
    /// The kind of view of `subresource`, already resolved, of a texture
    /// of `dimension`: one layer is a 2D view and several an array view,
    /// except that the six faces of one cube of a cube texture, sampled
    /// together, are a cube view. Several cubes are a 2D array view, since
    /// cube array views need a feature device selection doesn't require.
    pub(crate) fn of(
        dimension: TextureDimension,
        storage: bool,
        subresource: &Subresource,
    ) -> Self {
        let layers = &subresource.layers;
        match dimension {
            TextureDimension::D3 => Self::D3,
            TextureDimension::Cube
                if !storage && layers.len() == 6 && layers.start.is_multiple_of(6) =>
            {
                Self::Cube
            }
            TextureDimension::D2 | TextureDimension::Cube if layers.len() == 1 => Self::D2,
            TextureDimension::D2 | TextureDimension::Cube => Self::D2Array,
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of view checks and view types (design 0002, R21).

    use super::*;

    /// A texture of `dimension`, with 4 mips and `layer_count` layers,
    /// that can be sampled and stored.
    fn texture(dimension: TextureDimension, layer_count: u32) -> ViewedTexture<'static> {
        ViewedTexture {
            label: "albedo",
            dimension,
            mip_count: 4,
            layer_count,
            usage: TextureUsage::SAMPLED | TextureUsage::STORAGE,
        }
    }

    /// The view type of `view` of `texture`.
    fn view_type(view: &TextureView, texture: &ViewedTexture<'_>) -> ViewType {
        view.resolve(texture).1
    }

    /// Views resolve their open ranges, so the same part named two ways is
    /// one view.
    #[test]
    fn views_resolve_their_ranges() {
        let texture = texture(TextureDimension::D2, 3);
        let whole = TextureView::Sampled(Subresource {
            mips: 0..4,
            layers: 0..3,
        });
        assert_eq!(TextureView::sampled().resolve(&texture).0, whole);
        assert_eq!(
            TextureView::Storage(Subresource::mip(2))
                .resolve(&texture)
                .0,
            TextureView::Storage(Subresource {
                mips: 2..3,
                layers: 0..3,
            }),
        );
    }

    /// One layer of a 2D texture is a 2D view, and several an array view.
    #[test]
    fn layers_of_2d_textures_are_2d_or_array_views() {
        let texture = texture(TextureDimension::D2, 4);
        assert_eq!(
            view_type(&TextureView::Sampled(Subresource::layer(2)), &texture),
            ViewType::D2
        );
        assert_eq!(
            view_type(&TextureView::sampled(), &texture),
            ViewType::D2Array
        );
        let single = self::texture(TextureDimension::D2, 1);
        assert_eq!(view_type(&TextureView::sampled(), &single), ViewType::D2);
    }

    /// A 3D texture is always a 3D view.
    #[test]
    fn volume_textures_are_3d_views() {
        let texture = texture(TextureDimension::D3, 1);
        assert_eq!(view_type(&TextureView::sampled(), &texture), ViewType::D3);
        assert_eq!(
            view_type(&TextureView::Storage(Subresource::mip(1)), &texture),
            ViewType::D3
        );
    }

    /// The six faces of one cube, sampled, are a cube view. A face is a 2D
    /// view, and several cubes, faces that don't start a cube, or a
    /// storage view of all six faces are 2D array views.
    #[test]
    fn cube_faces_sampled_together_are_cube_views() {
        let cubes = texture(TextureDimension::Cube, 12);
        let sampled = |layers| TextureView::Sampled(Subresource { mips: 0..1, layers });
        assert_eq!(view_type(&sampled(0..6), &cubes), ViewType::Cube);
        assert_eq!(view_type(&sampled(6..12), &cubes), ViewType::Cube);
        assert_eq!(view_type(&sampled(4..5), &cubes), ViewType::D2);
        assert_eq!(view_type(&sampled(0..12), &cubes), ViewType::D2Array);
        assert_eq!(view_type(&sampled(3..9), &cubes), ViewType::D2Array);
        let storage = TextureView::Storage(Subresource {
            mips: 0..1,
            layers: 0..6,
        });
        assert_eq!(view_type(&storage, &cubes), ViewType::D2Array);
    }

    /// A storage view of more than one mip panics, naming the texture and
    /// the mips.
    #[test]
    #[should_panic(
        expected = "texture \"albedo\" has a storage view of mips 0..4, but a storage view covers \
                    exactly one mip"
    )]
    fn storage_views_of_several_mips_panic() {
        TextureView::Storage(Subresource::ALL).resolve(&texture(TextureDimension::D2, 1));
    }

    /// A view the texture's usage doesn't allow panics, naming the usage.
    #[test]
    fn views_need_their_usage() {
        for (usage, view, message) in [
            (
                TextureUsage::SAMPLED,
                TextureView::Storage(Subresource::mip(0)),
                "texture \"albedo\" has no STORAGE usage, so it can't have a storage view",
            ),
            (
                TextureUsage::STORAGE,
                TextureView::sampled(),
                "texture \"albedo\" has no SAMPLED usage, so it can't have a sampled view",
            ),
        ] {
            let texture = ViewedTexture {
                usage,
                ..texture(TextureDimension::D2, 1)
            };
            let panic = std::panic::catch_unwind(|| view.resolve(&texture))
                .expect_err("the view should panic");
            assert_eq!(panic.downcast_ref::<String>().unwrap(), message);
        }
    }

    /// A view past the texture's mips panics, as any subresource does.
    #[test]
    #[should_panic(
        expected = "texture \"albedo\" has 4 mips, so a subresource can't name mips 4..5"
    )]
    fn views_out_of_range_panic() {
        TextureView::Sampled(Subresource::mip(4)).resolve(&texture(TextureDimension::D2, 1));
    }
}
