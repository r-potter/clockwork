//! Textures: their handles, and how they're created (design 0002,
//! "Handles", "Memory" and "Data structures and layout").
//!
//! Textures are always in GPU-only memory. Their contents arrive by copies
//! from upload buffers, and shaders reach them through views in the
//! descriptor heaps.

use std::ops::{BitOr, BitOrAssign};

use crate::handle::Handle;

/// A texture. Refers to an entry in the backend's texture table, and is
/// stale once destroyed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Texture(Handle);

impl Texture {
    /// The texture for `handle`, issued by a backend's texture table.
    pub(crate) fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    /// The handle into the backend's texture table.
    pub(crate) fn handle(self) -> Handle {
        self.0
    }
}

/// The shape of a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    /// A 2D texture, or an array of them. Its depth is 1.
    D2,
    /// A 3D texture. It has one layer.
    D3,
    /// A cube map, or an array of them: square 2D layers, six per cube, in
    /// the order +X, -X, +Y, -Y, +Z, -Z.
    Cube,
}

/// The size of a texture, or of a region of one, in texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent3d {
    /// The width, in texels.
    pub width: u32,
    /// The height, in texels.
    pub height: u32,
    /// The depth, in texels. 1 for anything but a 3D texture.
    pub depth: u32,
}

impl Extent3d {
    /// A 2D extent, with a depth of 1.
    pub const fn new_2d(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }

    /// The most mips a texture of this extent can have: a full chain, down
    /// to 1×1×1. An empty extent, which no texture may have, has none.
    pub const fn max_mip_count(self) -> u32 {
        if self.is_empty() {
            return 0;
        }
        // One mip per bit up to the largest side's highest set bit: the
        // number of times it can be halved, plus one.
        let largest = max(max(self.width, self.height), self.depth);
        u32::BITS - largest.leading_zeros()
    }

    /// Whether any side is zero, so the extent holds no texels. No texture
    /// may be empty.
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0 || self.depth == 0
    }
}

/// The larger of `a` and `b`, in a `const fn`.
const fn max(a: u32, b: u32) -> u32 {
    if a > b { a } else { b }
}

/// A texel format.
///
/// Only the formats in use are listed. Block-compressed formats and the
/// window surface's format are added by the slices that use them (design
/// 0002, "Data structures and layout").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    /// Four 8-bit channels, normalized to `0..=1`.
    Rgba8Unorm,
    /// Four 8-bit channels, with colour in sRGB.
    Rgba8Srgb,
    /// Four 8-bit channels, blue first, normalized to `0..=1`.
    Bgra8Unorm,
    /// Four 8-bit channels, blue first, with colour in sRGB.
    Bgra8Srgb,
    /// Four 16-bit floats.
    Rgba16Float,
    /// Four 32-bit floats.
    Rgba32Float,
    /// One 32-bit unsigned integer.
    R32Uint,
    /// 32-bit floating-point depth, with no stencil.
    Depth32Float,
}

impl Format {
    /// Every format, for tests.
    #[cfg(test)]
    pub(crate) const ALL: [Self; 8] = [
        Self::Rgba8Unorm,
        Self::Rgba8Srgb,
        Self::Bgra8Unorm,
        Self::Bgra8Srgb,
        Self::Rgba16Float,
        Self::Rgba32Float,
        Self::R32Uint,
        Self::Depth32Float,
    ];

    /// Whether the format holds depth, rather than colour.
    pub const fn is_depth(self) -> bool {
        matches!(self, Self::Depth32Float)
    }

    /// The format's block layout, and what it holds (design 0002, "Texture
    /// copies"): the one table of format information that copies, views and
    /// barriers use.
    pub const fn info(self) -> FormatInfo {
        let block_bytes = match self {
            Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Bgra8Unorm
            | Self::Bgra8Srgb
            | Self::R32Uint
            | Self::Depth32Float => 4,
            Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
        };
        FormatInfo {
            block_extent: [1, 1],
            block_bytes,
            depth: self.is_depth(),
            srgb: matches!(self, Self::Rgba8Srgb | Self::Bgra8Srgb),
        }
    }
}

/// How a format lays out its texels, and what they hold.
///
/// Texels are stored in blocks: one texel for uncompressed formats, and 4×4
/// for block-compressed ones. Copies count rows and images in blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FormatInfo {
    /// A block's width and height, in texels.
    pub block_extent: [u32; 2],
    /// A block's size, in bytes.
    pub block_bytes: u32,
    /// Whether it holds depth, rather than colour.
    pub depth: bool,
    /// Whether its colour is in sRGB, which sampling converts to linear.
    pub srgb: bool,
}

/// How a texture is used. A set of the constants below, combined with `|`,
/// or with [`union`](Self::union) in a `const`. A texture must have at least
/// one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TextureUsage {
    /// Sampled by shaders.
    sampled: bool,
    /// Read and written by shaders, as storage.
    storage: bool,
    /// Rendered to, as a colour attachment.
    colour_attachment: bool,
    /// Rendered to, as a depth attachment.
    depth_attachment: bool,
    /// Copied from.
    copy_source: bool,
    /// Copied to.
    copy_destination: bool,
}

impl TextureUsage {
    /// No usage, which no texture may have.
    pub const NONE: Self = Self {
        sampled: false,
        storage: false,
        colour_attachment: false,
        depth_attachment: false,
        copy_source: false,
        copy_destination: false,
    };
    /// Sampled by shaders.
    pub const SAMPLED: Self = Self {
        sampled: true,
        ..Self::NONE
    };
    /// Read and written by shaders, as storage.
    pub const STORAGE: Self = Self {
        storage: true,
        ..Self::NONE
    };
    /// Rendered to, as a colour attachment.
    pub const COLOUR_ATTACHMENT: Self = Self {
        colour_attachment: true,
        ..Self::NONE
    };
    /// Rendered to, as a depth attachment.
    pub const DEPTH_ATTACHMENT: Self = Self {
        depth_attachment: true,
        ..Self::NONE
    };
    /// Copied from.
    pub const COPY_SOURCE: Self = Self {
        copy_source: true,
        ..Self::NONE
    };
    /// Copied to.
    pub const COPY_DESTINATION: Self = Self {
        copy_destination: true,
        ..Self::NONE
    };

    /// The usages in either set.
    pub const fn union(self, other: Self) -> Self {
        Self {
            sampled: self.sampled || other.sampled,
            storage: self.storage || other.storage,
            colour_attachment: self.colour_attachment || other.colour_attachment,
            depth_attachment: self.depth_attachment || other.depth_attachment,
            copy_source: self.copy_source || other.copy_source,
            copy_destination: self.copy_destination || other.copy_destination,
        }
    }

    /// Whether every usage in `other` is in this set.
    pub const fn contains(self, other: Self) -> bool {
        (self.sampled || !other.sampled)
            && (self.storage || !other.storage)
            && (self.colour_attachment || !other.colour_attachment)
            && (self.depth_attachment || !other.depth_attachment)
            && (self.copy_source || !other.copy_source)
            && (self.copy_destination || !other.copy_destination)
    }
}

impl BitOr for TextureUsage {
    type Output = Self;

    /// The usages in either set, as [`union`](Self::union).
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for TextureUsage {
    /// Adds the usages in `other`.
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

/// How a texture is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc<'a> {
    /// The texture's name, in debug tools and messages.
    pub label: &'a str,
    /// Its shape.
    pub dimension: TextureDimension,
    /// Its texel format.
    pub format: Format,
    /// Its size, in texels, at mip 0.
    pub extent: Extent3d,
    /// The number of mips, from 1 to [`Extent3d::max_mip_count`].
    pub mip_levels: u32,
    /// The number of layers: 1 for a 3D texture, and a multiple of 6 for
    /// a cube map.
    pub array_layers: u32,
    /// How it's used.
    pub usage: TextureUsage,
}

impl TextureDesc<'_> {
    /// Checks that the description is valid, as
    /// [`GpuResources::create_texture`] requires.
    ///
    /// [`GpuResources::create_texture`]: crate::GpuResources::create_texture
    ///
    /// # Panics
    ///
    /// Panics on the first problem, naming the texture.
    pub(crate) fn validate(&self) {
        let label = self.label;
        let Extent3d {
            width,
            height,
            depth,
        } = self.extent;
        assert!(
            !self.extent.is_empty(),
            "texture \"{label}\" has an empty extent, {width}x{height}x{depth}",
        );
        assert!(self.array_layers > 0, "texture \"{label}\" has no layers");
        let max_mips = self.extent.max_mip_count();
        assert!(
            (1..=max_mips).contains(&self.mip_levels),
            "texture \"{label}\" has {} mips, but a {width}x{height}x{depth} texture has 1 to {max_mips}",
            self.mip_levels,
        );
        match self.dimension {
            TextureDimension::D2 => assert!(
                depth == 1,
                "2D texture \"{label}\" has a depth of {depth}, not 1"
            ),
            TextureDimension::D3 => assert!(
                self.array_layers == 1,
                "3D texture \"{label}\" has {} layers, not 1",
                self.array_layers,
            ),
            TextureDimension::Cube => {
                assert!(
                    width == height && depth == 1,
                    "cube texture \"{label}\" is {width}x{height}x{depth}, not square with a depth of 1",
                );
                assert!(
                    self.array_layers.is_multiple_of(6),
                    "cube texture \"{label}\" has {} layers, not a multiple of 6",
                    self.array_layers,
                );
            }
        }
        assert!(
            self.usage != TextureUsage::NONE,
            "texture \"{label}\" has no usage"
        );
        assert!(
            !(self.usage.contains(TextureUsage::COLOUR_ATTACHMENT) && self.format.is_depth()),
            "texture \"{label}\" is a colour attachment with the depth format {:?}",
            self.format,
        );
        assert!(
            !self.usage.contains(TextureUsage::DEPTH_ATTACHMENT) || self.format.is_depth(),
            "texture \"{label}\" is a depth attachment with the colour format {:?}",
            self.format,
        );
    }
}

// Handles and descriptions are plain data, shared between threads (design
// 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Texture>();
    assert_send_sync::<TextureDesc<'_>>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of extents, usage sets and texture descriptions.

    use super::*;

    /// The description tests start from, changing a field or two: a valid
    /// 256×256 2D texture with a full mip chain.
    fn test_desc() -> TextureDesc<'static> {
        TextureDesc {
            label: "albedo",
            dimension: TextureDimension::D2,
            format: Format::Rgba8Srgb,
            extent: Extent3d::new_2d(256, 256),
            mip_levels: 9,
            array_layers: 1,
            usage: TextureUsage::SAMPLED | TextureUsage::COPY_DESTINATION,
        }
    }

    /// A full chain halves the largest side down to 1, rounding down, and
    /// an empty extent has none.
    #[test]
    fn full_mip_chains_reach_one_texel() {
        assert_eq!(Extent3d::new_2d(1, 1).max_mip_count(), 1);
        assert_eq!(Extent3d::new_2d(256, 256).max_mip_count(), 9);
        assert_eq!(Extent3d::new_2d(300, 7).max_mip_count(), 9);
        let volume = Extent3d {
            width: 4,
            height: 4,
            depth: 64,
        };
        assert_eq!(volume.max_mip_count(), 7);
        for empty in [
            Extent3d::new_2d(0, 256),
            Extent3d::new_2d(256, 0),
            Extent3d { depth: 0, ..volume },
        ] {
            assert!(empty.is_empty());
            assert_eq!(empty.max_mip_count(), 0, "{empty:?}");
        }
        assert!(!volume.is_empty());
    }

    /// A union contains each of its parts, and no other usage, whether made
    /// with `|`, `|=` or `union`.
    #[test]
    fn usage_unions_contain_exactly_their_parts() {
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
        assert!(usage.contains(TextureUsage::SAMPLED));
        assert!(usage.contains(TextureUsage::STORAGE));
        assert!(!usage.contains(TextureUsage::COPY_SOURCE));
        assert_eq!(TextureUsage::default(), TextureUsage::NONE);
        assert_eq!(usage, TextureUsage::SAMPLED.union(TextureUsage::STORAGE));
        let mut added = TextureUsage::SAMPLED;
        added |= TextureUsage::STORAGE;
        assert_eq!(added, usage);
    }

    /// Only depth formats are depth formats.
    #[test]
    fn only_depth32_holds_depth() {
        for format in Format::ALL {
            assert_eq!(
                format.is_depth(),
                format == Format::Depth32Float,
                "{format:?}"
            );
        }
    }

    /// Valid descriptions of each dimension pass.
    #[test]
    fn valid_descriptions_pass() {
        test_desc().validate();
        TextureDesc {
            array_layers: 4,
            mip_levels: 1,
            ..test_desc()
        }
        .validate();
        TextureDesc {
            dimension: TextureDimension::Cube,
            array_layers: 12,
            ..test_desc()
        }
        .validate();
        TextureDesc {
            dimension: TextureDimension::D3,
            extent: Extent3d {
                width: 32,
                height: 32,
                depth: 32,
            },
            mip_levels: 6,
            ..test_desc()
        }
        .validate();
        TextureDesc {
            format: Format::Depth32Float,
            mip_levels: 1,
            usage: TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLED,
            ..test_desc()
        }
        .validate();
    }

    /// Checks that `desc` panics with a message containing `expected`.
    fn assert_rejected(desc: TextureDesc<'_>, expected: &str) {
        let message =
            std::panic::catch_unwind(|| desc.validate()).expect_err("the description was accepted");
        let message = message
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert!(
            message.contains(expected),
            "{message:?} doesn't contain {expected:?}"
        );
    }

    /// Each kind of invalid description panics, naming the texture and the
    /// problem.
    #[test]
    fn invalid_descriptions_panic() {
        assert_rejected(
            TextureDesc {
                extent: Extent3d::new_2d(0, 4),
                ..test_desc()
            },
            "texture \"albedo\" has an empty extent, 0x4x1",
        );
        assert_rejected(
            TextureDesc {
                extent: Extent3d::new_2d(4, 0),
                ..test_desc()
            },
            "texture \"albedo\" has an empty extent, 4x0x1",
        );
        assert_rejected(
            TextureDesc {
                dimension: TextureDimension::D3,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth: 0,
                },
                mip_levels: 1,
                ..test_desc()
            },
            "texture \"albedo\" has an empty extent, 4x4x0",
        );
        assert_rejected(
            TextureDesc {
                array_layers: 0,
                ..test_desc()
            },
            "texture \"albedo\" has no layers",
        );
        assert_rejected(
            TextureDesc {
                mip_levels: 10,
                ..test_desc()
            },
            "has 10 mips, but a 256x256x1 texture has 1 to 9",
        );
        assert_rejected(
            TextureDesc {
                mip_levels: 0,
                ..test_desc()
            },
            "has 0 mips",
        );
        assert_rejected(
            TextureDesc {
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth: 2,
                },
                mip_levels: 1,
                ..test_desc()
            },
            "2D texture \"albedo\" has a depth of 2, not 1",
        );
        assert_rejected(
            TextureDesc {
                dimension: TextureDimension::D3,
                array_layers: 2,
                ..test_desc()
            },
            "3D texture \"albedo\" has 2 layers, not 1",
        );
        assert_rejected(
            TextureDesc {
                dimension: TextureDimension::Cube,
                extent: Extent3d::new_2d(256, 128),
                array_layers: 6,
                ..test_desc()
            },
            "is 256x128x1, not square",
        );
        assert_rejected(
            TextureDesc {
                dimension: TextureDimension::Cube,
                array_layers: 4,
                ..test_desc()
            },
            "has 4 layers, not a multiple of 6",
        );
        assert_rejected(
            TextureDesc {
                usage: TextureUsage::NONE,
                ..test_desc()
            },
            "texture \"albedo\" has no usage",
        );
        assert_rejected(
            TextureDesc {
                usage: TextureUsage::COLOUR_ATTACHMENT,
                format: Format::Depth32Float,
                ..test_desc()
            },
            "is a colour attachment with the depth format Depth32Float",
        );
        assert_rejected(
            TextureDesc {
                usage: TextureUsage::DEPTH_ATTACHMENT,
                ..test_desc()
            },
            "is a depth attachment with the colour format Rgba8Srgb",
        );
    }
}
