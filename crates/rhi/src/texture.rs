//! Textures: their handles, and how they're created (design 0002,
//! "Handles", "Memory" and "Data structures and layout").
//!
//! Textures are always in GPU-only memory. Their contents arrive by copies
//! from upload buffers, and shaders reach them through views in the
//! descriptor heaps.

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

    /// The number of mips in a full chain for this extent, down to 1×1×1.
    pub const fn full_mip_count(self) -> u32 {
        let largest = max(max(self.width, self.height), self.depth);
        if largest == 0 {
            0
        } else {
            u32::BITS - largest.leading_zeros()
        }
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
}

/// How a texture is used. A set of the constants below, combined with
/// [`union`](Self::union). A texture must have at least one.
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
    /// The number of mips, from 1 to [`Extent3d::full_mip_count`].
    pub mip_levels: u32,
    /// The number of layers: 1 for a 3D texture, and a multiple of 6 for
    /// a cube map.
    pub array_layers: u32,
    /// How it's used.
    pub usage: TextureUsage,
}

impl TextureDesc<'_> {
    /// Checks that the description is valid, as [`Device::create_texture`]
    /// requires.
    ///
    /// [`Device::create_texture`]: crate::Device::create_texture
    ///
    /// # Panics
    ///
    /// Panics on the first problem, naming the texture.
    pub(crate) fn check(&self) {
        let label = self.label;
        let Extent3d {
            width,
            height,
            depth,
        } = self.extent;
        assert!(
            width > 0 && height > 0 && depth > 0,
            "texture \"{label}\" has an empty extent, {width}x{height}x{depth}",
        );
        assert!(self.array_layers > 0, "texture \"{label}\" has no layers");
        let full = self.extent.full_mip_count();
        assert!(
            (1..=full).contains(&self.mip_levels),
            "texture \"{label}\" has {} mips, but a {width}x{height}x{depth} texture has 1 to {full}",
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

    /// A valid description of a 256×256 2D texture with a full mip chain,
    /// for tests to change one field of.
    fn valid() -> TextureDesc<'static> {
        TextureDesc {
            label: "albedo",
            dimension: TextureDimension::D2,
            format: Format::Rgba8Srgb,
            extent: Extent3d::new_2d(256, 256),
            mip_levels: 9,
            array_layers: 1,
            usage: TextureUsage::SAMPLED.union(TextureUsage::COPY_DESTINATION),
        }
    }

    /// A full chain halves the largest side down to 1, rounding down.
    #[test]
    fn full_mip_chains_reach_one_texel() {
        assert_eq!(Extent3d::new_2d(1, 1).full_mip_count(), 1);
        assert_eq!(Extent3d::new_2d(256, 256).full_mip_count(), 9);
        assert_eq!(Extent3d::new_2d(300, 7).full_mip_count(), 9);
        let volume = Extent3d {
            width: 4,
            height: 4,
            depth: 64,
        };
        assert_eq!(volume.full_mip_count(), 7);
    }

    /// A union contains each of its parts, and no other usage.
    #[test]
    fn usage_unions_contain_exactly_their_parts() {
        let usage = TextureUsage::SAMPLED.union(TextureUsage::STORAGE);
        assert!(usage.contains(TextureUsage::SAMPLED));
        assert!(usage.contains(TextureUsage::STORAGE));
        assert!(!usage.contains(TextureUsage::COPY_SOURCE));
        assert_eq!(TextureUsage::default(), TextureUsage::NONE);
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
        valid().check();
        TextureDesc {
            array_layers: 4,
            mip_levels: 1,
            ..valid()
        }
        .check();
        TextureDesc {
            dimension: TextureDimension::Cube,
            array_layers: 12,
            ..valid()
        }
        .check();
        TextureDesc {
            dimension: TextureDimension::D3,
            extent: Extent3d {
                width: 32,
                height: 32,
                depth: 32,
            },
            mip_levels: 6,
            ..valid()
        }
        .check();
        TextureDesc {
            format: Format::Depth32Float,
            mip_levels: 1,
            usage: TextureUsage::DEPTH_ATTACHMENT.union(TextureUsage::SAMPLED),
            ..valid()
        }
        .check();
    }

    /// Checks that `desc` panics with a message containing `expected`.
    fn assert_rejected(desc: TextureDesc<'_>, expected: &str) {
        let message =
            std::panic::catch_unwind(|| desc.check()).expect_err("the description was accepted");
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
                ..valid()
            },
            "texture \"albedo\" has an empty extent, 0x4x1",
        );
        assert_rejected(
            TextureDesc {
                array_layers: 0,
                ..valid()
            },
            "texture \"albedo\" has no layers",
        );
        assert_rejected(
            TextureDesc {
                mip_levels: 10,
                ..valid()
            },
            "has 10 mips, but a 256x256x1 texture has 1 to 9",
        );
        assert_rejected(
            TextureDesc {
                mip_levels: 0,
                ..valid()
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
                ..valid()
            },
            "2D texture \"albedo\" has a depth of 2, not 1",
        );
        assert_rejected(
            TextureDesc {
                dimension: TextureDimension::D3,
                array_layers: 2,
                ..valid()
            },
            "3D texture \"albedo\" has 2 layers, not 1",
        );
        assert_rejected(
            TextureDesc {
                dimension: TextureDimension::Cube,
                extent: Extent3d::new_2d(256, 128),
                array_layers: 6,
                ..valid()
            },
            "is 256x128x1, not square",
        );
        assert_rejected(
            TextureDesc {
                dimension: TextureDimension::Cube,
                array_layers: 4,
                ..valid()
            },
            "has 4 layers, not a multiple of 6",
        );
        assert_rejected(
            TextureDesc {
                usage: TextureUsage::NONE,
                ..valid()
            },
            "texture \"albedo\" has no usage",
        );
        assert_rejected(
            TextureDesc {
                usage: TextureUsage::COLOUR_ATTACHMENT,
                format: Format::Depth32Float,
                ..valid()
            },
            "is a colour attachment with the depth format Depth32Float",
        );
        assert_rejected(
            TextureDesc {
                usage: TextureUsage::DEPTH_ATTACHMENT,
                ..valid()
            },
            "is a depth attachment with the colour format Rgba8Srgb",
        );
    }
}
