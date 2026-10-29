//! Copies between buffers and textures: how texture data is laid out in a
//! buffer, the region of a texture copied, and the checks every copy passes
//! (design 0002, "Texture copies").
//!
//! Copies are where Vulkan is strictest about layout, and where mistakes
//! give skewed textures rather than errors, so every copy states its layout,
//! counted in texel blocks, and one pure function checks it, panicking with
//! the texture's label on a violation.

use std::ops::Range;

use crate::{Extent3d, Format, FormatInfo, Texture};

/// How texture data is laid out in a buffer, for a copy. Rows and images
/// are counted in texel blocks: one texel for uncompressed formats, and a
/// 4×4 block for block-compressed ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureDataLayout {
    /// Where the data starts, in bytes from the buffer address given with
    /// it.
    pub offset: u64,
    /// The distance, in bytes, between the starts of two rows of blocks.
    pub bytes_per_row: u32,
    /// The rows of blocks between the starts of two layers, or of two depth
    /// slices of a 3D texture.
    pub rows_per_image: u32,
}

impl TextureDataLayout {
    /// The layout of tightly packed data for `extent` texels of `format`,
    /// starting at offset 0.
    pub const fn tight(format: Format, extent: Extent3d) -> Self {
        tight(format.info(), extent)
    }
}

/// [`TextureDataLayout::tight`] for a format described by `format`.
const fn tight(format: FormatInfo, extent: Extent3d) -> TextureDataLayout {
    TextureDataLayout {
        offset: 0,
        bytes_per_row: extent.width.div_ceil(format.block_extent[0]) * format.block_bytes,
        rows_per_image: extent.height.div_ceil(format.block_extent[1]),
    }
}

/// The texture side of a copy: a region of one mip, in some layers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureCopy {
    /// The texture copied to or from.
    pub texture: Texture,
    /// The mip, from the largest, 0.
    pub mip: u32,
    /// The array layers. A 3D texture has one, and copies depth slices
    /// through `origin` and `extent` instead.
    pub layers: Range<u32>,
    /// The region's first texel in the mip.
    pub origin: [u32; 3],
    /// The region's size, in texels.
    pub extent: Extent3d,
}

/// What a copy check needs to know about the texture it copies.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CopiedTexture<'a> {
    /// The texture's label, for messages.
    pub label: &'a str,
    /// Its format's block layout.
    pub format: FormatInfo,
    /// The size of its largest mip.
    pub extent: Extent3d,
    /// How many mips it has.
    pub mip_count: u32,
    /// How many array layers it has.
    pub layer_count: u32,
}

/// Checks a copy between `texture`, in the region `copy` names, and buffer
/// data laid out as `layout`, and returns the number of bytes of the buffer,
/// from the address given with the layout, that the copy reads or writes.
///
/// # Panics
///
/// Panics, naming the texture, if:
/// - the mip or layers are outside the texture, or the region is empty or
///   outside the mip
/// - the origin isn't a multiple of the format's block extent, or the
///   extent isn't, other than where it reaches the edge of the mip, as
///   Vulkan allows for mips smaller than a block
/// - the offset isn't a multiple of the block size in bytes, or of 4 for a
///   depth format
/// - `bytes_per_row` isn't a multiple of the block size, or is less than a
///   row of blocks
/// - `rows_per_image` is less than the region's rows of blocks
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "translation checks texture copies in the next PR")
)]
pub(crate) fn check_texture_copy(
    texture: &CopiedTexture<'_>,
    copy: &TextureCopy,
    layout: &TextureDataLayout,
) -> u64 {
    let label = texture.label;
    let format = texture.format;
    assert!(
        copy.mip < texture.mip_count,
        "texture \"{label}\" has {} mips, so a copy can't use mip {}",
        texture.mip_count,
        copy.mip,
    );
    assert!(
        copy.layers.start < copy.layers.end && copy.layers.end <= texture.layer_count,
        "texture \"{label}\" has {} layers, so a copy can't use layers {:?}",
        texture.layer_count,
        copy.layers,
    );
    assert!(
        !copy.extent.is_empty(),
        "a copy of texture \"{label}\" has an empty extent, {:?}",
        copy.extent,
    );

    let mip = mip_extent(texture.extent, copy.mip);
    let sizes = [mip.width, mip.height, mip.depth];
    let extents = [copy.extent.width, copy.extent.height, copy.extent.depth];
    for axis in 0..3 {
        assert!(
            copy.origin[axis]
                .checked_add(extents[axis])
                .is_some_and(|end| end <= sizes[axis]),
            "a copy of texture \"{label}\" at {:?}, of {:?}, passes the edge of mip {}, which is \
             {mip:?}",
            copy.origin,
            copy.extent,
            copy.mip,
        );
    }
    for axis in 0..2 {
        let block = format.block_extent[axis];
        let reaches_edge = copy.origin[axis] + extents[axis] == sizes[axis];
        assert!(
            copy.origin[axis].is_multiple_of(block)
                && (extents[axis].is_multiple_of(block) || reaches_edge),
            "a copy of texture \"{label}\" at {:?}, of {:?}, isn't aligned to its {}×{} blocks",
            copy.origin,
            copy.extent,
            format.block_extent[0],
            format.block_extent[1],
        );
    }

    let block_bytes = u64::from(format.block_bytes);
    let offset_alignment = if format.depth {
        block_bytes.max(4)
    } else {
        block_bytes
    };
    assert!(
        layout.offset.is_multiple_of(offset_alignment),
        "a copy of texture \"{label}\" starts at offset {}, which isn't a multiple of \
         {offset_alignment} bytes",
        layout.offset,
    );
    let tight = tight(format, copy.extent);
    assert!(
        layout.bytes_per_row.is_multiple_of(format.block_bytes)
            && layout.bytes_per_row >= tight.bytes_per_row,
        "a copy of texture \"{label}\" has {} bytes per row, but needs a multiple of {} that's \
         at least {}",
        layout.bytes_per_row,
        format.block_bytes,
        tight.bytes_per_row,
    );
    assert!(
        layout.rows_per_image >= tight.rows_per_image,
        "a copy of texture \"{label}\" has {} rows per image, but needs at least {}",
        layout.rows_per_image,
        tight.rows_per_image,
    );

    // Every image but the last is `rows_per_image` rows long, and the last
    // row of the last image is only as long as the region's row of blocks.
    let images = u64::from(copy.layers.end - copy.layers.start) * u64::from(copy.extent.depth);
    let bytes_per_row = u64::from(layout.bytes_per_row);
    let rows = (images - 1) * u64::from(layout.rows_per_image) + u64::from(tight.rows_per_image);
    layout.offset + (rows - 1) * bytes_per_row + u64::from(tight.bytes_per_row)
}

/// The extent of mip `mip` of a texture whose largest mip is `extent`: each
/// side halved for each mip, rounding down, and at least 1.
fn mip_extent(extent: Extent3d, mip: u32) -> Extent3d {
    let side = |size: u32| size.checked_shr(mip).unwrap_or(0).max(1);
    Extent3d {
        width: side(extent.width),
        height: side(extent.height),
        depth: side(extent.depth),
    }
}

// Copies are plain data, recorded on any thread (design 0002, "Threading",
// R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TextureDataLayout>();
    assert_send_sync::<TextureCopy>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of copy layouts and their checks (design 0002, R25), for an
    //! uncompressed, a block-compressed and a depth format.
    //!
    //! `Format` has no block-compressed format until `rhi-streaming` adds
    //! BC1 to BC7, so the block-compressed cases describe one with a
    //! [`FormatInfo`]: 4×4 blocks of 16 bytes, as BC7's are.

    use super::*;
    use crate::handle::HandleTable;

    /// 4×4 blocks of 16 bytes, as BC7's are.
    const BC7: FormatInfo = FormatInfo {
        block_extent: [4, 4],
        block_bytes: 16,
        depth: false,
        srgb: false,
    };

    /// A texture of `format`, `extent` at mip 0, with four mips and two
    /// layers, labelled `albedo`.
    fn texture(format: FormatInfo, extent: Extent3d) -> CopiedTexture<'static> {
        CopiedTexture {
            label: "albedo",
            format,
            extent,
            mip_count: 4,
            layer_count: 2,
        }
    }

    /// A 2D texture of RGBA8 texels, 64×32.
    fn rgba8() -> CopiedTexture<'static> {
        texture(Format::Rgba8Unorm.info(), Extent3d::new_2d(64, 32))
    }

    /// A copy of `extent` at `origin` in layer 0 of `mip`.
    fn copy(mip: u32, origin: [u32; 2], extent: Extent3d) -> TextureCopy {
        TextureCopy {
            texture: Texture::from_handle(HandleTable::new("texture").insert("albedo", ())),
            mip,
            layers: 0..1,
            origin: [origin[0], origin[1], 0],
            extent,
        }
    }

    /// Tight layouts have rows of whole blocks, with no padding.
    #[test]
    fn tight_layouts_pack_blocks() {
        assert_eq!(
            TextureDataLayout::tight(Format::Rgba16Float, Extent3d::new_2d(10, 3)),
            TextureDataLayout {
                offset: 0,
                bytes_per_row: 80,
                rows_per_image: 3,
            },
        );
        // A partial block at the edge still takes a whole block.
        assert_eq!(
            tight(BC7, Extent3d::new_2d(10, 6)),
            TextureDataLayout {
                offset: 0,
                bytes_per_row: 48,
                rows_per_image: 2,
            },
        );
    }

    /// The format table gives each format's block size, and says which are
    /// depth and which sRGB.
    #[test]
    fn formats_describe_their_blocks() {
        for format in Format::ALL {
            let info = format.info();
            assert_eq!(info.block_extent, [1, 1], "{format:?}");
            assert_eq!(info.depth, format.is_depth(), "{format:?}");
        }
        assert_eq!(Format::Rgba32Float.info().block_bytes, 16);
        assert!(Format::Bgra8Srgb.info().srgb);
        assert!(!Format::Bgra8Unorm.info().srgb);
    }

    /// A tightly packed copy of a whole mip reads exactly its bytes.
    #[test]
    fn tight_copies_use_their_bytes() {
        let extent = Extent3d::new_2d(32, 16);
        let layout = TextureDataLayout::tight(Format::Rgba8Unorm, extent);
        assert_eq!(
            check_texture_copy(&rgba8(), &copy(1, [0, 0], extent), &layout),
            32 * 16 * 4,
        );
    }

    /// Padded rows and images, an offset, and several layers count every
    /// row but the last in full.
    #[test]
    fn padded_copies_count_their_padding() {
        let layout = TextureDataLayout {
            offset: 256,
            bytes_per_row: 128,
            rows_per_image: 10,
        };
        let copy = TextureCopy {
            layers: 0..2,
            ..copy(0, [8, 4], Extent3d::new_2d(16, 8))
        };
        // Ten rows of the first layer, seven full rows of the second, then
        // a row of 16 texels.
        assert_eq!(
            check_texture_copy(&rgba8(), &copy, &layout),
            256 + 17 * 128 + 16 * 4,
        );
    }

    /// Block-compressed copies are counted in blocks, and a mip smaller than
    /// a block is copied whole, though its extent isn't a multiple of one.
    #[test]
    fn block_compressed_copies_count_blocks() {
        let texture = texture(BC7, Extent3d::new_2d(16, 8));
        let layout = tight(BC7, Extent3d::new_2d(16, 8));
        assert_eq!(
            check_texture_copy(&texture, &copy(0, [0, 0], Extent3d::new_2d(16, 8)), &layout),
            4 * 2 * 16,
        );
        // Mip 2 is 4×2, and mip 3 is 2×1: both smaller than a block.
        let small = Extent3d::new_2d(2, 1);
        assert_eq!(
            check_texture_copy(&texture, &copy(3, [0, 0], small), &tight(BC7, small)),
            16,
        );
        // A region that ends at a mip's edge needn't be whole blocks.
        let edge = Extent3d::new_2d(4, 2);
        check_texture_copy(&texture, &copy(2, [0, 0], edge), &tight(BC7, edge));
    }

    /// A depth format's offset must be a multiple of 4.
    #[test]
    fn depth_copies_align_their_offset() {
        let texture = texture(Format::Depth32Float.info(), Extent3d::new_2d(8, 8));
        let extent = Extent3d::new_2d(8, 8);
        let layout = TextureDataLayout {
            offset: 4,
            ..TextureDataLayout::tight(Format::Depth32Float, extent)
        };
        assert_eq!(
            check_texture_copy(&texture, &copy(0, [0, 0], extent), &layout),
            4 + 8 * 8 * 4
        );
    }

    /// A mip the texture doesn't have panics, naming the texture.
    #[test]
    #[should_panic(expected = "texture \"albedo\" has 4 mips, so a copy can't use mip 4")]
    fn mips_past_the_last_panic() {
        let extent = Extent3d::new_2d(1, 1);
        check_texture_copy(
            &rgba8(),
            &copy(4, [0, 0], extent),
            &TextureDataLayout::tight(Format::Rgba8Unorm, extent),
        );
    }

    /// Layers past the texture's last panic.
    #[test]
    #[should_panic(expected = "texture \"albedo\" has 2 layers, so a copy can't use layers 1..3")]
    fn layers_past_the_last_panic() {
        let extent = Extent3d::new_2d(1, 1);
        let copy = TextureCopy {
            layers: 1..3,
            ..copy(0, [0, 0], extent)
        };
        check_texture_copy(
            &rgba8(),
            &copy,
            &TextureDataLayout::tight(Format::Rgba8Unorm, extent),
        );
    }

    /// A region that passes the edge of its mip panics.
    #[test]
    #[should_panic(
        expected = "a copy of texture \"albedo\" at [24, 0, 0], of Extent3d { width: 16, height: \
                    1, depth: 1 }, passes the edge of mip 1"
    )]
    fn regions_past_the_edge_panic() {
        let extent = Extent3d::new_2d(16, 1);
        check_texture_copy(
            &rgba8(),
            &copy(1, [24, 0], extent),
            &TextureDataLayout::tight(Format::Rgba8Unorm, extent),
        );
    }

    /// An empty region panics.
    #[test]
    #[should_panic(expected = "a copy of texture \"albedo\" has an empty extent")]
    fn empty_regions_panic() {
        let extent = Extent3d::new_2d(0, 4);
        check_texture_copy(
            &rgba8(),
            &copy(0, [0, 0], extent),
            &TextureDataLayout::tight(Format::Rgba8Unorm, extent),
        );
    }

    /// A block-compressed region that starts inside a block panics.
    #[test]
    #[should_panic(expected = "isn't aligned to its 4×4 blocks")]
    fn unaligned_origins_panic() {
        let texture = texture(BC7, Extent3d::new_2d(16, 16));
        let extent = Extent3d::new_2d(4, 4);
        check_texture_copy(&texture, &copy(0, [2, 0], extent), &tight(BC7, extent));
    }

    /// A block-compressed region that ends inside a block, short of the
    /// mip's edge, panics.
    #[test]
    #[should_panic(expected = "isn't aligned to its 4×4 blocks")]
    fn unaligned_extents_panic() {
        let texture = texture(BC7, Extent3d::new_2d(16, 16));
        let extent = Extent3d::new_2d(6, 4);
        check_texture_copy(&texture, &copy(0, [0, 0], extent), &tight(BC7, extent));
    }

    /// An offset that isn't a multiple of the block size panics.
    #[test]
    #[should_panic(
        expected = "a copy of texture \"albedo\" starts at offset 8, which isn't a multiple of 16 \
                    bytes"
    )]
    fn unaligned_offsets_panic() {
        let texture = texture(BC7, Extent3d::new_2d(16, 16));
        let extent = Extent3d::new_2d(16, 16);
        let layout = TextureDataLayout {
            offset: 8,
            ..tight(BC7, extent)
        };
        check_texture_copy(&texture, &copy(0, [0, 0], extent), &layout);
    }

    /// Rows shorter than a row of blocks panic.
    #[test]
    #[should_panic(
        expected = "a copy of texture \"albedo\" has 60 bytes per row, but needs a multiple of 4 \
                    that's at least 64"
    )]
    fn short_rows_panic() {
        let extent = Extent3d::new_2d(16, 4);
        let layout = TextureDataLayout {
            bytes_per_row: 60,
            ..TextureDataLayout::tight(Format::Rgba8Unorm, extent)
        };
        check_texture_copy(&rgba8(), &copy(0, [0, 0], extent), &layout);
    }

    /// Rows that aren't a multiple of the block size panic, even when
    /// they're long enough.
    #[test]
    #[should_panic(expected = "has 66 bytes per row, but needs a multiple of 4")]
    fn rows_of_partial_blocks_panic() {
        let extent = Extent3d::new_2d(16, 4);
        let layout = TextureDataLayout {
            bytes_per_row: 66,
            ..TextureDataLayout::tight(Format::Rgba8Unorm, extent)
        };
        check_texture_copy(&rgba8(), &copy(0, [0, 0], extent), &layout);
    }

    /// Images shorter than the region panic.
    #[test]
    #[should_panic(
        expected = "a copy of texture \"albedo\" has 3 rows per image, but needs at least 4"
    )]
    fn short_images_panic() {
        let extent = Extent3d::new_2d(16, 4);
        let layout = TextureDataLayout {
            rows_per_image: 3,
            ..TextureDataLayout::tight(Format::Rgba8Unorm, extent)
        };
        check_texture_copy(&rgba8(), &copy(0, [0, 0], extent), &layout);
    }

    /// Mips halve each side, rounding down, to at least 1.
    #[test]
    fn mips_halve_their_extent() {
        let extent = Extent3d {
            width: 64,
            height: 20,
            depth: 3,
        };
        assert_eq!(mip_extent(extent, 0), extent);
        assert_eq!(
            mip_extent(extent, 2),
            Extent3d {
                width: 16,
                height: 5,
                depth: 1
            }
        );
        assert_eq!(mip_extent(extent, 40), Extent3d::new_2d(1, 1));
    }
}
