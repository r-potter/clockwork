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
    pub const fn tightly_packed(format: Format, extent: Extent3d) -> Self {
        tightly_packed(format.info(), extent)
    }
}

/// [`TextureDataLayout::tightly_packed`] for a format described by `format`.
const fn tightly_packed(format: FormatInfo, extent: Extent3d) -> TextureDataLayout {
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
/// Those bytes aren't checked against a buffer, since copies name buffers by
/// address (design 0002, R5): a range past the end of the buffer at that
/// address is a bug in the caller that the GPU reads or writes through
/// unreported. `TODO.md` tracks checking it in debug builds.
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
/// - `rows_per_image` is less than the region's rows of blocks, for a copy
///   of more than one image, the only kind that uses it
/// - the bytes the copy uses reach past the end of the address space
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
        assert!(
            aligned_or_reaches_edge(
                copy.origin[axis],
                extents[axis],
                sizes[axis],
                u64::from(format.block_extent[axis]),
            ),
            "a copy of texture \"{label}\" at {:?}, of {:?}, isn't aligned to its {}×{} blocks",
            copy.origin,
            copy.extent,
            format.block_extent[0],
            format.block_extent[1],
        );
    }

    let offset_alignment = required_offset_alignment(format);
    assert!(
        layout.offset.is_multiple_of(offset_alignment),
        "a copy of texture \"{label}\" starts at offset {}, which isn't a multiple of \
         {offset_alignment} bytes",
        layout.offset,
    );
    let packed = tightly_packed(format, copy.extent);
    assert!(
        layout.bytes_per_row.is_multiple_of(format.block_bytes)
            && layout.bytes_per_row >= packed.bytes_per_row,
        "a copy of texture \"{label}\" has {} bytes per row, but needs a multiple of {} that's \
         at least {}",
        layout.bytes_per_row,
        format.block_bytes,
        packed.bytes_per_row,
    );
    // Each layer, or depth slice of a 3D texture, is an image.
    let images = u64::from(copy.layers.end - copy.layers.start) * u64::from(copy.extent.depth);
    assert!(
        images == 1 || layout.rows_per_image >= packed.rows_per_image,
        "a copy of texture \"{label}\" has {} rows per image, but needs at least {}",
        layout.rows_per_image,
        packed.rows_per_image,
    );

    // Every image but the last is `rows_per_image` rows long, and the last
    // row of the last image is only as long as the region's row of blocks.
    let end = (images - 1)
        .checked_mul(u64::from(layout.rows_per_image))
        .and_then(|rows| rows.checked_add(u64::from(packed.rows_per_image) - 1))
        .and_then(|rows| rows.checked_mul(u64::from(layout.bytes_per_row)))
        .and_then(|bytes| bytes.checked_add(u64::from(packed.bytes_per_row)))
        .and_then(|bytes| bytes.checked_add(layout.offset));
    end.unwrap_or_else(|| {
        panic!(
            "a copy of texture \"{label}\" at offset {} reaches past the end of the address \
             space",
            layout.offset,
        )
    })
}

/// Whether a region along one axis of a mip of `size` texels, from `origin`
/// for `extent` texels, starts at a multiple of `unit` texels, and either
/// is a multiple of it long or reaches the mip's edge: the rule Vulkan
/// gives both block-compressed copies and transfer granularities.
fn aligned_or_reaches_edge(origin: u32, extent: u32, size: u32, unit: u64) -> bool {
    let (origin, extent) = (u64::from(origin), u64::from(extent));
    origin.is_multiple_of(unit)
        && (extent.is_multiple_of(unit) || origin + extent == u64::from(size))
}

/// The alignment every copy's buffer offset needs for `format`: its block
/// size in bytes, and at least 4 for a depth format.
pub(crate) const fn required_offset_alignment(format: FormatInfo) -> u64 {
    let block_bytes = format.block_bytes as u64;
    if format.depth && block_bytes < 4 {
        4
    } else {
        block_bytes
    }
}

/// The texture regions a queue can copy (design 0002, "Texture copies"):
/// on Vulkan, its family's `minImageTransferGranularity`. Graphics and
/// compute queues can copy any region, but a dedicated transfer queue may
/// be coarser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferGranularity {
    /// Regions whose origin, along each axis, is a multiple of this many
    /// texel blocks, and whose extent is too, or reaches the edge of the
    /// mip. Each is a power of two, and `[1, 1, 1]` allows any region.
    /// Vulkan's `(0, 0, 0)` is [`WholeMips`](Self::WholeMips), not
    /// `Blocks`.
    Blocks([u32; 3]),
    /// Only whole mips.
    WholeMips,
}

impl TransferGranularity {
    /// The granularity that allows any region.
    pub const ANY: Self = Self::Blocks([1, 1, 1]);

    /// Whether a queue with this granularity can copy the region `copy`
    /// names of a texture of `format`, whose largest mip is `extent`.
    ///
    /// A region it can't copy can still be copied on a queue that allows
    /// any region, such as the graphics queue. A whole mip is always
    /// allowed. The rest of the copy, such as its layers and whether it
    /// stays inside the mip, is checked when it's translated.
    ///
    /// # Panics
    ///
    /// Panics if `copy.mip` is past the last mip a texture of `extent` can
    /// have, or if a granularity in `Blocks` isn't a power of two.
    pub fn allows(self, format: Format, extent: Extent3d, copy: &TextureCopy) -> bool {
        assert!(
            copy.mip < extent.max_mip_count(),
            "a copy of mip {} can't be of a texture whose largest mip is {extent:?}",
            copy.mip,
        );
        let mip = mip_extent(extent, copy.mip);
        let sizes = [mip.width, mip.height, mip.depth];
        let extents = [copy.extent.width, copy.extent.height, copy.extent.depth];
        let [block_width, block_height] = format.info().block_extent;
        let block = [block_width, block_height, 1];
        (0..3).all(|axis| match self {
            Self::WholeMips => copy.origin[axis] == 0 && extents[axis] == sizes[axis],
            Self::Blocks(granularity) => {
                assert!(
                    granularity[axis].is_power_of_two(),
                    "the transfer granularity {granularity:?} isn't a power of two on each \
                     axis; whole mips are `TransferGranularity::WholeMips`",
                );
                let unit = u64::from(granularity[axis]) * u64::from(block[axis]);
                aligned_or_reaches_edge(copy.origin[axis], extents[axis], sizes[axis], unit)
            }
        })
    }
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
    assert_send_sync::<TransferGranularity>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of copy layouts and their checks (design 0002, R25), for an
    //! uncompressed, a block-compressed and a depth format.

    use super::*;
    use crate::handle::HandleTable;

    /// BC7's 4×4 blocks of 16 bytes.
    const BC7: FormatInfo = Format::Bc7Unorm.info();

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

    /// Tightly packed layouts have rows of whole blocks, with no padding.
    #[test]
    fn tightly_packed_layouts_pack_blocks() {
        assert_eq!(
            TextureDataLayout::tightly_packed(Format::Rgba16Float, Extent3d::new_2d(10, 3)),
            TextureDataLayout {
                offset: 0,
                bytes_per_row: 80,
                rows_per_image: 3,
            },
        );
        // A partial block at the edge still takes a whole block.
        assert_eq!(
            tightly_packed(BC7, Extent3d::new_2d(10, 6)),
            TextureDataLayout {
                offset: 0,
                bytes_per_row: 48,
                rows_per_image: 2,
            },
        );
    }

    /// The format table gives each format's block size, one texel unless
    /// it's block-compressed, and says which are depth.
    #[test]
    fn formats_describe_their_blocks() {
        for format in Format::ALL {
            let info = format.info();
            if !format.is_block_compressed() {
                assert_eq!(info.block_extent, [1, 1], "{format:?}");
            }
            assert_eq!(info.depth, format.is_depth(), "{format:?}");
        }
        assert_eq!(Format::Rgba32Float.info().block_bytes, 16);
    }

    /// A tightly packed copy of a whole mip reads exactly its bytes.
    #[test]
    fn tightly_packed_copies_use_their_bytes() {
        let extent = Extent3d::new_2d(32, 16);
        let layout = TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent);
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
        let layout = tightly_packed(BC7, Extent3d::new_2d(16, 8));
        assert_eq!(
            check_texture_copy(&texture, &copy(0, [0, 0], Extent3d::new_2d(16, 8)), &layout),
            4 * 2 * 16,
        );
        // Mip 2 is 4×2, and mip 3 is 2×1: both smaller than a block.
        let small = Extent3d::new_2d(2, 1);
        assert_eq!(
            check_texture_copy(
                &texture,
                &copy(3, [0, 0], small),
                &tightly_packed(BC7, small)
            ),
            16,
        );
        // A region that ends at a mip's edge needn't be whole blocks.
        let edge = Extent3d::new_2d(4, 2);
        check_texture_copy(&texture, &copy(2, [0, 0], edge), &tightly_packed(BC7, edge));
    }

    /// A depth format's offset must be a multiple of 4.
    #[test]
    fn depth_copies_align_their_offset() {
        let texture = texture(Format::Depth32Float.info(), Extent3d::new_2d(8, 8));
        let extent = Extent3d::new_2d(8, 8);
        let layout = TextureDataLayout {
            offset: 4,
            ..TextureDataLayout::tightly_packed(Format::Depth32Float, extent)
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
            &TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent),
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
            &TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent),
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
            &TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent),
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
            &TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent),
        );
    }

    /// A block-compressed region that starts inside a block panics.
    #[test]
    #[should_panic(expected = "isn't aligned to its 4×4 blocks")]
    fn unaligned_origins_panic() {
        let texture = texture(BC7, Extent3d::new_2d(16, 16));
        let extent = Extent3d::new_2d(4, 4);
        check_texture_copy(
            &texture,
            &copy(0, [2, 0], extent),
            &tightly_packed(BC7, extent),
        );
    }

    /// A block-compressed region that ends inside a block, short of the
    /// mip's edge, panics.
    #[test]
    #[should_panic(expected = "isn't aligned to its 4×4 blocks")]
    fn unaligned_extents_panic() {
        let texture = texture(BC7, Extent3d::new_2d(16, 16));
        let extent = Extent3d::new_2d(6, 4);
        check_texture_copy(
            &texture,
            &copy(0, [0, 0], extent),
            &tightly_packed(BC7, extent),
        );
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
            ..tightly_packed(BC7, extent)
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
            ..TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent)
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
            ..TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent)
        };
        check_texture_copy(&rgba8(), &copy(0, [0, 0], extent), &layout);
    }

    /// Images shorter than the region panic, when a copy has several.
    #[test]
    #[should_panic(
        expected = "a copy of texture \"albedo\" has 3 rows per image, but needs at least 4"
    )]
    fn short_images_panic() {
        let extent = Extent3d::new_2d(16, 4);
        let layout = TextureDataLayout {
            rows_per_image: 3,
            ..TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent)
        };
        let copy = TextureCopy {
            layers: 0..2,
            ..copy(0, [0, 0], extent)
        };
        check_texture_copy(&rgba8(), &copy, &layout);
    }

    /// A copy of one image doesn't use `rows_per_image`, so it may be 0, as
    /// Vulkan allows.
    #[test]
    fn single_images_ignore_rows_per_image() {
        let extent = Extent3d::new_2d(16, 4);
        let layout = TextureDataLayout {
            rows_per_image: 0,
            ..TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent)
        };
        assert_eq!(
            check_texture_copy(&rgba8(), &copy(0, [0, 0], extent), &layout),
            16 * 4 * 4,
        );
    }

    /// A 3D texture's depth slices are its images: a copy of two slices,
    /// starting at the second, counts the first slice's padded rows in full.
    #[test]
    fn depth_slices_are_images() {
        let texture = CopiedTexture {
            label: "volume",
            format: Format::Rgba8Unorm.info(),
            extent: Extent3d {
                width: 8,
                height: 8,
                depth: 4,
            },
            mip_count: 1,
            layer_count: 1,
        };
        let copy = TextureCopy {
            origin: [0, 0, 1],
            extent: Extent3d {
                width: 8,
                height: 8,
                depth: 2,
            },
            ..copy(0, [0, 0], Extent3d::new_2d(8, 8))
        };
        let layout = TextureDataLayout {
            offset: 0,
            bytes_per_row: 64,
            rows_per_image: 10,
        };
        // Ten rows of the first slice, seven full rows of the second, then a
        // row of 8 texels.
        assert_eq!(
            check_texture_copy(&texture, &copy, &layout),
            17 * 64 + 8 * 4
        );
    }

    /// A copy whose bytes reach past the end of the address space panics,
    /// naming the texture, rather than wrapping.
    #[test]
    #[should_panic(
        expected = "a copy of texture \"albedo\" at offset 18446744073709551612 reaches past the \
                    end of the address space"
    )]
    fn copies_past_the_address_space_panic() {
        let extent = Extent3d::new_2d(16, 4);
        let layout = TextureDataLayout {
            offset: u64::MAX - 3,
            ..TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent)
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

    /// Any granularity of single texels allows every region.
    #[test]
    fn any_granularity_allows_every_region() {
        let extent = Extent3d::new_2d(64, 32);
        let region = copy(1, [3, 5], Extent3d::new_2d(7, 2));
        assert!(TransferGranularity::ANY.allows(Format::Rgba8Unorm, extent, &region));
    }

    /// A coarser granularity allows regions whose origin is a multiple of
    /// it, and whose extent is too, or reaches the edge of the mip.
    #[test]
    fn granularities_allow_aligned_regions() {
        let granularity = TransferGranularity::Blocks([8, 8, 1]);
        let allows = |origin, extent| {
            granularity.allows(
                Format::Rgba8Unorm,
                Extent3d::new_2d(64, 30),
                &copy(0, origin, extent),
            )
        };
        assert!(allows([8, 16], Extent3d::new_2d(16, 8)));
        // The bottom edge, at 30 rows, isn't a multiple of 8.
        assert!(allows([0, 24], Extent3d::new_2d(64, 6)));
        assert!(!allows([4, 16], Extent3d::new_2d(16, 8)), "origin");
        assert!(!allows([8, 16], Extent3d::new_2d(12, 8)), "width");
        assert!(!allows([8, 0], Extent3d::new_2d(16, 6)), "height");
    }

    /// A block-compressed format's granularity counts blocks, so it's a
    /// multiple of 4 texels.
    #[test]
    fn granularities_count_blocks_of_compressed_formats() {
        let granularity = TransferGranularity::Blocks([2, 2, 1]);
        let extent = Extent3d::new_2d(64, 64);
        let allows = |origin, size| {
            granularity.allows(
                Format::Bc7Unorm,
                extent,
                &copy(0, origin, Extent3d::new_2d(size, size)),
            )
        };
        assert!(allows([8, 8], 16));
        assert!(!allows([4, 4], 16), "origin of one block");
        assert!(!allows([8, 8], 4), "extent of one block");
    }

    /// The depth axis of a 3D texture has its own granularity, and a 2D
    /// texture's single slice is a whole one.
    #[test]
    fn granularities_apply_to_depth() {
        let granularity = TransferGranularity::Blocks([1, 1, 4]);
        let volume = Extent3d {
            width: 8,
            height: 8,
            depth: 16,
        };
        let slices = |origin, depth| TextureCopy {
            origin: [0, 0, origin],
            extent: Extent3d {
                width: 8,
                height: 8,
                depth,
            },
            ..copy(0, [0, 0], Extent3d::new_2d(8, 8))
        };
        assert!(granularity.allows(Format::Rgba8Unorm, volume, &slices(4, 8)));
        assert!(!granularity.allows(Format::Rgba8Unorm, volume, &slices(2, 8)));
        assert!(granularity.allows(
            Format::Rgba8Unorm,
            Extent3d::new_2d(8, 8),
            &copy(0, [0, 0], Extent3d::new_2d(8, 8)),
        ));
    }

    /// A queue that copies only whole mips allows a whole mip at any level,
    /// and nothing less.
    #[test]
    fn whole_mip_granularities_allow_only_whole_mips() {
        let extent = Extent3d::new_2d(64, 32);
        let allows =
            |region| TransferGranularity::WholeMips.allows(Format::Rgba8Unorm, extent, &region);
        assert!(allows(copy(0, [0, 0], extent)));
        assert!(allows(copy(2, [0, 0], Extent3d::new_2d(16, 8))));
        assert!(!allows(copy(2, [0, 0], Extent3d::new_2d(16, 4))));
        assert!(!allows(copy(2, [8, 0], Extent3d::new_2d(8, 8))));
    }

    /// A mip past the last a texture of the extent can have panics, rather
    /// than being judged as a 1×1 mip.
    #[test]
    #[should_panic(
        expected = "a copy of mip 7 can't be of a texture whose largest mip is Extent3d { width: 64, height: 32, depth: 1 }"
    )]
    fn granularities_of_missing_mips_panic() {
        TransferGranularity::WholeMips.allows(
            Format::Rgba8Unorm,
            Extent3d::new_2d(64, 32),
            &copy(7, [0, 0], Extent3d::new_2d(1, 1)),
        );
    }

    /// A granularity that isn't a power of two on each axis panics, naming
    /// `WholeMips` for Vulkan's `(0, 0, 0)`.
    #[test]
    fn granularities_must_be_powers_of_two() {
        for granularity in [[0, 0, 0], [3, 1, 1]] {
            let panic = std::panic::catch_unwind(|| {
                TransferGranularity::Blocks(granularity).allows(
                    Format::Rgba8Unorm,
                    Extent3d::new_2d(64, 32),
                    &copy(0, [0, 0], Extent3d::new_2d(64, 32)),
                )
            })
            .expect_err("the granularity was accepted");
            let message = panic.downcast_ref::<String>().cloned().unwrap_or_default();
            assert!(
                message.contains("isn't a power of two on each axis")
                    && message.contains("TransferGranularity::WholeMips"),
                "{granularity:?}: {message:?}",
            );
        }
    }

    /// Every copy's offset is aligned to its format's block size, and to 4
    /// bytes for a depth format.
    #[test]
    fn required_offset_alignments_suit_the_format() {
        assert_eq!(required_offset_alignment(Format::Rgba8Unorm.info()), 4);
        assert_eq!(required_offset_alignment(Format::Rgba32Float.info()), 16);
        assert_eq!(required_offset_alignment(Format::Bc1RgbaSrgb.info()), 8);
        assert_eq!(required_offset_alignment(Format::Depth32Float.info()), 4);
        let depth16 = FormatInfo {
            block_extent: [1, 1],
            block_bytes: 2,
            depth: true,
        };
        assert_eq!(required_offset_alignment(depth16), 4);
    }
}
