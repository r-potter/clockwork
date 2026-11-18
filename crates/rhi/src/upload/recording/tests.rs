//! T0 tests of what an uploader records: its copies, the barriers around
//! them, its acquire barriers when the transfer queue is and isn't the
//! graphics queue, and its choice of queue under each transfer granularity
//! (design 0002, R17 and R25).

use super::*;
use crate::command_list::{BufferTextureCopy, RecordedCommand};
use crate::upload::fake::FakeResources;
use crate::{Format, TextureDimension, TextureInfo};

/// The transfer queue's ID in the tests where it has its own queue.
const TRANSFER: QueueId = QueueId::new(2);

/// The graphics queue's ID.
const GRAPHICS: QueueId = QueueId::new(0);

/// A recording over a fake whose buffers start 8 bytes past a multiple of
/// 2^32, copying on a transfer queue with `granularity`, which is the
/// graphics queue if `shared`, and the fake.
fn recording(granularity: TransferGranularity, shared: bool) -> (Recording, Arc<FakeResources>) {
    let resources = Arc::new(FakeResources::new(8));
    let transfer = TransferQueue {
        id: if shared { GRAPHICS } else { TRANSFER },
        granularity,
    };
    let recording = Recording::new(resources.clone(), transfer, GRAPHICS, "streaming");
    (recording, resources)
}

/// A recording on a dedicated transfer queue that copies any region.
fn dedicated() -> (Recording, Arc<FakeResources>) {
    recording(TransferGranularity::ANY, false)
}

/// A destination buffer of 1024 bytes, whose address and size the
/// uploader reads.
fn buffer(resources: &FakeResources) -> Buffer {
    resources
        .create_buffer(&crate::BufferDesc {
            label: "vertices",
            size: 1024,
            usage: crate::BufferUsage::STORAGE,
            location: crate::MemoryLocation::Device,
        })
        .unwrap()
}

/// A 2D texture of `format`, 64×64 with 7 mips and 2 layers, sampled and
/// copied to.
fn texture(resources: &FakeResources, format: Format) -> Texture {
    resources.add_texture(
        "albedo",
        TextureInfo {
            dimension: TextureDimension::D2,
            format,
            extent: Extent3d::new_2d(64, 64),
            mip_levels: 7,
            array_layers: 2,
            usage: TextureUsage::SAMPLED | TextureUsage::COPY_DESTINATION,
        },
    )
}

/// The region of mip `mip` of layer 0 of `texture` at `origin`, of
/// `width`×`height`.
fn region(texture: Texture, mip: u32, origin: [u32; 2], width: u32, height: u32) -> TextureCopy {
    TextureCopy {
        texture,
        mip,
        layers: 0..1,
        origin: [origin[0], origin[1], 0],
        extent: Extent3d::new_2d(width, height),
    }
}

/// `count` RGBA8 texels' worth of bytes.
fn texels(count: usize) -> Vec<u8> {
    (0..count * 4).map(|i| i as u8).collect()
}

/// The barriers in a recorded barrier command.
fn barriers<'a>(command: &RecordedCommand<'a>) -> &'a [Barrier] {
    match command {
        RecordedCommand::Barrier(barriers) => barriers,
        other => panic!("expected a barrier, found {other:?}"),
    }
}

/// The copy in a recorded copy to a texture.
fn texture_copy<'a>(command: &RecordedCommand<'a>) -> &'a BufferTextureCopy {
    match command {
        RecordedCommand::CopyBufferToTexture(copy) => copy,
        other => panic!("expected a copy to a texture, found {other:?}"),
    }
}

/// A buffer upload copies the staged bytes to the buffer's address plus
/// the offset, then releases just those bytes to the graphics queue for
/// their next accesses, which is also the acquire barrier, so the rest of
/// the buffer keeps its owner and contents.
#[test]
fn buffer_uploads_copy_and_release() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    recording
        .upload_buffer(buffer, 16, &[1, 2, 3, 4], Access::VertexShaderRead.into())
        .unwrap();
    let recorded = recording.finish();
    let list = recorded.transfer.unwrap();
    assert_eq!(list.label(), "streaming");
    let commands = list.recorded();
    assert_eq!(commands.len(), 2, "{commands:?}");
    let RecordedCommand::CopyBuffer {
        source,
        destination,
        size,
    } = commands[0]
    else {
        panic!("expected a copy, found {:?}", commands[0]);
    };
    assert_eq!(resources.read(source, 4), [1, 2, 3, 4]);
    assert_eq!(destination.0, resources.buffer_address(buffer).0 + 16);
    assert_eq!(size, 4);
    let release =
        Barrier::buffer_range(buffer, 16..20, Access::CopyWrite, Access::VertexShaderRead)
            .with_queue_transfer(TRANSFER, GRAPHICS);
    assert_eq!(barriers(&commands[1]), std::slice::from_ref(&release));
    assert_eq!(recorded.acquire, [release]);
    assert!(recorded.graphics.is_none());
}

/// When the transfer queue is the graphics queue, the barrier after the
/// copies moves nothing between queues, and rendering needs no acquire
/// barriers.
#[test]
fn shared_queues_need_no_acquire_barriers() {
    let (mut recording, resources) = recording(TransferGranularity::ANY, true);
    let buffer = buffer(&resources);
    let texture = texture(&resources, Format::Rgba8Unorm);
    recording
        .upload_buffer(buffer, 0, &[1], Access::IndexRead.into())
        .unwrap();
    recording
        .upload_texture(
            region(texture, 6, [0, 0], 1, 1),
            &texels(1),
            Access::FragmentShaderSampledRead.into(),
        )
        .unwrap();
    let recorded = recording.finish();
    let commands = recorded.transfer.unwrap();
    let after = barriers(commands.recorded().last().unwrap()).to_vec();
    assert_eq!(after.len(), 2);
    assert!(
        after
            .iter()
            .all(|barrier| barrier.queue_transfer().is_none())
    );
    assert!(recorded.acquire.is_empty());
}

/// A texture upload discards the subresource's contents, copies the
/// staged, tightly packed bytes from an address aligned for the format's
/// copies, then releases the subresource for its next accesses.
#[test]
fn texture_uploads_discard_copy_and_release() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    let data = texels(16 * 16);
    recording
        .upload_texture(
            region(texture, 2, [0, 0], 16, 16),
            &data,
            Access::FragmentShaderSampledRead.into(),
        )
        .unwrap();
    let recorded = recording.finish();
    let list = recorded.transfer.unwrap();
    let commands = list.recorded();
    assert_eq!(commands.len(), 3, "{commands:?}");
    let subresource = Subresource::mip_layer(2, 0);
    assert_eq!(
        barriers(&commands[0]),
        [Barrier::subresource(
            texture,
            subresource.clone(),
            Access::None,
            Access::CopyWrite
        )],
    );
    let copy = texture_copy(&commands[1]);
    assert_eq!(copy.texture, region(texture, 2, [0, 0], 16, 16));
    assert_eq!(
        copy.layout,
        TextureDataLayout::tightly_packed(Format::Rgba8Unorm, Extent3d::new_2d(16, 16))
    );
    let alignment = resources
        .capabilities()
        .limits
        .copy_offset_alignment(Format::Rgba8Unorm);
    assert_eq!(copy.buffer.0 % alignment, 0);
    assert_eq!(resources.read(copy.buffer, data.len()), data);
    let release = Barrier::subresource(
        texture,
        subresource,
        Access::CopyWrite,
        Access::FragmentShaderSampledRead,
    )
    .with_queue_transfer(TRANSFER, GRAPHICS);
    assert_eq!(barriers(&commands[2]), std::slice::from_ref(&release));
    assert_eq!(recorded.acquire, [release]);
}

/// Block-compressed data is staged at an address aligned for its blocks
/// and the device's preference together.
#[test]
fn block_compressed_uploads_are_aligned_for_their_blocks() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Bc7Unorm);
    // An unaligned write first, so the next can't be aligned by chance.
    let buffer = buffer(&resources);
    recording
        .upload_buffer(buffer, 0, &[1], Access::IndexRead.into())
        .unwrap();
    recording
        .upload_texture(
            region(texture, 4, [0, 0], 4, 4),
            &[7; 16],
            Access::FragmentShaderSampledRead.into(),
        )
        .unwrap();
    let list = recording.finish().transfer.unwrap();
    let commands = list.recorded();
    let copy = texture_copy(&commands[2]);
    let alignment = resources
        .capabilities()
        .limits
        .copy_offset_alignment(Format::Bc7Unorm);
    assert_eq!(alignment, 64);
    assert_eq!(copy.buffer.0 % alignment, 0);
}

/// A mip filled in several regions in one flush gets one barrier before
/// its copies and one after them.
#[test]
fn regions_of_one_subresource_share_its_barriers() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    for origin in [[0, 0], [0, 4]] {
        recording
            .upload_texture(
                region(texture, 3, origin, 8, 4),
                &texels(32),
                Access::FragmentShaderSampledRead.into(),
            )
            .unwrap();
    }
    let recorded = recording.finish();
    let list = recorded.transfer.unwrap();
    let commands = list.recorded();
    assert_eq!(commands.len(), 4, "{commands:?}");
    assert_eq!(barriers(&commands[0]).len(), 1);
    texture_copy(&commands[1]);
    texture_copy(&commands[2]);
    assert_eq!(barriers(&commands[3]).len(), 1);
    assert_eq!(recorded.acquire.len(), 1);
}

/// Regions the transfer queue's granularity rules out are copied on the
/// graphics queue, with no queue transfer and no acquire barrier, while a
/// whole mip stays on the transfer queue.
#[test]
fn coarse_granularity_moves_partial_regions_to_graphics() {
    let (mut recording, resources) = recording(TransferGranularity::Blocks([8, 8, 1]), false);
    let texture = texture(&resources, Format::Rgba8Unorm);
    // Neither region is in whole 8×8 blocks of mip 2, which is 16×16.
    for (origin, width) in [([0, 0], 4), ([4, 0], 12)] {
        recording
            .upload_texture(
                region(texture, 2, origin, width, 16),
                &texels(width as usize * 16),
                Access::FragmentShaderSampledRead.into(),
            )
            .unwrap();
    }
    recording
        .upload_texture(
            region(texture, 1, [0, 0], 32, 32),
            &texels(32 * 32),
            Access::FragmentShaderSampledRead.into(),
        )
        .unwrap();
    let recorded = recording.finish();

    let graphics = recorded.graphics.unwrap();
    assert_eq!(graphics.label(), "streaming (graphics)");
    let commands = graphics.recorded();
    assert_eq!(texture_copy(&commands[1]).texture.mip, 2);
    assert_eq!(texture_copy(&commands[2]).texture.mip, 2);
    assert!(
        barriers(&commands[3])
            .iter()
            .all(|barrier| barrier.queue_transfer().is_none())
    );

    let transfer = recorded.transfer.unwrap();
    let commands = transfer.recorded();
    assert_eq!(texture_copy(&commands[1]).texture.mip, 1);
    assert_eq!(recorded.acquire.len(), 1);
    assert_eq!(
        recorded.acquire[0].queue_transfer(),
        Some((TRANSFER, GRAPHICS))
    );
}

/// A transfer queue that copies only whole mips copies a whole mip, and
/// leaves any part of one to the graphics queue.
#[test]
fn whole_mip_granularity_copies_whole_mips() {
    let (mut recording, resources) = recording(TransferGranularity::WholeMips, false);
    let texture = texture(&resources, Format::Rgba8Unorm);
    recording
        .upload_texture(
            region(texture, 5, [0, 0], 2, 2),
            &texels(4),
            Access::FragmentShaderSampledRead.into(),
        )
        .unwrap();
    for origin in [[0, 0], [2, 0]] {
        recording
            .upload_texture(
                region(texture, 4, origin, 2, 4),
                &texels(8),
                Access::FragmentShaderSampledRead.into(),
            )
            .unwrap();
    }
    let recorded = recording.finish();
    assert_eq!(
        texture_copy(&recorded.transfer.unwrap().recorded()[1])
            .texture
            .mip,
        5
    );
    assert_eq!(
        texture_copy(&recorded.graphics.unwrap().recorded()[1])
            .texture
            .mip,
        4
    );
}

/// Finishing leaves nothing recorded, so a flush with no uploads has no
/// lists.
#[test]
fn finishing_leaves_nothing_recorded() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    recording
        .upload_buffer(buffer, 0, &[1], Access::IndexRead.into())
        .unwrap();
    recording.finish();
    let recorded = recording.finish();
    assert!(recorded.transfer.is_none());
    assert!(recorded.graphics.is_none());
    assert!(recorded.acquire.is_empty());
}

/// Ranges of one buffer that touch, uploaded in any order for the same
/// accesses, share one barrier after their copies, for all their bytes.
#[test]
fn touching_buffer_ranges_share_a_barrier() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    for offset in [4, 0, 8] {
        recording
            .upload_buffer(buffer, offset, &[1; 4], Access::IndexRead.into())
            .unwrap();
    }
    let recorded = recording.finish();
    assert_eq!(recorded.transfer.unwrap().recorded().len(), 4);
    assert_eq!(
        recorded.acquire,
        [
            Barrier::buffer_range(buffer, 0..12, Access::CopyWrite, Access::IndexRead)
                .with_queue_transfer(TRANSFER, GRAPHICS)
        ]
    );
}

/// Ranges of one buffer with bytes between them each get a barrier, so the
/// bytes between keep their owner and contents.
#[test]
fn separate_buffer_ranges_get_a_barrier_each() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    for offset in [64, 0] {
        recording
            .upload_buffer(buffer, offset, &[1; 4], Access::IndexRead.into())
            .unwrap();
    }
    let barrier = |range| {
        Barrier::buffer_range(buffer, range, Access::CopyWrite, Access::IndexRead)
            .with_queue_transfer(TRANSFER, GRAPHICS)
    };
    assert_eq!(recording.finish().acquire, [barrier(0..4), barrier(64..68)]);
}

/// A mip filled by many tiles in one flush, in any order, passes the
/// checks, with one barrier before the copies and one after.
#[test]
fn tiles_that_fill_a_mip_pass() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    for index in [5, 0, 15, 10, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14] {
        recording
            .upload_texture(
                region(texture, 0, [index % 4 * 16, index / 4 * 16], 16, 16),
                &texels(256),
                Access::FragmentShaderSampledRead.into(),
            )
            .unwrap();
    }
    let recorded = recording.finish();
    assert_eq!(recorded.transfer.unwrap().recorded().len(), 18);
    assert_eq!(recorded.acquire.len(), 1);
}

/// A staging buffer that can't be created is an error, and records
/// nothing.
#[test]
fn staging_errors_record_nothing() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    let texture = texture(&resources, Format::Rgba8Unorm);
    resources.fail_after(0);
    assert!(
        recording
            .upload_buffer(buffer, 0, &[1], Access::IndexRead.into())
            .is_err()
    );
    assert!(
        recording
            .upload_texture(
                region(texture, 6, [0, 0], 1, 1),
                &texels(1),
                Access::FragmentShaderSampledRead.into(),
            )
            .is_err()
    );
    let recorded = recording.finish();
    assert!(recorded.transfer.is_none());
}

/// Data that isn't the region's tightly packed blocks panics.
#[test]
#[should_panic(expected = "uploader \"streaming\" uploads 12 bytes to a region")]
fn wrongly_sized_data_panics() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    recording
        .upload_texture(
            region(texture, 0, [0, 0], 2, 2),
            &[0; 12],
            Access::FragmentShaderSampledRead.into(),
        )
        .unwrap();
}

/// A texture that can't be copied to panics.
#[test]
#[should_panic(expected = "uploads to a texture that can't be copied to")]
fn textures_that_cant_be_copied_to_panic() {
    let (mut recording, resources) = dedicated();
    let texture = resources.add_texture(
        "target",
        TextureInfo {
            dimension: TextureDimension::D2,
            format: Format::Rgba8Unorm,
            extent: Extent3d::new_2d(4, 4),
            mip_levels: 1,
            array_layers: 1,
            usage: TextureUsage::SAMPLED,
        },
    );
    let _ = recording.upload_texture(
        region(texture, 0, [0, 0], 4, 4),
        &texels(16),
        Access::FragmentShaderSampledRead.into(),
    );
}

/// Accesses a buffer can't have panic at the upload, naming the uploader.
#[test]
#[should_panic(expected = "command list \"streaming\" has a barrier")]
fn texture_accesses_for_buffers_panic() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    let _ = recording.upload_buffer(buffer, 0, &[1], Access::FragmentShaderSampledRead.into());
}

/// One subresource uploaded for different accesses in one flush panics.
#[test]
#[should_panic(
    expected = "uploads to mip 3 of layers 0..1 of a texture for ComputeShaderSampledRead, having \
                uploaded to it for FragmentShaderSampledRead in this flush"
)]
fn one_subresource_for_two_access_sets_panics() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    let _ = recording.upload_texture(
        region(texture, 3, [0, 0], 4, 4),
        &texels(16),
        Access::FragmentShaderSampledRead.into(),
    );
    let _ = recording.upload_texture(
        region(texture, 3, [4, 0], 4, 4),
        &texels(16),
        Access::ComputeShaderSampledRead.into(),
    );
}

/// Overlapping but different layers of one mip in one flush panic, since
/// their barriers would overlap, saying so rather than blaming the
/// accesses, which are the same.
#[test]
#[should_panic(
    expected = "uploads to mip 0 of layers 0..2 of a texture, which overlap layers 0..1 it \
                uploaded to in this flush"
)]
fn overlapping_layers_panic() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    let _ = recording.upload_texture(
        region(texture, 0, [0, 0], 64, 64),
        &texels(64 * 64),
        Access::FragmentShaderSampledRead.into(),
    );
    let _ = recording.upload_texture(
        TextureCopy {
            layers: 0..2,
            ..region(texture, 0, [0, 0], 64, 64)
        },
        &texels(2 * 64 * 64),
        Access::FragmentShaderSampledRead.into(),
    );
}

/// Overlapping regions in one flush panic at the finish, since copies in a
/// list aren't ordered.
#[test]
#[should_panic(
    expected = "uploads to regions at [0, 0, 0] of Extent3d { width: 4, height: 4, depth: 1 } \
                and at [2, 2, 0] of Extent3d { width: 4, height: 4, depth: 1 } of mip 0 of \
                layers 0..1 of a texture in one flush, which overlap"
)]
fn overlapping_regions_panic() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    for origin in [[0, 0], [2, 2]] {
        recording
            .upload_texture(
                region(texture, 0, origin, 4, 4),
                &texels(16),
                Access::FragmentShaderSampledRead.into(),
            )
            .unwrap();
    }
    recording.finish();
}

/// Overlapping rows of a mip panic too, though they all start at the same
/// place across it.
#[test]
#[should_panic(expected = "of mip 0 of layers 0..1 of a texture in one flush, which overlap")]
fn overlapping_rows_panic() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    for origin in [[0, 0], [0, 32], [0, 4]] {
        recording
            .upload_texture(
                region(texture, 0, origin, 64, 8),
                &texels(64 * 8),
                Access::FragmentShaderSampledRead.into(),
            )
            .unwrap();
    }
    recording.finish();
}

/// A mip only partly filled in a flush panics at the finish, since the
/// barrier before its copies discards the rest of it.
#[test]
#[should_panic(
    expected = "uploads 32 of the 64 texels of mip 3 of layers 0..1 of a texture in one flush, \
                but a flush discards what a mip held, so it fills the mip whole"
)]
fn partly_filled_mips_panic() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    recording
        .upload_texture(
            region(texture, 3, [0, 0], 8, 4),
            &texels(32),
            Access::FragmentShaderSampledRead.into(),
        )
        .unwrap();
    recording.finish();
}

/// A region that isn't in its mip panics at the upload.
#[test]
#[should_panic(
    expected = "uploader \"streaming\" uploads to a region at [4, 0, 0] of Extent3d { width: 8, \
                height: 8, depth: 1 }, which isn't in mip 3 of Extent3d { width: 8, height: 8, \
                depth: 1 }"
)]
fn regions_outside_their_mip_panic() {
    let (mut recording, resources) = dedicated();
    let texture = texture(&resources, Format::Rgba8Unorm);
    let _ = recording.upload_texture(
        region(texture, 3, [4, 0], 8, 8),
        &texels(64),
        Access::FragmentShaderSampledRead.into(),
    );
}

/// A region whose tightly packed size passes 2^64 bytes panics, rather
/// than wrapping to a size the data might match.
#[test]
#[should_panic(expected = "which needs more than 2^64 bytes of tightly packed Rgba8Unorm")]
fn regions_too_large_to_count_panic() {
    let (mut recording, resources) = dedicated();
    let texture = resources.add_texture(
        "huge",
        TextureInfo {
            dimension: TextureDimension::D2,
            format: Format::Rgba8Unorm,
            extent: Extent3d::new_2d(1 << 28, 1 << 28),
            mip_levels: 1,
            array_layers: 128,
            usage: TextureUsage::SAMPLED | TextureUsage::COPY_DESTINATION,
        },
    );
    let _ = recording.upload_texture(
        TextureCopy {
            layers: 0..128,
            ..region(texture, 0, [0, 0], 1 << 28, 1 << 28)
        },
        &[0; 4],
        Access::FragmentShaderSampledRead.into(),
    );
}

/// One subresource with regions on both queues in one flush panics, since
/// each queue's barriers would discard the other's copies.
#[test]
#[should_panic(expected = "having uploaded to it on another in this flush")]
fn one_subresource_on_two_queues_panics() {
    let (mut recording, resources) = recording(TransferGranularity::Blocks([8, 8, 1]), false);
    let texture = texture(&resources, Format::Rgba8Unorm);
    let _ = recording.upload_texture(
        region(texture, 0, [0, 0], 8, 8),
        &texels(64),
        Access::FragmentShaderSampledRead.into(),
    );
    let _ = recording.upload_texture(
        region(texture, 0, [8, 0], 4, 4),
        &texels(16),
        Access::FragmentShaderSampledRead.into(),
    );
}

/// Overlapping bytes of one buffer in one flush panic at the finish.
#[test]
#[should_panic(
    expected = "uploads to bytes 2..6 of a buffer, which overlap bytes it uploaded to in this \
                flush"
)]
fn overlapping_buffer_ranges_panic() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    for offset in [0, 2] {
        recording
            .upload_buffer(buffer, offset, &[1; 4], Access::IndexRead.into())
            .unwrap();
    }
    recording.finish();
}

/// Bytes that don't fit in the buffer at the offset panic at the upload,
/// since a copy by address isn't checked against its buffer.
#[test]
#[should_panic(
    expected = "uploader \"streaming\" uploads 32 bytes at offset 1000 of a buffer of 1024 bytes"
)]
fn uploads_past_a_buffers_end_panic() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    let _ = recording.upload_buffer(buffer, 1000, &[1; 32], Access::IndexRead.into());
}

/// One buffer uploaded for different accesses in one flush panics.
#[test]
#[should_panic(expected = "which it uploaded to for")]
fn one_buffer_for_two_access_sets_panics() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    let _ = recording.upload_buffer(buffer, 0, &[1], Access::IndexRead.into());
    let _ = recording.upload_buffer(buffer, 4, &[1], Access::VertexShaderRead.into());
}

/// Finishing leaves the staged bytes in the staging's batch, for the
/// flush to end once it has submitted the copies, or to discard.
#[test]
fn finishing_leaves_the_batch_to_the_flush() {
    let (mut recording, resources) = dedicated();
    let buffer = buffer(&resources);
    recording
        .upload_buffer(buffer, 0, &[1], Access::IndexRead.into())
        .unwrap();
    recording.finish();
    assert_eq!(resources.live(), 2, "the destination and a staging buffer");
    recording.staging().discard();
    assert_eq!(resources.live(), 1);
}
