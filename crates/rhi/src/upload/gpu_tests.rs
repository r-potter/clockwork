//! T1 tests of the uploader on a real device: streaming from another thread
//! and handing over to rendering, buffers across flushes, and BC blocks
//! copied in and back out unchanged (design 0002, R17 and R26).
//!
//! On a device whose transfer queue is the graphics queue, such as
//! lavapipe, they check the path with no queue transfer, and on one with a
//! transfer queue of its own, the release and acquire.

use std::sync::Arc;
use std::thread;

use clockwork_diagnostics::capture::TestSession;

use crate::test_shaders;
use crate::vulkan::test_support::{Capability, VulkanTestBackend, require_capabilities};
use crate::{
    Access, AddressMode, Barrier, Buffer, BufferDesc, BufferUsage, CommandList, CullMode,
    DescriptorIndex, Extent2d, Extent3d, Filter, Format, GpuAddress, GpuResources, LoadOp,
    MemoryLocation, Queue, QueueKind, Rect2d, RenderHardwareInterface, RenderPassDesc, RenderState,
    SamplerDesc, SubmitDesc, TextureCopy, TextureDataLayout, TextureDesc, TextureDimension,
    TextureUsage, TextureView, Upload, Uploader, Viewport,
};

/// A buffer of `size` bytes in `location`, for copies.
fn create_buffer(
    resources: &dyn GpuResources,
    label: &str,
    size: u64,
    location: MemoryLocation,
) -> Buffer {
    resources
        .create_buffer(&BufferDesc {
            label,
            size,
            usage: BufferUsage::COPIES,
            location,
        })
        .unwrap()
}

/// An uploader on the backend's transfer and graphics queues.
fn uploader(backend: &VulkanTestBackend) -> Uploader {
    Uploader::new(
        backend.resources().clone(),
        backend.queue(QueueKind::Transfer).clone(),
        backend.queue(QueueKind::Graphics).clone(),
        "streaming",
    )
}

/// Checks that `upload` acquires `count` resources if the backend's
/// transfer queue is a queue of its own, and none otherwise.
fn check_acquire(backend: &VulkanTestBackend, upload: &Upload, count: usize) {
    let transfer = backend.queue(QueueKind::Transfer).id();
    let graphics = backend.queue(QueueKind::Graphics).id();
    if transfer == graphics {
        assert!(upload.acquire.is_empty(), "{:?}", upload.acquire);
    } else {
        assert_eq!(upload.acquire.len(), count, "{:?}", upload.acquire);
        assert!(
            upload
                .acquire
                .iter()
                .all(|barrier| barrier.queue_transfer() == Some((transfer, graphics)))
        );
    }
}

/// Translates `list` for the graphics queue, submits it waiting for
/// `upload`, and waits for it.
fn run_after(backend: &VulkanTestBackend, upload: &Upload, list: &CommandList) {
    let graphics = backend.queue(QueueKind::Graphics);
    let done = graphics
        .submit(SubmitDesc {
            command_buffers: vec![graphics.translate(list).unwrap()],
            wait_for: &[upload.timepoint],
        })
        .unwrap();
    graphics.wait(done).unwrap();
}

/// A texture uploaded on a spawned thread is sampled by a draw on the
/// test's thread, after waiting for the upload's timepoint and recording
/// its acquire barriers, and renders the colour uploaded, with no
/// validation errors (R17's `streamed_texture_is_sampled`). The spawned
/// thread enters the test's session span, so validation messages reach the
/// test.
#[test]
fn streamed_texture_is_sampled() {
    const SIZE: u32 = 4;
    const COLOUR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let extent = Extent3d::new_2d(SIZE, SIZE);
    let texture = |label, usage| {
        resources
            .create_texture(&TextureDesc {
                label,
                dimension: TextureDimension::D2,
                format: Format::Rgba32Float,
                extent,
                mip_levels: 1,
                array_layers: 1,
                usage,
            })
            .unwrap()
    };
    let source = texture(
        "streamed",
        TextureUsage::SAMPLED | TextureUsage::COPY_DESTINATION,
    );
    let target = texture(
        "target",
        TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE,
    );
    let whole = TextureCopy {
        texture: source,
        mip: 0,
        layers: 0..1,
        origin: [0; 3],
        extent,
    };
    let texels: Vec<u8> = (0..SIZE * SIZE)
        .flat_map(|_| COLOUR)
        .flat_map(f32::to_le_bytes)
        .collect();

    let mut uploader = uploader(&backend);
    let streamed = whole.clone();
    let upload = thread::scope(|scope| {
        let span = session.span().clone();
        scope
            .spawn(move || {
                span.in_scope(|| {
                    uploader
                        .upload_texture(streamed, &texels, Access::FragmentShaderSampledRead)
                        .unwrap();
                    uploader.flush().unwrap()
                })
            })
            .join()
            .unwrap()
    });
    check_acquire(&backend, &upload, 1);

    let program = resources
        .create_program(&test_shaders::sample_texture().desc())
        .unwrap();
    let sampler = resources
        .create_sampler(&SamplerDesc {
            label: "nearest",
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mip_filter: Filter::Nearest,
            address_modes: [AddressMode::ClampToEdge; 3],
        })
        .unwrap();
    let indices: [DescriptorIndex; 2] = [
        resources
            .texture_index(source, TextureView::sampled())
            .unwrap(),
        resources.sampler_index(sampler),
    ];
    let push_data: Vec<u8> = indices
        .iter()
        .flat_map(|index| index.0.to_le_bytes())
        .collect();
    let layout = TextureDataLayout::tightly_packed(Format::Rgba32Float, extent);
    let size = u64::from(layout.bytes_per_row * SIZE);
    let readback = create_buffer(&**resources, "readback", size, MemoryLocation::Readback);
    let attachment = Extent2d {
        width: SIZE,
        height: SIZE,
    };

    let mut list = CommandList::new("sample streamed texture");
    let mut barriers = upload.acquire.clone();
    barriers.push(Barrier::texture(
        target,
        Access::None,
        Access::ColorAttachmentWrite,
    ));
    list.barrier(&barriers);
    list.bind_program(program);
    list.push_data(0, &push_data);
    list.set_render_state(&RenderState::default().cull(CullMode::None));
    list.set_viewport(&Viewport::from_extent(attachment));
    list.set_scissor(Rect2d::from_extent(attachment));
    list.begin_render_pass(&RenderPassDesc::colour(target, LoadOp::DontCare));
    list.draw(3, 1);
    list.end_render_pass();
    list.barrier(&[Barrier::texture(
        target,
        Access::ColorAttachmentWrite,
        Access::CopyRead,
    )]);
    list.copy_texture_to_buffer(
        TextureCopy {
            texture: target,
            ..whole
        },
        resources.buffer_address(readback),
        layout,
    );
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run_after(&backend, &upload, &list);

    let mut bytes = vec![0; size as usize];
    resources.read_buffer(readback, 0, &mut bytes).unwrap();
    let (rendered, rest) = bytes.as_chunks::<16>();
    assert!(rest.is_empty());
    for texel in rendered {
        let (channels, _) = texel.as_chunks::<4>();
        let texel: Vec<f32> = channels
            .iter()
            .map(|&channel| f32::from_le_bytes(channel))
            .collect();
        assert_eq!(texel, COLOUR);
    }
    resources.destroy_program(program);
    resources.destroy_sampler(sampler);
    resources.destroy_buffer(readback);
    resources.destroy_texture(source);
    resources.destroy_texture(target);
}

/// Bytes uploaded to a buffer at an offset are copied out by the graphics
/// queue after it acquires them, in several flushes, each reusing the
/// staging buffer of the one before once it's complete. The bytes before
/// the offset, which the graphics queue filled and the uploads don't
/// touch, keep their contents, since only the uploaded bytes move between
/// queues.
#[test]
fn streamed_buffers_reach_the_graphics_queue() {
    const SIZE: u64 = 256;
    const OFFSET: u64 = 16;
    const FILL: u32 = 0xabab_abab;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let device = create_buffer(
        &**resources,
        "streamed",
        OFFSET + SIZE,
        MemoryLocation::Device,
    );
    let readback = create_buffer(
        &**resources,
        "readback",
        OFFSET + SIZE,
        MemoryLocation::Readback,
    );
    let address = |buffer| resources.buffer_address(buffer);
    let graphics = backend.queue(QueueKind::Graphics);
    let mut fill = CommandList::new("fill before the offset");
    fill.fill_buffer(address(device), OFFSET, FILL);
    let filled = graphics
        .submit(SubmitDesc::new(vec![graphics.translate(&fill).unwrap()]))
        .unwrap();
    graphics.wait(filled).unwrap();
    let mut uploader = uploader(&backend);

    for round in 0..3u8 {
        let bytes: Vec<u8> = (0..SIZE).map(|i| (i as u8).wrapping_add(round)).collect();
        uploader
            .upload_buffer(device, OFFSET, &bytes, Access::CopyRead)
            .unwrap();
        let upload = uploader.flush().unwrap();
        check_acquire(&backend, &upload, 1);

        let mut list = CommandList::new("read streamed buffer");
        let mut barriers = upload.acquire.clone();
        // The fill, or the last round's copy, wrote or read the bytes
        // before the offset, which stayed on the graphics queue.
        barriers.push(Barrier::buffer_range(
            device,
            0..OFFSET,
            Access::CopyRead | Access::CopyWrite,
            Access::CopyRead,
        ));
        list.barrier(&barriers);
        list.copy_buffer(address(device), address(readback), OFFSET + SIZE);
        list.barrier(&[
            Barrier::buffer(readback, Access::CopyWrite, Access::HostRead),
            // The next round's upload writes the buffer again.
            Barrier::buffer(device, Access::CopyRead, Access::CopyWrite),
        ]);
        run_after(&backend, &upload, &list);

        let mut out = vec![0; (OFFSET + SIZE) as usize];
        resources.read_buffer(readback, 0, &mut out).unwrap();
        let (before, uploaded) = out.split_at(OFFSET as usize);
        assert!(
            before.iter().all(|&byte| byte == 0xab),
            "round {round}: {before:?}"
        );
        assert_eq!(uploaded, bytes, "round {round}");
    }
    drop(uploader);
    resources.destroy_buffer(device);
    resources.destroy_buffer(readback);
}

/// BC7 blocks uploaded to every mip of a texture, down to mips smaller
/// than a block, are copied back out unchanged (R26's
/// `bc7_texture_round_trip`).
#[test]
fn bc7_texture_round_trip() {
    const MIPS: u32 = 4;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    require_capabilities!(backend, Capability::BcCompression);
    let resources = backend.resources();
    let extent = Extent3d::new_2d(8, 8);
    let texture = resources
        .create_texture(&TextureDesc {
            label: "compressed",
            dimension: TextureDimension::D2,
            format: Format::Bc7Unorm,
            extent,
            mip_levels: MIPS,
            array_layers: 1,
            usage: TextureUsage::SAMPLED
                | TextureUsage::COPY_SOURCE
                | TextureUsage::COPY_DESTINATION,
        })
        .unwrap();
    let mip = |mip: u32| TextureCopy {
        texture,
        mip,
        layers: 0..1,
        origin: [0; 3],
        extent: Extent3d::new_2d((8 >> mip).max(1), (8 >> mip).max(1)),
    };
    // Mip 0 is 2×2 blocks, and each smaller mip is one block, whatever its
    // size, so each mip's blocks start at a multiple of 16 bytes.
    let mip_bytes = |mip: u32| if mip == 0 { 64 } else { 16 };
    let blocks: Vec<Vec<u8>> = (0..MIPS)
        .map(|level| {
            (0..mip_bytes(level))
                .map(|i| (i * 7 + level as usize * 31) as u8)
                .collect()
        })
        .collect();
    let total: usize = blocks.iter().map(Vec::len).sum();

    let mut uploader = uploader(&backend);
    for (level, data) in (0..MIPS).zip(&blocks) {
        uploader
            .upload_texture(mip(level), data, Access::CopyRead)
            .unwrap();
    }
    let upload = uploader.flush().unwrap();
    check_acquire(&backend, &upload, MIPS as usize);

    let readback = create_buffer(
        &**resources,
        "readback",
        total as u64,
        MemoryLocation::Readback,
    );
    let mut list = CommandList::new("read compressed blocks");
    list.barrier(&upload.acquire);
    let mut offset = 0;
    for level in 0..MIPS {
        let region = mip(level);
        let layout = TextureDataLayout::tightly_packed(Format::Bc7Unorm, region.extent);
        list.copy_texture_to_buffer(
            region,
            GpuAddress(resources.buffer_address(readback).0 + offset),
            layout,
        );
        offset += mip_bytes(level) as u64;
    }
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run_after(&backend, &upload, &list);

    let mut out = vec![0; total];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, blocks.concat());
    drop(uploader);
    resources.destroy_buffer(readback);
    resources.destroy_texture(texture);
}

/// A flush with nothing uploaded still returns a timepoint, which is
/// reached, and needs no acquire barriers. Later ones return the last
/// flush's timepoint again, rather than submitting nothing for one.
#[test]
fn empty_flushes_complete() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let queue: &Arc<dyn Queue> = backend.queue(QueueKind::Transfer);
    let mut uploader = uploader(&backend);
    let first = uploader.flush().unwrap();
    assert!(first.acquire.is_empty());
    queue.wait(first.timepoint).unwrap();
    assert_eq!(uploader.flush().unwrap().timepoint, first.timepoint);

    let buffer = create_buffer(&**resources, "streamed", 16, MemoryLocation::Device);
    uploader
        .upload_buffer(buffer, 0, &[1; 16], Access::CopyRead)
        .unwrap();
    let upload = uploader.flush().unwrap();
    assert_ne!(upload.timepoint, first.timepoint);
    let empty = uploader.flush().unwrap();
    assert_eq!(empty.timepoint, upload.timepoint);
    assert!(empty.acquire.is_empty());
    queue.wait(empty.timepoint).unwrap();
    drop(uploader);
    resources.destroy_buffer(buffer);
}

/// An uploader given a queue other than the graphics queue as its graphics
/// queue panics at once, naming the queue's kind, rather than when
/// rendering records its acquire barriers.
#[test]
fn uploaders_need_a_graphics_queue() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    // Without a compute queue of its own, the compute queue is the
    // graphics queue.
    require_capabilities!(backend, Capability::DedicatedComputeQueue);
    let resources = backend.resources().clone();
    let transfer = backend.queue(QueueKind::Transfer).clone();
    let compute = backend.queue(QueueKind::Compute).clone();
    // Nothing is shared with the closure but what it consumes, so a panic
    // leaves nothing half-changed behind.
    let created = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        Uploader::new(resources, transfer, compute, "streaming")
    }));
    let message = created.expect_err("a compute queue isn't a graphics queue");
    let message = message
        .downcast_ref::<String>()
        .expect("the panic has a formatted message");
    assert!(
        message.contains("uploader \"streaming\" is given a Compute queue as its graphics queue"),
        "{message}"
    );
}

/// A flush with nothing uploaded reuses the staging of earlier flushes
/// whose copies are complete, so an uploader that stops uploading doesn't
/// hold its staging buffers in flight.
#[test]
fn empty_flushes_reclaim_staging() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let buffer = create_buffer(&**resources, "streamed", 16, MemoryLocation::Device);
    let mut uploader = uploader(&backend);
    uploader
        .upload_buffer(buffer, 0, &[1; 16], Access::CopyRead)
        .unwrap();
    let upload = uploader.flush().unwrap();
    assert_eq!(uploader.recording.staging().in_flight_len(), 1);
    // A buffer's copy is always on the transfer queue.
    backend
        .queue(QueueKind::Transfer)
        .wait(upload.timepoint)
        .unwrap();
    uploader.flush().unwrap();
    assert_eq!(uploader.recording.staging().in_flight_len(), 0);
    drop(uploader);
    resources.destroy_buffer(buffer);
}
