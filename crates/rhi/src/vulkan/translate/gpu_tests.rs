//! T1 tests of translating and submitting command lists on a real device
//! (design 0002, R4, R7, R9 and R16).
//!
//! Synchronization validation doesn't track copies and fills by address
//! (`VK_KHR_device_address_commands`), so a missing barrier between them
//! isn't reported. R8's `missing_barrier_is_reported` waits for texture
//! copies, which it does track.

use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::test_support::VulkanTestBackend;
use crate::{
    Access, Barrier, Buffer, BufferDesc, BufferUsage, CommandList, Extent3d, Format, GpuResources,
    MemoryLocation, QueueKind, RenderHardwareInterface, SubmitDesc, Subresource, Texture,
    TextureDesc, TextureDimension, TextureUsage,
};

/// A buffer of `size` bytes in `location`, for copies.
fn create(
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

/// Translates `list` for the graphics queue, submits it, and waits for it.
fn run(backend: &VulkanTestBackend, list: &CommandList) {
    let graphics = backend.queue(QueueKind::Graphics);
    let command_buffer = graphics.translate(list).unwrap();
    let done = graphics
        .submit(SubmitDesc::new(vec![command_buffer]))
        .unwrap();
    graphics.wait(done).unwrap();
}

/// Bytes written to an upload buffer, copied on the GPU to a GPU-only
/// buffer and then to a readback buffer, read back unchanged (R4's
/// `buffer_round_trip`), with a barrier before each access, so
/// synchronization validation reports nothing.
#[test]
fn buffer_round_trip() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let bytes: Vec<u8> = (0..=255).collect();
    let size = bytes.len() as u64;
    let upload = create(&**resources, "upload", size, MemoryLocation::Upload);
    let device = create(&**resources, "device", size, MemoryLocation::Device);
    let readback = create(&**resources, "readback", size, MemoryLocation::Readback);
    resources.write_buffer(upload, 0, &bytes);

    let address = |buffer| resources.buffer_address(buffer);
    let mut list = CommandList::new("round trip");
    list.barrier(&[Barrier::buffer(upload, Access::HostWrite, Access::CopyRead)]);
    list.copy_buffer(address(upload), address(device), size);
    list.barrier(&[Barrier::buffer(device, Access::CopyWrite, Access::CopyRead)]);
    list.copy_buffer(address(device), address(readback), size);
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, &list);

    let mut out = vec![0; bytes.len()];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, bytes);
    for buffer in [upload, device, readback] {
        resources.destroy_buffer(buffer);
    }
}

/// A copy on the transfer queue, waited on by a copy on the graphics queue,
/// with the buffer released by one and acquired by the other (R9's
/// `cross_queue_wait`). On a device without a dedicated transfer family,
/// both run on the graphics queue, the transfer is dropped, and the test
/// still passes.
#[test]
fn cross_queue_wait() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let transfer = backend.queue(QueueKind::Transfer);
    let graphics = backend.queue(QueueKind::Graphics);
    let bytes: Vec<u8> = (0..=255).rev().collect();
    let size = bytes.len() as u64;
    let upload = create(&**resources, "upload", size, MemoryLocation::Upload);
    let device = create(&**resources, "device", size, MemoryLocation::Device);
    let readback = create(&**resources, "readback", size, MemoryLocation::Readback);
    resources.write_buffer(upload, 0, &bytes);
    let address = |buffer| resources.buffer_address(buffer);
    let handover = Barrier::buffer(device, Access::CopyWrite, Access::CopyRead)
        .with_queue_transfer(QueueKind::Transfer, QueueKind::Graphics);

    let mut upload_list = CommandList::new("upload");
    upload_list.barrier(&[Barrier::buffer(upload, Access::HostWrite, Access::CopyRead)]);
    upload_list.copy_buffer(address(upload), address(device), size);
    upload_list.barrier(std::slice::from_ref(&handover));
    let uploaded = transfer
        .submit(SubmitDesc::new(vec![
            transfer.translate(&upload_list).unwrap(),
        ]))
        .unwrap();

    let mut read_list = CommandList::new("read");
    read_list.barrier(std::slice::from_ref(&handover));
    read_list.copy_buffer(address(device), address(readback), size);
    read_list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    let read = graphics
        .submit(SubmitDesc {
            command_buffers: vec![graphics.translate(&read_list).unwrap()],
            wait_for: &[uploaded],
        })
        .unwrap();
    graphics.wait(read).unwrap();

    let mut out = vec![0; bytes.len()];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, bytes);
    for buffer in [upload, device, readback] {
        resources.destroy_buffer(buffer);
    }
}

/// Eight threads each translate a list that fills a different region of
/// one buffer, and the command buffers are submitted as one batch, in a
/// chosen order (R16's `parallel_translation`). Each thread enters the
/// test's session span, so validation messages reach the test.
#[test]
fn parallel_translation() {
    const THREADS: u64 = 8;
    /// The bytes each thread fills.
    const REGION: u64 = 16;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let graphics = backend.queue(QueueKind::Graphics);
    let readback = create(
        &**resources,
        "readback",
        THREADS * REGION,
        MemoryLocation::Readback,
    );
    let base = resources.buffer_address(readback).0;

    let lists: Vec<_> = (0..THREADS)
        .map(|index| {
            let mut list = CommandList::new(&format!("region {index}"));
            let value = u32::from_le_bytes([index as u8 + 1; 4]);
            list.fill_buffer(crate::GpuAddress(base + index * REGION), REGION, value);
            list
        })
        .collect();
    let mut command_buffers: Vec<_> = std::thread::scope(|scope| {
        let threads: Vec<_> = lists
            .iter()
            .map(|list| {
                let span = session.span().clone();
                scope.spawn(move || span.in_scope(|| graphics.translate(list).unwrap()))
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    });
    let mut last = CommandList::new("make visible");
    last.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    command_buffers.push(graphics.translate(&last).unwrap());
    let done = graphics.submit(SubmitDesc::new(command_buffers)).unwrap();
    graphics.wait(done).unwrap();

    let mut out = vec![0; (THREADS * REGION) as usize];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    for (index, region) in out.chunks(REGION as usize).enumerate() {
        assert!(
            region.iter().all(|&byte| byte == index as u8 + 1),
            "region {index}: {region:?}"
        );
    }
    resources.destroy_buffer(readback);
}

/// A fill writes its value to every word of its range, and nothing
/// outside it, with a global barrier making it visible to the host. Debug
/// regions and labels around the fills translate with no validation error.
#[test]
fn fills_write_their_range() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let readback = create(&**resources, "readback", 16, MemoryLocation::Readback);
    let address = resources.buffer_address(readback).0;

    let mut list = CommandList::new("fill");
    list.begin_debug_region("clear");
    list.fill_buffer(crate::GpuAddress(address), 16, 0);
    list.insert_debug_label("cleared");
    list.end_debug_region();
    list.insert_debug_label("patterned");
    list.barrier(&[Barrier::global(Access::CopyWrite, Access::CopyWrite)]);
    list.fill_buffer(crate::GpuAddress(address + 4), 8, 0x0403_0201);
    list.barrier(&[Barrier::global(Access::CopyWrite, Access::HostRead)]);
    run(&backend, &list);

    let mut out = [0xFF; 16];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, [0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 0]);
    resources.destroy_buffer(readback);
}

/// A 4×4 texture of `format`, with three mips and two layers, for
/// `usage`.
fn create_texture(
    resources: &dyn GpuResources,
    label: &str,
    format: Format,
    usage: TextureUsage,
) -> Texture {
    resources
        .create_texture(&TextureDesc {
            label,
            dimension: TextureDimension::D2,
            format,
            extent: Extent3d::new_2d(4, 4),
            mip_levels: 3,
            array_layers: 2,
            usage,
        })
        .unwrap()
}

/// Texture barriers move each subresource they name between the layouts
/// their accesses need, so different mips can be in different layouts,
/// which validation checks against each image's tracked layouts when it's
/// submitted. A depth texture's barriers use its depth aspect.
#[test]
fn texture_barriers_move_subresources_between_layouts() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let colour = create_texture(
        &**resources,
        "colour",
        Format::Rgba8Unorm,
        TextureUsage::SAMPLED | TextureUsage::COPY_SOURCE | TextureUsage::COPY_DESTINATION,
    );
    let depth = create_texture(
        &**resources,
        "depth",
        Format::Depth32Float,
        TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLED,
    );

    let mut list = CommandList::new("layouts");
    list.barrier(&[
        Barrier::texture(colour, Access::None, Access::CopyWrite),
        Barrier::texture(depth, Access::None, Access::DepthAttachmentWrite),
    ]);
    list.barrier(&[
        Barrier::subresource(
            colour,
            Subresource::mip(0),
            Access::CopyWrite,
            Access::CopyRead,
        ),
        Barrier::subresource(
            colour,
            Subresource {
                mips: 1..Subresource::REMAINING,
                layers: 0..Subresource::REMAINING,
            },
            Access::CopyWrite,
            Access::FragmentShaderSampledRead,
        ),
    ]);
    list.barrier(&[
        Barrier::subresource(
            colour,
            Subresource::mip_layer(0, 1),
            Access::CopyRead,
            Access::FragmentShaderSampledRead,
        ),
        Barrier::texture(
            depth,
            Access::DepthAttachmentWrite,
            Access::FragmentShaderSampledRead,
        ),
    ]);
    run(&backend, &list);
    resources.destroy_texture(colour);
    resources.destroy_texture(depth);
}

/// A barrier naming a mip the texture doesn't have panics when the list is
/// translated, naming the texture.
#[test]
#[should_panic(expected = "texture \"colour\" has 3 mips, so a subresource can't name mips 3..4")]
fn subresources_outside_their_texture_panic() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let colour = create_texture(
        &**resources,
        "colour",
        Format::Rgba8Unorm,
        TextureUsage::SAMPLED,
    );
    let mut list = CommandList::new("past the last mip");
    list.barrier(&[Barrier::subresource(
        colour,
        Subresource::mip(3),
        Access::None,
        Access::FragmentShaderSampledRead,
    )]);
    let _ = backend.queue(QueueKind::Graphics).translate(&list);
}

/// A command buffer's pool is reused once its submission completes, and an
/// unsubmitted one's as soon as it's dropped, so translating every frame
/// doesn't create pools.
#[test]
fn command_pools_are_reused() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let graphics = backend.queue(QueueKind::Graphics);
    let list = CommandList::new("empty");
    let pool = |command_buffer: &crate::CommandBuffer| command_buffer.0.commands.unwrap().pool;

    let submitted = graphics.translate(&list).unwrap();
    let first = pool(&submitted);
    let done = graphics.submit(SubmitDesc::new(vec![submitted])).unwrap();
    graphics.wait(done).unwrap();
    let dropped = graphics.translate(&list).unwrap();
    assert_eq!(pool(&dropped), first);
    drop(dropped);
    assert_eq!(pool(&graphics.translate(&list).unwrap()), first);
}

/// A command buffer can only be submitted to the queue it was translated
/// for, since it was recorded for that queue's family.
#[test]
#[should_panic(
    expected = "command list \"copies\" was translated for the graphics queue, so it can't be \
                submitted to the transfer queue"
)]
fn command_buffers_belong_to_their_queue() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let command_buffer = backend
        .queue(QueueKind::Graphics)
        .translate(&CommandList::new("copies"))
        .unwrap();
    command_buffer.0.check_queue(QueueKind::Transfer);
}
