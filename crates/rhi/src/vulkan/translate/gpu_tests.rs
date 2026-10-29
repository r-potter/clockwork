//! T1 tests of translating and submitting command lists on a real device
//! (design 0002, R4 and R7).
//!
//! Synchronization validation doesn't track copies and fills by address
//! (`VK_KHR_device_address_commands`), so a missing barrier between them
//! isn't reported. R8's `missing_barrier_is_reported` waits for texture
//! copies, which it does track.

use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::test_support::VulkanTestBackend;
use crate::{
    Access, Barrier, Buffer, BufferDesc, BufferUsage, CommandList, GpuResources, MemoryLocation,
    QueueKind, RenderHardwareInterface, SubmitDesc,
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

/// A fill writes its value to every word of its range, and nothing
/// outside it, with a global barrier making it visible to the host.
#[test]
fn fills_write_their_range() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let readback = create(&**resources, "readback", 16, MemoryLocation::Readback);
    let address = resources.buffer_address(readback).0;

    let mut list = CommandList::new("fill");
    list.begin_label("clear");
    list.fill_buffer(crate::GpuAddress(address), 16, 0);
    list.end_label();
    list.barrier(&[Barrier::global(Access::CopyWrite, Access::CopyWrite)]);
    list.fill_buffer(crate::GpuAddress(address + 4), 8, 0x0403_0201);
    list.barrier(&[Barrier::global(Access::CopyWrite, Access::HostRead)]);
    run(&backend, &list);

    let mut out = [0xFF; 16];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, [0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 0]);
    resources.destroy_buffer(readback);
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
