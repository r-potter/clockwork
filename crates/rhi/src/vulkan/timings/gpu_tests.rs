//! T1 tests of timing scopes on a real device: their timestamps written,
//! kept and read back, and their query chunks recycled (design 0002,
//! "Timestamps", R22).

use clockwork_diagnostics::capture::TestSession;

use crate::timing::RETAINED_SUBMISSIONS;
use crate::vulkan::test_support::VulkanTestBackend;
use crate::{
    Access, Barrier, Buffer, BufferDesc, BufferUsage, CommandList, GpuAddress, GpuResources,
    GpuTimepoint, MemoryLocation, QueueKind, RenderHardwareInterface, SubmitDesc, Timing,
};

/// A buffer of 4 KiB to fill, and its address.
fn create_buffer(resources: &dyn GpuResources) -> (Buffer, GpuAddress) {
    let buffer = resources
        .create_buffer(&BufferDesc {
            label: "timed fills",
            size: 4096,
            usage: BufferUsage::COPIES,
            location: MemoryLocation::Readback,
        })
        .unwrap();
    (buffer, resources.buffer_address(buffer))
}

/// Translates `list` for the graphics queue, submits it, and waits for it,
/// returning the timepoint it reached.
fn run(backend: &VulkanTestBackend, list: &CommandList) -> GpuTimepoint {
    let graphics = backend.queue(QueueKind::Graphics);
    let command_buffer = graphics.translate(list).unwrap();
    let done = graphics
        .submit(SubmitDesc::new(vec![command_buffer]))
        .unwrap();
    graphics.wait(done).unwrap();
    done
}

/// A scope's label, depth and whether it has a duration.
fn shape(timing: &Timing) -> (&str, u32, bool) {
    (
        timing.label.as_str(),
        timing.depth,
        timing.duration.is_some(),
    )
}

/// A fill timed inside an outer scope reports both durations, nested, with
/// the list's label, once, and recycles the chunk the timestamps used
/// (R22's `timings_are_reported`).
#[test]
fn timings_are_reported() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let (buffer, address) = create_buffer(&**resources);

    let mut list = CommandList::new("frame");
    list.begin_timing("frame");
    list.begin_timing("fill");
    list.fill_buffer(address, 4096, 0);
    list.end_timing();
    list.end_timing();
    let done = run(&backend, &list);

    let graphics = backend.queue(QueueKind::Graphics);
    let timings = graphics.take_timings(done).unwrap().unwrap();
    assert_eq!(timings.command_buffers.len(), 1);
    let command_buffer = &timings.command_buffers[0];
    assert_eq!(command_buffer.label, "frame");
    let shapes: Vec<_> = command_buffer.scopes.iter().map(shape).collect();
    assert_eq!(shapes, [("frame", 0, true), ("fill", 1, true)]);
    let [outer, inner] = [0, 1].map(|index| command_buffer.scopes[index].duration.unwrap());
    assert!(inner <= outer, "{inner:?} inside {outer:?}");

    assert_eq!(
        graphics
            .take_timings(done)
            .unwrap()
            .unwrap()
            .command_buffers,
        []
    );
    assert_eq!(backend.core.query_pools.free_len(), 1);
    resources.destroy_buffer(buffer);
}

/// A list with more scopes than one chunk has queries for claims another,
/// and every scope still gets a duration.
#[test]
fn scopes_beyond_one_chunk_claim_another() {
    const SCOPES: usize = 200;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let (buffer, address) = create_buffer(&**resources);

    let mut list = CommandList::new("many scopes");
    for index in 0..SCOPES {
        if index > 0 {
            // Each fill overwrites the last, as the RHI's users would order.
            list.barrier(&[Barrier::buffer(
                buffer,
                Access::CopyWrite,
                Access::CopyWrite,
            )]);
        }
        list.begin_timing(&format!("fill {index}"));
        list.fill_buffer(address, 4096, 0);
        list.end_timing();
    }
    let done = run(&backend, &list);

    let graphics = backend.queue(QueueKind::Graphics);
    let timings = graphics.take_timings(done).unwrap().unwrap();
    let scopes = &timings.command_buffers[0].scopes;
    assert_eq!(scopes.len(), SCOPES);
    assert!(scopes.iter().all(|scope| scope.duration.is_some()));
    assert_eq!(scopes[SCOPES - 1].label, format!("fill {}", SCOPES - 1));
    assert_eq!(backend.core.query_pools.free_len(), 2);
    resources.destroy_buffer(buffer);
}

/// A command buffer dropped unsubmitted recycles its chunks straight away.
#[test]
fn unsubmitted_command_buffers_recycle_their_chunks() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let mut list = CommandList::new("dropped");
    list.begin_timing("nothing");
    list.end_timing();
    let command_buffer = backend.queue(QueueKind::Graphics).translate(&list).unwrap();
    assert_eq!(backend.core.query_pools.free_len(), 0);
    drop(command_buffer);
    assert_eq!(backend.core.query_pools.free_len(), 1);
}

/// Once more completed submissions with timings are left untaken than a
/// queue keeps, the oldest's are discarded, and taking them returns none,
/// while the newest's are still there. The rest are still kept when the
/// backend is dropped, which recycles their chunks.
#[test]
fn untaken_timings_are_discarded_beyond_the_limit() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let mut list = CommandList::new("untaken");
    list.begin_timing("nothing");
    list.end_timing();
    let timepoints: Vec<_> = (0..=RETAINED_SUBMISSIONS)
        .map(|_| run(&backend, &list))
        .collect();

    let graphics = backend.queue(QueueKind::Graphics);
    let oldest = graphics.take_timings(timepoints[0]).unwrap().unwrap();
    assert_eq!(oldest.command_buffers, []);
    let newest = graphics
        .take_timings(*timepoints.last().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(newest.command_buffers.len(), 1);
}
