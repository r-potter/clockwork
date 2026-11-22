//! Allocating transient memory from several threads at once (design 0002,
//! "Performance"): an allocation within a chunk, with eight threads
//! allocating at once, is budgeted at under 50 ns.
//!
//! Each run, eight threads each make [`PER_THREAD`] allocations from one
//! frame. They all fit in the frame's first chunk, which is taken before
//! the threads are released, so every allocation timed is within a chunk.
//! Between runs, the frame is ended and a new one begun, which isn't
//! timed, and garbage is collected, so that the new frame reuses the
//! chunk.

use std::hint::black_box;
use std::sync::RwLock;

use clockwork_rhi::{GpuResources, TransientFrame};
use criterion::Criterion;

use crate::{Context, THREADS, bench_threads};

/// The allocations each thread makes in a run.
const PER_THREAD: u64 = 1_024;

/// The size of each allocation, and its alignment: a typical draw's
/// constants.
const SIZE: u64 = 256;

/// Benchmarks allocating from one frame on [`THREADS`] threads at once, as
/// `transient/contended`. The time is for one allocation on one thread,
/// while the others allocate too.
pub(crate) fn bench(criterion: &mut Criterion, context: &Context) {
    let resources = context.resources();
    let frame = RwLock::new(Some(begin_frame(resources)));
    bench_threads(
        criterion,
        context,
        "transient/contended",
        &[(); THREADS],
        PER_THREAD,
        |()| {
            let frame = frame.read().expect("no thread panicked holding the lock");
            let frame = frame.as_ref().expect("a frame is begun between runs");
            for _ in 0..PER_THREAD {
                black_box(frame.allocate(SIZE, SIZE).expect("the chunk has room"));
            }
        },
        || {
            let mut frame = frame.write().expect("no thread panicked holding the lock");
            end_frame(resources, frame.take());
            *frame = Some(begin_frame(resources));
        },
    );
    let frame = frame
        .into_inner()
        .expect("no thread panicked holding the lock");
    end_frame(resources, frame);
}

/// Begins a frame, and takes its first chunk, so that the allocations
/// timed are all within it.
fn begin_frame(resources: &dyn GpuResources) -> TransientFrame {
    let frame = resources.begin_transient_frame();
    let _taken = frame.allocate(SIZE, SIZE).expect("a chunk can be taken");
    frame
}

/// Ends `frame`, if there is one, then collects garbage, so that its chunk
/// can be reused.
fn end_frame(resources: &dyn GpuResources, frame: Option<TransientFrame>) {
    if let Some(frame) = frame {
        resources.end_transient_frame(frame);
    }
    resources.collect_garbage();
}
