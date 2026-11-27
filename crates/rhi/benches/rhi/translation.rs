//! Translating command lists on the graphics queue (design 0002,
//! "Performance"): a list of 10,000 draws is budgeted at under 1 ms on one
//! thread, and eight such lists, each on its own thread at once, at under
//! 2 ms in total, so that lock contention doesn't erase the parallelism.
//!
//! The same lists are translated again and again, and each command buffer
//! is dropped unsubmitted, which returns its pool for the next translation
//! to claim, so after the first, translation allocates no pools. Only the
//! translation is timed, not dropping its command buffer.

use std::time::{Duration, Instant};

use clockwork_rhi::{CommandList, Program, Queue, QueueKind, RenderState, Texture};
use criterion::Criterion;

use crate::recording::{BATCH, begin_drawing, record_draw};
use crate::{Context, THREADS, bench_threads, render_target, triangle};

/// Benchmarks translating a list of [`BATCH`] draws, on one thread as
/// `translation/serial`, and on [`THREADS`] at once, each translating its
/// own, as `translation/parallel`.
pub(crate) fn bench(criterion: &mut Criterion, context: &Context) {
    let resources = context.resources();
    let queue = context.rhi.queue(QueueKind::Graphics).as_ref();
    let program = triangle(resources);
    let target = render_target(resources);
    let lists: [CommandList; THREADS] =
        std::array::from_fn(|thread| draw_list(&format!("draws {thread}"), program, target));

    criterion.bench_function("translation/serial", |bencher| {
        bencher.iter_custom(|iters| translate(queue, &lists[0], iters));
    });
    bench_threads(
        criterion,
        context,
        "translation/parallel",
        &lists,
        1,
        |list| {
            queue
                .translate_lists(&[list])
                .expect("translation succeeds")
        },
        || {},
    );

    resources.destroy_texture(target);
    resources.destroy_program(program);
}

/// A list called `label` of [`BATCH`] draws with `program`, rendering to
/// `target`, as the recording benchmark records them.
fn draw_list(label: &str, program: Program, target: Texture) -> CommandList {
    let state = RenderState::default();
    let mut list = CommandList::new(label);
    begin_drawing(&mut list, program, target);
    for draw in 0..BATCH {
        record_draw(&mut list, &state, draw);
    }
    list.end_render_pass();
    list
}

/// The time translating `list` on `queue` `iters` times takes.
fn translate(queue: &dyn Queue, list: &CommandList, iters: u64) -> Duration {
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let start = Instant::now();
        let commands = queue.translate_lists(&[list]);
        total += start.elapsed();
        drop(commands.expect("translation succeeds"));
    }
    total
}
