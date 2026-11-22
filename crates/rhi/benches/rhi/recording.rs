//! Recording draws into a command list (design 0002, "Performance"): a
//! draw's `set_render_state`, `push_data` and `draw` together are budgeted
//! at under 50 ns.
//!
//! The draws are recorded into one list, reset before each batch, so
//! once the first batch has grown its arenas, recording allocates nothing,
//! as with a list recorded again each frame. Only the draws are timed, not
//! resetting the list or beginning and ending its render pass.

use std::hint::black_box;
use std::time::{Duration, Instant};

use clockwork_rhi::{
    CommandList, LoadOp, Program, Rect2d, RenderPassDesc, RenderState, Texture, Viewport,
};
use criterion::Criterion;

use crate::{Context, TARGET_EXTENT, render_target, triangle};

/// The most draws in one batch: as many as a list of a busy pass holds.
pub(crate) const BATCH: u64 = 10_000;

/// Benchmarks recording a draw, as `recording/draw`.
pub(crate) fn bench(criterion: &mut Criterion, context: &Context) {
    let resources = context.resources();
    let program = triangle(resources);
    let target = render_target(resources);
    let mut list = CommandList::new("recording");
    criterion.bench_function("recording/draw", |bencher| {
        bencher.iter_custom(|iters| record(&mut list, program, target, iters));
    });
    resources.destroy_texture(target);
    resources.destroy_program(program);
}

/// The time recording `draws` draws into `list` takes, with `program`,
/// rendering to `target`, in batches of at most [`BATCH`].
fn record(list: &mut CommandList, program: Program, target: Texture, draws: u64) -> Duration {
    let state = RenderState::default();
    let mut total = Duration::ZERO;
    let mut done = 0;
    while done < draws {
        let batch = (draws - done).min(BATCH);
        list.reset();
        begin_drawing(list, program, target);
        let start = Instant::now();
        for draw in 0..batch {
            record_draw(list, black_box(&state), draw);
        }
        total += start.elapsed();
        list.end_render_pass();
        done += batch;
    }
    total
}

/// Binds `program` in `list`, sets the viewport and scissor to cover
/// `target`, and begins a render pass to it, ready to draw.
pub(crate) fn begin_drawing(list: &mut CommandList, program: Program, target: Texture) {
    list.bind_program(program);
    list.set_viewport(&Viewport::from_extent(TARGET_EXTENT));
    list.set_scissor(Rect2d::from_extent(TARGET_EXTENT));
    list.begin_render_pass(&RenderPassDesc::colour(target, LoadOp::Clear([0.0; 4])));
}

/// Records the draw numbered `draw` into `list`, with `state`, and push
/// data typical of a draw: the address of its constants, and an index.
pub(crate) fn record_draw(list: &mut CommandList, state: &RenderState, draw: u64) {
    let mut data = [0; 16];
    data[..8].copy_from_slice(&(0x1_0000_0000 + draw * 256).to_le_bytes());
    data[8..12].copy_from_slice(&(draw as u32).to_le_bytes());
    list.set_render_state(state);
    list.push_data(0, black_box(&data));
    list.draw(3, 1);
}
