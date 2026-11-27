//! Benchmarks of compiling (design 0003, "Performance"): a graph of 64
//! passes and 128 resources, budgeted at under 1 ms, including scheduling
//! and placement.
//!
//! Placement comes with the `graph-aliasing` slice, and the `Overlap`
//! schedule with `graph-queues`, so for now this times every other step
//! under `AsWritten`: validation and expansion, the dependency graph,
//! culling, scheduling, barriers, and load and store operations. Benchmarks
//! are built in release, so the plan checker, which debug builds run after
//! every compile, isn't timed.
//!
//! Run it with `cargo bench -p clockwork-frame-graph`. It needs no GPU.

use std::hint::black_box;

use clockwork_frame_graph::{
    Attachment, BufferId, CompileInputs, ComputeRecorder, GraphDesc, MipCount, OutputDesc,
    QueueLayout, Schedule, TextureId, TextureSize, TransientTexture, compile,
};
use clockwork_rhi::{Access, Extent2d, Extent3d, Format, QueueId};
use criterion::{Criterion, criterion_group, criterion_main};

/// The passes before the one that writes the output, which make 64 with
/// it.
const PASSES: usize = 63;

/// The index of the colour target that holds the last frame, after those
/// the passes write.
const HISTORY: usize = PASSES;

/// The output's extent: a 1080p surface.
const OUTPUT: Extent2d = Extent2d {
    width: 1920,
    height: 1080,
};

/// A graph of 64 passes and 128 resources, shaped like a frame's: raster
/// passes that render to full-size targets with depth, and sample earlier
/// targets and mips; compute passes that write targets, buffers and mips;
/// and transfer passes that copy between mips and into targets.
///
/// - 64 colour targets, the size of the output: one written by each pass
///   but the last, every 16th of them persistent, and a persistent
///   history, which the first pass samples from the last frame and the
///   last pass copies the frame into
/// - 16 depth targets, one per group of four passes, cleared by its first
///   pass and tested by its last
/// - 16 textures of 4 mips, each filled at mip 0 by a compute pass, copied
///   to mip 1 by a transfer pass, and sampled there by a raster pass
/// - 32 buffers, the even ones transient, written by compute passes, and
///   the odd ones persistent, read before any pass writes them
///
/// Pass `i` writes target `i`, and every pass has side effects, so culling
/// keeps them all, and each step has the whole graph to work on. The last
/// pass copies the last target to the history and the output.
fn graph() -> GraphDesc {
    let mut graph = GraphDesc::new("benchmark");
    let full = TextureSize::Output { scale: 1.0 };
    let targets: Vec<TextureId> = (0..64)
        .map(|i| {
            let name = format!("target {i}");
            let desc = TransientTexture::new(Format::Rgba16Float, full);
            if i % 16 == 0 || i == HISTORY {
                graph.persistent_texture(&name, desc)
            } else {
                graph.transient_texture(&name, desc)
            }
        })
        .collect();
    let depths: Vec<TextureId> = (0..16)
        .map(|i| {
            let desc = TransientTexture::new(Format::Depth32Float, full);
            graph.transient_texture(&format!("depth {i}"), desc)
        })
        .collect();
    let chains: Vec<TextureId> = (0..16)
        .map(|i| {
            let desc = TransientTexture {
                mips: MipCount::Count(4),
                ..TransientTexture::new(
                    Format::Rgba16Float,
                    TextureSize::Fixed(Extent3d::new_2d(512, 512)),
                )
            };
            graph.transient_texture(&format!("chain {i}"), desc)
        })
        .collect();
    let buffers: Vec<BufferId> = (0..32)
        .map(|i| {
            let name = format!("buffer {i}");
            if i % 2 == 0 {
                graph.transient_buffer(&name, 4096)
            } else {
                graph.persistent_buffer(&name, 4096)
            }
        })
        .collect();

    for i in 0..PASSES {
        let name = format!("pass {i}");
        let (target, depth, chain) = (targets[i], depths[i / 4], chains[i / 4]);
        match i % 4 {
            0 => {
                let mut pass = graph.raster_pass(&name, "draw");
                pass.colour(target, Attachment::Clear([0.0; 4]))
                    .depth(depth, Attachment::Clear(0.0))
                    .read(buffers[i / 2 + 1], Access::FragmentShaderRead)
                    .side_effects();
                if i >= 4 {
                    pass.read(targets[i - 1], Access::FragmentShaderSampledRead)
                        .read(targets[i - 3], Access::FragmentShaderSampledRead);
                } else {
                    pass.read(targets[HISTORY], Access::FragmentShaderSampledRead);
                }
            }
            1 => {
                graph
                    .compute_pass(&name, ComputeRecorder::recorder("compute"))
                    .read(targets[i - 1], Access::ComputeShaderSampledRead)
                    .write(target, Access::ComputeShaderWrite)
                    .write(chain.mip(0), Access::ComputeShaderWrite)
                    .write(buffers[i / 2], Access::ComputeShaderWrite)
                    .side_effects();
            }
            2 => {
                graph
                    .transfer_pass(&name, "copy")
                    .read(chain.mip(0), Access::CopyRead)
                    .write(chain.mip(1), Access::CopyWrite)
                    .read(buffers[i / 2 - 1], Access::CopyRead)
                    .write(target, Access::CopyWrite)
                    .side_effects();
            }
            _ => {
                graph
                    .raster_pass(&name, "draw")
                    .colour(target, Attachment::Discard)
                    .depth(depth, Attachment::Keep)
                    .read(targets[i - 2], Access::FragmentShaderSampledRead)
                    .read(chain.mip(1), Access::FragmentShaderSampledRead)
                    .read(buffers[i / 2], Access::VertexShaderRead)
                    .side_effects();
            }
        }
    }
    let output = graph.output();
    graph
        .transfer_pass("present", "copy")
        .read(targets[PASSES - 1], Access::CopyRead)
        .write(targets[HISTORY], Access::CopyWrite)
        .write(output, Access::CopyWrite);
    graph
}

/// Times compiling [`graph`] for a 1080p surface on one queue.
fn compiling(c: &mut Criterion) {
    let desc = graph();
    let inputs = CompileInputs {
        output: OutputDesc::surface(Format::Bgra8Srgb, OUTPUT),
        queues: QueueLayout::shared(QueueId::new(0)),
        schedule: Schedule::AsWritten,
    };
    compile(&desc, &inputs).expect("the benchmark's graph is valid");
    c.bench_function("compile/64 passes", |b| {
        b.iter(|| compile(black_box(&desc), black_box(&inputs)));
    });
}

criterion_group!(benches, compiling);
criterion_main!(benches);
