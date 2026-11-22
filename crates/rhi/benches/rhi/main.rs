//! Benchmarks of the RHI's own CPU time (design 0002, "Performance"),
//! measured through its public API, as callers pay for it.
//!
//! Every benchmark shares one headless backend, created without
//! validation, since the validation layers would time themselves rather
//! than the RHI. It's created in a [`TestSession`], as the T1 tests'
//! backends are, so a warning or error it logs fails the run when the
//! session ends. The device is the one device selection prefers.
//!
//! Each benchmark group is a module: [`recording`] times recording draws
//! into a command list, and [`translation`] translating lists of them, on
//! one thread and on several at once.
//!
//! Run them with `cargo bench -p clockwork-rhi`. They need a GPU.

mod recording;
mod translation;

use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use std::{panic, process};

use clockwork_diagnostics::capture::TestSession;
use clockwork_rhi::{
    BackendOptions, Extent2d, Extent3d, Format, GpuResources, GraphicsApi, Program, ProgramDesc,
    RenderHardwareInterface, ShaderStage, Texture, TextureDesc, TextureDimension, TextureUsage,
    Validation, create,
};
use criterion::Criterion;
use tracing::Span;

/// The size of the texture draws render to. Nothing is submitted, so it
/// only needs to be valid.
const TARGET_EXTENT: Extent2d = Extent2d {
    width: 64,
    height: 64,
};

/// The threads that work at once in benchmarks of contention, as many as
/// design 0002's budgets assume.
const THREADS: usize = 8;

/// What every benchmark group is given.
struct Context {
    /// The backend.
    rhi: Box<dyn RenderHardwareInterface>,
    /// The session's span, for threads a benchmark starts to enter, so that
    /// what they log is attributed to the session too.
    span: Span,
}

impl Context {
    /// The backend's resources.
    fn resources(&self) -> &dyn GpuResources {
        self.rhi.resources().as_ref()
    }
}

/// Creates the backend, runs every benchmark group with it, then reports
/// criterion's summary.
fn main() {
    abort_on_panic();
    let session = TestSession::new();
    let context = Context {
        rhi: backend(&session),
        span: session.span().clone(),
    };
    let mut criterion = Criterion::default().configure_from_args();
    recording::bench(&mut criterion, &context);
    translation::bench(&mut criterion, &context);
    criterion.final_summary();
}

/// Makes any panic, on any thread, abort the process once its message is
/// printed, so that a benchmark that fails exits with an error rather than
/// hanging.
///
/// [`bench_threads`] needs this: its threads and the main thread wait for
/// each other at barriers, which a panic doesn't release, so a thread that
/// panics would leave the others waiting forever.
fn abort_on_panic() {
    let print = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        print(info);
        process::abort();
    }));
}

/// Benchmarks running `work` on [`THREADS`] threads at once, as `name`,
/// with each thread's input from `inputs`, and counting each run as
/// `operations` iterations.
///
/// The threads are started once, rather than for each run, so that
/// starting them isn't timed. Each run releases them together, and is
/// timed from their release until the last has finished its work. Each
/// then drops what its work returned, and `between` prepares the next run,
/// while the threads wait to be released again, and neither is timed.
///
/// Whole runs are timed, so the time is scaled to the iterations criterion
/// asks for. It asks for thousands or more, so the last run's remainder
/// makes no difference.
fn bench_threads<I: Sync, R>(
    criterion: &mut Criterion,
    context: &Context,
    name: &str,
    inputs: &[I; THREADS],
    operations: u64,
    work: impl Fn(&I) -> R + Sync,
    mut between: impl FnMut(),
) {
    let start = Barrier::new(THREADS + 1);
    let end = Barrier::new(THREADS + 1);
    let stop = AtomicBool::new(false);
    thread::scope(|scope| {
        for input in inputs {
            let (start, end, stop, work) = (&start, &end, &stop, &work);
            let span = context.span.clone();
            scope.spawn(move || {
                let _entered = span.enter();
                loop {
                    start.wait();
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let result = work(input);
                    end.wait();
                    drop(result);
                }
            });
        }
        criterion.bench_function(name, |bencher| {
            bencher.iter_custom(|iters| {
                let runs = iters.div_ceil(operations);
                let mut total = Duration::ZERO;
                for _ in 0..runs {
                    start.wait();
                    let released = Instant::now();
                    end.wait();
                    total += released.elapsed();
                    between();
                }
                total.mul_f64(iters as f64 / (runs * operations) as f64)
            });
        });
        stop.store(true, Ordering::Relaxed);
        start.wait();
    });
}

/// A headless Vulkan backend in `session`, on the device selection prefers,
/// without validation.
///
/// # Panics
///
/// Panics if the backend can't be created.
fn backend(session: &TestSession) -> Box<dyn RenderHardwareInterface> {
    let options = BackendOptions {
        session: session.id(),
        device_filter: None,
        validation: Validation::Off,
    };
    create(GraphicsApi::Vulkan, None, &options)
        .unwrap_or_else(|err| panic!("failed to create a Vulkan backend: {err:?}"))
}

/// The vertex and fragment program that draws a triangle with no bound
/// resources, which the RHI's T1 tests draw too
/// (`shaders/tests/triangle.slang`, compiled by `build.rs`).
///
/// # Panics
///
/// Panics if the program can't be created.
fn triangle(resources: &dyn GpuResources) -> Program {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.spv"));
    // `include_bytes!` doesn't align the bytes for `u32`s, so they're
    // copied into words. SPIR-V that `slangc` writes is little-endian.
    let (words, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "the triangle's SPIR-V isn't whole words");
    let spirv: Vec<u32> = words.iter().map(|&word| u32::from_le_bytes(word)).collect();
    resources
        .create_program(&ProgramDesc {
            label: "triangle",
            spirv: &spirv,
            stages: &[
                (ShaderStage::Vertex, c"vertexMain"),
                (ShaderStage::Fragment, c"fragmentMain"),
            ],
        })
        .unwrap_or_else(|err| panic!("failed to create the triangle program: {err:?}"))
}

/// A texture for draws to render to, [`TARGET_EXTENT`] in size.
///
/// # Panics
///
/// Panics if the texture can't be created.
fn render_target(resources: &dyn GpuResources) -> Texture {
    resources
        .create_texture(&TextureDesc {
            label: "render target",
            dimension: TextureDimension::D2,
            format: Format::Rgba8Unorm,
            extent: Extent3d::new_2d(TARGET_EXTENT.width, TARGET_EXTENT.height),
            mip_levels: 1,
            array_layers: 1,
            usage: TextureUsage::COLOUR_ATTACHMENT,
        })
        .unwrap_or_else(|err| panic!("failed to create the render target: {err:?}"))
}
