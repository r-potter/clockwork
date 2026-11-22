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
//! into a command list.
//!
//! Run them with `cargo bench -p clockwork-rhi`. They need a GPU.

mod recording;

use clockwork_diagnostics::capture::TestSession;
use clockwork_rhi::{
    BackendOptions, Extent2d, Extent3d, Format, GpuResources, GraphicsApi, Program, ProgramDesc,
    RenderHardwareInterface, ShaderStage, Texture, TextureDesc, TextureDimension, TextureUsage,
    Validation, create,
};
use criterion::Criterion;

/// The size of the texture draws render to. Recording never renders, so
/// it only needs to be valid.
const TARGET_EXTENT: Extent2d = Extent2d {
    width: 64,
    height: 64,
};

/// Creates the backend, runs every benchmark group with it, then reports
/// criterion's summary.
fn main() {
    let session = TestSession::new();
    let rhi = backend(&session);
    let mut criterion = Criterion::default().configure_from_args();
    recording::bench(&mut criterion, rhi.resources().as_ref());
    criterion.final_summary();
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
