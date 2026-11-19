//! Compiles the test shaders in `shaders/tests/` to SPIR-V with `slangc`,
//! writing `<name>.spv` files to `OUT_DIR` for the tests to embed, with heap
//! access through `SPV_EXT_descriptor_heap`. The engine draws nothing
//! itself, so it has no shaders of its own: `shaders/` holds only the
//! modules shaders import, which the application's shaders import too.
//!
//! `slangc` is found by [`slangc::find`]: from the `SLANGC` environment
//! variable, then the Vulkan SDK, then `PATH`.

#[path = "build/compile.rs"]
mod compile;
#[path = "build/slangc.rs"]
mod slangc;

use std::env;
use std::path::{Path, PathBuf};

/// The shaders only tests use (design 0002, "Test plan"), as file names in
/// `shaders/tests/` without the `.slang` extension. The tests describe them
/// in `src/test_shaders.rs`.
const TEST_SHADERS: &[&str] = &[
    "triangle",
    "write_value",
    "count_groups",
    "fill_texels",
    "copy_texels",
    "downsample",
    "sample_texture",
    "read_colour",
];

/// The Slang modules shaders import, as file names in `shaders/` without
/// the `.slang` extension. They're compiled as part of each shader that
/// imports them, so they only need to rebuild the shaders when they change.
/// `shaders/` is on the include path, so the test shaders find them too.
const MODULES: &[&str] = &["heap"];

/// Finds `slangc`, then compiles each shader in [`TEST_SHADERS`].
///
/// # Panics
///
/// Panics, failing the build, if `slangc` can't be found or run, or if a
/// shader fails to compile.
fn main() {
    let slangc = compile::find_slangc();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    for name in MODULES {
        let source = PathBuf::from_iter(["shaders", &format!("{name}.slang")]);
        println!("cargo::rerun-if-changed={}", source.display());
    }
    let shaders = Path::new("shaders");
    let tests = shaders.join("tests");
    for name in TEST_SHADERS {
        let source = tests.join(format!("{name}.slang"));
        let output = out_dir.join(format!("{name}.spv"));
        compile::compile(&slangc, &source, shaders, &output);
    }
}
