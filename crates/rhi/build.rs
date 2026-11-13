//! Compiles the Slang shaders in `shaders/`, and the test shaders in
//! `shaders/tests/`, to SPIR-V with `slangc`, writing `<name>.spv` files to
//! `OUT_DIR` for the Vulkan backend and the tests to embed, with heap
//! access through `SPV_EXT_descriptor_heap`.
//!
//! `slangc` is found by [`slangc::find`]: from the `SLANGC` environment
//! variable, then the Vulkan SDK, then `PATH`.

#[path = "build/slangc.rs"]
mod slangc;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The engine's shaders, as file names in `shaders/` without the `.slang`
/// extension.
const SHADERS: &[&str] = &["triangle"];

/// The shaders only tests use (design 0002, "Test plan"), as file names in
/// `shaders/tests/` without the `.slang` extension. The tests describe them
/// in `src/test_shaders.rs`, and share the engine's `triangle`.
const TEST_SHADERS: &[&str] = &[
    "write_value",
    "count_groups",
    "fill_texels",
    "copy_texels",
    "downsample",
    "sample_texture",
];

/// The Slang modules shaders import, as file names in `shaders/` without
/// the `.slang` extension. They're compiled as part of each shader that
/// imports them, so they only need to rebuild the shaders when they change.
/// `shaders/` is on the include path, so the test shaders find them too.
const MODULES: &[&str] = &["heap"];

/// Finds `slangc`, then compiles each shader in [`SHADERS`] and
/// [`TEST_SHADERS`].
///
/// # Panics
///
/// Panics, failing the build, if `slangc` can't be found or run, or if a
/// shader fails to compile.
fn main() {
    // `PATH` isn't tracked, though the lookup reads it: `cargo run` changes
    // it for its child on Windows, so tracking it would rebuild the engine
    // whenever `cargo xtask` and plain `cargo` alternate.
    println!("cargo::rerun-if-env-changed=SLANGC");
    println!("cargo::rerun-if-env-changed=VULKAN_SDK");
    let slangc = slangc::find(
        slangc::Platform::current(),
        &slangc::Environment::from_process(),
        |path| path.is_file(),
    )
    .unwrap_or_else(|err| panic!("{err}"))
    .path;
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    for name in MODULES {
        let source = PathBuf::from_iter(["shaders", &format!("{name}.slang")]);
        println!("cargo::rerun-if-changed={}", source.display());
    }
    let shaders = Path::new("shaders");
    let tests = shaders.join("tests");
    let sources = SHADERS
        .iter()
        .map(|name| (shaders, name))
        .chain(TEST_SHADERS.iter().map(|name| (tests.as_path(), name)));
    for (directory, name) in sources {
        let source = directory.join(format!("{name}.slang"));
        println!("cargo::rerun-if-changed={}", source.display());
        let output = out_dir.join(format!("{name}.spv"));

        // Every entry point in the file (marked with `[shader(...)]`) is
        // compiled into the one SPIR-V module, keeping its name. Shaders
        // read the descriptor heaps directly, through
        // `SPV_EXT_descriptor_heap`, so no mapping is given when their
        // shader objects are created (design 0002, "Binding model"). The
        // capability changes nothing in a shader that doesn't read them.
        let status = Command::new(&slangc)
            .arg(&source)
            .arg("-I")
            .arg(shaders)
            .args([
                "-target",
                "spirv",
                "-capability",
                "spvDescriptorHeapEXT",
                "-o",
            ])
            .arg(&output)
            .status();
        match status {
            Ok(status) if status.success() => {}
            Ok(status) => panic!("slangc failed to compile {} ({status})", source.display()),
            Err(err) => panic!(
                "failed to run {} ({err}). Install the Slang shader compiler from \
                 the Vulkan SDK, or set SLANGC to its path. See docs/testing.md.",
                slangc.display(),
            ),
        }
    }
}
