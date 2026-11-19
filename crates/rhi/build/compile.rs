//! Compiles Slang shaders to SPIR-V with `slangc`, as every build script
//! that compiles shaders does.
//!
//! Shared by `clockwork-rhi`'s `build.rs` and the application's, each of
//! which includes this file as a module with `#[path]`, beside
//! [`slangc`](super::slangc), so both compile with the same options.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Finds `slangc` with [`slangc::find`](super::slangc::find), and has Cargo
/// rerun the build script when the variables that choose it change.
///
/// # Panics
///
/// Panics, failing the build, if `slangc` can't be found.
pub(crate) fn find_slangc() -> PathBuf {
    // `PATH` isn't tracked, though the lookup reads it: `cargo run` changes
    // it for its child on Windows, so tracking it would rebuild the engine
    // whenever `cargo xtask` and plain `cargo` alternate.
    println!("cargo::rerun-if-env-changed=SLANGC");
    println!("cargo::rerun-if-env-changed=VULKAN_SDK");
    super::slangc::find(
        super::slangc::Platform::current(),
        &super::slangc::Environment::from_process(),
        |path| path.is_file(),
    )
    .unwrap_or_else(|err| panic!("{err}"))
    .path
}

/// Compiles `source` to SPIR-V at `output` with `slangc`, finding the
/// modules it imports in `include`, and has Cargo rerun the build script
/// when `source` changes.
///
/// # Panics
///
/// Panics, failing the build, if `slangc` can't be run, or the shader
/// fails to compile.
pub(crate) fn compile(slangc: &Path, source: &Path, include: &Path, output: &Path) {
    println!("cargo::rerun-if-changed={}", source.display());
    // Every entry point in the file (marked with `[shader(...)]`) is
    // compiled into the one SPIR-V module, keeping its name. Shaders read
    // the descriptor heaps directly, through `SPV_EXT_descriptor_heap`, so
    // no mapping is given when their shader objects are created (design
    // 0002, "Binding model"). The capability changes nothing in a shader
    // that doesn't read them.
    let status = Command::new(slangc)
        .arg(source)
        .arg("-I")
        .arg(include)
        .args([
            "-target",
            "spirv",
            "-capability",
            "spvDescriptorHeapEXT",
            "-o",
        ])
        .arg(output)
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
