//! Compiles the Slang shaders in `shaders/` to SPIR-V with `slangc`, writing
//! `<name>.spv` files to `OUT_DIR` for the Vulkan backend to embed.
//!
//! `slangc` is found by [`slangc::find`]: from the `SLANGC` environment
//! variable, then the Vulkan SDK, then `PATH`.

#[path = "build/slangc.rs"]
mod slangc;

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// The shaders to compile, as file names in `shaders/` without the `.slang`
/// extension.
const SHADERS: &[&str] = &["triangle"];

/// Finds `slangc`, then compiles each shader in [`SHADERS`].
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

    for name in SHADERS {
        let source = PathBuf::from_iter(["shaders", &format!("{name}.slang")]);
        println!("cargo::rerun-if-changed={}", source.display());
        let output = out_dir.join(format!("{name}.spv"));

        // Every entry point in the file (marked with `[shader(...)]`) is
        // compiled into the one SPIR-V module, keeping its name.
        let status = Command::new(&slangc)
            .arg(&source)
            .args(["-target", "spirv", "-o"])
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
