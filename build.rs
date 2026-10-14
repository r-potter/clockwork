//! Compiles the Slang shaders in `shaders/` to SPIR-V with `slangc`, writing
//! `<name>.spv` files to `OUT_DIR` for the Vulkan backend to embed.
//!
//! `slangc` is looked up on `PATH`, unless the `SLANGC` environment variable
//! gives its path.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// The shaders to compile, as file names in `shaders/` without the `.slang`
/// extension.
const SHADERS: &[&str] = &["triangle"];

fn main() {
    println!("cargo::rerun-if-env-changed=SLANGC");
    let slangc = env::var_os("SLANGC").unwrap_or_else(|| "slangc".into());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    for name in SHADERS {
        let source = format!("shaders/{name}.slang");
        println!("cargo::rerun-if-changed={source}");
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
            Ok(status) => panic!("slangc failed to compile {source} ({status})"),
            Err(err) => panic!(
                "failed to run {} ({err}). Install the Slang shader compiler (for \
                 example with `sudo apt install slang-compiler`, or from the Vulkan \
                 SDK), or set SLANGC to its path.",
                slangc.to_string_lossy(),
            ),
        }
    }
}
