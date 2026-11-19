//! Compiles the application's Slang shaders in `assets/shaders/` to SPIR-V
//! with `slangc`, writing `<name>.spv` files to `OUT_DIR` for the
//! application to embed.
//!
//! This is a stop gap: a later design loads shaders with the rest of the
//! assets, at run time. Until then, the application compiles its shaders
//! as `clockwork-rhi` compiles its test shaders, with the same lookup of
//! `slangc` and the same options, and with `clockwork-rhi`'s `shaders/` on
//! the include path, so a shader can `import heap;` to reach the
//! descriptor heaps (design 0002, "Binding model").

#[path = "crates/rhi/build/compile.rs"]
mod compile;
#[path = "crates/rhi/build/slangc.rs"]
mod slangc;

use std::env;
use std::path::{Path, PathBuf};

/// The application's shaders, as file names in `assets/shaders/` without
/// the `.slang` extension.
const SHADERS: &[&str] = &["triangle"];

/// The Slang modules from `clockwork-rhi` that shaders import, as file names
/// in its `shaders/` without the `.slang` extension. They're compiled as
/// part of each shader that imports them, so they only need to rebuild the
/// shaders when they change.
const RHI_MODULES: &[&str] = &["heap"];

/// Finds `slangc`, then compiles each shader in [`SHADERS`].
///
/// # Panics
///
/// Panics, failing the build, if `slangc` can't be found or run, or if a
/// shader fails to compile.
fn main() {
    let slangc = compile::find_slangc();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));

    let rhi_shaders = Path::new("crates/rhi/shaders");
    for name in RHI_MODULES {
        let source = rhi_shaders.join(format!("{name}.slang"));
        println!("cargo::rerun-if-changed={}", source.display());
    }
    let shaders = Path::new("assets/shaders");
    for name in SHADERS {
        let source = shaders.join(format!("{name}.slang"));
        let output = out_dir.join(format!("{name}.spv"));
        compile::compile(&slangc, &source, rhi_shaders, &output);
    }
}
