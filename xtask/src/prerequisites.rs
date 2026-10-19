//! The prerequisites check that starts every `cargo xtask check`.
//!
//! It prints what the build will use, so that a failure on one machine can be
//! compared with another (see "Risks" in design 0001):
//! - the `slangc` the engine's `build.rs` will find, where it came from, and
//!   its version
//! - the Vulkan SDK's version, from `VULKAN_SDK`
//!
//! A missing `slangc` is an error, since nothing builds without it. A missing
//! Vulkan SDK is only reported: `slangc` can come from elsewhere, and the
//! validation layer it provides is only needed by GPU tests.

use std::path::Path;

use anyhow::{Context, Result};
use xshell::Shell;

use crate::slangc;

/// Reports the prerequisites, failing if `slangc` can't be found or run.
///
/// # Errors
///
/// Returns an error, pointing to `docs/testing.md`, if `slangc` isn't found
/// or doesn't run.
pub fn check(sh: &Shell) -> Result<()> {
    eprintln!("==> prerequisites");
    let env = slangc::Environment::from_process();

    let found = slangc::find(slangc::Platform::current(), &env, |path| path.is_file())?;
    let output = sh
        .cmd(&found.path)
        .arg("-version")
        .quiet()
        .ignore_status()
        .output()
        .with_context(|| {
            format!(
                "failed to run {} to read its version. See docs/testing.md.",
                found.path.display()
            )
        })?;
    // `slangc` has printed its version on stderr in some releases, and on
    // stdout in others.
    let version = [&output.stdout, &output.stderr]
        .into_iter()
        .map(|bytes| String::from_utf8_lossy(bytes).trim().to_owned())
        .find(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    eprintln!(
        "slangc: {} ({}), version {version}",
        found.path.display(),
        found.source,
    );

    match &env.vulkan_sdk {
        Some(sdk) => {
            let sdk = Path::new(sdk);
            let version = sdk_version(sdk).unwrap_or("unknown");
            eprintln!("Vulkan SDK: {} (version {version})", sdk.display());
        }
        None => eprintln!(
            "Vulkan SDK: VULKAN_SDK is not set. GPU tests need the SDK's \
             validation layer. See docs/testing.md."
        ),
    }
    Ok(())
}

/// The Vulkan SDK's version, from the name of its directory, given the value
/// of `VULKAN_SDK`.
///
/// The SDK's installers put it in a directory named after its version, such
/// as `1.4.341.1`. On Windows, `VULKAN_SDK` is that directory. On Linux, it's
/// the `x86_64` directory inside it. So this looks at `sdk` and its parent,
/// and returns the first whose name is a version.
pub fn sdk_version(sdk: &Path) -> Option<&str> {
    [Some(sdk), sdk.parent()]
        .into_iter()
        .flatten()
        .filter_map(|dir| dir.file_name()?.to_str())
        .find(|name| is_version(name))
}

/// Returns whether `name` looks like a version: at least three
/// dot-separated, non-empty runs of digits.
fn is_version(name: &str) -> bool {
    let parts: Vec<_> = name.split('.').collect();
    parts.len() >= 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    //! Tests of reading the Vulkan SDK's version.

    use std::path::PathBuf;

    use super::*;

    /// Windows: `VULKAN_SDK` is the versioned directory itself.
    #[test]
    fn sdk_version_from_the_sdk_directory() {
        let sdk = PathBuf::from_iter(["VulkanSDK", "1.4.341.1"]);
        assert_eq!(sdk_version(&sdk), Some("1.4.341.1"));
    }

    /// Linux: `VULKAN_SDK` is the architecture directory inside it.
    #[test]
    fn sdk_version_from_the_parent_directory() {
        let sdk = PathBuf::from_iter(["vulkan", "1.4.341.1", "x86_64"]);
        assert_eq!(sdk_version(&sdk), Some("1.4.341.1"));
    }

    /// An SDK installed somewhere unversioned has no version to report.
    #[test]
    fn sdk_version_unknown() {
        let sdk = PathBuf::from_iter(["opt", "vulkan-sdk"]);
        assert_eq!(sdk_version(&sdk), None);
    }

    /// Names with too few parts, empty parts, or other characters aren't
    /// versions.
    #[test]
    fn version_shapes() {
        assert!(is_version("1.4.341"));
        assert!(is_version("1.4.341.1"));
        assert!(!is_version("1.4"));
        assert!(!is_version("1..341"));
        assert!(!is_version("1.4.341-beta"));
        assert!(!is_version("x86_64"));
    }
}
