//! The prerequisites check that starts every `cargo xtask check`.
//!
//! It prints what the build will use, so that a failure on one machine can be
//! compared with another (see "Risks" in design 0001):
//! - the `slangc` the engine's `build.rs` will find, where it came from, and
//!   its version
//! - the Vulkan SDK's version, from `VULKAN_SDK`
//! - with `--lavapipe`, the manifest of the lavapipe build used
//! - with `--gpu`, the layers and devices the Vulkan loader finds
//!
//! A missing `slangc` is an error, since nothing builds without it. A missing
//! Vulkan SDK is only reported: `slangc` can come from elsewhere, and the
//! validation layer it provides is only needed by GPU tests. With `--gpu`, a
//! missing validation layer, or no devices, is an error, since every GPU test
//! would fail.

use std::ffi::CStr;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ash::{Entry, vk};
use xshell::Shell;

use crate::slangc;

/// The Khronos validation layer, which GPU tests require.
const VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";

/// Reports the prerequisites, failing if `slangc` can't be found or run, or,
/// if `gpu` is set, if the validation layer or a device is missing. `driver`
/// is the manifest of the only Vulkan driver the loader is to use, if any,
/// as with `--lavapipe`.
///
/// # Errors
///
/// Returns an error, pointing to `docs/testing.md`, if `slangc` isn't found
/// or doesn't run, or if `gpu` is set and the GPU tests can't run.
pub fn check(sh: &Shell, gpu: bool, driver: Option<&Path>) -> Result<()> {
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

    if let Some(driver) = driver {
        eprintln!("Vulkan driver: {} alone", driver.display());
    }
    if gpu {
        check_vulkan()?;
    }
    Ok(())
}

/// Prints the layers and devices the Vulkan loader finds, and fails if the
/// GPU tests can't run with them.
///
/// # Errors
///
/// Returns an error if the Vulkan loader can't be loaded or queried, or if
/// [`gpu_problem`] finds a problem.
fn check_vulkan() -> Result<()> {
    // SAFETY: Loading the Vulkan loader runs its library initialisation code,
    // which we trust to be well behaved.
    let entry = unsafe { Entry::load() }
        .context("failed to load the Vulkan loader, which GPU tests need. See docs/testing.md.")?;

    // SAFETY: `entry` holds a valid, loaded Vulkan loader.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }
        .context("failed to enumerate Vulkan layers")?;
    let layer_names: Vec<String> = layers
        .iter()
        .map(|layer| {
            layer
                .layer_name_as_c_str()
                .map_or("<invalid name>".into(), CStr::to_string_lossy)
                .into_owned()
        })
        .collect();
    let layer_descriptions: Vec<String> = layers
        .iter()
        .zip(&layer_names)
        .map(|(layer, name)| format!("{name} {}", format_version(layer.spec_version)))
        .collect();
    eprintln!("Vulkan layers: {}", list(&layer_descriptions));

    let devices = devices(&entry)?;
    eprintln!("Vulkan devices: {}", list(&devices));

    if let Some(problem) = gpu_problem(&layer_names, devices.len()) {
        bail!("{problem}. See docs/testing.md.");
    }
    Ok(())
}

/// Returns why GPU tests can't run with these layers and this many devices,
/// or `None` if they can. Pure.
fn gpu_problem(layers: &[String], device_count: usize) -> Option<String> {
    if !layers.iter().any(|layer| layer == VALIDATION_LAYER) {
        Some(format!(
            "GPU tests require validation, but the Vulkan loader can't find {VALIDATION_LAYER}"
        ))
    } else if device_count == 0 {
        Some("GPU tests need a Vulkan device, but the Vulkan loader found none".to_owned())
    } else {
        None
    }
}

/// Creates a minimal instance, with no layers or extensions, and describes
/// each physical device it finds by its name, type and Vulkan version.
///
/// # Errors
///
/// Returns an error if instance creation or device enumeration fails.
fn devices(entry: &Entry) -> Result<Vec<String>> {
    let application_info = vk::ApplicationInfo::default()
        .application_name(c"clockwork xtask")
        .api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&application_info);
    // SAFETY: `entry` holds a valid, loaded Vulkan loader, and `create_info`
    // and the structure it points to are valid for the call.
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .context("failed to create a Vulkan instance to list the devices")?;

    // SAFETY: `instance` is a valid instance.
    let handles = unsafe { instance.enumerate_physical_devices() };
    let devices = handles.map(|handles| {
        handles
            .into_iter()
            .map(|handle| {
                // SAFETY: `handle` was enumerated from `instance`, which is
                // still alive.
                let properties = unsafe { instance.get_physical_device_properties(handle) };
                let name = properties
                    .device_name_as_c_str()
                    .map_or("<invalid name>".into(), CStr::to_string_lossy);
                format!(
                    "{name} ({:?}, Vulkan {})",
                    properties.device_type,
                    format_version(properties.api_version),
                )
            })
            .collect()
    });
    // SAFETY: the instance was created above and is destroyed once, here.
    // Nothing created from it outlives it: physical devices are only
    // enumerated, not created.
    unsafe { instance.destroy_instance(None) };
    devices.context("failed to enumerate Vulkan devices")
}

/// Formats a Vulkan version number as `major.minor.patch`.
fn format_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version),
    )
}

/// Joins `items` with commas, or says there are none.
fn list(items: &[String]) -> String {
    if items.is_empty() {
        "none".to_owned()
    } else {
        items.join(", ")
    }
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
    //! Tests of reading the Vulkan SDK's version, and of deciding whether GPU
    //! tests can run.

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

    /// Layer names for the GPU checks.
    fn layers(names: &[&str]) -> Vec<String> {
        names.iter().map(|&name| name.to_owned()).collect()
    }

    /// With the validation layer and a device, GPU tests can run.
    #[test]
    fn gpu_checks_pass_with_the_layer_and_a_device() {
        let layers = layers(&["VK_LAYER_OBS_HOOK", "VK_LAYER_KHRONOS_validation"]);
        assert_eq!(gpu_problem(&layers, 1), None);
    }

    /// Without the validation layer, GPU tests can't run, since they always
    /// require validation.
    #[test]
    fn gpu_checks_need_the_validation_layer() {
        let problem = gpu_problem(&layers(&["VK_LAYER_OBS_HOOK"]), 1).unwrap();
        assert!(problem.contains(VALIDATION_LAYER), "{problem}");
    }

    /// Without a device, GPU tests can't run.
    #[test]
    fn gpu_checks_need_a_device() {
        let problem = gpu_problem(&layers(&[VALIDATION_LAYER]), 0).unwrap();
        assert!(problem.contains("found none"), "{problem}");
    }
}
