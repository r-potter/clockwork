//! Finds the Slang shader compiler, `slangc` (R18 in design 0001).
//!
//! Shared by the engine's `build.rs`, which compiles the shaders with it, and
//! by `xtask`, whose prerequisites check reports it and whose unit tests cover
//! this module (build scripts can't have tests of their own). Each includes
//! this file as a module with `#[path]`.
//!
//! The lookup is a pure function of the platform, the environment and a file
//! probe, so every platform's behaviour is tested on any machine (ADR 0003).
//! The order is:
//! 1. the `SLANGC` environment variable, taken as given
//! 2. the Vulkan SDK's binary directory, if `VULKAN_SDK` is set:
//!    `$VULKAN_SDK/bin/slangc` on Linux, `%VULKAN_SDK%\Bin\slangc.exe` on
//!    Windows
//! 3. each directory on `PATH`, in order

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};

/// The operating systems the lookup knows about (ADR 0003).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Linux: `slangc` in `$VULKAN_SDK/bin`, and `:` between `PATH` entries.
    Linux,
    /// Windows: `slangc.exe` in `%VULKAN_SDK%\Bin`, and `;` between `PATH`
    /// entries.
    Windows,
}

impl Platform {
    /// The platform this code was compiled for. Build scripts run on the host,
    /// which is the platform the engine is built on.
    pub fn current() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else {
            Self::Linux
        }
    }

    /// The file name of the `slangc` executable.
    fn executable(self) -> &'static str {
        match self {
            Self::Linux => "slangc",
            Self::Windows => "slangc.exe",
        }
    }

    /// The name of the Vulkan SDK's directory of executables, relative to
    /// `VULKAN_SDK`.
    fn sdk_bin_dir(self) -> &'static str {
        match self {
            Self::Linux => "bin",
            Self::Windows => "Bin",
        }
    }

    /// The character separating directories in `PATH`.
    fn path_separator(self) -> char {
        match self {
            Self::Linux => ':',
            Self::Windows => ';',
        }
    }
}

/// The environment variables the lookup reads. Unset and empty variables are
/// both `None`.
#[derive(Debug, Clone, Default)]
pub struct Environment {
    /// `SLANGC`: the path to `slangc`, overriding the other places.
    pub slangc: Option<OsString>,
    /// `VULKAN_SDK`: the root of an installed Vulkan SDK.
    pub vulkan_sdk: Option<OsString>,
    /// `PATH`: the directories searched for executables.
    pub path: Option<OsString>,
}

impl Environment {
    /// Reads the variables from this process's environment.
    pub fn from_process() -> Self {
        /// Reads `name`, treating an empty value as unset.
        fn var(name: &str) -> Option<OsString> {
            std::env::var_os(name).filter(|value| !value.is_empty())
        }
        Self {
            slangc: var("SLANGC"),
            vulkan_sdk: var("VULKAN_SDK"),
            path: var("PATH"),
        }
    }
}

/// Where `slangc` was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The `SLANGC` environment variable.
    SlangcVariable,
    /// The Vulkan SDK's binary directory.
    VulkanSdk,
    /// A directory on `PATH`.
    Path,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SlangcVariable => "from SLANGC",
            Self::VulkanSdk => "from VULKAN_SDK",
            Self::Path => "from PATH",
        })
    }
}

/// A `slangc` executable that the lookup found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    /// The path to run. Taken from `SLANGC` as given, which may be a bare
    /// command name; otherwise a path to an existing file.
    pub path: PathBuf,
    /// Which of the three places it came from.
    pub source: Source,
}

/// The error when `slangc` is in none of the places the lookup searches. Its
/// message names all three, and how to install the compiler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotFound {
    /// The platform searched, which decides the paths the message names.
    pub platform: Platform,
}

impl fmt::Display for NotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sdk_path = match self.platform {
            Platform::Linux => "$VULKAN_SDK/bin/slangc",
            Platform::Windows => "%VULKAN_SDK%\\Bin\\slangc.exe",
        };
        write!(
            f,
            "could not find the Slang shader compiler: SLANGC is not set, \
             {sdk_path} does not exist (or VULKAN_SDK is not set), and {} is not \
             on PATH. Install the Vulkan SDK and set up its environment, or set \
             SLANGC to the compiler's path. See docs/testing.md.",
            self.platform.executable(),
        )
    }
}

impl std::error::Error for NotFound {}

/// Finds `slangc` on `platform`, given the environment `env`, in the order
/// described in the module documentation. `is_file` reports whether a path is
/// an existing file; pass `|path| path.is_file()` outside tests.
///
/// `PATH` is split on the platform's separator after converting it to UTF-8
/// lossily, so a directory whose name isn't valid UTF-8 is never matched.
///
/// # Errors
///
/// Returns [`NotFound`] if `SLANGC` is unset and neither the SDK nor any
/// directory on `PATH` has the executable.
pub fn find(
    platform: Platform,
    env: &Environment,
    is_file: impl Fn(&Path) -> bool,
) -> Result<Found, NotFound> {
    if let Some(slangc) = &env.slangc {
        return Ok(Found {
            path: PathBuf::from(slangc),
            source: Source::SlangcVariable,
        });
    }

    if let Some(sdk) = &env.vulkan_sdk {
        let candidate = Path::new(sdk)
            .join(platform.sdk_bin_dir())
            .join(platform.executable());
        if is_file(&candidate) {
            return Ok(Found {
                path: candidate,
                source: Source::VulkanSdk,
            });
        }
    }

    if let Some(path) = &env.path {
        let path = path.to_string_lossy();
        let found = path
            .split(platform.path_separator())
            // An empty entry means the current directory on Linux. Building
            // shouldn't depend on where it was started, so it's skipped.
            .filter(|dir| !dir.is_empty())
            .map(|dir| Path::new(dir).join(platform.executable()))
            .find(|candidate| is_file(candidate));
        if let Some(candidate) = found {
            return Ok(Found {
                path: candidate,
                source: Source::Path,
            });
        }
    }

    Err(NotFound { platform })
}

#[cfg(test)]
mod tests {
    //! Tests of the lookup order on both platforms (R18). They use a fake file
    //! probe, so they pass on any machine, whatever it has installed.

    use super::*;

    /// A file probe that reports only `existing` as files.
    fn files(existing: &[PathBuf]) -> impl Fn(&Path) -> bool + '_ {
        |path| existing.iter().any(|file| file == path)
    }

    /// An environment with the given variables set.
    fn env(slangc: Option<&str>, vulkan_sdk: Option<&str>, path: Option<&str>) -> Environment {
        Environment {
            slangc: slangc.map(OsString::from),
            vulkan_sdk: vulkan_sdk.map(OsString::from),
            path: path.map(OsString::from),
        }
    }

    /// The SDK's `slangc` on Linux.
    fn linux_sdk_slangc() -> PathBuf {
        Path::new("/sdk").join("bin").join("slangc")
    }

    /// The SDK's `slangc` on Windows.
    fn windows_sdk_slangc() -> PathBuf {
        Path::new(r"C:\VulkanSDK\1.4.341.1")
            .join("Bin")
            .join("slangc.exe")
    }

    /// `SLANGC` wins over the SDK and `PATH`, and is used as given, even if
    /// no such file exists (it may be a command name).
    #[test]
    fn slangc_variable_comes_first() {
        for platform in [Platform::Linux, Platform::Windows] {
            let found = find(
                platform,
                &env(Some("my-slangc"), Some("/sdk"), Some("/usr/bin")),
                |_| true,
            );
            assert_eq!(
                found,
                Ok(Found {
                    path: PathBuf::from("my-slangc"),
                    source: Source::SlangcVariable,
                })
            );
        }
    }

    /// Linux: `$VULKAN_SDK/bin/slangc`, ahead of `PATH`.
    #[test]
    fn linux_sdk_comes_before_path() {
        let existing = [linux_sdk_slangc(), Path::new("/usr/bin").join("slangc")];
        let found = find(
            Platform::Linux,
            &env(None, Some("/sdk"), Some("/usr/bin")),
            files(&existing),
        );
        assert_eq!(
            found,
            Ok(Found {
                path: linux_sdk_slangc(),
                source: Source::VulkanSdk,
            })
        );
    }

    /// Windows: `%VULKAN_SDK%\Bin\slangc.exe`, ahead of `PATH`.
    #[test]
    fn windows_sdk_comes_before_path() {
        let existing = [
            windows_sdk_slangc(),
            Path::new(r"C:\tools").join("slangc.exe"),
        ];
        let found = find(
            Platform::Windows,
            &env(None, Some(r"C:\VulkanSDK\1.4.341.1"), Some(r"C:\tools")),
            files(&existing),
        );
        assert_eq!(
            found,
            Ok(Found {
                path: windows_sdk_slangc(),
                source: Source::VulkanSdk,
            })
        );
    }

    /// Linux: an SDK without `slangc` falls through to the first directory
    /// on `PATH` that has it, splitting on `:` and skipping empty entries.
    #[test]
    fn linux_path_is_searched_in_order() {
        let existing = [
            Path::new("/second").join("slangc"),
            Path::new("/third").join("slangc"),
        ];
        let found = find(
            Platform::Linux,
            &env(None, Some("/sdk"), Some("::/first:/second:/third")),
            files(&existing),
        );
        assert_eq!(
            found,
            Ok(Found {
                path: Path::new("/second").join("slangc"),
                source: Source::Path,
            })
        );
    }

    /// Windows: `PATH` is split on `;`, and the executable is `slangc.exe`.
    #[test]
    fn windows_path_is_searched_in_order() {
        let existing = [Path::new(r"C:\second").join("slangc.exe")];
        let found = find(
            Platform::Windows,
            &env(None, None, Some(r"C:\first;;C:\second")),
            files(&existing),
        );
        assert_eq!(
            found,
            Ok(Found {
                path: Path::new(r"C:\second").join("slangc.exe"),
                source: Source::Path,
            })
        );
    }

    /// Windows never looks for an executable without `.exe`.
    #[test]
    fn windows_ignores_files_without_exe() {
        let existing = [Path::new(r"C:\tools").join("slangc")];
        let found = find(
            Platform::Windows,
            &env(None, None, Some(r"C:\tools")),
            files(&existing),
        );
        assert_eq!(
            found,
            Err(NotFound {
                platform: Platform::Windows
            })
        );
    }

    /// With nothing found, the error names all three places, with the
    /// platform's SDK path.
    #[test]
    fn not_found_names_every_place() {
        for (platform, sdk_path, executable) in [
            (Platform::Linux, "$VULKAN_SDK/bin/slangc", "slangc"),
            (
                Platform::Windows,
                r"%VULKAN_SDK%\Bin\slangc.exe",
                "slangc.exe",
            ),
        ] {
            let err = find(platform, &env(None, Some("sdk"), Some("dir")), |_| false).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("SLANGC"), "{message}");
            assert!(message.contains(sdk_path), "{message}");
            assert!(
                message.contains(&format!("{executable} is not on PATH")),
                "{message}"
            );
            assert!(message.contains("docs/testing.md"), "{message}");
        }
    }

    /// The platform `build.rs` passes is the one it's running on.
    #[test]
    fn current_platform_matches_the_host() {
        assert_eq!(Platform::current() == Platform::Windows, cfg!(windows));
    }
}
