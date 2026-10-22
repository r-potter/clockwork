//! `cargo xtask lavapipe`: builds lavapipe, Mesa's CPU Vulkan driver, from a
//! pinned commit (R20 in design 0001).
//!
//! CI has no GPU, so its GPU tests run on lavapipe. No released Mesa has
//! every extension ADR 0002 requires on lavapipe yet, so it's built from
//! [`PIN`], a commit on Mesa's `main` branch. The same build is used locally,
//! so a lavapipe failure in CI can be reproduced on any Linux machine.
//!
//! [`build`]:
//! 1. checks that the build's prerequisites are installed, naming every one
//!    that's missing
//! 2. fetches the pinned commit, and nothing else, into
//!    `target/lavapipe/<commit>/source`
//! 3. configures it with Meson, using [`MESON_OPTIONS`], in
//!    `target/lavapipe/<commit>/build`
//! 4. builds and installs it in `target/lavapipe/<commit>/install`, then
//!    removes the source and build directories, which are large and not
//!    needed again
//!
//! A stamp file in the installation records the pin and the options it was
//! built with, so running it again does nothing until either changes. Builds
//! for different pins are in different directories, so they coexist, and
//! `cargo clean` removes them all.
//!
//! Linux only: Windows GPU tests run on the Windows machine's GPU instead
//! (see "Non-goals" in design 0001).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use xshell::Shell;

use crate::slangc::Platform;

/// A Mesa commit to build lavapipe from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pin {
    /// The commit's full hash.
    pub commit: &'static str,
    /// The commit's date, for reading the pin's age at a glance.
    pub date: &'static str,
}

/// The Mesa commit lavapipe is built from.
///
/// It's a `main` commit, because no Mesa release has `VK_EXT_descriptor_heap`
/// and `VK_KHR_device_address_commands` on lavapipe yet: 26.2.5, the latest,
/// has neither. Lavapipe's entries in Mesa's `docs/features.txt` were updated
/// to list both on 2026-10-14. This was `main`'s head on 2026-10-19, so it
/// has passed Mesa's own CI, which every merge to `main` must.
///
/// When a release has both, pin its tag's commit instead. Changing the pin is
/// a reviewed change like any other: see "Lavapipe" in `docs/testing.md`.
pub(crate) const PIN: Pin = Pin {
    commit: "44fd0fde8187b5bb1e919625f1ba24cb36c18cbc",
    date: "2026-10-19",
};

/// The capabilities lavapipe at [`PIN`] has, as `CLOCKWORK_REQUIRED_CAPABILITIES`
/// names them (see `Capability` in `crates/rhi/src/vulkan/test_support.rs`).
/// With `--lavapipe`, a GPU test needing one of these fails rather than
/// skipping.
///
/// Every capability is either here or in the tests' list of those lavapipe
/// lacks, which a test checks, so that one added to `Capability` can't
/// quietly skip in CI. The engine's `every_required_capability_is_present`
/// GPU test fails if lavapipe lacks one listed here.
pub(crate) const CAPABILITIES: &[&str] = &["swapchain-maintenance", "bc-compression"];

/// Mesa's repository.
const REPOSITORY: &str = "https://gitlab.freedesktop.org/mesa/mesa.git";

/// The options passed to `meson setup`: lavapipe, with the X11 and Wayland
/// presentation the smoke runs need, and nothing else Mesa can build.
const MESON_OPTIONS: &[&str] = &[
    "--buildtype=release",
    // Installed libraries go in `lib`, whatever the distribution's
    // convention, so the installation looks the same everywhere.
    "--libdir=lib",
    "-Dvulkan-drivers=swrast",
    "-Dgallium-drivers=llvmpipe",
    "-Dplatforms=x11,wayland",
    // Lavapipe compiles shaders with LLVM, so a missing LLVM must fail the
    // configuration, not quietly change the build.
    "-Dllvm=enabled",
    // No OpenGL, or the window system glue that comes with it.
    "-Dopengl=false",
    "-Dgles1=disabled",
    "-Dgles2=disabled",
    "-Degl=disabled",
    "-Dglx=disabled",
    "-Dgbm=disabled",
    "-Dvideo-codecs=[]",
];

/// The programs the build runs, with a description of each for the error
/// when one is missing.
const PROGRAMS: &[(&str, &str)] = &[
    ("git", "Git, to fetch Mesa"),
    ("meson", "Meson"),
    ("ninja", "Ninja"),
    ("pkg-config", "pkg-config"),
    ("cc", "a C compiler"),
    ("c++", "a C++ compiler"),
    ("llvm-config", "LLVM, with its development files"),
    ("bison", "Bison"),
    ("flex", "Flex"),
    ("python3", "Python 3"),
];

/// The Python modules Mesa needs in the `python3` on `PATH`, which runs its
/// code generators.
const PYTHON_MODULES: &[&str] = &["mako", "packaging", "yaml"];

/// The libraries the build links, as `pkg-config` names them: those for
/// presenting to X11 and Wayland, and those Mesa needs whatever the drivers.
const LIBRARIES: &[&str] = &[
    "expat",
    "zlib",
    "libdrm",
    "x11",
    "x11-xcb",
    "xcb",
    "xcb-dri3",
    "xcb-present",
    "xcb-randr",
    "xcb-shm",
    "xcb-xfixes",
    "xrandr",
    "xshmfence",
    "wayland-client",
    "wayland-protocols",
    "wayland-scanner",
];

/// Where one pin's build lives, under `target/lavapipe/<commit>/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    /// The pin's directory.
    root: PathBuf,
}

impl Layout {
    /// The layout for `pin`, in the Cargo target directory `target_dir`.
    pub(crate) fn new(target_dir: &Path, pin: Pin) -> Self {
        Self {
            root: target_dir.join("lavapipe").join(pin.commit),
        }
    }

    /// Mesa's source, checked out at the pin.
    fn source(&self) -> PathBuf {
        self.root.join("source")
    }

    /// Meson's build directory.
    fn build(&self) -> PathBuf {
        self.root.join("build")
    }

    /// Where the build is installed.
    fn install(&self) -> PathBuf {
        self.root.join("install")
    }

    /// The stamp file, written once the installation is complete.
    fn stamp(&self) -> PathBuf {
        self.install().join("clockwork-build.txt")
    }

    /// The installed driver's ICD manifest, which points the Vulkan loader at
    /// the driver, for the CPU architecture `arch` (as in
    /// [`std::env::consts::ARCH`], which Mesa uses in the file name too).
    pub(crate) fn manifest(&self, arch: &str) -> PathBuf {
        self.install()
            .join("share")
            .join("vulkan")
            .join("icd.d")
            .join(format!("lvp_icd.{arch}.json"))
    }
}

/// What the stamp file records: the pin and the options, one per line. A
/// build whose stamp doesn't match this is out of date.
fn stamp_contents(pin: Pin) -> String {
    let mut contents = format!("{}\n", pin.commit);
    for option in MESON_OPTIONS {
        contents.push_str(option);
        contents.push('\n');
    }
    contents
}

/// The arguments to `meson setup`, configuring the source in `layout` to
/// build in its build directory, and install in its installation directory.
fn meson_setup_args(layout: &Layout) -> Vec<String> {
    let mut args = vec![
        "setup".to_owned(),
        layout.build().display().to_string(),
        layout.source().display().to_string(),
        format!("--prefix={}", layout.install().display()),
    ];
    args.extend(MESON_OPTIONS.iter().map(|&option| option.to_owned()));
    args
}

/// Names the build's missing prerequisites, given probes for whether a
/// program is on `PATH`, a library is known to `pkg-config`, and a Python
/// module can be imported. Pure.
fn missing_prerequisites(
    has_program: impl Fn(&str) -> bool,
    has_library: impl Fn(&str) -> bool,
    has_python_module: impl Fn(&str) -> bool,
) -> Vec<String> {
    let programs = PROGRAMS
        .iter()
        .filter(|(program, _)| !has_program(program))
        .map(|(program, description)| format!("{description} (`{program}`)"));
    // Without `pkg-config`, every library would look missing: it's already
    // reported as a missing program, which is enough.
    let libraries = LIBRARIES
        .iter()
        .filter(|_| has_program("pkg-config"))
        .filter(|library| !has_library(library))
        .map(|library| format!("the `{library}` library, with its development files"));
    let python_modules = PYTHON_MODULES
        .iter()
        .filter(|_| has_program("python3"))
        .filter(|module| !has_python_module(module))
        .map(|module| format!("the `{module}` Python module"));
    programs.chain(libraries).chain(python_modules).collect()
}

/// Whether `program` is in a directory on `PATH`.
fn on_path(program: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join(program).is_file()))
}

/// Checks that everything the build needs is installed.
///
/// # Errors
///
/// Returns an error naming every missing prerequisite, pointing to
/// `docs/testing.md`, which lists the packages that provide them.
fn check_prerequisites(sh: &Shell) -> Result<()> {
    let missing = missing_prerequisites(
        on_path,
        |library| {
            sh.cmd("pkg-config")
                .args(["--exists", library])
                .quiet()
                .run()
                .is_ok()
        },
        |module| {
            sh.cmd("python3")
                .args(["-c", &format!("import {module}")])
                .quiet()
                .ignore_stderr()
                .run()
                .is_ok()
        },
    );
    if !missing.is_empty() {
        bail!(
            "building lavapipe needs:\n  - {}\nSee \"Lavapipe\" in docs/testing.md for the \
             packages that provide them.",
            missing.join("\n  - ")
        );
    }
    Ok(())
}

/// Whether lavapipe at [`PIN`] is completely installed in `layout`, with the
/// current options.
fn is_built(layout: &Layout) -> bool {
    fs::read_to_string(layout.stamp()).is_ok_and(|contents| contents == stamp_contents(PIN))
}

/// The ICD manifest of lavapipe at [`PIN`], built by [`build`] in the Cargo
/// target directory `target_dir`, for `--lavapipe`.
///
/// # Errors
///
/// Returns an error, saying to run `cargo xtask lavapipe`, if this pin isn't
/// built with the current options.
pub(crate) fn manifest(target_dir: &Path) -> Result<PathBuf> {
    let layout = Layout::new(target_dir, PIN);
    if !is_built(&layout) {
        bail!(
            "lavapipe at Mesa {} isn't built. Run `cargo xtask lavapipe` first.",
            PIN.commit
        );
    }
    Ok(layout.manifest(std::env::consts::ARCH))
}

/// Builds lavapipe at [`PIN`] in the workspace at `root`, unless it's
/// already built, and prints the path to its ICD manifest.
///
/// # Errors
///
/// Returns an error if this isn't Linux, if a prerequisite is missing, or if
/// fetching, configuring, building or installing Mesa fails.
pub(crate) fn build(root: &Path) -> Result<()> {
    if Platform::current() != Platform::Linux {
        bail!("lavapipe is only built on Linux. See \"Lavapipe\" in docs/testing.md.");
    }
    let layout = Layout::new(&root.join("target"), PIN);
    let manifest = layout.manifest(std::env::consts::ARCH);
    let stamp = stamp_contents(PIN);
    if is_built(&layout) {
        eprintln!(
            "lavapipe at Mesa {} ({}) is already built: {}",
            PIN.commit,
            PIN.date,
            manifest.display()
        );
        return Ok(());
    }

    let sh = Shell::new()?;
    check_prerequisites(&sh)?;
    eprintln!(
        "==> building lavapipe at Mesa {} ({})",
        PIN.commit, PIN.date
    );
    // Start from nothing, so that a failed or outdated build can't leave
    // anything behind in this one.
    for dir in [layout.source(), layout.build(), layout.install()] {
        if dir.exists() {
            fs::remove_dir_all(&dir)
                .with_context(|| format!("failed to remove {}", dir.display()))?;
        }
    }

    // Fetch only the pinned commit, without its history.
    let source = layout.source();
    fs::create_dir_all(&source)
        .with_context(|| format!("failed to create {}", source.display()))?;
    sh.change_dir(&source);
    sh.cmd("git").args(["init", "--quiet"]).run()?;
    sh.cmd("git")
        .args(["fetch", "--depth", "1", REPOSITORY, PIN.commit])
        .run()
        .context("failed to fetch Mesa")?;
    sh.cmd("git")
        .args(["checkout", "--quiet", "--detach", "FETCH_HEAD"])
        .run()?;

    sh.change_dir(root);
    sh.cmd("meson")
        .args(meson_setup_args(&layout))
        .run()
        .context("failed to configure Mesa")?;
    let build = layout.build().display().to_string();
    sh.cmd("meson")
        .args(["install", "-C", &build])
        .run()
        .context("failed to build and install lavapipe")?;

    if !manifest.is_file() {
        bail!(
            "the build finished, but didn't install {}",
            manifest.display()
        );
    }
    fs::write(layout.stamp(), stamp)
        .with_context(|| format!("failed to write {}", layout.stamp().display()))?;
    for dir in [layout.source(), layout.build()] {
        fs::remove_dir_all(&dir).with_context(|| format!("failed to remove {}", dir.display()))?;
    }
    eprintln!("lavapipe is built: {}", manifest.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Tests of the build's configuration. Building needs Linux and the
    //! prerequisites, and takes minutes, so the build itself is covered by
    //! the `linux-lavapipe` CI job instead.

    use super::*;

    /// The pin is a full commit hash, not a branch, tag or abbreviation, so
    /// it can't change under the build.
    #[test]
    fn the_pin_is_a_full_commit_hash() {
        assert_eq!(PIN.commit.len(), 40);
        assert!(
            PIN.commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

    /// Only lavapipe is built, with LLVM and both Linux window systems, as
    /// "Lavapipe" in design 0001 describes.
    #[test]
    fn meson_builds_lavapipe_alone() {
        for option in [
            "--buildtype=release",
            "-Dvulkan-drivers=swrast",
            "-Dgallium-drivers=llvmpipe",
            "-Dplatforms=x11,wayland",
            "-Dllvm=enabled",
            "-Dopengl=false",
        ] {
            assert!(MESON_OPTIONS.contains(&option), "{option} is missing");
        }
    }

    /// Meson configures the pin's source, to build and install under the
    /// pin's own directory.
    #[test]
    fn meson_setup_uses_the_pins_directories() {
        let layout = Layout::new(Path::new("target"), PIN);
        let root = Path::new("target").join("lavapipe").join(PIN.commit);
        let args = meson_setup_args(&layout);
        assert_eq!(
            args[..4],
            [
                "setup".to_owned(),
                root.join("build").display().to_string(),
                root.join("source").display().to_string(),
                format!("--prefix={}", root.join("install").display()),
            ]
        );
        assert!(args[4..].iter().eq(MESON_OPTIONS), "{args:?}");
    }

    /// Builds for different pins are in different directories, so they
    /// coexist.
    #[test]
    fn pins_have_their_own_directories() {
        let other = Pin {
            commit: "0000000000000000000000000000000000000000",
            date: "2026-01-01",
        };
        assert_ne!(
            Layout::new(Path::new("target"), PIN),
            Layout::new(Path::new("target"), other)
        );
    }

    /// The manifest is where Mesa installs it, named for the architecture.
    #[test]
    fn the_manifest_is_in_the_installation() {
        let layout = Layout::new(Path::new("target"), PIN);
        assert_eq!(
            layout.manifest("x86_64"),
            Path::new("target")
                .join("lavapipe")
                .join(PIN.commit)
                .join("install")
                .join("share")
                .join("vulkan")
                .join("icd.d")
                .join("lvp_icd.x86_64.json")
        );
    }

    /// The stamp changes when the pin or an option does, so either makes
    /// `cargo xtask lavapipe` rebuild.
    #[test]
    fn the_stamp_records_the_pin_and_the_options() {
        let stamp = stamp_contents(PIN);
        assert!(stamp.starts_with(PIN.commit), "{stamp}");
        for option in MESON_OPTIONS {
            assert!(stamp.lines().any(|line| line == *option), "{stamp}");
        }
    }

    /// The capabilities lavapipe lacks, so tests needing them skip on it
    /// ("Lavapipe" in design 0001).
    const LACKS: &[&str] = &[
        // Lavapipe has a single queue family.
        "dedicated-compute-queue",
        "dedicated-transfer-queue",
        // Lavapipe doesn't support `VK_EXT_present_timing`.
        "present-timing",
    ];

    /// The capability names in the engine's `Capability::name`, read from its
    /// source, since `xtask` can't depend on the engine.
    fn engine_capabilities() -> Vec<String> {
        let path = PathBuf::from_iter([
            crate::workspace_root(),
            PathBuf::from_iter(["crates", "rhi", "src", "vulkan", "test_support.rs"]),
        ]);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        source
            .lines()
            .filter_map(|line| {
                let (_, name) = line.trim().strip_prefix("Self::")?.split_once("=> \"")?;
                Some(name.strip_suffix("\",")?.to_owned())
            })
            .collect()
    }

    /// Every engine capability is either one lavapipe has, and so required
    /// under `--lavapipe`, or one it lacks, so that a new capability can't
    /// quietly skip in CI. And no name is in both lists, or unknown to the
    /// engine, which would fail every GPU test.
    #[test]
    fn every_capability_is_classified_for_lavapipe() {
        let engine = engine_capabilities();
        assert!(
            engine.len() >= CAPABILITIES.len() + LACKS.len(),
            "{engine:?}"
        );
        for name in &engine {
            let has = CAPABILITIES.contains(&name.as_str());
            let lacks = LACKS.contains(&name.as_str());
            assert!(
                has != lacks,
                "{name} must be in exactly one of CAPABILITIES and LACKS"
            );
        }
        for name in CAPABILITIES.iter().chain(LACKS) {
            assert!(
                engine.iter().any(|known| known == name),
                "{name} is unknown"
            );
        }
    }

    /// With everything installed, nothing is missing.
    #[test]
    fn nothing_missing() {
        assert!(missing_prerequisites(|_| true, |_| true, |_| true).is_empty());
    }

    /// Every missing program, library and module is named, not only the
    /// first.
    #[test]
    fn every_missing_prerequisite_is_named() {
        let missing = missing_prerequisites(
            |program| program != "meson" && program != "flex",
            |library| library != "wayland-client",
            |module| module != "mako",
        );
        assert_eq!(
            missing,
            [
                "Meson (`meson`)",
                "Flex (`flex`)",
                "the `wayland-client` library, with its development files",
                "the `mako` Python module",
            ]
        );
    }

    /// Without `pkg-config` or Python, their libraries and modules can't be
    /// checked, so only the missing programs are named.
    #[test]
    fn missing_tools_hide_what_they_would_check() {
        let missing = missing_prerequisites(
            |program| program != "pkg-config" && program != "python3",
            |_| false,
            |_| false,
        );
        assert_eq!(
            missing,
            ["pkg-config (`pkg-config`)", "Python 3 (`python3`)"]
        );
    }
}
