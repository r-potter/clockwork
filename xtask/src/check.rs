//! `cargo xtask check`: every check a branch must pass (R16 in design 0001).
//!
//! The checks run in order, stopping at the first failure:
//! 1. the prerequisites check (see [`crate::prerequisites`])
//! 2. the commands from [`steps`]: formatting, clippy in debug and release,
//!    rustdoc, each crate built on its own, and the tests: those that need no
//!    GPU, or, with `--gpu`, every tier, followed by a smoke run under each
//!    window system
//!
//! The list of commands is built by a pure function, so it's tested without
//! running anything.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use xshell::Shell;

use crate::slangc::Platform;
use crate::virtual_displays::{self, VirtualDisplays};
use crate::window_systems::{self, SmokeRun, WindowSystem};
use crate::{lavapipe, prerequisites};

/// What `cargo xtask check` was asked to check.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Options {
    /// Also run the T1 tests, which need a GPU, with every capability
    /// required, and the smoke runs.
    pub gpu: bool,
    /// With `gpu`, leave out the windowed smoke runs, on purpose.
    pub no_window: bool,
    /// With `gpu`, on Linux: run the smoke runs on virtual displays, rather
    /// than the desktop session's.
    pub virtual_displays: bool,
    /// With `gpu`, on Linux: run the GPU tests and smoke runs on the pinned
    /// lavapipe alone, requiring the capabilities it has rather than all of
    /// them.
    pub lavapipe: bool,
}

impl Options {
    /// Whether these options ask for the smoke runs.
    fn smoke_runs(self) -> bool {
        self.gpu && !self.no_window
    }

    /// Checks that these options make sense on `platform`. The command line
    /// already rejects `--virtual-displays` without `--gpu`, or with
    /// `--no-window`, and `--lavapipe` without `--gpu`.
    ///
    /// # Errors
    ///
    /// Returns an error for `--virtual-displays` on Windows, whose one window
    /// system needs no display server, and for `--lavapipe` on Windows, where
    /// it isn't built.
    pub(crate) fn validate(self, platform: Platform) -> Result<()> {
        if self.virtual_displays && platform != Platform::Linux {
            bail!("--virtual-displays is only for Linux, where Xvfb and Weston provide them");
        }
        if self.lavapipe && platform != Platform::Linux {
            bail!("--lavapipe is only for Linux, where `cargo xtask lavapipe` builds it");
        }
        Ok(())
    }

    /// The capabilities whose absence fails a GPU test, as
    /// `CLOCKWORK_REQUIRED_CAPABILITIES` takes them: those lavapipe has with
    /// `--lavapipe`, and otherwise all of them.
    fn required_capabilities(self) -> String {
        if self.lavapipe {
            lavapipe::CAPABILITIES.join(",")
        } else {
            "all".to_owned()
        }
    }
}

/// The variables through which the Vulkan loader finds drivers. With
/// `--lavapipe`, [`run`] sets the first to lavapipe's manifest, so that it's
/// the only driver, and unsets the others, which would add drivers or be
/// ignored.
const DRIVER_VARIABLES: [&str; 3] = ["VK_DRIVER_FILES", "VK_ADD_DRIVER_FILES", "VK_ICD_FILENAMES"];

/// The arguments to `cargo` for a smoke run: the application, in a window,
/// for a fixed number of frames rather than a fixed time, failing on any
/// warning or error (R16 in design 0001).
const SMOKE_RUN_ARGS: [&str; 5] = [
    "run",
    "--",
    "--exit-after-frames",
    "300",
    "--fail-on-warnings",
];

/// One command that `cargo xtask check` runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Step {
    /// What the step checks, for progress and failure messages.
    pub name: String,
    /// The arguments to `cargo`.
    pub args: Vec<String>,
    /// Environment variables set for this command only.
    pub env: Vec<(&'static str, String)>,
    /// Environment variables unset for this command only.
    pub unset: Vec<&'static str>,
}

impl Step {
    /// A step running `cargo` with `args`, and no change to the environment.
    fn cargo(name: &str, args: &[&str]) -> Self {
        Self {
            name: name.to_owned(),
            args: args.iter().map(|&arg| arg.to_owned()).collect(),
            env: Vec::new(),
            unset: Vec::new(),
        }
    }

    /// The steps that build `package` on its own: its library and binaries,
    /// then every target, as `cargo test --package` would. In a workspace
    /// build, a crate gets the features every other crate turns on for their
    /// shared dependencies, so it can build there and not on its own. And
    /// dev-dependencies' features apply only to builds with tests, so a
    /// library that relies on one builds with every target and not without.
    fn alone(package: &str) -> [Self; 2] {
        [
            Self::cargo(
                &format!("{package} on its own"),
                &["check", "--package", package],
            ),
            Self::cargo(
                &format!("{package} on its own, with every target"),
                &["check", "--package", package, "--all-targets"],
            ),
        ]
    }

    /// The step for `run`: the application under its window system, with
    /// `display_env` set, except for the variables `run` unsets.
    fn smoke_run(run: &SmokeRun, display_env: &[(&'static str, String)]) -> Self {
        let name = match run.window_system {
            WindowSystem::Win32 => "smoke run under Win32",
            WindowSystem::Wayland => "smoke run under Wayland",
            WindowSystem::X11 => "smoke run under X11",
        };
        Self {
            env: display_env
                .iter()
                .filter(|(key, _)| !run.unset.contains(key))
                .cloned()
                .collect(),
            unset: run.unset.to_vec(),
            ..Self::cargo(name, &SMOKE_RUN_ARGS)
        }
    }
}

/// The commands `cargo xtask check` runs after the prerequisites check, in
/// order. Each of `packages` is built on its own before the tests.
/// `smoke_runs` are added after the tests, if `options` asks for them, with
/// `display_env` set: the virtual displays' variables, if there are any.
pub(crate) fn steps(
    options: Options,
    packages: &[String],
    smoke_runs: &[SmokeRun],
    display_env: &[(&'static str, String)],
) -> Vec<Step> {
    let tests = if options.gpu {
        // Every tier, and every capability the device should have required,
        // so that no T1 test is skipped on the machines that run this.
        Step {
            env: vec![(
                "CLOCKWORK_REQUIRED_CAPABILITIES",
                options.required_capabilities(),
            )],
            ..Step::cargo("tests, including GPU tests", &["test", "--workspace"])
        }
    } else {
        // T1 tests live in modules named `gpu_tests`, so this runs every
        // test that needs no GPU.
        Step::cargo(
            "tests without a GPU",
            &["test", "--workspace", "--", "--skip", "gpu_tests"],
        )
    };
    let mut steps = vec![
        Step::cargo("formatting", &["fmt", "--all", "--check"]),
        Step::cargo(
            "clippy (debug)",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        ),
        Step::cargo(
            "clippy (release)",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--release",
                "--",
                "-D",
                "warnings",
            ],
        ),
        Step {
            // `-D warnings` makes broken intra-doc links fail the check.
            env: vec![("RUSTDOCFLAGS", "-D warnings".to_owned())],
            ..Step::cargo(
                "rustdoc",
                &[
                    "doc",
                    "--workspace",
                    "--no-deps",
                    "--document-private-items",
                ],
            )
        },
    ];
    steps.extend(packages.iter().flat_map(|package| Step::alone(package)));
    steps.push(tests);
    if options.smoke_runs() {
        steps.extend(
            smoke_runs
                .iter()
                .map(|run| Step::smoke_run(run, display_env)),
        );
    }
    steps
}

/// The names of the workspace's default members, from `metadata`, the output
/// of `cargo metadata --no-deps --format-version 1`: the engine's crates and
/// the application, without `xtask`, which `cargo xtask` already builds on
/// its own.
///
/// # Errors
///
/// Returns an error if `metadata` isn't JSON, or lacks a default member or
/// its package's name.
fn default_members(metadata: &str) -> Result<Vec<String>> {
    let metadata: Value =
        serde_json::from_str(metadata).context("`cargo metadata` printed invalid JSON")?;
    let ids = metadata["workspace_default_members"]
        .as_array()
        .context("`cargo metadata` listed no default members")?;
    let packages = metadata["packages"]
        .as_array()
        .context("`cargo metadata` listed no packages")?;
    ids.iter()
        .map(|id| {
            packages
                .iter()
                .find(|package| package["id"] == *id)
                .and_then(|package| package["name"].as_str())
                .map(str::to_owned)
                .with_context(|| format!("`cargo metadata` has no package named for {id}"))
        })
        .collect()
}

/// Runs every check `options` asks for in the workspace at `root`.
///
/// # Errors
///
/// Returns an error if the options don't apply to this platform, if the
/// virtual displays don't start, if a window system has no display for its
/// smoke run, if a prerequisite is missing, or if any check fails, naming the
/// check.
pub(crate) fn run(root: &Path, options: Options) -> Result<()> {
    let platform = Platform::current();
    options.validate(platform)?;
    // Set in this process, before anything else runs, so that the loader
    // queries in the prerequisites check see lavapipe alone too, and every
    // command inherits it. The caller's environment is unchanged.
    let driver = if options.lavapipe {
        let manifest = lavapipe::manifest(&root.join("target"))?;
        // SAFETY: `xtask` is single-threaded, and has started no threads
        // yet, so nothing can read the environment while it's changed.
        unsafe {
            std::env::set_var(DRIVER_VARIABLES[0], &manifest);
            for variable in &DRIVER_VARIABLES[1..] {
                std::env::remove_var(variable);
            }
        }
        Some(manifest)
    } else {
        None
    };
    // Started and decided first, so that a missing display fails at once,
    // not after the tests. The virtual displays are stopped when `displays`
    // is dropped, on return.
    let displays = if options.virtual_displays {
        Some(VirtualDisplays::start(&root.join("target"))?)
    } else {
        None
    };
    let display_env = displays
        .as_ref()
        .map(VirtualDisplays::env)
        .unwrap_or_default();
    let smoke_runs = if options.smoke_runs() {
        window_systems::smoke_runs(platform, |name| {
            if displays.is_some() {
                virtual_displays::is_set_in(&display_env, name)
            } else {
                std::env::var_os(name).is_some_and(|value| !value.is_empty())
            }
        })?
    } else {
        Vec::new()
    };
    let sh = Shell::new()?;
    sh.change_dir(root);

    prerequisites::check(&sh, options.gpu, driver.as_deref())?;

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let metadata = sh
        .cmd(&cargo)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .quiet()
        .read()
        .context("failed to run `cargo metadata`")?;
    let packages = default_members(&metadata)?;
    for step in steps(options, &packages, &smoke_runs, &display_env) {
        eprintln!("\n==> {}", step.name);
        let mut cmd = sh.cmd(&cargo).args(&step.args);
        for (key, value) in &step.env {
            cmd = cmd.env(key, value);
        }
        for key in &step.unset {
            cmd = cmd.env_remove(key);
        }
        cmd.run()
            .with_context(|| format!("check failed: {}", step.name))?;
    }
    eprintln!("\nall checks passed");
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Tests of the list of checks.

    use super::*;

    /// `--gpu`, with smoke runs.
    const GPU: Options = Options {
        gpu: true,
        no_window: false,
        virtual_displays: false,
        lavapipe: false,
    };

    /// `--gpu --no-window`.
    const GPU_NO_WINDOW: Options = Options {
        gpu: true,
        no_window: true,
        virtual_displays: false,
        lavapipe: false,
    };

    /// `--gpu --lavapipe`.
    const GPU_LAVAPIPE: Options = Options {
        lavapipe: true,
        ..GPU
    };

    /// The smoke runs for a Linux desktop session, with both window systems.
    fn linux_smoke_runs() -> Vec<SmokeRun> {
        window_systems::smoke_runs(Platform::Linux, |_| true).unwrap()
    }

    /// Two of the workspace's packages, to build on their own.
    fn packages() -> Vec<String> {
        vec!["clockwork".to_owned(), "clockwork-rhi".to_owned()]
    }

    /// The steps for `options`, on a Linux desktop.
    fn steps_for(options: Options) -> Vec<Step> {
        steps(options, &packages(), &linux_smoke_runs(), &[])
    }

    /// The names of the steps for `options`, on a Linux desktop.
    fn names_for(options: Options) -> Vec<String> {
        steps_for(options)
            .into_iter()
            .map(|step| step.name)
            .collect()
    }

    /// Whether `args` contains `arg`.
    fn has(args: &[String], arg: &str) -> bool {
        args.iter().any(|a| a == arg)
    }

    /// Whether `args` ends with `suffix`.
    fn ends_with(args: &[String], suffix: &[&str]) -> bool {
        args.len() >= suffix.len() && args[args.len() - suffix.len()..] == *suffix
    }

    /// Finds the step called `name`, in the steps for `options`.
    fn step_for(options: Options, name: &str) -> Step {
        steps_for(options)
            .into_iter()
            .find(|step| step.name == name)
            .unwrap_or_else(|| panic!("no step called {name}"))
    }

    /// Finds the step called `name`, in the default steps.
    fn step(name: &str) -> Step {
        step_for(Options::default(), name)
    }

    /// The design's order: cheap checks first, tests last.
    #[test]
    fn steps_run_in_the_designed_order() {
        assert_eq!(
            names_for(Options::default()),
            [
                "formatting",
                "clippy (debug)",
                "clippy (release)",
                "rustdoc",
                "clockwork on its own",
                "clockwork on its own, with every target",
                "clockwork-rhi on its own",
                "clockwork-rhi on its own, with every target",
                "tests without a GPU",
            ]
        );
    }

    /// Clippy warnings, including the workspace's lints, fail the check in
    /// both profiles, and cover tests and build scripts too.
    #[test]
    fn clippy_denies_warnings_for_all_targets() {
        for name in ["clippy (debug)", "clippy (release)"] {
            let args = step(name).args;
            assert!(has(&args, "--all-targets"), "{name}: {args:?}");
            assert!(
                ends_with(&args, &["--", "-D", "warnings"]),
                "{name}: {args:?}"
            );
        }
        assert!(has(&step("clippy (release)").args, "--release"));
    }

    /// Rustdoc covers private items, and its warnings fail the check.
    #[test]
    fn rustdoc_denies_warnings_on_private_items() {
        let doc = step("rustdoc");
        assert!(has(&doc.args, "--document-private-items"));
        assert_eq!(doc.env, [("RUSTDOCFLAGS", "-D warnings".to_owned())]);
    }

    /// Each package is built by itself, so with only the features it turns
    /// on: first without dev-dependencies, then with every target.
    #[test]
    fn each_package_builds_on_its_own() {
        assert_eq!(
            step("clockwork-rhi on its own").args,
            ["check", "--package", "clockwork-rhi"]
        );
        assert_eq!(
            step("clockwork-rhi on its own, with every target").args,
            ["check", "--package", "clockwork-rhi", "--all-targets"]
        );
    }

    /// The test step skips T1 tests, which need a GPU, by their module name.
    #[test]
    fn tests_skip_t1_modules() {
        let args = step("tests without a GPU").args;
        assert!(ends_with(&args, &["--", "--skip", "gpu_tests"]), "{args:?}");
    }

    /// With `--gpu`, the tests run every tier, in place of the T0-only step,
    /// with every capability required, so missing capabilities fail rather
    /// than skip.
    #[test]
    fn the_gpu_option_runs_every_tier_with_all_capabilities_required() {
        let names = names_for(GPU_NO_WINDOW);
        assert!(!has(&names, "tests without a GPU"), "{names:?}");
        let tests = step_for(GPU_NO_WINDOW, "tests, including GPU tests");
        assert_eq!(tests.args, ["test", "--workspace"]);
        assert_eq!(
            tests.env,
            [("CLOCKWORK_REQUIRED_CAPABILITIES", "all".to_owned())]
        );
    }

    /// `--gpu --no-window` changes only the test step: the other checks are
    /// the same.
    #[test]
    fn gpu_keeps_the_other_checks() {
        let default = steps_for(Options::default());
        let gpu = steps_for(GPU_NO_WINDOW);
        assert_eq!(default.len(), gpu.len());
        assert_eq!(default[..default.len() - 1], gpu[..gpu.len() - 1]);
    }

    /// `--gpu` adds a smoke run for each window system after the tests, so
    /// that a broken test fails before a window opens.
    #[test]
    fn gpu_ends_with_a_smoke_run_per_window_system() {
        let names = names_for(GPU);
        assert_eq!(
            names[names.len() - 3..],
            [
                "tests, including GPU tests",
                "smoke run under Wayland",
                "smoke run under X11",
            ]
        );
    }

    /// Each smoke run renders a fixed number of frames, failing on any
    /// warning, and unsets what its window system needs unset.
    #[test]
    fn smoke_runs_fail_on_warnings_under_their_window_system() {
        let x11 = step_for(GPU, "smoke run under X11");
        assert_eq!(
            x11.args,
            [
                "run",
                "--",
                "--exit-after-frames",
                "300",
                "--fail-on-warnings"
            ]
        );
        assert_eq!(x11.unset, ["WAYLAND_DISPLAY", "WAYLAND_SOCKET"]);
        assert!(x11.env.is_empty());
    }

    /// Windows has a single smoke run, under Win32.
    #[test]
    fn windows_smoke_runs_under_win32() {
        let runs = window_systems::smoke_runs(Platform::Windows, |_| false).unwrap();
        let names: Vec<_> = steps(GPU, &packages(), &runs, &[])
            .into_iter()
            .map(|step| step.name)
            .collect();
        assert_eq!(names.last().unwrap(), "smoke run under Win32");
        assert_eq!(
            names.iter().filter(|name| name.contains("smoke")).count(),
            1
        );
    }

    /// `--no-window` leaves the smoke runs out, and without `--gpu` there
    /// are none.
    #[test]
    fn smoke_runs_need_gpu_and_a_window() {
        for options in [Options::default(), GPU_NO_WINDOW] {
            let names = names_for(options);
            assert!(
                !names.iter().any(|name| name.contains("smoke")),
                "{options:?}: {names:?}"
            );
        }
    }

    /// Variables like those [`VirtualDisplays::env`] returns.
    fn virtual_display_env() -> Vec<(&'static str, String)> {
        vec![
            ("DISPLAY", ":99".to_owned()),
            ("WAYLAND_DISPLAY", "clockwork-wayland-42".to_owned()),
            ("XDG_RUNTIME_DIR", "runtime".to_owned()),
        ]
    }

    /// Each smoke run gets the virtual displays' variables, except those it
    /// unsets to choose its window system.
    #[test]
    fn smoke_runs_use_the_virtual_displays() {
        let steps = steps(
            GPU,
            &packages(),
            &linux_smoke_runs(),
            &virtual_display_env(),
        );
        let env_of = |name| {
            let step = steps.iter().find(|step| step.name == name).unwrap();
            step.env.iter().map(|(key, _)| *key).collect::<Vec<_>>()
        };
        assert_eq!(
            env_of("smoke run under Wayland"),
            ["WAYLAND_DISPLAY", "XDG_RUNTIME_DIR"]
        );
        assert_eq!(
            env_of("smoke run under X11"),
            ["DISPLAY", "XDG_RUNTIME_DIR"]
        );
    }

    /// The virtual displays' variables are for the smoke runs only.
    #[test]
    fn other_steps_dont_use_the_virtual_displays() {
        let with = steps(
            GPU,
            &packages(),
            &linux_smoke_runs(),
            &virtual_display_env(),
        );
        let without = steps_for(GPU);
        let count = with.len() - 2;
        assert_eq!(with[..count], without[..count]);
    }

    /// With `--lavapipe`, a GPU test fails for a missing capability that
    /// lavapipe has, and skips for one it lacks, rather than failing for all
    /// of them.
    #[test]
    fn lavapipe_requires_the_capabilities_it_has() {
        let tests = step_for(GPU_LAVAPIPE, "tests, including GPU tests");
        assert_eq!(
            tests.env,
            [(
                "CLOCKWORK_REQUIRED_CAPABILITIES",
                "swapchain-maintenance,bc-compression".to_owned()
            )]
        );
    }

    /// `--lavapipe` changes only which capabilities are required: the steps
    /// are the same, smoke runs included, since the driver is chosen by the
    /// environment every step inherits.
    #[test]
    fn lavapipe_keeps_the_steps() {
        let names = names_for(GPU_LAVAPIPE);
        assert_eq!(names, names_for(GPU));
    }

    /// `--lavapipe` is for Linux only, since it's only built there.
    #[test]
    fn lavapipe_is_for_linux() {
        assert!(GPU_LAVAPIPE.validate(Platform::Linux).is_ok());
        assert!(GPU_LAVAPIPE.validate(Platform::Windows).is_err());
    }

    /// Output like `cargo metadata --no-deps --format-version 1`'s, cut down
    /// to the fields read. As in the real output, `xtask`'s ID leaves out its
    /// name, which is the same as its directory's.
    const METADATA: &str = r#"{
        "packages": [
            {"name": "clockwork", "id": "path+file:///w#clockwork@0.1.0"},
            {"name": "clockwork-rhi", "id": "path+file:///w/crates/rhi#clockwork-rhi@0.1.0"},
            {"name": "xtask", "id": "path+file:///w/xtask#0.1.0"}
        ],
        "workspace_members": [
            "path+file:///w#clockwork@0.1.0",
            "path+file:///w/crates/rhi#clockwork-rhi@0.1.0",
            "path+file:///w/xtask#0.1.0"
        ],
        "workspace_default_members": [
            "path+file:///w#clockwork@0.1.0",
            "path+file:///w/crates/rhi#clockwork-rhi@0.1.0"
        ]
    }"#;

    /// The packages built on their own are the default members, by name,
    /// leaving out `xtask`.
    #[test]
    fn default_members_are_named() {
        assert_eq!(
            default_members(METADATA).unwrap(),
            ["clockwork", "clockwork-rhi"]
        );
    }

    /// A default member with no package, or output that isn't JSON, is an
    /// error, rather than a crate quietly left out.
    #[test]
    fn unreadable_metadata_is_an_error() {
        let missing = METADATA.replace(
            r#"{"name": "clockwork-rhi", "id": "path+file:///w/crates/rhi#clockwork-rhi@0.1.0"},"#,
            "",
        );
        assert_ne!(missing, METADATA);
        assert!(default_members(&missing).is_err());
        assert!(default_members("not JSON").is_err());
    }

    /// `--virtual-displays` is for Linux only.
    #[test]
    fn virtual_displays_are_for_linux() {
        let options = Options {
            virtual_displays: true,
            ..GPU
        };
        assert!(options.validate(Platform::Linux).is_ok());
        assert!(options.validate(Platform::Windows).is_err());
        assert!(GPU.validate(Platform::Windows).is_ok());
    }
}
