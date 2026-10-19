//! `cargo xtask check`: every check a branch must pass (R16 in design 0001).
//!
//! The checks run in order, stopping at the first failure:
//! 1. the prerequisites check (see [`crate::prerequisites`])
//! 2. the commands from [`steps`]: formatting, clippy in debug and release,
//!    rustdoc, and the tests: those that need no GPU, or, with `--gpu`, every
//!    tier, followed by a smoke run under each window system
//!
//! The list of commands is built by a pure function, so it's tested without
//! running anything.

use std::path::Path;

use anyhow::{Context, Result, bail};
use xshell::Shell;

use crate::prerequisites;
use crate::slangc::Platform;
use crate::virtual_displays::{self, VirtualDisplays};
use crate::window_systems::{self, SmokeRun, WindowSystem};

/// What `cargo xtask check` was asked to check.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// Also run the T1 tests, which need a GPU, with every capability
    /// required, and the smoke runs.
    pub gpu: bool,
    /// With `gpu`, leave out the windowed smoke runs, on purpose.
    pub no_window: bool,
    /// With `gpu`, on Linux: run the smoke runs on virtual displays, rather
    /// than the desktop session's.
    pub virtual_displays: bool,
}

impl Options {
    /// Whether these options ask for the smoke runs.
    fn smoke_runs(self) -> bool {
        self.gpu && !self.no_window
    }

    /// Checks that these options make sense on `platform`. The command line
    /// already rejects `--virtual-displays` without `--gpu`, or with
    /// `--no-window`.
    ///
    /// # Errors
    ///
    /// Returns an error for `--virtual-displays` on Windows, whose one window
    /// system needs no display server.
    pub fn validate(self, platform: Platform) -> Result<()> {
        if self.virtual_displays && platform != Platform::Linux {
            bail!("--virtual-displays is only for Linux, where Xvfb and Weston provide them");
        }
        Ok(())
    }
}

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
pub struct Step {
    /// What the step checks, for progress and failure messages.
    pub name: &'static str,
    /// The arguments to `cargo`.
    pub args: Vec<&'static str>,
    /// Environment variables set for this command only.
    pub env: Vec<(&'static str, String)>,
    /// Environment variables unset for this command only.
    pub unset: Vec<&'static str>,
}

impl Step {
    /// A step running `cargo` with `args`, and no change to the environment.
    fn cargo(name: &'static str, args: &[&'static str]) -> Self {
        Self {
            name,
            args: args.to_vec(),
            env: Vec::new(),
            unset: Vec::new(),
        }
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
/// order. `smoke_runs` are added after the tests, if `options` asks for them,
/// with `display_env` set: the virtual displays' variables, if there are
/// any.
pub fn steps(
    options: Options,
    smoke_runs: &[SmokeRun],
    display_env: &[(&'static str, String)],
) -> Vec<Step> {
    let tests = if options.gpu {
        // Every tier, and every capability required, so that no T1 test is
        // skipped on the machines that run this.
        Step {
            env: vec![("CLOCKWORK_REQUIRED_CAPABILITIES", "all".to_owned())],
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
        tests,
    ];
    if options.smoke_runs() {
        steps.extend(
            smoke_runs
                .iter()
                .map(|run| Step::smoke_run(run, display_env)),
        );
    }
    steps
}

/// Runs every check `options` asks for in the workspace at `root`.
///
/// # Errors
///
/// Returns an error if the options don't apply to this platform, if the
/// virtual displays don't start, if a window system has no display for its
/// smoke run, if a prerequisite is missing, or if any check fails, naming the
/// check.
pub fn run(root: &Path, options: Options) -> Result<()> {
    let platform = Platform::current();
    options.validate(platform)?;
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

    prerequisites::check(&sh, options.gpu)?;

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    for step in steps(options, &smoke_runs, &display_env) {
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
    };

    /// `--gpu --no-window`.
    const GPU_NO_WINDOW: Options = Options {
        gpu: true,
        no_window: true,
        virtual_displays: false,
    };

    /// The smoke runs for a Linux desktop session, with both window systems.
    fn linux_smoke_runs() -> Vec<SmokeRun> {
        window_systems::smoke_runs(Platform::Linux, |_| true).unwrap()
    }

    /// The steps for `options`, on a Linux desktop.
    fn steps_for(options: Options) -> Vec<Step> {
        steps(options, &linux_smoke_runs(), &[])
    }

    /// The names of the steps for `options`, on a Linux desktop.
    fn names_for(options: Options) -> Vec<&'static str> {
        steps_for(options).iter().map(|step| step.name).collect()
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
            assert!(args.contains(&"--all-targets"), "{name}: {args:?}");
            assert!(
                args.ends_with(&["--", "-D", "warnings"]),
                "{name}: {args:?}"
            );
        }
        assert!(step("clippy (release)").args.contains(&"--release"));
    }

    /// Rustdoc covers private items, and its warnings fail the check.
    #[test]
    fn rustdoc_denies_warnings_on_private_items() {
        let doc = step("rustdoc");
        assert!(doc.args.contains(&"--document-private-items"));
        assert_eq!(doc.env, [("RUSTDOCFLAGS", "-D warnings".to_owned())]);
    }

    /// The test step skips T1 tests, which need a GPU, by their module name.
    #[test]
    fn tests_skip_t1_modules() {
        let args = step("tests without a GPU").args;
        assert!(args.ends_with(&["--", "--skip", "gpu_tests"]), "{args:?}");
    }

    /// With `--gpu`, the tests run every tier, in place of the T0-only step,
    /// with every capability required, so missing capabilities fail rather
    /// than skip.
    #[test]
    fn the_gpu_option_runs_every_tier_with_all_capabilities_required() {
        let names = names_for(GPU_NO_WINDOW);
        assert!(!names.contains(&"tests without a GPU"), "{names:?}");
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
        let names: Vec<_> = steps(GPU, &runs, &[])
            .iter()
            .map(|step| step.name)
            .collect();
        assert_eq!(names.last(), Some(&"smoke run under Win32"));
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
        let steps = steps(GPU, &linux_smoke_runs(), &virtual_display_env());
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
        let with = steps(GPU, &linux_smoke_runs(), &virtual_display_env());
        let without = steps_for(GPU);
        let count = with.len() - 2;
        assert_eq!(with[..count], without[..count]);
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
