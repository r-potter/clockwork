//! `cargo xtask check`: every check a branch must pass (R16 in design 0001).
//!
//! The checks run in order, stopping at the first failure:
//! 1. the prerequisites check (see [`crate::prerequisites`])
//! 2. the commands from [`steps`]: formatting, clippy in debug and release,
//!    rustdoc, and the tests: those that need no GPU, or, with `--gpu`, every
//!    tier
//!
//! The list of commands is built by a pure function, so it's tested without
//! running anything.

use std::path::Path;

use anyhow::{Context, Result, bail};
use xshell::Shell;

use crate::prerequisites;

/// What `cargo xtask check` was asked to check.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// Also run the T1 tests, which need a GPU, with every capability
    /// required.
    pub gpu: bool,
    /// With `gpu`, leave out the windowed smoke runs, on purpose.
    pub no_window: bool,
}

impl Options {
    /// Checks that these options ask for something this version can do.
    ///
    /// # Errors
    ///
    /// Returns an error for `--gpu` without `--no-window`: the smoke runs
    /// arrive with the application's command-line options, in
    /// `foundations-cli`, and running fewer checks than were asked for
    /// mustn't look like a pass.
    pub fn validate(self) -> Result<()> {
        if self.gpu && !self.no_window {
            bail!(
                "the windowed smoke runs that --gpu includes aren't implemented yet (they arrive \
                 in foundations-cli); run `cargo xtask check --gpu --no-window` to leave them out"
            );
        }
        Ok(())
    }
}

/// One command that `cargo xtask check` runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// What the step checks, for progress and failure messages.
    pub name: &'static str,
    /// The arguments to `cargo`.
    pub args: Vec<&'static str>,
    /// Environment variables set for this command only.
    pub env: Vec<(&'static str, &'static str)>,
}

impl Step {
    /// A step running `cargo` with `args`, and no extra environment.
    fn cargo(name: &'static str, args: &[&'static str]) -> Self {
        Self {
            name,
            args: args.to_vec(),
            env: Vec::new(),
        }
    }
}

/// The commands `cargo xtask check` runs after the prerequisites check, in
/// order.
pub fn steps(options: Options) -> Vec<Step> {
    let tests = if options.gpu {
        // Every tier, and every capability required, so that no T1 test is
        // skipped on the machines that run this.
        Step {
            env: vec![("CLOCKWORK_REQUIRED_CAPABILITIES", "all")],
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
    vec![
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
            env: vec![("RUSTDOCFLAGS", "-D warnings")],
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
    ]
}

/// Runs every check `options` asks for in the workspace at `root`.
///
/// # Errors
///
/// Returns an error if the options ask for something not implemented, if a
/// prerequisite is missing, or if any check fails, naming the check.
pub fn run(root: &Path, options: Options) -> Result<()> {
    options.validate()?;
    let sh = Shell::new()?;
    sh.change_dir(root);

    prerequisites::check(&sh, options.gpu)?;

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    for step in steps(options) {
        eprintln!("\n==> {}", step.name);
        let mut cmd = sh.cmd(&cargo).args(&step.args);
        for (key, value) in &step.env {
            cmd = cmd.env(key, value);
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

    /// `--gpu --no-window`.
    const GPU: Options = Options {
        gpu: true,
        no_window: true,
    };

    /// Finds the step called `name`, in the steps for `options`.
    fn step_for(options: Options, name: &str) -> Step {
        steps(options)
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
        let names: Vec<_> = steps(Options::default())
            .iter()
            .map(|step| step.name)
            .collect();
        assert_eq!(
            names,
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
        assert_eq!(doc.env, [("RUSTDOCFLAGS", "-D warnings")]);
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
        let names: Vec<_> = steps(GPU).iter().map(|step| step.name).collect();
        assert!(!names.contains(&"tests without a GPU"), "{names:?}");
        let tests = step_for(GPU, "tests, including GPU tests");
        assert_eq!(tests.args, ["test", "--workspace"]);
        assert_eq!(tests.env, [("CLOCKWORK_REQUIRED_CAPABILITIES", "all")]);
    }

    /// `--gpu` changes only the test step: the other checks are the same.
    #[test]
    fn gpu_keeps_the_other_checks() {
        let default = steps(Options::default());
        let gpu = steps(GPU);
        assert_eq!(default.len(), gpu.len());
        assert_eq!(default[..default.len() - 1], gpu[..gpu.len() - 1]);
    }

    /// `--gpu` without `--no-window` asks for smoke runs, which don't exist
    /// yet, so it's an error rather than a quieter pass.
    #[test]
    fn gpu_without_no_window_is_an_error() {
        let options = Options {
            gpu: true,
            no_window: false,
        };
        assert!(options.validate().is_err());
        assert!(GPU.validate().is_ok());
        assert!(Options::default().validate().is_ok());
    }
}
