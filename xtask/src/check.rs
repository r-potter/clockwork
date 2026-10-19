//! `cargo xtask check`: every check a branch must pass (R16 in design 0001).
//!
//! The checks run in order, stopping at the first failure:
//! 1. the prerequisites check (see [`crate::prerequisites`])
//! 2. the commands from [`steps`]: formatting, clippy in debug and release,
//!    rustdoc, and the tests that need no GPU
//!
//! The list of commands is built by a pure function, so it's tested without
//! running anything.

use std::path::Path;

use anyhow::{Context, Result};
use xshell::Shell;

use crate::prerequisites;

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
pub fn steps() -> Vec<Step> {
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
        // T1 tests live in modules named `gpu_tests`, so this runs every
        // test that needs no GPU.
        Step::cargo(
            "tests without a GPU",
            &["test", "--workspace", "--", "--skip", "gpu_tests"],
        ),
    ]
}

/// Runs every check in the workspace at `root`.
///
/// # Errors
///
/// Returns an error if a prerequisite is missing or any check fails, naming
/// the check.
pub fn run(root: &Path) -> Result<()> {
    let sh = Shell::new()?;
    sh.change_dir(root);

    prerequisites::check(&sh)?;

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    for step in steps() {
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

    /// Finds the step called `name`.
    fn step(name: &str) -> Step {
        steps()
            .into_iter()
            .find(|step| step.name == name)
            .unwrap_or_else(|| panic!("no step called {name}"))
    }

    /// The design's order: cheap checks first, tests last.
    #[test]
    fn steps_run_in_the_designed_order() {
        let names: Vec<_> = steps().iter().map(|step| step.name).collect();
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
}
