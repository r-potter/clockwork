//! Developer tooling for Clockwork, run as `cargo xtask` (design 0001).
//!
//! `xtask` isn't part of the engine: it's a separate binary in the same Cargo
//! workspace, so that the checks a branch must pass are written once, in
//! Rust, and run the same way on Linux and Windows (ADR 0003). Its logic is
//! kept in pure functions with unit tests; the commands it runs are run
//! through `xshell`.
//!
//! See `docs/testing.md` for how to use it.

mod check;
mod prerequisites;
#[path = "../../build/slangc.rs"]
mod slangc;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by the smoke runs, in the next branch")
)]
mod window_systems;

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand};

/// Developer tooling for Clockwork. See docs/testing.md.
#[derive(Debug, Parser)]
#[command(name = "cargo xtask")]
struct Cli {
    /// The task to run.
    #[command(subcommand)]
    command: Command,
}

/// The tasks `xtask` can run.
#[derive(Debug, PartialEq, Eq, Subcommand)]
enum Command {
    /// Run every check a branch must pass: prerequisites, formatting, clippy
    /// (debug and release), rustdoc, and the tests that need no GPU.
    Check {
        /// Also run the GPU (T1) tests, with every capability required. Needs
        /// a Vulkan driver and the validation layer.
        #[arg(long)]
        gpu: bool,
        /// With --gpu, leave out the windowed smoke runs.
        #[arg(long, requires = "gpu")]
        no_window: bool,
    },
}

/// Runs the task named on the command line.
///
/// # Errors
///
/// Returns an error if a prerequisite is missing, or a check fails.
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Check { gpu, no_window } => {
            check::run(&workspace_root(), check::Options { gpu, no_window })
        }
    }
}

/// The workspace's root directory, where the commands run.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask is in a subdirectory of the workspace")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    //! Tests of the command-line interface.

    use clap::CommandFactory;

    use super::*;

    /// `clap`'s own consistency checks of the command definition, which
    /// otherwise only run when the matching options are used.
    #[test]
    fn command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    /// `cargo xtask check` selects the check task, with no GPU checks.
    #[test]
    fn check_parses() {
        let cli = Cli::try_parse_from(["xtask", "check"]).unwrap();
        assert_eq!(
            cli.command,
            Command::Check {
                gpu: false,
                no_window: false,
            }
        );
    }

    /// `--gpu --no-window` asks for the GPU tests without smoke runs.
    #[test]
    fn check_gpu_no_window_parses() {
        let cli = Cli::try_parse_from(["xtask", "check", "--gpu", "--no-window"]).unwrap();
        assert_eq!(
            cli.command,
            Command::Check {
                gpu: true,
                no_window: true,
            }
        );
    }

    /// `--no-window` only means something with `--gpu`, so it's an error
    /// without it.
    #[test]
    fn no_window_requires_gpu() {
        assert!(Cli::try_parse_from(["xtask", "check", "--no-window"]).is_err());
    }

    /// A task is required: running with none is an error, not a silent pass.
    #[test]
    fn no_task_is_an_error() {
        assert!(Cli::try_parse_from(["xtask"]).is_err());
    }

    /// Options from later slices, such as `--lavapipe`, must fail until
    /// they exist, rather than run fewer checks than requested.
    #[test]
    fn unknown_options_are_errors() {
        assert!(Cli::try_parse_from(["xtask", "check", "--gpu", "--lavapipe"]).is_err());
    }

    /// The workspace root holds the root `Cargo.toml` and this crate.
    #[test]
    fn workspace_root_contains_xtask() {
        let root = workspace_root();
        assert!(root.join("Cargo.toml").is_file());
        assert!(root.join("xtask").join("Cargo.toml").is_file());
    }
}
