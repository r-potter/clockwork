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
mod lavapipe;
mod prerequisites;
#[path = "../../build/slangc.rs"]
mod slangc;
mod virtual_displays;
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
        /// Also run the GPU (T1) tests, with every capability required, then
        /// a smoke run of the application under each window system the
        /// platform supports. Needs a Vulkan driver, the validation layer,
        /// and a display for each window system.
        #[arg(long)]
        gpu: bool,
        /// With --gpu, leave out the windowed smoke runs.
        #[arg(long, requires = "gpu")]
        no_window: bool,
        /// Linux, with --gpu: run the smoke runs on virtual displays, Xvfb
        /// and a headless Weston, started for them and stopped afterwards,
        /// rather than the desktop session's.
        #[arg(long, requires = "gpu", conflicts_with = "no_window")]
        virtual_displays: bool,
        /// Linux, with --gpu: run the GPU tests and smoke runs on lavapipe,
        /// built by `cargo xtask lavapipe`, as the only Vulkan driver,
        /// requiring only the capabilities lavapipe has.
        #[arg(long, requires = "gpu")]
        lavapipe: bool,
    },
    /// Linux: build lavapipe, Mesa's CPU Vulkan driver, from the pinned
    /// commit, in target/lavapipe/. Does nothing if it's already built.
    Lavapipe,
}

/// Runs the task named on the command line.
///
/// # Errors
///
/// Returns an error if a prerequisite is missing, or a check fails.
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Check {
            gpu,
            no_window,
            virtual_displays,
            lavapipe,
        } => check::run(
            &workspace_root(),
            check::Options {
                gpu,
                no_window,
                virtual_displays,
                lavapipe,
            },
        ),
        Command::Lavapipe => lavapipe::build(&workspace_root()),
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
                virtual_displays: false,
                lavapipe: false,
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
                virtual_displays: false,
                lavapipe: false,
            }
        );
    }

    /// `--no-window` only means something with `--gpu`, so it's an error
    /// without it.
    #[test]
    fn no_window_requires_gpu() {
        assert!(Cli::try_parse_from(["xtask", "check", "--no-window"]).is_err());
    }

    /// `--gpu --virtual-displays` asks for the smoke runs on virtual
    /// displays.
    #[test]
    fn check_gpu_virtual_displays_parses() {
        let cli = Cli::try_parse_from(["xtask", "check", "--gpu", "--virtual-displays"]).unwrap();
        assert_eq!(
            cli.command,
            Command::Check {
                gpu: true,
                no_window: false,
                virtual_displays: true,
                lavapipe: false,
            }
        );
    }

    /// CI's form: `--gpu --lavapipe --virtual-displays`.
    #[test]
    fn check_gpu_lavapipe_virtual_displays_parses() {
        let cli = Cli::try_parse_from([
            "xtask",
            "check",
            "--gpu",
            "--lavapipe",
            "--virtual-displays",
        ])
        .unwrap();
        assert_eq!(
            cli.command,
            Command::Check {
                gpu: true,
                no_window: false,
                virtual_displays: true,
                lavapipe: true,
            }
        );
    }

    /// `--lavapipe` chooses the driver for the GPU tests, so it's an error
    /// without `--gpu`.
    #[test]
    fn lavapipe_requires_gpu() {
        assert!(Cli::try_parse_from(["xtask", "check", "--lavapipe"]).is_err());
    }

    /// `--virtual-displays` is only for smoke runs, so it's an error without
    /// `--gpu`, or with `--no-window`.
    #[test]
    fn virtual_displays_need_smoke_runs() {
        assert!(Cli::try_parse_from(["xtask", "check", "--virtual-displays"]).is_err());
        assert!(
            Cli::try_parse_from([
                "xtask",
                "check",
                "--gpu",
                "--no-window",
                "--virtual-displays"
            ])
            .is_err()
        );
    }

    /// `cargo xtask lavapipe` selects the lavapipe build, which takes no
    /// options.
    #[test]
    fn lavapipe_parses() {
        let cli = Cli::try_parse_from(["xtask", "lavapipe"]).unwrap();
        assert_eq!(cli.command, Command::Lavapipe);
    }

    /// A task is required: running with none is an error, not a silent pass.
    #[test]
    fn no_task_is_an_error() {
        assert!(Cli::try_parse_from(["xtask"]).is_err());
    }

    /// Unknown options, such as a misspelt one, are errors, rather than
    /// running fewer checks than requested.
    #[test]
    fn unknown_options_are_errors() {
        assert!(Cli::try_parse_from(["xtask", "check", "--gpu", "--lavapipes"]).is_err());
    }

    /// The workspace root holds the root `Cargo.toml` and this crate.
    #[test]
    fn workspace_root_contains_xtask() {
        let root = workspace_root();
        assert!(root.join("Cargo.toml").is_file());
        assert!(root.join("xtask").join("Cargo.toml").is_file());
    }
}
