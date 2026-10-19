//! Which window systems the smoke runs use, and how each run selects its own
//! (R21 in design 0001, ADR 0003).
//!
//! Each supported window system gets a smoke run:
//!
//! | Platform | Window system | Needs | Selected by |
//! |---|---|---|---|
//! | Windows | Win32 | Nothing | It's the only one |
//! | Linux | Wayland | `WAYLAND_DISPLAY` | Unsetting `DISPLAY`, so that X11 can't be used |
//! | Linux | X11 | `DISPLAY` | Unsetting `WAYLAND_DISPLAY` and `WAYLAND_SOCKET`, since `winit` prefers Wayland whenever either is set |
//!
//! A window system whose display isn't available is an error, not a silent
//! skip, so that a check can't pass having run fewer smoke runs than it
//! should.
//!
//! The decision is a pure function of the platform and the environment, so
//! that every platform's behaviour is tested on any machine (ADR 0003).

use std::fmt;

use anyhow::{Result, bail};

use crate::slangc::Platform;

/// The window systems the engine supports (ADR 0003).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowSystem {
    /// Windows' only window system.
    Win32,
    /// Wayland, on Linux.
    Wayland,
    /// X11, on Linux, including through Xwayland in a Wayland session.
    X11,
}

impl WindowSystem {
    /// The window systems `platform` supports, in the order their smoke runs
    /// run.
    pub fn supported(platform: Platform) -> &'static [Self] {
        match platform {
            Platform::Windows => &[Self::Win32],
            Platform::Linux => &[Self::Wayland, Self::X11],
        }
    }

    /// The environment variable that names this window system's display,
    /// without which the smoke run can't open a window. `None` for Win32,
    /// which needs none.
    fn display_variable(self) -> Option<&'static str> {
        match self {
            Self::Win32 => None,
            Self::Wayland => Some("WAYLAND_DISPLAY"),
            Self::X11 => Some("DISPLAY"),
        }
    }

    /// The environment variables to unset for a run under this window
    /// system, so that `winit` can't choose another one.
    fn variables_to_unset(self) -> &'static [&'static str] {
        match self {
            Self::Win32 => &[],
            Self::Wayland => &["DISPLAY"],
            Self::X11 => &["WAYLAND_DISPLAY", "WAYLAND_SOCKET"],
        }
    }
}

impl fmt::Display for WindowSystem {
    /// Formats the window system's usual name, such as `X11`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Win32 => "Win32",
            Self::Wayland => "Wayland",
            Self::X11 => "X11",
        })
    }
}

/// One smoke run: the application, in a window under one window system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmokeRun {
    /// The window system the run uses.
    pub window_system: WindowSystem,
    /// Environment variables unset for the run, so that it uses
    /// `window_system`.
    pub unset: &'static [&'static str],
}

/// The smoke runs for `platform`, one for each window system it supports.
/// `is_set` says whether an environment variable is set to a non-empty value.
///
/// # Errors
///
/// Returns an error naming each window system whose display isn't available,
/// and the variable that would name it.
pub fn smoke_runs(platform: Platform, is_set: impl Fn(&str) -> bool) -> Result<Vec<SmokeRun>> {
    let supported = WindowSystem::supported(platform);
    let missing: Vec<_> = supported
        .iter()
        .filter_map(|window_system| {
            let variable = window_system.display_variable()?;
            (!is_set(variable)).then(|| format!("{window_system} ({variable} isn't set)"))
        })
        .collect();
    if !missing.is_empty() {
        bail!(
            "no display for the smoke runs under {}; run in a desktop session with every \
             window system, or leave the smoke runs out with --no-window",
            missing.join(" or "),
        );
    }
    Ok(supported
        .iter()
        .map(|&window_system| SmokeRun {
            window_system,
            unset: window_system.variables_to_unset(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    //! Tests of the window systems chosen for each platform and environment,
    //! all of which run on any machine.

    use super::*;

    /// An environment in which exactly the variables in `set` are set.
    fn env(set: &'static [&'static str]) -> impl Fn(&str) -> bool {
        move |name| set.contains(&name)
    }

    /// The window systems of `runs`, in order.
    fn window_systems(runs: &[SmokeRun]) -> Vec<WindowSystem> {
        runs.iter().map(|run| run.window_system).collect()
    }

    /// Windows has one smoke run, under Win32, which needs no display
    /// variable and unsets nothing.
    #[test]
    fn windows_runs_win32() {
        let runs = smoke_runs(Platform::Windows, env(&[])).unwrap();
        assert_eq!(
            runs,
            [SmokeRun {
                window_system: WindowSystem::Win32,
                unset: &[],
            }]
        );
    }

    /// Display variables that happen to be set on Windows don't add runs.
    #[test]
    fn windows_ignores_linux_displays() {
        let runs = smoke_runs(Platform::Windows, env(&["DISPLAY", "WAYLAND_DISPLAY"])).unwrap();
        assert_eq!(window_systems(&runs), [WindowSystem::Win32]);
    }

    /// A Linux desktop session with both displays runs Wayland, then X11.
    #[test]
    fn linux_runs_wayland_and_x11() {
        let runs = smoke_runs(Platform::Linux, env(&["DISPLAY", "WAYLAND_DISPLAY"])).unwrap();
        assert_eq!(
            window_systems(&runs),
            [WindowSystem::Wayland, WindowSystem::X11]
        );
    }

    /// The Wayland run unsets `DISPLAY`, so that it can't use X11 instead.
    #[test]
    fn wayland_run_hides_x11() {
        let runs = smoke_runs(Platform::Linux, env(&["DISPLAY", "WAYLAND_DISPLAY"])).unwrap();
        assert_eq!(runs[0].unset, ["DISPLAY"]);
    }

    /// The X11 run unsets both of the variables that make `winit` choose
    /// Wayland.
    #[test]
    fn x11_run_hides_wayland() {
        let runs = smoke_runs(Platform::Linux, env(&["DISPLAY", "WAYLAND_DISPLAY"])).unwrap();
        assert_eq!(runs[1].unset, ["WAYLAND_DISPLAY", "WAYLAND_SOCKET"]);
    }

    /// An X11-only session can't run the Wayland smoke run, which is an
    /// error naming the missing variable, not a skip.
    #[test]
    fn linux_without_wayland_is_an_error() {
        let err = smoke_runs(Platform::Linux, env(&["DISPLAY"])).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("Wayland (WAYLAND_DISPLAY isn't set)"),
            "{message}"
        );
        assert!(!message.contains("X11"), "{message}");
    }

    /// A Wayland session without Xwayland can't run the X11 smoke run.
    #[test]
    fn linux_without_x11_is_an_error() {
        let err = smoke_runs(Platform::Linux, env(&["WAYLAND_DISPLAY"])).unwrap_err();
        assert!(err.to_string().contains("X11 (DISPLAY isn't set)"), "{err}");
    }

    /// With no display at all, as over SSH, the error names both, and says
    /// how to leave the smoke runs out.
    #[test]
    fn linux_without_displays_names_both() {
        let err = smoke_runs(Platform::Linux, env(&[])).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("Wayland") && message.contains("X11"),
            "{message}"
        );
        assert!(message.contains("--no-window"), "{message}");
    }

    /// Every supported window system's name, as used in messages.
    #[test]
    fn window_systems_display_their_names() {
        let names: Vec<_> = [
            WindowSystem::Win32,
            WindowSystem::Wayland,
            WindowSystem::X11,
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        assert_eq!(names, ["Win32", "Wayland", "X11"]);
    }
}
