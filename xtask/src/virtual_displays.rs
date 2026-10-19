//! `--virtual-displays`: an X11 and a Wayland display for the smoke runs, on
//! a Linux machine with no desktop session, such as a CI runner (R16 in
//! design 0001).
//!
//! [`VirtualDisplays::start`] starts `Xvfb`, a virtual X server, and Weston,
//! the reference Wayland compositor, with its headless backend. Each runs
//! until the [`VirtualDisplays`] is dropped, after the smoke runs. Their
//! output goes to log files under `target/virtual-displays/`, named in any
//! error about them.
//!
//! The smoke runs find them through the variables [`VirtualDisplays::env`]
//! returns, which are set for the smoke runs only. The arguments and the
//! parsing of `Xvfb`'s output are pure functions, tested on any machine.

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

/// How long to wait for Weston to create its socket.
const WESTON_TIMEOUT: Duration = Duration::from_secs(10);

/// The arguments to `Xvfb`. `-displayfd 1` makes it choose a free display
/// number, and write it to its standard output once it's ready for clients,
/// so there's no fixed number to collide with, and no polling.
///
/// It listens only on Linux's abstract socket, which clients try first, and
/// not on a socket in `/tmp/.X11-unix`. Under WSL, that directory is WSLg's,
/// and other users can't create sockets in it, so `Xvfb` would find no
/// display number it could use.
const XVFB_ARGS: [&str; 9] = [
    "-displayfd",
    "1",
    "-screen",
    "0",
    "1280x720x24",
    "-nolisten",
    "tcp",
    "-nolisten",
    "unix",
];

/// The arguments to Weston, listening on `socket` in `XDG_RUNTIME_DIR`.
fn weston_args(socket: &str) -> Vec<String> {
    vec![
        "--backend=headless".to_owned(),
        format!("--socket={socket}"),
        // Never blank the output, however long the checks before the smoke
        // runs take.
        "--idle-time=0".to_owned(),
    ]
}

/// The name of Weston's socket for the `xtask` process `pid`, unique to it so
/// that two checks on one machine don't collide.
fn weston_socket(pid: u32) -> String {
    format!("clockwork-wayland-{pid}")
}

/// The X11 display, such as `:99`, from the line `Xvfb` writes for
/// `-displayfd`.
///
/// # Errors
///
/// Returns an error if the line isn't a display number, which happens if
/// `Xvfb` exits without choosing one.
fn x11_display(line: &str) -> Result<String> {
    let number = line.trim();
    if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("Xvfb didn't report a display number, but {line:?}");
    }
    Ok(format!(":{number}"))
}

/// The environment variables pointing a smoke run at the virtual displays.
fn display_env(
    x11_display: &str,
    wayland_socket: &str,
    runtime_dir: &Path,
) -> Vec<(&'static str, String)> {
    vec![
        ("DISPLAY", x11_display.to_owned()),
        ("WAYLAND_DISPLAY", wayland_socket.to_owned()),
        ("XDG_RUNTIME_DIR", runtime_dir.display().to_string()),
    ]
}

/// A running `Xvfb` and headless Weston, stopped when this is dropped.
#[derive(Debug)]
pub struct VirtualDisplays {
    /// The `Xvfb` process.
    xvfb: Child,
    /// The Weston process.
    weston: Child,
    /// The X11 display `Xvfb` chose, such as `:99`.
    x11_display: String,
    /// The name of Weston's socket, in `runtime_dir`.
    wayland_socket: String,
    /// The `XDG_RUNTIME_DIR` Weston and the smoke runs use.
    runtime_dir: PathBuf,
}

impl VirtualDisplays {
    /// Starts both displays, with logs and, if `XDG_RUNTIME_DIR` isn't set,
    /// a runtime directory, under `target_dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if either program can't be started, or isn't ready
    /// in time.
    pub fn start(target_dir: &Path) -> Result<Self> {
        let dir = target_dir.join("virtual-displays");
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
        let runtime_dir = match std::env::var_os("XDG_RUNTIME_DIR") {
            Some(runtime_dir) if !runtime_dir.is_empty() => PathBuf::from(runtime_dir),
            _ => private_dir(&dir.join("runtime"))?,
        };

        let xvfb_log = dir.join("xvfb.log");
        let mut xvfb = Command::new("Xvfb")
            .args(XVFB_ARGS)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(log_file(&xvfb_log)?)
            .spawn()
            .context("failed to start Xvfb; is it installed (the xvfb package)?")?;
        let x11_display = match read_x11_display(&mut xvfb) {
            Ok(display) => display,
            Err(err) => {
                stop(&mut xvfb);
                return Err(err.context(format!("Xvfb didn't start; see {}", xvfb_log.display())));
            }
        };

        let wayland_socket = weston_socket(std::process::id());
        let weston_log = dir.join("weston.log");
        let weston = Command::new("weston")
            .args(weston_args(&wayland_socket))
            .env("XDG_RUNTIME_DIR", &runtime_dir)
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_DISPLAY")
            .stdin(Stdio::null())
            .stdout(log_file(&weston_log)?)
            .stderr(log_file(&weston_log)?)
            .spawn();
        let weston = match weston {
            Ok(weston) => weston,
            Err(err) => {
                stop(&mut xvfb);
                return Err(err)
                    .context("failed to start Weston; is it installed (the weston package)?");
            }
        };
        // From here, dropping `displays` stops both.
        let mut displays = Self {
            xvfb,
            weston,
            x11_display,
            wayland_socket,
            runtime_dir,
        };
        displays
            .wait_for_weston()
            .with_context(|| format!("Weston didn't start; see {}", weston_log.display()))?;
        eprintln!(
            "virtual displays: X11 {}, Wayland {} (logs in {})",
            displays.x11_display,
            displays.wayland_socket,
            dir.display()
        );
        Ok(displays)
    }

    /// Waits for Weston's socket to appear, which it creates once it's ready
    /// for clients.
    ///
    /// # Errors
    ///
    /// Returns an error if Weston exits, or doesn't create its socket within
    /// [`WESTON_TIMEOUT`].
    fn wait_for_weston(&mut self) -> Result<()> {
        let socket = self.runtime_dir.join(&self.wayland_socket);
        let start = Instant::now();
        while !socket.exists() {
            if let Some(status) = self.weston.try_wait()? {
                bail!("Weston exited with {status}");
            }
            if start.elapsed() > WESTON_TIMEOUT {
                bail!(
                    "Weston didn't create {} within {WESTON_TIMEOUT:?}",
                    socket.display()
                );
            }
            thread::sleep(Duration::from_millis(50));
        }
        Ok(())
    }

    /// The environment variables that point a smoke run at these displays.
    pub fn env(&self) -> Vec<(&'static str, String)> {
        display_env(&self.x11_display, &self.wayland_socket, &self.runtime_dir)
    }
}

impl Drop for VirtualDisplays {
    /// Stops both displays.
    fn drop(&mut self) {
        stop(&mut self.weston);
        stop(&mut self.xvfb);
    }
}

/// Reads the display `xvfb` chose from its standard output, which blocks
/// until it's ready for clients, or has exited.
///
/// # Errors
///
/// Returns an error if `xvfb` exits without reporting a display.
fn read_x11_display(xvfb: &mut Child) -> Result<String> {
    let stdout = xvfb.stdout.take().context("Xvfb's output isn't piped")?;
    let mut line = String::new();
    BufReader::new(stdout).read_line(&mut line)?;
    x11_display(&line)
}

/// Stops `child` and waits for it to exit. Errors are ignored: the process
/// may already have exited, and nothing more can be done about it if not.
fn stop(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Creates the log file at `path`, replacing the previous run's.
///
/// # Errors
///
/// Returns an error if the file can't be created.
fn log_file(path: &Path) -> Result<File> {
    File::create(path).with_context(|| format!("failed to create {}", path.display()))
}

/// Creates the directory at `path`, if needed, readable only by this user,
/// as Wayland requires of `XDG_RUNTIME_DIR`.
///
/// # Errors
///
/// Returns an error if the directory can't be created, or its permissions
/// set.
fn private_dir(path: &Path) -> Result<PathBuf> {
    fs::create_dir_all(path).with_context(|| format!("failed to create {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to set the permissions of {}", path.display()))?;
    }
    Ok(path.to_path_buf())
}

/// Whether `name` is set to a non-empty value in `env`, the environment the
/// virtual displays give the smoke runs.
pub fn is_set_in(env: &[(&'static str, String)], name: &str) -> bool {
    env.iter()
        .any(|(key, value)| *key == name && !value.is_empty())
}

#[cfg(test)]
mod tests {
    //! Tests of the virtual displays' configuration. Starting them needs
    //! Linux with `Xvfb` and Weston installed, so it's covered by the smoke
    //! runs in CI instead.

    use super::*;

    /// `Xvfb` chooses its own display number, reports it on its standard
    /// output when ready, and accepts only local connections, through the
    /// abstract socket alone.
    #[test]
    fn xvfb_reports_a_local_display() {
        assert!(XVFB_ARGS.starts_with(&["-displayfd", "1"]));
        assert!(XVFB_ARGS.ends_with(&["-nolisten", "tcp", "-nolisten", "unix"]));
    }

    /// The number `Xvfb` writes becomes an X11 display name.
    #[test]
    fn display_number_becomes_a_display() {
        assert_eq!(x11_display("99\n").unwrap(), ":99");
        assert_eq!(x11_display("0").unwrap(), ":0");
    }

    /// No output, as when `Xvfb` fails to start, or anything other than a
    /// number, is an error.
    #[test]
    fn missing_display_number_is_an_error() {
        for line in ["", "\n", "Fatal server error", ":99", "-1"] {
            assert!(x11_display(line).is_err(), "{line:?} was accepted");
        }
    }

    /// Weston runs headless, on its own socket.
    #[test]
    fn weston_runs_headless_on_its_own_socket() {
        let args = weston_args("clockwork-wayland-42");
        assert!(args.contains(&"--backend=headless".to_owned()), "{args:?}");
        assert!(
            args.contains(&"--socket=clockwork-wayland-42".to_owned()),
            "{args:?}"
        );
    }

    /// Each `xtask` process has its own Weston socket.
    #[test]
    fn weston_sockets_are_per_process() {
        assert_ne!(weston_socket(1), weston_socket(2));
    }

    /// The smoke runs get both displays, and the runtime directory where
    /// Weston's socket is.
    #[test]
    fn env_names_both_displays_and_the_runtime_dir() {
        let runtime_dir = Path::new("run").join("user");
        let env = display_env(":99", "clockwork-wayland-42", &runtime_dir);
        assert_eq!(
            env,
            [
                ("DISPLAY", ":99".to_owned()),
                ("WAYLAND_DISPLAY", "clockwork-wayland-42".to_owned()),
                ("XDG_RUNTIME_DIR", runtime_dir.display().to_string()),
            ]
        );
    }

    /// The window systems see the virtual displays as set.
    #[test]
    fn displays_count_as_set() {
        let env = display_env(":99", "clockwork-wayland-42", Path::new("run"));
        assert!(is_set_in(&env, "DISPLAY"));
        assert!(is_set_in(&env, "WAYLAND_DISPLAY"));
        assert!(!is_set_in(&env, "WAYLAND_SOCKET"));
    }
}
