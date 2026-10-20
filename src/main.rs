//! Clockwork application entry point.

mod diagnostics;
mod rhi;
#[cfg(test)]
mod test_support;

use std::io::IsTerminal;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use diagnostics::{DiagnosticCounter, Session, SessionLayer, is_session_span};
use rhi::{
    BackendOptions, GraphicsApi, RenderHardwareInterface, Validation, ValidationFeatureSet,
    WindowSize, WindowTarget,
};
use tracing::level_filters::LevelFilter;
use tracing::{info, trace};
use tracing_subscriber::filter::{FilterExt, filter_fn};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Title of the application window.
const WINDOW_TITLE: &str = "Clockwork";

/// Clockwork, a small game engine.
///
/// Logs go to stderr, at `INFO` and above by default. Set `RUST_LOG` to change
/// what's printed, for example `RUST_LOG=debug`.
// The doc comments on this struct and its fields are the `--help` text.
#[derive(Debug, PartialEq, Eq, Parser)]
#[command(name = "clockwork", version)]
struct Options {
    /// Only consider GPUs whose name contains SUBSTRING, ignoring case.
    /// [default: the best GPU]
    #[arg(long, value_name = "SUBSTRING", value_parser = parse_device)]
    device: Option<String>,
    /// Validation features: off, or a comma-separated list of core, sync,
    /// gpu and best-practices (each implies core). Used if the validation
    /// layer is available, unless --fail-on-warnings is given. [default:
    /// sync in debug builds, off in release builds]
    #[arg(long, value_name = "FEATURES")]
    validation: Option<ValidationFeatureSet>,
    /// Exit cleanly after rendering N frames. [default: run until the window
    /// is closed]
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..))]
    exit_after_frames: Option<u64>,
    /// Require validation, unless --validation is off, and exit with an error
    /// if any warning or error was logged.
    #[arg(long)]
    fail_on_warnings: bool,
}

impl Options {
    /// The validation the application asks for (design 0001, "Validation"):
    /// the `--validation` features, or by default synchronization validation
    /// in debug builds and none in release builds. They're required under
    /// `--fail-on-warnings`, and otherwise used if they're available.
    fn validation(&self) -> Validation {
        validation(
            self.validation,
            self.fail_on_warnings,
            cfg!(debug_assertions),
        )
    }
}

/// Parses `--device`, rejecting a value that's empty or only whitespace. An
/// empty filter would match every device, and a blank one every device with a
/// space in its name, so either is surely a mistake.
fn parse_device(value: &str) -> Result<String, &'static str> {
    if value.trim().is_empty() {
        Err("must contain something other than whitespace")
    } else {
        Ok(value.to_owned())
    }
}

/// The validation for the requested features, `None` if `--validation`
/// wasn't given, under `--fail-on-warnings` if `fail_on_warnings` is set, in
/// a debug build if `debug` is set. Pure, so that both kinds of build are
/// tested in either.
///
/// `--fail-on-warnings` requires validation, so that a run with no warnings
/// can't be one that wasn't validated, unless the features are `off`: then it
/// fails only on what the engine itself logs.
fn validation(
    requested: Option<ValidationFeatureSet>,
    fail_on_warnings: bool,
    debug: bool,
) -> Validation {
    let default = if debug {
        ValidationFeatureSet::SYNC
    } else {
        ValidationFeatureSet::OFF
    };
    let features = requested.unwrap_or(default);
    if features.is_off() {
        Validation::Off
    } else if fail_on_warnings {
        Validation::Required(features)
    } else {
        Validation::Preferred(features)
    }
}

/// Starts the application.
///
/// Any error that reaches this point is printed by `anyhow` and the process
/// exits with a non-zero status.
fn main() -> Result<()> {
    // An invalid command line, or `--help`, exits here, with `clap`'s message.
    let options = Options::parse();

    // Log to stderr at `INFO` and above by default. Set `RUST_LOG` to change
    // this, e.g. `RUST_LOG=debug` or `RUST_LOG=clockwork::rhi::vulkan=trace`.
    // The application has a single session, so its span is left out of the
    // output rather than prefixing every line.
    let fmt_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy()
        .and(filter_fn(|metadata| !is_session_span(metadata)));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        // Only colour the output when it is going to a terminal.
        .with_ansi(std::io::stderr().is_terminal())
        .with_filter(fmt_filter);
    // Counts warnings and errors whatever `RUST_LOG` says, since it has its
    // own filter.
    let counter = DiagnosticCounter::default();
    tracing_subscriber::registry()
        .with(SessionLayer::layer())
        .with(fmt_layer)
        .with(counter.layer())
        .init();

    // Every event on this thread is attributed to the application's session,
    // including those from the `log` crate.
    let session = Session::new();
    let _entered = session.span().enter();

    // The summary is logged however the run ended, so that the warnings
    // leading up to an error are counted too.
    let result = run(&options, &session);
    let (errors, warnings) = (counter.errors(), counter.warnings());
    info!("{}", summary(errors, warnings));
    result?;
    check_warnings(errors, warnings, options.fail_on_warnings)
}

/// Runs the application in `session` until its window is closed, or it has
/// run the frames `--exit-after-frames` asked for.
///
/// # Errors
///
/// Returns an error if the event loop, the window or the renderer backend
/// can't be created, or if rendering fails.
fn run(options: &Options, session: &Session) -> Result<()> {
    let event_loop = EventLoop::new().context("failed to create event loop")?;
    // Run the loop continuously rather than waiting for events, so that the
    // engine can pace frames itself.
    event_loop.set_control_flow(ControlFlow::Poll);

    let backend_options = BackendOptions {
        session: session.id(),
        device_filter: options.device.clone(),
        validation: options.validation(),
    };
    let mut app = App::new(backend_options, options.exit_after_frames);
    event_loop
        .run_app(&mut app)
        .context("failed to run event loop")?;
    app.error.map_or(Ok(()), Err)
}

/// The summary logged at exit (R15), such as `0 errors and 2 warnings were
/// logged`. Suppressed and advisory messages aren't counted.
fn summary(errors: u32, warnings: u32) -> String {
    format!(
        "{} and {} were logged",
        count(errors, "error"),
        count(warnings, "warning")
    )
}

/// `n` followed by `noun`, made plural unless `n` is 1.
fn count(n: u32, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

/// Checks the counts at exit, under `--fail-on-warnings` if
/// `fail_on_warnings` is set.
///
/// # Errors
///
/// Returns an error under `--fail-on-warnings` if any error or warning was
/// counted.
fn check_warnings(errors: u32, warnings: u32, fail_on_warnings: bool) -> Result<()> {
    if fail_on_warnings && errors + warnings > 0 {
        bail!(
            "failing because of --fail-on-warnings: {}",
            summary(errors, warnings)
        );
    }
    Ok(())
}

/// Application state driven by the `winit` event loop.
struct App {
    /// How to create the renderer backend.
    options: BackendOptions,
    /// Exit after this many frames, or `None` to run until the window is
    /// closed.
    exit_after_frames: Option<u64>,
    /// Created when the application is first resumed.
    window: Option<Arc<Window>>,
    /// Renders to `window`. Created along with it.
    rhi: Option<Box<dyn RenderHardwareInterface>>,
    /// When the previous frame started. `None` before the first frame, and
    /// after suspending, so that time spent suspended is not counted.
    last_frame_start: Option<Instant>,
    /// The number of frames run so far.
    frame_count: u64,
    /// An error that stopped the event loop, returned from `main`.
    error: Option<anyhow::Error>,
}

impl App {
    /// An application that will create its backend with `options`, once the
    /// event loop first resumes it, and exit after `exit_after_frames`
    /// frames, if that's set.
    fn new(options: BackendOptions, exit_after_frames: Option<u64>) -> Self {
        Self {
            options,
            exit_after_frames,
            window: None,
            rhi: None,
            last_frame_start: None,
            frame_count: 0,
            error: None,
        }
    }

    /// Creates the window and the renderer backend that renders to it.
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attributes = Window::default_attributes().with_title(WINDOW_TITLE);
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("failed to create window")?,
        );

        // Only Vulkan is available for now.
        let rhi = rhi::create(
            GraphicsApi::Vulkan,
            Some(window_target(&window)),
            &self.options,
        )?;
        info!("using {:?} backend", rhi.api());

        self.window = Some(window);
        self.rhi = Some(rhi);
        Ok(())
    }

    /// Runs one frame: updates the game state, then renders it.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    fn frame(&mut self) -> Result<()> {
        let start = Instant::now();
        let delta = self
            .last_frame_start
            .map_or(Duration::ZERO, |last| start - last);
        self.last_frame_start = Some(start);
        self.frame_count += 1;
        trace!(frame = self.frame_count, ?delta, "starting frame");

        self.update(delta);
        if let Some(rhi) = &mut self.rhi {
            rhi.render_frame()?;
        }
        Ok(())
    }

    /// Whether the application has run the frames `--exit-after-frames` asked
    /// for. Always `false` without it.
    fn ran_requested_frames(&self) -> bool {
        self.exit_after_frames
            .is_some_and(|frames| self.frame_count >= frames)
    }

    /// Advances the game state by `delta`, the time since the previous frame.
    fn update(&mut self, _delta: Duration) {
        // TODO: update the game state.
    }

    /// Stores `err` to be returned from `main`, and stops the event loop.
    fn fail(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.error = Some(err);
        event_loop.exit();
    }
}

/// Describes `window` and its current size for the renderer backend.
fn window_target(window: &Arc<Window>) -> WindowTarget {
    WindowTarget {
        window: window.clone(),
        size: window_size(window.inner_size()),
    }
}

/// Converts a `winit` window size to the renderer backend's representation.
fn window_size(size: PhysicalSize<u32>) -> WindowSize {
    WindowSize {
        width: size.width,
        height: size.height,
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // The first resume creates the window and backend. Some platforms
        // (such as Android) suspend and resume again later, and the backend
        // then needs the window back.
        let result = if let (Some(window), Some(rhi)) = (&self.window, &mut self.rhi) {
            rhi.resume(window_target(window))
        } else {
            self.init(event_loop)
        };
        if let Err(err) = result {
            self.fail(event_loop, err);
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // On some platforms (such as Android) the window's native handle is
        // destroyed while suspended, so the backend must release everything
        // that uses it.
        if let Some(rhi) = &mut self.rhi {
            rhi.suspend();
        }
        self.last_frame_start = None;
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Called once per event loop iteration, after pending events have
        // been handled. In `ControlFlow::Poll` mode the loop does not wait
        // for events, so this runs a frame on every iteration.
        //
        // No frames run before the backend is created, or while suspended,
        // when the game is paused and there is nothing to render to. Nor do
        // they once the loop is exiting: `winit` may call this again after
        // `exit`, which would run more frames than `--exit-after-frames`
        // asked for.
        let Some(rhi) = &self.rhi else {
            return;
        };
        if rhi.is_suspended() || event_loop.exiting() {
            return;
        }
        if let Err(err) = self.frame() {
            self.fail(event_loop, err);
        } else if self.ran_requested_frames() {
            info!("exiting after {} frames, as requested", self.frame_count);
            event_loop.exit();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Destroy the backend, and with it the window surface and swapchain,
        // while the event loop's connection to the window system is still
        // open.
        self.rhi = None;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(rhi) = &mut self.rhi {
                    rhi.resize(window_size(size));
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests of the command-line options (R14).

    use clap::CommandFactory;
    use clap::error::ErrorKind;

    use super::*;

    /// Parses the command line `args`, which leaves out the program name.
    fn parse(args: &[&str]) -> Result<Options, clap::Error> {
        Options::try_parse_from(std::iter::once("clockwork").chain(args.iter().copied()))
    }

    /// `clap`'s own consistency checks of the command definition, which
    /// otherwise only run when the matching options are used.
    #[test]
    fn command_definition_is_valid() {
        Options::command().debug_assert();
    }

    /// With no options, nothing is set, so each option's default applies.
    #[test]
    fn no_options_leaves_every_default() {
        assert_eq!(
            parse(&[]).unwrap(),
            Options {
                device: None,
                validation: None,
                exit_after_frames: None,
                fail_on_warnings: false,
            }
        );
    }

    /// `--help` prints the help text and exits successfully, which `clap`
    /// reports as an error of its own kind.
    #[test]
    fn help_is_available() {
        let err = parse(&["--help"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
        assert_eq!(err.exit_code(), 0);
    }

    /// Every option appears in the help text.
    #[test]
    fn help_lists_every_option() {
        let help = Options::command().render_long_help().to_string();
        for option in [
            "--device",
            "--validation",
            "--exit-after-frames",
            "--fail-on-warnings",
        ] {
            assert!(help.contains(option), "{option} missing from:\n{help}");
        }
    }

    /// `--device` sets the device filter.
    #[test]
    fn device_sets_the_filter() {
        let options = parse(&["--device", "4090"]).unwrap();
        assert_eq!(options.device.as_deref(), Some("4090"));
    }

    /// An empty `--device` would match every device, and a blank one every
    /// device with a space in its name, so both are rejected.
    #[test]
    fn empty_or_blank_device_is_an_error() {
        for value in ["", " ", "\t", "  \t "] {
            let err = parse(&["--device", value]).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::ValueValidation, "{value:?}");
        }
    }

    /// A filter with spaces in it, as GPU names have, is kept as given.
    #[test]
    fn device_with_spaces_is_kept() {
        let options = parse(&["--device", "RTX 4090"]).unwrap();
        assert_eq!(options.device.as_deref(), Some("RTX 4090"));
    }

    /// `--validation` takes the same syntax as `CLOCKWORK_VALIDATION`,
    /// through the same parser.
    #[test]
    fn validation_parses_features() {
        let options = parse(&["--validation", "sync,gpu"]).unwrap();
        assert_eq!(
            options.validation,
            Some(ValidationFeatureSet::SYNC.with(rhi::ValidationFeature::Gpu))
        );
        let options = parse(&["--validation", "off"]).unwrap();
        assert_eq!(options.validation, Some(ValidationFeatureSet::OFF));
    }

    /// An unknown feature is an error, naming the valid ones.
    #[test]
    fn invalid_validation_is_an_error() {
        let err = parse(&["--validation", "sink"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ValueValidation);
        assert!(err.to_string().contains("best-practices"), "{err}");
    }

    /// `--exit-after-frames` takes a frame count.
    #[test]
    fn exit_after_frames_parses() {
        let options = parse(&["--exit-after-frames", "300"]).unwrap();
        assert_eq!(options.exit_after_frames, Some(300));
    }

    /// Zero frames would exit before rendering anything, so it's rejected,
    /// as are values that aren't counts.
    #[test]
    fn invalid_frame_counts_are_errors() {
        for value in ["0", "-1", "many"] {
            assert!(
                parse(&["--exit-after-frames", value]).is_err(),
                "{value} was accepted"
            );
        }
    }

    /// Options that don't exist are errors.
    #[test]
    fn unknown_options_are_errors() {
        let err = parse(&["--frames", "3"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
    }

    /// Debug builds prefer synchronization validation by default, and
    /// release builds don't validate.
    #[test]
    fn default_validation_depends_on_the_build() {
        assert_eq!(
            validation(None, false, true),
            Validation::Preferred(ValidationFeatureSet::SYNC)
        );
        assert_eq!(validation(None, false, false), Validation::Off);
    }

    /// `--validation` is preferred in either kind of build, so release
    /// builds can be validated too.
    #[test]
    fn requested_validation_is_preferred_in_any_build() {
        let features = ValidationFeatureSet::SYNC.with(rhi::ValidationFeature::BestPractices);
        for debug in [true, false] {
            assert_eq!(
                validation(Some(features), false, debug),
                Validation::Preferred(features)
            );
        }
    }

    /// `--validation off` turns validation off, even in debug builds.
    #[test]
    fn validation_off_turns_it_off() {
        assert_eq!(
            validation(Some(ValidationFeatureSet::OFF), false, true),
            Validation::Off
        );
    }

    /// `--fail-on-warnings` is a flag.
    #[test]
    fn fail_on_warnings_parses() {
        assert!(parse(&["--fail-on-warnings"]).unwrap().fail_on_warnings);
    }

    /// `--fail-on-warnings` requires validation, with the default features
    /// or the requested ones, in either kind of build.
    #[test]
    fn fail_on_warnings_requires_validation() {
        assert_eq!(
            validation(None, true, true),
            Validation::Required(ValidationFeatureSet::SYNC)
        );
        let features = ValidationFeatureSet::SYNC.with(rhi::ValidationFeature::Gpu);
        for debug in [true, false] {
            assert_eq!(
                validation(Some(features), true, debug),
                Validation::Required(features)
            );
        }
    }

    /// `--fail-on-warnings` doesn't turn validation on where it would be off:
    /// in release builds by default, or with `--validation off`. The run then
    /// fails only on what the engine itself logs.
    #[test]
    fn fail_on_warnings_keeps_validation_off() {
        assert_eq!(validation(None, true, false), Validation::Off);
        assert_eq!(
            validation(Some(ValidationFeatureSet::OFF), true, true),
            Validation::Off
        );
    }

    /// The summary counts each kind, in the singular for one.
    #[test]
    fn summary_counts_errors_and_warnings() {
        assert_eq!(summary(0, 2), "0 errors and 2 warnings were logged");
        assert_eq!(summary(1, 1), "1 error and 1 warning were logged");
    }

    /// Without `--fail-on-warnings`, warnings and errors are only summarised.
    #[test]
    fn counts_pass_without_fail_on_warnings() {
        assert!(check_warnings(3, 4, false).is_ok());
    }

    /// Under `--fail-on-warnings`, a single warning or error fails the run,
    /// and the message gives the counts.
    #[test]
    fn fail_on_warnings_fails_on_any_count() {
        assert!(check_warnings(0, 0, true).is_ok());
        let err = check_warnings(0, 1, true).unwrap_err();
        assert_eq!(
            err.to_string(),
            "failing because of --fail-on-warnings: 0 errors and 1 warning were logged"
        );
        assert!(check_warnings(1, 0, true).is_err());
    }
}
