//! Clockwork application entry point.

mod diagnostics;
mod rhi;
#[cfg(test)]
mod test_support;

use std::io::IsTerminal;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use diagnostics::{DiagnosticCounter, Session, SessionLayer, is_session_span};
use rhi::{
    BackendOptions, GraphicsApi, RenderHardwareInterface, Validation, ValidationFeatures,
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

/// Starts the application.
///
/// Any error that reaches this point is printed by `anyhow` and the process
/// exits with a non-zero status.
fn main() -> Result<()> {
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

    let event_loop = EventLoop::new().context("failed to create event loop")?;
    // Run the loop continuously rather than waiting for events, so that the
    // engine can pace frames itself.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(BackendOptions {
        session: session.id(),
        device_filter: None,
        validation: default_validation(),
    });
    event_loop
        .run_app(&mut app)
        .context("failed to run event loop")?;
    app.error.map_or(Ok(()), Err)
}

/// The validation the application asks for (design 0001, "Validation"):
/// synchronization validation in debug builds if it's available, and none in
/// release builds.
fn default_validation() -> Validation {
    if cfg!(debug_assertions) {
        Validation::Preferred(ValidationFeatures::SYNC)
    } else {
        Validation::Off
    }
}

/// Application state driven by the `winit` event loop.
struct App {
    /// How to create the renderer backend.
    options: BackendOptions,
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
    /// event loop first resumes it.
    fn new(options: BackendOptions) -> Self {
        Self {
            options,
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
        // when the game is paused and there is nothing to render to.
        let Some(rhi) = &self.rhi else {
            return;
        };
        if rhi.is_suspended() {
            return;
        }
        if let Err(err) = self.frame() {
            self.fail(event_loop, err);
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
