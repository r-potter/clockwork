//! Clockwork application entry point.

mod rhi;

use std::io::IsTerminal;
use std::sync::Arc;

use anyhow::{Context, Result};
use rhi::{GraphicsApi, RenderHardwareInterface};
use tracing::info;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use winit::application::ApplicationHandler;
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
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .with_writer(std::io::stderr)
        // Only colour the output when it is going to a terminal.
        .with_ansi(std::io::stderr().is_terminal())
        .init();

    let event_loop = EventLoop::new().context("failed to create event loop")?;
    // Run the loop continuously rather than waiting for events, so that the
    // engine can pace frames itself.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .context("failed to run event loop")?;
    app.error.map_or(Ok(()), Err)
}

/// Application state driven by the `winit` event loop.
#[derive(Default)]
struct App {
    /// Created when the application is first resumed.
    window: Option<Arc<Window>>,
    /// Renders to `window`. Created along with it.
    rhi: Option<Box<dyn RenderHardwareInterface>>,
    /// An error that stopped the event loop, returned from `main`.
    error: Option<anyhow::Error>,
}

impl App {
    /// Creates the window and the renderer backend that renders to it.
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attributes = Window::default_attributes().with_title(WINDOW_TITLE);
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("failed to create window")?,
        );

        // Only Vulkan is available for now.
        let rhi = rhi::create(GraphicsApi::Vulkan, Some(window.clone()))?;
        info!("using {:?} backend", rhi.api());

        self.window = Some(window);
        self.rhi = Some(rhi);
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // The first resume creates the window and backend. Some platforms
        // (such as Android) suspend and resume again later, and the backend
        // then needs the window back.
        let result = if let (Some(window), Some(rhi)) = (&self.window, &mut self.rhi) {
            rhi.resume(window.clone())
        } else {
            self.init(event_loop)
        };
        if let Err(err) = result {
            self.error = Some(err);
            event_loop.exit();
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // On some platforms (such as Android) the window's native handle is
        // destroyed while suspended, so the backend must release everything
        // that uses it.
        if let Some(rhi) = &mut self.rhi {
            rhi.suspend();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Destroy the backend, and with it the window surface, while the
        // event loop's connection to the window system is still open.
        self.rhi = None;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if event == WindowEvent::CloseRequested {
            event_loop.exit();
        }
    }
}
