//! Clockwork application entry point.

mod rhi;

use std::io::IsTerminal;

use anyhow::{Context, Result};
use rhi::GraphicsApi;
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

    // Initialise the renderer backend. Only Vulkan is available for now.
    let rhi = rhi::create(GraphicsApi::Vulkan)?;
    info!("using {:?} backend", rhi.api());

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
    window: Option<Window>,
    /// An error that stopped the event loop, returned from `main`.
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Some platforms (such as Android) resume more than once.
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes().with_title(WINDOW_TITLE);
        match event_loop.create_window(attributes) {
            Ok(window) => self.window = Some(window),
            Err(err) => {
                self.error = Some(anyhow::Error::new(err).context("failed to create window"));
                event_loop.exit();
            }
        }
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
