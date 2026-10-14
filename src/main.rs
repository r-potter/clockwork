//! Clockwork application entry point.

mod rhi;

use std::io::IsTerminal;

use anyhow::Result;
use rhi::GraphicsApi;
use tracing::info;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

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

    // Initialise the renderer backend. Only Vulkan is available for now.
    let rhi = rhi::create(GraphicsApi::Vulkan)?;
    info!("using {:?} backend", rhi.api());
    Ok(())
}
