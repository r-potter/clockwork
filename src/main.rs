//! Clockwork application entry point.

mod rhi;

use anyhow::Result;
use rhi::GraphicsApi;

/// Starts the application.
///
/// Any error that reaches this point is printed by `anyhow` and the process
/// exits with a non-zero status.
fn main() -> Result<()> {
    // Initialise the renderer backend. Only Vulkan is available for now.
    let rhi = rhi::create(GraphicsApi::Vulkan)?;
    println!("Using {:?} backend", rhi.api());
    Ok(())
}
