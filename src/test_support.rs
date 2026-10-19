//! Helpers shared by the engine's tests. Compiled only for tests.
//!
//! This is the only place tests read the environment (design 0001,
//! "Configuration"): the engine itself never does, so its behaviour depends
//! only on its options.

use std::sync::Once;

use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, fmt, registry};

use crate::diagnostics::SessionLayer;
use crate::diagnostics::capture::TestCapture;

/// Installs the test subscriber as the process's global default, once per
/// test process. [`TestSession::new`](crate::diagnostics::capture::TestSession::new)
/// calls it, so tests don't need to.
///
/// The subscriber has three layers, each with its own filter:
/// 1. [`SessionLayer`], which attributes events to sessions
/// 2. a fmt layer using the test writer, so that logs appear in the output of
///    failing tests only, filtered by `RUST_LOG` (`INFO` and above by default)
/// 3. [`TestCapture`], which takes warnings and errors whatever `RUST_LOG`
///    says, and fails the tests they're attributed to
///
/// It also routes the `log` crate's records into `tracing`.
///
/// # Panics
///
/// Panics if another global subscriber, or another `log` logger, was
/// installed first. Tests must not install their own subscribers, which would
/// hide events from the capture layer.
pub fn install_test_subscriber() {
    /// Guards the installation.
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let fmt_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();
        registry()
            .with(SessionLayer::layer())
            .with(fmt::layer().with_test_writer().with_filter(fmt_filter))
            .with(TestCapture::layer())
            .try_init()
            .expect("another global subscriber was installed before the test subscriber");
    });
}
