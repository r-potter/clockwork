//! T1 tests of creating backends through [`create`]. They create real
//! graphics API instances, so they need a driver; `cargo xtask check` skips
//! them by this module's name, and `cargo xtask check --gpu` runs them.
//!
//! Each runs in a [`TestSession`], with validation required, so any warning
//! or error a backend logs while it's created, used or destroyed fails the
//! test.

use std::sync::Arc;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::tests::ALL_GRAPHICS_APIS;
use super::{WindowSize, WindowTarget, create};
use crate::diagnostics::capture::TestSession;
use crate::test_support::backend_options;

/// Every graphics API's backend can be created headless, so it needs no
/// display, and reports itself as headless and not suspended.
#[test]
fn create_succeeds_for_every_graphics_api() {
    let session = TestSession::new();
    let options = backend_options(&session);
    for &api in ALL_GRAPHICS_APIS {
        let rhi = create(api, None, &options)
            .unwrap_or_else(|err| panic!("failed to create {api:?} backend: {err:?}"));
        assert_eq!(rhi.api(), api);
        assert!(rhi.is_headless());
        assert!(!rhi.is_suspended());
    }
}

/// A window whose handles are never available. Enough to check that a
/// headless backend rejects windows without using them.
struct UnavailableWindow;

impl HasDisplayHandle for UnavailableWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

impl HasWindowHandle for UnavailableWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

/// A headless backend has no window to suspend, resize or resume, so it
/// ignores the first two and rejects the third, staying headless.
#[test]
fn headless_backends_ignore_suspend_and_resize_and_reject_resume() {
    let session = TestSession::new();
    let options = backend_options(&session);
    let size = WindowSize {
        width: 640,
        height: 480,
    };
    for &api in ALL_GRAPHICS_APIS {
        let mut rhi = create(api, None, &options)
            .unwrap_or_else(|err| panic!("failed to create {api:?} backend: {err:?}"));
        rhi.suspend();
        assert!(rhi.is_headless());
        assert!(!rhi.is_suspended());
        rhi.resize(size);
        assert!(rhi.render_frame().is_ok());
        let target = WindowTarget {
            window: Arc::new(UnavailableWindow),
            size,
        };
        assert!(rhi.resume(target).is_err());
        assert!(rhi.is_headless());
    }
}
