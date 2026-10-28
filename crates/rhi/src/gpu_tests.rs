//! T1 tests of creating backends through [`create`]. They create real
//! graphics API instances, so they need a driver; `cargo xtask check` skips
//! them by this module's name, and `cargo xtask check --gpu` runs them.
//!
//! Each runs in a [`TestSession`], with validation required, so any warning
//! or error a backend logs while it's created, used or destroyed fails the
//! test.

use std::sync::Arc;
use std::thread;

use clockwork_diagnostics::capture::TestSession;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::tests::ALL_GRAPHICS_APIS;
use super::{WindowSize, WindowTarget, create};
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

/// Every backend's resources can be used from another thread, and outlive
/// the backend they came from: here the backend is dropped first, and the
/// shared core is destroyed with the last `Arc`, on the other thread, with
/// nothing logged (design 0002, "Ownership and lifetimes").
#[test]
fn resources_are_shared_and_outlive_their_backend() {
    let session = TestSession::new();
    let options = backend_options(&session);
    for &api in ALL_GRAPHICS_APIS {
        let rhi = create(api, None, &options)
            .unwrap_or_else(|err| panic!("failed to create {api:?} backend: {err:?}"));
        let resources = Arc::clone(rhi.resources());
        let device_name = resources.capabilities().device_name.clone();
        drop(rhi);

        let span = session.span().clone();
        let name_on_thread = thread::spawn(move || {
            let _entered = span.entered();
            resources.capabilities().device_name.clone()
        })
        .join()
        .expect("the thread using the resources panicked");
        assert_eq!(name_on_thread, device_name, "{api:?}");
    }
}
