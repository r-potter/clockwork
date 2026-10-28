//! Render hardware interface: an abstraction over the underlying graphics API.
//!
//! Vulkan is currently the only supported backend, but the interface is kept
//! API-agnostic so that others (e.g. Metal, DirectX) can be added later.
//!
//! The application obtains a backend by calling [`create`] with the desired
//! [`GraphicsApi`], and from then on uses it only through the
//! [`RenderHardwareInterface`] trait. Concrete backend types are private to
//! this crate.
//!
//! A backend either renders to a [`Window`], or runs headless with no window
//! (for example, in tests and on CI machines without a display). A windowed
//! backend can be suspended, releasing its window, and later resumed with a
//! window again, without recreating the backend.
//!
//! Backends are configured with [`BackendOptions`]: the session their events
//! are attributed to, which devices to consider, and whether to validate
//! (design 0001, "Validation" and "Configuration"). The application and the
//! tests configure them the same way; only where the values come from
//! differs.
//!
//! What the selected device can do beyond the baseline is described by
//! [`Capabilities`] and [`PresentCapabilities`] (design 0002,
//! "Capabilities").
//!
//! RHI calls return an [`RhiError`], which tells apart the failures a caller
//! can act on (design 0002, "Error handling").
//!
//! To add a backend:
//! 1. add a variant to [`GraphicsApi`],
//! 2. add a submodule with a type implementing [`RenderHardwareInterface`],
//! 3. add a match arm for it in [`create`].

mod buffer;
mod capabilities;
mod error;
#[cfg(test)]
mod gpu_tests;
mod handle;
#[cfg(test)]
mod test_support;
mod texture;
mod vulkan;

use std::fmt::{self, Debug};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Result, bail};
use clockwork_diagnostics::SessionId;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub use buffer::{Buffer, BufferDesc, BufferUsage, GpuAddress};
pub use capabilities::{
    Capabilities, Limits, MAX_PUSH_DATA_SIZE, PresentCapabilities, QueueKind, QueueKindSet,
};
pub use error::{RhiError, RhiResult};
pub use texture::{Extent3d, Format, Texture, TextureDesc, TextureDimension, TextureUsage};
use vulkan::VulkanRenderHardwareInterface;

/// A window a backend can render to.
///
/// Implemented for any type that provides `raw-window-handle` display and
/// window handles, such as `winit::window::Window`, so this module does not
/// depend on a particular windowing library.
pub trait Window: HasDisplayHandle + HasWindowHandle {}

impl<T: HasDisplayHandle + HasWindowHandle + ?Sized> Window for T {}

/// The size of a window's drawable area, in physical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSize {
    /// The width, in physical pixels.
    pub width: u32,
    /// The height, in physical pixels.
    pub height: u32,
}

/// A window for a backend to render to, and its current size.
///
/// The size is needed because some window systems (such as Wayland) leave the
/// size of the rendered images to the application.
pub struct WindowTarget {
    /// The window, which the backend keeps alive while it renders to it.
    pub window: Arc<dyn Window>,
    /// The window's drawable size when the target was made.
    pub size: WindowSize,
}

/// Graphics API used by the render hardware interface.
///
/// Passed to [`create`] to choose a backend, and returned by
/// [`RenderHardwareInterface::api`] to identify one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsApi {
    /// The Vulkan API, implemented by `VulkanRenderHardwareInterface`.
    Vulkan,
}

/// Options for creating a backend.
#[derive(Debug, Clone)]
pub struct BackendOptions {
    /// The session the backend's events are attributed to, including those the
    /// graphics API reports from its own threads.
    pub session: SessionId,
    /// Only consider devices whose name contains this, ignoring case. `None`
    /// considers every device.
    pub device_filter: Option<String>,
    /// Whether to validate, and with which features.
    pub validation: Validation,
}

/// Whether validation must, may or mustn't be enabled.
///
/// The guarantee comes from [`create`] succeeding: there's deliberately no
/// way to ask a backend afterwards whether it validates, since an answer
/// that's never checked fails open, and zero messages would look the same as
/// no validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validation {
    /// Don't validate.
    Off,
    /// Enable these features if available. Otherwise log a warning, and
    /// continue with what is available.
    Preferred(ValidationFeatures),
    /// Enable these features, or fail to create the backend.
    Required(ValidationFeatures),
}

/// One validation feature, enabled as part of [`ValidationFeatures`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationFeature {
    /// Checks API usage against the specification's valid usage rules. Every
    /// other feature implies it.
    Core,
    /// Synchronization validation: hazards between commands, such as two
    /// writes with no barrier between them.
    Sync,
    /// GPU-assisted validation: checks inside shaders, such as out-of-bounds
    /// accesses. Slow, so off by default.
    Gpu,
    /// Best-practices validation: advice, which is logged but never counted
    /// or failing (`clockwork-diagnostics` classifies them as advisory).
    BestPractices,
}

impl ValidationFeature {
    /// Every feature, in the order they're listed in.
    pub const ALL: [Self; 4] = [Self::Core, Self::Sync, Self::Gpu, Self::BestPractices];

    /// The feature's name, as written in `--validation` and
    /// `CLOCKWORK_VALIDATION`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Sync => "sync",
            Self::Gpu => "gpu",
            Self::BestPractices => "best-practices",
        }
    }
}

/// A set of validation features.
///
/// Parsed from `off`, or a comma-separated list of `core`, `sync`, `gpu` and
/// `best-practices` (see [`ValidationFeature::name`]), and displayed the same
/// way.
///
/// Invariant: any feature implies [`ValidationFeature::Core`], so the set is
/// either empty ([`OFF`](Self::OFF)) or contains `Core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationFeatures {
    /// Whether [`ValidationFeature::Core`] is enabled.
    core: bool,
    /// Whether [`ValidationFeature::Sync`] is enabled.
    sync: bool,
    /// Whether [`ValidationFeature::Gpu`] is enabled.
    gpu: bool,
    /// Whether [`ValidationFeature::BestPractices`] is enabled.
    best_practices: bool,
}

impl ValidationFeatures {
    /// No validation at all.
    pub const OFF: Self = Self {
        core: false,
        sync: false,
        gpu: false,
        best_practices: false,
    };

    /// Core and synchronization validation: the default in debug builds and
    /// tests (R6).
    pub const SYNC: Self = Self::OFF.with(ValidationFeature::Sync);

    /// This set with `feature` added, and [`ValidationFeature::Core`], which
    /// every feature implies.
    pub const fn with(mut self, feature: ValidationFeature) -> Self {
        self.core = true;
        match feature {
            ValidationFeature::Core => {}
            ValidationFeature::Sync => self.sync = true,
            ValidationFeature::Gpu => self.gpu = true,
            ValidationFeature::BestPractices => self.best_practices = true,
        }
        self
    }

    /// Whether `feature` is in the set.
    pub const fn contains(self, feature: ValidationFeature) -> bool {
        match feature {
            ValidationFeature::Core => self.core,
            ValidationFeature::Sync => self.sync,
            ValidationFeature::Gpu => self.gpu,
            ValidationFeature::BestPractices => self.best_practices,
        }
    }

    /// Whether the set is empty, meaning no validation.
    pub const fn is_off(self) -> bool {
        !self.core
    }

    /// The features in the set, in [`ValidationFeature::ALL`]'s order.
    pub fn iter(self) -> impl Iterator<Item = ValidationFeature> {
        ValidationFeature::ALL
            .into_iter()
            .filter(move |&feature| self.contains(feature))
    }
}

impl fmt::Display for ValidationFeatures {
    /// Formats the set as it's parsed: `off`, or the features' names,
    /// separated by commas.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_off() {
            return f.write_str("off");
        }
        let names: Vec<_> = self.iter().map(ValidationFeature::name).collect();
        f.write_str(&names.join(","))
    }
}

impl FromStr for ValidationFeatures {
    type Err = anyhow::Error;

    /// Parses `off`, or a comma-separated list of feature names. Spaces
    /// around names are ignored.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty value or list item, an unknown feature
    /// name, or `off` combined with features.
    fn from_str(value: &str) -> Result<Self> {
        if value.trim() == "off" {
            return Ok(Self::OFF);
        }
        let mut features = Self::OFF;
        for name in value.split(',').map(str::trim) {
            let Some(feature) = ValidationFeature::ALL
                .into_iter()
                .find(|feature| feature.name() == name)
            else {
                let expected: Vec<_> = ValidationFeature::ALL.map(ValidationFeature::name).into();
                bail!(
                    "invalid validation features \"{value}\": expected off, or a comma-separated \
                     list of {}",
                    expected.join(", "),
                );
            };
            features = features.with(feature);
        }
        Ok(features)
    }
}

/// Where a buffer's memory is, and which processor can reach it (design
/// 0002, "Memory").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// GPU-only memory, for anything the CPU doesn't touch.
    Device,
    /// Memory the CPU writes and the GPU reads. Device-local where the
    /// device lets the CPU see most of its memory, such as with resizable
    /// BAR, and system memory otherwise.
    Upload,
    /// Memory the GPU writes and the CPU reads, cached for the CPU where the
    /// device allows.
    Readback,
}

/// The GPU's resources, shared by every thread that creates, fills or
/// destroys them (design 0002, "Architecture").
///
/// Obtained from [`RenderHardwareInterface::resources`], and shared as an
/// `Arc<dyn GpuResources>`. Every method may be called from any thread. The
/// resources live as long as their last `Arc`, which may be after the
/// [`RenderHardwareInterface`] they came from is dropped.
///
/// Used as a trait object, so it must stay object-safe, as
/// [`RenderHardwareInterface`] describes. Methods are added with the code
/// that implements them.
pub trait GpuResources: Debug + Send + Sync {
    /// The device's name, and its optional features and limits for resources
    /// and shaders.
    fn capabilities(&self) -> &Capabilities;

    /// Creates a buffer, with memory in `desc.location`.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`] if device memory runs out,
    /// or [`RhiError::Other`] if the buffer can't be created otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `desc.size` is zero.
    fn create_buffer(&self, desc: &BufferDesc<'_>) -> RhiResult<Buffer>;

    /// Destroys the buffer. Its handle is stale from then on.
    ///
    /// Nothing can submit work through a `Queue` yet, so the buffer is
    /// freed straight away. Once it can, design 0002 frees it only once
    /// nothing translated before now can use it.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    fn destroy_buffer(&self, buffer: Buffer);

    /// The buffer's address on the GPU.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    fn buffer_address(&self, buffer: Buffer) -> GpuAddress;

    /// Copies `data` into an upload buffer at `offset`.
    ///
    /// It doesn't synchronize with the GPU: the caller makes sure no
    /// submitted work still uses the range, as Vulkan requires. Upload
    /// memory is coherent, so the GPU sees the bytes in work submitted
    /// afterwards. Overlapping writes from two threads are serialized, in
    /// no particular order.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale, the buffer isn't in upload memory, or
    /// the range is out of bounds.
    fn write_buffer(&self, buffer: Buffer, offset: u64, data: &[u8]);

    /// Copies from a readback buffer at `offset` into `out`.
    ///
    /// It doesn't synchronize with the GPU: the caller waits for the work
    /// that last wrote the range first, as Vulkan requires.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::Other`] if memory that isn't coherent can't be
    /// invalidated before it's read.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale, the buffer isn't in readback memory,
    /// or the range is out of bounds.
    fn read_buffer(&self, buffer: Buffer, offset: u64, out: &mut [u8]) -> RhiResult<()>;

    /// Creates a texture, in GPU-only memory. Its contents are undefined
    /// until it's written.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`] if device memory runs out,
    /// or [`RhiError::Other`] if the texture can't be created otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the description is invalid: an empty extent, no layers or
    /// usage, mips outside the full chain, a shape its dimension doesn't
    /// allow, or an attachment usage that doesn't match its format. Also
    /// panics if the device doesn't support the format for the usage.
    fn create_texture(&self, desc: &TextureDesc<'_>) -> RhiResult<Texture>;

    /// Destroys the texture. Its handle is stale from then on.
    ///
    /// Nothing can submit work through a `Queue` yet, so the texture is
    /// freed straight away, as [`destroy_buffer`](Self::destroy_buffer)
    /// describes.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    fn destroy_texture(&self, texture: Texture);
}

// The resources are shared between threads, as an `Arc<dyn GpuResources>`
// (design 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync + ?Sized>() {}
    assert_send_sync::<dyn GpuResources>();
    assert_send_sync::<Arc<dyn GpuResources>>();
};

/// Operations every graphics API backend must provide.
///
/// Used as a trait object, so it must stay object-safe: no generic methods,
/// and any method returning `Self` (such as a constructor) needs a
/// `where Self: Sized` bound. GPU resources should therefore be exposed as
/// opaque handles rather than as backend-specific associated types.
///
/// A backend stays on the thread that created it, since it owns the window's
/// surface and swapchain. What other threads need is its [`GpuResources`].
pub trait RenderHardwareInterface: Debug {
    /// Returns the graphics API this backend implements.
    fn api(&self) -> GraphicsApi;

    /// The GPU resources, to share with other threads.
    fn resources(&self) -> &Arc<dyn GpuResources>;

    /// What presenting to a window can do on the selected device. A headless
    /// backend reports it too, since it describes the device.
    fn present_capabilities(&self) -> &PresentCapabilities;

    /// Returns whether this backend was created without a window.
    ///
    /// A headless backend can never render to a window.
    fn is_headless(&self) -> bool;

    /// Returns whether this backend is suspended: created with a window, but
    /// not currently able to render to one. Always `false` when headless.
    fn is_suspended(&self) -> bool;

    /// Releases the window and everything that depends on it, such as the
    /// window surface and swapchain, while keeping all other resources.
    ///
    /// Must be called when the window becomes unavailable, for example when
    /// the application is suspended on Android. Does nothing if already
    /// suspended, or if headless.
    fn suspend(&mut self);

    /// Starts rendering to `target` again after [`suspend`](Self::suspend).
    ///
    /// The window must be on the same display as the window the backend was
    /// created with. If the backend is not suspended, its current window is
    /// replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend is headless, if the selected device
    /// cannot present to the window, or if the window's resources (such as
    /// its surface and swapchain) cannot be created.
    fn resume(&mut self, target: WindowTarget) -> Result<()>;

    /// Records a new window size. Size-dependent resources, such as the
    /// swapchain, are recreated for it at the start of the next
    /// [`render_frame`](Self::render_frame), so a stream of resizes (as during
    /// an interactive resize) causes only one recreation per frame.
    ///
    /// Must be called whenever the window is resized. Does nothing if headless
    /// or suspended. A size with no area (for example, when minimised) pauses
    /// presenting until a later resize.
    fn resize(&mut self, size: WindowSize);

    /// Renders and presents one frame.
    ///
    /// Currently a stub: it only applies any pending [`resize`](Self::resize).
    /// Does nothing else if headless, suspended, or while the window has no
    /// area.
    ///
    /// # Errors
    ///
    /// Returns an error if size-dependent resources cannot be recreated.
    fn render_frame(&mut self) -> Result<()>;
}

/// Creates the render hardware interface for the given graphics API.
///
/// If `window` is `Some`, the backend renders to it, and keeps it alive until
/// it is suspended or dropped. If it is `None`, the backend runs headless.
/// `options` sets its session, device filter and validation.
///
/// # Errors
///
/// Returns an error if the backend fails to initialise, for example because
/// the graphics API is not supported on this system, if no device matches
/// the device filter, or if validation is required but not available.
pub fn create(
    api: GraphicsApi,
    window: Option<WindowTarget>,
    options: &BackendOptions,
) -> Result<Box<dyn RenderHardwareInterface>> {
    Ok(match api {
        GraphicsApi::Vulkan => Box::new(VulkanRenderHardwareInterface::new(window, options)?),
    })
}

#[cfg(test)]
mod tests {
    //! T0 checks of this module, which need no GPU.

    use super::*;

    /// Every supported graphics API.
    pub(super) const ALL_GRAPHICS_APIS: &[GraphicsApi] = &[GraphicsApi::Vulkan];

    // Fails to compile when a variant is added to `GraphicsApi`, as a reminder
    // to add it to `ALL_GRAPHICS_APIS`.
    const _: () = match GraphicsApi::Vulkan {
        GraphicsApi::Vulkan => (),
    };

    /// Parses `value`, panicking with the error if it's invalid.
    fn parse(value: &str) -> ValidationFeatures {
        value
            .parse()
            .unwrap_or_else(|err| panic!("{value:?}: {err:?}"))
    }

    /// `off` is the empty set.
    #[test]
    fn off_parses_as_no_features() {
        assert_eq!(parse("off"), ValidationFeatures::OFF);
        assert!(parse("off").is_off());
    }

    /// Every feature implies core validation, so `sync` alone means core and
    /// sync, the same as listing both.
    #[test]
    fn every_feature_implies_core() {
        for feature in ValidationFeature::ALL {
            let features = parse(feature.name());
            assert!(features.contains(ValidationFeature::Core), "{feature:?}");
            assert!(features.contains(feature), "{feature:?}");
        }
        assert_eq!(parse("sync"), parse("core,sync"));
        assert_eq!(parse("sync"), ValidationFeatures::SYNC);
    }

    /// A list enables each feature it names and no other, ignoring spaces
    /// and order.
    #[test]
    fn lists_enable_exactly_their_features() {
        let features = parse("best-practices, gpu");
        assert_eq!(
            features.iter().collect::<Vec<_>>(),
            [
                ValidationFeature::Core,
                ValidationFeature::Gpu,
                ValidationFeature::BestPractices,
            ],
        );
    }

    /// Unknown names, empty values and items, and `off` mixed with features
    /// are errors, rather than being ignored.
    #[test]
    fn invalid_features_are_errors() {
        for value in [
            "",
            "fast",
            "sync,",
            "sync,,gpu",
            "off,sync",
            "Sync",
            "best_practices",
        ] {
            assert!(
                value.parse::<ValidationFeatures>().is_err(),
                "{value:?} parsed"
            );
        }
    }

    /// The error names the value and lists what's valid.
    #[test]
    fn invalid_feature_errors_list_the_valid_names() {
        let err = "fast".parse::<ValidationFeatures>().unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid validation features \"fast\": expected off, or a comma-separated list of \
             core, sync, gpu, best-practices",
        );
    }

    /// Displaying a set gives a value that parses back to it.
    #[test]
    fn features_display_as_they_parse() {
        for value in ["off", "core", "core,sync", "core,sync,gpu,best-practices"] {
            assert_eq!(parse(value).to_string(), value);
        }
    }
}
