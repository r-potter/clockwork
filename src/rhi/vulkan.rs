//! Vulkan implementation of the render hardware interface.
//!
//! This backend is selected with [`GraphicsApi::Vulkan`] and is only
//! constructed through [`super::create`]; the rest of the application interacts
//! with it solely via the [`RenderHardwareInterface`] trait.
//!
//! The Vulkan loader is loaded at runtime rather than linked, so the
//! application still starts on systems without Vulkan; creating this backend
//! fails with an error instead.
//!
//! In builds with debug assertions enabled, the Khronos validation layer is
//! enabled and its messages are emitted as `tracing` events from this module.
//! See [`ENABLE_VALIDATION`] and [`debug_callback`].

mod device;
mod device_selection;
mod physical_device;
mod surface;
mod swapchain;

use std::ffi::{CStr, c_void};
use std::fmt;

use anyhow::{Context, Result, bail};
use ash::{Entry, Instance, ext::debug_utils, vk};
use raw_window_handle::RawDisplayHandle;
use tracing::{Level, debug, error, info, trace, warn};

use super::{GraphicsApi, RenderHardwareInterface, WindowSize, WindowTarget};
use device::LogicalDevice;
use device_selection::{Candidate, Selection};
use physical_device::PhysicalDevice;
use surface::Surface;
use swapchain::Swapchain;

/// Name reported to the Vulkan driver for both the application and the engine.
const NAME: &CStr = c"clockwork";

/// Vulkan version the backend targets.
///
/// `ash` 0.38 was generated from the Vulkan 1.3 headers and has no
/// `API_VERSION_1_4` constant, so the version is built by hand. Its bindings
/// also lack the functions and types added to core in 1.4; use the equivalent
/// extension bindings instead.
const API_VERSION: u32 = vk::make_api_version(0, 1, 4, 0);

/// Whether to enable the validation layer and a debug messenger.
///
/// On in builds with debug assertions (such as the `dev` profile) and off in
/// release builds. If the validation layer is not installed, a warning is
/// logged and the backend is created without it.
const ENABLE_VALIDATION: bool = cfg!(debug_assertions);

/// Khronos validation layer, shipped with the Vulkan SDK.
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Vulkan render hardware interface.
///
/// Owns the Vulkan instance, the selected physical device, the logical device
/// and, unless headless or suspended, the window surface and swapchain.
///
/// Fields are dropped in declaration order, so each is declared before the
/// objects it was created from.
pub struct VulkanRenderHardwareInterface {
    /// The window size from the latest [`resize`](RenderHardwareInterface::resize),
    /// if the swapchain has not been recreated for it yet. Applied at the
    /// start of the next frame.
    pending_window_size: Option<WindowSize>,
    /// `None` when headless, while suspended, or while the window has no area.
    swapchain: Option<Swapchain>,
    device: LogicalDevice,
    /// `None` when headless, or while suspended.
    surface: Option<Surface>,
    physical_device: PhysicalDevice,
    /// Declared last so that it is dropped last, after every object created
    /// from it.
    instance: VulkanInstance,
}

/// A Vulkan instance and, when validation is enabled, its debug messenger.
///
/// Both are destroyed when this is dropped.
struct VulkanInstance {
    /// Function pointers for the Vulkan loader. Must outlive `instance`, as it
    /// keeps the loader library loaded. Also used to create surfaces.
    entry: Entry,
    instance: Instance,
    /// Present only when validation is enabled and available.
    debug_messenger: Option<DebugMessenger>,
    /// Whether the surface extensions for a display were enabled, so that
    /// surfaces can be created. `false` when headless.
    supports_surfaces: bool,
}

/// A debug messenger and the extension functions needed to destroy it.
struct DebugMessenger {
    debug_utils: debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanRenderHardwareInterface {
    /// Initialises the Vulkan backend: creates an instance (see
    /// [`VulkanInstance::new`]) and a surface for the window if there is one,
    /// selects a physical device to render with (see [`device_selection`]),
    /// then creates a logical device and, if there is a window, a swapchain.
    ///
    /// If `window` is `None`, runs headless: no surface or swapchain is
    /// created, and no surface or swapchain extensions are enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if instance, surface, device or swapchain creation
    /// fails, if the physical devices cannot be queried, or if none meets the
    /// minimum requirements.
    pub fn new(window: Option<WindowTarget>) -> Result<Self> {
        let display_handle = window
            .as_ref()
            .map(|target| target.window.display_handle().map(|handle| handle.as_raw()))
            .transpose()
            .context("failed to get the window's display handle")?;
        let instance = VulkanInstance::new(display_handle)?;

        let window_size = window.as_ref().map(|target| target.size);
        let surface = window
            .map(|target| Surface::new(&instance.entry, &instance.instance, target.window))
            .transpose()?;

        let candidates = PhysicalDevice::enumerate(&instance.instance)?
            .into_iter()
            .map(|device| {
                info!(
                    "found {:?} {}, supporting Vulkan {}",
                    device.device_type(),
                    device.name(),
                    format_version(device.api_version()),
                );
                let surface_support = surface
                    .as_ref()
                    .map(|surface| surface.support(&device))
                    .transpose()?;
                Ok(Candidate {
                    device,
                    surface_support,
                })
            })
            .collect::<Result<_>>()?;
        let Selection {
            device: physical_device,
            queue_families,
        } = device_selection::select(candidates)?;

        let device = LogicalDevice::new(&instance.instance, &physical_device, queue_families)?;

        let mut rhi = Self {
            pending_window_size: None,
            swapchain: None,
            device,
            surface,
            physical_device,
            instance,
        };
        if let Some(size) = window_size {
            rhi.recreate_swapchain(size)?;
        }
        Ok(rhi)
    }

    /// Replaces the swapchain with one for the current surface, sized to fit
    /// `window_size`. Removes it if the surface has no area. Does nothing if
    /// there is no surface.
    ///
    /// # Errors
    ///
    /// Returns an error if swapchain creation fails.
    fn recreate_swapchain(&mut self, window_size: WindowSize) -> Result<()> {
        let Some(surface) = &self.surface else {
            return Ok(());
        };
        // Pass the old swapchain to the new one so that its resources can be
        // reused, then drop it.
        self.swapchain = Swapchain::new(
            &self.instance.instance,
            &self.device,
            &self.physical_device,
            surface,
            window_size,
            self.swapchain.as_ref(),
        )?;
        Ok(())
    }
}

impl VulkanInstance {
    /// Loads the Vulkan loader and creates an instance, with validation if
    /// [`ENABLE_VALIDATION`] is set.
    ///
    /// If `display_handle` is `Some`, the instance extensions needed to create
    /// surfaces for windows on that display are enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the Vulkan loader cannot be found, if it is older
    /// than [`API_VERSION`], if the display's window system is not supported,
    /// or if instance or debug messenger creation fails (for example, because
    /// no Vulkan driver is installed).
    fn new(display_handle: Option<RawDisplayHandle>) -> Result<Self> {
        // SAFETY: Loading the Vulkan loader runs its library initialisation
        // code, which we trust to be well behaved.
        let entry = unsafe { Entry::load() }.context("failed to load the Vulkan loader")?;

        // Requesting a newer version than the driver supports does not make
        // instance creation fail, so check the loader's version explicitly.
        // SAFETY: `entry` holds a valid, loaded Vulkan loader.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .context("failed to query the Vulkan loader version")?
            // Vulkan 1.0 loaders do not provide `vkEnumerateInstanceVersion`.
            .unwrap_or(vk::API_VERSION_1_0);
        if loader_version < API_VERSION {
            bail!(
                "Vulkan {} is required, but the Vulkan loader only supports {}",
                format_version(API_VERSION),
                format_version(loader_version),
            );
        }

        let validation = ENABLE_VALIDATION && validation_layer_available(&entry)?;

        let mut layers = Vec::new();
        let mut extensions = Vec::new();
        if validation {
            layers.push(VALIDATION_LAYER.as_ptr());
            // Provided by the validation layer.
            extensions.push(debug_utils::NAME.as_ptr());
        }
        if let Some(display_handle) = display_handle {
            let surface_extensions = ash_window::enumerate_required_extensions(display_handle)
                .context("the window system is not supported by the Vulkan backend")?;
            extensions.extend_from_slice(surface_extensions);
        }

        // Only ask Vulkan for messages at levels `tracing` will record, using
        // the same mapping as `debug_callback`. Errors are always requested, as
        // the severity mask must not be empty.
        let mut message_severity = vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
        if tracing::enabled!(Level::WARN) {
            message_severity |= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING;
        }
        if tracing::enabled!(Level::DEBUG) {
            message_severity |= vk::DebugUtilsMessageSeverityFlagsEXT::INFO;
        }
        if tracing::enabled!(Level::TRACE) {
            message_severity |= vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE;
        }

        // Which messages the debug messenger reports, and the callback that
        // reports them.
        let mut messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(message_severity)
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        let application_info = vk::ApplicationInfo::default()
            .application_name(NAME)
            .engine_name(NAME)
            .api_version(API_VERSION);

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        if validation {
            // Also report problems in instance creation and destruction, which
            // the messenger created below cannot cover.
            create_info = create_info.push_next(&mut messenger_info);
        }

        // SAFETY: `create_info` and the structures it points to are valid for
        // the duration of the call.
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .context("failed to create Vulkan instance")?;

        // Construct `Self` before creating the messenger, so that the instance
        // is destroyed by `Drop` if messenger creation fails.
        let mut vulkan_instance = Self {
            entry,
            instance,
            debug_messenger: None,
            supports_surfaces: display_handle.is_some(),
        };
        if validation {
            let debug_utils =
                debug_utils::Instance::new(&vulkan_instance.entry, &vulkan_instance.instance);
            // SAFETY: `messenger_info` is valid, and `VK_EXT_debug_utils` was
            // enabled on the instance.
            let messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&messenger_info, None) }
                    .context("failed to create Vulkan debug messenger")?;
            vulkan_instance.debug_messenger = Some(DebugMessenger {
                debug_utils,
                messenger,
            });
        }

        Ok(vulkan_instance)
    }
}

/// Returns whether the validation layer is installed, logging a warning if
/// it is not.
fn validation_layer_available(entry: &Entry) -> Result<bool> {
    // SAFETY: `entry` holds a valid, loaded Vulkan loader.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }
        .context("failed to enumerate Vulkan instance layers")?;
    let available = layers
        .iter()
        .any(|layer| layer.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
    if !available {
        warn!(
            "{} is not installed; continuing without Vulkan validation",
            VALIDATION_LAYER.to_string_lossy(),
        );
    }
    Ok(available)
}

/// Emits messages from the Vulkan debug messenger as `tracing` events.
///
/// Vulkan severities map to `tracing` levels as follows:
///
/// | Vulkan    | `tracing` |
/// |-----------|-----------|
/// | `ERROR`   | `ERROR`   |
/// | `WARNING` | `WARN`    |
/// | `INFO`    | `DEBUG`   |
/// | `VERBOSE` | `TRACE`   |
///
/// Vulkan's `INFO` messages are mostly loader and layer diagnostics, so they
/// are logged at `DEBUG` to keep them out of the default `INFO` output.
///
/// # Safety
///
/// Called only by the Vulkan implementation, which guarantees that
/// `callback_data` is null or points to valid data for the duration of the
/// call.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: Guaranteed by the caller; see the function's safety section.
    let message = unsafe {
        callback_data
            .as_ref()
            .and_then(|data| data.message_as_c_str())
    };
    let message = message.map_or("<no message>".into(), CStr::to_string_lossy);
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        error!(?types, "{message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!(?types, "{message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        debug!(?types, "{message}");
    } else {
        trace!(?types, "{message}");
    }
    // Returning `FALSE` lets the Vulkan call that triggered the message
    // continue, as the specification requires.
    vk::FALSE
}

/// Formats a Vulkan version number as `major.minor.patch`.
fn format_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version),
    )
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        if let Some(debug_messenger) = &self.debug_messenger {
            // SAFETY: The messenger was created by `new` from this instance and
            // is destroyed exactly once here, before the instance.
            unsafe {
                debug_messenger
                    .debug_utils
                    .destroy_debug_utils_messenger(debug_messenger.messenger, None);
            }
        }
        // SAFETY: The instance was created by `new`, is destroyed exactly once
        // here, and no objects created from it outlive it.
        unsafe { self.instance.destroy_instance(None) };
    }
}

impl fmt::Debug for VulkanRenderHardwareInterface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanRenderHardwareInterface")
            .field("instance", &self.instance.instance.handle())
            .field("validation", &self.instance.debug_messenger.is_some())
            .field("headless", &self.is_headless())
            .field("suspended", &self.is_suspended())
            .field("physical_device", &self.physical_device.name())
            .field("queue_families", &self.device.queue_families())
            .field(
                "swapchain",
                &self.swapchain.as_ref().map(|swapchain| {
                    let extent = swapchain.extent();
                    format!(
                        "{}x{} {:?}",
                        extent.width,
                        extent.height,
                        swapchain.format().format,
                    )
                }),
            )
            .finish_non_exhaustive()
    }
}

impl RenderHardwareInterface for VulkanRenderHardwareInterface {
    /// Always [`GraphicsApi::Vulkan`].
    fn api(&self) -> GraphicsApi {
        GraphicsApi::Vulkan
    }

    fn is_headless(&self) -> bool {
        !self.instance.supports_surfaces
    }

    fn is_suspended(&self) -> bool {
        !self.is_headless() && self.surface.is_none()
    }

    fn suspend(&mut self) {
        // `resume` provides the window's size at that point.
        self.pending_window_size = None;
        // The swapchain must be destroyed before its surface.
        self.swapchain = None;
        self.surface = None;
    }

    fn resume(&mut self, target: WindowTarget) -> Result<()> {
        if self.is_headless() {
            bail!("a headless Vulkan backend cannot render to a window");
        }
        // Destroy any existing swapchain and surface first, as some platforms
        // (such as Android) allow only one surface per window at a time.
        self.suspend();

        let surface = Surface::new(&self.instance.entry, &self.instance.instance, target.window)?;
        // The device was selected for the previous surface, so check that it
        // can still present to this one.
        let present_family = self
            .device
            .queue_families()
            .present
            .context("a windowed Vulkan backend has no present queue family")?;
        if !surface.can_present(self.physical_device.handle, present_family)? {
            bail!(
                "{} cannot present to the resumed window",
                self.physical_device.name(),
            );
        }
        self.surface = Some(surface);

        self.recreate_swapchain(target.size)
    }

    fn resize(&mut self, size: WindowSize) {
        if self.surface.is_some() {
            self.pending_window_size = Some(size);
        }
    }

    fn render_frame(&mut self) -> Result<()> {
        // Recreate the swapchain once for all resizes since the last frame.
        // Window systems may also report a resize without a change in size.
        if let Some(size) = self.pending_window_size.take() {
            let unchanged = self
                .swapchain
                .as_ref()
                .is_some_and(|swapchain| swapchain.window_size() == size);
            if !unchanged {
                self.recreate_swapchain(size)?;
            }
        }

        // TODO: acquire a swapchain image, record and submit rendering work,
        // and present.
        Ok(())
    }
}
