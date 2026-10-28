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
//! Validation is enabled as [`BackendOptions::validation`] asks, through the
//! Khronos validation layer, and its messages are emitted as `tracing` events
//! on [`VALIDATION_TARGET`](clockwork_diagnostics::VALIDATION_TARGET), attributed
//! to [`BackendOptions::session`]. See the [`validation`] module.

mod buffers;
mod capabilities;
mod device;
mod device_selection;
mod error;
mod frames;
#[cfg(test)]
mod gpu_tests;
mod memory;
mod physical_device;
mod shaders;
mod surface;
mod swapchain;
#[cfg(test)]
pub(crate) mod test_support;
mod validation;

use std::ffi::CStr;
use std::sync::{Arc, PoisonError, RwLock};
use std::{fmt, slice};

use anyhow::{Context, Result, bail};
use ash::vk::TaggedStructure;
use ash::{Entry, Instance, ext::debug_utils, vk};
use raw_window_handle::RawDisplayHandle;
use tracing::{info, warn};

use super::{
    BackendOptions, Buffer, BufferDesc, Capabilities, GpuAddress, GpuResources, GraphicsApi,
    PresentCapabilities, RenderHardwareInterface, RhiResult, WindowSize, WindowTarget,
};
use crate::handle::HandleTable;
use buffers::BufferEntry;
use device::LogicalDevice;
use device_selection::{Candidate, Selection};
use frames::Frames;
use memory::MemoryAllocator;
use physical_device::PhysicalDevice;
use shaders::TriangleShaders;
use surface::Surface;
use swapchain::Swapchain;
use validation::{Availability, LayerSettings, VALIDATION_LAYER};

/// Name reported to the Vulkan driver for both the application and the engine.
const NAME: &CStr = c"clockwork";

/// Vulkan version the backend targets (ADR 0002).
const API_VERSION: u32 = vk::API_VERSION_1_4;

/// The colour each frame is cleared to before drawing, as linear RGBA.
/// Swapchain images use an sRGB format, so this appears brighter on screen
/// than the values suggest.
const CLEAR_COLOUR: [f32; 4] = [0.02, 0.05, 0.1, 1.0];

/// Vulkan render hardware interface.
///
/// Owns what belongs to the window: unless headless or suspended, the
/// window surface and swapchain. Also owns, until design 0002 moves them
/// out, the resources for frames in flight and the triangle's shader
/// objects. Everything else is in the shared [`VulkanDevice`], which it
/// hands out as the [`GpuResources`].
///
/// Fields are dropped in declaration order, so each is declared before the
/// objects it was created from, and the core, with the instance, last.
pub(crate) struct VulkanRenderHardwareInterface {
    /// The window's latest size, from creation, [`resume`] or [`resize`].
    /// `None` when headless or suspended.
    ///
    /// [`resume`]: RenderHardwareInterface::resume
    /// [`resize`]: RenderHardwareInterface::resize
    window_size: Option<WindowSize>,
    /// Set when acquiring or presenting reports that the swapchain no longer
    /// matches the surface, so that it is recreated at the start of the next
    /// frame.
    swapchain_out_of_date: bool,
    /// The command buffers and synchronization objects for each frame in
    /// flight.
    frames: Frames,
    /// The shader objects that draw the triangle.
    triangle_shaders: TriangleShaders,
    /// `None` when headless, while suspended, or while the window has no area.
    swapchain: Option<Swapchain>,
    /// `None` when headless, or while suspended.
    surface: Option<Surface>,
    /// What presenting can do on the selected device.
    present_capabilities: PresentCapabilities,
    /// `core`, as the resources the application shares between threads.
    resources: Arc<dyn GpuResources>,
    /// The instance and devices. Declared last so that this backend's
    /// reference to it is dropped last, after every object created from it.
    /// Holders of `resources` may keep it alive longer.
    core: Arc<VulkanDevice>,
}

/// The Vulkan backend's shared core (design 0002, "Ownership and
/// lifetimes"): the instance, the selected device and the logical device,
/// the memory allocator, and the resource tables. The
/// [`VulkanRenderHardwareInterface`] and every holder of its
/// [`GpuResources`] share it.
///
/// It's destroyed with its last `Arc`, on whichever thread drops it: it
/// waits for the device to be idle, destroys every resource left in its
/// tables, then drops its fields in declaration order, so each is declared
/// before the objects it was created from.
///
/// Its queues aren't locked yet, so only the owner thread submits to them,
/// through the [`VulkanRenderHardwareInterface`]. Submitting through a
/// `Queue` comes with the locks it needs, in `rhi-commands`.
pub(crate) struct VulkanDevice {
    /// Every live buffer.
    buffers: RwLock<HandleTable<BufferEntry>>,
    /// Allocates memory for resources on `device`.
    allocator: MemoryAllocator,
    /// The logical device, and its queues, created on `physical_device`.
    device: LogicalDevice,
    /// The physical device chosen by [`device_selection::select`].
    physical_device: PhysicalDevice,
    /// What `physical_device` can do beyond the requirements.
    capabilities: Capabilities,
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
    /// The instance, created from `entry`.
    instance: Instance,
    /// Present only when validation is enabled and available.
    debug_messenger: Option<DebugMessenger>,
    /// Whether the surface extensions for a display were enabled, so that
    /// surfaces can be created. `false` when headless.
    supports_surfaces: bool,
}

/// A debug messenger and the extension functions needed to destroy it.
struct DebugMessenger {
    /// The `VK_EXT_debug_utils` functions for the instance.
    debug_utils: debug_utils::Instance,
    /// The messenger, created with `debug_utils`.
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
    /// fails, if the physical devices cannot be queried, if none matches
    /// `options`' device filter and meets the minimum requirements, or if
    /// `options` requires validation that isn't available.
    pub(crate) fn new(window: Option<WindowTarget>, options: &BackendOptions) -> Result<Self> {
        let display_handle = window
            .as_ref()
            .map(|target| target.window.display_handle().map(|handle| handle.as_raw()))
            .transpose()
            .context("failed to get the window's display handle")?;
        let instance = VulkanInstance::new(display_handle, options)?;

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
            capabilities,
            present_capabilities,
        } = device_selection::select(candidates, options.device_filter.as_deref())?;

        let device = LogicalDevice::new(&instance.instance, &physical_device, queue_families)?;
        // Create these before the core takes the instance, so that if either
        // fails, the surface is still destroyed before the instance.
        let frames = Frames::new(&device)?;
        let triangle_shaders = TriangleShaders::new(&device)?;

        let allocator = MemoryAllocator::new(device.handle(), physical_device.memory_properties);
        let core = Arc::new(VulkanDevice {
            buffers: RwLock::new(HandleTable::new("buffer")),
            allocator,
            device,
            physical_device,
            capabilities,
            instance,
        });
        let mut rhi = Self {
            window_size,
            swapchain_out_of_date: false,
            frames,
            triangle_shaders,
            swapchain: None,
            surface,
            present_capabilities,
            resources: core.clone(),
            core,
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
            &self.core.instance.instance,
            &self.core.device,
            &self.core.physical_device,
            surface,
            window_size,
            self.swapchain.as_ref(),
        )?;
        self.swapchain_out_of_date = false;
        Ok(())
    }

    /// Acquires a swapchain image, clears it to [`CLEAR_COLOUR`], draws the
    /// triangle, and presents it, using the current frame's
    /// resources.
    ///
    /// Returns whether the swapchain should be recreated before the next
    /// frame. If it is out of date, nothing is rendered or presented.
    ///
    /// # Errors
    ///
    /// Returns an error if any Vulkan call fails, other than with
    /// `ERROR_OUT_OF_DATE_KHR`.
    fn draw_frame(&self, swapchain: &Swapchain) -> Result<bool> {
        let device = self.core.device.handle();
        let frame = self.frames.current();
        let present_queue = self
            .core
            .device
            .present_queue()
            .context("a windowed Vulkan backend has no present queue")?;

        // Every handle used below was created on `device`, or by `swapchain`
        // on `device`, and each info structure is valid for the duration of
        // its call. The `SAFETY` comments below rely on this.

        // Wait until the GPU has finished the last frame that used these
        // resources.
        // SAFETY: `frame.in_flight` is a fence created on `device`.
        unsafe { device.wait_for_fences(&[frame.in_flight], true, u64::MAX) }
            .context("failed to wait for Vulkan frame fence")?;

        // `image_acquired` is unsignalled: the submit that waited on it last
        // time has finished, as its fence has signalled.
        // SAFETY: `image_acquired` was created on `device`, and is
        // unsignalled with no pending operations, as above.
        let (image_index, mut out_of_date) =
            match unsafe { swapchain.acquire(frame.image_acquired) } {
                Ok(result) => result,
                // Leave the fence signalled, as nothing will be submitted to
                // signal it again.
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
                Err(err) => return Err(err).context("failed to acquire Vulkan swapchain image"),
            };
        let (image, image_view) = swapchain.image(image_index);

        // SAFETY: the fence was created on `device`, and has signalled, so no
        // queue submission still uses it.
        unsafe { device.reset_fences(&[frame.in_flight]) }
            .context("failed to reset Vulkan frame fence")?;
        // SAFETY: the fence wait above guarantees that the GPU has finished
        // with the command buffer.
        unsafe { frame.reset_commands(device) }?;

        let command_buffer = frame.command_buffer;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was allocated on `device` and has just
        // been reset, and `begin_info` is valid for the call.
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
            .context("failed to begin Vulkan command buffer")?;
        // SAFETY: the command buffer is recording, `image` and `image_view`
        // are a swapchain image of the swapchain's extent and its view, and
        // everything was created on `device`.
        unsafe {
            record_frame(
                device,
                command_buffer,
                image,
                image_view,
                swapchain.extent(),
                &self.triangle_shaders,
            )
        };
        // SAFETY: the command buffer is recording, on `device`.
        unsafe { device.end_command_buffer(command_buffer) }
            .context("failed to end Vulkan command buffer")?;

        // Wait for the image to be acquired before writing to it, and signal
        // when every command, including the final layout transition, is done.
        let wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.image_acquired)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(swapchain.render_finished(image_index))
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let command_buffer_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(slice::from_ref(&wait))
            .command_buffer_infos(slice::from_ref(&command_buffer_info))
            .signal_semaphore_infos(slice::from_ref(&signal));
        // SAFETY: the graphics queue, semaphores, command buffer and fence
        // were all created on `device`, the command buffer has ended, the
        // fence is unsignalled, and the info structures outlive the call.
        unsafe {
            device.queue_submit2(
                self.core.device.graphics_queue(),
                slice::from_ref(&submit_info),
                frame.in_flight,
            )
        }
        .context("failed to submit Vulkan frame")?;

        // SAFETY: the present queue was created on `device`, and its family
        // can present to the surface, as checked by device selection and on
        // resume. `image_index` was acquired above.
        match unsafe { swapchain.present(present_queue, image_index) } {
            Ok(suboptimal) => out_of_date |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => out_of_date = true,
            Err(err) => return Err(err).context("failed to present Vulkan swapchain image"),
        }
        Ok(out_of_date)
    }
}

/// Records commands into `command_buffer` that clear `image` to
/// [`CLEAR_COLOUR`], draw the triangle with `shaders`, and leave the image
/// ready to present.
///
/// # Safety
///
/// `command_buffer` must be recording, `image` and `image_view` must be a
/// swapchain image of size `extent` and a view of it, and `shaders` must
/// have been created on `device`, as must the others.
unsafe fn record_frame(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    image_view: vk::ImageView,
    extent: vk::Extent2D,
    shaders: &TriangleShaders,
) {
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    // Transition the image for rendering. Its previous contents are not
    // needed, so the old layout is `UNDEFINED`. The source stage matches the
    // stage the submit waits for the acquire semaphore at, so that the
    // transition happens after the image has been acquired.
    let to_attachment = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .image(image)
        .subresource_range(subresource_range);

    // Clearing happens when rendering begins, as the attachment's load
    // operation.
    let colour_attachment = vk::RenderingAttachmentInfo::default()
        .image_view(image_view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: CLEAR_COLOUR,
            },
        });
    let rendering_info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        })
        .layer_count(1)
        .color_attachments(slice::from_ref(&colour_attachment));

    // Transition the image for presenting, after rendering has written to it.
    // Presenting is ordered by the render-finished semaphore, so there is no
    // destination stage.
    let to_present = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::NONE)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(image)
        .subresource_range(subresource_range);

    // SAFETY: Guaranteed by the caller, and each info structure is valid for
    // the duration of its call.
    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&to_attachment)),
        );
        device.cmd_begin_rendering(command_buffer, &rendering_info);
        shaders.bind(command_buffer, extent);
        // Three vertices, generated by the vertex shader, and one instance.
        device.cmd_draw(command_buffer, 3, 1, 0, 0);
        device.cmd_end_rendering(command_buffer);
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&to_present)),
        );
    }
}

impl VulkanInstance {
    /// Loads the Vulkan loader and creates an instance, with the validation
    /// `options` asks for (see [`validation::decide`]), and a debug messenger
    /// whose messages are attributed to `options.session`.
    ///
    /// If `display_handle` is `Some`, the instance extensions needed to create
    /// surfaces for windows on that display are enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the Vulkan loader cannot be found, if it is older
    /// than [`API_VERSION`], if the display's window system is not supported,
    /// if validation is required but not available, or if instance or debug
    /// messenger creation fails (for example, because no Vulkan driver is
    /// installed).
    fn new(display_handle: Option<RawDisplayHandle>, options: &BackendOptions) -> Result<Self> {
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

        let plan = validation::decide(options.validation, Availability::query(&entry)?)?;
        if let Some(warning) = &plan.warning {
            warn!("{warning}");
        }
        let validate = !plan.features.is_off();
        if validate {
            info!("Vulkan validation: {}", validation::describe(&plan));
        } else {
            info!("Vulkan validation: off");
        }

        let mut layers = Vec::new();
        let mut extensions = Vec::new();
        if validate {
            layers.push(VALIDATION_LAYER.as_ptr());
            // Provided by the validation layer.
            extensions.push(debug_utils::NAME.as_ptr());
            if plan.use_settings {
                // Also provided by the validation layer.
                extensions.push(vk::EXT_LAYER_SETTINGS_NAME.as_ptr());
            }
        }
        if let Some(display_handle) = display_handle {
            let surface_extensions = ash_window::enumerate_required_extensions(display_handle)
                .context("the window system is not supported by the Vulkan backend")?;
            extensions.extend_from_slice(surface_extensions);
        }

        let mut messenger_create_info = validation::messenger_create_info(options.session);
        let layer_settings = LayerSettings::new(plan.features);
        let settings = layer_settings.settings();
        let mut layer_settings_create_info =
            vk::LayerSettingsCreateInfoEXT::default().settings(&settings);

        let application_info = vk::ApplicationInfo::default()
            .application_name(NAME)
            .engine_name(NAME)
            .api_version(API_VERSION);

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        if validate {
            // Also report problems in instance creation and destruction, which
            // the messenger created below cannot cover.
            create_info = create_info.push(&mut messenger_create_info);
        }
        if plan.use_settings {
            create_info = create_info.push(&mut layer_settings_create_info);
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
        if validate {
            let debug_utils =
                debug_utils::Instance::load(&vulkan_instance.entry, &vulkan_instance.instance);
            // SAFETY: `messenger_create_info` is valid, and `VK_EXT_debug_utils`
            // was enabled on the instance.
            let messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&messenger_create_info, None) }
                    .context("failed to create Vulkan debug messenger")?;
            vulkan_instance.debug_messenger = Some(DebugMessenger {
                debug_utils,
                messenger,
            });
        }

        Ok(vulkan_instance)
    }
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
            .field("instance", &self.core.instance.instance.handle())
            .field("validation", &self.core.instance.debug_messenger.is_some())
            .field("headless", &self.is_headless())
            .field("suspended", &self.is_suspended())
            .field("physical_device", &self.core.physical_device.name())
            .field(
                "capabilities",
                &crate::capabilities::summary(&self.core.capabilities, &self.present_capabilities),
            )
            .field("queue_families", &self.core.device.queue_families())
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

impl fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("physical_device", &self.physical_device.name())
            .field("capabilities", &self.capabilities)
            .field("queue_families", &self.device.queue_families())
            .finish_non_exhaustive()
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // Nothing useful can be done if this fails while dropping.
        // SAFETY: The device was created on this core's instance, and is
        // still alive.
        let _ = unsafe { self.device.handle().device_wait_idle() };
        let buffers = self
            .buffers
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .drain();
        for (_, entry) in buffers {
            // SAFETY: The device is idle, so no submitted work uses the
            // buffer.
            unsafe { self.free_buffer(entry) };
        }
    }
}

impl GpuResources for VulkanDevice {
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn create_buffer(&self, desc: &BufferDesc<'_>) -> RhiResult<Buffer> {
        self.create_buffer_entry(desc)
    }

    fn destroy_buffer(&self, buffer: Buffer) {
        self.destroy_buffer_entry(buffer);
    }

    fn buffer_address(&self, buffer: Buffer) -> GpuAddress {
        self.buffer_entry_address(buffer)
    }

    fn write_buffer(&self, buffer: Buffer, offset: u64, data: &[u8]) {
        self.write_buffer_entry(buffer, offset, data);
    }

    fn read_buffer(&self, buffer: Buffer, offset: u64, out: &mut [u8]) -> RhiResult<()> {
        self.read_buffer_entry(buffer, offset, out)
    }
}

impl RenderHardwareInterface for VulkanRenderHardwareInterface {
    /// Always [`GraphicsApi::Vulkan`].
    fn api(&self) -> GraphicsApi {
        GraphicsApi::Vulkan
    }

    fn resources(&self) -> &Arc<dyn GpuResources> {
        &self.resources
    }

    fn present_capabilities(&self) -> &PresentCapabilities {
        &self.present_capabilities
    }

    fn is_headless(&self) -> bool {
        !self.core.instance.supports_surfaces
    }

    fn is_suspended(&self) -> bool {
        !self.is_headless() && self.surface.is_none()
    }

    fn suspend(&mut self) {
        // `resume` provides the window's size at that point.
        self.window_size = None;
        self.swapchain_out_of_date = false;
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

        let surface = Surface::new(
            &self.core.instance.entry,
            &self.core.instance.instance,
            target.window,
        )?;
        // The device was selected for the previous surface, so check that it
        // can still present to this one.
        let present_family = self
            .core
            .device
            .queue_families()
            .present
            .context("a windowed Vulkan backend has no present queue family")?;
        if !surface.can_present(self.core.physical_device.handle, present_family)? {
            bail!(
                "{} cannot present to the resumed window",
                self.core.physical_device.name(),
            );
        }
        self.surface = Some(surface);
        self.window_size = Some(target.size);

        self.recreate_swapchain(target.size)
    }

    fn resize(&mut self, size: WindowSize) {
        if self.window_size.is_some() {
            self.window_size = Some(size);
        }
    }

    fn render_frame(&mut self) -> Result<()> {
        let Some(window_size) = self.window_size else {
            // Headless or suspended.
            return Ok(());
        };

        // Recreate the swapchain once for everything since the last frame: any
        // number of resizes, or an out-of-date result from acquiring or
        // presenting. Window systems may report a resize without a change in
        // size, so compare with the size the swapchain was created for.
        let size_changed = self
            .swapchain
            .as_ref()
            .is_none_or(|swapchain| swapchain.window_size() != window_size);
        if size_changed || self.swapchain_out_of_date {
            self.recreate_swapchain(window_size)?;
        }

        let Some(swapchain) = &self.swapchain else {
            // The window has no area, so there is nothing to render to.
            return Ok(());
        };
        let out_of_date = self.draw_frame(swapchain)?;
        self.swapchain_out_of_date = out_of_date;
        self.frames.advance();
        Ok(())
    }
}
