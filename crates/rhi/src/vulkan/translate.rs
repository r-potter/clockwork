//! Translating command lists into Vulkan command buffers (design 0002,
//! "Command lists and translation").
//!
//! Translation translates, and nothing more: each command becomes the
//! Vulkan commands it stands for, in order, with no merging, reordering or
//! removal (design 0002, "Keeping the recording model reversible"). It
//! looks up every handle the list names, holding each table's read lock
//! once for the whole list, and panics on a stale one, and on a subresource
//! outside its texture.
//!
//! Lock order: the tables' read locks are taken while a command pool is
//! checked out, never the other way around, and the queue's lock is never
//! taken. Holding the buffer and texture tables' read locks together can't
//! deadlock, since nothing takes either's write lock while holding the
//! other.

#[cfg(test)]
mod gpu_tests;

use std::fmt;
use std::sync::Arc;

use ash::vk;

use super::VulkanDevice;
use super::access::{AccessInfo, access_info};
use super::command_pools::PooledCommands;
use super::device_selection::QueueFamilies;
use super::error::error;
use crate::barrier::BarrierTarget;
use crate::command_list::Command;
use crate::copy::check_texture_copy;
use crate::deferred::Translation;
use crate::ffi::truncate_to_cstring;
use crate::{Access, CommandBuffer, CommandList, GpuPoint, QueueKind, RhiResult};

/// A command list translated for one queue, holding the command pool its
/// command buffer was recorded from, and holding back the destruction of
/// what it may use, until it's submitted or dropped.
pub(crate) struct VulkanCommandBuffer {
    /// The core, which owns the pool, so it outlives this.
    core: Arc<VulkanDevice>,
    /// The kind of the queue it was translated for, as that queue's
    /// [`Queue::kind`](crate::Queue::kind) reports.
    kind: QueueKind,
    /// The pool and its command buffer, or `None` once submitted.
    commands: Option<PooledCommands>,
    /// Its translation's epoch, given back to the core's deferred queue
    /// when it's submitted or dropped, or `None` once it has been.
    translation: Option<Translation>,
    /// The command list's label.
    label: Box<str>,
}

impl VulkanCommandBuffer {
    /// The recorded command buffer.
    pub(super) fn buffer(&self) -> vk::CommandBuffer {
        self.commands
            .expect("a command buffer is submitted only once")
            .buffer
    }

    /// Checks that it was translated for the queue of the kind `kind`.
    ///
    /// # Panics
    ///
    /// Panics, naming the command list, if it was translated for another.
    pub(super) fn check_queue(&self, kind: QueueKind) {
        assert!(
            self.kind == kind,
            "command list \"{}\" was translated for the {} queue, so it can't be submitted to \
             the {} queue",
            self.label,
            self.kind.name(),
            kind.name(),
        );
    }

    /// Gives its pool back to its queue, reusable once the timeline reaches
    /// `value`, which the submission it was in signals, and its translation
    /// back to the deferred queue, which keeps what it may use until then.
    pub(super) fn submitted(&mut self, value: u64) {
        if let Some(commands) = self.commands.take() {
            self.core.queues.give_back_pool(self.kind, commands, value);
        }
        if let Some(translation) = self.translation.take() {
            self.core
                .end_translation(translation, Some(GpuPoint::new(self.kind, value)));
        }
    }
}

impl Drop for VulkanCommandBuffer {
    /// Gives an unsubmitted command buffer's pool back to its queue, for
    /// reuse straight away, and its translation back to the deferred queue,
    /// since no submission uses either.
    fn drop(&mut self) {
        if let Some(commands) = self.commands.take() {
            self.core.queues.give_back_pool(self.kind, commands, 0);
        }
        if let Some(translation) = self.translation.take() {
            self.core.end_translation(translation, None);
        }
    }
}

impl fmt::Debug for VulkanCommandBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandBuffer")
            .field("label", &self.label)
            .field("queue", &self.kind)
            .field("submitted", &self.commands.is_none())
            .finish()
    }
}

impl VulkanDevice {
    /// Translates `list` into a command buffer for the queue of the kind
    /// `kind`, which serves itself.
    ///
    /// # Errors
    ///
    /// Returns an error if a command pool can't be checked out, or the
    /// command buffer can't be begun or ended.
    ///
    /// # Panics
    ///
    /// Panics if `list` has a debug region open, or names a stale handle.
    pub(super) fn translate(
        self: &Arc<Self>,
        kind: QueueKind,
        list: &CommandList,
    ) -> RhiResult<CommandBuffer> {
        list.check_complete();
        let device = self.device.handle();
        let commands = self.queues.check_out_pool(device, kind)?;
        // From here, returning early or panicking gives the pool and the
        // translation back. The translation starts before any handle is
        // looked up, so an object destroyed after its lookup is retired in
        // this epoch or later, and kept while this is outstanding.
        let command_buffer = VulkanCommandBuffer {
            core: self.clone(),
            kind,
            commands: Some(commands),
            translation: Some(self.begin_translation()),
            label: list.label().into(),
        };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: The command buffer was allocated on `device` from a pool
        // that's checked out, so no other thread uses it, and it's in the
        // initial state, since its pool was just created or reset.
        unsafe { device.begin_command_buffer(commands.buffer, &begin_info) }
            .map_err(|result| error(result, "begin the command buffer for", list.label()))?;
        // SAFETY: The command buffer is recording, for this queue's family.
        unsafe { self.record(commands.buffer, kind, list) };
        // SAFETY: The command buffer is recording.
        unsafe { device.end_command_buffer(commands.buffer) }
            .map_err(|result| error(result, "record the command buffer for", list.label()))?;
        self.device.set_object_name(commands.buffer, list.label());
        Ok(CommandBuffer(command_buffer))
    }

    /// Records `list`'s commands into `buffer`, for the queue of the kind
    /// `kind`.
    ///
    /// # Safety
    ///
    /// `buffer` must be recording, on this device, for that queue's family,
    /// and no other thread may use it.
    ///
    /// # Panics
    ///
    /// Panics if `list` names a stale handle, or has a queue transfer
    /// neither from nor to that queue.
    unsafe fn record(&self, buffer: vk::CommandBuffer, kind: QueueKind, list: &CommandList) {
        let device = self.device.handle();
        let families = self.device.queue_families();
        let buffers = self.lock_buffers_for_reading();
        let textures = self.lock_textures_for_reading();
        // Reused for each barrier command, so recording them doesn't allocate
        // once they've grown.
        let mut memory_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();
        let mut image_barriers = Vec::new();
        for command in list.commands() {
            match command {
                Command::Barrier { barriers } => {
                    memory_barriers.clear();
                    buffer_barriers.clear();
                    image_barriers.clear();
                    for barrier in list.barriers(barriers) {
                        let mut before = access_info(barrier.before());
                        let mut after = access_info(barrier.after());
                        let transfer =
                            queue_transfer(families, kind, barrier.queue_transfer(), list.label());
                        // A release's later accesses, and an acquire's
                        // earlier ones, happen on the other queue, so each
                        // half orders only its own queue's.
                        let (src_family, dst_family) = match transfer {
                            Transfer::None => (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                            Transfer::Release { from, to } => {
                                after.stages = vk::PipelineStageFlags2::NONE;
                                after.access = vk::AccessFlags2::NONE;
                                (from, to)
                            }
                            Transfer::Acquire { from, to } => {
                                before.stages = vk::PipelineStageFlags2::NONE;
                                before.access = vk::AccessFlags2::NONE;
                                (from, to)
                            }
                        };
                        match barrier.target() {
                            BarrierTarget::Global => memory_barriers.push(
                                vk::MemoryBarrier2::default()
                                    .src_stage_mask(before.stages)
                                    .src_access_mask(before.access)
                                    .dst_stage_mask(after.stages)
                                    .dst_access_mask(after.access),
                            ),
                            BarrierTarget::Buffer(target) => buffer_barriers.push(
                                vk::BufferMemoryBarrier2::default()
                                    .src_stage_mask(before.stages)
                                    .src_access_mask(before.access)
                                    .dst_stage_mask(after.stages)
                                    .dst_access_mask(after.access)
                                    .src_queue_family_index(src_family)
                                    .dst_queue_family_index(dst_family)
                                    .buffer(buffers.get(target.handle()).buffer())
                                    .offset(0)
                                    .size(vk::WHOLE_SIZE),
                            ),
                            BarrierTarget::Texture(target, subresource) => {
                                let (label, entry) = textures
                                    .lookup(target.handle())
                                    .unwrap_or_else(|err| panic!("{err}"));
                                let layout = |info: AccessInfo| {
                                    info.layout
                                        .expect("recording checked the access applies to textures")
                                };
                                image_barriers.push(
                                    vk::ImageMemoryBarrier2::default()
                                        .src_stage_mask(before.stages)
                                        .src_access_mask(before.access)
                                        .dst_stage_mask(after.stages)
                                        .dst_access_mask(after.access)
                                        .old_layout(layout(before))
                                        .new_layout(layout(after))
                                        .src_queue_family_index(src_family)
                                        .dst_queue_family_index(dst_family)
                                        .image(entry.image())
                                        .subresource_range(
                                            entry.subresource_range(label, subresource),
                                        ),
                                );
                            }
                        }
                    }
                    let dependency_info = vk::DependencyInfo::default()
                        .memory_barriers(&memory_barriers)
                        .buffer_memory_barriers(&buffer_barriers)
                        .image_memory_barriers(&image_barriers);
                    // SAFETY: Guaranteed by the caller, each buffer and
                    // image is live, since the tables are locked, each
                    // subresource range is inside its image, and
                    // `dependency_info` is valid for the call.
                    unsafe { device.cmd_pipeline_barrier2(buffer, &dependency_info) };
                }
                &Command::FillBuffer {
                    destination,
                    size,
                    value,
                } => {
                    let range = vk::DeviceAddressRangeKHR::default()
                        .address(destination.0)
                        .size(size);
                    // SAFETY: Guaranteed by the caller. Recording checked
                    // the range's alignment.
                    unsafe {
                        self.device.device_address_commands().cmd_fill_memory(
                            buffer,
                            &range,
                            vk::AddressCommandFlagsKHR::empty(),
                            value,
                        );
                    }
                }
                &Command::CopyBuffer {
                    source,
                    destination,
                    size,
                } => {
                    let region = vk::DeviceMemoryCopyKHR::default()
                        .src_range(
                            vk::DeviceAddressRangeKHR::default()
                                .address(source.0)
                                .size(size),
                        )
                        .dst_range(
                            vk::DeviceAddressRangeKHR::default()
                                .address(destination.0)
                                .size(size),
                        );
                    let copy_info = vk::CopyDeviceMemoryInfoKHR::default()
                        .regions(std::slice::from_ref(&region));
                    // SAFETY: Guaranteed by the caller, and `copy_info` is
                    // valid for the call. Recording checked that the ranges
                    // don't overlap.
                    unsafe {
                        self.device
                            .device_address_commands()
                            .cmd_copy_memory(buffer, &copy_info);
                    }
                }
                &(Command::CopyBufferToTexture { copy }
                | Command::CopyTextureToBuffer { copy }) => {
                    let to_texture = matches!(command, Command::CopyBufferToTexture { .. });
                    let copy = list.texture_copy(copy);
                    let region = &copy.texture;
                    let (label, entry) = textures
                        .lookup(region.texture.handle())
                        .unwrap_or_else(|err| panic!("{err}"));
                    let end = check_texture_copy(&entry.copied(label), region, &copy.layout);
                    let start = copy.buffer.0.checked_add(copy.layout.offset);
                    let start = start
                        .filter(|start| start.checked_add(end - copy.layout.offset).is_some())
                        .unwrap_or_else(|| {
                            panic!(
                                "command list \"{}\" copies texture \"{label}\" past the end of \
                                 the address space",
                                list.label(),
                            )
                        });
                    // The texture is in the layout its copy access needs,
                    // as the barriers before the copy put it.
                    let access = if to_texture {
                        Access::CopyWrite
                    } else {
                        Access::CopyRead
                    };
                    let layout = access_info(access)
                        .layout
                        .expect("copy accesses apply to textures");
                    let format = entry.format();
                    let texel =
                        |value: u32| i32::try_from(value).expect("texel offsets fit an i32");
                    let memory_copy = vk::DeviceMemoryImageCopyKHR::default()
                        .address_range(
                            vk::DeviceAddressRangeKHR::default()
                                .address(start)
                                .size(end - copy.layout.offset),
                        )
                        .address_row_length(
                            copy.layout.bytes_per_row / format.block_bytes * format.block_extent[0],
                        )
                        .address_image_height(copy.layout.rows_per_image * format.block_extent[1])
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: entry.aspect(),
                            mip_level: region.mip,
                            base_array_layer: region.layers.start,
                            layer_count: region.layers.end - region.layers.start,
                        })
                        .image_layout(layout)
                        .image_offset(vk::Offset3D {
                            x: texel(region.origin[0]),
                            y: texel(region.origin[1]),
                            z: texel(region.origin[2]),
                        })
                        .image_extent(vk::Extent3D {
                            width: region.extent.width,
                            height: region.extent.height,
                            depth: region.extent.depth,
                        });
                    let copy_info = vk::CopyDeviceMemoryImageInfoKHR::default()
                        .image(entry.image())
                        .regions(std::slice::from_ref(&memory_copy));
                    let device_address_commands = self.device.device_address_commands();
                    // SAFETY: Guaranteed by the caller, the image is live,
                    // since the table is locked, the copy passed its checks,
                    // so it's inside the image and laid out in whole
                    // blocks, and `copy_info` is valid for the call.
                    unsafe {
                        if to_texture {
                            device_address_commands.cmd_copy_memory_to_image(buffer, &copy_info);
                        } else {
                            device_address_commands.cmd_copy_image_to_memory(buffer, &copy_info);
                        }
                    }
                }
                Command::BeginDebugRegion { text } => {
                    if let Some(debug_utils) = self.device.debug_utils() {
                        let name = truncate_to_cstring(list.text(text));
                        let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
                        // SAFETY: Guaranteed by the caller, and `label` is
                        // valid for the call.
                        unsafe { debug_utils.cmd_begin_debug_utils_label(buffer, &label) };
                    }
                }
                Command::EndDebugRegion => {
                    if let Some(debug_utils) = self.device.debug_utils() {
                        // SAFETY: Guaranteed by the caller, and recording
                        // checked that a region is open.
                        unsafe { debug_utils.cmd_end_debug_utils_label(buffer) };
                    }
                }
                Command::InsertDebugLabel { text } => {
                    if let Some(debug_utils) = self.device.debug_utils() {
                        let name = truncate_to_cstring(list.text(text));
                        let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
                        // SAFETY: Guaranteed by the caller, and `label` is
                        // valid for the call.
                        unsafe { debug_utils.cmd_insert_debug_utils_label(buffer, &label) };
                    }
                }
            }
        }
    }
}

/// How a barrier's queue transfer is recorded on one queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transfer {
    /// No transfer: the barrier has none, or both kinds share a family.
    None,
    /// The releasing half, on the queue ownership moves from.
    Release {
        /// The family ownership moves from.
        from: u32,
        /// The family ownership moves to.
        to: u32,
    },
    /// The acquiring half, on the queue ownership moves to.
    Acquire {
        /// The family ownership moves from.
        from: u32,
        /// The family ownership moves to.
        to: u32,
    },
}

/// How `transfer`, a barrier's queue transfer, is recorded in the command
/// list `list` translated for the queue of the kind `translating`, on a
/// device whose queues are from `families`. Pure, so it's tested without a
/// GPU.
///
/// # Panics
///
/// Panics, naming the list, if the transfer is between two families and
/// the list is translated for neither.
fn queue_transfer(
    families: QueueFamilies,
    translating: QueueKind,
    transfer: Option<(QueueKind, QueueKind)>,
    list: &str,
) -> Transfer {
    let Some((from, to)) = transfer else {
        return Transfer::None;
    };
    let (from_family, to_family) = (families.family(from), families.family(to));
    let family = families.family(translating);
    if from_family == to_family {
        Transfer::None
    } else if family == from_family {
        Transfer::Release {
            from: from_family,
            to: to_family,
        }
    } else if family == to_family {
        Transfer::Acquire {
            from: from_family,
            to: to_family,
        }
    } else {
        panic!(
            "command list \"{list}\" has a queue transfer from the {} queue to the {} queue, but \
             is translated for the {} queue",
            from.name(),
            to.name(),
            translating.name(),
        );
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of recording queue transfers (design 0002,
    //! "Synchronization").

    use super::*;

    /// A device with graphics, compute and transfer families of their own.
    const DEDICATED: QueueFamilies = QueueFamilies {
        graphics: 0,
        compute: 2,
        transfer: 1,
        present: None,
    };

    /// A device whose kinds all share one family, as lavapipe's do.
    const SHARED: QueueFamilies = QueueFamilies {
        graphics: 0,
        compute: 0,
        transfer: 0,
        present: None,
    };

    /// From transfers to graphics.
    const UPLOAD: Option<(QueueKind, QueueKind)> = Some((QueueKind::Transfer, QueueKind::Graphics));

    /// A transfer between families is a release on the queue it's from, and
    /// an acquire on the queue it's to, naming both families.
    #[test]
    fn transfers_between_families_release_and_acquire() {
        assert_eq!(
            queue_transfer(DEDICATED, QueueKind::Transfer, UPLOAD, "upload"),
            Transfer::Release { from: 1, to: 0 },
        );
        assert_eq!(
            queue_transfer(DEDICATED, QueueKind::Graphics, UPLOAD, "upload"),
            Transfer::Acquire { from: 1, to: 0 },
        );
    }

    /// A transfer between kinds that share a family is dropped, on any of
    /// their queues, as is a barrier with no transfer.
    #[test]
    fn transfers_within_a_family_are_dropped() {
        for kind in QueueKind::ALL {
            assert_eq!(
                queue_transfer(SHARED, kind, UPLOAD, "upload"),
                Transfer::None
            );
            assert_eq!(
                queue_transfer(DEDICATED, kind, None, "upload"),
                Transfer::None
            );
        }
    }

    /// Translating a transfer for a queue it doesn't involve panics, naming
    /// the list and all three queues.
    #[test]
    #[should_panic(
        expected = "command list \"upload\" has a queue transfer from the transfer queue to the \
                    graphics queue, but is translated for the compute queue"
    )]
    fn transfers_for_another_queue_panic() {
        queue_transfer(DEDICATED, QueueKind::Compute, UPLOAD, "upload");
    }
}
