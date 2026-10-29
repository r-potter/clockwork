//! Translating command lists into Vulkan command buffers (design 0002,
//! "Command lists and translation").
//!
//! Translation translates, and nothing more: each command becomes the
//! Vulkan commands it stands for, in order, with no merging, reordering or
//! removal (design 0002, "Keeping the recording model reversible"). It
//! looks up every handle the list names, holding each table's read lock
//! once for the whole list, and panics on a stale one.
//!
//! Lock order: a table's read lock is taken while a command pool is checked
//! out, never the other way around, and the queue's lock is never taken.

#[cfg(test)]
mod gpu_tests;

use std::fmt;
use std::sync::Arc;

use ash::vk;

use super::VulkanDevice;
use super::access::access_info;
use super::command_pools::PooledCommands;
use super::error::error;
use crate::barrier::BarrierTarget;
use crate::command_list::Command;
use crate::ffi::truncate_to_cstring;
use crate::{CommandBuffer, CommandList, QueueKind, RhiResult};

/// A command list translated for one queue, holding the command pool its
/// command buffer was recorded from until it's submitted or dropped.
pub(crate) struct VulkanCommandBuffer {
    /// The core, which owns the pool, so it outlives this.
    core: Arc<VulkanDevice>,
    /// The kind of the queue it was translated for, as that queue's
    /// [`Queue::kind`](crate::Queue::kind) reports.
    kind: QueueKind,
    /// The pool and its command buffer, or `None` once submitted.
    commands: Option<PooledCommands>,
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
    /// `value`, which the submission it was in signals.
    pub(super) fn submitted(&mut self, value: u64) {
        if let Some(commands) = self.commands.take() {
            self.core.queues.give_back_pool(self.kind, commands, value);
        }
    }
}

impl Drop for VulkanCommandBuffer {
    /// Gives an unsubmitted command buffer's pool back to its queue, for
    /// reuse straight away, since no submission uses it.
    fn drop(&mut self) {
        if let Some(commands) = self.commands.take() {
            self.core.queues.give_back_pool(self.kind, commands, 0);
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
    /// Panics if `list` has a debug label open, or names a stale handle.
    pub(super) fn translate(
        self: &Arc<Self>,
        kind: QueueKind,
        list: &CommandList,
    ) -> RhiResult<CommandBuffer> {
        list.check_complete();
        let device = self.device.handle();
        let commands = self.queues.check_out_pool(device, kind)?;
        // From here, returning early or panicking gives the pool back.
        let command_buffer = VulkanCommandBuffer {
            core: self.clone(),
            kind,
            commands: Some(commands),
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
        unsafe { self.record(commands.buffer, list) };
        // SAFETY: The command buffer is recording.
        unsafe { device.end_command_buffer(commands.buffer) }
            .map_err(|result| error(result, "record the command buffer for", list.label()))?;
        self.device.set_object_name(commands.buffer, list.label());
        Ok(CommandBuffer(command_buffer))
    }

    /// Records `list`'s commands into `buffer`.
    ///
    /// # Safety
    ///
    /// `buffer` must be recording, on this device, and no other thread may
    /// use it.
    ///
    /// # Panics
    ///
    /// Panics if `list` names a stale handle.
    unsafe fn record(&self, buffer: vk::CommandBuffer, list: &CommandList) {
        let device = self.device.handle();
        let buffers = self.lock_buffers_for_reading();
        // Reused for each barrier command, so recording them doesn't allocate
        // once they've grown.
        let mut memory_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();
        for command in list.commands() {
            match command {
                Command::Barrier { barriers } => {
                    memory_barriers.clear();
                    buffer_barriers.clear();
                    for barrier in list.barriers(barriers) {
                        let before = access_info(barrier.before());
                        let after = access_info(barrier.after());
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
                                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .buffer(buffers.get(target.handle()).buffer())
                                    .offset(0)
                                    .size(vk::WHOLE_SIZE),
                            ),
                            BarrierTarget::Texture(..) => {
                                unimplemented!("texture barriers are translated from the next PR")
                            }
                        }
                    }
                    let dependency_info = vk::DependencyInfo::default()
                        .memory_barriers(&memory_barriers)
                        .buffer_memory_barriers(&buffer_barriers);
                    // SAFETY: Guaranteed by the caller, each buffer is live,
                    // since the table is locked, and `dependency_info` is
                    // valid for the call.
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
                Command::BeginLabel { text } => {
                    if let Some(debug_utils) = self.device.debug_utils() {
                        let name = truncate_to_cstring(list.text(text));
                        let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
                        // SAFETY: Guaranteed by the caller, and `label` is
                        // valid for the call.
                        unsafe { debug_utils.cmd_begin_debug_utils_label(buffer, &label) };
                    }
                }
                Command::EndLabel => {
                    if let Some(debug_utils) = self.device.debug_utils() {
                        // SAFETY: Guaranteed by the caller, and recording
                        // checked that a label is open.
                        unsafe { debug_utils.cmd_end_debug_utils_label(buffer) };
                    }
                }
            }
        }
    }
}
