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
//! claimed, never the other way around, and the queue's lock is never
//! taken. The buffer table's read lock is always taken before the texture
//! table's, and translation is the only code that holds both. A fixed order
//! is what keeps them from deadlocking: the standard `RwLock` makes a new
//! reader wait behind a waiting writer, so two threads taking the read
//! locks in opposite orders could each wait on the other, with a writer
//! queued on each table.

#[cfg(test)]
mod gpu_tests;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use ash::vk;

use super::VulkanDevice;
use super::access::{Side, scope};
use super::command_pools::CommandPool;
use super::error::error;
use super::queues::Queues;
use crate::barrier::BarrierTarget;
use crate::command_list::Command;
use crate::ffi::truncate_to_cstring;
use crate::{AccessSet, CommandBuffer, CommandList, QueueId, RhiResult};

/// A command list translated for one queue, holding the command pool its
/// command buffer was recorded from until it's submitted or dropped.
pub(crate) struct VulkanCommandBuffer {
    /// The core, which owns the pool, so it outlives this.
    core: Arc<VulkanDevice>,
    /// The queue it was translated for.
    queue: QueueId,
    /// The pool and its command buffer, or `None` once submitted.
    commands: Option<CommandPool>,
    /// The command list's label.
    label: Box<str>,
}

impl VulkanCommandBuffer {
    /// The recorded command buffer.
    pub(super) fn buffer(&self) -> vk::CommandBuffer {
        self.commands
            .as_ref()
            .expect("a command buffer is submitted only once")
            .buffer()
    }

    /// Checks that it was translated for the queue `id`.
    ///
    /// # Panics
    ///
    /// Panics, naming the command list and its queue, if it was translated
    /// for another.
    pub(super) fn check_queue(&self, id: QueueId) {
        assert!(
            self.queue == id,
            "command list \"{}\" was translated for the {} queue, so it can't be submitted to \
             another queue",
            self.label,
            self.core.queues.name(self.queue),
        );
    }

    /// Recycles its pool to its queue, reusable once the timeline reaches
    /// `value`, which the submission it was in signals.
    pub(super) fn submitted(&mut self, value: u64) {
        if let Some(commands) = self.commands.take() {
            self.core.queues.recycle_pool(self.queue, commands, value);
            self.core
                .outstanding_command_buffers
                .fetch_sub(1, Ordering::Release);
        }
    }
}

impl Drop for VulkanCommandBuffer {
    /// Recycles an unsubmitted command buffer's pool to its queue, for
    /// reuse straight away, since no submission uses it.
    fn drop(&mut self) {
        if let Some(commands) = self.commands.take() {
            self.core.queues.recycle_pool(self.queue, commands, 0);
            self.core
                .outstanding_command_buffers
                .fetch_sub(1, Ordering::Release);
        }
    }
}

impl fmt::Debug for VulkanCommandBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandBuffer")
            .field("label", &self.label)
            .field("queue", &self.queue)
            .field("submitted", &self.commands.is_none())
            .finish()
    }
}

impl VulkanDevice {
    /// Translates `list` into a command buffer for the queue `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if a command pool can't be claimed, or the
    /// command buffer can't be begun or ended.
    ///
    /// # Panics
    ///
    /// Panics if `list` has a debug region open, or names a stale handle.
    pub(super) fn translate(
        self: &Arc<Self>,
        id: QueueId,
        list: &CommandList,
    ) -> RhiResult<CommandBuffer> {
        list.check_complete();
        let device = self.device.handle();
        let commands = self.queues.claim_pool(device, id)?;
        let buffer = commands.buffer();
        // From here, returning early or panicking recycles the pool, and
        // counts the command buffer as no longer outstanding.
        self.outstanding_command_buffers
            .fetch_add(1, Ordering::AcqRel);
        let command_buffer = VulkanCommandBuffer {
            core: self.clone(),
            queue: id,
            commands: Some(commands),
            label: list.label().into(),
        };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: The command buffer was allocated on `device` from a pool
        // that's claimed, so no other thread uses it, and claiming it put it
        // in the initial state.
        unsafe { device.begin_command_buffer(buffer, &begin_info) }
            .map_err(|result| error(result, "begin the command buffer for", list.label()))?;
        // SAFETY: The command buffer is recording, for this queue's family.
        unsafe { self.record(buffer, id, list) };
        // SAFETY: The command buffer is recording.
        unsafe { device.end_command_buffer(buffer) }
            .map_err(|result| error(result, "record the command buffer for", list.label()))?;
        self.device.set_object_name(buffer, list.label());
        Ok(CommandBuffer(command_buffer))
    }

    /// Records `list`'s commands into `buffer`, for the queue `id`.
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
    unsafe fn record(&self, buffer: vk::CommandBuffer, id: QueueId, list: &CommandList) {
        let device = self.device.handle();
        let buffers = self.lock_buffers_for_reading();
        // After the buffer table's, as the lock order above requires.
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
                        let mut before = scope(barrier.before(), Side::Before);
                        let mut after = scope(barrier.after(), Side::After);
                        let transfer = queue_transfer(
                            &self.queues,
                            id,
                            barrier.queue_transfer(),
                            list.label(),
                        );
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
                                let layout = |set: AccessSet| {
                                    vk::ImageLayout::from(
                                        set.layout()
                                            .expect("recording checked the set has one layout"),
                                    )
                                };
                                image_barriers.push(
                                    vk::ImageMemoryBarrier2::default()
                                        .src_stage_mask(before.stages)
                                        .src_access_mask(before.access)
                                        .dst_stage_mask(after.stages)
                                        .dst_access_mask(after.access)
                                        .old_layout(layout(barrier.before()))
                                        .new_layout(layout(barrier.after()))
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
    /// No transfer: the barrier has none, or both ends are on one family.
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

/// What [`queue_transfer`] needs to know about a device's queues, so that
/// its tests can describe a device without creating one.
trait QueueLookup {
    /// The family of the queue `id`.
    fn family(&self, id: QueueId) -> u32;

    /// The name of the queue `id`, for messages.
    fn name(&self, id: QueueId) -> &'static str;
}

impl QueueLookup for Queues {
    fn family(&self, id: QueueId) -> u32 {
        Queues::family(self, id)
    }

    fn name(&self, id: QueueId) -> &'static str {
        Queues::name(self, id)
    }
}

/// How `transfer`, a barrier's queue transfer, is recorded in the command
/// list `list` translated for the queue `translating`, on a device whose
/// queues are `queues`. Pure, so it's tested without a GPU.
///
/// # Panics
///
/// Panics, naming the list, if the transfer is between two families and
/// the list is translated for neither end's queue.
fn queue_transfer(
    queues: &impl QueueLookup,
    translating: QueueId,
    transfer: Option<(QueueId, QueueId)>,
    list: &str,
) -> Transfer {
    let Some((from, to)) = transfer else {
        return Transfer::None;
    };
    let (from_family, to_family) = (queues.family(from), queues.family(to));
    if from_family == to_family {
        Transfer::None
    } else if translating == from {
        Transfer::Release {
            from: from_family,
            to: to_family,
        }
    } else if translating == to {
        Transfer::Acquire {
            from: from_family,
            to: to_family,
        }
    } else {
        panic!(
            "command list \"{list}\" has a queue transfer from the {} queue to the {} queue, but \
             is translated for the {} queue",
            queues.name(from),
            queues.name(to),
            queues.name(translating),
        );
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of recording queue transfers (design 0002,
    //! "Synchronization").

    use super::*;

    /// A device's queues, each a family and a name, indexed by ID.
    struct FakeQueues(&'static [(u32, &'static str)]);

    impl QueueLookup for FakeQueues {
        fn family(&self, id: QueueId) -> u32 {
            self.0[id.index()].0
        }

        fn name(&self, id: QueueId) -> &'static str {
            self.0[id.index()].1
        }
    }

    /// A graphics queue.
    const GRAPHICS: QueueId = QueueId::new(0);

    /// A compute queue.
    const COMPUTE: QueueId = QueueId::new(1);

    /// A transfer queue.
    const TRANSFER: QueueId = QueueId::new(2);

    /// A device with graphics, compute and transfer queues on families of
    /// their own.
    const DEDICATED: FakeQueues = FakeQueues(&[(0, "graphics"), (2, "compute"), (1, "transfer")]);

    /// A device with a graphics and a compute queue on one family, which a
    /// later design may give a device.
    const ONE_FAMILY: FakeQueues = FakeQueues(&[(0, "graphics"), (0, "compute")]);

    /// From the transfer queue to the graphics queue.
    const UPLOAD: Option<(QueueId, QueueId)> = Some((TRANSFER, GRAPHICS));

    /// A transfer between families is a release on the queue it's from, and
    /// an acquire on the queue it's to, naming both families.
    #[test]
    fn transfers_between_families_release_and_acquire() {
        assert_eq!(
            queue_transfer(&DEDICATED, TRANSFER, UPLOAD, "upload"),
            Transfer::Release { from: 1, to: 0 },
        );
        assert_eq!(
            queue_transfer(&DEDICATED, GRAPHICS, UPLOAD, "upload"),
            Transfer::Acquire { from: 1, to: 0 },
        );
    }

    /// A transfer is dropped when both ends are one queue, as when every
    /// kind shares lavapipe's, or two queues on one family, as is a barrier
    /// with no transfer.
    #[test]
    fn transfers_within_a_family_are_dropped() {
        for queue in [GRAPHICS, COMPUTE] {
            for transfer in [(GRAPHICS, GRAPHICS), (COMPUTE, GRAPHICS)] {
                assert_eq!(
                    queue_transfer(&ONE_FAMILY, queue, Some(transfer), "upload"),
                    Transfer::None
                );
            }
        }
        for queue in [GRAPHICS, COMPUTE, TRANSFER] {
            assert_eq!(
                queue_transfer(&DEDICATED, queue, None, "upload"),
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
        queue_transfer(&DEDICATED, COMPUTE, UPLOAD, "upload");
    }
}
