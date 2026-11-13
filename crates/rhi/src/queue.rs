//! Queues, and the timepoints on their timelines that submissions reach (design
//! 0002, "Queues and submission").
//!
//! There's a [`Queue`] for each [`QueueKind`]. A kind with no dedicated
//! family is served by the queue of a more capable kind it shares one with,
//! whose kind [`Queue::kind`] reports. Each queue has a [`QueueId`], which
//! says which queue it is, and orders its work on a timeline. A submission
//! returns the [`GpuTimepoint`] its completion reaches, which other
//! submissions, on any queue, and the CPU can wait for.
//!
//! A queue translates [`CommandList`]s into [`CommandBuffer`]s for its
//! family, on any thread, which are then submitted to it in a chosen order.

use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::vulkan::VulkanCommandBuffer;
use crate::{CommandList, QueueKind, RhiResult, SubmissionTimings, TransferGranularity};

/// Which of the backend's queues something is on, as [`Queue::id`] returns.
/// Kinds that share a queue share its ID.
///
/// A queue's kind says what it can run, not which queue it is, so timepoints
/// name queues by ID (design 0002, "Queues and submission").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueueId(u32);

impl QueueId {
    /// The ID of the backend's queue at `index`.
    pub(crate) const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The index of the backend's queue it names.
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A timepoint on a queue's timeline, reached when a submission completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuTimepoint {
    /// The queue whose timeline it's on.
    queue: QueueId,
    /// The value its timeline reaches.
    value: u64,
}

impl GpuTimepoint {
    /// The timepoint where `queue`'s timeline reaches `value`.
    pub(crate) fn new(queue: QueueId, value: u64) -> Self {
        Self { queue, value }
    }

    /// The queue whose timeline it's on.
    pub fn queue(self) -> QueueId {
        self.queue
    }

    /// The value its timeline reaches.
    pub(crate) fn value(self) -> u64 {
        self.value
    }
}

/// A command list translated for one queue, ready to submit to it.
///
/// Its command pool is returned to the queue when it's submitted, for reuse
/// once the submission completes, or straight away when it's dropped
/// unsubmitted.
pub struct CommandBuffer(pub(crate) VulkanCommandBuffer);

impl Debug for CommandBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

/// What to submit to a queue.
#[derive(Debug, Default)]
pub struct SubmitDesc<'a> {
    /// Command buffers translated by the queue, run in this order. With
    /// none, the submission reaches its timepoint once everything it waits for
    /// is reached.
    pub command_buffers: Vec<CommandBuffer>,
    /// Timepoints, on any queue, that must be reached before the submission
    /// runs.
    pub wait_for: &'a [GpuTimepoint],
}

impl SubmitDesc<'_> {
    /// A submission of `command_buffers`, waiting for nothing.
    pub fn new(command_buffers: Vec<CommandBuffer>) -> Self {
        Self {
            command_buffers,
            wait_for: &[],
        }
    }
}

/// A queue that work is submitted to, one for each [`QueueKind`]. Shared by
/// every thread that submits to it, as an `Arc<dyn Queue>`.
pub trait Queue: Debug + Send + Sync {
    /// Which queue this is. Two `Queue`s are the same queue exactly when
    /// their IDs are equal.
    fn id(&self) -> QueueId;

    /// The kind of queue this is. A kind with no dedicated family is served
    /// by the queue of the kind it shares with, whose kind this returns, so
    /// `rhi.queue(QueueKind::Transfer).kind()` is `Transfer` only if
    /// transfers have a queue of their own.
    fn kind(&self) -> QueueKind;

    /// How long one timestamp tick lasts on this queue, in nanoseconds, or
    /// `None` if it can't write timestamps (design 0002, "Timestamps").
    /// Device selection requires the graphics queue to write them.
    fn timestamp_period_ns(&self) -> Option<f32>;

    /// The texture regions this queue can copy (design 0002, "Texture
    /// copies"). Graphics and compute queues can copy any region, but a
    /// dedicated transfer queue may only copy coarser ones.
    fn transfer_granularity(&self) -> TransferGranularity;

    /// Translates `commands` into a command buffer for this queue. May be
    /// called on any thread, any number at once.
    ///
    /// # Errors
    ///
    /// Returns an error if a command buffer can't be allocated or recorded.
    ///
    /// # Panics
    ///
    /// Panics if `commands` names a stale handle, or has a debug region or
    /// timing scope still open.
    fn translate(&self, commands: &CommandList) -> RhiResult<CommandBuffer>;

    /// Submits command buffers, which run once every timepoint in
    /// `desc.wait_for` is reached, and returns the timepoint their completion
    /// reaches. Timepoints are returned in submission order, from any thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the submission fails.
    ///
    /// # Panics
    ///
    /// Panics if a command buffer was translated by another queue, or a
    /// timepoint to wait for wasn't returned by its queue.
    fn submit(&self, desc: SubmitDesc<'_>) -> RhiResult<GpuTimepoint>;

    /// Blocks until `timepoint` is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting fails.
    ///
    /// # Panics
    ///
    /// Panics if `timepoint` isn't one this queue returned.
    fn wait(&self, timepoint: GpuTimepoint) -> RhiResult<()>;

    /// Whether `timepoint` has been reached, without blocking.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline can't be read.
    ///
    /// # Panics
    ///
    /// Panics if `timepoint` isn't one this queue returned.
    fn is_complete(&self, timepoint: GpuTimepoint) -> RhiResult<bool>;

    /// The timings recorded in the submission that reaches `timepoint`, or
    /// `None` until it's reached, without blocking (design 0002,
    /// "Timestamps").
    ///
    /// Each submission's timings are returned once. A reached submission
    /// whose timings were already taken, or discarded because more than 64
    /// completed submissions with timings were left untaken, returns no
    /// command buffers, as does one that recorded no timings.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline or the results can't be read.
    ///
    /// # Panics
    ///
    /// Panics if `timepoint` isn't one this queue returned.
    fn take_timings(&self, timepoint: GpuTimepoint) -> RhiResult<Option<SubmissionTimings>>;
}

// Timepoints are plain data, and queues are shared between threads, as an
// `Arc<dyn Queue>` (design 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync + ?Sized>() {}
    assert_send_sync::<QueueId>();
    assert_send_sync::<GpuTimepoint>();
    assert_send_sync::<SubmitDesc<'_>>();
    /// Compiles only for types that are `Send`.
    const fn assert_send<T: Send>() {}
    assert_send::<CommandBuffer>();
    assert_send_sync::<dyn Queue>();
    assert_send_sync::<Arc<dyn Queue>>();
};
