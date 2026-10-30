//! Queues, and the points on their timelines that submissions reach (design
//! 0002, "Queues and submission").
//!
//! There's a [`Queue`] for each [`QueueKind`]. A kind with no dedicated
//! family is served by the queue of a more capable kind it shares one with,
//! whose kind [`Queue::kind`] reports. Each queue orders its work on a
//! timeline, and a submission returns the [`GpuPoint`] its completion
//! reaches, which other submissions, on any queue, and the CPU can wait for.
//!
//! A queue translates [`CommandList`]s into [`CommandBuffer`]s for its
//! family, on any thread, which are then submitted to it in a chosen order.

use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::vulkan::VulkanCommandBuffer;
use crate::{CommandList, QueueKind, RhiResult};

/// A point on a queue's timeline, reached when a submission completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuPoint {
    /// The kind of the queue whose timeline it's on, as its
    /// [`Queue::kind`] reports.
    queue: QueueKind,
    /// The value its timeline reaches.
    value: u64,
}

impl GpuPoint {
    /// The point where `queue`'s timeline reaches `value`.
    pub(crate) fn new(queue: QueueKind, value: u64) -> Self {
        Self { queue, value }
    }

    /// The kind of the queue whose timeline it's on.
    pub fn queue(self) -> QueueKind {
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
    /// none, the submission reaches its point once everything it waits for
    /// is reached.
    pub command_buffers: Vec<CommandBuffer>,
    /// Points, on any queue, that must be reached before the submission
    /// runs.
    pub wait_for: &'a [GpuPoint],
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
    /// The kind of queue this is. A kind with no dedicated family is served
    /// by the queue of the kind it shares with, whose kind this returns, so
    /// `rhi.queue(QueueKind::Transfer).kind()` is `Transfer` only if
    /// transfers have a queue of their own.
    fn kind(&self) -> QueueKind;

    /// Translates `commands` into a command buffer for this queue. May be
    /// called on any thread, any number at once.
    ///
    /// # Errors
    ///
    /// Returns an error if a command buffer can't be allocated or recorded.
    ///
    /// # Panics
    ///
    /// Panics if `commands` names a stale handle, or has a debug region still
    /// open.
    fn translate(&self, commands: &CommandList) -> RhiResult<CommandBuffer>;

    /// Submits command buffers, which run once every point in
    /// `desc.wait_for` is reached, and returns the point their completion
    /// reaches. Points are returned in submission order, from any thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the submission fails.
    ///
    /// # Panics
    ///
    /// Panics if a command buffer was translated by another queue, or a
    /// point to wait for wasn't returned by its queue.
    fn submit(&self, desc: SubmitDesc<'_>) -> RhiResult<GpuPoint>;

    /// Blocks until `point` is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting fails.
    ///
    /// # Panics
    ///
    /// Panics if `point` isn't one this queue returned.
    fn wait(&self, point: GpuPoint) -> RhiResult<()>;

    /// Whether `point` has been reached, without blocking.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline can't be read.
    ///
    /// # Panics
    ///
    /// Panics if `point` isn't one this queue returned.
    fn is_complete(&self, point: GpuPoint) -> RhiResult<bool>;
}

// Points are plain data, and queues are shared between threads, as an
// `Arc<dyn Queue>` (design 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync + ?Sized>() {}
    assert_send_sync::<GpuPoint>();
    assert_send_sync::<SubmitDesc<'_>>();
    /// Compiles only for types that are `Send`.
    const fn assert_send<T: Send>() {}
    assert_send::<CommandBuffer>();
    assert_send_sync::<dyn Queue>();
    assert_send_sync::<Arc<dyn Queue>>();
};
