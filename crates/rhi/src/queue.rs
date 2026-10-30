//! Queues, and the points on their timelines that submissions reach (design
//! 0002, "Queues and submission").
//!
//! There's a [`Queue`] for each [`QueueKind`]. A kind with no dedicated
//! family is served by the queue of a more capable kind it shares one with,
//! whose kind [`Queue::kind`] reports. Each queue orders its work on a
//! timeline, and a submission returns the [`GpuPoint`] its completion
//! reaches, which other submissions, on any queue, and the CPU can wait for.

use std::fmt::Debug;
use std::sync::Arc;

use crate::{QueueKind, RhiResult};

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

/// What to submit to a queue.
///
/// Command buffers to submit arrive with translation. Until then, a
/// submission runs no commands, and reaches its point once everything it
/// waits for is reached.
#[derive(Debug, Clone, Copy, Default)]
pub struct SubmitDesc<'a> {
    /// Points, on any queue, that must be reached before the submission
    /// runs.
    pub wait_for: &'a [GpuPoint],
}

/// A queue that work is submitted to, one for each [`QueueKind`]. Shared by
/// every thread that submits to it, as an `Arc<dyn Queue>`.
pub trait Queue: Debug + Send + Sync {
    /// The kind of queue this is. A kind with no dedicated family is served
    /// by the queue of the kind it shares with, whose kind this returns, so
    /// `rhi.queue(QueueKind::Transfer).kind()` is `Transfer` only if
    /// transfers have a queue of their own.
    fn kind(&self) -> QueueKind;

    /// Submits work, which runs once every point in `desc.wait_for` is
    /// reached, and returns the point its completion reaches. Points are
    /// returned in submission order, from any thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the submission fails.
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
    assert_send_sync::<dyn Queue>();
    assert_send_sync::<Arc<dyn Queue>>();
};
