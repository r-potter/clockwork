//! What a plan is compiled for (design 0003, "Compiling" and "API"): the
//! output's format, size and accesses, the queue each kind runs on, and the
//! choice of schedule.

use clockwork_rhi::{Access, AccessSet, Extent2d, Format, QueueId};

/// What a plan is compiled for. A plan is compiled again when any of it
/// changes.
#[derive(Debug, Clone, PartialEq)]
pub struct CompileInputs {
    /// The texture the frame renders to.
    pub output: OutputDesc,
    /// The queue each kind of pass runs on, which kinds sharing a queue
    /// share.
    pub queues: QueueLayout,
    /// How the order passes run in is chosen.
    pub schedule: Schedule,
}

/// The texture a frame renders to: its format and size, and the accesses
/// it's in before the frame and must be left in after it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputDesc {
    /// Its texel format.
    pub format: Format,
    /// Its size, in texels.
    pub extent: Extent2d,
    /// The accesses it's in when the frame starts.
    pub before: AccessSet,
    /// The accesses it must be left in when the frame ends.
    pub after: AccessSet,
}

impl OutputDesc {
    /// A surface image of `format` and `extent`: acquired before the frame,
    /// and presented after it.
    pub fn surface(format: Format, extent: Extent2d) -> Self {
        Self {
            format,
            extent,
            before: Access::Acquired.into(),
            after: Access::Present.into(),
        }
    }
}

/// The queue each kind of pass runs on, from [`Queue::id`]. Kinds that
/// share a queue have the same ID.
///
/// [`Queue::id`]: clockwork_rhi::Queue::id
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueueLayout {
    /// The graphics queue.
    pub graphics: QueueId,
    /// The queue compute passes run on.
    pub compute: QueueId,
    /// The queue transfer passes run on.
    pub transfer: QueueId,
}

impl QueueLayout {
    /// Every kind on `queue`, as on a device with one queue family.
    pub const fn shared(queue: QueueId) -> Self {
        Self {
            graphics: queue,
            compute: queue,
            transfer: queue,
        }
    }
}

/// How the compiler chooses the order passes run in. Every schedule is a
/// topological order of the dependency graph the written order defines.
///
/// `Overlap`, which overlaps work across queues and groups independent
/// passes, arrives with the queues it's for, and then becomes the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Schedule {
    /// The written order, each pass in a group of its own.
    #[default]
    AsWritten,
}
