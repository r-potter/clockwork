//! The device's queues, their locks and their timelines (design 0002,
//! "Queues and submission").
//!
//! Each family a queue kind uses has one queue and one timeline semaphore,
//! which counts its submissions: the `n`th submission signals `n`. A kind
//! that shares a family with a more capable kind is served by that kind's
//! queue (see [`QueueFamilies::serving_kind`]), so a wait on one of its own
//! points is dropped, since the queue already orders its submissions.
//!
//! Vulkan requires a queue to be used by one thread at a time, so each is
//! behind a `Mutex`, held for each submission and present. Timeline values
//! are assigned under the same lock, so they increase in submission order.
//! Waiting and polling only read the semaphore, so they take no lock.

#[cfg(test)]
mod gpu_tests;

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::{Context, Result};
use ash::vk;
use ash::vk::TaggedStructure;

use super::VulkanDevice;
use super::device::LogicalDevice;
use super::device_selection::QueueFamilies;
use super::error::error;
use crate::{GpuPoint, Queue, QueueKind, RhiResult, SubmitDesc};

/// The queue of each family in use, with the timelines of those that queue
/// kinds use. Owned by the core, which destroys it with
/// [`destroy`](Self::destroy).
pub(super) struct Queues {
    /// The queue of each kind that serves itself, in the order of
    /// [`QueueKind::ALL`].
    queues: Vec<KindQueue>,
    /// The queue to present on, if its family isn't one a kind uses.
    present_only: Option<Mutex<vk::Queue>>,
    /// The families the queues are from.
    families: QueueFamilies,
}

/// A queue that a kind's work is submitted to, and its timeline.
struct KindQueue {
    /// The kind it serves as, which [`Queue::kind`] reports.
    kind: QueueKind,
    /// The queue, locked for each submission and present.
    queue: Mutex<vk::Queue>,
    /// A timeline semaphore, signalled with the number of each submission.
    timeline: vk::Semaphore,
    /// The number of the last submission, so the value its completion
    /// signals. Written only while `queue` is locked.
    submitted: AtomicU64,
}

impl Queues {
    /// Wraps `device`'s queues, creating a timeline for each kind that
    /// serves itself, named after the kind.
    ///
    /// # Errors
    ///
    /// Returns an error if a timeline semaphore can't be created.
    pub(super) fn new(device: &LogicalDevice) -> Result<Self> {
        let families = device.queue_families();
        let mut queues = Self {
            queues: Vec::new(),
            present_only: None,
            families,
        };
        for kind in QueueKind::ALL {
            if families.serving_kind(kind) != kind {
                continue;
            }
            let mut type_create_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let create_info = vk::SemaphoreCreateInfo::default().push(&mut type_create_info);
            // SAFETY: `create_info` is valid for the call, and timeline
            // semaphores are enabled on the device.
            let timeline = match unsafe { device.handle().create_semaphore(&create_info, None) } {
                Ok(timeline) => timeline,
                Err(result) => {
                    // SAFETY: Nothing has been submitted to wait on or
                    // signal the timelines created so far.
                    unsafe { queues.destroy(device.handle()) };
                    return Err(result).with_context(|| {
                        format!("failed to create the {} queue's timeline", kind.name())
                    });
                }
            };
            let queue = device.queue(families.family(kind));
            device.set_object_name(queue, kind.name());
            device.set_object_name(timeline, &format!("{} timeline", kind.name()));
            queues.queues.push(KindQueue {
                kind,
                queue: Mutex::new(queue),
                timeline,
                submitted: AtomicU64::new(0),
            });
        }
        if let Some(present) = families.present
            && !QueueKind::ALL
                .into_iter()
                .any(|kind| families.family(kind) == present)
        {
            let queue = device.queue(present);
            device.set_object_name(queue, "present");
            queues.present_only = Some(Mutex::new(queue));
        }
        Ok(queues)
    }

    /// The queue `kind`'s work is submitted to.
    fn get(&self, kind: QueueKind) -> &KindQueue {
        let serving = self.families.serving_kind(kind);
        self.queues
            .iter()
            .find(|queue| queue.kind == serving)
            .expect("every kind that serves itself has a timeline")
    }

    /// The kind whose queue `kind`'s work is submitted to, which
    /// [`Queue::kind`] reports.
    pub(super) fn serving_kind(&self, kind: QueueKind) -> QueueKind {
        self.families.serving_kind(kind)
    }

    /// Locks the queue `kind`'s work is submitted to, for one submission.
    pub(super) fn lock(&self, kind: QueueKind) -> MutexGuard<'_, vk::Queue> {
        lock(&self.get(kind).queue)
    }

    /// Locks the queue to present on, or returns `None` when headless.
    pub(super) fn lock_present(&self) -> Option<MutexGuard<'_, vk::Queue>> {
        let present = self.families.present?;
        let queue = match &self.present_only {
            Some(queue) => queue,
            None => {
                let kind = QueueKind::ALL
                    .into_iter()
                    .find(|&kind| self.families.family(kind) == present)
                    .expect("the present family is a kind's, or has its own queue");
                &self.get(kind).queue
            }
        };
        Some(lock(queue))
    }

    /// Submits `desc` to the queue `kind`'s work is submitted to, on
    /// `device`, and returns the point its completion reaches.
    ///
    /// # Errors
    ///
    /// Returns an error if the submission fails.
    ///
    /// # Panics
    ///
    /// Panics if a point to wait for wasn't returned by its queue.
    pub(super) fn submit(
        &self,
        device: &ash::Device,
        kind: QueueKind,
        desc: SubmitDesc<'_>,
    ) -> RhiResult<GpuPoint> {
        let queue = self.get(kind);
        let waits: Vec<_> = desc
            .wait_for
            .iter()
            .filter_map(|&point| {
                let waited = self.get(point.queue());
                waited.check_returned(point);
                // A queue runs its own submissions in order.
                (waited.kind != queue.kind).then(|| {
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(waited.timeline)
                        .value(point.value())
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                })
            })
            .collect();

        let locked = lock(&queue.queue);
        let value = queue.submitted.load(Ordering::Relaxed) + 1;
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(queue.timeline)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .signal_semaphore_infos(std::slice::from_ref(&signal));
        // SAFETY: The queue and semaphores were created on `device`, the
        // queue is locked, so no other thread uses it, the info structures
        // outlive the call, and `value` is greater than any value signalled
        // or waited for on this timeline so far.
        unsafe {
            device.queue_submit2(
                *locked,
                std::slice::from_ref(&submit_info),
                vk::Fence::null(),
            )
        }
        .map_err(|result| error(result, "submit to queue", queue.kind.name()))?;
        queue.submitted.store(value, Ordering::Release);
        Ok(GpuPoint::new(queue.kind, value))
    }

    /// Blocks until `point`, which the queue `kind`'s work is submitted to
    /// returned, is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting fails.
    ///
    /// # Panics
    ///
    /// Panics if that queue didn't return `point`.
    pub(super) fn wait(
        &self,
        device: &ash::Device,
        kind: QueueKind,
        point: GpuPoint,
    ) -> RhiResult<()> {
        let queue = self.get(kind);
        queue.check_own(point);
        let value = point.value();
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&queue.timeline))
            .values(std::slice::from_ref(&value));
        // SAFETY: The semaphore is a timeline created on `device`, and
        // `wait_info` is valid for the call. Waiting needs no lock.
        unsafe { device.wait_semaphores(&wait_info, u64::MAX) }
            .map_err(|result| error(result, "wait for queue", queue.kind.name()))
    }

    /// Whether `point`, which the queue `kind`'s work is submitted to
    /// returned, has been reached.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline can't be read.
    ///
    /// # Panics
    ///
    /// Panics if that queue didn't return `point`.
    pub(super) fn is_complete(
        &self,
        device: &ash::Device,
        kind: QueueKind,
        point: GpuPoint,
    ) -> RhiResult<bool> {
        let queue = self.get(kind);
        queue.check_own(point);
        // SAFETY: The semaphore is a timeline created on `device`.
        let reached = unsafe { device.get_semaphore_counter_value(queue.timeline) }
            .map_err(|result| error(result, "read the timeline of queue", queue.kind.name()))?;
        Ok(reached >= point.value())
    }

    /// Destroys the timelines.
    ///
    /// # Safety
    ///
    /// The timelines must have been created on `device`, and no pending
    /// submission may wait on or signal them, for example because the
    /// device is idle.
    pub(super) unsafe fn destroy(&mut self, device: &ash::Device) {
        for queue in self.queues.drain(..) {
            // SAFETY: Guaranteed by the caller.
            unsafe { device.destroy_semaphore(queue.timeline, None) };
        }
    }
}

impl KindQueue {
    /// Checks that `point` is one this queue returned.
    ///
    /// # Panics
    ///
    /// Panics if it isn't.
    fn check_own(&self, point: GpuPoint) {
        assert_eq!(
            point.queue(),
            self.kind,
            "{point:?} is on the {} queue's timeline, not the {} queue's",
            point.queue().name(),
            self.kind.name(),
        );
        self.check_returned(point);
    }

    /// Checks that `point`, which is on this queue's timeline, was returned
    /// by a submission.
    ///
    /// # Panics
    ///
    /// Panics if no submission returned it.
    fn check_returned(&self, point: GpuPoint) {
        let submitted = self.submitted.load(Ordering::Acquire);
        assert!(
            (1..=submitted).contains(&point.value()),
            "the {} queue never returned {point:?}: it has made {submitted} submissions",
            self.kind.name(),
        );
    }
}

/// Locks `queue`, recovering it if a thread panicked while holding it, which
/// leaves nothing inconsistent (design 0002, "Error handling").
fn lock(queue: &Mutex<vk::Queue>) -> MutexGuard<'_, vk::Queue> {
    queue.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A [`Queue`] of the Vulkan backend: a view of one of the core's queues.
pub(super) struct VulkanQueue {
    /// The core, whose queues this is a view of.
    core: Arc<VulkanDevice>,
    /// The kind this queue serves as.
    kind: QueueKind,
}

impl VulkanQueue {
    /// The queue on `core` that `kind`'s work is submitted to.
    pub(super) fn new(core: Arc<VulkanDevice>, kind: QueueKind) -> Self {
        let kind = core.queues.serving_kind(kind);
        Self { core, kind }
    }
}

impl Queue for VulkanQueue {
    fn kind(&self) -> QueueKind {
        self.kind
    }

    fn submit(&self, desc: SubmitDesc<'_>) -> RhiResult<GpuPoint> {
        self.core
            .queues
            .submit(self.core.device.handle(), self.kind, desc)
    }

    fn wait(&self, point: GpuPoint) -> RhiResult<()> {
        self.core
            .queues
            .wait(self.core.device.handle(), self.kind, point)
    }

    fn is_complete(&self, point: GpuPoint) -> RhiResult<bool> {
        self.core
            .queues
            .is_complete(self.core.device.handle(), self.kind, point)
    }
}

impl fmt::Debug for VulkanQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanQueue")
            .field("kind", &self.kind)
            .field(
                "family",
                &self.core.device.queue_families().family(self.kind),
            )
            .finish_non_exhaustive()
    }
}
