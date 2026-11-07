//! T1 tests of queues, submission and timelines on a real device (design
//! 0002, "Queues and submission", part of R9).

use std::collections::BTreeSet;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

use ash::vk;
use ash::vk::TaggedStructure;
use clockwork_diagnostics::capture::TestSession;

use super::lock;
use crate::vulkan::test_support::VulkanTestBackend;
use crate::{GpuTimepoint, QueueId, QueueKind, RenderHardwareInterface, SubmitDesc};

/// A submission held back by a gate, a timeline semaphore only the host
/// signals, so the timepoint it reaches isn't reached until the gate opens.
///
/// Opens the gate when dropped, if the test hasn't, and waits for the
/// submission, so that a failing test doesn't leave the device unable to go
/// idle.
struct HeldSubmission<'a> {
    /// The device the gate and the submission are on.
    device: &'a ash::Device,
    /// The gate, which the submission waits to reach 1.
    gate: vk::Semaphore,
    /// The timeline of the queue the submission is on.
    timeline: vk::Semaphore,
    /// The timepoint the submission reaches once the gate opens.
    timepoint: GpuTimepoint,
}

impl<'a> HeldSubmission<'a> {
    /// Submits to `backend`'s queue for `kind` behind a new gate, assigning
    /// the queue's next timepoint as [`Queues::submit`](super::Queues::submit)
    /// does.
    fn new(backend: &'a VulkanTestBackend, kind: QueueKind) -> Self {
        let device = backend.core.device.handle();
        let mut type_create_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let create_info = vk::SemaphoreCreateInfo::default().push(&mut type_create_info);
        // SAFETY: `create_info` is valid for the call, and timeline
        // semaphores are enabled on the device.
        let gate = unsafe { device.create_semaphore(&create_info, None) }.unwrap();

        let queues = &backend.core.queues;
        let id = queues.id(kind);
        let queue = queues.queue(id);
        let locked = lock(&queues.locks.0[id.index()]);
        let value = queue.submitted.load(Ordering::Relaxed) + 1;
        let wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(gate)
            .value(1)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(queue.timeline)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(slice::from_ref(&wait))
            .signal_semaphore_infos(slice::from_ref(&signal));
        // SAFETY: The queue and semaphores were created on `device`, the
        // queue is locked, the info structures outlive the call, and `value`
        // is the queue's next timeline value. The host signals the gate
        // later, which timeline semaphores allow.
        unsafe { device.queue_submit2(*locked, slice::from_ref(&submit_info), vk::Fence::null()) }
            .unwrap();
        queue.submitted.store(value, Ordering::Release);
        Self {
            device,
            gate,
            timeline: queue.timeline,
            timepoint: GpuTimepoint::new(id, value),
        }
    }

    /// Opens the gate, letting the submission run.
    fn open(&self) {
        let signal_info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.gate)
            .value(1);
        // SAFETY: The gate is a timeline created on `device`, and is at 0:
        // only this signals it, once, from the test or from `drop`, which
        // checks it's still at 0.
        unsafe { self.device.signal_semaphore(&signal_info) }.unwrap();
    }
}

impl Drop for HeldSubmission<'_> {
    fn drop(&mut self) {
        // SAFETY: The gate is a timeline created on `device`.
        if unsafe { self.device.get_semaphore_counter_value(self.gate) } == Ok(0) {
            self.open();
        }
        let value = self.timepoint.value();
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(slice::from_ref(&self.timeline))
            .values(slice::from_ref(&value));
        // SAFETY: The timeline was created on `device`, and its value was
        // submitted to be signalled once the gate opened, which it has. Once
        // it's reached, nothing waits on the gate, so it can be destroyed.
        unsafe {
            let _ = self.device.wait_semaphores(&wait_info, u64::MAX);
            self.device.destroy_semaphore(self.gate, None);
        }
    }
}

/// A submission that waits for a timepoint isn't complete until the timepoint
/// is, on every queue, whether the timepoint is on another queue or its own.
/// With the wait missing, an empty submission would complete at once.
#[test]
fn a_wait_holds_back_the_submission() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let graphics = backend.queue(QueueKind::Graphics);
    for kind in QueueKind::ALL {
        let held = HeldSubmission::new(&backend, QueueKind::Graphics);
        let queue = backend.queue(kind);
        let after = queue
            .submit(SubmitDesc {
                wait_for: &[held.timepoint],
            })
            .unwrap();
        thread::sleep(Duration::from_millis(50));
        assert!(!graphics.is_complete(held.timepoint).unwrap(), "{kind:?}");
        assert!(!queue.is_complete(after).unwrap(), "{kind:?}");
        held.open();
        queue.wait(after).unwrap();
        assert!(graphics.is_complete(held.timepoint).unwrap(), "{kind:?}");
    }
}

/// Several threads submitting to one queue at once each get increasing
/// timepoints, and together the timepoints count every submission once, so each
/// is assigned under the queue's lock.
#[test]
fn threads_submit_to_one_queue() {
    const THREADS: u64 = 4;
    const SUBMISSIONS: u64 = 25;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    for kind in QueueKind::ALL {
        let queue = backend.queue(kind);
        let start = queue.submit(SubmitDesc::default()).unwrap().value();
        let per_thread: Vec<Vec<GpuTimepoint>> = thread::scope(|scope| {
            let threads: Vec<_> = (0..THREADS)
                .map(|_| {
                    scope.spawn(|| {
                        (0..SUBMISSIONS)
                            .map(|_| queue.submit(SubmitDesc::default()).unwrap())
                            .collect()
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect()
        });
        for timepoints in &per_thread {
            assert!(
                timepoints
                    .windows(2)
                    .all(|pair| pair[0].value() < pair[1].value()),
                "{kind:?}"
            );
        }
        let timepoints: Vec<GpuTimepoint> = per_thread.into_iter().flatten().collect();
        let values: BTreeSet<u64> = timepoints
            .iter()
            .map(|timepoint| timepoint.value())
            .collect();
        let expected: BTreeSet<u64> = (start + 1..=start + THREADS * SUBMISSIONS).collect();
        assert_eq!(values, expected, "{kind:?}");
        let last = timepoints
            .iter()
            .max_by_key(|timepoint| timepoint.value())
            .unwrap();
        queue.wait(*last).unwrap();
        assert!(
            timepoints
                .iter()
                .all(|&timepoint| queue.is_complete(timepoint).unwrap()),
            "{kind:?}"
        );
    }
}

/// Each kind gets the queue its work is submitted to, and kinds share one
/// `Arc`, so one timeline, exactly when they share a queue ID. A queue
/// reports the most capable kind that uses it.
#[test]
fn each_kind_gets_the_queue_that_serves_it() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    for kind in QueueKind::ALL {
        let queue = backend.queue(kind);
        assert_eq!(queue.id(), backend.core.queues.id(kind), "{kind:?}");
        let most_capable = QueueKind::ALL
            .into_iter()
            .find(|&other| backend.queue(other).id() == queue.id())
            .unwrap();
        assert_eq!(queue.kind(), most_capable, "{kind:?}");
        for other in QueueKind::ALL {
            assert_eq!(
                Arc::ptr_eq(queue, backend.queue(other)),
                queue.id() == backend.queue(other).id(),
                "{kind:?} and {other:?}",
            );
        }
    }
}

/// A queue's submissions reach increasing timepoints, in submission order, and
/// waiting for the last means every earlier one is reached too.
#[test]
fn submissions_reach_their_timepoints_in_order() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let graphics = backend.queue(QueueKind::Graphics);
    let first = graphics.submit(SubmitDesc::default()).unwrap();
    let second = graphics.submit(SubmitDesc::default()).unwrap();
    assert_eq!(first.queue(), graphics.id());
    assert!(second.value() > first.value());
    graphics.wait(second).unwrap();
    assert!(graphics.is_complete(first).unwrap());
    assert!(graphics.is_complete(second).unwrap());
}

/// Waiting on a queue for a timepoint on another queue's timeline panics,
/// naming the queue.
#[test]
#[should_panic(
    expected = "GpuTimepoint { queue: QueueId(1), value: 1 } isn't on the graphics queue's timeline"
)]
fn waiting_for_another_queues_timepoint_panics() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let graphics = backend.queue(QueueKind::Graphics);
    assert_eq!(graphics.id(), QueueId::new(0));
    let _ = graphics.wait(GpuTimepoint::new(QueueId::new(1), 1));
}

/// Waiting for a timepoint the queue hasn't returned panics, rather than
/// waiting forever.
#[test]
#[should_panic(
    expected = "the graphics queue never returned GpuTimepoint { queue: QueueId(0), value: 1 }: it \
                has made 0 submissions"
)]
fn waiting_for_a_timepoint_never_returned_panics() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let graphics = backend.queue(QueueKind::Graphics);
    let _ = graphics.wait(GpuTimepoint::new(graphics.id(), 1));
}
