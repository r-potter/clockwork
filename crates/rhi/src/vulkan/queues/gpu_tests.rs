//! T1 tests of queues, submission and timelines on a real device (design
//! 0002, "Queues and submission", part of R9).

use std::sync::Arc;

use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::test_support::VulkanTestBackend;
use crate::{GpuPoint, QueueId, QueueKind, RenderHardwareInterface, SubmitDesc};

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

/// A queue's submissions reach increasing points, in submission order, and
/// waiting for the last means every earlier one is reached too.
#[test]
fn submissions_reach_their_points_in_order() {
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

/// Each queue can wait for a point on the graphics queue, whether it has a
/// family of its own, when the wait is a semaphore wait, or shares the
/// graphics family, when the wait is dropped.
#[test]
fn queues_wait_for_points_on_other_queues() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let graphics = backend.queue(QueueKind::Graphics);
    for kind in QueueKind::ALL {
        let queue = backend.queue(kind);
        let before = graphics.submit(SubmitDesc::default()).unwrap();
        let after = queue
            .submit(SubmitDesc {
                wait_for: &[before],
                ..SubmitDesc::default()
            })
            .unwrap();
        queue.wait(after).unwrap();
        assert!(graphics.is_complete(before).unwrap(), "{kind:?}");
    }
}

/// Waiting on a queue for a point on another queue's timeline panics,
/// naming the queue.
#[test]
#[should_panic(
    expected = "GpuPoint { queue: QueueId(1), value: 1 } isn't on the graphics queue's timeline"
)]
fn waiting_for_another_queues_point_panics() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let graphics = backend.queue(QueueKind::Graphics);
    assert_eq!(graphics.id(), QueueId::new(0));
    let _ = graphics.wait(GpuPoint::new(QueueId::new(1), 1));
}

/// Waiting for a point the queue hasn't returned panics, rather than
/// waiting forever.
#[test]
#[should_panic(
    expected = "the graphics queue never returned GpuPoint { queue: QueueId(0), value: 1 }: it \
                has made 0 submissions"
)]
fn waiting_for_a_point_never_returned_panics() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let graphics = backend.queue(QueueKind::Graphics);
    let _ = graphics.wait(GpuPoint::new(graphics.id(), 1));
}
