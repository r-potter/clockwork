//! The device's queues and their locks (design 0002, "Queues and
//! submission").
//!
//! Each family a queue kind uses has one queue. A kind that shares a family
//! with a more capable kind is served by that kind's queue (see
//! [`QueueFamilies::serving_kind`]).
//!
//! Vulkan requires a queue to be used by one thread at a time, so each is
//! behind a `Mutex`, held for each submission and present.

use std::sync::{Mutex, MutexGuard, PoisonError};

use ash::vk;

use super::device::LogicalDevice;
use super::device_selection::QueueFamilies;
use crate::QueueKind;

/// The queue of each family in use, locked for each use.
pub(super) struct Queues {
    /// The queue of each kind that serves itself, in the order of
    /// [`QueueKind::ALL`].
    queues: Vec<KindQueue>,
    /// The queue to present on, if its family isn't one a kind uses.
    present_only: Option<Mutex<vk::Queue>>,
    /// The families the queues are from.
    families: QueueFamilies,
}

/// A queue that a kind's work is submitted to.
struct KindQueue {
    /// The kind it serves as.
    kind: QueueKind,
    /// The queue, locked for each submission and present.
    queue: Mutex<vk::Queue>,
}

impl Queues {
    /// Wraps `device`'s queues, naming each after the kind it serves, or
    /// `present` for one only presenting uses.
    pub(super) fn new(device: &LogicalDevice) -> Self {
        let families = device.queue_families();
        let mut queues = Vec::new();
        for kind in QueueKind::ALL {
            if families.serving_kind(kind) != kind {
                continue;
            }
            let queue = device.queue(families.family(kind));
            device.set_object_name(queue, kind.name());
            queues.push(KindQueue {
                kind,
                queue: Mutex::new(queue),
            });
        }
        let mut present_only = None;
        if let Some(present) = families.present
            && !QueueKind::ALL
                .into_iter()
                .any(|kind| families.family(kind) == present)
        {
            let queue = device.queue(present);
            device.set_object_name(queue, "present");
            present_only = Some(Mutex::new(queue));
        }
        Self {
            queues,
            present_only,
            families,
        }
    }

    /// The queue `kind`'s work is submitted to.
    fn get(&self, kind: QueueKind) -> &KindQueue {
        let serving = self.families.serving_kind(kind);
        self.queues
            .iter()
            .find(|queue| queue.kind == serving)
            .expect("every kind that serves itself has a queue")
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
}

/// Locks `queue`, recovering it if a thread panicked while holding it, which
/// leaves nothing inconsistent (design 0002, "Error handling").
fn lock(queue: &Mutex<vk::Queue>) -> MutexGuard<'_, vk::Queue> {
    queue.lock().unwrap_or_else(PoisonError::into_inner)
}
