//! The device's queues and their locks (design 0002, "Queues and
//! submission").
//!
//! Each family in use has one queue. Each [`QueueRole`], a kind's work or
//! presenting, is mapped to its family's queue when the device is created,
//! so a kind that shares a family with a more capable kind is served by that
//! kind's queue (see
//! [`QueueFamilies::serving_kind`](super::device_selection::QueueFamilies::serving_kind)).
//!
//! Vulkan requires a queue to be used by one thread at a time, so each is
//! behind a `Mutex`, held for each submission and present.

use std::sync::{Mutex, MutexGuard, PoisonError};

use ash::vk;

use super::device::LogicalDevice;
use super::device_selection::QueueFamilies;
use crate::QueueKind;

/// What a queue is used for: one kind's work, or presenting.
///
/// Presenting isn't a [`QueueKind`], since kinds rank queues by the commands
/// they can run, and a family of any kind may or may not be able to present.
/// It's a role only inside the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QueueRole {
    /// Submitting a kind's work.
    Kind(QueueKind),
    /// Presenting to the window's surface.
    Present,
}

impl QueueRole {
    /// Every role: the kinds, from the most capable to the least, then
    /// presenting.
    const ALL: [Self; QueueKind::ALL.len() + 1] = [
        Self::Kind(QueueKind::Graphics),
        Self::Kind(QueueKind::Compute),
        Self::Kind(QueueKind::Transfer),
        Self::Present,
    ];

    /// The role's position in [`QueueRole::ALL`].
    const fn index(self) -> usize {
        match self {
            Self::Kind(kind) => kind as usize,
            Self::Present => QueueKind::ALL.len(),
        }
    }

    /// The family the role uses, or `None` for presenting when headless.
    fn family(self, families: QueueFamilies) -> Option<u32> {
        match self {
            Self::Kind(kind) => Some(families.family(kind)),
            Self::Present => families.present,
        }
    }

    /// The role's name, which its queue is named after.
    const fn name(self) -> &'static str {
        match self {
            Self::Kind(kind) => kind.name(),
            Self::Present => "present",
        }
    }
}

// `QueueRole::index` takes a kind's position from its discriminant, which
// needs each role's index to be its position in `QueueRole::ALL`.
const _: () = {
    let mut position = 0;
    while position < QueueRole::ALL.len() {
        assert!(QueueRole::ALL[position].index() == position);
        position += 1;
    }
};

/// The queue of each family in use, locked for each use.
pub(super) struct Queues {
    /// One queue from each distinct family in use.
    queues: Vec<Mutex<vk::Queue>>,
    /// The index in `queues` of the queue each role uses, in the order of
    /// [`QueueRole::ALL`], or `None` for presenting when headless.
    roles: [Option<usize>; QueueRole::ALL.len()],
}

impl Queues {
    /// Wraps `device`'s queues, naming each after the first role in
    /// [`QueueRole::ALL`] that uses it: the most capable kind it serves, or
    /// `present` for one only presenting uses.
    pub(super) fn new(device: &LogicalDevice) -> Self {
        let families = device.queue_families();
        let mut queues = Vec::new();
        let mut queue_families = Vec::new();
        let roles = QueueRole::ALL.map(|role| {
            let family = role.family(families)?;
            if let Some(index) = queue_families.iter().position(|&other| other == family) {
                return Some(index);
            }
            let queue = device.queue(family);
            device.set_object_name(queue, role.name());
            queue_families.push(family);
            queues.push(Mutex::new(queue));
            Some(queues.len() - 1)
        });
        Self { queues, roles }
    }

    /// Locks the queue `role` uses, for one submission or present, or
    /// returns `None` if it has none, which is only presenting when
    /// headless.
    pub(super) fn lock(&self, role: QueueRole) -> Option<MutexGuard<'_, vk::Queue>> {
        let index = self.roles[role.index()]?;
        Some(lock(&self.queues[index]))
    }
}

/// Locks `queue`, recovering it if a thread panicked while holding it, which
/// leaves nothing inconsistent (design 0002, "Error handling").
fn lock(queue: &Mutex<vk::Queue>) -> MutexGuard<'_, vk::Queue> {
    queue.lock().unwrap_or_else(PoisonError::into_inner)
}
