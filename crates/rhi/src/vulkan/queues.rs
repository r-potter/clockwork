//! The device's queues and their locks (design 0002, "Queues and
//! submission").
//!
//! Each family in use has one queue. Each [`QueueRole`], a kind's work or
//! presenting, is mapped to its family's queue when the device is created,
//! so roles that share a family share the queue the first of them in
//! [`QueueRole::ALL`] created (see [`QueueLayout`]).
//!
//! Vulkan requires a queue to be used by one thread at a time, so each is
//! behind a `Mutex`, held for each submission and present. Waiting for the
//! whole device to be idle uses every queue, so it holds every lock (see
//! [`QueueLocks::wait_idle`]).

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ash::{VkResult, vk};

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
    pub(super) const fn name(self) -> &'static str {
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

/// The queues to create for a device's [`QueueFamilies`], and the queue each
/// role uses. Pure, so it's tested without a device.
#[derive(Debug)]
pub(super) struct QueueLayout {
    /// One entry per queue, each from a distinct family: the family, and the
    /// role the queue is named after, which is the first role in
    /// [`QueueRole::ALL`] that uses it. So a queue is named after the most
    /// capable kind it serves, or `present` if only presenting uses it.
    pub queues: Vec<(u32, QueueRole)>,
    /// The index in `queues` of the queue each role uses, in the order of
    /// [`QueueRole::ALL`], or `None` for presenting when headless.
    roles: [Option<usize>; QueueRole::ALL.len()],
}

impl QueueLayout {
    /// Lays out one queue for each distinct family in `families`.
    pub(super) fn new(families: QueueFamilies) -> Self {
        let mut queues: Vec<(u32, QueueRole)> = Vec::new();
        let roles = QueueRole::ALL.map(|role| {
            let family = role.family(families)?;
            let index = queues
                .iter()
                .position(|&(other, _)| other == family)
                .unwrap_or_else(|| {
                    queues.push((family, role));
                    queues.len() - 1
                });
            Some(index)
        });
        Self { queues, roles }
    }
}

/// Every queue on a device, each behind its lock, in the order of the
/// device's [`QueueLayout::queues`].
///
/// Cloning shares the locks, so that objects that wait for the device to be
/// idle when they're dropped can hold every lock while they wait.
#[derive(Clone)]
pub(super) struct QueueLocks(Arc<[Mutex<vk::Queue>]>);

impl QueueLocks {
    /// Wraps `queues`, in the order of the device's [`QueueLayout::queues`].
    pub(super) fn new(queues: impl IntoIterator<Item = vk::Queue>) -> Self {
        Self(queues.into_iter().map(Mutex::new).collect())
    }

    /// Waits for `device` to be idle, holding every queue's lock, since
    /// Vulkan requires every queue on a device to be used by no other thread
    /// while it waits.
    ///
    /// Locks the queues in order. Everything else holds one queue's lock at a
    /// time, so this can't deadlock with other threads.
    ///
    /// The locks aren't reentrant, so the calling thread must not hold a
    /// queue's lock, from [`Queues::lock`] or [`QueueLock::lock`], or this
    /// waits for itself forever. That includes dropping an object that
    /// waits for the device to be idle, such as a swapchain, while holding
    /// one.
    ///
    /// # Safety
    ///
    /// `device` must be the live device the queues were retrieved from.
    pub(super) unsafe fn wait_idle(&self, device: &ash::Device) -> VkResult<()> {
        let _guards: Vec<_> = self.0.iter().map(lock).collect();
        // SAFETY: `device` is live, as the caller guarantees, and every one of
        // its queues is locked for the call.
        unsafe { device.device_wait_idle() }
    }
}

/// The queue each role uses, locked for each use.
pub(super) struct Queues {
    /// The device's queues.
    locks: QueueLocks,
    /// The index in `locks` of the queue each role uses, in the order of
    /// [`QueueRole::ALL`], or `None` for presenting when headless.
    roles: [Option<usize>; QueueRole::ALL.len()],
}

impl Queues {
    /// Maps each role to the queue it uses among `device`'s queues, as the
    /// layout `device` created them from says.
    pub(super) fn new(device: &LogicalDevice) -> Self {
        Self {
            locks: device.queue_locks().clone(),
            roles: device.queue_layout().roles,
        }
    }

    /// The queue `role` uses, or `None` if it has none, which is only
    /// presenting when headless. For finding the queue before it's needed,
    /// and locking it for each use.
    pub(super) fn get(&self, role: QueueRole) -> Option<QueueLock<'_>> {
        let index = self.roles[role.index()]?;
        Some(QueueLock(&self.locks.0[index]))
    }

    /// Locks the queue `role` uses, for one submission or present, or
    /// returns `None` if it has none, which is only presenting when
    /// headless. Don't wait for the device to be idle while holding it (see
    /// [`QueueLocks::wait_idle`]).
    pub(super) fn lock(&self, role: QueueRole) -> Option<MutexGuard<'_, vk::Queue>> {
        self.get(role).map(QueueLock::lock)
    }
}

/// One of the device's queues, from [`Queues::get`], not yet locked.
#[derive(Clone, Copy)]
pub(super) struct QueueLock<'a>(&'a Mutex<vk::Queue>);

impl<'a> QueueLock<'a> {
    /// Locks the queue, for one submission or present. Don't wait for the
    /// device to be idle while holding it (see [`QueueLocks::wait_idle`]).
    pub(super) fn lock(self) -> MutexGuard<'a, vk::Queue> {
        lock(self.0)
    }
}

/// Locks `queue`, recovering it if a thread panicked while holding it, which
/// leaves nothing inconsistent (design 0002, "Error handling").
fn lock(queue: &Mutex<vk::Queue>) -> MutexGuard<'_, vk::Queue> {
    queue.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The families when every kind uses family 0, headless.
    const FAMILY_0: QueueFamilies = QueueFamilies {
        graphics: 0,
        compute: 0,
        transfer: 0,
        present: None,
    };

    /// The queue `role` uses in `layout`, as its family and the role it's
    /// named after.
    fn queue_for(layout: &QueueLayout, role: QueueRole) -> Option<(u32, QueueRole)> {
        layout.roles[role.index()].map(|index| layout.queues[index])
    }

    /// Headless with one family, every kind uses one queue, named after the
    /// graphics kind, and presenting has none.
    #[test]
    fn one_family_has_one_queue() {
        let layout = QueueLayout::new(FAMILY_0);
        assert_eq!(layout.queues, [(0, QueueRole::Kind(QueueKind::Graphics))]);
        for kind in QueueKind::ALL {
            assert_eq!(layout.roles[QueueRole::Kind(kind).index()], Some(0));
        }
        assert_eq!(queue_for(&layout, QueueRole::Present), None);
    }

    /// Presenting on the graphics family shares the graphics queue, and so
    /// its lock.
    #[test]
    fn presenting_shares_the_graphics_queue() {
        let layout = QueueLayout::new(QueueFamilies {
            present: Some(0),
            ..FAMILY_0
        });
        assert_eq!(layout.queues.len(), 1);
        assert_eq!(
            layout.roles[QueueRole::Present.index()],
            layout.roles[QueueRole::Kind(QueueKind::Graphics).index()],
        );
    }

    /// Dedicated families each get a queue named after their kind, and
    /// transfers sharing the compute family use the compute queue.
    #[test]
    fn dedicated_families_have_their_own_queues() {
        let layout = QueueLayout::new(QueueFamilies {
            compute: 1,
            transfer: 1,
            present: Some(0),
            ..FAMILY_0
        });
        assert_eq!(
            layout.queues,
            [
                (0, QueueRole::Kind(QueueKind::Graphics)),
                (1, QueueRole::Kind(QueueKind::Compute)),
            ],
        );
        assert_eq!(
            queue_for(&layout, QueueRole::Kind(QueueKind::Transfer)),
            Some((1, QueueRole::Kind(QueueKind::Compute))),
        );
    }

    /// A family only presenting uses gets its own queue, named after
    /// presenting, while one a kind also uses is named after the kind.
    #[test]
    fn a_present_only_family_is_named_after_presenting() {
        let layout = QueueLayout::new(QueueFamilies {
            transfer: 1,
            present: Some(2),
            ..FAMILY_0
        });
        assert_eq!(
            queue_for(&layout, QueueRole::Present),
            Some((2, QueueRole::Present)),
        );
        let layout = QueueLayout::new(QueueFamilies {
            transfer: 1,
            present: Some(1),
            ..FAMILY_0
        });
        assert_eq!(
            queue_for(&layout, QueueRole::Present),
            Some((1, QueueRole::Kind(QueueKind::Transfer))),
        );
    }
}
