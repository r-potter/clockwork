//! The device's queues, their locks and their timelines (design 0002,
//! "Queues and submission").
//!
//! Each family in use has one queue. Each [`QueueRole`], a kind's work or
//! presenting, is mapped to its family's queue when the device is created,
//! so a kind that shares a family with a more capable kind is served by that
//! kind's queue (see
//! [`QueueFamilies::serving_kind`](super::device_selection::QueueFamilies::serving_kind)).
//!
//! A queue's [`QueueId`] is its index among the device's queues, so kinds
//! that share a queue share its ID, and the ID, not the kind, is how a point
//! or a command buffer names its queue.
//!
//! Each queue has one timeline semaphore, which counts its
//! submissions: the `n`th submission signals `n`. A wait on one of a queue's
//! own points is dropped, since the queue already orders its submissions.
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
use super::command_pools::{CommandPools, PooledCommands};
use super::device::LogicalDevice;
use super::device_selection::QueueFamilies;
use super::error::error;
use crate::{GpuPoint, Queue, QueueId, QueueKind, RhiResult, SubmitDesc};

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

/// The queue of each family in use. Owned by the core, which destroys it
/// with [`destroy`](Self::destroy).
///
/// # Mapping roles to queues
///
/// Roles and queues aren't one to one. There are always four roles, but the
/// device has one queue for each distinct family in use, so between one and
/// four, and roles on the same family share its queue. On a device with a
/// single family, every role uses one queue. A shared queue must have one
/// lock, one timeline and one set of command pools, so the queues can't be
/// stored once for each role.
/// Instead, two fields record the mapping, one in each direction:
/// - [`roles`](Self::roles) maps each role to the queue it uses. Several
///   roles may map to one queue. Every lookup by role or kind goes through
///   it.
/// - [`DeviceQueue::role`] maps each queue to the role that owns it: the
///   first role in [`QueueRole::ALL`] that uses it. That's the kind
///   [`Queue::kind`] reports for every kind sharing the queue, and the name
///   the queue and its timeline are given.
///
/// A queue's index in `queues` is its [`QueueId`], so once a kind is mapped
/// to its queue's ID, everything else looks the queue up by ID.
///
/// Both are filled in once, by [`new`](Self::new), which goes through the
/// roles in the order of [`QueueRole::ALL`]: the kinds from the most capable
/// to the least, then presenting. A role whose family an earlier role uses is
/// mapped to that role's queue. Kinds find it through
/// [`QueueFamilies::serving_kind`](super::device_selection::QueueFamilies::serving_kind),
/// which device selection's tests cover, and presenting through the first
/// kind on its family. Any other role gets a new queue, which it owns. So a
/// queue's owner always comes before the roles sharing it, and the queue
/// exists by the time they're mapped. When headless, presenting has no
/// family, and isn't mapped.
///
/// For example, on a device with a dedicated compute family 1 but no
/// dedicated transfer family, and with graphics family 0 able to present:
///
/// | Role | Family | Index in `queues` | Owned by |
/// |------|--------|-------------------|----------|
/// | `Kind(Graphics)` | 0 | 0 | itself |
/// | `Kind(Compute)` | 1 | 1 | itself |
/// | `Kind(Transfer)` | 1 | 1 | `Kind(Compute)` |
/// | `Present` | 0 | 0 | `Kind(Graphics)` |
///
/// So `queues` holds the graphics queue and then the compute queue, `roles`
/// is `[Some(0), Some(1), Some(1), Some(0)]`, the compute and transfer
/// queues are both `QueueId(1)`, and their [`Queue::kind`] is `Compute`.
pub(super) struct Queues {
    /// One queue from each distinct family in use, in the order their owners
    /// come in [`QueueRole::ALL`], each at the index its [`QueueId`] names.
    queues: Vec<DeviceQueue>,
    /// The index in `queues` of the queue each role uses, in the order of
    /// [`QueueRole::ALL`], or `None` for presenting when headless. Several
    /// roles share an index when they share a family (see "Mapping roles to
    /// queues" above).
    roles: [Option<usize>; QueueRole::ALL.len()],
}

/// One of the device's queues, with its lock, its timeline and its command
/// pools.
///
/// A queue only presenting uses has a timeline and pools too, which nothing
/// uses, so that every queue is the same. Its pools are created only when
/// checked out, so they cost nothing.
struct DeviceQueue {
    /// The first role in [`QueueRole::ALL`] that uses it, which owns it: the
    /// kind it serves as, which [`Queue::kind`] reports, or presenting if no
    /// kind uses it. It's the reverse of [`Queues::roles`] (see "Mapping
    /// roles to queues" on [`Queues`]).
    role: QueueRole,
    /// The queue, locked for each submission and present.
    queue: Mutex<vk::Queue>,
    /// A timeline semaphore, signalled with the number of each submission.
    timeline: vk::Semaphore,
    /// The number of the last submission, so the value its completion
    /// signals. Written only while `queue` is locked.
    submitted: AtomicU64,
    /// The command pools for its family that aren't checked out.
    pools: CommandPools,
}

impl Queues {
    /// Wraps `device`'s queues, with a timeline for each, naming both after
    /// the first role in [`QueueRole::ALL`] that uses the queue: the most
    /// capable kind it serves, or `present` for one only presenting uses.
    ///
    /// # Errors
    ///
    /// Returns an error if a timeline semaphore can't be created.
    pub(super) fn new(device: &LogicalDevice) -> Result<Self> {
        let families = device.queue_families();
        let mut queues = Self {
            queues: Vec::new(),
            roles: [None; QueueRole::ALL.len()],
        };
        for role in QueueRole::ALL {
            let Some(family) = role.family(families) else {
                continue;
            };
            // A kind served by another is served by a more capable one, and
            // presenting may share a kind's family. Either comes later in
            // `ALL` than the role it shares with, whose queue already exists.
            let shared = match role {
                QueueRole::Kind(kind) => {
                    Some(families.serving_kind(kind)).filter(|&serving| serving != kind)
                }
                QueueRole::Present => QueueKind::ALL
                    .into_iter()
                    .find(|&kind| families.family(kind) == family),
            };
            if let Some(kind) = shared {
                queues.roles[role.index()] = queues.roles[QueueRole::Kind(kind).index()];
                continue;
            }
            if let Err(err) = queues.add(device, role, family) {
                // SAFETY: Nothing has been submitted to wait on or signal
                // the timelines created so far.
                unsafe { queues.destroy(device.handle()) };
                return Err(err);
            }
        }
        Ok(queues)
    }

    /// Adds the queue of `family` on `device`, with a timeline, both named
    /// after `role`, and maps `role` to it.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline semaphore can't be created.
    fn add(&mut self, device: &LogicalDevice, role: QueueRole, family: u32) -> Result<()> {
        let mut type_create_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let create_info = vk::SemaphoreCreateInfo::default().push(&mut type_create_info);
        // SAFETY: `create_info` is valid for the call, and timeline
        // semaphores are enabled on the device.
        let timeline = unsafe { device.handle().create_semaphore(&create_info, None) }
            .with_context(|| format!("failed to create the {} queue's timeline", role.name()))?;
        device.set_object_name(timeline, &format!("{} timeline", role.name()));
        let queue = device.queue(family);
        device.set_object_name(queue, role.name());
        self.queues.push(DeviceQueue {
            role,
            queue: Mutex::new(queue),
            timeline,
            submitted: AtomicU64::new(0),
            pools: CommandPools::new(family),
        });
        self.roles[role.index()] = Some(self.queues.len() - 1);
        Ok(())
    }

    /// The ID of the queue `kind`'s work is submitted to.
    pub(super) fn id(&self, kind: QueueKind) -> QueueId {
        let index = self.roles[QueueRole::Kind(kind).index()].expect("every kind has a queue");
        QueueId::new(u32::try_from(index).expect("a device has at most four queues"))
    }

    /// The kind the queue `id` serves as, which [`Queue::kind`] reports: the
    /// most capable kind whose work is submitted to it.
    pub(super) fn kind(&self, id: QueueId) -> QueueKind {
        self.get(id).kind()
    }

    /// The queue `id`.
    ///
    /// # Panics
    ///
    /// Panics if the device has no queue `id`, which only a point from
    /// another device can name.
    fn get(&self, id: QueueId) -> &DeviceQueue {
        self.queues
            .get(id.index())
            .unwrap_or_else(|| panic!("the device has no queue {id:?}"))
    }

    /// Locks the queue `role` uses, for one submission or present, or
    /// returns `None` if it has none, which is only presenting when
    /// headless.
    pub(super) fn lock(&self, role: QueueRole) -> Option<MutexGuard<'_, vk::Queue>> {
        let index = self.roles[role.index()]?;
        Some(lock(&self.queues[index].queue))
    }

    /// Checks out a command pool for the queue `id`, on `device` (see
    /// [`CommandPools::check_out`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline can't be read, or a pool can't be
    /// reset or created.
    #[expect(dead_code, reason = "translation checks pools out in the next PR")]
    pub(super) fn check_out_pool(
        &self,
        device: &ash::Device,
        id: QueueId,
    ) -> RhiResult<PooledCommands> {
        let queue = self.get(id);
        queue.pools.check_out(device, || {
            // SAFETY: The semaphore is a timeline created on `device`.
            unsafe { device.get_semaphore_counter_value(queue.timeline) }
                .map_err(|result| error(result, "read the timeline of queue", queue.role.name()))
        })
    }

    /// Gives `commands` back to the queue `id`, reusable once its timeline
    /// reaches `reusable_at`, or at once for 0.
    #[expect(dead_code, reason = "command buffers give pools back in the next PR")]
    pub(super) fn give_back_pool(&self, id: QueueId, commands: PooledCommands, reusable_at: u64) {
        self.get(id).pools.give_back(commands, reusable_at);
    }

    /// Submits `desc` to the queue `id`, on `device`, and returns the point
    /// its completion reaches.
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
        id: QueueId,
        desc: SubmitDesc<'_>,
    ) -> RhiResult<GpuPoint> {
        let queue = self.get(id);
        let waits: Vec<_> = desc
            .wait_for
            .iter()
            .filter_map(|&point| {
                let waited = self.get(point.queue());
                waited.check_returned(point);
                // A queue runs its own submissions in order.
                (point.queue() != id).then(|| {
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
        .map_err(|result| error(result, "submit to queue", queue.role.name()))?;
        queue.submitted.store(value, Ordering::Release);
        Ok(GpuPoint::new(id, value))
    }

    /// Blocks until `point`, which the queue `id` returned, is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting fails.
    ///
    /// # Panics
    ///
    /// Panics if that queue didn't return `point`.
    pub(super) fn wait(&self, device: &ash::Device, id: QueueId, point: GpuPoint) -> RhiResult<()> {
        let queue = self.check_own(id, point);
        let value = point.value();
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&queue.timeline))
            .values(std::slice::from_ref(&value));
        // SAFETY: The semaphore is a timeline created on `device`, and
        // `wait_info` is valid for the call. Waiting needs no lock.
        unsafe { device.wait_semaphores(&wait_info, u64::MAX) }
            .map_err(|result| error(result, "wait for queue", queue.role.name()))
    }

    /// Whether `point`, which the queue `id` returned, has been reached.
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
        id: QueueId,
        point: GpuPoint,
    ) -> RhiResult<bool> {
        let queue = self.check_own(id, point);
        // SAFETY: The semaphore is a timeline created on `device`.
        let reached = unsafe { device.get_semaphore_counter_value(queue.timeline) }
            .map_err(|result| error(result, "read the timeline of queue", queue.role.name()))?;
        Ok(reached >= point.value())
    }

    /// Checks that `point` is one the queue `id` returned, and returns the
    /// queue.
    ///
    /// # Panics
    ///
    /// Panics if it isn't.
    fn check_own(&self, id: QueueId, point: GpuPoint) -> &DeviceQueue {
        let queue = self.get(id);
        assert!(
            point.queue() == id,
            "{point:?} isn't on the {} queue's timeline",
            queue.role.name(),
        );
        queue.check_returned(point);
        queue
    }

    /// Destroys the timelines and command pools.
    ///
    /// # Safety
    ///
    /// They must have been created on `device`, no command pool may be
    /// checked out, and no pending submission may use them, for example
    /// because the device is idle.
    pub(super) unsafe fn destroy(&mut self, device: &ash::Device) {
        for mut queue in self.queues.drain(..) {
            // SAFETY: Guaranteed by the caller.
            unsafe {
                queue.pools.destroy(device);
                device.destroy_semaphore(queue.timeline, None);
            }
        }
    }
}

impl DeviceQueue {
    /// The kind the queue serves as.
    ///
    /// # Panics
    ///
    /// Panics if only presenting uses the queue. Only a queue that kinds
    /// use has a [`VulkanQueue`], which is what asks, so this can't happen.
    fn kind(&self) -> QueueKind {
        match self.role {
            QueueRole::Kind(kind) => kind,
            QueueRole::Present => panic!("no kind's work is submitted to the present queue"),
        }
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
            self.role.name(),
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
    /// Which of the core's queues this is.
    id: QueueId,
}

impl VulkanQueue {
    /// A view of the queue `id` on `core`.
    pub(super) fn new(core: Arc<VulkanDevice>, id: QueueId) -> Self {
        Self { core, id }
    }
}

impl Queue for VulkanQueue {
    fn id(&self) -> QueueId {
        self.id
    }

    fn kind(&self) -> QueueKind {
        self.core.queues.kind(self.id)
    }

    fn submit(&self, desc: SubmitDesc<'_>) -> RhiResult<GpuPoint> {
        let core = &self.core;
        core.lost
            .guard(|| core.queues.submit(core.device.handle(), self.id, desc))
    }

    fn wait(&self, point: GpuPoint) -> RhiResult<()> {
        let core = &self.core;
        core.lost
            .guard(|| core.queues.wait(core.device.handle(), self.id, point))
    }

    fn is_complete(&self, point: GpuPoint) -> RhiResult<bool> {
        let core = &self.core;
        core.lost.guard(|| {
            core.queues
                .is_complete(core.device.handle(), self.id, point)
        })
    }
}

impl fmt::Debug for VulkanQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = self.kind();
        f.debug_struct("VulkanQueue")
            .field("id", &self.id)
            .field("kind", &kind)
            .field("family", &self.core.device.queue_families().family(kind))
            .finish_non_exhaustive()
    }
}
