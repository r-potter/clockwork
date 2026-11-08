//! The device's queues, their locks and their timelines (design 0002,
//! "Queues and submission").
//!
//! Each family in use has one queue. Each [`QueueRole`], a kind's work or
//! presenting, is mapped to its family's queue when the device is created,
//! so roles that share a family share the queue the first of them in
//! [`QueueRole::ALL`] created (see [`QueueLayout`]).
//!
//! A queue's [`QueueId`] is its index among the device's queues, so kinds that
//! share a queue share its ID, and the ID, not the kind, is how a timepoint or
//! a command buffer names its queue.
//!
//! Each queue has one timeline semaphore, which counts its submissions: the
//! `n`th submission signals `n`. A wait on one of a queue's own timepoints is
//! kept, like any other: Vulkan starts a queue's submissions in order, but
//! doesn't wait for one to complete before running the next, so kinds that
//! share a queue are ordered as if they had their own.
//!
//! Vulkan requires a queue to be used by one thread at a time, so each is
//! behind a `Mutex`, held for each submission and present. Timeline values
//! are assigned under the same lock, so they increase in submission order.
//! Waiting and polling only read the semaphore, so they take no lock.
//! Waiting for the whole device to be idle uses every queue, so it holds
//! every lock (see [`QueueLocks::wait_idle`]).

#[cfg(test)]
mod gpu_tests;

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::{Context, Result};
use ash::vk::TaggedStructure;
use ash::{VkResult, vk};

use super::VulkanDevice;
use super::command_pools::{CommandPool, CommandPools};
use super::device::LogicalDevice;
use super::device_selection::QueueFamilies;
use super::error::error;
use super::physical_device::PhysicalDevice;
use super::timings::KeptTimings;
use super::translate::VulkanCommandBuffer;
use crate::timing::{RetainedTimings, TimestampClock};
use crate::{
    CommandBuffer, CommandList, GpuTimepoint, Queue, QueueId, QueueKind, RhiResult,
    SubmissionTimings, SubmitDesc,
};

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

/// The device's queues, each with its lock and its timeline. Owned by the
/// core, which destroys the timelines with [`destroy`](Self::destroy).
///
/// # Mapping roles to queues
///
/// Roles and queues aren't one to one. There are always four roles, but the
/// device has one queue for each distinct family in use, so between one and
/// four, and roles on the same family share its queue. On a device with a
/// single family, every role uses one queue. A shared queue must have one
/// lock, one timeline and one set of command pools, so the queues can't be
/// stored once for each role. Instead, the device's [`QueueLayout`] records
/// the mapping, one way in each direction:
/// - [`roles`](Self::roles) maps each role to the queue it uses. Several
///   roles may map to one queue. Every lookup by role or kind goes through
///   it.
/// - [`DeviceQueue::role`] maps each queue to the role that owns it: the
///   first role in [`QueueRole::ALL`] that uses it. That's the kind
///   [`Queue::kind`] reports for every kind sharing the queue, and the name
///   the queue and its timeline are given.
///
/// A queue's index in `queues`, which is also its index in `locks`, is its
/// [`QueueId`], so once a kind is mapped to its queue's ID, everything else
/// looks the queue up by ID. When headless, presenting has no family, and
/// isn't mapped.
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
    /// The device's queues, each behind its lock, at the index its
    /// [`QueueId`] names.
    locks: QueueLocks,
    /// The index in `queues` of the queue each role uses, in the order of
    /// [`QueueRole::ALL`], or `None` for presenting when headless. Several
    /// roles share an index when they share a family (see "Mapping roles to
    /// queues" above).
    roles: [Option<usize>; QueueRole::ALL.len()],
    /// What each queue in `locks` is used for, its timeline and its command
    /// pools, at the same index.
    queues: Vec<DeviceQueue>,
}

/// What one of the device's queues is used for, its timeline and its
/// command pools. Its lock is at the same index in [`Queues::locks`].
///
/// A queue only presenting uses has a timeline and pools too, which nothing
/// uses, so that every queue is the same. Its pools are created only when
/// claimed, so they cost nothing.
struct DeviceQueue {
    /// The first role in [`QueueRole::ALL`] that uses it, which owns it: the
    /// kind it serves as, which [`Queue::kind`] reports, or presenting if no
    /// kind uses it. It's the reverse of [`Queues::roles`] (see "Mapping
    /// roles to queues" on [`Queues`]).
    role: QueueRole,
    /// Its family.
    family: u32,
    /// Its family's timestamp counter, or `None` if the family can't write
    /// timestamps.
    clock: Option<TimestampClock>,
    /// A timeline semaphore, signalled with the number of each submission.
    timeline: vk::Semaphore,
    /// The number of the last submission, so the value its completion
    /// signals. Written only while the queue is locked.
    submitted: AtomicU64,
    /// The command pools for its family that aren't claimed.
    pools: CommandPools,
    /// The timings of its submissions that haven't been taken. Never held
    /// while recycling query chunks.
    timings: Mutex<RetainedTimings<KeptTimings>>,
}

impl Queues {
    /// Maps each role to the queue it uses among `device`'s queues, as the
    /// layout `device` created them from says, and creates a timeline for
    /// each queue, named after the role that owns it. `physical_device`,
    /// which `device` was created on, gives each queue's timestamp counter.
    ///
    /// # Errors
    ///
    /// Returns an error if a timeline semaphore can't be created.
    pub(super) fn new(device: &LogicalDevice, physical_device: &PhysicalDevice) -> Result<Self> {
        let layout = device.queue_layout();
        let mut queues = Self {
            locks: device.queue_locks().clone(),
            roles: layout.roles,
            queues: Vec::with_capacity(layout.queues.len()),
        };
        let period_ns = physical_device.limits().timestamp_period;
        for &(family, role) in &layout.queues {
            let valid_bits = physical_device.queue_families[family as usize].timestamp_valid_bits;
            let clock = TimestampClock::new(valid_bits, period_ns);
            if let Err(err) = queues.add(device, role, family, clock) {
                // SAFETY: Nothing has been submitted to wait on or signal
                // the timelines created so far.
                unsafe { queues.destroy(device.handle()) };
                return Err(err);
            }
        }
        Ok(queues)
    }

    /// Adds the next queue, of `family`, which `role` owns, with a timeline
    /// named after it, and the family's timestamp counter, `clock`.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline semaphore can't be created.
    fn add(
        &mut self,
        device: &LogicalDevice,
        role: QueueRole,
        family: u32,
        clock: Option<TimestampClock>,
    ) -> Result<()> {
        let mut type_create_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let create_info = vk::SemaphoreCreateInfo::default().push(&mut type_create_info);
        // SAFETY: `create_info` is valid for the call, and timeline
        // semaphores are enabled on the device.
        let timeline = unsafe { device.handle().create_semaphore(&create_info, None) }
            .with_context(|| format!("failed to create the {} queue's timeline", role.name()))?;
        device.set_object_name(timeline, &format!("{} timeline", role.name()));
        self.queues.push(DeviceQueue {
            role,
            family,
            clock,
            timeline,
            submitted: AtomicU64::new(0),
            pools: CommandPools::new(family),
            timings: Mutex::new(RetainedTimings::new()),
        });
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
        self.queue(id).kind()
    }

    /// The name of the queue `id`: the most capable kind it serves, or
    /// `present` for one only presenting uses.
    pub(super) fn name(&self, id: QueueId) -> &'static str {
        self.queue(id).role.name()
    }

    /// The family of the queue `id`.
    pub(super) fn family(&self, id: QueueId) -> u32 {
        self.queue(id).family
    }

    /// The timestamp counter of the queue `id`, or `None` if it can't write
    /// timestamps.
    pub(super) fn clock(&self, id: QueueId) -> Option<TimestampClock> {
        self.queue(id).clock
    }

    /// The queue `id`.
    ///
    /// # Panics
    ///
    /// Panics if the device has no queue `id`, which only a timepoint from
    /// another device can name.
    fn queue(&self, id: QueueId) -> &DeviceQueue {
        self.queues
            .get(id.index())
            .unwrap_or_else(|| panic!("the device has no queue {id:?}"))
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

    /// Claims a command pool for the queue `id`, on `device` (see
    /// [`CommandPools::claim`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline can't be read, or a pool can't be
    /// reset or created.
    pub(super) fn claim_pool(&self, device: &ash::Device, id: QueueId) -> RhiResult<CommandPool> {
        self.queue(id)
            .pools
            .claim(device, || self.reached(device, id))
    }

    /// Recycles `pool` for the queue `id`, reusable once its timeline
    /// reaches `reusable_at`, or at once for 0.
    pub(super) fn recycle_pool(&self, id: QueueId, pool: CommandPool, reusable_at: u64) {
        self.queue(id).pools.recycle(pool, reusable_at);
    }

    /// Submits `desc` to the queue `id`, on `device`, and returns the timepoint
    /// its completion reaches. The command buffers' pools go back to the
    /// queue, reusable once it's reached.
    ///
    /// # Errors
    ///
    /// Returns an error if the submission fails.
    ///
    /// # Panics
    ///
    /// Panics if a command buffer was translated for another queue, or a
    /// timepoint to wait for wasn't returned by its queue.
    pub(super) fn submit(
        &self,
        device: &ash::Device,
        id: QueueId,
        desc: SubmitDesc<'_>,
    ) -> RhiResult<GpuTimepoint> {
        let queue = self.queue(id);
        let command_buffer_infos: Vec<_> = desc
            .command_buffers
            .iter()
            .map(|CommandBuffer(command_buffer)| {
                command_buffer.check_queue(id);
                vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer.buffer())
            })
            .collect();
        // Timepoints on this queue are waited for too: a queue starts its
        // submissions in order, but a later one may run before an earlier one
        // completes, and the wait is what makes the earlier one's writes
        // visible. Even a timepoint already reached is waited for, for its
        // writes.
        let waits: Vec<_> = desc
            .wait_for
            .iter()
            .map(|&timepoint| {
                let waited = self.queue(timepoint.queue());
                waited.check_returned(timepoint);
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(waited.timeline)
                    .value(timepoint.value())
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();

        let locked = lock(&self.locks.0[id.index()]);
        let value = queue.submitted.load(Ordering::Relaxed) + 1;
        let signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(queue.timeline)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&command_buffer_infos)
            .signal_semaphore_infos(std::slice::from_ref(&signal));
        // SAFETY: The queue, semaphores and command buffers were created on
        // `device`, the queue is locked, so no other thread uses it, each
        // command buffer was recorded for its family and has ended, and is
        // owned by `desc`, so it's not pending elsewhere, the info structures
        // outlive the call, and `value` is greater than any value signalled
        // or waited for on this timeline so far. Every value waited for was
        // returned by a submission, so its signal has been submitted.
        unsafe {
            device.queue_submit2(
                *locked,
                std::slice::from_ref(&submit_info),
                vk::Fence::null(),
            )
        }
        .map_err(|result| error(result, "submit to queue", queue.role.name()))?;
        queue.submitted.store(value, Ordering::Release);
        // The command buffers are given back once the queue is unlocked, so
        // other threads submitting to it don't wait on the pools' and the
        // deferred queue's locks.
        drop(locked);
        let timepoint = GpuTimepoint::new(id, value);
        VulkanCommandBuffer::all_submitted(desc.command_buffers, timepoint);
        Ok(timepoint)
    }

    /// Blocks until `timepoint`, which the queue `id` returned, is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting fails.
    ///
    /// # Panics
    ///
    /// Panics if that queue didn't return `timepoint`.
    pub(super) fn wait(
        &self,
        device: &ash::Device,
        id: QueueId,
        timepoint: GpuTimepoint,
    ) -> RhiResult<()> {
        let queue = self.check_own(id, timepoint);
        let value = timepoint.value();
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&queue.timeline))
            .values(std::slice::from_ref(&value));
        // SAFETY: The semaphore is a timeline created on `device`, and
        // `wait_info` is valid for the call. Waiting needs no lock.
        unsafe { device.wait_semaphores(&wait_info, u64::MAX) }
            .map_err(|result| error(result, "wait for queue", queue.role.name()))
    }

    /// Whether `timepoint`, which the queue `id` returned, has been reached.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline can't be read.
    ///
    /// # Panics
    ///
    /// Panics if that queue didn't return `timepoint`.
    pub(super) fn is_complete(
        &self,
        device: &ash::Device,
        id: QueueId,
        timepoint: GpuTimepoint,
    ) -> RhiResult<bool> {
        let queue = self.check_own(id, timepoint);
        // SAFETY: The semaphore is a timeline created on `device`.
        let reached = unsafe { device.get_semaphore_counter_value(queue.timeline) }
            .map_err(|result| error(result, "read the timeline of queue", queue.role.name()))?;
        Ok(reached >= timepoint.value())
    }

    /// Checks that `timepoint` is one the queue `id` returned, and returns the
    /// queue.
    ///
    /// # Panics
    ///
    /// Panics if it isn't.
    fn check_own(&self, id: QueueId, timepoint: GpuTimepoint) -> &DeviceQueue {
        let queue = self.queue(id);
        assert!(
            timepoint.queue() == id,
            "{timepoint:?} isn't on the {} queue's timeline",
            queue.role.name(),
        );
        queue.check_returned(timepoint);
        queue
    }

    /// Keeps `timings` for the submission to the queue `id` that reaches
    /// `value`, on `device`. If the queue then keeps too many, discards the
    /// oldest its timeline has reached, and returns them, with the values
    /// they reach, for their chunks to be recycled. The timeline is read only
    /// then, and if it can't be read, nothing is discarded until a later
    /// submission.
    pub(super) fn keep_timings(
        &self,
        device: &ash::Device,
        id: QueueId,
        value: u64,
        timings: KeptTimings,
    ) -> Vec<(u64, KeptTimings)> {
        let mut kept = lock_timings(&self.queue(id).timings);
        kept.insert(value, timings);
        if !kept.needs_pruning() {
            return Vec::new();
        }
        match self.reached(device, id) {
            Ok(reached) => kept.prune(reached),
            Err(_) => Vec::new(),
        }
    }

    /// Takes the timings kept for the submission to the queue `id` that
    /// reaches `value`, or `None` if none are kept.
    pub(super) fn take_kept_timings(&self, id: QueueId, value: u64) -> Option<KeptTimings> {
        lock_timings(&self.queue(id).timings).take(value)
    }

    /// Takes every queue's kept timings, for their chunks to be recycled
    /// once the device is idle.
    pub(super) fn drain_timings(&mut self) -> Vec<KeptTimings> {
        self.queues
            .iter_mut()
            .flat_map(|queue| {
                queue
                    .timings
                    .get_mut()
                    .unwrap_or_else(PoisonError::into_inner)
                    .drain()
            })
            .collect()
    }

    /// The value the timeline of the queue `id` has reached.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline can't be read.
    pub(super) fn reached(&self, device: &ash::Device, id: QueueId) -> RhiResult<u64> {
        let queue = self.queue(id);
        // SAFETY: The semaphore is a timeline created on `device`.
        unsafe { device.get_semaphore_counter_value(queue.timeline) }
            .map_err(|result| error(result, "read the timeline of queue", queue.role.name()))
    }

    /// The value each queue's timeline has reached, with a timeline that
    /// can't be read, as on a lost device, counting as having reached
    /// nothing.
    pub(super) fn all_reached(&self, device: &ash::Device) -> Vec<(QueueId, u64)> {
        (0..self.queues.len())
            .map(|index| {
                let id =
                    QueueId::new(u32::try_from(index).expect("a device has at most four queues"));
                (id, self.reached(device, id).unwrap_or(0))
            })
            .collect()
    }

    /// Destroys the timelines and command pools.
    ///
    /// # Safety
    ///
    /// They must have been created on `device`, no command pool may be
    /// claimed, and no pending submission may use them, for example
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

    /// Checks that `timepoint`, which is on this queue's timeline, was returned
    /// by a submission.
    ///
    /// # Panics
    ///
    /// Panics if no submission returned it.
    fn check_returned(&self, timepoint: GpuTimepoint) {
        let submitted = self.submitted.load(Ordering::Acquire);
        assert!(
            (1..=submitted).contains(&timepoint.value()),
            "the {} queue never returned {timepoint:?}: it has made {submitted} submissions",
            self.role.name(),
        );
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

/// Locks a queue's kept timings, recovering them if a thread panicked while
/// holding them: each call leaves them consistent before it can panic.
fn lock_timings(
    timings: &Mutex<RetainedTimings<KeptTimings>>,
) -> MutexGuard<'_, RetainedTimings<KeptTimings>> {
    timings.lock().unwrap_or_else(PoisonError::into_inner)
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

    fn timestamp_period_ns(&self) -> Option<f32> {
        self.core
            .queues
            .clock(self.id)
            .map(TimestampClock::period_ns)
    }

    fn translate(&self, commands: &CommandList) -> RhiResult<CommandBuffer> {
        self.core.translate(self.id, commands)
    }

    /// Then frees what retired objects it can, as design 0002's "Deferred
    /// destruction" describes, after submitting, so the GPU gets its work
    /// first.
    fn submit(&self, desc: SubmitDesc<'_>) -> RhiResult<GpuTimepoint> {
        let core = &self.core;
        let submitted = core.queues.submit(core.device.handle(), self.id, desc);
        core.collect_retired_if_needed();
        submitted
    }

    fn wait(&self, timepoint: GpuTimepoint) -> RhiResult<()> {
        self.core
            .queues
            .wait(self.core.device.handle(), self.id, timepoint)
    }

    fn is_complete(&self, timepoint: GpuTimepoint) -> RhiResult<bool> {
        self.core
            .queues
            .is_complete(self.core.device.handle(), self.id, timepoint)
    }

    fn take_timings(&self, timepoint: GpuTimepoint) -> RhiResult<Option<SubmissionTimings>> {
        self.core.take_timings(self.id, timepoint)
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
