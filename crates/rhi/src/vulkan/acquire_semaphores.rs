//! The semaphores that acquiring a swapchain image signals, and the
//! submission that renders to it waits for (design 0002, "Presentation").
//!
//! A binary semaphore can be signalled again only once the wait on its last
//! signal has completed. Frames in flight are the application's choice, so
//! rather than one semaphore per frame, guarded by a fence, each semaphore
//! remembers the timepoint of the submission that waited for it, and is
//! reused once that's reached. A new one is created when none is free, so
//! the pool holds about as many as the application keeps frames in flight.
//!
//! A semaphore an acquire signalled, but no submission waited for, because
//! submitting failed, or the swapchain was destroyed first, is never reused,
//! since its signal may still be pending. It's destroyed with the pool.

use anyhow::Context;
use ash::vk;

use super::device::LogicalDevice;
use super::queues::QueueLocks;
use crate::{GpuTimepoint, RhiResult};

/// The acquire semaphores, and what each is doing.
///
/// Destroyed when dropped, once the device is idle, so the owner must drop it
/// before the logical device it was created from.
pub(super) struct AcquireSemaphores {
    /// Kept to create and destroy the semaphores.
    device: ash::Device,
    /// The device's queues, locked while waiting for it to be idle.
    queue_locks: QueueLocks,
    /// Each semaphore, at the index of its slot in `slots`.
    semaphores: Vec<vk::Semaphore>,
    /// What each semaphore is doing.
    slots: Slots,
}

impl AcquireSemaphores {
    /// An empty pool for `device`.
    pub(super) fn new(device: &LogicalDevice) -> Self {
        Self {
            device: device.handle().clone(),
            queue_locks: device.queue_locks().clone(),
            semaphores: Vec::new(),
            slots: Slots::default(),
        }
    }

    /// A semaphore for an acquire to signal, and its index, to pass to
    /// [`waited`](Self::waited) or [`unsignalled`](Self::unsignalled). It's
    /// one whose last wait is complete, as `is_reached` says, or a new one.
    ///
    /// # Errors
    ///
    /// Returns an error if `is_reached` does, or creating a semaphore fails.
    pub(super) fn take(
        &mut self,
        is_reached: impl FnMut(GpuTimepoint) -> RhiResult<bool>,
    ) -> RhiResult<(usize, vk::Semaphore)> {
        if let Some(index) = self.slots.take(is_reached)? {
            return Ok((index, self.semaphores[index]));
        }
        // SAFETY: The create info is valid for the call.
        let semaphore = unsafe {
            self.device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        }
        .context("failed to create a Vulkan semaphore to acquire swapchain images with")?;
        self.semaphores.push(semaphore);
        Ok((self.slots.push_taken(), semaphore))
    }

    /// Semaphore `index`.
    pub(super) fn semaphore(&self, index: usize) -> vk::Semaphore {
        self.semaphores[index]
    }

    /// Records that semaphore `index` was waited for by the submission that
    /// reaches `timepoint`, so it's reused once that's reached.
    pub(super) fn waited(&mut self, index: usize, timepoint: GpuTimepoint) {
        self.slots.waited(index, timepoint);
    }

    /// Records that the acquire semaphore `index` was taken for didn't
    /// signal it, so it's free straight away.
    pub(super) fn unsignalled(&mut self, index: usize) {
        self.slots.unsignalled(index);
    }
}

impl Drop for AcquireSemaphores {
    fn drop(&mut self) {
        // SAFETY: The semaphores were created on this device, and are
        // destroyed exactly once here, once the device, whose queues these
        // are, is idle, so no submission still waits for one.
        unsafe {
            // Nothing useful can be done if this fails while dropping.
            let _ = self.queue_locks.wait_idle(&self.device);
            for &semaphore in &self.semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
        }
    }
}

/// What a semaphore in the pool is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// Free to be signalled by an acquire once the submission that last
    /// waited for it, if any, reaches its timepoint.
    Free(Option<GpuTimepoint>),
    /// Taken by an acquire, and not yet waited for. A semaphore that's never
    /// waited for stays taken.
    Taken,
}

/// The pool's bookkeeping, apart from the semaphores themselves, so it's
/// tested without a device.
#[derive(Debug, Default)]
struct Slots(Vec<Slot>);

impl Slots {
    /// Takes the first free slot whose last wait is complete, as
    /// `is_reached` says, or returns `None` if there's none.
    ///
    /// # Errors
    ///
    /// Returns an error if `is_reached` does.
    fn take(
        &mut self,
        mut is_reached: impl FnMut(GpuTimepoint) -> RhiResult<bool>,
    ) -> RhiResult<Option<usize>> {
        for (index, slot) in self.0.iter_mut().enumerate() {
            let free = match *slot {
                Slot::Free(None) => true,
                Slot::Free(Some(timepoint)) => is_reached(timepoint)?,
                Slot::Taken => false,
            };
            if free {
                *slot = Slot::Taken;
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    /// Adds a slot, taken, and returns its index.
    fn push_taken(&mut self) -> usize {
        self.0.push(Slot::Taken);
        self.0.len() - 1
    }

    /// Frees taken slot `index`, once `timepoint` is reached.
    ///
    /// # Panics
    ///
    /// Panics if the slot isn't taken.
    fn waited(&mut self, index: usize, timepoint: GpuTimepoint) {
        self.free(index, Some(timepoint));
    }

    /// Frees taken slot `index` straight away.
    ///
    /// # Panics
    ///
    /// Panics if the slot isn't taken.
    fn unsignalled(&mut self, index: usize) {
        self.free(index, None);
    }

    /// Frees taken slot `index`, once `timepoint`, if any, is reached.
    ///
    /// # Panics
    ///
    /// Panics if the slot isn't taken.
    fn free(&mut self, index: usize, timepoint: Option<GpuTimepoint>) {
        let slot = &mut self.0[index];
        assert_eq!(*slot, Slot::Taken, "acquire semaphore {index} isn't taken");
        *slot = Slot::Free(timepoint);
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the acquire semaphores' bookkeeping.

    use super::*;
    use crate::QueueId;

    /// The graphics queue's timepoint `value`, for tests.
    fn timepoint(value: u64) -> GpuTimepoint {
        GpuTimepoint::new(QueueId::new(0), value)
    }

    /// A timeline that has reached `reached`.
    fn reached(reached: u64) -> impl FnMut(GpuTimepoint) -> RhiResult<bool> {
        move |timepoint| Ok(timepoint.value() <= reached)
    }

    /// An empty pool has nothing to take.
    #[test]
    fn empty_pools_have_nothing_free() {
        assert_eq!(Slots::default().take(reached(0)).unwrap(), None);
    }

    /// A slot is reused once the submission that waited for it completes,
    /// and not before.
    #[test]
    fn slots_are_reused_once_their_wait_completes() {
        let mut slots = Slots::default();
        let first = slots.push_taken();
        slots.waited(first, timepoint(5));
        assert_eq!(slots.take(reached(4)).unwrap(), None);
        assert_eq!(slots.take(reached(5)).unwrap(), Some(first));
        assert_eq!(slots.take(reached(5)).unwrap(), None, "it's taken again");
    }

    /// With two frames in flight, the pool holds two slots, each reused
    /// once its frame completes.
    #[test]
    fn frames_in_flight_take_turns() {
        let mut slots = Slots::default();
        let first = slots.push_taken();
        slots.waited(first, timepoint(1));
        assert_eq!(slots.take(reached(0)).unwrap(), None);
        let second = slots.push_taken();
        slots.waited(second, timepoint(2));
        assert_eq!(slots.take(reached(1)).unwrap(), Some(first));
        slots.waited(first, timepoint(3));
        assert_eq!(slots.take(reached(2)).unwrap(), Some(second));
    }

    /// A slot an acquire didn't signal is free straight away, without
    /// reading the timeline.
    #[test]
    fn unsignalled_slots_are_free_at_once() {
        let mut slots = Slots::default();
        let index = slots.push_taken();
        slots.unsignalled(index);
        let take = slots.take(|_| panic!("the timeline isn't read"));
        assert_eq!(take.unwrap(), Some(index));
    }

    /// A slot never waited for stays taken.
    #[test]
    fn slots_never_waited_for_stay_taken() {
        let mut slots = Slots::default();
        slots.push_taken();
        assert_eq!(slots.take(reached(u64::MAX)).unwrap(), None);
    }

    /// An error reading the timeline is returned, taking nothing.
    #[test]
    fn timeline_errors_are_returned() {
        let mut slots = Slots::default();
        let index = slots.push_taken();
        slots.waited(index, timepoint(1));
        let taken = slots.take(|_| Err(crate::RhiError::DeviceLost));
        assert!(matches!(taken, Err(crate::RhiError::DeviceLost)));
        assert_eq!(slots.take(reached(1)).unwrap(), Some(index));
    }

    /// Freeing a slot that isn't taken is a bug in the backend.
    #[test]
    #[should_panic(expected = "acquire semaphore 0 isn't taken")]
    fn freeing_free_slots_panics() {
        let mut slots = Slots::default();
        let index = slots.push_taken();
        slots.unsignalled(index);
        slots.unsignalled(index);
    }
}
