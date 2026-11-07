//! Deferred destruction in the Vulkan backend (design 0002, "Deferred
//! destruction", R10).
//!
//! Destroying a buffer, texture or sampler takes it out of its table, so
//! its handle is stale at once, and retires it into the core's
//! [`DeferredQueue`]. It's freed once no command buffer translated before
//! the destruction is outstanding: after a submission, once its work is
//! submitted, if anything is retired or enough submissions are pending to
//! prune; in
//! [`GpuResources::collect_garbage`](crate::GpuResources::collect_garbage);
//! or when the core is dropped, once the device is idle.
//!
//! Lock order: the deferred queue's lock is taken on its own, never while
//! holding a queue's or a table's lock, and what it returns is freed after
//! it's released.

#[cfg(test)]
mod gpu_tests;

use std::sync::{MutexGuard, PoisonError};

use tracing::debug;

use super::VulkanDevice;
use super::buffers::BufferEntry;
use super::samplers::SamplerEntry;
use super::textures::TextureEntry;
use crate::GpuTimepoint;
use crate::deferred::{DeferredQueue, Translation};

/// A destroyed object waiting to be freed.
#[derive(Debug)]
pub(super) enum Retired {
    /// A buffer and its memory.
    Buffer(BufferEntry),
    /// A texture and its memory.
    Texture(TextureEntry),
    /// A sampler's slot in the sampler heap.
    Sampler(SamplerEntry),
}

impl VulkanDevice {
    /// Retires `object`, taken out of its table, to be freed once nothing
    /// translated before now can use it.
    pub(super) fn retire(&self, object: Retired) {
        self.lock_deferred().retire(object);
    }

    /// Starts a translation in the current epoch. Call it before looking up
    /// any handle the list names.
    pub(super) fn begin_translation(&self) -> Translation {
        self.lock_deferred().translating()
    }

    /// Gives back a command buffer dropped unsubmitted.
    pub(super) fn end_dropped_translation(&self, translation: Translation) {
        self.lock_deferred().dropped(translation);
    }

    /// Gives back the command buffers of the submission that reaches
    /// `timepoint`, taking the deferred queue's lock once for all of them.
    pub(super) fn end_submitted_translations(
        &self,
        translations: impl IntoIterator<Item = Translation>,
        timepoint: GpuTimepoint,
    ) {
        let mut deferred = self.lock_deferred();
        for translation in translations {
            deferred.submitted(translation, timepoint);
        }
    }

    /// Frees every retired object that nothing outstanding can use, if
    /// anything is retired, as each submission does. With nothing retired,
    /// and few pending submissions to prune, it reads no timelines.
    pub(super) fn collect_retired_if_needed(&self) {
        if self.lock_deferred().needs_collecting() {
            self.collect_retired();
        }
    }

    /// Frees every retired object that nothing outstanding can use, then
    /// logs at `DEBUG` how many are left and the oldest epoch holding them
    /// back (design 0002, "Risks"), as
    /// [`GpuResources::collect_garbage`](crate::GpuResources::collect_garbage)
    /// does once a frame.
    pub(super) fn collect_garbage_and_report(&self) {
        self.collect_retired();
        let deferred = self.lock_deferred();
        debug!(
            retired = deferred.retired_len(),
            oldest_outstanding_epoch = ?deferred.oldest_outstanding(),
            "deferred destruction",
        );
    }

    /// Frees every retired object that nothing outstanding can use.
    ///
    /// The timelines are read before the deferred queue is locked, so
    /// translations starting meanwhile don't wait on the driver. A timeline
    /// that can't be read counts as having reached nothing, so on a lost
    /// device nothing more is freed until the core is dropped.
    pub(super) fn collect_retired(&self) {
        let reached = self.queues.all_reached(self.device.handle());
        let freed = self.lock_deferred().collect(&reached);
        for object in freed {
            // SAFETY: The deferred queue returned it, so no outstanding
            // command buffer can use it.
            unsafe { self.free_retired(object) };
        }
    }

    /// Frees every retired object, when the core is dropped.
    ///
    /// # Safety
    ///
    /// The device must be idle.
    pub(super) unsafe fn free_all_retired(&mut self) {
        let retired = self
            .deferred
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .drain();
        for object in retired {
            // SAFETY: Guaranteed by the caller, and nothing can translate,
            // since the core is being dropped.
            unsafe { self.free_retired(object) };
        }
    }

    /// Frees a retired object.
    ///
    /// # Safety
    ///
    /// No submitted or translated work may still use it.
    unsafe fn free_retired(&self, object: Retired) {
        match object {
            // SAFETY: Guaranteed by the caller.
            Retired::Buffer(entry) => unsafe { self.free_buffer(entry) },
            // SAFETY: Guaranteed by the caller.
            Retired::Texture(entry) => unsafe { self.free_texture(entry) },
            Retired::Sampler(entry) => self.free_sampler(entry),
        }
    }

    /// Locks the deferred queue, recovering it if a panic poisoned it: no
    /// queue operation panics halfway through, so it's consistent.
    pub(super) fn lock_deferred(&self) -> MutexGuard<'_, DeferredQueue<Retired>> {
        self.deferred.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
