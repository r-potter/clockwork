//! Deferred destruction in the Vulkan backend (design 0002, "Deferred
//! destruction", R10).
//!
//! Destroying a buffer, texture or sampler takes it out of its table, so
//! its handle is stale at once, and retires it into the core's
//! [`DeferredQueue`]. It's freed once no command buffer translated before
//! the destruction is outstanding: at the start of each submission, in
//! [`GpuResources::collect_garbage`](crate::GpuResources::collect_garbage),
//! or when the core is dropped, once the device is idle.
//!
//! Lock order: the deferred queue's lock is taken on its own, or inside a
//! queue's lock when a submission gives back its command buffers. It's
//! never held while taking a table's lock, and what it returns is freed
//! after it's released.

#[cfg(test)]
mod gpu_tests;

use std::sync::{MutexGuard, PoisonError};

use super::VulkanDevice;
use super::buffers::BufferEntry;
use super::samplers::SamplerEntry;
use super::textures::TextureEntry;
use crate::GpuPoint;
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

    /// Gives back a translated command buffer: submitted in the submission
    /// that reaches `point`, or dropped unsubmitted if it's `None`.
    pub(super) fn end_translation(&self, translation: Translation, point: Option<GpuPoint>) {
        let mut deferred = self.lock_deferred();
        match point {
            Some(point) => deferred.submitted(translation, point),
            None => deferred.dropped(translation),
        }
    }

    /// Frees every retired object that nothing outstanding can use.
    ///
    /// A timeline that can't be read counts as having reached nothing, so
    /// on a lost device nothing more is freed until the core is dropped.
    pub(super) fn collect_retired(&self) {
        let device = self.device.handle();
        let freed = self
            .lock_deferred()
            .collect(|kind| self.queues.reached(device, kind).unwrap_or(0));
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
