//! Timestamp query pools, in chunks claimed by translation and reset from
//! the host when they're recycled (design 0002, "Timestamps").
//!
//! Each chunk is a Vulkan query pool of [`QUERIES_PER_CHUNK`] timestamp
//! queries. A translation that times its commands claims chunks as it needs
//! queries, and keeps them until the timings written into them have been
//! read back, or the command buffer is dropped unsubmitted. Then the chunk
//! is recycled: reset with `vkResetQueryPool` (Vulkan 1.2's
//! `hostQueryReset`), so command buffers need no reset commands, and put
//! back on the free list.
//!
//! A query pool isn't tied to a queue family, so one free list serves every
//! queue on the device.

#[cfg(test)]
mod gpu_tests;

use std::mem;
use std::sync::{Mutex, PoisonError};
use std::thread;

use ash::vk;

use super::error::error;
use crate::RhiResult;

/// How many timestamp queries a chunk holds: a scope's two timestamps for
/// 128 scopes, so a typical command list needs one chunk.
pub(super) const QUERIES_PER_CHUNK: u32 = 256;

/// A query pool of [`QUERIES_PER_CHUNK`] timestamp queries.
///
/// There's one for each Vulkan pool, moved rather than copied: it's on the
/// free list, or held by the one command buffer that claimed it, so only
/// that command buffer's thread uses the pool. Every query is reset while
/// it's on the free list.
///
/// A claimed chunk must be recycled, since only the free list's chunks are
/// destroyed with the device, so dropping one panics in debug builds.
#[derive(Debug)]
pub(super) struct QueryChunk {
    /// The pool.
    pool: vk::QueryPool,
}

impl QueryChunk {
    /// The pool, for destroying it, without the check for a dropped chunk.
    fn into_pool(self) -> vk::QueryPool {
        let pool = self.pool;
        mem::forget(self);
        pool
    }
}

impl Drop for QueryChunk {
    /// Panics in debug builds, since a chunk that isn't recycled leaks its
    /// pool, which validation reports only when the device is destroyed.
    /// Its owner recycles it when dropped, so one dropped while unwinding
    /// from a panic isn't reported again.
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking(),
            "a timestamp query chunk was dropped without being recycled, leaking its query pool",
        );
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "translation writes timestamps later in `rhi-timing`"
    )
)]
impl QueryChunk {
    /// The pool, whose queries are all reset when it's claimed.
    pub(super) fn pool(&self) -> vk::QueryPool {
        self.pool
    }
}

/// The device's timestamp query chunks that aren't claimed.
pub(super) struct QueryPools {
    /// The chunks not claimed, most recently recycled last. Every query in
    /// them is reset.
    free: Mutex<Vec<QueryChunk>>,
}

impl QueryPools {
    /// An empty free list.
    pub(super) fn new() -> Self {
        Self {
            free: Mutex::new(Vec::new()),
        }
    }

    /// Claims a chunk whose queries are all reset: the one most recently
    /// recycled, or else a new one.
    ///
    /// # Errors
    ///
    /// Returns an error if a query pool can't be created.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "translation writes timestamps later in `rhi-timing`"
        )
    )]
    pub(super) fn claim(&self, device: &ash::Device) -> RhiResult<QueryChunk> {
        let free = self
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        match free {
            Some(chunk) => Ok(chunk),
            None => Self::create(device),
        }
    }

    /// Resets every query in `chunk` from the host, and puts it back on the
    /// free list.
    ///
    /// # Safety
    ///
    /// `chunk` must have been claimed from this free list on `device`, and
    /// no pending work may use its queries: its command buffer was never
    /// submitted, or its submission has completed.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "translation writes timestamps later in `rhi-timing`"
        )
    )]
    pub(super) unsafe fn recycle(&self, device: &ash::Device, chunk: QueryChunk) {
        // SAFETY: Guaranteed by the caller, and the chunk is claimed, so no
        // other thread uses it.
        unsafe { reset(device, chunk.pool) };
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(chunk);
    }

    /// Creates a chunk, with every query reset, since a new pool's queries
    /// are in an undefined state until they are.
    ///
    /// # Errors
    ///
    /// Returns an error if the query pool can't be created.
    fn create(device: &ash::Device) -> RhiResult<QueryChunk> {
        let pool_create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(QUERIES_PER_CHUNK);
        // SAFETY: `pool_create_info` is valid for the call.
        let pool = unsafe { device.create_query_pool(&pool_create_info, None) }
            .map_err(|result| error(result, "create query pool", "timestamp queries"))?;
        // SAFETY: The pool was just created on `device`, so nothing uses it.
        unsafe { reset(device, pool) };
        Ok(QueryChunk { pool })
    }

    /// Destroys every chunk on the free list.
    ///
    /// # Safety
    ///
    /// The chunks must have been created on `device`, none may be claimed,
    /// and no pending work may use them, for example because the device is
    /// idle.
    pub(super) unsafe fn destroy(&mut self, device: &ash::Device) {
        let free = self.free.get_mut().unwrap_or_else(PoisonError::into_inner);
        for chunk in free.drain(..) {
            // SAFETY: Guaranteed by the caller.
            unsafe { device.destroy_query_pool(chunk.into_pool(), None) };
        }
    }

    /// How many chunks are on the free list, for tests.
    #[cfg(test)]
    pub(super) fn free_len(&self) -> usize {
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// Resets every query in `pool` from the host.
///
/// # Safety
///
/// `pool` must be a chunk's pool created on `device`, used by no other
/// thread, and no pending work may use its queries.
unsafe fn reset(device: &ash::Device, pool: vk::QueryPool) {
    // SAFETY: Guaranteed by the caller, and `hostQueryReset` is enabled on
    // every device.
    unsafe { device.reset_query_pool(pool, 0, QUERIES_PER_CHUNK) };
}

#[cfg(test)]
mod tests {
    //! T0 tests of the check for a chunk that's dropped rather than recycled.

    use ash::vk::Handle;

    use super::*;

    /// Dropping a claimed chunk panics in debug builds, since its pool
    /// would leak.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a timestamp query chunk was dropped without being recycled")]
    fn dropped_chunks_are_reported() {
        drop(QueryChunk {
            pool: vk::QueryPool::from_raw(1),
        });
    }

    /// Taking a chunk's pool to destroy it isn't a dropped chunk.
    #[test]
    fn taking_the_pool_is_not_a_drop() {
        let chunk = QueryChunk {
            pool: vk::QueryPool::from_raw(1),
        };
        assert_eq!(chunk.into_pool(), vk::QueryPool::from_raw(1));
    }
}
