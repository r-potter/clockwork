//! Transient memory in the Vulkan backend (design 0002, "Transient memory",
//! R23).
//!
//! Each chunk is a buffer in upload memory, mapped persistently through its
//! allocation, with storage, index and indirect usage, so transient data
//! can be read through pointers, bound as indices, or used as indirect
//! arguments. Chunks live outside the buffer table, since nothing names
//! them by handle.
//!
//! A chunk must start at a multiple of [`MAX_TRANSIENT_ALIGNMENT`]. A chunk
//! of [`CHUNK_SIZE`] is a range of a memory block, which the allocator
//! places at that alignment. A larger one may have memory of its own, at an
//! address only the driver chooses, so it's created with room to start at
//! the first aligned address in its buffer.
//!
//! The [`VulkanDevice`] is the frames' [`ChunkSource`]. While a frame holds
//! a chunk, it's in the in-use map, keyed by its address, which is how the
//! frame gives it back. When the frame ends, the pool records its end at
//! once, and its chunks are retired into the deferred-destruction queue,
//! like any destroyed object. Once nothing translated before then is
//! outstanding, chunks of [`CHUNK_SIZE`] go back to the pool, and larger
//! ones are freed.
//!
//! Lock order: the transient lock is taken on its own. Chunks are created,
//! and those it hands back are freed, after it's released, so it's never
//! held with the allocator's, the deferred queue's, or a table's.

#[cfg(test)]
mod gpu_tests;

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{MutexGuard, PoisonError};

use super::VulkanDevice;
use super::buffers::BufferEntry;
use super::destruction::Retired;
use crate::transient::pool::ChunkPool;
use crate::transient::{
    CHUNK_SIZE, ChunkSource, MAX_TRANSIENT_ALIGNMENT, RetiredChunks, TransientChunk,
};
use crate::{BufferDesc, BufferUsage, MemoryLocation, RhiResult};

/// The backend's transient chunks: those free in the pool, and those frames
/// hold.
#[derive(Debug)]
pub(super) struct TransientChunks {
    /// The free chunks of [`CHUNK_SIZE`], and how many recent frames used.
    pool: ChunkPool<BufferEntry>,
    /// The chunks frames hold, of every size, by the address frames know
    /// them by.
    in_use: HashMap<u64, BufferEntry>,
    /// How many chunks of [`CHUNK_SIZE`] have been created, which tests
    /// check to tell reuse from churn.
    created: usize,
}

/// How many transient chunks a device has, for tests to check the pool's
/// size.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TransientCounts {
    /// Chunks free in the pool.
    pub(super) free: usize,
    /// Chunks frames hold, of every size.
    pub(super) in_use: usize,
    /// Chunks of [`CHUNK_SIZE`] ever created.
    pub(super) created: usize,
}

impl TransientChunks {
    /// No chunks.
    pub(super) fn new() -> Self {
        Self {
            pool: ChunkPool::new(),
            in_use: HashMap::new(),
            created: 0,
        }
    }

    /// Hands `entry` to a frame, as a chunk from the first address in it
    /// that's a multiple of [`MAX_TRANSIENT_ALIGNMENT`] to its end, and keeps
    /// it in the in-use map until the frame gives it back.
    fn hold(&mut self, entry: BufferEntry) -> TransientChunk {
        let skip = alignment_padding(entry.address().0);
        let mapped = entry.mapped().expect("upload memory is mapped").as_ptr();
        let chunk = TransientChunk {
            address: crate::GpuAddress(entry.address().0 + skip),
            // The padding is inside the buffer, which is mapped whole, so
            // the pointer stays in the mapping, and isn't null.
            mapped: NonNull::new(mapped.wrapping_add(
                usize::try_from(skip).expect("the padding is less than the alignment"),
            ))
            .expect("a mapping doesn't wrap the address space"),
            size: entry.size().saturating_sub(skip),
        };
        self.in_use.insert(chunk.address.0, entry);
        chunk
    }
}

impl VulkanDevice {
    /// Creates a chunk to hold `size` bytes from an address that's a
    /// multiple of [`MAX_TRANSIENT_ALIGNMENT`]. A chunk of [`CHUNK_SIZE`] is
    /// a range of a block, aligned by the allocator. A larger one has room
    /// to start at an aligned address wherever its memory is.
    ///
    /// # Errors
    ///
    /// As [`create_buffer_object`](Self::create_buffer_object).
    fn create_chunk(&self, size: u64) -> RhiResult<BufferEntry> {
        let padding = if size == CHUNK_SIZE {
            0
        } else {
            MAX_TRANSIENT_ALIGNMENT - 1
        };
        self.create_buffer_object(
            &BufferDesc {
                label: "transient chunk",
                size: size + padding,
                usage: BufferUsage::STORAGE | BufferUsage::INDEX | BufferUsage::INDIRECT,
                location: MemoryLocation::Upload,
            },
            MAX_TRANSIENT_ALIGNMENT,
        )
    }

    /// Takes back chunks retired when their frame ended, now that nothing
    /// can use them: chunks of [`CHUNK_SIZE`] go back to the pool, and the
    /// rest, and any the pool no longer needs, are freed.
    ///
    /// # Safety
    ///
    /// No submitted or translated work may still use the chunks.
    pub(super) unsafe fn recycle_transient_chunks(
        &self,
        chunks: Vec<BufferEntry>,
        large: Vec<BufferEntry>,
    ) {
        let released = self.lock_transient().pool.recycle(chunks);
        for chunk in released.into_iter().chain(large) {
            // SAFETY: Guaranteed by the caller.
            unsafe { self.free_buffer(chunk) };
        }
    }

    /// Frees every chunk in the pool, when the core is dropped, after the
    /// retired chunks have come back to it.
    ///
    /// # Safety
    ///
    /// The device must be idle.
    pub(super) unsafe fn free_transient_chunks(&mut self) {
        let transient = self
            .transient
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        // Each frame holds the core, so none is left to hold a chunk.
        debug_assert!(transient.in_use.is_empty());
        let chunks: Vec<_> = transient
            .pool
            .drain()
            .into_iter()
            .chain(transient.in_use.drain().map(|(_, chunk)| chunk))
            .collect();
        for chunk in chunks {
            // SAFETY: Guaranteed by the caller.
            unsafe { self.free_buffer(chunk) };
        }
    }

    /// How many transient chunks the device has, for tests to check the
    /// pool's size.
    #[cfg(test)]
    pub(super) fn transient_counts(&self) -> TransientCounts {
        let transient = self.lock_transient();
        TransientCounts {
            free: transient.pool.free_len(),
            in_use: transient.in_use.len(),
            created: transient.created,
        }
    }

    /// Locks the transient chunks, recovering them if a panic poisoned them:
    /// no operation on them panics halfway through, so they're consistent.
    fn lock_transient(&self) -> MutexGuard<'_, TransientChunks> {
        self.transient
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

// SAFETY: Each chunk is a range of a buffer in upload memory, mapped whole
// for as long as the buffer lives, and the buffer lives at least until the
// chunk is retired. While a frame holds a chunk, its buffer is in the in-use
// map, not the pool, so it's handed to no other frame, and the backend never
// touches its memory on the CPU.
unsafe impl ChunkSource for VulkanDevice {
    fn take_chunk(&self, size: u64) -> RhiResult<TransientChunk> {
        if size == CHUNK_SIZE {
            let mut transient = self.lock_transient();
            if let Some(entry) = transient.pool.take() {
                return Ok(transient.hold(entry));
            }
        }
        // Creating a chunk may allocate a block, so it's done without the
        // lock, which frames taking chunks and submissions recycling them
        // need.
        let entry = self.create_chunk(size)?;
        let mut transient = self.lock_transient();
        if size == CHUNK_SIZE {
            transient.pool.created();
            transient.created += 1;
        }
        Ok(transient.hold(entry))
    }

    fn retire_chunks(&self, retired: RetiredChunks) {
        let (released, chunks, large) = {
            let mut transient = self.lock_transient();
            let mut release = |chunk: TransientChunk| {
                transient
                    .in_use
                    .remove(&chunk.address.0)
                    .expect("a frame gives back only chunks this device handed it")
            };
            let chunks: Vec<_> = retired.chunks.into_iter().map(&mut release).collect();
            let large: Vec<_> = retired.large.into_iter().map(release).collect();
            (transient.pool.frame_ended(), chunks, large)
        };
        for chunk in released {
            // SAFETY: The pool keeps only chunks that were recycled, once
            // nothing could use them, and no frame has taken this one since.
            unsafe { self.free_buffer(chunk) };
        }
        // An idle frame ends no epoch.
        if !chunks.is_empty() || !large.is_empty() {
            self.retire(Retired::TransientChunks { chunks, large });
        }
    }
}

/// How far past `address` the first multiple of [`MAX_TRANSIENT_ALIGNMENT`]
/// is.
fn alignment_padding(address: u64) -> u64 {
    address.next_multiple_of(MAX_TRANSIENT_ALIGNMENT) - address
}

#[cfg(test)]
mod tests {
    //! T0 tests of placing chunks at aligned addresses.

    use super::*;

    /// An aligned address needs no padding, and any other is padded to the
    /// next multiple of the alignment, never by as much as the alignment.
    #[test]
    fn padding_reaches_the_next_aligned_address() {
        assert_eq!(alignment_padding(0), 0);
        assert_eq!(alignment_padding(4 * MAX_TRANSIENT_ALIGNMENT), 0);
        assert_eq!(alignment_padding(64), MAX_TRANSIENT_ALIGNMENT - 64);
        assert_eq!(
            alignment_padding(4 * MAX_TRANSIENT_ALIGNMENT + 1),
            MAX_TRANSIENT_ALIGNMENT - 1
        );
    }
}
