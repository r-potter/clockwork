//! Per-frame transient memory: a linear allocator over upload memory, shared
//! by every thread recording a frame (design 0002, "Transient memory", R23).
//!
//! A [`TransientFrame`] bumps an offset through its current chunk, with a
//! compare-and-swap, so allocations from many threads take no lock. When the
//! chunk is full, the thread that finds it so takes the frame's lock, gets
//! a new chunk from the backend's [`ChunkSource`], and makes it current.
//! Chunks are kept in slots that are each written once, so a thread that
//! read the current index can use its chunk without the lock.
//!
//! An allocation larger than [`CHUNK_SIZE`] gets a chunk of its own, which
//! never becomes current.
//!
//! When the frame ends, or is dropped, its chunks go back to the source,
//! which retires them until nothing translated in the frame is outstanding,
//! then keeps those of [`CHUNK_SIZE`] in its [`ChunkPool`](pool::ChunkPool)
//! for later frames. A frame that used no chunks ends too, so that the pool
//! sees every frame.
//!
//! The frame is pure logic over chunks the source describes, so it's tested
//! with a fake source over host memory.

pub(crate) mod pool;

use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::{mem, slice};

use anyhow::anyhow;

use crate::{GpuAddress, RhiError, RhiResult};

/// The size of a chunk, in bytes, unless one allocation needs more.
pub(crate) const CHUNK_SIZE: u64 = 4 << 20;

/// The largest alignment a transient allocation can ask for, in bytes.
/// Chunks start at a multiple of it, so an aligned offset is an aligned
/// address, and an allocation as large as a chunk always fits in an empty
/// one. It's the largest alignment Vulkan lets a device require of a
/// storage buffer's offset (`minStorageBufferOffsetAlignment`).
pub const MAX_TRANSIENT_ALIGNMENT: u64 = 256;

/// The most chunks of [`CHUNK_SIZE`] one frame can use: 1 GiB in all.
/// Allocations larger than a chunk don't count.
const MAX_CHUNKS_PER_FRAME: usize = 256;

/// How many chunk slots a [`Segment`] holds.
const SEGMENT_LEN: usize = 16;

/// A run of a frame's chunk slots, allocated when the frame first needs one
/// of them, so a frame that uses one chunk allocates one segment, not
/// [`MAX_CHUNKS_PER_FRAME`] slots.
type Segment = [OnceLock<CurrentChunk>; SEGMENT_LEN];

/// A buffer of upload memory, persistently mapped, that a frame allocates
/// from. The source that made it knows it by its address.
#[derive(Debug)]
pub(crate) struct TransientChunk {
    /// The buffer's address on the GPU, a multiple of
    /// [`MAX_TRANSIENT_ALIGNMENT`].
    pub(crate) address: GpuAddress,
    /// Where the buffer is mapped for the CPU to write, for all of `size`.
    pub(crate) mapped: NonNull<u8>,
    /// Its size, in bytes.
    pub(crate) size: u64,
}

// SAFETY: A chunk only describes memory. Its mapping can be written from any
// thread, and the frame hands each allocation a disjoint range of it.
unsafe impl Send for TransientChunk {}
// SAFETY: As above. Sharing a chunk only shares its description.
unsafe impl Sync for TransientChunk {}

/// Where a frame's chunks come from, and go back to: the backend's pool.
///
/// # Safety
///
/// [`TransientAllocation::bytes_mut`] writes through a chunk's mapping with
/// no other check, so each chunk [`take_chunk`](Self::take_chunk) returns
/// must be mapped for writing at `mapped` for its whole `size`, and nothing
/// else may read or write that memory on the CPU, or hold it in another
/// chunk, until the chunk is given back through
/// [`retire_chunks`](Self::retire_chunks).
pub(crate) unsafe trait ChunkSource: Debug + Send + Sync {
    /// A chunk of at least `size` bytes, from the pool or newly created,
    /// at an address that's a multiple of [`MAX_TRANSIENT_ALIGNMENT`].
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`] if device memory runs out, or
    /// [`RhiError::Other`] if a chunk can't be created otherwise.
    fn take_chunk(&self, size: u64) -> RhiResult<TransientChunk>;

    /// Takes back the chunks of a frame that has ended, to reuse once
    /// nothing translated before now is outstanding.
    ///
    /// Called once for every frame, even one that took no chunks, so that
    /// the source can record the frame's end with its pool's
    /// [`frame_ended`](pool::ChunkPool::frame_ended) at once. Only the
    /// chunks wait for the deferred-destruction queue, and a frame with
    /// none needs nothing retired there, where each retirement ends an
    /// epoch.
    fn retire_chunks(&self, retired: RetiredChunks);
}

/// The chunks a frame took, given back to its source when it ends.
#[derive(Debug)]
pub(crate) struct RetiredChunks {
    /// Its chunks of [`CHUNK_SIZE`], in the order it took them, for the
    /// source's pool to reuse. How many there are is how many the frame
    /// used.
    pub(crate) chunks: Vec<TransientChunk>,
    /// The chunks of its allocations larger than [`CHUNK_SIZE`], for the
    /// source to free.
    pub(crate) large: Vec<TransientChunk>,
}

/// Checks that a chunk taken for `size` bytes is as large and as aligned as
/// allocations rely on, so that a source that gets it wrong fails at once,
/// rather than handing out misaligned addresses.
///
/// # Panics
///
/// Panics if the chunk is smaller than `size`, or its address isn't a
/// multiple of [`MAX_TRANSIENT_ALIGNMENT`].
fn check_chunk(chunk: &TransientChunk, size: u64) {
    assert!(
        chunk.size >= size,
        "a transient chunk of {} bytes was taken for {size}",
        chunk.size,
    );
    assert!(
        chunk.address.0.is_multiple_of(MAX_TRANSIENT_ALIGNMENT),
        "a transient chunk's address, {:#x}, isn't a multiple of {MAX_TRANSIENT_ALIGNMENT}",
        chunk.address.0,
    );
}

/// A chunk of [`CHUNK_SIZE`] in a frame, and how much of it is allocated.
#[derive(Debug)]
struct CurrentChunk {
    /// The chunk.
    chunk: TransientChunk,
    /// The offset of its first unallocated byte.
    offset: AtomicU64,
}

impl CurrentChunk {
    /// Allocates `size` bytes at a multiple of `align`, returning their
    /// offset, or `None` if they don't fit in what's left.
    fn bump(&self, size: u64, align: u64) -> Option<u64> {
        let mut offset = self.offset.load(Ordering::Relaxed);
        loop {
            let start = offset.next_multiple_of(align);
            let end = start.checked_add(size)?;
            if end > self.chunk.size {
                return None;
            }
            // The offset only orders allocations against each other, and the
            // memory they write isn't read on the CPU, so it needs no
            // stronger ordering.
            match self.offset.compare_exchange_weak(
                offset,
                end,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(start),
                Err(actual) => offset = actual,
            }
        }
    }
}

/// One frame's transient allocations. Shared by every thread recording the
/// frame, and ended, by dropping it, once every list that uses its
/// allocations is translated. Its memory is reused once nothing translated
/// before then is outstanding.
pub struct TransientFrame {
    /// Where its chunks come from.
    source: Arc<dyn ChunkSource>,
    /// Its chunks of [`CHUNK_SIZE`], in the order it took them, in
    /// segments. Each segment and slot is set once, under `taking`, and the
    /// last slot set is `current`.
    segments: [OnceLock<Box<Segment>>; MAX_CHUNKS_PER_FRAME / SEGMENT_LEN],
    /// The index in `chunks` of the chunk allocations bump through. Its slot
    /// is empty until the first allocation.
    current: AtomicUsize,
    /// Held while taking a chunk, and holding the chunks of allocations
    /// larger than [`CHUNK_SIZE`].
    taking: Mutex<Vec<TransientChunk>>,
}

impl TransientFrame {
    /// A frame with no chunks yet, taking them from `source`.
    pub(crate) fn new(source: Arc<dyn ChunkSource>) -> Self {
        Self {
            source,
            segments: Default::default(),
            current: AtomicUsize::new(0),
            taking: Mutex::new(Vec::new()),
        }
    }

    /// Allocates `size` bytes at a multiple of `align`, for this frame only.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`] if a new chunk can't be
    /// allocated, or [`RhiError::Other`] if one can't be created otherwise,
    /// or the frame has used its 1 GiB of chunks.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero, or `align` isn't a power of two up to
    /// [`MAX_TRANSIENT_ALIGNMENT`].
    pub fn allocate(&self, size: u64, align: u64) -> RhiResult<TransientAllocation<'_>> {
        assert!(size > 0, "a transient allocation has a size of zero");
        assert!(
            align.is_power_of_two() && align <= MAX_TRANSIENT_ALIGNMENT,
            "a transient allocation's alignment, {align}, isn't a power of two up to \
             {MAX_TRANSIENT_ALIGNMENT}",
        );
        if size > CHUNK_SIZE {
            return self.allocate_large(size);
        }
        loop {
            let index = self.current.load(Ordering::Acquire);
            let current = self.chunk(index);
            if let Some(current) = current
                && let Some(offset) = current.bump(size, align)
            {
                return Ok(TransientAllocation::new(&current.chunk, offset, size));
            }
            self.take_next_chunk(index, current.is_some())?;
        }
    }

    /// Makes a new chunk current, unless another thread already has since
    /// this one found the chunk at `index` full, or, if `was_set` is false,
    /// found its slot empty.
    ///
    /// # Errors
    ///
    /// Returns the source's error, or [`RhiError::Other`] if the frame has
    /// no slot left.
    fn take_next_chunk(&self, index: usize, was_set: bool) -> RhiResult<()> {
        let _taking = self.taking.lock().unwrap_or_else(PoisonError::into_inner);
        if self.current.load(Ordering::Acquire) != index || self.chunk(index).is_some() != was_set {
            return Ok(());
        }
        let next = if was_set { index + 1 } else { index };
        let Some(segment) = self.segments.get(next / SEGMENT_LEN) else {
            return Err(RhiError::Other(anyhow!(
                "a transient frame used more than {MAX_CHUNKS_PER_FRAME} chunks of \
                 {CHUNK_SIZE} bytes"
            )));
        };
        let chunk = self.source.take_chunk(CHUNK_SIZE)?;
        check_chunk(&chunk, CHUNK_SIZE);
        // Only this thread sets segments and slots, while holding the lock,
        // and `next` is past every slot set so far.
        let segment = segment.get_or_init(Box::default);
        let set = segment[next % SEGMENT_LEN].set(CurrentChunk {
            chunk,
            offset: AtomicU64::new(0),
        });
        assert!(set.is_ok(), "transient chunk slot {next} was set twice");
        self.current.store(next, Ordering::Release);
        Ok(())
    }

    /// Allocates `size` bytes, more than [`CHUNK_SIZE`], in a chunk of
    /// their own.
    ///
    /// # Errors
    ///
    /// Returns the source's error.
    fn allocate_large(&self, size: u64) -> RhiResult<TransientAllocation<'_>> {
        let chunk = self.source.take_chunk(size)?;
        check_chunk(&chunk, size);
        let allocation = TransientAllocation::new(&chunk, 0, size);
        self.taking
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(chunk);
        Ok(allocation)
    }

    /// Gives the frame's chunks back to its source, as ending or dropping it
    /// does, even if it took none.
    fn retire(&mut self) {
        let large = mem::take(
            self.taking
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner),
        );
        // Only the segments the frame used are allocated, so this walks
        // only their slots.
        let chunks = mem::take(&mut self.segments)
            .into_iter()
            .filter_map(OnceLock::into_inner)
            .flat_map(|segment| *segment)
            .filter_map(|slot| slot.into_inner().map(|current| current.chunk))
            .collect();
        self.source.retire_chunks(RetiredChunks { chunks, large });
    }

    /// The chunk of [`CHUNK_SIZE`] at `index`, if the frame has taken it.
    fn chunk(&self, index: usize) -> Option<&CurrentChunk> {
        self.segments.get(index / SEGMENT_LEN)?.get()?[index % SEGMENT_LEN].get()
    }

    /// Whether the frame takes its chunks from `source`, so that a backend
    /// can check a frame ended on it was begun on it.
    pub(crate) fn is_from(&self, source: &dyn ChunkSource) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(&self.source), source)
    }

    /// How many chunks of [`CHUNK_SIZE`] the frame has taken.
    fn chunk_count(&self) -> usize {
        let current = self.current.load(Ordering::Acquire);
        current + usize::from(self.chunk(current).is_some())
    }
}

impl Drop for TransientFrame {
    /// Ends the frame, if it wasn't ended, giving its chunks back to be
    /// reused once nothing translated before now is outstanding.
    fn drop(&mut self) {
        self.retire();
    }
}

impl Debug for TransientFrame {
    /// Formats the frame as its number of chunks.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransientFrame")
            .field("chunks", &self.chunk_count())
            .finish_non_exhaustive()
    }
}

/// Memory for one frame only, from [`TransientFrame::allocate`]: an
/// address for shaders and commands, and bytes to write. Writable until the
/// frame ends, which the borrow enforces.
///
/// Writes are visible to work submitted after them, with no barrier, since
/// upload memory is coherent. Writing after submitting work that reads the
/// allocation is a bug.
pub struct TransientAllocation<'frame> {
    /// The allocation's address on the GPU.
    address: GpuAddress,
    /// Where it's mapped for the CPU to write.
    mapped: NonNull<u8>,
    /// Its size, in bytes.
    size: usize,
    /// Borrows the frame, whose chunks hold the memory.
    frame: PhantomData<&'frame TransientFrame>,
}

// SAFETY: The allocation is an exclusive range of mapped memory, which can be
// written from any thread. It isn't `Sync`, since `bytes_mut` needs `&mut`.
unsafe impl Send for TransientAllocation<'_> {}

impl<'frame> TransientAllocation<'frame> {
    /// The `size` bytes at `offset` in `chunk`, which the caller has
    /// allocated to this allocation alone.
    fn new(chunk: &TransientChunk, offset: u64, size: u64) -> Self {
        debug_assert!(offset + size <= chunk.size);
        // The chunk is mapped whole, so its offsets and sizes fit in the
        // CPU's address space, and the pointer doesn't wrap to null.
        let in_mapping = |value: u64| {
            usize::try_from(value)
                .expect("a transient chunk is mapped whole, so it fits in `usize`")
        };
        Self {
            address: GpuAddress(chunk.address.0 + offset),
            mapped: NonNull::new(chunk.mapped.as_ptr().wrapping_add(in_mapping(offset)))
                .expect("a chunk's mapping doesn't wrap the address space"),
            size: in_mapping(size),
            frame: PhantomData,
        }
    }

    /// The allocation's address on the GPU.
    pub fn address(&self) -> GpuAddress {
        self.address
    }

    /// The allocation's bytes, to write what the GPU will read. Their
    /// contents are undefined until written.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: The range is inside a chunk's mapping, which its source
        // keeps mapped and untouched until the frame ends and gives it back
        // (see `ChunkSource`), and the frame is borrowed for as long as this
        // allocation. The frame's atomic bump gave this range to this
        // allocation alone, and `&mut self` keeps this the only reference to
        // it. Mapped memory is plain bytes, so any contents are valid `u8`s.
        unsafe { slice::from_raw_parts_mut(self.mapped.as_ptr(), self.size) }
    }
}

impl Debug for TransientAllocation<'_> {
    /// Formats the allocation as its address and size.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransientAllocation")
            .field("address", &self.address)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

// The frame is shared by every thread recording it, and each allocation can
// be handed to another thread to write (design 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    /// Compiles only for types that are `Send`.
    const fn assert_send<T: Send>() {}
    assert_send_sync::<TransientFrame>();
    assert_send::<TransientAllocation<'_>>();
};

#[cfg(test)]
mod tests {
    //! T0 unit and property tests of transient allocation against a fake
    //! source over host memory (design 0002, R23).

    use std::collections::HashMap;
    use std::sync::Barrier;
    use std::thread;
    use std::time::Duration;

    use proptest::prelude::*;

    use super::*;

    /// A source whose chunks are host memory at fake addresses, recording
    /// what it hands out and takes back.
    #[derive(Debug, Default)]
    struct FakeSource {
        /// The memory behind each chunk handed out, by address, and how many
        /// chunks it has made, which places the next.
        state: Mutex<FakeState>,
    }

    /// What a [`FakeSource`] has handed out and taken back.
    #[derive(Debug, Default)]
    struct FakeState {
        /// The memory of each chunk handed out and not taken back, by
        /// address.
        memory: HashMap<u64, Box<[u8]>>,
        /// The size of each chunk ever handed out, in order.
        taken: Vec<u64>,
        /// The addresses of the chunks of [`CHUNK_SIZE`] and the large
        /// chunks taken back, in each retirement.
        retired: Vec<(Vec<u64>, Vec<u64>)>,
        /// Fail every take after this many, if set.
        fail_after: Option<usize>,
        /// Hand out chunks with no memory behind them, for tests that take
        /// many and never write.
        unbacked: bool,
        /// Hand out chunks 16 bytes past an aligned address, as a broken
        /// source would.
        misaligned: bool,
        /// Sleep this long in each take, so that other threads find the
        /// chunk full meanwhile and wait to take one too.
        take_delay: Option<Duration>,
    }

    impl FakeSource {
        /// A source that hands out chunks without limit.
        fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        /// Locks the source's state.
        fn state(&self) -> std::sync::MutexGuard<'_, FakeState> {
            self.state.lock().unwrap()
        }
    }

    // SAFETY: Each chunk's memory is a boxed slice of its size, which the
    // source holds, untouched, until the chunk is retired, and never hands
    // out twice. Unbacked chunks break this, so only tests that never call
    // `bytes_mut` ask for them.
    unsafe impl ChunkSource for FakeSource {
        /// A chunk of exactly `size` bytes, in 4 GiB of fake address space
        /// of its own past the last chunk's, so that addresses tell chunks
        /// apart. Its memory is zeroed host memory, unless the source is
        /// unbacked. Fails once `fail_after` chunks have been taken.
        fn take_chunk(&self, size: u64) -> RhiResult<TransientChunk> {
            let mut state = self.state();
            if state
                .fail_after
                .is_some_and(|limit| state.taken.len() >= limit)
            {
                return Err(RhiError::OutOfDeviceMemory {
                    label: "transient chunk".to_owned(),
                });
            }
            // Each chunk gets its own 4 GiB of fake address space, past the
            // first, so addresses tell chunks apart and are all aligned.
            let mut address = (state.taken.len() as u64 + 1) << 32;
            if state.misaligned {
                address += 16;
            }
            state.taken.push(size);
            if let Some(delay) = state.take_delay {
                thread::sleep(delay);
            }
            let mapped = if state.unbacked {
                NonNull::dangling()
            } else {
                let mut memory = vec![0; size as usize].into_boxed_slice();
                let mapped = NonNull::new(memory.as_mut_ptr()).unwrap();
                state.memory.insert(address, memory);
                mapped
            };
            Ok(TransientChunk {
                address: GpuAddress(address),
                mapped,
                size,
            })
        }

        /// Records the chunks' addresses as one retirement, and frees their
        /// memory, so a later write through them would be caught by Miri or
        /// a sanitizer rather than passing.
        fn retire_chunks(&self, retired: RetiredChunks) {
            let mut state = self.state();
            let addresses = |chunks: &[TransientChunk]| -> Vec<u64> {
                chunks.iter().map(|chunk| chunk.address.0).collect()
            };
            let entry = (addresses(&retired.chunks), addresses(&retired.large));
            for chunk in retired.chunks.iter().chain(&retired.large) {
                state.memory.remove(&chunk.address.0);
            }
            state.retired.push(entry);
        }
    }

    /// The chunk an address is in, and its offset there, for the fake
    /// source's addresses.
    fn chunk_and_offset(address: GpuAddress) -> (u64, u64) {
        (address.0 >> 32 << 32, address.0 & 0xFFFF_FFFF)
    }

    /// Allocations bump through one chunk, each at the next multiple of its
    /// alignment, with the bytes in between skipped.
    #[test]
    fn allocations_bump_through_a_chunk() {
        let source = FakeSource::new();
        let frame = TransientFrame::new(source.clone());
        let offsets: Vec<_> = [(3, 1), (8, 8), (1, 1), (16, 256)]
            .into_iter()
            .map(|(size, align)| frame.allocate(size, align).unwrap().address())
            .map(chunk_and_offset)
            .collect();
        assert_eq!(
            offsets,
            [(1 << 32, 0), (1 << 32, 8), (1 << 32, 16), (1 << 32, 256)]
        );
        assert_eq!(source.state().taken, [CHUNK_SIZE]);
    }

    /// A frame takes no chunk until its first allocation, and one that took
    /// none still ends, so that the source's pool counts it.
    #[test]
    fn frames_take_chunks_when_first_used() {
        let source = FakeSource::new();
        let frame = TransientFrame::new(source.clone());
        assert!(source.state().taken.is_empty());
        drop(frame);
        assert_eq!(source.state().retired, [(vec![], vec![])]);
    }

    /// An allocation that doesn't fit in what's left of the chunk moves to a
    /// new one, which later allocations bump through.
    #[test]
    fn full_chunks_move_to_a_new_one() {
        let source = FakeSource::new();
        let frame = TransientFrame::new(source.clone());
        frame.allocate(CHUNK_SIZE - 16, 1).unwrap();
        let next = frame.allocate(32, 16).unwrap().address();
        let after = frame.allocate(4, 4).unwrap().address();
        assert_eq!(chunk_and_offset(next), (2 << 32, 0));
        assert_eq!(chunk_and_offset(after), (2 << 32, 32));
        assert_eq!(source.state().taken, [CHUNK_SIZE, CHUNK_SIZE]);
    }

    /// An allocation that exactly fills what's left stays in the chunk.
    #[test]
    fn allocations_can_fill_a_chunk() {
        let source = FakeSource::new();
        let frame = TransientFrame::new(source.clone());
        frame.allocate(CHUNK_SIZE - 16, 1).unwrap();
        let last = frame.allocate(16, 16).unwrap().address();
        assert_eq!(chunk_and_offset(last), (1 << 32, CHUNK_SIZE - 16));
        assert_eq!(source.state().taken, [CHUNK_SIZE]);
    }

    /// An allocation larger than a chunk gets a chunk of its own size,
    /// leaving the current chunk to the allocations after it.
    #[test]
    fn large_allocations_get_their_own_chunk() {
        let source = FakeSource::new();
        let frame = TransientFrame::new(source.clone());
        frame.allocate(16, 1).unwrap();
        let large = frame.allocate(CHUNK_SIZE + 1, 1).unwrap().address();
        let after = frame.allocate(16, 1).unwrap().address();
        assert_eq!(chunk_and_offset(large), (2 << 32, 0));
        assert_eq!(chunk_and_offset(after), (1 << 32, 16));
        assert_eq!(source.state().taken, [CHUNK_SIZE, CHUNK_SIZE + 1]);
    }

    /// Ending a frame, by dropping it, gives back every chunk it took, once,
    /// with large ones apart, since the source frees them rather than
    /// keeping them.
    #[test]
    fn ending_a_frame_retires_its_chunks() {
        let source = FakeSource::new();
        let frame = TransientFrame::new(source.clone());
        frame.allocate(CHUNK_SIZE, 1).unwrap();
        frame.allocate(1, 1).unwrap();
        frame.allocate(2 * CHUNK_SIZE, 1).unwrap();
        drop(frame);
        let state = source.state();
        assert_eq!(state.retired, [(vec![1 << 32, 2 << 32], vec![3 << 32])]);
        assert!(state.memory.is_empty());
    }

    /// A frame knows the source it takes chunks from, and no other.
    #[test]
    fn frames_know_their_source() {
        let source = FakeSource::new();
        let other = FakeSource::new();
        let frame = TransientFrame::new(source.clone());
        assert!(frame.is_from(&*source));
        assert!(!frame.is_from(&*other));
    }

    /// Writes through an allocation reach the chunk's memory at its offset.
    #[test]
    fn writes_reach_the_chunk() {
        let source = FakeSource::new();
        let frame = TransientFrame::new(source.clone());
        frame.allocate(4, 1).unwrap();
        let mut allocation = frame.allocate(4, 4).unwrap();
        allocation.bytes_mut().copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(allocation.bytes_mut().len(), 4);
        let state = source.state();
        assert_eq!(state.memory[&(1 << 32)][..8], [0, 0, 0, 0, 1, 2, 3, 4]);
    }

    /// A source's error is returned, and the frame can still allocate from
    /// the chunk it has.
    #[test]
    fn source_errors_are_returned() {
        let source = FakeSource::new();
        source.state().fail_after = Some(1);
        let frame = TransientFrame::new(source.clone());
        frame.allocate(CHUNK_SIZE - 8, 1).unwrap();
        assert!(matches!(
            frame.allocate(16, 1),
            Err(RhiError::OutOfDeviceMemory { .. })
        ));
        assert!(matches!(
            frame.allocate(CHUNK_SIZE + 1, 1),
            Err(RhiError::OutOfDeviceMemory { .. })
        ));
        frame.allocate(8, 1).unwrap();
    }

    /// A frame that has used all its chunks returns an error, rather than
    /// growing without bound.
    #[test]
    fn frames_have_a_chunk_limit() {
        let source = FakeSource::new();
        source.state().unbacked = true;
        let frame = TransientFrame::new(source.clone());
        for _ in 0..MAX_CHUNKS_PER_FRAME {
            frame.allocate(CHUNK_SIZE, 1).unwrap();
        }
        let err = frame.allocate(1, 1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "a transient frame used more than 256 chunks of 4194304 bytes"
        );
    }

    /// An allocation of nothing panics.
    #[test]
    #[should_panic(expected = "a transient allocation has a size of zero")]
    fn empty_allocations_panic() {
        let frame = TransientFrame::new(FakeSource::new());
        let _ = frame.allocate(0, 1);
    }

    /// An alignment that isn't a power of two panics.
    #[test]
    #[should_panic(expected = "a transient allocation's alignment, 12, isn't a power of two")]
    fn alignments_must_be_powers_of_two() {
        let frame = TransientFrame::new(FakeSource::new());
        let _ = frame.allocate(4, 12);
    }

    /// An alignment past the largest a chunk guarantees panics.
    #[test]
    #[should_panic(expected = "alignment, 512, isn't a power of two up to 256")]
    fn alignments_have_a_limit() {
        let frame = TransientFrame::new(FakeSource::new());
        let _ = frame.allocate(4, 512);
    }

    /// A chunk taken from a source at an address that isn't a multiple of
    /// the largest alignment panics, rather than handing out misaligned
    /// allocations.
    #[test]
    #[should_panic(expected = "a transient chunk's address, 0x100000010, isn't a multiple of 256")]
    fn misaligned_chunks_panic() {
        let source = FakeSource::new();
        source.state().misaligned = true;
        let frame = TransientFrame::new(source);
        let _ = frame.allocate(4, 4);
    }

    /// A large allocation's chunk is checked the same way.
    #[test]
    #[should_panic(expected = "a transient chunk's address, 0x100000010, isn't a multiple of 256")]
    fn misaligned_large_chunks_panic() {
        let source = FakeSource::new();
        source.state().misaligned = true;
        let frame = TransientFrame::new(source);
        let _ = frame.allocate(CHUNK_SIZE + 1, 256);
    }

    /// The threads the concurrent tests allocate from.
    const THREADS: usize = 8;

    /// Threads that fill chunk after chunk at once take exactly as many
    /// chunks as the allocations fill, so no two threads that find the
    /// same chunk full, or the first slot empty, both move past it. The
    /// threads start together, and each take is slow, so several threads
    /// find each chunk full while one takes the next.
    #[test]
    fn concurrent_threads_take_each_chunk_once() {
        const PER_CHUNK: u64 = 16;
        const PER_THREAD: u64 = 32;
        let source = FakeSource::new();
        source.state().unbacked = true;
        source.state().take_delay = Some(Duration::from_millis(2));
        let frame = TransientFrame::new(source.clone());
        let start = Barrier::new(THREADS);
        thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    start.wait();
                    for _ in 0..PER_THREAD {
                        frame.allocate(CHUNK_SIZE / PER_CHUNK, 1).unwrap();
                    }
                });
            }
        });
        let chunks = THREADS as u64 * PER_THREAD / PER_CHUNK;
        assert_eq!(source.state().taken.len() as u64, chunks);
        assert_eq!(frame.chunk_count() as u64, chunks);
    }

    /// A strategy for an allocation's size and the log2 of its alignment:
    /// half small, as per-draw data is, and half up to an eighth of a
    /// chunk, so each case fills several chunks.
    fn request() -> impl Strategy<Value = (u64, u32)> {
        (prop_oneof![1..=256_u64, 1..=CHUNK_SIZE / 8], 0..=8_u32)
    }

    proptest! {
        /// Several threads allocate random sizes and alignments from one
        /// frame at once, filling several chunks, and writing their own
        /// byte at each allocation's ends. Every allocation is aligned,
        /// inside a chunk, and overlaps no other, which the bytes left in
        /// the chunks confirm.
        #[test]
        fn concurrent_allocations_never_overlap(
            requests in prop::collection::vec(prop::collection::vec(request(), 1..32), THREADS),
        ) {
            let source = FakeSource::new();
            let frame = TransientFrame::new(source.clone());
            let allocations: Vec<Vec<(GpuAddress, u64, u64)>> = thread::scope(|scope| {
                let threads: Vec<_> = requests
                    .iter()
                    .enumerate()
                    .map(|(thread, requests)| {
                        let frame = &frame;
                        scope.spawn(move || {
                            requests
                                .iter()
                                .map(|&(size, align_log2)| {
                                    let align = 1 << align_log2;
                                    let mut allocation = frame.allocate(size, align).unwrap();
                                    let bytes = allocation.bytes_mut();
                                    bytes[0] = thread as u8 + 1;
                                    bytes[size as usize - 1] = thread as u8 + 1;
                                    (allocation.address(), size, align)
                                })
                                .collect()
                        })
                    })
                    .collect();
                threads.into_iter().map(|thread| thread.join().unwrap()).collect()
            });

            let state = source.state();
            let mut ranges = Vec::new();
            for (thread, allocations) in allocations.iter().enumerate() {
                for &(address, size, align) in allocations {
                    prop_assert_eq!(address.0 % align, 0, "allocation is aligned");
                    let (chunk, offset) = chunk_and_offset(address);
                    let memory = &state.memory[&chunk];
                    prop_assert!(offset + size <= memory.len() as u64, "allocation is inside its chunk");
                    let ends = [memory[offset as usize], memory[(offset + size) as usize - 1]];
                    prop_assert_eq!(
                        ends,
                        [thread as u8 + 1; 2],
                        "no other allocation wrote over this one's ends",
                    );
                    ranges.push((address.0, address.0 + size));
                }
            }
            ranges.sort_unstable();
            for pair in ranges.windows(2) {
                prop_assert!(pair[0].1 <= pair[1].0, "allocations don't overlap");
            }
        }
    }
}
