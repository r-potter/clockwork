//! The pool of transient chunks a backend keeps for reuse, and how large it
//! may grow (design 0002, "Transient memory", R23).
//!
//! Frames take chunks of [`CHUNK_SIZE`] from the pool, which creates them
//! through the backend when it has none free, and the chunks come back to
//! it once nothing can use them. The pool counts the chunks taken and not
//! yet back as in use: those of the frame being recorded, and of the frames
//! still in flight.
//!
//! So that a one-off spike doesn't hold its memory for the rest of the run,
//! the pool keeps no more chunks, in use and free together, than were in use
//! when the busiest of the last [`RECENT_FRAMES`] frames ended, and hands
//! the rest back for the backend to free, the oldest first. Counting the
//! chunks in use, rather than only what one frame used, means that when
//! several frames' chunks come back at once, the pool keeps what the frames
//! recorded meanwhile will need, rather than freeing chunks only to create
//! them again. Chunks of large allocations never come back to it: the
//! backend frees them.
//!
//! The pool is pure logic over whatever the backend calls a chunk, so it's
//! tested with plain values.

#[cfg(doc)]
use super::CHUNK_SIZE;

/// How many of the most recent frames the pool's size follows.
pub(crate) const RECENT_FRAMES: usize = 8;

/// Free chunks of [`CHUNK_SIZE`], how many are in use, and how many were in
/// use as recent frames ended.
#[derive(Debug)]
pub(crate) struct ChunkPool<C> {
    /// Chunks nothing uses, the most recently recycled last.
    free: Vec<C>,
    /// How many chunks have been taken and not yet recycled.
    in_use: usize,
    /// How many chunks were in use as each of the last [`RECENT_FRAMES`]
    /// frames ended, as a ring, with frames that haven't happened yet as 0.
    recent: [usize; RECENT_FRAMES],
    /// The index in `recent` the next frame to end is recorded at.
    next: usize,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the Vulkan backend keeps its chunks in a pool in a later PR"
    )
)]
impl<C> ChunkPool<C> {
    /// An empty pool, as though the last [`RECENT_FRAMES`] frames used
    /// nothing.
    pub(crate) fn new() -> Self {
        Self {
            free: Vec::new(),
            in_use: 0,
            recent: [0; RECENT_FRAMES],
            next: 0,
        }
    }

    /// A free chunk, the one most recently recycled, or a new one from
    /// `create` if none is free. Either way, the chunk is in use until it's
    /// recycled.
    ///
    /// # Errors
    ///
    /// Returns `create`'s error, with nothing taken.
    pub(crate) fn take<E>(&mut self, create: impl FnOnce() -> Result<C, E>) -> Result<C, E> {
        let chunk = match self.free.pop() {
            Some(chunk) => chunk,
            None => create()?,
        };
        self.in_use += 1;
        Ok(chunk)
    }

    /// Records that a frame ended, with the chunks in use then. Returns the
    /// free chunks beyond what the last [`RECENT_FRAMES`] frames now need,
    /// for the backend to free. Called for every frame, even one that took
    /// no chunks, so that idle frames count.
    pub(crate) fn frame_ended(&mut self) -> Vec<C> {
        self.recent[self.next] = self.in_use;
        self.next = (self.next + 1) % RECENT_FRAMES;
        self.trim()
    }

    /// Takes back chunks of [`CHUNK_SIZE`] that nothing can use any more.
    /// Returns the free chunks beyond what the last [`RECENT_FRAMES`]
    /// frames need, for the backend to free.
    ///
    /// # Panics
    ///
    /// Panics if more chunks come back than are in use, which means the
    /// backend recycled a chunk the pool didn't hand out.
    pub(crate) fn recycle(&mut self, chunks: impl IntoIterator<Item = C>) -> Vec<C> {
        let free = self.free.len();
        self.free.extend(chunks);
        let recycled = self.free.len() - free;
        self.in_use = self
            .in_use
            .checked_sub(recycled)
            .expect("only chunks the pool handed out are recycled");
        self.trim()
    }

    /// Every free chunk, for the backend to free when it's dropped.
    pub(crate) fn drain(&mut self) -> Vec<C> {
        std::mem::take(&mut self.free)
    }

    /// How many chunks are free.
    pub(crate) fn free_len(&self) -> usize {
        self.free.len()
    }

    /// The most chunks the pool keeps, in use and free together: as many as
    /// were in use when the busiest of the last [`RECENT_FRAMES`] frames
    /// ended.
    fn limit(&self) -> usize {
        self.recent.iter().copied().max().unwrap_or(0)
    }

    /// Removes the free chunks beyond [`limit`](Self::limit), counting those
    /// in use, the least recently recycled first, and returns them.
    fn trim(&mut self) -> Vec<C> {
        let keep = self.limit().saturating_sub(self.in_use);
        let excess = self.free.len().saturating_sub(keep);
        self.free.drain(..excess).collect()
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the chunk pool's reuse and size (design 0002, R23).

    use std::convert::Infallible;

    use super::*;

    /// The frame loop the tests model, numbering the chunks it creates from
    /// 1.
    struct Frames {
        /// The pool.
        pool: ChunkPool<u32>,
        /// How many chunks have been created.
        created: u32,
    }

    impl Frames {
        /// A loop with an empty pool, which has created nothing.
        fn new() -> Self {
            Self {
                pool: ChunkPool::new(),
                created: 0,
            }
        }

        /// Takes a chunk from the pool, creating the next one if it has
        /// none free.
        fn take(&mut self) -> u32 {
            let created = &mut self.created;
            self.pool
                .take(|| {
                    *created += 1;
                    Ok::<_, Infallible>(*created)
                })
                .unwrap()
        }

        /// Runs a frame that takes `used` chunks and ends. Returns its
        /// chunks, which the caller recycles once nothing can use them, and
        /// what ending it released.
        fn run(&mut self, used: usize) -> (Vec<u32>, Vec<u32>) {
            let chunks = (0..used).map(|_| self.take()).collect();
            (chunks, self.pool.frame_ended())
        }
    }

    /// No chunks, for comparing with what's released.
    const NONE: [u32; 0] = [];

    /// An empty pool creates chunks, and recycled chunks are taken again,
    /// the most recent first.
    #[test]
    fn recycled_chunks_are_taken_again() {
        let mut frames = Frames::new();
        assert_eq!(frames.take(), 1);
        assert_eq!(frames.take(), 2);
        assert_eq!(frames.pool.frame_ended(), NONE);
        assert_eq!(frames.pool.recycle([1, 2]), NONE);
        assert_eq!(frames.take(), 2);
        assert_eq!(frames.take(), 1);
        assert_eq!(frames.take(), 3);
    }

    /// A chunk that can't be created is an error, with nothing counted as
    /// in use.
    #[test]
    fn creation_errors_take_nothing() {
        let mut pool = ChunkPool::<u32>::new();
        assert_eq!(pool.take(|| Err("out of memory")), Err("out of memory"));
        assert_eq!(pool.frame_ended(), NONE);
        assert_eq!(pool.limit(), 0);
    }

    /// With no frame having ended, the pool keeps nothing, so recycling
    /// releases every chunk.
    #[test]
    fn idle_pools_keep_nothing() {
        let mut frames = Frames::new();
        frames.take();
        frames.take();
        assert_eq!(frames.pool.recycle([1, 2]), [1, 2]);
        assert_eq!(frames.pool.free_len(), 0);
    }

    /// The pool keeps as many chunks as were in use when the busiest recent
    /// frame ended, releasing the least recently recycled beyond that.
    #[test]
    fn pools_keep_what_the_busiest_frame_needed() {
        let mut frames = Frames::new();
        frames.run(3);
        frames.take();
        assert_eq!(frames.pool.recycle([1, 2, 3, 4]), [1]);
        assert_eq!(frames.pool.free_len(), 3);
    }

    /// Recycling a chunk the pool didn't hand out panics, rather than
    /// miscounting what's in use.
    #[test]
    #[should_panic(expected = "only chunks the pool handed out are recycled")]
    fn recycling_untaken_chunks_panics() {
        let mut frames = Frames::new();
        frames.take();
        frames.pool.recycle([1, 2]);
    }

    /// In a steady frame loop, with three frames in flight each using the
    /// same number of chunks, frames reuse the pool's chunks rather than
    /// creating more, and nothing is released.
    #[test]
    fn steady_frames_reuse_chunks() {
        let mut frames = Frames::new();
        let mut in_flight = Vec::new();
        for _ in 0..20 {
            let (chunks, released) = frames.run(2);
            assert_eq!(released, NONE);
            in_flight.push(chunks);
            if in_flight.len() > 3 {
                assert_eq!(frames.pool.recycle(in_flight.remove(0)), NONE);
            }
        }
        // Four frames' chunks exist at once, the three in flight and the one
        // being recorded, and none more were ever created.
        assert_eq!(frames.created, 8);
    }

    /// When the chunks of several frames in flight come back at once, the
    /// pool keeps them all for the frames that follow, rather than freeing
    /// all but one frame's worth and creating the rest again.
    #[test]
    fn batched_recycles_keep_chunks_for_frames_in_flight() {
        let mut frames = Frames::new();
        let mut in_flight = Vec::new();
        for _ in 0..3 {
            let (chunks, released) = frames.run(4);
            assert_eq!(released, NONE);
            in_flight.extend(chunks);
        }
        assert_eq!(frames.pool.recycle(in_flight.drain(..)), NONE);
        assert_eq!(frames.pool.free_len(), 12);

        for _ in 0..3 {
            let (chunks, released) = frames.run(4);
            assert_eq!(released, NONE);
            in_flight.extend(chunks);
        }
        assert_eq!(frames.pool.recycle(in_flight), NONE);
        assert_eq!(frames.created, 12, "the later frames reused every chunk");
    }

    /// After a spike, the pool keeps the spike's chunks while it's among the
    /// recent frames, then releases what the frames since don't need, the
    /// oldest first, as the frame that pushes the spike out of the window
    /// ends.
    #[test]
    fn pools_shrink_after_a_spike() {
        let mut frames = Frames::new();
        let (spike, released) = frames.run(10);
        assert_eq!(released, NONE);
        assert_eq!(frames.pool.recycle(spike), NONE);
        assert_eq!(frames.pool.free_len(), 10);

        // The next frames each use one chunk, the most recently recycled,
        // and recycle it once they end. While the spike is among the last
        // eight frames, the pool keeps its ten.
        for _ in 1..RECENT_FRAMES {
            let (chunks, released) = frames.run(1);
            assert_eq!(chunks, [10]);
            assert_eq!(released, NONE);
            assert_eq!(frames.pool.recycle(chunks), NONE);
            assert_eq!(frames.pool.free_len(), 10);
        }

        // The eighth frame after the spike pushes it out, so the pool keeps
        // only the one chunk the recent frames had in use, which this frame
        // holds: ending it releases the other nine, the oldest first.
        let (chunks, released) = frames.run(1);
        assert_eq!(released, (1..=9).collect::<Vec<_>>());
        assert_eq!(frames.pool.recycle(chunks), NONE);
        assert_eq!(frames.pool.free_len(), 1);
        assert_eq!(
            frames.created, 10,
            "the frames after the spike reused its chunks"
        );
    }

    /// Frames that use no chunks count too, so a pool whose frames stop
    /// allocating releases everything once they've filled the window.
    #[test]
    fn frames_that_use_nothing_count() {
        let mut frames = Frames::new();
        let (chunks, _) = frames.run(4);
        assert_eq!(frames.pool.recycle(chunks), NONE);
        for _ in 1..RECENT_FRAMES {
            assert_eq!(frames.pool.frame_ended(), NONE);
        }
        assert_eq!(frames.pool.frame_ended(), [1, 2, 3, 4]);
    }

    /// Draining returns every free chunk, and leaves the pool empty.
    #[test]
    fn draining_returns_every_free_chunk() {
        let mut frames = Frames::new();
        let (chunks, _) = frames.run(2);
        frames.pool.recycle(chunks);
        assert_eq!(frames.pool.drain(), [1, 2]);
        assert_eq!(frames.take(), 3);
    }
}
