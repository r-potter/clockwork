//! The staging buffers an uploader writes the bytes it uploads to, before
//! copying them on the GPU (design 0002, "Streaming").
//!
//! Each uploader has its own [`Staging`], apart from the transient memory
//! rendering frames use, so that streaming doesn't change how much memory
//! the frames' pool keeps. Bytes are written into buffers of
//! [`CHUNK_SIZE`] in upload memory, bumping an offset through the current
//! one, and bytes too large for one get a buffer of their own. The buffers
//! written between two flushes are a batch. Once a flush submits the copies
//! that read a batch, the batch is in flight until that submission's
//! timepoint is reached, and then its buffers of [`CHUNK_SIZE`] go back to
//! a [`ChunkPool`], each flush counting as one of its frames, and the
//! others are destroyed.
//!
//! The staging is ordinary code over [`GpuResources`], so it's tested with a
//! fake of it.

use std::sync::Arc;

use crate::transient::CHUNK_SIZE;
use crate::transient::pool::ChunkPool;
use crate::{
    Buffer, BufferDesc, BufferUsage, GpuAddress, GpuResources, GpuTimepoint, MemoryLocation,
    RhiResult,
};

/// A staging buffer of [`CHUNK_SIZE`] being written, and how much of it is.
#[derive(Debug)]
struct Chunk {
    /// The buffer.
    buffer: Buffer,
    /// Its address on the GPU.
    address: GpuAddress,
    /// How many bytes from its start are written or skipped for alignment.
    used: u64,
}

/// The staging buffers written between two flushes.
#[derive(Debug, Default)]
struct Batch {
    /// Its buffers of [`CHUNK_SIZE`], for the pool to reuse.
    chunks: Vec<Buffer>,
    /// The buffers of bytes too large for one of [`CHUNK_SIZE`], destroyed
    /// rather than reused.
    large: Vec<Buffer>,
}

impl Batch {
    /// Whether the batch has no buffers.
    fn is_empty(&self) -> bool {
        self.chunks.is_empty() && self.large.is_empty()
    }
}

/// An uploader's staging buffers: the batch being written, the batches in
/// flight, and the pool of free buffers.
#[derive(Debug)]
pub(crate) struct Staging {
    /// Where the buffers come from.
    resources: Arc<dyn GpuResources>,
    /// The buffers' label: the uploader's, followed by `staging`.
    label: String,
    /// The free buffers of [`CHUNK_SIZE`], and how many to keep.
    pool: ChunkPool<Buffer>,
    /// The buffer of [`CHUNK_SIZE`] being written, also in `batch`, if the
    /// batch has one with room left.
    current: Option<Chunk>,
    /// The buffers written since the last flush.
    batch: Batch,
    /// The batches submitted, each with the timepoint its copies complete
    /// at.
    in_flight: Vec<(GpuTimepoint, Batch)>,
}

impl Staging {
    /// Staging for the uploader labelled `label`, with no buffers yet.
    pub(crate) fn new(resources: Arc<dyn GpuResources>, label: &str) -> Self {
        Self {
            resources,
            label: format!("{label} staging"),
            pool: ChunkPool::new(),
            current: None,
            batch: Batch::default(),
            in_flight: Vec::new(),
        }
    }

    /// Writes `data` to a staging buffer, at an address that's a multiple of
    /// `alignment`, and returns the address. The buffer is part of the batch
    /// until it's [`submitted`](Self::submitted).
    ///
    /// # Errors
    ///
    /// Returns an error if a staging buffer can't be created.
    ///
    /// # Panics
    ///
    /// Panics if `data` is empty, or `alignment` is 0.
    pub(crate) fn write(&mut self, data: &[u8], alignment: u64) -> RhiResult<GpuAddress> {
        assert!(!data.is_empty(), "staging writes have bytes to write");
        assert!(alignment > 0, "staging writes have an alignment");
        let len = data.len() as u64;
        if let Some(chunk) = &mut self.current {
            let offset = aligned_offset(chunk.address, chunk.used, alignment);
            if offset.checked_add(len).is_some_and(|end| end <= CHUNK_SIZE) {
                return Ok(self.write_at(data, offset));
            }
        }

        // However a new buffer's address is aligned, the bytes fit in one of
        // `CHUNK_SIZE` after at most `alignment - 1` bytes of padding.
        let worst_case = len.saturating_add(alignment - 1);
        if worst_case > CHUNK_SIZE {
            let buffer = self.create(worst_case)?;
            self.batch.large.push(buffer);
            let address = self.resources.buffer_address(buffer);
            let offset = aligned_offset(address, 0, alignment);
            self.resources.write_buffer(buffer, offset, data);
            return Ok(GpuAddress(address.0 + offset));
        }

        let buffer = match self.pool.take() {
            Some(buffer) => buffer,
            None => {
                let buffer = self.create(CHUNK_SIZE)?;
                self.pool.created();
                buffer
            }
        };
        self.batch.chunks.push(buffer);
        let address = self.resources.buffer_address(buffer);
        self.current = Some(Chunk {
            buffer,
            address,
            used: 0,
        });
        Ok(self.write_at(data, aligned_offset(address, 0, alignment)))
    }

    /// Writes `data` at `offset` in the current buffer, which has room for
    /// it, and returns its address.
    fn write_at(&mut self, data: &[u8], offset: u64) -> GpuAddress {
        let chunk = self
            .current
            .as_mut()
            .expect("only the current buffer is written at an offset");
        self.resources.write_buffer(chunk.buffer, offset, data);
        chunk.used = offset + data.len() as u64;
        GpuAddress(chunk.address.0 + offset)
    }

    /// Creates a staging buffer of `size` bytes.
    fn create(&self, size: u64) -> RhiResult<Buffer> {
        self.resources.create_buffer(&BufferDesc {
            label: &self.label,
            size,
            usage: BufferUsage::COPIES,
            location: MemoryLocation::Upload,
        })
    }

    /// Whether nothing has been written since the last flush.
    pub(crate) fn batch_is_empty(&self) -> bool {
        self.batch.is_empty()
    }

    /// How many batches are in flight, for the uploader's tests to check
    /// that flushes reclaim them.
    #[cfg(test)]
    pub(crate) fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    /// Ends the batch, whose copies complete at `timepoint`, so its buffers
    /// are in flight until then. Every flush ends one, even an empty one,
    /// so that the pool sees every flush.
    pub(crate) fn submitted(&mut self, timepoint: GpuTimepoint) {
        self.current = None;
        let batch = std::mem::take(&mut self.batch);
        if !batch.is_empty() {
            self.in_flight.push((timepoint, batch));
        }
        let released = self.pool.frame_ended();
        self.destroy(released);
    }

    /// Ends the batch without submitting it, when its copies couldn't be
    /// submitted, destroying its buffers. Destruction waits for anything
    /// translated before now, so the buffers stay alive for any of the
    /// copies that were submitted.
    pub(crate) fn discard(&mut self) {
        self.current = None;
        let batch = std::mem::take(&mut self.batch);
        self.pool.forget(batch.chunks.len());
        self.destroy(batch.chunks);
        self.destroy(batch.large);
        let released = self.pool.frame_ended();
        self.destroy(released);
    }

    /// Reuses the buffers of every batch in flight whose timepoint
    /// `is_complete` says is reached: those of [`CHUNK_SIZE`] go back to the
    /// pool, and the others are destroyed.
    ///
    /// # Errors
    ///
    /// Returns the first error `is_complete` returns, having reused the
    /// batches it reported reached before it.
    pub(crate) fn reclaim(
        &mut self,
        mut is_complete: impl FnMut(GpuTimepoint) -> RhiResult<bool>,
    ) -> RhiResult<()> {
        // Batches may be on different queues, so a batch can be reached
        // before an earlier one.
        let mut index = 0;
        while let Some((timepoint, _)) = self.in_flight.get(index) {
            if is_complete(*timepoint)? {
                let (_, batch) = self.in_flight.remove(index);
                let released = self.pool.recycle(batch.chunks);
                self.destroy(released);
                self.destroy(batch.large);
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    /// Destroys `buffers`.
    fn destroy(&self, buffers: impl IntoIterator<Item = Buffer>) {
        for buffer in buffers {
            self.resources.destroy_buffer(buffer);
        }
    }
}

impl Drop for Staging {
    /// Destroys every staging buffer. Destruction waits for anything
    /// translated before now, so the batches in flight stay alive until
    /// their copies complete.
    fn drop(&mut self) {
        let batch = std::mem::take(&mut self.batch);
        let in_flight = std::mem::take(&mut self.in_flight);
        let free = self.pool.drain();
        for batch in std::iter::once(batch).chain(in_flight.into_iter().map(|(_, batch)| batch)) {
            self.destroy(batch.chunks);
            self.destroy(batch.large);
        }
        self.destroy(free);
    }
}

/// The least offset from `used` onwards, in a buffer at `address`, that's
/// at an address that's a multiple of `alignment`.
fn aligned_offset(address: GpuAddress, used: u64, alignment: u64) -> u64 {
    (address.0 + used).next_multiple_of(alignment) - address.0
}

#[cfg(test)]
mod tests {
    //! T0 tests of the uploader's staging buffers (design 0002, R17).

    use super::*;
    use crate::QueueId;
    use crate::transient::pool::RECENT_FRAMES;
    use crate::upload::fake::FakeResources;

    /// Staging over a fake whose buffers start 8 bytes past a multiple of
    /// 2^32, and the fake.
    fn staging() -> (Staging, Arc<FakeResources>) {
        let resources = Arc::new(FakeResources::new(8));
        (Staging::new(resources.clone(), "streaming"), resources)
    }

    /// The timepoint where the queue with index `queue` reaches `value`.
    fn timepoint(queue: u32, value: u64) -> GpuTimepoint {
        GpuTimepoint::new(QueueId::new(queue), value)
    }

    /// Bytes are written to upload buffers, at aligned addresses, one after
    /// another in a buffer of `CHUNK_SIZE` while they fit.
    #[test]
    fn writes_are_aligned_and_packed() {
        let (mut staging, resources) = staging();
        let first = staging.write(&[1, 2, 3], 16).unwrap();
        let second = staging.write(&[4, 5], 64).unwrap();
        let third = staging.write(&[6], 1).unwrap();
        assert_eq!(first.0 % 16, 0);
        assert_eq!(second.0 % 64, 0);
        assert_eq!(first.0 >> 32, second.0 >> 32, "both are in one buffer");
        assert_eq!(third.0, second.0 + 2, "an unaligned write follows");
        assert_eq!(resources.read(first, 3), [1, 2, 3]);
        assert_eq!(resources.read(second, 2), [4, 5]);
        assert_eq!(resources.read(third, 1), [6]);
        assert_eq!(resources.created(), 1);
    }

    /// Staging buffers are upload buffers used only for copies, labelled
    /// after the uploader.
    #[test]
    fn buffers_are_upload_buffers_for_copies() {
        let (mut staging, resources) = staging();
        staging.write(&[1], 1).unwrap();
        let buffer = staging.batch.chunks[0];
        resources.with_buffer(buffer, |buffer| {
            assert_eq!(buffer.location, MemoryLocation::Upload);
            assert_eq!(buffer.usage, BufferUsage::COPIES);
            assert_eq!(buffer.bytes.len() as u64, CHUNK_SIZE);
        });
        assert_eq!(staging.label, "streaming staging");
    }

    /// Bytes that don't fit in the rest of the current buffer go to a new
    /// one, at its first aligned address.
    #[test]
    fn full_buffers_move_to_a_new_one() {
        let (mut staging, resources) = staging();
        let len = usize::try_from(CHUNK_SIZE - 64).unwrap();
        staging.write(&vec![1; len], 16).unwrap();
        let next = staging.write(&[2; 64], 16).unwrap();
        assert_eq!(resources.created(), 2);
        assert_eq!(next.0, (2 << 32) + 16, "the first aligned address");
        assert_eq!(staging.batch.chunks.len(), 2);
    }

    /// Bytes that, aligned, may not fit in a buffer of `CHUNK_SIZE` get a
    /// buffer of their own, with room to align them, which never becomes
    /// current.
    #[test]
    fn large_writes_get_their_own_buffer() {
        let (mut staging, resources) = staging();
        let len = usize::try_from(CHUNK_SIZE).unwrap();
        let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let large = staging.write(&data, 256).unwrap();
        assert_eq!(large.0 % 256, 0);
        assert_eq!(resources.read(large, len), data);
        assert_eq!(staging.batch.large.len(), 1);
        resources.with_buffer(staging.batch.large[0], |buffer| {
            assert_eq!(buffer.bytes.len() as u64, CHUNK_SIZE + 255);
        });
        assert!(staging.current.is_none());

        // A write that may not fit in an empty buffer once aligned is large
        // too.
        let len = usize::try_from(CHUNK_SIZE - 14).unwrap();
        staging.write(&vec![0; len], 16).unwrap();
        assert_eq!(staging.batch.large.len(), 2);
        assert!(staging.batch.chunks.is_empty());
    }

    /// A batch's buffers are reused once its timepoint is reached, and not
    /// before.
    #[test]
    fn buffers_are_reused_once_their_copies_complete() {
        let (mut staging, resources) = staging();
        assert!(staging.batch_is_empty());
        staging.write(&[1], 1).unwrap();
        assert!(!staging.batch_is_empty());
        staging.submitted(timepoint(0, 1));
        assert!(staging.batch_is_empty());
        staging.reclaim(|_| Ok(false)).unwrap();
        staging.write(&[2], 1).unwrap();
        assert_eq!(resources.created(), 2, "the first batch is in flight");
        staging.submitted(timepoint(0, 2));

        staging.reclaim(|_| Ok(true)).unwrap();
        // Each write fills more than half a buffer, so they need one each.
        let half = vec![3; usize::try_from(CHUNK_SIZE / 2 + 1).unwrap()];
        staging.write(&half, 1).unwrap();
        staging.write(&half, 1).unwrap();
        assert_eq!(staging.batch.chunks.len(), 2);
        assert_eq!(resources.created(), 2, "both batches' buffers were reused");
        assert_eq!(resources.destroyed(), [] as [u64; 0]);
    }

    /// A large write leaves the current buffer current, so the bytes
    /// written after it carry on filling that buffer.
    #[test]
    fn large_writes_leave_the_current_buffer() {
        let (mut staging, resources) = staging();
        let before = staging.write(&[1], 1).unwrap();
        staging
            .write(&vec![0; usize::try_from(CHUNK_SIZE + 1).unwrap()], 1)
            .unwrap();
        let after = staging.write(&[2], 1).unwrap();
        assert_eq!(after.0, before.0 + 1, "the write follows the one before");
        assert_eq!(staging.batch.chunks.len(), 1);
        assert_eq!(staging.batch.large.len(), 1);
        assert_eq!(resources.created(), 2);
    }

    /// Batches on different queues are each reused once their own
    /// timepoint is reached, whatever order they were submitted in.
    #[test]
    fn batches_are_reclaimed_out_of_order() {
        let (mut staging, _) = staging();
        staging.write(&[1], 1).unwrap();
        staging.submitted(timepoint(1, 5));
        staging.write(&[2], 1).unwrap();
        staging.submitted(timepoint(0, 3));
        staging
            .reclaim(|timepoint| Ok(timepoint.queue() == QueueId::new(0)))
            .unwrap();
        assert_eq!(staging.in_flight.len(), 1);
        assert_eq!(staging.in_flight[0].0, timepoint(1, 5));
    }

    /// The buffers of large writes are destroyed once their copies
    /// complete, rather than reused.
    #[test]
    fn large_buffers_are_destroyed_once_their_copies_complete() {
        let (mut staging, resources) = staging();
        staging
            .write(&vec![0; usize::try_from(CHUNK_SIZE + 1).unwrap()], 1)
            .unwrap();
        staging.submitted(timepoint(0, 1));
        assert_eq!(resources.live(), 1);
        staging.reclaim(|_| Ok(true)).unwrap();
        assert_eq!(resources.destroyed(), [CHUNK_SIZE + 1]);
        assert_eq!(resources.live(), 0);
    }

    /// An error reading a timepoint is returned, and leaves its batch in
    /// flight.
    #[test]
    fn reclaim_errors_leave_batches_in_flight() {
        let (mut staging, _) = staging();
        staging.write(&[1], 1).unwrap();
        staging.submitted(timepoint(0, 1));
        assert!(
            staging
                .reclaim(|_| Err(crate::RhiError::DeviceLost))
                .is_err()
        );
        assert_eq!(staging.in_flight.len(), 1);
    }

    /// The pool keeps what recent flushes had in use, and once a busy flush
    /// leaves the window, idle flushes release its buffers.
    #[test]
    fn idle_flushes_release_buffers() {
        let (mut staging, resources) = staging();
        let len = usize::try_from(CHUNK_SIZE / 2).unwrap();
        for _ in 0..3 {
            staging.write(&vec![0; len + 1], 1).unwrap();
        }
        staging.submitted(timepoint(0, 1));
        staging.reclaim(|_| Ok(true)).unwrap();
        assert_eq!(resources.live(), 3);
        for _ in 1..RECENT_FRAMES {
            staging.submitted(timepoint(0, 1));
        }
        assert_eq!(resources.live(), 3);
        staging.submitted(timepoint(0, 1));
        assert_eq!(resources.live(), 0);
        assert_eq!(resources.destroyed(), [CHUNK_SIZE; 3]);
    }

    /// A batch that couldn't be submitted is destroyed, and no longer
    /// counts as in use, so the pool doesn't keep buffers for it.
    #[test]
    fn discarded_batches_are_destroyed() {
        let (mut staging, resources) = staging();
        staging.write(&[1], 1).unwrap();
        staging
            .write(&vec![0; usize::try_from(CHUNK_SIZE + 1).unwrap()], 1)
            .unwrap();
        staging.discard();
        assert_eq!(resources.live(), 0);
        assert_eq!(resources.destroyed(), [CHUNK_SIZE, CHUNK_SIZE + 1]);
        staging.write(&[2], 1).unwrap();
        assert_eq!(resources.created(), 3, "nothing was pooled");
        assert!(staging.in_flight.is_empty());
    }

    /// A buffer that can't be created is an error, and leaves the batch as
    /// it was.
    #[test]
    fn creation_errors_are_returned() {
        let (mut staging, resources) = staging();
        resources.fail_after(0);
        assert!(staging.write(&[1], 1).is_err());
        assert!(staging.batch.is_empty());
        staging.submitted(timepoint(0, 1));
        assert!(staging.in_flight.is_empty());
    }

    /// Dropping the staging destroys every buffer: the batch's, those in
    /// flight, and the free ones.
    #[test]
    fn dropping_destroys_every_buffer() {
        let (mut staging, resources) = staging();
        staging.write(&[1], 1).unwrap();
        staging.submitted(timepoint(0, 1));
        staging.reclaim(|_| Ok(true)).unwrap();
        staging
            .write(&vec![0; usize::try_from(CHUNK_SIZE + 1).unwrap()], 1)
            .unwrap();
        staging.submitted(timepoint(0, 2));
        staging.write(&[2], 1).unwrap();
        staging.write(&[3], 1).unwrap();
        assert_eq!(resources.live(), 2);
        drop(staging);
        assert_eq!(resources.live(), 0);
    }
}
