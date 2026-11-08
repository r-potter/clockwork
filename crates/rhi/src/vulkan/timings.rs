//! Writing timing scopes' timestamps during translation, keeping them with
//! the queue once submitted, and reading them back (design 0002,
//! "Timestamps").
//!
//! Translation records each scope of a command list into the command
//! buffer's [`RecordedTimings`]: its label, its depth, and, on a queue that
//! can write timestamps, the two queries its timestamps are written to.
//! Queries are numbered from 0 in each command buffer, and come from query
//! chunks claimed as they're needed, so query `n` is query
//! `n % QUERIES_PER_CHUNK` of chunk `n / QUERIES_PER_CHUNK`.
//!
//! When the command buffer is submitted, its timings move to its queue's
//! [`RetainedTimings`](crate::timing::RetainedTimings), under the
//! submission's timeline value, if it recorded any scopes. They're read back
//! by [`Queue::take_timings`](crate::Queue::take_timings) once the
//! submission completes, and their chunks recycled then, or when they're
//! discarded. A command buffer dropped unsubmitted recycles its chunks at
//! once.
//!
//! The retained timings' lock is never held while recycling, which takes
//! the query pools' lock.

#[cfg(test)]
mod gpu_tests;

use ash::vk;
use tracing::debug;

use super::VulkanDevice;
use super::error::error;
use super::query_pools::{QUERIES_PER_CHUNK, QueryChunk};
use crate::timing::{ScopeRecord, resolve};
use crate::{CommandBufferTimings, GpuTimepoint, QueueId, RhiResult, SubmissionTimings};

/// The timing scopes one command buffer recorded, and the query chunks
/// their timestamps are written to.
#[derive(Debug, Default)]
pub(super) struct RecordedTimings {
    /// Every scope, in the order they began.
    scopes: Vec<ScopeRecord>,
    /// The indices in `scopes` of the scopes still open, innermost last,
    /// while recording.
    open: Vec<usize>,
    /// The chunks claimed for the queries, in query order.
    chunks: Vec<QueryChunk>,
    /// How many queries have been written.
    queries: u32,
}

impl RecordedTimings {
    /// Opens a scope called `label`, whose beginning is written to `query`,
    /// or isn't written for `None`.
    pub(super) fn begin(&mut self, label: &str, query: Option<u32>) {
        let depth = u32::try_from(self.open.len()).expect("scopes nest fewer than 2^32 deep");
        self.open.push(self.scopes.len());
        self.scopes.push(ScopeRecord {
            label: label.to_owned(),
            depth,
            queries: query.map(|query| [query, query]),
        });
    }

    /// Closes the innermost open scope, whose end is written to `query`, or
    /// isn't written for `None`.
    ///
    /// # Panics
    ///
    /// Panics if no scope is open, which recording rules out.
    pub(super) fn end(&mut self, query: Option<u32>) {
        let index = self.open.pop().expect("recording checks a scope is open");
        if let (Some(queries), Some(query)) = (&mut self.scopes[index].queries, query) {
            queries[1] = query;
        }
    }

    /// The next query, claiming a chunk from `core`'s query pools if the
    /// last is full: its number among this command buffer's queries, and
    /// the pool and index it's written to.
    ///
    /// # Errors
    ///
    /// Returns an error if a chunk can't be claimed.
    pub(super) fn next_query(
        &mut self,
        core: &VulkanDevice,
    ) -> RhiResult<(u32, vk::QueryPool, u32)> {
        let query = self.queries;
        if query.is_multiple_of(QUERIES_PER_CHUNK) {
            self.chunks
                .push(core.query_pools.claim(core.device.handle())?);
        }
        self.queries += 1;
        let chunk = &self.chunks[(query / QUERIES_PER_CHUNK) as usize];
        Ok((query, chunk.pool(), query % QUERIES_PER_CHUNK))
    }

    /// Whether no scope was recorded.
    fn is_empty(&self) -> bool {
        self.scopes.is_empty()
    }
}

/// The timings of one submission, kept by its queue until they're taken:
/// each command buffer that recorded a scope, with its label, in
/// submission order.
pub(super) type KeptTimings = Vec<(Box<str>, RecordedTimings)>;

impl VulkanDevice {
    /// Keeps the timings of `command_buffers`, labelled, submitted to the
    /// queue `id` in the submission that reaches `timepoint`, if any
    /// recorded a scope. Then, if the queue keeps too many, discards the
    /// oldest completed ones, recycling their chunks.
    pub(super) fn keep_submitted_timings(
        &self,
        id: QueueId,
        timepoint: GpuTimepoint,
        command_buffers: Vec<(Box<str>, RecordedTimings)>,
    ) {
        let kept: KeptTimings = command_buffers
            .into_iter()
            .filter(|(_, timings)| !timings.is_empty())
            .collect();
        if kept.is_empty() {
            return;
        }
        let discarded = self
            .queues
            .keep_timings(self.device.handle(), id, timepoint.value(), kept);
        for (value, timings) in discarded {
            debug!(
                "discarded the timings of the {} queue's submission {value}, which weren't taken",
                self.queues.name(id),
            );
            for (_, timings) in timings {
                // SAFETY: The submission was discarded only once its queue's
                // timeline had reached it.
                unsafe { self.recycle_timings(timings) };
            }
        }
    }

    /// The timings of the submission to the queue `id` that reaches
    /// `timepoint`, once it has, recycling their chunks (see
    /// [`Queue::take_timings`](crate::Queue::take_timings)).
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline or the results can't be read.
    ///
    /// # Panics
    ///
    /// Panics if that queue didn't return `timepoint`.
    pub(super) fn take_timings(
        &self,
        id: QueueId,
        timepoint: GpuTimepoint,
    ) -> RhiResult<Option<SubmissionTimings>> {
        let device = self.device.handle();
        if !self.queues.is_complete(device, id, timepoint)? {
            return Ok(None);
        }
        let Some(kept) = self.queues.take_kept_timings(id, timepoint.value()) else {
            return Ok(Some(SubmissionTimings::default()));
        };
        let clock = self.queues.clock(id);
        let mut command_buffers = Vec::with_capacity(kept.len());
        let mut result = Ok(());
        for (label, timings) in kept {
            let RecordedTimings {
                scopes,
                chunks,
                queries,
                ..
            } = timings;
            if result.is_ok() {
                // SAFETY: The submission that wrote the queries has
                // completed, and the chunks were created on `device`.
                match unsafe { read_timestamps(device, &chunks, queries) } {
                    Ok(timestamps) => command_buffers.push(CommandBufferTimings {
                        label: label.into(),
                        scopes: resolve(scopes, clock, &timestamps),
                    }),
                    Err(err) => result = Err(error(err, "read the timings of", &label)),
                }
            }
            // Recycled whether or not they were read, so a failure doesn't
            // leak them.
            // SAFETY: As above.
            unsafe { self.recycle_chunks(chunks) };
        }
        result.map(|()| Some(SubmissionTimings { command_buffers }))
    }

    /// Recycles the chunks of `timings`.
    ///
    /// # Safety
    ///
    /// No pending work may use their queries: their command buffer was
    /// never submitted, or its submission has completed.
    pub(super) unsafe fn recycle_timings(&self, timings: RecordedTimings) {
        // SAFETY: Guaranteed by the caller.
        unsafe { self.recycle_chunks(timings.chunks) };
    }

    /// Recycles `chunks`.
    ///
    /// # Safety
    ///
    /// As for [`recycle_timings`](Self::recycle_timings).
    unsafe fn recycle_chunks(&self, chunks: Vec<QueryChunk>) {
        for chunk in chunks {
            // SAFETY: Guaranteed by the caller, and every chunk was claimed
            // from the core's query pools.
            unsafe { self.query_pools.recycle(self.device.handle(), chunk) };
        }
    }
}

/// Reads the first `queries` timestamps written to `chunks`, indexed by
/// query number.
///
/// # Safety
///
/// The submission that wrote them must have completed, and the chunks must
/// have been created on `device`.
unsafe fn read_timestamps(
    device: &ash::Device,
    chunks: &[QueryChunk],
    queries: u32,
) -> Result<Vec<u64>, vk::Result> {
    let mut timestamps = vec![0; queries as usize];
    for (chunk, timestamps) in chunks
        .iter()
        .zip(timestamps.chunks_mut(QUERIES_PER_CHUNK as usize))
    {
        // `WAIT` never blocks, since the submission has completed, and
        // makes sure every query is reported available.
        // SAFETY: Guaranteed by the caller, and `timestamps` holds a 64-bit
        // result for each query written to the chunk, starting at its first.
        unsafe {
            device.get_query_pool_results(
                chunk.pool(),
                0,
                timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }?;
    }
    Ok(timestamps)
}
