//! T1 tests of timestamp query chunks on a real device (design 0002,
//! "Timestamps").

use std::slice;

use ash::vk;
use clockwork_diagnostics::capture::TestSession;

use super::QueryChunk;
use crate::QueueKind;
use crate::vulkan::queues::QueueRole;
use crate::vulkan::test_support::{TestCommands, VulkanTestBackend};

/// A chunk claimed from `backend`'s free list, recycled when dropped, so a
/// failing assertion doesn't leak its pool.
struct TestChunk<'a> {
    /// The backend whose free list the chunk was claimed from.
    backend: &'a VulkanTestBackend,
    /// The chunk, taken when it's recycled.
    chunk: Option<QueryChunk>,
}

impl<'a> TestChunk<'a> {
    /// Claims a chunk from `backend`'s free list.
    ///
    /// # Panics
    ///
    /// Panics if a query pool can't be created.
    fn claim(backend: &'a VulkanTestBackend) -> Self {
        let chunk = backend
            .core
            .query_pools
            .claim(backend.core.device.handle())
            .unwrap();
        Self {
            backend,
            chunk: Some(chunk),
        }
    }

    /// The chunk's pool.
    fn pool(&self) -> vk::QueryPool {
        self.chunk.as_ref().unwrap().pool()
    }
}

impl Drop for TestChunk<'_> {
    /// Waits for the device to be idle, so that no work uses the chunk,
    /// even if a test panicked with a submission pending, then recycles it.
    fn drop(&mut self) {
        let device = &self.backend.core.device;
        // A failure here leaves nothing better to do than recycle the chunk
        // anyway, which only a lost device's pending work could still use.
        // SAFETY: The device is live, its queues were retrieved from it, and
        // this thread holds none of their locks.
        let _ = unsafe { device.queue_locks().wait_idle(device.handle()) };
        if let Some(chunk) = self.chunk.take() {
            // SAFETY: The chunk was claimed from this free list, on this
            // device, which is idle, so no pending work uses it.
            unsafe {
                self.backend
                    .core
                    .query_pools
                    .recycle(device.handle(), chunk)
            };
        }
    }
}

/// Writes two timestamps into the first queries of `pool` on `backend`'s
/// graphics queue, recording into `commands`, and waits for them.
///
/// # Panics
///
/// Panics if any Vulkan call fails.
fn write_timestamps(backend: &VulkanTestBackend, commands: &TestCommands<'_>, pool: vk::QueryPool) {
    let device = backend.core.device.handle();
    let command_buffer = commands.command_buffer;
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: The pool was created on `device`, and no work recorded from it
    // is pending, since every submission is waited for. Resetting it returns
    // the command buffer to the initial state.
    unsafe {
        device
            .reset_command_pool(commands.pool, vk::CommandPoolResetFlags::empty())
            .unwrap();
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap();
    }
    // SAFETY: The command buffer is recording, for the graphics family,
    // which can write timestamps, and the queries are within the pool.
    unsafe {
        for query in 0..2 {
            device.cmd_write_timestamp2(
                command_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                pool,
                query,
            );
        }
        device.end_command_buffer(command_buffer).unwrap();
    }
    // SAFETY: The create info is valid for the call.
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
    let command_buffer_info = vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);
    let submit_info =
        vk::SubmitInfo2::default().command_buffer_infos(slice::from_ref(&command_buffer_info));
    let queue = backend
        .core
        .queues
        .lock(QueueRole::Kind(QueueKind::Graphics))
        .unwrap();
    // SAFETY: The queue is locked, the command buffer has ended and was
    // recorded for its family, and the fence is unsignalled. The submission
    // signals no timeline, so the queue's timepoints are unaffected.
    unsafe { device.queue_submit2(*queue, slice::from_ref(&submit_info), fence) }.unwrap();
    drop(queue);
    // SAFETY: The fence was created on `device`, and is destroyed once the
    // submission that signals it has completed.
    unsafe {
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        device.destroy_fence(fence, None);
    }
}

/// The first two queries of `pool`, each with its availability, read
/// without waiting: `NOT_READY` if either is unavailable.
fn first_two_queries(
    backend: &VulkanTestBackend,
    pool: vk::QueryPool,
) -> (Result<(), vk::Result>, [[u64; 2]; 2]) {
    let mut queries = [[0_u64; 2]; 2];
    // SAFETY: The pool was created on the device, with at least two
    // queries, and `queries` holds a 64-bit result and its availability for
    // each.
    let result = unsafe {
        backend.core.device.handle().get_query_pool_results(
            pool,
            0,
            &mut queries,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
        )
    };
    (result, queries)
}

/// A recycled query chunk is reset from the host and claimed again, so its
/// queries can be written again with no reset command, which validation
/// would report otherwise.
#[test]
fn recycled_query_chunks_are_reset_for_reuse() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let pools = &backend.core.query_pools;
    let commands = TestCommands::new(&backend);

    let chunk = TestChunk::claim(&backend);
    let pool = chunk.pool();
    write_timestamps(&backend, &commands, pool);
    let (result, written) = first_two_queries(&backend, pool);
    assert_eq!(result, Ok(()));
    assert!(written.iter().all(|&[_, available]| available != 0));

    drop(chunk);
    assert_eq!(pools.free_len(), 1);
    let chunk = TestChunk::claim(&backend);
    assert_eq!(chunk.pool(), pool);
    assert_eq!(pools.free_len(), 0);
    let (result, reset) = first_two_queries(&backend, pool);
    assert_eq!(result, Err(vk::Result::NOT_READY));
    assert!(reset.iter().all(|&[_, available]| available == 0));

    write_timestamps(&backend, &commands, chunk.pool());
}
