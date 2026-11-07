//! Command pools, claimed for each translation and recycled once the work
//! recorded from them completes (design 0002, "Command lists and
//! translation" and "Queues and submission").
//!
//! Vulkan requires a command pool, and the command buffers allocated from
//! it, to be used by one thread at a time. Each translation claims a pool
//! from its queue's free list for the whole call, so translations on
//! different threads never share one, and records into one of its command
//! buffers. The pool goes back on the list when that command buffer is
//! submitted, or dropped unsubmitted.
//!
//! A pool hands out several command buffers between resets, up to
//! [`BUFFERS_PER_POOL`], so it's claimed again while the work recorded from
//! it earlier is still pending. It's reset only once all of that work has
//! completed, which returns every command buffer to the initial state to be
//! reused, rather than freed and allocated again. So the number of pools
//! follows how many translations run at once, as vendors recommend a pool
//! for each recording thread and frame in flight, without the RHI knowing
//! about either.

use std::sync::{Mutex, PoisonError};

use ash::vk;

use super::error::error;
use crate::{RhiError, RhiResult};

/// How many command buffers a pool hands out before it's reset. A pool that
/// has handed out this many waits until all their work completes, so this
/// bounds the memory a busy pool holds, while letting a pool serve many
/// translations a frame.
const BUFFERS_PER_POOL: usize = 16;

/// A command pool, and what it has handed out since it was last reset.
///
/// There's one for each Vulkan pool, moved rather than copied: it's on its
/// family's free list, or held by the one translation that claimed it, so
/// only that translation's thread uses the pool.
#[derive(Debug)]
pub(super) struct CommandPool {
    /// The pool.
    pool: vk::CommandPool,
    /// Every command buffer allocated from `pool`, in the order they're
    /// handed out after each reset.
    buffers: Vec<vk::CommandBuffer>,
    /// How many of `buffers` have been handed out since the last reset. The
    /// last of them is the claimed one.
    used: usize,
    /// The timeline value after which no work recorded from the pool since
    /// its last reset is pending: the highest value a submission of one of
    /// its command buffers signals, or 0 if none has been submitted.
    reusable_at: u64,
}

impl CommandPool {
    /// The command buffer claimed with the pool: a primary command buffer
    /// in the initial state when claimed. A claimed pool has always handed
    /// out at least one.
    pub(super) fn buffer(&self) -> vk::CommandBuffer {
        self.buffers[self.used - 1]
    }

    /// The pool, for tests that check which pool is claimed.
    #[cfg(test)]
    pub(super) fn pool(&self) -> vk::CommandPool {
        self.pool
    }
}

/// How [`CommandPools::claim`] uses a free pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Choice {
    /// Reset the pool at this index, whose work has all completed, and use
    /// its first command buffer.
    Reset(usize),
    /// Use the next command buffer of the pool at this index, which has some
    /// left, while its earlier work may still be pending.
    Next(usize),
}

/// One queue family's command pools that aren't claimed.
pub(super) struct CommandPools {
    /// The family the pools are created for.
    family: u32,
    /// The pools not claimed, in the order they were recycled.
    free: Mutex<Vec<CommandPool>>,
}

impl CommandPools {
    /// An empty free list of pools for `family`.
    pub(super) fn new(family: u32) -> Self {
        Self {
            family,
            free: Mutex::new(Vec::new()),
        }
    }

    /// Claims a pool, and a command buffer from it in the initial state: a
    /// pool whose work has all completed, reset, or else one with command
    /// buffers left, or else a new one. `reached` reads the value the
    /// queue's timeline has reached, and is called only if a free pool has
    /// been submitted.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline can't be read, a pool can't be reset
    /// or created, or a command buffer can't be allocated.
    pub(super) fn claim(
        &self,
        device: &ash::Device,
        reached: impl FnOnce() -> RhiResult<u64>,
    ) -> RhiResult<CommandPool> {
        let chosen = {
            let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
            let reached = if free.iter().any(|pool| pool.reusable_at > 0) {
                reached()?
            } else {
                0
            };
            choose(&free, reached).map(|choice| match choice {
                Choice::Reset(index) => (free.swap_remove(index), true),
                Choice::Next(index) => (free.swap_remove(index), false),
            })
        };
        let mut pool = match chosen {
            Some((pool, false)) => pool,
            Some((mut pool, true)) => {
                // SAFETY: The pool was created on `device`, it's claimed, so
                // no other thread uses it, and the timeline has passed every
                // submission of its command buffers, so no pending work uses
                // them.
                let reset = unsafe {
                    device.reset_command_pool(pool.pool, vk::CommandPoolResetFlags::empty())
                };
                if let Err(result) = reset {
                    self.recycle(pool, 0);
                    return Err(self.error(result, "reset the command pool of queue family"));
                }
                pool.used = 0;
                pool.reusable_at = 0;
                pool
            }
            None => self.create(device)?,
        };
        if pool.used == pool.buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: The pool was created on `device`, it's claimed, so no
            // other thread uses it, and `allocate_info` is valid for the call.
            match unsafe { device.allocate_command_buffers(&allocate_info) } {
                Ok(buffers) => pool.buffers.extend(buffers),
                Err(result) => {
                    self.recycle(pool, 0);
                    return Err(self.error(result, "allocate a command buffer for queue family"));
                }
            }
        }
        // Every command buffer after those handed out since the last reset
        // is in the initial state: the reset returned it there, or it was
        // just allocated.
        pool.used += 1;
        Ok(pool)
    }

    /// Puts `pool` back on the free list, with its command buffer submitted
    /// in the submission that reaches `reusable_at`, or not submitted for 0.
    pub(super) fn recycle(&self, mut pool: CommandPool, reusable_at: u64) {
        pool.reusable_at = pool.reusable_at.max(reusable_at);
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(pool);
    }

    /// Creates a pool for the family, with no command buffers yet.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool can't be created.
    fn create(&self, device: &ash::Device) -> RhiResult<CommandPool> {
        // Command buffers are only reset with their pool, so the pool doesn't
        // need `RESET_COMMAND_BUFFER`, and each lives for one submission.
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(self.family);
        // SAFETY: `pool_create_info` is valid for the call, and the family is
        // one the device created a queue from.
        let pool = unsafe { device.create_command_pool(&pool_create_info, None) }
            .map_err(|result| self.error(result, "create a command pool for queue family"))?;
        Ok(CommandPool {
            pool,
            buffers: Vec::with_capacity(BUFFERS_PER_POOL),
            used: 0,
            reusable_at: 0,
        })
    }

    /// The error for `result` from `action`, naming the family.
    fn error(&self, result: vk::Result, action: &str) -> RhiError {
        error(result, action, &self.family.to_string())
    }

    /// Destroys every pool on the free list, and their command buffers.
    ///
    /// # Safety
    ///
    /// The pools must have been created on `device`, none may be claimed,
    /// and no pending work may use their command buffers, for example
    /// because the device is idle.
    pub(super) unsafe fn destroy(&mut self, device: &ash::Device) {
        let free = self.free.get_mut().unwrap_or_else(PoisonError::into_inner);
        for pool in free.drain(..) {
            // SAFETY: Guaranteed by the caller. Destroying a pool frees its
            // command buffers.
            unsafe { device.destroy_command_pool(pool.pool, None) };
        }
    }
}

/// How to claim from `free`, when the timeline has reached `reached`: the
/// first pool whose work has all completed, reset, or else the first with
/// command buffers left, or `None` to create a pool.
///
/// Resetting comes first, so pools are reused from their first command
/// buffer and stay small, and a pool with buffers left serves a claim while
/// every other pool's work is pending.
fn choose(free: &[CommandPool], reached: u64) -> Option<Choice> {
    free.iter()
        .position(|pool| pool.reusable_at <= reached)
        .map(Choice::Reset)
        .or_else(|| {
            free.iter()
                .position(|pool| pool.used < BUFFERS_PER_POOL)
                .map(Choice::Next)
        })
}

#[cfg(test)]
mod tests {
    //! T0 tests of choosing a free pool to claim, and of recycling one.

    use ash::vk::Handle;

    use super::*;

    /// A free pool, numbered `id`, that has handed out `used` command
    /// buffers, reusable at `reusable_at`.
    fn free(id: u64, used: usize, reusable_at: u64) -> CommandPool {
        CommandPool {
            pool: vk::CommandPool::from_raw(id),
            buffers: Vec::new(),
            used,
            reusable_at,
        }
    }

    /// A pool is reset once the timeline reaches the last submission of its
    /// command buffers, and a pool none of whose command buffers was
    /// submitted is reset at once.
    #[test]
    fn pools_whose_work_completed_are_reset() {
        let pools = [free(1, 3, 5), free(2, 3, 3), free(3, 1, 0)];
        assert_eq!(choose(&pools, 3), Some(Choice::Reset(1)));
        assert_eq!(choose(&pools, 5), Some(Choice::Reset(0)));
        assert_eq!(choose(&pools[..1], 4), Some(Choice::Next(0)));
        assert_eq!(choose(&pools[2..], 0), Some(Choice::Reset(0)));
    }

    /// While every pool's work is pending, the first with command buffers
    /// left hands out its next, and a pool that has handed out all of them
    /// waits for its reset.
    #[test]
    fn pending_pools_hand_out_buffers_they_have_left() {
        let pools = [free(1, BUFFERS_PER_POOL, 5), free(2, 4, 6), free(3, 2, 7)];
        assert_eq!(choose(&pools, 4), Some(Choice::Next(1)));
        assert_eq!(choose(&pools[..1], 4), None);
        assert_eq!(choose(&pools[..1], 5), Some(Choice::Reset(0)));
        assert_eq!(choose(&[], 10), None);
    }

    /// Recycling keeps the highest value of the pool's submissions since its
    /// last reset, so a command buffer dropped unsubmitted, or submitted
    /// earlier, doesn't let the pool be reset while later work is pending.
    #[test]
    fn recycling_keeps_the_latest_submission() {
        let pools = CommandPools::new(0);
        pools.recycle(free(1, 2, 7), 4);
        pools.recycle(free(2, 2, 7), 0);
        pools.recycle(free(3, 1, 0), 9);
        let free = pools.free.lock().unwrap();
        let reusable_at: Vec<_> = free.iter().map(|pool| pool.reusable_at).collect();
        assert_eq!(reusable_at, [7, 7, 9]);
    }
}
