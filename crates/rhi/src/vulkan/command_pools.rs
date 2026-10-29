//! Command pools, checked out for each translation and recycled once the
//! work recorded from them completes (design 0002, "Command lists and
//! translation" and "Queues and submission").
//!
//! Vulkan requires a command pool, and the command buffers allocated from
//! it, to be used by one thread at a time. Each translation checks a pool
//! out of its queue's free list for the whole call, so translations on
//! different threads never share one. A pool holds one command buffer, and
//! goes back on the list when its command buffer is submitted, marked with
//! the timeline value that submission signals, or straight away when it's
//! dropped unsubmitted. It's reset when it's next checked out, once its
//! timeline has reached that value.

use std::sync::{Mutex, PoisonError};

use ash::vk;

use super::error::error;
use crate::RhiResult;

/// A command pool, and the one command buffer allocated from it.
#[derive(Debug, Clone, Copy)]
pub(super) struct PooledCommands {
    /// The pool.
    pub pool: vk::CommandPool,
    /// The primary command buffer allocated from `pool`.
    #[expect(dead_code, reason = "translation records into it in the next PR")]
    pub buffer: vk::CommandBuffer,
}

/// A pool on the free list.
#[derive(Debug)]
struct FreePool {
    /// The pool and its command buffer.
    commands: PooledCommands,
    /// The timeline value after which nothing pending uses it: the value its
    /// last submission signals, or 0 if it was never submitted.
    reusable_at: u64,
}

/// One queue family's command pools that aren't checked out.
pub(super) struct CommandPools {
    /// The family the pools are created for.
    family: u32,
    /// The pools not checked out, in the order they were given back.
    free: Mutex<Vec<FreePool>>,
}

impl CommandPools {
    /// An empty free list of pools for `family`.
    pub(super) fn new(family: u32) -> Self {
        Self {
            family,
            free: Mutex::new(Vec::new()),
        }
    }

    /// Checks out a pool whose work has completed, reset, or creates one.
    /// `reached` reads the value the queue's timeline has reached, and is
    /// called only if a free pool has been submitted.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline can't be read, or a pool can't be
    /// reset or created.
    pub(super) fn check_out(
        &self,
        device: &ash::Device,
        reached: impl FnOnce() -> RhiResult<u64>,
    ) -> RhiResult<PooledCommands> {
        let reusable = {
            let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
            let reached = if free.iter().any(|pool| pool.reusable_at > 0) {
                reached()?
            } else {
                0
            };
            first_reusable(&free, reached).map(|index| free.swap_remove(index))
        };
        let Some(pool) = reusable else {
            return self.create(device);
        };
        // SAFETY: The pool was created on `device`, it's checked out, so no
        // other thread uses it, and the timeline has passed its last
        // submission, so no pending work uses its command buffer.
        let reset = unsafe {
            device.reset_command_pool(pool.commands.pool, vk::CommandPoolResetFlags::empty())
        };
        if let Err(result) = reset {
            self.give_back(pool.commands, pool.reusable_at);
            return Err(error(
                result,
                "reset the command pool of queue family",
                &self.family.to_string(),
            ));
        }
        Ok(pool.commands)
    }

    /// Puts `commands` back on the free list, reusable once the queue's
    /// timeline reaches `reusable_at`, or straight away for 0.
    pub(super) fn give_back(&self, commands: PooledCommands, reusable_at: u64) {
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(FreePool {
                commands,
                reusable_at,
            });
    }

    /// Creates a pool for the family, with one primary command buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if either can't be created.
    fn create(&self, device: &ash::Device) -> RhiResult<PooledCommands> {
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(self.family);
        // SAFETY: `pool_create_info` is valid for the call, and the family is
        // one the device created a queue from.
        let pool =
            unsafe { device.create_command_pool(&pool_create_info, None) }.map_err(|result| {
                error(
                    result,
                    "create a command pool for queue family",
                    &self.family.to_string(),
                )
            })?;
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: The pool was just created on `device`, and `allocate_info`
        // is valid for the call.
        match unsafe { device.allocate_command_buffers(&allocate_info) } {
            Ok(buffers) => Ok(PooledCommands {
                pool,
                buffer: buffers[0],
            }),
            Err(result) => {
                // SAFETY: The pool was just created, and nothing uses it.
                unsafe { device.destroy_command_pool(pool, None) };
                Err(error(
                    result,
                    "allocate a command buffer for queue family",
                    &self.family.to_string(),
                ))
            }
        }
    }

    /// Destroys every pool on the free list, and their command buffers.
    ///
    /// # Safety
    ///
    /// The pools must have been created on `device`, none may be checked
    /// out, and no pending work may use their command buffers, for example
    /// because the device is idle.
    pub(super) unsafe fn destroy(&mut self, device: &ash::Device) {
        let free = self.free.get_mut().unwrap_or_else(PoisonError::into_inner);
        for pool in free.drain(..) {
            // SAFETY: Guaranteed by the caller. Destroying a pool frees its
            // command buffers.
            unsafe { device.destroy_command_pool(pool.commands.pool, None) };
        }
    }
}

/// The index of the first pool in `free` that no pending work uses, when the
/// timeline has reached `reached`.
fn first_reusable(free: &[FreePool], reached: u64) -> Option<usize> {
    free.iter().position(|pool| pool.reusable_at <= reached)
}

#[cfg(test)]
mod tests {
    //! T0 tests of choosing a free pool to reuse.

    use ash::vk::Handle;

    use super::*;

    /// A free pool, numbered `id`, reusable at `reusable_at`.
    fn free(id: u64, reusable_at: u64) -> FreePool {
        FreePool {
            commands: PooledCommands {
                pool: vk::CommandPool::from_raw(id),
                buffer: vk::CommandBuffer::null(),
            },
            reusable_at,
        }
    }

    /// A pool is reused once the timeline reaches its last submission's
    /// value, and a pool never submitted is reused at once.
    #[test]
    fn pools_are_reused_once_their_work_completes() {
        let pools = [free(1, 5), free(2, 3), free(3, 0)];
        assert_eq!(first_reusable(&pools, 2), Some(2));
        assert_eq!(first_reusable(&pools, 3), Some(1));
        assert_eq!(first_reusable(&pools, 5), Some(0));
        assert_eq!(first_reusable(&pools[..2], 2), None);
        assert_eq!(first_reusable(&[], 10), None);
    }
}
