//! The pure parts of the GPU memory allocator (design 0002, "Memory
//! allocator", and ADR 0006).
//!
//! Vulkan limits how many memory allocations a device may have at once, so
//! the RHI takes large blocks of memory from Vulkan and suballocates from
//! them. This crate is that policy, with no Vulkan types, so it's tested and
//! measured without a GPU:
//! - [`TlsfSuballocator`] suballocates a block's offsets
//! - a [`Pool`] holds the blocks of one memory type, and decides which block
//!   an allocation goes in and when a block is given back
//!
//! The Vulkan shell around them, which allocates, maps and frees the blocks,
//! is `clockwork-rhi`'s `MemoryAllocator`.

mod pool;
mod tlsf;

pub use pool::{Pool, PoolAllocation, PoolStatistics, ResourceKind};
pub use tlsf::{Allocation, TlsfSuballocator};
