//! The blocks of one memory type, and the policy for using them (design
//! 0002, "Memory allocator").
//!
//! A [`Pool`] decides which block an allocation goes in, when a new block is
//! needed, and when an empty one is given back. It's generic over the
//! block's memory handle, and asks its caller for new blocks, so it does no
//! Vulkan calls itself, and is tested with fake handles.

#[cfg(debug_assertions)]
use std::collections::BTreeMap;

use crate::tlsf::{Allocation, TlsfSuballocator};

/// What a range of memory holds. Buffers and images are kept in separate
/// blocks, so that `bufferImageGranularity`, which can forbid linear and
/// optimal resources from sharing a page, never needs to be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    /// A buffer, or another linear resource.
    Buffer,
    /// An image with optimal tiling.
    Image,
}

/// A block of memory and the suballocator over it.
struct Block<M> {
    /// Identifies the block within its pool, stable while it exists.
    id: u64,
    /// The block's memory handle, given by the pool's caller.
    memory: M,
    /// Suballocates the block's range.
    suballocator: TlsfSuballocator,
    /// The live ranges, as start to end, so that debug builds check that no
    /// two overlap (design 0002, "Risks").
    #[cfg(debug_assertions)]
    live: BTreeMap<u64, u64>,
}

/// A range allocated from a [`Pool`], to give back to [`Pool::free`].
#[derive(Debug)]
pub struct PoolAllocation {
    /// The block list the range is in.
    kind: ResourceKind,
    /// The [`Block::id`] of the block the range is in.
    block: u64,
    /// The range within the block.
    allocation: Allocation,
}

impl PoolAllocation {
    /// Where the range starts within its block.
    pub fn offset(&self) -> u64 {
        self.allocation.offset()
    }

    /// The range's length, in bytes.
    pub fn size(&self) -> u64 {
        self.allocation.size()
    }
}

/// What a pool holds, for statistics and tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStatistics {
    /// The blocks the pool holds, in both lists.
    pub blocks: usize,
    /// The bytes in those blocks.
    pub reserved: u64,
    /// The bytes in live allocations.
    pub allocated: u64,
}

/// The blocks of one memory type, in a list for buffers and one for images,
/// each block a fixed size.
///
/// An empty block is given back, unless it's the last one in its list, so
/// that a frame that frees and reallocates the same resources doesn't churn
/// block allocations. Each list therefore holds at most one empty block.
pub struct Pool<M> {
    /// The size of every block, in bytes.
    block_size: u64,
    /// The blocks holding buffers.
    buffers: Vec<Block<M>>,
    /// The blocks holding images.
    images: Vec<Block<M>>,
    /// The [`Block::id`] the next block gets.
    next_id: u64,
}

impl<M: Copy> Pool<M> {
    /// A pool of blocks of `block_size` bytes, holding none yet.
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is zero.
    pub fn new(block_size: u64) -> Self {
        assert!(block_size > 0, "a memory pool needs a block size");
        Self {
            block_size,
            buffers: Vec::new(),
            images: Vec::new(),
            next_id: 0,
        }
    }

    /// The size of every block, in bytes.
    pub fn block_size(&self) -> u64 {
        self.block_size
    }

    /// Allocates `size` bytes at a multiple of `alignment`, for a resource of
    /// `kind`, from the first block of that kind with room. If none has
    /// room, calls `new_block` with the block size for a new block's memory,
    /// and allocates from that.
    ///
    /// Returns the memory of the block the range is in, and the range.
    ///
    /// # Errors
    ///
    /// Returns `new_block`'s error, leaving the pool unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the allocation can't fit in an empty block, which the
    /// caller avoids by giving larger resources dedicated memory.
    pub fn allocate<E>(
        &mut self,
        kind: ResourceKind,
        size: u64,
        alignment: u64,
        new_block: impl FnOnce(u64) -> Result<M, E>,
    ) -> Result<(M, PoolAllocation), E> {
        assert!(
            size.checked_add(alignment - 1)
                .is_some_and(|needed| needed <= self.block_size),
            "{size} bytes at alignment {alignment} can't fit in a {}-byte block",
            self.block_size,
        );
        let blocks = match kind {
            ResourceKind::Buffer => &mut self.buffers,
            ResourceKind::Image => &mut self.images,
        };
        for block in blocks.iter_mut() {
            if let Some(allocation) = block.suballocator.allocate(size, alignment) {
                return Ok(Self::record(kind, block, allocation));
            }
        }

        let memory = new_block(self.block_size)?;
        let mut block = Block {
            id: self.next_id,
            memory,
            suballocator: TlsfSuballocator::new(self.block_size),
            #[cfg(debug_assertions)]
            live: BTreeMap::new(),
        };
        self.next_id += 1;
        let allocation = block
            .suballocator
            .allocate(size, alignment)
            .expect("an empty block holds any allocation that fits its size");
        let result = Self::record(kind, &mut block, allocation);
        blocks.push(block);
        Ok(result)
    }

    /// Frees `allocation`. Returns the memory of its block, for the caller
    /// to give back, if that leaves the block empty and it isn't the last
    /// block in its list.
    ///
    /// # Panics
    ///
    /// Panics if `allocation` didn't come from this pool.
    pub fn free(&mut self, allocation: PoolAllocation) -> Option<M> {
        let blocks = match allocation.kind {
            ResourceKind::Buffer => &mut self.buffers,
            ResourceKind::Image => &mut self.images,
        };
        let index = blocks
            .iter()
            .position(|block| block.id == allocation.block)
            .expect("freed an allocation from a block this pool doesn't hold");
        let block = &mut blocks[index];
        #[cfg(debug_assertions)]
        block.live.remove(&allocation.allocation.offset());
        block.suballocator.free(allocation.allocation);
        if block.suballocator.allocated() == 0 && blocks.len() > 1 {
            Some(blocks.remove(index).memory)
        } else {
            None
        }
    }

    /// The pool's blocks and bytes.
    pub fn statistics(&self) -> PoolStatistics {
        let blocks = self.buffers.iter().chain(&self.images);
        PoolStatistics {
            blocks: self.buffers.len() + self.images.len(),
            reserved: (self.buffers.len() + self.images.len()) as u64 * self.block_size,
            allocated: blocks.map(|block| block.suballocator.allocated()).sum(),
        }
    }

    /// Removes every block, whether or not it's empty, and returns their
    /// memory, for the caller to give back when it's destroyed.
    pub fn drain(&mut self) -> Vec<M> {
        self.buffers
            .drain(..)
            .chain(self.images.drain(..))
            .map(|block| block.memory)
            .collect()
    }

    /// Wraps `allocation`, from `block`, as a [`PoolAllocation`], checking
    /// in debug builds that it overlaps no live range.
    fn record(
        kind: ResourceKind,
        block: &mut Block<M>,
        allocation: Allocation,
    ) -> (M, PoolAllocation) {
        #[cfg(debug_assertions)]
        {
            let (start, end) = (allocation.offset(), allocation.offset() + allocation.size());
            let before = block.live.range(..end).next_back();
            assert!(
                before.is_none_or(|(_, &before_end)| before_end <= start),
                "the suballocator returned {start}..{end}, overlapping a live range",
            );
            block.live.insert(start, end);
        }
        (
            block.memory,
            PoolAllocation {
                kind,
                block: block.id,
                allocation,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the pool's block policy, with fake memory handles (design
    //! 0002, R19).

    use std::cell::Cell;

    use proptest::prelude::*;

    use super::*;

    /// A pool of 1 KiB blocks, and a source of fake memory handles that
    /// counts the blocks it has made.
    struct Fixture {
        /// The pool under test, with `u32` handles.
        pool: Pool<u32>,
        /// The next handle to give a new block.
        next_memory: Cell<u32>,
    }

    impl Fixture {
        /// A pool of 1 KiB blocks.
        fn new() -> Self {
            Self {
                pool: Pool::new(1_024),
                next_memory: Cell::new(0),
            }
        }

        /// Allocates `size` bytes of `kind`, making new blocks as needed.
        fn allocate(&mut self, kind: ResourceKind, size: u64) -> (u32, PoolAllocation) {
            let next = &self.next_memory;
            self.pool
                .allocate(kind, size, 1, |block_size| {
                    assert_eq!(block_size, 1_024);
                    next.set(next.get() + 1);
                    Ok::<_, ()>(next.get() - 1)
                })
                .unwrap()
        }
    }

    /// The first allocation makes a block, and later ones share it.
    #[test]
    fn allocations_share_a_block() {
        let mut fixture = Fixture::new();
        let (first, _a) = fixture.allocate(ResourceKind::Buffer, 100);
        let (second, b) = fixture.allocate(ResourceKind::Buffer, 100);
        assert_eq!((first, second), (0, 0));
        assert_eq!(b.offset(), 100);
        assert_eq!(fixture.pool.statistics().blocks, 1);
    }

    /// A new block is made only when no block has room.
    #[test]
    fn a_full_block_makes_another() {
        let mut fixture = Fixture::new();
        let (first, _a) = fixture.allocate(ResourceKind::Buffer, 1_000);
        let (second, _b) = fixture.allocate(ResourceKind::Buffer, 100);
        let (third, _c) = fixture.allocate(ResourceKind::Buffer, 24);
        assert_eq!((first, second, third), (0, 1, 0));
    }

    /// Buffers and images never share a block.
    #[test]
    fn buffers_and_images_use_separate_blocks() {
        let mut fixture = Fixture::new();
        let (buffer, _a) = fixture.allocate(ResourceKind::Buffer, 100);
        let (image, _b) = fixture.allocate(ResourceKind::Image, 100);
        assert_ne!(buffer, image);
        assert_eq!(fixture.pool.statistics().blocks, 2);
    }

    /// An empty block is given back unless it's the last in its list.
    #[test]
    fn empty_blocks_are_given_back_except_the_last() {
        let mut fixture = Fixture::new();
        let (_, a) = fixture.allocate(ResourceKind::Buffer, 1_000);
        let (_, b) = fixture.allocate(ResourceKind::Buffer, 1_000);
        assert_eq!(fixture.pool.free(a), Some(0));
        assert_eq!(fixture.pool.free(b), None);
        assert_eq!(fixture.pool.statistics().blocks, 1);
    }

    /// The statistics count blocks, their bytes and the bytes allocated.
    #[test]
    fn statistics_count_blocks_and_bytes() {
        let mut fixture = Fixture::new();
        let (_, a) = fixture.allocate(ResourceKind::Buffer, 300);
        let _b = fixture.allocate(ResourceKind::Image, 200);
        assert_eq!(
            fixture.pool.statistics(),
            PoolStatistics {
                blocks: 2,
                reserved: 2_048,
                allocated: 500,
            }
        );
        assert_eq!(a.size(), 300);
        fixture.pool.free(a);
        assert_eq!(fixture.pool.statistics().allocated, 200);
    }

    /// A failure to make a block is returned, and leaves the pool unchanged.
    #[test]
    fn a_failed_block_leaves_the_pool_unchanged() {
        let mut pool = Pool::<u32>::new(1_024);
        let result = pool.allocate(ResourceKind::Buffer, 100, 1, |_| Err("out of memory"));
        assert_eq!(result.unwrap_err(), "out of memory");
        assert_eq!(pool.statistics(), PoolStatistics::default());
    }

    /// Draining returns every block's memory, empty or not.
    #[test]
    fn draining_returns_every_block() {
        let mut fixture = Fixture::new();
        let _a = fixture.allocate(ResourceKind::Buffer, 1_000);
        let _b = fixture.allocate(ResourceKind::Buffer, 1_000);
        let _c = fixture.allocate(ResourceKind::Image, 10);
        let mut memory = fixture.pool.drain();
        memory.sort_unstable();
        assert_eq!(memory, [0, 1, 2]);
        assert_eq!(fixture.pool.statistics(), PoolStatistics::default());
    }

    /// An allocation that can't fit in an empty block is a bug in the
    /// caller, which should have used dedicated memory.
    #[test]
    #[should_panic(expected = "can't fit")]
    fn oversized_allocations_panic() {
        Fixture::new().allocate(ResourceKind::Buffer, 1_025);
    }

    /// The block size is what the pool was made with.
    #[test]
    fn the_block_size_is_fixed() {
        assert_eq!(Pool::<u32>::new(4_096).block_size(), 4_096);
    }

    proptest! {
        /// Random allocations and frees of buffers and images: the bytes
        /// allocated always match the live allocations, no list ever holds
        /// more than one empty block, and draining returns every block made
        /// and not given back.
        #[test]
        fn blocks_match_the_live_allocations(
            steps in prop::collection::vec((any::<bool>(), 1..=512_u64, any::<usize>()), 1..200),
        ) {
            let mut fixture = Fixture::new();
            let mut live = Vec::new();
            let mut given_back = 0;
            for (allocate, size, index) in steps {
                if allocate || live.is_empty() {
                    let kind = if size % 2 == 0 { ResourceKind::Buffer } else { ResourceKind::Image };
                    live.push(fixture.allocate(kind, size).1);
                } else {
                    let allocation = live.swap_remove(index % live.len());
                    if fixture.pool.free(allocation).is_some() {
                        given_back += 1;
                    }
                }
                let allocated: u64 = live.iter().map(PoolAllocation::size).sum();
                prop_assert_eq!(fixture.pool.statistics().allocated, allocated);
                for blocks in [&fixture.pool.buffers, &fixture.pool.images] {
                    let empty = blocks.iter().filter(|block| block.suballocator.allocated() == 0).count();
                    prop_assert!(empty <= 1);
                }
            }
            let made = fixture.next_memory.get() as usize;
            prop_assert_eq!(fixture.pool.drain().len(), made - given_back);
        }
    }
}
