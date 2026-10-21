//! A TLSF (two-level segregated fit) suballocator over a range of offsets
//! (design 0002, "Memory allocator").
//!
//! [`TlsfSuballocator`] manages the offsets `0..capacity` of something it
//! doesn't own, such as a block of device memory. It hands out aligned ranges
//! of it, and takes them back, with no knowledge of what the range is.
//!
//! Free ranges are kept in lists indexed by size class. The first level has
//! a class per power of two, and the second splits each into
//! [`SL_COUNT`] classes. Two bitmaps record which lists are non-empty, so
//! allocating finds a large enough free range with two bit scans, and freeing
//! merges a range with its free neighbours, both in constant time.
//!
//! Unlike textbook TLSF, an allocation fails only when no free range can
//! hold it: when the constant-time search finds nothing,
//! [`TlsfSuballocator::allocate`] looks through the lists that might still
//! hold a range that fits (see [`TlsfSuballocator::find_in_lower_classes`]).
//! That only happens when the caller would otherwise need a new block.
//!
//! # How it works
//!
//! Every range, free or allocated, has a [`RangeNode`]. The ranges tile
//! `0..capacity` with no gaps, and each node links to its physical
//! neighbours, the ranges just before and after it. Each free node is also in
//! a doubly linked list of the free ranges in its size class. Nodes live in
//! a `Vec` and link to each other by index, and nodes freed by merging are
//! reused.
//!
//! ## Size classes
//!
//! A size's first-level class is its power of two, given by its highest set
//! bit. Its second-level class is given by the [`SL_BITS`] bits below that,
//! which split the power of two into [`SL_COUNT`] equal steps. For example,
//! 1,000 is `0b11_1110_1000`. Its highest set bit is bit 9, so it's in the
//! power of two from 512 to 1,023, which is split into steps of 32 bytes.
//! The next four bits are `1111`, so it's in the last step, the class of
//! ranges from 992 to 1,023 bytes. Sizes below [`SL_COUNT`] don't have
//! [`SL_BITS`] bits below their highest bit, so they share first-level class
//! 0, with a second-level class for each size.
//!
//! The classes are in order of size, so all the ranges in one class are
//! smaller than all the ranges in the next.
//!
//! ## Finding a free range
//!
//! Bit `f` of the first-level bitmap is set when any list in first-level
//! class `f` is non-empty, and bit `s` of the second-level bitmap for `f` is
//! set when the list for class `(f, s)` is non-empty. To find the smallest
//! non-empty class at or above a class `(f, s)`, mask off the bits below `s`
//! in the second-level bitmap for `f`, and take the lowest bit left. If there
//! isn't one, mask off the bits up to and including `f` in the first-level
//! bitmap, take the lowest bit left, and then the lowest bit of that first
//! level's second-level bitmap. Each step is a single bit scan.
//!
//! ## Allocating
//!
//! 1. Work out the smallest size a free range can be and still hold the
//!    allocation, wherever it starts: the size asked for, plus the most that
//!    aligning its start could skip, which is `alignment - 1` bytes.
//! 2. Round that up to the start of a class. The class the size itself is
//!    in also holds ranges smaller than it, so it's skipped. Every range in
//!    the class rounded up to, and in the classes above, is large enough.
//! 3. Take the first range in the smallest non-empty class at or above the
//!    rounded-up class. This is a good fit rather than the best one, since
//!    the class may have smaller ranges that would also do, but it takes
//!    constant time.
//! 4. If there's no such class, look through the ranges in the classes
//!    skipped in step 2 for one that fits. If none does, the allocation
//!    fails.
//! 5. Remove the range from its free list. Split off the gap before the
//!    aligned offset, if there is one, and the space after the allocation,
//!    if there's any, and add each to the free list for its own size.
//!
//! ## Freeing
//!
//! 1. Check that the allocation belongs to this suballocator, and that it
//!    hasn't already been freed.
//! 2. If the range after it is free, remove that range from its free list,
//!    and merge it into the freed range.
//! 3. If the range before it is free, remove that range from its free list,
//!    and merge the freed range into it.
//! 4. Add the merged range to the free list for its size.
//!
//! No two free ranges are ever next to each other, since freeing always
//! merges them, so freeing needs at most these two merges.

use std::fmt;

/// The second-level classes per first level, as a power of two.
const SL_BITS: u32 = 4;

/// The second-level classes per first level.
const SL_COUNT: usize = 1 << SL_BITS;

/// The first-level classes: sizes below [`SL_COUNT`] share the first, with
/// a second-level class per size, and each power of two from [`SL_COUNT`]
/// up to 2⁶³ has its own.
const FL_COUNT: usize = 64 - SL_BITS as usize + 1;

/// An index into [`TlsfSuballocator::nodes`].
type RangeNodeIndex = u32;

/// A size class: a first-level and a second-level index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SizeClass {
    /// The first-level index, below [`FL_COUNT`].
    first: usize,
    /// The second-level index, below [`SL_COUNT`].
    second: usize,
}

impl SizeClass {
    /// The class holding ranges of `size` bytes: the class whose sizes start
    /// at or below `size`.
    fn containing(size: u64) -> Self {
        // First-level class 0 holds the sizes below `SL_COUNT`, with a
        // second-level class for each, so small sizes map to it directly.
        if size < SL_COUNT as u64 {
            return Self {
                first: 0,
                second: size as usize,
            };
        }
        let log2 = size.ilog2();
        Self {
            // `SL_COUNT`, the power of two just above the small sizes, is
            // first-level class 1.
            first: (log2 - SL_BITS + 1) as usize,
            // Shifting right leaves the highest set bit, which is worth
            // `SL_COUNT`, and the `SL_BITS` bits below it, which give the
            // step within the power of two.
            second: (size >> (log2 - SL_BITS)) as usize - SL_COUNT,
        }
    }

    /// The first class every range in which holds at least `size` bytes, or
    /// `None` if there's none, for sizes too close to `u64::MAX`.
    fn at_least(size: u64) -> Option<Self> {
        if size < SL_COUNT as u64 {
            return Some(Self::containing(size));
        }
        // The classes for this power of two are `step` bytes wide, so the
        // next multiple of `step` starts a class. That's `size` itself if
        // it starts its class. Otherwise it's the start of the next class,
        // skipping `size`'s own class, which also holds smaller ranges.
        let step = 1 << (size.ilog2() - SL_BITS);
        Some(Self::containing(size.checked_next_multiple_of(step)?))
    }

    /// The class's position among all classes, in order of size.
    fn flat(self) -> usize {
        self.first * SL_COUNT + self.second
    }

    /// The class at position `flat` among all classes.
    fn from_flat(flat: usize) -> Self {
        Self {
            first: flat / SL_COUNT,
            second: flat % SL_COUNT,
        }
    }
}

/// A range of offsets, free or allocated, and its neighbours.
///
/// The ranges tile `0..capacity`: each one's `next_physical` starts where it
/// ends. No two neighbours are both free, since freeing merges them.
#[derive(Debug, Clone)]
struct RangeNode {
    /// Where the range starts.
    offset: u64,
    /// The range's length, in bytes. Never zero.
    size: u64,
    /// Whether the range is free, and so in a free list.
    free: bool,
    /// The range that ends where this one starts, if any.
    prev_physical: Option<RangeNodeIndex>,
    /// The range that starts where this one ends, if any.
    next_physical: Option<RangeNodeIndex>,
    /// The previous range in this one's free list. Unused when allocated.
    prev_free: Option<RangeNodeIndex>,
    /// The next range in this one's free list. Unused when allocated.
    next_free: Option<RangeNodeIndex>,
}

/// An allocated range, returned by [`TlsfSuballocator::allocate`] and given
/// back to [`TlsfSuballocator::free`].
///
/// It isn't `Clone`, so that the same range can't be freed twice.
#[derive(Debug, PartialEq, Eq)]
pub struct Allocation {
    /// Where the range starts. A multiple of the alignment asked for.
    offset: u64,
    /// The range's length, in bytes, as asked for.
    size: u64,
    /// The node that holds the range.
    node: RangeNodeIndex,
}

impl Allocation {
    /// Where the range starts.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The range's length, in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// A TLSF suballocator over the offsets `0..capacity`. See the module
/// documentation.
pub struct TlsfSuballocator {
    /// The offsets managed are `0..capacity`.
    capacity: u64,
    /// The bytes in allocated ranges.
    allocated: u64,
    /// Every range, free or allocated, and unused nodes, indexed by
    /// [`RangeNodeIndex`].
    nodes: Vec<RangeNode>,
    /// Nodes in [`nodes`](Self::nodes) that hold no range, for reuse.
    unused_nodes: Vec<RangeNodeIndex>,
    /// Bit `f` is set when some list in first-level class `f` is non-empty.
    first_level: u64,
    /// For each first-level class, bit `s` is set when the list for
    /// second-level class `s` is non-empty.
    second_level: [u16; FL_COUNT],
    /// The first range in each class's free list.
    heads: [[Option<RangeNodeIndex>; SL_COUNT]; FL_COUNT],
}

impl TlsfSuballocator {
    /// A suballocator over `0..capacity`, all of it free.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(capacity: u64) -> Self {
        assert!(capacity > 0, "a TLSF suballocator needs a capacity");
        let mut tlsf = Self {
            capacity,
            allocated: 0,
            nodes: Vec::new(),
            unused_nodes: Vec::new(),
            first_level: 0,
            second_level: [0; FL_COUNT],
            heads: [[None; SL_COUNT]; FL_COUNT],
        };
        let node = tlsf.new_node(0, capacity);
        tlsf.insert_free_node(node);
        tlsf
    }

    /// The size of the range managed, in bytes.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The bytes in allocated ranges.
    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    /// Allocates `size` bytes at a multiple of `alignment`, or returns
    /// `None` if no free range can hold them.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero, or `alignment` isn't a power of two.
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Option<Allocation> {
        assert!(size > 0, "can't allocate zero bytes");
        assert!(
            alignment.is_power_of_two(),
            "alignment {alignment} isn't a power of two"
        );

        // Any range of at least `size + alignment - 1` bytes can hold the
        // allocation, whatever its offset, so look in the first class whose
        // ranges are all that large, or a larger one. Either step fails only
        // for sizes too close to `u64::MAX` to have such a class, and then
        // only the slow search below can find a range.
        let worst_case = size.checked_add(alignment - 1);
        let large_enough = worst_case.and_then(SizeClass::at_least);
        // Take the first range in the smallest non-empty class that's large
        // enough, in constant time. If there's none, fall back to checking
        // ranges one by one in the classes that might hold a fit.
        let node = large_enough
            .and_then(|class| self.first_non_empty(class))
            .and_then(|class| self.heads[class.first][class.second])
            .or_else(|| self.find_in_lower_classes(size, alignment, large_enough))?;

        self.remove_free_node(node);
        let start = self.nodes[node as usize].offset;
        let offset = start.next_multiple_of(alignment);
        // Return the gap before the aligned offset, and the space after the
        // allocation, to the free lists. Each goes in the list for its own
        // size, which is usually a smaller class than the range's was.
        let node = if offset > start {
            let (gap, rest) = self.split_node(node, offset - start);
            self.insert_free_node(gap);
            rest
        } else {
            node
        };
        if self.nodes[node as usize].size > size {
            let (_, rest) = self.split_node(node, size);
            self.insert_free_node(rest);
        }
        self.allocated += size;
        Some(Allocation { offset, size, node })
    }

    /// Frees `allocation`, merging its range with any free neighbours.
    ///
    /// # Panics
    ///
    /// Panics if `allocation` didn't come from this suballocator.
    pub fn free(&mut self, allocation: Allocation) {
        let mut node = allocation.node;
        let is_live = self.nodes.get(node as usize).is_some_and(|live| {
            !live.free && live.offset == allocation.offset && live.size == allocation.size
        });
        assert!(
            is_live,
            "freed an allocation this TLSF suballocator didn't make: {allocation:?}"
        );
        self.allocated -= allocation.size;

        // Merge with the free neighbours on either side, so that no two
        // free ranges are next to each other. Each neighbour leaves its
        // free list first, since its size, and so its class, changes. The
        // next neighbour merges into this node, and this node into the
        // previous one, so the merged range keeps the lowest node's offset.
        if let Some(next) = self.nodes[node as usize].next_physical
            && self.nodes[next as usize].free
        {
            self.remove_free_node(next);
            self.merge_next_node(node);
        }
        if let Some(prev) = self.nodes[node as usize].prev_physical
            && self.nodes[prev as usize].free
        {
            self.remove_free_node(prev);
            self.merge_next_node(prev);
            node = prev;
        }
        self.insert_free_node(node);
    }

    /// The first non-empty class at or above `class`, if any.
    fn first_non_empty(&self, class: SizeClass) -> Option<SizeClass> {
        // Look in `class`'s own first level, with the bits of the smaller
        // second-level classes masked off. The lowest bit left is the
        // smallest non-empty class.
        let second = self.second_level[class.first] & (u16::MAX << class.second);
        if second != 0 {
            return Some(SizeClass {
                first: class.first,
                second: second.trailing_zeros() as usize,
            });
        }
        // Otherwise take the smallest non-empty first level above it. Every
        // class in it is larger than `class`, so its smallest non-empty
        // class will do, and its bit is set because its list is non-empty.
        let first = self.first_level & u64::MAX.checked_shl(class.first as u32 + 1).unwrap_or(0);
        if first == 0 {
            return None;
        }
        let first = first.trailing_zeros() as usize;
        Some(SizeClass {
            first,
            second: self.second_level[first].trailing_zeros() as usize,
        })
    }

    /// Looks for a free range that holds `size` bytes at a multiple of
    /// `alignment`, in the classes below `large_enough` (or every class from
    /// `size`'s up, if it's `None`). These are the classes that may hold
    /// ranges large enough, but not all of them, so their ranges are checked
    /// one by one.
    fn find_in_lower_classes(
        &self,
        size: u64,
        alignment: u64,
        large_enough: Option<SizeClass>,
    ) -> Option<RangeNodeIndex> {
        // The classes below `size`'s hold only smaller ranges, and the
        // constant-time search has found the classes from `large_enough` up
        // empty, so only the classes between them can hold a fit. They're
        // searched smallest first, to leave larger ranges for larger
        // requests.
        let end = large_enough.map_or(FL_COUNT * SL_COUNT, SizeClass::flat);
        (SizeClass::containing(size).flat()..end)
            .map(SizeClass::from_flat)
            .find_map(|class| {
                let mut next = self.heads[class.first][class.second];
                while let Some(node) = next {
                    let range = &self.nodes[node as usize];
                    // A range fits if `size` bytes remain in it after
                    // skipping to its first aligned offset.
                    let fits = range
                        .offset
                        .checked_next_multiple_of(alignment)
                        .and_then(|offset| (offset - range.offset).checked_add(size))
                        .is_some_and(|needed| needed <= range.size);
                    if fits {
                        return Some(node);
                    }
                    next = range.next_free;
                }
                None
            })
    }

    /// Adds a node for the range `offset..offset + size`, allocated and with
    /// no neighbours.
    fn new_node(&mut self, offset: u64, size: u64) -> RangeNodeIndex {
        let node = RangeNode {
            offset,
            size,
            free: false,
            prev_physical: None,
            next_physical: None,
            prev_free: None,
            next_free: None,
        };
        if let Some(index) = self.unused_nodes.pop() {
            self.nodes[index as usize] = node;
            index
        } else {
            let index = RangeNodeIndex::try_from(self.nodes.len())
                .expect("a TLSF suballocator has more than u32::MAX ranges");
            self.nodes.push(node);
            index
        }
    }

    /// Splits `node`'s range after `size` bytes, into `node` and a new node
    /// for the rest, and returns both. Neither is in a free list.
    fn split_node(&mut self, node: RangeNodeIndex, size: u64) -> (RangeNodeIndex, RangeNodeIndex) {
        let (offset, total, next) = {
            let range = &self.nodes[node as usize];
            (range.offset, range.size, range.next_physical)
        };
        // Link the new node in between `node` and its old next neighbour.
        let rest = self.new_node(offset + size, total - size);
        self.nodes[rest as usize].prev_physical = Some(node);
        self.nodes[rest as usize].next_physical = next;
        if let Some(next) = next {
            self.nodes[next as usize].prev_physical = Some(rest);
        }
        let range = &mut self.nodes[node as usize];
        range.size = size;
        range.next_physical = Some(rest);
        (node, rest)
    }

    /// Merges `node`'s next neighbour into it, and frees the neighbour's
    /// node. Neither may be in a free list.
    fn merge_next_node(&mut self, node: RangeNodeIndex) {
        let next = self.nodes[node as usize]
            .next_physical
            .expect("merging with a next neighbour that exists");
        let (size, after) = {
            let range = &self.nodes[next as usize];
            (range.size, range.next_physical)
        };
        // Grow `node` over its neighbour, and unlink the neighbour by
        // linking `node` to the range after it.
        let range = &mut self.nodes[node as usize];
        range.size += size;
        range.next_physical = after;
        if let Some(after) = after {
            self.nodes[after as usize].prev_physical = Some(node);
        }
        self.unused_nodes.push(next);
    }

    /// Marks `node` free and adds it to the front of its class's list.
    fn insert_free_node(&mut self, node: RangeNodeIndex) {
        let class = SizeClass::containing(self.nodes[node as usize].size);
        let head = self.heads[class.first][class.second];
        let range = &mut self.nodes[node as usize];
        range.free = true;
        range.prev_free = None;
        range.next_free = head;
        if let Some(head) = head {
            self.nodes[head as usize].prev_free = Some(node);
        }
        self.heads[class.first][class.second] = Some(node);
        // The list is non-empty now, and so is its first level.
        self.second_level[class.first] |= 1 << class.second;
        self.first_level |= 1 << class.first;
    }

    /// Marks `node` allocated and removes it from its class's list.
    fn remove_free_node(&mut self, node: RangeNodeIndex) {
        let (class, prev, next) = {
            let range = &mut self.nodes[node as usize];
            range.free = false;
            (
                SizeClass::containing(range.size),
                range.prev_free,
                range.next_free,
            )
        };
        // Unlink the node from its list. With no previous node, it's the
        // list's head, so its next node becomes the head.
        match prev {
            Some(prev) => self.nodes[prev as usize].next_free = next,
            None => self.heads[class.first][class.second] = next,
        }
        if let Some(next) = next {
            self.nodes[next as usize].prev_free = prev;
        }
        // If the list is empty now, clear its bit, and its first level's bit
        // if that was the first level's last non-empty list.
        if self.heads[class.first][class.second].is_none() {
            self.second_level[class.first] &= !(1 << class.second);
            if self.second_level[class.first] == 0 {
                self.first_level &= !(1 << class.first);
            }
        }
    }
}

impl fmt::Debug for TlsfSuballocator {
    /// Shows the capacity and how much is allocated, rather than every
    /// node.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TlsfSuballocator")
            .field("capacity", &self.capacity)
            .field("allocated", &self.allocated)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the TLSF suballocator (design 0002, R19).

    use super::*;

    /// Small sizes each have a class of their own, and larger ones a
    /// sixteenth of their power of two.
    #[test]
    fn classes_split_each_power_of_two_into_sixteen() {
        let class = |first, second| SizeClass { first, second };
        assert_eq!(SizeClass::containing(0), class(0, 0));
        assert_eq!(SizeClass::containing(15), class(0, 15));
        assert_eq!(SizeClass::containing(16), class(1, 0));
        assert_eq!(SizeClass::containing(31), class(1, 15));
        assert_eq!(SizeClass::containing(32), class(2, 0));
        assert_eq!(SizeClass::containing(33), class(2, 0));
        assert_eq!(SizeClass::containing(34), class(2, 1));
        assert_eq!(
            SizeClass::containing(u64::MAX),
            class(FL_COUNT - 1, SL_COUNT - 1)
        );
    }

    /// Rounding up gives the first class whose every range is large enough.
    #[test]
    fn at_least_rounds_up_to_a_class_boundary() {
        assert_eq!(SizeClass::at_least(15), Some(SizeClass::containing(15)));
        assert_eq!(SizeClass::at_least(32), Some(SizeClass::containing(32)));
        assert_eq!(SizeClass::at_least(33), Some(SizeClass::containing(34)));
        assert_eq!(SizeClass::at_least(u64::MAX), None);
    }

    /// Flat positions order classes by size, and convert back.
    #[test]
    fn flat_positions_round_trip() {
        for size in [0, 1, 15, 16, 33, 1 << 20, u64::MAX] {
            let class = SizeClass::containing(size);
            assert_eq!(SizeClass::from_flat(class.flat()), class);
        }
        assert!(SizeClass::containing(1_000).flat() < SizeClass::containing(1_100).flat());
    }

    /// Allocations are placed one after another from the start.
    #[test]
    fn allocations_fill_from_the_start() {
        let mut tlsf = TlsfSuballocator::new(1_024);
        let first = tlsf.allocate(100, 1).unwrap();
        let second = tlsf.allocate(100, 1).unwrap();
        assert_eq!((first.offset(), second.offset()), (0, 100));
        assert_eq!((first.size(), second.size()), (100, 100));
        assert_eq!((tlsf.allocated(), tlsf.capacity()), (200, 1_024));
    }

    /// An aligned allocation starts at a multiple of its alignment, and the
    /// gap before it can be allocated again.
    ///
    /// The gap is 246 bytes, in the class of ranges from 240 to 247 bytes.
    /// A request for 240 bytes looks in that class first, and finds the gap.
    /// (One for all 246 bytes would look in the class above, and take the
    /// space after the aligned allocation instead, since TLSF is a good fit,
    /// not a best fit.)
    #[test]
    fn alignment_leaves_a_reusable_gap() {
        let mut tlsf = TlsfSuballocator::new(1_024);
        let _first = tlsf.allocate(10, 1).unwrap();
        let aligned = tlsf.allocate(16, 256).unwrap();
        assert_eq!(aligned.offset(), 256);
        let gap = tlsf.allocate(240, 1).unwrap();
        assert_eq!(gap.offset(), 10);
    }

    /// Allocation fails once nothing fits, and succeeds again after a free.
    #[test]
    fn allocation_fails_when_full() {
        let mut tlsf = TlsfSuballocator::new(100);
        let all = tlsf.allocate(100, 1).unwrap();
        assert_eq!(tlsf.allocate(1, 1), None);
        tlsf.free(all);
        assert!(tlsf.allocate(100, 1).is_some());
    }

    /// Freeing ranges in any order merges them, so the whole capacity can
    /// be allocated again.
    #[test]
    fn freeing_merges_neighbours() {
        let mut tlsf = TlsfSuballocator::new(300);
        let a = tlsf.allocate(100, 1).unwrap();
        let b = tlsf.allocate(100, 1).unwrap();
        let c = tlsf.allocate(100, 1).unwrap();
        tlsf.free(a);
        tlsf.free(c);
        tlsf.free(b);
        assert_eq!(tlsf.allocated(), 0);
        assert_eq!(tlsf.allocate(300, 1).unwrap().offset(), 0);
    }

    /// A free range in a class the constant-time search skips is still
    /// found: 33 bytes are free, in the class of ranges from 32 to 33 bytes,
    /// and the search for 33 bytes starts at the class above.
    #[test]
    fn a_fitting_range_in_a_lower_class_is_found() {
        let mut tlsf = TlsfSuballocator::new(64);
        let first = tlsf.allocate(33, 1).unwrap();
        let _rest = tlsf.allocate(31, 1).unwrap();
        tlsf.free(first);
        assert_eq!(tlsf.allocate(33, 1).unwrap().offset(), 0);
    }

    /// An allocation as large as any `u64` can be can't overflow the search.
    #[test]
    fn huge_requests_fail_cleanly() {
        let mut tlsf = TlsfSuballocator::new(1 << 20);
        assert_eq!(tlsf.allocate(u64::MAX, 1 << 63), None);
        assert_eq!(tlsf.allocate(u64::MAX, 1), None);
        let mut whole = TlsfSuballocator::new(u64::MAX);
        assert_eq!(whole.allocate(u64::MAX, 1).unwrap().offset(), 0);
    }

    /// Asking for nothing is a bug in the caller.
    #[test]
    #[should_panic(expected = "can't allocate zero bytes")]
    fn zero_sized_allocations_panic() {
        TlsfSuballocator::new(16).allocate(0, 1);
    }

    /// Alignments must be powers of two, as Vulkan's are.
    #[test]
    #[should_panic(expected = "isn't a power of two")]
    fn non_power_of_two_alignments_panic() {
        TlsfSuballocator::new(16).allocate(4, 3);
    }

    /// An allocation from another suballocator is rejected, rather than
    /// corrupting this one.
    #[test]
    #[should_panic(expected = "didn't make")]
    fn freeing_a_foreign_allocation_panics() {
        let mut first = TlsfSuballocator::new(64);
        let mut second = TlsfSuballocator::new(64);
        let _ = second.allocate(8, 1).unwrap();
        let foreign = second.allocate(8, 1).unwrap();
        let _ = first.allocate(8, 1).unwrap();
        first.free(foreign);
    }
}
