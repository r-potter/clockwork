//! Benchmarks of the allocator's pure parts (design 0002, "Performance"):
//! allocating and freeing with the TLSF suballocator, which is budgeted at
//! under 100 ns each, and with the pool, which adds the search for a block
//! with room.
//!
//! Each benchmark runs in a steady state, on memory fragmented the way a
//! long-running application's is. It allocates a batch of ranges of random
//! sizes, then frees them all in a random order, and times only the
//! operation it measures. Every batch gives back what it took, so the state
//! is the same at the start of each one. The sizes come from a fixed seed,
//! so every run measures the same work.
//!
//! Run them with `cargo bench -p clockwork-gpu-memory`.

use std::convert::Infallible;
use std::hint::black_box;
use std::time::{Duration, Instant};

use clockwork_gpu_memory::{Allocation, Pool, PoolAllocation, ResourceKind, TlsfSuballocator};
use criterion::{Criterion, criterion_group, criterion_main};

/// The TLSF benchmark's block size: the allocator's block size for
/// device-local memory.
const TLSF_BLOCK_SIZE: u64 = 256 << 20;

/// The pool benchmark's block size, small enough that the pool holds several
/// blocks and searches them.
const POOL_BLOCK_SIZE: u64 = 16 << 20;

/// About how many blocks the pool benchmark's pool holds, before it's
/// fragmented.
const POOL_BLOCKS: u64 = 8;

/// The alignment of every allocation, a common buffer alignment.
const ALIGNMENT: u64 = 256;

/// The smallest allocation, in bytes.
const MIN_SIZE: u64 = 256;

/// The powers of two above [`MIN_SIZE`] that sizes span, so that the
/// largest are just under 128 KiB.
const SIZE_OCTAVES: u64 = 8;

/// The allocations in one batch.
const BATCH: usize = 1_024;

/// The batches generated, which the benchmarks cycle through.
const BATCHES: usize = 16;

/// The seed for every benchmark's random sizes and orders.
const SEED: u64 = 0x5eed_c10c_4057_a110;

/// A xorshift64 generator, so that every run sees the same sizes, with no
/// dependency for random numbers.
struct Rng(u64);

impl Rng {
    /// The next 64 random bits.
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A random number below `n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A random allocation size, from [`MIN_SIZE`] to just under 128 KiB.
    /// Its power of two is uniform, so small sizes are as common as large
    /// ones, as with a scene's resources.
    fn size(&mut self) -> u64 {
        let octave = MIN_SIZE << self.below(SIZE_OCTAVES + 1);
        octave + self.below(octave)
    }

    /// The indices `0..len`, shuffled.
    fn permutation(&mut self, len: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..len).collect();
        for i in (1..len).rev() {
            let j = self.below(i as u64 + 1) as usize;
            indices.swap(i, j);
        }
        indices
    }
}

/// One batch's work: the sizes to allocate, in order, and the order to free
/// them in.
struct Batch {
    /// The size of each allocation.
    sizes: Vec<u64>,
    /// Indices into [`sizes`](Self::sizes), in the order they're freed.
    free_order: Vec<usize>,
}

/// [`BATCHES`] batches of random sizes and orders.
fn batches(rng: &mut Rng) -> Vec<Batch> {
    (0..BATCHES)
        .map(|_| Batch {
            sizes: (0..BATCH).map(|_| rng.size()).collect(),
            free_order: rng.permutation(BATCH),
        })
        .collect()
}

/// What a benchmark times.
#[derive(Debug, Clone, Copy)]
enum Operation {
    /// Allocating each range.
    Allocate,
    /// Freeing each range.
    Free,
}

/// Something the benchmarks allocate from: the suballocator or the pool.
trait Allocator {
    /// What an allocation returns, to give back to [`free`](Self::free).
    type Allocation;

    /// Allocates `size` bytes at [`ALIGNMENT`].
    ///
    /// # Panics
    ///
    /// Panics if there's no room, which the benchmarks' sizes avoid.
    fn allocate(&mut self, size: u64) -> Self::Allocation;

    /// Frees `allocation`.
    fn free(&mut self, allocation: Self::Allocation);
}

impl Allocator for TlsfSuballocator {
    type Allocation = Allocation;

    fn allocate(&mut self, size: u64) -> Allocation {
        TlsfSuballocator::allocate(self, size, ALIGNMENT)
            .expect("the fragmented block has room for a batch")
    }

    fn free(&mut self, allocation: Allocation) {
        TlsfSuballocator::free(self, allocation);
    }
}

/// A pool of buffer blocks, with fake memory handles.
struct BufferPool {
    /// The pool, whose handles count the blocks made.
    pool: Pool<u64>,
    /// The handle the next new block gets.
    next_memory: u64,
}

impl Allocator for BufferPool {
    type Allocation = PoolAllocation;

    fn allocate(&mut self, size: u64) -> PoolAllocation {
        let next_memory = &mut self.next_memory;
        let (_, allocation) = self
            .pool
            .allocate(ResourceKind::Buffer, size, ALIGNMENT, |_| {
                *next_memory += 1;
                Ok::<_, Infallible>(*next_memory)
            })
            .unwrap_or_else(|never| match never {});
        allocation
    }

    fn free(&mut self, allocation: PoolAllocation) {
        // A block given back needs nothing done with its fake handle.
        let _ = self.pool.free(allocation);
    }
}

/// Fragments `allocator`: allocates random sizes until `fill` bytes are in
/// use, then frees a random half of the allocations, leaving holes of every
/// size between those kept.
///
/// Returns the allocations kept, which must outlive the benchmark so that
/// the holes stay.
fn fragment<A: Allocator>(allocator: &mut A, rng: &mut Rng, fill: u64) -> Vec<A::Allocation> {
    let mut allocations = Vec::new();
    let mut allocated = 0;
    while allocated < fill {
        let size = rng.size();
        allocations.push(Some(allocator.allocate(size)));
        allocated += size;
    }
    for index in rng.permutation(allocations.len()) {
        if index % 2 == 0 {
            let allocation = allocations[index].take().expect("freed once");
            allocator.free(allocation);
        }
    }
    allocations.into_iter().flatten().collect()
}

/// The time `iters` of `operation` take on `allocator`, cycling through
/// `batches`.
///
/// Whole batches are run, so the time measured is scaled to `iters`
/// operations. Criterion asks for thousands of iterations or more, so the
/// last batch's remainder makes no difference.
fn measure<A: Allocator>(
    allocator: &mut A,
    batches: &[Batch],
    operation: Operation,
    iters: u64,
) -> Duration {
    let mut live: Vec<Option<A::Allocation>> = Vec::with_capacity(BATCH);
    let mut total = Duration::ZERO;
    let mut done = 0;
    for batch in batches.iter().cycle() {
        if done >= iters {
            break;
        }
        let start = Instant::now();
        for &size in &batch.sizes {
            live.push(Some(allocator.allocate(black_box(size))));
        }
        let allocating = start.elapsed();

        let start = Instant::now();
        for &index in &batch.free_order {
            allocator.free(black_box(live[index].take().expect("freed once")));
        }
        let freeing = start.elapsed();
        live.clear();

        total += match operation {
            Operation::Allocate => allocating,
            Operation::Free => freeing,
        };
        done += BATCH as u64;
    }
    total.mul_f64(iters as f64 / done as f64)
}

/// Benchmarks allocating and freeing with `allocator`, fragmented to `fill`
/// bytes, as `group/allocate` and `group/free`.
fn bench_allocator<A: Allocator>(
    criterion: &mut Criterion,
    group: &str,
    mut allocator: A,
    fill: u64,
) {
    let mut rng = Rng(SEED);
    let _kept = fragment(&mut allocator, &mut rng, fill);
    let batches = batches(&mut rng);
    let mut group = criterion.benchmark_group(group);
    for (name, operation) in [("allocate", Operation::Allocate), ("free", Operation::Free)] {
        group.bench_function(name, |bencher| {
            bencher.iter_custom(|iters| measure(&mut allocator, &batches, operation, iters));
        });
    }
    group.finish();
}

/// The TLSF suballocator alone, over one device-local block, half filled
/// before it's fragmented. The budget is under 100 ns for each operation.
fn tlsf(criterion: &mut Criterion) {
    bench_allocator(
        criterion,
        "tlsf",
        TlsfSuballocator::new(TLSF_BLOCK_SIZE),
        TLSF_BLOCK_SIZE / 2,
    );
}

/// The pool, with [`POOL_BLOCKS`] blocks filled before they're fragmented,
/// so that allocations search several blocks for room. It has no budget of
/// its own: it shows what the pool adds to the suballocator.
fn pool(criterion: &mut Criterion) {
    let pool = BufferPool {
        pool: Pool::new(POOL_BLOCK_SIZE),
        next_memory: 0,
    };
    bench_allocator(criterion, "pool", pool, POOL_BLOCKS * POOL_BLOCK_SIZE);
}

criterion_group!(benches, tlsf, pool);
criterion_main!(benches);
