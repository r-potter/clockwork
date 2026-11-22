//! Creating resources and looking them up (design 0002, "Performance"):
//! a handle lookup, through `buffer_address`, is budgeted at under 20 ns,
//! `texture_index` for a view that already exists at under 50 ns, and
//! creating and destroying a buffer, when its pool has room, at under
//! 10 µs.

use std::hint::black_box;
use std::time::{Duration, Instant};

use clockwork_rhi::{
    Buffer, BufferDesc, BufferUsage, Extent3d, Format, GpuResources, MemoryLocation, TextureDesc,
    TextureDimension, TextureUsage, TextureView,
};
use criterion::Criterion;

use crate::Context;

/// The buffers the lookup benchmark looks up, so that the table holds as
/// many handles as a modest scene's.
const LOOKUP_BUFFERS: usize = 1_024;

/// The buffers created and destroyed between garbage collections in the
/// creation benchmark.
const CREATION_BATCH: u64 = 256;

/// The size of each buffer the benchmarks create: a typical small
/// buffer, which the allocator suballocates from a block.
const BUFFER_SIZE: u64 = 64 << 10;

/// Benchmarks looking up a handle as `resources/buffer_address`, a cached
/// `texture_index` as `resources/texture_index`, and creating and
/// destroying a buffer as `resources/create_buffer`.
pub(crate) fn bench(criterion: &mut Criterion, context: &Context) {
    let resources = context.resources();
    lookup(criterion, resources);
    texture_index(criterion, resources);
    creation(criterion, resources);
}

/// A buffer called `label` of [`BUFFER_SIZE`] bytes, in device memory.
///
/// # Panics
///
/// Panics if the buffer can't be created.
fn buffer(resources: &dyn GpuResources, label: &str) -> Buffer {
    resources
        .create_buffer(&BufferDesc {
            label,
            size: BUFFER_SIZE,
            usage: BufferUsage::STORAGE,
            location: MemoryLocation::Device,
        })
        .unwrap_or_else(|err| panic!("failed to create buffer \"{label}\": {err:?}"))
}

/// Benchmarks `buffer_address`, which looks the buffer up in the buffer
/// table under its read lock, and does nothing else.
///
/// The RHI holds the lock once for many lookups only inside translation,
/// which callers can't reach on its own, so this budget includes taking
/// the lock. The lookups cycle through [`LOOKUP_BUFFERS`] buffers, a step
/// apart that's coprime with their number, so that consecutive lookups
/// don't hit the same entry.
fn lookup(criterion: &mut Criterion, resources: &dyn GpuResources) {
    let buffers: Vec<Buffer> = (0..LOOKUP_BUFFERS)
        .map(|index| buffer(resources, &format!("lookup {index}")))
        .collect();
    let mut index = 0;
    criterion.bench_function("resources/buffer_address", |bencher| {
        bencher.iter(|| {
            index = (index + 257) % LOOKUP_BUFFERS;
            resources.buffer_address(black_box(buffers[index]))
        });
    });
    for buffer in buffers {
        resources.destroy_buffer(buffer);
    }
    resources.collect_garbage();
}

/// Benchmarks `texture_index` for a sampled view of a texture, which the
/// first call creates, so that every call timed finds it.
fn texture_index(criterion: &mut Criterion, resources: &dyn GpuResources) {
    let texture = resources
        .create_texture(&TextureDesc {
            label: "sampled",
            dimension: TextureDimension::D2,
            format: Format::Rgba8Unorm,
            extent: Extent3d::new_2d(64, 64),
            mip_levels: 1,
            array_layers: 1,
            usage: TextureUsage::SAMPLED,
        })
        .unwrap_or_else(|err| panic!("failed to create the sampled texture: {err:?}"));
    let index = resources
        .texture_index(texture, TextureView::sampled())
        .expect("the heap has room");
    criterion.bench_function("resources/texture_index", |bencher| {
        bencher.iter(|| {
            let cached = resources.texture_index(black_box(texture), TextureView::sampled());
            assert_eq!(cached.expect("the view exists"), index);
        });
    });
    resources.destroy_texture(texture);
    resources.collect_garbage();
}

/// Benchmarks creating a buffer and destroying it, in batches of
/// [`CREATION_BATCH`]. Destroying a buffer only retires it, so garbage is
/// collected after each batch, which frees them and leaves the pool room
/// for the next. That's timed too, since it's where the destruction is
/// done.
fn creation(criterion: &mut Criterion, resources: &dyn GpuResources) {
    criterion.bench_function("resources/create_buffer", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let mut done = 0;
            while done < iters {
                let batch = (iters - done).min(CREATION_BATCH);
                let start = Instant::now();
                for _ in 0..batch {
                    resources.destroy_buffer(buffer(resources, "created"));
                }
                resources.collect_garbage();
                total += start.elapsed();
                done += batch;
            }
            total
        });
    });
}
