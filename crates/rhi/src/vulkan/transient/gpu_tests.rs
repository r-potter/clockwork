//! T1 tests of transient memory on a real device (design 0002, R23).

use std::panic::{self, AssertUnwindSafe};
use std::thread;

use clockwork_diagnostics::capture::TestSession;

use crate::test_shaders;
use crate::transient::CHUNK_SIZE;
use crate::vulkan::test_support::VulkanTestBackend;
use crate::{
    Access, Barrier, Buffer, BufferDesc, BufferUsage, CommandList, CullMode, Extent2d, Extent3d,
    Format, GpuAddress, GpuResources, LoadOp, MemoryLocation, Offset2d, QueueKind, Rect2d,
    RenderHardwareInterface, RenderPassDesc, RenderState, SubmitDesc, TextureCopy,
    TextureDataLayout, TextureDesc, TextureDimension, TextureUsage, Viewport,
};

/// Four threads each allocate and write one quadrant's colour in a frame's
/// transient memory, and four draws, each scissored to its quadrant, read
/// them. Every tenth frame, the first quadrant's colour is at the end of an
/// allocation larger than a chunk. Over 100 frames, each rendered as its
/// threads wrote it, the frames reuse one chunk of the pool, never
/// creating another, and the large chunks are freed (R23's
/// `transient_data_reaches_shaders`).
#[test]
fn transient_data_reaches_shaders() {
    const SIZE: u32 = 8;
    const HALF: u32 = SIZE / 2;
    const FRAMES: u32 = 100;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let target = resources
        .create_texture(&TextureDesc {
            label: "target",
            dimension: TextureDimension::D2,
            format: Format::Rgba32Float,
            extent: Extent3d::new_2d(SIZE, SIZE),
            mip_levels: 1,
            array_layers: 1,
            usage: TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE,
        })
        .unwrap();
    let extent = Extent3d::new_2d(SIZE, SIZE);
    let layout = TextureDataLayout::tightly_packed(Format::Rgba32Float, extent);
    let size = u64::from(layout.bytes_per_row * SIZE);
    let readback = resources
        .create_buffer(&BufferDesc {
            label: "readback",
            size,
            usage: BufferUsage::COPIES,
            location: MemoryLocation::Readback,
        })
        .unwrap();
    let program = resources
        .create_program(&test_shaders::read_colour().desc())
        .unwrap();
    let attachment = Extent2d {
        width: SIZE,
        height: SIZE,
    };
    // Each quadrant's top-left corner, in the order the threads draw them.
    let quadrants = [(0, 0), (HALF, 0), (0, HALF), (HALF, HALF)];
    let colour = |frame: u32, quadrant: usize| {
        [
            frame as f32 / FRAMES as f32,
            quadrant as f32 / 4.0,
            0.5,
            1.0,
        ]
    };
    let graphics = backend.queue(QueueKind::Graphics);

    for frame_index in 0..FRAMES {
        // As a frame loop does, collect what earlier frames left, which
        // brings the last frame's chunk back to the pool.
        resources.collect_garbage();
        let counts = backend.core.transient_counts();
        assert_eq!(counts.in_use, 0, "frame {frame_index}: {counts:?}");
        assert!(counts.free <= 1, "frame {frame_index}: {counts:?}");

        let frame = resources.begin_transient_frame();
        let large = frame_index % 10 == 0;
        let addresses: Vec<GpuAddress> = thread::scope(|scope| {
            let threads: Vec<_> = (0..quadrants.len())
                .map(|quadrant| {
                    let frame = &frame;
                    scope.spawn(move || {
                        let bytes: Vec<u8> = colour(frame_index, quadrant)
                            .iter()
                            .flat_map(|channel| channel.to_le_bytes())
                            .collect();
                        let offset = if large && quadrant == 0 {
                            CHUNK_SIZE
                        } else {
                            0
                        };
                        let mut allocation = frame.allocate(offset + 16, 16).unwrap();
                        allocation.bytes_mut()[offset as usize..].copy_from_slice(&bytes);
                        GpuAddress(allocation.address().0 + offset)
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect()
        });

        let mut list = CommandList::new("transient colours");
        list.barrier(&[Barrier::texture(
            target,
            Access::None,
            Access::ColorAttachmentWrite,
        )]);
        list.bind_program(program);
        list.set_render_state(&RenderState::default().cull(CullMode::None));
        list.set_viewport(&Viewport::from_extent(attachment));
        list.begin_render_pass(&RenderPassDesc::colour(target, LoadOp::DontCare));
        for (&(x, y), address) in quadrants.iter().zip(&addresses) {
            list.set_scissor(Rect2d {
                offset: Offset2d { x, y },
                extent: Extent2d {
                    width: HALF,
                    height: HALF,
                },
            });
            list.push_data(0, &address.0.to_le_bytes());
            list.draw(3, 1);
        }
        list.end_render_pass();
        list.barrier(&[Barrier::texture(
            target,
            Access::ColorAttachmentWrite,
            Access::CopyRead,
        )]);
        list.copy_texture_to_buffer(
            TextureCopy {
                texture: target,
                mip: 0,
                layers: 0..1,
                origin: [0; 3],
                extent,
            },
            resources.buffer_address(readback),
            layout,
        );
        list.barrier(&[Barrier::buffer(
            readback,
            Access::CopyWrite,
            Access::HostRead,
        )]);
        // Every list that uses the frame's allocations is translated before
        // the frame ends.
        let command_buffer = graphics.translate(&list).unwrap();
        resources.end_transient_frame(frame);
        let done = graphics
            .submit(SubmitDesc::new(vec![command_buffer]))
            .unwrap();
        graphics.wait(done).unwrap();

        let texels = read_texels(&**resources, readback, size);
        for (quadrant, &(left, top)) in quadrants.iter().enumerate() {
            let expected = colour(frame_index, quadrant);
            for y in top..top + HALF {
                for x in left..left + HALF {
                    let at = ((y * SIZE + x) * 4) as usize;
                    assert_eq!(
                        texels[at..at + 4],
                        expected,
                        "frame {frame_index}, texel ({x}, {y}) of quadrant {quadrant}",
                    );
                }
            }
        }
    }

    resources.collect_garbage();
    let counts = backend.core.transient_counts();
    assert_eq!(counts.in_use, 0, "{counts:?}");
    assert_eq!(counts.free, 1, "{counts:?}");
    assert_eq!(counts.created, 1, "the frames reused one chunk: {counts:?}");
    resources.destroy_program(program);
    resources.destroy_buffer(readback);
    resources.destroy_texture(target);
}

/// A frame's chunks aren't reused while a list translated in the frame is
/// outstanding, and come back to the pool once its submission completes.
#[test]
fn chunks_wait_for_lists_translated_in_their_frame() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let graphics = backend.queue(QueueKind::Graphics);

    let frame = resources.begin_transient_frame();
    let first = frame.allocate(16, 16).unwrap().address();
    let command_buffer = graphics
        .translate(&CommandList::new("uses the frame"))
        .unwrap();
    resources.end_transient_frame(frame);
    resources.collect_garbage();
    let counts = backend.core.transient_counts();
    assert_eq!((counts.free, counts.in_use), (0, 0), "{counts:?}");

    // A frame begun meanwhile gets a new chunk. Each allocation is at the
    // start of its chunk, so their addresses are the chunks'.
    let frame = resources.begin_transient_frame();
    let second = frame.allocate(16, 16).unwrap().address();
    assert_ne!(first, second);
    resources.end_transient_frame(frame);

    let done = graphics
        .submit(SubmitDesc::new(vec![command_buffer]))
        .unwrap();
    graphics.wait(done).unwrap();
    resources.collect_garbage();
    let counts = backend.core.transient_counts();
    assert_eq!(counts.free, 2, "both frames' chunks are back: {counts:?}");
    assert_eq!(counts.created, 2, "{counts:?}");
}

/// An allocation larger than half a block of upload memory, which gets
/// memory of its own, starts at an aligned address, and a draw reads what
/// was written at its end.
#[test]
fn allocations_larger_than_a_block_are_aligned() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let colour = [0.25, 0.5, 0.75, 1.0];
    let size = 40 << 20;

    let frame = resources.begin_transient_frame();
    let mut allocation = frame.allocate(size, 256).unwrap();
    assert_eq!(
        allocation.address().0 % 256,
        0,
        "{:?}",
        allocation.address()
    );
    let bytes: Vec<u8> = colour
        .iter()
        .flat_map(|channel: &f32| channel.to_le_bytes())
        .collect();
    let end = size as usize - bytes.len();
    allocation.bytes_mut()[end..].copy_from_slice(&bytes);
    let address = GpuAddress(allocation.address().0 + end as u64);
    assert_eq!(render_colour(&backend, address), colour);
    resources.end_transient_frame(frame);
}

/// Ending a frame on a device other than the one it was begun on panics,
/// rather than ending it there.
#[test]
fn frames_end_on_their_own_device() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let other = VulkanTestBackend::headless(&session);
    let frame = backend.resources().begin_transient_frame();
    let panic = panic::catch_unwind(AssertUnwindSafe(|| {
        other.resources().end_transient_frame(frame);
    }))
    .expect_err("ending the frame on another device should panic");
    assert_eq!(
        panic.downcast_ref::<&str>().copied(),
        Some("a transient frame is ended on the device it was begun on"),
    );
}

/// Draws a 1×1 attachment with the colour at `address`, and returns what
/// was rendered.
fn render_colour(backend: &VulkanTestBackend, address: GpuAddress) -> [f32; 4] {
    let resources = backend.resources();
    let extent = Extent3d::new_2d(1, 1);
    let target = resources
        .create_texture(&TextureDesc {
            label: "target",
            dimension: TextureDimension::D2,
            format: Format::Rgba32Float,
            extent,
            mip_levels: 1,
            array_layers: 1,
            usage: TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE,
        })
        .unwrap();
    let readback = resources
        .create_buffer(&BufferDesc {
            label: "readback",
            size: 16,
            usage: BufferUsage::COPIES,
            location: MemoryLocation::Readback,
        })
        .unwrap();
    let program = resources
        .create_program(&test_shaders::read_colour().desc())
        .unwrap();
    let attachment = Extent2d {
        width: 1,
        height: 1,
    };
    let mut list = CommandList::new("read colour");
    list.barrier(&[Barrier::texture(
        target,
        Access::None,
        Access::ColorAttachmentWrite,
    )]);
    list.bind_program(program);
    list.set_render_state(&RenderState::default().cull(CullMode::None));
    list.set_viewport(&Viewport::from_extent(attachment));
    list.set_scissor(Rect2d::from_extent(attachment));
    list.push_data(0, &address.0.to_le_bytes());
    list.begin_render_pass(&RenderPassDesc::colour(target, LoadOp::DontCare));
    list.draw(3, 1);
    list.end_render_pass();
    list.barrier(&[Barrier::texture(
        target,
        Access::ColorAttachmentWrite,
        Access::CopyRead,
    )]);
    list.copy_texture_to_buffer(
        TextureCopy {
            texture: target,
            mip: 0,
            layers: 0..1,
            origin: [0; 3],
            extent,
        },
        resources.buffer_address(readback),
        TextureDataLayout::tightly_packed(Format::Rgba32Float, extent),
    );
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    let graphics = backend.queue(QueueKind::Graphics);
    let command_buffer = graphics.translate(&list).unwrap();
    let done = graphics
        .submit(SubmitDesc::new(vec![command_buffer]))
        .unwrap();
    graphics.wait(done).unwrap();
    let texels = read_texels(&**resources, readback, 16);
    resources.destroy_program(program);
    resources.destroy_buffer(readback);
    resources.destroy_texture(target);
    texels.try_into().unwrap()
}

/// The `f32`s in a readback buffer of `size` bytes.
fn read_texels(resources: &dyn GpuResources, buffer: Buffer, size: u64) -> Vec<f32> {
    let mut bytes = vec![0; size as usize];
    resources.read_buffer(buffer, 0, &mut bytes).unwrap();
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "the buffer holds whole f32s");
    chunks
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect()
}
