//! T1 tests of translating and submitting command lists on a real device
//! (design 0002, R4, R5, R6, R7, R9, R16, R21 and R25).
//!
//! Synchronization validation doesn't track copies and fills by address
//! (`VK_KHR_device_address_commands`), on either side, so a missing
//! barrier between them isn't reported, and R8's
//! `missing_barrier_is_reported`, which renders twice, is still to come in
//! `rhi-shaders`.

use std::panic::{self, AssertUnwindSafe};

use clockwork_diagnostics::capture::TestSession;

use crate::test_shaders;
use crate::vulkan::test_support::{Capability, VulkanTestBackend, require_capabilities};
use crate::{
    Access, AddressMode, Barrier, Blend, Buffer, BufferDesc, BufferUsage, ColorAttachment,
    ColourMask, ColourWrite, CommandList, CompareOp, CullMode, DepthAttachment, DepthTest,
    DescriptorIndex, Extent2d, Extent3d, Filter, Format, FrontFace, GpuAddress, GpuResources,
    IndexFormat, LoadOp, MemoryLocation, Offset2d, QueueId, QueueKind, Rect2d,
    RenderHardwareInterface, RenderPassDesc, RenderState, SamplerDesc, StoreOp, SubmitDesc,
    Subresource, Texture, TextureCopy, TextureDataLayout, TextureDesc, TextureDimension,
    TextureUsage, TextureView, Topology, Viewport,
};

/// A buffer of `size` bytes in `location`, for copies.
fn create(
    resources: &dyn GpuResources,
    label: &str,
    size: u64,
    location: MemoryLocation,
) -> Buffer {
    resources
        .create_buffer(&BufferDesc {
            label,
            size,
            usage: BufferUsage::COPIES,
            location,
        })
        .unwrap()
}

/// Translates `list` for the queue `kind`'s work goes to, submits it, and
/// waits for it.
fn run(backend: &VulkanTestBackend, kind: QueueKind, list: &CommandList) {
    let queue = backend.queue(kind);
    let command_buffer = queue.translate(list).unwrap();
    let done = queue.submit(SubmitDesc::new(vec![command_buffer])).unwrap();
    queue.wait(done).unwrap();
}

/// The message `translate` panics with, translating `list` for the queue
/// `kind`'s work goes to.
fn translation_panic(backend: &VulkanTestBackend, kind: QueueKind, list: &CommandList) -> String {
    let queue = backend.queue(kind);
    let panic = panic::catch_unwind(AssertUnwindSafe(|| queue.translate(list)))
        .expect_err("translation should panic");
    panic
        .downcast_ref::<String>()
        .expect("the panic has a formatted message")
        .clone()
}

/// A compute program writes through the address in its push data, written
/// in two blocks at their offsets, on the graphics queue and on the compute
/// queue, with the heaps bound, and the value is read back (R5's
/// `compute_writes_through_device_address`).
#[test]
fn compute_writes_through_device_address() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::write_value().desc())
        .unwrap();
    for kind in [QueueKind::Graphics, QueueKind::Compute] {
        let output = resources
            .create_buffer(&BufferDesc {
                label: "output",
                size: 4,
                usage: BufferUsage::STORAGE,
                location: MemoryLocation::Readback,
            })
            .unwrap();
        let mut list = CommandList::new("write value");
        list.bind_program(program);
        // In two halves, the high one first, so the shader finds the
        // address only if each lands at its offset, and the first is kept
        // when the second is written.
        let address = resources.buffer_address(output).0.to_le_bytes();
        list.push_data(4, &address[4..]);
        list.push_data(0, &address[..4]);
        list.dispatch([1, 1, 1]);
        list.barrier(&[Barrier::buffer(
            output,
            Access::ComputeShaderWrite,
            Access::HostRead,
        )]);
        run(&backend, kind, &list);
        let mut value = [0; 4];
        resources.read_buffer(output, 0, &mut value).unwrap();
        assert_eq!(u32::from_le_bytes(value), 0xC10C_C0DE, "{kind:?}");
        resources.destroy_buffer(output);
    }
    resources.destroy_program(program);
}

/// A compute program runs on workgroup counts the GPU wrote, read from
/// their address, on the graphics queue and on the compute queue, and the
/// number of workgroups it ran is read back. The arguments are in an
/// indirect buffer, and in one that's a storage buffer too, as one a
/// compute shader writes would be.
#[test]
fn compute_dispatches_indirectly() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::count_groups().desc())
        .unwrap();
    let usages = [
        BufferUsage::INDIRECT,
        BufferUsage::INDIRECT | BufferUsage::STORAGE,
    ];
    for (kind, usage) in [QueueKind::Graphics, QueueKind::Compute]
        .into_iter()
        .flat_map(|kind| usages.map(|usage| (kind, usage)))
    {
        let arguments = resources
            .create_buffer(&BufferDesc {
                label: "arguments",
                size: 16,
                usage,
                location: MemoryLocation::Device,
            })
            .unwrap();
        let counter = resources
            .create_buffer(&BufferDesc {
                label: "counter",
                size: 4,
                usage: BufferUsage::STORAGE,
                location: MemoryLocation::Readback,
            })
            .unwrap();
        let arguments_address = resources.buffer_address(arguments);
        let counter_address = resources.buffer_address(counter);
        let mut list = CommandList::new("count groups");
        // Two workgroups in each dimension, from 4 bytes in, so the count
        // is 8 only if all three are read, from there.
        list.fill_buffer(arguments_address, 4, 0);
        list.fill_buffer(GpuAddress(arguments_address.0 + 4), 12, 2);
        list.fill_buffer(counter_address, 4, 0);
        list.barrier(&[
            Barrier::buffer(arguments, Access::CopyWrite, Access::IndirectRead),
            Barrier::buffer(
                counter,
                Access::CopyWrite,
                Access::ComputeShaderRead | Access::ComputeShaderWrite,
            ),
        ]);
        list.bind_program(program);
        list.push_data(0, &counter_address.0.to_le_bytes());
        list.dispatch_indirect(GpuAddress(arguments_address.0 + 4));
        list.barrier(&[Barrier::buffer(
            counter,
            Access::ComputeShaderWrite,
            Access::HostRead,
        )]);
        run(&backend, kind, &list);
        let mut count = [0; 4];
        resources.read_buffer(counter, 0, &mut count).unwrap();
        assert_eq!(u32::from_le_bytes(count), 8, "{kind:?}, {usage:?}");
        resources.destroy_buffer(arguments);
        resources.destroy_buffer(counter);
    }
    resources.destroy_program(program);
}

/// An indirect dispatch needs a compute program bound too.
#[test]
fn indirect_dispatches_need_a_compute_program() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let mut list = CommandList::new("draw");
    list.bind_program(program);
    list.dispatch_indirect(GpuAddress(0x1000));
    assert_eq!(
        translation_panic(&backend, QueueKind::Graphics, &list),
        "command list \"draw\" dispatches with no compute program bound",
    );
    resources.destroy_program(program);
}

/// The push data for heap slots `indices`, as the test shaders declare it:
/// each a 4-byte `ResourceIndex` or `SamplerIndex`, in order.
fn index_push_data(indices: &[DescriptorIndex]) -> Vec<u8> {
    indices
        .iter()
        .flat_map(|index| index.0.to_le_bytes())
        .collect()
}

/// A compute program writes a storage view through its heap slot, and
/// another samples it through a sampled view and a sampler, and writes
/// what it read to a storage view of mip 1 of a second texture, which is
/// copied out and read back (part of R6 and R21).
#[test]
fn compute_reads_and_writes_heap_textures() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let source = resources
        .create_texture(&TextureDesc {
            label: "source",
            dimension: TextureDimension::D2,
            format: Format::Rgba32Float,
            extent: Extent3d::new_2d(4, 4),
            mip_levels: 1,
            array_layers: 1,
            usage: TextureUsage::SAMPLED | TextureUsage::STORAGE,
        })
        .unwrap();
    let destination = resources
        .create_texture(&TextureDesc {
            label: "destination",
            dimension: TextureDimension::D2,
            format: Format::Rgba32Float,
            extent: Extent3d::new_2d(8, 8),
            mip_levels: 2,
            array_layers: 1,
            usage: TextureUsage::STORAGE | TextureUsage::COPY_SOURCE,
        })
        .unwrap();
    let sampler = resources
        .create_sampler(&SamplerDesc {
            label: "nearest",
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mip_filter: Filter::Nearest,
            address_modes: [AddressMode::ClampToEdge; 3],
        })
        .unwrap();
    let extent = Extent3d::new_2d(4, 4);
    let layout = TextureDataLayout::tightly_packed(Format::Rgba32Float, extent);
    let size = u64::from(layout.bytes_per_row * 4);
    let readback = create(&**resources, "readback", size, MemoryLocation::Readback);
    let fill = resources
        .create_program(&test_shaders::fill_texels().desc())
        .unwrap();
    let copy = resources
        .create_program(&test_shaders::copy_texels().desc())
        .unwrap();
    let index = |texture, view| resources.texture_index(texture, view).unwrap();
    let source_storage = index(source, TextureView::Storage(Subresource::mip(0)));
    let source_sampled = index(source, TextureView::sampled());
    let mip_1 = index(destination, TextureView::Storage(Subresource::mip(1)));

    let mut list = CommandList::new("heap textures");
    list.barrier(&[
        Barrier::texture(source, Access::None, Access::ComputeShaderWrite),
        Barrier::subresource(
            destination,
            Subresource::mip(1),
            Access::None,
            Access::ComputeShaderWrite,
        ),
    ]);
    list.bind_program(fill);
    list.push_data(0, &index_push_data(&[source_storage]));
    list.dispatch([1, 1, 1]);
    list.barrier(&[Barrier::texture(
        source,
        Access::ComputeShaderWrite,
        Access::ComputeShaderSampledRead,
    )]);
    list.bind_program(copy);
    list.push_data(
        0,
        &index_push_data(&[source_sampled, resources.sampler_index(sampler), mip_1]),
    );
    list.dispatch([1, 1, 1]);
    list.barrier(&[Barrier::subresource(
        destination,
        Subresource::mip(1),
        Access::ComputeShaderWrite,
        Access::CopyRead,
    )]);
    list.copy_texture_to_buffer(
        TextureCopy {
            texture: destination,
            mip: 1,
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
    run(&backend, QueueKind::Graphics, &list);

    assert_eq!(
        read_texels(&**resources, readback, size),
        coordinate_texels(4)
    );

    resources.destroy_program(fill);
    resources.destroy_program(copy);
    resources.destroy_sampler(sampler);
    resources.destroy_buffer(readback);
    resources.destroy_texture(source);
    resources.destroy_texture(destination);
}

/// Each mip of an 8×8 texture is written by compute from the mip above,
/// sampled through a view of that mip, into a storage view of its own,
/// with a barrier on each mip between its write and its read. The
/// smallest is then sampled through a view of only that mip, and mip 1
/// and the sample are read back (R21's `mip_chain_by_compute`).
#[test]
fn mip_chain_by_compute() {
    const MIPS: u32 = 4;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = |label, size, mip_levels, usage| {
        resources
            .create_texture(&TextureDesc {
                label,
                dimension: TextureDimension::D2,
                format: Format::Rgba32Float,
                extent: Extent3d::new_2d(size, size),
                mip_levels,
                array_layers: 1,
                usage,
            })
            .unwrap()
    };
    let chain = texture(
        "chain",
        8,
        MIPS,
        TextureUsage::SAMPLED | TextureUsage::STORAGE | TextureUsage::COPY_SOURCE,
    );
    let smallest = texture(
        "smallest",
        1,
        1,
        TextureUsage::STORAGE | TextureUsage::COPY_SOURCE,
    );
    let sampler = resources
        .create_sampler(&SamplerDesc {
            label: "nearest",
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mip_filter: Filter::Nearest,
            address_modes: [AddressMode::ClampToEdge; 3],
        })
        .unwrap();
    let fill = resources
        .create_program(&test_shaders::fill_texels().desc())
        .unwrap();
    let downsample = resources
        .create_program(&test_shaders::downsample().desc())
        .unwrap();
    let copy = resources
        .create_program(&test_shaders::copy_texels().desc())
        .unwrap();
    let index = |texture, view| resources.texture_index(texture, view).unwrap();
    let sampled = |mip| index(chain, TextureView::Sampled(Subresource::mip(mip)));
    let storage = |mip| index(chain, TextureView::Storage(Subresource::mip(mip)));

    let mut list = CommandList::new("mip chain");
    list.barrier(&[
        Barrier::texture(chain, Access::None, Access::ComputeShaderWrite),
        Barrier::texture(smallest, Access::None, Access::ComputeShaderWrite),
    ]);
    list.bind_program(fill);
    list.push_data(0, &index_push_data(&[storage(0)]));
    list.dispatch([1, 1, 1]);
    list.bind_program(downsample);
    for mip in 1..MIPS {
        list.barrier(&[Barrier::subresource(
            chain,
            Subresource::mip(mip - 1),
            Access::ComputeShaderWrite,
            Access::ComputeShaderSampledRead,
        )]);
        list.push_data(0, &index_push_data(&[sampled(mip - 1), storage(mip)]));
        list.dispatch([1, 1, 1]);
    }
    list.barrier(&[Barrier::subresource(
        chain,
        Subresource::mip(MIPS - 1),
        Access::ComputeShaderWrite,
        Access::ComputeShaderSampledRead,
    )]);
    list.bind_program(copy);
    list.push_data(
        0,
        &index_push_data(&[
            sampled(MIPS - 1),
            resources.sampler_index(sampler),
            index(smallest, TextureView::Storage(Subresource::mip(0))),
        ]),
    );
    list.dispatch([1, 1, 1]);
    list.barrier(&[
        Barrier::subresource(
            chain,
            Subresource::mip(1),
            Access::ComputeShaderSampledRead,
            Access::CopyRead,
        ),
        Barrier::texture(smallest, Access::ComputeShaderWrite, Access::CopyRead),
    ]);
    let mut read_back = Vec::new();
    for (texture, mip, size) in [(chain, 1, 4), (smallest, 0, 1)] {
        let extent = Extent3d::new_2d(size, size);
        let layout = TextureDataLayout::tightly_packed(Format::Rgba32Float, extent);
        let bytes = u64::from(layout.bytes_per_row * size);
        let readback = create(&**resources, "readback", bytes, MemoryLocation::Readback);
        list.copy_texture_to_buffer(
            TextureCopy {
                texture,
                mip,
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
        read_back.push((readback, bytes));
    }
    run(&backend, QueueKind::Graphics, &list);

    // Mip 0's texel (x, y) is (x, y, x + y, 1), so each of mip 1's is its
    // 2×2 block's average, and the smallest is the average of all of mip 0.
    let mip_1: Vec<f32> = (0..4)
        .flat_map(|y| {
            (0..4).flat_map(move |x| {
                let (x, y) = (2.0 * x as f32 + 0.5, 2.0 * y as f32 + 0.5);
                [x, y, x + y, 1.0]
            })
        })
        .collect();
    let read = |(buffer, size)| read_texels(&**resources, buffer, size);
    assert_eq!(read(read_back[0]), mip_1);
    assert_eq!(read(read_back[1]), [3.5, 3.5, 7.0, 1.0]);

    for (readback, _) in read_back {
        resources.destroy_buffer(readback);
    }
    for program in [fill, downsample, copy] {
        resources.destroy_program(program);
    }
    resources.destroy_sampler(sampler);
    resources.destroy_texture(chain);
    resources.destroy_texture(smallest);
}

/// The `size` bytes of `buffer`, a readback buffer, as `f32`s, such as the
/// channels of `Rgba32Float` texels copied into it.
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

/// Dispatching with only a vertex and fragment program bound panics when
/// translated, naming the list.
#[test]
fn dispatches_need_a_compute_program() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let mut list = CommandList::new("draw");
    list.bind_program(program);
    list.dispatch([1, 1, 1]);
    assert_eq!(
        translation_panic(&backend, QueueKind::Graphics, &list),
        "command list \"draw\" dispatches with no compute program bound",
    );
    resources.destroy_program(program);
}

/// Binding a vertex and fragment program binds only its own stages, so a
/// compute program bound before it still runs for a later dispatch, which
/// writes its value.
#[test]
fn compute_programs_stay_bound_across_graphics_binds() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let compute = resources
        .create_program(&test_shaders::write_value().desc())
        .unwrap();
    let graphics = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let output = resources
        .create_buffer(&BufferDesc {
            label: "output",
            size: 4,
            usage: BufferUsage::STORAGE,
            location: MemoryLocation::Readback,
        })
        .unwrap();
    let mut list = CommandList::new("both bound");
    list.bind_program(compute);
    list.bind_program(graphics);
    list.push_data(0, &resources.buffer_address(output).0.to_le_bytes());
    list.dispatch([1, 1, 1]);
    list.barrier(&[Barrier::buffer(
        output,
        Access::ComputeShaderWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);
    let mut value = [0; 4];
    resources.read_buffer(output, 0, &mut value).unwrap();
    assert_eq!(u32::from_le_bytes(value), 0xC10C_C0DE);
    resources.destroy_buffer(output);
    resources.destroy_program(compute);
    resources.destroy_program(graphics);
}

/// A transfer queue of its own runs no programs, so binding one, or
/// pushing data, panics when translated for it.
#[test]
fn the_transfer_queue_runs_no_programs() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    require_capabilities!(backend, Capability::DedicatedTransferQueue);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::write_value().desc())
        .unwrap();
    let mut list = CommandList::new("upload");
    list.bind_program(program);
    assert_eq!(
        translation_panic(&backend, QueueKind::Transfer, &list),
        "command list \"upload\" binds program \"write value\", which the transfer queue can't \
         run",
    );
    let mut list = CommandList::new("upload");
    list.push_data(0, &[0; 8]);
    assert_eq!(
        translation_panic(&backend, QueueKind::Transfer, &list),
        "command list \"upload\" pushes data, but the transfer queue runs no programs",
    );
    resources.destroy_program(program);
}

/// A compute queue of its own may be on a family with no graphics, so
/// binding a vertex and fragment program panics when translated for it.
#[test]
fn the_compute_queue_runs_only_compute_programs() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    require_capabilities!(backend, Capability::DedicatedComputeQueue);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let mut list = CommandList::new("draw");
    list.bind_program(program);
    assert_eq!(
        translation_panic(&backend, QueueKind::Compute, &list),
        "command list \"draw\" binds program \"triangle\", which the compute queue can't run",
    );
    resources.destroy_program(program);
}

/// Render state, viewports and scissors are set on the graphics queue,
/// with the defaults and with every piece changed, and validation reports
/// nothing.
#[test]
fn render_state_is_set_on_the_graphics_queue() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let blended = ColourWrite {
        blend: Some(Blend::PREMULTIPLIED_ALPHA),
        mask: ColourMask {
            alpha: false,
            ..ColourMask::ALL
        },
    };
    let changed = RenderState::default()
        .topology(Topology::TriangleStrip)
        .cull(CullMode::Front)
        .front_face(FrontFace::Clockwise)
        .depth(Some(DepthTest {
            compare: CompareOp::GreaterOrEqual,
            write: false,
        }))
        .colour(0, blended)
        .colour(1, ColourWrite::default());
    let mut list = CommandList::new("draw");
    list.bind_program(program);
    list.set_render_state(&RenderState::default());
    let extent = Extent2d {
        width: 64,
        height: 32,
    };
    list.set_viewport(&Viewport::from_extent(extent));
    list.set_scissor(Rect2d::from_extent(extent));
    list.set_viewport(&Viewport {
        x: -8.5,
        y: 4.0,
        width: 32.0,
        height: 16.0,
        min_depth: 0.75,
        max_depth: 0.25,
    });
    list.set_scissor(Rect2d {
        offset: Offset2d { x: 8, y: 4 },
        extent: Extent2d {
            width: 16,
            height: 0,
        },
    });
    list.set_render_state(&changed);
    list.set_render_state(&changed.depth(None));
    list.set_render_state(&changed.topology(Topology::LineList));
    run(&backend, QueueKind::Graphics, &list);
    resources.destroy_program(program);
}

/// Only the graphics queue draws, so setting render state, a viewport, a
/// scissor or an index buffer, or beginning a render pass, panics when
/// translated for a compute queue of its own.
#[test]
fn only_the_graphics_queue_sets_render_state() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    require_capabilities!(backend, Capability::DedicatedComputeQueue);
    // Checks that a list recorded by `record` panics, saying it `does`.
    let check = |does: &str, record: &dyn Fn(&mut CommandList)| {
        let mut list = CommandList::new("draw");
        record(&mut list);
        assert_eq!(
            translation_panic(&backend, QueueKind::Compute, &list),
            format!(
                "command list \"draw\" {does}, but only the graphics queue draws, not the \
                 compute queue"
            ),
        );
    };
    let extent = Extent2d {
        width: 64,
        height: 32,
    };
    check("sets render state", &|list| {
        list.set_render_state(&RenderState::default());
    });
    check("sets a viewport", &|list| {
        list.set_viewport(&Viewport::from_extent(extent));
    });
    check("sets a scissor", &|list| {
        list.set_scissor(Rect2d::from_extent(extent));
    });
    check("binds an index buffer", &|list| {
        list.bind_index_buffer(GpuAddress(0x1000), 12, IndexFormat::Uint16);
    });
    let resources = backend.resources();
    let texture = create_texture(
        &**resources,
        "colour",
        Format::Rgba8Unorm,
        TextureUsage::COLOUR_ATTACHMENT,
    );
    check("begins a render pass", &|list| {
        list.begin_render_pass(&RenderPassDesc::colour(texture, LoadOp::DontCare));
        list.end_render_pass();
    });
    resources.destroy_texture(texture);
}

/// A render pass clears each colour attachment to its own clear value, a
/// float format's to its floats and an integer format's to its integers,
/// and its depth attachment to its depth, in the mip and layer each names,
/// and stores them, so that copying that subresource out reads the clear
/// value in every texel.
#[test]
fn render_passes_clear_their_attachments() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let usage = TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE;
    let float = create_texture(&**resources, "float", Format::Rgba32Float, usage);
    let integer = create_texture(&**resources, "integer", Format::R32Uint, usage);
    let depth = create_texture(
        &**resources,
        "depth",
        Format::Depth32Float,
        TextureUsage::DEPTH_ATTACHMENT | TextureUsage::COPY_SOURCE,
    );
    // Mip 1 of the 4×4 textures, in their second layer.
    let mip = Extent3d::new_2d(2, 2);
    let subresource = Subresource::mip_layer(1, 1);
    let float_size = 2 * 2 * 16;
    let integer_size = 2 * 2 * 4;
    let depth_readback = create(
        &**resources,
        "depth texels",
        integer_size,
        MemoryLocation::Readback,
    );
    let float_readback = create(
        &**resources,
        "float texels",
        float_size,
        MemoryLocation::Readback,
    );
    let integer_readback = create(
        &**resources,
        "integer texels",
        integer_size,
        MemoryLocation::Readback,
    );
    let colour = [0.25, 0.5, 0.75, 1.0];
    let attachment = |texture, load| ColorAttachment {
        texture,
        mip: 1,
        layers: 1..2,
        load,
        store: StoreOp::Store,
    };
    let render_pass = RenderPassDesc {
        colour: vec![
            attachment(float, LoadOp::Clear(colour)),
            attachment(integer, LoadOp::Clear([7.0, 0.0, 0.0, 0.0])),
        ],
        depth: Some(DepthAttachment {
            texture: depth,
            mip: 1,
            layers: 1..2,
            load: LoadOp::Clear(0.5),
            store: StoreOp::Store,
        }),
    };
    let depth_attachment = Access::DepthAttachmentRead | Access::DepthAttachmentWrite;
    let region = |texture| TextureCopy {
        texture,
        mip: 1,
        layers: 1..2,
        origin: [0; 3],
        extent: mip,
    };
    let barriers = |before, after| {
        [float, integer]
            .map(|texture| Barrier::subresource(texture, subresource.clone(), before, after))
    };

    let mut list = CommandList::new("clear");
    let depth_barrier =
        |before, after| Barrier::subresource(depth, subresource.clone(), before, after);
    list.barrier(&barriers(Access::None, Access::ColorAttachmentWrite));
    list.barrier(&[depth_barrier(Access::None.into(), depth_attachment)]);
    list.begin_render_pass(&render_pass);
    list.end_render_pass();
    list.barrier(&barriers(Access::ColorAttachmentWrite, Access::CopyRead));
    list.barrier(&[depth_barrier(depth_attachment, Access::CopyRead.into())]);
    list.copy_texture_to_buffer(
        region(depth),
        resources.buffer_address(depth_readback),
        TextureDataLayout::tightly_packed(Format::Depth32Float, mip),
    );
    list.copy_texture_to_buffer(
        region(float),
        resources.buffer_address(float_readback),
        TextureDataLayout::tightly_packed(Format::Rgba32Float, mip),
    );
    list.copy_texture_to_buffer(
        region(integer),
        resources.buffer_address(integer_readback),
        TextureDataLayout::tightly_packed(Format::R32Uint, mip),
    );
    list.barrier(&[
        Barrier::buffer(float_readback, Access::CopyWrite, Access::HostRead),
        Barrier::buffer(integer_readback, Access::CopyWrite, Access::HostRead),
        Barrier::buffer(depth_readback, Access::CopyWrite, Access::HostRead),
    ]);
    run(&backend, QueueKind::Graphics, &list);

    assert_eq!(
        read_texels(&**resources, float_readback, float_size),
        colour.repeat(4),
    );
    let mut bytes = vec![0; integer_size as usize];
    resources
        .read_buffer(integer_readback, 0, &mut bytes)
        .unwrap();
    assert_eq!(bytes, 7_u32.to_le_bytes().repeat(4));
    assert_eq!(
        read_texels(&**resources, depth_readback, integer_size),
        [0.5; 4]
    );
    for buffer in [float_readback, integer_readback, depth_readback] {
        resources.destroy_buffer(buffer);
    }
    for texture in [float, integer, depth] {
        resources.destroy_texture(texture);
    }
}

/// The triangle program draws into an 8×8 colour attachment cleared to
/// transparent black, twice, as two instances, which write the same
/// texels. The texel at the centre is its corners' colours, red, green and
/// blue, weighted by where the texel's centre is in the triangle, and the
/// corner texel, outside the triangle, keeps the clear value. A compute
/// program is bound after the triangle's, which leaves the triangle's
/// bound for the draw.
#[test]
fn draws_render_to_the_render_pass() {
    const SIZE: u32 = 8;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let compute = resources
        .create_program(&test_shaders::write_value().desc())
        .unwrap();
    let texture = resources
        .create_texture(&TextureDesc {
            label: "triangle",
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
    let readback = create(&**resources, "readback", size, MemoryLocation::Readback);
    let attachment = Extent2d {
        width: SIZE,
        height: SIZE,
    };

    let mut list = CommandList::new("triangle");
    list.barrier(&[Barrier::texture(
        texture,
        Access::None,
        Access::ColorAttachmentWrite,
    )]);
    list.bind_program(program);
    list.bind_program(compute);
    list.set_render_state(&RenderState::default().cull(CullMode::None));
    list.set_viewport(&Viewport::from_extent(attachment));
    list.set_scissor(Rect2d::from_extent(attachment));
    list.begin_render_pass(&RenderPassDesc::colour(texture, LoadOp::Clear([0.0; 4])));
    list.draw(3, 2);
    list.end_render_pass();
    list.barrier(&[Barrier::texture(
        texture,
        Access::ColorAttachmentWrite,
        Access::CopyRead,
    )]);
    list.copy_texture_to_buffer(
        whole(texture, extent),
        resources.buffer_address(readback),
        layout,
    );
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);

    let texels = read_texels(&**resources, readback, size);
    assert_triangle_centre(texel(&texels, SIZE, 4, 4));
    assert_eq!(texel(&texels, SIZE, 0, 0), [0.0; 4]);
    resources.destroy_buffer(readback);
    resources.destroy_texture(texture);
    resources.destroy_program(program);
    resources.destroy_program(compute);
}

/// The triangle program draws into mip 1 of layer 2 of a 16×16 array
/// texture with two mips and three layers, and nothing else in the texture
/// changes (R21's `render_to_mip_and_layer`).
///
/// Every layer of each mip is first cleared to a marker colour by a render
/// pass to all three layers, rather than copied into, since NVIDIA's 617.42
/// doesn't order a copy out of an image after a copy into it (design 0002,
/// "Risks"). The draw's render pass then clears mip 1 of layer 2, 8×8, to
/// transparent black, and draws the triangle. Each mip of each layer is
/// copied out: that subresource has the triangle at its centre and the
/// clear value in its corner, and every other is still the marker colour.
#[test]
fn render_to_mip_and_layer() {
    const SIZE: u32 = 16;
    const MIPS: u32 = 2;
    const LAYERS: u32 = 3;
    // The mip and layer drawn into. The mip is 8×8, which
    // `assert_triangle_centre` expects.
    const MIP: u32 = 1;
    const LAYER: u32 = 2;
    const MARKER: [f32; 4] = [0.5, 0.25, 0.125, 1.0];
    /// One mip of one layer, copied out to check.
    struct Copied {
        /// The mip.
        mip: u32,
        /// The layer.
        layer: u32,
        /// The mip's width and height.
        size: u32,
        /// Where it's copied to in the readback buffer, in bytes.
        offset: u64,
    }
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let texture = resources
        .create_texture(&TextureDesc {
            label: "array",
            dimension: TextureDimension::D2,
            format: Format::Rgba32Float,
            extent: Extent3d::new_2d(SIZE, SIZE),
            mip_levels: MIPS,
            array_layers: LAYERS,
            usage: TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE,
        })
        .unwrap();
    // Every mip of a layer, before the next layer's.
    let mut subresources = Vec::new();
    let mut offset = 0;
    for layer in 0..LAYERS {
        for mip in 0..MIPS {
            let size = SIZE >> mip;
            subresources.push(Copied {
                mip,
                layer,
                size,
                offset,
            });
            offset += u64::from(size * size * 16);
        }
    }
    let readback = create(&**resources, "readback", offset, MemoryLocation::Readback);
    let attachment = |mip, layers, load| RenderPassDesc {
        colour: vec![ColorAttachment {
            texture,
            mip,
            layers,
            load,
            store: StoreOp::Store,
        }],
        depth: None,
    };
    let drawn = Extent2d {
        width: SIZE >> MIP,
        height: SIZE >> MIP,
    };

    let mut list = CommandList::new("mip and layer");
    list.barrier(&[Barrier::texture(
        texture,
        Access::None,
        Access::ColorAttachmentWrite,
    )]);
    for mip in 0..MIPS {
        list.begin_render_pass(&attachment(mip, 0..LAYERS, LoadOp::Clear(MARKER)));
        list.end_render_pass();
    }
    list.barrier(&[Barrier::subresource(
        texture,
        Subresource::mip_layer(MIP, LAYER),
        Access::ColorAttachmentWrite,
        Access::ColorAttachmentWrite,
    )]);
    list.bind_program(program);
    list.set_render_state(&RenderState::default().cull(CullMode::None));
    list.set_viewport(&Viewport::from_extent(drawn));
    list.set_scissor(Rect2d::from_extent(drawn));
    list.begin_render_pass(&attachment(MIP, LAYER..LAYER + 1, LoadOp::Clear([0.0; 4])));
    list.draw(3, 1);
    list.end_render_pass();
    list.barrier(&[Barrier::texture(
        texture,
        Access::ColorAttachmentWrite,
        Access::CopyRead,
    )]);
    for &Copied {
        mip,
        layer,
        size,
        offset,
    } in &subresources
    {
        let extent = Extent3d::new_2d(size, size);
        list.copy_texture_to_buffer(
            TextureCopy {
                texture,
                mip,
                layers: layer..layer + 1,
                origin: [0; 3],
                extent,
            },
            GpuAddress(resources.buffer_address(readback).0 + offset),
            TextureDataLayout::tightly_packed(Format::Rgba32Float, extent),
        );
    }
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);

    let texels = read_texels(&**resources, readback, offset);
    for Copied {
        mip,
        layer,
        size,
        offset,
    } in subresources
    {
        let start = (offset / 4) as usize;
        let texels = &texels[start..start + (size * size * 4) as usize];
        if (mip, layer) == (MIP, LAYER) {
            assert_triangle_centre(texel(texels, size, 4, 4));
            assert_eq!(texel(texels, size, 0, 0), [0.0; 4]);
        } else {
            assert!(
                texels.chunks(4).all(|texel| texel == MARKER),
                "mip {mip} of layer {layer} isn't all {MARKER:?}",
            );
        }
    }
    resources.destroy_buffer(readback);
    resources.destroy_texture(texture);
    resources.destroy_program(program);
}

/// Texel (`x`, `y`) of `Rgba32Float` texels in rows `width` texels wide,
/// read back tightly packed.
fn texel(texels: &[f32], width: u32, x: u32, y: u32) -> [f32; 4] {
    let at = ((y * width + x) * 4) as usize;
    [texels[at], texels[at + 1], texels[at + 2], texels[at + 3]]
}

/// Checks that `centre`, the centre texel of an 8×8 attachment the
/// triangle program drew into, is its corners' colours, red, green and
/// blue, weighted by where the texel's centre is in the triangle.
///
/// # Panics
///
/// Panics, naming both colours, if any channel differs by more than
/// rounding.
fn assert_triangle_centre(centre: [f32; 4]) {
    // The centre of texel (4, 4) is at (0.125, 0.125) in clip space, which
    // is 0.375 of the way to the top corner, 0.4375 to the bottom right and
    // 0.1875 to the bottom left.
    let expected = [0.375, 0.4375, 0.1875, 1.0];
    assert!(
        centre
            .iter()
            .zip(expected)
            .all(|(texel, expected)| (texel - expected).abs() < 1e-3),
        "the centre texel is {centre:?}, not {expected:?}",
    );
}

/// The triangle program draws indexed into three layers of an 8×8 colour
/// attachment, each cleared to transparent black, with the vertex index
/// each corner's colour comes from read from the index buffer bound:
///
/// - layer 0: 16-bit indices, bound from partway into the buffer, after
///   indices that would draw nothing
/// - layer 1: 32-bit indices, in another order
/// - layer 2: 16-bit indices that all name one vertex, so nothing is drawn
///
/// The first two layers' centres are the triangle's colours, and the
/// third's is the clear value.
#[test]
fn indexed_draws_read_their_index_buffer() {
    const SIZE: u32 = 8;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let texture = resources
        .create_texture(&TextureDesc {
            label: "triangles",
            dimension: TextureDimension::D2,
            format: Format::Rgba32Float,
            extent: Extent3d::new_2d(SIZE, SIZE),
            mip_levels: 1,
            array_layers: 3,
            usage: TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE,
        })
        .unwrap();
    // Six 16-bit indices, the first three naming one vertex, then three
    // 32-bit indices, which start 4-byte aligned.
    let indices: Vec<u8> = [0_u16, 0, 0, 2, 1, 0]
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .chain([1_u32, 2, 0].iter().flat_map(|index| index.to_le_bytes()))
        .collect();
    let index_buffer = resources
        .create_buffer(&BufferDesc {
            label: "indices",
            size: indices.len() as u64,
            usage: BufferUsage::INDEX,
            location: MemoryLocation::Upload,
        })
        .unwrap();
    resources.write_buffer(index_buffer, 0, &indices);
    let base = resources.buffer_address(index_buffer).0;
    let extent = Extent3d::new_2d(SIZE, SIZE);
    let layout = TextureDataLayout::tightly_packed(Format::Rgba32Float, extent);
    let layer_size = u64::from(layout.bytes_per_row * SIZE);
    let readback = create(
        &**resources,
        "readback",
        3 * layer_size,
        MemoryLocation::Readback,
    );
    let attachment = Extent2d {
        width: SIZE,
        height: SIZE,
    };

    let mut list = CommandList::new("indexed triangles");
    list.barrier(&[
        Barrier::texture(texture, Access::None, Access::ColorAttachmentWrite),
        Barrier::buffer(index_buffer, Access::HostWrite, Access::IndexRead),
    ]);
    list.bind_program(program);
    list.set_render_state(&RenderState::default().cull(CullMode::None));
    list.set_viewport(&Viewport::from_extent(attachment));
    list.set_scissor(Rect2d::from_extent(attachment));
    for (layer, address, size, format) in [
        (0, base + 6, 6, IndexFormat::Uint16),
        (1, base + 12, 12, IndexFormat::Uint32),
        (2, base, 6, IndexFormat::Uint16),
    ] {
        let mut render_pass = RenderPassDesc::colour(texture, LoadOp::Clear([0.0; 4]));
        render_pass.colour[0].layers = layer..layer + 1;
        list.bind_index_buffer(GpuAddress(address), size, format);
        list.begin_render_pass(&render_pass);
        list.draw_indexed(3, 1);
        list.end_render_pass();
    }
    list.barrier(&[Barrier::texture(
        texture,
        Access::ColorAttachmentWrite,
        Access::CopyRead,
    )]);
    for layer in 0..3 {
        list.copy_texture_to_buffer(
            TextureCopy {
                layers: layer..layer + 1,
                ..whole(texture, extent)
            },
            GpuAddress(resources.buffer_address(readback).0 + u64::from(layer) * layer_size),
            layout,
        );
    }
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);

    let texels = read_texels(&**resources, readback, 3 * layer_size);
    let layers: Vec<&[f32]> = texels.chunks(texels.len() / 3).collect();
    assert_triangle_centre(texel(layers[0], SIZE, 4, 4));
    assert_triangle_centre(texel(layers[1], SIZE, 4, 4));
    assert_eq!(texel(layers[2], SIZE, 4, 4), [0.0; 4]);
    resources.destroy_buffer(readback);
    resources.destroy_buffer(index_buffer);
    resources.destroy_texture(texture);
    resources.destroy_program(program);
}

/// The triangle program draws indirectly into two layers of an 8×8 colour
/// attachment, each cleared to transparent black, from the same arguments,
/// 4 bytes into a buffer: a draw of no vertices, 16 bytes of zeros, then a
/// draw of the triangle, 32 bytes after the first, from a first vertex of
/// 3 and a first instance of 5.
///
/// - layer 0: two draws 32 bytes apart, so the triangle is drawn only if
///   the second draw's arguments are read a stride after the first's, and
///   `SV_VertexID` counts from 0, without the first vertex, which would
///   name vertices past the triangle's three
/// - layer 1: one draw, so only the first draw's arguments are read, and
///   nothing is drawn
#[test]
fn indirect_draws_read_their_arguments() {
    const SIZE: u32 = 8;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let texture = resources
        .create_texture(&TextureDesc {
            label: "triangles",
            dimension: TextureDimension::D2,
            format: Format::Rgba32Float,
            extent: Extent3d::new_2d(SIZE, SIZE),
            mip_levels: 1,
            array_layers: 2,
            usage: TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE,
        })
        .unwrap();
    // Each draw's vertex count, instance count, first vertex and first
    // instance, after 4 bytes that aren't read.
    let arguments: Vec<u8> = [0_u32, 0, 1, 0, 0, 0, 0, 0, 0, 3, 1, 3, 5]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let arguments_buffer = resources
        .create_buffer(&BufferDesc {
            label: "arguments",
            size: arguments.len() as u64,
            usage: BufferUsage::INDIRECT,
            location: MemoryLocation::Upload,
        })
        .unwrap();
    resources.write_buffer(arguments_buffer, 0, &arguments);
    let first = GpuAddress(resources.buffer_address(arguments_buffer).0 + 4);
    let extent = Extent3d::new_2d(SIZE, SIZE);
    let layout = TextureDataLayout::tightly_packed(Format::Rgba32Float, extent);
    let layer_size = u64::from(layout.bytes_per_row * SIZE);
    let readback = create(
        &**resources,
        "readback",
        2 * layer_size,
        MemoryLocation::Readback,
    );
    let attachment = Extent2d {
        width: SIZE,
        height: SIZE,
    };

    let mut list = CommandList::new("indirect triangles");
    list.barrier(&[
        Barrier::texture(texture, Access::None, Access::ColorAttachmentWrite),
        Barrier::buffer(arguments_buffer, Access::HostWrite, Access::IndirectRead),
    ]);
    list.bind_program(program);
    list.set_render_state(&RenderState::default().cull(CullMode::None));
    list.set_viewport(&Viewport::from_extent(attachment));
    list.set_scissor(Rect2d::from_extent(attachment));
    for (layer, draws) in [(0, 2), (1, 1)] {
        let mut render_pass = RenderPassDesc::colour(texture, LoadOp::Clear([0.0; 4]));
        render_pass.colour[0].layers = layer..layer + 1;
        list.begin_render_pass(&render_pass);
        list.draw_indirect(first, draws, 32);
        list.end_render_pass();
    }
    list.barrier(&[Barrier::texture(
        texture,
        Access::ColorAttachmentWrite,
        Access::CopyRead,
    )]);
    for layer in 0..2 {
        list.copy_texture_to_buffer(
            TextureCopy {
                layers: layer..layer + 1,
                ..whole(texture, extent)
            },
            GpuAddress(resources.buffer_address(readback).0 + u64::from(layer) * layer_size),
            layout,
        );
    }
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);

    let texels = read_texels(&**resources, readback, 2 * layer_size);
    let layers: Vec<&[f32]> = texels.chunks(texels.len() / 2).collect();
    assert_triangle_centre(texel(layers[0], SIZE, 4, 4));
    assert_eq!(texel(layers[1], SIZE, 4, 4), [0.0; 4]);
    resources.destroy_buffer(readback);
    resources.destroy_buffer(arguments_buffer);
    resources.destroy_texture(texture);
    resources.destroy_program(program);
}

/// The triangle program draws indexed and indirectly into two layers of an
/// 8×8 colour attachment, each cleared to transparent black, from 16-bit
/// indices that name one vertex three times, then each vertex, and from
/// the same arguments, 4 bytes into a buffer: a draw of no indices, 4
/// bytes of zeros, then a draw of three indices from the fourth, 24 bytes
/// after the first, with a vertex offset of 3 and a first instance of 5.
///
/// - layer 0: two draws 24 bytes apart, so the triangle is drawn only if
///   the second draw's arguments are read a stride after the first's, its
///   first index is used, and `SV_VertexID` is the index read, without the
///   vertex offset, which would name vertices past the triangle's three
/// - layer 1: one draw, so only the first draw's arguments are read, and
///   nothing is drawn
#[test]
fn indexed_indirect_draws_read_their_arguments() {
    const SIZE: u32 = 8;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::triangle().desc())
        .unwrap();
    let texture = resources
        .create_texture(&TextureDesc {
            label: "triangles",
            dimension: TextureDimension::D2,
            format: Format::Rgba32Float,
            extent: Extent3d::new_2d(SIZE, SIZE),
            mip_levels: 1,
            array_layers: 2,
            usage: TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE,
        })
        .unwrap();
    let indices: Vec<u8> = [0_u16, 0, 0, 2, 1, 0]
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect();
    // Each draw's index count, instance count, first index, vertex offset
    // and first instance, after 4 bytes that aren't read.
    let arguments: Vec<u8> = [0_u32, 0, 1, 0, 0, 0, 0, 3, 1, 3, 3, 5]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let index_buffer = resources
        .create_buffer(&BufferDesc {
            label: "indices",
            size: indices.len() as u64,
            usage: BufferUsage::INDEX,
            location: MemoryLocation::Upload,
        })
        .unwrap();
    let arguments_buffer = resources
        .create_buffer(&BufferDesc {
            label: "arguments",
            size: arguments.len() as u64,
            usage: BufferUsage::INDIRECT,
            location: MemoryLocation::Upload,
        })
        .unwrap();
    resources.write_buffer(index_buffer, 0, &indices);
    resources.write_buffer(arguments_buffer, 0, &arguments);
    let first = GpuAddress(resources.buffer_address(arguments_buffer).0 + 4);
    let extent = Extent3d::new_2d(SIZE, SIZE);
    let layout = TextureDataLayout::tightly_packed(Format::Rgba32Float, extent);
    let layer_size = u64::from(layout.bytes_per_row * SIZE);
    let readback = create(
        &**resources,
        "readback",
        2 * layer_size,
        MemoryLocation::Readback,
    );
    let attachment = Extent2d {
        width: SIZE,
        height: SIZE,
    };

    let mut list = CommandList::new("indexed indirect triangles");
    list.barrier(&[
        Barrier::texture(texture, Access::None, Access::ColorAttachmentWrite),
        Barrier::buffer(index_buffer, Access::HostWrite, Access::IndexRead),
        Barrier::buffer(arguments_buffer, Access::HostWrite, Access::IndirectRead),
    ]);
    list.bind_program(program);
    list.set_render_state(&RenderState::default().cull(CullMode::None));
    list.set_viewport(&Viewport::from_extent(attachment));
    list.set_scissor(Rect2d::from_extent(attachment));
    list.bind_index_buffer(
        resources.buffer_address(index_buffer),
        indices.len() as u64,
        IndexFormat::Uint16,
    );
    for (layer, draws) in [(0, 2), (1, 1)] {
        let mut render_pass = RenderPassDesc::colour(texture, LoadOp::Clear([0.0; 4]));
        render_pass.colour[0].layers = layer..layer + 1;
        list.begin_render_pass(&render_pass);
        list.draw_indexed_indirect(first, draws, 24);
        list.end_render_pass();
    }
    list.barrier(&[Barrier::texture(
        texture,
        Access::ColorAttachmentWrite,
        Access::CopyRead,
    )]);
    for layer in 0..2 {
        list.copy_texture_to_buffer(
            TextureCopy {
                layers: layer..layer + 1,
                ..whole(texture, extent)
            },
            GpuAddress(resources.buffer_address(readback).0 + u64::from(layer) * layer_size),
            layout,
        );
    }
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);

    let texels = read_texels(&**resources, readback, 2 * layer_size);
    let layers: Vec<&[f32]> = texels.chunks(texels.len() / 2).collect();
    assert_triangle_centre(texel(layers[0], SIZE, 4, 4));
    assert_eq!(texel(layers[1], SIZE, 4, 4), [0.0; 4]);
    resources.destroy_buffer(readback);
    resources.destroy_buffer(arguments_buffer);
    resources.destroy_buffer(index_buffer);
    resources.destroy_texture(texture);
    resources.destroy_program(program);
}

/// A draw samples a texture through its slot in the resource heap, with a
/// sampler through its slot in the sampler heap, and renders what it
/// samples (R6's `fragment_samples_heap_texture`).
///
/// The texture's texels, each `(x, y, x + y, 1)`, are copied in from a
/// buffer. The draw covers a colour attachment of the same size, sampling
/// at each fragment's lower-right corner with a linear sampler that clamps
/// to the edge, so each texel rendered is the average of the texture's at
/// the same coordinates and the three after it, clamped to the edge.
///
/// A decoy texture and a nearest sampler take the first slot in each heap,
/// so the texture's and the sampler's slots aren't 0, and the result is
/// right only if the draw uses the slots in its push data.
#[test]
fn fragment_samples_heap_texture() {
    const SIZE: u32 = 4;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = |label, usage| {
        resources
            .create_texture(&TextureDesc {
                label,
                dimension: TextureDimension::D2,
                format: Format::Rgba32Float,
                extent: Extent3d::new_2d(SIZE, SIZE),
                mip_levels: 1,
                array_layers: 1,
                usage,
            })
            .unwrap()
    };
    let sampler = |label, filter| {
        resources
            .create_sampler(&SamplerDesc {
                label,
                mag_filter: filter,
                min_filter: filter,
                mip_filter: Filter::Nearest,
                address_modes: [AddressMode::ClampToEdge; 3],
            })
            .unwrap()
    };
    let decoy_texture = texture("decoy", TextureUsage::SAMPLED);
    let decoy_index = resources
        .texture_index(decoy_texture, TextureView::sampled())
        .unwrap();
    let decoy_sampler = sampler("nearest", Filter::Nearest);
    let source = texture(
        "source",
        TextureUsage::SAMPLED | TextureUsage::COPY_DESTINATION,
    );
    let source_index = resources
        .texture_index(source, TextureView::sampled())
        .unwrap();
    let linear = sampler("linear", Filter::Linear);
    assert_ne!(source_index, decoy_index);
    assert_ne!(
        resources.sampler_index(linear),
        resources.sampler_index(decoy_sampler),
    );
    let target = texture(
        "target",
        TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE,
    );
    let program = resources
        .create_program(&test_shaders::sample_texture().desc())
        .unwrap();
    let texels = coordinate_texels(SIZE);
    let bytes: Vec<u8> = texels
        .iter()
        .flat_map(|texel| texel.to_le_bytes())
        .collect();
    // Each texel's average with the three after it in x and y, clamped to
    // the edge, as a linear sampler blends them where they meet.
    let texel = |x: u32, y: u32| {
        let at = ((y.min(SIZE - 1) * SIZE + x.min(SIZE - 1)) * 4) as usize;
        &texels[at..at + 4]
    };
    let expected: Vec<f32> = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            let corners = [
                texel(x, y),
                texel(x + 1, y),
                texel(x, y + 1),
                texel(x + 1, y + 1),
            ];
            (0..4)
                .map(move |channel| corners.iter().map(|corner| corner[channel]).sum::<f32>() / 4.0)
        })
        .collect();
    let extent = Extent3d::new_2d(SIZE, SIZE);
    let layout = TextureDataLayout::tightly_packed(Format::Rgba32Float, extent);
    let size = u64::from(layout.bytes_per_row * SIZE);
    assert_eq!(bytes.len() as u64, size);
    let upload = create(&**resources, "texels", size, MemoryLocation::Upload);
    resources.write_buffer(upload, 0, &bytes);
    let readback = create(&**resources, "readback", size, MemoryLocation::Readback);
    let attachment = Extent2d {
        width: SIZE,
        height: SIZE,
    };

    let mut list = CommandList::new("sample texture");
    list.barrier(&[
        Barrier::texture(source, Access::None, Access::CopyWrite),
        Barrier::buffer(upload, Access::HostWrite, Access::CopyRead),
    ]);
    list.copy_buffer_to_texture(
        resources.buffer_address(upload),
        layout,
        whole(source, extent),
    );
    list.barrier(&[
        Barrier::texture(source, Access::CopyWrite, Access::FragmentShaderSampledRead),
        Barrier::texture(target, Access::None, Access::ColorAttachmentWrite),
    ]);
    list.bind_program(program);
    list.push_data(
        0,
        &index_push_data(&[source_index, resources.sampler_index(linear)]),
    );
    list.set_render_state(&RenderState::default().cull(CullMode::None));
    list.set_viewport(&Viewport::from_extent(attachment));
    list.set_scissor(Rect2d::from_extent(attachment));
    list.begin_render_pass(&RenderPassDesc::colour(target, LoadOp::DontCare));
    list.draw(3, 1);
    list.end_render_pass();
    list.barrier(&[Barrier::texture(
        target,
        Access::ColorAttachmentWrite,
        Access::CopyRead,
    )]);
    list.copy_texture_to_buffer(
        whole(target, extent),
        resources.buffer_address(readback),
        layout,
    );
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);

    // Filtering weights each texel to the precision the device's subtexel
    // grid allows, so the averages may differ by rounding.
    let rendered = read_texels(&**resources, readback, size);
    assert!(
        rendered
            .iter()
            .zip(&expected)
            .all(|(rendered, expected)| (rendered - expected).abs() < 1e-3),
        "rendered {rendered:?}, not {expected:?}",
    );
    resources.destroy_program(program);
    resources.destroy_sampler(linear);
    resources.destroy_sampler(decoy_sampler);
    resources.destroy_buffer(upload);
    resources.destroy_buffer(readback);
    resources.destroy_texture(source);
    resources.destroy_texture(decoy_texture);
    resources.destroy_texture(target);
}

/// The texels of a `size`×`size` `Rgba32Float` texture whose texel at (`x`,
/// `y`) is `(x, y, x + y, 1)`, tightly packed, so each texel says where it
/// is.
fn coordinate_texels(size: u32) -> Vec<f32> {
    (0..size)
        .flat_map(|y| (0..size).flat_map(move |x| [x as f32, y as f32, (x + y) as f32, 1.0]))
        .collect()
}

/// Drawing, directly or indirectly, indexed or not, with only a compute
/// program bound panics when translated, naming the list.
#[test]
fn draws_need_a_vertex_and_fragment_program() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let program = resources
        .create_program(&test_shaders::write_value().desc())
        .unwrap();
    let texture = create_texture(
        &**resources,
        "colour",
        Format::Rgba8Unorm,
        TextureUsage::COLOUR_ATTACHMENT,
    );
    let extent = Extent2d {
        width: 4,
        height: 4,
    };
    // Binding an index buffer is translated before the draw panics, so it
    // must be a real one. The draws aren't, so their arguments can be
    // anywhere.
    let index_buffer = resources
        .create_buffer(&BufferDesc {
            label: "indices",
            size: 6,
            usage: BufferUsage::INDEX,
            location: MemoryLocation::Device,
        })
        .unwrap();
    let indices = resources.buffer_address(index_buffer);
    let draws: [&dyn Fn(&mut CommandList); 3] = [
        &|list| list.draw(3, 1),
        &|list| list.draw_indirect(GpuAddress(0x1000), 1, 16),
        &|list| {
            list.bind_index_buffer(indices, 6, IndexFormat::Uint16);
            list.draw_indexed_indirect(GpuAddress(0x1000), 1, 20);
        },
    ];
    for draw in draws {
        let mut list = CommandList::new("draw");
        list.bind_program(program);
        list.set_render_state(&RenderState::default());
        list.set_viewport(&Viewport::from_extent(extent));
        list.set_scissor(Rect2d::from_extent(extent));
        list.begin_render_pass(&RenderPassDesc::colour(texture, LoadOp::DontCare));
        draw(&mut list);
        list.end_render_pass();
        assert_eq!(
            translation_panic(&backend, QueueKind::Graphics, &list),
            "command list \"draw\" draws with no vertex and fragment program bound",
        );
    }
    resources.destroy_buffer(index_buffer);
    resources.destroy_texture(texture);
    resources.destroy_program(program);
}

/// Attachments that can't be rendered to together panic when translated,
/// naming the list and the textures: here, mips of different sizes.
#[test]
fn mismatched_attachments_panic_when_translated() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let usage = TextureUsage::COLOUR_ATTACHMENT;
    let albedo = create_texture(&**resources, "albedo", Format::Rgba8Unorm, usage);
    let normals = create_texture(&**resources, "normals", Format::Rgba8Unorm, usage);
    let mut render_pass = RenderPassDesc::colour(albedo, LoadOp::DontCare);
    render_pass.colour.push(ColorAttachment {
        texture: normals,
        mip: 1,
        ..render_pass.colour[0].clone()
    });
    let mut list = CommandList::new("gbuffer");
    list.begin_render_pass(&render_pass);
    list.end_render_pass();
    assert_eq!(
        translation_panic(&backend, QueueKind::Graphics, &list),
        "command list \"gbuffer\" renders to attachments of different sizes: \"albedo\" is 4x4 \
         and \"normals\" is 2x2, at the mips they name",
    );
    resources.destroy_texture(albedo);
    resources.destroy_texture(normals);
}

/// Bytes written to an upload buffer, copied on the GPU to a GPU-only
/// buffer and then to a readback buffer, read back unchanged (R4's
/// `buffer_round_trip`), with a barrier before each access, so
/// synchronization validation reports nothing.
#[test]
fn buffer_round_trip() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let bytes: Vec<u8> = (0..=255).collect();
    let size = bytes.len() as u64;
    let upload = create(&**resources, "upload", size, MemoryLocation::Upload);
    let device = create(&**resources, "device", size, MemoryLocation::Device);
    let readback = create(&**resources, "readback", size, MemoryLocation::Readback);
    resources.write_buffer(upload, 0, &bytes);

    let address = |buffer| resources.buffer_address(buffer);
    let mut list = CommandList::new("round trip");
    list.barrier(&[Barrier::buffer(upload, Access::HostWrite, Access::CopyRead)]);
    list.copy_buffer(address(upload), address(device), size);
    list.barrier(&[Barrier::buffer(device, Access::CopyWrite, Access::CopyRead)]);
    list.copy_buffer(address(device), address(readback), size);
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);

    let mut out = vec![0; bytes.len()];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, bytes);
    for buffer in [upload, device, readback] {
        resources.destroy_buffer(buffer);
    }
}

/// A copy on the transfer queue, waited on by a copy on the graphics queue,
/// with the buffer released by one and acquired by the other (R9's
/// `cross_queue_wait`). On a device without a dedicated transfer family,
/// both run on the graphics queue, the transfer is dropped, and the test
/// still passes.
#[test]
fn cross_queue_wait() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let transfer = backend.queue(QueueKind::Transfer);
    let graphics = backend.queue(QueueKind::Graphics);
    let bytes: Vec<u8> = (0..=255).rev().collect();
    let size = bytes.len() as u64;
    let upload = create(&**resources, "upload", size, MemoryLocation::Upload);
    let device = create(&**resources, "device", size, MemoryLocation::Device);
    let readback = create(&**resources, "readback", size, MemoryLocation::Readback);
    resources.write_buffer(upload, 0, &bytes);
    let address = |buffer| resources.buffer_address(buffer);
    let handover = Barrier::buffer(device, Access::CopyWrite, Access::CopyRead)
        .with_queue_transfer(transfer.id(), graphics.id());

    let mut upload_list = CommandList::new("upload");
    upload_list.barrier(&[Barrier::buffer(upload, Access::HostWrite, Access::CopyRead)]);
    upload_list.copy_buffer(address(upload), address(device), size);
    upload_list.barrier(std::slice::from_ref(&handover));
    let uploaded = transfer
        .submit(SubmitDesc::new(vec![
            transfer.translate(&upload_list).unwrap(),
        ]))
        .unwrap();

    let mut read_list = CommandList::new("read");
    read_list.barrier(std::slice::from_ref(&handover));
    read_list.copy_buffer(address(device), address(readback), size);
    read_list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    let read = graphics
        .submit(SubmitDesc {
            command_buffers: vec![graphics.translate(&read_list).unwrap()],
            wait_for: &[uploaded],
        })
        .unwrap();
    graphics.wait(read).unwrap();

    let mut out = vec![0; bytes.len()];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, bytes);
    for buffer in [upload, device, readback] {
        resources.destroy_buffer(buffer);
    }
}

/// Eight threads each translate a list that fills a different region of
/// one buffer, and the command buffers are submitted as one batch, in a
/// chosen order (R16's `parallel_translation`). Each thread enters the
/// test's session span, so validation messages reach the test.
#[test]
fn parallel_translation() {
    const THREADS: u64 = 8;
    /// The bytes each thread fills.
    const REGION: u64 = 16;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let graphics = backend.queue(QueueKind::Graphics);
    let readback = create(
        &**resources,
        "readback",
        THREADS * REGION,
        MemoryLocation::Readback,
    );
    let base = resources.buffer_address(readback).0;

    let lists: Vec<_> = (0..THREADS)
        .map(|index| {
            let mut list = CommandList::new(&format!("region {index}"));
            let value = u32::from_le_bytes([index as u8 + 1; 4]);
            list.fill_buffer(crate::GpuAddress(base + index * REGION), REGION, value);
            list
        })
        .collect();
    let mut command_buffers: Vec<_> = std::thread::scope(|scope| {
        let threads: Vec<_> = lists
            .iter()
            .map(|list| {
                let span = session.span().clone();
                scope.spawn(move || span.in_scope(|| graphics.translate(list).unwrap()))
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    });
    let mut last = CommandList::new("make visible");
    last.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    command_buffers.push(graphics.translate(&last).unwrap());
    let done = graphics.submit(SubmitDesc::new(command_buffers)).unwrap();
    graphics.wait(done).unwrap();

    let mut out = vec![0; (THREADS * REGION) as usize];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    for (index, region) in out.chunks(REGION as usize).enumerate() {
        assert!(
            region.iter().all(|&byte| byte == index as u8 + 1),
            "region {index}: {region:?}"
        );
    }
    resources.destroy_buffer(readback);
}

/// A fill writes its value to every word of its range, and nothing
/// outside it, with a global barrier making it visible to the host. Debug
/// regions and labels around the fills translate with no validation error.
#[test]
fn fills_write_their_range() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let readback = create(&**resources, "readback", 16, MemoryLocation::Readback);
    let address = resources.buffer_address(readback).0;

    let mut list = CommandList::new("fill");
    list.begin_debug_region("clear");
    list.fill_buffer(crate::GpuAddress(address), 16, 0);
    list.insert_debug_label("cleared");
    list.end_debug_region();
    list.insert_debug_label("patterned");
    list.barrier(&[Barrier::global(Access::CopyWrite, Access::CopyWrite)]);
    list.fill_buffer(crate::GpuAddress(address + 4), 8, 0x0403_0201);
    list.barrier(&[Barrier::global(Access::CopyWrite, Access::HostRead)]);
    run(&backend, QueueKind::Graphics, &list);

    let mut out = [0xFF; 16];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, [0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 0]);
    resources.destroy_buffer(readback);
}

/// Storage buffers and other buffers are filled and copied on every queue,
/// with no validation error, with their ranges flagged as possibly in a
/// storage buffer, as translation flags every range. NVIDIA's driver 617.42
/// writes nothing for such copies on its transfer-only family, which device
/// selection doesn't use there, so this checks the data too.
#[test]
fn storage_buffers_are_filled_and_copied() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    for kind in [QueueKind::Graphics, QueueKind::Compute, QueueKind::Transfer] {
        for usage in [BufferUsage::STORAGE, BufferUsage::COPIES] {
            let create = |label, location| {
                resources
                    .create_buffer(&BufferDesc {
                        label,
                        size: 8,
                        usage,
                        location,
                    })
                    .unwrap()
            };
            let source = create("source", MemoryLocation::Device);
            let readback = create("readback", MemoryLocation::Readback);

            let mut list = CommandList::new("storage");
            list.fill_buffer(resources.buffer_address(source), 8, 0x0403_0201);
            list.barrier(&[Barrier::buffer(source, Access::CopyWrite, Access::CopyRead)]);
            list.copy_buffer(
                resources.buffer_address(source),
                resources.buffer_address(readback),
                8,
            );
            list.barrier(&[Barrier::buffer(
                readback,
                Access::CopyWrite,
                Access::HostRead,
            )]);
            run(&backend, kind, &list);

            let mut out = [0; 8];
            resources.read_buffer(readback, 0, &mut out).unwrap();
            assert_eq!(out, [1, 2, 3, 4, 1, 2, 3, 4], "{kind:?}, {usage:?}");
            resources.destroy_buffer(source);
            resources.destroy_buffer(readback);
        }
    }
}

/// A 4×4 texture of `format`, with three mips and two layers, for
/// `usage`.
fn create_texture(
    resources: &dyn GpuResources,
    label: &str,
    format: Format,
    usage: TextureUsage,
) -> Texture {
    resources
        .create_texture(&TextureDesc {
            label,
            dimension: TextureDimension::D2,
            format,
            extent: Extent3d::new_2d(4, 4),
            mip_levels: 3,
            array_layers: 2,
            usage,
        })
        .unwrap()
}

/// Texture barriers move each subresource they name between the layouts
/// their accesses need, so different mips can be in different layouts,
/// which validation checks against each image's tracked layouts when it's
/// submitted. A depth texture's barriers use its depth aspect, and move it
/// between sets of accesses: tested and written, in the attachment layout,
/// then tested and sampled, in the read-only layout.
#[test]
fn texture_barriers_move_subresources_between_layouts() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let colour = create_texture(
        &**resources,
        "colour",
        Format::Rgba8Unorm,
        TextureUsage::SAMPLED | TextureUsage::COPY_SOURCE | TextureUsage::COPY_DESTINATION,
    );
    let depth = create_texture(
        &**resources,
        "depth",
        Format::Depth32Float,
        TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLED,
    );

    let mut list = CommandList::new("layouts");
    list.barrier(&[
        Barrier::texture(colour, Access::None, Access::CopyWrite),
        Barrier::texture(
            depth,
            Access::None,
            Access::DepthAttachmentRead | Access::DepthAttachmentWrite,
        ),
    ]);
    list.barrier(&[
        Barrier::subresource(
            colour,
            Subresource::mip(0),
            Access::CopyWrite,
            Access::CopyRead,
        ),
        Barrier::subresource(
            colour,
            Subresource {
                mips: 1..Subresource::REMAINING,
                layers: 0..Subresource::REMAINING,
            },
            Access::CopyWrite,
            Access::FragmentShaderSampledRead,
        ),
    ]);
    list.barrier(&[
        Barrier::subresource(
            colour,
            Subresource::mip_layer(0, 1),
            Access::CopyRead,
            Access::FragmentShaderSampledRead,
        ),
        Barrier::texture(
            depth,
            Access::DepthAttachmentRead | Access::DepthAttachmentWrite,
            Access::DepthAttachmentRead | Access::FragmentShaderSampledRead,
        ),
    ]);
    run(&backend, QueueKind::Graphics, &list);
    resources.destroy_texture(colour);
    resources.destroy_texture(depth);
}

/// A barrier naming a mip the texture doesn't have panics when the list is
/// translated, naming the texture.
#[test]
#[should_panic(expected = "texture \"colour\" has 3 mips, so a subresource can't name mips 3..4")]
fn subresources_outside_their_texture_panic() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let colour = create_texture(
        &**resources,
        "colour",
        Format::Rgba8Unorm,
        TextureUsage::SAMPLED,
    );
    let mut list = CommandList::new("past the last mip");
    list.barrier(&[Barrier::subresource(
        colour,
        Subresource::mip(3),
        Access::None,
        Access::FragmentShaderSampledRead,
    )]);
    let _ = backend.queue(QueueKind::Graphics).translate(&list);
}

/// The whole of mip 0, layer 0, of `texture`, which is `extent` there.
fn whole(texture: Texture, extent: Extent3d) -> TextureCopy {
    TextureCopy {
        texture,
        mip: 0,
        layers: 0..1,
        origin: [0; 3],
        extent,
    }
}

/// Data with padded rows, after an offset, copied into a texture and back
/// out tightly packed, matches the texels it held (R25's
/// `padded_rows_round_trip`).
///
/// On NVIDIA's driver 617.42, the copy out of the texture doesn't wait for
/// the copy into it, despite the barrier between them, and reads zeros,
/// unless the copy into it is flagged as possibly in a storage buffer,
/// which translation does to every range. Split into two submissions, the
/// round trip passes with no flags, so the copy in writes its data, and the
/// fault is in ordering the two, as the synchronization bug NVIDIA has
/// acknowledged in `vkCmdCopyImageToMemoryKHR` would be. `TODO.md` tracks
/// it.
#[test]
fn padded_rows_round_trip() {
    const WIDTH: u32 = 4;
    const HEIGHT: u32 = 4;
    const PADDED_ROW: u32 = 256;
    const OFFSET: u64 = 64;
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let extent = Extent3d::new_2d(WIDTH, HEIGHT);
    let texture = create_texture(
        &**resources,
        "padded",
        Format::Rgba8Unorm,
        TextureUsage::COPY_SOURCE | TextureUsage::COPY_DESTINATION,
    );
    let tight_size = u64::from(WIDTH * HEIGHT * 4);
    let padded_size = OFFSET + u64::from(PADDED_ROW * HEIGHT);
    let upload = create(
        &**resources,
        "padded rows",
        padded_size,
        MemoryLocation::Upload,
    );
    let readback = create(
        &**resources,
        "tightly packed rows",
        tight_size,
        MemoryLocation::Readback,
    );

    // Each texel's bytes are its row and column, and the padding is 0xEE,
    // which no texel holds.
    let mut padded = vec![0xEE; padded_size as usize];
    let mut packed = Vec::new();
    for row in 0..HEIGHT {
        for column in 0..WIDTH {
            let texel = [row as u8, column as u8, 0x10, 0x20];
            let at = (OFFSET + u64::from(row * PADDED_ROW + column * 4)) as usize;
            padded[at..at + 4].copy_from_slice(&texel);
            packed.extend_from_slice(&texel);
        }
    }
    resources.write_buffer(upload, 0, &padded);

    let padded_layout = TextureDataLayout {
        offset: OFFSET,
        bytes_per_row: PADDED_ROW,
        rows_per_image: HEIGHT,
    };
    let mut list = CommandList::new("padded round trip");
    list.barrier(&[
        Barrier::buffer(upload, Access::HostWrite, Access::CopyRead),
        Barrier::texture(texture, Access::None, Access::CopyWrite),
    ]);
    list.copy_buffer_to_texture(
        resources.buffer_address(upload),
        padded_layout,
        whole(texture, extent),
    );
    list.barrier(&[Barrier::texture(
        texture,
        Access::CopyWrite,
        Access::CopyRead,
    )]);
    list.copy_texture_to_buffer(
        whole(texture, extent),
        resources.buffer_address(readback),
        TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent),
    );
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);

    let mut out = vec![0; tight_size as usize];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, packed);
    resources.destroy_buffer(upload);
    resources.destroy_buffer(readback);
    resources.destroy_texture(texture);
}

/// A copy into one layer of a mip other than 0, read back from an offset
/// region of it, gives the texels at that offset, so the mip, layer and
/// origin each reach Vulkan where they belong, rather than mistakes in both
/// directions cancelling out.
#[test]
fn subresources_round_trip() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = create_texture(
        &**resources,
        "layered",
        Format::Rgba8Unorm,
        TextureUsage::COPY_SOURCE | TextureUsage::COPY_DESTINATION,
    );
    let mip = Extent3d::new_2d(2, 2);
    let upload = create(&**resources, "mip 1 of layer 1", 16, MemoryLocation::Upload);
    let readback = create(&**resources, "column 1", 8, MemoryLocation::Readback);
    // Each texel's first two bytes are its column and row.
    let texels: Vec<u8> = (0..2u8)
        .flat_map(|row| (0..2u8).flat_map(move |column| [column, row, 0x30, 0x40]))
        .collect();
    resources.write_buffer(upload, 0, &texels);

    let layer_1_of_mip_1 = |origin: [u32; 3], extent: Extent3d| TextureCopy {
        texture,
        mip: 1,
        layers: 1..2,
        origin,
        extent,
    };
    let column = Extent3d::new_2d(1, 2);
    let mut list = CommandList::new("subresources");
    list.barrier(&[
        Barrier::buffer(upload, Access::HostWrite, Access::CopyRead),
        Barrier::subresource(
            texture,
            Subresource::mip_layer(1, 1),
            Access::None,
            Access::CopyWrite,
        ),
    ]);
    list.copy_buffer_to_texture(
        resources.buffer_address(upload),
        TextureDataLayout::tightly_packed(Format::Rgba8Unorm, mip),
        layer_1_of_mip_1([0; 3], mip),
    );
    list.barrier(&[Barrier::subresource(
        texture,
        Subresource::mip_layer(1, 1),
        Access::CopyWrite,
        Access::CopyRead,
    )]);
    list.copy_texture_to_buffer(
        layer_1_of_mip_1([1, 0, 0], column),
        resources.buffer_address(readback),
        TextureDataLayout::tightly_packed(Format::Rgba8Unorm, column),
    );
    list.barrier(&[Barrier::buffer(
        readback,
        Access::CopyWrite,
        Access::HostRead,
    )]);
    run(&backend, QueueKind::Graphics, &list);

    let mut out = [0; 8];
    resources.read_buffer(readback, 0, &mut out).unwrap();
    assert_eq!(out, [1, 0, 0x30, 0x40, 1, 1, 0x30, 0x40]);
    resources.destroy_buffer(upload);
    resources.destroy_buffer(readback);
    resources.destroy_texture(texture);
}

/// Copies out of a depth texture, and out of an offset region of one layer
/// of a mip other than 0, name their aspect and subresource as Vulkan
/// requires, which validation checks. Copying out of a texture works on
/// every driver, so this runs where `padded_rows_round_trip` skips.
#[test]
fn copies_from_textures_name_their_subresources() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let colour = create_texture(
        &**resources,
        "colour",
        Format::Rgba8Unorm,
        TextureUsage::COPY_SOURCE,
    );
    let depth = create_texture(
        &**resources,
        "depth",
        Format::Depth32Float,
        TextureUsage::COPY_SOURCE,
    );
    let extent = Extent3d::new_2d(4, 4);
    let texel = Extent3d::new_2d(1, 1);
    let readback = create(
        &**resources,
        "copied out",
        256 + 64,
        MemoryLocation::Readback,
    );

    let mut list = CommandList::new("copies out");
    list.barrier(&[
        Barrier::subresource(
            colour,
            Subresource::mip_layer(1, 1),
            Access::None,
            Access::CopyRead,
        ),
        Barrier::subresource(depth, Subresource::layer(0), Access::None, Access::CopyRead),
    ]);
    list.copy_texture_to_buffer(
        TextureCopy {
            texture: colour,
            mip: 1,
            layers: 1..2,
            origin: [1, 1, 0],
            extent: texel,
        },
        resources.buffer_address(readback),
        TextureDataLayout::tightly_packed(Format::Rgba8Unorm, texel),
    );
    list.copy_texture_to_buffer(
        whole(depth, extent),
        resources.buffer_address(readback),
        TextureDataLayout {
            offset: 256,
            ..TextureDataLayout::tightly_packed(Format::Depth32Float, extent)
        },
    );
    run(&backend, QueueKind::Graphics, &list);
    resources.destroy_buffer(readback);
    resources.destroy_texture(colour);
    resources.destroy_texture(depth);
}

/// A copy that fails its checks panics when it's translated, naming the
/// texture.
#[test]
#[should_panic(
    expected = "a copy of texture \"colour\" has 12 bytes per row, but needs a multiple of 4 \
                that's at least 16"
)]
fn texture_copies_are_checked_when_translated() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = create_texture(
        &**resources,
        "colour",
        Format::Rgba8Unorm,
        TextureUsage::COPY_DESTINATION,
    );
    let extent = Extent3d::new_2d(4, 4);
    let mut list = CommandList::new("short rows");
    list.copy_buffer_to_texture(
        crate::GpuAddress(0x1000),
        TextureDataLayout {
            bytes_per_row: 12,
            ..TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent)
        },
        whole(texture, extent),
    );
    let _ = backend.queue(QueueKind::Graphics).translate(&list);
}

/// A command buffer's pool is claimed again for its next command buffer
/// while its submission is pending, reset and reused once it completes, and
/// reused as soon as an unsubmitted one is dropped, so translating every
/// frame doesn't create pools.
#[test]
fn command_pools_are_reused() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let graphics = backend.queue(QueueKind::Graphics);
    let list = CommandList::new("empty");
    let pool =
        |command_buffer: &crate::CommandBuffer| command_buffer.0.commands.as_ref().unwrap().pool();

    let submitted = graphics.translate(&list).unwrap();
    let first = pool(&submitted);
    let done = graphics.submit(SubmitDesc::new(vec![submitted])).unwrap();
    let pending = graphics.translate(&list).unwrap();
    assert_eq!(pool(&pending), first);
    drop(pending);
    graphics.wait(done).unwrap();
    let dropped = graphics.translate(&list).unwrap();
    assert_eq!(pool(&dropped), first);
    drop(dropped);
    assert_eq!(pool(&graphics.translate(&list).unwrap()), first);
}

/// A command buffer can only be submitted to the queue it was translated
/// for, since it was recorded for that queue's family.
#[test]
#[should_panic(
    expected = "command list \"copies\" was translated for the graphics queue, so it can't be \
                submitted to another queue"
)]
fn command_buffers_belong_to_their_queue() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let command_buffer = backend
        .queue(QueueKind::Graphics)
        .translate(&CommandList::new("copies"))
        .unwrap();
    command_buffer.0.check_queue(QueueId::new(1));
}
