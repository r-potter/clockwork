//! T1 tests of deferred destruction on a real device (design 0002, R10).
//!
//! Each test destroys objects that work may still use, with validation on,
//! so a buffer freed while a submission or a translated command buffer
//! still names it fails the test.

use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::test_support::VulkanTestBackend;
use crate::{
    Access, AddressMode, Barrier, Buffer, BufferDesc, BufferUsage, CommandList, Extent3d, Filter,
    Format, GpuResources, MemoryLocation, QueueKind, RenderHardwareInterface, SamplerDesc,
    SubmitDesc, TextureDesc, TextureDimension, TextureUsage,
};

/// A 256-byte GPU-only buffer.
fn buffer(resources: &dyn GpuResources, label: &str) -> Buffer {
    resources
        .create_buffer(&BufferDesc {
            label,
            size: 256,
            usage: BufferUsage::COPIES,
            location: MemoryLocation::Device,
        })
        .unwrap()
}

/// A list that fills `buffer` and orders the fill, so that it names the
/// buffer by handle as well as by address.
fn fill(resources: &dyn GpuResources, buffer: Buffer) -> CommandList {
    let mut list = CommandList::new("fill");
    list.fill_buffer(resources.buffer_address(buffer), 256, 0);
    list.barrier(&[Barrier::buffer(buffer, Access::CopyWrite, Access::CopyRead)]);
    list
}

/// How many destroyed objects are waiting to be freed.
fn retired(backend: &VulkanTestBackend) -> usize {
    backend.core.lock_deferred().retired_len()
}

/// A buffer destroyed straight after submitting work that uses it is kept
/// until the work completes, then freed, with no validation error (R10's
/// `destroy_while_in_flight`).
#[test]
fn destroy_while_in_flight() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let graphics = backend.queue(QueueKind::Graphics);
    let target = buffer(&**resources, "in flight");

    let command_buffer = graphics.translate(&fill(&**resources, target)).unwrap();
    let done = graphics
        .submit(SubmitDesc::new(vec![command_buffer]))
        .unwrap();
    resources.destroy_buffer(target);
    assert_eq!(retired(&backend), 1);
    graphics.wait(done).unwrap();
    resources.collect_garbage();
    assert_eq!(retired(&backend), 0);
}

/// A buffer destroyed after a list naming it is translated, and before the
/// list is submitted, is kept until that submission completes.
#[test]
fn lists_translated_before_a_destruction_keep_what_they_use() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let graphics = backend.queue(QueueKind::Graphics);
    let target = buffer(&**resources, "translated");

    let command_buffer = graphics.translate(&fill(&**resources, target)).unwrap();
    resources.destroy_buffer(target);
    resources.collect_garbage();
    assert_eq!(retired(&backend), 1);
    let done = graphics
        .submit(SubmitDesc::new(vec![command_buffer]))
        .unwrap();
    assert_eq!(retired(&backend), 1);
    graphics.wait(done).unwrap();
    resources.collect_garbage();
    assert_eq!(retired(&backend), 0);
}

/// A list dropped unsubmitted stops keeping what it named.
#[test]
fn dropped_lists_keep_nothing() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let target = buffer(&**resources, "dropped");

    let command_buffer = backend
        .queue(QueueKind::Graphics)
        .translate(&fill(&**resources, target))
        .unwrap();
    resources.destroy_buffer(target);
    resources.collect_garbage();
    assert_eq!(retired(&backend), 1);
    drop(command_buffer);
    resources.collect_garbage();
    assert_eq!(retired(&backend), 0);
}

/// Textures and samplers are retired too, and with nothing outstanding
/// they're freed at the next collection.
#[test]
fn textures_and_samplers_are_retired() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = resources
        .create_texture(&TextureDesc {
            label: "albedo",
            dimension: TextureDimension::D2,
            format: Format::Rgba8Unorm,
            extent: Extent3d::new_2d(4, 4),
            mip_levels: 1,
            array_layers: 1,
            usage: TextureUsage::SAMPLED,
        })
        .unwrap();
    let sampler = resources
        .create_sampler(&SamplerDesc {
            label: "linear",
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mip_filter: Filter::Linear,
            address_modes: [AddressMode::Repeat; 3],
        })
        .unwrap();
    let slots = backend.core.heaps.sampler.slots_in_use();
    resources.destroy_texture(texture);
    resources.destroy_sampler(sampler);
    assert_eq!(retired(&backend), 2);
    assert_eq!(backend.core.heaps.sampler.slots_in_use(), slots);
    resources.collect_garbage();
    assert_eq!(retired(&backend), 0);
    assert_eq!(backend.core.heaps.sampler.slots_in_use(), slots - 1);
}

/// Objects still retired when the backend is dropped are freed with it,
/// once the device is idle, and aren't reported as leaks.
#[test]
fn retired_objects_are_freed_at_drop() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let graphics = backend.queue(QueueKind::Graphics);
    let target = buffer(&**resources, "retired at drop");
    let command_buffer = graphics.translate(&fill(&**resources, target)).unwrap();
    graphics
        .submit(SubmitDesc::new(vec![command_buffer]))
        .unwrap();
    resources.destroy_buffer(target);
    assert_eq!(retired(&backend), 1);
}
