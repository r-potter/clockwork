//! T1 tests of textures on a real device (design 0002, R3 and part of R6).

use std::sync::PoisonError;

use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::test_support::VulkanTestBackend;
use crate::{
    Extent3d, Format, RenderHardwareInterface, Texture, TextureDesc, TextureDimension, TextureUsage,
};

/// A sampled 2D texture of `width` × `height` texels, with one mip.
fn desc_2d(label: &str, width: u32, height: u32) -> TextureDesc<'_> {
    TextureDesc {
        label,
        dimension: TextureDimension::D2,
        format: Format::Rgba8Unorm,
        extent: Extent3d::new_2d(width, height),
        mip_levels: 1,
        array_layers: 1,
        usage: TextureUsage::SAMPLED | TextureUsage::COPY_DESTINATION,
    }
}

/// The memory type `texture`'s memory is of.
fn memory_type(backend: &VulkanTestBackend, texture: Texture) -> u32 {
    backend
        .core
        .textures
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .get(texture.handle())
        .allocation
        .memory_type()
}

/// Textures of every dimension, with mips, layers and each kind of usage,
/// can be created, and destroying them gives their memory back.
#[test]
fn textures_of_every_dimension_are_created() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    // The core's own memory, such as the descriptor heaps, may share a
    // type with the textures, as on lavapipe's only type, so destroying
    // the textures returns each type to what it held before.
    let statistics = |memory_type| backend.core.allocator.statistics(memory_type);
    let before: Vec<_> = (0..backend
        .core
        .physical_device
        .memory_properties
        .memory_type_count)
        .map(statistics)
        .collect();

    let descs = [
        TextureDesc {
            mip_levels: 9,
            usage: TextureUsage::SAMPLED
                | TextureUsage::STORAGE
                | TextureUsage::COPY_SOURCE
                | TextureUsage::COPY_DESTINATION,
            ..desc_2d("mipped", 256, 256)
        },
        TextureDesc {
            array_layers: 4,
            format: Format::Rgba16Float,
            usage: TextureUsage::COLOUR_ATTACHMENT | TextureUsage::SAMPLED,
            ..desc_2d("cascades", 512, 512)
        },
        TextureDesc {
            dimension: TextureDimension::Cube,
            array_layers: 6,
            ..desc_2d("sky", 64, 64)
        },
        TextureDesc {
            dimension: TextureDimension::D3,
            extent: Extent3d {
                width: 16,
                height: 16,
                depth: 16,
            },
            ..desc_2d("volume", 16, 16)
        },
        TextureDesc {
            format: Format::Depth32Float,
            usage: TextureUsage::DEPTH_ATTACHMENT | TextureUsage::SAMPLED,
            ..desc_2d("depth", 640, 480)
        },
    ];
    let textures: Vec<_> = descs
        .iter()
        .map(|desc| resources.create_texture(desc).unwrap())
        .collect();
    let memory_types: Vec<_> = textures
        .iter()
        .map(|&texture| memory_type(&backend, texture))
        .collect();
    for texture in textures {
        resources.destroy_texture(texture);
    }
    for memory_type in memory_types {
        let (before, after) = (before[memory_type as usize], statistics(memory_type));
        assert_eq!(
            (after.pool.allocated, after.dedicated_count),
            (before.pool.allocated, before.dedicated_count),
            "memory type {memory_type}",
        );
    }
}

/// Using a destroyed texture's handle panics, naming the texture (R3).
#[test]
#[should_panic(expected = "stale texture handle: \"albedo\" was destroyed")]
fn stale_texture_handles_panic() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = resources.create_texture(&desc_2d("albedo", 4, 4)).unwrap();
    resources.destroy_texture(texture);
    resources.destroy_texture(texture);
}

/// An invalid description panics before anything is created.
#[test]
#[should_panic(expected = "cube texture \"sky\" has 4 layers, not a multiple of 6")]
fn invalid_textures_panic() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let _ = backend.resources().create_texture(&TextureDesc {
        dimension: TextureDimension::Cube,
        array_layers: 4,
        ..desc_2d("sky", 64, 64)
    });
}

/// Textures left when the device is dropped are destroyed with it, before
/// their memory and the device, so validation reports nothing.
#[test]
fn textures_left_at_drop_are_destroyed() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    backend
        .resources()
        .create_texture(&desc_2d("left behind", 64, 64))
        .unwrap();
    drop(backend);
}
