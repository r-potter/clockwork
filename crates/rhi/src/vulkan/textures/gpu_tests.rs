//! T1 tests of textures on a real device (design 0002, R3 and part of R6).

use std::iter;
use std::sync::PoisonError;

use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::test_support::{Capability, VulkanTestBackend, require_capabilities};
use crate::{
    ColorAttachment, CommandList, DescriptorIndex, Extent3d, Format, HeapKind, LoadOp, QueueKind,
    RenderHardwareInterface, RenderPassDesc, RhiError, StoreOp, Subresource, Texture, TextureDesc,
    TextureDimension, TextureUsage, TextureView,
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
/// can be created, with a view of each kind their dimension allows, and
/// destroying them gives their memory and views' slots back.
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
    let slots = backend.core.heaps.resource.slots_in_use();
    // A 2D view, a storage view of one mip, a 2D array view and a layer
    // of it, a cube view and a face of it, a 3D view, and a depth view.
    let views = [
        (0, TextureView::sampled()),
        (0, TextureView::Storage(Subresource::mip(3))),
        (1, TextureView::sampled()),
        (1, TextureView::Sampled(Subresource::layer(2))),
        (2, TextureView::sampled()),
        (2, TextureView::Sampled(Subresource::layer(5))),
        (3, TextureView::sampled()),
        (4, TextureView::sampled()),
    ];
    for (texture, view) in &views {
        resources
            .texture_index(textures[*texture], view.clone())
            .unwrap();
    }
    assert_eq!(
        backend.core.heaps.resource.slots_in_use(),
        slots + views.len() as u32
    );
    let memory_types: Vec<_> = textures
        .iter()
        .map(|&texture| memory_type(&backend, texture))
        .collect();
    for texture in textures {
        resources.destroy_texture(texture);
    }
    resources.collect_garbage();
    assert_eq!(backend.core.heaps.resource.slots_in_use(), slots);
    for memory_type in memory_types {
        let (before, after) = (before[memory_type as usize], statistics(memory_type));
        assert_eq!(
            (after.pool.allocated, after.dedicated_count),
            (before.pool.allocated, before.dedicated_count),
            "memory type {memory_type}",
        );
    }
}

/// Textures of each BC format, with mips down to 1×1, smaller than a
/// block, can be created for sampling and copying, as a 2D texture, an
/// array and a cube map, each with a sampled view in the heap, as the
/// importers will load them (part of R26).
#[test]
fn bc_textures_are_created() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    require_capabilities!(backend, Capability::BcCompression);
    let resources = backend.resources();
    for format in Format::ALL
        .into_iter()
        .filter(|format| format.is_block_compressed())
    {
        let base = TextureDesc {
            format,
            mip_levels: 7,
            usage: TextureUsage::SAMPLED
                | TextureUsage::COPY_SOURCE
                | TextureUsage::COPY_DESTINATION,
            ..desc_2d("compressed", 64, 64)
        };
        let descs = [
            base,
            TextureDesc {
                array_layers: 4,
                ..base
            },
            TextureDesc {
                dimension: TextureDimension::Cube,
                array_layers: 6,
                ..base
            },
        ];
        for desc in descs {
            let texture = resources.create_texture(&desc).unwrap();
            resources
                .texture_index(texture, TextureView::sampled())
                .unwrap();
            resources.destroy_texture(texture);
        }
    }
}

/// Each view is written once, into a slot of its own, which later calls
/// return, however they name the same part, and the slots are freed only
/// once the destroyed texture is collected (part of R6).
#[test]
fn views_are_cached_slots() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = resources
        .create_texture(&TextureDesc {
            mip_levels: 4,
            array_layers: 2,
            usage: TextureUsage::SAMPLED | TextureUsage::STORAGE,
            ..desc_2d("albedo", 64, 64)
        })
        .unwrap();
    let heap = &backend.core.heaps.resource;
    let slots = heap.slots_in_use();
    let index = |view| resources.texture_index(texture, view).unwrap();
    let sampled = index(TextureView::sampled());
    let storage = index(TextureView::Storage(Subresource::mip(1)));
    assert_ne!(sampled, storage);
    assert_eq!(heap.slots_in_use(), slots + 2);
    let same_views: [(DescriptorIndex, TextureView); 3] = [
        (sampled, TextureView::sampled()),
        (
            sampled,
            TextureView::Sampled(Subresource {
                mips: 0..4,
                layers: 0..2,
            }),
        ),
        (
            storage,
            TextureView::Storage(Subresource {
                mips: 1..2,
                layers: 0..Subresource::REMAINING,
            }),
        ),
    ];
    for (expected, view) in same_views {
        assert_eq!(index(view.clone()), expected, "{view:?}");
    }
    assert_eq!(heap.slots_in_use(), slots + 2);
    resources.destroy_texture(texture);
    assert_eq!(heap.slots_in_use(), slots + 2);
    resources.collect_garbage();
    assert_eq!(heap.slots_in_use(), slots);
}

/// Each attachment's view is created once, the first time a translation
/// renders to it, and reused by later translations, however they name the
/// same layers. The views are destroyed with the texture, or validation
/// would report them when the device is destroyed.
#[test]
fn attachment_views_are_cached() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = resources
        .create_texture(&TextureDesc {
            mip_levels: 2,
            array_layers: 2,
            usage: TextureUsage::COLOUR_ATTACHMENT,
            ..desc_2d("target", 8, 8)
        })
        .unwrap();
    let attachment_views = || {
        backend
            .core
            .textures
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(texture.handle())
            .attachment_views
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    };
    let render = |mip, layers| {
        let mut list = CommandList::new("render");
        list.begin_render_pass(&RenderPassDesc {
            colour: vec![ColorAttachment {
                texture,
                mip,
                layers,
                load: LoadOp::DontCare,
                store: StoreOp::DontCare,
            }],
            depth: None,
        });
        list.end_render_pass();
        backend.queue(QueueKind::Graphics).translate(&list).unwrap();
    };
    assert_eq!(attachment_views(), 0);
    render(0, 0..2);
    render(0, 0..Subresource::REMAINING);
    assert_eq!(attachment_views(), 1);
    render(1, 1..2);
    assert_eq!(attachment_views(), 2);
    resources.destroy_texture(texture);
    resources.collect_garbage();
}

/// Asking for a view with the resource heap full returns
/// `DescriptorHeapFull`, naming the heap and the texture, and caches
/// nothing, so the view is written once a slot is freed (R24).
#[test]
fn full_resource_heap_is_reported_for_views() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = resources.create_texture(&desc_2d("albedo", 4, 4)).unwrap();
    let heap = &backend.core.heaps.resource;
    let fillers: Vec<_> = iter::from_fn(|| heap.allocate_slot()).collect();

    let err = resources
        .texture_index(texture, TextureView::sampled())
        .unwrap_err();
    assert!(
        matches!(
            &err,
            RhiError::DescriptorHeapFull { heap: HeapKind::Resource, label } if label == "albedo"
        ),
        "{err:?}",
    );

    let (&freed, rest) = fillers.split_last().unwrap();
    heap.free_slot(freed);
    assert_eq!(
        resources
            .texture_index(texture, TextureView::sampled())
            .unwrap(),
        DescriptorIndex(freed),
    );
    for &slot in rest {
        heap.free_slot(slot);
    }
    resources.destroy_texture(texture);
}

/// Asking for a view of a destroyed texture panics, naming the texture
/// (R3).
#[test]
#[should_panic(expected = "stale texture handle: \"albedo\" was destroyed")]
fn views_of_stale_textures_panic() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = resources.create_texture(&desc_2d("albedo", 4, 4)).unwrap();
    resources.destroy_texture(texture);
    let _ = resources.texture_index(texture, TextureView::sampled());
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

/// A texture's info is what it was created as, for each dimension, so the
/// `Uploader` can look up a texture's format and extent.
#[test]
fn textures_report_what_they_were_created_as() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let descs = [
        TextureDesc {
            mip_levels: 3,
            array_layers: 2,
            format: Format::Rgba16Float,
            ..desc_2d("layers", 8, 4)
        },
        TextureDesc {
            dimension: TextureDimension::Cube,
            array_layers: 6,
            usage: TextureUsage::SAMPLED,
            ..desc_2d("sky", 16, 16)
        },
        TextureDesc {
            dimension: TextureDimension::D3,
            extent: Extent3d {
                width: 8,
                height: 4,
                depth: 2,
            },
            ..desc_2d("volume", 8, 4)
        },
    ];
    for desc in descs {
        let texture = resources.create_texture(&desc).unwrap();
        assert_eq!(
            resources.texture_info(texture),
            desc.info(),
            "{}",
            desc.label
        );
        resources.destroy_texture(texture);
    }
}

/// Looking up a destroyed texture's info panics, naming the texture.
#[test]
#[should_panic(expected = "stale texture handle: \"albedo\" was destroyed")]
fn stale_texture_info_panics() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let texture = resources.create_texture(&desc_2d("albedo", 4, 4)).unwrap();
    resources.destroy_texture(texture);
    resources.texture_info(texture);
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
