//! Throwaway prototype for the first rhi-shaders PR: how shaders reach the
//! descriptor heaps. Not for merging.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::slice;

use ash::vk;
use clockwork_diagnostics::capture::TestSession;

use crate::vulkan::queues::QueueRole;
use crate::vulkan::test_support::{TestCommands, VulkanTestBackend};
use crate::{
    AddressMode, BufferDesc, BufferUsage, Extent3d, Filter, Format, MemoryLocation, QueueKind,
    RenderHardwareInterface, SamplerDesc, TextureDesc, TextureDimension, TextureUsage,
};

/// `DescriptorHandle<T>`, with `-capability spvDescriptorHeapEXT`.
const HANDLE: &[u8] = include_bytes!("heap_prototype/heap_ext.spv");
/// `ResourceDescriptorHeap[uint]`, with `-capability spvDescriptorHeapEXT`.
const INDEX: &[u8] = include_bytes!("heap_prototype/heap_index.spv");
/// Ordinary bindings, mapped onto the heaps.
const MAPPED: &[u8] = include_bytes!("heap_prototype/heap_mapped.spv");
/// Typed `ResourceIndex<T>` and `SamplerIndex` wrappers around a `uint`.
const WRAPPED: &[u8] = include_bytes!("heap_prototype/heap_wrapped.spv");

/// The colour the texture is cleared to.
const COLOUR: [f32; 4] = [1.0, 0.5, 0.25, 1.0];

#[derive(Clone, Copy, Debug)]
enum Variant {
    Handle,
    Index,
    Mapped,
    Wrapped,
}

fn words(spirv: &[u8]) -> Vec<u32> {
    ash::util::read_spv(&mut std::io::Cursor::new(spirv)).unwrap()
}

fn run(variant: Variant) {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let core = &backend.core;
    let device = core.device.handle();
    let heap_ext = core.device.descriptor_heap();
    let props = &core.physical_device.properties.descriptor_heap;
    eprintln!(
        "{variant:?} on {}: image size {} align {}, sampler size {} align {}, buffer size {} \
         align {}, resource reserved {}, sampler reserved {}",
        backend.resources().capabilities().device_name,
        props.image_descriptor_size,
        props.image_descriptor_alignment,
        props.sampler_descriptor_size,
        props.sampler_descriptor_alignment,
        props.buffer_descriptor_size,
        props.buffer_descriptor_alignment,
        props.min_resource_heap_reserved_range,
        props.min_sampler_heap_reserved_range,
    );
    let untyped = core
        .physical_device
        .supports_extension(ash::khr::shader_untyped_pointers::NAME);
    eprintln!("{variant:?}: VK_KHR_shader_untyped_pointers supported: {untyped}");
    if !untyped && !matches!(variant, Variant::Mapped) {
        panic!("{variant:?}: no VK_KHR_shader_untyped_pointers");
    }
    let resources = backend.resources();

    let texture = resources
        .create_texture(&TextureDesc {
            label: "texel",
            dimension: TextureDimension::D2,
            format: Format::Rgba8Unorm,
            extent: Extent3d::new_2d(1, 1),
            mip_levels: 1,
            array_layers: 1,
            usage: TextureUsage::SAMPLED | TextureUsage::COPY_DESTINATION,
        })
        .unwrap();
    let image = core
        .lock_textures_for_reading()
        .get(texture.handle())
        .image();
    let sampler = resources
        .create_sampler(&SamplerDesc {
            label: "linear",
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mip_filter: Filter::Linear,
            address_modes: [AddressMode::ClampToEdge; 3],
        })
        .unwrap();
    let sampler_index = resources.sampler_index(sampler).0;
    let output = resources
        .create_buffer(&BufferDesc {
            label: "output",
            size: 32,
            usage: BufferUsage::STORAGE,
            location: MemoryLocation::Readback,
        })
        .unwrap();
    let output_address = resources.buffer_address(output).0;

    // Write the texture's descriptor into a resource heap slot.
    let heap = &core.heaps.resource;
    let slot = heap.allocate_slot().unwrap();
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let view = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(range);
    let image_info = vk::ImageDescriptorInfoEXT {
        s_type: vk::StructureType::IMAGE_DESCRIPTOR_INFO_EXT,
        p_next: std::ptr::null(),
        p_view: &view,
        layout: vk::ImageLayout::READ_ONLY_OPTIMAL,
        _marker: PhantomData,
    };
    let resource_info = vk::ResourceDescriptorInfoEXT {
        s_type: vk::StructureType::RESOURCE_DESCRIPTOR_INFO_EXT,
        p_next: std::ptr::null(),
        ty: vk::DescriptorType::SAMPLED_IMAGE,
        data: vk::ResourceDescriptorDataEXT {
            p_image: &image_info,
        },
        _marker: PhantomData,
    };
    let slot_range = heap.slot_address_range(slot);
    unsafe {
        heap_ext
            .write_resource_descriptors(
                slice::from_ref(&resource_info),
                slice::from_ref(&slot_range),
            )
            .unwrap();
    }
    eprintln!("texture slot {slot}, sampler slot {sampler_index}");

    // The shader object.
    let resource_stride = heap.layout().stride() as u32;
    let sampler_stride = core.heaps.sampler.layout().stride() as u32;
    let mappings = [
        vk::DescriptorSetAndBindingMappingEXT {
            s_type: vk::StructureType::DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
            p_next: std::ptr::null(),
            descriptor_set: 0,
            first_binding: 0,
            binding_count: 1,
            resource_mask: vk::SpirvResourceTypeFlagsEXT::SAMPLED_IMAGE,
            source: vk::DescriptorMappingSourceEXT::HEAP_WITH_CONSTANT_OFFSET,
            source_data: vk::DescriptorMappingSourceDataEXT {
                constant_offset: vk::DescriptorMappingSourceConstantOffsetEXT {
                    heap_offset: 0,
                    heap_array_stride: resource_stride,
                    p_embedded_sampler: std::ptr::null(),
                    sampler_heap_offset: 0,
                    sampler_heap_array_stride: 0,
                    _marker: PhantomData,
                },
            },
            _marker: PhantomData,
        },
        vk::DescriptorSetAndBindingMappingEXT {
            s_type: vk::StructureType::DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
            p_next: std::ptr::null(),
            descriptor_set: 0,
            first_binding: 1,
            binding_count: 1,
            resource_mask: vk::SpirvResourceTypeFlagsEXT::SAMPLER,
            source: vk::DescriptorMappingSourceEXT::HEAP_WITH_CONSTANT_OFFSET,
            source_data: vk::DescriptorMappingSourceDataEXT {
                constant_offset: vk::DescriptorMappingSourceConstantOffsetEXT {
                    heap_offset: 0,
                    heap_array_stride: sampler_stride,
                    p_embedded_sampler: std::ptr::null(),
                    sampler_heap_offset: 0,
                    sampler_heap_array_stride: sampler_stride,
                    _marker: PhantomData,
                },
            },
            _marker: PhantomData,
        },
    ];
    let mapping_info = vk::ShaderDescriptorSetAndBindingMappingInfoEXT {
        s_type: vk::StructureType::SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
        p_next: std::ptr::null(),
        mapping_count: mappings.len() as u32,
        p_mappings: mappings.as_ptr(),
        _marker: PhantomData,
    };
    let code = words(match variant {
        Variant::Handle => HANDLE,
        Variant::Index => INDEX,
        Variant::Mapped => MAPPED,
        Variant::Wrapped => WRAPPED,
    });
    let code_bytes =
        unsafe { slice::from_raw_parts(code.as_ptr().cast::<u8>(), size_of_val(code.as_slice())) };
    let mut create_info = vk::ShaderCreateInfoEXT::default()
        .flags(vk::ShaderCreateFlagsEXT::DESCRIPTOR_HEAP)
        .stage(vk::ShaderStageFlags::COMPUTE)
        .code_type(vk::ShaderCodeTypeEXT::SPIRV)
        .code(code_bytes)
        .name(c"main");
    if matches!(variant, Variant::Mapped) {
        create_info.p_next = (&raw const mapping_info).cast::<c_void>();
    }
    let shader_object = core.device.shader_object();
    let shader = unsafe { shader_object.create_shaders(slice::from_ref(&create_info), None) }
        .map_err(|(_, err)| err)
        .unwrap()[0];

    // Push data.
    let mut push = Vec::new();
    push.extend_from_slice(&output_address.to_le_bytes());
    match variant {
        Variant::Handle => {
            push.extend_from_slice(&[slot, 0].map(u32::to_le_bytes).concat());
            push.extend_from_slice(&[sampler_index, 0].map(u32::to_le_bytes).concat());
        }
        Variant::Index | Variant::Mapped | Variant::Wrapped => {
            push.extend_from_slice(&slot.to_le_bytes());
            push.extend_from_slice(&sampler_index.to_le_bytes());
        }
    }

    // Record.
    let commands = TestCommands::new(&backend);
    let command_buffer = commands.command_buffer;
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    let bind = |heap: &crate::vulkan::descriptor_heaps::DescriptorHeap| vk::BindHeapInfoEXT {
        s_type: vk::StructureType::BIND_HEAP_INFO_EXT,
        p_next: std::ptr::null(),
        heap_range: vk::DeviceAddressRangeEXT {
            address: heap.address(),
            size: heap.layout().reserved_offset() + heap.layout().reserved(),
        },
        reserved_range_offset: heap.layout().reserved_offset(),
        reserved_range_size: heap.layout().reserved(),
        _marker: PhantomData,
    };
    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap();
        let to_clear = vk::ImageMemoryBarrier2::default()
            .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(range);
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&to_clear)),
        );
        device.cmd_clear_color_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue { float32: COLOUR },
            slice::from_ref(&range),
        );
        let to_read = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
            .image(image)
            .subresource_range(range);
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&to_read)),
        );
        heap_ext.cmd_bind_resource_heap(command_buffer, &bind(&core.heaps.resource));
        heap_ext.cmd_bind_sampler_heap(command_buffer, &bind(&core.heaps.sampler));
        shader_object.cmd_bind_shaders(command_buffer, &[vk::ShaderStageFlags::COMPUTE], &[shader]);
        heap_ext.cmd_push_data(
            command_buffer,
            &vk::PushDataInfoEXT {
                s_type: vk::StructureType::PUSH_DATA_INFO_EXT,
                p_next: std::ptr::null(),
                offset: 0,
                data: vk::HostAddressRangeConstEXT {
                    address: push.as_ptr().cast(),
                    size: push.len(),
                    _marker: PhantomData,
                },
                _marker: PhantomData,
            },
        );
        device.cmd_dispatch(command_buffer, 1, 1, 1);
        let to_host = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().memory_barriers(slice::from_ref(&to_host)),
        );
        device.end_command_buffer(command_buffer).unwrap();
    }
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
    let command_buffer_info = vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);
    let submit_info =
        vk::SubmitInfo2::default().command_buffer_infos(slice::from_ref(&command_buffer_info));
    let queue = core
        .queues
        .lock(QueueRole::Kind(QueueKind::Graphics))
        .unwrap();
    unsafe { device.queue_submit2(*queue, slice::from_ref(&submit_info), fence) }.unwrap();
    drop(queue);
    unsafe {
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        device.destroy_fence(fence, None);
    }

    let mut bytes = [0_u8; 32];
    resources.read_buffer(output, 0, &mut bytes).unwrap();
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    eprintln!(
        "{variant:?}: load {:?}, sample {:?}",
        &floats[..4],
        &floats[4..]
    );
    for (got, want) in floats.iter().zip(COLOUR.iter().chain(COLOUR.iter())) {
        assert!((got - want).abs() < 1.0 / 255.0, "{floats:?}");
    }

    unsafe { shader_object.destroy_shader(shader, None) };
    heap.free_slot(slot);
    resources.destroy_buffer(output);
    resources.destroy_sampler(sampler);
    resources.destroy_texture(texture);
}

#[test]
fn handle() {
    run(Variant::Handle);
}

#[test]
fn index() {
    run(Variant::Index);
}

#[test]
fn mapped() {
    run(Variant::Mapped);
}

#[test]
fn wrapped() {
    run(Variant::Wrapped);
}
