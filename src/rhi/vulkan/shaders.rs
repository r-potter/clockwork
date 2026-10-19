//! The shader objects that draw the triangle.

use std::ffi::CStr;
use std::io::Cursor;
use std::slice;

use anyhow::{Context, Result};
use ash::{ext, vk};

use super::device::LogicalDevice;

/// The triangle shaders, compiled from `shaders/triangle.slang` by
/// `build.rs`.
const TRIANGLE_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.spv"));

/// The shader entry points in [`TRIANGLE_SPIRV`].
const VERTEX_ENTRY_POINT: &CStr = c"vertexMain";
const FRAGMENT_ENTRY_POINT: &CStr = c"fragmentMain";

/// Linked vertex and fragment shader objects (`VK_EXT_shader_object`) that
/// draw a triangle from three vertices and no bound resources.
///
/// Unlike a pipeline, shader objects hold no other state, so they do not
/// depend on the attachment format, and all rendering state is set when
/// drawing (see [`TriangleShaders::bind`]).
///
/// Destroyed when dropped, so the owner must drop it before the logical
/// device it was created from.
pub struct TriangleShaders {
    /// Kept to destroy the shaders.
    shader_object: ext::shader_object::Device,
    device: ash::Device,
    vertex: vk::ShaderEXT,
    fragment: vk::ShaderEXT,
}

impl TriangleShaders {
    /// Creates the shader objects.
    ///
    /// # Errors
    ///
    /// Returns an error if the shaders cannot be created.
    pub fn new(device: &LogicalDevice) -> Result<Self> {
        // SPIR-V code must be 4-byte aligned, which `include_bytes!` does not
        // guarantee, so copy it into 32-bit words and pass a view of those.
        let words = ash::util::read_spv(&mut Cursor::new(TRIANGLE_SPIRV))
            .context("the compiled triangle shader is not valid SPIR-V")?;
        // SAFETY: The bytes of `words` are initialised and live as long as
        // `code`, and `u8` has no alignment or validity requirements.
        let code = unsafe {
            slice::from_raw_parts(words.as_ptr().cast::<u8>(), size_of_val(words.as_slice()))
        };

        // Linking the stages lets the driver optimise them together, as it
        // would for a pipeline. Linked shaders must be created in one call.
        let create_infos = [
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::LINK_STAGE)
                .stage(vk::ShaderStageFlags::VERTEX)
                .next_stage(vk::ShaderStageFlags::FRAGMENT)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(code)
                .name(VERTEX_ENTRY_POINT),
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::LINK_STAGE)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(code)
                .name(FRAGMENT_ENTRY_POINT),
        ];
        let shader_object = device.shader_object().clone();
        // SAFETY: `VK_EXT_shader_object` and its feature are enabled on the
        // device, and `create_infos` and the code they point to are valid for
        // the duration of the call.
        let shaders = match unsafe { shader_object.create_shaders(&create_infos, None) } {
            Ok(shaders) => shaders,
            Err((shaders, err)) => {
                // Destroy any shaders that were created before the failure.
                for shader in shaders {
                    // SAFETY: The shader was just created and is unused.
                    unsafe { shader_object.destroy_shader(shader, None) };
                }
                return Err(err).context("failed to create Vulkan shader objects");
            }
        };

        Ok(Self {
            shader_object,
            device: device.handle().clone(),
            vertex: shaders[0],
            fragment: shaders[1],
        })
    }

    /// Binds the shaders, and sets the rendering state that a pipeline would
    /// otherwise provide, for drawing to a single colour attachment of
    /// `extent`.
    ///
    /// # Safety
    ///
    /// `command_buffer` must be recording, inside dynamic rendering to a
    /// single colour attachment, on the device the shaders were created on.
    pub unsafe fn bind(&self, command_buffer: vk::CommandBuffer, extent: vk::Extent2D) {
        let shader_object = &self.shader_object;
        let device = &self.device;
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };

        // SAFETY: Guaranteed by the caller, and every slice is valid for the
        // duration of its call.
        unsafe {
            shader_object.cmd_bind_shaders(
                command_buffer,
                &[vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT],
                &[self.vertex, self.fragment],
            );

            // No vertex buffers: the vertex shader generates the vertices.
            shader_object.cmd_set_vertex_input(command_buffer, &[], &[]);
            device.cmd_set_primitive_topology(command_buffer, vk::PrimitiveTopology::TRIANGLE_LIST);
            device.cmd_set_primitive_restart_enable(command_buffer, false);

            // Draw to the whole attachment.
            device.cmd_set_viewport_with_count(command_buffer, slice::from_ref(&viewport));
            device.cmd_set_scissor_with_count(command_buffer, slice::from_ref(&scissor));

            // Rasterise filled triangles from both sides, one sample per pixel.
            device.cmd_set_rasterizer_discard_enable(command_buffer, false);
            shader_object.cmd_set_polygon_mode(command_buffer, vk::PolygonMode::FILL);
            device.cmd_set_cull_mode(command_buffer, vk::CullModeFlags::NONE);
            device.cmd_set_front_face(command_buffer, vk::FrontFace::COUNTER_CLOCKWISE);
            device.cmd_set_depth_bias_enable(command_buffer, false);
            shader_object
                .cmd_set_rasterization_samples(command_buffer, vk::SampleCountFlags::TYPE_1);
            shader_object.cmd_set_sample_mask(command_buffer, vk::SampleCountFlags::TYPE_1, &[!0]);
            shader_object.cmd_set_alpha_to_coverage_enable(command_buffer, false);

            // No depth or stencil attachment.
            device.cmd_set_depth_test_enable(command_buffer, false);
            device.cmd_set_depth_write_enable(command_buffer, false);
            device.cmd_set_stencil_test_enable(command_buffer, false);

            // Write the fragment colour without blending.
            shader_object.cmd_set_color_blend_enable(command_buffer, 0, &[vk::FALSE]);
            shader_object.cmd_set_color_write_mask(
                command_buffer,
                0,
                &[vk::ColorComponentFlags::RGBA],
            );
        }
    }
}

impl Drop for TriangleShaders {
    fn drop(&mut self) {
        // SAFETY: The shaders were created by `new` and are destroyed exactly
        // once here. Waiting for the device to be idle ensures that no
        // submitted work still uses them.
        unsafe {
            // Nothing useful can be done if this fails while dropping.
            let _ = self.device.device_wait_idle();
            self.shader_object.destroy_shader(self.vertex, None);
            self.shader_object.destroy_shader(self.fragment, None);
        }
    }
}
