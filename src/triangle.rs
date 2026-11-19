//! The triangle: a placeholder for the renderer, drawn by application code
//! through the RHI (design 0002, "The triangle"). `0007-pbr`'s renderer
//! replaces it.
//!
//! It draws three vertices from a GPU-only vertex buffer, filled through an
//! [`Uploader`], which its shader reads through the buffer's address in
//! push data, so every frame exercises device addresses and the uploader.

use std::mem::size_of;
use std::sync::Arc;

use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use clockwork_rhi::{
    Access, AccessSet, Barrier, Buffer, BufferDesc, BufferUsage, CommandList, CullMode, Extent2d,
    GpuResources, GpuTimepoint, LoadOp, MemoryLocation, Program, ProgramDesc, QueueKind, Rect2d,
    RenderHardwareInterface, RenderPassDesc, RenderState, ShaderStage, Texture, Upload, Uploader,
    Viewport,
};
use glam::Vec4;

/// The triangle's shaders (`assets/shaders/triangle.slang`), compiled to
/// SPIR-V by `build.rs`.
const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/triangle.spv"));

/// The colour each frame is cleared to before drawing, as linear RGBA.
/// Surface images use an sRGB format, so this appears brighter on screen
/// than the values suggest.
const CLEAR_COLOUR: [f32; 4] = [0.02, 0.05, 0.1, 1.0];

/// One corner of the triangle, as `Vertex` in the shader lays it out.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vertex {
    /// Clip-space position. Vulkan's clip space has +Y pointing down.
    position: Vec4,
    /// Linear RGBA colour, interpolated across the triangle.
    colour: Vec4,
}

// Two `float4`s, as the shader's `Vertex` is.
const _: () = assert!(size_of::<Vertex>() == 32);

/// The shader's push data, as `PushData` in the shader lays it out.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PushData {
    /// The address of the three vertices.
    vertices: u64,
}

// One pointer, as the shader's `PushData` is.
const _: () = assert!(size_of::<PushData>() == 8);

/// The corners: red at the top, then green and blue at the bottom right and
/// bottom left.
const VERTICES: [Vertex; 3] = [
    Vertex {
        position: Vec4::new(0.0, -0.5, 0.0, 1.0),
        colour: Vec4::new(1.0, 0.0, 0.0, 1.0),
    },
    Vertex {
        position: Vec4::new(0.5, 0.5, 0.0, 1.0),
        colour: Vec4::new(0.0, 1.0, 0.0, 1.0),
    },
    Vertex {
        position: Vec4::new(-0.5, 0.5, 0.0, 1.0),
        colour: Vec4::new(0.0, 0.0, 1.0, 1.0),
    },
];

/// The triangle's program and vertex buffer, destroyed when it's dropped.
#[derive(Debug)]
pub(crate) struct Triangle {
    /// Where the program and vertex buffer were created.
    resources: Arc<dyn GpuResources>,
    /// The triangle's vertex and fragment shaders.
    program: Program,
    /// The three vertices, in GPU-only memory.
    vertices: Buffer,
    /// The vertices' upload, until the first draw acquires it.
    upload: Option<Upload>,
}

impl Triangle {
    /// Creates the triangle's program and vertex buffer on `rhi`, and
    /// starts uploading the vertices on its transfer queue.
    ///
    /// # Errors
    ///
    /// Returns an error if the program or buffer can't be created, or the
    /// upload fails.
    pub(crate) fn new(rhi: &dyn RenderHardwareInterface) -> Result<Self> {
        let resources = rhi.resources().clone();
        let spirv = spirv_words(SPIRV)?;
        let program = resources.create_program(&ProgramDesc {
            label: "triangle",
            spirv: &spirv,
            stages: &[
                (ShaderStage::Vertex, c"vertexMain"),
                (ShaderStage::Fragment, c"fragmentMain"),
            ],
        })?;
        let bytes = bytemuck::bytes_of(&VERTICES);
        let vertices = match resources.create_buffer(&BufferDesc {
            label: "triangle vertices",
            size: bytes.len() as u64,
            usage: BufferUsage::STORAGE,
            location: MemoryLocation::Device,
        }) {
            Ok(vertices) => vertices,
            Err(err) => {
                resources.destroy_program(program);
                return Err(err.into());
            }
        };
        // Destroys what's been created if the upload fails.
        let mut triangle = Self {
            resources: resources.clone(),
            program,
            vertices,
            upload: None,
        };
        let mut uploader = Uploader::new(
            resources,
            rhi.queue(QueueKind::Transfer).clone(),
            rhi.queue(QueueKind::Graphics).clone(),
            "triangle",
        );
        uploader.upload_buffer(vertices, 0, bytes, Access::VertexShaderRead)?;
        triangle.upload = Some(uploader.flush()?);
        Ok(triangle)
    }

    /// Records drawing the triangle to all of `target`, a 2D colour
    /// attachment `extent` in size, into `commands`, for the graphics
    /// queue. The target is cleared first, and moves from `before` to
    /// `after`.
    ///
    /// The first draw also acquires the vertices' upload, and returns its
    /// timepoint, which the submission must wait for. Later draws return
    /// `None`.
    pub(crate) fn draw(
        &mut self,
        commands: &mut CommandList,
        target: Texture,
        extent: Extent2d,
        before: impl Into<AccessSet>,
        after: impl Into<AccessSet>,
    ) -> Option<GpuTimepoint> {
        let upload = self.upload.take();
        if let Some(upload) = &upload {
            commands.barrier(&upload.acquire);
        }
        commands.barrier(&[Barrier::texture(
            target,
            before,
            Access::ColorAttachmentWrite,
        )]);
        commands.begin_render_pass(&RenderPassDesc::colour(target, LoadOp::Clear(CLEAR_COLOUR)));
        commands.bind_program(self.program);
        // The corners are clockwise on screen, which the default state would
        // cull as a back face, so culling is off. There's no depth
        // attachment, so there's no depth test either.
        commands.set_render_state(&RenderState::default().cull(CullMode::None).depth(None));
        commands.set_viewport(&Viewport::from_extent(extent));
        commands.set_scissor(Rect2d::from_extent(extent));
        let push_data = PushData {
            vertices: self.resources.buffer_address(self.vertices).0,
        };
        commands.push_data(0, bytemuck::bytes_of(&push_data));
        commands.draw(3, 1);
        commands.end_render_pass();
        commands.barrier(&[Barrier::texture(
            target,
            Access::ColorAttachmentWrite,
            after,
        )]);
        upload.map(|upload| upload.timepoint)
    }
}

impl Drop for Triangle {
    fn drop(&mut self) {
        self.resources.destroy_program(self.program);
        self.resources.destroy_buffer(self.vertices);
    }
}

/// Reads `bytes`, the SPIR-V `slangc` wrote, as words, which
/// `include_bytes!` doesn't align for.
///
/// # Errors
///
/// Returns an error if `bytes` isn't a whole number of words.
fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>> {
    let (words, rest) = bytes.as_chunks::<4>();
    ensure!(
        rest.is_empty(),
        "the triangle's SPIR-V is {} bytes, which isn't a whole number of words",
        bytes.len(),
    );
    Ok(words.iter().map(|&word| u32::from_le_bytes(word)).collect())
}

#[cfg(test)]
mod gpu_tests {
    //! T1 tests of the triangle, on a headless backend (R13).

    use clockwork_diagnostics::capture::TestSession;
    use clockwork_rhi::test_support::backend_options;
    use clockwork_rhi::{
        Extent3d, Format, GraphicsApi, SubmitDesc, TextureCopy, TextureDataLayout, TextureDesc,
        TextureDimension, TextureUsage,
    };

    use super::*;

    /// The offscreen target's width and height, in pixels. Odd, so the
    /// centre pixel's centre is at the centre of clip space.
    const SIZE: u32 = 63;

    /// The triangle renders to an offscreen texture: the centre pixel is
    /// its corners' colours blended, and a corner pixel is the clear
    /// colour.
    #[test]
    fn triangle_renders_offscreen() {
        let session = TestSession::new();
        let rhi = clockwork_rhi::create(GraphicsApi::Vulkan, None, &backend_options(&session))
            .expect("failed to create a headless Vulkan backend");
        let resources = rhi.resources();
        let graphics = rhi.queue(QueueKind::Graphics);
        let mut triangle = Triangle::new(&*rhi).unwrap();

        let extent = Extent2d {
            width: SIZE,
            height: SIZE,
        };
        let target = resources
            .create_texture(&TextureDesc {
                label: "offscreen",
                dimension: TextureDimension::D2,
                format: Format::Rgba8Unorm,
                extent: Extent3d::new_2d(SIZE, SIZE),
                mip_levels: 1,
                array_layers: 1,
                usage: TextureUsage::COLOUR_ATTACHMENT | TextureUsage::COPY_SOURCE,
            })
            .unwrap();
        let layout =
            TextureDataLayout::tightly_packed(Format::Rgba8Unorm, Extent3d::new_2d(SIZE, SIZE));
        let readback = resources
            .create_buffer(&BufferDesc {
                label: "readback",
                size: u64::from(layout.bytes_per_row * SIZE),
                usage: BufferUsage::COPIES,
                location: MemoryLocation::Readback,
            })
            .unwrap();

        let mut commands = CommandList::new("offscreen triangle");
        let upload = triangle.draw(
            &mut commands,
            target,
            extent,
            Access::None,
            Access::CopyRead,
        );
        commands.copy_texture_to_buffer(
            TextureCopy {
                texture: target,
                mip: 0,
                layers: 0..1,
                origin: [0, 0, 0],
                extent: Extent3d::new_2d(SIZE, SIZE),
            },
            resources.buffer_address(readback),
            layout,
        );
        commands.barrier(&[Barrier::buffer(
            readback,
            Access::CopyWrite,
            Access::HostRead,
        )]);
        let command_buffer = graphics.translate(&commands).unwrap();
        let wait_for: Vec<_> = upload.into_iter().collect();
        assert_eq!(wait_for.len(), 1, "the first draw waits for the upload");
        let done = graphics
            .submit(SubmitDesc {
                command_buffers: vec![command_buffer],
                wait_for: &wait_for,
            })
            .unwrap();
        graphics.wait(done).unwrap();

        let mut pixels = vec![0; (layout.bytes_per_row * SIZE) as usize];
        resources.read_buffer(readback, 0, &mut pixels).unwrap();
        let pixel = |x: u32, y: u32| {
            let start = (y * layout.bytes_per_row + x * 4) as usize;
            <[u8; 4]>::try_from(&pixels[start..start + 4]).unwrap()
        };
        // The centre is half way down from the top corner to the middle of
        // the bottom edge, so it's half the top's colour and a quarter of
        // each bottom corner's.
        let centre = [0.5, 0.25, 0.25, 1.0];
        assert_close(pixel(SIZE / 2, SIZE / 2), centre, 1, "the centre");
        assert_close(pixel(0, 0), CLEAR_COLOUR, 1, "a corner");

        resources.destroy_buffer(readback);
        resources.destroy_texture(target);
    }

    /// Asserts that every channel of `actual`, a `Rgba8Unorm` pixel, is
    /// within `tolerance` of `expected`'s.
    fn assert_close(actual: [u8; 4], expected: [f32; 4], tolerance: u8, what: &str) {
        let expected = expected.map(|channel| (channel * 255.0).round() as u8);
        let close = actual
            .iter()
            .zip(expected)
            .all(|(&actual, expected)| actual.abs_diff(expected) <= tolerance);
        assert!(
            close,
            "{what} is {actual:?}, but should be about {expected:?}"
        );
    }
}
