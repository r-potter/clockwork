//! Setting render state, viewports and scissors with dynamic state, since
//! shader objects have no pipeline to bake them into (design 0002,
//! "Programs").

use std::slice;

use ash::vk;

use super::device::LogicalDevice;
use crate::{
    BlendFactor, BlendOp, ColourMask, CompareOp, CullMode, FrontFace, MAX_COLOUR_ATTACHMENTS,
    Rect2d, RenderState, Topology, Viewport,
};

impl From<Topology> for vk::PrimitiveTopology {
    fn from(topology: Topology) -> Self {
        match topology {
            Topology::PointList => Self::POINT_LIST,
            Topology::LineList => Self::LINE_LIST,
            Topology::LineStrip => Self::LINE_STRIP,
            Topology::TriangleList => Self::TRIANGLE_LIST,
            Topology::TriangleStrip => Self::TRIANGLE_STRIP,
        }
    }
}

impl From<CullMode> for vk::CullModeFlags {
    fn from(cull: CullMode) -> Self {
        match cull {
            CullMode::None => Self::NONE,
            CullMode::Front => Self::FRONT,
            CullMode::Back => Self::BACK,
        }
    }
}

impl From<FrontFace> for vk::FrontFace {
    fn from(front_face: FrontFace) -> Self {
        match front_face {
            FrontFace::CounterClockwise => Self::COUNTER_CLOCKWISE,
            FrontFace::Clockwise => Self::CLOCKWISE,
        }
    }
}

impl From<CompareOp> for vk::CompareOp {
    fn from(op: CompareOp) -> Self {
        match op {
            CompareOp::Never => Self::NEVER,
            CompareOp::Less => Self::LESS,
            CompareOp::Equal => Self::EQUAL,
            CompareOp::LessOrEqual => Self::LESS_OR_EQUAL,
            CompareOp::Greater => Self::GREATER,
            CompareOp::NotEqual => Self::NOT_EQUAL,
            CompareOp::GreaterOrEqual => Self::GREATER_OR_EQUAL,
            CompareOp::Always => Self::ALWAYS,
        }
    }
}

impl From<BlendFactor> for vk::BlendFactor {
    fn from(factor: BlendFactor) -> Self {
        match factor {
            BlendFactor::Zero => Self::ZERO,
            BlendFactor::One => Self::ONE,
            BlendFactor::SourceColour => Self::SRC_COLOR,
            BlendFactor::OneMinusSourceColour => Self::ONE_MINUS_SRC_COLOR,
            BlendFactor::DestinationColour => Self::DST_COLOR,
            BlendFactor::OneMinusDestinationColour => Self::ONE_MINUS_DST_COLOR,
            BlendFactor::SourceAlpha => Self::SRC_ALPHA,
            BlendFactor::OneMinusSourceAlpha => Self::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DestinationAlpha => Self::DST_ALPHA,
            BlendFactor::OneMinusDestinationAlpha => Self::ONE_MINUS_DST_ALPHA,
        }
    }
}

impl From<BlendOp> for vk::BlendOp {
    fn from(op: BlendOp) -> Self {
        match op {
            BlendOp::Add => Self::ADD,
            BlendOp::Subtract => Self::SUBTRACT,
            BlendOp::ReverseSubtract => Self::REVERSE_SUBTRACT,
            BlendOp::Min => Self::MIN,
            BlendOp::Max => Self::MAX,
        }
    }
}

impl From<ColourMask> for vk::ColorComponentFlags {
    fn from(mask: ColourMask) -> Self {
        let channel = |written: bool, flag: Self| if written { flag } else { Self::empty() };
        channel(mask.red, Self::R)
            | channel(mask.green, Self::G)
            | channel(mask.blue, Self::B)
            | channel(mask.alpha, Self::A)
    }
}

/// The colour blend enables, equations and write masks that set `state`'s
/// first `attachments` colour attachments. Pure, so it's tested without a
/// GPU.
fn colour_writes(
    state: &RenderState,
    attachments: usize,
) -> (
    Vec<vk::Bool32>,
    Vec<vk::ColorBlendEquationEXT>,
    Vec<vk::ColorComponentFlags>,
) {
    let writes = &state.colour[..attachments];
    let enables = writes
        .iter()
        .map(|write| vk::Bool32::from(write.blend.is_some()))
        .collect();
    // An attachment that doesn't blend gets an equation that replaces, which
    // Vulkan ignores.
    let equations = writes
        .iter()
        .map(|write| {
            let Some(blend) = write.blend else {
                return vk::ColorBlendEquationEXT::default()
                    .src_color_blend_factor(vk::BlendFactor::ONE)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE);
            };
            vk::ColorBlendEquationEXT::default()
                .src_color_blend_factor(blend.colour.source.into())
                .dst_color_blend_factor(blend.colour.destination.into())
                .color_blend_op(blend.colour.operation.into())
                .src_alpha_blend_factor(blend.alpha.source.into())
                .dst_alpha_blend_factor(blend.alpha.destination.into())
                .alpha_blend_op(blend.alpha.operation.into())
        })
        .collect();
    let masks = writes
        .iter()
        .map(|write| vk::ColorComponentFlags::from(write.mask))
        .collect();
    (enables, equations, masks)
}

/// How many colour attachments render state is set for on a device that
/// renders to at most `max_color_attachments`: every one a [`RenderState`]
/// describes, or as many as the device can render to, if fewer.
pub(super) fn colour_attachments(max_color_attachments: u32) -> usize {
    MAX_COLOUR_ATTACHMENTS.min(max_color_attachments as usize)
}

impl LogicalDevice {
    /// Records commands into `buffer` that set every piece of `state`, and
    /// the state a [`RenderState`] fixes, for draws to up to `attachments`
    /// colour attachments.
    ///
    /// # Safety
    ///
    /// `buffer` must be recording, on this device, for a family that
    /// supports graphics, no other thread may use it, and `attachments`
    /// must be at most the device's `maxColorAttachments`.
    pub(super) unsafe fn set_render_state(
        &self,
        buffer: vk::CommandBuffer,
        state: &RenderState,
        attachments: usize,
    ) {
        let device = self.handle();
        let shader_object = self.shader_object();
        let (enables, equations, masks) = colour_writes(state, attachments);
        let depth_compare = state
            .depth
            .map_or(vk::CompareOp::ALWAYS, |depth| depth.compare.into());
        // SAFETY: Guaranteed by the caller, and every slice is valid for the
        // duration of its call.
        unsafe {
            // No vertex buffers: vertex shaders read vertices by address.
            shader_object.cmd_set_vertex_input(buffer, &[], &[]);
            device.cmd_set_primitive_topology(buffer, state.topology.into());
            device.cmd_set_primitive_restart_enable(buffer, false);
            // Shader objects need a line width whenever the topology is a
            // line list or strip, and 1 is the only width without
            // `wideLines`. Set for every topology, like the rest.
            device.cmd_set_line_width(buffer, 1.0);

            // Rasterize filled polygons, one sample per pixel.
            device.cmd_set_rasterizer_discard_enable(buffer, false);
            shader_object.cmd_set_polygon_mode(buffer, vk::PolygonMode::FILL);
            device.cmd_set_cull_mode(buffer, state.cull.into());
            device.cmd_set_front_face(buffer, state.front_face.into());
            device.cmd_set_depth_bias_enable(buffer, false);
            shader_object.cmd_set_rasterization_samples(buffer, vk::SampleCountFlags::TYPE_1);
            shader_object.cmd_set_sample_mask(buffer, vk::SampleCountFlags::TYPE_1, &[!0]);
            shader_object.cmd_set_alpha_to_coverage_enable(buffer, false);

            // Depth as `state` says, and no stencil.
            device.cmd_set_depth_test_enable(buffer, state.depth.is_some());
            device.cmd_set_depth_write_enable(buffer, state.depth.is_some_and(|depth| depth.write));
            device.cmd_set_depth_compare_op(buffer, depth_compare);
            device.cmd_set_stencil_test_enable(buffer, false);

            shader_object.cmd_set_color_blend_enable(buffer, 0, &enables);
            shader_object.cmd_set_color_blend_equation(buffer, 0, &equations);
            shader_object.cmd_set_color_write_mask(buffer, 0, &masks);
        }
    }

    /// Records a command into `buffer` that sets the viewport to `viewport`.
    ///
    /// # Safety
    ///
    /// `buffer` must be recording, on this device, for a family that
    /// supports graphics, and no other thread may use it.
    pub(super) unsafe fn set_viewport(&self, buffer: vk::CommandBuffer, viewport: &Viewport) {
        let viewport = vk::Viewport::from(*viewport);
        // SAFETY: Guaranteed by the caller, and the slice is valid for the
        // duration of the call.
        unsafe {
            self.handle()
                .cmd_set_viewport_with_count(buffer, slice::from_ref(&viewport));
        }
    }

    /// Records a command into `buffer` that sets the scissor to `scissor`.
    ///
    /// # Safety
    ///
    /// `buffer` must be recording, on this device, for a family that
    /// supports graphics, and no other thread may use it.
    pub(super) unsafe fn set_scissor(&self, buffer: vk::CommandBuffer, scissor: Rect2d) {
        let scissor = vk::Rect2D::from(scissor);
        // SAFETY: Guaranteed by the caller, and the slice is valid for the
        // duration of the call.
        unsafe {
            self.handle()
                .cmd_set_scissor_with_count(buffer, slice::from_ref(&scissor));
        }
    }
}

impl From<Viewport> for vk::Viewport {
    fn from(viewport: Viewport) -> Self {
        Self {
            x: viewport.x,
            y: viewport.y,
            width: viewport.width,
            height: viewport.height,
            min_depth: viewport.min_depth,
            max_depth: viewport.max_depth,
        }
    }
}

impl From<Rect2d> for vk::Rect2D {
    fn from(rect: Rect2d) -> Self {
        // Recording checks that a scissor's edges are at most `i32::MAX`.
        let signed = |value: u32| i32::try_from(value).expect("scissor offsets fit in an i32");
        Self {
            offset: vk::Offset2D {
                x: signed(rect.offset.x),
                y: signed(rect.offset.y),
            },
            extent: vk::Extent2D {
                width: rect.extent.width,
                height: rect.extent.height,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of translating colour writes.

    use super::*;
    use crate::{Blend, ColourWrite};

    /// Each attachment gets its own blend enable, equation and mask, and
    /// only as many attachments as asked for are set.
    #[test]
    fn colour_writes_are_per_attachment() {
        let state = RenderState::default().colour(
            1,
            ColourWrite {
                blend: Some(Blend::ALPHA),
                mask: ColourMask {
                    alpha: false,
                    ..ColourMask::ALL
                },
            },
        );
        let (enables, equations, masks) = colour_writes(&state, 4);
        assert_eq!(enables, [vk::FALSE, vk::TRUE, vk::FALSE, vk::FALSE]);
        assert_eq!(
            equations[1].src_color_blend_factor,
            vk::BlendFactor::SRC_ALPHA
        );
        assert_eq!(
            equations[1].dst_color_blend_factor,
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        );
        assert_eq!(equations[1].src_alpha_blend_factor, vk::BlendFactor::ONE);
        assert_eq!(equations.len(), 4);
        assert_eq!(
            masks,
            [
                vk::ColorComponentFlags::RGBA,
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B,
                vk::ColorComponentFlags::RGBA,
                vk::ColorComponentFlags::RGBA,
            ],
        );
    }

    /// Render state is set for every attachment it describes, or as many as
    /// the device can render to, if fewer.
    #[test]
    fn colour_attachments_are_capped_by_the_device() {
        assert_eq!(colour_attachments(4), 4);
        assert_eq!(colour_attachments(8), MAX_COLOUR_ATTACHMENTS);
        assert_eq!(colour_attachments(16), MAX_COLOUR_ATTACHMENTS);
    }
}
