//! A plan's barriers and render passes as the RHI's (design 0003,
//! "Executing"): each frame, the executor turns the plan's indices into
//! the RHI handles it bound for the frame.
//!
//! Two things are decided here rather than when compiling, since they
//! depend on the frame:
//!
//! - **Barriers past an import's shape.** The compiler doesn't know an
//!   import's mips and layers, so it splits its state at its uses' bounds
//!   as written, where a whole use ends at
//!   [`Subresource::REMAINING`](clockwork_rhi::Subresource::REMAINING). A
//!   barrier on a part from an explicit bound to the end, such as mips `1..`
//!   of an import used whole and as mip 0, covers nothing when the import
//!   has no more mips, and the RHI panics on a barrier over an empty range.
//!   So a barrier that starts past the texture's last mip or layer is left
//!   out.
//! - **The first frame of a resource.** A barrier that orders a first use
//!   after the last frame is from the accesses the frame ends with, which
//!   the last frame left the resource in. A resource created for this frame
//!   has no last frame, and no contents, so its barrier is from
//!   [`Access::None`] instead, which discards them.
//!
//! Render passes are filled into a [`RenderPassDesc`] the executor keeps,
//! so a frame allocates nothing once each has its attachments.

use clockwork_rhi::{
    Access, Barrier, Buffer, ColorAttachment, DepthAttachment, RenderPassDesc, Texture,
};

use crate::barriers::{BarrierTarget, PlannedBarrier};
use crate::render_passes::PlannedRenderPass;

/// The RHI handles a frame binds for a plan's resources, by their indices
/// in the description.
#[derive(Debug, Clone, Default)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "`FrameGraph`, later in `graph-execute`, builds it"
    )
)]
pub(crate) struct Bound {
    /// Each texture's, indexed by `TextureId`, or `None` for one the plan
    /// doesn't use.
    pub(crate) textures: Vec<Option<BoundTexture>>,
    /// Each buffer's, indexed by `BufferId`, or `None` for one the plan
    /// doesn't use.
    pub(crate) buffers: Vec<Option<BoundBuffer>>,
}

/// A texture bound for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "`FrameGraph`, later in `graph-execute`, builds it"
    )
)]
pub(crate) struct BoundTexture {
    /// Its handle.
    pub(crate) texture: Texture,
    /// How many mips it has: its shape's, or for an import, the caller's.
    pub(crate) mips: u32,
    /// How many layers it has.
    pub(crate) layers: u32,
    /// Whether it was created for this frame, so has no last frame.
    pub(crate) fresh: bool,
}

/// A buffer bound for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "`FrameGraph`, later in `graph-execute`, builds it"
    )
)]
pub(crate) struct BoundBuffer {
    /// Its handle.
    pub(crate) buffer: Buffer,
    /// Whether it was created for this frame, so has no last frame.
    pub(crate) fresh: bool,
}

/// Appends to `barriers` the RHI barriers `planned` stands for, with the
/// handles `bound` gives, leaving out those past an import's shape, and
/// moving a fresh resource from [`Access::None`] rather than from the last
/// frame's accesses.
///
/// # Panics
///
/// Panics if a barrier names a resource `bound` has no handle for, which
/// is a bug in the executor.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
)]
pub(crate) fn barriers(planned: &[PlannedBarrier], bound: &Bound, barriers: &mut Vec<Barrier>) {
    let from = |barrier: &PlannedBarrier, fresh: bool| {
        if barrier.from_last_frame && fresh {
            Access::None.into()
        } else {
            barrier.before
        }
    };
    for barrier in planned {
        match &barrier.target {
            BarrierTarget::Texture(index, part) => {
                let texture =
                    bound.textures[*index].expect("every texture a barrier names is bound");
                if part.mips.start >= texture.mips || part.layers.start >= texture.layers {
                    continue;
                }
                barriers.push(Barrier::subresource(
                    texture.texture,
                    part.clone(),
                    from(barrier, texture.fresh),
                    barrier.after,
                ));
            }
            BarrierTarget::Buffer(index) => {
                let buffer = bound.buffers[*index].expect("every buffer a barrier names is bound");
                barriers.push(Barrier::buffer(
                    buffer.buffer,
                    from(barrier, buffer.fresh),
                    barrier.after,
                ));
            }
            BarrierTarget::Global => barriers.push(Barrier::global(barrier.before, barrier.after)),
        }
    }
}

/// Fills `render_pass` with the attachments of `planned`, with the handles
/// `bound` gives, keeping its memory.
///
/// # Panics
///
/// Panics if an attachment names a texture `bound` has no handle for,
/// which is a bug in the executor.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
)]
pub(crate) fn fill_render_pass(
    planned: &PlannedRenderPass,
    bound: &Bound,
    render_pass: &mut RenderPassDesc,
) {
    let texture = |index: usize| {
        bound.textures[index]
            .expect("every texture a render pass names is bound")
            .texture
    };
    render_pass.colour.clear();
    render_pass
        .colour
        .extend(planned.colour.iter().map(|attachment| ColorAttachment {
            texture: texture(attachment.texture),
            mip: attachment.mip,
            layers: attachment.layer..attachment.layer + 1,
            load: attachment.load,
            store: attachment.store,
        }));
    render_pass.depth = planned.depth.as_ref().map(|attachment| DepthAttachment {
        texture: texture(attachment.texture),
        mip: attachment.mip,
        layers: attachment.layer..attachment.layer + 1,
        load: attachment.load,
        store: attachment.store,
    });
}

#[cfg(test)]
mod tests {
    //! T0 tests of a plan's barriers and render passes as the RHI's.

    use clockwork_rhi::test_support::{unissued_buffer, unissued_texture};
    use clockwork_rhi::{AccessSet, LoadOp, StoreOp, Subresource};

    use super::*;
    use crate::render_passes::PlannedAttachment;

    /// A frame's handles: texture 0 an import of one mip and two layers,
    /// texture 1 created for this frame, texture 2 kept from the last, and
    /// buffer 0 created for this frame.
    fn bound() -> Bound {
        let texture = |index, mips, layers, fresh| {
            Some(BoundTexture {
                texture: unissued_texture(index),
                mips,
                layers,
                fresh,
            })
        };
        Bound {
            textures: vec![
                texture(0, 1, 2, false),
                texture(1, 4, 1, true),
                texture(2, 1, 1, false),
            ],
            buffers: vec![Some(BoundBuffer {
                buffer: unissued_buffer(0),
                fresh: true,
            })],
        }
    }

    /// A barrier on `target`, from `before` to `after`.
    fn planned(
        target: BarrierTarget,
        before: impl Into<AccessSet>,
        after: impl Into<AccessSet>,
        from_last_frame: bool,
    ) -> PlannedBarrier {
        PlannedBarrier {
            target,
            before: before.into(),
            after: after.into(),
            from_last_frame,
        }
    }

    /// The RHI barriers `planned` stands for, with [`bound`]'s handles.
    fn converted(planned: &[PlannedBarrier]) -> Vec<Barrier> {
        let mut converted = Vec::new();
        barriers(planned, &bound(), &mut converted);
        converted
    }

    /// A barrier on part of an import is kept while it starts within the
    /// import's mips and layers, whatever its end, and left out once it
    /// starts past them, as mips `1..` of a one-mip import does.
    #[test]
    fn barriers_past_an_imports_shape_are_left_out() {
        let part = |mips, layers| {
            planned(
                BarrierTarget::Texture(0, Subresource { mips, layers }),
                Access::CopyWrite,
                Access::FragmentShaderSampledRead,
                false,
            )
        };
        let remaining = Subresource::REMAINING;
        let kept = [part(0..1, 0..remaining), part(0..remaining, 1..2)];
        let left_out = [part(1..remaining, 0..remaining), part(0..1, 2..remaining)];
        let all: Vec<_> = kept.iter().chain(&left_out).cloned().collect();
        let expected: Vec<_> = kept
            .iter()
            .map(|barrier| {
                let BarrierTarget::Texture(_, subresource) = &barrier.target else {
                    unreachable!("every barrier here is on a texture")
                };
                Barrier::subresource(
                    unissued_texture(0),
                    subresource.clone(),
                    Access::CopyWrite,
                    Access::FragmentShaderSampledRead,
                )
            })
            .collect();
        assert_eq!(converted(&all), expected);
    }

    /// A barrier from the last frame's accesses is from `Access::None` for
    /// a resource created for this frame, and stands for one kept from the
    /// last, and any other barrier keeps its accesses.
    #[test]
    fn a_fresh_resources_first_barrier_is_from_none() {
        let whole = Subresource::ALL;
        let barriers = [
            planned(
                BarrierTarget::Texture(1, whole.clone()),
                Access::CopyRead,
                Access::CopyWrite,
                true,
            ),
            planned(
                BarrierTarget::Texture(2, whole.clone()),
                Access::CopyRead,
                Access::CopyWrite,
                true,
            ),
            planned(
                BarrierTarget::Texture(1, whole.clone()),
                Access::CopyWrite,
                Access::CopyRead,
                false,
            ),
            planned(
                BarrierTarget::Buffer(0),
                Access::ComputeShaderRead,
                Access::ComputeShaderWrite,
                true,
            ),
            planned(
                BarrierTarget::Global,
                Access::CopyWrite,
                Access::ComputeShaderRead,
                false,
            ),
        ];
        assert_eq!(
            converted(&barriers),
            [
                Barrier::subresource(
                    unissued_texture(1),
                    whole.clone(),
                    Access::None,
                    Access::CopyWrite
                ),
                Barrier::subresource(
                    unissued_texture(2),
                    whole.clone(),
                    Access::CopyRead,
                    Access::CopyWrite
                ),
                Barrier::subresource(
                    unissued_texture(1),
                    whole,
                    Access::CopyWrite,
                    Access::CopyRead
                ),
                Barrier::buffer(unissued_buffer(0), Access::None, Access::ComputeShaderWrite),
                Barrier::global(Access::CopyWrite, Access::ComputeShaderRead),
            ]
        );
    }

    /// An attachment of `texture`'s `mip` and `layer`.
    fn attachment<T>(
        texture: usize,
        mip: u32,
        layer: u32,
        load: LoadOp<T>,
    ) -> PlannedAttachment<T> {
        PlannedAttachment {
            texture,
            mip,
            layer,
            load,
            store: StoreOp::Store,
        }
    }

    /// A render pass is filled with each attachment's handle, mip and one
    /// layer, and its load and store, replacing what it held before, in the
    /// same memory.
    #[test]
    fn render_passes_are_filled_in_place() {
        let mut render_pass = RenderPassDesc {
            colour: Vec::new(),
            depth: None,
        };
        let two = PlannedRenderPass {
            colour: vec![
                attachment(1, 2, 0, LoadOp::Clear([0.0, 0.0, 0.0, 1.0])),
                attachment(0, 0, 1, LoadOp::Load),
            ],
            depth: Some(attachment(2, 0, 0, LoadOp::Clear(0.0))),
        };
        fill_render_pass(&two, &bound(), &mut render_pass);
        assert_eq!(
            render_pass,
            RenderPassDesc {
                colour: vec![
                    ColorAttachment {
                        texture: unissued_texture(1),
                        mip: 2,
                        layers: 0..1,
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                    },
                    ColorAttachment {
                        texture: unissued_texture(0),
                        mip: 0,
                        layers: 1..2,
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                ],
                depth: Some(DepthAttachment {
                    texture: unissued_texture(2),
                    mip: 0,
                    layers: 0..1,
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Store,
                }),
            }
        );

        let capacity = render_pass.colour.capacity();
        let one = PlannedRenderPass {
            colour: vec![attachment(2, 0, 0, LoadOp::DontCare)],
            depth: None,
        };
        fill_render_pass(&one, &bound(), &mut render_pass);
        assert_eq!(render_pass.colour.len(), 1);
        assert_eq!(render_pass.colour[0].texture, unissued_texture(2));
        assert_eq!(render_pass.depth, None);
        assert_eq!(render_pass.colour.capacity(), capacity);
    }
}
