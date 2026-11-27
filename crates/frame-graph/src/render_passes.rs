//! Load and store operations, the tenth step of compiling (design 0003,
//! "Compiling"): the render pass each scheduled pass with attachments
//! begins, with what each attachment holds when it begins and what happens
//! to what's rendered when it ends.
//!
//! An attachment's load operation follows from how it's declared: a clear
//! if the pass asks for one, a load if it keeps the contents, and
//! otherwise "don't care". Its store operation is a store if a pass that
//! runs later in the frame reads what it renders, before a cleared or
//! discarded attachment replaces it, or if it outlasts the frame, as the
//! output, a persistent texture or an import does. Otherwise it's "don't
//! care", which lets a tiling GPU skip writing it out.

use std::ops::Range;

use clockwork_rhi::{LoadOp, StoreOp};

use crate::desc::TextureKind;
use crate::expand::ExpandedPass;
use crate::pass::{Use, UseRole};
use crate::schedule::Scheduled;
use crate::{Attachment, GraphDesc, ResourceUse, TextureUse};

/// One attachment of a planned render pass: a mip of a layer of a texture.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlannedAttachment<T> {
    /// The texture, by its index, as its `TextureId` gives.
    pub(crate) texture: usize,
    /// The mip rendered to.
    pub(crate) mip: u32,
    /// The layer rendered to.
    pub(crate) layer: u32,
    /// What it holds when the render pass begins.
    pub(crate) load: LoadOp<T>,
    /// What happens to what's rendered when the render pass ends.
    pub(crate) store: StoreOp,
}

/// The render pass a pass begins: its attachments, in the order they're
/// declared.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlannedRenderPass {
    /// The colour attachments, by number.
    pub(crate) colour: Vec<PlannedAttachment<[f32; 4]>>,
    /// The depth attachment, if there's one.
    pub(crate) depth: Option<PlannedAttachment<f32>>,
}

/// For each pass `scheduled` runs, in the order it runs them, the render
/// pass it begins, or `None` if it has no attachments. `passes` are the
/// expanded passes of `desc`.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "compiling, in a later pull request, calls it")
)]
pub(crate) fn render_passes(
    desc: &GraphDesc,
    passes: &[ExpandedPass],
    scheduled: &Scheduled,
) -> Vec<Option<PlannedRenderPass>> {
    scheduled
        .passes
        .iter()
        .enumerate()
        .map(|(place, &pass)| {
            let later = &scheduled.passes[place + 1..];
            let store = |target: &TextureUse| store(desc, passes, later, target);
            let mut render_pass = PlannedRenderPass {
                colour: Vec::new(),
                depth: None,
            };
            for used in &passes[pass].pass.uses {
                let ResourceUse::Texture(target) = &used.resource else {
                    continue;
                };
                match used.role {
                    UseRole::Colour(attachment) => render_pass.colour.push(PlannedAttachment {
                        load: load(attachment),
                        store: store(target),
                        ..attached(target)
                    }),
                    UseRole::Depth(attachment) => {
                        render_pass.depth = Some(PlannedAttachment {
                            load: load(attachment),
                            store: store(target),
                            ..attached(target)
                        });
                    }
                    UseRole::Read | UseRole::Write => {}
                }
            }
            let empty = render_pass.colour.is_empty() && render_pass.depth.is_none();
            (!empty).then_some(render_pass)
        })
        .collect()
}

/// The attachment `target` names, loaded and stored as "don't care", for
/// its caller to fill in.
fn attached<T>(target: &TextureUse) -> PlannedAttachment<T> {
    PlannedAttachment {
        texture: target.texture.index(),
        mip: target.subresource.mips.start,
        layer: target.subresource.layers.start,
        load: LoadOp::DontCare,
        store: StoreOp::DontCare,
    }
}

/// The load operation of an attachment that starts as `attachment` says.
fn load<T>(attachment: Attachment<T>) -> LoadOp<T> {
    match attachment {
        Attachment::Clear(value) => LoadOp::Clear(value),
        Attachment::Keep => LoadOp::Load,
        Attachment::Discard => LoadOp::DontCare,
    }
}

/// The store operation of an attachment to `target`, one mip of one layer
/// of a texture of `desc`, given the passes that run after it, `later`, of
/// `passes`: a store if one of them reads what it renders before a use
/// replaces it, or if it outlasts the frame.
///
/// A use that writes part of it, such as a copy, leaves the rest of what
/// it renders, so the passes after that use still count.
fn store(
    desc: &GraphDesc,
    passes: &[ExpandedPass],
    later: &[usize],
    target: &TextureUse,
) -> StoreOp {
    let overlap = |a: &Range<u32>, b: &Range<u32>| a.start < b.end && b.start < a.end;
    let covers = |used: &Use| match &used.resource {
        ResourceUse::Texture(part) => {
            part.texture == target.texture
                && overlap(&part.subresource.mips, &target.subresource.mips)
                && overlap(&part.subresource.layers, &target.subresource.layers)
        }
        ResourceUse::Buffer(_) | ResourceUse::Repeated(_) => false,
    };
    for &pass in later {
        let uses = passes[pass].pass.uses.iter().filter(|used| covers(used));
        let (mut reads, mut replaces) = (false, false);
        for used in uses {
            reads |= used.role.reads_contents(used.access);
            replaces |= used.role.replaces_contents();
        }
        // A pass's reads see what came before its writes.
        if reads {
            return StoreOp::Store;
        }
        if replaces {
            return StoreOp::DontCare;
        }
    }
    match desc.textures[target.texture.index()].kind {
        TextureKind::Output | TextureKind::Persistent(_) | TextureKind::Imported(_) => {
            StoreOp::Store
        }
        TextureKind::Transient(_) => StoreOp::DontCare,
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of load and store operations.

    use clockwork_rhi::{Access, Extent3d, Format};

    use super::*;
    use crate::cull::cull;
    use crate::dependencies::dependencies;
    use crate::schedule::schedule;
    use crate::validate::test_support::{finished, inputs};
    use crate::validate::validate;
    use crate::{ComputeRecorder, ImportDesc, MipCount, Schedule, TextureSize, TransientTexture};

    /// The render passes of `desc`, finished, compiled as written, one line
    /// per pass that begins one, naming each attachment's texture, with
    /// its load and store operations.
    fn planned(desc: &GraphDesc) -> Vec<String> {
        let desc = finished(desc);
        let validated = validate(&desc, &inputs()).expect("the graph is valid");
        let passes = &validated.passes;
        let mut dependencies = dependencies(&desc, passes);
        let kept = cull(&desc, passes, &dependencies);
        dependencies.reduce(&kept);
        let scheduled = schedule(Schedule::AsWritten, &kept);
        let render_passes = render_passes(&desc, passes, &scheduled);

        /// One attachment, of `kind`, as the test lists it.
        fn line<T: std::fmt::Debug>(
            desc: &GraphDesc,
            kind: &str,
            attachment: &PlannedAttachment<T>,
        ) -> String {
            format!(
                "{kind} {}: {:?}, {:?}",
                desc.textures[attachment.texture].name, attachment.load, attachment.store
            )
        }
        scheduled
            .passes
            .iter()
            .zip(&render_passes)
            .filter_map(|(&pass, render_pass)| {
                let render_pass = render_pass.as_ref()?;
                let mut attachments: Vec<_> = render_pass
                    .colour
                    .iter()
                    .map(|colour| line(&desc, "colour", colour))
                    .collect();
                attachments
                    .extend((render_pass.depth.iter()).map(|depth| line(&desc, "depth", depth)));
                Some(format!(
                    "pass \"{}\": {}",
                    passes[pass].pass.name,
                    attachments.join("; ")
                ))
            })
            .collect()
    }

    /// A 256 × 256 texture of `format`.
    fn texture(format: Format) -> TransientTexture {
        TransientTexture::new(format, TextureSize::Fixed(Extent3d::new_2d(256, 256)))
    }

    /// Loads follow how each attachment starts, and an attachment is stored
    /// if a later pass reads it, by sampling it or keeping its contents,
    /// and not otherwise. The output is stored, though no later pass reads
    /// it.
    #[test]
    fn loads_follow_how_attachments_start() {
        let mut graph = GraphDesc::new("loads");
        let scene = graph.transient_texture("scene", texture(Format::Rgba16Float));
        let normals = graph.transient_texture("normals", texture(Format::Rgba16Float));
        let depth = graph.transient_texture("depth", texture(Format::Depth32Float));
        let output = graph.output();
        graph
            .raster_pass("draw", "draw")
            .colour(scene, Attachment::Clear([0.0, 0.0, 0.0, 1.0]))
            .colour(normals, Attachment::Discard)
            .depth(depth, Attachment::Clear(0.0));
        graph
            .raster_pass("decals", "decals")
            .colour(scene, Attachment::Keep)
            .depth(depth, Attachment::Keep);
        graph
            .raster_pass("tone map", "tone map")
            .read(scene, Access::FragmentShaderSampledRead)
            .read(normals, Access::FragmentShaderSampledRead)
            .colour(output, Attachment::Discard);

        assert_eq!(
            planned(&graph),
            [
                "pass \"draw\": colour scene: Clear([0.0, 0.0, 0.0, 1.0]), Store; \
                 colour normals: DontCare, Store; depth depth: Clear(0.0), Store",
                "pass \"decals\": colour scene: Load, Store; depth depth: Load, DontCare",
                "pass \"tone map\": colour output: DontCare, Store",
            ]
        );
    }

    /// An attachment that a later attachment clears before any pass reads
    /// it isn't stored, and one whose contents a later copy writes part of
    /// is, if a pass reads them after the copy.
    #[test]
    fn stores_look_past_partial_writes_to_the_next_read() {
        let mut graph = GraphDesc::new("stores");
        let hdr = graph.transient_texture("hdr", texture(Format::Rgba16Float));
        graph
            .raster_pass("first", "first")
            .colour(hdr, Attachment::Clear([0.0; 4]))
            .side_effects();
        graph
            .raster_pass("second", "second")
            .colour(hdr, Attachment::Clear([0.0; 4]));
        graph
            .transfer_pass("patch", "patch")
            .write(hdr, Access::CopyWrite);
        graph
            .compute_pass("measure", ComputeRecorder::recorder("measure"))
            .read(hdr, Access::ComputeShaderSampledRead)
            .side_effects();

        let clear = "Clear([0.0, 0.0, 0.0, 0.0])";
        assert_eq!(
            planned(&graph),
            [
                format!("pass \"first\": colour hdr: {clear}, DontCare"),
                format!("pass \"second\": colour hdr: {clear}, Store"),
            ]
        );
    }

    /// An attachment to one mip isn't stored because a later pass reads
    /// another mip of its texture.
    #[test]
    fn reads_of_other_mips_dont_store_an_attachment() {
        let mut graph = GraphDesc::new("other mips");
        let chain = TransientTexture {
            mips: MipCount::Count(2),
            ..texture(Format::Rgba16Float)
        };
        let bloom = graph.transient_texture("bloom", chain);
        graph
            .transfer_pass("fill", "fill")
            .write(bloom.mip(1), Access::CopyWrite);
        graph
            .raster_pass("draw", "draw")
            .colour(bloom.attachment(0, 0), Attachment::Discard)
            .side_effects();
        graph
            .compute_pass("measure", ComputeRecorder::recorder("measure"))
            .read(bloom.mip(1), Access::ComputeShaderSampledRead)
            .side_effects();

        assert_eq!(
            planned(&graph),
            ["pass \"draw\": colour bloom: DontCare, DontCare"]
        );
    }

    /// Attachments to persistent and imported textures are stored, though
    /// no later pass reads them, since they outlast the frame, and one to a
    /// transient texture nothing reads isn't.
    #[test]
    fn attachments_that_outlast_the_frame_are_stored() {
        let mut graph = GraphDesc::new("outlast");
        let history = graph.persistent_texture("history", texture(Format::Rgba16Float));
        let target = graph.import_texture(
            "target",
            ImportDesc {
                before: Access::None.into(),
                after: Access::FragmentShaderSampledRead.into(),
            },
        );
        let scratch = graph.transient_texture("scratch", texture(Format::Rgba16Float));
        for (name, texture) in [("keep", history), ("export", target), ("scratch", scratch)] {
            graph
                .raster_pass(name, name)
                .colour(texture, Attachment::Discard)
                .side_effects();
        }

        assert_eq!(
            planned(&graph),
            [
                "pass \"keep\": colour history: DontCare, Store",
                "pass \"export\": colour target: DontCare, Store",
                "pass \"scratch\": colour scratch: DontCare, DontCare",
            ]
        );
    }
}
