//! Validation of the order of reads and writes (design 0003, "Compiling",
//! step 2): on the expanded passes, every read of a transient resource
//! follows, in the written order, a pass that writes what it reads, so a
//! repetition's read of the previous mip follows the repetition that wrote
//! it, and some pass writes the output.

use std::ops::Range;

use clockwork_rhi::Subresource;

use super::uses::describe_range;
use super::{Checked, CheckedUse, Found, TextureShape};
use crate::desc::{BufferKind, TextureKind};
use crate::error::Errors;
use crate::expand::ExpandedPass;
use crate::pass::{Attachment, UseRole};
use crate::{GraphDesc, ResourceUse};

/// Records in `found` each read of a transient resource's contents that no
/// earlier pass of `passes`, expanded, wrote. `checked` holds each expanded
/// pass's uses that passed the checks before these, and only those reads
/// are checked here. Every write counts, even one with a mistake of its
/// own, since that's reported already, and what it writes shouldn't be
/// reported again as unwritten.
pub(super) fn check_flow(
    desc: &GraphDesc,
    shapes: &[Option<TextureShape>],
    passes: &[ExpandedPass],
    checked: &[Vec<CheckedUse>],
    found: &mut Found,
) {
    let mut written = Written::new(desc, shapes);
    for (index, (expanded, uses)) in passes.iter().zip(checked).enumerate() {
        // A pass reads what earlier passes wrote, so its own writes count
        // only from the next pass on.
        for used in uses.iter().filter(|used| reads_contents(used)) {
            if let Some(name) = written.unwritten(desc, &used.resource) {
                found.push(index, format!("reads {name}, which no earlier pass writes"));
            }
        }
        for used in expanded
            .pass
            .uses
            .iter()
            .filter(|used| used.access.is_write())
        {
            written.mark(&used.resource);
        }
    }
}

/// Reports a graph that never writes its output.
pub(super) fn check_output(desc: &GraphDesc, errors: &mut Errors) {
    let output = desc.output();
    let output_written = desc.passes.iter().flat_map(|pass| &pass.uses).any(|used| {
        used.access.is_write()
            && matches!(&used.resource, ResourceUse::Texture(part) if part.texture == output)
    });
    if !output_written {
        errors.push("no pass writes the output".to_owned());
    }
}

/// Whether `used` reads the resource's contents from before the pass: a
/// read, a write that also reads, or an attachment that keeps its
/// contents. An attachment that's cleared or discarded doesn't, though its
/// accesses include depth testing or blending, since it starts afresh.
fn reads_contents(used: &CheckedUse) -> bool {
    match used.role {
        UseRole::Read => true,
        UseRole::Write => used.access.iter().any(|access| !access.is_write()),
        UseRole::Colour(attachment) => matches!(attachment, Attachment::Keep),
        UseRole::Depth(attachment) => matches!(attachment, Attachment::Keep),
    }
}

/// Which parts of each transient resource the passes so far have written.
/// Other resources have contents from before the frame, so they're never
/// tracked.
struct Written {
    /// For each texture, `None` if it isn't transient, and otherwise
    /// whether each mip of each layer is written, indexed by
    /// `mip * layers + layer`, with its layer count.
    textures: Vec<Option<(Vec<bool>, u32)>>,
    /// For each buffer, `None` if it isn't transient, and otherwise whether
    /// it's written.
    buffers: Vec<Option<bool>>,
}

impl Written {
    /// Nothing written yet, for `desc`'s transient resources, whose shapes
    /// are `shapes`.
    fn new(desc: &GraphDesc, shapes: &[Option<TextureShape>]) -> Self {
        let textures = desc
            .textures
            .iter()
            .zip(shapes)
            .map(|(texture, shape)| match (&texture.kind, shape) {
                (TextureKind::Transient(_), Some(shape)) => {
                    let cells = (shape.mips * shape.layers) as usize;
                    Some((vec![false; cells], shape.layers))
                }
                _ => None,
            })
            .collect();
        let buffers = desc
            .buffers
            .iter()
            .map(|buffer| matches!(buffer.kind, BufferKind::Transient { .. }).then_some(false))
            .collect();
        Self { textures, buffers }
    }

    /// The cells of `subresource`, in a texture with `layers` layers.
    fn cells(subresource: &Subresource, layers: u32) -> impl Iterator<Item = usize> + '_ {
        subresource.mips.clone().flat_map(move |mip| {
            subresource
                .layers
                .clone()
                .map(move |layer| (mip * layers + layer) as usize)
        })
    }

    /// How a read of `resource` is named in a message, if part of what it
    /// reads is transient and unwritten: the resource's name, and the
    /// part of a texture read where that's less than all of it.
    fn unwritten(&self, desc: &GraphDesc, resource: &Checked) -> Option<String> {
        match resource {
            Checked::Texture(texture, part) => {
                let (cells, layers) = self.textures[texture.index()].as_ref()?;
                if Self::cells(part, *layers).all(|cell| cells[cell]) {
                    return None;
                }
                let name = &desc.textures[texture.index()].name;
                let mips =
                    u32::try_from(cells.len()).expect("a texture's cells fit a u32") / layers;
                let mut read = Vec::new();
                if part.mips != (0..mips) {
                    read.push(describe_range("mip", &part.mips));
                }
                if part.layers != (0..*layers) {
                    read.push(describe_range("layer", &part.layers));
                }
                read.push(format!("\"{name}\""));
                Some(read.join(" of "))
            }
            Checked::Buffer(buffer) => {
                let written = self.buffers[buffer.index()]?;
                (!written).then(|| format!("\"{}\"", desc.buffers[buffer.index()].name))
            }
        }
    }

    /// Records that `resource` is written. The use needn't have passed its
    /// checks: one naming a resource the graph doesn't have is ignored, and
    /// its mips and layers are limited to those the texture has.
    fn mark(&mut self, resource: &ResourceUse) {
        let (texture, part) = match resource {
            ResourceUse::Texture(part) => (part.texture, part.subresource.clone()),
            // Expanding a pass resolves its relative mips, so one is left
            // only in a pass that isn't repeated, or whose repeat has a
            // mistake. Either is reported already, so the write counts as
            // writing every mip it could reach, rather than reads of them
            // being reported again as unwritten.
            ResourceUse::Repeated(part) => (
                part.texture,
                Subresource {
                    mips: Subresource::ALL.mips,
                    layers: part.layers.clone(),
                },
            ),
            ResourceUse::Buffer(buffer) => {
                if let Some(Some(written)) = self.buffers.get_mut(buffer.index()) {
                    *written = true;
                }
                return;
            }
        };
        if let Some(Some((cells, layers))) = self.textures.get_mut(texture.index()) {
            let mips = u32::try_from(cells.len()).expect("a texture's cells fit a u32") / *layers;
            let limit =
                |range: &Range<u32>, count: u32| range.start.min(count)..range.end.min(count);
            let within = Subresource {
                mips: limit(&part.mips, mips),
                layers: limit(&part.layers, *layers),
            };
            for cell in Self::cells(&within, *layers) {
                cells[cell] = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of validating the order of reads and writes.

    use clockwork_rhi::{Access, Extent3d, Format};

    use super::*;
    use crate::validate::test_support::{inputs, messages};
    use crate::validate::validate;
    use crate::{
        BufferId, CompileError, ComputeRecorder, ImportDesc, MipCount, MipDirection, TextureId,
        TextureSize, TransientTexture,
    };

    /// The resources of [`graph`].
    struct Resources {
        /// A transient texture.
        gbuffer: TextureId,
        /// A transient texture with four mips.
        bloom: TextureId,
        /// A persistent texture.
        history: TextureId,
        /// An imported texture.
        font: TextureId,
        /// A transient buffer.
        lights: BufferId,
    }

    /// A graph with the resources it returns, and no passes.
    fn graph() -> (GraphDesc, Resources) {
        let mut graph = GraphDesc::new("flow");
        let size = TextureSize::Output { scale: 1.0 };
        let texture = TransientTexture::new(Format::Rgba16Float, size);
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(256, 256)),
            )
        };
        let import = ImportDesc {
            before: Access::CopyWrite.into(),
            after: Access::CopyWrite.into(),
        };
        let resources = Resources {
            gbuffer: graph.transient_texture("gbuffer", texture),
            bloom: graph.transient_texture("bloom", chain),
            history: graph.persistent_texture("history", texture),
            font: graph.import_texture("font", import),
            lights: graph.transient_buffer("lights", 64),
        };
        (graph, resources)
    }

    /// Reads of what earlier passes wrote, and of persistent and imported
    /// resources' older contents, report nothing.
    #[test]
    fn reads_after_writes_and_of_older_contents_have_no_errors() {
        let (mut graph, resources) = graph();
        let Resources {
            gbuffer,
            bloom,
            history,
            font,
            lights,
        } = resources;
        graph
            .transfer_pass("fill", "fill")
            .write(gbuffer, Access::CopyWrite)
            .write(bloom.mip(0), Access::CopyWrite)
            .write(lights, Access::CopyWrite);
        graph
            .compute_pass("downsample", ComputeRecorder::recorder("downsample"))
            .read(bloom.mip(0), Access::ComputeShaderSampledRead)
            .write(bloom.mip(1), Access::ComputeShaderWrite);
        graph
            .compute_pass("light", ComputeRecorder::recorder("light"))
            .read(gbuffer, Access::ComputeShaderSampledRead)
            .read(bloom.mip(1), Access::ComputeShaderSampledRead)
            .read(history, Access::ComputeShaderSampledRead)
            .read(font, Access::ComputeShaderSampledRead)
            .write(
                lights,
                Access::ComputeShaderRead | Access::ComputeShaderWrite,
            );
        assert_eq!(messages(&graph), Vec::<String>::new());
    }

    /// Reads of transient contents no earlier pass wrote are reported, per
    /// mip, and a pass's own writes don't count for its reads.
    #[test]
    fn reads_of_unwritten_transients_are_reported() {
        let (mut graph, resources) = graph();
        let Resources {
            gbuffer,
            bloom,
            lights,
            ..
        } = resources;
        graph
            .compute_pass("lighting", ComputeRecorder::recorder("lighting"))
            .read(gbuffer, Access::ComputeShaderSampledRead)
            .write(
                lights,
                Access::ComputeShaderRead | Access::ComputeShaderWrite,
            );
        graph
            .compute_pass("downsample", ComputeRecorder::recorder("downsample"))
            .write(bloom.mip(0), Access::ComputeShaderWrite)
            .read(bloom.mip(0), Access::ComputeShaderRead);
        graph
            .compute_pass("blur", ComputeRecorder::recorder("blur"))
            .read(bloom.mip(1), Access::ComputeShaderRead);
        assert_eq!(
            messages(&graph),
            [
                "pass \"lighting\" reads \"gbuffer\", which no earlier pass writes",
                "pass \"lighting\" reads \"lights\", which no earlier pass writes",
                "pass \"downsample\" reads mip 0 of \"bloom\", which no earlier pass writes",
                "pass \"blur\" reads mip 1 of \"bloom\", which no earlier pass writes",
            ]
        );
    }

    /// A kept attachment reads its contents, and a cleared one doesn't.
    #[test]
    fn a_kept_attachment_reads_and_a_cleared_one_does_not() {
        let (mut graph, Resources { gbuffer, .. }) = graph();
        graph
            .raster_pass("cleared", "cleared")
            .colour(gbuffer, Attachment::Clear([0.0; 4]));
        graph
            .raster_pass("kept", "kept")
            .colour(gbuffer, Attachment::Keep);
        assert_eq!(messages(&graph), Vec::<String>::new());

        let (mut unwritten, _) = self::graph();
        unwritten
            .raster_pass("kept", "kept")
            .colour(gbuffer, Attachment::Keep);
        assert_eq!(
            messages(&unwritten),
            ["pass \"kept\" reads \"gbuffer\", which no earlier pass writes"]
        );
    }

    /// Each repetition writes only its own mip, and reads the previous one
    /// after the repetition before wrote it, so a read of a mip past the
    /// range is reported.
    #[test]
    fn each_repetition_writes_its_own_mip() {
        let (mut graph, Resources { bloom, .. }) = graph();
        graph
            .transfer_pass("fill", "fill")
            .write(bloom.mip(0), Access::CopyWrite);
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..3)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        graph
            .compute_pass("composite", ComputeRecorder::dispatch("composite"))
            .bind(bloom.mip(2), Access::ComputeShaderSampledRead)
            .bind(bloom.mip(3), Access::ComputeShaderSampledRead);
        assert_eq!(
            messages(&graph),
            ["pass \"composite\" reads mip 3 of \"bloom\", which no earlier pass writes"]
        );
    }

    /// A relative write in a pass whose repeat has a mistake, which is
    /// reported, counts as writing every mip, so later reads of the chain
    /// aren't reported as well.
    #[test]
    fn a_mistaken_repeats_write_counts_as_writing_every_mip() {
        let (mut graph, Resources { bloom, .. }) = graph();
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, 9..)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        graph
            .compute_pass("composite", ComputeRecorder::dispatch("composite"))
            .bind(bloom, Access::ComputeShaderSampledRead);
        assert_eq!(
            messages(&graph),
            ["pass \"downsample\" is repeated from mip 9 of \"bloom\", whose last mip is 3"]
        );
    }

    /// A fullscreen pass's bound read is ordered as any other read is, so
    /// one of a transient texture no earlier pass writes is reported.
    #[test]
    fn bound_reads_must_follow_a_write() {
        let (mut graph, Resources { gbuffer, .. }) = graph();
        let output = graph.output();
        graph
            .fullscreen_pass("lighting", "lighting")
            .bind(gbuffer, Access::FragmentShaderSampledRead)
            .colour(output, Attachment::Discard);
        assert_eq!(
            messages(&graph),
            ["pass \"lighting\" reads \"gbuffer\", which no earlier pass writes"]
        );
    }

    /// A write reported for another mistake still counts as writing, so
    /// later reads of it aren't reported as well.
    #[test]
    fn a_write_with_a_mistake_still_writes() {
        let (mut graph, Resources { gbuffer, .. }) = graph();
        let output = graph.output();
        graph
            .transfer_pass("fill", "fill")
            .write(gbuffer, Access::CopyRead | Access::CopyWrite);
        graph
            .compute_pass("light", ComputeRecorder::recorder("light"))
            .read(gbuffer, Access::ComputeShaderSampledRead);
        graph
            .transfer_pass("present", "present")
            .read(output, Access::CopyWrite);
        let errors = validate(&graph, &inputs()).unwrap_err();
        let messages: Vec<_> = errors.errors().iter().map(CompileError::message).collect();
        assert_eq!(
            messages,
            [
                "pass \"fill\" uses \"gbuffer\" with CopyRead | CopyWrite, which share no layout",
                "pass \"present\" reads \"output\" with CopyWrite, which writes, so it's declared \
                 with `write`",
            ]
        );
    }

    /// A graph that never writes its output is reported.
    #[test]
    fn the_output_must_be_written() {
        let (graph, _) = graph();
        let errors = validate(&graph, &inputs()).unwrap_err();
        let messages: Vec<_> = errors.errors().iter().map(CompileError::message).collect();
        assert_eq!(messages, ["no pass writes the output"]);
    }
}
