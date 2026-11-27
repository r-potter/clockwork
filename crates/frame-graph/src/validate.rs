//! Validation, the first step of compiling (design 0003, "Compiling").
//!
//! It checks a description against the compile inputs and reports every
//! mistake it finds at once, so a graph built in Rust gets the same checks
//! as one loaded from a technique file. What it works out on the way, such
//! as each texture's size and mip count, is kept for the steps after it.
//!
//! The checks run in layers: resources first, then each use on its own,
//! then, once passes repeated for each mip are expanded, what involves
//! several uses' mips, such as whether overlapping uses share a layout, a
//! render pass's attachments and the order of reads and writes. Each later
//! layer sees only the uses that passed the earlier ones, so one mistake is
//! reported once, rather than again by every check that depends on it.

mod attachments;
mod flow;
mod repeat;
mod resources;
mod uses;

use std::ops::Range;

use clockwork_rhi::{AccessSet, Extent3d, Format, Subresource, TextureDimension};

use crate::error::{CompileErrors, Errors};
use crate::expand::{ExpandedPass, expand};
use crate::pass::UseRole;
use crate::{BufferId, CompileInputs, GraphDesc, ResourceUse, TextureId};

/// A texture's shape, worked out from its description and the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TextureShape {
    /// Its shape.
    pub(crate) dimension: TextureDimension,
    /// Its texel format.
    pub(crate) format: Format,
    /// Its size at mip 0.
    pub(crate) extent: Extent3d,
    /// How many mips it has.
    pub(crate) mips: u32,
    /// How many layers it has.
    pub(crate) layers: u32,
}

impl TextureShape {
    /// The width and height of mip `mip`: each halved per mip, rounded
    /// down, and at least 1.
    pub(crate) fn mip_size(&self, mip: u32) -> (u32, u32) {
        let halve = |length: u32| length.checked_shr(mip).unwrap_or(0).max(1);
        (halve(self.extent.width), halve(self.extent.height))
    }
}

/// What validation worked out about a description that has no mistakes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Validated {
    /// Each texture's shape, indexed by `TextureId`, or `None` for an
    /// imported texture, whose shape is the caller's.
    pub(crate) textures: Vec<Option<TextureShape>>,
    /// The passes, with each pass repeated for each mip expanded into its
    /// repetitions.
    pub(crate) passes: Vec<ExpandedPass>,
}

/// What the checks of uses on their own found of a pass.
#[derive(Debug)]
struct PassUses {
    /// Each of its uses, in order, if it has no mistakes. A use of a mip
    /// relative to a repetition is resolved at the first repetition.
    uses: Vec<Option<CheckedUse>>,
    /// The mips its repetitions make, if it's repeated for each mip, and
    /// neither how nor which texture its relative mips name has a mistake.
    mips: Option<Range<u32>>,
}

/// A use that passed the checks of uses on their own.
#[derive(Debug, Clone)]
struct CheckedUse {
    /// What it uses.
    resource: Checked,
    /// How it uses it.
    access: AccessSet,
    /// How it was declared.
    role: UseRole,
}

/// What a checked use uses.
#[derive(Debug, Clone)]
enum Checked {
    /// Part of a texture, with its ranges resolved where its shape is known,
    /// and otherwise ending at [`Subresource::REMAINING`] where they did.
    Texture(TextureId, Subresource),
    /// A buffer.
    Buffer(BufferId),
}

/// Checks `desc` against `inputs`.
///
/// # Errors
///
/// Returns every mistake found.
pub(crate) fn validate(
    desc: &GraphDesc,
    inputs: &CompileInputs,
) -> Result<Validated, CompileErrors> {
    let mut errors = Errors::default();
    resources::check_names(desc, &mut errors);
    let textures = resources::texture_shapes(desc, inputs, &mut errors);
    resources::check_buffers(desc, &mut errors);
    let checked = uses::check_uses(desc, &textures, &mut errors);
    attachments::check_attachments(desc, &textures, &checked, &mut errors);

    // Step 2 (design 0003, "Compiling"): expand, then check what involves
    // several uses' mips on the expanded passes.
    let mips: Vec<_> = checked.iter().map(|pass| pass.mips.clone()).collect();
    let passes = expand(&desc.passes, &mips);
    let mut uses = expanded_uses(desc, &passes, &checked);
    let mut found = Found::new(passes.len());
    for (index, (pass, uses)) in passes.iter().zip(&mut uses).enumerate() {
        let conflicting = uses::check_overlaps(desc, uses, |message| found.push(index, message));
        uses.retain(|used| {
            !matches!(used.resource, Checked::Texture(texture, _) if conflicting.contains(&texture))
        });
        attachments::check_together(desc, &textures, &pass.pass, uses, |message| {
            found.push(index, message);
        });
    }
    flow::check_flow(desc, &textures, &passes, &uses, &mut found);
    found.report(desc, &passes, &mut errors);
    flow::check_output(desc, &mut errors);
    errors.finish(desc.label(), Validated { textures, passes })
}

/// The uses of each of `passes`, expanded from `desc`'s, that passed the
/// checks of uses on their own, as `checked` holds them for the passes as
/// written. A repetition's uses are its pass's, with each relative mip
/// moved to the one the repetition uses.
fn expanded_uses(
    desc: &GraphDesc,
    passes: &[ExpandedPass],
    checked: &[PassUses],
) -> Vec<Vec<CheckedUse>> {
    passes
        .iter()
        .map(|expanded| {
            let written = &desc.passes[expanded.source.index()];
            let uses = written.uses.iter().zip(&expanded.pass.uses);
            uses.zip(&checked[expanded.source.index()].uses)
                .filter_map(|((written, used), checked)| {
                    let mut checked = checked.clone()?;
                    if let (
                        ResourceUse::Repeated(_),
                        ResourceUse::Texture(part),
                        Checked::Texture(_, subresource),
                    ) = (&written.resource, &used.resource, &mut checked.resource)
                    {
                        subresource.mips = part.subresource.mips.clone();
                    }
                    Some(checked)
                })
                .collect()
        })
        .collect()
}

/// The mistakes found in each expanded pass, each as a message that
/// follows the pass's name, kept until they're all found so those of a
/// repeated pass's repetitions can be reported together.
struct Found(Vec<Vec<String>>);

impl Found {
    /// Nothing found yet, in any of `count` expanded passes.
    fn new(count: usize) -> Self {
        Self(vec![Vec::new(); count])
    }

    /// Records the mistake `message` in the expanded pass at `index`.
    fn push(&mut self, index: usize, message: String) {
        self.0[index].push(message);
    }

    /// Reports every mistake found, in the order of `passes`, each with its
    /// pass's name. A mistake every repetition of a pass makes is reported
    /// once, first, with the name of the pass as written, since it doesn't
    /// depend on the mip, and the rest with their repetition's name.
    fn report(self, desc: &GraphDesc, passes: &[ExpandedPass], errors: &mut Errors) {
        let mut found = self.0.into_iter();
        for group in passes.chunk_by(|a, b| a.source == b.source) {
            let messages: Vec<_> = found.by_ref().take(group.len()).collect();
            let mut common = Vec::new();
            if group[0].mip.is_some() {
                for message in &messages[0] {
                    let everywhere = messages.iter().all(|others| others.contains(message));
                    if everywhere && !common.contains(message) {
                        common.push(message.clone());
                    }
                }
            }
            let name = &desc.passes[group[0].source.index()].name;
            for message in &common {
                errors.push(format!("pass \"{name}\" {message}"));
            }
            for (pass, messages) in group.iter().zip(messages) {
                for message in messages.iter().filter(|message| !common.contains(message)) {
                    errors.push(format!("pass \"{}\" {message}", pass.pass.name));
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Helpers for validation's tests.

    use clockwork_rhi::{Access, Extent2d, Format, QueueId};

    use super::*;
    use crate::{OutputDesc, QueueLayout, Schedule};

    /// The name of the pass [`finished`] adds.
    pub(crate) const OUTPUT_WRITER: &str = "writes the output";

    /// Inputs for a 1920 × 1080 BGRA8 sRGB surface, on one queue.
    pub(crate) fn inputs() -> CompileInputs {
        CompileInputs {
            output: OutputDesc::surface(
                Format::Bgra8Srgb,
                Extent2d {
                    width: 1920,
                    height: 1080,
                },
            ),
            queues: QueueLayout::shared(QueueId::new(0)),
            schedule: Schedule::AsWritten,
        }
    }

    /// `desc` with a last pass that writes the output, as a valid graph
    /// must, so that tests of other rules needn't.
    pub(crate) fn finished(desc: &GraphDesc) -> GraphDesc {
        let mut desc = desc.clone();
        let output = desc.output();
        desc.transfer_pass(OUTPUT_WRITER, "copy")
            .write(output, Access::CopyWrite);
        desc
    }

    /// The messages of the errors validating `desc`, [`finished`], with
    /// [`inputs`] reports, or none if it's valid.
    pub(crate) fn messages(desc: &GraphDesc) -> Vec<String> {
        match validate(&finished(desc), &inputs()) {
            Ok(_) => Vec::new(),
            Err(errors) => errors
                .errors()
                .iter()
                .map(|error| error.message().to_owned())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of checking the expanded passes.

    use clockwork_rhi::Access;

    use super::test_support::{OUTPUT_WRITER, finished, inputs, messages};
    use super::*;
    use crate::{
        Attachment, ComputeRecorder, MipCount, MipDirection, TextureSize, TransientTexture,
    };

    /// A graph with a 256 × 256 texture of four mips, "bloom", and a pass
    /// that writes its mip 0.
    fn graph() -> (GraphDesc, TextureId) {
        let mut graph = GraphDesc::new("expanded");
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(256, 256)),
            )
        };
        let bloom = graph.transient_texture("bloom", chain);
        graph
            .transfer_pass("fill", "fill")
            .write(bloom.mip(0), Access::CopyWrite);
        (graph, bloom)
    }

    /// A valid graph's passes are its passes expanded, each repetition
    /// named with its mip.
    #[test]
    fn a_valid_graph_keeps_its_expanded_passes() {
        let (mut graph, bloom) = graph();
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);

        let validated = validate(&finished(&graph), &inputs()).unwrap();
        let names: Vec<_> = validated
            .passes
            .iter()
            .map(|pass| pass.pass.name.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "fill",
                "downsample (mip 1)",
                "downsample (mip 2)",
                "downsample (mip 3)",
                OUTPUT_WRITER,
            ]
        );
    }

    /// Overlapping uses, and reads of what no earlier pass wrote, are
    /// checked at each repetition, and a mistake at only some is reported
    /// with each one's name.
    #[test]
    fn each_repetition_is_checked() {
        let (mut graph, bloom) = graph();
        // Mip 2 is sampled by every repetition, which conflicts with the
        // second writing it, and is read by the first before it's written.
        graph
            .compute_pass("overlap", ComputeRecorder::dispatch("overlap"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite)
            .bind(bloom.mip(2), Access::ComputeShaderSampledRead);
        assert_eq!(
            messages(&graph),
            [
                "pass \"overlap (mip 1)\" reads mip 2 of \"bloom\", which no earlier pass writes",
                "pass \"overlap (mip 2)\" uses overlapping parts of \"bloom\" with \
                 ComputeShaderSampledRead | ComputeShaderWrite, which share no layout",
            ]
        );
    }

    /// A render pass's attachments are checked for size at each
    /// repetition, since the current mip's size changes from one to the
    /// next.
    #[test]
    fn attachment_sizes_are_checked_at_each_repetition() {
        let (mut graph, bloom) = graph();
        let small = graph.transient_texture(
            "small",
            TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(64, 64)),
            ),
        );
        graph
            .fullscreen_pass("blur", "blur")
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::FragmentShaderSampledRead)
            .colour(bloom.current_mip().attachment(0), Attachment::Discard)
            .colour(small, Attachment::Discard);
        assert_eq!(
            messages(&graph),
            [
                "pass \"blur (mip 1)\" renders to \"bloom\" at 128 × 128 and \"small\" at 64 × \
                 64, but a render pass's attachments are the same size",
                "pass \"blur (mip 3)\" renders to \"bloom\" at 32 × 32 and \"small\" at 64 × 64, \
                 but a render pass's attachments are the same size",
            ]
        );
    }

    /// A pass with a relative mip of another texture, or of one the graph
    /// doesn't have, is reported and kept as written, even where the
    /// repeated texture's range gives that mip no previous one.
    #[test]
    fn a_relative_mip_of_another_texture_isnt_expanded() {
        let (mut graph, bloom) = graph();
        let hdr = graph.transient_texture(
            "hdr",
            TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(256, 256)),
            ),
        );
        for (name, other) in [("other", hdr), ("missing", TextureId(9))] {
            graph
                .compute_pass(name, ComputeRecorder::dispatch(name))
                .for_each_mip(bloom, MipDirection::Down, 0..)
                .bind(other.previous_mip(), Access::ComputeShaderSampledRead)
                .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        }
        assert_eq!(
            messages(&graph),
            [
                "pass \"other\" uses the previous mip of \"hdr\", but is repeated for each mip of \
                 \"bloom\"",
                "pass \"missing\" uses texture 9, which this graph doesn't have",
            ]
        );
    }

    /// A mistake every repetition makes, such as overlapping fixed uses or
    /// a read of what's never written, is reported once, first, with the
    /// name of the pass as written.
    #[test]
    fn a_mistake_every_repetition_makes_is_reported_once() {
        let (mut graph, bloom) = graph();
        let lights = graph.transient_buffer("lights", 64);
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite)
            .bind(lights, Access::ComputeShaderRead)
            .bind(bloom.mip(0), Access::ComputeShaderSampledRead)
            .bind(bloom.mip(0), Access::ComputeShaderRead);
        assert_eq!(
            messages(&graph),
            [
                "pass \"downsample\" uses overlapping parts of \"bloom\" with ComputeShaderRead | \
                 ComputeShaderSampledRead, which share no layout",
                "pass \"downsample\" reads \"lights\", which no earlier pass writes",
            ]
        );
    }
}
