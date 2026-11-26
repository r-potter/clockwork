//! Validation, the first step of compiling (design 0003, "Compiling").
//!
//! It checks a description against the compile inputs and reports every
//! mistake it finds at once, so a graph built in Rust gets the same checks
//! as one loaded from a technique file. What it works out on the way, such
//! as each texture's size and mip count, is kept for the steps after it.
//!
//! The checks run in layers: resources first, then each use on its own,
//! then what involves several uses, such as a render pass's attachments
//! and the order of reads and writes. Each later layer sees only the uses
//! that passed the earlier ones, so one mistake is reported once, rather
//! than again by every check that depends on it.

mod attachments;
mod flow;
mod repeat;
mod resources;
mod uses;

use clockwork_rhi::{AccessSet, Extent3d, Format, Subresource, TextureDimension};

use crate::error::{CompileErrors, Errors};
use crate::pass::UseRole;
use crate::{BufferId, CompileInputs, GraphDesc, TextureId};

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Validated {
    /// Each texture's shape, indexed by `TextureId`, or `None` for an
    /// imported texture, whose shape is the caller's.
    pub(crate) textures: Vec<Option<TextureShape>>,
}

/// A use that passed the checks of uses on their own.
#[derive(Debug)]
struct CheckedUse {
    /// What it uses.
    resource: Checked,
    /// How it uses it.
    access: AccessSet,
    /// How it was declared.
    role: UseRole,
}

/// What a checked use uses.
#[derive(Debug)]
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
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "compiling, in a later pull request, calls it")
)]
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
    flow::check_flow(desc, &textures, &checked, &mut errors);
    errors.finish(desc.label(), Validated { textures })
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
