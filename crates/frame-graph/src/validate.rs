//! Validation, the first step of compiling (design 0003, "Compiling").
//!
//! It checks a description against the compile inputs and reports every
//! mistake it finds at once, so a graph built in Rust gets the same checks
//! as one loaded from a technique file. What it works out on the way, such
//! as each texture's size and mip count, is kept for the steps after it.

mod resources;
mod uses;

use clockwork_rhi::{Extent3d, Format, TextureDimension};

use crate::error::{CompileErrors, Errors};
use crate::{CompileInputs, GraphDesc};

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

/// What validation worked out about a description that has no mistakes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Validated {
    /// Each texture's shape, indexed by `TextureId`, or `None` for an
    /// imported texture, whose shape is the caller's.
    pub(crate) textures: Vec<Option<TextureShape>>,
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
    uses::check_uses(desc, &textures, &mut errors);
    errors.finish(desc.label(), Validated { textures })
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Helpers for validation's tests.

    use clockwork_rhi::{Extent2d, Format, QueueId};

    use super::*;
    use crate::{OutputDesc, QueueLayout, Schedule};

    /// Inputs for a 1920 × 1080 RGBA8 surface, on one queue.
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

    /// The messages of the errors validating `desc` with [`inputs`]
    /// reports, or none if it's valid.
    pub(crate) fn messages(desc: &GraphDesc) -> Vec<String> {
        match validate(desc, &inputs()) {
            Ok(_) => Vec::new(),
            Err(errors) => errors
                .errors()
                .iter()
                .map(|error| error.message().to_owned())
                .collect(),
        }
    }
}
