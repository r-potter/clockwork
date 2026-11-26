//! Expansion, the second step of compiling (design 0003, "Compiling"): each
//! pass repeated for each mip of a texture becomes a pass per mip, named
//! with its mip, whose uses name that mip and the one the repetition before
//! made in place of the relative ones. The steps after it see only plain
//! passes, so a mip chain's barriers come from the usual rules.

use std::ops::Range;

use clockwork_rhi::Subresource;

use crate::pass::{PassDesc, Use};
use crate::{MipDirection, PassId, RelativeMip, ResourceUse, TextureUse};

/// A pass after expansion: a pass as written, or one repetition of a
/// repeated pass.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpandedPass {
    /// The pass it comes from, as written.
    pub(crate) source: PassId,
    /// The mip it makes, if it's a repetition.
    pub(crate) mip: Option<u32>,
    /// The pass, with no repeat and no relative mips, unless its repeat has
    /// a mistake, which validation reports. A repetition is named with its
    /// mip, and its uses and bindings keep their order, so a binding still
    /// names its use by index.
    pub(crate) pass: PassDesc,
}

/// Expands `passes`, in order, each repeated pass into its repetitions in
/// the order they walk the chain: going down, from the lowest-numbered mip
/// of its range to the highest, each reading the mip numbered one less, and
/// going up, from the highest-numbered to the lowest, each reading the mip
/// numbered one more. `mips` holds, for each repeated pass, the mips its
/// repetitions make, as validation resolved them, or `None` if its repeat
/// has a mistake, or a relative mip names another texture, which
/// validation reports. That pass is kept as written, so the checks after
/// validation's first step still see its other uses.
///
/// # Panics
///
/// Panics if a pass with a range uses a previous mip the range doesn't
/// give it, which validation reports.
pub(crate) fn expand(passes: &[PassDesc], mips: &[Option<Range<u32>>]) -> Vec<ExpandedPass> {
    let mut expanded = Vec::new();
    for (index, pass) in passes.iter().enumerate() {
        let source = PassId(u32::try_from(index).expect("pass IDs fit a u32"));
        let (Some(repeat), Some(range)) = (pass.repeat, mips[index].clone()) else {
            expanded.push(ExpandedPass {
                source,
                mip: None,
                pass: pass.clone(),
            });
            continue;
        };
        for step in 0..range.len() {
            let step = u32::try_from(step).expect("a chain has at most 32 mips");
            let (mip, previous) = match repeat.direction {
                MipDirection::Down => {
                    let mip = range.start + step;
                    (mip, mip.checked_sub(1))
                }
                MipDirection::Up => {
                    let mip = range.end - 1 - step;
                    (mip, mip.checked_add(1))
                }
            };
            let uses = pass
                .uses
                .iter()
                .map(|used| repetition_use(used, mip, previous))
                .collect();
            expanded.push(ExpandedPass {
                source,
                mip: Some(mip),
                pass: PassDesc {
                    name: format!("{} (mip {mip})", pass.name),
                    uses,
                    repeat: None,
                    ..pass.clone()
                },
            });
        }
    }
    expanded
}

/// `used` in the repetition that makes mip `current` from mip `previous`:
/// a relative mip becomes that mip, and any other use is unchanged.
fn repetition_use(used: &Use, current: u32, previous: Option<u32>) -> Use {
    let ResourceUse::Repeated(part) = &used.resource else {
        return used.clone();
    };
    let mip = match part.mip {
        RelativeMip::Current => current,
        RelativeMip::Previous => {
            previous.expect("validation gives every repetition that reads it a previous mip")
        }
    };
    Use {
        resource: ResourceUse::Texture(TextureUse {
            texture: part.texture,
            subresource: Subresource {
                mips: mip..mip + 1,
                layers: part.layers.clone(),
            },
        }),
        ..used.clone()
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of expanding passes repeated for each mip.

    use clockwork_rhi::{Access, Extent3d, Format};

    use super::*;
    use crate::pass::{Binding, RepeatedUse, UseRole};
    use crate::{
        Attachment, ComputeRecorder, GraphDesc, MipCount, TextureId, TextureSize, TransientTexture,
    };

    /// A graph with a four-mip texture "bloom".
    fn graph() -> (GraphDesc, TextureId) {
        let mut graph = GraphDesc::new("expand");
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(256, 256)),
            )
        };
        let bloom = graph.transient_texture("bloom", chain);
        (graph, bloom)
    }

    /// The mip of each texture use in `pass`, in order, or `None` for any
    /// other use.
    fn mips_used(pass: &PassDesc) -> Vec<Option<Range<u32>>> {
        pass.uses
            .iter()
            .map(|used| match &used.resource {
                ResourceUse::Texture(part) => Some(part.subresource.mips.clone()),
                ResourceUse::Buffer(_) | ResourceUse::Repeated(_) => None,
            })
            .collect()
    }

    /// A pass repeated down the chain becomes a pass per mip, from the
    /// lowest-numbered, each reading the mip numbered one less and writing
    /// its own, between the passes as written, which are unchanged.
    #[test]
    fn a_pass_repeated_down_becomes_one_per_mip() {
        let (mut graph, bloom) = graph();
        graph
            .transfer_pass("fill", "fill")
            .write(bloom.mip(0), Access::CopyWrite);
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        graph
            .transfer_pass("read", "read")
            .read(bloom, Access::CopyRead);

        let expanded = expand(&graph.passes, &[None, Some(1..4), None]);
        let names: Vec<_> = expanded
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
                "read"
            ]
        );
        let sources: Vec<_> = expanded
            .iter()
            .map(|pass| (pass.source, pass.mip))
            .collect();
        assert_eq!(
            sources,
            [
                (PassId(0), None),
                (PassId(1), Some(1)),
                (PassId(1), Some(2)),
                (PassId(1), Some(3)),
                (PassId(2), None),
            ]
        );
        assert_eq!(expanded[0].pass, graph.passes[0]);
        assert_eq!(expanded[4].pass, graph.passes[2]);
        for (repetition, mip) in expanded[1..4].iter().zip(1..) {
            assert_eq!(repetition.pass.repeat, None);
            assert_eq!(
                mips_used(&repetition.pass),
                [Some(mip - 1..mip), Some(mip..mip + 1)]
            );
            assert_eq!(repetition.pass.bindings, graph.passes[1].bindings);
            assert_eq!(repetition.pass.uses[1].role, UseRole::Write);
        }
    }

    /// A pass repeated up the chain walks it from the highest-numbered mip
    /// of its range, each repetition reading the mip numbered one more.
    #[test]
    fn a_pass_repeated_up_walks_from_the_highest_numbered_mip() {
        let (mut graph, bloom) = graph();
        graph
            .compute_pass("upsample", ComputeRecorder::dispatch("upsample"))
            .for_each_mip(bloom, MipDirection::Up, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);

        let expanded = expand(&graph.passes, &[Some(0..3)]);
        let made: Vec<_> = expanded.iter().map(|pass| pass.mip).collect();
        assert_eq!(made, [Some(2), Some(1), Some(0)]);
        assert_eq!(expanded[0].pass.name, "upsample (mip 2)");
        assert_eq!(mips_used(&expanded[0].pass), [Some(3..4), Some(2..3)]);
        assert_eq!(mips_used(&expanded[2].pass), [Some(1..2), Some(0..1)]);
    }

    /// A relative attachment renders to the repetition's mip, a use of a
    /// fixed mip is the same in every repetition, and a repetition that
    /// doesn't read the previous mip needs none, so a pass may make mip 0
    /// going down.
    #[test]
    fn attachments_and_fixed_uses_follow_the_repetition() {
        let (mut graph, bloom) = graph();
        let lights = graph.transient_buffer("lights", 16);
        graph
            .fullscreen_pass("clear", "clear")
            .for_each_mip(bloom, MipDirection::Down, 0..2)
            .bind(lights, Access::FragmentShaderRead)
            .colour(
                bloom.current_mip().attachment(0),
                Attachment::Clear([0.0; 4]),
            );

        let expanded = expand(&graph.passes, &[Some(0..2)]);
        assert_eq!(expanded.len(), 2);
        for (repetition, mip) in expanded.iter().zip(0..) {
            let pass = &repetition.pass;
            assert_eq!(pass.uses[0], graph.passes[0].uses[0]);
            assert_eq!(mips_used(pass), [None, Some(mip..mip + 1)]);
            assert_eq!(
                pass.uses[1].role,
                UseRole::Colour(Attachment::Clear([0.0; 4]))
            );
            assert_eq!(pass.bindings, [Binding::Use(0)]);
        }
    }

    /// A repeated pass with no range, since its repeat has a mistake, is
    /// kept as written, as one pass.
    #[test]
    fn a_pass_with_no_range_is_kept_as_written() {
        let (mut graph, bloom) = graph();
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, 9..)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);

        let expanded = expand(&graph.passes, &[None]);
        assert_eq!(
            expanded,
            [ExpandedPass {
                source: PassId(0),
                mip: None,
                pass: graph.passes[0].clone(),
            }]
        );
    }

    /// A relative use keeps its layers, and a range that starts partway
    /// down the chain makes only the mips in it.
    #[test]
    fn a_relative_use_keeps_its_layers() {
        let mut graph = GraphDesc::new("expand");
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            layers: 4,
            ..TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(256, 256)),
            )
        };
        let shadows = graph.transient_texture("shadows", chain);
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(shadows, MipDirection::Down, 2..)
            .bind(
                RepeatedUse {
                    layers: 1..3,
                    ..shadows.previous_mip()
                },
                Access::ComputeShaderSampledRead,
            )
            .bind(
                RepeatedUse {
                    layers: 1..3,
                    ..shadows.current_mip()
                },
                Access::ComputeShaderWrite,
            );

        let expanded = expand(&graph.passes, &[Some(2..4)]);
        let made: Vec<_> = expanded.iter().map(|pass| pass.mip).collect();
        assert_eq!(made, [Some(2), Some(3)]);
        for (repetition, mip) in expanded.iter().zip(2..) {
            let subresources: Vec<_> = repetition
                .pass
                .uses
                .iter()
                .map(|used| match &used.resource {
                    ResourceUse::Texture(part) => part.subresource.clone(),
                    ResourceUse::Buffer(_) | ResourceUse::Repeated(_) => {
                        panic!("a repetition's uses are of fixed mips")
                    }
                })
                .collect();
            assert_eq!(
                subresources,
                [
                    Subresource {
                        mips: mip - 1..mip,
                        layers: 1..3,
                    },
                    Subresource {
                        mips: mip..mip + 1,
                        layers: 1..3,
                    },
                ]
            );
        }
    }
}
