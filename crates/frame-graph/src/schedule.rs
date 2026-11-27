//! Scheduling, the sixth step of compiling (design 0003, "Scheduling"): the
//! order the kept passes run in, split into groups, runs of consecutive
//! passes with no dependencies between them, whose barriers are recorded
//! together before the group's first pass.

use std::ops::Range;

use crate::Schedule;
#[cfg(test)]
use crate::dependencies::Dependencies;
#[cfg(test)]
use crate::expand::ExpandedPass;

/// The order the kept passes run in, and their groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Scheduled {
    /// The kept passes, by their indices in the expanded passes, in the
    /// order they run.
    pub(crate) passes: Vec<usize>,
    /// The groups, in the order they run, each a range of `passes`. Every
    /// pass is in one.
    pub(crate) groups: Vec<Range<usize>>,
}

/// The order the passes that culling keeps, as `kept` says by their
/// indices, run in under the schedule `kind`.
///
/// `AsWritten` runs them in the written order, which is always a
/// topological order of the dependency graph, since every dependency is
/// on an earlier pass, and puts each in a group of its own, so its
/// barriers are recorded before it, as they would be by hand.
pub(crate) fn schedule(kind: Schedule, kept: &[bool]) -> Scheduled {
    match kind {
        Schedule::AsWritten => {
            let passes: Vec<_> = (0..kept.len()).filter(|&pass| kept[pass]).collect();
            let groups = (0..passes.len()).map(|place| place..place + 1).collect();
            Scheduled { passes, groups }
        }
    }
}

#[cfg(test)]
impl Scheduled {
    /// Why this isn't a valid schedule of the passes `kept` keeps, of
    /// `passes`, whose dependency graph, reduced, is `dependencies`, if it
    /// isn't: each kept pass runs once, after every pass it depends on, and
    /// the groups cover the passes in order, with no dependency between two
    /// passes of one group, and no part of a texture used by two of them in
    /// different layouts.
    ///
    /// The design also puts a group's passes on one queue. Until queues are
    /// assigned, in the `graph-queues` slice, every pass is on one queue, so
    /// there's nothing to check; that slice adds the check.
    pub(crate) fn mistake(
        &self,
        passes: &[ExpandedPass],
        dependencies: &Dependencies,
        kept: &[bool],
    ) -> Option<String> {
        let mut place = vec![None; kept.len()];
        for (at, &pass) in self.passes.iter().enumerate() {
            match kept.get(pass) {
                None => return Some(format!("pass {pass} runs, but the graph has no such pass")),
                Some(false) => return Some(format!("pass {pass} runs, but it's culled")),
                Some(true) => {}
            }
            if place[pass].replace(at).is_some() {
                return Some(format!("pass {pass} runs more than once"));
            }
        }
        if let Some(pass) = (0..kept.len()).find(|&pass| kept[pass] && place[pass].is_none()) {
            return Some(format!("pass {pass} is kept, but doesn't run"));
        }
        let mut group = vec![0; self.passes.len()];
        let mut next = 0;
        for (index, range) in self.groups.iter().enumerate() {
            if range.start != next || range.is_empty() || range.end > self.passes.len() {
                return Some(format!("group {index} is {range:?}, after {next} passes"));
            }
            group[range.clone()].fill(index);
            next = range.end;
        }
        if next != self.passes.len() {
            return Some(format!(
                "the groups cover {next} of {} passes",
                self.passes.len()
            ));
        }
        for (at, &pass) in self.passes.iter().enumerate() {
            for dependency in &dependencies.on[pass] {
                let Some(earlier) = place[dependency.on] else {
                    return Some(format!(
                        "pass {pass} depends on pass {}, which doesn't run",
                        dependency.on
                    ));
                };
                if earlier > at {
                    return Some(format!("pass {pass} runs before pass {}", dependency.on));
                }
                if group[earlier] == group[at] {
                    return Some(format!(
                        "pass {pass} shares a group with pass {}, which it depends on",
                        dependency.on
                    ));
                }
            }
        }
        for range in &self.groups {
            let members = &self.passes[range.clone()];
            for (at, &first) in members.iter().enumerate() {
                for &second in &members[at + 1..] {
                    if let Some(texture) = layout_clash(&passes[first], &passes[second]) {
                        return Some(format!(
                            "passes {first} and {second} share a group, but use texture \
                             {texture} in different layouts"
                        ));
                    }
                }
            }
        }
        None
    }
}

/// The index of a texture that `first` and `second` use overlapping parts
/// of in different layouts, if there's one: such passes can't share a
/// group, since its barriers, recorded before all of its passes, can leave
/// the part in only one layout. Each pass's uses of the part are joined
/// first, as barrier derivation joins them, so the layouts compared are
/// those its barriers move the part to.
#[cfg(test)]
pub(crate) fn layout_clash(first: &ExpandedPass, second: &ExpandedPass) -> Option<usize> {
    use clockwork_rhi::{AccessSet, Layout, Subresource};

    use crate::{ResourceUse, TextureUse};

    /// Whether two ranges share a mip or layer.
    fn overlap(a: &Range<u32>, b: &Range<u32>) -> bool {
        a.start < b.end && b.start < a.end
    }
    /// The part both `a` and `b` cover, or `None` if they're of different
    /// textures or don't overlap.
    fn shared(a: &TextureUse, b: &TextureUse) -> Option<TextureUse> {
        let (a_part, b_part) = (&a.subresource, &b.subresource);
        let both = a.texture == b.texture
            && overlap(&a_part.mips, &b_part.mips)
            && overlap(&a_part.layers, &b_part.layers);
        both.then(|| TextureUse {
            texture: a.texture,
            subresource: Subresource {
                mips: a_part.mips.start.max(b_part.mips.start)
                    ..a_part.mips.end.min(b_part.mips.end),
                layers: a_part.layers.start.max(b_part.layers.start)
                    ..a_part.layers.end.min(b_part.layers.end),
            },
        })
    }
    /// The layout the accesses of `pass`'s uses that overlap `area` join
    /// into.
    fn layout(pass: &ExpandedPass, area: &TextureUse) -> Option<Layout> {
        pass.pass
            .uses
            .iter()
            .filter_map(|used| match &used.resource {
                ResourceUse::Texture(part) if shared(part, area).is_some() => Some(used.access),
                _ => None,
            })
            .reduce(AccessSet::union)
            .and_then(AccessSet::layout)
    }

    /// The parts of textures `pass` uses.
    fn textures(pass: &ExpandedPass) -> impl Iterator<Item = &TextureUse> {
        pass.pass
            .uses
            .iter()
            .filter_map(|used| match &used.resource {
                ResourceUse::Texture(part) => Some(part),
                _ => None,
            })
    }

    textures(first).find_map(|a| {
        textures(second).find_map(|b| {
            let area = shared(a, b)?;
            (layout(first, &area) != layout(second, &area)).then(|| a.texture.index())
        })
    })
}

#[cfg(test)]
mod tests {
    //! T0 tests of scheduling.

    use clockwork_rhi::{Access, Extent3d, Format};

    use super::*;
    use crate::cull::cull;
    use crate::dependencies::dependencies;
    use crate::validate::test_support::{finished, inputs};
    use crate::validate::validate;
    use crate::{
        ComputeRecorder, GraphDesc, MipCount, MipDirection, TextureSize, TransientTexture,
    };

    /// `desc`, a valid description, compiled as far as scheduling it as
    /// written.
    struct Compiled {
        /// Its schedule.
        scheduled: Scheduled,
        /// Its expanded passes.
        passes: Vec<ExpandedPass>,
        /// Its dependency graph, reduced.
        dependencies: Dependencies,
        /// Which of its passes are kept.
        kept: Vec<bool>,
    }

    /// `desc`, a valid description, compiled as far as scheduling it as
    /// written.
    fn compiled(desc: &GraphDesc) -> Compiled {
        let validated = validate(desc, &inputs()).expect("the graph is valid");
        let dependencies = dependencies(desc, &validated.passes);
        let kept = cull(desc, &validated.passes, &dependencies);
        Compiled {
            scheduled: schedule(Schedule::AsWritten, &kept),
            dependencies: Dependencies::of_kept(desc, &validated.passes, &kept),
            passes: validated.passes,
            kept,
        }
    }

    impl Compiled {
        /// Why the schedule of `passes` and `groups` isn't valid for the
        /// graph, with `kept` in place of the passes culling keeps.
        fn mistake(
            &self,
            passes: Vec<usize>,
            groups: Vec<Range<usize>>,
            kept: &[bool],
        ) -> Option<String> {
            Scheduled { passes, groups }.mistake(&self.passes, &self.dependencies, kept)
        }
    }

    /// `AsWritten` runs the kept passes in the written order, each in a
    /// group of its own, which is a valid schedule, and leaves culled
    /// passes out.
    #[test]
    fn as_written_runs_the_kept_passes_in_order() {
        let mut graph = GraphDesc::new("schedule");
        let size = TextureSize::Fixed(Extent3d::new_2d(256, 256));
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..TransientTexture::new(Format::Rgba16Float, size)
        };
        let bloom = graph.transient_texture("bloom", chain);
        let unread =
            graph.transient_texture("unread", TransientTexture::new(Format::Rgba8Unorm, size));
        graph
            .transfer_pass("fill", "fill")
            .write(bloom.mip(0), Access::CopyWrite);
        graph
            .transfer_pass("culled", "culled")
            .write(unread, Access::CopyWrite);
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        graph
            .compute_pass("composite", ComputeRecorder::dispatch("composite"))
            .bind(bloom, Access::ComputeShaderSampledRead)
            .side_effects();

        let compiled = compiled(&finished(&graph));
        let scheduled = &compiled.scheduled;
        // Pass 1 is culled, passes 2 to 4 are the downsample's
        // repetitions, and pass 6 writes the output.
        assert_eq!(scheduled.passes, [0, 2, 3, 4, 5, 6]);
        assert_eq!(scheduled.groups, [0..1, 1..2, 2..3, 3..4, 4..5, 5..6]);
        let mistake = scheduled.mistake(&compiled.passes, &compiled.dependencies, &compiled.kept);
        assert_eq!(mistake, None);
    }

    /// The check of a schedule finds a pass that runs before one it depends
    /// on or depends on one that doesn't run, a pass that runs twice, is
    /// missing, is culled or isn't in the graph, groups that don't cover
    /// the passes in order, and two passes of one group with a dependency
    /// between them.
    #[test]
    fn invalid_schedules_are_found() {
        let mut graph = GraphDesc::new("invalid");
        let size = TextureSize::Fixed(Extent3d::new_2d(16, 16));
        let hdr = graph.transient_texture("hdr", TransientTexture::new(Format::Rgba16Float, size));
        graph
            .transfer_pass("write", "a")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("read", "b")
            .read(hdr, Access::CopyRead)
            .side_effects();
        let compiled = compiled(&finished(&graph));
        assert_eq!(compiled.scheduled.passes, [0, 1, 2]);

        let kept = compiled.kept.clone();
        let mistake = |passes, groups, kept: &[bool]| compiled.mistake(passes, groups, kept);
        let singles = || vec![0..1, 1..2, 2..3];
        let message = |text: &str| Some(text.to_owned());
        assert_eq!(
            mistake(vec![1, 0, 2], singles(), &kept),
            message("pass 1 runs before pass 0")
        );
        assert_eq!(
            mistake(vec![0, 0, 2], singles(), &kept),
            message("pass 0 runs more than once")
        );
        assert_eq!(
            mistake(vec![0, 1], vec![0..1, 1..2], &kept),
            message("pass 2 is kept, but doesn't run")
        );
        assert_eq!(
            mistake(vec![0, 1, 2], singles(), &[true, false, true]),
            message("pass 1 runs, but it's culled")
        );
        assert_eq!(
            mistake(vec![1, 2], vec![0..1, 1..2], &[false, true, true]),
            message("pass 1 depends on pass 0, which doesn't run")
        );
        assert_eq!(
            mistake(vec![0, 1, 7], singles(), &kept),
            message("pass 7 runs, but the graph has no such pass")
        );
        assert_eq!(
            mistake(vec![0, 1, 2], vec![0..1, 2..3], &kept),
            message("group 1 is 2..3, after 1 passes")
        );
        assert_eq!(
            mistake(vec![0, 1, 2], vec![0..1, 1..1, 1..3], &kept),
            message("group 1 is 1..1, after 1 passes")
        );
        assert_eq!(
            mistake(vec![0, 1, 2], vec![0..1, 1..5], &kept),
            message("group 1 is 1..5, after 1 passes")
        );
        assert_eq!(
            mistake(vec![0, 1, 2], vec![0..1, 1..2], &kept),
            message("the groups cover 2 of 3 passes")
        );
        assert_eq!(
            mistake(vec![0, 1, 2], vec![0..2, 2..3], &kept),
            message("pass 1 shares a group with pass 0, which it depends on")
        );
    }

    /// The check of a schedule finds two passes of one group that use a
    /// part of a texture in different layouts, though they have no
    /// dependency between them, and allows two that share a layout.
    #[test]
    fn a_group_uses_a_texture_in_one_layout() {
        let mut graph = GraphDesc::new("layouts");
        let size = TextureSize::Fixed(Extent3d::new_2d(16, 16));
        let hdr = graph.transient_texture("hdr", TransientTexture::new(Format::Rgba16Float, size));
        graph
            .transfer_pass("write", "a")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("copy", "b")
            .read(hdr, Access::CopyRead)
            .side_effects();
        graph
            .compute_pass("sample", ComputeRecorder::recorder("c"))
            .read(hdr, Access::ComputeShaderSampledRead)
            .side_effects();
        graph
            .compute_pass("sample again", ComputeRecorder::recorder("d"))
            .read(hdr, Access::ComputeShaderSampledRead)
            .side_effects();
        let compiled = compiled(&finished(&graph));
        let kept = &compiled.kept;
        assert_eq!(compiled.scheduled.passes, [0, 1, 2, 3, 4]);

        assert_eq!(
            compiled.mistake(vec![0, 1, 2, 3, 4], vec![0..1, 1..3, 3..4, 4..5], kept),
            Some("passes 1 and 2 share a group, but use texture 1 in different layouts".to_owned())
        );
        assert_eq!(
            compiled.mistake(vec![0, 1, 2, 3, 4], vec![0..1, 1..2, 2..4, 4..5], kept),
            None
        );
    }
}
