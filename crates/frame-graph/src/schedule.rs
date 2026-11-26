//! Scheduling, the sixth step of compiling (design 0003, "Scheduling"): the
//! order the kept passes run in, split into groups, runs of consecutive
//! passes with no dependencies between them, whose barriers are recorded
//! together before the group's first pass.

use std::ops::Range;

use crate::Schedule;
#[cfg(test)]
use crate::dependencies::Dependencies;

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
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "compiling, in a later pull request, calls it")
)]
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
    /// Why this isn't a valid schedule of the passes `kept` keeps, whose
    /// dependency graph, reduced, is `dependencies`, if it isn't: each kept
    /// pass runs once, after every pass it depends on, and the groups cover
    /// the passes in order, with no dependency between two passes of one
    /// group.
    ///
    /// The design also puts a group's passes on one queue. Until queues are
    /// assigned, in the `graph-queues` slice, every pass is on one queue, so
    /// there's nothing to check; that slice adds the check.
    pub(crate) fn mistake(&self, dependencies: &Dependencies, kept: &[bool]) -> Option<String> {
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
        None
    }
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
    /// written: its schedule, its reduced dependency graph, and which of
    /// its passes are kept.
    fn compiled(desc: &GraphDesc) -> (Scheduled, Dependencies, Vec<bool>) {
        let validated = validate(desc, &inputs()).expect("the graph is valid");
        let mut dependencies = dependencies(desc, &validated.passes);
        let kept = cull(desc, &validated.passes, &dependencies);
        dependencies.reduce(&kept);
        (schedule(Schedule::AsWritten, &kept), dependencies, kept)
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

        let (scheduled, dependencies, kept) = compiled(&finished(&graph));
        // Pass 1 is culled, passes 2 to 4 are the downsample's
        // repetitions, and pass 6 writes the output.
        assert_eq!(scheduled.passes, [0, 2, 3, 4, 5, 6]);
        assert_eq!(scheduled.groups, [0..1, 1..2, 2..3, 3..4, 4..5, 5..6]);
        assert_eq!(scheduled.mistake(&dependencies, &kept), None);
    }

    /// The check of a schedule finds a pass that runs before one it depends
    /// on or depends on one that doesn't run, a pass that runs twice, is
    /// missing, is culled or isn't in the graph, groups that don't cover the passes in order, and two passes of
    /// one group with a dependency between them.
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
        let (valid, dependencies, kept) = compiled(&finished(&graph));
        assert_eq!(valid.passes, [0, 1, 2]);

        let mistake = |passes: Vec<usize>, groups: Vec<Range<usize>>, kept: &[bool]| {
            Scheduled { passes, groups }.mistake(&dependencies, kept)
        };
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
}
