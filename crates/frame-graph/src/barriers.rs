//! Barrier derivation, the seventh step of compiling (design 0003,
//! "Deriving barriers"): the barriers the scheduled passes need, recorded
//! before each group.
//!
//! It walks the groups in the schedule's order, keeping each subresource's
//! state: the barriers into it this frame. Reads after reads, in one
//! layout, need no barrier, and widen the latest barrier into the
//! subresource to take them in, so a texture written once and sampled by
//! three passes has one barrier, not three. Anything else, a write, a read
//! after a write, or a change of layout, needs a barrier from the accesses
//! the subresource has been used with since its latest barrier.
//!
//! Every use is ordered by the barriers alone, whatever order the schedule
//! chose, since each barrier is from the accesses before it in that order.
//! Everything runs on one queue until queues are assigned, so there are no
//! queue transfers yet.
//!
//! The first use of each subresource in a frame is ordered after the last
//! use before the frame: the accesses its import or the output gives, or
//! for a resource the graph creates, the accesses the frame ends with,
//! which the last frame, running the same plan, left it in.

use std::ops::Range;

use clockwork_rhi::{Access, AccessSet, Subresource};

use crate::desc::{BufferKind, TextureKind};
use crate::expand::ExpandedPass;
use crate::grid::Grid;
use crate::schedule::Scheduled;
use crate::validate::TextureShape;
use crate::{GraphDesc, OutputDesc, ResourceUse};

/// What a planned barrier applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BarrierTarget {
    /// Part of the texture at this index, as its `TextureId` gives. For an
    /// import, whose shape is the caller's, a range ending at
    /// [`Subresource::REMAINING`] may start past its last mip or layer and
    /// cover nothing, such as mips `1..` of a one-mip import used whole and
    /// as mip 0. Executing skips it.
    Texture(usize, Subresource),
    /// The buffer at this index, as its `BufferId` gives.
    Buffer(usize),
}

/// A barrier in a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedBarrier {
    /// What it applies to.
    pub(crate) target: BarrierTarget,
    /// The accesses before it.
    pub(crate) before: AccessSet,
    /// The accesses after it.
    pub(crate) after: AccessSet,
    /// Whether it orders the first use in a frame of a resource the graph
    /// creates after the last frame's uses, from the accesses the frame
    /// ends with.
    ///
    /// The first frame after a compile has no last frame of this plan. A
    /// resource created for it has no contents, so its barrier is from
    /// [`Access::None`] instead. A persistent resource kept from the old
    /// plan, with its contents, is wherever earlier plans left each part
    /// of it, split differently from this plan's, so the executor keeps
    /// each persistent resource's latest accesses per part across plans,
    /// and before that frame moves each part from them, or from
    /// [`Access::None`] for a part no plan has used, to the accesses this
    /// plan ends with. Its barrier then stands.
    pub(crate) from_last_frame: bool,
}

/// The barriers of a schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Barriers {
    /// Every barrier, in the order they're recorded.
    pub(crate) barriers: Vec<PlannedBarrier>,
    /// For each of the schedule's groups, in order, the range of `barriers`
    /// recorded before its first pass.
    pub(crate) groups: Vec<Range<usize>>,
}

/// The barriers into a subresource this frame, by their indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Used {
    /// The first, ordering it after the last use before the frame.
    first: usize,
    /// The latest, whose accesses after are those it's been used with
    /// since.
    latest: usize,
}

/// Where a subresource's state is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Slot {
    /// A cell of the texture at this index, by the cell's index in its grid.
    Texture(usize, usize),
    /// The buffer at this index.
    Buffer(usize),
}

/// The barriers the passes `scheduled` runs need, of `passes`, the expanded
/// passes of `desc`, whose textures' shapes are `shapes`, rendering to the
/// output `output` describes.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "compiling, in a later pull request, calls it")
)]
pub(crate) fn barriers(
    desc: &GraphDesc,
    output: &OutputDesc,
    shapes: &[Option<TextureShape>],
    passes: &[ExpandedPass],
    scheduled: &Scheduled,
) -> Barriers {
    let resolve = |texture: usize, subresource: &Subresource| {
        let resolve = |range: &Range<u32>, count: Option<u32>| match count {
            Some(count) if range.end == Subresource::REMAINING => range.start..count,
            _ => range.clone(),
        };
        let shape = shapes[texture];
        Subresource {
            mips: resolve(&subresource.mips, shape.map(|shape| shape.mips)),
            layers: resolve(&subresource.layers, shape.map(|shape| shape.layers)),
        }
    };

    let mut textures: Vec<Grid<Option<Used>>> =
        desc.textures.iter().map(|_| Grid::default()).collect();
    for &pass in &scheduled.passes {
        for used in &passes[pass].pass.uses {
            if let ResourceUse::Texture(part) = &used.resource {
                let index = part.texture.index();
                let part = resolve(index, &part.subresource);
                textures[index].split(&part.mips, &part.layers);
            }
        }
    }
    let mut textures: Vec<_> = textures.into_iter().map(Grid::finish).collect();
    let mut buffers = vec![None; desc.buffers.len()];

    let mut barriers = Vec::new();
    let mut groups = Vec::with_capacity(scheduled.groups.len());
    // Each subresource a pass uses, with every access its uses of it have,
    // kept from one pass to the next so it's allocated once.
    let mut covered = Vec::new();
    for group in &scheduled.groups {
        let start = barriers.len();
        for &pass in &scheduled.passes[group.clone()] {
            covered.clear();
            for used in &passes[pass].pass.uses {
                match &used.resource {
                    ResourceUse::Texture(part) => {
                        let index = part.texture.index();
                        let part = resolve(index, &part.subresource);
                        let cells = textures[index].cells(&part.mips, &part.layers);
                        covered.extend(cells.map(|cell| (Slot::Texture(index, cell), used.access)));
                    }
                    ResourceUse::Buffer(buffer) => {
                        covered.push((Slot::Buffer(buffer.index()), used.access));
                    }
                    ResourceUse::Repeated(_) => {
                        unreachable!("expanding a valid description resolves every relative mip")
                    }
                }
            }
            covered.sort_unstable_by_key(|&(slot, _)| slot);
            covered.dedup_by(|later, earlier| {
                let same = later.0 == earlier.0;
                if same {
                    earlier.1 |= later.1;
                }
                same
            });
            for &(slot, accesses) in &covered {
                let (state, target, before_frame, texture) = match slot {
                    Slot::Texture(index, cell) => {
                        let grid = &mut textures[index];
                        let (mips, layers) = grid.bounds(cell);
                        let target = BarrierTarget::Texture(index, Subresource { mips, layers });
                        let before_frame = texture_start(&desc.textures[index].kind, output);
                        (&mut grid.cells[cell], target, before_frame, true)
                    }
                    Slot::Buffer(index) => {
                        let before_frame = buffer_start(&desc.buffers[index].kind);
                        let target = BarrierTarget::Buffer(index);
                        (&mut buffers[index], target, before_frame, false)
                    }
                };
                apply(
                    &mut barriers,
                    state,
                    target,
                    before_frame,
                    accesses,
                    texture,
                );
            }
        }
        groups.push(start..barriers.len());
    }

    for (index, texture) in desc.textures.iter().enumerate() {
        if let TextureKind::Transient(_) | TextureKind::Persistent(_) = texture.kind {
            carry_over(&mut barriers, &textures[index].cells);
        }
    }
    for (index, buffer) in desc.buffers.iter().enumerate() {
        if let BufferKind::Transient { .. } | BufferKind::Persistent { .. } = buffer.kind {
            carry_over(&mut barriers, &buffers[index..=index]);
        }
    }
    Barriers { barriers, groups }
}

/// The accesses a texture of `kind` is in before its first use in a
/// frame: its import's or the output's, or `None` for one the graph
/// creates, which the last frame left in the accesses the frame ends with.
fn texture_start(kind: &TextureKind, output: &OutputDesc) -> Option<AccessSet> {
    match kind {
        TextureKind::Output => Some(output.before),
        TextureKind::Imported(import) => Some(import.before),
        TextureKind::Transient(_) | TextureKind::Persistent(_) => None,
    }
}

/// The accesses a buffer of `kind` is in before its first use in a frame:
/// its import's, or `None` for one the graph creates, which the last frame
/// left in the accesses the frame ends with.
fn buffer_start(kind: &BufferKind) -> Option<AccessSet> {
    match kind {
        BufferKind::Imported(import) => Some(import.before),
        BufferKind::Transient { .. } | BufferKind::Persistent { .. } => None,
    }
}

/// Applies a pass's use of a subresource, `target`, with `accesses`, every
/// access its uses of it have, to its state, `used`, recording the barrier
/// it needs, if any, in `barriers`. `start` is the accesses it's in before
/// its first use in a frame, or `None` if they're those the frame ends
/// with, and `texture` says whether it's part of a texture, which has a
/// layout.
fn apply(
    barriers: &mut Vec<PlannedBarrier>,
    used: &mut Option<Used>,
    target: BarrierTarget,
    start: Option<AccessSet>,
    accesses: AccessSet,
    texture: bool,
) {
    let Some(state) = used else {
        // The frame's last accesses aren't known until the walk ends, so the
        // barrier from them is completed then.
        barriers.push(PlannedBarrier {
            target,
            before: start.unwrap_or(Access::None.into()),
            after: accesses,
            from_last_frame: start.is_none(),
        });
        let index = barriers.len() - 1;
        *used = Some(Used {
            first: index,
            latest: index,
        });
        return;
    };
    let latest = &mut barriers[state.latest];
    let since = latest.after;
    let joined = since | accesses;
    let reads = !since.is_write() && !accesses.is_write();
    if reads && (!texture || joined.layout() == since.layout()) {
        latest.after = joined;
        return;
    }
    barriers.push(PlannedBarrier {
        target,
        before: since,
        after: accesses,
        from_last_frame: false,
    });
    state.latest = barriers.len() - 1;
}

/// Completes the first barrier of each of `cells`, the subresources of a
/// resource the graph creates, with the accesses the frame ends with,
/// which the last frame left it in.
fn carry_over(barriers: &mut [PlannedBarrier], cells: &[Option<Used>]) {
    for used in cells.iter().flatten() {
        barriers[used.first].before = barriers[used.latest].after;
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of barrier derivation.

    use clockwork_rhi::{Extent3d, Format};

    use super::*;
    use crate::cull::cull;
    use crate::dependencies::dependencies;
    use crate::schedule::schedule;
    use crate::validate::test_support::{finished, inputs};
    use crate::validate::validate;
    use crate::{
        Attachment, ComputeRecorder, ImportDesc, MipCount, MipDirection, Schedule, TextureSize,
        TransientTexture,
    };

    /// The barriers of `desc`, finished, compiled as written, one line per
    /// pass and barrier, in the order they're recorded.
    fn derived(desc: &GraphDesc) -> Vec<String> {
        let desc = finished(desc);
        let inputs = inputs();
        let validated = validate(&desc, &inputs).expect("the graph is valid");
        let passes = &validated.passes;
        let mut dependencies = dependencies(&desc, passes);
        let kept = cull(&desc, passes, &dependencies);
        dependencies.reduce(&kept);
        let scheduled = schedule(Schedule::AsWritten, &kept);
        let derived = barriers(
            &desc,
            &inputs.output,
            &validated.textures,
            passes,
            &scheduled,
        );

        let line = |barrier: &PlannedBarrier| {
            let range = |range: &Range<u32>| match range.end {
                Subresource::REMAINING => format!("{}..", range.start),
                end => format!("{}..{end}", range.start),
            };
            let target = match &barrier.target {
                BarrierTarget::Texture(index, part) => format!(
                    "{} mips {} layers {}",
                    desc.textures[*index].name,
                    range(&part.mips),
                    range(&part.layers)
                ),
                BarrierTarget::Buffer(index) => desc.buffers[*index].name.clone(),
            };
            let carried = if barrier.from_last_frame {
                " (from the last frame)"
            } else {
                ""
            };
            format!(
                "  {target}: {:?} -> {:?}{carried}",
                barrier.before, barrier.after
            )
        };
        let mut lines = Vec::new();
        for (group, recorded) in scheduled.groups.iter().zip(&derived.groups) {
            lines.extend(derived.barriers[recorded.clone()].iter().map(line));
            for &pass in &scheduled.passes[group.clone()] {
                lines.push(format!("pass \"{}\"", passes[pass].pass.name));
            }
        }
        lines
    }

    /// The lines [`derived`] gives for the pass `finished` adds, which
    /// writes the output.
    fn output_lines() -> [&'static str; 2] {
        [
            "  output mips 0..1 layers 0..1: Acquired -> CopyWrite",
            "pass \"writes the output\"",
        ]
    }

    /// `lines`, then [`output_lines`].
    fn then_output(lines: &[&str]) -> Vec<String> {
        lines
            .iter()
            .chain(&output_lines())
            .map(|line| (*line).to_owned())
            .collect()
    }

    /// A 256 × 256 texture of one mip and layer.
    fn texture() -> TransientTexture {
        let size = TextureSize::Fixed(Extent3d::new_2d(256, 256));
        TransientTexture::new(Format::Rgba16Float, size)
    }

    /// A write after a write, a read after a write and a write after a read
    /// each need a barrier, from the accesses before to those after, and
    /// the first use is ordered after the accesses the frame ends with.
    #[test]
    fn conflicting_uses_need_barriers() {
        let mut graph = GraphDesc::new("conflicts");
        let hdr = graph.transient_texture("hdr", texture());
        graph
            .transfer_pass("fill", "fill")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("patch", "patch")
            .write(hdr, Access::CopyWrite);
        graph
            .compute_pass("measure", ComputeRecorder::recorder("measure"))
            .read(hdr, Access::ComputeShaderSampledRead)
            .side_effects();
        graph
            .transfer_pass("refill", "refill")
            .write(hdr, Access::CopyWrite)
            .side_effects();

        assert_eq!(
            derived(&graph),
            then_output(&[
                "  hdr mips 0..1 layers 0..1: CopyWrite -> CopyWrite (from the last frame)",
                "pass \"fill\"",
                "  hdr mips 0..1 layers 0..1: CopyWrite -> CopyWrite",
                "pass \"patch\"",
                "  hdr mips 0..1 layers 0..1: CopyWrite -> ComputeShaderSampledRead",
                "pass \"measure\"",
                "  hdr mips 0..1 layers 0..1: ComputeShaderSampledRead -> CopyWrite",
                "pass \"refill\"",
            ])
        );
    }

    /// Reads after a write that share a layout need one barrier, which
    /// takes in each of them, and a read in another layout needs a barrier
    /// of its own.
    #[test]
    fn reads_in_one_layout_share_a_barrier() {
        let mut graph = GraphDesc::new("reads");
        let hdr = graph.transient_texture("hdr", texture());
        let ldr = graph.transient_texture("ldr", texture());
        graph
            .transfer_pass("fill", "fill")
            .write(hdr, Access::CopyWrite);
        graph
            .compute_pass("measure", ComputeRecorder::recorder("measure"))
            .read(hdr, Access::ComputeShaderSampledRead)
            .side_effects();
        graph
            .raster_pass("tone map", "tone map")
            .read(hdr, Access::FragmentShaderSampledRead)
            .colour(ldr, Attachment::Discard)
            .side_effects();
        graph
            .transfer_pass("save", "save")
            .read(hdr, Access::CopyRead)
            .side_effects();

        assert_eq!(
            derived(&graph),
            then_output(&[
                "  hdr mips 0..1 layers 0..1: CopyRead -> CopyWrite (from the last frame)",
                "pass \"fill\"",
                "  hdr mips 0..1 layers 0..1: CopyWrite -> FragmentShaderSampledRead \
                 | ComputeShaderSampledRead",
                "pass \"measure\"",
                "  ldr mips 0..1 layers 0..1: ColorAttachmentWrite -> ColorAttachmentWrite \
                 (from the last frame)",
                "pass \"tone map\"",
                "  hdr mips 0..1 layers 0..1: FragmentShaderSampledRead \
                 | ComputeShaderSampledRead -> CopyRead",
                "pass \"save\"",
            ])
        );
    }

    /// A pass's uses of one subresource need one barrier, into every access
    /// they have.
    #[test]
    fn a_pass_uses_a_subresource_once() {
        let mut graph = GraphDesc::new("together");
        let hdr = graph.persistent_texture("hdr", texture());
        graph
            .compute_pass("adjust", ComputeRecorder::recorder("adjust"))
            .read(hdr, Access::ComputeShaderRead)
            .write(hdr, Access::ComputeShaderWrite)
            .side_effects();

        let accesses = "ComputeShaderRead | ComputeShaderWrite";
        assert_eq!(
            derived(&graph),
            then_output(&[
                &format!(
                    "  hdr mips 0..1 layers 0..1: {accesses} -> {accesses} (from the last frame)"
                ),
                "pass \"adjust\"",
            ])
        );
    }

    /// A pass repeated down a mip chain reads one mip while writing the
    /// next, each with barriers of its own, and a later read of the whole
    /// chain needs a barrier only on the mip the last repetition wrote.
    #[test]
    fn mips_have_barriers_of_their_own() {
        let mut graph = GraphDesc::new("chain");
        let chain = TransientTexture {
            mips: MipCount::Count(3),
            ..texture()
        };
        let bloom = graph.transient_texture("bloom", chain);
        graph
            .transfer_pass("fill", "fill")
            .write(bloom.mip(0), Access::CopyWrite);
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        graph
            .compute_pass("composite", ComputeRecorder::dispatch("composite"))
            .bind(bloom, Access::ComputeShaderSampledRead)
            .side_effects();

        let read = "ComputeShaderSampledRead";
        assert_eq!(
            derived(&graph),
            then_output(&[
                &format!(
                    "  bloom mips 0..1 layers 0..1: {read} -> CopyWrite (from the last frame)"
                ),
                "pass \"fill\"",
                &format!("  bloom mips 0..1 layers 0..1: CopyWrite -> {read}"),
                &format!(
                    "  bloom mips 1..2 layers 0..1: {read} -> ComputeShaderWrite \
                     (from the last frame)"
                ),
                "pass \"downsample (mip 1)\"",
                &format!("  bloom mips 1..2 layers 0..1: ComputeShaderWrite -> {read}"),
                &format!(
                    "  bloom mips 2..3 layers 0..1: {read} -> ComputeShaderWrite \
                     (from the last frame)"
                ),
                "pass \"downsample (mip 2)\"",
                &format!("  bloom mips 2..3 layers 0..1: ComputeShaderWrite -> {read}"),
                "pass \"composite\"",
            ])
        );
    }

    /// Reads of a buffer after a write share a barrier, whatever accesses
    /// they have, since a buffer has no layout.
    #[test]
    fn buffer_reads_share_a_barrier() {
        let mut graph = GraphDesc::new("buffer reads");
        let lights = graph.transient_buffer("lights", 64);
        graph
            .transfer_pass("fill", "fill")
            .write(lights, Access::CopyWrite);
        graph
            .compute_pass("cull", ComputeRecorder::recorder("cull"))
            .read(lights, Access::ComputeShaderRead)
            .side_effects();
        graph
            .transfer_pass("save", "save")
            .read(lights, Access::CopyRead)
            .side_effects();

        let reads = "ComputeShaderRead | CopyRead";
        assert_eq!(
            derived(&graph),
            then_output(&[
                &format!("  lights: {reads} -> CopyWrite (from the last frame)"),
                "pass \"fill\"",
                &format!("  lights: CopyWrite -> {reads}"),
                "pass \"cull\"",
                "pass \"save\"",
            ])
        );
    }

    /// A persistent resource's first use is ordered after the accesses the
    /// frame ends with, as a transient one's is. An import's is ordered
    /// after the accesses it starts in. The parts no pass uses, and imports
    /// no pass uses, have no barriers.
    #[test]
    fn first_uses_follow_what_came_before_the_frame() {
        let mut graph = GraphDesc::new("first uses");
        let history = graph.persistent_texture("history", texture());
        let target = graph.import_texture(
            "target",
            ImportDesc {
                before: Access::CopyRead.into(),
                after: Access::FragmentShaderSampledRead.into(),
            },
        );
        graph.import_texture(
            "spare",
            ImportDesc {
                before: Access::CopyRead.into(),
                after: Access::FragmentShaderSampledRead.into(),
            },
        );
        let counts = graph.transient_buffer("counts", 64);
        graph.import_buffer(
            "uploads",
            ImportDesc {
                before: Access::HostWrite.into(),
                after: Access::ComputeShaderRead.into(),
            },
        );
        graph
            .transfer_pass("resolve", "resolve")
            .read(history, Access::CopyRead)
            .write(target.mip(0), Access::CopyWrite)
            .write(counts, Access::CopyWrite);
        graph
            .compute_pass("count", ComputeRecorder::recorder("count"))
            .read(counts, Access::ComputeShaderRead)
            .side_effects();
        graph
            .transfer_pass("keep", "keep")
            .write(history, Access::CopyWrite);

        let lines = [
            "  history mips 0..1 layers 0..1: CopyWrite -> CopyRead (from the last frame)",
            "  target mips 0..1 layers 0..: CopyRead -> CopyWrite",
            "  counts: ComputeShaderRead -> CopyWrite (from the last frame)",
            "pass \"resolve\"",
            "  counts: CopyWrite -> ComputeShaderRead",
            "pass \"count\"",
            "  history mips 0..1 layers 0..1: CopyRead -> CopyWrite",
            "pass \"keep\"",
            "  output mips 0..1 layers 0..1: Acquired -> CopyWrite",
            "pass \"writes the output\"",
        ];
        assert_eq!(derived(&graph), lines);
    }
}
