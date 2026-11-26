//! The dependency graph, the third step of compiling (design 0003,
//! "Scheduling"): an edge from one pass to a later one, in the written
//! order, wherever their uses of a subresource conflict. A read sees the
//! writes before it back to the latest that replaced the contents, a write
//! follows the reads of what it overwrites, and a write follows the write
//! before it. Reads after reads add no edge, and nor do uses of different
//! mips or layers of one texture.
//!
//! Only a cleared or discarded attachment replaces the contents, since its
//! load operation drops them. Any other write, such as a copy or a storage
//! write, may cover only part of a subresource, so the writes before it
//! stay visible, unless its pass read them too, in which case what it
//! writes carries them on.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::expand::ExpandedPass;
use crate::pass::Use;
use crate::{GraphDesc, ResourceUse};

/// How a pass depends on an earlier one: which conflicts their uses have.
/// A pair of passes can have several, on different subresources.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Hazards {
    /// The later pass reads what the earlier one wrote, which no write
    /// between them replaced.
    pub(crate) read_after_write: bool,
    /// The later pass overwrites what the earlier one read.
    pub(crate) write_after_read: bool,
    /// The later pass overwrites what the earlier one wrote.
    pub(crate) write_after_write: bool,
}

/// A pass's dependency on an earlier one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dependency {
    /// The earlier pass, by its index in the expanded passes.
    pub(crate) on: usize,
    /// Why.
    pub(crate) hazards: Hazards,
}

/// The dependency graph of a description's expanded passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dependencies {
    /// For each expanded pass, by its index, the earlier passes it depends
    /// on, in the written order: every one at first, which culling follows,
    /// and once [`reduce`](Self::reduce)d, only those between kept passes
    /// that no path implies.
    pub(crate) on: Vec<Vec<Dependency>>,
}

/// The dependency graph of `passes`, the expanded passes of `desc`, which
/// validation found have no mistakes, with every dependency, not yet
/// reduced.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "compiling, in a later pull request, calls it")
)]
pub(crate) fn dependencies(desc: &GraphDesc, passes: &[ExpandedPass]) -> Dependencies {
    let mut textures: Vec<_> = desc.textures.iter().map(|_| Grid::default()).collect();
    for used in passes.iter().flat_map(|expanded| &expanded.pass.uses) {
        if let ResourceUse::Texture(part) = &used.resource {
            textures[part.texture.index()].split(&part.subresource.mips, &part.subresource.layers);
        }
    }
    let mut textures: Vec<_> = textures.into_iter().map(Grid::finish).collect();
    let mut buffers = vec![Cell::default(); desc.buffers.len()];

    let mut found = Vec::with_capacity(passes.len());
    // Each cell a pass's uses cover, with the use, kept from one pass to the
    // next so it's allocated once.
    let mut covered = Vec::new();
    for (index, expanded) in passes.iter().enumerate() {
        covered.clear();
        for used in &expanded.pass.uses {
            match &used.resource {
                ResourceUse::Texture(part) => {
                    let texture = part.texture.index();
                    let (mips, layers) = (&part.subresource.mips, &part.subresource.layers);
                    let cells = textures[texture].cells(mips, layers);
                    covered.extend(cells.map(|cell| (Slot::Texture(texture, cell), used)));
                }
                ResourceUse::Buffer(buffer) => covered.push((Slot::Buffer(buffer.index()), used)),
                ResourceUse::Repeated(_) => {
                    unreachable!("expanding a valid description resolves every relative mip")
                }
            }
        }
        let mut on = BTreeMap::new();
        // A pass's reads see what was written before it, so they're found
        // before its writes change that, and its own uses never conflict.
        for step in [Step::Read, Step::Write, Step::Join] {
            for &(slot, used) in &covered {
                let cell = match slot {
                    Slot::Texture(texture, cell) => &mut textures[texture].cells[cell],
                    Slot::Buffer(buffer) => &mut buffers[buffer],
                };
                cell.apply(step, index, used, &mut on);
            }
        }
        let on = on
            .into_iter()
            .map(|(on, hazards)| Dependency { on, hazards });
        found.push(on.collect());
    }
    Dependencies { on: found }
}

/// Where a subresource's state is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// A cell of the texture at this index, by the cell's index in its grid.
    Texture(usize, usize),
    /// The buffer at this index.
    Buffer(usize),
}

/// One step of applying a pass's uses to the subresources' state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Its reads depend on the visible writes before them.
    Read,
    /// Its writes depend on the reads and the write before them, and
    /// become visible.
    Write,
    /// Its reads of what it didn't write join the reads since the latest
    /// write.
    Join,
}

/// A subresource's uses so far: a mip of a layer of a texture, or a buffer.
#[derive(Debug, Clone, Default)]
struct Cell {
    /// The passes whose writes are visible, by their indices, in order: the
    /// latest that replaced the contents, or whose pass read them, and
    /// each after it, which may have covered only part of it.
    writers: Vec<usize>,
    /// The passes that read it since the latest write, by their indices,
    /// in order.
    readers: Vec<usize>,
    /// The last pass that read it, by its index, so its write is known to
    /// carry on what it read.
    read_by: Option<usize>,
}

impl Cell {
    /// Applies `step` of `used`, by the pass at `index`, recording each
    /// earlier pass it depends on in `on`.
    fn apply(&mut self, step: Step, index: usize, used: &Use, on: &mut BTreeMap<usize, Hazards>) {
        let reads = used.role.reads_contents(used.access);
        let writes = used.access.is_write();
        let mut depend = |pass: usize, hazard: fn(&mut Hazards)| {
            if pass != index {
                hazard(on.entry(pass).or_default());
            }
        };
        match step {
            Step::Read if reads => {
                for &writer in &self.writers {
                    depend(writer, |hazards| hazards.read_after_write = true);
                }
                self.read_by = Some(index);
            }
            Step::Write if writes => {
                for &reader in &self.readers {
                    depend(reader, |hazards| hazards.write_after_read = true);
                }
                if let Some(&writer) = self.writers.last() {
                    depend(writer, |hazards| hazards.write_after_write = true);
                }
                // A write that replaces the contents, or follows its pass's
                // read of them, leaves nothing earlier to see.
                if used.role.replaces_contents() || self.read_by == Some(index) {
                    self.writers.clear();
                }
                if self.writers.last() != Some(&index) {
                    self.writers.push(index);
                }
                self.readers.clear();
            }
            Step::Join if reads && self.writers.last() != Some(&index) => {
                if self.readers.last() != Some(&index) {
                    self.readers.push(index);
                }
            }
            Step::Read | Step::Write | Step::Join => {}
        }
    }
}

/// A texture's subresources, split only as finely as its uses need: its
/// mips at every bound a use's mips have, and its layers likewise, so a
/// texture whose size the graph doesn't know is split as one it does.
#[derive(Debug, Clone, Default)]
struct Grid {
    /// The bounds the mips are split at, in order.
    mips: Vec<u32>,
    /// The bounds the layers are split at, in order.
    layers: Vec<u32>,
    /// Each cell's uses, by mip interval then layer interval.
    cells: Vec<Cell>,
}

impl Grid {
    /// Splits the texture at the bounds of `mips` and `layers`.
    fn split(&mut self, mips: &Range<u32>, layers: &Range<u32>) {
        self.mips.extend([mips.start, mips.end]);
        self.layers.extend([layers.start, layers.end]);
    }

    /// The texture, split at every bound, with nothing used yet.
    fn finish(mut self) -> Self {
        for bounds in [&mut self.mips, &mut self.layers] {
            bounds.sort_unstable();
            bounds.dedup();
        }
        let count = |bounds: &[u32]| bounds.len().saturating_sub(1);
        self.cells = vec![Cell::default(); count(&self.mips) * count(&self.layers)];
        self
    }

    /// The indices of the cells `mips` and `layers` cover, whose bounds are
    /// among the texture's.
    fn cells(&self, mips: &Range<u32>, layers: &Range<u32>) -> impl Iterator<Item = usize> {
        let intervals = |bounds: &[u32], range: &Range<u32>| {
            let at = |bound| {
                bounds
                    .binary_search(&bound)
                    .expect("the texture is split at every use's bounds")
            };
            at(range.start)..at(range.end)
        };
        let columns = self.layers.len() - 1;
        let layers = intervals(&self.layers, layers);
        intervals(&self.mips, mips)
            .flat_map(move |mip| layers.clone().map(move |layer| mip * columns + layer))
    }
}

impl Dependencies {
    /// Leaves the graph of the passes `kept` keeps, by their indices,
    /// transitively reduced, so it stays small: a pass that isn't kept
    /// loses its dependencies, and so does each pass on it, and a
    /// dependency that a path through other kept passes implies is dropped.
    /// Culling comes first, since a path through a pass it then removed
    /// would no longer imply the dependency it stood for.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "compiling, in a later pull request, calls it")
    )]
    pub(crate) fn reduce(&mut self, kept: &[bool]) {
        for (on, &kept_pass) in self.on.iter_mut().zip(kept) {
            on.retain(|dependency| kept_pass && kept[dependency.on]);
        }
        let count = self.on.len();
        let mut dependents = vec![Vec::new(); count];
        for (pass, on) in self.on.iter().enumerate() {
            for dependency in on {
                dependents[dependency.on].push(pass);
            }
        }
        // What each pass reaches, worked out from the last pass back, since
        // a pass reaches only later ones. A pass's dependents, taken in
        // order, reach every later pass reachable through an earlier
        // dependent, so one already reached is implied.
        let mut reaches = vec![Bits::new(count); count];
        let mut implied = vec![Vec::new(); count];
        for pass in (0..count).rev() {
            let mut reached = Bits::new(count);
            for &dependent in &dependents[pass] {
                if reached.contains(dependent) {
                    implied[dependent].push(pass);
                }
                reached.insert(dependent);
                reached.union(&reaches[dependent]);
            }
            reaches[pass] = reached;
        }
        for (on, implied) in self.on.iter_mut().zip(implied) {
            on.retain(|dependency| !implied.contains(&dependency.on));
        }
    }
}

/// A set of passes, by their indices.
#[derive(Debug, Clone)]
struct Bits(Vec<u64>);

impl Bits {
    /// No passes, of `count`.
    fn new(count: usize) -> Self {
        Self(vec![0; count.div_ceil(64)])
    }

    /// Whether `pass` is in the set.
    fn contains(&self, pass: usize) -> bool {
        self.0[pass / 64] & (1 << (pass % 64)) != 0
    }

    /// Adds `pass`.
    fn insert(&mut self, pass: usize) {
        self.0[pass / 64] |= 1 << (pass % 64);
    }

    /// Adds every pass in `other`.
    fn union(&mut self, other: &Self) {
        for (word, other) in self.0.iter_mut().zip(&other.0) {
            *word |= other;
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the dependency graph.

    use clockwork_rhi::{Access, Extent3d, Format, Subresource};

    use super::*;
    use crate::expand::expand;
    use crate::{
        Attachment, BufferId, ComputeRecorder, ImportDesc, MipCount, MipDirection, TextureId,
        TextureSize, TransientTexture,
    };

    /// No hazards.
    const NONE: Hazards = Hazards {
        read_after_write: false,
        write_after_read: false,
        write_after_write: false,
    };
    /// A read after a write.
    const RAW: Hazards = Hazards {
        read_after_write: true,
        ..NONE
    };
    /// A write after a read.
    const WAR: Hazards = Hazards {
        write_after_read: true,
        ..NONE
    };
    /// A write after a write.
    const WAW: Hazards = Hazards {
        write_after_write: true,
        ..NONE
    };

    /// A graph with a texture "hdr", a four-mip texture "bloom" and a
    /// buffer "lights", all transient.
    fn graph() -> (GraphDesc, TextureId, TextureId, BufferId) {
        let mut graph = GraphDesc::new("dependencies");
        let size = TextureSize::Fixed(Extent3d::new_2d(256, 256));
        let hdr = graph.transient_texture("hdr", TransientTexture::new(Format::Rgba16Float, size));
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..TransientTexture::new(Format::Rgba16Float, size)
        };
        let bloom = graph.transient_texture("bloom", chain);
        let lights = graph.transient_buffer("lights", 64);
        (graph, hdr, bloom, lights)
    }

    /// The edges of `dependencies`, as (earlier, later, hazards).
    fn listed(dependencies: &Dependencies) -> Vec<(usize, usize, Hazards)> {
        dependencies
            .on
            .iter()
            .enumerate()
            .flat_map(|(pass, on)| {
                on.iter()
                    .map(move |dependency| (dependency.on, pass, dependency.hazards))
            })
            .collect()
    }

    /// The dependency graph of `graph`, whose passes aren't repeated, not
    /// yet reduced.
    fn unreduced(graph: &GraphDesc) -> Dependencies {
        let passes = expand(&graph.passes, &vec![None; graph.passes.len()]);
        dependencies(graph, &passes)
    }

    /// The edges of `graph`'s dependency graph, reduced with every pass
    /// kept, with its passes expanded as `mips` gives each.
    fn edges_with(graph: &GraphDesc, mips: &[Option<Range<u32>>]) -> Vec<(usize, usize, Hazards)> {
        let passes = expand(&graph.passes, mips);
        let mut dependencies = dependencies(graph, &passes);
        dependencies.reduce(&vec![true; passes.len()]);
        listed(&dependencies)
    }

    /// The edges of `graph`'s dependency graph, whose passes aren't
    /// repeated, reduced with every pass kept.
    fn edges(graph: &GraphDesc) -> Vec<(usize, usize, Hazards)> {
        edges_with(graph, &vec![None; graph.passes.len()])
    }

    /// A read depends on the write before it, a write on the reads before
    /// it, and a write on the write before it.
    #[test]
    fn each_conflict_is_an_edge() {
        let (mut graph, hdr, ..) = graph();
        graph
            .transfer_pass("write", "write")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("read", "read")
            .read(hdr, Access::CopyRead);
        graph
            .transfer_pass("overwrite", "overwrite")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("again", "again")
            .write(hdr, Access::CopyWrite);
        assert_eq!(edges(&graph), [(0, 1, RAW), (1, 2, WAR), (2, 3, WAW)]);
    }

    /// Reads after reads add no edge, so passes that read what one pass
    /// wrote each depend on it alone, and the next write on each of them.
    #[test]
    fn reads_after_reads_are_independent() {
        let (mut graph, hdr, ..) = graph();
        graph
            .transfer_pass("write", "write")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("first", "first")
            .read(hdr, Access::CopyRead);
        graph
            .transfer_pass("second", "second")
            .read(hdr, Access::CopyRead);
        graph
            .transfer_pass("overwrite", "overwrite")
            .write(hdr, Access::CopyWrite);
        assert_eq!(
            edges(&graph),
            [(0, 1, RAW), (0, 2, RAW), (1, 3, WAR), (2, 3, WAR)]
        );
    }

    /// A read sees the latest write before it in the written order that
    /// replaced the contents, not an earlier one.
    #[test]
    fn a_read_sees_the_latest_write() {
        let (mut graph, hdr, ..) = graph();
        graph
            .raster_pass("first", "first")
            .colour(hdr, Attachment::Clear([0.0; 4]));
        graph
            .raster_pass("second", "second")
            .colour(hdr, Attachment::Discard);
        graph
            .transfer_pass("read", "read")
            .read(hdr, Access::CopyRead);
        assert_eq!(edges(&graph), [(0, 1, WAW), (1, 2, RAW)]);
    }

    /// A write that may cover only part of a subresource, such as a copy,
    /// leaves the writes before it visible, so a later read depends on each
    /// of them, back to the one that replaced the contents. A pass that
    /// reads the contents before writing them carries them on, so a read
    /// after it depends on it alone. The graph is checked before it's
    /// reduced, which would drop the read that the path through the partial
    /// write implies.
    #[test]
    fn partial_writes_leave_earlier_writes_visible() {
        let (mut graph, hdr, _, lights) = graph();
        graph
            .raster_pass("clear", "clear")
            .colour(hdr, Attachment::Clear([0.0; 4]));
        graph
            .transfer_pass("overlay", "overlay")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("read", "read")
            .read(hdr, Access::CopyRead);
        graph
            .transfer_pass("fill", "fill")
            .write(lights, Access::CopyWrite);
        graph
            .compute_pass("update", ComputeRecorder::recorder("update"))
            .read(lights, Access::ComputeShaderRead)
            .write(lights, Access::ComputeShaderWrite);
        graph
            .transfer_pass("use", "use")
            .read(lights, Access::CopyRead);
        let both = Hazards {
            read_after_write: true,
            write_after_write: true,
            ..NONE
        };
        assert_eq!(
            listed(&unreduced(&graph)),
            [
                (0, 1, WAW),
                (0, 2, RAW),
                (1, 2, RAW),
                (3, 4, both),
                (4, 5, RAW)
            ]
        );
    }

    /// Uses of different mips or layers of one texture don't conflict,
    /// while a use of the whole texture conflicts with each.
    #[test]
    fn different_mips_and_layers_are_independent() {
        let (mut graph, _, bloom, _) = graph();
        let array = graph.transient_texture(
            "array",
            TransientTexture {
                layers: 2,
                ..TransientTexture::new(
                    Format::Rgba16Float,
                    TextureSize::Fixed(Extent3d::new_2d(16, 16)),
                )
            },
        );
        graph
            .transfer_pass("mip 0", "a")
            .write(bloom.mip(0), Access::CopyWrite);
        graph
            .transfer_pass("mip 1", "b")
            .write(bloom.mip(1), Access::CopyWrite);
        graph
            .transfer_pass("layer 0", "c")
            .write(array.subresource(Subresource::layer(0)), Access::CopyWrite);
        graph
            .transfer_pass("layer 1", "d")
            .write(array.subresource(Subresource::layer(1)), Access::CopyWrite);
        graph
            .transfer_pass("whole", "e")
            .read(bloom, Access::CopyRead);
        assert_eq!(edges(&graph), [(0, 4, RAW), (1, 4, RAW)]);
    }

    /// A pass that reads and writes one subresource depends on the write
    /// before it as a read, not only as a write, and a later read depends
    /// on it rather than on that earlier write.
    #[test]
    fn a_pass_that_reads_and_writes_reads_first() {
        let (mut graph, _, _, lights) = graph();
        graph
            .transfer_pass("fill", "fill")
            .write(lights, Access::CopyWrite);
        graph
            .compute_pass("update", ComputeRecorder::recorder("update"))
            .write(lights, Access::ComputeShaderWrite)
            .read(lights, Access::ComputeShaderRead);
        graph
            .compute_pass("use", ComputeRecorder::recorder("use"))
            .read(lights, Access::ComputeShaderRead);
        let both = Hazards {
            read_after_write: true,
            write_after_write: true,
            ..NONE
        };
        assert_eq!(edges(&graph), [(0, 1, both), (1, 2, RAW)]);
    }

    /// A kept attachment reads its contents, so it depends on the write
    /// before it as a read, while a cleared one only overwrites them. The
    /// kept attachment's read is its own pass's, which wrote it last, so
    /// the clear after it is a write after a write alone.
    #[test]
    fn a_kept_attachment_reads_and_a_cleared_one_writes() {
        let (mut graph, hdr, ..) = graph();
        graph
            .raster_pass("clear", "clear")
            .colour(hdr, Attachment::Clear([0.0; 4]));
        graph
            .raster_pass("blend", "blend")
            .colour(hdr, Attachment::Keep);
        graph
            .raster_pass("cleared again", "cleared")
            .colour(hdr, Attachment::Clear([0.0; 4]));
        let both = Hazards {
            read_after_write: true,
            write_after_write: true,
            ..NONE
        };
        assert_eq!(edges(&graph), [(0, 1, both), (1, 2, WAW)]);
    }

    /// A read of a persistent texture before any write sees the last
    /// frame's contents, so the frame's first write depends on it, and an
    /// imported texture's mips are split by its uses, as a created one's
    /// are, though the graph doesn't know how many it has.
    #[test]
    fn older_contents_are_read_before_they_are_overwritten() {
        let (mut graph, ..) = graph();
        let size = TextureSize::Fixed(Extent3d::new_2d(16, 16));
        let history =
            graph.persistent_texture("history", TransientTexture::new(Format::Rgba16Float, size));
        let import = ImportDesc {
            before: Access::CopyWrite.into(),
            after: Access::CopyRead.into(),
        };
        let font = graph.import_texture("font", import);
        graph
            .transfer_pass("resolve", "a")
            .read(history, Access::CopyRead);
        graph
            .transfer_pass("accumulate", "b")
            .write(history, Access::CopyWrite);
        graph
            .transfer_pass("glyphs", "c")
            .read(font.mip(0), Access::CopyRead);
        graph.transfer_pass("upload", "d").write(
            font.subresource(Subresource {
                mips: 1..Subresource::REMAINING,
                layers: 0..1,
            }),
            Access::CopyWrite,
        );
        graph
            .transfer_pass("all glyphs", "e")
            .read(font, Access::CopyRead);
        assert_eq!(edges(&graph), [(0, 1, WAR), (3, 4, RAW)]);
    }

    /// Every dependency is found, for culling to follow, and reducing the
    /// graph drops each that a path through other passes implies, reads
    /// after writes included.
    #[test]
    fn reducing_drops_implied_dependencies() {
        let (mut graph, hdr, _, lights) = graph();
        graph
            .transfer_pass("fill", "fill")
            .write(hdr, Access::CopyWrite)
            .write(lights, Access::CopyWrite);
        // Reads both, so the overwrite of both below depends on it, which
        // implies the overwrite's dependency on the fill.
        graph
            .transfer_pass("read", "read")
            .read(hdr, Access::CopyRead)
            .read(lights, Access::CopyRead);
        graph
            .transfer_pass("overwrite", "overwrite")
            .write(hdr, Access::CopyWrite)
            .write(lights, Access::CopyWrite);
        // The overwrite may cover only part of hdr, so the late read sees
        // the fill's write too, which the path through the overwrite
        // implies.
        graph
            .transfer_pass("late read", "late")
            .read(hdr, Access::CopyRead);
        assert_eq!(
            listed(&unreduced(&graph)),
            [
                (0, 1, RAW),
                (0, 2, WAW),
                (1, 2, WAR),
                (0, 3, RAW),
                (2, 3, RAW)
            ]
        );
        assert_eq!(edges(&graph), [(0, 1, RAW), (1, 2, WAR), (2, 3, RAW)]);
    }

    /// Reducing the graph of the passes culling keeps drops the passes it
    /// doesn't, with their dependencies, so a dependency between kept
    /// passes that a path through a culled one implied is kept.
    #[test]
    fn reducing_keeps_what_culled_passes_implied() {
        let (mut graph, hdr, ..) = graph();
        graph
            .transfer_pass("first", "first")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("culled", "culled")
            .read(hdr, Access::CopyRead);
        graph
            .transfer_pass("second", "second")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("read", "read")
            .read(hdr, Access::CopyRead);
        let mut dependencies = unreduced(&graph);
        dependencies.reduce(&[true, false, true, true]);
        assert_eq!(listed(&dependencies), [(0, 2, WAW), (2, 3, RAW)]);
    }

    /// Reduction works across more passes than one word of its sets holds:
    /// in a chain of 70, pass 64's overwrite of what pass 0 read is implied
    /// by the chain, and dropped. Pass 64 is the first bit of the second
    /// word, which, read from the first word instead, would be pass 0, which
    /// pass 0 doesn't reach.
    #[test]
    fn reduction_spans_many_passes() {
        let (mut graph, hdr, _, lights) = graph();
        graph
            .transfer_pass("pass 0", "pass 0")
            .read(lights, Access::CopyRead)
            .write(hdr, Access::CopyWrite);
        for pass in 1..70 {
            let name = format!("pass {pass}");
            let mut builder = graph.transfer_pass(&name, &name);
            builder
                .read(hdr, Access::CopyRead)
                .write(hdr, Access::CopyWrite);
            if pass == 64 {
                builder.write(lights, Access::CopyWrite);
            }
        }
        let both = Hazards {
            read_after_write: true,
            write_after_write: true,
            ..NONE
        };
        let chain: Vec<_> = (1..70).map(|pass| (pass - 1, pass, both)).collect();
        assert_eq!(edges(&graph), chain);
    }

    /// Each repetition of a repeated pass reads the mip the one before it
    /// wrote, so a chain is a line of dependencies.
    #[test]
    fn a_mip_chain_is_a_line() {
        let (mut graph, _, bloom, _) = graph();
        graph
            .transfer_pass("fill", "fill")
            .write(bloom.mip(0), Access::CopyWrite);
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        assert_eq!(
            edges_with(&graph, &[None, Some(1..4)]),
            [(0, 1, RAW), (1, 2, RAW), (2, 3, RAW)]
        );
    }
}
