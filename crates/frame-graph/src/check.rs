//! The plan checker (design 0003, "Deriving barriers"): a second,
//! independent implementation of the rules a plan must follow, written for
//! clarity rather than speed.
//!
//! It knows nothing of schedules, groups or how the compiler derived the
//! barriers: only the order the plan runs its passes and records its
//! barriers in. For each mip of each layer of each texture, and each
//! buffer, it lists what happens to it in that order, then checks that:
//!
//! - every pair of uses that conflict, where either writes, is ordered by
//!   a chain of barriers, and a write is made available and visible to
//!   the use after it;
//! - every layout change, which writes the texture, is ordered after the
//!   uses before it and before the uses after it, as a write is;
//! - every use finds the texture in the layout its accesses need, and
//!   every barrier moves it from the layout it's in.
//!
//! A barrier orders what comes before it, in the stages of its accesses
//! before, against what comes after it, in the stages of its accesses
//! after. Barriers chain: a later barrier whose stages before share one
//! with an earlier barrier's stages after is ordered after it, as Vulkan's
//! execution dependencies chain. Each access is in one stage of its own,
//! which this module gives without reference to how the RHI maps them.
//!
//! Resources the graph creates are checked over two frames running the
//! plan one after the other, so a first use is checked against the last
//! frame's uses. The first frame starts from `Access::None`, as the first
//! frame after a compile does. The output and imports are checked over one
//! frame, starting in the accesses the caller gives and ending in those it
//! asks for, for the parts the frame uses.

use std::fmt;
use std::ops::Range;

use clockwork_rhi::{Access, AccessSet, Layout, Subresource};

use crate::barriers::{BarrierTarget, PlannedBarrier};
use crate::desc::{BufferKind, TextureKind};
use crate::describe::{accesses, kebab};
use crate::{GraphDesc, Plan, ResourceUse};

/// A hazard in a plan: two uses of a resource that conflict with nothing
/// ordering them, or a use that finds a texture in the wrong layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hazard {
    /// What's wrong, as a sentence without a full stop, naming the
    /// resource and the passes involved.
    message: String,
}

impl Hazard {
    /// What's wrong.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Hazard {
    /// Writes the message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// Checks that `plan`, compiled from `desc`, orders every pair of
/// conflicting uses, and that every use finds its layout.
///
/// # Errors
///
/// Returns each hazard found, once, in the order of the resources and of
/// the plan.
pub fn check(desc: &GraphDesc, plan: &Plan) -> Result<(), Vec<Hazard>> {
    let steps = steps(plan);
    let mut found = Vec::new();
    for (texture, decl) in desc.textures.iter().enumerate() {
        let frame = match &decl.kind {
            TextureKind::Output => Frame::Given(plan.output.before, plan.output.after),
            TextureKind::Imported(import) => Frame::Given(import.before, import.after),
            TextureKind::Transient(_) | TextureKind::Persistent(_) => Frame::Repeated,
        };
        let (mips, layers) = extent(plan, texture);
        for mip in 0..mips {
            for layer in 0..layers {
                let unit = Unit::Texture {
                    texture,
                    mip,
                    layer,
                };
                Resource::new(plan, &steps, unit, frame).check(&mut found);
            }
        }
    }
    for (buffer, decl) in desc.buffers.iter().enumerate() {
        let frame = match &decl.kind {
            BufferKind::Imported(import) => Frame::Given(import.before, import.after),
            BufferKind::Transient { .. } | BufferKind::Persistent { .. } => Frame::Repeated,
        };
        Resource::new(plan, &steps, Unit::Buffer(buffer), frame).check(&mut found);
    }
    let mut hazards: Vec<Hazard> = Vec::new();
    for message in found {
        if hazards.iter().all(|hazard| hazard.message != message) {
            hazards.push(Hazard { message });
        }
    }
    if hazards.is_empty() {
        Ok(())
    } else {
        Err(hazards)
    }
}

/// One thing a plan does, in the order it does them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Records the barrier at this index of the plan's.
    Barrier(usize),
    /// Runs the pass at this index of the plan's.
    Pass(usize),
}

/// What `plan` does each frame, in order: each group's barriers then its
/// passes, then the barriers at the end of the frame.
fn steps(plan: &Plan) -> Vec<Step> {
    let mut steps = Vec::new();
    for batch in &plan.batches {
        for group in &plan.groups[batch.groups.clone()] {
            steps.extend(group.barriers.clone().map(Step::Barrier));
            steps.extend(group.passes.clone().map(Step::Pass));
        }
    }
    steps.extend(plan.end_of_frame.clone().map(Step::Barrier));
    steps
}

/// The smallest part of a resource the checker follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    /// A mip of a layer of the texture at this index.
    Texture {
        /// The texture.
        texture: usize,
        /// The mip.
        mip: u32,
        /// The layer.
        layer: u32,
    },
    /// The buffer at this index.
    Buffer(usize),
}

/// How a resource's frames begin and end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frame {
    /// The graph creates it, and each frame starts where the last ended.
    Repeated,
    /// The caller gives it, in the first accesses, and it must be left in
    /// the second.
    Given(AccessSet, AccessSet),
}

/// The mips and layers of the texture at `texture`: its shape's, or for an
/// import, whose shape is the caller's, as many as the plan names.
fn extent(plan: &Plan, texture: usize) -> (u32, u32) {
    if let Some(shape) = plan.textures[texture] {
        return (shape.mips, shape.layers);
    }
    let bound = |range: &Range<u32>| {
        if range.end == Subresource::REMAINING {
            range.start + 1
        } else {
            range.end
        }
    };
    let mut extent = (1, 1);
    let mut name = |part: &Subresource| {
        extent.0 = extent.0.max(bound(&part.mips));
        extent.1 = extent.1.max(bound(&part.layers));
    };
    for planned in &plan.passes {
        for used in &plan.expanded[planned.pass].pass.uses {
            if let ResourceUse::Texture(part) = &used.resource
                && part.texture.index() == texture
            {
                name(&part.subresource);
            }
        }
    }
    for barrier in &plan.barriers {
        if let BarrierTarget::Texture(index, part) = &barrier.target
            && *index == texture
        {
            name(part);
        }
    }
    extent
}

/// Whether `range` includes `index`.
fn includes(range: &Range<u32>, index: u32) -> bool {
    range.start <= index && (range.end == Subresource::REMAINING || index < range.end)
}

/// Who uses a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum User {
    /// The pass at this index of the plan's.
    Pass(usize),
    /// The caller, before the frame.
    Start,
    /// The caller, after the frame.
    End,
}

/// What happens to a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum What {
    /// It's used with the accesses.
    Use(User, AccessSet),
    /// A barrier, from the first accesses to the second, on part of the
    /// texture if the last field says so, and otherwise on the buffer or
    /// all memory, which have no layout.
    Barrier(AccessSet, AccessSet, bool),
}

impl What {
    /// Whether it's a barrier that changes a texture's layout, which
    /// writes the texture. A barrier that discards the contents changes it
    /// from whatever it was.
    fn changes_layout(self) -> bool {
        match self {
            Self::Barrier(before, after, texture) => {
                texture && (discards(before) || layout(before) != layout(after))
            }
            Self::Use(..) => false,
        }
    }
}

/// Something that happens to a resource, in a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Event {
    /// Whether it's in the last frame, rather than the frame being checked.
    last_frame: bool,
    /// What happens.
    what: What,
}

/// One unit of a resource, and what happens to it, in order.
struct Resource<'a> {
    /// The plan.
    plan: &'a Plan,
    /// The unit.
    unit: Unit,
    /// How its frames begin and end.
    frame: Frame,
    /// What happens to it, in order.
    events: Vec<Event>,
}

impl<'a> Resource<'a> {
    /// What happens to `unit` in `plan`, which does `steps` each frame.
    fn new(plan: &'a Plan, steps: &[Step], unit: Unit, frame: Frame) -> Self {
        let mut events = Vec::new();
        let frames: &[bool] = match frame {
            Frame::Repeated => &[true, false],
            Frame::Given(before, _) => {
                if !discards(before) {
                    events.push(Event {
                        last_frame: false,
                        what: What::Use(User::Start, before),
                    });
                }
                &[false]
            }
        };
        let mut resource = Self {
            plan,
            unit,
            frame,
            events,
        };
        for &last_frame in frames {
            for &step in steps {
                let what = match step {
                    Step::Pass(index) => resource
                        .used(index)
                        .map(|set| What::Use(User::Pass(index), set)),
                    Step::Barrier(index) => resource.barrier(&plan.barriers[index], last_frame),
                };
                if let Some(what) = what {
                    resource.events.push(Event { last_frame, what });
                }
            }
        }
        let used = resource
            .events
            .iter()
            .any(|event| matches!(event.what, What::Use(User::Pass(_), _)));
        if let Frame::Given(_, after) = frame
            && used
        {
            resource.events.push(Event {
                last_frame: false,
                what: What::Use(User::End, after),
            });
        }
        resource
    }

    /// Every access the pass at `index` of the plan's uses the unit with,
    /// if it uses it.
    fn used(&self, index: usize) -> Option<AccessSet> {
        let pass = &self.plan.expanded[self.plan.passes[index].pass].pass;
        pass.uses
            .iter()
            .filter(|used| match (&used.resource, self.unit) {
                (
                    ResourceUse::Texture(part),
                    Unit::Texture {
                        texture,
                        mip,
                        layer,
                    },
                ) => {
                    part.texture.index() == texture
                        && includes(&part.subresource.mips, mip)
                        && includes(&part.subresource.layers, layer)
                }
                (ResourceUse::Buffer(buffer), Unit::Buffer(index)) => buffer.index() == index,
                _ => false,
            })
            .map(|used| used.access)
            .reduce(AccessSet::union)
    }

    /// What `barrier` does to the unit, if it covers it, in the last frame
    /// if `last_frame`, where one from the last frame's accesses is from
    /// `Access::None` instead, as in the first frame after a compile.
    fn barrier(&self, barrier: &PlannedBarrier, last_frame: bool) -> Option<What> {
        let texture = match (&barrier.target, self.unit) {
            (
                BarrierTarget::Texture(index, part),
                Unit::Texture {
                    texture,
                    mip,
                    layer,
                },
            ) => {
                if *index != texture || !includes(&part.mips, mip) || !includes(&part.layers, layer)
                {
                    return None;
                }
                true
            }
            (BarrierTarget::Buffer(index), Unit::Buffer(buffer)) if *index == buffer => false,
            (BarrierTarget::Global, _) => false,
            _ => return None,
        };
        let before = if last_frame && barrier.from_last_frame {
            Access::None.into()
        } else {
            barrier.before
        };
        Some(What::Barrier(before, barrier.after, texture))
    }

    /// Checks what happens to the unit, adding the message of each hazard
    /// to `found`.
    fn check(&self, found: &mut Vec<String>) {
        self.check_layouts(found);
        for later in 0..self.events.len() {
            if self.events[later].last_frame {
                continue;
            }
            for earlier in 0..later {
                if self.conflict(earlier, later) && !self.ordered(earlier, later) {
                    found.push(format!(
                        "{}: {} isn't ordered after {}",
                        self.name(),
                        self.describe(later),
                        self.describe(earlier)
                    ));
                }
            }
        }
    }

    /// Checks that every use finds a texture in its layout, and every
    /// barrier moves it from the layout it's in.
    fn check_layouts(&self, found: &mut Vec<String>) {
        let Unit::Texture { .. } = self.unit else {
            return;
        };
        let mut current = match self.frame {
            Frame::Repeated => Layout::Undefined,
            Frame::Given(before, _) => layout(before),
        };
        for event in &self.events {
            match event.what {
                What::Use(user, set) => {
                    if layout(set) != current && !event.last_frame {
                        found.push(format!(
                            "{}: {} needs the {} layout, but it's in {}",
                            self.name(),
                            self.user(user, set, false),
                            name(layout(set)),
                            name(current)
                        ));
                    }
                }
                What::Barrier(before, after, texture) => {
                    if !texture {
                        continue;
                    }
                    if !discards(before) && layout(before) != current && !event.last_frame {
                        found.push(format!(
                            "{}: a barrier moves it from {} ({}), but it's in {}",
                            self.name(),
                            accesses(before),
                            name(layout(before)),
                            name(current)
                        ));
                    }
                    current = layout(after);
                }
            }
        }
    }

    /// Whether the events at `earlier` and `later` conflict: two uses where
    /// either writes, or a layout change, which writes, and a use or
    /// another layout change.
    fn conflict(&self, earlier: usize, later: usize) -> bool {
        let writes = |what: What| match what {
            What::Use(_, set) => Some(set.is_write()),
            What::Barrier(..) if what.changes_layout() => Some(true),
            What::Barrier(..) => None,
        };
        match (
            writes(self.events[earlier].what),
            writes(self.events[later].what),
        ) {
            (Some(first), Some(second)) => first || second,
            _ => false,
        }
    }

    /// Whether a chain of barriers orders the event at `earlier` before the
    /// one at `later`, and makes what the earlier wrote visible to the
    /// later, if it's a use.
    ///
    /// The chain's first barrier has the earlier use's stages among its
    /// stages before, and its writes among its accesses before, so it makes
    /// them available, or is the barrier whose layout change is the earlier
    /// event. Each barrier after it shares a stage before with the stages
    /// after of a barrier already in the chain. The later use's stages are
    /// among the stages after of a barrier in the chain, and if the earlier
    /// event wrote, its accesses among that barrier's accesses after, so
    /// the write is visible to it. A later layout change must itself be in
    /// the chain.
    fn ordered(&self, earlier: usize, later: usize) -> bool {
        let first = self.events[earlier].what;
        let last = self.events[later].what;
        let start = if first.changes_layout() {
            earlier
        } else {
            earlier + 1
        };
        let end = if last.changes_layout() {
            later + 1
        } else {
            later
        };
        // Each barrier in the chain, by index, with its accesses after.
        let mut chain: Vec<(usize, AccessSet)> = Vec::new();
        for (index, event) in self.events.iter().enumerate().take(end).skip(start) {
            let What::Barrier(before, after, _) = event.what else {
                continue;
            };
            let starts = match first {
                What::Use(_, set) => {
                    stages(set) & !stages(before) == 0
                        && writes(set).into_iter().all(|write| before.contains(write))
                }
                What::Barrier(..) => index == earlier,
            };
            let follows = chain
                .iter()
                .any(|&(_, end)| stages(end) & stages(before) != 0);
            if starts || follows {
                chain.push((index, after));
            }
        }
        match last {
            What::Barrier(..) => chain.iter().any(|&(index, _)| index == later),
            What::Use(_, set) => {
                let wrote = match first {
                    What::Use(_, earlier) => earlier.is_write(),
                    What::Barrier(..) => true,
                };
                chain.iter().any(|&(_, after)| {
                    stages(set) & !stages(after) == 0
                        && (!wrote || set.iter().all(|access| after.contains(access)))
                })
            }
        }
    }

    /// The unit's name, in messages: its resource's, and for a texture with
    /// more than one, its mip and layer.
    fn name(&self) -> String {
        match self.unit {
            Unit::Texture {
                texture,
                mip,
                layer,
            } => {
                let (mips, layers) = extent(self.plan, texture);
                let mut name = self.plan.texture_names[texture].clone();
                if mips > 1 {
                    name.push_str(&format!(" mip {mip}"));
                }
                if layers > 1 {
                    name.push_str(&format!(" layer {layer}"));
                }
                name
            }
            Unit::Buffer(buffer) => self.plan.buffer_names[buffer].clone(),
        }
    }

    /// The event at `index`, in messages.
    fn describe(&self, index: usize) -> String {
        let event = self.events[index];
        match event.what {
            What::Use(user, set) => self.user(user, set, event.last_frame),
            What::Barrier(_, after, _) => {
                let frame = if event.last_frame {
                    "the last frame's "
                } else {
                    ""
                };
                format!("{frame}layout change to {}", name(layout(after)))
            }
        }
    }

    /// `user`'s use with `set`, in messages, in the last frame if
    /// `last_frame`.
    fn user(&self, user: User, set: AccessSet, last_frame: bool) -> String {
        let frame = if last_frame { "the last frame's " } else { "" };
        match user {
            User::Pass(index) => {
                let pass = &self.plan.expanded[self.plan.passes[index].pass].pass;
                format!("{frame}pass \"{}\" ({})", pass.name, accesses(set))
            }
            User::Start => format!("the accesses it starts the frame in ({})", accesses(set)),
            User::End => format!("the accesses it must end the frame in ({})", accesses(set)),
        }
    }
}

/// Whether a barrier from `set` discards a texture's contents, from any
/// layout, ordering nothing before it.
fn discards(set: AccessSet) -> bool {
    set == Access::None.into() || set == Access::Acquired.into()
}

/// The layout a texture is in for `set`.
fn layout(set: AccessSet) -> Layout {
    set.layout().unwrap_or(Layout::Undefined)
}

/// `layout`'s name, in messages.
fn name(layout: Layout) -> String {
    kebab(&format!("{layout:?}"))
}

/// The writes among `set`'s accesses.
fn writes(set: AccessSet) -> Vec<Access> {
    set.iter().filter(|access| access.is_write()).collect()
}

/// The stages `set`'s accesses happen in, as a mask, each access in one
/// stage of its own. `None` and `Acquired` happen in none.
fn stages(set: AccessSet) -> u16 {
    set.iter().map(stage).fold(0, |mask, stage| mask | stage)
}

/// The stage `access` happens in, as a bit, or 0 for none.
fn stage(access: Access) -> u16 {
    /// Reading indirect arguments.
    const INDIRECT: u16 = 1;
    /// Reading indices.
    const INDEX: u16 = 1 << 1;
    /// Vertex shading.
    const VERTEX: u16 = 1 << 2;
    /// Depth testing, early and late.
    const DEPTH_TESTS: u16 = 1 << 3;
    /// Fragment shading.
    const FRAGMENT: u16 = 1 << 4;
    /// Writing colour attachments.
    const COLOUR: u16 = 1 << 5;
    /// Compute shading.
    const COMPUTE: u16 = 1 << 6;
    /// Copies.
    const COPY: u16 = 1 << 7;
    /// The CPU.
    const HOST: u16 = 1 << 8;
    /// Presenting, after everything.
    const PRESENT: u16 = 1 << 9;
    match access {
        Access::None | Access::Acquired => 0,
        Access::IndirectRead => INDIRECT,
        Access::IndexRead => INDEX,
        Access::VertexShaderRead => VERTEX,
        Access::FragmentShaderRead | Access::FragmentShaderSampledRead => FRAGMENT,
        Access::ComputeShaderRead
        | Access::ComputeShaderSampledRead
        | Access::ComputeShaderWrite => COMPUTE,
        Access::ColorAttachmentRead | Access::ColorAttachmentWrite => COLOUR,
        Access::DepthAttachmentRead | Access::DepthAttachmentWrite => DEPTH_TESTS,
        Access::CopyRead | Access::CopyWrite => COPY,
        Access::HostRead | Access::HostWrite => HOST,
        Access::Present => PRESENT,
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the plan checker, on compiled plans and on plans broken
    //! on purpose.

    use clockwork_rhi::{Extent3d, Format};

    use super::*;
    use crate::validate::test_support::{finished, inputs};
    use crate::{
        Attachment, ComputeRecorder, ImportDesc, MipCount, MipDirection, TextureSize,
        TransientTexture, compile,
    };

    /// A texture of `format`, 256 × 256.
    fn texture(format: Format) -> TransientTexture {
        TransientTexture::new(format, TextureSize::Fixed(Extent3d::new_2d(256, 256)))
    }

    /// A bloom chain: a scene, its bright parts, downsampled and upsampled
    /// mip by mip, and composited into the output.
    fn bloom() -> GraphDesc {
        let mut graph = GraphDesc::new("bloom");
        let scene = graph.transient_texture("scene", texture(Format::Rgba16Float));
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..texture(Format::Rgba16Float)
        };
        let bloom = graph.transient_texture("bloom", chain);
        let output = graph.output();
        graph
            .raster_pass("scene", "scene")
            .colour(scene, Attachment::Clear([0.0; 4]));
        graph
            .compute_pass("bright", ComputeRecorder::dispatch("bright"))
            .bind(scene, Access::ComputeShaderSampledRead)
            .bind(bloom.mip(0), Access::ComputeShaderWrite);
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        graph
            .compute_pass("upsample", ComputeRecorder::dispatch("upsample"))
            .for_each_mip(bloom, MipDirection::Up, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(
                bloom.current_mip(),
                Access::ComputeShaderRead | Access::ComputeShaderWrite,
            );
        graph
            .fullscreen_pass("composite", "composite")
            .bind(scene, Access::FragmentShaderSampledRead)
            .bind(bloom.mip(0), Access::FragmentShaderSampledRead)
            .colour(output, Attachment::Discard);
        graph
    }

    /// Depth tested by two passes, a persistent texture read before it's
    /// written, buffers, and an import written and left for the caller.
    fn mixed() -> GraphDesc {
        let mut graph = GraphDesc::new("mixed");
        let depth = graph.transient_texture("depth", texture(Format::Depth32Float));
        let colour = graph.transient_texture("colour", texture(Format::Rgba16Float));
        let history = graph.persistent_texture("history", texture(Format::Rgba16Float));
        let target = graph.import_texture(
            "target",
            ImportDesc {
                before: Access::CopyRead.into(),
                after: Access::FragmentShaderSampledRead.into(),
            },
        );
        let lights = graph.transient_buffer("lights", 64);
        let readback = graph.import_buffer(
            "readback",
            ImportDesc {
                before: Access::HostRead.into(),
                after: Access::HostRead.into(),
            },
        );
        graph
            .compute_pass("cull", ComputeRecorder::recorder("cull"))
            .write(lights, Access::ComputeShaderWrite);
        graph
            .raster_pass("prepass", "prepass")
            .depth(depth, Attachment::Clear(0.0));
        graph
            .raster_pass("shade", "shade")
            .read(lights, Access::FragmentShaderRead)
            .read(history, Access::FragmentShaderSampledRead)
            .colour(colour, Attachment::Clear([0.0; 4]))
            .depth(depth, Attachment::Keep);
        graph
            .transfer_pass("resolve", "resolve")
            .read(colour, Access::CopyRead)
            .write(history, Access::CopyWrite)
            .write(target, Access::CopyWrite)
            .write(readback, Access::CopyWrite);
        finished(&graph)
    }

    /// `plan` with its barrier at `index` left out.
    fn without(plan: &Plan, index: usize) -> Plan {
        let mut plan = plan.clone();
        plan.barriers.remove(index);
        let shift = |range: &mut Range<usize>| {
            if range.start > index {
                range.start -= 1;
            }
            if range.end > index {
                range.end -= 1;
            }
        };
        for group in &mut plan.groups {
            shift(&mut group.barriers);
        }
        shift(&mut plan.end_of_frame);
        plan
    }

    /// The messages of the hazards `check` finds in `plan`, of `desc`.
    fn hazards(desc: &GraphDesc, plan: &Plan) -> Vec<String> {
        match check(desc, plan) {
            Ok(()) => Vec::new(),
            Err(hazards) => hazards
                .iter()
                .map(|hazard| hazard.message().to_owned())
                .collect(),
        }
    }

    /// The compiler's plans have no hazards.
    #[test]
    fn compiled_plans_have_no_hazards() {
        for desc in [bloom(), mixed()] {
            let plan = compile(&desc, &inputs()).expect("the graph is valid");
            assert_eq!(
                hazards(&desc, &plan),
                Vec::<String>::new(),
                "{}",
                plan.describe()
            );
        }
    }

    /// Every barrier the compiler records is needed: with any one left out,
    /// the checker finds a hazard.
    #[test]
    fn a_plan_missing_any_barrier_has_hazards() {
        for desc in [bloom(), mixed()] {
            let plan = compile(&desc, &inputs()).expect("the graph is valid");
            for index in 0..plan.barriers.len() {
                assert!(
                    !hazards(&desc, &without(&plan, index)).is_empty(),
                    "barrier {index} of {}:\n{}",
                    desc.label(),
                    plan.describe()
                );
            }
        }
    }

    /// The index in `plan` of the first barrier on the texture named
    /// `name`, from `before`.
    fn barrier_on(plan: &Plan, name: &str, before: AccessSet) -> usize {
        plan.barriers
            .iter()
            .position(|barrier| {
                matches!(barrier.target, BarrierTarget::Texture(texture, _)
                    if plan.texture_names[texture] == name)
                    && barrier.before == before
            })
            .expect("the plan has the barrier")
    }

    /// A read after a write with no barrier between them is unordered, and
    /// finds the texture in the write's layout.
    #[test]
    fn a_missing_barrier_leaves_a_read_unordered() {
        let desc = mixed();
        let plan = compile(&desc, &inputs()).expect("the graph is valid");
        let index = barrier_on(&plan, "colour", Access::ColorAttachmentWrite.into());
        let found = hazards(&desc, &without(&plan, index));
        for expected in [
            "colour: pass \"resolve\" (copy-read) needs the copy-source layout, but it's in \
             attachment",
            "colour: pass \"resolve\" (copy-read) isn't ordered after pass \"shade\" \
             (color-attachment-write)",
        ] {
            assert!(found.contains(&expected.to_owned()), "{found:#?}");
        }
    }

    /// A barrier whose accesses after are in another stage than the use
    /// after it orders nothing for that use.
    #[test]
    fn a_barrier_in_the_wrong_stage_orders_nothing() {
        let desc = mixed();
        let mut plan = compile(&desc, &inputs()).expect("the graph is valid");
        let index = barrier_on(&plan, "colour", Access::ColorAttachmentWrite.into());
        // Sampling, in the copy's place, is in the fragment shader rather
        // than the copy, and moves the texture to the read-only layout, so
        // the copy finds the wrong layout too.
        plan.barriers[index].after = Access::FragmentShaderSampledRead.into();
        let found = hazards(&desc, &plan);
        assert!(
            found.contains(
                &"colour: pass \"resolve\" (copy-read) isn't ordered after pass \"shade\" \
                  (color-attachment-write)"
                    .to_owned()
            ),
            "{found:#?}"
        );
    }

    /// A barrier from accesses of another layout than the texture is in
    /// moves it from the wrong layout.
    #[test]
    fn a_barrier_must_move_from_the_layout_a_texture_is_in() {
        let desc = mixed();
        let mut plan = compile(&desc, &inputs()).expect("the graph is valid");
        let index = barrier_on(&plan, "colour", Access::ColorAttachmentWrite.into());
        plan.barriers[index].before = Access::CopyWrite.into();
        let found = hazards(&desc, &plan);
        assert!(
            found.contains(
                &"colour: a barrier moves it from copy-write (copy-destination), but it's in \
                  attachment"
                    .to_owned()
            ),
            "{found:#?}"
        );
    }

    /// The first use in a frame of a persistent texture is unordered after
    /// the last frame's write without its barrier.
    #[test]
    fn a_first_use_needs_ordering_after_the_last_frame() {
        let desc = mixed();
        let plan = compile(&desc, &inputs()).expect("the graph is valid");
        let index = barrier_on(&plan, "history", Access::CopyWrite.into());
        let found = hazards(&desc, &without(&plan, index));
        assert!(
            found.contains(
                &"history: pass \"shade\" (fragment-shader-sampled-read) isn't ordered after the \
                  last frame's pass \"resolve\" (copy-write)"
                    .to_owned()
            ),
            "{found:#?}"
        );
    }

    /// An import must be left in the accesses the caller asks for, after
    /// the frame's last use of it and its last layout change.
    #[test]
    fn an_import_must_end_the_frame_as_asked() {
        let desc = mixed();
        let plan = compile(&desc, &inputs()).expect("the graph is valid");
        let index = barrier_on(&plan, "target", Access::CopyWrite.into());
        assert_eq!(
            hazards(&desc, &without(&plan, index)),
            [
                "target: the accesses it must end the frame in (fragment-shader-sampled-read) \
                 needs the read-only layout, but it's in copy-destination",
                "target: the accesses it must end the frame in (fragment-shader-sampled-read) \
                 isn't ordered after layout change to copy-destination",
                "target: the accesses it must end the frame in (fragment-shader-sampled-read) \
                 isn't ordered after pass \"resolve\" (copy-write)",
            ]
        );
    }

    /// Hazards display as their messages.
    #[test]
    fn hazards_display_their_message() {
        let hazard = Hazard {
            message: "hdr: a hazard".to_owned(),
        };
        assert_eq!(hazard.to_string(), "hdr: a hazard");
    }
}
