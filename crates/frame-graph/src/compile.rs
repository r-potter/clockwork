//! Compiling (design 0003, "Compiling"): a description and its inputs into
//! a [`Plan`], by each step in turn.
//!
//! The steps this slice has run here: validation and expansion, the
//! dependency graph, culling, scheduling, barriers, and load and store
//! operations. Queue assignment and batching come with `graph-queues`,
//! and placing transient resources with `graph-aliasing`. Until then,
//! every pass runs on the graphics queue, which can run every kind, in one
//! batch, so a pass that asks for the compute or transfer queue still runs,
//! only not alongside the graphics work.

use std::ops::Range;

use clockwork_rhi::QueueId;

use crate::barriers::{PlannedBarrier, barriers};
use crate::cull::cull;
use crate::dependencies::dependencies;
use crate::expand::ExpandedPass;
use crate::render_passes::{PlannedRenderPass, render_passes};
use crate::schedule::schedule;
use crate::validate::{TextureShape, validate};
use crate::{CompileErrors, CompileInputs, GraphDesc};

/// A compiled frame graph: what runs each frame, in order, with every
/// barrier and render pass worked out. It's plain data, and the same every
/// frame until something it was compiled from changes.
///
/// It's flat: batches, each a range of the groups, each a range of the
/// barriers recorded before it and a range of the passes it runs, so
/// executing it walks it in order and allocates nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// The graph's label.
    pub(crate) label: String,
    /// Each texture's shape, indexed by `TextureId`, or `None` for an
    /// import, whose shape is the caller's.
    pub(crate) textures: Vec<Option<TextureShape>>,
    /// The description's passes, with each pass repeated for each mip
    /// expanded into its repetitions, culled or not.
    pub(crate) expanded: Vec<ExpandedPass>,
    /// The batches, in the order they're submitted.
    pub(crate) batches: Vec<Batch>,
    /// The groups, in the order they run.
    pub(crate) groups: Vec<Group>,
    /// The barriers, in the order they're recorded.
    pub(crate) barriers: Vec<PlannedBarrier>,
    /// The passes that run, in the order they run.
    pub(crate) passes: Vec<PlannedPass>,
    /// The range of `barriers` recorded after the last group.
    pub(crate) end_of_frame: Range<usize>,
}

/// A run of groups on one queue, submitted together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Batch {
    /// The queue it's submitted to.
    pub(crate) queue: QueueId,
    /// Its groups, a range of the plan's.
    pub(crate) groups: Range<usize>,
}

/// Consecutive passes with no dependencies between them, whose barriers are
/// recorded together before the first of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Group {
    /// The barriers recorded before its first pass, a range of the plan's.
    pub(crate) barriers: Range<usize>,
    /// Its passes, a range of the plan's.
    pub(crate) passes: Range<usize>,
}

/// A pass that runs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlannedPass {
    /// The pass, by its index in the plan's expanded passes.
    pub(crate) pass: usize,
    /// The render pass it begins, if it has attachments.
    pub(crate) render_pass: Option<PlannedRenderPass>,
}

/// Compiles `desc` into a plan for `inputs`.
///
/// # Errors
///
/// Returns every mistake found in `desc`, each naming the passes and
/// resources involved.
pub fn compile(desc: &GraphDesc, inputs: &CompileInputs) -> Result<Plan, CompileErrors> {
    let validated = validate(desc, inputs)?;
    let expanded = validated.passes;
    let dependencies = dependencies(desc, &expanded);
    let kept = cull(desc, &expanded, &dependencies);
    let scheduled = schedule(inputs.schedule, &kept);
    let derived = barriers(
        desc,
        &inputs.output,
        &validated.textures,
        &expanded,
        &scheduled,
    );
    let render_passes = render_passes(desc, &expanded, &scheduled);

    let groups: Vec<_> = scheduled
        .groups
        .iter()
        .zip(derived.groups)
        .map(|(passes, barriers)| Group {
            barriers,
            passes: passes.clone(),
        })
        .collect();
    let passes = scheduled
        .passes
        .iter()
        .zip(render_passes)
        .map(|(&pass, render_pass)| PlannedPass { pass, render_pass })
        .collect();
    let batches = if groups.is_empty() {
        Vec::new()
    } else {
        vec![Batch {
            queue: inputs.queues.graphics,
            groups: 0..groups.len(),
        }]
    };
    Ok(Plan {
        label: desc.label().to_owned(),
        textures: validated.textures,
        expanded,
        batches,
        groups,
        barriers: derived.barriers,
        passes,
        end_of_frame: derived.end_of_frame,
    })
}

#[cfg(test)]
mod tests {
    //! T0 tests of compiling.

    use clockwork_rhi::{Access, Extent3d, Format};

    use super::*;
    use crate::barriers::BarrierTarget;
    use crate::validate::test_support::{OUTPUT_WRITER, finished, inputs};
    use clockwork_rhi::{QueueKind, Subresource};

    use crate::{Attachment, ComputeRecorder, QueueLayout, TextureSize, TransientTexture};

    /// The names of the passes `plan` runs, in order.
    fn names(plan: &Plan) -> Vec<&str> {
        let name = |planned: &PlannedPass| plan.expanded[planned.pass].pass.name.as_str();
        plan.passes.iter().map(name).collect()
    }

    /// A graph that draws a scene, culls a pass nothing reads, and copies
    /// the scene to the output.
    fn scene() -> GraphDesc {
        let mut graph = GraphDesc::new("scene");
        let size = TextureSize::Fixed(Extent3d::new_2d(256, 256));
        let scene =
            graph.transient_texture("scene", TransientTexture::new(Format::Rgba16Float, size));
        let unread =
            graph.transient_texture("unread", TransientTexture::new(Format::Rgba16Float, size));
        graph
            .raster_pass("draw", "draw")
            .colour(scene, Attachment::Clear([0.0; 4]));
        graph
            .transfer_pass("unread", "unread")
            .write(unread, Access::CopyWrite);
        graph
            .compute_pass("measure", ComputeRecorder::recorder("measure"))
            .read(scene, Access::ComputeShaderSampledRead)
            .side_effects();
        finished(&graph)
    }

    /// Compiling runs the steps in turn: the kept passes run in the
    /// written order, each in a group of its own with the barriers before
    /// it, in one batch on the graphics queue, and the raster pass begins
    /// its render pass.
    #[test]
    fn compiling_runs_each_step() {
        let inputs = inputs();
        let plan = compile(&scene(), &inputs).expect("the graph is valid");

        assert_eq!(plan.label, "scene");
        assert_eq!(names(&plan), ["draw", "measure", OUTPUT_WRITER]);
        assert_eq!(
            plan.batches,
            [Batch {
                queue: inputs.queues.graphics,
                groups: 0..3,
            }]
        );
        let passes: Vec<_> = plan
            .groups
            .iter()
            .map(|group| group.passes.clone())
            .collect();
        assert_eq!(passes, [0..1, 1..2, 2..3]);
        // Each group's barriers follow the last group's.
        let mut next = 0;
        for group in &plan.groups {
            assert_eq!(group.barriers.start, next);
            next = group.barriers.end;
        }
        assert_eq!(plan.end_of_frame, next..plan.barriers.len());
        // The output moves to `Present` at the end of the frame.
        let last = &plan.barriers[plan.end_of_frame.clone()];
        assert_eq!(last.len(), 1);
        assert_eq!(
            last[0].target,
            BarrierTarget::Texture(0, Subresource::mip_layer(0, 0))
        );
        assert_eq!(last[0].after, Access::Present.into());

        let render_pass = plan.passes[0].render_pass.as_ref().expect("draw renders");
        assert_eq!(render_pass.colour.len(), 1);
        assert!(
            plan.passes[1..]
                .iter()
                .all(|pass| pass.render_pass.is_none())
        );
    }

    /// A graph with mistakes compiles to every mistake validation finds.
    #[test]
    fn mistakes_are_reported() {
        let mut graph = GraphDesc::new("mistakes");
        let size = TextureSize::Fixed(Extent3d::new_2d(16, 16));
        let hdr = graph.transient_texture("hdr", TransientTexture::new(Format::Rgba16Float, size));
        graph
            .compute_pass("measure", ComputeRecorder::recorder("measure"))
            .read(hdr, Access::ComputeShaderSampledRead)
            .side_effects();

        let errors = compile(&graph, &inputs()).expect_err("the graph has mistakes");
        let messages: Vec<_> = errors
            .errors()
            .iter()
            .map(|error| error.message())
            .collect();
        assert_eq!(
            messages,
            [
                "pass \"measure\" reads \"hdr\", which no earlier pass writes",
                "no pass writes the output",
            ]
        );
    }

    /// Until queues are assigned, a pass that asks for the compute queue
    /// runs on the graphics queue, even where the compute queue is
    /// dedicated.
    #[test]
    fn every_pass_runs_on_the_graphics_queue() {
        let mut graph = GraphDesc::new("queues");
        let lights = graph.transient_buffer("lights", 64);
        graph
            .compute_pass("cull", ComputeRecorder::recorder("cull"))
            .queue(QueueKind::Compute)
            .write(lights, Access::ComputeShaderWrite);
        graph
            .compute_pass("light", ComputeRecorder::recorder("light"))
            .read(lights, Access::ComputeShaderRead)
            .side_effects();
        let graphics = QueueId::new(0);
        let inputs = CompileInputs {
            queues: QueueLayout {
                graphics,
                compute: QueueId::new(1),
                transfer: QueueId::new(2),
            },
            ..inputs()
        };

        let plan = compile(&finished(&graph), &inputs).expect("the graph is valid");
        assert_eq!(plan.batches.len(), 1);
        assert_eq!(plan.batches[0].queue, graphics);
        assert_eq!(names(&plan), ["cull", "light", OUTPUT_WRITER]);
    }

    /// Plans and their inputs are plain data, shared freely between
    /// threads, as design 0003's "Threading" says.
    #[test]
    fn plans_and_inputs_are_send_and_sync() {
        /// Compiles only if `T` is `Send` and `Sync`.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Plan>();
        assert_send_sync::<CompileInputs>();
    }
}
