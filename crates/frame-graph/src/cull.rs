//! Culling, the fourth step of compiling (design 0003, "Compiling"): a pass
//! whose writes nothing reads is left out of the plan, unless it's marked as
//! having side effects.

use tracing::debug;

use crate::GraphDesc;
use crate::dependencies::Dependencies;
use crate::desc::{BufferKind, TextureKind};
use crate::expand::ExpandedPass;

/// Which of `passes`, the expanded passes of `desc`, whose dependency graph
/// is `dependencies`, not yet reduced, are kept, by their indices.
///
/// The passes kept to start with are those with side effects, and those
/// whose writes outlast the frame: those still visible when it ends, of the
/// output, of imported resources, which the caller uses after the frame,
/// and of persistent resources, which the next frame reads. A write that
/// may cover only part of a subresource leaves the writes before it
/// visible, so they're kept too. Then a pass is kept if a kept pass reads
/// something it writes, which the graph shows before it's reduced. Each
/// pass culled is logged at `DEBUG`, since a pass culled because, say, a
/// debug view isn't shown is normal.
pub(crate) fn cull(
    desc: &GraphDesc,
    passes: &[ExpandedPass],
    dependencies: &Dependencies,
) -> Vec<bool> {
    let mut pending: Vec<usize> = passes
        .iter()
        .enumerate()
        .filter(|(_, expanded)| expanded.pass.side_effects)
        .map(|(index, _)| index)
        .collect();
    let textures = desc.textures.iter().zip(&dependencies.last_texture_writers);
    for (texture, writers) in textures {
        let outlasts = match texture.kind {
            TextureKind::Output | TextureKind::Persistent(_) | TextureKind::Imported(_) => true,
            TextureKind::Transient(_) => false,
        };
        if outlasts {
            pending.extend(writers);
        }
    }
    let buffers = desc.buffers.iter().zip(&dependencies.last_buffer_writers);
    for (buffer, writers) in buffers {
        let outlasts = match buffer.kind {
            BufferKind::Persistent { .. } | BufferKind::Imported(_) => true,
            BufferKind::Transient { .. } => false,
        };
        if outlasts {
            pending.extend(writers);
        }
    }

    let mut kept = vec![false; passes.len()];
    while let Some(pass) = pending.pop() {
        if std::mem::replace(&mut kept[pass], true) {
            continue;
        }
        let reads = dependencies.on[pass]
            .iter()
            .filter(|dependency| dependency.hazards.read_after_write);
        pending.extend(reads.map(|dependency| dependency.on));
    }
    for (expanded, _) in passes.iter().zip(&kept).filter(|(_, kept)| !**kept) {
        debug!(
            pass = expanded.pass.name,
            "culled pass \"{}\", since nothing reads what it writes", expanded.pass.name
        );
    }
    kept
}

#[cfg(test)]
mod tests {
    //! T0 tests of culling.

    use clockwork_rhi::{Access, Extent3d, Format};

    use super::*;
    use crate::dependencies::dependencies;
    use crate::validate::test_support::{finished, inputs};
    use crate::validate::validate;
    use crate::{
        Attachment, ComputeRecorder, ImportDesc, MipCount, MipDirection, TextureId, TextureSize,
        TransientTexture,
    };

    /// A graph with a transient texture "hdr" and a four-mip transient
    /// texture "bloom".
    fn graph() -> (GraphDesc, TextureId, TextureId) {
        let mut graph = GraphDesc::new("cull");
        let size = TextureSize::Fixed(Extent3d::new_2d(256, 256));
        let hdr = graph.transient_texture("hdr", TransientTexture::new(Format::Rgba16Float, size));
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..TransientTexture::new(Format::Rgba16Float, size)
        };
        let bloom = graph.transient_texture("bloom", chain);
        (graph, hdr, bloom)
    }

    /// The names of the passes culled from `desc`, a valid description, in
    /// order.
    fn culled(desc: &GraphDesc) -> Vec<String> {
        let validated = validate(desc, &inputs()).expect("the graph is valid");
        let dependencies = dependencies(desc, &validated.passes);
        let kept = cull(desc, &validated.passes, &dependencies);
        validated
            .passes
            .iter()
            .zip(kept)
            .filter(|(_, kept)| !kept)
            .map(|(expanded, _)| expanded.pass.name.clone())
            .collect()
    }

    /// A pass is kept if what it writes reaches the output's final contents
    /// through reads, and culled if nothing reads it, or what's read is a
    /// write that replaced it. A write of the output that a later write
    /// replaces is culled too, but one that a later write may cover only
    /// part of is kept.
    #[test]
    fn passes_nothing_reads_are_culled() {
        let (mut graph, hdr, bloom) = graph();
        let output = graph.output();
        graph
            .raster_pass("overwritten", "a")
            .colour(hdr, Attachment::Clear([0.0; 4]));
        graph
            .raster_pass("scene", "b")
            .colour(hdr, Attachment::Clear([0.0; 4]));
        graph
            .transfer_pass("unread", "c")
            .write(bloom, Access::CopyWrite);
        graph
            .raster_pass("replaced", "d")
            .colour(output, Attachment::Clear([0.0; 4]));
        graph
            .raster_pass("present", "e")
            .read(hdr, Access::FragmentShaderSampledRead)
            .colour(output, Attachment::Discard);
        // A copy may cover only part of the output, so what "present" drew
        // stays visible beside it.
        graph
            .transfer_pass("overlay", "f")
            .write(output, Access::CopyWrite);
        assert_eq!(culled(&graph), ["overwritten", "unread", "replaced"]);
    }

    /// A pass with side effects is kept, and so are the passes whose writes
    /// it reads.
    #[test]
    fn side_effects_keep_a_pass_and_its_inputs() {
        let (mut graph, hdr, _) = graph();
        graph
            .transfer_pass("scene", "a")
            .write(hdr, Access::CopyWrite);
        graph
            .compute_pass("readback", ComputeRecorder::recorder("readback"))
            .read(hdr, Access::ComputeShaderSampledRead)
            .side_effects();
        assert_eq!(culled(&finished(&graph)), Vec::<String>::new());
    }

    /// The last writes of persistent and imported resources are kept, since
    /// they outlast the frame, while a transient resource's aren't.
    #[test]
    fn writes_that_outlast_the_frame_are_kept() {
        let (mut graph, hdr, _) = graph();
        let size = TextureSize::Fixed(Extent3d::new_2d(16, 16));
        let history =
            graph.persistent_texture("history", TransientTexture::new(Format::Rgba16Float, size));
        let counters = graph.persistent_buffer("counters", 16);
        let import = ImportDesc {
            before: Access::CopyRead.into(),
            after: Access::CopyRead.into(),
        };
        let readback = graph.import_buffer("readback", import);
        graph
            .transfer_pass("history", "a")
            .write(history, Access::CopyWrite);
        graph
            .transfer_pass("counters", "b")
            .write(counters, Access::CopyWrite);
        graph
            .transfer_pass("readback", "c")
            .write(readback, Access::CopyWrite);
        graph
            .transfer_pass("transient", "d")
            .write(hdr, Access::CopyWrite);
        assert_eq!(culled(&finished(&graph)), ["transient"]);
    }

    /// Each repetition of a repeated pass is culled or kept on its own, so
    /// a chain's mips that nothing reads aren't made.
    #[test]
    fn repetitions_are_culled_on_their_own() {
        let (mut graph, _, bloom) = graph();
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
            .bind(bloom.mip(2), Access::ComputeShaderSampledRead)
            .side_effects();
        assert_eq!(culled(&finished(&graph)), ["downsample (mip 3)"]);
    }

    /// Two kept passes that both write a texture stay ordered once the pass
    /// between them is culled, since the kept passes' graph is built after
    /// culling, so the culled pass's path doesn't stand in for their
    /// dependency.
    #[test]
    fn kept_passes_stay_ordered_around_a_culled_one() {
        let (mut graph, hdr, bloom) = graph();
        let output = graph.output();
        graph
            .transfer_pass("first", "a")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("culled", "b")
            .read(hdr, Access::CopyRead)
            .write(bloom, Access::CopyWrite);
        graph
            .transfer_pass("second", "c")
            .write(hdr, Access::CopyWrite);
        graph
            .transfer_pass("present", "d")
            .read(hdr, Access::CopyRead)
            .write(output, Access::CopyWrite);
        let desc = finished(&graph);
        let validated = validate(&desc, &inputs()).expect("the graph is valid");
        let dependencies = dependencies(&desc, &validated.passes);
        let kept = cull(&desc, &validated.passes, &dependencies);
        assert_eq!(kept, [true, false, true, true, true]);

        let dependencies = Dependencies::of_kept(&desc, &validated.passes, &kept);
        let second: Vec<_> = dependencies.on[2]
            .iter()
            .map(|dependency| (dependency.on, dependency.hazards.write_after_write))
            .collect();
        assert_eq!(second, [(0, true)]);
    }
}
