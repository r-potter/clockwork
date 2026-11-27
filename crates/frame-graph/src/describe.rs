//! Readable plans (design 0003, "Readable plans" and "Data structures and
//! layout"): a plan as text, one line per batch, group, barrier and pass,
//! in the order they run, so a snapshot's diff shows which barrier changed
//! or which pass moved.

use std::ops::Range;

use clockwork_rhi::{AccessSet, LoadOp, QueueKind, StoreOp, Subresource};

use crate::Plan;
use crate::barriers::{BarrierTarget, PlannedBarrier};
use crate::render_passes::{PlannedAttachment, PlannedRenderPass};

impl Plan {
    /// A readable description of the plan, for snapshot tests and
    /// debugging: each batch and the queue it's submitted to, each group's
    /// barriers and passes, each render pass's load and store operations,
    /// and the barriers at the end of the frame.
    ///
    /// Accesses are written as their names are in the RHI, in lower case
    /// with hyphens, such as `compute-shader-write`. A barrier names the
    /// mips and layers it covers, unless it covers the whole texture. One
    /// that orders a resource's first use in a frame after the last frame
    /// says so, since the first frame after a compile has none.
    pub fn describe(&self) -> String {
        let mut lines = Vec::new();
        for (number, batch) in self.batches.iter().enumerate() {
            lines.push(format!("batch {number} on {}", queue(batch.kind)));
            for (number, group) in self.groups[batch.groups.clone()].iter().enumerate() {
                lines.push(format!("  group {number}"));
                for barrier in &self.barriers[group.barriers.clone()] {
                    lines.push(format!("    {}", self.barrier(barrier)));
                }
                for planned in &self.passes[group.passes.clone()] {
                    let pass = &self.expanded[planned.pass].pass;
                    lines.push(format!("    pass \"{}\" ({})", pass.name, pass.kind.name()));
                    if let Some(render_pass) = &planned.render_pass {
                        lines.push(format!(
                            "      render pass: {}",
                            self.render_pass(render_pass)
                        ));
                    }
                }
            }
        }
        lines.push("  end of frame".to_owned());
        for barrier in &self.barriers[self.end_of_frame.clone()] {
            lines.push(format!("    {}", self.barrier(barrier)));
        }
        let mut text = lines.join("\n");
        text.push('\n');
        text
    }

    /// `barrier`, as a line of [`describe`](Self::describe).
    fn barrier(&self, barrier: &PlannedBarrier) -> String {
        let target = match &barrier.target {
            BarrierTarget::Texture(texture, part) => {
                format!(
                    "{}{}",
                    self.texture_names[*texture],
                    self.part(*texture, part)
                )
            }
            BarrierTarget::Buffer(buffer) => self.buffer_names[*buffer].clone(),
            BarrierTarget::Global => "all memory".to_owned(),
        };
        let carried = if barrier.from_last_frame {
            " (after the last frame)"
        } else {
            ""
        };
        format!(
            "barrier {target}: {} -> {}{carried}",
            accesses(barrier.before),
            accesses(barrier.after)
        )
    }

    /// The mips and layers `part` of the texture at `texture` covers, with
    /// a space before them, or nothing if it covers the whole texture.
    fn part(&self, texture: usize, part: &Subresource) -> String {
        let shape = self.textures[texture];
        let mips = range("mip", &part.mips, shape.map(|shape| shape.mips));
        let layers = range("layer", &part.layers, shape.map(|shape| shape.layers));
        mips.into_iter()
            .chain(layers)
            .map(|axis| format!(" {axis}"))
            .collect()
    }

    /// `render_pass`'s attachments, with their load and store operations.
    fn render_pass(&self, render_pass: &PlannedRenderPass) -> String {
        let colour = render_pass.colour.iter().map(|attachment| {
            let load = match attachment.load {
                LoadOp::Clear([r, g, b, a]) => format!("clear ({r}, {g}, {b}, {a})"),
                LoadOp::Load => "load".to_owned(),
                LoadOp::DontCare => "don't care".to_owned(),
            };
            self.attachment("colour", attachment, &load)
        });
        let depth = render_pass.depth.iter().map(|attachment| {
            let load = match attachment.load {
                LoadOp::Clear(depth) => format!("clear {depth}"),
                LoadOp::Load => "load".to_owned(),
                LoadOp::DontCare => "don't care".to_owned(),
            };
            self.attachment("depth", attachment, &load)
        });
        colour.chain(depth).collect::<Vec<_>>().join("; ")
    }

    /// `attachment`, a `role` attachment loaded as `load` says, naming its
    /// mip and layer where the texture may have more than one.
    fn attachment<T>(&self, role: &str, attachment: &PlannedAttachment<T>, load: &str) -> String {
        let shape = self.textures[attachment.texture];
        let mip = index("mip", attachment.mip, shape.map(|shape| shape.mips));
        let layer = index("layer", attachment.layer, shape.map(|shape| shape.layers));
        let part: String = mip
            .into_iter()
            .chain(layer)
            .map(|axis| format!(" {axis}"))
            .collect();
        let store = match attachment.store {
            StoreOp::Store => "store",
            StoreOp::DontCare => "don't care",
        };
        format!(
            "{role} {}{part}: {load}, {store}",
            self.texture_names[attachment.texture]
        )
    }
}

/// The name of a kind of queue.
fn queue(kind: QueueKind) -> &'static str {
    match kind {
        QueueKind::Graphics => "graphics",
        QueueKind::Compute => "compute",
        QueueKind::Transfer => "transfer",
    }
}

/// `set`'s accesses, each as its name in lower case with hyphens, joined
/// with `|`.
fn accesses(set: AccessSet) -> String {
    let names: Vec<_> = set
        .iter()
        .map(|access| kebab(&format!("{access:?}")))
        .collect();
    names.join(" | ")
}

/// `name`, in upper camel case, in lower case with a hyphen before each
/// word after the first.
fn kebab(name: &str) -> String {
    let mut kebab = String::with_capacity(name.len() + 4);
    for (index, letter) in name.chars().enumerate() {
        if letter.is_ascii_uppercase() && index > 0 {
            kebab.push('-');
        }
        kebab.push(letter.to_ascii_lowercase());
    }
    kebab
}

/// Mip or layer `index`, as `noun` and its number, of a texture with
/// `count` of them, `None` if that's not known, as an import's isn't.
/// `None` if it's the texture's only one, or index 0 of an import, which
/// usually has only one.
fn index(noun: &str, index: u32, count: Option<u32>) -> Option<String> {
    let only = index == 0 && count.is_none_or(|count| count == 1);
    (!only).then(|| format!("{noun} {index}"))
}

/// `range` of the `count` mips or layers a texture has, `None` if that's
/// not known, as `noun` and its numbers: `mip 2`, `mips 1..3` or `mips 1..`.
/// `None` if it covers them all.
fn range(noun: &str, range: &Range<u32>, count: Option<u32>) -> Option<String> {
    let open = range.end == Subresource::REMAINING;
    if range.start == 0 && (open || Some(range.end) == count) {
        return None;
    }
    Some(if open {
        format!("{noun}s {}..", range.start)
    } else if range.end == range.start + 1 {
        format!("{noun} {}", range.start)
    } else {
        format!("{noun}s {}..{}", range.start, range.end)
    })
}

#[cfg(test)]
mod tests {
    //! T0 snapshot tests of readable plans: the example techniques of
    //! design 0003's test plan, built in Rust until technique files
    //! arrive, compiled as written on one queue. Review a changed snapshot
    //! as `docs/testing.md`'s "Snapshot tests" describes.

    use clockwork_rhi::{Access, Extent3d, Format};
    use insta::assert_snapshot;

    use super::*;
    use crate::validate::test_support::inputs;
    use crate::{
        Attachment, ComputeRecorder, GraphDesc, MipCount, MipDirection, TextureSize,
        TransientTexture, compile,
    };

    /// `desc` compiled as written on one queue, and described.
    fn described(desc: &GraphDesc) -> String {
        compile(desc, &inputs())
            .expect("the graph is valid")
            .describe()
    }

    /// A texture of `format`, the output's size.
    fn full_size(format: Format) -> TransientTexture {
        TransientTexture::new(format, TextureSize::Output { scale: 1.0 })
    }

    /// The triangle: one raster pass that clears the output and draws.
    #[test]
    fn the_triangle() {
        let mut graph = GraphDesc::new("triangle");
        let output = graph.output();
        graph
            .raster_pass("triangle", "triangle")
            .colour(output, Attachment::Clear([0.02, 0.05, 0.1, 1.0]));
        assert_snapshot!(described(&graph));
    }

    /// A deferred renderer's passes: a depth prepass, a G-buffer, a shadow
    /// map, lighting from all three, and tone mapping to the output.
    #[test]
    fn a_deferred_renderer() {
        let mut graph = GraphDesc::new("deferred");
        let depth = graph.transient_texture("depth", full_size(Format::Depth32Float));
        let albedo = graph.transient_texture("albedo", full_size(Format::Rgba8Unorm));
        let normals = graph.transient_texture("normals", full_size(Format::Rgba16Float));
        let shadow = graph.transient_texture(
            "shadow",
            TransientTexture::new(
                Format::Depth32Float,
                TextureSize::Fixed(Extent3d::new_2d(2048, 2048)),
            ),
        );
        let hdr = graph.transient_texture("hdr", full_size(Format::Rgba16Float));
        let output = graph.output();
        graph
            .raster_pass("depth prepass", "depth prepass")
            .depth(depth, Attachment::Clear(0.0));
        graph
            .raster_pass("g-buffer", "g-buffer")
            .colour(albedo, Attachment::Discard)
            .colour(normals, Attachment::Discard)
            .depth(depth, Attachment::Keep);
        graph
            .raster_pass("shadow map", "shadow map")
            .depth(shadow, Attachment::Clear(0.0));
        graph
            .fullscreen_pass("lighting", "lighting")
            .bind(albedo, Access::FragmentShaderSampledRead)
            .bind(normals, Access::FragmentShaderSampledRead)
            .bind(depth, Access::FragmentShaderSampledRead)
            .bind(shadow, Access::FragmentShaderSampledRead)
            .colour(hdr, Attachment::Discard);
        graph
            .fullscreen_pass("tone map", "tone map")
            .bind(hdr, Access::FragmentShaderSampledRead)
            .colour(output, Attachment::Discard);
        assert_snapshot!(described(&graph));
    }

    /// A bloom chain: the bright parts of a scene, downsampled mip by mip,
    /// upsampled back up, and composited over the scene into the output.
    #[test]
    fn a_bloom_chain() {
        let mut graph = GraphDesc::new("bloom");
        let scene = graph.transient_texture("scene", full_size(Format::Rgba16Float));
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(512, 512)),
            )
        };
        let bloom = graph.transient_texture("bloom", chain);
        let output = graph.output();
        graph
            .raster_pass("scene", "scene")
            .colour(scene, Attachment::Clear([0.0, 0.0, 0.0, 1.0]));
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
        assert_snapshot!(described(&graph));
    }

    /// A compute pass that asks for the compute queue, culling lights into
    /// a buffer the shading pass reads. It runs on the graphics queue until
    /// queues are assigned.
    #[test]
    fn a_compute_pass_on_the_compute_queue() {
        let mut graph = GraphDesc::new("light culling");
        let lights = graph.transient_buffer("lights", 4096);
        let output = graph.output();
        graph
            .compute_pass("cull lights", ComputeRecorder::recorder("cull lights"))
            .queue(clockwork_rhi::QueueKind::Compute)
            .write(lights, Access::ComputeShaderWrite);
        graph
            .raster_pass("shade", "shade")
            .read(lights, Access::FragmentShaderRead)
            .colour(output, Attachment::Clear([0.0, 0.0, 0.0, 1.0]));
        assert_snapshot!(described(&graph));
    }

    /// Accesses are written in lower case with hyphens, parts of a texture
    /// by the mips and layers they cover, or not at all if they cover it
    /// whole, and an attachment's mip and layer only where the texture may
    /// have more than one.
    #[test]
    fn names_and_parts_are_written_plainly() {
        assert_eq!(
            accesses(Access::ColorAttachmentRead | Access::ColorAttachmentWrite),
            "color-attachment-read | color-attachment-write"
        );
        assert_eq!(accesses(Access::None.into()), "none");
        assert_eq!(range("mip", &(0..4), Some(4)), None);
        assert_eq!(range("mip", &(0..Subresource::REMAINING), None), None);
        assert_eq!(range("mip", &(2..3), Some(4)), Some("mip 2".to_owned()));
        assert_eq!(
            range("layer", &(1..3), Some(6)),
            Some("layers 1..3".to_owned())
        );
        assert_eq!(
            range("mip", &(1..Subresource::REMAINING), None),
            Some("mips 1..".to_owned())
        );
        assert_eq!(range("mip", &(0..2), None), Some("mips 0..2".to_owned()));
        // An attachment's mip or layer is named unless it's the only one,
        // or index 0 of an import, whose count isn't known.
        assert_eq!(index("mip", 0, Some(1)), None);
        assert_eq!(index("mip", 0, None), None);
        assert_eq!(index("mip", 0, Some(4)), Some("mip 0".to_owned()));
        assert_eq!(index("layer", 2, None), Some("layer 2".to_owned()));
    }
}
