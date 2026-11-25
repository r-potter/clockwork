//! Validation of each pass's queue and uses (design 0003, "Compiling",
//! step 1): that each use names a resource the graph has, with accesses
//! that suit the resource, the pass's kind and how the use was declared,
//! and parts of a texture that exist, in one layout.

use std::ops::Range;

use clockwork_rhi::{Access, AccessSet, Layout, QueueKind, Subresource};

use super::{Checked, CheckedUse, TextureShape};
use crate::error::Errors;
use crate::pass::{PassDesc, Use, UseRole};
use crate::{GraphDesc, ResourceUse, TextureId};

/// Reports every mistake in each pass's queue and uses, and returns each
/// pass's uses that have none, for the checks after these. A texture whose
/// overlapping uses share no layout is left out of its pass's, so its
/// mistake is reported once.
pub(super) fn check_uses(
    desc: &GraphDesc,
    shapes: &[Option<TextureShape>],
    errors: &mut Errors,
) -> Vec<Vec<CheckedUse>> {
    desc.passes
        .iter()
        .map(|pass| {
            let mut check = PassCheck {
                desc,
                shapes,
                pass,
                errors,
            };
            check.queue();
            let mut checked: Vec<_> = pass
                .uses
                .iter()
                .filter_map(|used| check.one(used))
                .collect();
            let conflicting = check.overlaps(&checked);
            checked.retain(|used| {
                !matches!(used.resource, Checked::Texture(texture, _) if conflicting.contains(&texture))
            });
            checked
        })
        .collect()
}

/// Checks one pass, reporting each mistake with the pass's name.
struct PassCheck<'a> {
    /// The description the pass is in.
    desc: &'a GraphDesc,
    /// Each texture's shape, where it's known.
    shapes: &'a [Option<TextureShape>],
    /// The pass.
    pass: &'a PassDesc,
    /// Where mistakes are reported.
    errors: &'a mut Errors,
}

impl PassCheck<'_> {
    /// Records the error `message`, which follows the pass's name.
    fn report(&mut self, message: &str) {
        self.errors
            .push(format!("pass \"{}\" {message}", self.pass.name));
    }

    /// Checks that the pass's kind can run on the queue it asks for.
    fn queue(&mut self) {
        let (kind, queue) = (&self.pass.kind, self.pass.queue);
        if !kind.allows_queue(queue) {
            self.report(&format!(
                "is a {} pass, so it can't run on the {} queue",
                kind.name(),
                queue_name(queue)
            ));
        }
    }

    /// Checks `used`, and returns it with its subresource resolved if it has
    /// no mistakes.
    fn one(&mut self, used: &Use) -> Option<CheckedUse> {
        let (name, texture) = match &used.resource {
            ResourceUse::Texture(part) => {
                let Some(decl) = self.desc.textures.get(part.texture.index()) else {
                    self.report(&format!(
                        "uses texture {}, which this graph doesn't have",
                        part.texture.0
                    ));
                    return None;
                };
                (decl.name.as_str(), Some(part))
            }
            ResourceUse::Buffer(buffer) => {
                let Some(decl) = self.desc.buffers.get(buffer.index()) else {
                    self.report(&format!(
                        "uses buffer {}, which this graph doesn't have",
                        buffer.0
                    ));
                    return None;
                };
                (decl.name.as_str(), None)
            }
        };
        let accesses_ok = self.accesses(name, texture.is_some(), used);
        let role_ok = self.role(name, used);
        let resource = match (texture, &used.resource) {
            (Some(part), _) => {
                let shape = self.shapes[part.texture.index()];
                let subresource = self.subresource(name, &part.subresource, shape.as_ref());
                // The layout is checked only once every access suits a
                // texture, since one that doesn't gives it none, but it
                // doesn't wait for the rest.
                let laid_out = accesses_ok && self.layout(name, subresource.as_ref(), used.access);
                if !(role_ok && laid_out) {
                    return None;
                }
                Checked::Texture(part.texture, subresource?)
            }
            (None, ResourceUse::Buffer(buffer)) => Checked::Buffer(*buffer),
            (None, ResourceUse::Texture(_)) => unreachable!("a texture use has its part"),
        };
        (accesses_ok && role_ok).then_some(CheckedUse {
            resource,
            access: used.access,
            role: used.role,
        })
    }

    /// Whether each access in `used` suits the resource `name`, a texture if
    /// `is_texture`, and the pass, reporting each that doesn't.
    fn accesses(&mut self, name: &str, is_texture: bool, used: &Use) -> bool {
        let kind = &self.pass.kind;
        let attachment = matches!(used.role, UseRole::Colour(_) | UseRole::Depth(_));
        let mut ok = true;
        for access in used.access.iter() {
            let problem = if is_never_a_pass_use(access) {
                Some(format!(
                    "uses \"{name}\" with {access:?}, which no pass can"
                ))
            } else if is_texture && !access.applies_to_textures() {
                Some(format!(
                    "uses texture \"{name}\" with {access:?}, which only a buffer can have"
                ))
            } else if !is_texture && !access.applies_to_buffers() {
                Some(format!(
                    "uses buffer \"{name}\" with {access:?}, which only a texture can have"
                ))
            } else if is_attachment_access(access) && !attachment {
                Some(format!(
                    "uses \"{name}\" with {access:?}, which only an attachment can have"
                ))
            } else if !attachment && !kind.allows_access(access) {
                Some(format!(
                    "is a {} pass, so it can't use \"{name}\" with {access:?}",
                    kind.name()
                ))
            } else {
                None
            };
            if let Some(problem) = problem {
                self.report(&problem);
                ok = false;
            }
        }
        ok
    }

    /// Whether `used` was declared as what it does: with `read` only if it
    /// doesn't write, and with `write` only if it does.
    fn role(&mut self, name: &str, used: &Use) -> bool {
        let writes = used.access.is_write();
        let access = used.access;
        match used.role {
            UseRole::Read if writes => self.report(&format!(
                "reads \"{name}\" with {access:?}, which writes, so it's declared with `write`"
            )),
            UseRole::Write if !writes => self.report(&format!(
                "writes \"{name}\" with {access:?}, which only reads, so it's declared with `read`"
            )),
            _ => return true,
        }
        false
    }

    /// `subresource` of the texture `name`, with its ranges resolved for
    /// `shape` if it's known, or `None` if they name mips or layers it
    /// doesn't have, which is reported.
    fn subresource(
        &mut self,
        name: &str,
        subresource: &Subresource,
        shape: Option<&TextureShape>,
    ) -> Option<Subresource> {
        let mips = self.range(
            name,
            "mip",
            &subresource.mips,
            shape.map(|shape| shape.mips),
        );
        let layers = self.range(
            name,
            "layer",
            &subresource.layers,
            shape.map(|shape| shape.layers),
        );
        Some(Subresource {
            mips: mips?,
            layers: layers?,
        })
    }

    /// `range` of the texture `name`'s `count` items (mips or layers, as
    /// `item` says), with an end of [`Subresource::REMAINING`] resolved to
    /// `count` where it's known, or `None` if it's empty or passes the last,
    /// which is reported.
    fn range(
        &mut self,
        name: &str,
        item: &str,
        range: &Range<u32>,
        count: Option<u32>,
    ) -> Option<Range<u32>> {
        let end = match (range.end, count) {
            (Subresource::REMAINING, Some(count)) => count,
            (end, _) => end,
        };
        let in_range = count.is_none_or(|count| end <= count);
        if range.start < end && in_range {
            return Some(range.start..end);
        }
        let described = describe_range(item, range);
        let has = match count {
            Some(count) => format!("it has {count}"),
            None => format!("that names no {item}s"),
        };
        self.report(&format!("uses {described} of \"{name}\", but {has}"));
        None
    }

    /// Whether `access` has one layout, and a storage use names one mip, as
    /// a storage view has, reporting why not. The mips are checked only if
    /// `subresource` is known, since [`subresource`](Self::subresource)
    /// reported why it isn't.
    fn layout(&mut self, name: &str, subresource: Option<&Subresource>, access: AccessSet) -> bool {
        match (access.layout(), subresource) {
            (None, _) => {
                self.report(&format!(
                    "uses \"{name}\" with {access:?}, which share no layout"
                ));
                false
            }
            (Some(Layout::General), Some(subresource)) if subresource.mips.len() != 1 => {
                let described = describe_range("mip", &subresource.mips);
                self.report(&format!(
                    "uses {described} of \"{name}\" with {access:?}, through a storage view, \
                     which has one mip"
                ));
                false
            }
            (Some(_), _) => true,
        }
    }

    /// Checks that uses of overlapping parts of a texture share a layout,
    /// since a subresource is in one layout at a time, and returns the
    /// textures whose uses don't, each reported once.
    fn overlaps(&mut self, uses: &[CheckedUse]) -> Vec<TextureId> {
        let parts: Vec<_> = uses
            .iter()
            .filter_map(|used| match &used.resource {
                Checked::Texture(texture, subresource) => {
                    Some((*texture, subresource, used.access))
                }
                Checked::Buffer(_) => None,
            })
            .collect();
        let mut reported = Vec::new();
        for &(texture, subresource, access) in &parts {
            if reported.contains(&texture) {
                continue;
            }
            let together = parts
                .iter()
                .filter(|&&(other, other_part, _)| {
                    other == texture && overlap(subresource, other_part)
                })
                .fold(access, |access, &(_, _, other)| access | other);
            if together.layout().is_none() {
                reported.push(texture);
                let name = &self.desc.textures[texture.index()].name;
                self.report(&format!(
                    "uses overlapping parts of \"{name}\" with {together:?}, which share no \
                     layout"
                ));
            }
        }
        reported
    }
}

/// Whether two parts of a texture share a mip of a layer.
pub(super) fn overlap(a: &Subresource, b: &Subresource) -> bool {
    let ranges_meet = |a: &Range<u32>, b: &Range<u32>| a.start < b.end && b.start < a.end;
    ranges_meet(&a.mips, &b.mips) && ranges_meet(&a.layers, &b.layers)
}

/// Whether `access` can never be a pass's use: one that only starts or
/// ends a frame, or one the CPU makes.
fn is_never_a_pass_use(access: Access) -> bool {
    matches!(
        access,
        Access::None | Access::Present | Access::Acquired | Access::HostRead | Access::HostWrite
    )
}

/// Whether `access` is a colour or depth attachment's.
fn is_attachment_access(access: Access) -> bool {
    matches!(
        access,
        Access::ColorAttachmentRead
            | Access::ColorAttachmentWrite
            | Access::DepthAttachmentRead
            | Access::DepthAttachmentWrite
    )
}

/// The name of `queue`'s kind, in messages.
fn queue_name(queue: QueueKind) -> &'static str {
    match queue {
        QueueKind::Graphics => "graphics",
        QueueKind::Compute => "compute",
        QueueKind::Transfer => "transfer",
    }
}

/// `range` of a texture's `item`s (mips or layers) as written in messages:
/// "mip 2", "mips 2..4" or, for a range to the last, "mips 2..".
pub(super) fn describe_range(item: &str, range: &Range<u32>) -> String {
    if range.end == Subresource::REMAINING {
        format!("{item}s {}..", range.start)
    } else if range.end == range.start.saturating_add(1) {
        format!("{item} {}", range.start)
    } else {
        format!("{item}s {}..{}", range.start, range.end)
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of validating passes' queues and uses.

    use clockwork_rhi::{Extent3d, Format, QueueKind};

    use super::*;
    use crate::validate::test_support::messages;
    use crate::{
        Attachment, BufferId, ComputeRecorder, ImportDesc, MipCount, TextureSize, TransientTexture,
    };

    /// A graph with a one-mip texture "hdr", a four-mip texture "bloom", an
    /// imported texture "font" and a buffer "lights".
    fn graph() -> (GraphDesc, TextureId, TextureId, TextureId, BufferId) {
        let mut graph = GraphDesc::new("uses");
        let size = TextureSize::Output { scale: 1.0 };
        let hdr = graph.transient_texture("hdr", TransientTexture::new(Format::Rgba16Float, size));
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(256, 256)),
            )
        };
        let bloom = graph.transient_texture("bloom", chain);
        let import = ImportDesc {
            before: Access::CopyWrite.into(),
            after: Access::FragmentShaderSampledRead.into(),
        };
        let font = graph.import_texture("font", import);
        let lights = graph.transient_buffer("lights", 256);
        (graph, hdr, bloom, font, lights)
    }

    /// Valid uses of every kind of resource, by every kind of pass, report
    /// nothing.
    #[test]
    fn valid_uses_have_no_errors() {
        let (mut graph, hdr, bloom, font, lights) = graph();
        let output = graph.output();
        graph
            .transfer_pass("fill", "fill")
            .write(bloom.mip(0), Access::CopyWrite)
            .write(lights, Access::CopyWrite);
        graph
            .compute_pass("downsample", ComputeRecorder::recorder("downsample"))
            .queue(QueueKind::Compute)
            .read(bloom.mip(0), Access::ComputeShaderSampledRead)
            .write(bloom.mip(1), Access::ComputeShaderWrite)
            .write(
                lights,
                Access::ComputeShaderRead | Access::ComputeShaderWrite,
            );
        graph
            .raster_pass("scene", "scene")
            .colour(hdr, Attachment::Clear([0.0; 4]))
            .read(font, Access::FragmentShaderSampledRead)
            .read(lights, Access::IndirectRead | Access::VertexShaderRead);
        graph
            .transfer_pass("copy", "copy")
            .queue(QueueKind::Transfer)
            .read(hdr, Access::CopyRead)
            .write(output, Access::CopyWrite);
        assert_eq!(messages(&graph), Vec::<String>::new());
    }

    /// A pass that asks for a queue its kind can't run on is reported.
    #[test]
    fn each_kind_runs_only_on_its_queues() {
        let (mut graph, ..) = graph();
        graph
            .compute_pass("cull", ComputeRecorder::dispatch("cull"))
            .queue(QueueKind::Transfer);
        assert_eq!(
            messages(&graph),
            ["pass \"cull\" is a compute pass, so it can't run on the transfer queue"]
        );
    }

    /// Uses of textures and buffers the graph doesn't have are reported.
    #[test]
    fn uses_must_name_resources_the_graph_has() {
        let (mut graph, ..) = graph();
        graph
            .transfer_pass("stray", "stray")
            .read(TextureId(99), Access::CopyRead)
            .write(BufferId(7), Access::CopyWrite);
        assert_eq!(
            messages(&graph),
            [
                "pass \"stray\" uses texture 99, which this graph doesn't have",
                "pass \"stray\" uses buffer 7, which this graph doesn't have",
            ]
        );
    }

    /// Accesses that don't suit the resource, the kind of pass, or any pass
    /// at all are reported, each with why.
    #[test]
    fn accesses_must_suit_the_resource_and_the_kind() {
        let (mut graph, hdr, _, _, lights) = graph();
        let output = graph.output();
        graph
            .raster_pass("draw", "draw")
            .colour(output, Attachment::Discard)
            .read(lights, Access::FragmentShaderSampledRead)
            .read(hdr, Access::IndexRead)
            .read(hdr, Access::ColorAttachmentRead)
            .read(hdr, Access::CopyRead)
            .read(lights, Access::HostRead);
        graph
            .compute_pass("blur", ComputeRecorder::recorder("blur"))
            .read(hdr, Access::FragmentShaderSampledRead)
            .read(hdr, Access::None);
        assert_eq!(
            messages(&graph),
            [
                "pass \"draw\" uses buffer \"lights\" with FragmentShaderSampledRead, which only a \
                 texture can have",
                "pass \"draw\" uses texture \"hdr\" with IndexRead, which only a buffer can have",
                "pass \"draw\" uses \"hdr\" with ColorAttachmentRead, which only an attachment can \
                 have",
                "pass \"draw\" is a raster pass, so it can't use \"hdr\" with CopyRead",
                "pass \"draw\" uses \"lights\" with HostRead, which no pass can",
                "pass \"blur\" is a compute pass, so it can't use \"hdr\" with \
                 FragmentShaderSampledRead",
                "pass \"blur\" uses \"hdr\" with None, which no pass can",
            ]
        );
    }

    /// A read that writes and a write that only reads are reported.
    #[test]
    fn reads_and_writes_are_declared_as_what_they_do() {
        let (mut graph, hdr, ..) = graph();
        graph
            .transfer_pass("copy", "copy")
            .read(hdr, Access::CopyWrite)
            .write(hdr, Access::CopyRead);
        assert_eq!(
            messages(&graph),
            [
                "pass \"copy\" reads \"hdr\" with CopyWrite, which writes, so it's declared with \
                 `write`",
                "pass \"copy\" writes \"hdr\" with CopyRead, which only reads, so it's declared \
                 with `read`",
            ]
        );
    }

    /// Mips and layers past a texture's last are reported, and an empty
    /// range even for an import.
    #[test]
    fn subresources_must_be_in_the_texture() {
        let (mut graph, hdr, bloom, font, _) = graph();
        graph
            .compute_pass("past the end", ComputeRecorder::recorder("past"))
            .read(bloom.mip(4), Access::ComputeShaderSampledRead)
            .read(bloom.mip(u32::MAX), Access::ComputeShaderSampledRead)
            .read(
                hdr.subresource(Subresource::layer(1)),
                Access::ComputeShaderSampledRead,
            )
            .read(
                font.subresource(Subresource {
                    mips: 2..2,
                    layers: 0..1,
                }),
                Access::ComputeShaderSampledRead,
            )
            .read(font.mip(9), Access::ComputeShaderRead);
        assert_eq!(
            messages(&graph),
            [
                "pass \"past the end\" uses mip 4 of \"bloom\", but it has 4",
                "pass \"past the end\" uses mips 4294967295.. of \"bloom\", but it has 4",
                "pass \"past the end\" uses layer 1 of \"hdr\", but it has 1",
                "pass \"past the end\" uses mips 2..2 of \"font\", but that names no mips",
            ]
        );
    }

    /// A use whose accesses share no layout, and a storage use of more than
    /// one mip, are reported.
    #[test]
    fn a_use_has_one_layout_and_storage_one_mip() {
        let (mut graph, _, bloom, font, _) = graph();
        graph
            .compute_pass("mixed", ComputeRecorder::recorder("mixed"))
            .read(
                bloom.mip(0),
                Access::ComputeShaderSampledRead | Access::ComputeShaderRead,
            )
            .write(bloom, Access::ComputeShaderWrite)
            .read(font, Access::ComputeShaderRead);
        assert_eq!(
            messages(&graph),
            [
                "pass \"mixed\" uses \"bloom\" with ComputeShaderRead | ComputeShaderSampledRead, \
                 which share no layout",
                "pass \"mixed\" uses mips 0..4 of \"bloom\" with ComputeShaderWrite, through a \
                 storage view, which has one mip",
                "pass \"mixed\" uses mips 0.. of \"font\" with ComputeShaderRead, through a \
                 storage view, which has one mip",
            ]
        );
    }

    /// A use with another mistake, such as a mip past the end or a read
    /// that writes, is still checked for a layout its accesses share.
    #[test]
    fn other_mistakes_hide_no_layout_mistakes() {
        let (mut graph, hdr, bloom, ..) = graph();
        graph
            .compute_pass("past the end", ComputeRecorder::recorder("past"))
            .read(
                bloom.mip(4),
                Access::ComputeShaderSampledRead | Access::ComputeShaderRead,
            );
        graph
            .transfer_pass("copy", "copy")
            .read(hdr, Access::CopyRead | Access::CopyWrite);
        assert_eq!(
            messages(&graph),
            [
                "pass \"past the end\" uses mip 4 of \"bloom\", but it has 4",
                "pass \"past the end\" uses \"bloom\" with ComputeShaderRead | \
                 ComputeShaderSampledRead, which share no layout",
                "pass \"copy\" reads \"hdr\" with CopyRead | CopyWrite, which writes, so it's \
                 declared with `write`",
                "pass \"copy\" uses \"hdr\" with CopyRead | CopyWrite, which share no layout",
            ]
        );
    }

    /// Uses of overlapping parts of a texture must share a layout, but
    /// uses of different mips needn't.
    #[test]
    fn overlapping_uses_share_a_layout() {
        let (mut graph, _, bloom, ..) = graph();
        graph
            .transfer_pass("fill", "fill")
            .write(bloom, Access::CopyWrite);
        graph
            .compute_pass("feedback", ComputeRecorder::recorder("feedback"))
            .read(
                bloom.subresource(Subresource::ALL),
                Access::ComputeShaderSampledRead,
            )
            .write(bloom.mip(1), Access::ComputeShaderWrite)
            .read(bloom.mip(2), Access::ComputeShaderRead);
        graph
            .compute_pass("separate mips", ComputeRecorder::recorder("separate"))
            .read(bloom.mip(0), Access::ComputeShaderSampledRead)
            .write(bloom.mip(1), Access::ComputeShaderWrite);
        assert_eq!(
            messages(&graph),
            [
                "pass \"feedback\" uses overlapping parts of \"bloom\" with \
                 ComputeShaderRead | ComputeShaderSampledRead | ComputeShaderWrite, which share \
                 no layout"
            ]
        );
    }
}
