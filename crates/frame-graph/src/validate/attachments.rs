//! Validation of render passes' attachments (design 0003, "Compiling", step
//! 1): that each raster and fullscreen pass renders to something, with no
//! more attachments than a render pass has, in formats it can render to,
//! all the same size.

use clockwork_rhi::{MAX_COLOUR_ATTACHMENTS, TextureDimension};

use super::uses::overlap;
use super::{Checked, CheckedUse, TextureShape};
use crate::GraphDesc;
use crate::error::Errors;
use crate::pass::{PassDesc, PassKind, UseRole};

/// Reports every mistake in each raster and fullscreen pass's
/// attachments. `checked` holds each pass's uses that passed the checks of
/// uses on their own, and only those are checked here.
pub(super) fn check_attachments(
    desc: &GraphDesc,
    shapes: &[Option<TextureShape>],
    checked: &[Vec<CheckedUse>],
    errors: &mut Errors,
) {
    for (pass, uses) in desc.passes.iter().zip(checked) {
        if !matches!(
            pass.kind,
            PassKind::Raster { .. } | PassKind::Fullscreen { .. }
        ) {
            continue;
        }
        let mut check = AttachmentCheck {
            desc,
            shapes,
            pass,
            errors,
        };
        check.counts();
        let attachments: Vec<_> = uses
            .iter()
            .filter_map(|used| match (&used.resource, used.role) {
                (Checked::Texture(texture, subresource), UseRole::Colour(_)) => {
                    Some((texture.index(), subresource, false))
                }
                (Checked::Texture(texture, subresource), UseRole::Depth(_)) => {
                    Some((texture.index(), subresource, true))
                }
                _ => None,
            })
            .collect();
        for &(texture, _, depth) in &attachments {
            check.format(texture, depth);
        }
        check.distinct(&attachments);
        check.sizes(&attachments);
    }
}

/// An attachment that passed the checks of uses: its texture's index, its
/// mip and layer, and whether it's the depth attachment.
type AttachmentUse<'a> = (usize, &'a clockwork_rhi::Subresource, bool);

/// Checks one pass's attachments, reporting each mistake with the pass's
/// name.
struct AttachmentCheck<'a> {
    /// The description the pass is in.
    desc: &'a GraphDesc,
    /// Each texture's shape, where it's known.
    shapes: &'a [Option<TextureShape>],
    /// The pass.
    pass: &'a PassDesc,
    /// Where mistakes are reported.
    errors: &'a mut Errors,
}

impl AttachmentCheck<'_> {
    /// Records the error `message`, which follows the pass's name.
    fn report(&mut self, message: &str) {
        self.errors
            .push(format!("pass \"{}\" {message}", self.pass.name));
    }

    /// The name of the texture at `index`.
    fn name(&self, index: usize) -> &str {
        &self.desc.textures[index].name
    }

    /// Checks that the pass has at least one attachment, at most
    /// [`MAX_COLOUR_ATTACHMENTS`] colour attachments, and at most one depth
    /// attachment. These count every attachment declared, whether or not
    /// it passed the checks of uses.
    fn counts(&mut self) {
        let roles = || self.pass.uses.iter().map(|used| used.role);
        let colour = roles()
            .filter(|role| matches!(role, UseRole::Colour(_)))
            .count();
        let depth = roles()
            .filter(|role| matches!(role, UseRole::Depth(_)))
            .count();
        let kind = self.pass.kind.name();
        if colour + depth == 0 {
            self.report(&format!(
                "is a {kind} pass with no attachments, so it renders to nothing"
            ));
        }
        if colour > MAX_COLOUR_ATTACHMENTS {
            self.report(&format!(
                "has {colour} colour attachments, but a render pass can have at most \
                 {MAX_COLOUR_ATTACHMENTS}"
            ));
        }
        if depth > 1 {
            self.report(&format!(
                "has {depth} depth attachments, but a render pass can have one"
            ));
        }
    }

    /// Checks that the texture at `index` can be rendered to, as the depth
    /// attachment if `depth`: not 3D, not block-compressed, and in a depth
    /// format exactly when it's the depth attachment. An imported texture's
    /// format is the caller's, so it isn't checked.
    fn format(&mut self, index: usize, depth: bool) {
        let Some(shape) = self.shapes[index] else {
            return;
        };
        let name = self.name(index).to_owned();
        let format = shape.format;
        let problem = if shape.dimension == TextureDimension::D3 {
            Some(format!("renders to \"{name}\", which is a 3D texture"))
        } else if format.is_block_compressed() {
            Some(format!(
                "renders to \"{name}\", which is block-compressed, as {format:?}"
            ))
        } else if depth && !format.is_depth() {
            Some(format!(
                "tests depth against \"{name}\", whose format, {format:?}, has no depth"
            ))
        } else if !depth && format.is_depth() {
            Some(format!(
                "renders colour to \"{name}\", whose format, {format:?}, is a depth format"
            ))
        } else {
            None
        };
        if let Some(problem) = problem {
            self.report(&problem);
        }
    }

    /// Checks that no two attachments are the same mip of the same layer.
    fn distinct(&mut self, attachments: &[AttachmentUse<'_>]) {
        for (i, &(texture, part, _)) in attachments.iter().enumerate() {
            let repeated = attachments[..i]
                .iter()
                .any(|&(other, other_part, _)| other == texture && overlap(part, other_part));
            if repeated {
                let name = self.name(texture).to_owned();
                self.report(&format!(
                    "renders to the same mip and layer of \"{name}\" more than once"
                ));
            }
        }
    }

    /// Checks that every attachment whose texture's shape is known is the
    /// same size at its mip, as a render pass's attachments are, reporting
    /// the first that differs from the first.
    fn sizes(&mut self, attachments: &[AttachmentUse<'_>]) {
        let mut sized = attachments.iter().filter_map(|&(texture, part, _)| {
            let shape = self.shapes[texture]?;
            Some((texture, shape.mip_size(part.mips.start)))
        });
        let Some((first, size)) = sized.next() else {
            return;
        };
        if let Some((other, other_size)) = sized.find(|&(_, other_size)| other_size != size) {
            let (first, other) = (self.name(first).to_owned(), self.name(other).to_owned());
            self.report(&format!(
                "renders to \"{first}\" at {} × {} and \"{other}\" at {} × {}, but a render \
                 pass's attachments are the same size",
                size.0, size.1, other_size.0, other_size.1
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of validating attachments.

    use clockwork_rhi::{Access, Extent3d, Format};

    use super::*;
    use crate::validate::test_support::messages;
    use crate::{Attachment, ImportDesc, MipCount, TextureSize, TransientTexture};

    /// A texture of `format`, the output's size.
    fn full_size(format: Format) -> TransientTexture {
        TransientTexture::new(format, TextureSize::Output { scale: 1.0 })
    }

    #[test]
    fn matching_attachments_have_no_errors() {
        let mut graph = GraphDesc::new("attachments");
        let output = graph.output();
        let depth = graph.transient_texture("depth", full_size(Format::Depth32Float));
        let chain = TransientTexture {
            mips: MipCount::Full,
            ..TransientTexture::new(Format::Rgba16Float, TextureSize::Output { scale: 2.0 })
        };
        let bloom = graph.transient_texture("bloom", chain);
        graph
            .raster_pass("scene", "scene")
            .colour(output, Attachment::Clear([0.0; 4]))
            .colour(bloom.attachment(1, 0), Attachment::Clear([0.0; 4]))
            .depth(depth, Attachment::Clear(0.0), true);
        assert_eq!(messages(&graph), Vec::<String>::new());
    }

    #[test]
    fn a_render_pass_needs_attachments_and_has_few() {
        let mut graph = GraphDesc::new("counts");
        let depth = graph.transient_texture("depth", full_size(Format::Depth32Float));
        let colours: Vec<_> = (0..9)
            .map(|i| graph.transient_texture(&format!("colour {i}"), full_size(Format::Rgba8Unorm)))
            .collect();
        graph.raster_pass("empty", "empty");
        graph.fullscreen_pass("empty fullscreen", "nothing");
        let mut many = graph.raster_pass("many", "many");
        for &colour in &colours {
            many.colour(colour, Attachment::Clear([0.0; 4]));
        }
        many.depth(depth, Attachment::Clear(0.0), true)
            .depth(depth, Attachment::Keep, false);

        let messages = messages(&graph);
        assert_eq!(
            messages[..4],
            [
                "pass \"empty\" is a raster pass with no attachments, so it renders to nothing",
                "pass \"empty fullscreen\" is a fullscreen pass with no attachments, so it renders \
                 to nothing",
                "pass \"many\" has 9 colour attachments, but a render pass can have at most 8",
                "pass \"many\" has 2 depth attachments, but a render pass can have one",
            ]
        );
    }

    #[test]
    fn attachments_must_be_renderable() {
        let mut graph = GraphDesc::new("formats");
        let volume = TransientTexture {
            dimension: TextureDimension::D3,
            ..TransientTexture::new(
                Format::Rgba8Unorm,
                TextureSize::Fixed(Extent3d::new_2d(4, 4)),
            )
        };
        let fog = graph.transient_texture("fog", volume);
        let compressed = graph.transient_texture("compressed", full_size(Format::Bc7Unorm));
        let hdr = graph.transient_texture("hdr", full_size(Format::Rgba16Float));
        let depth = graph.transient_texture("depth", full_size(Format::Depth32Float));
        graph
            .raster_pass("3D", "a")
            .colour(fog, Attachment::Discard);
        graph
            .raster_pass("BC", "b")
            .colour(compressed, Attachment::Discard);
        graph
            .raster_pass("swapped", "c")
            .colour(depth, Attachment::Discard)
            .depth(hdr, Attachment::Clear(0.0), true);

        let messages = messages(&graph);
        assert_eq!(
            messages[..4],
            [
                "pass \"3D\" renders to \"fog\", which is a 3D texture",
                "pass \"BC\" renders to \"compressed\", which is block-compressed, as Bc7Unorm",
                "pass \"swapped\" renders colour to \"depth\", whose format, Depth32Float, is a \
                 depth format",
                "pass \"swapped\" tests depth against \"hdr\", whose format, Rgba16Float, has no \
                 depth",
            ]
        );
    }

    #[test]
    fn attachments_are_distinct_and_the_same_size() {
        let mut graph = GraphDesc::new("sizes");
        let output = graph.output();
        let half = graph.transient_texture(
            "half",
            TransientTexture::new(Format::Rgba8Unorm, TextureSize::Output { scale: 0.5 }),
        );
        let import = ImportDesc {
            before: Access::ColorAttachmentWrite.into(),
            after: Access::ColorAttachmentWrite.into(),
        };
        let overlay = graph.import_texture("overlay", import);
        graph
            .raster_pass("mismatched", "a")
            .colour(output, Attachment::Discard)
            .colour(overlay, Attachment::Discard)
            .colour(half, Attachment::Discard);
        graph
            .raster_pass("twice", "b")
            .colour(output, Attachment::Discard)
            .colour(output, Attachment::Discard);

        assert_eq!(
            messages(&graph),
            [
                "pass \"mismatched\" renders to \"output\" at 1920 × 1080 and \"half\" at 960 × \
                 540, but a render pass's attachments are the same size",
                "pass \"twice\" renders to the same mip and layer of \"output\" more than once",
            ]
        );
    }
}
