//! Checks of a pass repeated for each mip of a texture: that the texture's
//! mips are known and more than one, and that the range names mips the
//! texture has, the first repetition's previous mip included, if the pass
//! uses it.

use std::ops::Range;

use super::TextureShape;
use crate::desc::TextureKind;
use crate::pass::{PassDesc, Repeat};
use crate::{GraphDesc, MipDirection, RelativeMip, ResourceUse, TextureSize};

/// A mistake in how a pass is repeated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RepeatError {
    /// One to report, with this message, which follows the pass's name.
    Report(String),
    /// The texture's description has a mistake, which is reported already.
    Reported,
}

/// Checks `pass`, repeated as `repeat`, and returns the mips its
/// repetitions make.
///
/// The range is checked in order, so its first mistake is the one
/// reported: whether it starts past the last mip before whether it's
/// empty, since `9..` going down a chain of 4 mips is empty because it
/// starts past it.
///
/// # Errors
///
/// Returns the first mistake found.
pub(super) fn check_repeat(
    desc: &GraphDesc,
    shapes: &[Option<TextureShape>],
    pass: &PassDesc,
    repeat: Repeat,
) -> Result<Range<u32>, RepeatError> {
    let report = |message: String| Err(RepeatError::Report(message));
    let Some(decl) = desc.textures.get(repeat.texture.index()) else {
        return report(format!(
            "is repeated for each mip of texture {}, which this graph doesn't have",
            repeat.texture.0
        ));
    };
    let name = &decl.name;
    let mips = match (shapes[repeat.texture.index()], &decl.kind) {
        (Some(shape), _) => shape.mips,
        (None, TextureKind::Imported(_)) => {
            return report(format!(
                "is repeated for each mip of \"{name}\", which is imported, so the graph doesn't \
                 know its mips"
            ));
        }
        (None, _) => return Err(RepeatError::Reported),
    };
    if mips == 1 {
        return report(format!(
            "is repeated for each mip of \"{name}\", which has one mip"
        ));
    }
    let last = mips - 1;
    // A chain sized by the output gets shorter as the window shrinks, so a
    // closed end is clamped to it rather than reported.
    let clamp = matches!(
        &decl.kind,
        TextureKind::Transient(created) | TextureKind::Persistent(created)
            if matches!(created.size, TextureSize::Output { .. })
    );
    let range = repeat.resolve(mips, clamp);
    if range.start > last {
        return report(format!(
            "is repeated from mip {} of \"{name}\", whose last mip is {last}",
            range.start
        ));
    }
    if range.is_empty() {
        let why = match repeat.end {
            None => format!(", since an open end going up stops before the last mip, {last}"),
            Some(end) if end > range.end => format!(
                ", since its end is clamped to its chain, which the output's size makes {mips} \
                 mips"
            ),
            Some(_) => String::new(),
        };
        return report(format!(
            "is repeated for mips {}..{} of \"{name}\", which is none{why}",
            range.start, range.end
        ));
    }
    if range.end > mips {
        // An end that saturated may have been one further than it says, so
        // it's given without a mip.
        let end = match range.end {
            u32::MAX => "past the end of the chain".to_owned(),
            end => format!("up to mip {}", end - 1),
        };
        return report(format!(
            "is repeated {end} of \"{name}\", whose last mip is {last}"
        ));
    }
    let uses_previous = pass.uses.iter().any(|used| {
        matches!(
            &used.resource,
            ResourceUse::Repeated(part)
                if part.texture == repeat.texture && part.mip == RelativeMip::Previous
        )
    });
    if uses_previous {
        match repeat.direction {
            MipDirection::Down if range.start == 0 => {
                return report(format!(
                    "uses the previous mip of \"{name}\", but its first repetition makes mip 0, \
                     which has none above it"
                ));
            }
            MipDirection::Up if range.end == mips => {
                return report(format!(
                    "uses the previous mip of \"{name}\", but its first repetition makes mip \
                     {last}, the last, which has none below it"
                ));
            }
            MipDirection::Down | MipDirection::Up => {}
        }
    }
    Ok(range)
}

#[cfg(test)]
mod tests {
    //! T0 tests of validating passes repeated for each mip.

    use std::ops::{Bound, RangeBounds};

    use clockwork_rhi::{Access, Extent3d, Format};

    use crate::validate::test_support::messages;
    use crate::{
        Attachment, ComputeRecorder, GraphDesc, ImportDesc, MipCount, MipDirection, TextureId,
        TextureSize, TransientTexture,
    };

    /// A graph with a four-mip texture "bloom".
    fn graph() -> (GraphDesc, TextureId) {
        let mut graph = GraphDesc::new("repeat");
        let chain = TransientTexture {
            mips: MipCount::Count(4),
            ..TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(256, 256)),
            )
        };
        let bloom = graph.transient_texture("bloom", chain);
        (graph, bloom)
    }

    /// Adds a pass `name`, repeated for each mip of `texture` in `mips` in
    /// `direction`, which writes the current mip and, if `previous`, reads
    /// the previous one.
    fn repeated(
        graph: &mut GraphDesc,
        name: &str,
        texture: TextureId,
        direction: MipDirection,
        mips: impl RangeBounds<u32>,
        previous: bool,
    ) {
        let mut pass = graph.compute_pass(name, ComputeRecorder::dispatch(name));
        pass.for_each_mip(texture, direction, mips);
        if previous {
            pass.bind(texture.previous_mip(), Access::ComputeShaderSampledRead);
        }
        pass.bind(texture.current_mip(), Access::ComputeShaderWrite);
    }

    /// The message a valid repeated pass `name` reports until compiling
    /// supports it.
    fn unsupported(name: &str) -> String {
        format!(
            "pass \"{name}\" is repeated for each mip of \"bloom\", which compiling doesn't \
             support yet"
        )
    }

    /// Open bounds go as far as the chain allows, in either direction, and
    /// closed bounds within it are valid.
    #[test]
    fn ranges_within_the_chain_are_valid() {
        let (mut graph, bloom) = graph();
        repeated(&mut graph, "down", bloom, MipDirection::Down, .., true);
        repeated(&mut graph, "up", bloom, MipDirection::Up, .., true);
        repeated(&mut graph, "middle", bloom, MipDirection::Down, 2..=3, true);
        repeated(&mut graph, "low", bloom, MipDirection::Up, 1..3, true);
        // A fullscreen pass draws to the current mip, by its attachment.
        graph
            .fullscreen_pass("blur", "blur")
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::FragmentShaderSampledRead)
            .colour(bloom.current_mip().attachment(0), Attachment::Discard);
        assert_eq!(
            messages(&graph),
            [
                unsupported("down"),
                unsupported("up"),
                unsupported("middle"),
                unsupported("low"),
                unsupported("blur"),
            ]
        );
    }

    /// A range must start within the chain, make at least one mip, and end
    /// within the chain, and a range that starts past the chain is
    /// reported as that, not as empty.
    #[test]
    fn ranges_must_name_mips_the_texture_has() {
        let (mut graph, bloom) = graph();
        repeated(&mut graph, "past", bloom, MipDirection::Down, 9.., true);
        repeated(&mut graph, "at end", bloom, MipDirection::Up, 4.., false);
        repeated(&mut graph, "none", bloom, MipDirection::Down, 2..2, true);
        // Written as bounds, since Clippy rejects a literal `3..1`.
        let backwards = (Bound::Included(3), Bound::Excluded(1));
        repeated(
            &mut graph,
            "backwards",
            bloom,
            MipDirection::Up,
            backwards,
            true,
        );
        repeated(&mut graph, "top", bloom, MipDirection::Up, 3.., true);
        repeated(&mut graph, "long", bloom, MipDirection::Down, 1..=4, true);
        repeated(
            &mut graph,
            "max",
            bloom,
            MipDirection::Down,
            ..=u32::MAX,
            true,
        );
        assert_eq!(
            messages(&graph),
            [
                "pass \"past\" is repeated from mip 9 of \"bloom\", whose last mip is 3",
                "pass \"at end\" is repeated from mip 4 of \"bloom\", whose last mip is 3",
                "pass \"none\" is repeated for mips 2..2 of \"bloom\", which is none",
                "pass \"backwards\" is repeated for mips 3..1 of \"bloom\", which is none",
                "pass \"top\" is repeated for mips 3..3 of \"bloom\", which is none, since an \
                 open end going up stops before the last mip, 3",
                "pass \"long\" is repeated up to mip 4 of \"bloom\", whose last mip is 3",
                "pass \"max\" is repeated past the end of the chain of \"bloom\", whose last mip \
                 is 3",
            ]
        );
    }

    /// A closed end on a chain sized by the output is clamped to the chain,
    /// so a range that fits a large window still compiles in a small one,
    /// making fewer mips, unless the clamp leaves it none.
    #[test]
    fn closed_ends_are_clamped_to_a_chain_sized_by_the_output() {
        let mut graph = GraphDesc::new("clamped");
        let chain = TransientTexture {
            mips: MipCount::Full,
            ..TransientTexture::new(Format::Rgba16Float, TextureSize::Output { scale: 1.0 })
        };
        let bloom = graph.transient_texture("bloom", chain);
        // The test inputs' 1920 × 1080 output makes a chain of 11 mips.
        repeated(&mut graph, "down", bloom, MipDirection::Down, 1..=20, true);
        repeated(&mut graph, "up", bloom, MipDirection::Up, ..=20, true);
        repeated(&mut graph, "empty", bloom, MipDirection::Up, 10..=20, true);
        repeated(&mut graph, "past", bloom, MipDirection::Down, 12..=20, true);
        assert_eq!(
            messages(&graph),
            [
                unsupported("down"),
                unsupported("up"),
                "pass \"empty\" is repeated for mips 10..10 of \"bloom\", which is none, since \
                 its end is clamped to its chain, which the output's size makes 11 mips"
                    .to_owned(),
                "pass \"past\" is repeated from mip 12 of \"bloom\", whose last mip is 10"
                    .to_owned(),
            ]
        );
    }

    /// A previous mip read only through an attachment needs one for the
    /// first repetition too.
    #[test]
    fn a_previous_mip_drawn_to_needs_one() {
        let (mut graph, bloom) = graph();
        graph
            .fullscreen_pass("blend", "blend")
            .for_each_mip(bloom, MipDirection::Down, 0..)
            .colour(bloom.previous_mip().attachment(0), Attachment::Keep);
        assert_eq!(
            messages(&graph),
            [
                "pass \"blend\" uses the previous mip of \"bloom\", but its first repetition \
                 makes mip 0, which has none above it"
            ]
        );
    }

    /// A pass that reads the previous mip needs one for its first
    /// repetition: one above going down, and one below going up. A pass
    /// that doesn't read it may make mip 0 going down, or the last mip
    /// going up.
    #[test]
    fn the_first_repetition_needs_a_previous_mip_if_it_reads_one() {
        let (mut graph, bloom) = graph();
        repeated(&mut graph, "from top", bloom, MipDirection::Down, 0.., true);
        repeated(
            &mut graph,
            "from bottom",
            bloom,
            MipDirection::Up,
            ..4,
            true,
        );
        repeated(
            &mut graph,
            "clear down",
            bloom,
            MipDirection::Down,
            0..,
            false,
        );
        repeated(&mut graph, "clear up", bloom, MipDirection::Up, ..4, false);
        assert_eq!(
            messages(&graph),
            [
                "pass \"from top\" uses the previous mip of \"bloom\", but its first repetition \
                 makes mip 0, which has none above it"
                    .to_owned(),
                "pass \"from bottom\" uses the previous mip of \"bloom\", but its first \
                 repetition makes mip 3, the last, which has none below it"
                    .to_owned(),
                unsupported("clear down"),
                unsupported("clear up"),
            ]
        );
    }

    /// A repeated texture's mips must be known, so not imported, and more
    /// than one, and a texture whose description has a mistake has only
    /// that mistake reported.
    #[test]
    fn repeated_textures_need_known_mips() {
        let (mut graph, _) = graph();
        let hdr = graph.transient_texture(
            "hdr",
            TransientTexture::new(Format::Rgba16Float, TextureSize::Output { scale: 1.0 }),
        );
        let import = ImportDesc {
            before: Access::CopyWrite.into(),
            after: Access::ComputeShaderSampledRead.into(),
        };
        let font = graph.import_texture("font", import);
        let broken = graph.transient_texture(
            "broken",
            TransientTexture {
                mips: MipCount::Count(4),
                ..TransientTexture::new(Format::Rgba16Float, TextureSize::Output { scale: 1.0 })
            },
        );
        let output = graph.output();
        for (name, texture) in [
            ("hdr", hdr),
            ("font", font),
            ("broken", broken),
            ("output", output),
            ("missing", TextureId(9)),
        ] {
            graph
                .compute_pass(name, ComputeRecorder::dispatch(name))
                .for_each_mip(texture, MipDirection::Down, ..);
        }
        assert_eq!(
            messages(&graph),
            [
                "texture \"broken\" is sized by the output, so it has 1 mip or a full chain, \
                 not 4",
                "pass \"hdr\" is repeated for each mip of \"hdr\", which has one mip",
                "pass \"font\" is repeated for each mip of \"font\", which is imported, so the \
                 graph doesn't know its mips",
                "pass \"output\" is repeated for each mip of \"output\", which has one mip",
                "pass \"missing\" is repeated for each mip of texture 9, which this graph doesn't \
                 have",
            ]
        );
    }
}
