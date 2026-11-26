//! Validation of each pass's queue and uses (design 0003, "Compiling",
//! steps 1 and 2): that each use names a resource the graph has, with
//! accesses that suit the resource, the pass's kind and how the use was
//! declared, and parts of a texture that exist, and, on the expanded
//! passes, that a pass's uses of one subresource share a layout.

use std::ops::Range;

use clockwork_rhi::{Access, AccessSet, Layout, MAX_PUSH_DATA_SIZE, QueueKind, Subresource};

use super::repeat::{RepeatError, check_repeat};
use super::{Checked, CheckedUse, PassUses, TextureShape};
use crate::error::Errors;
use crate::pass::{Binding, Constant, PassDesc, PassKind, Repeat, Use, UseRole};
use crate::{
    ComputeRecorder, GraphDesc, MipDirection, RelativeMip, RepeatedUse, ResourceUse, TextureId,
    TextureUse,
};

/// The size, in bytes, of the push data `pass`'s bindings lay out, in order
/// (design 0003, "Data structures and layout"): a texture view or a sampler
/// is a 4-byte heap index, a buffer an 8-byte address aligned to 8, and a
/// constant 4 bytes, or 16 aligned to 16 for a `float4`, as Slang aligns it.
fn push_data_size(pass: &PassDesc) -> u32 {
    pass.bindings.iter().fold(0, |offset, binding| {
        let (size, alignment) = match binding {
            Binding::Use(index) => match pass.uses[*index].resource {
                ResourceUse::Buffer(_) => (8, 8),
                ResourceUse::Texture(_) | ResourceUse::Repeated(_) => (4, 4),
            },
            Binding::Sampler(_)
            | Binding::Constant(Constant::Float(_) | Constant::Int(_) | Constant::Uint(_)) => {
                (4, 4)
            }
            Binding::Constant(Constant::Float4(_)) => (16, 16),
        };
        offset.next_multiple_of(alignment).saturating_add(size)
    })
}

/// Reports every mistake in each pass's queue and uses, and returns what
/// it found of each pass, for the checks after these: its uses that have
/// none, and, if it's repeated for each mip, the mips its repetitions make.
pub(super) fn check_uses(
    desc: &GraphDesc,
    shapes: &[Option<TextureShape>],
    errors: &mut Errors,
) -> Vec<PassUses> {
    desc.passes
        .iter()
        .map(|pass| {
            let mut check = PassCheck {
                desc,
                shapes,
                pass,
                repetition: Repetition::None,
                errors,
            };
            check.queue();
            check.bindings();
            let mut mips = None;
            if let Some(repeat) = pass.repeat {
                check.repetition = match check_repeat(desc, shapes, pass, repeat) {
                    Ok(range) => {
                        let chain = shapes[repeat.texture.index()]
                            .expect("a valid repeat's texture has a shape")
                            .mips;
                        mips = Some(range.clone());
                        Repetition::first(repeat, range, chain)
                    }
                    Err(RepeatError::Report(message)) => {
                        check.report(&message);
                        Repetition::Mistaken {
                            texture: repeat.texture,
                        }
                    }
                    Err(RepeatError::Reported) => Repetition::Mistaken {
                        texture: repeat.texture,
                    },
                };
            }
            let mut uses: Vec<_> = pass.uses.iter().map(|used| check.one(used)).collect();
            // A relative mip of another texture than the one the pass is
            // repeated for, which is reported, can't be resolved at each
            // repetition, since the range is the repeated texture's. The
            // pass isn't expanded, as one whose repeat has a mistake isn't,
            // and its relative uses aren't checked further.
            let foreign = pass.uses.iter().any(|used| {
                matches!(
                    &used.resource,
                    ResourceUse::Repeated(part)
                        if pass.repeat.is_some_and(|repeat| repeat.texture != part.texture)
                )
            });
            if foreign {
                mips = None;
                for (used, checked) in pass.uses.iter().zip(&mut uses) {
                    if matches!(used.resource, ResourceUse::Repeated(_)) {
                        *checked = None;
                    }
                }
            }
            PassUses { uses, mips }
        })
        .collect()
}

/// Checks that an expanded pass's `uses` of overlapping parts of a texture
/// share a layout, since a subresource is in one layout at a time, calling
/// `report` with each mistake, which follows the pass's name, and returns
/// the textures whose uses don't, each reported once.
pub(super) fn check_overlaps(
    desc: &GraphDesc,
    uses: &[CheckedUse],
    mut report: impl FnMut(String),
) -> Vec<TextureId> {
    let parts: Vec<_> = uses
        .iter()
        .filter_map(|used| match &used.resource {
            Checked::Texture(texture, subresource) => Some((*texture, subresource, used.access)),
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
            .filter(|&&(other, other_part, _)| other == texture && overlap(subresource, other_part))
            .fold(access, |access, &(_, _, other)| access | other);
        if together.layout().is_none() {
            reported.push(texture);
            let name = &desc.textures[texture.index()].name;
            report(format!(
                "uses overlapping parts of \"{name}\" with {together:?}, which share no layout"
            ));
        }
    }
    reported
}

/// Checks one pass, reporting each mistake with the pass's name.
struct PassCheck<'a> {
    /// The description the pass is in.
    desc: &'a GraphDesc,
    /// Each texture's shape, where it's known.
    shapes: &'a [Option<TextureShape>],
    /// The pass.
    pass: &'a PassDesc,
    /// The repetition its relative mips are checked at, if it's repeated.
    repetition: Repetition,
    /// Where mistakes are reported.
    errors: &'a mut Errors,
}

/// The repetition a pass's uses of relative mips are checked at. A repeated
/// pass's uses are each checked once, as written, rather than once per
/// repetition (design 0003, "Compiling", step 1), so a mistake in one is
/// reported once. That holds for every repetition, since what's checked of
/// a use on its own doesn't depend on which mip it is, once the range is
/// known to have it, so the first repetition stands for the rest. What
/// depends on several uses' mips is checked on the expanded passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Repetition {
    /// The pass isn't repeated.
    None,
    /// The pass is repeated for each mip of `texture`, but how has a
    /// mistake, which is reported, so its relative mips can't be resolved.
    /// What doesn't need the mip is still checked.
    Mistaken {
        /// The texture, which the graph may not have.
        texture: TextureId,
    },
    /// The first repetition of a pass repeated for each mip of `texture`.
    First {
        /// The texture.
        texture: TextureId,
        /// The mip the first repetition makes.
        current: u32,
        /// The mip it makes it from, if the chain has one there.
        previous: Option<u32>,
    },
}

impl Repetition {
    /// The first repetition of `repeat`, which makes `mips` of a chain of
    /// `chain` mips: the largest mip of the range going down, and the
    /// smallest going up.
    fn first(repeat: Repeat, mips: Range<u32>, chain: u32) -> Self {
        let (current, previous) = match repeat.direction {
            MipDirection::Down => (mips.start, mips.start.checked_sub(1)),
            MipDirection::Up => (mips.end - 1, (mips.end < chain).then_some(mips.end)),
        };
        Self::First {
            texture: repeat.texture,
            current,
            previous,
        }
    }
}

impl PassCheck<'_> {
    /// Records the error `message`, which follows the pass's name.
    fn report(&mut self, message: &str) {
        self.errors
            .push(format!("pass \"{}\" {message}", self.pass.name));
    }

    /// Checks the pass's bindings: that the graph records the pass, since a
    /// recorder sets its own push data, that each sampler is one the graph
    /// has, that no resource is bound as indirect arguments, that they fit
    /// the RHI's push data, and that a compute pass the graph dispatches
    /// binds every resource it uses, since its program can reach nothing
    /// else.
    fn bindings(&mut self) {
        let pass = self.pass;
        if let PassKind::Compute(ComputeRecorder::Recorder(recorder)) = &pass.kind
            && !pass.bindings.is_empty()
        {
            self.report(&format!(
                "has bindings, but the recorder \"{recorder}\" records it, which sets its own \
                 push data"
            ));
        }
        for binding in &pass.bindings {
            match binding {
                Binding::Sampler(sampler) if self.desc.samplers.get(sampler.index()).is_none() => {
                    self.report(&format!(
                        "binds sampler {}, which this graph doesn't have",
                        sampler.0
                    ));
                }
                // A bound resource is read by the pass's program, through a
                // shader. The graph's own dispatch takes no arguments from
                // a buffer, so a bound use read as indirect arguments would
                // be ordered against the wrong stage.
                Binding::Use(index) if pass.uses[*index].access.contains(Access::IndirectRead) => {
                    if let Some(name) = self.resource_name(&pass.uses[*index].resource) {
                        self.report(&format!(
                            "binds \"{name}\" with IndirectRead, but its program reads it through \
                             a shader"
                        ));
                    }
                }
                _ => {}
            }
        }
        let size = push_data_size(pass);
        if size > MAX_PUSH_DATA_SIZE {
            self.report(&format!(
                "binds {size} bytes of push data, but a pass has at most {MAX_PUSH_DATA_SIZE}"
            ));
        }
        if let PassKind::Compute(ComputeRecorder::Dispatch(_)) = &pass.kind {
            for (index, used) in pass.uses.iter().enumerate() {
                let bound = pass.bindings.contains(&Binding::Use(index));
                // A use of a resource the graph doesn't have is reported
                // by the checks of each use.
                if let (false, Some(name)) = (bound, self.resource_name(&used.resource)) {
                    self.report(&format!(
                        "uses \"{name}\" without binding it, but the graph dispatches it, so \
                         its program can't reach it"
                    ));
                }
            }
        }
    }

    /// The name of the resource `resource` uses, or `None` if the graph
    /// doesn't have it.
    fn resource_name(&self, resource: &ResourceUse) -> Option<&str> {
        let name = match resource {
            ResourceUse::Texture(TextureUse { texture, .. })
            | ResourceUse::Repeated(RepeatedUse { texture, .. }) => {
                &self.desc.textures.get(texture.index())?.name
            }
            ResourceUse::Buffer(buffer) => &self.desc.buffers.get(buffer.index())?.name,
        };
        Some(name)
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
                (decl.name.as_str(), Some(part.clone()))
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
            ResourceUse::Repeated(part) => {
                let Some(decl) = self.desc.textures.get(part.texture.index()) else {
                    self.report(&format!(
                        "uses texture {}, which this graph doesn't have",
                        part.texture.0
                    ));
                    return None;
                };
                let relative = match part.mip {
                    RelativeMip::Current => "current",
                    RelativeMip::Previous => "previous",
                };
                let (texture, first) = match self.repetition {
                    Repetition::None => {
                        self.report(&format!(
                            "uses the {relative} mip of \"{}\", but isn't repeated for each mip",
                            decl.name
                        ));
                        return None;
                    }
                    Repetition::Mistaken { texture } => (texture, None),
                    Repetition::First {
                        texture,
                        current,
                        previous,
                    } => (texture, Some((current, previous))),
                };
                if texture != part.texture {
                    // A repeat of a texture the graph doesn't have is
                    // reported already.
                    if let Some(repeated) = self.desc.textures.get(texture.index()) {
                        self.report(&format!(
                            "uses the {relative} mip of \"{}\", but is repeated for each mip of \
                             \"{}\"",
                            decl.name, repeated.name
                        ));
                    }
                    return None;
                }
                let Some((current, previous)) = first else {
                    // The mip can't be resolved, but the accesses, the role
                    // and whether the accesses share a layout don't need
                    // it. A relative mip is always one mip, so it suits a
                    // storage view. The layout waits for the accesses to
                    // suit a texture, as it does for any use.
                    let accesses_ok = self.accesses(&decl.name, true, used);
                    self.role(&decl.name, used);
                    if accesses_ok {
                        self.layout(&decl.name, None, used.access);
                    }
                    return None;
                };
                // Checking the repeat gave a first repetition that reads
                // the previous mip one.
                let mip = match part.mip {
                    RelativeMip::Current => current,
                    RelativeMip::Previous => previous?,
                };
                let resolved = TextureUse {
                    texture,
                    subresource: Subresource {
                        mips: mip..mip + 1,
                        layers: part.layers.clone(),
                    },
                };
                (decl.name.as_str(), Some(resolved))
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
            (None, ResourceUse::Texture(_) | ResourceUse::Repeated(_)) => {
                unreachable!("a texture use, relative or not, has its part")
            }
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

    use clockwork_rhi::{AddressMode, Extent3d, Filter, Format, QueueKind};

    use super::*;
    use crate::validate::test_support::messages;
    use crate::{
        Attachment, BufferId, ComputeRecorder, ImportDesc, MipCount, MipDirection, SamplerId,
        SamplerSettings, TextureSize, TransientTexture,
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

    /// A mip relative to a repetition is valid in a pass repeated for each
    /// mip of its texture, and a mistake in a pass that isn't repeated, or
    /// that's repeated for another texture's mips.
    #[test]
    fn relative_mips_need_a_repeated_pass() {
        let (mut graph, hdr, bloom, ..) = graph();
        graph
            .transfer_pass("fill", "fill")
            .write(bloom.mip(0), Access::CopyWrite);
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        graph
            .compute_pass("once", ComputeRecorder::dispatch("once"))
            .bind(hdr.previous_mip(), Access::ComputeShaderSampledRead)
            .bind(TextureId(9).current_mip(), Access::ComputeShaderWrite);
        // A relative mip of another texture than one the graph doesn't
        // have, which is reported already, isn't reported again.
        graph
            .compute_pass("missing", ComputeRecorder::dispatch("missing"))
            .for_each_mip(TextureId(9), MipDirection::Down, ..)
            .bind(bloom.current_mip(), Access::ComputeShaderWrite);
        graph
            .compute_pass("other", ComputeRecorder::dispatch("other"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(hdr.current_mip(), Access::ComputeShaderWrite);

        assert_eq!(
            messages(&graph),
            [
                "pass \"once\" uses the previous mip of \"hdr\", but isn't repeated for each mip",
                "pass \"once\" uses texture 9, which this graph doesn't have",
                "pass \"missing\" is repeated for each mip of texture 9, which this graph doesn't \
                 have",
                "pass \"other\" uses the current mip of \"hdr\", but is repeated for each mip of \
                 \"bloom\"",
            ]
        );
    }

    /// A repeated pass's uses are each checked once, at its first
    /// repetition, as any other use is, so a mistake in one is reported
    /// once, with the pass's name. A pass whose repeat has a mistake still
    /// has its relative uses' accesses checked, and whether they share a
    /// layout, which need no mip.
    #[test]
    fn repeated_uses_are_checked_on_their_own() {
        let (mut graph, _, bloom, _, lights) = graph();
        graph
            .compute_pass("copy", ComputeRecorder::dispatch("copy"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .bind(bloom.current_mip(), Access::CopyWrite);
        graph
            .compute_pass("past", ComputeRecorder::dispatch("past"))
            .for_each_mip(bloom, MipDirection::Down, 9..)
            .bind(bloom.current_mip(), Access::CopyWrite)
            .bind(
                bloom.previous_mip(),
                Access::ComputeShaderRead | Access::ComputeShaderSampledRead,
            )
            .bind(lights, Access::CopyRead);

        assert_eq!(
            messages(&graph),
            [
                "pass \"copy\" is a compute pass, so it can't use \"bloom\" with CopyWrite",
                "pass \"past\" is repeated from mip 9 of \"bloom\", whose last mip is 3",
                "pass \"past\" is a compute pass, so it can't use \"bloom\" with CopyWrite",
                "pass \"past\" uses \"bloom\" with ComputeShaderRead | ComputeShaderSampledRead, \
                 which share no layout",
                "pass \"past\" is a compute pass, so it can't use \"lights\" with CopyRead",
            ]
        );
    }

    /// The first repetition makes the largest mip of the range going down,
    /// from the one above, and the smallest going up, from the one below,
    /// which the chain may not have.
    #[test]
    fn the_first_repetition_starts_the_walk() {
        let bloom = TextureId(2);
        let first = |direction, mips: std::ops::Range<u32>| {
            let repeat = Repeat {
                texture: bloom,
                start: Some(mips.start),
                end: Some(mips.end),
                direction,
            };
            match Repetition::first(repeat, mips, 4) {
                Repetition::First {
                    current, previous, ..
                } => (current, previous),
                other => panic!("expected a first repetition, not {other:?}"),
            }
        };
        assert_eq!(first(MipDirection::Down, 1..4), (1, Some(0)));
        assert_eq!(first(MipDirection::Down, 0..2), (0, None));
        assert_eq!(first(MipDirection::Up, 0..3), (2, Some(3)));
        assert_eq!(first(MipDirection::Up, 1..4), (3, None));
    }

    /// Bindings are only for passes the graph records, name samplers the
    /// graph has, aren't read as indirect arguments, and cover every resource a compute pass the graph
    /// dispatches uses, while a fullscreen pass reads through its fragment
    /// shader.
    #[test]
    fn bindings_must_suit_the_pass() {
        let (mut graph, hdr, bloom, font, lights) = graph();
        let output = graph.output();
        let linear = graph.sampler(
            "linear",
            SamplerSettings::new(Filter::Linear, AddressMode::ClampToEdge),
        );
        graph
            .transfer_pass("fill", "fill")
            .write(hdr, Access::CopyWrite)
            .write(lights, Access::CopyWrite);
        graph
            .fullscreen_pass("composite", "composite")
            .bind(hdr, Access::FragmentShaderSampledRead)
            .bind(font, Access::FragmentShaderSampledRead)
            .bind_sampler(linear)
            .bind(lights, Access::FragmentShaderRead)
            .colour(output, Attachment::Discard);
        graph
            .compute_pass("recorded", ComputeRecorder::recorder("recorded"))
            .bind(lights, Access::ComputeShaderRead);
        graph
            .compute_pass("dispatched", ComputeRecorder::dispatch("dispatched"))
            .bind(bloom.mip(0), Access::ComputeShaderWrite)
            .bind_sampler(SamplerId(3))
            .read(lights, Access::ComputeShaderRead);
        graph
            .compute_pass("indirect", ComputeRecorder::dispatch("indirect"))
            .bind(lights, Access::IndirectRead | Access::ComputeShaderRead);
        graph
            .fullscreen_pass("wrong stage", "wrong stage")
            .bind(hdr, Access::ComputeShaderSampledRead)
            .colour(output, Attachment::Discard);

        assert_eq!(
            messages(&graph),
            [
                "pass \"recorded\" has bindings, but the recorder \"recorded\" records it, which \
                 sets its own push data",
                "pass \"dispatched\" binds sampler 3, which this graph doesn't have",
                "pass \"dispatched\" uses \"lights\" without binding it, but the graph \
                 dispatches it, so its program can't reach it",
                "pass \"indirect\" binds \"lights\" with IndirectRead, but its program reads it \
                 through a shader",
                "pass \"wrong stage\" is a fullscreen pass, so it can't use \"hdr\" with \
                 ComputeShaderSampledRead",
            ]
        );
    }

    /// Push data is laid out in binding order, with a buffer's address
    /// aligned to 8 and a `float4` to 16, and may fill the RHI's 128 bytes
    /// but not pass them.
    #[test]
    fn bindings_must_fit_the_push_data() {
        let (mut graph, _, _, font, lights) = graph();
        let output = graph.output();
        let linear = graph.sampler(
            "linear",
            SamplerSettings::new(Filter::Linear, AddressMode::ClampToEdge),
        );
        graph
            .transfer_pass("fill", "fill")
            .write(lights, Access::CopyWrite);
        for (name, extra) in [("fits", false), ("overflows", true)] {
            let mut pass = graph.fullscreen_pass(name, name);
            // Bytes 0 to 4, then 16 to 32 after padding, then 32 to 36, then
            // 40 to 48 after padding. Without either padding, the passes
            // come to 4 or 12 bytes less, and neither overflows.
            pass.bind(font, Access::FragmentShaderSampledRead)
                .bind_constant(Constant::Float4([0.0; 4]))
                .bind_sampler(linear)
                .bind(lights, Access::FragmentShaderRead);
            // 20 more of 4 bytes each fill the 128.
            for _ in 0..20 {
                pass.bind_sampler(linear);
            }
            if extra {
                pass.bind_constant(Constant::Uint(1));
            }
            pass.colour(output, Attachment::Discard);
        }

        assert_eq!(
            messages(&graph),
            ["pass \"overflows\" binds 132 bytes of push data, but a pass has at most 128"]
        );
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
