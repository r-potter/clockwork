//! Validation of a description's resources: their names, and the shapes and
//! sizes of the textures and buffers the graph creates.

use std::collections::HashSet;

use clockwork_rhi::{Extent3d, TextureDimension};

use super::TextureShape;
use crate::desc::{BufferKind, TextureKind};
use crate::error::Errors;
use crate::{CompileInputs, GraphDesc, MipCount, OUTPUT, TextureSize, TransientTexture};

/// Reports each name used by more than one texture, buffer or pass. Each
/// kind of item has names of its own, so a texture and a buffer may share
/// one. A texture named after the output is reported as taking its
/// reserved name.
pub(super) fn check_names(desc: &GraphDesc, errors: &mut Errors) {
    let textures = desc.textures.iter().map(|texture| texture.name.as_str());
    let buffers = desc.buffers.iter().map(|buffer| buffer.name.as_str());
    let passes = desc.passes.iter().map(|pass| pass.name.as_str());
    report_duplicates("texture", textures, errors);
    report_duplicates("buffer", buffers, errors);
    report_duplicates("pass", passes, errors);
}

/// Reports, once each, the names that more than one of `names` has, in the
/// order their second use is found. `noun` names what they're names of.
fn report_duplicates<'a>(noun: &str, names: impl Iterator<Item = &'a str>, errors: &mut Errors) {
    let mut seen = HashSet::new();
    let mut reported = HashSet::new();
    for name in names {
        if !seen.insert(name) && reported.insert(name) {
            if noun == "texture" && name == OUTPUT {
                errors.push(format!(
                    "a texture is named \"{OUTPUT}\", the name reserved for the output"
                ));
            } else {
                errors.push(format!("more than one {noun} is named \"{name}\""));
            }
        }
    }
}

/// Each texture's shape, indexed by `TextureId`, with `None` for imported
/// textures and those whose description has a mistake, which is reported.
pub(super) fn texture_shapes(
    desc: &GraphDesc,
    inputs: &CompileInputs,
    errors: &mut Errors,
) -> Vec<Option<TextureShape>> {
    let output = inputs.output;
    if output.extent.width == 0 || output.extent.height == 0 {
        errors.push(format!(
            "the output is {} × {}, so it has no texels",
            output.extent.width, output.extent.height
        ));
    }
    desc.textures
        .iter()
        .map(|texture| match &texture.kind {
            TextureKind::Output => Some(TextureShape {
                dimension: TextureDimension::D2,
                format: output.format,
                extent: Extent3d::new_2d(output.extent.width, output.extent.height),
                mips: 1,
                layers: 1,
            }),
            TextureKind::Transient(created) | TextureKind::Persistent(created) => {
                let mut check = ShapeCheck {
                    texture: &texture.name,
                    errors,
                };
                check.shape(created, inputs)
            }
            TextureKind::Imported(_) => None,
        })
        .collect()
}

/// Reports each buffer the graph creates with no bytes.
pub(super) fn check_buffers(desc: &GraphDesc, errors: &mut Errors) {
    for buffer in &desc.buffers {
        let size = match buffer.kind {
            BufferKind::Transient { size } | BufferKind::Persistent { size } => size,
            BufferKind::Imported(_) => continue,
        };
        if size == 0 {
            errors.push(format!("buffer \"{}\" has a size of 0 bytes", buffer.name));
        }
    }
}

/// Works out the shape of one texture the graph creates, reporting each
/// mistake in its description.
struct ShapeCheck<'a> {
    /// The texture's name.
    texture: &'a str,
    /// Where mistakes are reported.
    errors: &'a mut Errors,
}

impl ShapeCheck<'_> {
    /// Records the error `message`, which follows the texture's name.
    fn report(&mut self, message: &str) {
        self.errors
            .push(format!("texture \"{}\" {message}", self.texture));
    }

    /// The shape of `desc`, or `None` if it has a mistake, which is
    /// reported.
    fn shape(&mut self, desc: &TransientTexture, inputs: &CompileInputs) -> Option<TextureShape> {
        // Each check reports its own mistakes, so all of them run before
        // any failure is acted on. Only those that need the size wait for
        // it.
        let extent = self.extent(desc, inputs);
        let shaped = self.check_dimension(desc, extent);
        let layers = self.layers(desc);
        let mips = self.mips(desc, extent);
        if !shaped {
            return None;
        }
        Some(TextureShape {
            dimension: desc.dimension,
            format: desc.format,
            extent: extent?,
            mips: mips?,
            layers: layers?,
        })
    }

    /// Its size at mip 0, or `None` if it has none.
    fn extent(&mut self, desc: &TransientTexture, inputs: &CompileInputs) -> Option<Extent3d> {
        let extent = match desc.size {
            TextureSize::Fixed(extent) => extent,
            TextureSize::Output { scale } => {
                if !(scale.is_finite() && scale > 0.0) {
                    self.report(&format!(
                        "is sized by the output with a scale of {scale}, which must be positive"
                    ));
                    return None;
                }
                let output = inputs.output.extent;
                Extent3d::new_2d(
                    self.scaled(output.width, scale)?,
                    self.scaled(output.height, scale)?,
                )
            }
        };
        if extent.is_empty() {
            self.report(&format!(
                "is {} × {} × {}, so it has no texels",
                extent.width, extent.height, extent.depth
            ));
            return None;
        }
        Some(extent)
    }

    /// `length` times `scale`, rounded up, and at least 1, or `None` if
    /// that's more texels than a size can hold.
    ///
    /// A scale such as a third isn't exact as an `f32`, so the product may
    /// be just over the whole number it stands for. A product within the
    /// scale's own rounding error of a whole number, relative to its size,
    /// is that number: a third of 1920 is 640, not 641. One further past
    /// it, however little, rounds up, as 0.3334 of 3 rounds up to 2.
    fn scaled(&mut self, length: u32, scale: f32) -> Option<u32> {
        let product = f64::from(length) * f64::from(scale);
        let rounding_error = product * f64::from(f32::EPSILON);
        let scaled = (product - rounding_error).ceil().max(1.0);
        if scaled > f64::from(u32::MAX) {
            self.report(&format!(
                "is sized by the output with a scale of {scale}, which makes it too large"
            ));
            return None;
        }
        // Positive, whole and in range, as checked and rounded above.
        Some(scaled as u32)
    }

    /// Whether `extent` and the layers suit the texture's dimension,
    /// reporting why not. The size is checked only if it's known, since
    /// [`extent`](Self::extent) reported why it isn't, and a count of no
    /// layers is left to [`layers`](Self::layers).
    fn check_dimension(&mut self, desc: &TransientTexture, extent: Option<Extent3d>) -> bool {
        match desc.dimension {
            TextureDimension::D3 => {
                let fixed = matches!(desc.size, TextureSize::Fixed(_));
                if !fixed {
                    self.report("is 3D, so it needs a fixed size, not the output's");
                }
                let one_layer = desc.layers <= 1;
                if !one_layer {
                    self.report(&format!("is 3D, so it has one layer, not {}", desc.layers));
                }
                fixed && one_layer
            }
            TextureDimension::D2 => self.check_flat(extent),
            TextureDimension::Cube => {
                let flat = self.check_flat(extent);
                let square = match extent {
                    Some(extent) if extent.width != extent.height => {
                        self.report(&format!(
                            "is a cube map, so its faces must be square, not {} × {}",
                            extent.width, extent.height
                        ));
                        false
                    }
                    _ => true,
                };
                let whole_cubes = desc.layers.is_multiple_of(6);
                if !whole_cubes {
                    self.report(&format!(
                        "is a cube map, so its layers must be a multiple of 6, not {}",
                        desc.layers
                    ));
                }
                flat && square && whole_cubes
            }
        }
    }

    /// Whether `extent`, if it's known, has a depth of 1, as every texture
    /// but a 3D one has, reporting it if not.
    fn check_flat(&mut self, extent: Option<Extent3d>) -> bool {
        match extent {
            Some(extent) if extent.depth != 1 => {
                self.report(&format!(
                    "has a depth of {}, but only a 3D texture can have more than 1",
                    extent.depth
                ));
                false
            }
            _ => true,
        }
    }

    /// Its layer count, or `None` if it has none.
    fn layers(&mut self, desc: &TransientTexture) -> Option<u32> {
        if desc.layers == 0 {
            self.report("has 0 layers");
            return None;
        }
        Some(desc.layers)
    }

    /// How many mips the texture has at the size `extent`, or `None` if
    /// that's none, more than it can have, or a fixed count above 1 for a
    /// texture sized by the output. Such a texture has one mip or a full
    /// chain, since how many its size allows changes with the window's.
    /// Only the count's limit waits for the size, if it isn't known.
    fn mips(&mut self, desc: &TransientTexture, extent: Option<Extent3d>) -> Option<u32> {
        let scaled = matches!(desc.size, TextureSize::Output { .. });
        match desc.mips {
            MipCount::Count(0) => {
                self.report("has 0 mips");
                return None;
            }
            MipCount::Count(count) if count > 1 && scaled => {
                self.report(&format!(
                    "is sized by the output, so it has 1 mip or a full chain, not {count}"
                ));
                return None;
            }
            MipCount::Count(_) | MipCount::Full => {}
        }
        let extent = extent?;
        let most = extent.max_mip_count();
        match desc.mips {
            MipCount::Full => Some(most),
            MipCount::Count(count) if count > most => {
                self.report(&format!(
                    "has {count} mips, but one of {} × {} × {} can have at most {most}",
                    extent.width, extent.height, extent.depth
                ));
                None
            }
            MipCount::Count(count) => Some(count),
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of validating resources.

    use clockwork_rhi::{Access, Extent2d, Format};

    use super::*;
    use crate::validate::test_support::{finished, inputs, messages};
    use crate::validate::validate;
    use crate::{ImportDesc, TextureId};

    /// A texture of `size`, with one mip and one layer.
    fn sized(size: TextureSize) -> TransientTexture {
        TransientTexture::new(Format::Rgba16Float, size)
    }

    /// A texture of a fixed `width` × `height` × `depth`.
    fn fixed(width: u32, height: u32, depth: u32) -> TransientTexture {
        sized(TextureSize::Fixed(Extent3d {
            width,
            height,
            depth,
        }))
    }

    /// The shape validation gives `texture` in a graph of its own.
    fn shape_of(created: TransientTexture) -> Option<TextureShape> {
        let mut graph = GraphDesc::new("shapes");
        let texture = graph.transient_texture("texture", created);
        validate(&finished(&graph), &inputs()).ok()?.textures[texture.index()]
    }

    /// Transient, persistent and imported resources that are valid report nothing.
    #[test]
    fn a_graph_of_valid_resources_has_no_errors() {
        let mut graph = GraphDesc::new("valid");
        graph.transient_texture("hdr", sized(TextureSize::Output { scale: 1.0 }));
        graph.persistent_texture("history", fixed(64, 64, 1));
        graph.import_texture(
            "font",
            ImportDesc {
                before: Access::CopyWrite.into(),
                after: Access::FragmentShaderSampledRead.into(),
            },
        );
        graph.transient_buffer("lights", 256);
        assert_eq!(messages(&graph), Vec::<String>::new());
    }

    /// A name used more than once among textures, buffers or passes is
    /// reported once, and a texture may share a buffer's name.
    #[test]
    fn duplicate_names_are_reported_once_each() {
        let mut graph = GraphDesc::new("names");
        graph.transient_texture("hdr", fixed(4, 4, 1));
        graph.transient_texture("hdr", fixed(4, 4, 1));
        graph.transient_texture("hdr", fixed(4, 4, 1));
        graph.transient_texture(OUTPUT, fixed(4, 4, 1));
        graph.transient_buffer("hdr", 4);
        graph.transient_buffer("lights", 4);
        graph.transient_buffer("lights", 4);
        graph.transfer_pass("scene", "scene");
        graph.transfer_pass("scene", "scene");

        assert_eq!(
            messages(&graph),
            [
                "more than one texture is named \"hdr\"",
                "a texture is named \"output\", the name reserved for the output",
                "more than one buffer is named \"lights\"",
                "more than one pass is named \"scene\"",
            ]
        );
    }

    /// Sizes scaled from the output's round up to whole texels, and at
    /// least one, but a third of a size is exact despite floating point.
    #[test]
    fn sizes_relative_to_the_output_are_rounded_up() {
        let half = shape_of(sized(TextureSize::Output { scale: 0.5 })).unwrap();
        assert_eq!(half.extent, Extent3d::new_2d(960, 540));
        let third = shape_of(sized(TextureSize::Output { scale: 1.0 / 3.0 })).unwrap();
        assert_eq!(third.extent, Extent3d::new_2d(640, 360));
        let tiny = shape_of(sized(TextureSize::Output { scale: 1e-6 })).unwrap();
        assert_eq!(tiny.extent, Extent3d::new_2d(1, 1));
    }

    /// Only a product within its scale's rounding error of a whole number
    /// is taken as that number. Anything further past it rounds up.
    #[test]
    fn products_just_past_a_whole_number_round_up() {
        let mut graph = GraphDesc::new("rounding");
        let texture =
            graph.transient_texture("third", sized(TextureSize::Output { scale: 0.3334 }));
        let mut narrow = inputs();
        narrow.output.extent = Extent2d {
            width: 3,
            height: 3,
        };
        let validated = validate(&finished(&graph), &narrow).unwrap();
        let shape = validated.textures[texture.index()].unwrap();
        assert_eq!(shape.extent, Extent3d::new_2d(2, 2));
    }

    /// The output's shape comes from the compile inputs.
    #[test]
    fn the_output_has_the_inputs_shape() {
        let graph = GraphDesc::new("output");
        let validated = validate(&finished(&graph), &inputs()).unwrap();
        assert_eq!(
            validated.textures[graph.output().index()],
            Some(TextureShape {
                dimension: TextureDimension::D2,
                format: Format::Bgra8Srgb,
                extent: Extent3d::new_2d(1920, 1080),
                mips: 1,
                layers: 1,
            })
        );
    }

    /// An imported texture's shape is the caller's, so validation has none.
    #[test]
    fn imported_textures_have_no_shape() {
        let mut graph = GraphDesc::new("imports");
        let import = ImportDesc {
            before: Access::CopyWrite.into(),
            after: Access::CopyWrite.into(),
        };
        let font = graph.import_texture("font", import);
        let validated = validate(&finished(&graph), &inputs()).unwrap();
        assert_eq!(validated.textures[font.index()], None);
    }

    /// A full chain has a mip for each halving down to one texel.
    #[test]
    fn a_full_chain_reaches_one_texel() {
        let chain = TransientTexture {
            mips: MipCount::Full,
            ..sized(TextureSize::Output { scale: 1.0 })
        };
        assert_eq!(shape_of(chain).unwrap().mips, 11);
    }

    /// Scales that aren't positive, sizes too large or empty, and buffers
    /// of no bytes are reported.
    #[test]
    fn bad_sizes_are_reported() {
        let mut graph = GraphDesc::new("sizes");
        for (name, scale) in [("zero", 0.0), ("negative", -1.0), ("nan", f32::NAN)] {
            graph.transient_texture(name, sized(TextureSize::Output { scale }));
        }
        graph.transient_texture("huge", sized(TextureSize::Output { scale: 1e30 }));
        graph.transient_texture("empty", fixed(0, 4, 1));
        graph.transient_buffer("nothing", 0);

        assert_eq!(
            messages(&graph),
            [
                "texture \"zero\" is sized by the output with a scale of 0, which must be positive",
                "texture \"negative\" is sized by the output with a scale of -1, which must be \
                 positive",
                "texture \"nan\" is sized by the output with a scale of NaN, which must be positive",
                "texture \"huge\" is sized by the output with a scale of 1000000000000000000000000000000, \
                 which makes it too large",
                "texture \"empty\" is 0 × 4 × 1, so it has no texels",
                "buffer \"nothing\" has a size of 0 bytes",
            ]
        );
    }

    /// A texture's checks that don't need its size run even when its size
    /// is a mistake, so every mistake is reported at once.
    #[test]
    fn a_bad_size_hides_no_other_mistakes() {
        let mut graph = GraphDesc::new("sizes and shapes");
        let volume = TransientTexture {
            dimension: TextureDimension::D3,
            layers: 2,
            ..sized(TextureSize::Output { scale: 0.0 })
        };
        let cube = TransientTexture {
            dimension: TextureDimension::Cube,
            layers: 4,
            ..fixed(0, 4, 1)
        };
        let no_layers = TransientTexture {
            layers: 0,
            ..fixed(0, 4, 1)
        };
        graph.transient_texture("fog", volume);
        graph.transient_texture("sky", cube);
        graph.transient_texture("nothing", no_layers);

        assert_eq!(
            messages(&graph),
            [
                "texture \"fog\" is sized by the output with a scale of 0, which must be positive",
                "texture \"fog\" is 3D, so it needs a fixed size, not the output's",
                "texture \"fog\" is 3D, so it has one layer, not 2",
                "texture \"sky\" is 0 × 4 × 1, so it has no texels",
                "texture \"sky\" is a cube map, so its layers must be a multiple of 6, not 4",
                "texture \"nothing\" is 0 × 4 × 1, so it has no texels",
                "texture \"nothing\" has 0 layers",
            ]
        );
    }

    /// No mips, more mips than the size allows, and no layers are
    /// reported.
    #[test]
    fn bad_mip_and_layer_counts_are_reported() {
        let mut graph = GraphDesc::new("counts");
        let no_mips = TransientTexture {
            mips: MipCount::Count(0),
            ..fixed(8, 8, 1)
        };
        let too_many = TransientTexture {
            mips: MipCount::Count(5),
            ..fixed(8, 8, 1)
        };
        let no_layers = TransientTexture {
            layers: 0,
            ..fixed(8, 8, 1)
        };
        graph.transient_texture("no mips", no_mips);
        graph.transient_texture("too many", too_many);
        graph.persistent_texture("no layers", no_layers);

        assert_eq!(
            messages(&graph),
            [
                "texture \"no mips\" has 0 mips",
                "texture \"too many\" has 5 mips, but one of 8 × 8 × 1 can have at most 4",
                "texture \"no layers\" has 0 layers",
            ]
        );
    }

    /// A texture sized by the output has one mip or a full chain, since a
    /// fixed count could exceed what a smaller window's size allows.
    #[test]
    fn scaled_textures_have_one_mip_or_a_full_chain() {
        let mut graph = GraphDesc::new("scaled mips");
        let scaled = |mips| TransientTexture {
            mips,
            ..sized(TextureSize::Output { scale: 0.5 })
        };
        graph.transient_texture("single", scaled(MipCount::Count(1)));
        graph.transient_texture("chain", scaled(MipCount::Full));
        graph.transient_texture("bloom", scaled(MipCount::Count(6)));
        graph.transient_texture(
            "fixed",
            TransientTexture {
                mips: MipCount::Count(6),
                ..fixed(64, 64, 1)
            },
        );

        assert_eq!(
            messages(&graph),
            ["texture \"bloom\" is sized by the output, so it has 1 mip or a full chain, not 6"]
        );
    }

    /// A count of no mips, or of several on a texture sized by the output,
    /// is reported even when the size is a mistake.
    #[test]
    fn mip_counts_are_checked_without_a_size() {
        let mut graph = GraphDesc::new("mips without a size");
        let broken = |mips| TransientTexture {
            mips,
            ..sized(TextureSize::Output { scale: 0.0 })
        };
        graph.transient_texture("none", broken(MipCount::Count(0)));
        graph.transient_texture("several", broken(MipCount::Count(3)));

        assert_eq!(
            messages(&graph),
            [
                "texture \"none\" is sized by the output with a scale of 0, which must be positive",
                "texture \"none\" has 0 mips",
                "texture \"several\" is sized by the output with a scale of 0, which must be \
                 positive",
                "texture \"several\" is sized by the output, so it has 1 mip or a full chain, not 3",
            ]
        );
    }

    /// 3D textures have a fixed size and one layer, only they have depth,
    /// and cube maps are square, in whole cubes.
    #[test]
    fn shapes_must_suit_their_dimension() {
        let mut graph = GraphDesc::new("dimensions");
        let volume = |size, layers| TransientTexture {
            dimension: TextureDimension::D3,
            layers,
            ..sized(size)
        };
        let cube = |width, height, layers| TransientTexture {
            dimension: TextureDimension::Cube,
            layers,
            ..fixed(width, height, 1)
        };
        graph.transient_texture("fog", volume(TextureSize::Fixed(Extent3d::new_2d(4, 4)), 1));
        graph.transient_texture("scaled fog", volume(TextureSize::Output { scale: 0.25 }, 1));
        graph.transient_texture(
            "layered fog",
            volume(TextureSize::Fixed(Extent3d::new_2d(4, 4)), 2),
        );
        graph.transient_texture(
            "thin fog",
            volume(TextureSize::Fixed(Extent3d::new_2d(4, 4)), 0),
        );
        graph.transient_texture("deep", fixed(4, 4, 4));
        graph.transient_texture("sky", cube(16, 16, 12));
        graph.transient_texture("oblong", cube(16, 8, 6));
        graph.transient_texture("partial", cube(16, 16, 4));

        assert_eq!(
            messages(&graph),
            [
                "texture \"scaled fog\" is 3D, so it needs a fixed size, not the output's",
                "texture \"layered fog\" is 3D, so it has one layer, not 2",
                "texture \"thin fog\" has 0 layers",
                "texture \"deep\" has a depth of 4, but only a 3D texture can have more than 1",
                "texture \"oblong\" is a cube map, so its faces must be square, not 16 × 8",
                "texture \"partial\" is a cube map, so its layers must be a multiple of 6, not 4",
            ]
        );
    }

    /// An output with no texels is reported.
    #[test]
    fn an_empty_output_is_reported() {
        let graph = GraphDesc::new("empty output");
        let mut empty = inputs();
        empty.output.extent = Extent2d {
            width: 0,
            height: 1080,
        };
        let errors = validate(&graph, &empty).unwrap_err();
        assert_eq!(
            errors.errors()[0].message(),
            "the output is 0 × 1080, so it has no texels"
        );
        assert_eq!(graph.output(), TextureId(0));
    }
}
