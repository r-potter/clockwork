//! A frame graph's description: its resources and passes, by name (design
//! 0003, "Passes and resources").
//!
//! A [`GraphDesc`] is plain data. It holds no RHI handles, so it can be
//! built, compared and compiled with no GPU. Building one checks nothing:
//! compiling reports every mistake in it at once, so a graph built in Rust is
//! checked the same way as one loaded from a technique file.

use clockwork_rhi::{AccessSet, Extent3d, Format, QueueKind, TextureDimension};

use crate::pass::{
    ComputePassBuilder, ComputeRecorder, FullscreenPassBuilder, PassDesc, PassId, PassKind,
    RasterPassBuilder, TransferPassBuilder,
};

/// The reserved name of the texture a frame renders to, such as the surface
/// image. Every description has it, as [`GraphDesc::output`].
pub const OUTPUT: &str = "output";

/// A texture in a [`GraphDesc`]: an index into its textures.
///
/// An ID belongs to the description that made it. One from another
/// description names whatever texture has the same index in this one, and
/// compiling reports one with an index past its last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TextureId(pub(crate) u32);

impl TextureId {
    /// The index of the texture in its description's textures.
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A buffer in a [`GraphDesc`]: an index into its buffers.
///
/// An ID belongs to the description that made it, as a [`TextureId`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub(crate) u32);

impl BufferId {
    /// The index of the buffer in its description's buffers.
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A transient or persistent texture's description. Its usage isn't given:
/// the compiler derives it from the texture's accesses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientTexture {
    /// Its shape: a 2D texture or array, a cube map or array, or a 3D
    /// texture. A 3D texture has a fixed size, and one layer, and a cube map's
    /// layers are a multiple of six.
    pub dimension: TextureDimension,
    /// Its texel format.
    pub format: Format,
    /// Its size at mip 0.
    pub size: TextureSize,
    /// How many mips it has.
    pub mips: MipCount,
    /// How many array layers it has.
    pub layers: u32,
}

impl TransientTexture {
    /// A 2D texture of `format` and `size`, with one mip and one layer.
    pub const fn new(format: Format, size: TextureSize) -> Self {
        Self {
            dimension: TextureDimension::D2,
            format,
            size,
            mips: MipCount::Count(1),
            layers: 1,
        }
    }
}

/// A texture's size at mip 0: fixed, or the output's scaled and rounded up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TextureSize {
    /// This many texels, with a depth of 1 for anything but a 3D texture.
    Fixed(Extent3d),
    /// The output's size, times `scale`, rounded up: 1 for the output's own
    /// size, and 0.5 for half of it. Its depth is 1, so a 3D texture can't
    /// have it.
    Output {
        /// The factor the output's width and height are multiplied by.
        scale: f32,
    },
}

/// How many mips a texture has.
///
/// A texture sized by the output has one mip or a full chain: how many a
/// size allows changes with the output's, so a fixed count above 1 could
/// stop compiling when the window shrinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MipCount {
    /// This many, from 1. Above 1, only for a texture of a fixed size.
    Count(u32),
    /// A full chain, down to 1 × 1, which depends on the texture's size.
    Full,
}

/// The accesses an imported resource is in before each frame, and must be
/// left in after it. They're the same every frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportDesc {
    /// The accesses it's in when the frame starts, which its first use is
    /// ordered after.
    pub before: AccessSet,
    /// The accesses it must be left in when the frame ends.
    pub after: AccessSet,
}

/// A texture's name, and where it comes from.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TextureDecl {
    /// Its name, unique among the description's textures.
    pub(crate) name: String,
    /// Where it comes from.
    pub(crate) kind: TextureKind,
}

/// Where a texture comes from, and how long its contents live.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TextureKind {
    /// The texture the frame renders to, described by the compile inputs.
    Output,
    /// Created by the graph, with contents that live within a frame.
    Transient(TransientTexture),
    /// Created by the graph, with contents kept from one frame to the next.
    Persistent(TransientTexture),
    /// The caller's, given each frame.
    Imported(ImportDesc),
}

/// A buffer's name, and where it comes from.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BufferDecl {
    /// Its name, unique among the description's buffers.
    pub(crate) name: String,
    /// Where it comes from.
    pub(crate) kind: BufferKind,
}

/// Where a buffer comes from, and how long its contents live.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BufferKind {
    /// Created by the graph, with contents that live within a frame.
    Transient {
        /// Its size, in bytes.
        size: u64,
    },
    /// Created by the graph, with contents kept from one frame to the next.
    Persistent {
        /// Its size, in bytes.
        size: u64,
    },
    /// The caller's, given each frame.
    Imported(ImportDesc),
}

/// A frame's passes and the resources they use, by name. Built in Rust, or
/// loaded from a technique file.
///
/// Passes are kept in the order they're written, which defines what each
/// read sees: the most recent write before it.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphDesc {
    /// The graph's name, in messages and debug tools.
    label: String,
    /// Its textures, indexed by [`TextureId`]. The first is the output.
    pub(crate) textures: Vec<TextureDecl>,
    /// Its buffers, indexed by [`BufferId`].
    pub(crate) buffers: Vec<BufferDecl>,
    /// Its passes, in the order they're written, indexed by [`PassId`].
    pub(crate) passes: Vec<PassDesc>,
}

impl GraphDesc {
    /// An empty graph named `label`, with only its output.
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_owned(),
            textures: vec![TextureDecl {
                name: OUTPUT.to_owned(),
                kind: TextureKind::Output,
            }],
            buffers: Vec::new(),
            passes: Vec::new(),
        }
    }

    /// The graph's name.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The texture the frame renders to, given each frame. Its format, size
    /// and accesses before and after the frame are compile inputs.
    pub fn output(&self) -> TextureId {
        TextureId(0)
    }

    /// A texture created by the graph, whose contents live within a frame.
    pub fn transient_texture(&mut self, name: &str, desc: TransientTexture) -> TextureId {
        self.add_texture(name, TextureKind::Transient(desc))
    }

    /// A texture created by the graph, whose contents are kept from one
    /// frame to the next. A pass that reads it before any pass in the frame
    /// writes it sees the last frame's contents.
    pub fn persistent_texture(&mut self, name: &str, desc: TransientTexture) -> TextureId {
        self.add_texture(name, TextureKind::Persistent(desc))
    }

    /// A texture the caller owns, and gives each frame.
    pub fn import_texture(&mut self, name: &str, import: ImportDesc) -> TextureId {
        self.add_texture(name, TextureKind::Imported(import))
    }

    /// A buffer of `size` bytes created by the graph, whose contents live
    /// within a frame.
    pub fn transient_buffer(&mut self, name: &str, size: u64) -> BufferId {
        self.add_buffer(name, BufferKind::Transient { size })
    }

    /// A buffer of `size` bytes created by the graph, whose contents are
    /// kept from one frame to the next.
    pub fn persistent_buffer(&mut self, name: &str, size: u64) -> BufferId {
        self.add_buffer(name, BufferKind::Persistent { size })
    }

    /// A buffer the caller owns, and gives each frame.
    pub fn import_buffer(&mut self, name: &str, import: ImportDesc) -> BufferId {
        self.add_buffer(name, BufferKind::Imported(import))
    }

    /// A raster pass, recorded by the recorder registered as `recorder`.
    pub fn raster_pass(&mut self, name: &str, recorder: &str) -> RasterPassBuilder<'_> {
        let kind = PassKind::Raster {
            recorder: recorder.to_owned(),
        };
        RasterPassBuilder::new(self.add_pass(name, kind))
    }

    /// A fullscreen pass: one draw of a fullscreen triangle with the
    /// technique's program `program`, recorded by the graph.
    pub fn fullscreen_pass(&mut self, name: &str, program: &str) -> FullscreenPassBuilder<'_> {
        let kind = PassKind::Fullscreen {
            program: program.to_owned(),
        };
        FullscreenPassBuilder::new(self.add_pass(name, kind))
    }

    /// A compute pass, recorded by a recorder or dispatched by the graph.
    pub fn compute_pass(
        &mut self,
        name: &str,
        recorder: ComputeRecorder,
    ) -> ComputePassBuilder<'_> {
        ComputePassBuilder::new(self.add_pass(name, PassKind::Compute(recorder)))
    }

    /// A transfer pass, recorded by the recorder registered as `recorder`.
    pub fn transfer_pass(&mut self, name: &str, recorder: &str) -> TransferPassBuilder<'_> {
        let kind = PassKind::Transfer {
            recorder: recorder.to_owned(),
        };
        TransferPassBuilder::new(self.add_pass(name, kind))
    }

    /// The name of `texture`.
    ///
    /// # Panics
    ///
    /// Panics if `texture` isn't one of this description's.
    pub fn texture_name(&self, texture: TextureId) -> &str {
        &self.textures[texture.index()].name
    }

    /// The name of `buffer`.
    ///
    /// # Panics
    ///
    /// Panics if `buffer` isn't one of this description's.
    pub fn buffer_name(&self, buffer: BufferId) -> &str {
        &self.buffers[buffer.index()].name
    }

    /// The name of `pass`.
    ///
    /// # Panics
    ///
    /// Panics if `pass` isn't one of this description's.
    pub fn pass_name(&self, pass: PassId) -> &str {
        &self.passes[pass.index()].name
    }

    /// Adds a texture, and returns its ID.
    fn add_texture(&mut self, name: &str, kind: TextureKind) -> TextureId {
        let id = TextureId(next_index(self.textures.len()));
        self.textures.push(TextureDecl {
            name: name.to_owned(),
            kind,
        });
        id
    }

    /// Adds a buffer, and returns its ID.
    fn add_buffer(&mut self, name: &str, kind: BufferKind) -> BufferId {
        let id = BufferId(next_index(self.buffers.len()));
        self.buffers.push(BufferDecl {
            name: name.to_owned(),
            kind,
        });
        id
    }

    /// Adds a pass on the graphics queue, with no uses, after those already
    /// written, and returns its ID and description for a builder to fill in.
    fn add_pass(&mut self, name: &str, kind: PassKind) -> (PassId, &mut PassDesc) {
        let id = PassId(next_index(self.passes.len()));
        self.passes.push(PassDesc {
            name: name.to_owned(),
            kind,
            queue: QueueKind::Graphics,
            uses: Vec::new(),
            side_effects: false,
            repeat: None,
        });
        let pass = self.passes.last_mut().expect("a pass was just pushed");
        (id, pass)
    }
}

/// The index a table of `len` entries gives its next entry.
///
/// # Panics
///
/// Panics if the table already has `u32::MAX` entries, far more than any
/// frame has.
fn next_index(len: usize) -> u32 {
    u32::try_from(len).expect("a frame graph has fewer than 2^32 of each item")
}

#[cfg(test)]
mod tests {
    //! T0 tests of building descriptions.

    use clockwork_rhi::Access;

    use super::*;

    /// A one-mip texture the size of the output.
    fn full_size() -> TransientTexture {
        TransientTexture::new(Format::Rgba16Float, TextureSize::Output { scale: 1.0 })
    }

    /// Transient textures are 2D with one mip and layer by default, and
    /// can be cube maps or 3D.
    #[test]
    fn transient_textures_can_be_any_shape() {
        let texture = full_size();
        assert_eq!(texture.dimension, TextureDimension::D2);
        assert_eq!((texture.mips, texture.layers), (MipCount::Count(1), 1));

        let mut graph = GraphDesc::new("shapes");
        let environment = TransientTexture {
            dimension: TextureDimension::Cube,
            mips: MipCount::Full,
            layers: 6,
            ..TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d::new_2d(512, 512)),
            )
        };
        let fog = TransientTexture {
            dimension: TextureDimension::D3,
            ..TransientTexture::new(
                Format::Rgba16Float,
                TextureSize::Fixed(Extent3d {
                    width: 160,
                    height: 90,
                    depth: 64,
                }),
            )
        };
        let cube = graph.persistent_texture("environment", environment);
        let volume = graph.transient_texture("fog", fog);
        assert_eq!(
            graph.textures[cube.index()].kind,
            TextureKind::Persistent(environment)
        );
        assert_eq!(
            graph.textures[volume.index()].kind,
            TextureKind::Transient(fog)
        );
    }

    /// A new graph has its output and nothing else.
    #[test]
    fn a_new_graph_has_only_its_output() {
        let graph = GraphDesc::new("empty");
        assert_eq!(graph.label(), "empty");
        assert_eq!(graph.texture_name(graph.output()), OUTPUT);
        assert_eq!(graph.textures.len(), 1);
        assert_eq!(graph.textures[0].kind, TextureKind::Output);
        assert!(graph.buffers.is_empty());
        assert!(graph.passes.is_empty());
    }

    /// Textures and buffers get IDs in the order they're added, after the
    /// output, and keep their kinds.
    #[test]
    fn resources_are_numbered_in_the_order_they_are_added() {
        let mut graph = GraphDesc::new("resources");
        let import = ImportDesc {
            before: Access::CopyWrite.into(),
            after: Access::FragmentShaderSampledRead.into(),
        };
        let hdr = graph.transient_texture("hdr", full_size());
        let history = graph.persistent_texture("history", full_size());
        let font = graph.import_texture("font", import);
        let lights = graph.transient_buffer("lights", 256);
        let counters = graph.persistent_buffer("counters", 16);
        let instances = graph.import_buffer("instances", import);

        assert_eq!(
            [hdr, history, font],
            [TextureId(1), TextureId(2), TextureId(3)]
        );
        assert_eq!(
            [lights, counters, instances],
            [BufferId(0), BufferId(1), BufferId(2)]
        );
        assert_eq!(graph.texture_name(history), "history");
        assert_eq!(graph.buffer_name(instances), "instances");
        assert_eq!(graph.textures[2].kind, TextureKind::Persistent(full_size()));
        assert_eq!(graph.textures[3].kind, TextureKind::Imported(import));
        assert_eq!(graph.buffers[0].kind, BufferKind::Transient { size: 256 });
        assert_eq!(graph.buffers[1].kind, BufferKind::Persistent { size: 16 });
    }

    /// Passes keep the written order, on the graphics queue with no side
    /// effects unless asked.
    #[test]
    fn passes_are_kept_in_the_order_they_are_written() {
        let mut graph = GraphDesc::new("passes");
        let first = graph.raster_pass("scene", "scene").id();
        let second = graph.fullscreen_pass("tone map", "tone-map").id();
        let third = graph
            .compute_pass("blur", ComputeRecorder::dispatch("blur"))
            .id();
        let fourth = graph.transfer_pass("readback", "readback").id();

        assert_eq!(
            [first, second, third, fourth].map(|pass| pass.0),
            [0, 1, 2, 3]
        );
        let names: Vec<_> = graph.passes.iter().map(|pass| pass.name.as_str()).collect();
        assert_eq!(names, ["scene", "tone map", "blur", "readback"]);
        assert_eq!(graph.pass_name(second), "tone map");
        assert!(
            graph
                .passes
                .iter()
                .all(|pass| pass.queue == QueueKind::Graphics && !pass.side_effects)
        );
    }

    /// Descriptions are plain data, shared freely between threads.
    #[test]
    fn descriptions_are_send_and_sync() {
        /// Compiles only if `T` is `Send` and `Sync`.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GraphDesc>();
    }
}
