//! Passes and their uses of resources (design 0003, "Passes and
//! resources").
//!
//! A pass's kind decides what it can declare, so each kind has a builder
//! with only the methods it allows: a fullscreen pass has colour
//! attachments, a compute pass reads and writes through shaders, and so on.
//! Uses are RHI [`AccessSet`]s on a resource, or on a [`Subresource`] of a
//! texture. Attachments are uses too, with the accesses design 0002's
//! "Synchronization" gives them.

use std::ops::Range;

use clockwork_rhi::{Access, AccessSet, QueueKind, Subresource};

use crate::desc::{BufferId, TextureId};

/// A pass in a [`GraphDesc`](crate::GraphDesc): an index into its passes,
/// in the order they're written.
///
/// An ID belongs to the description that made it. One from another
/// description names whatever pass has the same index in this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PassId(pub(crate) u32);

impl PassId {
    /// The index of the pass in its description's passes.
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Part of a texture that a pass uses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureUse {
    /// The texture.
    pub texture: TextureId,
    /// The mips and layers used.
    pub subresource: Subresource,
}

impl TextureId {
    /// The texture's `subresource`, for a use of part of it.
    pub fn subresource(self, subresource: Subresource) -> TextureUse {
        TextureUse {
            texture: self,
            subresource,
        }
    }

    /// Mip `mip` of the texture, in every layer.
    pub fn mip(self, mip: u32) -> TextureUse {
        self.subresource(Subresource {
            mips: one(mip),
            layers: 0..Subresource::REMAINING,
        })
    }

    /// Mip `mip` of layer `layer` of the texture, for rendering to as an
    /// attachment.
    pub fn attachment(self, mip: u32, layer: u32) -> AttachmentTarget {
        AttachmentTarget {
            texture: self,
            mip,
            layer,
        }
    }
}

impl From<TextureId> for TextureUse {
    /// The whole texture: every mip of every layer.
    fn from(texture: TextureId) -> Self {
        texture.subresource(Subresource::ALL)
    }
}

/// The range of the one mip or layer `index`.
///
/// It's empty for `u32::MAX`, rather than overflowing, which compiling
/// reports as naming no mip or layer the texture has, as it does for any
/// index past its last.
fn one(index: u32) -> Range<u32> {
    index..index.saturating_add(1)
}

/// One mip of one layer of a texture, which a pass renders to as an
/// attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttachmentTarget {
    /// The texture.
    pub texture: TextureId,
    /// The mip rendered to.
    pub mip: u32,
    /// The layer rendered to.
    pub layer: u32,
}

impl AttachmentTarget {
    /// The part of the texture the attachment uses.
    fn texture_use(self) -> TextureUse {
        self.texture.subresource(Subresource {
            mips: one(self.mip),
            layers: one(self.layer),
        })
    }
}

impl From<TextureId> for AttachmentTarget {
    /// Mip 0 of layer 0 of the texture.
    fn from(texture: TextureId) -> Self {
        texture.attachment(0, 0)
    }
}

/// A resource, or part of one, that a pass uses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceUse {
    /// Part of a texture.
    Texture(TextureUse),
    /// A whole buffer. Buffers are tracked whole, since they're reached by
    /// address.
    Buffer(BufferId),
}

impl From<TextureUse> for ResourceUse {
    /// The part of the texture `texture` names.
    fn from(texture: TextureUse) -> Self {
        Self::Texture(texture)
    }
}

impl From<TextureId> for ResourceUse {
    /// The whole texture.
    fn from(texture: TextureId) -> Self {
        Self::Texture(texture.into())
    }
}

impl From<BufferId> for ResourceUse {
    /// The whole buffer.
    fn from(buffer: BufferId) -> Self {
        Self::Buffer(buffer)
    }
}

/// How an attachment's contents start. `T` is the clear value: four floats
/// for colour, one for depth, as in the RHI's `LoadOp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Attachment<T> {
    /// Cleared to this value.
    Clear(T),
    /// The pass reads the contents, by blending or drawing over them.
    Keep,
    /// The pass overwrites every pixel it needs.
    Discard,
}

/// What records a compute pass.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ComputeRecorder {
    /// The recorder registered under this name.
    Recorder(String),
    /// The graph, with one dispatch of the technique's program of this name.
    Dispatch(String),
}

impl ComputeRecorder {
    /// The recorder registered as `name`.
    pub fn recorder(name: &str) -> Self {
        Self::Recorder(name.to_owned())
    }

    /// One dispatch, by the graph, of the technique's program `program`.
    pub fn dispatch(program: &str) -> Self {
        Self::Dispatch(program.to_owned())
    }
}

/// A pass, as written.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PassDesc {
    /// Its name, unique among the description's passes.
    pub(crate) name: String,
    /// Its kind, and what records it.
    pub(crate) kind: PassKind,
    /// The kind of queue it asks to run on.
    pub(crate) queue: QueueKind,
    /// Its uses of resources, in the order they were declared.
    pub(crate) uses: Vec<Use>,
    /// Whether it's kept even if nothing reads what it writes.
    pub(crate) side_effects: bool,
}

/// A kind of pass, and what records it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PassKind {
    /// Draws, recorded by a recorder.
    Raster {
        /// The name the recorder is registered under.
        recorder: String,
    },
    /// One draw of a fullscreen triangle, recorded by the graph.
    Fullscreen {
        /// The technique's program it draws with.
        program: String,
    },
    /// Dispatches, recorded by a recorder or the graph.
    Compute(ComputeRecorder),
    /// Copies and fills, recorded by a recorder.
    Transfer {
        /// The name the recorder is registered under.
        recorder: String,
    },
}

impl PassKind {
    /// The kind's name, in messages: "raster", "fullscreen", "compute" or
    /// "transfer".
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::Raster { .. } => "raster",
            Self::Fullscreen { .. } => "fullscreen",
            Self::Compute(_) => "compute",
            Self::Transfer { .. } => "transfer",
        }
    }

    /// Whether a pass of this kind may use a resource with `access`, other
    /// than as an attachment, which only raster and fullscreen passes have.
    /// Raster passes read through vertex and fragment shaders and as draw
    /// arguments, compute passes read and write through compute shaders
    /// and read dispatch arguments, and transfer passes copy. A fullscreen
    /// pass's bindings arrive with the graph's own passes.
    pub(crate) fn allows_access(&self, access: Access) -> bool {
        match self {
            Self::Raster { .. } => matches!(
                access,
                Access::IndirectRead
                    | Access::IndexRead
                    | Access::VertexShaderRead
                    | Access::FragmentShaderRead
                    | Access::FragmentShaderSampledRead
            ),
            Self::Fullscreen { .. } => false,
            Self::Compute(_) => matches!(
                access,
                Access::IndirectRead
                    | Access::ComputeShaderRead
                    | Access::ComputeShaderSampledRead
                    | Access::ComputeShaderWrite
            ),
            Self::Transfer { .. } => matches!(access, Access::CopyRead | Access::CopyWrite),
        }
    }

    /// Whether a pass of this kind can ask for `queue`'s kind of queue:
    /// raster and fullscreen passes only graphics, compute passes graphics
    /// or compute, and transfer passes any. Builders let a pass ask for any,
    /// so that compiling reports a mistake with the rest.
    pub(crate) fn allows_queue(&self, queue: QueueKind) -> bool {
        match self {
            Self::Raster { .. } | Self::Fullscreen { .. } => queue == QueueKind::Graphics,
            Self::Compute(_) => matches!(queue, QueueKind::Graphics | QueueKind::Compute),
            Self::Transfer { .. } => true,
        }
    }
}

/// One use of a resource by a pass.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Use {
    /// The resource, or part of one.
    pub(crate) resource: ResourceUse,
    /// How the pass uses it.
    pub(crate) access: AccessSet,
    /// How the use was declared.
    pub(crate) role: UseRole,
}

/// How a use was declared, which validation checks its accesses against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum UseRole {
    /// With `read`, so it should only read.
    Read,
    /// With `write`, so it should write.
    Write,
    /// As a colour attachment, starting as given.
    Colour(Attachment<[f32; 4]>),
    /// As the depth attachment, starting as given.
    Depth(Attachment<f32>),
}

/// The accesses of a colour attachment that starts as `attachment`: a write,
/// and a read too if the pass keeps the contents.
fn colour_access(attachment: Attachment<[f32; 4]>) -> AccessSet {
    match attachment {
        Attachment::Keep => Access::ColorAttachmentRead | Access::ColorAttachmentWrite,
        Attachment::Clear(_) | Attachment::Discard => Access::ColorAttachmentWrite.into(),
    }
}

/// The accesses of a depth attachment that starts as `attachment`: depth
/// testing, and writing if the pass writes depth, or clears or discards the
/// contents, which writes them before the pass's first draw.
fn depth_access(attachment: Attachment<f32>, write: bool) -> AccessSet {
    let replaces_contents = !matches!(attachment, Attachment::Keep);
    if write || replaces_contents {
        Access::DepthAttachmentRead | Access::DepthAttachmentWrite
    } else {
        Access::DepthAttachmentRead.into()
    }
}

/// What every builder shares: the pass being built, and its ID.
#[derive(Debug)]
struct Builder<'a> {
    /// The pass's ID.
    id: PassId,
    /// The pass, in its description.
    pass: &'a mut PassDesc,
}

impl Builder<'_> {
    /// Adds a use of `resource`.
    fn add_use(&mut self, resource: ResourceUse, access: AccessSet, role: UseRole) {
        self.pass.uses.push(Use {
            resource,
            access,
            role,
        });
    }
}

/// Generates a builder struct for one kind of pass, with the methods every
/// kind has: `new`, `id` and `side_effects`.
macro_rules! builder {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug)]
        pub struct $name<'a>(Builder<'a>);

        impl<'a> $name<'a> {
            /// A builder for the pass `id`, which `pass` describes.
            pub(crate) fn new((id, pass): (PassId, &'a mut PassDesc)) -> Self {
                Self(Builder { id, pass })
            }

            /// The pass's ID.
            pub fn id(&self) -> PassId {
                self.0.id
            }

            /// Keeps the pass even if nothing reads what it writes.
            pub fn side_effects(&mut self) -> &mut Self {
                self.0.pass.side_effects = true;
                self
            }
        }
    };
}

/// Generates the `read` and `write` methods, for kinds that use resources
/// other than as attachments.
macro_rules! read_write {
    ($name:ident) => {
        impl $name<'_> {
            /// Reads `resource` with `access`, which should only read.
            pub fn read(
                &mut self,
                resource: impl Into<ResourceUse>,
                access: impl Into<AccessSet>,
            ) -> &mut Self {
                self.0
                    .add_use(resource.into(), access.into(), UseRole::Read);
                self
            }

            /// Writes `resource` with `access`, which should write, and may
            /// also read.
            pub fn write(
                &mut self,
                resource: impl Into<ResourceUse>,
                access: impl Into<AccessSet>,
            ) -> &mut Self {
                self.0
                    .add_use(resource.into(), access.into(), UseRole::Write);
                self
            }
        }
    };
}

/// Generates the `colour` method, for kinds that render.
macro_rules! colour {
    ($name:ident) => {
        impl $name<'_> {
            /// Renders to `target`, one mip of one layer, as the next colour
            /// attachment, starting as `attachment` says. A plain
            /// `TextureId` is its mip 0 of layer 0.
            pub fn colour(
                &mut self,
                target: impl Into<AttachmentTarget>,
                attachment: Attachment<[f32; 4]>,
            ) -> &mut Self {
                self.0.add_use(
                    ResourceUse::Texture(target.into().texture_use()),
                    colour_access(attachment),
                    UseRole::Colour(attachment),
                );
                self
            }
        }
    };
}

/// Generates the `queue` method, for kinds that may run on another queue,
/// documented with which kinds of queue the pass may ask for.
macro_rules! queue {
    ($name:ident, $allowed:literal) => {
        impl $name<'_> {
            /// Asks to run on `queue`'s kind of queue, rather than graphics.
            /// Where the device has no dedicated queue of that kind, the pass
            /// runs on the queue that kind shares.
            ///
            #[doc = $allowed]
            pub fn queue(&mut self, queue: QueueKind) -> &mut Self {
                self.0.pass.queue = queue;
                self
            }
        }
    };
}

builder! {
    /// Builds a raster pass: colour and depth attachments, and reads and
    /// writes through shaders.
    RasterPassBuilder
}
read_write!(RasterPassBuilder);
colour!(RasterPassBuilder);

impl RasterPassBuilder<'_> {
    /// Depth-tests against `target`, one mip of one layer, as the depth
    /// attachment, starting as `attachment` says, and writes depth if
    /// `write`. A plain `TextureId` is its mip 0 of layer 0. Clearing or
    /// discarding the contents writes them too, whatever `write` says.
    pub fn depth(
        &mut self,
        target: impl Into<AttachmentTarget>,
        attachment: Attachment<f32>,
        write: bool,
    ) -> &mut Self {
        self.0.add_use(
            ResourceUse::Texture(target.into().texture_use()),
            depth_access(attachment, write),
            UseRole::Depth(attachment),
        );
        self
    }
}

builder! {
    /// Builds a fullscreen pass: colour attachments, drawn to by the graph.
    FullscreenPassBuilder
}
colour!(FullscreenPassBuilder);

builder! {
    /// Builds a compute pass: reads and writes through shaders, on the
    /// graphics or compute queue.
    ComputePassBuilder
}
read_write!(ComputePassBuilder);
queue!(
    ComputePassBuilder,
    "A compute pass may ask for graphics or compute. Compiling reports one that asks for transfer."
);

builder! {
    /// Builds a transfer pass: copy reads and writes, and fills, on any
    /// queue.
    TransferPassBuilder
}
read_write!(TransferPassBuilder);
queue!(
    TransferPassBuilder,
    "A transfer pass may ask for any kind of queue."
);

#[cfg(test)]
mod tests {
    //! T0 tests of the pass builders.

    use clockwork_rhi::Format;

    use super::*;
    use crate::{GraphDesc, TextureSize, TransientTexture};

    /// A graph with a one-mip texture, a four-mip texture and a buffer.
    fn graph() -> (GraphDesc, TextureId, TextureId, BufferId) {
        let mut graph = GraphDesc::new("passes");
        let size = TextureSize::Output { scale: 1.0 };
        let depth =
            graph.transient_texture("depth", TransientTexture::new(Format::Depth32Float, size));
        let mut chain = TransientTexture::new(Format::Rgba16Float, size);
        chain.mips = crate::MipCount::Count(4);
        let bloom = graph.transient_texture("bloom", chain);
        let lights = graph.transient_buffer("lights", 1024);
        (graph, depth, bloom, lights)
    }

    #[test]
    fn attachments_have_the_accesses_their_start_implies() {
        let (mut graph, depth, _, _) = graph();
        let output = graph.output();
        graph
            .raster_pass("cleared", "scene")
            .colour(output, Attachment::Clear([0.0; 4]))
            .depth(depth, Attachment::Clear(0.0), true);
        graph
            .raster_pass("kept", "overlay")
            .colour(output, Attachment::Keep)
            .depth(depth, Attachment::Keep, false);
        graph
            .fullscreen_pass("discarded", "tone-map")
            .colour(output, Attachment::Discard);

        let accesses: Vec<Vec<AccessSet>> = graph
            .passes
            .iter()
            .map(|pass| pass.uses.iter().map(|used| used.access).collect())
            .collect();
        assert_eq!(
            accesses,
            [
                vec![
                    Access::ColorAttachmentWrite.into(),
                    Access::DepthAttachmentRead | Access::DepthAttachmentWrite,
                ],
                vec![
                    Access::ColorAttachmentRead | Access::ColorAttachmentWrite,
                    Access::DepthAttachmentRead.into(),
                ],
                vec![Access::ColorAttachmentWrite.into()],
            ]
        );
        assert_eq!(
            graph.passes[0].uses[0].role,
            UseRole::Colour(Attachment::Clear([0.0; 4]))
        );
        assert_eq!(
            graph.passes[1].uses[1].role,
            UseRole::Depth(Attachment::Keep)
        );
    }

    #[test]
    fn clearing_or_discarding_depth_writes_it_without_depth_writes() {
        let (mut graph, depth, _, _) = graph();
        graph
            .raster_pass("cleared", "test only")
            .depth(depth, Attachment::Clear(0.0), false);
        graph
            .raster_pass("discarded", "test only")
            .depth(depth, Attachment::Discard, false);

        for pass in &graph.passes {
            assert_eq!(
                pass.uses[0].access,
                Access::DepthAttachmentRead | Access::DepthAttachmentWrite,
                "pass \"{}\"",
                pass.name
            );
        }
    }

    #[test]
    fn attachments_render_to_one_mip_of_one_layer() {
        let (mut graph, depth, bloom, _) = graph();
        graph
            .raster_pass("upsample", "upsample")
            .colour(bloom.attachment(2, 0), Attachment::Discard)
            .depth(depth, Attachment::Clear(0.0), true);

        let uses = &graph.passes[0].uses;
        assert_eq!(
            uses[0].resource,
            ResourceUse::Texture(bloom.subresource(Subresource::mip_layer(2, 0)))
        );
        assert_eq!(
            uses[1].resource,
            ResourceUse::Texture(depth.subresource(Subresource::mip_layer(0, 0))),
            "a plain texture ID is mip 0 of layer 0"
        );
    }

    #[test]
    fn the_last_mip_index_gives_an_empty_range_rather_than_overflowing() {
        let (_, _, bloom, _) = graph();
        assert_eq!(bloom.mip(u32::MAX).subresource.mips, u32::MAX..u32::MAX);
        let target = bloom.attachment(u32::MAX, u32::MAX).texture_use();
        assert!(target.subresource.mips.is_empty() && target.subresource.layers.is_empty());
    }

    #[test]
    fn each_kind_of_pass_allows_its_queues() {
        let raster = PassKind::Raster {
            recorder: String::new(),
        };
        let fullscreen = PassKind::Fullscreen {
            program: String::new(),
        };
        let compute = PassKind::Compute(ComputeRecorder::recorder(""));
        let transfer = PassKind::Transfer {
            recorder: String::new(),
        };
        let allowed = |kind: &PassKind| QueueKind::ALL.map(|queue| kind.allows_queue(queue));

        // In the order graphics, compute, transfer.
        assert_eq!(allowed(&raster), [true, false, false]);
        assert_eq!(allowed(&fullscreen), [true, false, false]);
        assert_eq!(allowed(&compute), [true, true, false]);
        assert_eq!(allowed(&transfer), [true, true, true]);
    }

    #[test]
    fn reads_and_writes_keep_their_subresources_and_roles() {
        let (mut graph, _, bloom, lights) = graph();
        graph
            .compute_pass("downsample", ComputeRecorder::recorder("downsample"))
            .read(bloom.mip(0), Access::ComputeShaderSampledRead)
            .write(bloom.mip(1), Access::ComputeShaderWrite)
            .read(lights, Access::ComputeShaderRead);

        let uses = &graph.passes[0].uses;
        assert_eq!(uses[0].resource, ResourceUse::Texture(bloom.mip(0)));
        assert_eq!(uses[0].role, UseRole::Read);
        assert_eq!(uses[1].resource, ResourceUse::Texture(bloom.mip(1)));
        assert_eq!(uses[1].role, UseRole::Write);
        assert_eq!(uses[2].resource, ResourceUse::Buffer(lights));
        assert_eq!(uses[2].access, Access::ComputeShaderRead.into());
    }

    #[test]
    fn a_whole_texture_is_every_mip_of_every_layer() {
        let (_, _, bloom, _) = graph();
        assert_eq!(
            ResourceUse::from(bloom),
            ResourceUse::Texture(bloom.subresource(Subresource::ALL))
        );
    }

    #[test]
    fn queues_and_side_effects_are_recorded() {
        let (mut graph, _, _, lights) = graph();
        graph
            .compute_pass("cull", ComputeRecorder::dispatch("cull"))
            .queue(QueueKind::Compute)
            .write(lights, Access::ComputeShaderWrite);
        graph
            .transfer_pass("readback", "readback")
            .queue(QueueKind::Transfer)
            .read(lights, Access::CopyRead)
            .side_effects();

        assert_eq!(graph.passes[0].queue, QueueKind::Compute);
        assert!(!graph.passes[0].side_effects);
        assert_eq!(
            graph.passes[0].kind,
            PassKind::Compute(ComputeRecorder::Dispatch("cull".to_owned()))
        );
        assert_eq!(graph.passes[1].queue, QueueKind::Transfer);
        assert!(graph.passes[1].side_effects);
    }
}
