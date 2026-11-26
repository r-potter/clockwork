//! Passes and their uses of resources (design 0003, "Passes and
//! resources").
//!
//! A pass's kind decides what it can declare, so each kind has a builder
//! with only the methods it allows: a fullscreen pass has colour
//! attachments, a compute pass reads and writes through shaders, and so on.
//! Uses are RHI [`AccessSet`]s on a resource, or on a [`Subresource`] of a
//! texture. Attachments are uses too, with the accesses design 0002's
//! "Synchronization" gives them.

use std::ops::{Bound, Range, RangeBounds};

use clockwork_rhi::{Access, AccessSet, QueueKind, Subresource};

use crate::desc::{BufferId, SamplerId, TextureId};

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
            mip: TargetMip::Index(mip),
            layer,
        }
    }

    /// The mip a pass repeated for each mip of the texture is making, in
    /// every layer.
    pub fn current_mip(self) -> RepeatedUse {
        self.relative_mip(RelativeMip::Current)
    }

    /// The mip next to the current one on the side the chain is walked
    /// from, in every layer: the one above in a pass repeated down the
    /// chain, and the one below in a pass repeated up it. The repetition
    /// before made it, except in the first repetition, where an earlier
    /// pass must have.
    pub fn previous_mip(self) -> RepeatedUse {
        self.relative_mip(RelativeMip::Previous)
    }

    /// The texture's mip `mip`, relative to a repetition, in every layer.
    fn relative_mip(self, mip: RelativeMip) -> RepeatedUse {
        RepeatedUse {
            texture: self,
            mip,
            layers: 0..Subresource::REMAINING,
        }
    }
}

/// A mip of the texture a pass is repeated for, relative to the repetition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelativeMip {
    /// The mip the repetition makes.
    Current,
    /// The mip next to it on the side the chain is walked from: the one
    /// above in a pass repeated down the chain, and the one below in a pass
    /// repeated up it. The repetition before made it, except in the first
    /// repetition, where an earlier pass must have.
    Previous,
}

/// Which way a repeated pass walks its texture's mip chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MipDirection {
    /// From larger mips to smaller, each repetition reading the mip above
    /// the one it makes, as a downsample does.
    Down,
    /// From smaller mips to larger, each repetition reading the mip below
    /// the one it makes, as an upsample does.
    Up,
}

/// How a pass is repeated for each mip of a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Repeat {
    /// The texture.
    pub(crate) texture: TextureId,
    /// The first mip, counting from mip 0, that a repetition makes, or
    /// `None` for the first that has a previous mip.
    pub(crate) start: Option<u32>,
    /// One past the last mip, counting from mip 0, that a repetition makes,
    /// or `None` for one past the last that has a previous mip.
    pub(crate) end: Option<u32>,
    /// Which way the repetitions walk the chain.
    pub(crate) direction: MipDirection,
}

impl Repeat {
    /// Repetitions of `texture`'s mips in `mips`, in `direction`, with
    /// closed bounds as a start and an exclusive end. A bound that would
    /// pass `u32::MAX` saturates there, which is past any texture's last
    /// mip, since a texture has at most 32, so compiling reports it.
    fn new(texture: TextureId, mips: impl RangeBounds<u32>, direction: MipDirection) -> Self {
        let start = match mips.start_bound() {
            Bound::Included(&start) => Some(start),
            Bound::Excluded(&start) => Some(start.saturating_add(1)),
            Bound::Unbounded => None,
        };
        let end = match mips.end_bound() {
            Bound::Included(&end) => Some(end.saturating_add(1)),
            Bound::Excluded(&end) => Some(end),
            Bound::Unbounded => None,
        };
        Self {
            texture,
            start,
            end,
            direction,
        }
    }

    /// The mips the repetitions make, as a start and an exclusive end, for
    /// a texture of `mips` mips. An open bound goes as far as the chain
    /// allows while each repetition has a previous mip: going down, from
    /// mip 1 to the last, and going up, from mip 0 to the one before the
    /// last.
    ///
    /// If `clamp`, a closed end goes no further than an open one would, as
    /// for a texture sized by the output, whose chain gets shorter as the
    /// window shrinks, so a range that fits a large window still compiles
    /// in a small one, making fewer mips. Otherwise the range isn't checked
    /// against the chain, so it can be empty or pass the last mip.
    pub(crate) fn resolve(self, mips: u32, clamp: bool) -> Range<u32> {
        let (start, end) = match self.direction {
            MipDirection::Down => (1, mips),
            MipDirection::Up => (0, mips.saturating_sub(1)),
        };
        let closed_end = match self.end {
            Some(closed) if clamp => closed.min(end),
            Some(closed) => closed,
            None => end,
        };
        self.start.unwrap_or(start)..closed_end
    }
}

/// Part of the texture a pass is repeated for, naming its mip relative to
/// each repetition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RepeatedUse {
    /// The texture, which must be the one the pass is repeated for.
    pub texture: TextureId,
    /// The mip, relative to each repetition.
    pub mip: RelativeMip,
    /// The layers used.
    pub layers: Range<u32>,
}

impl RepeatedUse {
    /// Layer `layer` of the mip, for rendering to as an attachment.
    pub fn attachment(self, layer: u32) -> AttachmentTarget {
        AttachmentTarget {
            texture: self.texture,
            mip: TargetMip::Relative(self.mip),
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
    pub mip: TargetMip,
    /// The layer rendered to.
    pub layer: u32,
}

/// The mip an attachment renders to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetMip {
    /// This mip.
    Index(u32),
    /// A mip relative to each repetition of a pass repeated for each mip of
    /// the texture.
    Relative(RelativeMip),
}

impl AttachmentTarget {
    /// The part of the texture the attachment uses.
    fn resource_use(self) -> ResourceUse {
        match self.mip {
            TargetMip::Index(mip) => ResourceUse::Texture(self.texture.subresource(Subresource {
                mips: one(mip),
                layers: one(self.layer),
            })),
            TargetMip::Relative(mip) => ResourceUse::Repeated(RepeatedUse {
                texture: self.texture,
                mip,
                layers: one(self.layer),
            }),
        }
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
    /// Part of the texture a pass is repeated for, whose mip is relative to
    /// each repetition.
    Repeated(RepeatedUse),
}

impl From<RepeatedUse> for ResourceUse {
    /// The part of the repeated texture `texture` names.
    fn from(texture: RepeatedUse) -> Self {
        Self::Repeated(texture)
    }
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
    /// How it's repeated for each mip of a texture, if it is.
    pub(crate) repeat: Option<Repeat>,
    /// What the graph puts in its push data, in order, for a pass the graph
    /// records.
    pub(crate) bindings: Vec<Binding>,
}

/// A constant a binding puts in a pass's push data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Constant {
    /// A 4-byte `float`.
    Float(f32),
    /// A 4-byte `int`.
    Int(i32),
    /// A 4-byte `uint`.
    Uint(u32),
    /// A 16-byte `float4`.
    Float4([f32; 4]),
}

/// One entry in the push data of a pass the graph records.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Binding {
    /// The resource of the pass's use at this index in its uses: a texture
    /// view's heap index, or a buffer's address.
    Use(usize),
    /// A sampler's heap index.
    Sampler(SamplerId),
    /// A constant.
    Constant(Constant),
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
    /// and read dispatch arguments, transfer passes copy, and fullscreen
    /// passes read through their fragment shader, by their bindings.
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
            Self::Fullscreen { .. } => matches!(
                access,
                Access::FragmentShaderRead | Access::FragmentShaderSampledRead
            ),
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

impl UseRole {
    /// Whether a use declared so, with `access`, reads the resource's
    /// contents from before the pass: a read, a write that also reads, or
    /// an attachment that keeps its contents. An attachment that's cleared
    /// or discarded doesn't, though its accesses include depth testing or
    /// blending, since it starts afresh.
    pub(crate) fn reads_contents(self, access: AccessSet) -> bool {
        match self {
            Self::Read => true,
            Self::Write => access.iter().any(|access| !access.is_write()),
            Self::Colour(attachment) => matches!(attachment, Attachment::Keep),
            Self::Depth(attachment) => matches!(attachment, Attachment::Keep),
        }
    }

    /// Whether a use declared so replaces the resource's contents, whatever
    /// it covers: an attachment that's cleared or discarded, whose load
    /// operation drops the contents from before the pass. Any other write,
    /// such as a copy or a storage write, may cover only part of what it
    /// names, leaving the rest as it was.
    pub(crate) fn replaces_contents(self) -> bool {
        matches!(
            self,
            Self::Colour(Attachment::Clear(_) | Attachment::Discard)
                | Self::Depth(Attachment::Clear(_) | Attachment::Discard)
        )
    }
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

    /// Binds `resource` with `access` as the next entry of the push data,
    /// and adds it as a use: a write if `access` writes, and a read if not.
    fn bind(&mut self, resource: ResourceUse, access: AccessSet) {
        let role = if access.is_write() {
            UseRole::Write
        } else {
            UseRole::Read
        };
        let index = self.pass.uses.len();
        self.add_use(resource, access, role);
        self.pass.bindings.push(Binding::Use(index));
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
                    target.into().resource_use(),
                    colour_access(attachment),
                    UseRole::Colour(attachment),
                );
                self
            }
        }
    };
}

/// Generates the `for_each_mip` method, for kinds that can be repeated.
macro_rules! for_each_mip {
    ($name:ident) => {
        impl $name<'_> {
            /// Repeats the pass for each mip of `texture` in `mips`, walking
            /// the chain in `direction`, so a chain is built from one pass.
            /// Each repetition is a pass of its own, named with its mip,
            /// whose uses of [`TextureId::current_mip`] and
            /// [`TextureId::previous_mip`] name the mip it makes and the one
            /// it makes it from: the mip above it going
            /// [`Down`](MipDirection::Down), as a downsample does, and the
            /// one below going [`Up`](MipDirection::Up), as an upsample
            /// does. Each repetition's previous mip is the one the
            /// repetition before made, except the first's, which an earlier
            /// pass must write.
            ///
            /// `mips` names the mips the repetitions make, whichever way
            /// they walk. An open bound goes as far as the chain allows:
            /// going down, `..` is every mip after the first, and `1..6`
            /// makes mips 1 to 5; going up, `..` is every mip but the last,
            /// and `..5` makes mips 4 to 0.
            ///
            /// For a texture sized by the output, whose chain gets shorter
            /// as the window shrinks, a closed end goes no further than an
            /// open one would, so a small window makes fewer mips.
            ///
            /// Compiling reports a texture that's imported, whose mips the
            /// graph doesn't know, or that has one mip, and a range that
            /// passes the chain, makes no mips, or, if the pass reads the
            /// previous mip, starts where there's none.
            pub fn for_each_mip(
                &mut self,
                texture: TextureId,
                direction: MipDirection,
                mips: impl RangeBounds<u32>,
            ) -> &mut Self {
                self.0.pass.repeat = Some(Repeat::new(texture, mips, direction));
                self
            }
        }
    };
}

/// Generates the `bind`, `bind_sampler` and `bind_constant` methods, for
/// kinds the graph can record.
macro_rules! bindings {
    ($name:ident) => {
        impl $name<'_> {
            /// Binds `resource` with `access` as the next entry of the
            /// pass's push data: a texture's view, by its heap index, or a
            /// buffer's address. It's a use of the resource too: a write if
            /// `access` writes, and a read if not.
            ///
            /// Only a pass the graph records has bindings, since a recorder
            /// sets its own push data. Compiling reports bindings on a
            /// compute pass a recorder records.
            pub fn bind(
                &mut self,
                resource: impl Into<ResourceUse>,
                access: impl Into<AccessSet>,
            ) -> &mut Self {
                self.0.bind(resource.into(), access.into());
                self
            }

            /// Binds `sampler` as the next entry of the pass's push data, by
            /// its heap index.
            pub fn bind_sampler(&mut self, sampler: SamplerId) -> &mut Self {
                self.0.pass.bindings.push(Binding::Sampler(sampler));
                self
            }

            /// Binds `constant` as the next entry of the pass's push data.
            pub fn bind_constant(&mut self, constant: Constant) -> &mut Self {
                self.0.pass.bindings.push(Binding::Constant(constant));
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
            target.into().resource_use(),
            depth_access(attachment, write),
            UseRole::Depth(attachment),
        );
        self
    }
}

builder! {
    /// Builds a fullscreen pass: colour attachments, drawn to by the graph,
    /// and the bindings its program reads.
    FullscreenPassBuilder
}
colour!(FullscreenPassBuilder);
bindings!(FullscreenPassBuilder);
for_each_mip!(FullscreenPassBuilder);

builder! {
    /// Builds a compute pass: reads and writes through shaders, on the
    /// graphics or compute queue, and the bindings of one the graph
    /// dispatches.
    ComputePassBuilder
}
read_write!(ComputePassBuilder);
bindings!(ComputePassBuilder);
for_each_mip!(ComputePassBuilder);
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

    use clockwork_rhi::{Extent3d, Format};

    use super::*;
    use crate::{GraphDesc, TextureSize, TransientTexture};

    /// A graph with a one-mip texture, a four-mip texture and a buffer.
    fn graph() -> (GraphDesc, TextureId, TextureId, BufferId) {
        let mut graph = GraphDesc::new("passes");
        let size = TextureSize::Output { scale: 1.0 };
        let depth =
            graph.transient_texture("depth", TransientTexture::new(Format::Depth32Float, size));
        let mut chain = TransientTexture::new(
            Format::Rgba16Float,
            TextureSize::Fixed(Extent3d::new_2d(256, 256)),
        );
        chain.mips = crate::MipCount::Count(4);
        let bloom = graph.transient_texture("bloom", chain);
        let lights = graph.transient_buffer("lights", 1024);
        (graph, depth, bloom, lights)
    }

    /// Attachments get the accesses their start implies: a kept colour
    /// attachment reads as well as writes.
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

    /// Clearing or discarding depth writes it, even when the pass doesn't
    /// write depth.
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

    /// An attachment is one mip of one layer, mip 0 of layer 0 for a plain
    /// texture ID.
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

    /// The largest mip or layer index gives an empty range for validation
    /// to report, rather than overflowing.
    #[test]
    fn the_last_mip_index_gives_an_empty_range_rather_than_overflowing() {
        let (_, _, bloom, _) = graph();
        assert_eq!(bloom.mip(u32::MAX).subresource.mips, u32::MAX..u32::MAX);
        let ResourceUse::Texture(target) = bloom.attachment(u32::MAX, u32::MAX).resource_use()
        else {
            panic!("an attachment to a mip by index uses part of a texture");
        };
        assert!(target.subresource.mips.is_empty() && target.subresource.layers.is_empty());
    }

    /// Each kind of pass may ask only for the queues it can run on.
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

    /// Reads and writes keep the parts of resources they name and how
    /// they were declared.
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

    /// A repeated pass keeps the texture it's repeated for, and its uses
    /// and attachments keep their mips relative to each repetition.
    #[test]
    fn repeated_passes_name_mips_relative_to_each_repetition() {
        let (mut graph, _, bloom, _) = graph();
        graph
            .compute_pass("downsample", ComputeRecorder::dispatch("downsample"))
            .for_each_mip(bloom, MipDirection::Down, ..)
            .read(bloom.previous_mip(), Access::ComputeShaderSampledRead)
            .write(bloom.current_mip(), Access::ComputeShaderWrite);
        graph
            .fullscreen_pass("upsample", "upsample")
            .for_each_mip(bloom, MipDirection::Up, ..)
            .colour(bloom.current_mip().attachment(0), Attachment::Discard);
        graph.compute_pass("once", ComputeRecorder::dispatch("once"));

        let repeated = |mip, layers| {
            ResourceUse::Repeated(RepeatedUse {
                texture: bloom,
                mip,
                layers,
            })
        };
        let downsample = &graph.passes[0];
        assert_eq!(
            downsample
                .repeat
                .map(|repeat| (repeat.texture, repeat.direction)),
            Some((bloom, MipDirection::Down))
        );
        assert_eq!(
            downsample.uses[0].resource,
            repeated(RelativeMip::Previous, Subresource::ALL.layers)
        );
        assert_eq!(
            downsample.uses[1].resource,
            repeated(RelativeMip::Current, Subresource::ALL.layers)
        );
        let upsample = &graph.passes[1];
        assert_eq!(
            upsample
                .repeat
                .map(|repeat| (repeat.texture, repeat.direction)),
            Some((bloom, MipDirection::Up))
        );
        assert_eq!(
            upsample.uses[0].resource,
            repeated(RelativeMip::Current, 0..1),
            "an attachment is one layer of the mip"
        );
        assert_eq!(graph.passes[2].repeat, None);
    }

    /// A repeated pass's mips keep their bounds as a start and an exclusive
    /// end, with an open bound left for compiling to resolve, and a bound
    /// past `u32::MAX` saturating rather than overflowing.
    #[test]
    fn repeated_mips_keep_their_bounds() {
        let (_, _, bloom, _) = graph();
        let bounds = |repeat: Repeat| (repeat.start, repeat.end);
        let down = |mips| bounds(Repeat::new(bloom, mips, MipDirection::Down));
        assert_eq!(down((Bound::Unbounded, Bound::Unbounded)), (None, None));
        assert_eq!(
            down((Bound::Included(1), Bound::Excluded(6))),
            (Some(1), Some(6))
        );
        assert_eq!(
            down((Bound::Included(2), Bound::Unbounded)),
            (Some(2), None)
        );
        assert_eq!(
            down((Bound::Excluded(1), Bound::Included(5))),
            (Some(2), Some(6))
        );
        assert_eq!(
            down((Bound::Excluded(u32::MAX), Bound::Included(u32::MAX))),
            (Some(u32::MAX), Some(u32::MAX))
        );
        assert_eq!(
            bounds(Repeat::new(bloom, ..5, MipDirection::Up)),
            (None, Some(5))
        );
        assert_eq!(
            bounds(Repeat::new(bloom, 1..=3, MipDirection::Up)),
            (Some(1), Some(4))
        );
    }

    /// An open bound resolves to as far as the chain allows while each
    /// repetition has a previous mip, and a closed one is kept, even when
    /// it's past the chain, unless its end is clamped, when it goes no
    /// further than an open one.
    #[test]
    fn open_bounds_resolve_to_the_ends_of_the_chain() {
        let (_, _, bloom, _) = graph();
        let down = |mips| Repeat::new(bloom, mips, MipDirection::Down).resolve(4, false);
        let up = |mips| Repeat::new(bloom, mips, MipDirection::Up).resolve(4, false);
        let clamped = |direction, mips| Repeat::new(bloom, mips, direction).resolve(4, true);
        assert_eq!(clamped(MipDirection::Down, 1..=9), 1..4);
        assert_eq!(clamped(MipDirection::Down, 1..=2), 1..3);
        assert_eq!(clamped(MipDirection::Up, 0..=3), 0..3);
        assert_eq!(clamped(MipDirection::Up, 3..=5), 3..3);
        assert_eq!(down((Bound::Unbounded, Bound::Unbounded)), 1..4);
        assert_eq!(down((Bound::Included(2), Bound::Unbounded)), 2..4);
        assert_eq!(down((Bound::Included(0), Bound::Excluded(9))), 0..9);
        assert_eq!(up((Bound::Unbounded, Bound::Unbounded)), 0..3);
        assert_eq!(up((Bound::Unbounded, Bound::Excluded(2))), 0..2);
        assert_eq!(up((Bound::Included(3), Bound::Unbounded)), 3..3);
        assert_eq!(
            Repeat::new(bloom, .., MipDirection::Up).resolve(1, false),
            0..0
        );
    }

    /// Bindings keep their order, and a bound resource is a use too: a
    /// write if its accesses write, and a read if not.
    #[test]
    fn bindings_keep_their_order_and_resources_become_uses() {
        let (mut graph, depth, bloom, lights) = graph();
        let output = graph.output();
        let linear = graph.sampler(
            "linear",
            crate::SamplerSettings::new(
                clockwork_rhi::Filter::Linear,
                clockwork_rhi::AddressMode::ClampToEdge,
            ),
        );
        graph
            .fullscreen_pass("composite", "composite")
            .bind(depth, Access::FragmentShaderSampledRead)
            .bind_sampler(linear)
            .bind(lights, Access::FragmentShaderRead)
            .bind_constant(Constant::Float(0.5))
            .colour(output, Attachment::Discard);
        graph
            .compute_pass("blur", ComputeRecorder::dispatch("blur"))
            .bind(bloom.mip(0), Access::ComputeShaderSampledRead)
            .bind(bloom.mip(1), Access::ComputeShaderWrite)
            .bind_constant(Constant::Float4([1.0, 0.0, 0.0, 0.0]));

        let composite = &graph.passes[0];
        assert_eq!(
            composite.bindings,
            [
                Binding::Use(0),
                Binding::Sampler(linear),
                Binding::Use(1),
                Binding::Constant(Constant::Float(0.5)),
            ]
        );
        let roles: Vec<_> = composite.uses.iter().map(|used| used.role).collect();
        assert_eq!(
            roles,
            [
                UseRole::Read,
                UseRole::Read,
                UseRole::Colour(Attachment::Discard)
            ]
        );
        let blur = &graph.passes[1];
        assert_eq!(blur.bindings[..2], [Binding::Use(0), Binding::Use(1)]);
        assert_eq!(blur.uses[1].role, UseRole::Write);
        assert_eq!(blur.uses[1].resource, ResourceUse::Texture(bloom.mip(1)));
    }

    /// A plain texture ID used as a resource is the whole texture.
    #[test]
    fn a_whole_texture_is_every_mip_of_every_layer() {
        let (_, _, bloom, _) = graph();
        assert_eq!(
            ResourceUse::from(bloom),
            ResourceUse::Texture(bloom.subresource(Subresource::ALL))
        );
    }

    /// A pass's queue and side effects are kept as asked.
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
