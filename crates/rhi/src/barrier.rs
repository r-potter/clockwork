//! Barriers, described by how a resource is used before and after them
//! (design 0002, "Synchronization" and "Subresources and views").
//!
//! An [`Access`] is a way a resource is used, such as being sampled by a
//! fragment shader or written by a copy: a single read or write, in one
//! stage. An [`AccessSet`] is every way a resource is used on one side of a
//! barrier. A [`Barrier`] names a buffer, a texture's [`Subresource`], or all
//! memory, and the access sets before and after it. Each access names the
//! engine-level [`Layout`] a texture must be in for it, and recording
//! chooses one for each set, so a texture's layout always follows from its
//! accesses, rather than being stated separately. Backends map each access
//! and layout to their own synchronization vocabulary with a table.

use std::fmt;
use std::ops::{BitOr, BitOrAssign, Range};

use crate::{Buffer, QueueId, Texture};

/// A way a resource is used, which a barrier orders against another: a
/// single read or write, in one stage. A use that reads and writes names
/// both, in an [`AccessSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    /// Not used: a new resource, or one whose contents are discarded. A
    /// texture's contents are lost by a barrier from it. Only valid alone.
    None,
    /// Read as indirect draw or dispatch arguments. Buffers only.
    IndirectRead,
    /// Read as an index buffer. Buffers only.
    IndexRead,
    /// Read by a vertex shader, through a buffer's address or a storage
    /// view.
    VertexShaderRead,
    /// Read by a fragment shader, through a buffer's address or a storage
    /// view.
    FragmentShaderRead,
    /// Sampled by a fragment shader, through a sampled view.
    FragmentShaderSampledRead,
    /// Read by a compute shader, through a buffer's address or a storage
    /// view.
    ComputeShaderRead,
    /// Sampled by a compute shader, through a sampled view.
    ComputeShaderSampledRead,
    /// Written by a compute shader, through a buffer's address or a storage
    /// view. A shader that also reads what it writes, for example with
    /// atomics, is [`ComputeShaderRead`](Self::ComputeShaderRead) too.
    ComputeShaderWrite,
    /// Read as a colour attachment, to load its contents or blend with
    /// them.
    ColorAttachmentRead,
    /// Written as a colour attachment. Rendering that loads its contents or
    /// blends with them is [`ColorAttachmentRead`](Self::ColorAttachmentRead)
    /// too.
    ColorAttachmentWrite,
    /// Depth-tested against as a depth attachment.
    DepthAttachmentRead,
    /// Written as a depth attachment. Depth testing that writes is
    /// [`DepthAttachmentRead`](Self::DepthAttachmentRead) too.
    DepthAttachmentWrite,
    /// Read by a copy.
    CopyRead,
    /// Written by a copy or a fill.
    CopyWrite,
    /// Written by the CPU, through a mapping. Buffers only.
    HostWrite,
    /// Read by the CPU, through a mapping. Buffers only.
    HostRead,
    /// Presented to the window. Textures only: a surface image ends its
    /// frame here. Only valid alone.
    Present,
}

impl Access {
    /// Every access.
    pub const ALL: [Self; 18] = [
        Self::None,
        Self::IndirectRead,
        Self::IndexRead,
        Self::VertexShaderRead,
        Self::FragmentShaderRead,
        Self::FragmentShaderSampledRead,
        Self::ComputeShaderRead,
        Self::ComputeShaderSampledRead,
        Self::ComputeShaderWrite,
        Self::ColorAttachmentRead,
        Self::ColorAttachmentWrite,
        Self::DepthAttachmentRead,
        Self::DepthAttachmentWrite,
        Self::CopyRead,
        Self::CopyWrite,
        Self::HostWrite,
        Self::HostRead,
        Self::Present,
    ];

    /// Whether the access writes the resource.
    pub const fn is_write(self) -> bool {
        matches!(
            self,
            Self::ComputeShaderWrite
                | Self::ColorAttachmentWrite
                | Self::DepthAttachmentWrite
                | Self::CopyWrite
                | Self::HostWrite
        )
    }

    /// Whether a buffer can be used this way.
    pub const fn applies_to_buffers(self) -> bool {
        !matches!(
            self,
            Self::FragmentShaderSampledRead
                | Self::ComputeShaderSampledRead
                | Self::ColorAttachmentRead
                | Self::ColorAttachmentWrite
                | Self::DepthAttachmentRead
                | Self::DepthAttachmentWrite
                | Self::Present
        )
    }

    /// Whether a texture can be used this way.
    pub const fn applies_to_textures(self) -> bool {
        self.layout().is_some()
    }

    /// The layout a texture must be in for this access alone, or `None` for
    /// an access only buffers have.
    pub(crate) const fn layout(self) -> Option<Layout> {
        match self {
            Self::None => Some(Layout::Undefined),
            Self::IndirectRead | Self::IndexRead | Self::HostWrite | Self::HostRead => None,
            Self::VertexShaderRead
            | Self::FragmentShaderRead
            | Self::ComputeShaderRead
            | Self::ComputeShaderWrite => Some(Layout::General),
            Self::FragmentShaderSampledRead
            | Self::ComputeShaderSampledRead
            | Self::DepthAttachmentRead => Some(Layout::ReadOnly),
            Self::ColorAttachmentRead | Self::ColorAttachmentWrite | Self::DepthAttachmentWrite => {
                Some(Layout::Attachment)
            }
            Self::CopyRead => Some(Layout::CopySource),
            Self::CopyWrite => Some(Layout::CopyDestination),
            Self::Present => Some(Layout::Present),
        }
    }

    /// Another layout this access can also happen in, beside accesses that
    /// need it: depth testing happens in the attachment layout beside a
    /// depth write.
    const fn also_in(self) -> Option<Layout> {
        match self {
            Self::DepthAttachmentRead => Some(Layout::Attachment),
            _ => None,
        }
    }

    /// Whether the access is only valid alone in its set.
    const fn is_alone_only(self) -> bool {
        matches!(self, Self::None | Self::Present)
    }

    /// The access's bit in an [`AccessSet`].
    const fn bit(self) -> u32 {
        1 << self as u32
    }
}

/// The layouts a texture can be in, which each [`Access`] names and
/// recording chooses for each [`AccessSet`]. Backends map each to a layout
/// of their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Layout {
    /// No contents: a new texture, or one whose contents are discarded.
    Undefined,
    /// Read and written by shaders through storage views.
    General,
    /// Sampled, or depth-tested without writing.
    ReadOnly,
    /// Rendered to as a colour or depth attachment.
    Attachment,
    /// Read by a copy.
    CopySource,
    /// Written by a copy.
    CopyDestination,
    /// Presented to the window.
    Present,
}

/// Every way a resource is used on one side of a barrier, built with `|`
/// from [`Access`] values. A single `Access` is a set of one, and a set is
/// never empty.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccessSet(u32);

impl AccessSet {
    /// The accesses in either set.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether `access` is in the set.
    pub const fn contains(self, access: Access) -> bool {
        self.0 & access.bit() != 0
    }

    /// Whether any access in the set writes the resource.
    pub fn is_write(self) -> bool {
        self.iter().any(Access::is_write)
    }

    /// The accesses in the set, in the order of [`Access::ALL`].
    pub fn iter(self) -> impl Iterator<Item = Access> {
        Access::ALL
            .into_iter()
            .filter(move |&access| self.contains(access))
    }

    /// The one layout a texture can be in for every access in the set, or
    /// `None` if there isn't one, or an access is one only buffers have.
    ///
    /// A set with writes takes their layout, and its reads must be able to
    /// happen in it. A set of only reads takes the layout they share,
    /// preferring the one its accesses use alone.
    pub(crate) fn layout(self) -> Option<Layout> {
        let can_be_in = |layout: Layout| {
            self.iter()
                .all(|access| access.layout() == Some(layout) || access.also_in() == Some(layout))
        };
        let preferred = match self.iter().find(|access| access.is_write()) {
            Some(write) => write.layout(),
            None => self.iter().find_map(Access::layout),
        }?;
        if can_be_in(preferred) {
            return Some(preferred);
        }
        // Reads that use different layouts alone may share another, such
        // as a colour attachment read beside a depth test.
        if self.is_write() {
            return None;
        }
        self.iter()
            .filter_map(Access::also_in)
            .find(|&layout| can_be_in(layout))
    }
}

impl From<Access> for AccessSet {
    /// The set of `access` alone.
    fn from(access: Access) -> Self {
        Self(access.bit())
    }
}

impl<T: Into<AccessSet>> BitOr<T> for AccessSet {
    type Output = Self;

    /// The accesses in either set, as [`union`](Self::union).
    fn bitor(self, other: T) -> Self {
        self.union(other.into())
    }
}

impl<T: Into<AccessSet>> BitOr<T> for Access {
    type Output = AccessSet;

    /// The set of this access and those in `other`.
    fn bitor(self, other: T) -> AccessSet {
        AccessSet::from(self) | other
    }
}

impl<T: Into<AccessSet>> BitOrAssign<T> for AccessSet {
    /// Adds the accesses in `other`.
    fn bitor_assign(&mut self, other: T) {
        *self = self.union(other.into());
    }
}

impl fmt::Debug for AccessSet {
    /// Formats the set as its accesses joined by `|`, as it's built.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, access) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(" | ")?;
            }
            write!(f, "{access:?}")?;
        }
        Ok(())
    }
}

/// Part of a texture: a range of mips and a range of layers.
///
/// A range ending at [`Subresource::REMAINING`] reaches the texture's last
/// mip or layer, whatever it has. A range that passes the texture's mips or
/// layers, or is empty, panics when the list is translated, naming the
/// texture.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Subresource {
    /// The mips, from the largest, 0.
    pub mips: Range<u32>,
    /// The array layers. A cube texture has six per cube.
    pub layers: Range<u32>,
}

impl Subresource {
    /// The end of a range that reaches the texture's last mip or layer.
    pub const REMAINING: u32 = u32::MAX;

    /// Every mip of every layer.
    pub const ALL: Self = Self {
        mips: 0..Self::REMAINING,
        layers: 0..Self::REMAINING,
    };

    /// Mip `mip`, in every layer.
    pub const fn mip(mip: u32) -> Self {
        Self {
            mips: mip..mip + 1,
            layers: 0..Self::REMAINING,
        }
    }

    /// Every mip of layer `layer`.
    pub const fn layer(layer: u32) -> Self {
        Self {
            mips: 0..Self::REMAINING,
            layers: layer..layer + 1,
        }
    }

    /// Mip `mip` of layer `layer`.
    pub const fn mip_layer(mip: u32, layer: u32) -> Self {
        Self {
            mips: mip..mip + 1,
            layers: layer..layer + 1,
        }
    }

    /// The subresource with its ranges resolved for the texture `label`,
    /// which has `mip_count` mips and `layer_count` layers: each range
    /// ending at [`REMAINING`](Self::REMAINING) ends at the texture's last.
    ///
    /// # Panics
    ///
    /// Panics if either range is empty or passes the texture's last mip or
    /// layer, naming the texture.
    pub(crate) fn resolve(&self, label: &str, mip_count: u32, layer_count: u32) -> Self {
        Self {
            mips: resolve_range(label, "mips", &self.mips, mip_count),
            layers: resolve_range(label, "layers", &self.layers, layer_count),
        }
    }
}

/// `range` of the `count` `items` (mips or layers) of the texture `label`,
/// with an end of [`Subresource::REMAINING`] resolved to `count`.
///
/// # Panics
///
/// Panics if the resolved range is empty or ends after `count`.
fn resolve_range(label: &str, items: &str, range: &Range<u32>, count: u32) -> Range<u32> {
    let end = if range.end == Subresource::REMAINING {
        count
    } else {
        range.end
    };
    assert!(
        range.start < end && end <= count,
        "texture \"{label}\" has {count} {items}, so a subresource can't name {items} {}",
        describe_range(range),
    );
    range.start..end
}

/// `range` as written in messages, with an end of
/// [`Subresource::REMAINING`] written as an open range.
fn describe_range(range: &Range<u32>) -> String {
    if range.end == Subresource::REMAINING {
        format!("{}..", range.start)
    } else {
        format!("{range:?}")
    }
}

/// What a barrier applies to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum BarrierTarget {
    /// All memory, as a global memory barrier.
    Global,
    /// A buffer, whole.
    Buffer(Buffer),
    /// Part of a texture.
    Texture(Texture, Subresource),
}

/// Orders the accesses to a resource before it against those after it, and
/// moves a texture's subresource to the layout its next accesses need.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Barrier {
    /// What the barrier applies to.
    target: BarrierTarget,
    /// How the resource was used before the barrier.
    before: AccessSet,
    /// How it's used after it.
    after: AccessSet,
    /// The queues ownership moves from and to, if it moves.
    transfer: Option<(QueueId, QueueId)>,
}

impl Barrier {
    /// A barrier for all memory: every access in `before` comes before every
    /// access in `after`.
    pub fn global(before: impl Into<AccessSet>, after: impl Into<AccessSet>) -> Self {
        Self {
            target: BarrierTarget::Global,
            before: before.into(),
            after: after.into(),
            transfer: None,
        }
    }

    /// A barrier for `buffer`, whole.
    pub fn buffer(
        buffer: Buffer,
        before: impl Into<AccessSet>,
        after: impl Into<AccessSet>,
    ) -> Self {
        Self {
            target: BarrierTarget::Buffer(buffer),
            before: before.into(),
            after: after.into(),
            transfer: None,
        }
    }

    /// A barrier for every mip and layer of `texture`.
    pub fn texture(
        texture: Texture,
        before: impl Into<AccessSet>,
        after: impl Into<AccessSet>,
    ) -> Self {
        Self::subresource(texture, Subresource::ALL, before, after)
    }

    /// A barrier for `subresource` of `texture`. Only that subresource's
    /// layout changes, so different mips can be in different layouts.
    pub fn subresource(
        texture: Texture,
        subresource: Subresource,
        before: impl Into<AccessSet>,
        after: impl Into<AccessSet>,
    ) -> Self {
        Self {
            target: BarrierTarget::Texture(texture, subresource),
            before: before.into(),
            after: after.into(),
            transfer: None,
        }
    }

    /// The barrier, also moving the resource's ownership from the queue
    /// `from` to the queue `to` (design 0002, "Synchronization"). Record it
    /// on both queues, with the same accesses: on `from` it releases the
    /// resource, after the accesses before it, and on `to` it acquires it,
    /// before the accesses after it, once a wait orders the two. When both
    /// are one queue, as when two kinds share a queue, or are two queues on
    /// one family, the backend drops the transfer and keeps the accesses.
    pub fn with_queue_transfer(self, from: QueueId, to: QueueId) -> Self {
        Self {
            transfer: Some((from, to)),
            ..self
        }
    }

    /// The queues ownership moves from and to, if it moves.
    pub fn queue_transfer(&self) -> Option<(QueueId, QueueId)> {
        self.transfer
    }

    /// What the barrier applies to.
    pub(crate) fn target(&self) -> &BarrierTarget {
        &self.target
    }

    /// How the resource was used before the barrier.
    pub fn before(&self) -> AccessSet {
        self.before
    }

    /// How the resource is used after the barrier.
    pub fn after(&self) -> AccessSet {
        self.after
    }

    /// Checks that the access sets suit what the barrier applies to, for a
    /// barrier recorded into the command list `list`.
    ///
    /// # Panics
    ///
    /// Panics, naming the list, if:
    /// - [`Access::None`] or [`Access::Present`] isn't alone in its set;
    /// - a buffer is given an access only textures have, or a texture one
    ///   only buffers have;
    /// - all memory is given [`Access::Present`], which needs a layout;
    /// - a texture is moved to [`Access::None`], which has no layout to move
    ///   to;
    /// - a texture's set has no one layout, such as sampled and storage
    ///   access together.
    /// - a queue transfer is for all memory, which no queue owns.
    pub(crate) fn check(&self, list: &str) {
        for set in [self.before, self.after] {
            for access in set.iter() {
                assert!(
                    !access.is_alone_only() || set == AccessSet::from(access),
                    "command list \"{list}\" has a barrier with {set:?}, but {access:?} must be \
                     alone",
                );
            }
        }
        let (target, applies): (_, fn(Access) -> bool) = match self.target {
            // A global barrier orders memory accesses only, with no layout,
            // so it can't present.
            BarrierTarget::Global => ("all memory", |access| access != Access::Present),
            BarrierTarget::Buffer(_) => ("a buffer", Access::applies_to_buffers),
            BarrierTarget::Texture(..) => ("a texture", Access::applies_to_textures),
        };
        for access in self.before.iter().chain(self.after.iter()) {
            assert!(
                applies(access),
                "command list \"{list}\" has a barrier for {target} with {access:?}, which \
                 doesn't apply to {target}",
            );
        }
        if self.transfer.is_some() {
            assert!(
                self.target != BarrierTarget::Global,
                "command list \"{list}\" has a barrier for all memory with a queue transfer, but \
                 only a buffer or texture can move between queues",
            );
        }
        if let BarrierTarget::Texture(..) = self.target {
            assert!(
                !self.after.contains(Access::None),
                "command list \"{list}\" has a barrier for a texture to Access::None, but a \
                 texture's contents are only discarded by a barrier from it",
            );
            for set in [self.before, self.after] {
                assert!(
                    set.layout().is_some(),
                    "command list \"{list}\" has a barrier for a texture with {set:?}, which \
                     have no one layout",
                );
            }
        }
    }
}

// Barriers are plain data, recorded on any thread (design 0002,
// "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Access>();
    assert_send_sync::<AccessSet>();
    assert_send_sync::<Subresource>();
    assert_send_sync::<Barrier>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of accesses, access sets, subresources and barrier checks
    //! (design 0002, R8 and part of R21).

    use super::*;
    use crate::handle::HandleTable;

    /// A graphics queue's ID, for tests.
    const GRAPHICS: QueueId = QueueId::new(0);

    /// A transfer queue's ID, for tests.
    const TRANSFER: QueueId = QueueId::new(1);

    /// A texture handle for tests, from a table of its own.
    fn texture() -> Texture {
        Texture::from_handle(HandleTable::new("texture").insert("texture", ()))
    }

    /// A buffer handle for tests, from a table of its own.
    fn buffer() -> Buffer {
        Buffer::from_handle(HandleTable::new("buffer").insert("buffer", ()))
    }

    /// Every access applies to buffers, textures or both.
    #[test]
    fn every_access_applies_to_something() {
        for access in Access::ALL {
            assert!(
                access.applies_to_buffers() || access.applies_to_textures(),
                "{access:?}"
            );
        }
    }

    /// The writes are the accesses that write, and `None` isn't one.
    #[test]
    fn writes_are_the_writing_accesses() {
        let writes: Vec<_> = Access::ALL
            .into_iter()
            .filter(|access| access.is_write())
            .collect();
        assert_eq!(
            writes,
            [
                Access::ComputeShaderWrite,
                Access::ColorAttachmentWrite,
                Access::DepthAttachmentWrite,
                Access::CopyWrite,
                Access::HostWrite,
            ],
        );
    }

    /// Each access names the layout made for it: storage access is
    /// `General`, sampling and depth testing `ReadOnly`, and attachment
    /// writes and colour reads `Attachment`. Exactly the accesses only
    /// buffers have name none.
    #[test]
    fn accesses_name_their_layouts() {
        use Layout as L;
        let expected = [
            (Access::None, Some(L::Undefined)),
            (Access::IndirectRead, None),
            (Access::IndexRead, None),
            (Access::VertexShaderRead, Some(L::General)),
            (Access::FragmentShaderRead, Some(L::General)),
            (Access::FragmentShaderSampledRead, Some(L::ReadOnly)),
            (Access::ComputeShaderRead, Some(L::General)),
            (Access::ComputeShaderSampledRead, Some(L::ReadOnly)),
            (Access::ComputeShaderWrite, Some(L::General)),
            (Access::ColorAttachmentRead, Some(L::Attachment)),
            (Access::ColorAttachmentWrite, Some(L::Attachment)),
            (Access::DepthAttachmentRead, Some(L::ReadOnly)),
            (Access::DepthAttachmentWrite, Some(L::Attachment)),
            (Access::CopyRead, Some(L::CopySource)),
            (Access::CopyWrite, Some(L::CopyDestination)),
            (Access::HostWrite, None),
            (Access::HostRead, None),
            (Access::Present, Some(L::Present)),
        ];
        assert_eq!(
            expected.map(|(access, _)| access),
            Access::ALL,
            "every access is listed"
        );
        for (access, layout) in expected {
            assert_eq!(access.layout(), layout, "{access:?}");
        }
    }

    /// Sets are built with `|` from accesses, hold each access once, and
    /// list them in a fixed order.
    #[test]
    fn sets_combine_accesses() {
        let set = Access::ComputeShaderWrite | Access::ComputeShaderRead;
        assert_eq!(
            set.iter().collect::<Vec<_>>(),
            [Access::ComputeShaderRead, Access::ComputeShaderWrite],
        );
        assert_eq!(set | Access::ComputeShaderRead, set);
        let mut grown = AccessSet::from(Access::ComputeShaderRead);
        grown |= Access::ComputeShaderWrite;
        assert_eq!(grown, set);
        assert!(set.is_write());
        assert!(!AccessSet::from(Access::ComputeShaderRead).is_write());
        assert_eq!(format!("{set:?}"), "ComputeShaderRead | ComputeShaderWrite");
    }

    /// A set takes its writes' layout, reads share the layout they use
    /// alone, and depth tests move to the attachment layout beside a depth
    /// write.
    #[test]
    fn sets_have_one_layout() {
        use Layout as L;
        for (set, layout) in [
            (
                Access::ComputeShaderRead | Access::ComputeShaderWrite,
                L::General,
            ),
            (
                Access::ColorAttachmentRead | Access::ColorAttachmentWrite,
                L::Attachment,
            ),
            (
                Access::DepthAttachmentRead | Access::DepthAttachmentWrite,
                L::Attachment,
            ),
            (Access::DepthAttachmentRead.into(), L::ReadOnly),
            (
                Access::DepthAttachmentRead | Access::FragmentShaderSampledRead,
                L::ReadOnly,
            ),
            (
                Access::FragmentShaderSampledRead | Access::ComputeShaderSampledRead,
                L::ReadOnly,
            ),
            (
                Access::VertexShaderRead | Access::FragmentShaderRead,
                L::General,
            ),
            (
                Access::DepthAttachmentRead | Access::ColorAttachmentRead,
                L::Attachment,
            ),
        ] {
            assert_eq!(set.layout(), Some(layout), "{set:?}");
        }
    }

    /// Sets whose accesses can't share a layout have none.
    #[test]
    fn mixed_sets_have_no_layout() {
        for set in [
            Access::FragmentShaderSampledRead | Access::FragmentShaderRead,
            Access::CopyWrite | Access::ComputeShaderWrite,
            Access::DepthAttachmentWrite | Access::FragmentShaderSampledRead,
            Access::CopyRead | Access::FragmentShaderSampledRead,
            Access::HostRead.into(),
        ] {
            assert_eq!(set.layout(), None, "{set:?}");
        }
    }

    /// The named subresources cover what their names say, with
    /// `REMAINING` for the texture's last.
    #[test]
    fn subresource_constructors_name_their_ranges() {
        assert_eq!(
            Subresource::ALL.resolve("t", 4, 6),
            Subresource {
                mips: 0..4,
                layers: 0..6
            },
        );
        assert_eq!(
            Subresource::mip(2).resolve("t", 4, 6),
            Subresource {
                mips: 2..3,
                layers: 0..6
            },
        );
        assert_eq!(
            Subresource::layer(5).resolve("t", 4, 6),
            Subresource {
                mips: 0..4,
                layers: 5..6
            },
        );
        assert_eq!(
            Subresource::mip_layer(3, 1).resolve("t", 4, 6),
            Subresource {
                mips: 3..4,
                layers: 1..2
            },
        );
    }

    /// A range reaching exactly the texture's last mip or layer is in
    /// range.
    #[test]
    fn subresources_may_reach_the_last_mip_and_layer() {
        let subresource = Subresource {
            mips: 1..4,
            layers: 0..6,
        };
        assert_eq!(subresource.resolve("t", 4, 6), subresource);
    }

    /// A mip past the texture's last panics, naming the texture.
    #[test]
    #[should_panic(
        expected = "texture \"albedo\" has 4 mips, so a subresource can't name mips 4..5"
    )]
    fn mips_past_the_last_panic() {
        Subresource::mip(4).resolve("albedo", 4, 1);
    }

    /// A layer range past the texture's last panics.
    #[test]
    #[should_panic(
        expected = "texture \"shadows\" has 4 layers, so a subresource can't name layers 2..6"
    )]
    fn layers_past_the_last_panic() {
        Subresource {
            mips: 0..1,
            layers: 2..6,
        }
        .resolve("shadows", 1, 4);
    }

    /// A range starting past the texture's last, reaching `REMAINING`,
    /// panics, rather than resolving to nothing.
    #[test]
    #[should_panic(
        expected = "texture \"albedo\" has 1 layers, so a subresource can't name layers 1.."
    )]
    fn remaining_ranges_starting_past_the_last_panic() {
        Subresource {
            mips: 0..1,
            layers: 1..Subresource::REMAINING,
        }
        .resolve("albedo", 1, 1);
    }

    /// An empty range panics.
    #[test]
    #[should_panic(
        expected = "texture \"albedo\" has 4 mips, so a subresource can't name mips 2..2"
    )]
    fn empty_ranges_panic() {
        Subresource {
            mips: 2..2,
            layers: 0..1,
        }
        .resolve("albedo", 4, 1);
    }

    /// Barriers whose accesses suit what they apply to pass, including
    /// discarding a texture's contents with a barrier from `None`, and sets
    /// with one layout.
    #[test]
    fn suitable_barriers_pass() {
        Barrier::global(Access::CopyWrite, Access::HostRead).check("list");
        Barrier::global(Access::ColorAttachmentWrite, Access::FragmentShaderRead).check("list");
        Barrier::buffer(buffer(), Access::HostWrite, Access::CopyRead).check("list");
        Barrier::buffer(
            buffer(),
            Access::CopyWrite,
            Access::ComputeShaderRead | Access::ComputeShaderWrite | Access::IndirectRead,
        )
        .check("list");
        Barrier::texture(texture(), Access::None, Access::ColorAttachmentWrite).check("list");
        Barrier::texture(
            texture(),
            Access::DepthAttachmentRead | Access::DepthAttachmentWrite,
            Access::DepthAttachmentRead | Access::FragmentShaderSampledRead,
        )
        .check("list");
        Barrier::subresource(
            texture(),
            Subresource::mip(1),
            Access::ComputeShaderWrite,
            Access::FragmentShaderSampledRead,
        )
        .check("list");
    }

    /// Queue transfers are kept, and checked: a buffer or texture can move
    /// between two queues, or name one queue at both ends, as when two
    /// kinds share it.
    #[test]
    fn queue_transfers_are_kept() {
        let barrier = Barrier::buffer(buffer(), Access::CopyWrite, Access::CopyRead)
            .with_queue_transfer(TRANSFER, GRAPHICS);
        barrier.check("list");
        assert_eq!(barrier.queue_transfer(), Some((TRANSFER, GRAPHICS)));
        Barrier::texture(
            texture(),
            Access::CopyWrite,
            Access::FragmentShaderSampledRead,
        )
        .with_queue_transfer(TRANSFER, GRAPHICS)
        .check("list");
        Barrier::buffer(buffer(), Access::CopyWrite, Access::CopyRead)
            .with_queue_transfer(GRAPHICS, GRAPHICS)
            .check("list");
        assert_eq!(
            Barrier::global(Access::CopyWrite, Access::CopyRead).queue_transfer(),
            None
        );
    }

    /// A global barrier can't move between queues, since no queue owns all
    /// memory.
    #[test]
    #[should_panic(
        expected = "command list \"upload\" has a barrier for all memory with a queue transfer, \
                    but only a buffer or texture can move between queues"
    )]
    fn global_queue_transfers_panic() {
        Barrier::global(Access::CopyWrite, Access::CopyRead)
            .with_queue_transfer(TRANSFER, GRAPHICS)
            .check("upload");
    }

    /// A buffer barrier with an access only textures have panics, naming
    /// the list.
    #[test]
    #[should_panic(
        expected = "command list \"frame\" has a barrier for a buffer with ColorAttachmentWrite, \
                    which doesn't apply to a buffer"
    )]
    fn buffer_barriers_with_texture_accesses_panic() {
        Barrier::buffer(buffer(), Access::ColorAttachmentWrite, Access::CopyRead).check("frame");
    }

    /// A global barrier can't present, which needs a texture's layout.
    #[test]
    #[should_panic(
        expected = "command list \"frame\" has a barrier for all memory with Present, which \
                    doesn't apply to all memory"
    )]
    fn global_barriers_to_present_panic() {
        Barrier::global(Access::ColorAttachmentWrite, Access::Present).check("frame");
    }

    /// A texture barrier with an access only buffers have panics, in any
    /// position in its set.
    #[test]
    #[should_panic(
        expected = "command list \"frame\" has a barrier for a texture with HostRead, which \
                    doesn't apply to a texture"
    )]
    fn texture_barriers_with_buffer_accesses_panic() {
        Barrier::texture(
            texture(),
            Access::CopyWrite,
            Access::CopyRead | Access::HostRead,
        )
        .check("frame");
    }

    /// A texture barrier to `None` panics, since there's no layout to move
    /// it to.
    #[test]
    #[should_panic(
        expected = "command list \"frame\" has a barrier for a texture to Access::None, but a \
                    texture's contents are only discarded by a barrier from it"
    )]
    fn texture_barriers_to_none_panic() {
        Barrier::texture(texture(), Access::CopyRead, Access::None).check("frame");
    }

    /// `None` beside another access panics, naming the set.
    #[test]
    #[should_panic(
        expected = "command list \"frame\" has a barrier with None | CopyRead, but None must be \
                    alone"
    )]
    fn none_beside_another_access_panics() {
        Barrier::buffer(buffer(), Access::None | Access::CopyRead, Access::HostRead).check("frame");
    }

    /// `Present` beside another access panics, naming the set.
    #[test]
    #[should_panic(
        expected = "command list \"frame\" has a barrier with ColorAttachmentWrite | Present, \
                    but Present must be alone"
    )]
    fn present_beside_another_access_panics() {
        Barrier::texture(
            texture(),
            Access::ColorAttachmentWrite,
            Access::ColorAttachmentWrite | Access::Present,
        )
        .check("frame");
    }

    /// A texture barrier with a set that has no one layout panics, naming
    /// the list and the set.
    #[test]
    #[should_panic(expected = "command list \"frame\" has a barrier for a texture with \
                    FragmentShaderRead | FragmentShaderSampledRead, which have no one layout")]
    fn texture_sets_with_no_layout_panic() {
        Barrier::texture(
            texture(),
            Access::ComputeShaderWrite,
            Access::FragmentShaderSampledRead | Access::FragmentShaderRead,
        )
        .check("frame");
    }

    /// A buffer barrier's sets need no layout, so any accesses buffers can
    /// have can share one.
    #[test]
    fn buffer_sets_need_no_layout() {
        Barrier::buffer(
            buffer(),
            Access::CopyWrite,
            Access::VertexShaderRead | Access::IndexRead | Access::CopyRead,
        )
        .check("list");
    }
}
