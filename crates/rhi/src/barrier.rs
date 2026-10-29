//! Barriers, described by how a resource is used before and after them
//! (design 0002, "Synchronization" and "Subresources and views").
//!
//! An [`Access`] is a way a resource is used, such as being sampled by a
//! fragment shader or written by a copy. A [`Barrier`] names a buffer, a
//! texture's [`Subresource`], or all memory, and the accesses before and
//! after it. Backends map each access to their own synchronization
//! vocabulary with a table, so a texture's layout always follows from its
//! access, rather than being stated separately.

use std::ops::Range;

use crate::{Buffer, Texture};

/// A way a resource is used, which a barrier orders against another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    /// Not used: a new resource, or one whose contents are discarded. A
    /// texture's contents are lost by a barrier from it.
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
    /// view.
    ComputeShaderWrite,
    /// Rendered to as a colour attachment.
    ColorAttachmentWrite,
    /// Depth-tested against as a depth attachment, without writing it.
    DepthAttachmentRead,
    /// Depth-tested and written as a depth attachment.
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
    /// frame here.
    Present,
}

impl Access {
    /// Every access.
    pub const ALL: [Self; 17] = [
        Self::None,
        Self::IndirectRead,
        Self::IndexRead,
        Self::VertexShaderRead,
        Self::FragmentShaderRead,
        Self::FragmentShaderSampledRead,
        Self::ComputeShaderRead,
        Self::ComputeShaderSampledRead,
        Self::ComputeShaderWrite,
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
                | Self::ColorAttachmentWrite
                | Self::DepthAttachmentRead
                | Self::DepthAttachmentWrite
                | Self::Present
        )
    }

    /// Whether a texture can be used this way.
    pub const fn applies_to_textures(self) -> bool {
        !matches!(
            self,
            Self::IndirectRead | Self::IndexRead | Self::HostWrite | Self::HostRead
        )
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
/// moves a texture's subresource to the layout its next access needs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Barrier {
    /// What the barrier applies to.
    target: BarrierTarget,
    /// How the resource was used before the barrier.
    before: Access,
    /// How it's used after it.
    after: Access,
}

impl Barrier {
    /// A barrier for all memory: every access of the kind `before` comes
    /// before every access of the kind `after`.
    pub fn global(before: Access, after: Access) -> Self {
        Self {
            target: BarrierTarget::Global,
            before,
            after,
        }
    }

    /// A barrier for `buffer`, whole.
    pub fn buffer(buffer: Buffer, before: Access, after: Access) -> Self {
        Self {
            target: BarrierTarget::Buffer(buffer),
            before,
            after,
        }
    }

    /// A barrier for every mip and layer of `texture`.
    pub fn texture(texture: Texture, before: Access, after: Access) -> Self {
        Self::subresource(texture, Subresource::ALL, before, after)
    }

    /// A barrier for `subresource` of `texture`. Only that subresource's
    /// layout changes, so different mips can be in different layouts.
    pub fn subresource(
        texture: Texture,
        subresource: Subresource,
        before: Access,
        after: Access,
    ) -> Self {
        Self {
            target: BarrierTarget::Texture(texture, subresource),
            before,
            after,
        }
    }

    /// What the barrier applies to.
    pub(crate) fn target(&self) -> &BarrierTarget {
        &self.target
    }

    /// How the resource was used before the barrier.
    pub fn before(&self) -> Access {
        self.before
    }

    /// How the resource is used after the barrier.
    pub fn after(&self) -> Access {
        self.after
    }

    /// Checks that the accesses suit what the barrier applies to, for a
    /// barrier recorded into the command list `list`.
    ///
    /// # Panics
    ///
    /// Panics, naming the list, if a buffer is given an access only
    /// textures have, a texture is given one only buffers have, all memory
    /// is given [`Access::Present`], which needs a layout, or a texture is
    /// moved to [`Access::None`], which has no layout to move to.
    pub(crate) fn check(&self, list: &str) {
        let (target, applies): (_, fn(Access) -> bool) = match self.target {
            // A global barrier orders memory accesses only, with no layout,
            // so it can't present.
            BarrierTarget::Global => ("all memory", |access| access != Access::Present),
            BarrierTarget::Buffer(_) => ("a buffer", Access::applies_to_buffers),
            BarrierTarget::Texture(..) => ("a texture", Access::applies_to_textures),
        };
        for access in [self.before, self.after] {
            assert!(
                applies(access),
                "command list \"{list}\" has a barrier for {target} with {access:?}, which \
                 doesn't apply to {target}",
            );
        }
        if let BarrierTarget::Texture(..) = self.target {
            assert!(
                self.after != Access::None,
                "command list \"{list}\" has a barrier for a texture to Access::None, but a \
                 texture's contents are only discarded by a barrier from it",
            );
        }
    }
}

// Barriers are plain data, recorded on any thread (design 0002,
// "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Access>();
    assert_send_sync::<Subresource>();
    assert_send_sync::<Barrier>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of accesses, subresources and barrier checks (design 0002,
    //! R8 and part of R21).

    use super::*;
    use crate::handle::HandleTable;

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

    /// Barriers whose accesses suit what they apply to pass, including
    /// discarding a texture's contents with a barrier from `None`.
    #[test]
    fn suitable_barriers_pass() {
        Barrier::global(Access::CopyWrite, Access::HostRead).check("list");
        Barrier::global(Access::ColorAttachmentWrite, Access::FragmentShaderRead).check("list");
        Barrier::buffer(buffer(), Access::HostWrite, Access::CopyRead).check("list");
        Barrier::texture(texture(), Access::None, Access::ColorAttachmentWrite).check("list");
        Barrier::subresource(
            texture(),
            Subresource::mip(1),
            Access::ComputeShaderWrite,
            Access::FragmentShaderSampledRead,
        )
        .check("list");
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

    /// A texture barrier with an access only buffers have panics.
    #[test]
    #[should_panic(
        expected = "command list \"frame\" has a barrier for a texture with HostRead, which \
                    doesn't apply to a texture"
    )]
    fn texture_barriers_with_buffer_accesses_panic() {
        Barrier::texture(texture(), Access::CopyWrite, Access::HostRead).check("frame");
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
}
