//! The access table: what each [`Access`] and [`Layout`] means in Vulkan's
//! synchronization2 vocabulary (design 0002, "Synchronization").
//!
//! This is the one place the backend spells out pipeline stages, access
//! flags and image layouts, following Tobias Hector's
//! `simple_vulkan_synchronization`. It's pure data, so it's reviewed and
//! tested as data. A barrier combines the table's entries for the access
//! set on each side of it with [`scope`], and takes a texture's layouts from
//! the layout recording chose for each set, so a layout can't disagree with
//! its accesses.

use ash::vk;

use crate::barrier::Layout;
use crate::{Access, AccessSet};

/// What an [`Access`], or an [`AccessSet`] on one side of a barrier, means to
/// Vulkan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AccessInfo {
    /// The pipeline stages that perform the accesses.
    pub stages: vk::PipelineStageFlags2,
    /// The memory accesses those stages make.
    pub access: vk::AccessFlags2,
}

/// Which side of a barrier an [`AccessSet`] is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Side {
    /// The accesses before the barrier, which it waits for.
    Before,
    /// The accesses after the barrier, which wait for it.
    After,
}

/// The stages and access flags for `access` alone.
///
/// [`Access::None`] has no stages or flags, so a barrier from it waits for
/// nothing. [`Access::Present`] has none either, since the swapchain's
/// semaphores order presenting.
pub(super) fn access_info(access: Access) -> AccessInfo {
    use vk::AccessFlags2 as A;
    use vk::PipelineStageFlags2 as S;

    let (stages, access) = match access {
        Access::None | Access::Present => (S::NONE, A::NONE),
        Access::IndirectRead => (S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ),
        Access::IndexRead => (S::INDEX_INPUT, A::INDEX_READ),
        Access::VertexShaderRead => (S::VERTEX_SHADER, A::SHADER_STORAGE_READ),
        Access::FragmentShaderRead => (S::FRAGMENT_SHADER, A::SHADER_STORAGE_READ),
        Access::FragmentShaderSampledRead => (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ),
        Access::ComputeShaderRead => (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
        Access::ComputeShaderSampledRead => (S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ),
        Access::ComputeShaderWrite => (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
        Access::ColorAttachmentRead => (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_READ),
        Access::ColorAttachmentWrite => (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE),
        Access::DepthAttachmentRead => (
            S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
            A::DEPTH_STENCIL_ATTACHMENT_READ,
        ),
        Access::DepthAttachmentWrite => (
            S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
            A::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        Access::CopyRead => (S::COPY, A::TRANSFER_READ),
        // Fills are clear commands in synchronization2's stages.
        Access::CopyWrite => (S::COPY | S::CLEAR, A::TRANSFER_WRITE),
        Access::HostWrite => (S::HOST, A::HOST_WRITE),
        Access::HostRead => (S::HOST, A::HOST_READ),
    };
    AccessInfo { stages, access }
}

/// The stages and access flags for `set` on `side` of a barrier.
///
/// The stages are every access's, on both sides, so a barrier waits for
/// reads before it as well as writes. After a barrier, the flags are every
/// access's, so earlier writes are made visible to each. Before it, they're
/// only the writes', since only writes need making available, and Vulkan
/// ignores read flags there.
pub(super) fn scope(set: AccessSet, side: Side) -> AccessInfo {
    let mut scope = AccessInfo {
        stages: vk::PipelineStageFlags2::NONE,
        access: vk::AccessFlags2::NONE,
    };
    for access in set.iter() {
        let info = access_info(access);
        scope.stages |= info.stages;
        if side == Side::After || access.is_write() {
            scope.access |= info.access;
        }
    }
    scope
}

impl From<Layout> for vk::ImageLayout {
    /// The Vulkan layout for `layout`.
    ///
    /// Attachments and read-only accesses use synchronization2's
    /// `ATTACHMENT_OPTIMAL` and `READ_ONLY_OPTIMAL`, which take their meaning
    /// from the texture's format, rather than a layout for each kind of
    /// attachment or read. `UNDEFINED` discards a texture's contents.
    fn from(layout: Layout) -> Self {
        match layout {
            Layout::Undefined => Self::UNDEFINED,
            Layout::General => Self::GENERAL,
            Layout::ReadOnly => Self::READ_ONLY_OPTIMAL,
            Layout::Attachment => Self::ATTACHMENT_OPTIMAL,
            Layout::CopySource => Self::TRANSFER_SRC_OPTIMAL,
            Layout::CopyDestination => Self::TRANSFER_DST_OPTIMAL,
            Layout::Present => Self::PRESENT_SRC_KHR,
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the access table (design 0002, R8).

    use super::*;

    /// Every access flag that writes memory.
    const WRITE_FLAGS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_WRITE.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
            | vk::AccessFlags2::HOST_WRITE.as_raw()
            | vk::AccessFlags2::MEMORY_WRITE.as_raw(),
    );

    /// Read accesses map to no write flags, and write accesses only to write
    /// flags, so each access is a single read or write.
    #[test]
    fn writes_map_to_only_write_flags() {
        for access in Access::ALL {
            let flags = access_info(access).access;
            if access.is_write() {
                assert!(WRITE_FLAGS.contains(flags), "{access:?}");
            } else {
                assert!(!flags.intersects(WRITE_FLAGS), "{access:?}");
            }
        }
    }

    /// Every access but `None` and `Present` happens in some stage, with
    /// some memory access, so it's ordered by a barrier.
    #[test]
    fn accesses_have_stages_and_flags() {
        for access in Access::ALL {
            let info = access_info(access);
            let has_stages = !info.stages.is_empty() && !info.access.is_empty();
            let unordered = matches!(access, Access::None | Access::Present);
            assert_eq!(has_stages, !unordered, "{access:?}");
        }
    }

    /// Each layout maps to the Vulkan layout made for it, with
    /// synchronization2's layouts for attachments and read-only access.
    #[test]
    fn layouts_map_to_vulkan_layouts() {
        use vk::ImageLayout as L;
        for (layout, expected) in [
            (Layout::Undefined, L::UNDEFINED),
            (Layout::General, L::GENERAL),
            (Layout::ReadOnly, L::READ_ONLY_OPTIMAL),
            (Layout::Attachment, L::ATTACHMENT_OPTIMAL),
            (Layout::CopySource, L::TRANSFER_SRC_OPTIMAL),
            (Layout::CopyDestination, L::TRANSFER_DST_OPTIMAL),
            (Layout::Present, L::PRESENT_SRC_KHR),
        ] {
            assert_eq!(L::from(layout), expected, "{layout:?}");
        }
    }

    /// A set's stages are every access's on both sides, and its flags are
    /// every access's after a barrier, but only the writes' before it.
    #[test]
    fn sets_combine_stages_and_flags() {
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let read_write = Access::ComputeShaderRead | Access::ComputeShaderWrite;
        assert_eq!(
            scope(read_write, Side::Before),
            AccessInfo {
                stages: S::COMPUTE_SHADER,
                access: A::SHADER_STORAGE_WRITE,
            },
        );
        assert_eq!(
            scope(read_write, Side::After),
            AccessInfo {
                stages: S::COMPUTE_SHADER,
                access: A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            },
        );
        let reads = Access::VertexShaderRead | Access::FragmentShaderSampledRead;
        assert_eq!(
            scope(reads, Side::Before),
            AccessInfo {
                stages: S::VERTEX_SHADER | S::FRAGMENT_SHADER,
                access: A::NONE,
            },
        );
        assert_eq!(
            scope(reads, Side::After),
            AccessInfo {
                stages: S::VERTEX_SHADER | S::FRAGMENT_SHADER,
                access: A::SHADER_STORAGE_READ | A::SHADER_SAMPLED_READ,
            },
        );
    }

    /// A single access's scope after a barrier is its own entry.
    #[test]
    fn a_single_access_is_its_own_scope() {
        for access in Access::ALL {
            assert_eq!(
                scope(access.into(), Side::After),
                access_info(access),
                "{access:?}"
            );
        }
    }
}
