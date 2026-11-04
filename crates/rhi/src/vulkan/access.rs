//! The access table: what each [`Access`] means in Vulkan's synchronization2
//! vocabulary (design 0002, "Synchronization").
//!
//! This is the one place the backend spells out pipeline stages, access
//! flags and image layouts, following Tobias Hector's
//! `simple_vulkan_synchronization`. It's pure data, so it's reviewed and
//! tested as data. A barrier takes its stages and flags from the table for
//! the accesses on either side of it, and a texture's layouts too, so a
//! layout can't disagree with its access.

use ash::vk;

use crate::Access;

/// What an [`Access`] means to Vulkan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AccessInfo {
    /// The pipeline stages that perform the access.
    pub stages: vk::PipelineStageFlags2,
    /// The memory accesses those stages make.
    pub access: vk::AccessFlags2,
    /// The layout a texture must be in for the access, or `None` for an
    /// access only buffers have.
    pub layout: Option<vk::ImageLayout>,
}

/// The stages, access flags and layout for `access`.
///
/// [`Access::None`] has no stages or flags, so a barrier from it waits for
/// nothing, and its layout is `UNDEFINED`, so a barrier from it discards a
/// texture's contents. [`Access::Present`] has none either, since the
/// swapchain's semaphores order presenting.
pub(super) fn access_info(access: Access) -> AccessInfo {
    use vk::AccessFlags2 as A;
    use vk::ImageLayout as L;
    use vk::PipelineStageFlags2 as S;

    let (stages, access, layout) = match access {
        Access::None => (S::NONE, A::NONE, Some(L::UNDEFINED)),
        Access::IndirectRead => (S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ, None),
        Access::IndexRead => (S::INDEX_INPUT, A::INDEX_READ, None),
        Access::VertexShaderRead => (S::VERTEX_SHADER, A::SHADER_STORAGE_READ, Some(L::GENERAL)),
        Access::FragmentShaderRead => {
            (S::FRAGMENT_SHADER, A::SHADER_STORAGE_READ, Some(L::GENERAL))
        }
        Access::FragmentShaderSampledRead => (
            S::FRAGMENT_SHADER,
            A::SHADER_SAMPLED_READ,
            Some(L::SHADER_READ_ONLY_OPTIMAL),
        ),
        Access::ComputeShaderRead => (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ, Some(L::GENERAL)),
        Access::ComputeShaderSampledRead => (
            S::COMPUTE_SHADER,
            A::SHADER_SAMPLED_READ,
            Some(L::SHADER_READ_ONLY_OPTIMAL),
        ),
        Access::ComputeShaderWrite => {
            (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, Some(L::GENERAL))
        }
        Access::ColorAttachmentWrite => (
            S::COLOR_ATTACHMENT_OUTPUT,
            A::COLOR_ATTACHMENT_WRITE,
            Some(L::COLOR_ATTACHMENT_OPTIMAL),
        ),
        Access::DepthAttachmentRead => (
            S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
            A::DEPTH_STENCIL_ATTACHMENT_READ,
            Some(L::DEPTH_STENCIL_READ_ONLY_OPTIMAL),
        ),
        // Depth testing reads the attachment as well as writing it.
        Access::DepthAttachmentWrite => (
            S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
            A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
            Some(L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ),
        Access::CopyRead => (S::COPY, A::TRANSFER_READ, Some(L::TRANSFER_SRC_OPTIMAL)),
        // Fills are clear commands in synchronization2's stages.
        Access::CopyWrite => (
            S::COPY | S::CLEAR,
            A::TRANSFER_WRITE,
            Some(L::TRANSFER_DST_OPTIMAL),
        ),
        Access::HostWrite => (S::HOST, A::HOST_WRITE, None),
        Access::HostRead => (S::HOST, A::HOST_READ, None),
        Access::Present => (S::NONE, A::NONE, Some(L::PRESENT_SRC_KHR)),
    };
    AccessInfo {
        stages,
        access,
        layout,
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

    /// Read accesses map to no write flags, and write accesses to at least
    /// one, so a barrier after a read never makes memory available that
    /// wasn't written, and one after a write always does.
    #[test]
    fn only_writes_map_to_write_flags() {
        for access in Access::ALL {
            let writes = access_info(access).access.intersects(WRITE_FLAGS);
            assert_eq!(writes, access.is_write(), "{access:?}");
        }
    }

    /// Exactly the accesses textures can have give a layout, so a texture
    /// barrier always has one, and a buffer-only access never claims one.
    #[test]
    fn texture_accesses_have_layouts() {
        for access in Access::ALL {
            assert_eq!(
                access_info(access).layout.is_some(),
                access.applies_to_textures(),
                "{access:?}"
            );
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

    /// Attachments, sampling, copies and presenting use the layouts made
    /// for them, and storage access uses `GENERAL`.
    #[test]
    fn layouts_match_their_accesses() {
        use vk::ImageLayout as L;
        for (access, layout) in [
            (Access::None, L::UNDEFINED),
            (Access::ColorAttachmentWrite, L::COLOR_ATTACHMENT_OPTIMAL),
            (
                Access::DepthAttachmentRead,
                L::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ),
            (
                Access::DepthAttachmentWrite,
                L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ),
            (
                Access::FragmentShaderSampledRead,
                L::SHADER_READ_ONLY_OPTIMAL,
            ),
            (
                Access::ComputeShaderSampledRead,
                L::SHADER_READ_ONLY_OPTIMAL,
            ),
            (Access::ComputeShaderWrite, L::GENERAL),
            (Access::CopyRead, L::TRANSFER_SRC_OPTIMAL),
            (Access::CopyWrite, L::TRANSFER_DST_OPTIMAL),
            (Access::Present, L::PRESENT_SRC_KHR),
        ] {
            assert_eq!(access_info(access).layout, Some(layout), "{access:?}");
        }
    }
}
