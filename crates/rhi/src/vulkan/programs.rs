//! Programs in the Vulkan backend (design 0002, "Programs").
//!
//! A program is a set of shader objects (`VK_EXT_shader_object`): a vertex
//! and a fragment shader, linked, or a compute shader. Each is created to
//! reach the descriptor heaps directly, through `SPV_EXT_descriptor_heap`,
//! with push data rather than push constants, so no layout or mapping is
//! given (design 0002, "Binding model"). The [`VulkanDevice`]'s program
//! table holds each program's shader objects.

#[cfg(test)]
mod gpu_tests;

use std::slice;
use std::sync::{PoisonError, RwLockWriteGuard};

use ash::vk;

use super::VulkanDevice;
use super::destruction::Retired;
use super::error::error;
use crate::handle::HandleTable;
use crate::{Program, ProgramDesc, RhiResult, ShaderStage};

/// A program in the program table: its shader objects, each with its
/// stage, in the order they're bound.
#[derive(Debug)]
pub(super) struct ProgramEntry {
    /// The stage of each of `shaders`.
    #[expect(dead_code, reason = "translation binds programs in the next PR")]
    stages: Vec<vk::ShaderStageFlags>,
    /// The shader objects, created together, and linked if there are two.
    shaders: Vec<vk::ShaderEXT>,
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    /// The Vulkan stage `stage` runs in.
    fn from(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => Self::VERTEX,
            ShaderStage::Fragment => Self::FRAGMENT,
            ShaderStage::Compute => Self::COMPUTE,
        }
    }
}

/// The create infos for `desc`'s stages, from `code`, its SPIR-V as bytes:
/// each reaching the heaps directly, and, for a vertex and fragment pair,
/// linked, with the vertex stage followed by the fragment stage.
///
/// `desc` must be valid.
fn shader_create_infos<'a>(
    desc: &ProgramDesc<'a>,
    code: &'a [u8],
) -> Vec<vk::ShaderCreateInfoEXT<'a>> {
    // Linking a vertex and fragment pair lets the driver optimise them
    // together, as it would a pipeline. Linked shaders must be created in
    // one call.
    let link = if desc.stages.len() > 1 {
        vk::ShaderCreateFlagsEXT::LINK_STAGE
    } else {
        vk::ShaderCreateFlagsEXT::empty()
    };
    desc.stages
        .iter()
        .map(|&(stage, entry_point)| {
            let next_stage = match stage {
                ShaderStage::Vertex => vk::ShaderStageFlags::FRAGMENT,
                ShaderStage::Fragment | ShaderStage::Compute => vk::ShaderStageFlags::empty(),
            };
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::DESCRIPTOR_HEAP | link)
                .stage(vk::ShaderStageFlags::from(stage))
                .next_stage(next_stage)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(code)
                .name(entry_point)
        })
        .collect()
}

impl VulkanDevice {
    /// Creates the shader objects for `desc`, names them after it, and adds
    /// the program to the program table.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::OutOfDeviceMemory`](crate::RhiError::OutOfDeviceMemory)
    /// or [`RhiError::Other`](crate::RhiError::Other) if the shader objects
    /// can't be created.
    ///
    /// # Panics
    ///
    /// Panics if `desc` is invalid.
    pub(super) fn create_program_entry(&self, desc: &ProgramDesc<'_>) -> RhiResult<Program> {
        desc.validate();
        // SAFETY: Any initialised `u32`s are valid bytes, with no alignment
        // requirement, and the bytes live as long as `desc.spirv`.
        let code = unsafe {
            slice::from_raw_parts(desc.spirv.as_ptr().cast::<u8>(), size_of_val(desc.spirv))
        };
        let create_infos = shader_create_infos(desc, code);
        let shader_object = self.device.shader_object();
        // SAFETY: `VK_EXT_shader_object` and `VK_EXT_descriptor_heap` are
        // enabled with their features (device selection requires them), and
        // `create_infos` and the code they point to are valid for the call.
        let shaders = match unsafe { shader_object.create_shaders(&create_infos, None) } {
            Ok(shaders) => shaders,
            Err((shaders, result)) => {
                for shader in shaders {
                    // SAFETY: The shader was just created, and nothing uses
                    // it. A failed creation leaves a null handle, which
                    // destroying ignores.
                    unsafe { shader_object.destroy_shader(shader, None) };
                }
                return Err(error(result, "create the shaders of program", desc.label));
            }
        };
        for &shader in &shaders {
            self.device.set_object_name(shader, desc.label);
        }
        let entry = ProgramEntry {
            stages: create_infos
                .iter()
                .map(|create_info| create_info.stage)
                .collect(),
            shaders,
        };
        Ok(Program::from_handle(
            self.lock_programs_for_writing().insert(desc.label, entry),
        ))
    }

    /// Removes `program` from the table, and retires it, for its shader
    /// objects to be destroyed once nothing translated before now can use
    /// them.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale.
    pub(super) fn destroy_program_entry(&self, program: Program) {
        let (_, entry) = self.lock_programs_for_writing().remove(program.handle());
        self.retire(Retired::Program(entry));
    }

    /// Destroys the shader objects of a program taken out of the table.
    ///
    /// # Safety
    ///
    /// No submitted or translated work may still use them.
    pub(super) unsafe fn free_program(&self, entry: ProgramEntry) {
        for shader in entry.shaders {
            // SAFETY: The shader was created on this device, and the caller
            // guarantees that no work uses it.
            unsafe { self.device.shader_object().destroy_shader(shader, None) };
        }
    }

    /// Locks the program table for writing, recovering it if a panic
    /// poisoned it: no table operation panics halfway through, so it's
    /// consistent.
    fn lock_programs_for_writing(&self) -> RwLockWriteGuard<'_, HandleTable<ProgramEntry>> {
        self.programs
            .write()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the shader create infos.

    use std::ffi::CStr;

    use super::*;

    /// The create infos for `stages`, with their entry points.
    fn create_infos(
        stages: &'static [(ShaderStage, &'static CStr)],
    ) -> Vec<vk::ShaderCreateInfoEXT<'static>> {
        let desc = ProgramDesc {
            label: "program",
            spirv: &[0x0723_0203],
            stages,
        };
        shader_create_infos(&desc, &[3, 2, 35, 7])
    }

    /// A vertex and fragment pair is linked, the vertex stage followed by
    /// the fragment stage, each with its own entry point, in the order the
    /// description gives them.
    #[test]
    fn graphics_stages_are_linked() {
        let create_infos = create_infos(&[
            (ShaderStage::Fragment, c"fragmentMain"),
            (ShaderStage::Vertex, c"vertexMain"),
        ]);
        let [fragment, vertex] = create_infos.as_slice() else {
            panic!("expected two create infos");
        };
        for create_info in [fragment, vertex] {
            assert!(
                create_info
                    .flags
                    .contains(vk::ShaderCreateFlagsEXT::LINK_STAGE)
            );
        }
        assert_eq!(vertex.stage, vk::ShaderStageFlags::VERTEX);
        assert_eq!(vertex.next_stage, vk::ShaderStageFlags::FRAGMENT);
        // SAFETY: The name points at a `'static` C string.
        assert_eq!(unsafe { CStr::from_ptr(vertex.p_name) }, c"vertexMain");
        assert_eq!(fragment.stage, vk::ShaderStageFlags::FRAGMENT);
        assert_eq!(fragment.next_stage, vk::ShaderStageFlags::empty());
        // SAFETY: As above.
        assert_eq!(unsafe { CStr::from_ptr(fragment.p_name) }, c"fragmentMain");
    }

    /// A compute stage alone isn't linked, and has no next stage.
    #[test]
    fn a_compute_stage_is_alone() {
        let create_infos = create_infos(&[(ShaderStage::Compute, c"main")]);
        let [compute] = create_infos.as_slice() else {
            panic!("expected one create info");
        };
        assert!(!compute.flags.contains(vk::ShaderCreateFlagsEXT::LINK_STAGE));
        assert_eq!(compute.stage, vk::ShaderStageFlags::COMPUTE);
        assert_eq!(compute.next_stage, vk::ShaderStageFlags::empty());
    }

    /// Every stage reaches the heaps directly, from SPIR-V.
    #[test]
    fn every_stage_reaches_the_heaps() {
        let create_infos = create_infos(&[
            (ShaderStage::Vertex, c"vertexMain"),
            (ShaderStage::Fragment, c"fragmentMain"),
        ]);
        for create_info in &create_infos {
            assert!(
                create_info
                    .flags
                    .contains(vk::ShaderCreateFlagsEXT::DESCRIPTOR_HEAP)
            );
            assert_eq!(create_info.code_type, vk::ShaderCodeTypeEXT::SPIRV);
        }
    }

    /// Each stage maps to its own Vulkan stage.
    #[test]
    fn stages_map_to_vulkan_stages() {
        for (stage, vk_stage) in [
            (ShaderStage::Vertex, vk::ShaderStageFlags::VERTEX),
            (ShaderStage::Fragment, vk::ShaderStageFlags::FRAGMENT),
            (ShaderStage::Compute, vk::ShaderStageFlags::COMPUTE),
        ] {
            assert_eq!(vk::ShaderStageFlags::from(stage), vk_stage, "{stage:?}");
        }
    }
}
