//! The shaders T1 tests create programs from (design 0002, "Test plan"),
//! compiled by `build.rs` from `shaders/`.
//!
//! Each is a [`TestProgram`] holding its SPIR-V as words, since
//! `include_bytes!` doesn't align the bytes for `u32`s, from which
//! [`TestProgram::desc`] describes the program.

use std::ffi::CStr;
use std::io::Cursor;

use crate::{ProgramDesc, ShaderStage};

/// A test program's SPIR-V and stages.
pub(crate) struct TestProgram {
    /// The program's label.
    label: &'static str,
    /// Its SPIR-V, as words.
    spirv: Vec<u32>,
    /// Its stages and their entry points.
    stages: &'static [(ShaderStage, &'static CStr)],
}

impl TestProgram {
    /// Reads `spirv`, the bytes `slangc` wrote, as words.
    ///
    /// # Panics
    ///
    /// Panics if `spirv` isn't SPIR-V.
    fn new(
        label: &'static str,
        spirv: &[u8],
        stages: &'static [(ShaderStage, &'static CStr)],
    ) -> Self {
        let spirv = ash::util::read_spv(&mut Cursor::new(spirv))
            .unwrap_or_else(|err| panic!("{label} isn't SPIR-V: {err}"));
        Self {
            label,
            spirv,
            stages,
        }
    }

    /// The program's description, for
    /// [`GpuResources::create_program`](crate::GpuResources::create_program).
    pub(crate) fn desc(&self) -> ProgramDesc<'_> {
        ProgramDesc {
            label: self.label,
            spirv: &self.spirv,
            stages: self.stages,
        }
    }
}

/// A compute program that writes `0xC10CC0DE` through the address in its
/// push data (`shaders/write_value.slang`).
pub(crate) fn write_value() -> TestProgram {
    TestProgram::new(
        "write value",
        include_bytes!(concat!(env!("OUT_DIR"), "/write_value.spv")),
        &[(ShaderStage::Compute, c"main")],
    )
}

/// A compute program that adds one, atomically, to the `u32` counter at the
/// address in its push data for each workgroup it runs
/// (`shaders/count_groups.slang`).
pub(crate) fn count_groups() -> TestProgram {
    TestProgram::new(
        "count groups",
        include_bytes!(concat!(env!("OUT_DIR"), "/count_groups.spv")),
        &[(ShaderStage::Compute, c"main")],
    )
}

/// A compute program that writes each texel's coordinates, `(x, y, x + y,
/// 1)`, into a 2D storage view, through its slot in the resource heap
/// (`shaders/fill_texels.slang`). Its push data is the view's
/// [`DescriptorIndex`](crate::DescriptorIndex), and it runs in 8×8
/// workgroups.
pub(crate) fn fill_texels() -> TestProgram {
    TestProgram::new(
        "fill texels",
        include_bytes!(concat!(env!("OUT_DIR"), "/fill_texels.spv")),
        &[(ShaderStage::Compute, c"main")],
    )
}

/// A compute program that samples a 2D sampled view at each texel's
/// centre, and writes the result into a 2D storage view of the same size
/// (`shaders/copy_texels.slang`). Its push data is the sampled view's,
/// the sampler's and the storage view's
/// [`DescriptorIndex`](crate::DescriptorIndex), in that order, and it runs
/// in 8×8 workgroups.
pub(crate) fn copy_texels() -> TestProgram {
    TestProgram::new(
        "copy texels",
        include_bytes!(concat!(env!("OUT_DIR"), "/copy_texels.spv")),
        &[(ShaderStage::Compute, c"main")],
    )
}

/// The triangle's vertex and fragment program (`shaders/triangle.slang`).
pub(crate) fn triangle() -> TestProgram {
    TestProgram::new(
        "triangle",
        include_bytes!(concat!(env!("OUT_DIR"), "/triangle.spv")),
        &[
            (ShaderStage::Vertex, c"vertexMain"),
            (ShaderStage::Fragment, c"fragmentMain"),
        ],
    )
}
