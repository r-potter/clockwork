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
