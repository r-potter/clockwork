//! Programs: their handles, and how they're created (design 0002,
//! "Handles" and "Programs").
//!
//! A program is a set of linked shader stages from SPIR-V: a vertex and
//! fragment pair, or a single compute shader. It holds no other state, so
//! the same program draws to any attachment formats, with every piece of
//! rasterization state set when drawing.

use std::ffi::CStr;

use crate::handle::Handle;

/// A program. Refers to an entry in the backend's program table, and is
/// stale once destroyed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Program(Handle);

impl Program {
    /// The program for `handle`, issued by a backend's program table.
    pub(crate) fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    /// The handle into the backend's program table.
    pub(crate) fn handle(self) -> Handle {
        self.0
    }
}

/// A stage of a program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Runs once per vertex of a draw.
    Vertex,
    /// Runs once per fragment a draw covers.
    Fragment,
    /// Runs once per invocation of a dispatch.
    Compute,
}

/// How a program is created: linked shader stages from SPIR-V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramDesc<'a> {
    /// The program's name, in debug tools and messages.
    pub label: &'a str,
    /// SPIR-V words, holding every stage's entry point.
    pub spirv: &'a [u32],
    /// The stages and their entry points: a vertex and a fragment stage, in
    /// either order, or a compute stage alone.
    pub stages: &'a [(ShaderStage, &'a CStr)],
}

impl ProgramDesc<'_> {
    /// Checks that the description is valid, as
    /// [`GpuResources::create_program`] requires.
    ///
    /// [`GpuResources::create_program`]: crate::GpuResources::create_program
    ///
    /// # Panics
    ///
    /// Panics, naming the program, if `spirv` is empty, or if `stages`
    /// isn't a vertex and a fragment stage or a compute stage alone.
    pub(crate) fn validate(&self) {
        let label = self.label;
        assert!(!self.spirv.is_empty(), "program \"{label}\" has no SPIR-V");
        let stages: Vec<_> = self.stages.iter().map(|&(stage, _)| stage).collect();
        assert!(
            matches!(
                stages.as_slice(),
                [ShaderStage::Vertex, ShaderStage::Fragment]
                    | [ShaderStage::Fragment, ShaderStage::Vertex]
                    | [ShaderStage::Compute]
            ),
            "program \"{label}\" has stages {stages:?}, but a program is a vertex and a \
             fragment stage, or a compute stage alone",
        );
    }
}

// Handles and descriptions are plain data, shared between threads (design
// 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Program>();
    assert_send_sync::<ProgramDesc<'_>>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of program descriptions.

    use super::*;

    /// A description of `stages`, each with the entry point `main`, from
    /// one word of SPIR-V.
    fn desc(stages: &[ShaderStage]) -> ProgramDesc<'static> {
        let stages: Vec<_> = stages.iter().map(|&stage| (stage, c"main")).collect();
        ProgramDesc {
            label: "program",
            spirv: &[0x0723_0203],
            stages: Vec::leak(stages),
        }
    }

    /// A vertex and fragment pair, in either order, and a compute stage
    /// alone, are programs.
    #[test]
    fn graphics_and_compute_programs_are_valid() {
        use ShaderStage::{Compute, Fragment, Vertex};
        for stages in [&[Vertex, Fragment][..], &[Fragment, Vertex], &[Compute]] {
            desc(stages).validate();
        }
    }

    /// Any other set of stages panics, naming the program and its stages.
    #[test]
    fn other_stages_panic() {
        use ShaderStage::{Compute, Fragment, Vertex};
        for stages in [
            &[][..],
            &[Vertex],
            &[Fragment],
            &[Vertex, Vertex],
            &[Vertex, Fragment, Compute],
            &[Compute, Compute],
            &[Vertex, Compute],
        ] {
            let panic = std::panic::catch_unwind(|| desc(stages).validate())
                .expect_err(&format!("{stages:?} should panic"));
            let message = panic.downcast_ref::<String>().unwrap();
            assert!(
                message.starts_with("program \"program\" has stages"),
                "{message}"
            );
        }
    }

    /// The panic lists the stages in the order the description gives them.
    #[test]
    #[should_panic(expected = "has stages [Fragment, Compute, Vertex]")]
    fn invalid_stages_are_listed_as_given() {
        use ShaderStage::{Compute, Fragment, Vertex};
        desc(&[Fragment, Compute, Vertex]).validate();
    }

    /// A program needs SPIR-V.
    #[test]
    #[should_panic(expected = "program \"program\" has no SPIR-V")]
    fn empty_spirv_panics() {
        ProgramDesc {
            spirv: &[],
            ..desc(&[ShaderStage::Compute])
        }
        .validate();
    }
}
