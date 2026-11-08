//! T1 tests of programs on a real device (design 0002, "Programs", R3).

use clockwork_diagnostics::capture::TestSession;

use crate::test_shaders;
use crate::vulkan::test_support::VulkanTestBackend;
use crate::{GpuResources, RenderHardwareInterface};

/// A compute program and a linked vertex and fragment program are created
/// from SPIR-V that reaches the heaps, and destroyed, with no validation
/// error. Destruction tests check that programs are retired.
#[test]
fn compute_and_graphics_programs_are_created() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources = backend.resources();
    let programs = [test_shaders::write_value(), test_shaders::triangle()]
        .map(|program| resources.create_program(&program.desc()).unwrap());
    assert_ne!(programs[0], programs[1]);
    for program in programs {
        resources.destroy_program(program);
    }
    resources.collect_garbage();
}

/// A destroyed program's handle is stale, and using it panics with its
/// label.
#[test]
#[should_panic(expected = "stale program handle: \"write value\" was destroyed")]
fn destroyed_programs_are_stale() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    let resources: &dyn GpuResources = &**backend.resources();
    let program = resources
        .create_program(&test_shaders::write_value().desc())
        .unwrap();
    resources.destroy_program(program);
    resources.destroy_program(program);
}
