//! The frame graph (design 0003): a frame's passes and the resources they
//! use, compiled into an execution plan with every barrier, layout and load
//! and store operation derived, and executed through the RHI.
//!
//! It has three layers, following "pure cores, thin shells":
//! - the description, a [`GraphDesc`]: resources and passes, by name, built
//!   in Rust or loaded from a technique file, as plain data with no RHI
//!   handles
//! - the compiler, a pure function from a description to a plan
//! - the executor, which creates the plan's resources, and each frame records,
//!   translates and submits its passes
//!
//! So far the crate has the description and the compiler, [`compile()`],
//! for one queue, with [`check()`], which checks a plan against the rules
//! independently of the compiler, after every compile in debug builds. The
//! executor follows.

mod barriers;
mod check;
mod commands;
mod compile;
mod cull;
mod dependencies;
mod desc;
mod describe;
mod error;
mod expand;
mod grid;
mod inputs;
mod pass;
#[cfg(test)]
mod random_graphs;
mod render_passes;
mod resources;
mod schedule;
mod units;
mod validate;

pub use check::{Hazard, check};
pub use compile::{Plan, compile};
pub use desc::{
    BufferId, GraphDesc, ImportDesc, MipCount, OUTPUT, SamplerId, SamplerSettings, TextureId,
    TextureSize, TransientTexture,
};
pub use error::{CompileError, CompileErrors};
pub use inputs::{CompileInputs, OutputDesc, QueueLayout, Schedule};
pub use pass::{
    Attachment, AttachmentTarget, ComputePassBuilder, ComputeRecorder, Constant,
    FullscreenPassBuilder, MipDirection, PassId, RasterPassBuilder, RelativeMip, RepeatedUse,
    ResourceUse, TargetMip, TextureUse, TransferPassBuilder,
};
