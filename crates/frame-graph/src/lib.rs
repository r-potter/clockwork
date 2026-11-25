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
//! So far the crate has the description. The compiler and the executor
//! follow, slice by slice.

mod desc;
mod error;
mod inputs;
mod pass;
mod validate;

pub use desc::{
    BufferId, GraphDesc, ImportDesc, MipCount, OUTPUT, TextureId, TextureSize, TransientTexture,
};
pub use error::{CompileError, CompileErrors};
pub use inputs::{CompileInputs, OutputDesc, QueueLayout, Schedule};
pub use pass::{
    Attachment, AttachmentTarget, ComputePassBuilder, ComputeRecorder, FullscreenPassBuilder,
    PassId, RasterPassBuilder, RelativeMip, RepeatedUse, ResourceUse, TargetMip, TextureUse,
    TransferPassBuilder,
};
