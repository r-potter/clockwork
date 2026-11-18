//! Streaming: filling buffers and textures from any thread, on the transfer
//! queue (design 0002, "Streaming", R17).
//!
//! It's ordinary code over [`GpuResources`](crate::GpuResources) and
//! [`Queue`](crate::Queue), outside the `vulkan` module, so it's the same
//! for every backend. So far it has the staging buffers an uploader writes
//! its bytes to, and the recording of the copies and barriers that move
//! them into place, which are tested with a fake of
//! [`GpuResources`](crate::GpuResources).

#[cfg(test)]
mod fake;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the uploader records its uploads in a later PR")
)]
mod recording;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the uploader flushes its staging in a later PR")
)]
mod staging;
