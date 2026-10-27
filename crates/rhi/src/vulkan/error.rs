//! Turning Vulkan's results into [`RhiError`]s (design 0002, "Error
//! handling").
//!
//! The mapping is pure, so it's tested without a GPU. Only the results a
//! caller can act on will get a variant of their own, each added with the
//! code that returns it. Every other result becomes [`RhiError::Other`],
//! with the result's name in its message.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "RHI calls return RhiError from rhi-resources")
)]

use anyhow::anyhow;
use ash::vk;

use crate::RhiError;

/// The error for `result`, returned by a call that tried to `action` the
/// object labelled `label`, such as `error(result, "create buffer",
/// "triangle vertices")`.
pub(super) fn error(result: vk::Result, action: &str, label: &str) -> RhiError {
    RhiError::Other(anyhow!("failed to {action} \"{label}\": {result:?}"))
}

#[cfg(test)]
mod tests {
    //! T0 tests of the `VkResult` mapping (design 0002, R24).

    use super::*;

    /// Results with no variant are `Other`, with the action, the label and
    /// the result's name in its message.
    #[test]
    fn other_results_are_other_with_their_name() {
        for result in [
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_INITIALIZATION_FAILED,
            vk::Result::ERROR_TOO_MANY_OBJECTS,
            vk::Result::ERROR_UNKNOWN,
        ] {
            let RhiError::Other(err) = error(result, "create buffer", "vertices");
            assert_eq!(
                err.to_string(),
                format!("failed to create buffer \"vertices\": {result:?}"),
            );
        }
    }
}
