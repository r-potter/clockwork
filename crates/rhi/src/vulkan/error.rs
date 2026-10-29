//! Turning Vulkan's results into [`RhiError`]s (design 0002, "Error
//! handling").
//!
//! The mapping is pure, so it's tested without a GPU. Only the results a
//! caller can act on get a variant of their own, each added with the code
//! that returns it. Every other result becomes [`RhiError::Other`], with the
//! result's name in its message.

use anyhow::anyhow;
use ash::vk;

use crate::RhiError;

/// The error for `result`, returned by a call that tried to `action` the
/// object labelled `label`, such as `error(result, "create buffer",
/// "triangle vertices")`.
pub(super) fn error(result: vk::Result, action: &str, label: &str) -> RhiError {
    variant(result, label)
        .unwrap_or_else(|| RhiError::Other(anyhow!("failed to {action} \"{label}\": {result:?}")))
}

/// `err` as an [`RhiError`], for an error from code that returns
/// [`anyhow::Error`]s, such as the memory allocator: the variant for the
/// first [`vk::Result`] in its chain that has one, or [`RhiError::Other`]
/// with the whole chain.
pub(super) fn error_from_chain(err: anyhow::Error, label: &str) -> RhiError {
    let variant = err
        .chain()
        .find_map(|cause| cause.downcast_ref::<vk::Result>())
        .and_then(|&result| variant(result, label));
    variant.unwrap_or(RhiError::Other(err))
}

/// The variant for `result`, if it's one a caller can act on.
fn variant(result: vk::Result, label: &str) -> Option<RhiError> {
    match result {
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => Some(RhiError::OutOfDeviceMemory {
            label: label.to_owned(),
        }),
        vk::Result::ERROR_DEVICE_LOST => Some(RhiError::DeviceLost),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of the `VkResult` mapping (design 0002, R24).

    use anyhow::Context;

    use super::*;

    /// Running out of device memory names the object being created.
    #[test]
    fn out_of_device_memory_names_the_object() {
        let err = error(
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            "create buffer",
            "vertices",
        );
        assert!(matches!(
            err,
            RhiError::OutOfDeviceMemory { label } if label == "vertices"
        ));
    }

    /// A lost device is `DeviceLost`, whatever the call was doing, on its
    /// own or in a chain.
    #[test]
    fn device_lost_is_mapped() {
        assert!(matches!(
            error(vk::Result::ERROR_DEVICE_LOST, "submit to queue", "graphics"),
            RhiError::DeviceLost
        ));
        let err =
            anyhow::Error::from(vk::Result::ERROR_DEVICE_LOST).context("failed to map memory");
        assert!(matches!(
            error_from_chain(err, "vertices"),
            RhiError::DeviceLost
        ));
    }

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
            let RhiError::Other(err) = error(result, "create buffer", "vertices") else {
                panic!("{result:?} has a variant");
            };
            assert_eq!(
                err.to_string(),
                format!("failed to create buffer \"vertices\": {result:?}"),
            );
        }
    }

    /// A result inside an `anyhow` chain, as the memory allocator returns,
    /// is mapped as it would be on its own.
    #[test]
    fn results_in_a_chain_are_mapped() {
        let err = anyhow::Error::from(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
            .context("failed to allocate 256 bytes of memory type 3");
        assert!(matches!(
            error_from_chain(err, "vertices"),
            RhiError::OutOfDeviceMemory { label } if label == "vertices"
        ));
    }

    /// A chain with no result, or one with no variant, stays whole in
    /// `Other`.
    #[test]
    fn other_chains_are_kept_whole() {
        let result: anyhow::Result<()> =
            Err(vk::Result::ERROR_MEMORY_MAP_FAILED).context("failed to map memory");
        let RhiError::Other(err) = error_from_chain(result.unwrap_err(), "vertices") else {
            panic!("ERROR_MEMORY_MAP_FAILED has a variant");
        };
        assert_eq!(
            format!("{err:#}"),
            format!(
                "failed to map memory: {}",
                vk::Result::ERROR_MEMORY_MAP_FAILED
            ),
        );
    }
}
