//! The errors RHI calls return (design 0002, "Error handling").
//!
//! [`RhiError`] will have a variant for each failure a caller can do
//! something about, and has [`RhiError::Other`] for everything else. Each
//! variant is added with the code that returns it, and only if a caller
//! handles it differently from `Other`, so the set stays small. Bugs in the
//! caller, such as a stale handle, panic instead.
//!
//! A lost device is sticky: [`DeviceLoss`] records it, and every later call
//! that touches the GPU returns [`RhiError::DeviceLost`] without trying.

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::HeapKind;

/// A failure of an RHI call that the caller may handle.
///
/// `?` turns it into an [`anyhow::Error`] in application code, since it
/// implements [`Error`] and is `Send + Sync + 'static`.
#[derive(Debug)]
pub enum RhiError {
    /// Device memory ran out while creating the named object. Streaming can
    /// evict resources and try again, or load a lower mip level.
    OutOfDeviceMemory {
        /// The label of the object being created.
        label: String,
    },
    /// The heap has no free slot for the named texture view or sampler. The
    /// caller can free views or samplers, or report that the scene needs a
    /// bigger heap.
    DescriptorHeapFull {
        /// The heap that's full.
        heap: HeapKind,
        /// The label of the texture or sampler that needed a slot.
        label: String,
    },
    /// The device was lost, so nothing more can run on it. Every later call
    /// that touches the GPU returns this too. The caller can stop rendering
    /// and exit cleanly, since recovering is out of scope.
    DeviceLost,
    /// Anything with no specific recovery, with its context.
    Other(anyhow::Error),
}

/// The result of an RHI call.
pub type RhiResult<T> = Result<T, RhiError>;

impl fmt::Display for RhiError {
    /// Formats the error, naming the label involved. For
    /// [`Other`](Self::Other), this is the outermost context only, as for an
    /// [`anyhow::Error`], and the rest of the chain is its
    /// [`source`](Error::source).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfDeviceMemory { label } => {
                write!(f, "failed to create \"{label}\": out of device memory")
            }
            Self::DescriptorHeapFull { heap, label } => {
                write!(f, "the {heap} has no free slot for \"{label}\"")
            }
            Self::DeviceLost => f.write_str("the device was lost"),
            Self::Other(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl Error for RhiError {
    /// The cause of an [`Other`](Self::Other) error, from its context chain.
    /// The other variants have none.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Other(err) => err.source(),
            Self::OutOfDeviceMemory { .. } | Self::DescriptorHeapFull { .. } | Self::DeviceLost => {
                None
            }
        }
    }
}

impl From<anyhow::Error> for RhiError {
    /// Wraps the error as [`RhiError::Other`].
    fn from(err: anyhow::Error) -> Self {
        Self::Other(err)
    }
}

/// Whether a backend's device has been lost, which is sticky (design 0002,
/// "Error handling"): once any call returns [`RhiError::DeviceLost`], every
/// later call through [`guard`](Self::guard) returns it without running.
#[derive(Debug, Default)]
pub(crate) struct DeviceLoss {
    /// Set when a call first returns [`RhiError::DeviceLost`], and never
    /// cleared.
    lost: AtomicBool,
}

impl DeviceLoss {
    /// Runs `call`, a call that touches the GPU, unless the device is
    /// already lost, and records a loss it returns.
    ///
    /// # Errors
    ///
    /// Returns [`RhiError::DeviceLost`] if the device was already lost, and
    /// otherwise whatever `call` returns.
    pub(crate) fn guard<T>(&self, call: impl FnOnce() -> RhiResult<T>) -> RhiResult<T> {
        if self.lost.load(Ordering::Acquire) {
            return Err(RhiError::DeviceLost);
        }
        let result = call();
        if matches!(result, Err(RhiError::DeviceLost)) {
            self.lost.store(true, Ordering::Release);
        }
        result
    }
}

// Errors cross threads, and convert to `anyhow::Error` with `?` (design
// 0002, "Error handling").
const _: () = {
    /// Compiles only for types that can be an `anyhow::Error`.
    const fn assert_error<T: Error + Send + Sync + 'static>() {}
    assert_error::<RhiError>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of the errors' messages and their conversion to
    //! `anyhow::Error` (design 0002, R24).

    use anyhow::{Context, anyhow};

    use super::*;

    /// Running out of device memory names the object being created.
    #[test]
    fn out_of_device_memory_names_the_object() {
        let err = RhiError::OutOfDeviceMemory {
            label: "triangle vertices".into(),
        };
        assert_eq!(
            err.to_string(),
            "failed to create \"triangle vertices\": out of device memory",
        );
        assert!(err.source().is_none());
    }

    /// A full heap names the heap, and the object that needed a slot.
    #[test]
    fn descriptor_heap_full_names_the_heap_and_object() {
        let err = RhiError::DescriptorHeapFull {
            heap: HeapKind::Sampler,
            label: "shadow comparison".into(),
        };
        assert_eq!(
            err.to_string(),
            "the sampler heap has no free slot for \"shadow comparison\"",
        );
        assert!(err.source().is_none());
        let err = RhiError::DescriptorHeapFull {
            heap: HeapKind::Resource,
            label: "albedo".into(),
        };
        assert_eq!(
            err.to_string(),
            "the resource heap has no free slot for \"albedo\""
        );
    }

    /// A lost device has a message, and no cause.
    #[test]
    fn device_lost_says_so() {
        assert_eq!(RhiError::DeviceLost.to_string(), "the device was lost");
        assert!(RhiError::DeviceLost.source().is_none());
    }

    /// Calls run, and return what they return, until one loses the
    /// device. Errors other than a lost device aren't sticky.
    #[test]
    fn calls_run_until_the_device_is_lost() {
        let loss = DeviceLoss::default();
        assert!(matches!(loss.guard(|| Ok(1)), Ok(1)));
        assert!(matches!(
            loss.guard::<()>(|| Err(RhiError::Other(anyhow!("invalid SPIR-V")))),
            Err(RhiError::Other(_))
        ));
        assert!(matches!(loss.guard(|| Ok(2)), Ok(2)));
    }

    /// Once a call loses the device, every later call returns `DeviceLost`
    /// without running.
    #[test]
    fn a_lost_device_is_sticky() {
        let loss = DeviceLoss::default();
        assert!(matches!(
            loss.guard::<()>(|| Err(RhiError::DeviceLost)),
            Err(RhiError::DeviceLost)
        ));
        let mut ran = false;
        let result = loss.guard(|| {
            ran = true;
            Ok(())
        });
        assert!(matches!(result, Err(RhiError::DeviceLost)));
        assert!(!ran);
    }

    /// `Other` keeps its `anyhow` context chain, through `?` into an
    /// `anyhow::Error` and back out again.
    #[test]
    fn other_keeps_its_context_chain() {
        let err = RhiError::from(anyhow!("invalid SPIR-V").context("failed to create \"sky\""));
        assert_eq!(err.to_string(), "failed to create \"sky\"");
        let err = anyhow::Error::from(err).context("failed to load the sky");
        assert_eq!(
            format!("{err:#}"),
            "failed to load the sky: failed to create \"sky\": invalid SPIR-V",
        );
    }

    /// An `RhiError` can be found again in an `anyhow::Error`, so
    /// application code can react to its variant after `?`.
    #[test]
    fn errors_survive_conversion_to_anyhow() {
        let result: anyhow::Result<()> =
            Err(RhiError::Other(anyhow!("invalid SPIR-V"))).context("failed to render");
        let err = result.unwrap_err();
        assert!(matches!(
            err.downcast_ref::<RhiError>(),
            Some(RhiError::Other(_))
        ));
    }
}
