//! The errors RHI calls return (design 0002, "Error handling").
//!
//! [`RhiError`] will have a variant for each failure a caller can do
//! something about, and has [`RhiError::Other`] for everything else. Each
//! variant is added with the code that returns it, and only if a caller
//! handles it differently from `Other`, so the set stays small. Bugs in the
//! caller, such as a stale handle, panic instead.

use std::error::Error;
use std::fmt;

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
            Self::OutOfDeviceMemory { .. } => None,
        }
    }
}

impl From<anyhow::Error> for RhiError {
    /// Wraps the error as [`RhiError::Other`].
    fn from(err: anyhow::Error) -> Self {
        Self::Other(err)
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
