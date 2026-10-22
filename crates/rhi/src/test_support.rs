//! Helpers shared by the RHI's tests: reading the test environment. Compiled
//! only for tests.
//!
//! This is the only place tests read the environment (design 0001,
//! "Configuration"): the engine itself never does, so its behaviour depends
//! only on its options. `cargo test` can't pass options to tests, so T1 tests
//! are configured by these variables:
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | [`CLOCKWORK_DEVICE`] | Only consider devices whose names contain this, ignoring case | Every device |
//! | [`CLOCKWORK_VALIDATION`] | The validation features, as for [`ValidationFeatures`]'s parser, except that `off` is rejected | `sync` |
//! | [`CLOCKWORK_REQUIRED_CAPABILITIES`] | `all`, or a comma-separated list of capabilities whose absence fails tests rather than skipping them | None |
//!
//! Each variable is read by a pure function of its value, tested here, and a
//! wrapper that reads the environment.

use std::io::{self, Write};

use anyhow::{Result, bail};
use clockwork_diagnostics::capture::TestSession;

use crate::{BackendOptions, Validation, ValidationFeatures};

/// Restricts T1 tests to devices whose names contain its value.
pub(crate) const CLOCKWORK_DEVICE: &str = "CLOCKWORK_DEVICE";

/// The validation features T1 tests require.
pub(crate) const CLOCKWORK_VALIDATION: &str = "CLOCKWORK_VALIDATION";

/// The capabilities whose absence fails T1 tests, rather than skipping them.
pub(crate) const CLOCKWORK_REQUIRED_CAPABILITIES: &str = "CLOCKWORK_REQUIRED_CAPABILITIES";

/// The options for a backend in `session`, from the test environment: the
/// device filter from [`CLOCKWORK_DEVICE`], and validation always
/// [`Validation::Required`], with the features from [`CLOCKWORK_VALIDATION`].
///
/// There's deliberately no way to run T1 tests without validation: a pass
/// would prove less than it claims.
///
/// # Panics
///
/// Panics if [`CLOCKWORK_VALIDATION`] is invalid.
pub(crate) fn backend_options(session: &TestSession) -> BackendOptions {
    let features = validation_features(env(CLOCKWORK_VALIDATION).as_deref())
        .unwrap_or_else(|err| panic!("{CLOCKWORK_VALIDATION}: {err:#}"));
    BackendOptions {
        session: session.id(),
        device_filter: device_filter(env(CLOCKWORK_DEVICE).as_deref()),
        validation: Validation::Required(features),
    }
}

/// The capabilities required by [`CLOCKWORK_REQUIRED_CAPABILITIES`], each of
/// which must be one of `known`.
///
/// # Panics
///
/// Panics if the variable is invalid, or names an unknown capability.
pub(crate) fn required_capabilities(known: &[&str]) -> RequiredCapabilities {
    RequiredCapabilities::parse(env(CLOCKWORK_REQUIRED_CAPABILITIES).as_deref(), known)
        .unwrap_or_else(|err| panic!("{CLOCKWORK_REQUIRED_CAPABILITIES}: {err:#}"))
}

/// The value of the environment variable `name`, if it's set and valid
/// Unicode.
fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// The validation features T1 tests require, given [`CLOCKWORK_VALIDATION`]'s
/// value: `sync` if it's unset or empty.
///
/// # Errors
///
/// Returns an error if the value doesn't parse, or is `off`: T1 tests always
/// require validation.
fn validation_features(value: Option<&str>) -> Result<ValidationFeatures> {
    let Some(value) = value.filter(|value| !value.trim().is_empty()) else {
        return Ok(ValidationFeatures::SYNC);
    };
    let features: ValidationFeatures = value.parse()?;
    if features.is_off() {
        bail!("GPU tests always require validation, so \"off\" isn't allowed");
    }
    Ok(features)
}

/// The device filter, given [`CLOCKWORK_DEVICE`]'s value: `None`, meaning
/// every device, if it's unset or empty.
fn device_filter(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Which capabilities T1 tests require, from
/// [`CLOCKWORK_REQUIRED_CAPABILITIES`]. A test that needs a capability the
/// device lacks fails if it's required, and is skipped otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RequiredCapabilities {
    /// Every capability, as on the reference device.
    All,
    /// Only these, by name. Empty when the variable is unset.
    Listed(Vec<String>),
}

impl RequiredCapabilities {
    /// Parses [`CLOCKWORK_REQUIRED_CAPABILITIES`]'s value: `all`, or a
    /// comma-separated list of names from `known`. Unset or empty requires
    /// none. Spaces around names are ignored.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty list item or an unknown name, listing
    /// the known ones, so that a typo can't turn a failure into a skip.
    fn parse(value: Option<&str>, known: &[&str]) -> Result<Self> {
        let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(Self::Listed(Vec::new()));
        };
        if value == "all" {
            return Ok(Self::All);
        }
        let mut names = Vec::new();
        for name in value.split(',').map(str::trim) {
            if !known.contains(&name) {
                bail!(
                    "invalid capabilities \"{value}\": expected all, or a comma-separated list of \
                     {}",
                    known.join(", "),
                );
            }
            names.push(name.to_owned());
        }
        Ok(Self::Listed(names))
    }

    /// Whether the capability called `name` is required.
    fn includes(&self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Listed(names) => names.iter().any(|listed| listed == name),
        }
    }
}

/// What a T1 test does when the device lacks a capability it needs (R9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MissingCapability {
    /// Skip the test, with a message.
    Skip,
    /// Fail the test: the capability is required, so its absence is a bug or
    /// a misconfigured machine, not a reason to skip.
    Fail,
}

/// Decides what a test does when the device lacks the capability called
/// `name`. Pure.
pub(crate) fn on_missing_capability(
    name: &str,
    required: &RequiredCapabilities,
) -> MissingCapability {
    if required.includes(name) {
        MissingCapability::Fail
    } else {
        MissingCapability::Skip
    }
}

/// Handles the device `device` lacking the capability called `name`: prints
/// a skip message, or panics if the capability is required.
///
/// The skip message is written to the process's stderr directly, rather than
/// with `eprintln!`, which the test harness captures and shows only if the
/// test fails, so that skips are visible in passing runs.
///
/// # Panics
///
/// Panics if `required` includes the capability.
pub(crate) fn missing_capability(name: &str, device: &str, required: &RequiredCapabilities) {
    let test = std::thread::current()
        .name()
        .unwrap_or("<unnamed test>")
        .to_owned();
    match on_missing_capability(name, required) {
        MissingCapability::Fail => panic!(
            "{device} lacks the {name} capability, which {CLOCKWORK_REQUIRED_CAPABILITIES} \
             requires"
        ),
        MissingCapability::Skip => {
            // Nothing can be done if stderr is unwritable.
            let _ = writeln!(
                io::stderr(),
                "=== clockwork: skipped {test}: {device} lacks the {name} capability ==="
            );
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of reading the test environment, and of the skip-or-fail
    //! decision (R9).

    use super::*;

    /// Capability names for the tests.
    const KNOWN: &[&str] = &["descriptor-heap", "present-timing"];

    /// Unset or empty, `CLOCKWORK_VALIDATION` means synchronization
    /// validation, the default (R6).
    #[test]
    fn validation_defaults_to_sync() {
        for value in [None, Some(""), Some("  ")] {
            assert_eq!(
                validation_features(value).unwrap(),
                ValidationFeatures::SYNC
            );
        }
    }

    /// `CLOCKWORK_VALIDATION` takes the same features as `--validation`.
    #[test]
    fn validation_features_parse_like_the_option() {
        assert_eq!(
            validation_features(Some("core,gpu")).unwrap(),
            "core,gpu".parse().unwrap(),
        );
        assert!(validation_features(Some("fast")).is_err());
    }

    /// T1 tests can't be run without validation.
    #[test]
    fn validation_off_is_rejected() {
        let err = validation_features(Some("off")).unwrap_err();
        assert!(
            err.to_string().contains("always require validation"),
            "{err}"
        );
    }

    /// An unset or empty `CLOCKWORK_DEVICE` considers every device.
    #[test]
    fn device_filter_defaults_to_every_device() {
        assert_eq!(device_filter(None), None);
        assert_eq!(device_filter(Some(" ")), None);
        assert_eq!(device_filter(Some("4090")), Some("4090".to_owned()));
    }

    /// Unset or empty, no capability is required, so missing ones skip.
    #[test]
    fn no_capabilities_are_required_by_default() {
        for value in [None, Some("")] {
            let required = RequiredCapabilities::parse(value, KNOWN).unwrap();
            assert_eq!(
                on_missing_capability("descriptor-heap", &required),
                MissingCapability::Skip,
            );
        }
    }

    /// `all` requires every capability, so a missing one fails.
    #[test]
    fn all_requires_every_capability() {
        let required = RequiredCapabilities::parse(Some("all"), KNOWN).unwrap();
        for name in KNOWN {
            assert_eq!(
                on_missing_capability(name, &required),
                MissingCapability::Fail
            );
        }
    }

    /// A list requires the capabilities it names, and only those.
    #[test]
    fn lists_require_only_their_capabilities() {
        let required = RequiredCapabilities::parse(Some(" descriptor-heap "), KNOWN).unwrap();
        assert_eq!(
            on_missing_capability("descriptor-heap", &required),
            MissingCapability::Fail,
        );
        assert_eq!(
            on_missing_capability("present-timing", &required),
            MissingCapability::Skip,
        );
    }

    /// Unknown names and empty items are errors, so that a typo can't turn a
    /// required capability's failure into a skip.
    #[test]
    fn unknown_capabilities_are_errors() {
        for value in [
            "descriptor_heap",
            "descriptor-heap,",
            "ALL",
            "all,present-timing",
        ] {
            assert!(
                RequiredCapabilities::parse(Some(value), KNOWN).is_err(),
                "{value:?} parsed",
            );
        }
    }

    /// A missing capability that's required fails the test, naming it.
    #[test]
    #[should_panic(expected = "a test device lacks the present-timing capability")]
    fn missing_required_capabilities_fail() {
        missing_capability(
            "present-timing",
            "a test device",
            &RequiredCapabilities::All,
        );
    }
}
