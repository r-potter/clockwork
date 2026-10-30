//! Fixtures for the Vulkan backend's T1 tests: [`VulkanTestBackend`], the
//! test-only [`Capability`] list, and [`require_capabilities!`].
//!
//! [`Capability`] names the optional parts of the RHI's [`Capabilities`],
//! [`PresentCapabilities`] and queues, so that tests can require them by name in
//! `CLOCKWORK_REQUIRED_CAPABILITIES`. Whether a device has one is read from
//! the object that reports it, as the RHI's users read it (design 0002,
//! "Capabilities").

use clockwork_diagnostics::capture::TestSession;

use super::VulkanRenderHardwareInterface;
use crate::test_support::{self, RequiredCapabilities};
use crate::{Capabilities, PresentCapabilities, QueueKind, QueueKindSet, RenderHardwareInterface};

/// An optional capability some T1 tests need, which some devices lack.
///
/// A test that needs one asks [`VulkanTestBackend::has`], usually through
/// [`require_capabilities!`], and is skipped on devices without it, unless
/// `CLOCKWORK_REQUIRED_CAPABILITIES` requires it (design 0001, R9).
///
/// The capabilities device selection requires, such as descriptor heaps,
/// aren't listed: every selected device has them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Capability {
    /// [`PresentCapabilities::present_timing`].
    PresentTiming,
    /// [`PresentCapabilities::swapchain_maintenance`].
    SwapchainMaintenance,
    /// A compute queue of its own, whose [`Queue::kind`](crate::Queue::kind)
    /// is [`QueueKind::Compute`].
    DedicatedComputeQueue,
    /// A transfer queue of its own, whose [`Queue::kind`](crate::Queue::kind)
    /// is [`QueueKind::Transfer`].
    DedicatedTransferQueue,
    /// [`Capabilities::bc_compression`].
    BcCompression,
}

impl Capability {
    /// Every capability.
    pub(crate) const ALL: [Self; 5] = [
        Self::PresentTiming,
        Self::SwapchainMaintenance,
        Self::DedicatedComputeQueue,
        Self::DedicatedTransferQueue,
        Self::BcCompression,
    ];

    /// The capability's name, as written in
    /// `CLOCKWORK_REQUIRED_CAPABILITIES`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::PresentTiming => "present-timing",
            Self::SwapchainMaintenance => "swapchain-maintenance",
            Self::DedicatedComputeQueue => "dedicated-compute-queue",
            Self::DedicatedTransferQueue => "dedicated-transfer-queue",
            Self::BcCompression => "bc-compression",
        }
    }

    /// Whether a device with `capabilities` and `present`, whose queues of
    /// their own are of the kinds in `dedicated`, has the capability.
    pub(crate) fn is_in(
        self,
        capabilities: &Capabilities,
        present: &PresentCapabilities,
        dedicated: QueueKindSet,
    ) -> bool {
        match self {
            Self::PresentTiming => present.present_timing,
            Self::SwapchainMaintenance => present.swapchain_maintenance,
            Self::DedicatedComputeQueue => dedicated.contains(QueueKind::Compute),
            Self::DedicatedTransferQueue => dedicated.contains(QueueKind::Transfer),
            Self::BcCompression => capabilities.bc_compression,
        }
    }
}

/// A headless Vulkan backend in a test's session, with validation required.
///
/// Configured from the test environment (see [`crate::test_support`]), with
/// validation always required, so a test using it can't pass without
/// validation. It derefs to the backend, so tests can reach its device.
///
/// Drop it before the [`TestSession`], so that messages logged while the
/// backend is destroyed are attributed to the session while it's still
/// checking: declare the session first.
pub(crate) struct VulkanTestBackend {
    /// The backend.
    rhi: VulkanRenderHardwareInterface,
    /// The capabilities whose absence fails tests, rather than skipping them.
    required: RequiredCapabilities,
}

impl VulkanTestBackend {
    /// Creates a headless backend in `session`, configured from the test
    /// environment.
    ///
    /// # Panics
    ///
    /// Panics if no device meets the requirements, or if the requested
    /// validation isn't available, or if the test environment is invalid.
    pub(crate) fn headless(session: &TestSession) -> Self {
        let options = test_support::backend_options(session);
        let rhi = VulkanRenderHardwareInterface::new(None, &options)
            .unwrap_or_else(|err| panic!("failed to create a headless Vulkan backend: {err:#}"));
        let names = Capability::ALL.map(Capability::name);
        Self {
            rhi,
            required: test_support::required_capabilities(&names),
        }
    }

    /// Returns whether the device has `capability`. If it doesn't, prints a
    /// skip message, or panics if `CLOCKWORK_REQUIRED_CAPABILITIES` includes
    /// it.
    ///
    /// # Panics
    ///
    /// Panics if the device lacks `capability` and it's required.
    pub(crate) fn has(&self, capability: Capability) -> bool {
        let capabilities = self.rhi.resources().capabilities();
        let dedicated = QueueKind::ALL
            .into_iter()
            .filter(|&kind| self.rhi.queue(kind).kind() == kind)
            .fold(QueueKindSet::EMPTY, QueueKindSet::with);
        let supported = capability.is_in(capabilities, self.rhi.present_capabilities(), dedicated);
        if !supported {
            test_support::missing_capability(
                capability.name(),
                &capabilities.device_name,
                &self.required,
            );
        }
        supported
    }
}

impl std::ops::Deref for VulkanTestBackend {
    type Target = VulkanRenderHardwareInterface;

    fn deref(&self) -> &Self::Target {
        &self.rhi
    }
}

/// Returns from the test early if `backend` lacks any of the capabilities,
/// after reporting each one it lacks (see [`VulkanTestBackend::has`]).
///
/// ```ignore
/// let session = TestSession::new();
/// let backend = VulkanTestBackend::headless(&session);
/// require_capabilities!(backend, Capability::DedicatedTransferQueue);
/// ```
macro_rules! require_capabilities {
    ($backend:expr, $($capability:expr),+ $(,)?) => {
        // Ask about every capability, so that each missing one is reported.
        let mut has_all = true;
        $(has_all &= $backend.has($capability);)+
        if !has_all {
            return;
        }
    };
}
#[expect(
    unused_imports,
    reason = "no GPU test needs an optional capability until design 0002's rhi-commands"
)]
pub(crate) use require_capabilities;

#[cfg(test)]
mod tests {
    //! T0 tests of the capability list and [`require_capabilities!`].

    use std::cell::RefCell;

    use super::*;
    use crate::Limits;

    /// Every capability has a distinct name, so none can be required by
    /// mistake in place of another.
    #[test]
    fn capability_names_are_distinct() {
        let mut names = Capability::ALL.map(Capability::name).to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Capability::ALL.len());
    }

    /// Capabilities, present capabilities and dedicated queue kinds, with
    /// every optional one set to `has`.
    fn capabilities(has: bool) -> (Capabilities, PresentCapabilities, QueueKindSet) {
        let capabilities = Capabilities {
            device_name: "device".to_owned(),
            timestamps: QueueKindSet::EMPTY.with(QueueKind::Graphics),
            bc_compression: has,
            limits: Limits {
                max_texture_dimension_2d: 16_384,
                max_compute_workgroup_size: [1_024, 1_024, 64],
                timestamp_period_ns: 1.0,
            },
        };
        let present = PresentCapabilities {
            present_timing: has,
            swapchain_maintenance: has,
        };
        let dedicated = if has {
            QueueKindSet::EMPTY
                .with(QueueKind::Compute)
                .with(QueueKind::Transfer)
        } else {
            QueueKindSet::EMPTY
        };
        (capabilities, present, dedicated)
    }

    /// Each capability reads its own field: a device has it when that
    /// field is set, and only then.
    #[test]
    fn each_capability_reads_its_field() {
        for capability in Capability::ALL {
            let (with, with_present, with_dedicated) = capabilities(true);
            assert!(
                capability.is_in(&with, &with_present, with_dedicated),
                "{capability:?}"
            );
            let (without, without_present, without_dedicated) = capabilities(false);
            assert!(
                !capability.is_in(&without, &without_present, without_dedicated),
                "{capability:?}"
            );
            for other in Capability::ALL
                .into_iter()
                .filter(|&other| other != capability)
            {
                // Clearing another capability's field doesn't change this
                // one.
                let (mut capabilities, mut present, mut dedicated) = capabilities(true);
                match other {
                    Capability::PresentTiming => present.present_timing = false,
                    Capability::SwapchainMaintenance => present.swapchain_maintenance = false,
                    Capability::DedicatedComputeQueue => {
                        dedicated = QueueKindSet::EMPTY.with(QueueKind::Transfer);
                    }
                    Capability::DedicatedTransferQueue => {
                        dedicated = QueueKindSet::EMPTY.with(QueueKind::Compute);
                    }
                    Capability::BcCompression => capabilities.bc_compression = false,
                }
                assert!(
                    capability.is_in(&capabilities, &present, dedicated),
                    "{capability:?}, {other:?}"
                );
            }
        }
    }

    /// A fake backend, with the capabilities it has, recording which were
    /// asked about.
    struct FakeBackend {
        /// The capabilities the fake device has.
        capabilities: Vec<Capability>,
        /// The capabilities asked about, in order.
        asked: RefCell<Vec<Capability>>,
    }

    impl FakeBackend {
        /// Records the question, and answers it, without skipping or failing.
        fn has(&self, capability: Capability) -> bool {
            self.asked.borrow_mut().push(capability);
            self.capabilities.contains(&capability)
        }
    }

    /// Runs `require_capabilities!` for two capabilities against `backend`,
    /// in a function returning `()` as tests do, and returns whether the
    /// code after it ran.
    fn reaches_the_test_body(backend: &FakeBackend) -> bool {
        let mut reached = false;
        let mut test = || {
            require_capabilities!(
                backend,
                Capability::BcCompression,
                Capability::PresentTiming,
            );
            reached = true;
        };
        test();
        reached
    }

    /// The test continues when the device has every capability.
    #[test]
    fn require_capabilities_continues_with_every_capability() {
        let backend = FakeBackend {
            capabilities: vec![Capability::BcCompression, Capability::PresentTiming],
            asked: RefCell::default(),
        };
        assert!(reaches_the_test_body(&backend));
    }

    /// The test returns early when the device lacks one, but every
    /// capability is still asked about, so each missing one is reported.
    #[test]
    fn require_capabilities_returns_early_after_asking_about_all() {
        let backend = FakeBackend {
            capabilities: vec![Capability::PresentTiming],
            asked: RefCell::default(),
        };
        assert!(!reaches_the_test_body(&backend));
        assert_eq!(
            *backend.asked.borrow(),
            [Capability::BcCompression, Capability::PresentTiming],
        );
    }
}
