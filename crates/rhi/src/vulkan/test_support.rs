//! Fixtures for the Vulkan backend's T1 tests: [`VulkanTestBackend`], the
//! test-only [`Capability`] list, [`require_capabilities!`], and
//! [`TestCommands`], for recording raw Vulkan commands.
//!
//! [`Capability`] names the optional parts of the RHI's [`Capabilities`],
//! [`PresentCapabilities`] and queues, so that tests can require them by name in
//! `CLOCKWORK_REQUIRED_CAPABILITIES`. Whether a device has one is read from
//! the object that reports it, as the RHI's users read it (design 0002,
//! "Capabilities").

use ash::vk;
use clockwork_diagnostics::capture::TestSession;

use super::VulkanRenderHardwareInterface;
use super::device_selection::copies_break_on_transfer_families;
use crate::test_support::{self, RequiredCapabilities};
use crate::{Capabilities, PresentCapabilities, QueueKind, RenderHardwareInterface};

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
        dedicated: &[QueueKind],
    ) -> bool {
        match self {
            Self::PresentTiming => present.present_timing,
            Self::SwapchainMaintenance => present.swapchain_maintenance,
            Self::DedicatedComputeQueue => dedicated.contains(&QueueKind::Compute),
            Self::DedicatedTransferQueue => dedicated.contains(&QueueKind::Transfer),
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
    /// A dedicated transfer queue that device selection left unused because
    /// its driver breaks copies there is a driver bug, not a missing
    /// capability, so it skips even when the capability is required.
    ///
    /// # Panics
    ///
    /// Panics if the device lacks `capability` and it's required.
    pub(crate) fn has(&self, capability: Capability) -> bool {
        let capabilities = self.rhi.resources().capabilities();
        let dedicated: Vec<_> = QueueKind::ALL
            .into_iter()
            .filter(|&kind| self.rhi.queue(kind).kind() == kind)
            .collect();
        let supported = capability.is_in(capabilities, self.rhi.present_capabilities(), &dedicated);
        if !supported
            && capability == Capability::DedicatedTransferQueue
            && copies_break_on_transfer_families(&self.rhi.core.physical_device)
        {
            test_support::skip_for_driver_bug(
                &capabilities.device_name,
                "copies by address break on its transfer-only queue family, so transfers \
                 share the compute family (see TODO.md)",
            );
        } else if !supported {
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

/// A command pool with one primary command buffer, destroyed when dropped,
/// for tests that record raw Vulkan commands.
pub(crate) struct TestCommands<'a> {
    /// The device the pool was created on.
    device: &'a ash::Device,
    /// The pool, which frees `command_buffer` when destroyed.
    pub(crate) pool: vk::CommandPool,
    /// A command buffer allocated from `pool`.
    pub(crate) command_buffer: vk::CommandBuffer,
}

impl<'a> TestCommands<'a> {
    /// Creates a command pool for `backend`'s graphics queue family, and
    /// allocates one primary command buffer from it.
    ///
    /// # Panics
    ///
    /// Panics if any Vulkan call fails.
    pub(crate) fn new(backend: &'a VulkanTestBackend) -> Self {
        let device = backend.core.device.handle();
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(backend.core.device.queue_families().graphics);
        // SAFETY: `device` is a valid logical device, and `pool_create_info`
        // is valid for the call.
        let pool = unsafe { device.create_command_pool(&pool_create_info, None) }.unwrap();
        let mut commands = Self {
            device,
            pool,
            command_buffer: vk::CommandBuffer::null(),
        };
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was created on `device`, and `allocate_info` is
        // valid for the call.
        commands.command_buffer =
            unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap()[0];
        commands
    }
}

impl Drop for TestCommands<'_> {
    fn drop(&mut self) {
        // SAFETY: the pool was created on `device`, and its command buffer
        // is never left pending: it's submitted only by tests that wait for
        // it. Destroying the pool frees the command buffer.
        unsafe { self.device.destroy_command_pool(self.pool, None) };
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
    fn capabilities(has: bool) -> (Capabilities, PresentCapabilities, Vec<QueueKind>) {
        let capabilities = Capabilities {
            device_name: "device".to_owned(),
            bc_compression: has,
            limits: Limits {
                max_texture_dimension_2d: 16_384,
                max_compute_workgroup_size: [1_024, 1_024, 64],
            },
        };
        let present = PresentCapabilities {
            present_timing: has,
            swapchain_maintenance: has,
        };
        let dedicated = if has {
            vec![QueueKind::Compute, QueueKind::Transfer]
        } else {
            Vec::new()
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
                capability.is_in(&with, &with_present, &with_dedicated),
                "{capability:?}"
            );
            let (without, without_present, without_dedicated) = capabilities(false);
            assert!(
                !capability.is_in(&without, &without_present, &without_dedicated),
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
                    Capability::DedicatedComputeQueue => dedicated = vec![QueueKind::Transfer],
                    Capability::DedicatedTransferQueue => dedicated = vec![QueueKind::Compute],
                    Capability::BcCompression => capabilities.bc_compression = false,
                }
                assert!(
                    capability.is_in(&capabilities, &present, &dedicated),
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
