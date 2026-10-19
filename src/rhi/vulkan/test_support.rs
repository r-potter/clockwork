//! Fixtures for the Vulkan backend's T1 tests: [`VulkanTestBackend`], the
//! test-only [`Capability`] list, and [`require_capabilities!`].
//!
//! The capability list is a stand-in, for tests only. Design `0002-rhi`
//! replaces it with the RHI's own capability query.

use ash::vk;

use super::VulkanRenderHardwareInterface;
use super::physical_device::PhysicalDevice;
use crate::diagnostics::capture::TestSession;
use crate::test_support::{self, RequiredCapabilities};

/// A capability some T1 tests need, which some devices lack.
///
/// A test that needs one asks [`VulkanTestBackend::has`], usually through
/// [`require_capabilities!`], and is skipped on devices without it, unless
/// `CLOCKWORK_REQUIRED_CAPABILITIES` requires it (R9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// `VK_EXT_descriptor_heap` (ADR 0002).
    DescriptorHeap,
    /// `VK_KHR_device_address_commands` (ADR 0002).
    DeviceAddressCommands,
    /// `VK_EXT_present_timing` (ADR 0002).
    PresentTiming,
    /// A queue family that supports compute but not graphics.
    DedicatedComputeQueue,
    /// A queue family that supports transfers, but neither graphics nor
    /// compute.
    DedicatedTransferQueue,
}

impl Capability {
    /// Every capability.
    pub const ALL: [Self; 5] = [
        Self::DescriptorHeap,
        Self::DeviceAddressCommands,
        Self::PresentTiming,
        Self::DedicatedComputeQueue,
        Self::DedicatedTransferQueue,
    ];

    /// The capability's name, as written in
    /// `CLOCKWORK_REQUIRED_CAPABILITIES`.
    pub fn name(self) -> &'static str {
        match self {
            Self::DescriptorHeap => "descriptor-heap",
            Self::DeviceAddressCommands => "device-address-commands",
            Self::PresentTiming => "present-timing",
            Self::DedicatedComputeQueue => "dedicated-compute-queue",
            Self::DedicatedTransferQueue => "dedicated-transfer-queue",
        }
    }

    /// Whether `device` has the capability.
    ///
    /// The extensions are checked by name only: `ash` 0.38 has no bindings
    /// for their feature structures, so their features can't be queried until
    /// `foundations-ash-1.4`.
    pub fn is_supported_by(self, device: &PhysicalDevice) -> bool {
        let has_family = |wanted: vk::QueueFlags, unwanted: vk::QueueFlags| {
            device.queue_families.iter().any(|family| {
                family.queue_flags.contains(wanted) && !family.queue_flags.intersects(unwanted)
            })
        };
        match self {
            Self::DescriptorHeap => device.supports_extension(c"VK_EXT_descriptor_heap"),
            Self::DeviceAddressCommands => {
                device.supports_extension(c"VK_KHR_device_address_commands")
            }
            Self::PresentTiming => device.supports_extension(c"VK_EXT_present_timing"),
            Self::DedicatedComputeQueue => {
                has_family(vk::QueueFlags::COMPUTE, vk::QueueFlags::GRAPHICS)
            }
            Self::DedicatedTransferQueue => has_family(
                vk::QueueFlags::TRANSFER,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            ),
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
pub struct VulkanTestBackend {
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
    pub fn headless(session: &TestSession) -> Self {
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
    #[expect(
        dead_code,
        reason = "no T1 test needs a capability until foundations-ash-1.4's reference_device_reports_required_extensions"
    )]
    pub fn has(&self, capability: Capability) -> bool {
        let device = &self.rhi.physical_device;
        let supported = capability.is_supported_by(device);
        if !supported {
            test_support::missing_capability(capability.name(), &device.name(), &self.required);
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
/// require_capabilities!(backend, Capability::DescriptorHeap);
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
    reason = "no T1 test needs a capability until foundations-ash-1.4's reference_device_reports_required_extensions"
)]
pub(crate) use require_capabilities;

#[cfg(test)]
mod tests {
    //! T0 tests of the capability list and [`require_capabilities!`].

    use std::cell::RefCell;

    use super::*;

    /// Every capability has a distinct name, so none can be required by
    /// mistake in place of another.
    #[test]
    fn capability_names_are_distinct() {
        let mut names = Capability::ALL.map(Capability::name).to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Capability::ALL.len());
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
                Capability::DescriptorHeap,
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
            capabilities: vec![Capability::DescriptorHeap, Capability::PresentTiming],
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
            [Capability::DescriptorHeap, Capability::PresentTiming],
        );
    }
}
