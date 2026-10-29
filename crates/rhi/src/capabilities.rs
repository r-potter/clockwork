//! What the selected device can do beyond the baseline every backend
//! requires (design 0002, "Capabilities").
//!
//! A device without the required features is never selected, so they aren't
//! listed here. What varies between devices that were selected is reported
//! by the object it constrains: [`Capabilities`], from
//! [`GpuResources`](crate::GpuResources), for resources and shaders, and
//! [`PresentCapabilities`], from the
//! [`RenderHardwareInterface`](crate::RenderHardwareInterface), for
//! presenting. Backends derive both once, when the device is selected.

/// The most push data, in bytes, that one draw or dispatch can carry (design
/// 0002, "Binding model"). Device selection requires at least this much, so
/// recording, which needs no device, can check against it.
pub const MAX_PUSH_DATA_SIZE: u32 = 128;

/// A kind of queue that work is submitted to (design 0002, "Queues and
/// submission").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueKind {
    /// Rendering, and anything else. Every device has one.
    Graphics,
    /// Compute dispatches and transfers, on a dedicated queue where the
    /// device has one, and otherwise sharing the graphics queue.
    Compute,
    /// Transfers only, on a dedicated queue where the device has one, and
    /// otherwise sharing a more capable queue.
    Transfer,
}

impl QueueKind {
    /// Every kind, from the most capable to the least.
    pub const ALL: [Self; 3] = [Self::Graphics, Self::Compute, Self::Transfer];

    /// The kind's name, in messages and debug tools.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Graphics => "graphics",
            Self::Compute => "compute",
            Self::Transfer => "transfer",
        }
    }
}

/// A set of [`QueueKind`]s.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueueKindSet {
    /// Whether [`QueueKind::Graphics`] is in the set.
    graphics: bool,
    /// Whether [`QueueKind::Compute`] is in the set.
    compute: bool,
    /// Whether [`QueueKind::Transfer`] is in the set.
    transfer: bool,
}

impl QueueKindSet {
    /// The empty set.
    pub const EMPTY: Self = Self {
        graphics: false,
        compute: false,
        transfer: false,
    };

    /// This set with `kind` added.
    pub const fn with(mut self, kind: QueueKind) -> Self {
        match kind {
            QueueKind::Graphics => self.graphics = true,
            QueueKind::Compute => self.compute = true,
            QueueKind::Transfer => self.transfer = true,
        }
        self
    }

    /// Whether `kind` is in the set.
    pub const fn contains(self, kind: QueueKind) -> bool {
        match kind {
            QueueKind::Graphics => self.graphics,
            QueueKind::Compute => self.compute,
            QueueKind::Transfer => self.transfer,
        }
    }
}

/// The selected device's name, and its optional features and limits for
/// resources and shaders.
#[derive(Debug, Clone, PartialEq)]
pub struct Capabilities {
    /// The device's name, as its driver reports it.
    pub device_name: String,
    /// A queue that supports compute but not graphics, so that compute work
    /// can run alongside rendering.
    pub dedicated_compute_queue: bool,
    /// A queue that supports transfers but neither graphics nor compute, so
    /// that uploads can run alongside other work.
    pub dedicated_transfer_queue: bool,
    /// Which queue kinds can write timestamps. Always includes
    /// [`QueueKind::Graphics`], which device selection requires.
    pub timestamps: QueueKindSet,
    /// BC1 to BC7 compressed textures.
    pub bc_compression: bool,
    /// The device's limits that the RHI's users need to know.
    pub limits: Limits,
}

/// Limits of the selected device that the RHI's users need to know.
#[derive(Debug, Clone, PartialEq)]
pub struct Limits {
    /// The largest width or height of a 2D texture, in texels.
    pub max_texture_dimension_2d: u32,
    /// The largest compute workgroup, in invocations along each axis.
    pub max_compute_workgroup_size: [u32; 3],
    /// How long one timestamp tick lasts, in nanoseconds.
    pub timestamp_period_ns: f32,
}

/// What presenting to a window can do on the selected device. It describes
/// the device, so a headless backend reports it too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentCapabilities {
    /// Presentation timing (`VK_EXT_present_timing` on Vulkan), for frame
    /// pacing.
    pub present_timing: bool,
    /// Swapchain maintenance (`VK_KHR_swapchain_maintenance1` on Vulkan),
    /// which lets a swapchain be destroyed without waiting for the device to
    /// be idle.
    pub swapchain_maintenance: bool,
}

/// Summarises the optional capabilities the device has and lacks, for the
/// log: for example, `has dedicated compute queue, BC compression; lacks
/// present timing`.
///
/// The log contract counts an optional capability a device lacks as notable,
/// so both are listed.
pub(crate) fn summary(capabilities: &Capabilities, present: &PresentCapabilities) -> String {
    let capabilities = [
        ("present timing", present.present_timing),
        ("swapchain maintenance", present.swapchain_maintenance),
        (
            "dedicated compute queue",
            capabilities.dedicated_compute_queue,
        ),
        (
            "dedicated transfer queue",
            capabilities.dedicated_transfer_queue,
        ),
        ("BC compression", capabilities.bc_compression),
    ];
    let names = |wanted: bool| -> Vec<&str> {
        capabilities
            .iter()
            .filter(|(_, has)| *has == wanted)
            .map(|(name, _)| *name)
            .collect()
    };
    let (has, lacks) = (names(true), names(false));
    match (has.is_empty(), lacks.is_empty()) {
        (false, true) => format!("has {}", has.join(", ")),
        (true, false) => format!("lacks {}", lacks.join(", ")),
        _ => format!("has {}; lacks {}", has.join(", "), lacks.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of [`QueueKindSet`] and the capability summary.

    use super::*;

    /// Capabilities with nothing optional.
    fn none() -> Capabilities {
        Capabilities {
            device_name: "device".to_owned(),
            dedicated_compute_queue: false,
            dedicated_transfer_queue: false,
            timestamps: QueueKindSet::EMPTY.with(QueueKind::Graphics),
            bc_compression: false,
            limits: Limits {
                max_texture_dimension_2d: 16_384,
                max_compute_workgroup_size: [1_024, 1_024, 64],
                timestamp_period_ns: 1.0,
            },
        }
    }

    /// A set contains exactly the kinds added to it.
    #[test]
    fn queue_kind_sets_contain_what_was_added() {
        let kinds = QueueKindSet::EMPTY
            .with(QueueKind::Graphics)
            .with(QueueKind::Transfer);
        assert!(kinds.contains(QueueKind::Graphics));
        assert!(!kinds.contains(QueueKind::Compute));
        assert!(kinds.contains(QueueKind::Transfer));
        assert_eq!(QueueKindSet::default(), QueueKindSet::EMPTY);
    }

    /// The summary lists every optional capability a device has.
    #[test]
    fn the_summary_lists_what_a_device_has() {
        let capabilities = Capabilities {
            dedicated_compute_queue: true,
            dedicated_transfer_queue: true,
            bc_compression: true,
            ..none()
        };
        let present = PresentCapabilities {
            present_timing: true,
            swapchain_maintenance: true,
        };
        assert_eq!(
            summary(&capabilities, &present),
            "has present timing, swapchain maintenance, dedicated compute queue, dedicated \
             transfer queue, BC compression",
        );
    }

    /// The summary lists what a device lacks too.
    #[test]
    fn the_summary_lists_what_a_device_lacks() {
        let capabilities = Capabilities {
            bc_compression: true,
            ..none()
        };
        let present = PresentCapabilities {
            present_timing: false,
            swapchain_maintenance: true,
        };
        assert_eq!(
            summary(&capabilities, &present),
            "has swapchain maintenance, BC compression; lacks present timing, dedicated \
             compute queue, dedicated transfer queue",
        );
    }

    /// A device with nothing optional lists only what it lacks.
    #[test]
    fn the_summary_of_a_device_without_capabilities() {
        let present = PresentCapabilities {
            present_timing: false,
            swapchain_maintenance: false,
        };
        assert_eq!(
            summary(&none(), &present),
            "lacks present timing, swapchain maintenance, dedicated compute queue, dedicated \
             transfer queue, BC compression",
        );
    }
}
