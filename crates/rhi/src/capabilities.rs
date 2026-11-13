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
//!
//! Nor is what a queue can do: whether a kind has a queue of its own is
//! the kind of the queue it's given, [`Queue::kind`](crate::Queue::kind),
//! and whether it can write timestamps is
//! [`Queue::timestamp_period_ns`](crate::Queue::timestamp_period_ns).

use crate::Format;
use crate::copy::required_offset_alignment;

/// The most push data, in bytes, that one draw or dispatch can carry (design
/// 0002, "Binding model"). Device selection requires at least this much, so
/// recording, which needs no device, can check against it.
pub const MAX_PUSH_DATA_SIZE: u32 = 128;

/// The most draws that one indirect draw can run (design 0002, "Command
/// lists and translation"): the least Vulkan allows a device with
/// multi-draw indirect to support. Device selection requires at least this
/// many, so recording, which needs no device, can check against it.
pub const MAX_DRAW_INDIRECT_COUNT: u32 = 65_535;

/// A kind of queue that work is submitted to (design 0002, "Queues and
/// submission").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// The selected device's name, and its optional features and limits for
/// resources and shaders.
#[derive(Debug, Clone, PartialEq)]
pub struct Capabilities {
    /// The device's name, as its driver reports it.
    pub device_name: String,
    /// BC1 to BC7 compressed textures.
    pub bc_compression: bool,
    /// The device's limits that the RHI's users need to know.
    pub limits: Limits,
}

impl Capabilities {
    /// Capabilities for T0 tests: a device named `device`, with the limits
    /// of a typical desktop GPU, and every optional capability set to `has`.
    #[cfg(test)]
    pub(crate) fn for_tests(has: bool) -> Self {
        Self {
            device_name: "device".to_owned(),
            bc_compression: has,
            limits: Limits {
                max_texture_dimension_2d: 16_384,
                max_compute_workgroup_size: [1_024, 1_024, 64],
                optimal_copy_offset_alignment: 64,
            },
        }
    }
}

/// Limits of the selected device that the RHI's users need to know.
#[derive(Debug, Clone, PartialEq)]
pub struct Limits {
    /// The largest width or height of a 2D texture, in texels.
    pub max_texture_dimension_2d: u32,
    /// The largest compute workgroup, in invocations along each axis.
    pub max_compute_workgroup_size: [u32; 3],
    /// The buffer offset alignment, in bytes, at which copies to and from
    /// textures are fastest: a power of two, and a hint rather than a
    /// requirement. [`copy_offset_alignment`](Self::copy_offset_alignment)
    /// combines it with what a format needs, reading 0 as no preference.
    pub optimal_copy_offset_alignment: u64,
}

impl Limits {
    /// The alignment, in bytes, to place texture data of `format` at in a
    /// buffer, for copies to and from textures: the least multiple of both
    /// what every copy of the format needs and
    /// [`optimal_copy_offset_alignment`](Self::optimal_copy_offset_alignment)
    /// (design 0002, "Texture copies"). An optimal alignment of 0, which no
    /// device should report, is read as 1, so the result is never 0.
    pub const fn copy_offset_alignment(&self, format: Format) -> u64 {
        let optimal = if self.optimal_copy_offset_alignment == 0 {
            1
        } else {
            self.optimal_copy_offset_alignment
        };
        lcm(required_offset_alignment(format.info()), optimal)
    }
}

/// The least common multiple of `a` and `b`, which aren't 0, in a
/// `const fn`.
const fn lcm(a: u64, b: u64) -> u64 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a / x * b
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
/// log: for example, `has swapchain maintenance, BC compression; lacks
/// present timing`.
///
/// The log contract counts an optional capability a device lacks as notable,
/// so both are listed.
pub(crate) fn summary(capabilities: &Capabilities, present: &PresentCapabilities) -> String {
    let capabilities = [
        ("present timing", present.present_timing),
        ("swapchain maintenance", present.swapchain_maintenance),
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
    //! T0 tests of the capability summary and copy offset alignments.

    use super::*;

    /// Capabilities with nothing optional.
    fn none() -> Capabilities {
        Capabilities::for_tests(false)
    }

    /// The summary lists every optional capability a device has.
    #[test]
    fn the_summary_lists_what_a_device_has() {
        let capabilities = Capabilities {
            bc_compression: true,
            ..none()
        };
        let present = PresentCapabilities {
            present_timing: true,
            swapchain_maintenance: true,
        };
        assert_eq!(
            summary(&capabilities, &present),
            "has present timing, swapchain maintenance, BC compression",
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
            "has swapchain maintenance, BC compression; lacks present timing",
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
            "lacks present timing, swapchain maintenance, BC compression",
        );
    }

    /// The limits of a device whose optimal copy offset alignment is
    /// `optimal` bytes.
    fn limits(optimal: u64) -> Limits {
        Limits {
            optimal_copy_offset_alignment: optimal,
            ..Capabilities::for_tests(true).limits
        }
    }

    /// A copy's offset alignment is the optimal one where that's a multiple
    /// of what the format needs, and what the format needs where the
    /// optimal one is finer.
    #[test]
    fn copy_offset_alignments_suit_the_device_and_format() {
        assert_eq!(limits(64).copy_offset_alignment(Format::Rgba8Unorm), 64);
        assert_eq!(limits(64).copy_offset_alignment(Format::Bc7Unorm), 64);
        assert_eq!(limits(1).copy_offset_alignment(Format::Rgba32Float), 16);
        assert_eq!(limits(1).copy_offset_alignment(Format::Depth32Float), 4);
        assert_eq!(limits(4).copy_offset_alignment(Format::Bc1RgbaSrgb), 8);
    }

    /// An optimal alignment of 0 is no preference, so the format's own
    /// alignment applies, rather than an alignment of 0, which can't be
    /// rounded up to.
    #[test]
    fn zero_optimal_alignments_are_ignored() {
        assert_eq!(limits(0).copy_offset_alignment(Format::Rgba8Unorm), 4);
        assert_eq!(limits(0).copy_offset_alignment(Format::Bc7Unorm), 16);
    }

    /// The alignment is a multiple of both when neither divides the other,
    /// which Vulkan's powers of two never cause, but a format of three
    /// 32-bit channels would.
    #[test]
    fn copy_offset_alignments_are_common_multiples() {
        assert_eq!(lcm(12, 8), 24);
        assert_eq!(lcm(8, 12), 24);
        assert_eq!(lcm(16, 64), 64);
        assert_eq!(lcm(1, 1), 1);
    }
}
