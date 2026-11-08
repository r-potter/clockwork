//! Converting GPU timestamps to durations (design 0002, "Timestamps").
//!
//! A queue that can write timestamps has a [`TimestampClock`], from its
//! family's valid bits and the device's tick period. A scope's duration is
//! the difference of its two timestamps, masked to the valid bits so that a
//! counter that wraps between them still gives the right answer, times the
//! period. The conversion is pure, so it's tested without a device.

use std::time::Duration;

/// The timestamp counter of a queue that can write timestamps: how many of
/// its bits are valid, and how long a tick lasts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TimestampClock {
    /// The counter's valid bits, as a mask of the low bits.
    mask: u64,
    /// How long one tick lasts, in nanoseconds.
    period_ns: f32,
}

impl TimestampClock {
    /// The clock of a counter with `valid_bits` valid bits, whose ticks last
    /// `period_ns` nanoseconds, or `None` for 0 valid bits, which means the
    /// queue can't write timestamps.
    ///
    /// # Panics
    ///
    /// Panics if `valid_bits` is more than 64.
    pub(crate) fn new(valid_bits: u32, period_ns: f32) -> Option<Self> {
        assert!(
            valid_bits <= u64::BITS,
            "a timestamp counter has at most 64 valid bits, not {valid_bits}",
        );
        if valid_bits == 0 {
            return None;
        }
        Some(Self {
            mask: u64::MAX >> (u64::BITS - valid_bits),
            period_ns,
        })
    }

    /// How long one tick lasts, in nanoseconds.
    pub(crate) fn period_ns(self) -> f32 {
        self.period_ns
    }

    /// The time from the timestamp `begin` to the later timestamp `end`,
    /// rounded to the nearest nanosecond. Only the valid bits are used, so a
    /// counter that wrapped once in between still gives the right answer.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "timings are read back later in `rhi-timing`")
    )]
    pub(crate) fn duration(self, begin: u64, end: u64) -> Duration {
        let ticks = end.wrapping_sub(begin) & self.mask;
        let nanoseconds = (ticks as f64 * f64::from(self.period_ns)).round();
        // A float to integer cast saturates, which no real duration reaches.
        Duration::from_nanos(nanoseconds as u64)
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of converting timestamps to durations (design 0002, R22).

    use super::*;

    /// A queue whose family has no valid bits can't write timestamps, so
    /// it has no clock.
    #[test]
    fn no_valid_bits_means_no_clock() {
        assert_eq!(TimestampClock::new(0, 1.0), None);
    }

    /// The clock keeps the period, for `Queue::timestamp_period_ns`.
    #[test]
    fn the_clock_keeps_its_period() {
        let clock = TimestampClock::new(64, 0.833).unwrap();
        assert_eq!(clock.period_ns(), 0.833);
    }

    /// A duration is the difference in ticks times the period, rounded to
    /// the nearest nanosecond.
    #[test]
    fn durations_are_ticks_times_the_period() {
        let clock = TimestampClock::new(64, 1.0).unwrap();
        assert_eq!(clock.duration(1_000, 1_000), Duration::ZERO);
        assert_eq!(clock.duration(1_000, 3_500), Duration::from_nanos(2_500));
        let clock = TimestampClock::new(64, 41.667).unwrap();
        assert_eq!(clock.duration(10, 13), Duration::from_nanos(125));
    }

    /// A 64-bit counter that wraps between the timestamps gives the time
    /// across the wrap.
    #[test]
    fn a_wrapped_64_bit_counter_gives_the_right_duration() {
        let clock = TimestampClock::new(64, 1.0).unwrap();
        assert_eq!(clock.duration(u64::MAX - 9, 5), Duration::from_nanos(15));
    }

    /// A narrower counter wraps at its valid bits, and the bits above them
    /// are ignored, whatever they hold.
    #[test]
    fn a_wrapped_narrow_counter_gives_the_right_duration() {
        let clock = TimestampClock::new(36, 1.0).unwrap();
        let top = (1 << 36) - 1;
        assert_eq!(clock.duration(top - 9, 5), Duration::from_nanos(15));
        assert_eq!(
            clock.duration((top - 9) | (0xABC << 40), 5 | (0x123 << 40)),
            Duration::from_nanos(15),
        );
        assert_eq!(clock.duration(100, 250), Duration::from_nanos(150));
    }

    /// More than 64 valid bits isn't a counter Vulkan can report.
    #[test]
    #[should_panic(expected = "a timestamp counter has at most 64 valid bits, not 65")]
    fn more_than_64_valid_bits_panics() {
        TimestampClock::new(65, 1.0);
    }
}
