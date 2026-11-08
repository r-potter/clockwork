//! GPU timings: the results [`Queue::take_timings`](crate::Queue::take_timings)
//! returns, converting timestamps to durations, and keeping results until
//! they're taken (design 0002, "Timestamps").
//!
//! A queue that can write timestamps has a [`TimestampClock`], from its
//! family's valid bits and the device's tick period. A scope's duration is
//! the difference of its two timestamps, masked to the valid bits so that a
//! counter that wraps between them still gives the right answer, times the
//! period.
//!
//! Translation records each scope as a [`ScopeRecord`], naming the queries
//! its timestamps are written to, or none on a queue that can't write them.
//! Once the submission completes, [`resolve`] turns the records and the
//! timestamps read back into [`Timing`]s. Each queue keeps a submission's
//! records in a [`RetainedTimings`] until they're taken, or discarded once
//! more than [`RETAINED_SUBMISSIONS`] have completed untaken. All of it is
//! pure, so it's tested without a device.

use std::collections::{BTreeMap, btree_map};
use std::mem;
use std::time::Duration;

/// How many completed submissions with timings each queue keeps untaken
/// before discarding the oldest. Callers that time every frame take each
/// frame's results a frame or two later, so nothing is lost in practice,
/// and memory stays bounded when they don't.
///
/// [`Queue::take_timings`](crate::Queue::take_timings)'s documentation
/// states this number, since public documentation can't link to it, so a
/// change here goes there too.
pub(crate) const RETAINED_SUBMISSIONS: usize = 64;

/// The timings recorded in one submission (design 0002, "Timestamps").
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubmissionTimings {
    /// Each command buffer that recorded at least one timing scope, in
    /// submission order. Empty if none did, or if the timings were already
    /// taken or discarded.
    pub command_buffers: Vec<CommandBufferTimings>,
}

/// The timings recorded in one command buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandBufferTimings {
    /// The command list's label.
    pub label: String,
    /// Its timing scopes, in the order they began.
    pub scopes: Vec<Timing>,
}

/// One timing scope's result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timing {
    /// The scope's label.
    pub label: String,
    /// How many scopes it's nested in: 0 for an outermost scope.
    pub depth: u32,
    /// The GPU time from the scope's beginning to its end, or `None` on a
    /// queue that can't write timestamps.
    pub duration: Option<Duration>,
}

/// A timing scope as translation records it: its label, its depth, and the
/// queries its timestamps are written to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScopeRecord {
    /// The scope's label.
    pub label: String,
    /// How many scopes it's nested in: 0 for an outermost scope.
    pub depth: u32,
    /// The indices of the queries its beginning and end are written to,
    /// among the command buffer's queries, or `None` on a queue that can't
    /// write timestamps.
    pub queries: Option<[u32; 2]>,
}

/// The timings of `scopes`, whose queries were written with `timestamps`,
/// indexed as the scopes' queries are, on a queue with `clock`. A scope has
/// a duration only if it has queries and the queue has a clock.
///
/// # Panics
///
/// Panics if a scope's query is past the end of `timestamps`.
pub(crate) fn resolve(
    scopes: Vec<ScopeRecord>,
    clock: Option<TimestampClock>,
    timestamps: &[u64],
) -> Vec<Timing> {
    scopes
        .into_iter()
        .map(|scope| Timing {
            duration: scope.queries.zip(clock).map(|([begin, end], clock)| {
                clock.duration(timestamps[begin as usize], timestamps[end as usize])
            }),
            label: scope.label,
            depth: scope.depth,
        })
        .collect()
}

/// One queue's submissions with timings that haven't been taken, by the
/// timeline value each reaches, each holding `T`: what's needed to read its
/// timings back.
///
/// It holds at most [`RETAINED_SUBMISSIONS`] completed submissions once
/// pruned. Submissions still pending aren't discarded, however many there
/// are, since their queries are still in use. So it holds more only while
/// more than that many submissions with timings are in flight.
#[derive(Debug)]
pub(crate) struct RetainedTimings<T> {
    /// The submissions, by the timeline value each reaches.
    submissions: BTreeMap<u64, T>,
}

impl<T> RetainedTimings<T> {
    /// An empty store.
    pub(crate) fn new() -> Self {
        Self {
            submissions: BTreeMap::new(),
        }
    }

    /// Keeps `timings` for the submission that reaches `value`.
    ///
    /// # Panics
    ///
    /// Panics if a submission that reaches `value` is already kept.
    pub(crate) fn insert(&mut self, value: u64, timings: T) {
        let previous = self.submissions.insert(value, timings);
        assert!(
            previous.is_none(),
            "timings for the submission reaching {value} are kept twice",
        );
    }

    /// Takes the timings of the submission that reaches `value`, or `None`
    /// if none are kept.
    pub(crate) fn take(&mut self, value: u64) -> Option<T> {
        self.submissions.remove(&value)
    }

    /// Whether more than [`RETAINED_SUBMISSIONS`] submissions are kept, so
    /// [`prune`](Self::prune) may have some to discard. The timeline only
    /// needs reading when this says so.
    pub(crate) fn needs_pruning(&self) -> bool {
        self.submissions.len() > RETAINED_SUBMISSIONS
    }

    /// Discards the oldest submissions while more than
    /// [`RETAINED_SUBMISSIONS`] are kept, taking only those the timeline,
    /// at `reached`, has completed. Returns each one discarded, with the
    /// value it reaches, so its queries can be recycled.
    pub(crate) fn prune(&mut self, reached: u64) -> Vec<(u64, T)> {
        let mut discarded = Vec::new();
        while self.needs_pruning() {
            match self.submissions.first_entry() {
                Some(oldest) if *oldest.key() <= reached => discarded.push(oldest.remove_entry()),
                _ => break,
            }
        }
        discarded
    }

    /// Takes every submission's timings, for recycling their queries when
    /// the device is idle.
    pub(crate) fn drain(&mut self) -> btree_map::IntoValues<u64, T> {
        mem::take(&mut self.submissions).into_values()
    }

    /// How many submissions are kept, for tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.submissions.len()
    }
}

// Results are plain data, sent to whichever thread takes them (design 0002,
// "Threading").
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SubmissionTimings>();
};

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
    pub(crate) fn duration(self, begin: u64, end: u64) -> Duration {
        let ticks = end.wrapping_sub(begin) & self.mask;
        let nanoseconds = (ticks as f64 * f64::from(self.period_ns)).round();
        // A float to integer cast saturates, which no real duration reaches.
        Duration::from_nanos(nanoseconds as u64)
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of converting timestamps to durations, resolving scopes, and
    //! keeping results until they're taken (design 0002, R22).

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

    /// A scope recorded with `label` at `depth`, written to `queries`.
    fn scope(label: &str, depth: u32, queries: Option<[u32; 2]>) -> ScopeRecord {
        ScopeRecord {
            label: label.to_owned(),
            depth,
            queries,
        }
    }

    /// Nested scopes each get the time between their own two timestamps,
    /// keeping their labels and depths, in the order they began.
    #[test]
    fn nested_scopes_are_resolved_from_their_own_queries() {
        let clock = TimestampClock::new(64, 2.0);
        // The outer scope begins at query 0 and ends at 3, around the inner
        // one at 1 and 2.
        let scopes = vec![
            scope("frame", 0, Some([0, 3])),
            scope("shadows", 1, Some([1, 2])),
        ];
        assert_eq!(
            resolve(scopes, clock, &[100, 110, 160, 200]),
            [
                Timing {
                    label: "frame".to_owned(),
                    depth: 0,
                    duration: Some(Duration::from_nanos(200)),
                },
                Timing {
                    label: "shadows".to_owned(),
                    depth: 1,
                    duration: Some(Duration::from_nanos(100)),
                },
            ],
        );
    }

    /// On a queue that can't write timestamps, scopes have no queries and
    /// no clock, and are reported with no duration, keeping their labels
    /// and depths (design 0002, R22).
    #[test]
    fn scopes_without_timestamps_have_no_duration() {
        let scopes = vec![scope("upload", 0, None), scope("copy", 1, None)];
        let timings = resolve(scopes, None, &[]);
        assert_eq!(
            timings
                .iter()
                .map(|timing| (timing.label.as_str(), timing.depth, timing.duration))
                .collect::<Vec<_>>(),
            [("upload", 0, None), ("copy", 1, None)],
        );
    }

    /// A store with a submission for each value in `values`, each holding
    /// its own value.
    fn retained(values: impl IntoIterator<Item = u64>) -> RetainedTimings<u64> {
        let mut retained = RetainedTimings::new();
        for value in values {
            retained.insert(value, value);
        }
        retained
    }

    /// A submission's timings are taken once, and a submission with none
    /// kept has nothing to take.
    #[test]
    fn timings_are_taken_once() {
        let mut retained = retained([3, 5]);
        assert_eq!(retained.take(5), Some(5));
        assert_eq!(retained.take(5), None);
        assert_eq!(retained.take(4), None);
        assert_eq!(retained.len(), 1);
    }

    /// Once more than the limit have completed, the oldest are discarded,
    /// keeping the most recent, whatever order they were kept in.
    #[test]
    fn the_most_recent_completed_submissions_are_kept() {
        let limit = RETAINED_SUBMISSIONS as u64;
        let mut retained = retained((1..=limit + 2).rev());
        assert!(retained.needs_pruning());
        assert_eq!(retained.prune(limit + 2), [(1, 1), (2, 2)]);
        assert!(!retained.needs_pruning());
        assert_eq!(retained.len(), RETAINED_SUBMISSIONS);
        assert_eq!(retained.take(3), Some(3));
        assert_eq!(retained.prune(limit + 2), []);
    }

    /// A submission still pending isn't discarded, since its queries are
    /// still in use, even when that leaves more than the limit kept.
    #[test]
    fn pending_submissions_are_not_discarded() {
        let limit = RETAINED_SUBMISSIONS as u64;
        let mut retained = retained(1..=limit + 3);
        assert_eq!(retained.prune(1), [(1, 1)]);
        assert_eq!(retained.prune(0), []);
        assert!(retained.needs_pruning());
        assert_eq!(retained.len(), RETAINED_SUBMISSIONS + 2);
    }

    /// Draining takes every submission's timings, pending or not.
    #[test]
    fn draining_takes_every_submission() {
        let mut retained = retained([7, 2, 9]);
        let mut drained: Vec<_> = retained.drain().collect();
        drained.sort_unstable();
        assert_eq!(drained, [2, 7, 9]);
        assert_eq!(retained.len(), 0);
    }

    /// A submission's timings can't be kept twice.
    #[test]
    #[should_panic(expected = "timings for the submission reaching 4 are kept twice")]
    fn keeping_a_submission_twice_panics() {
        retained([4, 4]);
    }
}
