//! Partitioning a batch's command lists into translation units (design
//! 0003, "Coalescing", R10).
//!
//! Each pass is recorded into a list of its own, so passes record in
//! parallel. Translating each list into a command buffer of its own would
//! pay what every command buffer costs, a command pool, beginning and
//! ending it, and binding the descriptor heaps, once for each pass, which
//! for a fullscreen pass of a handful of commands can be most of its cost.
//! So consecutive lists are translated together, in units, each one job
//! and one command buffer.
//!
//! A unit takes lists, in the order they run, until it holds a threshold
//! of commands, [`UNIT_COMMANDS`] unless a test asks for another. A list
//! that holds that many on its own gets a unit of its own, and closes the
//! unit before it, so a pass with thousands of draws still gets a thread
//! to itself. The threshold isn't for the GPU, which doesn't mind a large
//! command buffer: it spreads translation's work across the worker pool,
//! since each command buffer is translated by one thread.
//!
//! It's a pure function of the lists' command counts, so a frame's units
//! are deterministic, and it's called for one batch at a time, so a unit
//! never crosses a batch, which is one submission on one queue. Units
//! change no synchronization: within a submission, a barrier orders every
//! command after it, whichever command buffer it's in.

use std::ops::Range;

/// The commands a translation unit holds before it closes: a starting
/// point, which the executing benchmarks tune.
#[expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, uses it")]
pub(crate) const UNIT_COMMANDS: usize = 1_000;

/// Appends to `units` the translation units of a batch whose lists hold
/// `counts` commands, in the order they run, with `threshold` the commands
/// a unit holds before it closes. Each unit is a range of the indices
/// among this batch's lists, counted from 0, so a caller holding several
/// batches' lists offsets each batch's units by where its lists start.
///
/// A threshold of 1 puts every list in a unit of its own, and
/// `usize::MAX` the whole batch in one, as the tests that render each way
/// ask for.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
)]
pub(crate) fn partition(
    counts: impl IntoIterator<Item = usize>,
    threshold: usize,
    units: &mut Vec<Range<usize>>,
) {
    // The first list of the open unit, how many commands it holds, and one
    // past the last list seen, so a unit still open at the end is closed.
    let mut start = 0;
    let mut held = 0;
    let mut end = 0;
    for (index, count) in counts.into_iter().enumerate() {
        end = index + 1;
        if count >= threshold {
            if start < index {
                units.push(start..index);
            }
            units.push(index..end);
            start = end;
            held = 0;
            continue;
        }
        held += count;
        if held >= threshold {
            units.push(start..end);
            start = end;
            held = 0;
        }
    }
    if start < end {
        units.push(start..end);
    }
}

#[cfg(test)]
#[expect(
    clippy::single_range_in_vec_init,
    reason = "each range is a unit, so a list of one is one unit, not a range to collect"
)]
mod tests {
    //! T0 tests of partitioning (R10).

    use super::*;

    /// The units of a batch whose lists hold `counts` commands, with a
    /// threshold of 10.
    fn units(counts: &[usize]) -> Vec<Range<usize>> {
        let mut units = Vec::new();
        partition(counts.iter().copied(), 10, &mut units);
        units
    }

    /// Small lists are joined until a unit holds the threshold, which
    /// closes it, and the last unit takes what's left.
    #[test]
    fn small_lists_are_joined_up_to_the_threshold() {
        assert_eq!(units(&[3, 3, 3, 1, 4, 4]), [0..4, 4..6]);
        assert_eq!(units(&[9, 2, 9]), [0..2, 2..3]);
    }

    /// A list that holds the threshold on its own gets a unit of its own,
    /// closing the unit before it, which keeps what it held.
    #[test]
    fn a_large_list_gets_a_unit_of_its_own() {
        assert_eq!(units(&[2, 2, 50, 2]), [0..2, 2..3, 3..4]);
        assert_eq!(units(&[50, 10, 3]), [0..1, 1..2, 2..3]);
    }

    /// An empty batch has no units, and lists with no commands join the
    /// unit they're in.
    #[test]
    fn an_empty_batch_has_no_units() {
        assert_eq!(units(&[]), Vec::<Range<usize>>::new());
        assert_eq!(units(&[0, 0]), [0..2]);
    }

    /// A threshold of 1 puts every list in a unit of its own, and
    /// `usize::MAX` the whole batch in one.
    #[test]
    fn thresholds_give_a_unit_per_list_or_per_batch() {
        let counts = [3, 0, 2000, 7];
        let mut each = Vec::new();
        partition(counts, 1, &mut each);
        assert_eq!(each, [0..1, 1..2, 2..3, 3..4]);
        let mut whole = Vec::new();
        partition(counts, usize::MAX, &mut whole);
        assert_eq!(whole, [0..4]);
    }

    /// Units are appended, so the units of each batch, partitioned in
    /// turn, follow the last batch's, and none crosses from one to the
    /// next.
    #[test]
    fn each_batch_appends_its_units() {
        let mut all = Vec::new();
        partition([4, 4], 10, &mut all);
        partition([4, 4], 10, &mut all);
        assert_eq!(all, [0..2, 0..2]);
    }
}
