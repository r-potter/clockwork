//! Deferred destruction, keyed by epochs (design 0002, "Deferred
//! destruction", R10).
//!
//! A destroyed object can't be freed straight away: work already submitted
//! may still use it, and, with translation on several threads, so may a
//! command buffer translated before the destruction and submitted after
//! it. So destruction retires the object, and it's freed later, once
//! nothing that could use it is outstanding.
//!
//! Time is counted in epochs. The queue has an epoch counter, which each
//! destruction advances. A command buffer records the epoch it was
//! translated in, and a retired object the epoch it was destroyed in. A
//! command buffer is outstanding from translation until its submission
//! completes, or until it's dropped unsubmitted. A retired object is freed
//! once no outstanding command buffer was translated in an epoch at or
//! before the one it was destroyed in.
//!
//! [`DeferredQueue`] is pure logic over epochs and timeline values, so it's
//! tested with fakes. The backend wraps it, and frees what it returns.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{mem, thread};

use crate::{GpuTimepoint, QueueId};

/// A command buffer being translated, or translated and not yet submitted,
/// in the epoch it records. It must be given back to the queue that issued
/// it, as [`submitted`](DeferredQueue::submitted) or
/// [`dropped`](DeferredQueue::dropped).
#[derive(Debug, PartialEq, Eq)]
#[must_use = "an outstanding command buffer holds back destruction until it's given back"]
pub(crate) struct Translation {
    /// The [`DeferredQueue::id`] of the queue that issued it.
    queue: u64,
    /// The epoch it was translated in.
    epoch: u64,
}

impl Translation {
    /// The epoch it was translated in, given back to the queue `queue`,
    /// which consumes it without the check its drop makes.
    ///
    /// # Panics
    ///
    /// Panics if another queue issued it, whose epochs mean nothing here.
    fn give_back(self, queue: u64) -> u64 {
        assert!(
            self.queue == queue,
            "a translation is given back to the queue that issued it"
        );
        let epoch = self.epoch;
        mem::forget(self);
        epoch
    }
}

impl Drop for Translation {
    /// Panics in debug builds, since a translation that isn't given back
    /// holds back every later destruction until the device is dropped. Its
    /// owner gives it back when dropped, so one dropped while unwinding from
    /// a panic isn't reported again.
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking(),
            "a translation from epoch {} was dropped without being given back as submitted or \
             dropped",
            self.epoch,
        );
    }
}

/// The source of [`DeferredQueue::id`]s.
static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(0);

/// Retired objects, and what may still use them.
#[derive(Debug)]
pub(crate) struct DeferredQueue<T> {
    /// What tells this queue's translations from another's, such as another
    /// device's.
    id: u64,
    /// The current epoch, which each retirement ends.
    epoch: u64,
    /// How many command buffers translated in each epoch are unsubmitted.
    unsubmitted: BTreeMap<u64, usize>,
    /// The submissions that may not have completed: the oldest epoch any of
    /// their command buffers was translated in, and the timepoint each
    /// reaches. A submission's command buffers share an entry.
    pending: Vec<(u64, GpuTimepoint)>,
    /// Retired objects, with the epoch each was destroyed in, oldest first.
    retired: VecDeque<(u64, T)>,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the backend retires objects through it in the next PR"
    )
)]
impl<T> DeferredQueue<T> {
    /// An empty queue, at the first epoch.
    pub(crate) fn new() -> Self {
        Self {
            id: NEXT_QUEUE_ID.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
            unsubmitted: BTreeMap::new(),
            pending: Vec::new(),
            retired: VecDeque::new(),
        }
    }

    /// Starts a translation in the current epoch. Call it before looking up
    /// any handle the list names.
    pub(crate) fn translating(&mut self) -> Translation {
        *self.unsubmitted.entry(self.epoch).or_default() += 1;
        Translation {
            queue: self.id,
            epoch: self.epoch,
        }
    }

    /// Gives back a command buffer dropped unsubmitted, which uses nothing
    /// from then on.
    ///
    /// # Panics
    ///
    /// Panics if another queue issued `translation`.
    pub(crate) fn dropped(&mut self, translation: Translation) {
        self.forget_unsubmitted(translation.give_back(self.id));
    }

    /// Gives back a command buffer submitted in the submission that reaches
    /// `timepoint`, which uses what it was translated with until then.
    ///
    /// # Panics
    ///
    /// Panics if another queue issued `translation`.
    pub(crate) fn submitted(&mut self, translation: Translation, timepoint: GpuTimepoint) {
        let epoch = translation.give_back(self.id);
        self.forget_unsubmitted(epoch);
        // A submission's command buffers are given back one after another,
        // and only the oldest epoch among them matters.
        match self.pending.last_mut() {
            Some((oldest, last)) if *last == timepoint => *oldest = (*oldest).min(epoch),
            _ => self.pending.push((epoch, timepoint)),
        }
    }

    /// Retires `object`, destroyed in the current epoch, which then ends.
    pub(crate) fn retire(&mut self, object: T) {
        self.retired.push_back((self.epoch, object));
        self.epoch += 1;
    }

    /// Returns the retired objects that nothing outstanding can use, oldest
    /// first, for the caller to free. `reached` gives the value each queue's
    /// timeline has reached, which the caller reads before locking the
    /// queue, so translations don't wait on the driver. A queue it doesn't
    /// list counts as having reached nothing.
    pub(crate) fn collect(&mut self, reached: &[(QueueId, u64)]) -> Vec<T> {
        self.pending.retain(|&(_, timepoint)| {
            let value = reached
                .iter()
                .find(|&&(queue, _)| queue == timepoint.queue())
                .map_or(0, |&(_, value)| value);
            timepoint.value() > value
        });
        let oldest_outstanding = self
            .unsubmitted
            .keys()
            .next()
            .copied()
            .into_iter()
            .chain(self.pending.iter().map(|&(epoch, _)| epoch))
            .min();
        let mut freed = Vec::new();
        while let Some(&(destroyed, _)) = self.retired.front() {
            // An outstanding command buffer translated in or before the
            // epoch the object was destroyed in may use it.
            if oldest_outstanding.is_some_and(|oldest| oldest <= destroyed) {
                break;
            }
            let (_, object) = self.retired.pop_front().expect("the front was just seen");
            freed.push(object);
        }
        freed
    }

    /// Returns every retired object, for the caller to free once the device
    /// is idle, when nothing can use them.
    pub(crate) fn drain(&mut self) -> Vec<T> {
        self.retired.drain(..).map(|(_, object)| object).collect()
    }

    /// Counts one fewer unsubmitted command buffer from `epoch`.
    ///
    /// # Panics
    ///
    /// Panics if none is unsubmitted, which means the queue's counts are
    /// wrong.
    fn forget_unsubmitted(&mut self, epoch: u64) {
        let count = self
            .unsubmitted
            .get_mut(&epoch)
            .expect("an unsubmitted translation is counted in its epoch");
        *count -= 1;
        if *count == 0 {
            self.unsubmitted.remove(&epoch);
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 unit and property tests of the deferred-destruction queue against
    //! fake timelines (design 0002, R10).

    use std::collections::HashMap;

    use proptest::prelude::*;

    use super::*;

    /// A graphics queue.
    const GRAPHICS: QueueId = QueueId::new(0);

    /// A transfer queue.
    const TRANSFER: QueueId = QueueId::new(2);

    /// The timepoint `value` on the graphics queue's timeline.
    fn graphics(value: u64) -> GpuTimepoint {
        GpuTimepoint::new(GRAPHICS, value)
    }

    /// Timelines that have reached `value` on the graphics and transfer
    /// queues.
    fn at(value: u64) -> [(QueueId, u64); 2] {
        [(GRAPHICS, value), (TRANSFER, value)]
    }

    /// With nothing outstanding, a retired object is freed at the next
    /// collection, and only once.
    #[test]
    fn objects_nothing_uses_are_freed() {
        let mut queue = DeferredQueue::new();
        queue.retire("buffer");
        assert_eq!(queue.collect(&at(0)), ["buffer"]);
        assert_eq!(queue.collect(&at(0)), [] as [&str; 0]);
    }

    /// An object is kept while a submission translated before its
    /// destruction is pending, and freed once that submission completes.
    #[test]
    fn objects_wait_for_submissions_translated_before() {
        let mut queue = DeferredQueue::new();
        let translation = queue.translating();
        queue.submitted(translation, graphics(1));
        queue.retire("buffer");
        assert_eq!(queue.collect(&at(0)), [] as [&str; 0]);
        assert_eq!(queue.collect(&at(1)), ["buffer"]);
    }

    /// The case epochs exist for: a list translated before a destruction
    /// and submitted after it keeps the object until its submission
    /// completes.
    #[test]
    fn lists_translated_before_and_submitted_after_hold_objects() {
        let mut queue = DeferredQueue::new();
        let translation = queue.translating();
        queue.retire("buffer");
        assert_eq!(queue.collect(&at(0)), [] as [&str; 0]);
        queue.submitted(translation, graphics(1));
        assert_eq!(queue.collect(&at(0)), [] as [&str; 0]);
        assert_eq!(queue.collect(&at(1)), ["buffer"]);
    }

    /// A list dropped unsubmitted stops holding objects back at once.
    #[test]
    fn dropped_lists_hold_nothing() {
        let mut queue = DeferredQueue::new();
        let translation = queue.translating();
        queue.retire("buffer");
        assert_eq!(queue.collect(&at(0)), [] as [&str; 0]);
        queue.dropped(translation);
        assert_eq!(queue.collect(&at(0)), ["buffer"]);
    }

    /// A list translated after a destruction can't use the object, so it
    /// doesn't hold it back, though it's still outstanding.
    #[test]
    fn lists_translated_after_hold_nothing_older() {
        let mut queue = DeferredQueue::new();
        queue.retire("buffer");
        let translation = queue.translating();
        assert_eq!(queue.collect(&at(0)), ["buffer"]);
        queue.dropped(translation);
    }

    /// Objects are freed oldest first, each once nothing older than its
    /// destruction is outstanding, and a queue the caller doesn't list
    /// counts as having reached nothing.
    #[test]
    fn objects_are_freed_in_order_as_work_completes() {
        let mut queue = DeferredQueue::new();
        let first = queue.translating();
        queue.submitted(first, graphics(1));
        queue.retire("first");
        let second = queue.translating();
        queue.submitted(second, GpuTimepoint::new(TRANSFER, 1));
        queue.retire("second");

        assert_eq!(queue.collect(&[(TRANSFER, 1)]), [] as [&str; 0]);
        assert_eq!(queue.collect(&[(GRAPHICS, 1)]), ["first", "second"]);
    }

    /// A submission's command buffers share one pending entry, which keeps
    /// the oldest epoch among them.
    #[test]
    fn submissions_keep_their_oldest_epoch() {
        let mut queue = DeferredQueue::new();
        let older = queue.translating();
        queue.retire("buffer");
        let newer = queue.translating();
        queue.submitted(newer, graphics(1));
        queue.submitted(older, graphics(1));
        assert_eq!(queue.pending, [(0, graphics(1))]);
        assert_eq!(queue.collect(&at(0)), [] as [&str; 0]);
        assert_eq!(queue.collect(&at(1)), ["buffer"]);
    }

    /// A translation given back to a queue that didn't issue it panics,
    /// rather than changing that queue's counts.
    #[test]
    #[should_panic(expected = "a translation is given back to the queue that issued it")]
    fn translations_belong_to_their_queue() {
        let mut issuer = DeferredQueue::<&str>::new();
        let mut other = DeferredQueue::<&str>::new();
        let kept = other.translating();
        let translation = issuer.translating();
        other.dropped(translation);
        other.dropped(kept);
    }

    /// A translation dropped without being given back panics in debug
    /// builds, rather than holding back destruction for good.
    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "a translation from epoch 0 was dropped without being given back")
    )]
    fn lost_translations_panic() {
        let mut queue = DeferredQueue::<&str>::new();
        drop(queue.translating());
    }

    /// Draining returns every retired object, whatever is outstanding.
    #[test]
    fn draining_returns_everything() {
        let mut queue = DeferredQueue::new();
        let translation = queue.translating();
        queue.retire("first");
        queue.retire("second");
        assert_eq!(queue.drain(), ["first", "second"]);
        queue.dropped(translation);
        assert_eq!(queue.collect(&at(0)), [] as [&str; 0]);
    }

    /// One step of [`no_object_is_freed_while_it_could_be_used`].
    #[derive(Debug, Clone)]
    enum Step {
        /// Translate a list.
        Translate,
        /// Submit the unsubmitted list at this index, modulo their number,
        /// to this queue.
        Submit(usize, QueueId),
        /// Drop the unsubmitted list at this index, modulo their number.
        Drop(usize),
        /// Complete every submission so far on this queue.
        Complete(QueueId),
        /// Destroy an object.
        Retire,
        /// Free what can be freed.
        Collect,
        /// Free everything retired, as when the device is idle.
        Drain,
    }

    /// A strategy for steps, weighted towards translating, submitting and
    /// retiring, which build up work for the others.
    fn step() -> impl Strategy<Value = Step> {
        let queue = (0..3_u32).prop_map(QueueId::new);
        prop_oneof![
            3 => Just(Step::Translate),
            3 => (any::<usize>(), queue.clone())
                .prop_map(|(index, queue)| Step::Submit(index, queue)),
            1 => any::<usize>().prop_map(Step::Drop),
            2 => queue.prop_map(Step::Complete),
            3 => Just(Step::Retire),
            2 => Just(Step::Collect),
            1 => Just(Step::Drain),
        ]
    }

    proptest! {
        /// Random interleavings of translations, submissions, completions,
        /// drops, destructions and collections, against a model that
        /// records which objects each list could use: every object alive
        /// when it was translated. No object is freed while a list that
        /// could use it is outstanding, and once everything completes,
        /// every object is freed.
        #[test]
        fn no_object_is_freed_while_it_could_be_used(steps in prop::collection::vec(step(), 1..200)) {
            let mut queue = DeferredQueue::new();
            // Each unsubmitted list: its translation, and the objects it
            // could use.
            let mut unsubmitted: Vec<(Translation, Vec<usize>)> = Vec::new();
            // Each pending submission's timepoint, and the objects its list
            // could use.
            let mut pending: Vec<(GpuTimepoint, Vec<usize>)> = Vec::new();
            let mut submitted: HashMap<QueueId, u64> = HashMap::new();
            let mut reached: HashMap<QueueId, u64> = HashMap::new();
            let mut alive: Vec<usize> = Vec::new();
            let mut retired = 0;
            let mut freed: Vec<usize> = Vec::new();
            let mut next_object = 0;
            // Objects exist from the start of a test until destroyed; create
            // a few up front, and one after each destruction, so lists
            // always have something to use.
            for _ in 0..3 {
                alive.push(next_object);
                next_object += 1;
            }
            for step in steps {
                match step {
                    Step::Translate => {
                        unsubmitted.push((queue.translating(), alive.clone()));
                    }
                    Step::Submit(index, id) => {
                        if unsubmitted.is_empty() {
                            continue;
                        }
                        let (translation, uses) = unsubmitted.remove(index % unsubmitted.len());
                        let value = submitted.entry(id).or_default();
                        *value += 1;
                        let timepoint = GpuTimepoint::new(id, *value);
                        queue.submitted(translation, timepoint);
                        pending.push((timepoint, uses));
                    }
                    Step::Drop(index) => {
                        if unsubmitted.is_empty() {
                            continue;
                        }
                        let (translation, _) = unsubmitted.remove(index % unsubmitted.len());
                        queue.dropped(translation);
                    }
                    Step::Complete(id) => {
                        reached.insert(id, submitted.get(&id).copied().unwrap_or(0));
                    }
                    Step::Retire => {
                        if alive.is_empty() {
                            continue;
                        }
                        let object = alive.remove(0);
                        queue.retire(object);
                        retired += 1;
                        alive.push(next_object);
                        next_object += 1;
                    }
                    Step::Collect => {
                        let now: Vec<(QueueId, u64)> = reached.iter().map(|(&id, &value)| (id, value)).collect();
                        let value = |id| reached.get(&id).copied().unwrap_or(0);
                        pending.retain(|(timepoint, _)| timepoint.value() > value(timepoint.queue()));
                        let newly_freed = queue.collect(&now);
                        for object in &newly_freed {
                            let used = unsubmitted
                                .iter()
                                .map(|(_, uses)| uses)
                                .chain(pending.iter().map(|(_, uses)| uses))
                                .any(|uses| uses.contains(object));
                            prop_assert!(!used, "object {object} was freed while a list could use it");
                        }
                        freed.extend(newly_freed);
                    }
                    Step::Drain => {
                        // The caller drains only when nothing can use what's
                        // retired, so drained objects aren't checked against
                        // the lists, but they mustn't be freed again.
                        freed.extend(queue.drain());
                    }
                }
            }
            // Once every list is given back and every submission completes,
            // everything retired is freed.
            for (translation, _) in unsubmitted {
                queue.dropped(translation);
            }
            let done: Vec<(QueueId, u64)> = submitted.iter().map(|(&id, &value)| (id, value)).collect();
            freed.extend(queue.collect(&done));
            prop_assert_eq!(freed.len(), retired, "each object is freed once");
            let mut sorted = freed.clone();
            sorted.sort_unstable();
            prop_assert_eq!(freed, sorted, "objects are freed oldest first");
        }
    }
}
