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

use crate::{GpuPoint, QueueKind};

/// A command buffer being translated, or translated and not yet submitted,
/// in the epoch it records. It must be given back to the queue that issued
/// it, as [`submitted`](DeferredQueue::submitted) or
/// [`dropped`](DeferredQueue::dropped).
#[derive(Debug, PartialEq, Eq)]
#[must_use = "an outstanding command buffer holds back destruction until it's given back"]
pub(crate) struct Translation {
    /// The epoch it was translated in.
    epoch: u64,
}

/// Retired objects, and what may still use them.
#[derive(Debug)]
pub(crate) struct DeferredQueue<T> {
    /// The current epoch, which each retirement ends.
    epoch: u64,
    /// How many command buffers translated in each epoch are unsubmitted.
    unsubmitted: BTreeMap<u64, usize>,
    /// The submitted command buffers whose submissions may not have
    /// completed: the epoch each was translated in, and the point its
    /// submission reaches.
    pending: Vec<(u64, GpuPoint)>,
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
        Translation { epoch: self.epoch }
    }

    /// Gives back a command buffer dropped unsubmitted, which uses nothing
    /// from then on.
    pub(crate) fn dropped(&mut self, translation: Translation) {
        self.forget_unsubmitted(translation.epoch);
    }

    /// Gives back a command buffer submitted in the submission that reaches
    /// `point`, which uses what it was translated with until then.
    pub(crate) fn submitted(&mut self, translation: Translation, point: GpuPoint) {
        self.forget_unsubmitted(translation.epoch);
        self.pending.push((translation.epoch, point));
    }

    /// Retires `object`, destroyed in the current epoch, which then ends.
    pub(crate) fn retire(&mut self, object: T) {
        self.retired.push_back((self.epoch, object));
        self.epoch += 1;
    }

    /// Returns the retired objects that nothing outstanding can use, oldest
    /// first, for the caller to free. `reached` gives the value each queue's
    /// timeline has reached, and is called once for each queue with a
    /// pending submission.
    pub(crate) fn collect(&mut self, mut reached: impl FnMut(QueueKind) -> u64) -> Vec<T> {
        let mut reached_by_kind: Vec<(QueueKind, u64)> = Vec::new();
        self.pending.retain(|&(_, point)| {
            let kind = point.queue();
            let value = match reached_by_kind.iter().find(|(known, _)| *known == kind) {
                Some(&(_, value)) => value,
                None => {
                    let value = reached(kind);
                    reached_by_kind.push((kind, value));
                    value
                }
            };
            point.value() > value
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
    /// Panics if none is unsubmitted, which means a [`Translation`] came
    /// from another queue.
    fn forget_unsubmitted(&mut self, epoch: u64) {
        let count = self
            .unsubmitted
            .get_mut(&epoch)
            .expect("a translation is given back to the queue that issued it");
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

    /// The point `value` on the graphics queue's timeline.
    fn graphics(value: u64) -> GpuPoint {
        GpuPoint::new(QueueKind::Graphics, value)
    }

    /// A timeline that has reached `value` on every queue.
    fn at(value: u64) -> impl FnMut(QueueKind) -> u64 {
        move |_| value
    }

    /// With nothing outstanding, a retired object is freed at the next
    /// collection, and only once.
    #[test]
    fn objects_nothing_uses_are_freed() {
        let mut queue = DeferredQueue::new();
        queue.retire("buffer");
        assert_eq!(queue.collect(at(0)), ["buffer"]);
        assert_eq!(queue.collect(at(0)), [] as [&str; 0]);
    }

    /// An object is kept while a submission translated before its
    /// destruction is pending, and freed once that submission completes.
    #[test]
    fn objects_wait_for_submissions_translated_before() {
        let mut queue = DeferredQueue::new();
        let translation = queue.translating();
        queue.submitted(translation, graphics(1));
        queue.retire("buffer");
        assert_eq!(queue.collect(at(0)), [] as [&str; 0]);
        assert_eq!(queue.collect(at(1)), ["buffer"]);
    }

    /// The case epochs exist for: a list translated before a destruction
    /// and submitted after it keeps the object until its submission
    /// completes.
    #[test]
    fn lists_translated_before_and_submitted_after_hold_objects() {
        let mut queue = DeferredQueue::new();
        let translation = queue.translating();
        queue.retire("buffer");
        assert_eq!(queue.collect(at(0)), [] as [&str; 0]);
        queue.submitted(translation, graphics(1));
        assert_eq!(queue.collect(at(0)), [] as [&str; 0]);
        assert_eq!(queue.collect(at(1)), ["buffer"]);
    }

    /// A list dropped unsubmitted stops holding objects back at once.
    #[test]
    fn dropped_lists_hold_nothing() {
        let mut queue = DeferredQueue::new();
        let translation = queue.translating();
        queue.retire("buffer");
        assert_eq!(queue.collect(at(0)), [] as [&str; 0]);
        queue.dropped(translation);
        assert_eq!(queue.collect(at(0)), ["buffer"]);
    }

    /// A list translated after a destruction can't use the object, so it
    /// doesn't hold it back, though it's still outstanding.
    #[test]
    fn lists_translated_after_hold_nothing_older() {
        let mut queue = DeferredQueue::new();
        queue.retire("buffer");
        let translation = queue.translating();
        assert_eq!(queue.collect(at(0)), ["buffer"]);
        queue.dropped(translation);
    }

    /// Objects are freed oldest first, each once nothing older than its
    /// destruction is outstanding, and each queue's timeline is read once.
    #[test]
    fn objects_are_freed_in_order_as_work_completes() {
        let mut queue = DeferredQueue::new();
        let first = queue.translating();
        queue.submitted(first, graphics(1));
        queue.retire("first");
        let second = queue.translating();
        queue.submitted(second, GpuPoint::new(QueueKind::Transfer, 1));
        queue.retire("second");

        let mut reads = Vec::new();
        let freed = queue.collect(|kind| {
            reads.push(kind);
            match kind {
                QueueKind::Graphics => 0,
                _ => 1,
            }
        });
        assert_eq!(freed, [] as [&str; 0]);
        reads.sort_by_key(|kind| *kind as u8);
        assert_eq!(reads, [QueueKind::Graphics, QueueKind::Transfer]);
        assert_eq!(queue.collect(at(1)), ["first", "second"]);
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
        assert_eq!(queue.collect(at(0)), [] as [&str; 0]);
    }

    /// One step of [`no_object_is_freed_while_it_could_be_used`].
    #[derive(Debug, Clone)]
    enum Step {
        /// Translate a list.
        Translate,
        /// Submit the unsubmitted list at this index, modulo their number,
        /// to the queue of this kind.
        Submit(usize, QueueKind),
        /// Drop the unsubmitted list at this index, modulo their number.
        Drop(usize),
        /// Complete every submission so far on the queue of this kind.
        Complete(QueueKind),
        /// Destroy an object.
        Retire,
        /// Free what can be freed.
        Collect,
    }

    /// A strategy for steps, weighted towards translating, submitting and
    /// retiring, which build up work for the others.
    fn step() -> impl Strategy<Value = Step> {
        let kind = prop_oneof![
            Just(QueueKind::Graphics),
            Just(QueueKind::Compute),
            Just(QueueKind::Transfer),
        ];
        prop_oneof![
            3 => Just(Step::Translate),
            3 => (any::<usize>(), kind.clone()).prop_map(|(index, kind)| Step::Submit(index, kind)),
            1 => any::<usize>().prop_map(Step::Drop),
            2 => kind.prop_map(Step::Complete),
            3 => Just(Step::Retire),
            2 => Just(Step::Collect),
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
            // Each pending submission's point, and the objects its list
            // could use.
            let mut pending: Vec<(GpuPoint, Vec<usize>)> = Vec::new();
            let mut submitted: HashMap<QueueKind, u64> = HashMap::new();
            let mut reached: HashMap<QueueKind, u64> = HashMap::new();
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
                    Step::Submit(index, kind) => {
                        if unsubmitted.is_empty() {
                            continue;
                        }
                        let (translation, uses) = unsubmitted.remove(index % unsubmitted.len());
                        let value = submitted.entry(kind).or_default();
                        *value += 1;
                        let point = GpuPoint::new(kind, *value);
                        queue.submitted(translation, point);
                        pending.push((point, uses));
                    }
                    Step::Drop(index) => {
                        if unsubmitted.is_empty() {
                            continue;
                        }
                        let (translation, _) = unsubmitted.remove(index % unsubmitted.len());
                        queue.dropped(translation);
                    }
                    Step::Complete(kind) => {
                        reached.insert(kind, submitted.get(&kind).copied().unwrap_or(0));
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
                        let now = |kind| reached.get(&kind).copied().unwrap_or(0);
                        pending.retain(|(point, _)| point.value() > now(point.queue()));
                        let newly_freed = queue.collect(now);
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
                }
            }
            // Once every list is given back and every submission completes,
            // everything retired is freed.
            for (translation, _) in unsubmitted {
                queue.dropped(translation);
            }
            freed.extend(queue.collect(|kind| submitted.get(&kind).copied().unwrap_or(0)));
            prop_assert_eq!(freed.len(), retired);
            let mut sorted = freed.clone();
            sorted.sort_unstable();
            prop_assert_eq!(freed, sorted, "objects are freed oldest first");
        }
    }
}
