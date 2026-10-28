//! Handles, and the tables they index (design 0002, "Handles").
//!
//! Every GPU object the application sees is a `Copy` handle: an index into a
//! table, and a generation. A [`HandleTable`] is a slot map. Each slot holds
//! an object or a link in the free list, and its generation is incremented
//! when its object is removed, so a handle to a removed object is stale from
//! then on, even once the slot holds another object.
//!
//! Looking up a stale handle is a bug in the caller, and panics. The message
//! names the object the handle was for, when the table still knows it: each
//! slot remembers the label of its previous object until its next removal,
//! which covers the usual bug of using a handle just after destroying it.
//! A handle older than that is still detected, but named by its slot and
//! generation.
//!
//! This is safe code, so it's checked with property tests against a model
//! rather than with `miri`.

use std::fmt;

/// An index into a [`HandleTable`], and the generation of the slot it was
/// issued for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Handle {
    /// The slot's index.
    index: u32,
    /// The slot's generation when the handle was issued.
    generation: u32,
}

/// A table of objects of one kind, indexed by [`Handle`]s.
#[derive(Debug)]
pub(crate) struct HandleTable<T> {
    /// What the table holds, such as `buffer`, for messages.
    kind: &'static str,
    /// Every slot, holding an object or free.
    slots: Vec<Slot<T>>,
    /// The most recently freed slot, which is reused first.
    free: Option<u32>,
    /// The number of objects in the table.
    len: usize,
}

/// One slot of a [`HandleTable`].
#[derive(Debug)]
struct Slot<T> {
    /// Incremented each time the slot's object is removed. A slot whose
    /// generation reaches `u32::MAX` is never reused, so generations never
    /// wrap.
    generation: u32,
    /// The slot's object, or its link in the free list.
    state: SlotState<T>,
    /// The label of the slot's previous object, issued with generation
    /// `generation - 1`, to name it when its stale handle is used.
    previous_label: Option<Box<str>>,
}

/// What a [`Slot`] holds.
#[derive(Debug)]
enum SlotState<T> {
    /// An object, and the label it was created with.
    Occupied {
        /// The object's name, in debug tools and messages.
        label: Box<str>,
        /// The object.
        value: T,
    },
    /// No object, and the next free slot.
    Free {
        /// The slot freed before this one, if it's still free.
        next: Option<u32>,
    },
}

/// Why a [`Handle`] doesn't refer to an object in a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HandleError<'a> {
    /// The handle's object was removed.
    Stale {
        /// What the table holds.
        kind: &'static str,
        /// The handle.
        handle: Handle,
        /// The object's label, if the table still knows it.
        label: Option<&'a str>,
    },
    /// The table never issued the handle.
    Invalid {
        /// What the table holds.
        kind: &'static str,
        /// The handle.
        handle: Handle,
    },
}

impl fmt::Display for HandleError<'_> {
    /// Names the object, or the slot and generation when the label isn't
    /// known.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stale {
                kind,
                label: Some(label),
                ..
            } => write!(f, "stale {kind} handle: \"{label}\" was destroyed"),
            Self::Stale {
                kind,
                handle,
                label: None,
            } => write!(
                f,
                "stale {kind} handle: the object in slot {}, generation {}, was destroyed",
                handle.index, handle.generation,
            ),
            Self::Invalid { kind, handle } => write!(
                f,
                "invalid {kind} handle: slot {}, generation {}, was never issued by this table",
                handle.index, handle.generation,
            ),
        }
    }
}

impl<T> HandleTable<T> {
    /// An empty table of objects of `kind`, such as `buffer`, which names
    /// them in messages.
    pub(crate) fn new(kind: &'static str) -> Self {
        Self {
            kind,
            slots: Vec::new(),
            free: None,
            len: 0,
        }
    }

    /// The number of objects in the table.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "leak reporting counts objects, later in rhi-resources"
        )
    )]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Adds `value`, labelled `label`, and returns its handle.
    pub(crate) fn insert(&mut self, label: &str, value: T) -> Handle {
        let label = label.into();
        self.len += 1;
        if let Some(index) = self.free {
            let slot = &mut self.slots[index as usize];
            let SlotState::Free { next } = slot.state else {
                unreachable!("the free list holds only free slots");
            };
            self.free = next;
            slot.state = SlotState::Occupied { label, value };
            return Handle {
                index,
                generation: slot.generation,
            };
        }
        let index = u32::try_from(self.slots.len()).expect("a table has fewer than 2^32 slots");
        self.slots.push(Slot {
            generation: 0,
            state: SlotState::Occupied { label, value },
            previous_label: None,
        });
        Handle {
            index,
            generation: 0,
        }
    }

    /// The object `handle` refers to.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale or invalid, naming its object.
    pub(crate) fn get(&self, handle: Handle) -> &T {
        match self.lookup(handle) {
            Ok((_, value)) => value,
            Err(err) => panic!("{err}"),
        }
    }

    /// The label of the object `handle` refers to.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale or invalid, naming its object.
    pub(crate) fn label(&self, handle: Handle) -> &str {
        match self.lookup(handle) {
            Ok((label, _)) => label,
            Err(err) => panic!("{err}"),
        }
    }

    /// Removes the object `handle` refers to, returning its label and the
    /// object. The handle is stale from then on.
    ///
    /// # Panics
    ///
    /// Panics if the handle is stale or invalid, naming its object.
    pub(crate) fn remove(&mut self, handle: Handle) -> (Box<str>, T) {
        if let Err(err) = self.lookup(handle) {
            panic!("{err}");
        }
        let slot = &mut self.slots[handle.index as usize];
        let SlotState::Occupied { label, value } =
            std::mem::replace(&mut slot.state, SlotState::Free { next: self.free })
        else {
            unreachable!("the lookup found an object");
        };
        slot.previous_label = Some(label.clone());
        slot.generation += 1;
        if slot.generation < u32::MAX {
            self.free = Some(handle.index);
        } else {
            // Retire the slot, rather than reissue a generation.
            slot.state = SlotState::Free { next: None };
        }
        self.len -= 1;
        (label, value)
    }

    /// Removes every object, returning their labels and the objects. Every
    /// handle the table issued is stale from then on.
    pub(crate) fn drain(&mut self) -> Vec<(Box<str>, T)> {
        let handles: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| matches!(slot.state, SlotState::Occupied { .. }))
            .map(|(index, slot)| Handle {
                index: u32::try_from(index).expect("a table has fewer than 2^32 slots"),
                generation: slot.generation,
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| self.remove(handle))
            .collect()
    }

    /// The label and object `handle` refers to, or why it doesn't refer to
    /// one.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is stale, or wasn't issued by this
    /// table.
    pub(crate) fn lookup(&self, handle: Handle) -> Result<(&str, &T), HandleError<'_>> {
        let invalid = HandleError::Invalid {
            kind: self.kind,
            handle,
        };
        let Some(slot) = self.slots.get(handle.index as usize) else {
            return Err(invalid);
        };
        match &slot.state {
            SlotState::Occupied { label, value } if handle.generation == slot.generation => {
                Ok((label, value))
            }
            _ if handle.generation >= slot.generation => Err(invalid),
            _ => Err(HandleError::Stale {
                kind: self.kind,
                handle,
                label: slot
                    .previous_label
                    .as_deref()
                    .filter(|_| handle.generation + 1 == slot.generation),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 unit and property tests of the handle table (design 0002, R3).

    use std::collections::HashMap;

    use proptest::prelude::*;

    use super::*;

    /// Objects can be looked up by their handles, with their labels, until
    /// they're removed.
    #[test]
    fn objects_are_found_by_their_handles() {
        let mut table = HandleTable::new("buffer");
        let vertices = table.insert("vertices", 1);
        let indices = table.insert("indices", 2);
        assert_eq!(table.len(), 2);
        assert_eq!(*table.get(vertices), 1);
        assert_eq!(table.label(indices), "indices");
        assert_eq!(table.remove(vertices), ("vertices".into(), 1));
        assert_eq!(table.len(), 1);
        assert_eq!(*table.get(indices), 2);
    }

    /// A removed object's slot is reused, with a new generation, so the old
    /// handle stays stale.
    #[test]
    fn slots_are_reused_with_a_new_generation() {
        let mut table = HandleTable::new("buffer");
        let old = table.insert("old", ());
        table.remove(old);
        let new = table.insert("new", ());
        assert_eq!(new.index, old.index);
        assert_ne!(new, old);
        assert!(table.lookup(old).is_err());
        assert_eq!(table.label(new), "new");
    }

    /// Using a handle just after removing its object panics, naming it,
    /// even once its slot holds another object.
    #[test]
    #[should_panic(expected = "stale buffer handle: \"vertices\" was destroyed")]
    fn stale_handles_panic_with_their_label() {
        let mut table = HandleTable::new("buffer");
        let vertices = table.insert("vertices", ());
        table.remove(vertices);
        table.insert("indices", ());
        table.get(vertices);
    }

    /// A handle older than the slot's previous object is still stale, and
    /// named by its slot and generation.
    #[test]
    fn older_stale_handles_name_their_slot() {
        let mut table = HandleTable::new("texture");
        let first = table.insert("first", ());
        table.remove(first);
        let second = table.insert("second", ());
        table.remove(second);
        assert_eq!(
            table.lookup(first).unwrap_err().to_string(),
            "stale texture handle: the object in slot 0, generation 0, was destroyed",
        );
        assert_eq!(
            table.lookup(second).unwrap_err().to_string(),
            "stale texture handle: \"second\" was destroyed",
        );
    }

    /// Draining returns every object, in slot order, and leaves every
    /// handle stale and the table empty.
    #[test]
    fn draining_removes_everything() {
        let mut table = HandleTable::new("buffer");
        let first = table.insert("first", 1);
        let second = table.insert("second", 2);
        let third = table.insert("third", 3);
        table.remove(second);
        assert_eq!(table.drain(), [("first".into(), 1), ("third".into(), 3)],);
        assert_eq!(table.len(), 0);
        for handle in [first, second, third] {
            assert!(table.lookup(handle).is_err());
        }
    }

    /// Removing an object twice panics, rather than freeing its slot twice.
    #[test]
    #[should_panic(expected = "stale sampler handle: \"linear\" was destroyed")]
    fn removing_twice_panics() {
        let mut table = HandleTable::new("sampler");
        let linear = table.insert("linear", ());
        table.remove(linear);
        table.remove(linear);
    }

    /// A handle another table issued, past this table's slots or ahead of
    /// a slot's generation, is invalid.
    #[test]
    fn foreign_handles_are_invalid() {
        let mut table = HandleTable::new("buffer");
        let handle = table.insert("vertices", ());
        for foreign in [
            Handle {
                index: 1,
                generation: 0,
            },
            Handle {
                index: 0,
                generation: 1,
            },
        ] {
            assert_eq!(
                table.lookup(foreign),
                Err(HandleError::Invalid {
                    kind: "buffer",
                    handle: foreign
                }),
            );
        }
        assert!(table.lookup(handle).is_ok());
    }

    /// A slot whose generation runs out is retired rather than reused, so
    /// a generation is never issued twice.
    #[test]
    fn exhausted_slots_are_retired() {
        let mut table = HandleTable::new("buffer");
        let handle = table.insert("old", ());
        table.slots[0].generation = u32::MAX - 1;
        let handle = Handle {
            generation: u32::MAX - 1,
            ..handle
        };
        table.remove(handle);
        let new = table.insert("new", ());
        assert_eq!(new.index, 1);
        assert!(table.lookup(handle).is_err());
    }

    /// One step of [`handles_match_a_model`].
    #[derive(Debug, Clone)]
    enum Step {
        /// Insert an object.
        Insert,
        /// Remove the live object at this index into the live handles,
        /// modulo their number.
        Remove(usize),
    }

    proptest! {
        /// Random inserts and removes, checked against a `HashMap` model:
        /// every live handle finds its own object and label, every handle
        /// ever removed stays stale, and the table's length matches.
        #[test]
        fn handles_match_a_model(
            steps in prop::collection::vec(
                prop_oneof![Just(Step::Insert), any::<usize>().prop_map(Step::Remove)],
                1..200,
            ),
        ) {
            let mut table = HandleTable::new("buffer");
            let mut live: HashMap<Handle, (String, usize)> = HashMap::new();
            let mut removed: Vec<Handle> = Vec::new();
            for (number, step) in steps.into_iter().enumerate() {
                match step {
                    Step::Insert => {
                        let label = format!("object {number}");
                        let handle = table.insert(&label, number);
                        // A handle is never issued twice.
                        prop_assert!(!live.contains_key(&handle));
                        prop_assert!(!removed.contains(&handle));
                        live.insert(handle, (label, number));
                    }
                    Step::Remove(index) => {
                        let mut handles: Vec<_> = live.keys().copied().collect();
                        if handles.is_empty() {
                            continue;
                        }
                        handles.sort_by_key(|handle| (handle.index, handle.generation));
                        let handle = handles[index % handles.len()];
                        let (label, value) = live.remove(&handle).unwrap();
                        prop_assert_eq!(table.remove(handle), (label.into_boxed_str(), value));
                        removed.push(handle);
                    }
                }
                prop_assert_eq!(table.len(), live.len());
                for (&handle, (label, value)) in &live {
                    prop_assert_eq!(table.lookup(handle), Ok((label.as_str(), value)));
                }
                for &handle in &removed {
                    let is_stale = matches!(table.lookup(handle), Err(HandleError::Stale { .. }));
                    prop_assert!(is_stale);
                }
            }
        }
    }
}
