//! Command lists: plain data, recorded with no device, then translated for a
//! queue (design 0002, "Command lists and translation").
//!
//! A [`CommandList`] holds its commands as a `Vec` of the private
//! [`Command`] enum, with variable-length data, such as label text, in an
//! arena beside it, so that each command stays small. Recording checks the
//! rules it can see on its own, and panics on misuse, which is a bug in the
//! caller. Rules that need the backend's state are checked when the list is
//! translated.
//!
//! Only this crate reads a list's commands (R18). Tests read them through
//! the test-only `CommandList::recorded`, which resolves the arena, so that the
//! recording model can be changed without rewriting them (design 0002,
//! "Keeping the recording model reversible").

use std::ops::Range;

use crate::GpuAddress;

/// The alignment, in bytes, of a fill's address and size, as Vulkan
/// requires of fills.
const FILL_ALIGNMENT: u64 = 4;

/// A list of GPU commands, recorded on any thread with no device, then
/// translated by a queue.
///
/// Buffers are named by their [`GpuAddress`]es, as everywhere in the RHI.
///
/// A list isn't `Clone`, since each is translated once, rather than
/// replayed (design 0002, "Keeping the recording model reversible").
#[derive(Debug)]
pub struct CommandList {
    /// The list's name, in messages and debug tools.
    label: String,
    /// The commands, in recording order.
    commands: Vec<Command>,
    /// The text of every debug label, which commands refer to by range.
    text: String,
    /// How many debug labels are open.
    open_labels: u32,
}

/// A recorded command. Variable-length data is in the list's arena.
#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "queues translate commands in the next PR")
)]
pub(crate) enum Command {
    /// Fills `size` bytes at `destination` with copies of `value`.
    FillBuffer {
        /// The first byte to fill.
        destination: GpuAddress,
        /// How many bytes to fill.
        size: u64,
        /// The value repeated across the range, in the device's byte order.
        value: u32,
    },
    /// Copies `size` bytes from `source` to `destination`.
    CopyBuffer {
        /// The first byte to copy.
        source: GpuAddress,
        /// Where the first byte is copied to.
        destination: GpuAddress,
        /// How many bytes to copy.
        size: u64,
    },
    /// Opens a debug label, whose text is `text` in the list's arena.
    BeginLabel {
        /// The label's text, as a range of the arena.
        text: Range<usize>,
    },
    /// Closes the innermost open debug label.
    EndLabel,
}

impl CommandList {
    /// An empty list called `label`.
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_owned(),
            commands: Vec::new(),
            text: String::new(),
            open_labels: 0,
        }
    }

    /// The list's name, in messages and debug tools.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Fills `size` bytes at `destination` with copies of `value`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero, if `destination` or `size` isn't a multiple
    /// of 4, or if the range passes the end of the address space.
    pub fn fill_buffer(&mut self, destination: GpuAddress, size: u64, value: u32) {
        self.check_range("fills", destination, size);
        assert!(
            destination.0.is_multiple_of(FILL_ALIGNMENT) && size.is_multiple_of(FILL_ALIGNMENT),
            "command list \"{}\" fills {size} bytes at {:#x}, but a fill's address and size \
             must be multiples of {FILL_ALIGNMENT}",
            self.label,
            destination.0,
        );
        self.commands.push(Command::FillBuffer {
            destination,
            size,
            value,
        });
    }

    /// Copies `size` bytes from `source` to `destination`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero, if either range passes the end of the
    /// address space, or if the ranges overlap.
    pub fn copy_buffer(&mut self, source: GpuAddress, destination: GpuAddress, size: u64) {
        self.check_range("copies from", source, size);
        self.check_range("copies to", destination, size);
        assert!(
            source.0 >= destination.0 + size || destination.0 >= source.0 + size,
            "command list \"{}\" copies {size} bytes from {:#x} to {:#x}, but a copy's \
             source and destination mustn't overlap",
            self.label,
            source.0,
            destination.0,
        );
        self.commands.push(Command::CopyBuffer {
            source,
            destination,
            size,
        });
    }

    /// Opens a debug label called `text`, which capture tools show around
    /// the commands recorded until the matching [`end_label`](Self::end_label).
    /// Labels nest.
    pub fn begin_label(&mut self, text: &str) {
        let start = self.text.len();
        self.text.push_str(text);
        self.commands.push(Command::BeginLabel {
            text: start..self.text.len(),
        });
        self.open_labels += 1;
    }

    /// Closes the innermost open debug label.
    ///
    /// # Panics
    ///
    /// Panics if no label is open.
    pub fn end_label(&mut self) {
        assert!(
            self.open_labels > 0,
            "command list \"{}\" ends a debug label, but none is open",
            self.label,
        );
        self.open_labels -= 1;
        self.commands.push(Command::EndLabel);
    }

    /// Checks that `size` bytes at `address` are a range a command can use:
    /// at least one byte, and not past the end of the address space. `verb`
    /// says what the command does with it, for the message.
    ///
    /// # Panics
    ///
    /// Panics if the range is empty or passes the end of the address space.
    fn check_range(&self, verb: &str, address: GpuAddress, size: u64) {
        assert!(
            size > 0,
            "command list \"{}\" {verb} 0 bytes at {:#x}, but a range must have at least one byte",
            self.label,
            address.0,
        );
        assert!(
            address.0.checked_add(size).is_some(),
            "command list \"{}\" {verb} {size} bytes at {:#x}, past the end of the address space",
            self.label,
            address.0,
        );
    }
}

/// A recorded command as tests see it, with its arena data resolved.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordedCommand<'a> {
    /// See [`CommandList::fill_buffer`].
    FillBuffer {
        /// The first byte to fill.
        destination: GpuAddress,
        /// How many bytes to fill.
        size: u64,
        /// The value repeated across the range.
        value: u32,
    },
    /// See [`CommandList::copy_buffer`].
    CopyBuffer {
        /// The first byte to copy.
        source: GpuAddress,
        /// Where the first byte is copied to.
        destination: GpuAddress,
        /// How many bytes to copy.
        size: u64,
    },
    /// See [`CommandList::begin_label`].
    BeginLabel(&'a str),
    /// See [`CommandList::end_label`].
    EndLabel,
}

#[cfg(test)]
impl CommandList {
    /// The recorded commands, in order, for tests to inspect.
    pub(crate) fn recorded(&self) -> Vec<RecordedCommand<'_>> {
        self.commands
            .iter()
            .map(|command| match *command {
                Command::FillBuffer {
                    destination,
                    size,
                    value,
                } => RecordedCommand::FillBuffer {
                    destination,
                    size,
                    value,
                },
                Command::CopyBuffer {
                    source,
                    destination,
                    size,
                } => RecordedCommand::CopyBuffer {
                    source,
                    destination,
                    size,
                },
                Command::BeginLabel { ref text } => {
                    RecordedCommand::BeginLabel(&self.text[text.clone()])
                }
                Command::EndLabel => RecordedCommand::EndLabel,
            })
            .collect()
    }
}

// A list is plain data, recorded on one thread at a time and sent to
// another to translate (design 0002, "Threading", R15). Commands stay small,
// so a list of thousands stays compact, with variable-length data in the
// arena instead.
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CommandList>();
    assert!(size_of::<Command>() <= 64);
};

#[cfg(test)]
mod tests {
    //! T0 tests of recording commands and its checks (design 0002, R7).

    use super::*;

    /// Commands are recorded in order, with their values, and labels keep
    /// their text, including when they nest.
    #[test]
    fn commands_are_recorded_in_order() {
        let mut list = CommandList::new("upload");
        list.begin_label("outer");
        list.fill_buffer(GpuAddress(0x1000), 256, 0xC10C_C0DE);
        list.begin_label("inner");
        list.copy_buffer(GpuAddress(0x1000), GpuAddress(0x2000), 256);
        list.end_label();
        list.end_label();
        assert_eq!(list.label(), "upload");
        assert_eq!(
            list.recorded(),
            [
                RecordedCommand::BeginLabel("outer"),
                RecordedCommand::FillBuffer {
                    destination: GpuAddress(0x1000),
                    size: 256,
                    value: 0xC10C_C0DE,
                },
                RecordedCommand::BeginLabel("inner"),
                RecordedCommand::CopyBuffer {
                    source: GpuAddress(0x1000),
                    destination: GpuAddress(0x2000),
                    size: 256,
                },
                RecordedCommand::EndLabel,
                RecordedCommand::EndLabel,
            ],
        );
    }

    /// A new list is empty.
    #[test]
    fn new_lists_are_empty() {
        assert_eq!(CommandList::new("empty").recorded(), []);
    }

    /// Copies whose ranges touch without overlapping are allowed, in either
    /// direction.
    #[test]
    fn adjacent_copies_are_allowed() {
        let mut list = CommandList::new("adjacent");
        list.copy_buffer(GpuAddress(0x1000), GpuAddress(0x1100), 0x100);
        list.copy_buffer(GpuAddress(0x1100), GpuAddress(0x1000), 0x100);
        assert_eq!(list.recorded().len(), 2);
    }

    /// A range may end exactly at the end of the address space.
    #[test]
    fn ranges_may_reach_the_end_of_the_address_space() {
        let mut list = CommandList::new("top");
        list.fill_buffer(GpuAddress(u64::MAX - 7), 4, 0);
        assert_eq!(list.recorded().len(), 1);
    }

    /// Ending a label with none open panics, naming the list.
    #[test]
    #[should_panic(expected = "command list \"unbalanced\" ends a debug label, but none is open")]
    fn ending_an_unopened_label_panics() {
        let mut list = CommandList::new("unbalanced");
        list.begin_label("one");
        list.end_label();
        list.end_label();
    }

    /// Filling no bytes panics, naming the list.
    #[test]
    #[should_panic(
        expected = "command list \"empty fill\" fills 0 bytes at 0x1000, but a range must have at \
                    least one byte"
    )]
    fn empty_fills_panic() {
        CommandList::new("empty fill").fill_buffer(GpuAddress(0x1000), 0, 0);
    }

    /// A fill at an address that isn't a multiple of 4 panics.
    #[test]
    #[should_panic(
        expected = "command list \"unaligned\" fills 8 bytes at 0x1002, but a fill's address and \
                    size must be multiples of 4"
    )]
    fn unaligned_fill_addresses_panic() {
        CommandList::new("unaligned").fill_buffer(GpuAddress(0x1002), 8, 0);
    }

    /// A fill whose size isn't a multiple of 4 panics.
    #[test]
    #[should_panic(
        expected = "command list \"unaligned\" fills 6 bytes at 0x1000, but a fill's address and \
                    size must be multiples of 4"
    )]
    fn unaligned_fill_sizes_panic() {
        CommandList::new("unaligned").fill_buffer(GpuAddress(0x1000), 6, 0);
    }

    /// Copying no bytes panics, naming the list.
    #[test]
    #[should_panic(
        expected = "command list \"empty copy\" copies from 0 bytes at 0x1000, but a range must \
                    have at least one byte"
    )]
    fn empty_copies_panic() {
        CommandList::new("empty copy").copy_buffer(GpuAddress(0x1000), GpuAddress(0x2000), 0);
    }

    /// A copy whose destination passes the end of the address space
    /// panics.
    #[test]
    #[should_panic(
        expected = "command list \"wrapped\" copies to 16 bytes at 0xfffffffffffffff8, past the \
                    end of the address space"
    )]
    fn copies_past_the_end_of_the_address_space_panic() {
        CommandList::new("wrapped").copy_buffer(GpuAddress(0x1000), GpuAddress(u64::MAX - 7), 16);
    }

    /// A copy whose source and destination overlap panics, whichever comes
    /// first.
    #[test]
    #[should_panic(
        expected = "command list \"overlap\" copies 256 bytes from 0x1080 to 0x1000, but a copy's \
                    source and destination mustn't overlap"
    )]
    fn overlapping_copies_panic() {
        CommandList::new("overlap").copy_buffer(GpuAddress(0x1080), GpuAddress(0x1000), 256);
    }
}
