//! Command lists: plain data, recorded with no device, then translated for a
//! queue (design 0002, "Command lists and translation").
//!
//! A [`CommandList`] holds its commands as a `Vec` of the private
//! [`Command`] enum, with larger or variable-length data, such as debug
//! text, barriers, texture copies and push data, in arenas beside it, so
//! that each command stays small. Recording checks the rules it can see on its own,
//! and panics on misuse, which is a bug in the caller. Rules that need the
//! backend's state, such as a texture copy's checks, are checked when the
//! list is translated.
//!
//! Only this crate reads a list's commands (R18). Tests read them through
//! the test-only `CommandList::recorded`, which resolves the arena, so that the
//! recording model can be changed without rewriting them (design 0002,
//! "Keeping the recording model reversible").

use std::ops::Range;

use crate::{Barrier, GpuAddress, MAX_PUSH_DATA_SIZE, Program, TextureCopy, TextureDataLayout};

/// The alignment, in bytes, of a fill's address and size, as Vulkan
/// requires of fills.
const FILL_ALIGNMENT: u64 = 4;

/// The alignment, in bytes, of a push data block's offset and size, as
/// Vulkan requires of push data.
const PUSH_DATA_ALIGNMENT: u32 = 4;

/// The size, in bytes, of an indirect dispatch's arguments: three `u32`
/// workgroup counts, as Vulkan's `VkDispatchIndirectCommand` lays them out.
pub(crate) const DISPATCH_ARGUMENTS_SIZE: u64 = 12;

/// The alignment, in bytes, of indirect arguments' address, as Vulkan
/// requires of indirect commands.
const INDIRECT_ALIGNMENT: u64 = 4;

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
    /// The text of every debug region and label, and every timing scope's
    /// label, which commands refer to by range.
    text: String,
    /// Every barrier, which commands refer to by range.
    barriers: Vec<Barrier>,
    /// Every copy between a buffer and a texture, which commands refer to
    /// by index.
    texture_copies: Vec<BufferTextureCopy>,
    /// Every push data block, which commands refer to by range. Each
    /// block's size is a multiple of 4, so each starts at a multiple of 4.
    push_data: Vec<u8>,
    /// Whether a program has been bound, so that dispatches can run.
    program_bound: bool,
    /// How many debug regions are open.
    open_regions: u32,
    /// How many timing scopes are open.
    open_timings: u32,
}

/// A recorded command. Variable-length data is in the list's arena.
#[derive(Debug)]
pub(crate) enum Command {
    /// Orders accesses with the barriers in `barriers` in the list's arena,
    /// together.
    Barrier {
        /// The barriers, as a range of the arena.
        barriers: Range<usize>,
    },
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
    /// Copies from a buffer to a texture, as `texture_copies[copy]` in the
    /// list's arena describes.
    CopyBufferToTexture {
        /// The copy's index in the arena.
        copy: usize,
    },
    /// Copies from a texture to a buffer, as `texture_copies[copy]` in the
    /// list's arena describes.
    CopyTextureToBuffer {
        /// The copy's index in the arena.
        copy: usize,
    },
    /// Binds `program`, for later dispatches to run.
    BindProgram {
        /// The program.
        program: Program,
    },
    /// Writes `data` in the list's arena to the push data later dispatches
    /// read, at `offset`.
    PushData {
        /// Where the block goes in the push data, in bytes.
        offset: u32,
        /// The block, as a range of the arena.
        data: Range<usize>,
    },
    /// Runs the compute program that's bound, on `groups` workgroups in
    /// each dimension.
    Dispatch {
        /// The number of workgroups in x, y and z.
        groups: [u32; 3],
    },
    /// Runs the compute program that's bound, on the workgroup counts the
    /// GPU reads at `arguments` when the dispatch runs.
    DispatchIndirect {
        /// The address of the arguments.
        arguments: GpuAddress,
    },
    /// Opens a debug region, whose name is `text` in the list's arena.
    BeginDebugRegion {
        /// The region's name, as a range of the arena.
        text: Range<usize>,
    },
    /// Closes the innermost open debug region.
    EndDebugRegion,
    /// Marks a point with a debug label, whose text is `text` in the list's
    /// arena.
    InsertDebugLabel {
        /// The label's text, as a range of the arena.
        text: Range<usize>,
    },
    /// Opens a timing scope, whose label is `text` in the list's arena.
    BeginTiming {
        /// The scope's label, as a range of the arena.
        text: Range<usize>,
    },
    /// Closes the innermost open timing scope.
    EndTiming,
}

impl CommandList {
    /// An empty list called `label`.
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_owned(),
            commands: Vec::new(),
            text: String::new(),
            barriers: Vec::new(),
            texture_copies: Vec::new(),
            push_data: Vec::new(),
            program_bound: false,
            open_regions: 0,
            open_timings: 0,
        }
    }

    /// The list's name, in messages and debug tools.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The commands, in recording order, for translation.
    pub(crate) fn commands(&self) -> &[Command] {
        &self.commands
    }

    /// The debug text at `range` of the arena, for translation.
    pub(crate) fn text(&self, range: &Range<usize>) -> &str {
        &self.text[range.clone()]
    }

    /// The barriers at `range` of the arena, for translation.
    pub(crate) fn barriers(&self, range: &Range<usize>) -> &[Barrier] {
        &self.barriers[range.clone()]
    }

    /// The texture copy at `index` in the arena, for translation.
    pub(crate) fn texture_copy(&self, index: usize) -> &BufferTextureCopy {
        &self.texture_copies[index]
    }

    /// The push data block at `range` of the arena, for translation.
    pub(crate) fn push_data_block(&self, range: &Range<usize>) -> &[u8] {
        &self.push_data[range.clone()]
    }

    /// Checks that the list can be translated: that every debug region and
    /// timing scope it opened is closed.
    ///
    /// # Panics
    ///
    /// Panics, naming the list, if a region or scope is still open.
    pub(crate) fn check_complete(&self) {
        assert!(
            self.open_regions == 0,
            "command list \"{}\" is translated with {} still open",
            self.label,
            count(self.open_regions, "debug region"),
        );
        assert!(
            self.open_timings == 0,
            "command list \"{}\" is translated with {} still open",
            self.label,
            count(self.open_timings, "timing scope"),
        );
    }

    /// Orders the accesses each barrier names, together (design 0002,
    /// "Synchronization"). Nothing is recorded for no barriers.
    ///
    /// # Panics
    ///
    /// Panics if a barrier's accesses don't suit what it applies to (see
    /// [`Barrier`]).
    pub fn barrier(&mut self, barriers: &[Barrier]) {
        if barriers.is_empty() {
            return;
        }
        for barrier in barriers {
            barrier.check(&self.label);
        }
        let start = self.barriers.len();
        self.barriers.extend_from_slice(barriers);
        self.commands.push(Command::Barrier {
            barriers: start..self.barriers.len(),
        });
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

    /// Copies buffer data at `source`, laid out as `layout`, to the region
    /// of a texture `destination` names (design 0002, "Texture copies").
    /// The texture must be in the layout of [`Access::CopyWrite`].
    ///
    /// The copy is checked when the list is translated, since that needs
    /// the texture (see [`TextureCopy`]).
    ///
    /// [`Access::CopyWrite`]: crate::Access::CopyWrite
    pub fn copy_buffer_to_texture(
        &mut self,
        source: GpuAddress,
        layout: TextureDataLayout,
        destination: TextureCopy,
    ) {
        let copy = self.push_texture_copy(source, layout, destination);
        self.commands.push(Command::CopyBufferToTexture { copy });
    }

    /// Copies the region of a texture `source` names to buffer data at
    /// `destination`, laid out as `layout`. The texture must be in the
    /// layout of [`Access::CopyRead`].
    ///
    /// The copy is checked when the list is translated, since that needs
    /// the texture.
    ///
    /// [`Access::CopyRead`]: crate::Access::CopyRead
    pub fn copy_texture_to_buffer(
        &mut self,
        source: TextureCopy,
        destination: GpuAddress,
        layout: TextureDataLayout,
    ) {
        let copy = self.push_texture_copy(destination, layout, source);
        self.commands.push(Command::CopyTextureToBuffer { copy });
    }

    /// Adds a copy between buffer data at `buffer`, laid out as `layout`,
    /// and the region `texture` names to the arena, and returns its index.
    fn push_texture_copy(
        &mut self,
        buffer: GpuAddress,
        layout: TextureDataLayout,
        texture: TextureCopy,
    ) -> usize {
        self.texture_copies.push(BufferTextureCopy {
            buffer,
            layout,
            texture,
        });
        self.texture_copies.len() - 1
    }

    /// Binds `program`, for later dispatches to run, until another is
    /// bound (design 0002, "Programs").
    ///
    /// Translation panics if the program is stale, or if the queue can't
    /// run it: the transfer queue runs no programs, and only the graphics
    /// queue runs a vertex and fragment program.
    pub fn bind_program(&mut self, program: Program) {
        self.commands.push(Command::BindProgram { program });
        self.program_bound = true;
    }

    /// Writes `data` to the push data later dispatches read, at `offset`
    /// bytes into it (design 0002, "Binding model"). The rest of the push
    /// data keeps what was last written there, so a list can change one
    /// field of a block, such as an index, between dispatches. Push data
    /// typically holds a single address of a structure, or a few addresses
    /// and indices. Nothing is recorded for an empty block.
    ///
    /// Translation panics if the queue runs no programs, as the transfer
    /// queue doesn't.
    ///
    /// # Panics
    ///
    /// Panics if `offset` or the size of `data` isn't a multiple of 4, or
    /// if the block passes the end of the push data, which holds
    /// [`MAX_PUSH_DATA_SIZE`] bytes.
    pub fn push_data(&mut self, offset: u32, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let size = data.len();
        assert!(
            offset.is_multiple_of(PUSH_DATA_ALIGNMENT)
                && size.is_multiple_of(PUSH_DATA_ALIGNMENT as usize),
            "command list \"{}\" pushes {size} bytes of data at offset {offset}, but a block's \
             offset and size must be multiples of {PUSH_DATA_ALIGNMENT}",
            self.label,
        );
        assert!(
            u64::from(offset) + size as u64 <= u64::from(MAX_PUSH_DATA_SIZE),
            "command list \"{}\" pushes {size} bytes of data at offset {offset}, past the end of \
             the {MAX_PUSH_DATA_SIZE} bytes of push data",
            self.label,
        );
        let start = self.push_data.len();
        self.push_data.extend_from_slice(data);
        self.commands.push(Command::PushData {
            offset,
            data: start..self.push_data.len(),
        });
    }

    /// Runs the compute program that's bound, on `groups` workgroups in x,
    /// y and z. A dimension of 0 runs nothing.
    ///
    /// Translation panics if the program that's bound isn't a compute
    /// program.
    ///
    /// # Panics
    ///
    /// Panics if no program has been bound.
    pub fn dispatch(&mut self, groups: [u32; 3]) {
        assert!(
            self.program_bound,
            "command list \"{}\" dispatches with no program bound",
            self.label,
        );
        self.commands.push(Command::Dispatch { groups });
    }

    /// Runs the compute program that's bound, on workgroup counts the GPU
    /// reads from `arguments` when the dispatch runs: three `u32`s, the
    /// counts in x, y and z, so that a dispatch's size can be decided on
    /// the GPU. A count of 0 runs nothing.
    ///
    /// The arguments must be in a buffer with [`BufferUsage::INDIRECT`],
    /// and be read with [`Access::IndirectRead`], so a barrier is needed
    /// after the GPU writes them.
    ///
    /// Translation panics if the program that's bound isn't a compute
    /// program.
    ///
    /// # Panics
    ///
    /// Panics if no program has been bound, if `arguments` isn't a multiple
    /// of 4, or if the arguments pass the end of the address space.
    ///
    /// [`BufferUsage::INDIRECT`]: crate::BufferUsage::INDIRECT
    /// [`Access::IndirectRead`]: crate::Access::IndirectRead
    pub fn dispatch_indirect(&mut self, arguments: GpuAddress) {
        assert!(
            self.program_bound,
            "command list \"{}\" dispatches with no program bound",
            self.label,
        );
        self.check_range(
            "dispatches with arguments",
            arguments,
            DISPATCH_ARGUMENTS_SIZE,
        );
        assert!(
            arguments.0.is_multiple_of(INDIRECT_ALIGNMENT),
            "command list \"{}\" dispatches with arguments at {:#x}, but indirect arguments' \
             address must be a multiple of {INDIRECT_ALIGNMENT}",
            self.label,
            arguments.0,
        );
        self.commands.push(Command::DispatchIndirect { arguments });
    }

    /// Opens a debug region called `name`, which capture tools show around
    /// the commands recorded until the matching
    /// [`end_debug_region`](Self::end_debug_region). Regions nest.
    pub fn begin_debug_region(&mut self, name: &str) {
        let text = self.push_text(name);
        self.commands.push(Command::BeginDebugRegion { text });
        self.open_regions += 1;
    }

    /// Closes the innermost open debug region.
    ///
    /// # Panics
    ///
    /// Panics if no region is open.
    pub fn end_debug_region(&mut self) {
        assert!(
            self.open_regions > 0,
            "command list \"{}\" ends a debug region, but none is open",
            self.label,
        );
        self.open_regions -= 1;
        self.commands.push(Command::EndDebugRegion);
    }

    /// Marks this point in the list with a debug label, `text`, which
    /// capture tools show between the commands recorded before and after it.
    pub fn insert_debug_label(&mut self, text: &str) {
        let text = self.push_text(text);
        self.commands.push(Command::InsertDebugLabel { text });
    }

    /// Opens a timing scope called `label`, which times, on the GPU, the
    /// commands recorded until the matching [`end_timing`](Self::end_timing)
    /// (design 0002, "Timestamps"). Scopes nest, independently of debug
    /// regions.
    pub fn begin_timing(&mut self, label: &str) {
        let text = self.push_text(label);
        self.commands.push(Command::BeginTiming { text });
        self.open_timings += 1;
    }

    /// Closes the innermost open timing scope.
    ///
    /// # Panics
    ///
    /// Panics if no scope is open.
    pub fn end_timing(&mut self) {
        assert!(
            self.open_timings > 0,
            "command list \"{}\" ends a timing scope, but none is open",
            self.label,
        );
        self.open_timings -= 1;
        self.commands.push(Command::EndTiming);
    }

    /// Appends `text` to the arena, returning its range.
    fn push_text(&mut self, text: &str) -> Range<usize> {
        let start = self.text.len();
        self.text.push_str(text);
        start..self.text.len()
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

/// `n` of `noun`, for a message: "1 debug region" or "2 debug regions".
fn count(n: u32, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

/// A copy between a buffer and a texture, in a list's arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BufferTextureCopy {
    /// The buffer data's address, which the layout's offset is from.
    pub buffer: GpuAddress,
    /// How the data is laid out in the buffer.
    pub layout: TextureDataLayout,
    /// The region of the texture.
    pub texture: TextureCopy,
}

/// A recorded command as tests see it, with its arena data resolved.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordedCommand<'a> {
    /// See [`CommandList::barrier`].
    Barrier(&'a [Barrier]),
    /// See [`CommandList::copy_buffer_to_texture`].
    CopyBufferToTexture(&'a BufferTextureCopy),
    /// See [`CommandList::copy_texture_to_buffer`].
    CopyTextureToBuffer(&'a BufferTextureCopy),
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
    /// See [`CommandList::bind_program`].
    BindProgram(Program),
    /// See [`CommandList::push_data`].
    PushData {
        /// Where the block goes in the push data, in bytes.
        offset: u32,
        /// The block.
        data: &'a [u8],
    },
    /// See [`CommandList::dispatch`].
    Dispatch([u32; 3]),
    /// See [`CommandList::dispatch_indirect`].
    DispatchIndirect(GpuAddress),
    /// See [`CommandList::begin_debug_region`].
    BeginDebugRegion(&'a str),
    /// See [`CommandList::end_debug_region`].
    EndDebugRegion,
    /// See [`CommandList::insert_debug_label`].
    InsertDebugLabel(&'a str),
    /// See [`CommandList::begin_timing`].
    BeginTiming(&'a str),
    /// See [`CommandList::end_timing`].
    EndTiming,
}

#[cfg(test)]
impl CommandList {
    /// The recorded commands, in order, for tests to inspect.
    pub(crate) fn recorded(&self) -> Vec<RecordedCommand<'_>> {
        self.commands
            .iter()
            .map(|command| match *command {
                Command::Barrier { ref barriers } => {
                    RecordedCommand::Barrier(&self.barriers[barriers.clone()])
                }
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
                Command::CopyBufferToTexture { copy } => {
                    RecordedCommand::CopyBufferToTexture(&self.texture_copies[copy])
                }
                Command::CopyTextureToBuffer { copy } => {
                    RecordedCommand::CopyTextureToBuffer(&self.texture_copies[copy])
                }
                Command::BindProgram { program } => RecordedCommand::BindProgram(program),
                Command::PushData { offset, ref data } => RecordedCommand::PushData {
                    offset,
                    data: &self.push_data[data.clone()],
                },
                Command::Dispatch { groups } => RecordedCommand::Dispatch(groups),
                Command::DispatchIndirect { arguments } => {
                    RecordedCommand::DispatchIndirect(arguments)
                }
                Command::BeginDebugRegion { ref text } => {
                    RecordedCommand::BeginDebugRegion(&self.text[text.clone()])
                }
                Command::EndDebugRegion => RecordedCommand::EndDebugRegion,
                Command::InsertDebugLabel { ref text } => {
                    RecordedCommand::InsertDebugLabel(&self.text[text.clone()])
                }
                Command::BeginTiming { ref text } => {
                    RecordedCommand::BeginTiming(&self.text[text.clone()])
                }
                Command::EndTiming => RecordedCommand::EndTiming,
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
    //! T0 tests of recording commands and its checks (design 0002, R7 and
    //! R22).

    use super::*;
    use crate::handle::HandleTable;
    use crate::{Access, Buffer, Extent3d, Format, Texture};

    /// A program, for recording, which nothing looks up without a backend.
    fn program() -> Program {
        Program::from_handle(HandleTable::new("program").insert("write value", ()))
    }

    /// Programs, push data and dispatches are recorded in order, with their
    /// values, push data keeps each block's offset and bytes, and an empty
    /// block records nothing.
    #[test]
    fn dispatches_are_recorded() {
        let program = program();
        let mut list = CommandList::new("compute");
        list.bind_program(program);
        list.push_data(0, &0x1000_u64.to_le_bytes());
        list.push_data(8, &[1, 0, 0, 0]);
        list.dispatch([4, 2, 1]);
        list.push_data(8, &[]);
        list.push_data(8, &[2, 0, 0, 0]);
        list.dispatch([1, 1, 1]);
        list.dispatch_indirect(GpuAddress(0x2004));
        list.push_data(0, &[7; MAX_PUSH_DATA_SIZE as usize]);
        assert_eq!(
            list.recorded(),
            [
                RecordedCommand::BindProgram(program),
                RecordedCommand::PushData {
                    offset: 0,
                    data: &0x1000_u64.to_le_bytes(),
                },
                RecordedCommand::PushData {
                    offset: 8,
                    data: &[1, 0, 0, 0],
                },
                RecordedCommand::Dispatch([4, 2, 1]),
                RecordedCommand::PushData {
                    offset: 8,
                    data: &[2, 0, 0, 0],
                },
                RecordedCommand::Dispatch([1, 1, 1]),
                RecordedCommand::DispatchIndirect(GpuAddress(0x2004)),
                RecordedCommand::PushData {
                    offset: 0,
                    data: &[7; MAX_PUSH_DATA_SIZE as usize],
                },
            ],
        );
    }

    /// A dispatch with no program bound panics, naming the list.
    #[test]
    #[should_panic(expected = "command list \"compute\" dispatches with no program bound")]
    fn dispatches_need_a_program() {
        let mut list = CommandList::new("compute");
        list.push_data(0, &[0; 8]);
        list.dispatch([1, 1, 1]);
    }

    /// An indirect dispatch with no program bound panics, as a direct one
    /// does.
    #[test]
    #[should_panic(expected = "command list \"compute\" dispatches with no program bound")]
    fn indirect_dispatches_need_a_program() {
        CommandList::new("compute").dispatch_indirect(GpuAddress(0x1000));
    }

    /// Indirect arguments at an address that isn't a multiple of 4 panic.
    #[test]
    #[should_panic(
        expected = "command list \"compute\" dispatches with arguments at 0x1002, but indirect \
                    arguments' address must be a multiple of 4"
    )]
    fn unaligned_dispatch_arguments_panic() {
        let mut list = CommandList::new("compute");
        list.bind_program(program());
        list.dispatch_indirect(GpuAddress(0x1002));
    }

    /// Indirect arguments that pass the end of the address space panic.
    #[test]
    #[should_panic(
        expected = "command list \"compute\" dispatches with arguments 12 bytes at \
                    0xfffffffffffffff8, past the end of the address space"
    )]
    fn dispatch_arguments_past_the_end_of_the_address_space_panic() {
        let mut list = CommandList::new("compute");
        list.bind_program(program());
        list.dispatch_indirect(GpuAddress(u64::MAX - 7));
    }

    /// A push data block that passes the end of the push data panics,
    /// naming the list and the limit, whether from its size or its offset.
    #[test]
    fn push_data_past_the_end_panics() {
        for (offset, size) in [(0, MAX_PUSH_DATA_SIZE as usize + 4), (124, 8)] {
            let panic = std::panic::catch_unwind(|| {
                CommandList::new("compute").push_data(offset, &vec![0; size]);
            })
            .expect_err("the block should panic");
            assert_eq!(
                panic.downcast_ref::<String>().unwrap(),
                &format!(
                    "command list \"compute\" pushes {size} bytes of data at offset {offset}, past \
                     the end of the 128 bytes of push data"
                ),
            );
        }
    }

    /// A push data block whose offset or size isn't a multiple of 4
    /// panics.
    #[test]
    fn unaligned_push_data_panics() {
        for (offset, size) in [(0, 6), (2, 4)] {
            let panic = std::panic::catch_unwind(|| {
                CommandList::new("compute").push_data(offset, &vec![0; size]);
            })
            .expect_err("the block should panic");
            assert_eq!(
                panic.downcast_ref::<String>().unwrap(),
                &format!(
                    "command list \"compute\" pushes {size} bytes of data at offset {offset}, but \
                     a block's offset and size must be multiples of 4"
                ),
            );
        }
    }

    /// Barriers are recorded together, in order, and an empty slice records
    /// nothing.
    #[test]
    fn barriers_are_recorded_together() {
        let buffer = Buffer::from_handle(HandleTable::new("buffer").insert("staging", ()));
        let texture = Texture::from_handle(HandleTable::new("texture").insert("albedo", ()));
        let barriers = [
            Barrier::buffer(buffer, Access::HostWrite, Access::CopyRead),
            Barrier::texture(texture, Access::None, Access::CopyWrite),
        ];
        let mut list = CommandList::new("upload");
        list.barrier(&[]);
        list.barrier(&barriers);
        list.barrier(&barriers[..1]);
        assert_eq!(
            list.recorded(),
            [
                RecordedCommand::Barrier(&barriers),
                RecordedCommand::Barrier(&barriers[..1]),
            ],
        );
    }

    /// A list whose debug regions are closed can be translated, and debug
    /// labels leave nothing to close.
    #[test]
    fn closed_regions_are_complete() {
        let mut list = CommandList::new("regions");
        list.begin_debug_region("outer");
        list.insert_debug_label("inside");
        list.end_debug_region();
        list.insert_debug_label("after");
        list.check_complete();
    }

    /// Translating a list with a debug region still open panics, naming the
    /// list.
    #[test]
    #[should_panic(expected = "command list \"open\" is translated with 1 debug region still open")]
    fn open_regions_are_incomplete() {
        let mut list = CommandList::new("open");
        list.begin_debug_region("outer");
        list.check_complete();
    }

    /// Timing scopes are recorded in order with their labels, nest, and
    /// interleave with debug regions, and a list whose scopes are closed
    /// can be translated.
    #[test]
    fn timing_scopes_are_recorded() {
        let mut list = CommandList::new("frame");
        list.begin_timing("frame");
        list.begin_debug_region("shadows");
        list.begin_timing("shadows");
        list.end_debug_region();
        list.end_timing();
        list.begin_timing("lighting");
        list.end_timing();
        list.end_timing();
        list.check_complete();
        assert_eq!(
            list.recorded(),
            [
                RecordedCommand::BeginTiming("frame"),
                RecordedCommand::BeginDebugRegion("shadows"),
                RecordedCommand::BeginTiming("shadows"),
                RecordedCommand::EndDebugRegion,
                RecordedCommand::EndTiming,
                RecordedCommand::BeginTiming("lighting"),
                RecordedCommand::EndTiming,
                RecordedCommand::EndTiming,
            ],
        );
    }

    /// Translating a list with a timing scope still open panics, naming
    /// the list and how many are open.
    #[test]
    #[should_panic(
        expected = "command list \"open\" is translated with 2 timing scopes still open"
    )]
    fn open_timing_scopes_are_incomplete() {
        let mut list = CommandList::new("open");
        list.begin_timing("outer");
        list.begin_timing("inner");
        list.begin_timing("closed");
        list.end_timing();
        list.check_complete();
    }

    /// Ending a timing scope with none open panics, naming the list.
    #[test]
    #[should_panic(expected = "command list \"unbalanced\" ends a timing scope, but none is open")]
    fn ending_an_unopened_timing_scope_panics() {
        let mut list = CommandList::new("unbalanced");
        list.begin_timing("one");
        list.end_timing();
        list.end_timing();
    }

    /// A debug region isn't a timing scope, so ending a scope can't close
    /// one.
    #[test]
    #[should_panic(expected = "command list \"region\" ends a timing scope, but none is open")]
    fn regions_open_no_timing_scope() {
        let mut list = CommandList::new("region");
        list.begin_debug_region("pass");
        list.end_timing();
    }

    /// A timing scope isn't a debug region, so ending a region can't close
    /// one.
    #[test]
    #[should_panic(expected = "command list \"scope\" ends a debug region, but none is open")]
    fn timing_scopes_open_no_region() {
        let mut list = CommandList::new("scope");
        list.begin_timing("pass");
        list.end_debug_region();
    }

    /// Copies between buffers and textures are recorded in order, each with
    /// its buffer address, layout and region.
    #[test]
    fn texture_copies_are_recorded() {
        let texture = Texture::from_handle(HandleTable::new("texture").insert("albedo", ()));
        let extent = Extent3d::new_2d(4, 4);
        let layout = TextureDataLayout::tightly_packed(Format::Rgba8Unorm, extent);
        let region = TextureCopy {
            texture,
            mip: 0,
            layers: 0..1,
            origin: [0; 3],
            extent,
        };
        let mut list = CommandList::new("upload");
        list.copy_buffer_to_texture(GpuAddress(0x1000), layout, region.clone());
        list.copy_texture_to_buffer(region.clone(), GpuAddress(0x2000), layout);
        let copy = |buffer| BufferTextureCopy {
            buffer: GpuAddress(buffer),
            layout,
            texture: region.clone(),
        };
        assert_eq!(
            list.recorded(),
            [
                RecordedCommand::CopyBufferToTexture(&copy(0x1000)),
                RecordedCommand::CopyTextureToBuffer(&copy(0x2000)),
            ],
        );
    }

    /// Recording a barrier checks it, naming the list.
    #[test]
    #[should_panic(
        expected = "command list \"upload\" has a barrier for all memory with Present, which \
                    doesn't apply to all memory"
    )]
    fn recorded_barriers_are_checked() {
        CommandList::new("upload").barrier(&[Barrier::global(Access::None, Access::Present)]);
    }

    /// Commands are recorded in order, with their values, and debug regions
    /// and labels keep their text, including when regions nest.
    #[test]
    fn commands_are_recorded_in_order() {
        let mut list = CommandList::new("upload");
        list.begin_debug_region("outer");
        list.fill_buffer(GpuAddress(0x1000), 256, 0xC10C_C0DE);
        list.insert_debug_label("filled");
        list.begin_debug_region("inner");
        list.copy_buffer(GpuAddress(0x1000), GpuAddress(0x2000), 256);
        list.end_debug_region();
        list.end_debug_region();
        list.insert_debug_label("copied");
        assert_eq!(list.label(), "upload");
        assert_eq!(
            list.recorded(),
            [
                RecordedCommand::BeginDebugRegion("outer"),
                RecordedCommand::FillBuffer {
                    destination: GpuAddress(0x1000),
                    size: 256,
                    value: 0xC10C_C0DE,
                },
                RecordedCommand::InsertDebugLabel("filled"),
                RecordedCommand::BeginDebugRegion("inner"),
                RecordedCommand::CopyBuffer {
                    source: GpuAddress(0x1000),
                    destination: GpuAddress(0x2000),
                    size: 256,
                },
                RecordedCommand::EndDebugRegion,
                RecordedCommand::EndDebugRegion,
                RecordedCommand::InsertDebugLabel("copied"),
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

    /// Ending a debug region with none open panics, naming the list.
    #[test]
    #[should_panic(expected = "command list \"unbalanced\" ends a debug region, but none is open")]
    fn ending_an_unopened_region_panics() {
        let mut list = CommandList::new("unbalanced");
        list.begin_debug_region("one");
        list.end_debug_region();
        list.end_debug_region();
    }

    /// A debug label doesn't open a region, so a region can't end it.
    #[test]
    #[should_panic(expected = "command list \"label\" ends a debug region, but none is open")]
    fn labels_open_no_region() {
        let mut list = CommandList::new("label");
        list.insert_debug_label("point");
        list.end_debug_region();
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
