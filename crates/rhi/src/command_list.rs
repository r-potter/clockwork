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

use crate::{
    Barrier, ColorAttachment, DepthAttachment, GpuAddress, MAX_COLOUR_ATTACHMENTS,
    MAX_PUSH_DATA_SIZE, Program, Rect2d, RenderPassDesc, RenderState, TextureCopy,
    TextureDataLayout, Viewport,
};

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
    /// Every render state set, which commands refer to by index.
    render_states: Vec<RenderState>,
    /// Every render pass's colour attachments, which commands refer to by
    /// range.
    colour_attachments: Vec<ColorAttachment>,
    /// Every render pass's depth attachment, which commands refer to by
    /// index.
    depth_attachments: Vec<DepthAttachment>,
    /// Every push data block, which commands refer to by range. Each
    /// block's size is a multiple of 4, so each starts at a multiple of 4.
    push_data: Vec<u8>,
    /// Whether a program has been bound, so that dispatches can run.
    program_bound: bool,
    /// Whether a render pass is open: begun, and not yet ended.
    in_render_pass: bool,
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
    /// Begins a render pass to the colour attachments `colour` and the depth
    /// attachment `depth` in the list's arena.
    BeginRenderPass {
        /// The colour attachments, as a range of the arena.
        colour: Range<usize>,
        /// The depth attachment's index in the arena, or `None` for none.
        depth: Option<usize>,
    },
    /// Ends the render pass.
    EndRenderPass,
    /// Binds `program`, for later dispatches to run.
    BindProgram {
        /// The program.
        program: Program,
    },
    /// Sets the render state later draws use to `render_states[state]` in
    /// the list's arena.
    SetRenderState {
        /// The state's index in the arena.
        state: usize,
    },
    /// Sets the viewport later draws use.
    SetViewport {
        /// The viewport.
        viewport: Viewport,
    },
    /// Sets the scissor later draws use.
    SetScissor {
        /// The scissor.
        scissor: Rect2d,
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
            render_states: Vec::new(),
            colour_attachments: Vec::new(),
            depth_attachments: Vec::new(),
            push_data: Vec::new(),
            program_bound: false,
            in_render_pass: false,
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

    /// The render state at `index` in the arena, for translation.
    pub(crate) fn render_state(&self, index: usize) -> &RenderState {
        &self.render_states[index]
    }

    /// The colour attachments at `range` of the arena, for translation.
    pub(crate) fn colour_attachments(&self, range: &Range<usize>) -> &[ColorAttachment] {
        &self.colour_attachments[range.clone()]
    }

    /// The depth attachment at `index` in the arena, for translation.
    pub(crate) fn depth_attachment(&self, index: usize) -> &DepthAttachment {
        &self.depth_attachments[index]
    }

    /// The push data block at `range` of the arena, for translation.
    pub(crate) fn push_data_block(&self, range: &Range<usize>) -> &[u8] {
        &self.push_data[range.clone()]
    }

    /// Checks that the list can be translated: that no render pass is open, and
    /// every debug region and timing scope it opened is closed.
    ///
    /// # Panics
    ///
    /// Panics, naming the list, if a render pass, a region or a scope is still
    /// open.
    pub(crate) fn check_complete(&self) {
        assert!(
            !self.in_render_pass,
            "command list \"{}\" is translated with a render pass still open",
            self.label,
        );
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
    /// Panics if a render pass is open, or if a barrier's accesses don't suit
    /// what it applies to (see [`Barrier`]).
    pub fn barrier(&mut self, barriers: &[Barrier]) {
        self.check_outside_render_pass("records a barrier");
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
    /// Panics if a render pass is open, if `size` is zero, if `destination` or
    /// `size` isn't a multiple of 4, or if the range passes the end of the
    /// address space.
    pub fn fill_buffer(&mut self, destination: GpuAddress, size: u64, value: u32) {
        self.check_outside_render_pass("fills a buffer");
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
    /// Panics if a render pass is open, if `size` is zero, if either range
    /// passes the end of the address space, or if the ranges overlap.
    pub fn copy_buffer(&mut self, source: GpuAddress, destination: GpuAddress, size: u64) {
        self.check_outside_render_pass("copies a buffer");
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
    /// # Panics
    ///
    /// Panics if a render pass is open.
    ///
    /// [`Access::CopyWrite`]: crate::Access::CopyWrite
    pub fn copy_buffer_to_texture(
        &mut self,
        source: GpuAddress,
        layout: TextureDataLayout,
        destination: TextureCopy,
    ) {
        self.check_outside_render_pass("copies to a texture");
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
    /// # Panics
    ///
    /// Panics if a render pass is open.
    ///
    /// [`Access::CopyRead`]: crate::Access::CopyRead
    pub fn copy_texture_to_buffer(
        &mut self,
        source: TextureCopy,
        destination: GpuAddress,
        layout: TextureDataLayout,
    ) {
        self.check_outside_render_pass("copies from a texture");
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

    /// Begins a render pass to `render_pass`'s attachments, which draws render to
    /// until [`end_render_pass`](Self::end_render_pass) (design 0002,
    /// "Subresources and views"). Each attachment is loaded as it says.
    ///
    /// The attachments are checked when the list is translated, since that
    /// needs the textures (see [`RenderPassDesc`]). Translation also panics if
    /// the queue doesn't draw: only the graphics queue does.
    ///
    /// # Panics
    ///
    /// Panics if a render pass is already open, or if `render_pass` has no
    /// attachments, or more than [`MAX_COLOUR_ATTACHMENTS`] colour
    /// attachments.
    pub fn begin_render_pass(&mut self, render_pass: &RenderPassDesc) {
        assert!(
            !self.in_render_pass,
            "command list \"{}\" begins a render pass, but one is already open",
            self.label,
        );
        assert!(
            !render_pass.colour.is_empty() || render_pass.depth.is_some(),
            "command list \"{}\" begins a render pass with no attachments",
            self.label,
        );
        assert!(
            render_pass.colour.len() <= MAX_COLOUR_ATTACHMENTS,
            "command list \"{}\" begins a render pass with {} colour attachments, but a render pass has \
             at most {MAX_COLOUR_ATTACHMENTS}",
            self.label,
            render_pass.colour.len(),
        );
        let start = self.colour_attachments.len();
        self.colour_attachments
            .extend_from_slice(&render_pass.colour);
        let depth = render_pass.depth.clone().map(|depth| {
            self.depth_attachments.push(depth);
            self.depth_attachments.len() - 1
        });
        self.commands.push(Command::BeginRenderPass {
            colour: start..self.colour_attachments.len(),
            depth,
        });
        self.in_render_pass = true;
    }

    /// Ends the render pass, storing each attachment as it says.
    ///
    /// # Panics
    ///
    /// Panics if no render pass is open.
    pub fn end_render_pass(&mut self) {
        assert!(
            self.in_render_pass,
            "command list \"{}\" ends a render pass, but none is open",
            self.label,
        );
        self.in_render_pass = false;
        self.commands.push(Command::EndRenderPass);
    }

    /// Binds `program`, for later dispatches to run, until another of its
    /// kind is bound (design 0002, "Programs"). A compute program and a
    /// vertex and fragment program are bound separately, as Vulkan binds
    /// them, so binding one leaves the other bound.
    ///
    /// Translation panics if the program is stale, or if the queue can't
    /// run it: the transfer queue runs no programs, and only the graphics
    /// queue runs a vertex and fragment program.
    pub fn bind_program(&mut self, program: Program) {
        self.commands.push(Command::BindProgram { program });
        self.program_bound = true;
    }

    /// Sets the render state later draws use, until another is set (design
    /// 0002, "Programs"). Every piece of it is set each time.
    ///
    /// Translation panics if the queue doesn't draw: only the graphics
    /// queue does.
    pub fn set_render_state(&mut self, state: &RenderState) {
        self.render_states.push(*state);
        self.commands.push(Command::SetRenderState {
            state: self.render_states.len() - 1,
        });
    }

    /// Sets the viewport later draws use, until it's set again: how they map
    /// to their attachments, and the range their depths map to.
    ///
    /// Translation panics if the queue doesn't draw: only the graphics
    /// queue does.
    ///
    /// # Panics
    ///
    /// Panics if any of the viewport's values isn't finite, if it's empty,
    /// or if a depth isn't from 0 to 1, which Vulkan doesn't allow.
    pub fn set_viewport(&mut self, viewport: &Viewport) {
        let Viewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        } = *viewport;
        assert!(
            [x, y, width, height, min_depth, max_depth]
                .iter()
                .all(|value| value.is_finite()),
            "command list \"{}\" sets a viewport with a value that isn't finite: {viewport:?}",
            self.label,
        );
        assert!(
            width > 0.0 && height > 0.0,
            "command list \"{}\" sets an empty {width}x{height} viewport",
            self.label,
        );
        assert!(
            (0.0..=1.0).contains(&min_depth) && (0.0..=1.0).contains(&max_depth),
            "command list \"{}\" sets a viewport with depths {min_depth} to {max_depth}, but \
             depths must be from 0 to 1",
            self.label,
        );
        self.commands.push(Command::SetViewport {
            viewport: *viewport,
        });
    }

    /// Sets the scissor later draws use, until it's set again: the
    /// rectangle of their attachments outside which they write nothing. An
    /// empty scissor discards every fragment.
    ///
    /// Translation panics if the queue doesn't draw: only the graphics
    /// queue does.
    ///
    /// # Panics
    ///
    /// Panics if the scissor's right or bottom edge is past `i32::MAX`,
    /// which Vulkan doesn't allow.
    pub fn set_scissor(&mut self, scissor: Rect2d) {
        let Rect2d { offset, extent } = scissor;
        let fits = |start: u32, size: u32| {
            u64::from(start) + u64::from(size) <= i32::MAX.unsigned_abs().into()
        };
        assert!(
            fits(offset.x, extent.width) && fits(offset.y, extent.height),
            "command list \"{}\" sets a {}x{} scissor at ({}, {}), whose edges must be at most \
             {}",
            self.label,
            extent.width,
            extent.height,
            offset.x,
            offset.y,
            i32::MAX,
        );
        self.commands.push(Command::SetScissor { scissor });
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
    /// Translation panics if no compute program is bound, as when only a
    /// vertex and fragment program is.
    ///
    /// # Panics
    ///
    /// Panics if a render pass is open, or if no program has been bound.
    pub fn dispatch(&mut self, groups: [u32; 3]) {
        self.check_outside_render_pass("dispatches");
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
    /// Translation panics if no compute program is bound, as when only a
    /// vertex and fragment program is.
    ///
    /// # Panics
    ///
    /// Panics if a render pass is open, if no program has been bound, if
    /// `arguments` isn't a multiple of 4, or if the arguments pass the end
    /// of the address space.
    ///
    /// [`BufferUsage::INDIRECT`]: crate::BufferUsage::INDIRECT
    /// [`Access::IndirectRead`]: crate::Access::IndirectRead
    pub fn dispatch_indirect(&mut self, arguments: GpuAddress) {
        self.check_outside_render_pass("dispatches");
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

    /// Checks that no render pass is open, for a command that Vulkan doesn't
    /// allow inside a render pass, which `does` describes for the message.
    ///
    /// # Panics
    ///
    /// Panics, naming the list, if a render pass is open.
    fn check_outside_render_pass(&self, does: &str) {
        assert!(
            !self.in_render_pass,
            "command list \"{}\" {does} inside a render pass, which Vulkan doesn't allow",
            self.label,
        );
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RecordedCommand<'a> {
    /// See [`CommandList::barrier`].
    Barrier(&'a [Barrier]),
    /// See [`CommandList::begin_render_pass`].
    BeginRenderPass {
        /// The colour attachments.
        colour: &'a [ColorAttachment],
        /// The depth attachment.
        depth: Option<&'a DepthAttachment>,
    },
    /// See [`CommandList::end_render_pass`].
    EndRenderPass,
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
    /// See [`CommandList::set_render_state`].
    SetRenderState(&'a RenderState),
    /// See [`CommandList::set_viewport`].
    SetViewport(Viewport),
    /// See [`CommandList::set_scissor`].
    SetScissor(Rect2d),
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
                Command::BeginRenderPass { ref colour, depth } => {
                    RecordedCommand::BeginRenderPass {
                        colour: &self.colour_attachments[colour.clone()],
                        depth: depth.map(|depth| &self.depth_attachments[depth]),
                    }
                }
                Command::EndRenderPass => RecordedCommand::EndRenderPass,
                Command::BindProgram { program } => RecordedCommand::BindProgram(program),
                Command::SetRenderState { state } => {
                    RecordedCommand::SetRenderState(&self.render_states[state])
                }
                Command::SetViewport { viewport } => RecordedCommand::SetViewport(viewport),
                Command::SetScissor { scissor } => RecordedCommand::SetScissor(scissor),
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
    use crate::{Access, Buffer, Extent3d, Format, LoadOp, StoreOp, Texture};

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

    /// A viewport over part of a 1920x1080 attachment, with a depth range of
    /// its own.
    const VIEWPORT: Viewport = Viewport {
        x: 960.0,
        y: 0.0,
        width: 960.0,
        height: 540.0,
        min_depth: 0.25,
        max_depth: 0.75,
    };

    /// Render states, viewports and scissors are recorded in order, each
    /// with its value as it was set.
    #[test]
    fn render_state_is_recorded() {
        let culled = RenderState::default();
        let unculled = culled.cull(crate::CullMode::None);
        let scissor = Rect2d {
            offset: crate::Offset2d { x: 1000, y: 20 },
            extent: crate::Extent2d {
                width: 100,
                height: 50,
            },
        };
        let mut list = CommandList::new("draws");
        list.set_render_state(&culled);
        list.set_viewport(&VIEWPORT);
        list.set_scissor(scissor);
        list.set_render_state(&unculled);
        assert_eq!(
            list.recorded(),
            [
                RecordedCommand::SetRenderState(&culled),
                RecordedCommand::SetViewport(VIEWPORT),
                RecordedCommand::SetScissor(scissor),
                RecordedCommand::SetRenderState(&unculled),
            ],
        );
    }

    /// A viewport may start off the attachment, and its depths may be
    /// reversed.
    #[test]
    fn viewports_may_start_off_the_attachment() {
        let viewport = Viewport {
            x: -10.5,
            y: -20.0,
            min_depth: 1.0,
            max_depth: 0.0,
            ..VIEWPORT
        };
        let mut list = CommandList::new("offset");
        list.set_viewport(&viewport);
        assert_eq!(list.recorded(), [RecordedCommand::SetViewport(viewport)]);
    }

    /// A render pass to a texture called `label`, with `count` colour
    /// attachments of mip 0 of layer 0, each cleared to black, and no depth
    /// attachment.
    fn render_pass(label: &str, count: usize) -> RenderPassDesc {
        let texture = Texture::from_handle(HandleTable::new("texture").insert(label, ()));
        RenderPassDesc {
            colour: vec![
                RenderPassDesc::colour(texture, LoadOp::Clear([0.0; 4])).colour[0].clone();
                count
            ],
            depth: None,
        }
    }

    /// Render passes are recorded in order, each with its attachments as they
    /// were given, including a depth attachment with no colour attachments,
    /// and a list whose render passes are ended can be translated.
    #[test]
    fn render_passes_are_recorded() {
        let mut shadows = render_pass("shadow map", 0);
        shadows.depth = Some(DepthAttachment {
            texture: Texture::from_handle(HandleTable::new("texture").insert("shadow map", ())),
            mip: 0,
            layers: 0..4,
            load: LoadOp::Clear(0.0),
            store: StoreOp::Store,
        });
        let mut gbuffer = render_pass("albedo", 2);
        gbuffer.colour[1].mip = 1;
        gbuffer.colour[1].layers = 2..4;
        gbuffer.colour[1].load = LoadOp::Load;
        gbuffer.colour[1].store = StoreOp::DontCare;
        let mut list = CommandList::new("frame");
        list.begin_render_pass(&shadows);
        list.end_render_pass();
        list.begin_render_pass(&gbuffer);
        list.end_render_pass();
        list.check_complete();
        assert_eq!(
            list.recorded(),
            [
                RecordedCommand::BeginRenderPass {
                    colour: &[],
                    depth: shadows.depth.as_ref(),
                },
                RecordedCommand::EndRenderPass,
                RecordedCommand::BeginRenderPass {
                    colour: &gbuffer.colour,
                    depth: None,
                },
                RecordedCommand::EndRenderPass,
            ],
        );
    }

    /// Beginning a render pass while one is open panics, naming the list.
    #[test]
    #[should_panic(
        expected = "command list \"frame\" begins a render pass, but one is already open"
    )]
    fn render_pass_begun_twice_panics() {
        let mut list = CommandList::new("frame");
        list.begin_render_pass(&render_pass("albedo", 1));
        list.begin_render_pass(&render_pass("albedo", 1));
    }

    /// Ending a render pass that isn't open panics, naming the list.
    #[test]
    #[should_panic(expected = "command list \"frame\" ends a render pass, but none is open")]
    fn ending_an_unopened_render_pass_panics() {
        let mut list = CommandList::new("frame");
        list.begin_render_pass(&render_pass("albedo", 1));
        list.end_render_pass();
        list.end_render_pass();
    }

    /// Translating a list with a render pass still open panics, naming the
    /// list.
    #[test]
    #[should_panic(expected = "command list \"frame\" is translated with a render pass still open")]
    fn open_render_pass_is_incomplete() {
        let mut list = CommandList::new("frame");
        list.begin_render_pass(&render_pass("albedo", 1));
        list.check_complete();
    }

    /// Dispatches, copies, fills and barriers inside a render pass panic,
    /// naming the list and the command, and are allowed once it's ended.
    #[test]
    fn transfers_dispatches_and_barriers_panic_inside_a_render_pass() {
        /// Records a command into a list.
        type Record<'a> = Box<dyn Fn(&mut CommandList) + 'a>;
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
        let commands: [(&str, Record<'_>); 7] = [
            (
                "records a barrier",
                Box::new(|list| {
                    list.barrier(&[Barrier::texture(texture, Access::None, Access::CopyWrite)]);
                }),
            ),
            (
                "fills a buffer",
                Box::new(|list| list.fill_buffer(GpuAddress(0x1000), 4, 0)),
            ),
            (
                "copies a buffer",
                Box::new(|list| list.copy_buffer(GpuAddress(0x1000), GpuAddress(0x2000), 4)),
            ),
            (
                "copies to a texture",
                Box::new(|list| {
                    list.copy_buffer_to_texture(GpuAddress(0x1000), layout, region.clone());
                }),
            ),
            (
                "copies from a texture",
                Box::new(|list| {
                    list.copy_texture_to_buffer(region.clone(), GpuAddress(0x1000), layout);
                }),
            ),
            ("dispatches", Box::new(|list| list.dispatch([1, 1, 1]))),
            (
                "dispatches",
                Box::new(|list| list.dispatch_indirect(GpuAddress(0x1000))),
            ),
        ];
        for (does, record) in &commands {
            let mut list = CommandList::new("frame");
            list.bind_program(program());
            list.begin_render_pass(&render_pass("albedo", 1));
            let panic =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| record(&mut list)))
                    .expect_err("the command should panic");
            assert_eq!(
                panic.downcast_ref::<String>().unwrap(),
                &format!(
                    "command list \"frame\" {does} inside a render pass, which Vulkan doesn't allow"
                ),
            );
            list.end_render_pass();
            record(&mut list);
        }
    }

    /// A render pass needs an attachment, and at most eight colour attachments.
    #[test]
    fn render_passes_need_one_to_eight_colour_attachments() {
        let mut list = CommandList::new("frame");
        list.begin_render_pass(&render_pass("albedo", MAX_COLOUR_ATTACHMENTS));
        list.end_render_pass();
        for (count, message) in [
            (
                0,
                "command list \"frame\" begins a render pass with no attachments",
            ),
            (
                MAX_COLOUR_ATTACHMENTS + 1,
                "command list \"frame\" begins a render pass with 9 colour attachments, but a render pass \
                 has at most 8",
            ),
        ] {
            let panic = std::panic::catch_unwind(|| {
                CommandList::new("frame").begin_render_pass(&render_pass("albedo", count));
            })
            .expect_err("the render pass should panic");
            assert_eq!(panic.downcast_ref::<String>().unwrap(), message);
        }
    }

    /// An empty viewport panics, naming its size.
    #[test]
    #[should_panic(expected = "command list \"empty\" sets an empty 0x540 viewport")]
    fn empty_viewports_panic() {
        CommandList::new("empty").set_viewport(&Viewport {
            width: 0.0,
            ..VIEWPORT
        });
    }

    /// A viewport with a value that isn't finite panics.
    #[test]
    #[should_panic(
        expected = "command list \"nan\" sets a viewport with a value that isn't finite"
    )]
    fn viewports_that_are_not_finite_panic() {
        CommandList::new("nan").set_viewport(&Viewport {
            y: f32::NAN,
            ..VIEWPORT
        });
    }

    /// A viewport with a depth outside 0 to 1 panics, naming its depths.
    #[test]
    #[should_panic(
        expected = "command list \"deep\" sets a viewport with depths 0 to 2, but depths must be \
                    from 0 to 1"
    )]
    fn viewport_depths_past_one_panic() {
        CommandList::new("deep").set_viewport(&Viewport {
            min_depth: 0.0,
            max_depth: 2.0,
            ..VIEWPORT
        });
    }

    /// An empty scissor is recorded, and discards every fragment.
    #[test]
    fn empty_scissors_are_recorded() {
        let scissor = Rect2d::from_extent(crate::Extent2d {
            width: 0,
            height: 0,
        });
        let mut list = CommandList::new("empty");
        list.set_scissor(scissor);
        assert_eq!(list.recorded(), [RecordedCommand::SetScissor(scissor)]);
    }

    /// A scissor whose edge is past `i32::MAX` panics, and one whose edge is
    /// exactly there doesn't.
    #[test]
    #[should_panic(
        expected = "command list \"wide\" sets a 2x1 scissor at (2147483646, 0), whose edges must \
                    be at most 2147483647"
    )]
    fn scissors_past_i32_max_panic() {
        let mut list = CommandList::new("wide");
        let at_the_edge = Rect2d {
            offset: crate::Offset2d {
                x: i32::MAX.unsigned_abs() - 1,
                y: 0,
            },
            extent: crate::Extent2d {
                width: 1,
                height: 1,
            },
        };
        list.set_scissor(at_the_edge);
        list.set_scissor(Rect2d {
            extent: crate::Extent2d {
                width: 2,
                height: 1,
            },
            ..at_the_edge
        });
    }
}
