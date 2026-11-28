//! What a recorder records with (design 0003, "Recorders", R2): a wrapper
//! of its pass's command list for each kind of pass, with only the
//! commands that kind may record. The executor labels each pass's list
//! with the pass's name, which the wrappers' panics name the pass by.
//!
//! None can record a barrier, or begin or end a render pass: the graph
//! records those itself, from the plan, before and after the recorder
//! runs, so a recorder can't synchronize, and a raster pass's render pass,
//! viewport and scissor are already set. Only a transfer pass copies or
//! fills, only a raster pass draws, and only a compute pass dispatches.
//! Each also has the debug labels and timing scopes of a command list,
//! which nest inside the pass's own: a recorder can close only those it
//! opened, and must close them all before it returns, so the pass's region
//! and timing scope stay the graph's.
//!
//! The types stop a recorder synchronizing, not using a resource its pass
//! didn't declare. That's a rule: every resource the GPU writes during a
//! frame is a graph resource, and every use of one is declared.

use clockwork_rhi::{
    CommandList, GpuAddress, IndexFormat, Program, Rect2d, RenderState, TextureCopy,
    TextureDataLayout, Viewport,
};

/// What a recorder has opened and not yet closed, so it can't close the
/// debug region and timing scope the graph opens for its pass.
#[derive(Debug, Default)]
struct OpenedRegions {
    /// How many debug regions are open.
    debug_regions: u32,
    /// How many timing scopes are open.
    timing_regions: u32,
}

impl OpenedRegions {
    /// Records that the recorder of the pass `pass` closes a debug region.
    ///
    /// # Panics
    ///
    /// Panics if it has none open.
    fn end_debug_region(&mut self, pass: &str) {
        assert!(
            self.debug_regions > 0,
            "the recorder of pass \"{pass}\" ends a debug region it didn't open"
        );
        self.debug_regions -= 1;
    }

    /// Records that the recorder of the pass `pass` closes a timing scope.
    ///
    /// # Panics
    ///
    /// Panics if it has none open.
    fn end_timing(&mut self, pass: &str) {
        assert!(
            self.timing_regions > 0,
            "the recorder of pass \"{pass}\" ends a timing scope it didn't open"
        );
        self.timing_regions -= 1;
    }

    /// Checks that the recorder of the pass `pass` closed everything it
    /// opened.
    ///
    /// # Panics
    ///
    /// Panics if a debug region or timing scope is still open.
    fn check_closed(&self, pass: &str) {
        let open: Vec<_> = [
            (self.debug_regions, "debug region"),
            (self.timing_regions, "timing scope"),
        ]
        .into_iter()
        .filter(|&(count, _)| count > 0)
        .map(|(count, noun)| match count {
            1 => format!("1 {noun}"),
            count => format!("{count} {noun}s"),
        })
        .collect();
        assert!(
            open.is_empty(),
            "the recorder of pass \"{pass}\" returns with {} still open",
            open.join(" and ")
        );
    }
}

/// Generates the methods every recorder type has, each recording the
/// command list method of the same name: debug regions and labels, and
/// timing scopes, which nest inside the ones the graph opens for the pass.
macro_rules! labels_and_timings {
    ($name:ident) => {
        impl $name<'_> {
            /// Opens a debug region inside the pass's, as
            /// [`CommandList::begin_debug_region`] does. It must be closed
            /// before the recorder returns.
            pub fn begin_debug_region(&mut self, name: &str) {
                self.list.begin_debug_region(name);
                self.opened.debug_regions += 1;
            }

            /// Closes the innermost debug region the recorder opened, as
            /// [`CommandList::end_debug_region`] does.
            ///
            /// # Panics
            ///
            /// Panics, naming the pass, if the recorder has no debug region
            /// open, since closing one would close the pass's own.
            pub fn end_debug_region(&mut self) {
                self.opened.end_debug_region(self.list.label());
                self.list.end_debug_region();
            }

            /// Marks a point with a debug label, as
            /// [`CommandList::insert_debug_label`] does.
            pub fn insert_debug_label(&mut self, text: &str) {
                self.list.insert_debug_label(text);
            }

            /// Opens a timing scope inside the pass's, as
            /// [`CommandList::begin_timing`] does. It must be closed before
            /// the recorder returns.
            pub fn begin_timing(&mut self, label: &str) {
                self.list.begin_timing(label);
                self.opened.timing_regions += 1;
            }

            /// Closes the innermost timing scope the recorder opened, as
            /// [`CommandList::end_timing`] does.
            ///
            /// # Panics
            ///
            /// Panics, naming the pass, if the recorder has no timing scope
            /// open, since closing one would close the pass's own.
            pub fn end_timing(&mut self) {
                self.opened.end_timing(self.list.label());
                self.list.end_timing();
            }

            /// Checks that the recorder closed every debug region and timing
            /// scope it opened, once it returns, before the graph closes the
            /// pass's own.
            ///
            /// # Panics
            ///
            /// Panics, naming the pass, if one is still open.
            #[cfg_attr(
                not(test),
                expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
            )]
            pub(crate) fn check_closed(&self) {
                self.opened.check_closed(self.list.label());
            }
        }
    };
}

/// The commands a raster pass's recorder may record: programs, render
/// state, push data and draws. The render pass, a viewport and a scissor
/// covering its attachments are already set.
///
/// It has no barrier, no render pass and no dispatches or copies:
///
/// ```
/// fn record(commands: &mut clockwork_frame_graph::RasterCommands<'_>) {
///     commands.draw(3, 1);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::RasterCommands<'_>) {
///     commands.barrier(&[]);
/// }
/// ```
///
/// ```compile_fail,E0599
/// # use clockwork_rhi::RenderPassDesc;
/// fn record(commands: &mut clockwork_frame_graph::RasterCommands<'_>, pass: &RenderPassDesc) {
///     commands.begin_render_pass(pass);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::RasterCommands<'_>) {
///     commands.end_render_pass();
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::RasterCommands<'_>) {
///     commands.dispatch([1, 1, 1]);
/// }
/// ```
///
/// ```compile_fail,E0599
/// # use clockwork_rhi::GpuAddress;
/// fn record(commands: &mut clockwork_frame_graph::RasterCommands<'_>) {
///     commands.copy_buffer(GpuAddress(0), GpuAddress(256), 4);
/// }
/// ```
#[derive(Debug)]
pub struct RasterCommands<'a> {
    /// The pass's list.
    list: &'a mut CommandList,
    /// What the recorder has opened and not yet closed.
    opened: OpenedRegions,
}

impl<'a> RasterCommands<'a> {
    /// Records into `list`, a raster pass's.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
    )]
    pub(crate) fn new(list: &'a mut CommandList) -> Self {
        Self {
            list,
            opened: OpenedRegions::default(),
        }
    }

    /// Binds `program` for later draws, as [`CommandList::bind_program`]
    /// does.
    pub fn bind_program(&mut self, program: Program) {
        self.list.bind_program(program);
    }

    /// Sets the render state later draws use, as
    /// [`CommandList::set_render_state`] does.
    pub fn set_render_state(&mut self, state: &RenderState) {
        self.list.set_render_state(state);
    }

    /// Sets the viewport later draws use, in place of the one covering the
    /// attachments, as [`CommandList::set_viewport`] does.
    pub fn set_viewport(&mut self, viewport: &Viewport) {
        self.list.set_viewport(viewport);
    }

    /// Sets the scissor later draws use, in place of the one covering the
    /// attachments, as [`CommandList::set_scissor`] does.
    pub fn set_scissor(&mut self, scissor: Rect2d) {
        self.list.set_scissor(scissor);
    }

    /// Writes `data` to the push data at `offset`, as
    /// [`CommandList::push_data`] does. It must write all the push data its
    /// draws read: what an earlier pass wrote isn't kept.
    pub fn push_data(&mut self, offset: u32, data: &[u8]) {
        self.list.push_data(offset, data);
    }

    /// Binds an index buffer for later indexed draws, as
    /// [`CommandList::bind_index_buffer`] does.
    pub fn bind_index_buffer(&mut self, address: GpuAddress, size: u64, format: IndexFormat) {
        self.list.bind_index_buffer(address, size, format);
    }

    /// Draws `vertex_count` vertices, `instance_count` times, as
    /// [`CommandList::draw`] does.
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32) {
        self.list.draw(vertex_count, instance_count);
    }

    /// Draws `index_count` indices, `instance_count` times, as
    /// [`CommandList::draw_indexed`] does.
    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32) {
        self.list.draw_indexed(index_count, instance_count);
    }

    /// Draws `draw_count` times, from arguments the GPU reads, as
    /// [`CommandList::draw_indirect`] does.
    pub fn draw_indirect(&mut self, arguments: GpuAddress, draw_count: u32, stride: u64) {
        self.list.draw_indirect(arguments, draw_count, stride);
    }

    /// Draws indexed `draw_count` times, from arguments the GPU reads, as
    /// [`CommandList::draw_indexed_indirect`] does.
    pub fn draw_indexed_indirect(&mut self, arguments: GpuAddress, draw_count: u32, stride: u64) {
        self.list
            .draw_indexed_indirect(arguments, draw_count, stride);
    }
}

labels_and_timings!(RasterCommands);

/// The commands a compute pass's recorder may record: programs, push data
/// and dispatches.
///
/// It has no barrier, no render pass and no draws or copies:
///
/// ```
/// fn record(commands: &mut clockwork_frame_graph::ComputeCommands<'_>) {
///     commands.dispatch([1, 1, 1]);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::ComputeCommands<'_>) {
///     commands.barrier(&[]);
/// }
/// ```
///
/// ```compile_fail,E0599
/// # use clockwork_rhi::RenderPassDesc;
/// fn record(commands: &mut clockwork_frame_graph::ComputeCommands<'_>, pass: &RenderPassDesc) {
///     commands.begin_render_pass(pass);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::ComputeCommands<'_>) {
///     commands.end_render_pass();
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::ComputeCommands<'_>) {
///     commands.draw(3, 1);
/// }
/// ```
///
/// ```compile_fail,E0599
/// # use clockwork_rhi::GpuAddress;
/// fn record(commands: &mut clockwork_frame_graph::ComputeCommands<'_>) {
///     commands.copy_buffer(GpuAddress(0), GpuAddress(256), 4);
/// }
/// ```
#[derive(Debug)]
pub struct ComputeCommands<'a> {
    /// The pass's list.
    list: &'a mut CommandList,
    /// What the recorder has opened and not yet closed.
    opened: OpenedRegions,
}

impl<'a> ComputeCommands<'a> {
    /// Records into `list`, a compute pass's.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
    )]
    pub(crate) fn new(list: &'a mut CommandList) -> Self {
        Self {
            list,
            opened: OpenedRegions::default(),
        }
    }

    /// Binds `program` for later dispatches, as
    /// [`CommandList::bind_program`] does.
    pub fn bind_program(&mut self, program: Program) {
        self.list.bind_program(program);
    }

    /// Writes `data` to the push data at `offset`, as
    /// [`CommandList::push_data`] does. It must write all the push data its
    /// dispatches read: what an earlier pass wrote isn't kept.
    pub fn push_data(&mut self, offset: u32, data: &[u8]) {
        self.list.push_data(offset, data);
    }

    /// Dispatches `groups` workgroups, as [`CommandList::dispatch`] does.
    pub fn dispatch(&mut self, groups: [u32; 3]) {
        self.list.dispatch(groups);
    }

    /// Dispatches workgroup counts the GPU reads, as
    /// [`CommandList::dispatch_indirect`] does.
    pub fn dispatch_indirect(&mut self, arguments: GpuAddress) {
        self.list.dispatch_indirect(arguments);
    }
}

labels_and_timings!(ComputeCommands);

/// The commands a transfer pass's recorder may record: copies and fills.
///
/// It has no barrier, no render pass and no draws or dispatches:
///
/// ```
/// # use clockwork_rhi::GpuAddress;
/// fn record(commands: &mut clockwork_frame_graph::TransferCommands<'_>) {
///     commands.copy_buffer(GpuAddress(0), GpuAddress(256), 4);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::TransferCommands<'_>) {
///     commands.barrier(&[]);
/// }
/// ```
///
/// ```compile_fail,E0599
/// # use clockwork_rhi::RenderPassDesc;
/// fn record(commands: &mut clockwork_frame_graph::TransferCommands<'_>, pass: &RenderPassDesc) {
///     commands.begin_render_pass(pass);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::TransferCommands<'_>) {
///     commands.end_render_pass();
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::TransferCommands<'_>) {
///     commands.draw(3, 1);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn record(commands: &mut clockwork_frame_graph::TransferCommands<'_>) {
///     commands.dispatch([1, 1, 1]);
/// }
/// ```
#[derive(Debug)]
pub struct TransferCommands<'a> {
    /// The pass's list.
    list: &'a mut CommandList,
    /// What the recorder has opened and not yet closed.
    opened: OpenedRegions,
}

impl<'a> TransferCommands<'a> {
    /// Records into `list`, a transfer pass's.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
    )]
    pub(crate) fn new(list: &'a mut CommandList) -> Self {
        Self {
            list,
            opened: OpenedRegions::default(),
        }
    }

    /// Fills `size` bytes at `destination` with copies of `value`, as
    /// [`CommandList::fill_buffer`] does.
    pub fn fill_buffer(&mut self, destination: GpuAddress, size: u64, value: u32) {
        self.list.fill_buffer(destination, size, value);
    }

    /// Copies `size` bytes from `source` to `destination`, as
    /// [`CommandList::copy_buffer`] does.
    pub fn copy_buffer(&mut self, source: GpuAddress, destination: GpuAddress, size: u64) {
        self.list.copy_buffer(source, destination, size);
    }

    /// Copies buffer data to a texture's region, as
    /// [`CommandList::copy_buffer_to_texture`] does.
    pub fn copy_buffer_to_texture(
        &mut self,
        source: GpuAddress,
        layout: TextureDataLayout,
        destination: TextureCopy,
    ) {
        self.list
            .copy_buffer_to_texture(source, layout, destination);
    }

    /// Copies a texture's region to buffer data, as
    /// [`CommandList::copy_texture_to_buffer`] does.
    pub fn copy_texture_to_buffer(
        &mut self,
        source: TextureCopy,
        destination: GpuAddress,
        layout: TextureDataLayout,
    ) {
        self.list
            .copy_texture_to_buffer(source, destination, layout);
    }
}

labels_and_timings!(TransferCommands);

#[cfg(test)]
mod tests {
    //! T0 tests of what the recorder types record.

    use super::*;

    /// Each recorder type records into its pass's list, in order, so the
    /// executor can count and translate what it recorded, and a recorder
    /// that closes what it opens passes the check after it returns.
    #[test]
    fn recorders_record_into_their_passs_list() {
        let mut list = CommandList::new("measure");
        let mut compute = ComputeCommands::new(&mut list);
        compute.begin_timing("measure");
        compute.push_data(0, &[0; 4]);
        compute.end_timing();
        compute.check_closed();
        assert_eq!(list.command_count(), 3);

        let mut list = CommandList::new("save");
        let mut transfer = TransferCommands::new(&mut list);
        transfer.begin_debug_region("save");
        transfer.fill_buffer(GpuAddress(256), 16, 0);
        transfer.copy_buffer(GpuAddress(256), GpuAddress(512), 16);
        transfer.insert_debug_label("copied");
        transfer.end_debug_region();
        transfer.check_closed();
        assert_eq!(list.command_count(), 5);

        let mut list = CommandList::new("draw");
        let mut raster = RasterCommands::new(&mut list);
        raster.set_render_state(&RenderState::default());
        raster.check_closed();
        assert_eq!(list.command_count(), 1);
    }

    /// The graph's own debug region and timing scope for a pass, opened on
    /// `list` before its recorder runs.
    fn opened(label: &str) -> CommandList {
        let mut list = CommandList::new(label);
        list.begin_debug_region(label);
        list.begin_timing(label);
        list
    }

    /// A recorder that closes a debug region it didn't open panics, naming
    /// its pass, rather than closing the pass's own.
    #[test]
    #[should_panic(expected = "the recorder of pass \"save\" ends a debug region it didn't open")]
    fn a_recorder_cant_close_the_passs_debug_region() {
        let mut list = opened("save");
        let mut transfer = TransferCommands::new(&mut list);
        transfer.begin_debug_region("copy");
        transfer.end_debug_region();
        transfer.end_debug_region();
    }

    /// A recorder that closes a timing scope it didn't open panics, naming
    /// its pass, rather than closing the pass's own.
    #[test]
    #[should_panic(
        expected = "the recorder of pass \"measure\" ends a timing scope it didn't open"
    )]
    fn a_recorder_cant_close_the_passs_timing_scope() {
        let mut list = opened("measure");
        ComputeCommands::new(&mut list).end_timing();
    }

    /// A recorder that returns with debug regions and timing scopes open is
    /// found, naming its pass and how many of each are open.
    #[test]
    #[should_panic(
        expected = "the recorder of pass \"draw\" returns with 1 debug region and 2 timing scopes \
                    still open"
    )]
    fn a_recorder_must_close_what_it_opens() {
        let mut list = opened("draw");
        let mut raster = RasterCommands::new(&mut list);
        raster.begin_debug_region("opaque");
        raster.begin_timing("opaque");
        raster.begin_timing("sorted");
        raster.check_closed();
    }

    /// A recorder that returns with only a timing scope open is told of
    /// that alone.
    #[test]
    #[should_panic(
        expected = "the recorder of pass \"measure\" returns with 1 timing scope still open"
    )]
    fn a_recorder_is_told_only_what_it_left_open() {
        let mut list = opened("measure");
        let mut compute = ComputeCommands::new(&mut list);
        compute.begin_timing("reduce");
        compute.check_closed();
    }

    /// A recorder type is `Send`, so the job that records its pass can run
    /// on any thread. Recording takes `&mut`, so one is never recorded into
    /// from two threads at once.
    #[test]
    fn recorders_are_send() {
        /// Compiles only if `T` is `Send`.
        fn assert_send<T: Send>() {}
        assert_send::<RasterCommands<'_>>();
        assert_send::<ComputeCommands<'_>>();
        assert_send::<TransferCommands<'_>>();
    }
}
