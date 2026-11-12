//! Translating command lists into Vulkan command buffers (design 0002,
//! "Command lists and translation").
//!
//! Translation translates, and nothing more: each command becomes the
//! Vulkan commands it stands for, in order, with no merging, reordering or
//! removal (design 0002, "Keeping the recording model reversible"). It
//! looks up every handle the list names, holding each table's read lock
//! once for the whole list, and panics on a stale one, on a subresource
//! outside its texture, on a program the queue can't run, on a draw or
//! dispatch with no program of its kind bound, on render state, an index
//! buffer or a render pass for a queue that doesn't draw, and on
//! attachments their textures can't be, or that can't be rendered to
//! together.
//!
//! Every command buffer for a queue that runs programs, the graphics or
//! compute queue, starts by binding both descriptor heaps, so each
//! translates independently (design 0002, "Command lists and
//! translation"). The transfer queue's don't, since its family may not
//! support them, and it runs no programs.
//!
//! Lock order: the tables' read locks are taken while a command pool is
//! claimed, never the other way around, and the queue's lock is never
//! taken. Query chunks are claimed while the tables' read locks are held,
//! and the query pools' lock is held only to take a chunk, so it's last in
//! any order. The buffer table's read lock is always taken before the
//! texture table's, and the texture table's before the program table's,
//! and translation is the only code that holds more than one. A fixed
//! order is what keeps them from deadlocking: the standard `RwLock` makes a
//! new reader wait behind a waiting writer, so two threads taking the read
//! locks in opposite orders could each wait on the other, with a writer
//! queued on each table.

#[cfg(test)]
mod gpu_tests;

use std::sync::Arc;
use std::{fmt, mem};

use ash::vk;

use super::VulkanDevice;
use super::access::{Side, scope};
use super::command_pools::CommandPool;
use super::error::error;
use super::queues::Queues;
use super::render_state::colour_attachments;
use super::timings::RecordedTimings;
use crate::barrier::BarrierTarget;
use crate::command_list::{
    Command, DISPATCH_ARGUMENTS_SIZE, DRAW_ARGUMENTS_SIZE, DRAW_INDEXED_ARGUMENTS_SIZE,
    indirect_draws_size,
};
use crate::copy::check_texture_copy;
use crate::deferred::Translation;
use crate::ffi::truncate_to_cstring;
use crate::{
    Access, AccessSet, CommandBuffer, CommandList, GpuAddress, GpuTimepoint, IndexFormat, QueueId,
    QueueKind, RhiResult,
};

/// The flags every address command passes for its ranges: possibly in a
/// storage buffer, which Vulkan allows for any range (design 0002, "Binding
/// model").
///
/// Commands name buffers only by address, so the exact flag, whether a
/// storage buffer holds the range, would need a lookup of each range in a
/// table of the storage buffers' ranges. The flag is a hint, and nothing
/// has shown the exact one to be faster.
const ADDRESS_FLAGS: vk::AddressCommandFlagsKHR =
    vk::AddressCommandFlagsKHR::UNKNOWN_STORAGE_BUFFER_USAGE;

impl From<IndexFormat> for vk::IndexType {
    fn from(format: IndexFormat) -> Self {
        match format {
            IndexFormat::Uint16 => Self::UINT16,
            IndexFormat::Uint32 => Self::UINT32,
        }
    }
}

/// The range `draws` indirect draws read their arguments from, each
/// `arguments_size` bytes, the first at `arguments` and each next `stride`
/// bytes after.
///
/// Vulkan requires a stride no larger than the range, and ignores it for a
/// single draw, whose range is only its arguments, so the stride is 0 for
/// fewer than two draws.
///
/// # Panics
///
/// Panics if the range's size overflows, which recording checks against.
fn indirect_draws_range(
    arguments: GpuAddress,
    draws: u32,
    stride: u64,
    arguments_size: u64,
) -> vk::StridedDeviceAddressRangeKHR {
    let size = indirect_draws_size(draws, stride, arguments_size)
        .expect("recording checked that the arguments end within the address space");
    vk::StridedDeviceAddressRangeKHR::default()
        .address(arguments.0)
        .size(size)
        .stride(if draws > 1 { stride } else { 0 })
}

/// The info for `draws` indirect draws, indexed or not, whose arguments
/// are each `arguments_size` bytes, the first at `arguments` and each next
/// `stride` bytes after.
///
/// # Panics
///
/// Panics if the arguments' size overflows, which recording checks
/// against.
fn indirect_draw_info(
    arguments: GpuAddress,
    draws: u32,
    stride: u64,
    arguments_size: u64,
) -> vk::DrawIndirect2InfoKHR<'static> {
    vk::DrawIndirect2InfoKHR::default()
        .address_range(indirect_draws_range(
            arguments,
            draws,
            stride,
            arguments_size,
        ))
        .address_flags(ADDRESS_FLAGS)
        .draw_count(draws)
}

/// A command list translated for one queue, holding the command pool its
/// command buffer was recorded from, and holding back the destruction of
/// what it may use, until it's submitted or dropped.
pub(crate) struct VulkanCommandBuffer {
    /// The core, which owns the pool, so it outlives this.
    core: Arc<VulkanDevice>,
    /// The queue it was translated for.
    queue: QueueId,
    /// The pool and its command buffer, or `None` once submitted.
    commands: Option<CommandPool>,
    /// Its translation's epoch, given back to the core's deferred queue
    /// when it's submitted or dropped, or `None` once it has been.
    translation: Option<Translation>,
    /// Its timing scopes and their query chunks, which move to the queue
    /// when it's submitted, or are recycled when it's dropped unsubmitted.
    timings: RecordedTimings,
    /// The command list's label.
    label: Box<str>,
}

impl VulkanCommandBuffer {
    /// The recorded command buffer.
    pub(super) fn buffer(&self) -> vk::CommandBuffer {
        self.commands
            .as_ref()
            .expect("a command buffer is submitted only once")
            .buffer()
    }

    /// Checks that it was translated for the queue `id`.
    ///
    /// # Panics
    ///
    /// Panics, naming the command list and its queue, if it was translated
    /// for another.
    pub(super) fn check_queue(&self, id: QueueId) {
        assert!(
            self.queue == id,
            "command list \"{}\" was translated for the {} queue, so it can't be submitted to \
             another queue",
            self.label,
            self.core.queues.name(self.queue),
        );
    }

    /// Recycles the pools of `command_buffers`, all in the submission that
    /// reaches `timepoint`, reusable once the timeline reaches it, gives
    /// their translations back to the deferred queue, which keeps what they
    /// may use until then, taking its lock once for them all, and gives
    /// their timings to the queue, until they're taken.
    pub(super) fn all_submitted(command_buffers: Vec<CommandBuffer>, timepoint: GpuTimepoint) {
        let Some(CommandBuffer(first)) = command_buffers.first() else {
            return;
        };
        let core = first.core.clone();
        let queue = first.queue;
        let mut command_buffers = command_buffers;
        // The translations and timings are given back before any pool is
        // recycled, so a panic while recycling can't drop one that's been
        // taken.
        let translations: Vec<_> = command_buffers
            .iter_mut()
            .filter_map(|CommandBuffer(command_buffer)| command_buffer.translation.take())
            .collect();
        core.end_submitted_translations(translations, timepoint);
        let timings = command_buffers
            .iter_mut()
            .map(|CommandBuffer(command_buffer)| {
                let timings = mem::take(&mut command_buffer.timings);
                (command_buffer.label.clone(), timings)
            })
            .collect();
        core.keep_submitted_timings(queue, timepoint, timings);
        for CommandBuffer(mut command_buffer) in command_buffers {
            if let Some(commands) = command_buffer.commands.take() {
                core.queues
                    .recycle_pool(command_buffer.queue, commands, timepoint.value());
            }
        }
    }
}

impl Drop for VulkanCommandBuffer {
    /// Recycles an unsubmitted command buffer's pool to its queue, for reuse
    /// straight away, and its query chunks, and gives its translation back
    /// to the deferred queue, since no submission uses any of them.
    fn drop(&mut self) {
        if let Some(commands) = self.commands.take() {
            self.core.queues.recycle_pool(self.queue, commands, 0);
        }
        // SAFETY: A submitted command buffer's timings were taken when it
        // was submitted, so these, if any, were never submitted.
        unsafe { self.core.recycle_timings(mem::take(&mut self.timings)) };
        if let Some(translation) = self.translation.take() {
            self.core.end_dropped_translation(translation);
        }
    }
}

impl fmt::Debug for VulkanCommandBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandBuffer")
            .field("label", &self.label)
            .field("queue", &self.queue)
            .field("submitted", &self.commands.is_none())
            .finish()
    }
}

impl VulkanDevice {
    /// Translates `list` into a command buffer for the queue `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if a command pool or query chunk can't be claimed,
    /// or the command buffer can't be begun or ended.
    ///
    /// # Panics
    ///
    /// Panics if `list` has a render pass, a debug region or a timing scope
    /// open, names a stale handle, binds a program the queue can't run,
    /// renders, or sets render state or an index buffer, on a queue that
    /// doesn't draw, or renders to attachments their textures can't be, or
    /// that can't be rendered to together, or if it draws with no vertex
    /// and fragment program bound, or dispatches with no compute program
    /// bound.
    pub(super) fn translate(
        self: &Arc<Self>,
        id: QueueId,
        list: &CommandList,
    ) -> RhiResult<CommandBuffer> {
        list.check_complete();
        let device = self.device.handle();
        let commands = self.queues.claim_pool(device, id)?;
        let buffer = commands.buffer();
        // From here, returning early or panicking recycles the pool and the
        // query chunks, and gives the translation back. The translation
        // starts before any handle is looked up, so an object destroyed after
        // its lookup is retired in this epoch or later, and kept while this
        // is outstanding.
        let mut command_buffer = VulkanCommandBuffer {
            core: self.clone(),
            queue: id,
            commands: Some(commands),
            translation: Some(self.begin_translation()),
            timings: RecordedTimings::default(),
            label: list.label().into(),
        };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: The command buffer was allocated on `device` from a pool
        // that's claimed, so no other thread uses it, and claiming it put it
        // in the initial state.
        unsafe { device.begin_command_buffer(buffer, &begin_info) }
            .map_err(|result| error(result, "begin the command buffer for", list.label()))?;
        // SAFETY: The command buffer is recording, for this queue's family.
        unsafe { self.record(buffer, id, list, &mut command_buffer.timings) }?;
        // SAFETY: The command buffer is recording.
        unsafe { device.end_command_buffer(buffer) }
            .map_err(|result| error(result, "record the command buffer for", list.label()))?;
        self.device.set_object_name(buffer, list.label());
        Ok(CommandBuffer(command_buffer))
    }

    /// Records `list`'s commands into `buffer`, for the queue `id`, and its
    /// timing scopes into `timings`.
    ///
    /// A timing scope's timestamps are written at `ALL_COMMANDS`, so its end
    /// is written once every command before it has completed. On a queue
    /// that can't write timestamps, scopes are recorded with no queries.
    ///
    /// # Errors
    ///
    /// Returns an error if a query chunk can't be claimed.
    ///
    /// # Safety
    ///
    /// `buffer` must be recording, on this device, for that queue's family,
    /// and no other thread may use it.
    ///
    /// # Panics
    ///
    /// Panics if `list` names a stale handle, has a queue transfer neither
    /// from nor to that queue, binds a program or pushes data the queue
    /// can't use, renders, or sets render state or an index buffer, on a
    /// queue that doesn't draw, renders to attachments their textures
    /// can't be, or that can't be rendered to together, or draws with no
    /// vertex and fragment program bound, or dispatches with no compute
    /// program bound.
    unsafe fn record(
        &self,
        buffer: vk::CommandBuffer,
        id: QueueId,
        list: &CommandList,
        timings: &mut RecordedTimings,
    ) -> RhiResult<()> {
        let device = self.device.handle();
        let writes_timestamps = self.queues.clock(id).is_some();
        let kind = self.queues.kind(id);
        let runs_programs = kind != QueueKind::Transfer;
        let colour_attachments =
            colour_attachments(self.physical_device.limits().max_color_attachments);
        if runs_programs {
            let descriptor_heap = self.device.descriptor_heap();
            // SAFETY: Guaranteed by the caller, the queue's family supports
            // graphics or compute, and the heaps live as long as the core,
            // which outlives every command buffer.
            unsafe {
                descriptor_heap.cmd_bind_resource_heap(buffer, &self.heaps.resource.bind_info());
                descriptor_heap.cmd_bind_sampler_heap(buffer, &self.heaps.sampler.bind_info());
            }
        }
        let buffers = self.lock_buffers_for_reading();
        // After the buffer table's, and the program table's after it, as the
        // lock order above requires.
        let textures = self.lock_textures_for_reading();
        let programs = self.lock_programs_for_reading();
        // The programs last bound at each bind point, by label. Binding a
        // program binds only its own stages, so a compute program and a
        // vertex and fragment program stay bound together.
        let mut bound = BoundPrograms::default();
        // Reused for each barrier command, so recording them doesn't allocate
        // once they've grown.
        let mut memory_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();
        let mut image_barriers = Vec::new();
        for command in list.commands() {
            match command {
                Command::Barrier { barriers } => {
                    memory_barriers.clear();
                    buffer_barriers.clear();
                    image_barriers.clear();
                    for barrier in list.barriers(barriers) {
                        let mut before = scope(barrier.before(), Side::Before);
                        let mut after = scope(barrier.after(), Side::After);
                        let transfer = queue_transfer(
                            &self.queues,
                            id,
                            barrier.queue_transfer(),
                            list.label(),
                        );
                        // A release's later accesses, and an acquire's
                        // earlier ones, happen on the other queue, so each
                        // half orders only its own queue's.
                        let (src_family, dst_family) = match transfer {
                            Transfer::None => (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                            Transfer::Release { from, to } => {
                                after.stages = vk::PipelineStageFlags2::NONE;
                                after.access = vk::AccessFlags2::NONE;
                                (from, to)
                            }
                            Transfer::Acquire { from, to } => {
                                before.stages = vk::PipelineStageFlags2::NONE;
                                before.access = vk::AccessFlags2::NONE;
                                (from, to)
                            }
                        };
                        match barrier.target() {
                            BarrierTarget::Global => memory_barriers.push(
                                vk::MemoryBarrier2::default()
                                    .src_stage_mask(before.stages)
                                    .src_access_mask(before.access)
                                    .dst_stage_mask(after.stages)
                                    .dst_access_mask(after.access),
                            ),
                            BarrierTarget::Buffer(target) => buffer_barriers.push(
                                vk::BufferMemoryBarrier2::default()
                                    .src_stage_mask(before.stages)
                                    .src_access_mask(before.access)
                                    .dst_stage_mask(after.stages)
                                    .dst_access_mask(after.access)
                                    .src_queue_family_index(src_family)
                                    .dst_queue_family_index(dst_family)
                                    .buffer(buffers.get(target.handle()).buffer())
                                    .offset(0)
                                    .size(vk::WHOLE_SIZE),
                            ),
                            BarrierTarget::Texture(target, subresource) => {
                                let (label, entry) = textures
                                    .lookup(target.handle())
                                    .unwrap_or_else(|err| panic!("{err}"));
                                let layout = |set: AccessSet| {
                                    vk::ImageLayout::from(
                                        set.layout()
                                            .expect("recording checked the set has one layout"),
                                    )
                                };
                                image_barriers.push(
                                    vk::ImageMemoryBarrier2::default()
                                        .src_stage_mask(before.stages)
                                        .src_access_mask(before.access)
                                        .dst_stage_mask(after.stages)
                                        .dst_access_mask(after.access)
                                        .old_layout(layout(barrier.before()))
                                        .new_layout(layout(barrier.after()))
                                        .src_queue_family_index(src_family)
                                        .dst_queue_family_index(dst_family)
                                        .image(entry.image())
                                        .subresource_range(
                                            entry.subresource_range(label, subresource),
                                        ),
                                );
                            }
                        }
                    }
                    let dependency_info = vk::DependencyInfo::default()
                        .memory_barriers(&memory_barriers)
                        .buffer_memory_barriers(&buffer_barriers)
                        .image_memory_barriers(&image_barriers);
                    // SAFETY: Guaranteed by the caller, each buffer and
                    // image is live, since the tables are locked, each
                    // subresource range is inside its image, and
                    // `dependency_info` is valid for the call.
                    unsafe { device.cmd_pipeline_barrier2(buffer, &dependency_info) };
                }
                &Command::FillBuffer {
                    destination,
                    size,
                    value,
                } => {
                    let range = vk::DeviceAddressRangeKHR::default()
                        .address(destination.0)
                        .size(size);
                    // SAFETY: Guaranteed by the caller. Recording checked
                    // the range's alignment.
                    unsafe {
                        self.device.device_address_commands().cmd_fill_memory(
                            buffer,
                            &range,
                            ADDRESS_FLAGS,
                            value,
                        );
                    }
                }
                &Command::CopyBuffer {
                    source,
                    destination,
                    size,
                } => {
                    let region = vk::DeviceMemoryCopyKHR::default()
                        .src_range(
                            vk::DeviceAddressRangeKHR::default()
                                .address(source.0)
                                .size(size),
                        )
                        .src_flags(ADDRESS_FLAGS)
                        .dst_range(
                            vk::DeviceAddressRangeKHR::default()
                                .address(destination.0)
                                .size(size),
                        )
                        .dst_flags(ADDRESS_FLAGS);
                    let copy_info = vk::CopyDeviceMemoryInfoKHR::default()
                        .regions(std::slice::from_ref(&region));
                    // SAFETY: Guaranteed by the caller, and `copy_info` is
                    // valid for the call. Recording checked that the ranges
                    // don't overlap.
                    unsafe {
                        self.device
                            .device_address_commands()
                            .cmd_copy_memory(buffer, &copy_info);
                    }
                }
                &(Command::CopyBufferToTexture { copy }
                | Command::CopyTextureToBuffer { copy }) => {
                    let to_texture = matches!(command, Command::CopyBufferToTexture { .. });
                    let copy = list.texture_copy(copy);
                    let region = &copy.texture;
                    let (label, entry) = textures
                        .lookup(region.texture.handle())
                        .unwrap_or_else(|err| panic!("{err}"));
                    let end = check_texture_copy(&entry.copied(label), region, &copy.layout);
                    let start = copy.buffer.0.checked_add(copy.layout.offset);
                    let start = start
                        .filter(|start| start.checked_add(end - copy.layout.offset).is_some())
                        .unwrap_or_else(|| {
                            panic!(
                                "command list \"{}\" copies texture \"{label}\" past the end of \
                                 the address space",
                                list.label(),
                            )
                        });
                    // The texture is in the layout its copy access needs,
                    // as the barriers before the copy put it.
                    let access = if to_texture {
                        Access::CopyWrite
                    } else {
                        Access::CopyRead
                    };
                    let layout = vk::ImageLayout::from(
                        access.layout().expect("copy accesses apply to textures"),
                    );
                    let format = entry.format();
                    let texel =
                        |value: u32| i32::try_from(value).expect("texel offsets fit an i32");
                    // Vulkan counts rows and images in texels. A copy of one
                    // image doesn't use its height, which the checks allow
                    // to be short, so it's passed as 0, which Vulkan reads
                    // as tightly packed.
                    let in_texels = |blocks: u32, block_extent: u32| {
                        blocks.checked_mul(block_extent).unwrap_or_else(|| {
                            panic!(
                                "command list \"{}\" copies texture \"{label}\" with a layout too \
                                 large for Vulkan to count in texels",
                                list.label(),
                            )
                        })
                    };
                    let images = (region.layers.end - region.layers.start) * region.extent.depth;
                    let row_length = in_texels(
                        copy.layout.bytes_per_row / format.block_bytes,
                        format.block_extent[0],
                    );
                    let image_height = if images == 1 {
                        0
                    } else {
                        in_texels(copy.layout.rows_per_image, format.block_extent[1])
                    };
                    let memory_copy = vk::DeviceMemoryImageCopyKHR::default()
                        .address_range(
                            vk::DeviceAddressRangeKHR::default()
                                .address(start)
                                .size(end - copy.layout.offset),
                        )
                        .address_flags(ADDRESS_FLAGS)
                        .address_row_length(row_length)
                        .address_image_height(image_height)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: entry.aspect(),
                            mip_level: region.mip,
                            base_array_layer: region.layers.start,
                            layer_count: region.layers.end - region.layers.start,
                        })
                        .image_layout(layout)
                        .image_offset(vk::Offset3D {
                            x: texel(region.origin[0]),
                            y: texel(region.origin[1]),
                            z: texel(region.origin[2]),
                        })
                        .image_extent(vk::Extent3D {
                            width: region.extent.width,
                            height: region.extent.height,
                            depth: region.extent.depth,
                        });
                    let copy_info = vk::CopyDeviceMemoryImageInfoKHR::default()
                        .image(entry.image())
                        .regions(std::slice::from_ref(&memory_copy));
                    let device_address_commands = self.device.device_address_commands();
                    // SAFETY: Guaranteed by the caller, the image is live,
                    // since the table is locked, the copy passed its checks,
                    // so it's inside the image and laid out in whole
                    // blocks, and `copy_info` is valid for the call.
                    unsafe {
                        if to_texture {
                            device_address_commands.cmd_copy_memory_to_image(buffer, &copy_info);
                        } else {
                            device_address_commands.cmd_copy_image_to_memory(buffer, &copy_info);
                        }
                    }
                }
                Command::BindProgram { program } => {
                    let (label, entry) = programs
                        .lookup(program.handle())
                        .unwrap_or_else(|err| panic!("{err}"));
                    assert!(
                        entry.runs_on(kind),
                        "command list \"{}\" binds program \"{label}\", which the {} queue can't \
                         run",
                        list.label(),
                        self.queues.name(id),
                    );
                    // SAFETY: Guaranteed by the caller, the shader objects
                    // are live, since the table is locked, and the queue can
                    // run them.
                    unsafe {
                        self.device.shader_object().cmd_bind_shaders(
                            buffer,
                            entry.stages(),
                            entry.shaders(),
                        );
                    }
                    if entry.is_compute() {
                        bound.compute = Some(label);
                    } else {
                        bound.graphics = Some(label);
                    }
                }
                &Command::BeginRenderPass { ref colour, depth } => {
                    check_draws(list, kind, "begins a render pass");
                    // SAFETY: Guaranteed by the caller, the queue's family
                    // supports graphics, recording checked that a render pass
                    // isn't already open, and `textures` is this device's
                    // table, locked until recording ends.
                    unsafe {
                        self.begin_render_pass(
                            buffer,
                            list,
                            list.colour_attachments(colour),
                            depth.map(|depth| list.depth_attachment(depth)),
                            &textures,
                        )
                    }?;
                }
                Command::EndRenderPass => {
                    // SAFETY: Guaranteed by the caller, and recording checked
                    // that a render pass is open, so it began on this queue,
                    // which draws.
                    unsafe { device.cmd_end_rendering(buffer) };
                }
                &Command::SetRenderState { state } => {
                    check_draws(list, kind, "sets render state");
                    // SAFETY: Guaranteed by the caller, the queue's family
                    // supports graphics, and `colour_attachments` is at most
                    // the device's limit.
                    unsafe {
                        self.device.set_render_state(
                            buffer,
                            list.render_state(state),
                            colour_attachments,
                        );
                    }
                }
                Command::SetViewport { viewport } => {
                    check_draws(list, kind, "sets a viewport");
                    // SAFETY: Guaranteed by the caller, and the queue's
                    // family supports graphics.
                    unsafe { self.device.set_viewport(buffer, viewport) };
                }
                &Command::SetScissor { scissor } => {
                    check_draws(list, kind, "sets a scissor");
                    // SAFETY: Guaranteed by the caller, and the queue's
                    // family supports graphics.
                    unsafe { self.device.set_scissor(buffer, scissor) };
                }
                &Command::PushData { offset, ref data } => {
                    assert!(
                        runs_programs,
                        "command list \"{}\" pushes data, but the {} queue runs no programs",
                        list.label(),
                        self.queues.name(id),
                    );
                    let push_data_info = vk::PushDataInfoEXT::default().offset(offset).data(
                        vk::HostAddressRangeConstEXT::default().address(list.push_data_block(data)),
                    );
                    // SAFETY: Guaranteed by the caller, the queue's family
                    // supports graphics or compute, recording checked the
                    // block's alignment, and that it ends within the limit
                    // device selection requires, and `push_data_info` is
                    // valid for the call.
                    unsafe {
                        self.device
                            .descriptor_heap()
                            .cmd_push_data(buffer, &push_data_info);
                    }
                }
                &Command::BindIndexBuffer {
                    address,
                    size,
                    format,
                } => {
                    check_draws(list, kind, "binds an index buffer");
                    let bind_info = vk::BindIndexBuffer3InfoKHR::default()
                        .address_range(
                            vk::DeviceAddressRangeKHR::default()
                                .address(address.0)
                                .size(size),
                        )
                        .address_flags(ADDRESS_FLAGS)
                        .index_type(vk::IndexType::from(format));
                    // SAFETY: Guaranteed by the caller, the queue's family
                    // supports graphics, recording checked that the address
                    // and size are multiples of the index size, and
                    // `bind_info` is valid for the call.
                    unsafe {
                        self.device
                            .device_address_commands()
                            .cmd_bind_index_buffer3(buffer, &bind_info);
                    }
                }
                &Command::DrawIndexed { indices, instances } => {
                    bound.check_draw(list);
                    // SAFETY: Guaranteed by the caller, a render pass is open,
                    // so the queue draws, a vertex and fragment program is
                    // bound, with the heaps, and recording checked that
                    // render state, a viewport and a scissor are set, and
                    // that an index buffer holding `indices` indices is
                    // bound.
                    unsafe { device.cmd_draw_indexed(buffer, indices, instances, 0, 0, 0) };
                }
                &Command::DrawIndirect {
                    arguments,
                    draws,
                    stride,
                } => {
                    bound.check_draw(list);
                    let draw_info =
                        indirect_draw_info(arguments, draws, stride, DRAW_ARGUMENTS_SIZE);
                    // SAFETY: Guaranteed by the caller, a render pass is open,
                    // so the queue draws, a vertex and fragment program is
                    // bound, with the heaps, recording checked that render
                    // state, a viewport and a scissor are set, that the draw
                    // count is within the limit device selection requires,
                    // and the arguments' address and stride, and
                    // `draw_info` is valid for the call.
                    unsafe {
                        self.device
                            .device_address_commands()
                            .cmd_draw_indirect2(buffer, &draw_info);
                    }
                }
                &Command::DrawIndexedIndirect {
                    arguments,
                    draws,
                    stride,
                } => {
                    bound.check_draw(list);
                    let draw_info =
                        indirect_draw_info(arguments, draws, stride, DRAW_INDEXED_ARGUMENTS_SIZE);
                    // SAFETY: Guaranteed by the caller, a render pass is open,
                    // so the queue draws, a vertex and fragment program is
                    // bound, with the heaps, recording checked that render
                    // state, a viewport, a scissor and an index buffer are
                    // set, that the draw count is within the limit device
                    // selection requires, and the arguments' address and
                    // stride, and `draw_info` is valid for the call.
                    unsafe {
                        self.device
                            .device_address_commands()
                            .cmd_draw_indexed_indirect2(buffer, &draw_info);
                    }
                }
                &Command::Draw {
                    vertices,
                    instances,
                } => {
                    bound.check_draw(list);
                    // SAFETY: Guaranteed by the caller, a render pass is open,
                    // so the queue draws, a vertex and fragment program is
                    // bound, with the heaps, and recording checked that
                    // render state, a viewport and a scissor are set.
                    unsafe { device.cmd_draw(buffer, vertices, instances, 0, 0) };
                }
                &Command::Dispatch { groups: [x, y, z] } => {
                    bound.check_dispatch(list);
                    // SAFETY: Guaranteed by the caller, and a compute
                    // program the queue can run is bound, with the heaps.
                    unsafe { device.cmd_dispatch(buffer, x, y, z) };
                }
                &Command::DispatchIndirect { arguments } => {
                    bound.check_dispatch(list);
                    let dispatch_info = vk::DispatchIndirect2InfoKHR::default()
                        .address_range(
                            vk::DeviceAddressRangeKHR::default()
                                .address(arguments.0)
                                .size(DISPATCH_ARGUMENTS_SIZE),
                        )
                        .address_flags(ADDRESS_FLAGS);
                    // SAFETY: Guaranteed by the caller, a compute program
                    // the queue can run is bound, with the heaps, recording
                    // checked the arguments' alignment, and
                    // `dispatch_info` is valid for the call.
                    unsafe {
                        self.device
                            .device_address_commands()
                            .cmd_dispatch_indirect2(buffer, &dispatch_info);
                    }
                }
                Command::BeginDebugRegion { text } => {
                    if let Some(debug_utils) = self.device.debug_utils() {
                        let name = truncate_to_cstring(list.text(text));
                        let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
                        // SAFETY: Guaranteed by the caller, and `label` is
                        // valid for the call.
                        unsafe { debug_utils.cmd_begin_debug_utils_label(buffer, &label) };
                    }
                }
                Command::EndDebugRegion => {
                    if let Some(debug_utils) = self.device.debug_utils() {
                        // SAFETY: Guaranteed by the caller, and recording
                        // checked that a region is open.
                        unsafe { debug_utils.cmd_end_debug_utils_label(buffer) };
                    }
                }
                Command::InsertDebugLabel { text } => {
                    if let Some(debug_utils) = self.device.debug_utils() {
                        let name = truncate_to_cstring(list.text(text));
                        let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
                        // SAFETY: Guaranteed by the caller, and `label` is
                        // valid for the call.
                        unsafe { debug_utils.cmd_insert_debug_utils_label(buffer, &label) };
                    }
                }
                Command::BeginTiming { text } => {
                    // SAFETY: Guaranteed by the caller, and `writes_timestamps`
                    // is whether this queue has a clock, so whether its family
                    // can write timestamps.
                    let query =
                        unsafe { self.write_timestamp(buffer, writes_timestamps, timings) }?;
                    timings.begin(list.text(text), query);
                }
                Command::EndTiming => {
                    // SAFETY: As above.
                    let query =
                        unsafe { self.write_timestamp(buffer, writes_timestamps, timings) }?;
                    timings.end(query);
                }
            }
        }
        Ok(())
    }

    /// Writes a timestamp into `buffer` at `ALL_COMMANDS`, to the next of
    /// `timings`' queries, and returns that query's number, if
    /// `writes_timestamps`. Otherwise writes nothing, and returns `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if a query chunk can't be claimed.
    ///
    /// # Safety
    ///
    /// `buffer` must be recording, on this device, and no other thread may
    /// use it. If `writes_timestamps`, it must be for a family that can
    /// write timestamps.
    unsafe fn write_timestamp(
        &self,
        buffer: vk::CommandBuffer,
        writes_timestamps: bool,
        timings: &mut RecordedTimings,
    ) -> RhiResult<Option<u32>> {
        if !writes_timestamps {
            return Ok(None);
        }
        let (query, pool, index) = timings.next_query(self)?;
        // SAFETY: Guaranteed by the caller, and the query is in a chunk this
        // command buffer claimed, so it's reset and no other command buffer
        // writes it.
        unsafe {
            self.device.handle().cmd_write_timestamp2(
                buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                pool,
                index,
            );
        }
        Ok(Some(query))
    }
}

/// How a barrier's queue transfer is recorded on one queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transfer {
    /// No transfer: the barrier has none, or both ends are on one family.
    None,
    /// The releasing half, on the queue ownership moves from.
    Release {
        /// The family ownership moves from.
        from: u32,
        /// The family ownership moves to.
        to: u32,
    },
    /// The acquiring half, on the queue ownership moves to.
    Acquire {
        /// The family ownership moves from.
        from: u32,
        /// The family ownership moves to.
        to: u32,
    },
}

/// What [`queue_transfer`] needs to know about a device's queues, so that
/// its tests can describe a device without creating one.
trait QueueLookup {
    /// The family of the queue `id`.
    fn family(&self, id: QueueId) -> u32;

    /// The name of the queue `id`, for messages.
    fn name(&self, id: QueueId) -> &'static str;
}

impl QueueLookup for Queues {
    fn family(&self, id: QueueId) -> u32 {
        Queues::family(self, id)
    }

    fn name(&self, id: QueueId) -> &'static str {
        Queues::name(self, id)
    }
}

/// Checks that `list`, translated for a queue of `kind`, can record a
/// command that sets state for draws, which `does` describes for the
/// message.
///
/// # Panics
///
/// Panics, naming the list, if the queue isn't the graphics queue, the only
/// one that draws.
fn check_draws(list: &CommandList, kind: QueueKind, does: &str) {
    assert!(
        kind == QueueKind::Graphics,
        "command list \"{}\" {does}, but only the graphics queue draws, not the {} queue",
        list.label(),
        kind.name(),
    );
}

/// The labels of the programs last bound at each of Vulkan's bind points,
/// while a list is translated. Binding a program binds only its own
/// stages, so binding a compute program leaves a vertex and fragment
/// program bound, and the other way around.
#[derive(Debug, Default)]
struct BoundPrograms<'a> {
    /// The compute program, for dispatches.
    compute: Option<&'a str>,
    /// The vertex and fragment program, for draws.
    graphics: Option<&'a str>,
}

impl BoundPrograms<'_> {
    /// Checks that a draw in `list` has a vertex and fragment program to
    /// run.
    ///
    /// # Panics
    ///
    /// Panics, naming the list, if no vertex and fragment program is
    /// bound, as when it binds only compute programs.
    fn check_draw(&self, list: &CommandList) {
        assert!(
            self.graphics.is_some(),
            "command list \"{}\" draws with no vertex and fragment program bound",
            list.label(),
        );
    }

    /// Checks that a dispatch in `list` has a compute program to run.
    ///
    /// # Panics
    ///
    /// Panics, naming the list, if no compute program is bound, as when
    /// it binds only vertex and fragment programs.
    fn check_dispatch(&self, list: &CommandList) {
        assert!(
            self.compute.is_some(),
            "command list \"{}\" dispatches with no compute program bound",
            list.label(),
        );
    }
}

/// How `transfer`, a barrier's queue transfer, is recorded in the command
/// list `list` translated for the queue `translating`, on a device whose
/// queues are `queues`. Pure, so it's tested without a GPU.
///
/// # Panics
///
/// Panics, naming the list, if the transfer is between two families and
/// the list is translated for neither end's queue.
fn queue_transfer(
    queues: &impl QueueLookup,
    translating: QueueId,
    transfer: Option<(QueueId, QueueId)>,
    list: &str,
) -> Transfer {
    let Some((from, to)) = transfer else {
        return Transfer::None;
    };
    let (from_family, to_family) = (queues.family(from), queues.family(to));
    if from_family == to_family {
        Transfer::None
    } else if translating == from {
        Transfer::Release {
            from: from_family,
            to: to_family,
        }
    } else if translating == to {
        Transfer::Acquire {
            from: from_family,
            to: to_family,
        }
    } else {
        panic!(
            "command list \"{list}\" has a queue transfer from the {} queue to the {} queue, but \
             is translated for the {} queue",
            queues.name(from),
            queues.name(to),
            queues.name(translating),
        );
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of recording queue transfers (design 0002,
    //! "Synchronization").

    use super::*;

    /// A device's queues, each a family and a name, indexed by ID.
    struct FakeQueues(&'static [(u32, &'static str)]);

    impl QueueLookup for FakeQueues {
        fn family(&self, id: QueueId) -> u32 {
            self.0[id.index()].0
        }

        fn name(&self, id: QueueId) -> &'static str {
            self.0[id.index()].1
        }
    }

    /// A graphics queue.
    const GRAPHICS: QueueId = QueueId::new(0);

    /// A compute queue.
    const COMPUTE: QueueId = QueueId::new(1);

    /// A transfer queue.
    const TRANSFER: QueueId = QueueId::new(2);

    /// A device with graphics, compute and transfer queues on families of
    /// their own.
    const DEDICATED: FakeQueues = FakeQueues(&[(0, "graphics"), (2, "compute"), (1, "transfer")]);

    /// A device with a graphics and a compute queue on one family, which a
    /// later design may give a device.
    const ONE_FAMILY: FakeQueues = FakeQueues(&[(0, "graphics"), (0, "compute")]);

    /// From the transfer queue to the graphics queue.
    const UPLOAD: Option<(QueueId, QueueId)> = Some((TRANSFER, GRAPHICS));

    /// A transfer between families is a release on the queue it's from, and
    /// an acquire on the queue it's to, naming both families.
    #[test]
    fn transfers_between_families_release_and_acquire() {
        assert_eq!(
            queue_transfer(&DEDICATED, TRANSFER, UPLOAD, "upload"),
            Transfer::Release { from: 1, to: 0 },
        );
        assert_eq!(
            queue_transfer(&DEDICATED, GRAPHICS, UPLOAD, "upload"),
            Transfer::Acquire { from: 1, to: 0 },
        );
    }

    /// A transfer is dropped when both ends are one queue, as when every
    /// kind shares lavapipe's, or two queues on one family, as is a barrier
    /// with no transfer.
    #[test]
    fn transfers_within_a_family_are_dropped() {
        for queue in [GRAPHICS, COMPUTE] {
            for transfer in [(GRAPHICS, GRAPHICS), (COMPUTE, GRAPHICS)] {
                assert_eq!(
                    queue_transfer(&ONE_FAMILY, queue, Some(transfer), "upload"),
                    Transfer::None
                );
            }
        }
        for queue in [GRAPHICS, COMPUTE, TRANSFER] {
            assert_eq!(
                queue_transfer(&DEDICATED, queue, None, "upload"),
                Transfer::None
            );
        }
    }

    /// Indirect draws read from the first draw's arguments to the end of
    /// the last's, with their stride only for two draws or more, since
    /// Vulkan requires it to be no larger than the range.
    #[test]
    fn indirect_draw_ranges_span_every_draw() {
        for (draws, stride, size, range_stride) in [
            (0, 32, 0, 0),
            (1, 32, 16, 0),
            (2, 32, 48, 32),
            (3, 16, 48, 16),
        ] {
            let range = indirect_draws_range(GpuAddress(0x1000), draws, stride, 16);
            assert_eq!(
                (range.address, range.size, range.stride),
                (0x1000, size, range_stride),
                "{draws} draws {stride} bytes apart",
            );
        }
    }

    /// Each index format maps to the Vulkan index type of its size.
    #[test]
    fn index_formats_map_to_vulkan_index_types() {
        for (format, index_type) in [
            (IndexFormat::Uint16, vk::IndexType::UINT16),
            (IndexFormat::Uint32, vk::IndexType::UINT32),
        ] {
            assert_eq!(vk::IndexType::from(format), index_type, "{format:?}");
        }
    }

    /// Translating a transfer for a queue it doesn't involve panics, naming
    /// the list and all three queues.
    #[test]
    #[should_panic(
        expected = "command list \"upload\" has a queue transfer from the transfer queue to the \
                    graphics queue, but is translated for the compute queue"
    )]
    fn transfers_for_another_queue_panic() {
        queue_transfer(&DEDICATED, COMPUTE, UPLOAD, "upload");
    }
}
