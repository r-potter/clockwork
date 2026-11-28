//! Recorders, the Rust code that records the passes a technique leaves to
//! code, and what each is told about its pass (design 0003, "Recorders").
//!
//! The renderer registers each recorder by name and kind in
//! [`Recorders<D>`], where `D` is the frame's data, such as the render
//! snapshot, which every recorder reads and none writes. A pass names its
//! recorder. Each frame, a recorder is given a [`PassContext`], the frame's
//! data, and the commands its kind of pass may record.
//!
//! A recorder returns an [`RhiResult`], so an error from the RHI, such as a
//! full descriptor heap when a view is first indexed, reaches the caller of
//! the frame graph's `execute` rather than panicking.

use std::collections::BTreeMap;
use std::fmt;

use clockwork_rhi::{
    Buffer, DescriptorIndex, Extent2d, GpuAddress, GpuResources, Program, RhiResult, Sampler,
    Texture, TextureView, TransientFrame,
};

use crate::commands::Bound;
use crate::pass::PassDesc;
use crate::recording::{ComputeCommands, RasterCommands, TransferCommands};
use crate::{GraphDesc, ResourceUse};

/// A raster pass's recorder.
pub(crate) type RasterRecordFn<D> =
    Box<dyn Fn(&PassContext<'_>, &D, &mut RasterCommands<'_>) -> RhiResult<()> + Send + Sync>;

/// A compute pass's recorder.
pub(crate) type ComputeRecordFn<D> =
    Box<dyn Fn(&PassContext<'_>, &D, &mut ComputeCommands<'_>) -> RhiResult<()> + Send + Sync>;

/// A transfer pass's recorder.
pub(crate) type TransferRecordFn<D> =
    Box<dyn Fn(&PassContext<'_>, &D, &mut TransferCommands<'_>) -> RhiResult<()> + Send + Sync>;

/// The recorders for the passes a technique leaves to code, by name and
/// kind. `D` is the frame's data, which every recorder reads.
///
/// It's shared by every recording job, so each recorder is `Fn`, `Send` and
/// `Sync`.
pub struct Recorders<D> {
    /// The raster passes' recorders, by name.
    raster: BTreeMap<String, RasterRecordFn<D>>,
    /// The compute passes' recorders, by name.
    compute: BTreeMap<String, ComputeRecordFn<D>>,
    /// The transfer passes' recorders, by name.
    transfer: BTreeMap<String, TransferRecordFn<D>>,
}

impl<D> Recorders<D> {
    /// No recorders.
    pub fn new() -> Self {
        Self {
            raster: BTreeMap::new(),
            compute: BTreeMap::new(),
            transfer: BTreeMap::new(),
        }
    }

    /// Registers `record` as the recorder of raster passes that name
    /// `name`.
    ///
    /// # Panics
    ///
    /// Panics if a raster recorder is already registered as `name`.
    pub fn raster(
        &mut self,
        name: &str,
        record: impl Fn(&PassContext<'_>, &D, &mut RasterCommands<'_>) -> RhiResult<()>
        + Send
        + Sync
        + 'static,
    ) {
        register(&mut self.raster, "raster", name, Box::new(record));
    }

    /// Registers `record` as the recorder of compute passes that name
    /// `name`.
    ///
    /// # Panics
    ///
    /// Panics if a compute recorder is already registered as `name`.
    pub fn compute(
        &mut self,
        name: &str,
        record: impl Fn(&PassContext<'_>, &D, &mut ComputeCommands<'_>) -> RhiResult<()>
        + Send
        + Sync
        + 'static,
    ) {
        register(&mut self.compute, "compute", name, Box::new(record));
    }

    /// Registers `record` as the recorder of transfer passes that name
    /// `name`.
    ///
    /// # Panics
    ///
    /// Panics if a transfer recorder is already registered as `name`.
    pub fn transfer(
        &mut self,
        name: &str,
        record: impl Fn(&PassContext<'_>, &D, &mut TransferCommands<'_>) -> RhiResult<()>
        + Send
        + Sync
        + 'static,
    ) {
        register(&mut self.transfer, "transfer", name, Box::new(record));
    }

    /// The raster recorder registered as `name`, if there's one.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
    )]
    pub(crate) fn raster_recorder(&self, name: &str) -> Option<&RasterRecordFn<D>> {
        self.raster.get(name)
    }

    /// The compute recorder registered as `name`, if there's one.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
    )]
    pub(crate) fn compute_recorder(&self, name: &str) -> Option<&ComputeRecordFn<D>> {
        self.compute.get(name)
    }

    /// The transfer recorder registered as `name`, if there's one.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
    )]
    pub(crate) fn transfer_recorder(&self, name: &str) -> Option<&TransferRecordFn<D>> {
        self.transfer.get(name)
    }
}

impl<D> Default for Recorders<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D> fmt::Debug for Recorders<D> {
    /// Lists the recorders' names by kind, since the recorders themselves
    /// are closures.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Recorders")
            .field("raster", &self.raster.keys().collect::<Vec<_>>())
            .field("compute", &self.compute.keys().collect::<Vec<_>>())
            .field("transfer", &self.transfer.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Adds `record` to `recorders`, of passes of `kind`, as `name`.
///
/// # Panics
///
/// Panics if one is already registered as `name`.
fn register<R>(recorders: &mut BTreeMap<String, R>, kind: &str, name: &str, record: R) {
    let replaced = recorders.insert(name.to_owned(), record);
    assert!(
        replaced.is_none(),
        "a {kind} recorder is already registered as \"{name}\""
    );
}

/// What a recorder can find out about its pass: its name and size, the
/// handles of the resources it declared, the technique's programs and
/// samplers, all by name, and the frame's transient memory.
///
/// A lookup of a resource the pass didn't declare, or of a program or
/// sampler the technique doesn't have, is a bug in the recorder, and
/// panics, naming the pass.
#[derive(Debug)]
pub struct PassContext<'a> {
    /// The pass's names and handles.
    names: PassNames<'a>,
    /// The resources, for heap indices and addresses.
    resources: &'a dyn GpuResources,
    /// The frame's transient memory.
    transient: &'a TransientFrame,
}

impl<'a> PassContext<'a> {
    /// The context of the pass `names` describes, looking up heap indices
    /// and addresses in `resources`, with the frame's transient memory
    /// `transient`.
    #[expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")]
    pub(crate) fn new(
        names: PassNames<'a>,
        resources: &'a dyn GpuResources,
        transient: &'a TransientFrame,
    ) -> Self {
        Self {
            names,
            resources,
            transient,
        }
    }

    /// The pass's name.
    pub fn name(&self) -> &str {
        &self.names.pass.name
    }

    /// The size of the pass's attachments, or for a pass with none, the
    /// output's.
    pub fn extent(&self) -> Extent2d {
        self.names.extent
    }

    /// The texture the pass declared as `name`.
    ///
    /// # Panics
    ///
    /// Panics if the pass declared no use of a texture named `name`.
    pub fn texture(&self, name: &str) -> Texture {
        self.names.texture(name)
    }

    /// The heap index of `view` of the texture the pass declared as `name`,
    /// for a shader to index.
    ///
    /// # Errors
    ///
    /// Returns an error if the view's descriptor can't be written, such as
    /// when the resource heap is full.
    ///
    /// # Panics
    ///
    /// Panics if the pass declared no use of a texture named `name`, or if
    /// the view isn't one the texture allows.
    pub fn texture_index(&self, name: &str, view: TextureView) -> RhiResult<DescriptorIndex> {
        self.resources.texture_index(self.texture(name), view)
    }

    /// The address of the buffer the pass declared as `name`.
    ///
    /// # Panics
    ///
    /// Panics if the pass declared no use of a buffer named `name`.
    pub fn buffer_address(&self, name: &str) -> GpuAddress {
        self.resources.buffer_address(self.names.buffer(name))
    }

    /// The technique's program named `name`.
    ///
    /// # Panics
    ///
    /// Panics if the technique has no program named `name`.
    pub fn program(&self, name: &str) -> Program {
        self.names.program(name)
    }

    /// The heap index of the graph's sampler named `name`, for a shader to
    /// index.
    ///
    /// # Panics
    ///
    /// Panics if the graph has no sampler named `name`.
    pub fn sampler_index(&self, name: &str) -> DescriptorIndex {
        self.resources.sampler_index(self.names.sampler(name))
    }

    /// The frame's transient memory, which the pass allocates per-draw
    /// data from.
    pub fn transient(&self) -> &TransientFrame {
        self.transient
    }
}

/// A pass's names, resolved to the handles the frame bound: the part of a
/// [`PassContext`] that needs no device.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PassNames<'a> {
    /// The description, whose resources and samplers are named.
    pub(crate) desc: &'a GraphDesc,
    /// The pass, expanded, whose uses are those it declared.
    pub(crate) pass: &'a PassDesc,
    /// The size of its attachments, or the output's for a pass with none.
    pub(crate) extent: Extent2d,
    /// The handles the frame bound for the plan's resources.
    pub(crate) bound: &'a Bound,
    /// The technique's programs, by name.
    pub(crate) programs: &'a [(String, Program)],
    /// The graph's samplers, indexed by `SamplerId`.
    pub(crate) samplers: &'a [Sampler],
}

impl PassNames<'_> {
    /// The texture the pass declared as `name`.
    ///
    /// # Panics
    ///
    /// Panics, naming the pass, if it declared no use of a texture named
    /// `name`.
    fn texture(&self, name: &str) -> Texture {
        let declared = self.desc.textures.iter().position(|texture| texture.name == name).filter(
            |&index| {
                self.pass.uses.iter().any(|used| {
                    matches!(&used.resource, ResourceUse::Texture(part) if part.texture.index() == index)
                })
            },
        );
        let index = declared.unwrap_or_else(|| {
            panic!(
                "pass \"{}\" looks up the texture \"{name}\", which it doesn't use",
                self.pass.name
            )
        });
        self.bound.textures[index]
            .expect("every texture a pass that runs uses is bound")
            .texture
    }

    /// The buffer the pass declared as `name`.
    ///
    /// # Panics
    ///
    /// Panics, naming the pass, if it declared no use of a buffer named
    /// `name`.
    fn buffer(&self, name: &str) -> Buffer {
        let declared = self
            .desc
            .buffers
            .iter()
            .position(|buffer| buffer.name == name)
            .filter(|&index| {
                self.pass.uses.iter().any(|used| {
                    matches!(used.resource, ResourceUse::Buffer(buffer) if buffer.index() == index)
                })
            });
        let index = declared.unwrap_or_else(|| {
            panic!(
                "pass \"{}\" looks up the buffer \"{name}\", which it doesn't use",
                self.pass.name
            )
        });
        self.bound.buffers[index]
            .expect("every buffer a pass that runs uses is bound")
            .buffer
    }

    /// The technique's program named `name`.
    ///
    /// # Panics
    ///
    /// Panics, naming the pass, if the technique has none.
    fn program(&self, name: &str) -> Program {
        self.programs
            .iter()
            .find(|(program, _)| program == name)
            .map(|&(_, program)| program)
            .unwrap_or_else(|| {
                panic!(
                    "pass \"{}\" looks up the program \"{name}\", which the technique doesn't have",
                    self.pass.name
                )
            })
    }

    /// The graph's sampler named `name`.
    ///
    /// # Panics
    ///
    /// Panics, naming the pass, if the graph has none.
    fn sampler(&self, name: &str) -> Sampler {
        let index = self
            .desc
            .samplers
            .iter()
            .position(|sampler| sampler.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "pass \"{}\" looks up the sampler \"{name}\", which the graph doesn't have",
                    self.pass.name
                )
            });
        self.samplers[index]
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of registering recorders, and of a pass's names.

    use clockwork_rhi::test_support::{
        unissued_buffer, unissued_program, unissued_sampler, unissued_texture,
    };
    use clockwork_rhi::{Access, AddressMode, Extent3d, Filter, Format};

    use super::*;
    use crate::commands::{BoundBuffer, BoundTexture};
    use crate::{Attachment, SamplerSettings, TextureSize, TransientTexture};

    /// Recorders are found by kind and name, and a name may be used once
    /// for each kind.
    #[test]
    fn recorders_are_found_by_kind_and_name() {
        let mut recorders = Recorders::<u32>::new();
        recorders.raster("scene", |_, _, commands| {
            commands.draw(3, 1);
            Ok(())
        });
        recorders.compute("scene", |_, _, _| Ok(()));
        recorders.transfer("readback", |_, _, _| Ok(()));
        assert!(recorders.raster_recorder("scene").is_some());
        assert!(recorders.compute_recorder("scene").is_some());
        assert!(recorders.transfer_recorder("scene").is_none());
        assert!(recorders.transfer_recorder("readback").is_some());
        assert!(recorders.raster_recorder("readback").is_none());
        assert_eq!(
            format!("{recorders:?}"),
            "Recorders { raster: [\"scene\"], compute: [\"scene\"], transfer: [\"readback\"] }"
        );
    }

    /// Registering a second recorder of one kind under a name panics.
    #[test]
    #[should_panic(expected = "a compute recorder is already registered as \"cull\"")]
    fn a_name_is_registered_once_for_each_kind() {
        let mut recorders = Recorders::<()>::default();
        recorders.compute("cull", |_, _, _| Ok(()));
        recorders.compute("cull", |_, _, _| Ok(()));
    }

    /// A graph with a pass "draw" that uses the texture "scene" and the
    /// buffer "lights", but not the texture "history" or the buffer
    /// "unused", and a sampler, "linear".
    fn graph() -> GraphDesc {
        let mut graph = GraphDesc::new("names");
        let size = TextureSize::Fixed(Extent3d::new_2d(16, 16));
        let texture = TransientTexture::new(Format::Rgba16Float, size);
        let scene = graph.transient_texture("scene", texture);
        graph.persistent_texture("history", texture);
        let lights = graph.transient_buffer("lights", 64);
        graph.transient_buffer("unused", 64);
        graph.sampler(
            "linear",
            SamplerSettings::new(Filter::Linear, AddressMode::ClampToEdge),
        );
        graph
            .raster_pass("draw", "draw")
            .read(lights, Access::FragmentShaderRead)
            .colour(scene, Attachment::Clear([0.0; 4]));
        graph
    }

    /// The handles a frame binds for [`graph`]: the output, scene and
    /// history are textures 0 to 2, and lights and unused buffers 0 and 1.
    fn bound() -> Bound {
        let texture = |index| {
            Some(BoundTexture {
                texture: unissued_texture(index),
                mips: 1,
                layers: 1,
                fresh: false,
            })
        };
        let buffer = |index| {
            Some(BoundBuffer {
                buffer: unissued_buffer(index),
                fresh: false,
            })
        };
        Bound {
            textures: vec![texture(0), texture(1), texture(2)],
            buffers: vec![buffer(0), buffer(1)],
        }
    }

    /// Calls `check` with the names of [`graph`]'s pass "draw", with
    /// [`bound`]'s handles, the program "tone map", and the sampler.
    fn with_names(check: impl FnOnce(PassNames<'_>)) {
        let desc = graph();
        let bound = bound();
        let programs = [("tone map".to_owned(), unissued_program(4))];
        let samplers = [unissued_sampler(2)];
        check(PassNames {
            desc: &desc,
            pass: &desc.passes[0],
            extent: Extent2d {
                width: 16,
                height: 16,
            },
            bound: &bound,
            programs: &programs,
            samplers: &samplers,
        });
    }

    /// A pass's names resolve to the frame's handles for the resources it
    /// declared, and to the technique's programs and the graph's samplers.
    #[test]
    fn names_resolve_to_the_frames_handles() {
        with_names(|names| {
            assert_eq!(names.texture("scene"), unissued_texture(1));
            assert_eq!(names.buffer("lights"), unissued_buffer(0));
            assert_eq!(names.program("tone map"), unissued_program(4));
            assert_eq!(names.sampler("linear"), unissued_sampler(2));
        });
    }

    /// Looking up a texture the pass doesn't use panics, naming the pass,
    /// whether the graph has it or not.
    #[test]
    #[should_panic(
        expected = "pass \"draw\" looks up the texture \"history\", which it doesn't use"
    )]
    fn a_texture_the_pass_doesnt_use_isnt_found() {
        with_names(|names| {
            names.texture("history");
        });
    }

    /// Looking up a buffer the pass doesn't use panics, naming the pass.
    #[test]
    #[should_panic(expected = "pass \"draw\" looks up the buffer \"unused\", which it doesn't use")]
    fn a_buffer_the_pass_doesnt_use_isnt_found() {
        with_names(|names| {
            names.buffer("unused");
        });
    }

    /// Looking up a program the technique doesn't have panics, naming the
    /// pass.
    #[test]
    #[should_panic(
        expected = "pass \"draw\" looks up the program \"bloom\", which the technique doesn't have"
    )]
    fn a_program_the_technique_doesnt_have_isnt_found() {
        with_names(|names| {
            names.program("bloom");
        });
    }

    /// Looking up a sampler the graph doesn't have panics, naming the pass.
    #[test]
    #[should_panic(
        expected = "pass \"draw\" looks up the sampler \"nearest\", which the graph doesn't have"
    )]
    fn a_sampler_the_graph_doesnt_have_isnt_found() {
        with_names(|names| {
            names.sampler("nearest");
        });
    }

    /// Recorders are shared by every recording job, whatever the frame's
    /// data, and a pass's context is read by the job that records it.
    #[test]
    fn recorders_and_contexts_are_shared_between_threads() {
        /// Compiles only if `T` is `Send` and `Sync`.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Recorders<std::cell::Cell<u32>>>();
        assert_send_sync::<PassContext<'_>>();
    }
}
