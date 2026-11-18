//! A fake of [`GpuResources`] for the uploader's T0 tests, over host
//! memory. Compiled only for tests.
//!
//! It fakes only what the uploader uses: buffers, whose handles it issues
//! from a handle table so that a stale handle panics as on a real backend.
//! The other methods panic.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::handle::HandleTable;
use crate::{
    Buffer, BufferDesc, BufferUsage, Capabilities, DescriptorIndex, GpuAddress, GpuResources,
    MemoryLocation, Program, ProgramDesc, RhiError, RhiResult, Sampler, SamplerDesc, Texture,
    TextureDesc, TextureInfo, TextureView, TransientFrame,
};

/// A buffer the fake created.
#[derive(Debug)]
pub(crate) struct FakeBuffer {
    /// Its address, which the fake chooses.
    pub(crate) address: GpuAddress,
    /// Its usage.
    pub(crate) usage: BufferUsage,
    /// Where its memory is.
    pub(crate) location: MemoryLocation,
    /// Its contents, as long as it is.
    pub(crate) bytes: Vec<u8>,
}

/// What the fake holds.
#[derive(Debug)]
struct State {
    /// Issues buffer handles, and finds stale ones.
    handles: HandleTable<()>,
    /// The buffers not yet destroyed.
    buffers: HashMap<Buffer, FakeBuffer>,
    /// How many buffers it has created.
    created: u64,
    /// The sizes of the buffers it destroyed, in the order it destroyed
    /// them.
    destroyed: Vec<u64>,
    /// How many more buffers it creates before failing, or `None` for no
    /// limit.
    remaining: Option<u64>,
}

/// A fake of [`GpuResources`] over host memory.
#[derive(Debug)]
pub(crate) struct FakeResources {
    /// The device's capabilities.
    capabilities: Capabilities,
    /// How far past a multiple of 2^32 each buffer's address is, so tests
    /// can check that writes are aligned by address rather than by offset.
    misalignment: u64,
    /// The buffers, and what's been done to them.
    state: Mutex<State>,
}

impl FakeResources {
    /// A fake whose buffers each start `misalignment` bytes past a multiple
    /// of 2^32.
    pub(crate) fn new(misalignment: u64) -> Self {
        Self {
            capabilities: Capabilities::for_tests(true),
            misalignment,
            state: Mutex::new(State {
                handles: HandleTable::new("buffer"),
                buffers: HashMap::new(),
                created: 0,
                destroyed: Vec::new(),
                remaining: None,
            }),
        }
    }

    /// Fails to create buffers once it has created `count` more.
    pub(crate) fn fail_after(&self, count: u64) {
        self.state().remaining = Some(count);
    }

    /// How many buffers it has created.
    pub(crate) fn created(&self) -> u64 {
        self.state().created
    }

    /// The sizes of the buffers it has destroyed, in the order it destroyed
    /// them.
    pub(crate) fn destroyed(&self) -> Vec<u64> {
        self.state().destroyed.clone()
    }

    /// How many buffers it holds that aren't destroyed.
    pub(crate) fn live(&self) -> usize {
        self.state().buffers.len()
    }

    /// Calls `inspect` with the buffer, if it hasn't been destroyed.
    pub(crate) fn with_buffer<R>(
        &self,
        buffer: Buffer,
        inspect: impl FnOnce(&FakeBuffer) -> R,
    ) -> R {
        let state = self.state();
        state.handles.get(buffer.handle());
        inspect(&state.buffers[&buffer])
    }

    /// The bytes at `address`, `len` long, in whichever live buffer holds
    /// them.
    ///
    /// # Panics
    ///
    /// Panics if no live buffer holds them all.
    pub(crate) fn read(&self, address: GpuAddress, len: usize) -> Vec<u8> {
        let state = self.state();
        state
            .buffers
            .values()
            .find_map(|buffer| {
                let start = usize::try_from(address.0.checked_sub(buffer.address.0)?).ok()?;
                buffer
                    .bytes
                    .get(start..start.checked_add(len)?)
                    .map(<[u8]>::to_vec)
            })
            .unwrap_or_else(|| panic!("no buffer holds {len} bytes at {address:?}"))
    }

    /// The fake's state.
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl GpuResources for FakeResources {
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn create_buffer(&self, desc: &BufferDesc<'_>) -> RhiResult<Buffer> {
        desc.validate();
        let mut state = self.state();
        if let Some(remaining) = &mut state.remaining {
            if *remaining == 0 {
                return Err(RhiError::OutOfDeviceMemory {
                    label: desc.label.to_owned(),
                });
            }
            *remaining -= 1;
        }
        state.created += 1;
        let address = GpuAddress((state.created << 32) + self.misalignment);
        let buffer = FakeBuffer {
            address,
            usage: desc.usage,
            location: desc.location,
            bytes: vec![0; usize::try_from(desc.size).expect("fake buffers fit in memory")],
        };
        let handle = Buffer::from_handle(state.handles.insert(desc.label, ()));
        state.buffers.insert(handle, buffer);
        Ok(handle)
    }

    fn destroy_buffer(&self, buffer: Buffer) {
        let mut state = self.state();
        state.handles.remove(buffer.handle());
        let buffer = state
            .buffers
            .remove(&buffer)
            .expect("live handles have buffers");
        state.destroyed.push(buffer.bytes.len() as u64);
    }

    fn buffer_address(&self, buffer: Buffer) -> GpuAddress {
        self.with_buffer(buffer, |buffer| buffer.address)
    }

    fn write_buffer(&self, buffer: Buffer, offset: u64, data: &[u8]) {
        let mut state = self.state();
        state.handles.get(buffer.handle());
        let buffer = state
            .buffers
            .get_mut(&buffer)
            .expect("live handles have buffers");
        assert_eq!(
            buffer.location,
            MemoryLocation::Upload,
            "only upload buffers are written"
        );
        let start = usize::try_from(offset).expect("fake buffers fit in memory");
        buffer.bytes[start..start + data.len()].copy_from_slice(data);
    }

    fn read_buffer(&self, _: Buffer, _: u64, _: &mut [u8]) -> RhiResult<()> {
        unimplemented!("the uploader doesn't read buffers")
    }

    fn create_texture(&self, _: &TextureDesc<'_>) -> RhiResult<Texture> {
        unimplemented!("the uploader doesn't create textures")
    }

    fn destroy_texture(&self, _: Texture) {
        unimplemented!("the uploader doesn't destroy textures")
    }

    fn texture_info(&self, _: Texture) -> TextureInfo {
        unimplemented!("the fake has no textures yet")
    }

    fn texture_index(&self, _: Texture, _: TextureView) -> RhiResult<DescriptorIndex> {
        unimplemented!("the uploader doesn't index textures")
    }

    fn create_sampler(&self, _: &SamplerDesc<'_>) -> RhiResult<Sampler> {
        unimplemented!("the uploader doesn't create samplers")
    }

    fn destroy_sampler(&self, _: Sampler) {
        unimplemented!("the uploader doesn't destroy samplers")
    }

    fn sampler_index(&self, _: Sampler) -> DescriptorIndex {
        unimplemented!("the uploader doesn't index samplers")
    }

    fn create_program(&self, _: &ProgramDesc<'_>) -> RhiResult<Program> {
        unimplemented!("the uploader doesn't create programs")
    }

    fn destroy_program(&self, _: Program) {
        unimplemented!("the uploader doesn't destroy programs")
    }

    fn collect_garbage(&self) {
        unimplemented!("the uploader doesn't collect garbage")
    }

    fn begin_transient_frame(&self) -> TransientFrame {
        unimplemented!("the uploader keeps its own staging buffers")
    }

    fn end_transient_frame(&self, _: TransientFrame) {
        unimplemented!("the uploader keeps its own staging buffers")
    }
}
