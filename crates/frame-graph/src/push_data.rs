//! The push data of the passes the graph records itself, fullscreen passes
//! and compute passes it dispatches (design 0003, "Data structures and
//! layout", R14).
//!
//! Their bindings are laid out in order: a texture view or a sampler is a
//! 4-byte heap index, a buffer an 8-byte address aligned to 8, and a
//! constant 4 bytes (`float`, `int` or `uint`), or 16 aligned to 16 for a
//! `float4`, as Slang lays out a struct of them. Validation checks that a
//! pass's fit the RHI's [`MAX_PUSH_DATA_SIZE`](clockwork_rhi::MAX_PUSH_DATA_SIZE)
//! bytes by the same layout, and each frame the executor fills them from
//! the handles it bound, so the layout is in one place.

use clockwork_rhi::{DescriptorIndex, GpuAddress};

use crate::ResourceUse;
use crate::SamplerId;
use crate::pass::{Binding, Constant, PassDesc};

/// Each of `pass`'s bindings, in order, with where it goes in the push data
/// and how many bytes it takes.
fn laid_out(pass: &PassDesc) -> impl Iterator<Item = (Binding, u32, u32)> + '_ {
    let mut offset = 0_u32;
    pass.bindings.iter().map(move |&binding| {
        let (size, alignment) = match binding {
            Binding::Use(index) => match pass.uses[index].resource {
                ResourceUse::Buffer(_) => (8, 8),
                ResourceUse::Texture(_) | ResourceUse::Repeated(_) => (4, 4),
            },
            Binding::Sampler(_)
            | Binding::Constant(Constant::Float(_) | Constant::Int(_) | Constant::Uint(_)) => {
                (4, 4)
            }
            Binding::Constant(Constant::Float4(_)) => (16, 16),
        };
        let at = offset.next_multiple_of(alignment);
        offset = at.saturating_add(size);
        (binding, at, size)
    })
}

/// The size, in bytes, of the push data `pass`'s bindings lay out: the end
/// of the last, or 0 for none. It's a multiple of 4, as push data must be.
pub(crate) fn size(pass: &PassDesc) -> u32 {
    laid_out(pass)
        .last()
        .map_or(0, |(_, offset, size)| offset.saturating_add(size))
}

/// What a bound use of a resource puts in the push data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "`FrameGraph`, later in `graph-execute`, builds it"
    )
)]
pub(crate) enum PushValue {
    /// A texture view's heap index.
    Index(DescriptorIndex),
    /// A buffer's address.
    Address(GpuAddress),
}

/// Fills `data` with `pass`'s push data, in the host's byte order, as push
/// data is copied as written, keeping `data`'s memory. `resource` gives
/// what the use at each index puts there, and `sampler` each sampler's heap
/// index. The bytes between bindings are 0.
///
/// # Panics
///
/// Panics if `resource` gives an address for a texture or a heap index for
/// a buffer, which is a bug in the executor.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "`FrameGraph`, later in `graph-execute`, calls it")
)]
pub(crate) fn fill(
    pass: &PassDesc,
    mut resource: impl FnMut(usize) -> PushValue,
    mut sampler: impl FnMut(SamplerId) -> DescriptorIndex,
    data: &mut Vec<u8>,
) {
    data.clear();
    data.resize(size(pass) as usize, 0);
    for (binding, offset, size) in laid_out(pass) {
        let at = offset as usize;
        match binding {
            Binding::Use(index) => match (resource(index), size) {
                (PushValue::Index(DescriptorIndex(slot)), 4) => put(data, at, &slot.to_ne_bytes()),
                (PushValue::Address(GpuAddress(address)), 8) => {
                    put(data, at, &address.to_ne_bytes());
                }
                (bound, _) => panic!(
                    "use {index} of pass \"{}\" is bound as {bound:?}, which isn't what its \
                     resource binds",
                    pass.name
                ),
            },
            Binding::Sampler(id) => put(data, at, &sampler(id).0.to_ne_bytes()),
            Binding::Constant(Constant::Float(value)) => put(data, at, &value.to_ne_bytes()),
            Binding::Constant(Constant::Int(value)) => put(data, at, &value.to_ne_bytes()),
            Binding::Constant(Constant::Uint(value)) => put(data, at, &value.to_ne_bytes()),
            Binding::Constant(Constant::Float4(values)) => {
                for (index, value) in values.iter().enumerate() {
                    put(data, at + 4 * index, &value.to_ne_bytes());
                }
            }
        }
    }
}

/// Writes `bytes` into `data` at `at`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "`FrameGraph`, later in `graph-execute`, calls `fill`"
    )
)]
fn put(data: &mut [u8], at: usize, bytes: &[u8]) {
    data[at..at + bytes.len()].copy_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    //! T0 tests of the push data the graph's passes lay out and fill (R14).

    use clockwork_rhi::{Access, AddressMode, Extent3d, Filter, Format};

    use super::*;
    use crate::{
        Attachment, ComputeRecorder, GraphDesc, SamplerSettings, TextureSize, TransientTexture,
    };

    /// A fullscreen pass that binds a texture, a buffer, a sampler, a
    /// `float` and a `float4`, so the buffer and the `float4` are aligned
    /// past padding.
    fn graph() -> GraphDesc {
        let mut graph = GraphDesc::new("push data");
        let size = TextureSize::Fixed(Extent3d::new_2d(16, 16));
        let scene =
            graph.transient_texture("scene", TransientTexture::new(Format::Rgba16Float, size));
        let lights = graph.transient_buffer("lights", 64);
        let linear = graph.sampler(
            "linear",
            SamplerSettings::new(Filter::Linear, AddressMode::ClampToEdge),
        );
        let output = graph.output();
        graph
            .fullscreen_pass("composite", "composite")
            .bind(scene, Access::FragmentShaderSampledRead)
            .bind(lights, Access::FragmentShaderRead)
            .bind_sampler(linear)
            .bind_constant(Constant::Float(0.5))
            .bind_constant(Constant::Float4([1.0, 2.0, 3.0, 4.0]))
            .colour(output, Attachment::Discard);
        graph
            .compute_pass("measure", ComputeRecorder::recorder("measure"))
            .read(scene, Access::ComputeShaderSampledRead);
        graph
    }

    /// The scene's heap index, the lights' address and the sampler's heap
    /// index, as an executor would look them up.
    fn bound(index: usize) -> PushValue {
        match index {
            0 => PushValue::Index(DescriptorIndex(7)),
            _ => PushValue::Address(GpuAddress(0x1122_3344_5566_7788)),
        }
    }

    /// Bindings are laid out in order, a buffer's address aligned to 8 and
    /// a `float4` to 16, and filled with what each binds, with 0 between
    /// them.
    #[test]
    fn bindings_are_laid_out_and_filled_in_order() {
        let graph = graph();
        let pass = &graph.passes[0];
        // The index at 0, padding to 8, the address at 8, the sampler at 16,
        // the float at 20, padding to 32, and the float4 at 32.
        assert_eq!(size(pass), 48);
        let mut data = vec![0xff; 3];
        fill(pass, bound, |_| DescriptorIndex(3), &mut data);
        let mut expected = Vec::new();
        expected.extend(7_u32.to_ne_bytes());
        expected.extend([0; 4]);
        expected.extend(0x1122_3344_5566_7788_u64.to_ne_bytes());
        expected.extend(3_u32.to_ne_bytes());
        expected.extend(0.5_f32.to_ne_bytes());
        expected.extend([0; 8]);
        for value in [1.0_f32, 2.0, 3.0, 4.0] {
            expected.extend(value.to_ne_bytes());
        }
        assert_eq!(data, expected);
    }

    /// A pass with no bindings has no push data, and filling it empties
    /// what was there.
    #[test]
    fn no_bindings_have_no_push_data() {
        let graph = graph();
        let pass = &graph.passes[1];
        assert_eq!(size(pass), 0);
        let mut data = vec![1, 2, 3, 4];
        fill(pass, bound, |_| DescriptorIndex(0), &mut data);
        assert_eq!(data, Vec::<u8>::new());
    }

    /// A texture given an address, rather than a heap index, is a bug in
    /// the executor, and panics, naming the pass.
    #[test]
    #[should_panic(
        expected = "use 0 of pass \"composite\" is bound as Address(GpuAddress(1)), which isn't \
                    what its resource binds"
    )]
    fn a_texture_bound_as_an_address_panics() {
        let graph = graph();
        let mut data = Vec::new();
        fill(
            &graph.passes[0],
            |_| PushValue::Address(GpuAddress(1)),
            |_| DescriptorIndex(0),
            &mut data,
        );
    }
}
