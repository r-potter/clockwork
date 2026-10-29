//! Samplers: their handles, and how they're created (design 0002,
//! "Handles" and "Binding model").
//!
//! A sampler is only a slot in the backend's sampler heap, holding a
//! descriptor written from its description. Shaders reach it by its
//! [`DescriptorIndex`](crate::DescriptorIndex).

use crate::handle::Handle;

/// A sampler. Refers to an entry in the backend's sampler table, and is
/// stale once destroyed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sampler(Handle);

impl Sampler {
    /// The sampler for `handle`, issued by a backend's sampler table.
    pub(crate) fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    /// The handle into the backend's sampler table.
    pub(crate) fn handle(self) -> Handle {
        self.0
    }
}

/// How a sampler is created. Every mip is sampled, with no bias.
///
/// It holds filtering and address modes only. Anisotropy, depth comparison,
/// border colours, LOD clamps and bias each arrive with the code that needs
/// them (design 0002, "Binding model").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerDesc<'a> {
    /// The sampler's name, in messages.
    pub label: &'a str,
    /// Filtering when the texture is magnified.
    pub mag_filter: Filter,
    /// Filtering when the texture is minified.
    pub min_filter: Filter,
    /// Filtering between mips.
    pub mip_filter: Filter,
    /// What coordinates outside 0 to 1 sample, along u, v and w.
    pub address_modes: [AddressMode; 3],
}

/// How texels are filtered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Filter {
    /// The nearest texel, or the nearest mip.
    Nearest,
    /// A linear blend of the nearest texels, or of the two nearest mips.
    Linear,
}

impl Filter {
    /// Every filter.
    pub const ALL: [Self; 2] = [Self::Nearest, Self::Linear];
}

/// What a coordinate outside 0 to 1 samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// The texture tiles.
    Repeat,
    /// The texture tiles, mirrored on every other tile.
    MirroredRepeat,
    /// The texel at the nearest edge.
    ClampToEdge,
}

impl AddressMode {
    /// Every address mode.
    pub const ALL: [Self; 3] = [Self::Repeat, Self::MirroredRepeat, Self::ClampToEdge];
}

// Handles and descriptions are plain data, shared between threads (design
// 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Sampler>();
    assert_send_sync::<SamplerDesc<'_>>();
};
