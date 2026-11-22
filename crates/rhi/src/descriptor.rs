//! The descriptor heaps as the application sees them (design 0002,
//! "Binding model").
//!
//! The backend owns two heaps, one for texture views and one for samplers.
//! Shaders reach each view or sampler by its slot in its heap, a
//! [`DescriptorIndex`].

use std::fmt;

/// A descriptor heap: one for texture views, and one for samplers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeapKind {
    /// The heap of texture views, for shaders to sample and write.
    Resource,
    /// The heap of samplers.
    Sampler,
}

impl fmt::Display for HeapKind {
    /// The heap's name in messages: `resource heap` or `sampler heap`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Resource => "resource heap",
            Self::Sampler => "sampler heap",
        })
    }
}

/// A slot in a descriptor heap, for shaders to index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorIndex(pub u32);

// Heap slots are plain data, written into push data on any thread (design
// 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HeapKind>();
    assert_send_sync::<DescriptorIndex>();
};
