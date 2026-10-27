//! Buffers: their handles, how they're created, and their addresses (design
//! 0002, "Handles", "Memory" and "Binding model").
//!
//! A buffer is never bound. Shaders, copies, index buffers and indirect
//! arguments all reach it through its [`GpuAddress`].

use crate::MemoryLocation;
use crate::handle::Handle;

/// A buffer. Refers to an entry in the backend's buffer table, and is stale
/// once destroyed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Buffer(Handle);

impl Buffer {
    /// The buffer for `handle`, issued by a backend's buffer table.
    pub(crate) fn from_handle(handle: Handle) -> Self {
        Self(handle)
    }

    /// The handle into the backend's buffer table.
    pub(crate) fn handle(self) -> Handle {
        self.0
    }
}

/// How a buffer is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc<'a> {
    /// The buffer's name, in debug tools and messages.
    pub label: &'a str,
    /// The size, in bytes. Must be greater than zero.
    pub size: u64,
    /// How it's used, beyond copies, which every buffer allows.
    pub usage: BufferUsage,
    /// Where its memory is.
    pub location: MemoryLocation,
}

/// How a buffer is used, beyond copies and its device address, which every
/// buffer allows. A set of the constants below, combined with
/// [`union`](Self::union).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct BufferUsage {
    /// Read and written by shaders, through its address.
    storage: bool,
    /// Indices for indexed draws.
    index: bool,
    /// Arguments for indirect draws and dispatches.
    indirect: bool,
}

impl BufferUsage {
    /// Copies only, such as an upload or readback buffer.
    pub const COPIES: Self = Self {
        storage: false,
        index: false,
        indirect: false,
    };
    /// Read and written by shaders, through its address.
    pub const STORAGE: Self = Self {
        storage: true,
        ..Self::COPIES
    };
    /// Indices for indexed draws.
    pub const INDEX: Self = Self {
        index: true,
        ..Self::COPIES
    };
    /// Arguments for indirect draws and dispatches.
    pub const INDIRECT: Self = Self {
        indirect: true,
        ..Self::COPIES
    };

    /// The usages in either set.
    pub const fn union(self, other: Self) -> Self {
        Self {
            storage: self.storage || other.storage,
            index: self.index || other.index,
            indirect: self.indirect || other.indirect,
        }
    }

    /// Whether every usage in `other` is in this set.
    pub const fn contains(self, other: Self) -> bool {
        (self.storage || !other.storage)
            && (self.index || !other.index)
            && (self.indirect || !other.indirect)
    }
}

/// A buffer's address on the GPU, for shaders, copies and indirect
/// arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuAddress(pub u64);

// Handles and descriptions are plain data, shared between threads (design
// 0002, "Threading", R15).
const _: () = {
    /// Compiles only for types that are `Send` and `Sync`.
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Buffer>();
    assert_send_sync::<BufferDesc<'_>>();
    assert_send_sync::<GpuAddress>();
};

#[cfg(test)]
mod tests {
    //! T0 tests of buffer usage sets.

    use super::*;

    /// A union contains each of its parts, and no other usage.
    #[test]
    fn unions_contain_exactly_their_parts() {
        let usage = BufferUsage::STORAGE.union(BufferUsage::INDIRECT);
        assert!(usage.contains(BufferUsage::STORAGE));
        assert!(usage.contains(BufferUsage::INDIRECT));
        assert!(!usage.contains(BufferUsage::INDEX));
        assert!(usage.contains(BufferUsage::COPIES));
        assert!(!BufferUsage::COPIES.contains(BufferUsage::STORAGE));
        assert_eq!(BufferUsage::default(), BufferUsage::COPIES);
    }
}
