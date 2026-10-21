//! GPU memory allocation (design 0002, "Memory allocator").
//!
//! Vulkan limits how many memory allocations a device may have at once, so
//! the allocator takes large blocks of memory from Vulkan and suballocates
//! from them. The suballocator is [`tlsf`], which is pure: it works on
//! offsets and sizes, with no Vulkan types, so it's tested without a GPU.

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the memory pools in rhi-memory's later PRs use it"
    )
)]
mod tlsf;
