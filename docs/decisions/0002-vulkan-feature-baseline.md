# 0002: Target a bleeding-edge Vulkan feature set with no fallbacks

- **Status:** Proposed
- **Date:** 2026-10-15

## Context

The engine is a showcase of current techniques. Supporting older GPUs or drivers would mean keeping a second binding model (descriptor sets), pipelines alongside shader objects, and render passes alongside dynamic rendering, each needing its own tests. The development GPU, an RTX 3090 on NVIDIA 615.71.09, supports the whole modern feature set (checked with `vulkaninfo` on 2026-10-15).

## Decision

**Required:**
- Vulkan 1.4
- `VK_EXT_descriptor_heap`: all shader-visible images and samplers live in descriptor heaps. There are no descriptor sets, pools or set layouts.
- `VK_KHR_device_address_commands`: buffers are identified by device address throughout.
- Buffer device addresses and timeline semaphores (core since 1.2)
- Dynamic rendering and synchronization2 (core since 1.3). There are no render passes or framebuffers.
- `VK_EXT_shader_object`. There are no pipelines, pipeline layouts or pipeline caches.

**Optional, with a fallback:**
- `VK_EXT_present_timing`. Without it, frame pacing relies on FIFO presentation.
- `VK_KHR_swapchain_maintenance1`. Without it, swapchain recreation waits for the device to be idle.

The exact list is maintained in device selection's `REQUIREMENTS` (`src/rhi/vulkan/device_selection.rs`), which is the source of truth. The requirements are added there as the subsystems that need them are implemented.

## Consequences

- Devices without the required extensions are rejected at start-up, with the reasons logged. As of this date that includes every released Mesa lavapipe (26.2.5 and older), and every non-recent driver.
- **Testing:** there's no GPU in CI, so CI runs T1 tests on lavapipe built from a pinned Mesa `main` commit, which supports every required extension (Mesa's `docs/features.txt`, 2026-10-15). Lavapipe has one queue family and no present timing, so those paths are tested only by the hardware checks before review. T1 tests that need a missing capability skip, and `CLOCKWORK_REQUIRED_CAPABILITIES` makes a skip fail for any capability the device is expected to have (see design 0001).
- `ash` 0.38 on crates.io lacks the bindings, so `ash` is pinned to a git revision (`0001-foundations`).
- Validation can't see accesses through descriptor heaps, so GPU-assisted validation is part of the test process for shader and binding changes.
- Older desktop GPUs and drivers are out of reach. Android (through Turnip) and macOS and iOS (through KosmicKrisp) are planned once those drivers support the required extensions: see ADR 0003. MoltenVK isn't planned, because it implements a more limited subset of Vulkan.
