# TODO

Outstanding tasks and future work.

## Vulkan

- [ ] **Update to the Vulkan 1.4 bindings.** The backend already requests
      Vulkan 1.4, but `ash` 0.38 (the latest crates.io release) only has
      bindings for Vulkan 1.3.281. Newer bindings are only on `ash`'s `master`
      branch, so this requires switching `ash` to a git dependency pinned to a
      commit (`rev = "..."`). Once switched:
  - replace `vk::make_api_version(0, 1, 4, 0)` with `vk::API_VERSION_1_4` in
    `src/rhi/vulkan.rs`
  - add the Vulkan 1.4 features and properties to `PhysicalDevice` in
    `src/rhi/vulkan/physical_device.rs`
  - update any API changes between 0.38 and `master`
- [ ] **Choose a physical device to render with.** `VulkanRenderHardwareInterface::new`
      currently enumerates and logs the devices, then drops them (see the
      `TODO` in `src/rhi/vulkan.rs`). Selection should filter and rank them
      using the information in `PhysicalDevice`. As part of this:
  - reject devices whose `api_version()` is below 1.4; the loader version
    check does not guarantee that each GPU supports it
  - decide whether the RHI keeps every `PhysicalDevice` or only the chosen one
  - remove the `#[expect(dead_code)]` attributes in
    `src/rhi/vulkan/physical_device.rs` as fields come into use
- [ ] **Add surface and swapchain support** for presenting to a window: the
      `VK_KHR_surface` and platform surface instance extensions, and the
      `VK_KHR_swapchain` device extension (which device selection should
      require).
- [ ] **Support macOS via MoltenVK.** This needs the
      `VK_KHR_portability_enumeration` instance extension and the
      `ENUMERATE_PORTABILITY_KHR` instance creation flag, and enabling
      `VK_KHR_portability_subset` on devices that report it.
- [ ] **Decide whether `PhysicalDevice` needs to be `Send`/`Sync`.** It is
      neither, because `ash`'s structures contain `p_next` raw pointers, even
      though they are always null. If the RHI needs to be used across threads,
      add `unsafe impl Send`/`Sync` with a safety comment.

## Testing

- [ ] **Make the unit tests work without a Vulkan driver.**
      `create_succeeds_for_every_graphics_api` creates a real Vulkan instance,
      so it fails on machines (such as CI runners) with no Vulkan driver.
      Options include installing a software driver such as lavapipe on CI,
      or marking the test `#[ignore]` and running it separately.
