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
- [x] **Choose a physical device to render with.** Done in
      `src/rhi/vulkan/device_selection.rs`: devices are filtered by
      `REQUIREMENTS` (Vulkan 1.4 and a graphics queue), then compared by
      device type and largest device-local heap. The RHI keeps only the chosen
      device.
- [ ] **Extend device selection as the renderer grows.** Add entries to
      `REQUIREMENTS` for the features, limits and extensions the renderer
      comes to depend on, and refine `compare` if needed. Remove the
      `#[expect(dead_code)]` attributes in `src/rhi/vulkan/physical_device.rs`
      as fields come into use.
- [x] **Create a window surface.** Done in `src/rhi/vulkan/surface.rs`,
      using `ash-window` to enable the surface instance extensions and create
      the surface. Passing no window runs the backend headless.
      `RenderHardwareInterface::suspend` and `resume` drop and recreate the
      surface, keeping the rest of the backend.
- [x] **Add swapchain support.** Done in `src/rhi/vulkan/swapchain.rs`,
      with the logical device and its graphics and present queues in
      `src/rhi/vulkan/device.rs`. When there is a window, device selection
      requires `VK_KHR_swapchain`, a queue family that can present, and a
      surface format and present mode. The swapchain is recreated on resize,
      and destroyed and recreated with the surface on suspend and resume.
- [ ] **Render and present frames.** Fill in the `TODO` in
      `VulkanRenderHardwareInterface::render_frame`: acquire swapchain
      images, record and submit command buffers to the graphics queue, and
      present on the present queue, with semaphores and fences to
      synchronise them.
  - Swapchain recreation is already deferred: `resize()` records the latest
    size in `pending_window_size`, and `render_frame` recreates the
    swapchain once at the start of the frame. Resize events are still
    required, because on Wayland the swapchain does not go out of date when
    the window is resized.
  - `ERROR_OUT_OF_DATE_KHR` and `SUBOPTIMAL_KHR` from acquire and present
    should also trigger recreation at the start of the next frame. They
    give no new size, so `render_frame` needs a way to recreate with the
    current size, for example by keeping the last known window size and a
    separate "needs recreation" flag.

  Also enable the device features the renderer uses (such as Vulkan 1.3's
  `dynamic_rendering` and `synchronization2`), and remove the
  `#[expect(dead_code)]` attributes on the queues in
  `src/rhi/vulkan/device.rs`.
- [ ] **Use `VK_EXT_swapchain_maintenance1` when available.** Depends on
      "Render and present frames". Without it, there is no way to know when
      the presentation engine has finished with a swapchain's images, so
      `Swapchain::drop` in `src/rhi/vulkan/swapchain.rs` waits for the whole
      device to be idle before destroying it, on every recreation. With it:
  - pass a fence with each present (`VkSwapchainPresentFenceInfoEXT`), and
    destroy a retired swapchain, and the semaphores used to present its
    images, only once the fences for its presents have signalled, instead
    of calling `device_wait_idle`
  - before recreating, release images that were acquired but not presented
    (`vkReleaseSwapchainImagesEXT`), so they are not left held by the old
    swapchain
  - query the `swapchainMaintenance1` feature
    (`VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT`) in `PhysicalDevice`,
    and enable the extension and feature when supported. It also needs the
    `VK_EXT_surface_maintenance1` and `VK_KHR_get_surface_capabilities2`
    instance extensions. Keep the current `device_wait_idle` path as a
    fallback for devices without it, rather than making it a requirement.
  - consider its other features: switching present modes without
    recreating the swapchain (`VkSwapchainPresentModesCreateInfoEXT`), which
    would help the frame pacing work, and controlling how images are scaled
    while a resize is in progress (`VkSwapchainPresentScalingCreateInfoEXT`)

  `ash` 0.38 has bindings for the EXT extensions. Newer Vulkan headers may
  also define KHR versions; check when updating to the Vulkan 1.4 bindings.
- [ ] **Support Android.** The app already handles `suspended` and
      `resumed`, destroying and recreating the surface, but it cannot be built
      for Android yet. This needs an `android_main` entry point (with the app
      in a library crate), one of `winit`'s `android-native-activity` or
      `android-game-activity` features, and a packaging tool such as
      `cargo-apk` or `xbuild`. Suspend and resume have only been tested on
      desktop, by calling the handlers manually.
- [ ] **Support macOS via MoltenVK.** This needs the
      `VK_KHR_portability_enumeration` instance extension and the
      `ENUMERATE_PORTABILITY_KHR` instance creation flag, and enabling
      `VK_KHR_portability_subset` on devices that report it.
- [ ] **Decide whether `PhysicalDevice` needs to be `Send`/`Sync`.** It is
      neither, because `ash`'s structures contain `p_next` raw pointers, even
      though they are always null. If the RHI needs to be used across threads,
      add `unsafe impl Send`/`Sync` with a safety comment.

## Engine

- [x] **Add a frame loop.** `App::about_to_wait` in `src/main.rs` runs
      `App::frame` on every event loop iteration (except before the backend
      exists and while suspended), which measures the time since the last
      frame, calls `App::update` (still a stub), then
      `RenderHardwareInterface::render_frame`.
- [ ] **Add frame pacing.** The `winit` event loop runs in
      `ControlFlow::Poll` mode so the engine can pace frames itself, but
      nothing limits the frame rate yet: the stub frame loop runs hundreds of
      thousands of frames per second and uses 100% of a CPU core while the
      window is open. Limit the frame rate, for example by waiting on vsync
      when presenting or by sleeping until the next frame is due. The swapchain uses the FIFO
      present mode, which waits on vsync; revisit this (for example, offering
      `MAILBOX` where supported) if the engine paces frames itself.
- [ ] **Upgrade to `winit` 0.31 once it is stable.** We are on 0.30.13;
      0.31 is in beta (0.31.0-beta.3 at the time of writing). 0.31 splits
      surface handling out of `resumed`/`suspended`, which become app
      lifecycle events emitted only on iOS, Web and Android. When upgrading,
      in `src/main.rs`:
  - move the contents of `resumed` (creating the window and RHI, or calling
    `rhi.resume`) to `can_create_surfaces`
  - move the contents of `suspended` (`rhi.suspend`) to `destroy_surfaces`
  - adapt to the other API changes in the 0.31 changelog, such as the new
    `run_app` signature

  This is easy to miss: code left in `resumed` still compiles, but is never
  called on desktop, so the window and RHI would never be created.

## Testing

- [ ] **Make the unit tests work without a Vulkan driver.**
      `create_succeeds_for_every_graphics_api` runs headless, so it does not
      need a display, but it still creates a real Vulkan instance, so it fails
      on machines (such as CI runners) with no Vulkan driver. Options include
      installing a software driver such as lavapipe on CI, or marking the test
      `#[ignore]` and running it separately.
