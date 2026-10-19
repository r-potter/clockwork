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

  This is the `foundations-ash-1.4` slice of design 0001.
- [x] **Choose a physical device to render with.** Done in
      `src/rhi/vulkan/device_selection.rs`: devices are filtered by
      `REQUIREMENTS` (Vulkan 1.4, `VK_EXT_shader_object`, shader draw
      parameters, dynamic rendering and synchronization2, and a graphics
      queue; plus, when there is a window, `VK_KHR_swapchain`, a queue family
      that can present, and a surface format and present mode), then
      compared by device type and total device-local memory. The RHI keeps
      only the chosen device.
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
- [x] **Render and present frames.** `render_frame` in
      `src/rhi/vulkan.rs` acquires a swapchain image, clears it to
      `CLEAR_COLOUR` with dynamic rendering, and presents it, with two frames
      in flight (`src/rhi/vulkan/frames.rs`) and a render-finished semaphore
      per swapchain image. Resizes and `OUT_OF_DATE`/`SUBOPTIMAL` results
      recreate the swapchain once at the start of the next frame. Vulkan
      1.3's `dynamic_rendering` and `synchronization2` are enabled and
      required.
- [x] **Draw a triangle.** `TriangleShaders` in
      `src/rhi/vulkan/shaders.rs` draws a triangle with per-corner colours
      from `shaders/triangle.slang`, with no vertex buffers, using linked
      `VK_EXT_shader_object` shader objects instead of a pipeline, with all
      rendering state set when drawing. `build.rs` compiles the shaders to
      SPIR-V with `slangc`, and the backend embeds them. Vulkan 1.1's
      `shaderDrawParameters` is enabled and required, because Slang compiles
      `SV_VertexID` using the base vertex.
- [x] **Decide what to do on drivers without `VK_EXT_shader_object`.**
      Decided in ADR 0002: it's required, with no emulation layer or
      pipeline fallback. Devices without it are rejected at start-up.
- [ ] **Add a rendering API to the RHI.** The triangle is hardcoded in the
      Vulkan backend (`record_frame` in `src/rhi/vulkan.rs`). Expose
      pipelines, shaders, buffers and descriptors through
      `RenderHardwareInterface` as opaque handles, so that the application
      decides what to draw.
- [x] **Compile shaders for other backends.** Not needed: ADR 0003 keeps
      Vulkan as the only graphics API on every planned platform, so SPIR-V
      stays the only shader target.
- [ ] **Use `VK_EXT_swapchain_maintenance1` when available.** Without it,
      there is no way to know when
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
- [ ] **Support Android.** Planned in ADR 0003, on Adreno GPUs only,
      using Mesa's Turnip driver bundled with the app. It's waiting for
      Turnip to support `VK_EXT_shader_object`, `VK_EXT_descriptor_heap` and
      `VK_KHR_device_address_commands` (ADR 0002). The app already handles
      `suspended` and `resumed`, destroying and recreating the surface, but
      it cannot be built for Android yet. This needs an `android_main` entry
      point, with the app in a library crate (which ADR 0001 allows), one of
      `winit`'s `android-native-activity` or `android-game-activity`
      features, and a packaging tool such as `cargo-apk` or `xbuild`. Suspend
      and resume have only been tested on desktop, by calling the handlers
      manually.
- [ ] **Support macOS and iOS via KosmicKrisp.** Planned in ADR 0003, using
      Mesa's KosmicKrisp (Vulkan on Metal). It's waiting for KosmicKrisp to
      support Vulkan 1.4, `VK_EXT_shader_object`, `VK_EXT_descriptor_heap`
      and `VK_KHR_device_address_commands`. MoltenVK isn't planned, because
      it implements a more limited subset of Vulkan (ADR 0002).
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
      `ControlFlow::Poll` mode so the engine can pace frames itself. For now,
      the swapchain's FIFO present mode limits the frame rate to the
      display's refresh rate when there is a window (about 60 frames per
      second and 6% of a CPU core on the development machine). Nothing limits
      it while the window has no area or when headless, when the loop spins
      at 100% of a CPU core. Decide how the engine paces frames (for example,
      sleeping until the next frame is due, or offering `MAILBOX` where
      supported, with a frame rate limit), and handle those cases too.
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
      on machines (such as CI runners) with no Vulkan driver. For now, it and
      the other test that creates a backend are in a `gpu_tests` module in
      `src/rhi.rs`, which `cargo xtask check` skips. The
      `foundations-validation` slice of design 0001 closes this, by splitting
      the tests into tiers properly.
- [x] **Install `slangc` on CI.** Done in the `foundations-checks` slice of
      design 0001: the `linux` and `windows` jobs in
      `.github/workflows/ci.yml` install a pinned Vulkan SDK, and `build.rs`
      finds `slangc` in it (see `build/slangc.rs`).
- [x] **Decide how to suppress messages that have no message ID.** A
      `SUPPRESSED` entry in `src/diagnostics.rs` now matches its message by
      ID (`MessageMatch::Id`) or, for messages without one, by the start of
      its text (`MessageMatch::Prefix`). Recorded as a revision of design
      0001 (R4).
- [x] **Stop the OBS Vulkan layer failing tests on the Windows machine.**
      The loader's warning about OBS's implicit layer, `VK_LAYER_OBS_HOOK`,
      is suppressed by prefix, so OBS can still capture the engine. Its
      entry's target is `clockwork::rhi::vulkan`, where the debug messenger
      logs today: `foundations-validation` must move it to
      `clockwork::validation` along with the messenger's messages.
