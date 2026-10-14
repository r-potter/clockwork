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
- [ ] **Add swapchain support** for presenting to the surface: the
      `VK_KHR_swapchain` device extension, which device selection should
      require when not headless, along with a queue family that can present
      to the surface. The swapchain must be destroyed in `suspend` and
      recreated in `resume`, along with the surface. On `resume`, check that
      the selected device can still present to the new surface.
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

- [ ] **Add a frame loop with frame pacing.** The `winit` event loop in
      `src/main.rs` runs in `ControlFlow::Poll` mode so the engine can pace
      frames itself, but there is no frame work or waiting yet, so it spins
      and uses 100% of a CPU core while the window is open. Run each frame
      (update, then render) from `ApplicationHandler::about_to_wait`, and
      limit the frame rate, for example by waiting on vsync when presenting
      or by sleeping until the next frame is due.
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
