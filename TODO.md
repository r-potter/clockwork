# TODO

Outstanding tasks and future work.

## Vulkan

- [x] **Update to the Vulkan 1.4 bindings.** Done in the
      `foundations-ash-1.4` slice of design 0001: `Cargo.toml` patches `ash`
      and `ash-window` to a commit on `ash`'s `master` branch, generated from
      the Vulkan 1.4.352 headers. `API_VERSION` is `vk::API_VERSION_1_4`, and
      `PhysicalDevice` has the Vulkan 1.4 features and properties. Remove the
      patch once `ash` releases these bindings.
- [x] **Choose a physical device to render with.** Done in
      `crates/rhi/src/vulkan/device_selection.rs`: devices are filtered by
      `REQUIREMENTS` (Vulkan 1.4, `VK_EXT_shader_object`, shader draw
      parameters, dynamic rendering and synchronization2, and a graphics
      queue; plus, when there is a window, `VK_KHR_swapchain`, a queue family
      that can present, and an sRGB surface format and a present mode), then
      compared by device type and total device-local memory. The RHI keeps
      only the chosen device.
- [ ] **Extend device selection as the renderer grows.** Add entries to
      `REQUIREMENTS` for the features, limits and extensions the renderer
      comes to depend on, and refine `compare` if needed. The ADR 0002
      baseline is required since design 0002's `rhi-baseline` slice, and
      every field of `PhysicalDevice` is now read.
- [ ] **Make the device filter a type that can't be blank.**
      `BackendOptions::device_filter` is an `Option<String>`, and its two
      sources each decide what a blank value means: `--device` rejects one
      (`parse_device` in `src/main.rs`), and the T1 tests treat a blank
      `CLOCKWORK_DEVICE` as unset (`device_filter` in
      `crates/rhi/src/test_support.rs`). `BackendOptions` itself accepts
      `Some("")`, which matches every device, and `Some(" ")`, which matches
      every device with a space in its name, the silent mistake
      `parse_device` exists to catch. Add a `DeviceFilter` newtype to
      `clockwork-rhi`, whose constructor (or `FromStr`) rejects a value
      that's empty or only whitespace, so the rule sits beside device
      selection (`filter` in `crates/rhi/src/vulkan/device_selection.rs`).
      `--device` would parse into it, and `backend_options` map a blank
      variable to `None` before constructing one.
- [x] **Create a window surface.** Done in `crates/rhi/src/vulkan/surface.rs`,
      using `ash-window` to enable the surface instance extensions and create
      the surface. Passing no window runs the backend headless.
      `RenderHardwareInterface::suspend` and `resume` drop and recreate the
      surface, keeping the rest of the backend.
- [x] **Add swapchain support.** Done in `crates/rhi/src/vulkan/swapchain.rs`,
      with the logical device and its graphics and present queues in
      `crates/rhi/src/vulkan/device.rs`. When there is a window, device
      selection requires `VK_KHR_swapchain`, a queue family that can present,
      and an sRGB surface format and a present mode. The swapchain is
      recreated on resize, and destroyed and recreated with the surface on
      suspend and resume.
- [x] **Render and present frames.** `acquire_surface_image` in
      `crates/rhi/src/vulkan.rs` acquires a swapchain image as a texture,
      with a semaphore from a pool that reuses each once the submission
      that waited for it completes, and `present` submits what renders to
      it and presents it, with a render-finished semaphore per swapchain
      image. The application keeps two frames in flight. Resizes and
      `OUT_OF_DATE`/`SUBOPTIMAL` results recreate the swapchain once at the
      start of the next frame. Vulkan 1.3's `dynamic_rendering` and
      `synchronization2` are enabled and required.
- [x] **Draw a triangle.** `src/triangle.rs` draws a triangle with
      per-corner colours through the RHI, from a vertex buffer its shader,
      `assets/shaders/triangle.slang`, reads by address. Programs are linked
      `VK_EXT_shader_object` shader objects instead of a pipeline, with all
      rendering state set when drawing. The application's `build.rs`
      compiles its shaders to SPIR-V with `slangc`, and it embeds them.
      Vulkan 1.1's `shaderDrawParameters` is enabled and required, because
      Slang compiles `SV_VertexID` using the base vertex.
- [x] **Decide what to do on drivers without `VK_EXT_shader_object`.**
      Decided in ADR 0002: it's required, with no emulation layer or
      pipeline fallback. Devices without it are rejected at start-up.
- [x] **Add a rendering API to the RHI.** Design 0002's `rhi-resources`
      slice added buffers, textures and samplers as opaque handles through
      `GpuResources`, with descriptor heap slots, debug names and leak
      reports. `rhi-commands` added command lists,
      translated on any thread and submitted to queues with timelines,
      barriers described by accesses, copies, and deferred destruction.
      `rhi-timing` added nested timing scopes, whose GPU durations each
      queue keeps until `take_timings` reads them back. `rhi-shaders`
      added programs, render state, push data, shader access to the
      descriptor heaps, views and attachments of any subresource, render
      passes, and draws, direct, indexed and indirect, and dispatches.
      `rhi-transient` added per-frame transient memory:
      `begin_transient_frame` hands out a `TransientFrame`, which any
      thread allocates from without a lock, in chunks of upload memory
      that a pool sized by recent frames recycles through deferred
      destruction. `rhi-streaming` added the BC formats and streaming: an
      `Uploader` on any thread stages bytes in upload buffers of its own,
      records the copies and the release to the graphics queue, falling
      back to the graphics queue for texture regions the transfer queue
      can't copy, and `flush` submits them, returning an `Upload` with
      their timepoint and acquire barriers. The design's remaining slice,
      `rhi-present`, added `acquire_surface_image` and `present`, and the
      application now draws the triangle itself (`src/triangle.rs`), as a
      `Program` from `assets/shaders/triangle.slang`, with its vertices
      filled through an `Uploader`, and the backend no longer draws
      anything itself. It also added benchmarks of the design's
      "Performance" budgets, in `crates/rhi/benches/rhi/`. Design 0002 is
      implemented; its "As built" section lists what differs.
- [ ] **Meet design 0002's translation budgets.** Translating a list of
      10,000 draws takes 4.0 ms on one thread, against a budget of 1 ms,
      and eight such lists on eight threads take 5.1 ms, against 2 ms
      (`translation/serial` and `translation/parallel`, on the Windows
      desktop's i9-13900K and RTX 4090). The benchmark's lists set the
      whole `RenderState` before every draw, and translation sets every
      piece of dynamic state each time, which is about 90% of the cost:
      with the state set once per list, a list takes 0.48 ms. Skip state
      that hasn't changed since the last one set, the design's first
      remedy, kept off in tests by its rule that translation translates
      and nothing more ("Keeping the recording model reversible").
      Recording also stores a copy of the state each time it's set, so
      storing each distinct state once, for example with a map from state
      to its index in the arena, may help too. Check with
      `cargo bench -p clockwork-rhi -- translation`.
- [ ] **Decide what design 0002's transient allocation budget measures.**
      With eight threads allocating from one frame at once, an
      allocation takes 312 ns on each thread, against a budget of 50 ns
      (`transient/contended`). Each allocation is one compare-and-swap on
      the chunk's offset, and eight threads doing nothing else contend for
      its cache line: one thread alone takes 32 ns, and the eight
      together make one allocation every 39 ns. Real threads record
      between allocations, so they contend far less. Either revise the
      budget to the eight threads' throughput, or give each thread a
      range of the chunk of its own to bump through, and revise
      "Performance" or "Transient memory" to match.
- [ ] **Run the benchmarks on the RTX 3090 machine.** Design 0002's
      budgets are for the RTX 3090 machine, and design 0003's are for the
      same machine, but the benchmarks have only run on the Windows
      desktop's i9-13900K and RTX 4090. Run `cargo bench -p clockwork-rhi`
      and `cargo bench -p clockwork-frame-graph` there, and update the two
      items above and design 0002's "As built" with the results. Compiling
      a 64-pass graph took 119 µs on the desktop, against a budget of
      1 ms.
- [ ] **Consider building the uploader's staging on the transient
      allocator.** `Staging` (`crates/rhi/src/upload/staging.rs`) has its
      own bump allocator over 4 MiB upload buffers, with the same
      large-allocation rule as `TransientFrame`. It writes each upload
      with `write_buffer`, which looks the buffer up every time, rather
      than through a persistent mapping, and it tracks each flush's
      timepoint to know when to reuse a batch. A `TransientFrame` per
      flush, from a chunk pool of the uploader's own, would remove all
      three: one allocator, writes through the mapping, and retirement by
      deferred destruction. It needs a `GpuResources` method that creates
      a pool, a Vulkan `ChunkSource` per pool, chunks that copies can read
      from, and either a way to reach copy offset alignments beyond
      `MAX_TRANSIENT_ALIGNMENT` or a reason to cap them there. Profile
      streaming many small uploads first, and revise design 0002's
      "Streaming" and "Transient memory" if it's worth doing.
- [ ] **Run the uploader's graphics-queue fallback on a device.** An
      `Uploader` copies texture regions its transfer queue's
      `minImageTransferGranularity` doesn't allow on the graphics queue
      instead, submitted after the transfer queue's copies and waiting for
      them. On NVIDIA, transfers share the compute family, which Vulkan
      requires to copy any region, and lavapipe has one queue family, so
      only T0 tests (`crates/rhi/src/upload/recording/tests.rs`) cover
      that path, and the T1 tests (`crates/rhi/src/upload/gpu_tests.rs`)
      upload whole mips. On a device whose transfer queue has a coarser
      granularity, add a T1 test that uploads a mip in partial regions,
      some the transfer queue can copy and some it can't, samples or
      copies it out after the `Upload`, and checks its acquire barriers
      cover only the transfer queue's regions.
- [ ] **Let a program's stages come from separate SPIR-V modules.**
      `ProgramDesc` has one `spirv` field, and every stage is created from
      that module, choosing its entry point by name. That suits `slangc`,
      which puts every entry point of a file in one module, but not stages
      compiled separately or loaded from a cache. When something needs it,
      move the SPIR-V into each stage (for example `stages: &[StageDesc {
      stage, spirv, entry_point }]`), have `shader_create_infos` in
      `crates/rhi/src/vulkan/programs.rs` point each create info at its
      stage's code (`VK_EXT_shader_object` links create infos, not modules),
      and check each stage's SPIR-V isn't empty. Revise design 0002's
      "Programs" with the change.
- [x] **Compile shaders for other backends.** Not needed: ADR 0003 keeps
      Vulkan as the only graphics API on every planned platform, so SPIR-V
      stays the only shader target.
- [ ] **Use `VK_KHR_swapchain_maintenance1` when available.** It's optional in
      ADR 0002. Without it, there is no way to know when the presentation
      engine has finished with a swapchain's images, so `Swapchain::drop` in
      `crates/rhi/src/vulkan/swapchain.rs` waits for the whole device to be
      idle before destroying it, on every recreation. With it:
  - pass a fence with each present (`VkSwapchainPresentFenceInfoKHR`), and
    destroy a retired swapchain, and the semaphores used to present its
    images, only once the fences for its presents have signalled, instead
    of calling `device_wait_idle`
  - before recreating, release images that were acquired but not presented
    (`vkReleaseSwapchainImagesKHR`), so they are not left held by the old
    swapchain
  - enable the extension and its `swapchainMaintenance1` feature when
    `PhysicalDevice::features.swapchain_maintenance1` says it's supported.
    It also needs the `VK_KHR_surface_maintenance1` and
    `VK_KHR_get_surface_capabilities2` instance extensions. Keep the current `device_wait_idle` path as a
    fallback for devices without it, rather than making it a requirement.
  - consider its other features: switching present modes without
    recreating the swapchain (`VkSwapchainPresentModesCreateInfoKHR`), which
    would help the frame pacing work, and controlling how images are scaled
    while a resize is in progress (`VkSwapchainPresentScalingCreateInfoKHR`)

  The KHR extension promoted the EXT ones, and the pinned `ash` has bindings
  for both.
- [ ] **Catch command buffers kept past their surface image's swapchain.**
      `Swapchain::drop` in `crates/rhi/src/vulkan/swapchain.rs` destroys its
      images' attachment views at once, not through deferred destruction, so
      a command buffer translated with a surface image, kept, and submitted
      after the swapchain is recreated uses destroyed views. Design 0002
      ("Presentation") makes that misuse, but it's undefined behaviour rather
      than a panic. Have translation record, under the texture table's read
      lock, each command buffer that uses a surface image until it's
      submitted or dropped, and have `remove_surface_images` check for one
      and remove the entries in one critical section under the write lock,
      panicking before the swapchain waits for the device or destroys
      anything, so every interleaving panics rather than racing. Removal
      while the backend itself is dropped mustn't panic.
- [x] **Order the first barrier on an acquired image after the acquire.**
      Decided at the start of `rhi-present`, and recorded as a revision of
      design 0002's "Synchronization": an acquired image starts at
      `Access::Acquired`, whose stage in the access table
      (`crates/rhi/src/vulkan/access.rs`), `COLOR_ATTACHMENT_OUTPUT`, is the
      one the acquire semaphore is waited for at, so the barrier from it is
      ordered after the wait.
- [ ] **Support Android.** Planned in ADR 0003, on Adreno GPUs only,
      using Mesa's Turnip driver bundled with the app. It's waiting for
      Turnip to support `VK_EXT_shader_object`, `VK_EXT_descriptor_heap` and
      `VK_KHR_device_address_commands` (ADR 0002). The app already handles
      `suspended` and `resumed`, destroying and recreating the surface, but
      it cannot be built for Android yet. This needs an `android_main` entry
      point, a second one over the engine's library crates (ADR 0006), one of
      `winit`'s `android-native-activity` or `android-game-activity`
      features, and a packaging tool such as `cargo-apk` or `xbuild`. Suspend
      and resume have only been tested on desktop, by calling the handlers
      manually.
- [ ] **Support macOS and iOS via KosmicKrisp.** Planned in ADR 0003, using
      Mesa's KosmicKrisp (Vulkan on Metal). It's waiting for KosmicKrisp to
      support Vulkan 1.4, `VK_EXT_shader_object`, `VK_EXT_descriptor_heap`
      and `VK_KHR_device_address_commands`. MoltenVK isn't planned, because
      it implements a more limited subset of Vulkan (ADR 0002).
- [x] **Decide whether `PhysicalDevice` needs to be `Send`/`Sync`.** It needs
      both, since design 0002 shares the device between threads. It already is
      both: the pinned `ash` declares its structures `Send` and `Sync`, though
      `ash` 0.38 didn't, so no `unsafe impl` is needed. A compile-time
      assertion in `crates/rhi/src/vulkan/physical_device.rs` keeps it so.
- [ ] **Support read-only depth attachments in the RHI.** The RHI renders
      every depth attachment in the layout of a depth write
      (`crates/rhi/src/vulkan/render_pass.rs`), and its store operations,
      `Store` and `DontCare`, both write the attachment. So the frame
      graph treats every depth attachment as written, and two passes that
      only depth-test one texture, such as a forward pass and a decal pass
      after a depth prepass, run one after the other rather than sharing a
      barrier. Read-only depth needs:
      - rendering with a depth attachment in a read-only layout when the
        depth test doesn't write, such as `Layout::ReadOnly`, which maps to
        `READ_ONLY_OPTIMAL` and already serves sampled depth
      - `StoreOp::None` (`VK_ATTACHMENT_STORE_OP_NONE`, core in Vulkan
        1.3), which writes nothing, so the attachment's barriers needn't
        order a write
      - a way for the frame graph to declare it, such as bringing back
        `RasterPassBuilder::depth`'s `write` argument, with
        `DepthAttachmentRead` alone for a depth test that doesn't write

      Revise design 0002 for the RHI's part and 0003 for the graph's.
      Measure first: it's worth it if serializing depth-tested passes
      shows in the example techniques' GPU times.

## Engine

- [x] **Add a frame loop.** `App::about_to_wait` in `src/main.rs` runs
      `App::frame` on every event loop iteration (except before the backend
      exists and while suspended), which measures the time since the last
      frame, calls `App::update` (still a stub), then draws the triangle
      to a surface image from `acquire_surface_image` and presents it,
      once the frame two before it is complete.
- [ ] **Revisit RON for technique files.** Design 0003 chose TOML for
      technique files, over RON, in review ("Alternatives considered"):
      the `toml` crate keeps each value's position with `Spanned<T>`, so
      errors found after parsing point at the value, TOML is familiar,
      and a name table keeps the format separate from the Rust API. RON
      reads as the description's Rust values, with a syntax for enums,
      which TOML lacks. Once `graph-techniques` has a loader, revisit it
      if descriptions have grown deeper than two levels, or the loader
      has much hand-written code mapping TOML strings and tables onto
      enums. Check first whether the `ron` crate now reports the
      positions of values. Both formats would parse into the same
      `GraphDesc`, so a switch changes only the loader, and the
      techniques in `assets/techniques/`.
- [ ] **Design asset access and hot reload with `0006-importers`.**
      Design 0003 deferred hot reload of techniques to the asset system
      ("Non-goals"), so that techniques, their shaders, textures and
      models are read and reloaded one way. The design should give:
      - one abstraction over where assets come from: the file system,
        and other sources such as compressed asset packages
      - hot reload for every kind of asset, techniques included,
        installing a reloaded technique with `FrameGraph::set_technique`,
        and keeping the running one, with an `ERROR`, when a reload fails
      - how shaders reach `slangc`, which reads sources from the file
        system itself, and whether packaged shaders are compiled ahead
        of time
      - a replacement for the one function that reads technique files in
        `clockwork-frame-graph`

      It should also choose between polling modification times and
      platform file events through `notify`, which design 0003's review
      compared. Polling costs a few microseconds a file, notices a save
      within about half a second, handles editors that save by renaming
      a new file over the old one, works under WSL across the Windows
      boundary, where inotify sees no events, and is T0-testable with a
      fake file probe and clock. `notify` reacts at once and costs
      nothing while idle, but still needs debouncing, has to watch
      directories rather than files to survive those renames, misses
      events on WSL and network drives, and is hard to test without
      real timing.
- [ ] **Revisit linking Slang once hot reload exists.** Design 0003
      compiles a technique's programs by running `slangc`, chosen over
      linking Slang as a library in review ("Alternatives considered"):
      it needs nothing new built, keeps compiler crashes out of the
      engine, and needs no `unsafe`. But every compile starts a process
      and loads Slang's core module. Once hot reload exists (see above),
      time reloading a technique after editing one of its shaders. If it takes
      more than about half a second, link the library behind the same
      compiler interface, with bindings of our own rather than the
      `shader-slang` crate:
      - the compiling interfaces (global session, session, module, entry
        point, component type and blob), declared as `#[repr(C)]` tables
        of `extern "system"` functions in `slang.h`'s order, and the C
        reflection functions (`spReflection*`) for push data
      - the library loaded with `libloading` from beside the `slangc` that
        `crates/rhi/build/slangc.rs` finds, so it's the build's version,
        and refused if its version isn't one the bindings were checked
        against, since a table that doesn't match the header is undefined
        behaviour
      - a session per worker, or a lock, since global sessions aren't
        meant to be shared between threads freely
      - the interfaces compared with `slang.h` whenever the pinned SDK
        moves

      Revise design 0003's "Techniques" and "Alternatives considered"
      with the change.
- [x] **Skip import barriers past the import's shape in `graph-execute`.**
      Done in `crates/frame-graph/src/commands.rs`: turning a plan's
      barriers into the RHI's each frame leaves out one that starts past
      its texture's last mip or layer, such as mips `1..` of a one-mip
      import used whole and as mip 0, which would cover nothing, and on
      which the RHI panics.
- [ ] **Leave out the barriers a plan doesn't need.** The property test
      `the_checker_finds_any_missing_barrier`
      (`crates/frame-graph/src/random_graphs.rs`) found barriers the
      compiler records that order nothing that another barrier, or the
      caller, doesn't order already, and skips them:
      - a global barrier whose buffers an earlier group's global barrier
        orders already, since each orders every buffer
      - one at the end of the frame, in one layout, whose accesses before
        are in the stages of those it leaves the resource in, such as
        `compute-shader-read` to `compute-shader-write`, which the
        caller's own barrier from them covers. The compiler would need
        each access's stage, which only the RHI's Vulkan backend has
      - one from the last frame's reads to the same reads of a persistent
        texture. Before the first frame after a compile the executor moves
        the texture to the accesses the frame ends with, so it orders
        nothing, but the checker starts that frame from `Access::None`.
        Model that in the checker first, with transient textures still
        starting from `Access::None`

      Each costs a little recording and translating, not correctness.
      Narrow the test's exceptions as each is left out.
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

- [x] **Run a T1 test that needs a capability.** Done in the
      `foundations-ash-1.4` slice of design 0001:
      `reference_device_reports_required_extensions` in
      `crates/rhi/src/vulkan/gpu_tests.rs` uses `require_capabilities!`, so
      the `#[expect]` attributes on it and `VulkanTestBackend::has` are gone.
      It passes on the RTX 4090 from driver 617.42, the first checked that has
      `VK_KHR_device_address_commands`.

- [x] **Make the unit tests work without a Vulkan driver.** Done in the
      `foundations-validation` slice of design 0001: every test that creates a
      backend is a T1 test, in a `gpu_tests` module
      (`crates/rhi/src/gpu_tests.rs` and
      `crates/rhi/src/vulkan/gpu_tests.rs`), which `cargo xtask check` skips
      and `cargo xtask check --gpu --no-window` runs, with validation
      required.
- [x] **Install `slangc` on CI.** Done in the `foundations-checks` slice of
      design 0001: the `linux` and `windows` jobs in
      `.github/workflows/ci.yml` install a pinned Vulkan SDK, and `build.rs`
      finds `slangc` in it (see `crates/rhi/build/slangc.rs`).
- [x] **Decide how to suppress messages that have no message ID.** A
      `SUPPRESSED` entry in `crates/diagnostics/src/lib.rs` now matches its
      message by ID (`MessageMatch::Id`) or, for messages without one, by the
      start of its text (`MessageMatch::Prefix`). Recorded as a revision of
      design 0001 (R4).
- [x] **Stop the OBS Vulkan layer failing tests on the Windows machine.**
      The loader's warning about OBS's implicit layer, `VK_LAYER_OBS_HOOK`,
      is suppressed by prefix, on `clockwork::validation`, where the debug
      messenger logs the loader's messages, so OBS can still capture the
      engine.
- [ ] **Run the RTX 3090 hardware check for `foundations-lavapipe`.** It
      was merged after checks on the Windows laptop (RTX 4070) and on the
      pinned lavapipe under WSL. Run `cargo xtask check --gpu` on the RTX
      3090 in a desktop session, with both window systems: the slice changed
      how warnings from the `log` crate are classified, and added two
      suppressions for messages logged under X11 and Wayland.
- [ ] **Move the lavapipe pin to Mesa 26.3.0.** The pin in
      `xtask/src/lavapipe.rs` is a commit on Mesa's `main`, because no
      release had `VK_EXT_descriptor_heap` and
      `VK_KHR_device_address_commands` on lavapipe. Mesa's
      `26.3-branchpoint` (2026-10-15) has both, so pin the `mesa-26.3.0`
      tag's commit once it's released, following "Moving the pin" in
      `docs/testing.md`. Then consider whether a packaged lavapipe (Arch, or
      a PPA on Ubuntu) could replace the build, which would need a revision
      of design 0001, since a rolling package isn't pinned.
- [x] **Pin CI's Linux runners to an Ubuntu version.** Both Linux jobs
      run on `ubuntu-24.04`, so GitHub's move of `ubuntu-latest` to 26.04
      (2026-10-19 to 2026-11-19) can't change the packages `linux-lavapipe`
      installs, or the libraries its lavapipe build links. Moving to 26.04 is
      a deliberate change, as "CI" in `docs/testing.md` describes.
- [ ] **Check texture copies' buffer ranges in debug builds.** Copies
      name buffers by address (design 0002, R5), so the bytes
      `check_texture_copy` returns aren't checked against a buffer, and a
      range past the end of one is read or written unreported. Translation
      could look the address up in the buffer table in debug builds and
      panic, naming the buffer, if the range doesn't fit inside it.
- [ ] **Fill recycled transient chunks with a pattern in debug builds, if
      late reads need catching.** A list translated after its frame ends
      reads transient data another frame may have overwritten, which
      validation can't detect (design 0002, "Risks"). The design proposed
      filling recycled chunks with a pattern in debug builds, so a late
      read shows as garbage, but `rhi-transient` left it out until such a
      bug turns up. The frame knows how far it bumped through each chunk,
      so only that part, and a new chunk once, would need filling, at
      about the cost of the data written.
- [ ] **Check copy hazards once synchronization validation sees copies
      and fills by address.** The RHI copies and fills with
      `VK_KHR_device_address_commands`, which synchronization validation
      doesn't track, on the buffer side or the texture side, so a missing
      barrier between copies isn't reported (see "Validation is required"
      in `docs/testing.md`). When a validation layer release tracks them,
      add a copy case to `missing_barrier_is_reported`.
- [ ] **Write `missing_barrier_is_reported` once synchronization
      validation reports a hazard between RHI commands.** Design 0002's
      R8 asks for a T1 test that a missing barrier is reported and that
      the barrier removes the report. The layer in SDK 1.4.363 reports no
      hazard between two render passes that write one attachment with no
      barrier between them, whether they clear, load or draw, and with
      copies and fills by address untracked (above), no pair of RHI
      commands produces one. A raw repro needs only a 4×4 colour image in
      `COLOR_ATTACHMENT_OPTIMAL` and two `vkCmdBeginRendering` and
      `vkCmdEndRendering` pairs that clear and store it, which report
      nothing, while two `vkCmdClearColorImage` calls on the same image
      report `SYNC-HAZARD-WRITE-AFTER-WRITE`. Find out whether the spec
      orders attachment writes across render pass instances, and if not,
      report it upstream. When the layer reports either case, write the
      test, rendering twice, or copying, with and without a barrier.
- [ ] **Remove the workarounds for NVIDIA's broken copies once they're
      fixed.** NVIDIA's proprietary driver, up to 617.42 at least, breaks
      copies by address, with no validation message:
      - On its transfer-only family, `vkCmdCopyMemoryKHR` writes nothing
        when a range is flagged `UNKNOWN_STORAGE_BUFFER_USAGE`, as
        translation flags every range.
      - `vkCmdCopyImageToMemoryKHR` isn't ordered after a copy into the
        same image, despite the barrier between them, so it reads stale
        data. On the graphics queue, `padded_rows_round_trip` reads zeros
        with no flags, and passes when split into two submissions, so the
        copy in does write. Flagging the copy in
        `UNKNOWN_STORAGE_BUFFER_USAGE`, as translation does, hides it
        there, which is why the round trips pass.
      - On the transfer-only family, a submission that copies into an
        image with `UNKNOWN_STORAGE_BUFFER_USAGE` fails. With no flag the
        round trip read the wrong data, likely the ordering bug.

      NVIDIA has acknowledged the first two, with fixes due in a later
      driver. Report the third if the fix doesn't cover it. Until then
      device selection doesn't use transfer-only families on that driver
      (`copies_break_on_transfer_families` in
      `crates/rhi/src/vulkan/device_selection.rs`), so transfers share the
      compute family, and tests that need a dedicated transfer queue skip
      there. When the fix ships, limit the check to earlier driver
      versions, and check `cross_queue_wait`,
      `storage_buffers_are_filled_and_copied`, the texture round trips and
      the uploader's tests (`crates/rhi/src/upload/gpu_tests.rs`) pass on
      the transfer-only family, and that the round trips pass on the
      graphics queue with no flag on the copy in.
- [ ] **Make `CLOCKWORK_VALIDATION=gpu` usable.** With GPU-assisted
      validation, every T1 test fails on the layer's own warnings at
      start-up: `VALIDATION-SETTINGS`, for running GPU-AV alongside core
      validation, and `WARNING-Setting-Limit-Adjusted`, as GPU-AV reserves
      some of the device's limits. Neither is a problem in the engine, so
      either suppress them when GPU-AV is requested, or have
      `test_support` turn core validation off with it. Design 0002 relies
      on GPU-AV for binding tests ("Risks"), such as
      `fragment_samples_heap_texture`, which its test plan asks to run
      with GPU-AV before review. Found while prototyping heap access, and
      seen again running that test, where in both GPU-AV reported nothing
      else.
- [ ] **Move CI off actions that run on Node.js 20.** GitHub warns that
      `actions/cache@v4` targets Node.js 20, which is deprecated, and runs
      it on Node.js 24 instead. Move to a version that targets Node.js 24
      when there is one.