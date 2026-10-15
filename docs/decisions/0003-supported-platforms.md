# 0003: Support Windows, X11 and Wayland from the start

- **Status:** Proposed
- **Date:** 2026-10-15

## Context

The engine has only ever run on Linux. Windows support is expected to be needed soon, and retrofitting a platform late is costly: path handling, tooling, shader compilation, CI and input all end up assuming one operating system. On Linux, Wayland and X11 behave differently enough (surface creation, window sizing, input) that both need testing.

Android, macOS and iOS are planned too, but their Vulkan drivers don't yet support ADR 0002's required extensions. They can't be supported yet, but the engine mustn't be designed in a way that rules them out.

## Decision

These platforms are supported from the start, and every design accounts for them:

| OS | Window system | Vulkan surface extension |
|---|---|---|
| Windows | Win32 | `VK_KHR_win32_surface` |
| Linux | Wayland | `VK_KHR_wayland_surface` |
| Linux | X11 | `VK_KHR_xlib_surface` or `VK_KHR_xcb_surface` (as `ash-window` chooses) |

"Supported" means:
- the engine and all tooling build and pass their GPU-free (T0) tests on both operating systems, in CI
- GPU (T1) tests and windowed smoke runs pass under each window system

**Where each is tested:**

| | Linux | Windows |
|---|---|---|
| Build and T0 | CI (hosted runner) | CI (hosted runner) |
| T1 and smoke runs | CI, on a pinned lavapipe, with X11 and Wayland on virtual displays; and on the RTX 3090 before review | On the Windows desktop with an RTX 4090, before review |

### Planned platforms

These are future work. They aren't supported yet, but designs must not rule them out.

| Platform | Vulkan driver | Waiting for |
|---|---|---|
| Android | Mesa's **Turnip** (Qualcomm Adreno), bundled with the app rather than using the device's own driver | Turnip support for `VK_EXT_shader_object`, `VK_EXT_descriptor_heap` and `VK_KHR_device_address_commands`. It already has Vulkan 1.4 on Adreno a7xx and later, and present timing on a750 and later. |
| macOS, iOS | Mesa's **KosmicKrisp** (Vulkan on Metal) | KosmicKrisp support for Vulkan 1.4, `VK_EXT_shader_object`, `VK_EXT_descriptor_heap` and `VK_KHR_device_address_commands`. It has Vulkan 1.3 today. |

The "waiting for" lists come from Mesa's `docs/features.txt` on `main`, on 2026-10-15. That file can lag behind the drivers, so recheck it when planning either platform.

What this means:
- **Vulkan stays the only graphics API.** Both planned drivers are Vulkan implementations, so neither needs a Metal or other backend, and SPIR-V from Slang stays the only shader target.
- **Android support covers Adreno devices only,** because Turnip drives only Adreno GPUs.
- **Bundling a driver** means loading it directly from the app, rather than through the system's Vulkan loader. The backend already loads Vulkan at run time (`Entry::load` in `src/rhi/vulkan.rs`), and `ash` can load a specific library instead (`Entry::load_from`). Validation layers on Android must be bundled too. The Android design settles the details, including how the bundled Turnip is built and pinned (as lavapipe is for CI).

## Consequences

- **Tooling is Rust (`xtask`), not shell scripts,** so that it runs on both operating systems and its logic can be tested.
- **Platform-specific code** (the `slangc` lookup, Vulkan SDK detection, window-system selection) is written as pure functions that take the platform as a parameter, so that every platform's behaviour is tested on any machine.
- **Text files use LF line endings everywhere,** enforced by `.gitattributes`.
- **Later designs must cover platform differences.** Examples already known:
  - **Importers:** Windows file systems ignore case. Case-insensitive asset resolution must behave, and report, the same way on both.
  - **Input:** raw mouse input and pointer capture differ between Win32, Wayland and X11.
  - **Presentation:** present modes and frame pacing differ between window systems.
- **Windows rendering isn't covered by CI** until a Mesa release with descriptor heaps on lavapipe is packaged for Windows. Revisit then.
- **Designs must leave room for the planned platforms.** Known examples:
  - **Crate layout:** Android needs the app in a library crate for its `android_main` entry point. ADR 0001 already allows that split when it's needed.
  - **Lifecycle:** mobile platforms suspend apps and take away their windows. The RHI's `suspend` and `resume` already handle this, and later subsystems must keep working through them.
  - **Input** (`0004-input`): touch and on-screen controls, as well as keyboard, mouse and gamepad.
  - **Frame graph** (`0003-frame-graph`): mobile GPUs render in tiles, so the graph must be able to express attachment load and store operations, and transient attachments that never leave tile memory.
  - **Tooling:** `xtask` builds Mesa for CI already, so building and pinning Turnip and KosmicKrisp later follows the same pattern.
- **Revisit this ADR** when Turnip or KosmicKrisp support the required extensions, recording each platform's move from planned to supported in a new ADR.
