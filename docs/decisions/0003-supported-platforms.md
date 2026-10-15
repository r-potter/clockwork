# 0003: Support Windows, X11 and Wayland from the start

- **Status:** Proposed
- **Date:** 2026-10-15

## Context

The engine has only ever run on Linux. Windows support is expected to be needed soon, and retrofitting a platform late is costly: path handling, tooling, shader compilation, CI and input all end up assuming one operating system. On Linux, Wayland and X11 behave differently enough (surface creation, window sizing, input) that both need testing.

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

macOS and Android aren't supported. ADR 0002's extensions aren't available there yet.

## Consequences

- **Tooling is Rust (`xtask`), not shell scripts,** so that it runs on both operating systems and its logic can be tested.
- **Platform-specific code** (the `slangc` lookup, Vulkan SDK detection, window-system selection) is written as pure functions that take the platform as a parameter, so that every platform's behaviour is tested on any machine.
- **Text files use LF line endings everywhere,** enforced by `.gitattributes`.
- **Later designs must cover platform differences.** Examples already known:
  - **Importers:** Windows file systems ignore case. Case-insensitive asset resolution must behave, and report, the same way on both.
  - **Input:** raw mouse input and pointer capture differ between Win32, Wayland and X11.
  - **Presentation:** present modes and frame pacing differ between window systems.
- **Windows rendering isn't covered by CI** until a Mesa release with descriptor heaps on lavapipe is packaged for Windows. Revisit then.
