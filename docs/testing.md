# Testing

How to set up a machine, run the checks, and write tests. This guide describes only tools that exist: each slice of [design 0001](design/0001-foundations.md) adds the section for what it builds (R19).

## Prerequisites

Every machine needs:
- **Rust**, stable, with `clippy` and `rustfmt` (`rustup component add clippy rustfmt`)
- **The Vulkan SDK** from [LunarG](https://vulkan.lunarg.com/sdk/home), which provides the Slang shader compiler (`slangc`) that `clockwork-rhi`'s `build.rs` runs, and the Khronos validation layer that GPU tests need. CI uses 1.4.363.0 (see `VULKAN_SDK_VERSION` in `.github/workflows/ci.yml`).

GPU tests also need a Vulkan 1.4 driver. The checks below don't.

### Minimum versions for GPU tests

The engine targets the extensions [ADR 0002](decisions/0002-vulkan-feature-baseline.md) lists, so GPU tests need a driver that has them and a validation layer that knows them:

| What | Minimum | Why |
|---|---|---|
| Vulkan SDK | 1.4.350.0 | The first SDK whose validation layer knows `VK_KHR_device_address_commands`, which arrived in the 1.4.346 headers. An older layer can't validate code that uses it. 1.4.341.1 knows `VK_EXT_descriptor_heap` and `VK_EXT_present_timing`, but not `VK_KHR_device_address_commands`. CI uses 1.4.363.0. |
| NVIDIA driver, Windows | 617.42 | Checked on the RTX 4090 on 2026-10-19 with `vulkaninfo`: it has every ADR 0002 extension, required and optional. 610.88 lacks `VK_KHR_device_address_commands`. |
| NVIDIA driver, Linux | 615.71.09 | Checked on the RTX 3090 on 2026-10-15 with `vulkaninfo` (ADR 0002). |

Device selection rejects a device without the required extensions and features, or whose limits are too small for the RHI (design 0002, R1), and logs why at `INFO`. GPU tests then fail, because no device can be selected. The application logs the optional extensions the selected device has and lacks, at `INFO`, when it starts.

The required extensions are enabled on every device the engine creates, so GPU tests need the minimum SDK above. An older validation layer reports the extensions it doesn't know, or misses errors in code that uses them.

### Linux

1. Download the Linux tarball and extract it, for example into `~/vulkan`, giving `~/vulkan/1.4.363.0/`.
2. Set up its environment in every shell that builds the engine, for example from `~/.profile`:

   ```sh
   source ~/vulkan/1.4.363.0/setup-env.sh
   ```

   This sets `VULKAN_SDK` (to `~/vulkan/1.4.363.0/x86_64`), and adds the SDK to `PATH`, `LD_LIBRARY_PATH` and the Vulkan loader's layer search path.

### Windows

1. Run the SDK installer. The defaults are enough.
2. Open a new terminal. The installer sets `VULKAN_SDK` (for example `C:\VulkanSDK\1.4.363.0`) and adds its `Bin` directory to `PATH`, and registers the validation layer in the registry, so there's nothing else to set up.

Check out the repository with Git's defaults: `.gitattributes` keeps every text file, shaders included, with LF line endings on both operating systems, so `cargo fmt --check` and diffs agree.

### How `slangc` is found

`clockwork-rhi`'s `build.rs` and `cargo xtask check` use the same lookup (`crates/rhi/build/slangc.rs`), in this order:
1. `SLANGC`, if set: the path to `slangc`, used as given
2. the Vulkan SDK: `$VULKAN_SDK/bin/slangc` on Linux, `%VULKAN_SDK%\Bin\slangc.exe` on Windows
3. each directory on `PATH`

If none has it, the build fails with a message naming all three. Changing `SLANGC` or `VULKAN_SDK` recompiles the shaders. Changing `PATH` doesn't, so after moving `slangc` on `PATH`, run `cargo clean` or touch `crates/rhi/build.rs`.

## `cargo xtask check`

`cargo xtask check` runs every check a branch must pass, and is what CI runs. Run it before asking for review. It runs, stopping at the first failure:
1. **Prerequisites:** prints the `slangc` it found, where from, and its version, and the Vulkan SDK's version. A missing `slangc` is an error; a missing `VULKAN_SDK` is only reported, since only GPU tests need the SDK's validation layer. With `--gpu`, it also prints the layers and devices the Vulkan loader finds, and fails if the validation layer is missing or there's no device.
2. **Formatting:** `cargo fmt --all --check`. Run `cargo fmt --all` to fix it.
3. **Clippy**, in debug and release: `cargo clippy --workspace --all-targets -- -D warnings`, so every warning, including the lints below, fails the check.
4. **Rustdoc:** `cargo doc --workspace --no-deps --document-private-items`, with `RUSTDOCFLAGS=-D warnings`, so broken intra-doc links fail the check.
5. **Each crate on its own:** `cargo check --package <crate>`, then the same with `--all-targets`, for the engine's crates and the application (the workspace's default members). A workspace build turns on, for each dependency, every feature any crate asks for, so a crate can build there and fail on its own, as in `cargo test --package <crate>`. Dev-dependencies' features apply only to builds with tests, hence the build without them first. Fix a failure by asking for the feature in the crate that needs it, or on the workspace dependency.
6. **Tests that need no GPU:** `cargo test --workspace -- --skip gpu_tests`. With `--gpu`, every tier instead (see "GPU tests" below).
7. **Smoke runs**, with `--gpu` only: the application in a window under each window system (see "Smoke runs" below).

Options:

| Option | Effect |
|---|---|
| `--gpu` | Run the GPU (T1) tests too, with `CLOCKWORK_REQUIRED_CAPABILITIES=all`: `cargo test --workspace` in place of step 6. Then the smoke runs. Needs a Vulkan 1.4 driver, the validation layer, and a display for each window system. |
| `--no-window` | With `--gpu`: leave out the smoke runs, on purpose, for example over SSH. |
| `--virtual-displays` | Linux, with `--gpu`: run the smoke runs on virtual displays, so they work with no desktop session (see "Virtual displays" below). |
| `--lavapipe` | Linux, with `--gpu`: run the GPU tests and smoke runs on the pinned lavapipe alone, requiring only the capabilities it has (see "Lavapipe" below). This is how CI runs them. |

Run `cargo xtask check --gpu` before asking for review on a machine with a GPU, in a desktop session, and say in the review which machine and window systems it passed on.

### Smoke runs

A smoke run starts the application in a window, and checks that it renders and exits cleanly, with nothing logged at `WARN` or above:

```sh
cargo run -- --exit-after-frames 300 --fail-on-warnings
```

It stops after a number of frames, not a time, so that a slow machine or a virtual display can't make it flaky. `--fail-on-warnings` requires validation, and fails the run on any counted warning or error, from any subsystem or dependency (see "The application's options" below).

`--gpu` runs one for each window system the platform supports (ADR 0003), each with the variables that would let `winit` choose another window system unset:

| Platform | Window system | Needs | Unsets |
|---|---|---|---|
| Windows | Win32 | Nothing | Nothing |
| Linux | Wayland | `WAYLAND_DISPLAY` | `DISPLAY` |
| Linux | X11 | `DISPLAY` | `WAYLAND_DISPLAY` and `WAYLAND_SOCKET` |

On Linux, that means a desktop session with both: a Wayland session with Xwayland, or an X11 session running a Wayland compositor. Otherwise, use `--virtual-displays`. Under WSL, use `--virtual-displays` too: WSLg's Wayland compositor closes the connection to the smoke run after a few frames. The X11 smoke run needs `libxkbcommon-x11`, which `winit` loads when it starts: desktops usually have it, but minimal installations may not (on Ubuntu and Debian, `sudo apt install libxkbcommon-x11-0`; on Fedora, `sudo dnf install libxkbcommon-x11`). A window system with no display is an error before anything else runs, not a skip, so that a check can't pass having run fewer smoke runs than it should. Use `--no-window` to leave them all out on purpose. The choice is made by `xtask/src/window_systems.rs`.

### Virtual displays

`cargo xtask check --gpu --virtual-displays` runs the Linux smoke runs with no desktop session, as on a CI runner or over SSH. Before the other checks, it starts:
- **`Xvfb`**, a virtual X server, which chooses a free display number itself (`-displayfd`), so it can't collide with another. It listens only on Linux's abstract socket, not in `/tmp/.X11-unix`, which under WSL is WSLg's, and where other users can't create sockets.
- **Weston**, the reference Wayland compositor, headless (`weston --backend=headless`), on a socket named for the `xtask` process

Both are stopped when the check ends, whether it passes or not. The smoke runs, and nothing else, get their `DISPLAY` and `WAYLAND_DISPLAY`, an `XDG_RUNTIME_DIR` for Weston's socket: the session's if it's set, otherwise `target/virtual-displays/runtime`; and `WINIT_X11_SCALE_FACTOR=1`, since `Xvfb`'s display reports no physical size, from which `winit` would otherwise derive the scale factor, warning that it can't. Their logs are in `target/virtual-displays/`, and an error starting either names its log.

They need the `xvfb` and `weston` packages (on Ubuntu and Debian, `sudo apt install xvfb weston`; on Fedora, `sudo dnf install xorg-x11-server-Xvfb weston`). A missing program is reported when the check starts.

`--virtual-displays` is an error on Windows, which needs no display server, and with `--no-window`. The configuration is in `xtask/src/virtual_displays.rs`.

To reproduce a failing smoke run, run the same command, unsetting the same variables. On Linux, for example, `env -u WAYLAND_DISPLAY -u WAYLAND_SOCKET cargo run -- --exit-after-frames 300 --fail-on-warnings` for X11.

### Lavapipe

Lavapipe is Mesa's Vulkan driver for the CPU. CI has no GPU, so it runs the GPU tests on lavapipe. No Mesa release has every extension [ADR 0002](decisions/0002-vulkan-feature-baseline.md) requires on lavapipe yet, so it's built from a pinned commit on Mesa's `main` branch, the same way in CI and locally. Linux only.

`cargo xtask lavapipe` builds it:
1. It checks the build's prerequisites (below), and names every one that's missing.
2. It fetches the pinned commit alone, without Mesa's history, and builds lavapipe with Meson, with the X11 and Wayland presentation the smoke runs need, and nothing else Mesa can build. The first build takes several minutes.
3. It installs it in `target/lavapipe/<commit>/install/`, removes the source and build directories, and prints the path to the driver's manifest, `share/vulkan/icd.d/lvp_icd.x86_64.json`, which points the Vulkan loader at the driver.

Running it again does nothing, unless the pin or the build options have changed since, in which case it builds again from scratch. Builds for different pins coexist, and `cargo clean` removes them all. The pin and the options are in `xtask/src/lavapipe.rs`.

The build needs Git, Meson 1.4 or later, Ninja, pkg-config, C and C++ compilers, LLVM with its development files, Bison and Flex, Python 3 with the `mako`, `packaging` and `yaml` modules, and the development files of the libraries Mesa uses to present to X11 and Wayland. On Ubuntu and Debian:

```sh
sudo apt install git ninja-build pkg-config g++ llvm-dev bison flex \
  python3-mako python3-packaging python3-yaml \
  libexpat1-dev zlib1g-dev libdrm-dev libx11-dev libx11-xcb-dev libxcb1-dev \
  libxcb-dri3-dev libxcb-present-dev libxcb-randr0-dev libxcb-shm0-dev \
  libxcb-xfixes0-dev libxrandr-dev libxshmfence-dev libwayland-dev wayland-protocols
pipx install meson
```

Ubuntu 24.04's own Meson is too old, hence `pipx`. On Fedora:

```sh
sudo dnf install git meson ninja-build pkgconf-pkg-config gcc gcc-c++ llvm-devel bison flex \
  python3-mako python3-packaging python3-pyyaml \
  expat-devel zlib-ng-compat-devel libdrm-devel libX11-devel libxcb-devel \
  libXrandr-devel libxshmfence-devel wayland-devel wayland-protocols-devel
```

#### Running the checks on it

`cargo xtask check --gpu --lavapipe` runs the GPU tests and smoke runs on lavapipe, as CI does, so a lavapipe failure in CI reproduces with it on any Linux machine. Build lavapipe first: if the pinned build isn't there, the check fails at once, saying so. Add `--virtual-displays` to run the smoke runs with no desktop session, exactly as CI does.

`xtask` points the Vulkan loader at lavapipe's manifest with `VK_DRIVER_FILES`, so it's the only driver, and unsets `VK_ADD_DRIVER_FILES` and `VK_ICD_FILENAMES`, which would add drivers or be ignored. It does this in its own environment, which every command it runs inherits, so the prerequisites check lists lavapipe alone too; your shell's environment is unchanged.

`CLOCKWORK_REQUIRED_CAPABILITIES` becomes the capabilities lavapipe has, rather than `all`: `swapchain-maintenance` and `bc-compression`. Tests needing the others skip on it, since it has one queue family and no present timing, so the hardware checks before review cover those. The list is `CAPABILITIES` in `xtask/src/lavapipe.rs`. A test there fails if a capability added to `Capability` isn't listed either as one lavapipe has or as one it lacks, and the GPU test `every_required_capability_is_present` fails if lavapipe lacks one listed as one it has.

To run a single test on lavapipe, set the same variables yourself:

```sh
VK_DRIVER_FILES=$PWD/target/lavapipe/<commit>/install/share/vulkan/icd.d/lvp_icd.x86_64.json \
  CLOCKWORK_REQUIRED_CAPABILITIES=swapchain-maintenance,bc-compression \
  cargo test <test name>
```

#### Moving the pin

The pin is `PIN` in `xtask/src/lavapipe.rs`: a full commit hash, its date, and, in its documentation, why that commit was chosen. Move it like any other reviewed change, to pick up a fix, or a feature the engine starts to need:
- Choose a commit on Mesa's `main` branch. Every merge to `main` has passed Mesa's own CI.
- Once a Mesa release has `VK_EXT_descriptor_heap` and `VK_KHR_device_address_commands` on lavapipe (see lavapipe's entries in Mesa's `docs/features.txt`), pin that release's tagged commit instead, and say so in `PIN`'s documentation.
- Run `cargo xtask lavapipe`, then `cargo xtask check --gpu --lavapipe`. If the new build gains a capability, add it to `CAPABILITIES`, and remove it from the test's list of those lavapipe lacks.

`cargo xtask` is an alias (in `.cargo/config.toml`) for running the `xtask` crate, the repository's developer tooling. The repository is a Cargo workspace of the engine's crates, the `clockwork` application at its root, and `xtask` (ADR 0006). Every package but `xtask` is a default member, so plain `cargo build`, `cargo run` and `cargo test` act on the whole engine, and `cargo test -p clockwork-gpu-memory` on one crate.

### Lints

`Cargo.toml` sets these lints for the whole workspace (R13, and ADR 0006):
- `clippy::undocumented_unsafe_blocks`: every `unsafe` block has a `// SAFETY:` comment directly above it, saying why it's sound. A comment covering several blocks isn't enough: give each its own, even if it says "as above".
- `clippy::missing_docs_in_private_items`: every item is documented, including private ones, struct fields and enum variants.
- `unsafe_op_in_unsafe_fn`: unsafe operations inside an `unsafe fn` still need an `unsafe` block, with its own `// SAFETY:` comment.
- `missing_docs`: every public item is documented, as part of its crate's API.
- `unreachable_pub`: every `pub` item is reachable from its crate's root. Anything else is `pub(crate)` or private, so that each crate's API stays minimal.

Clippy doesn't apply `missing_docs_in_private_items` to test code (`#[cfg(test)]` modules and `#[test]` functions), but the rule is the same there: document test modules, tests and test helpers, saying what each test checks and why it matters. Review checks this.

See "Documentation and lints" in design 0001 for what each kind of doc comment must say.

## Test tiers

Tests are unit tests inside the crate they test, so they can reach its private items (ADR 0006). A crate may also have `tests/`, for tests of its public API, and `benches/`, for benchmarks (see "Benchmarks"). Fixtures another crate's tests need sit behind a `test-support` feature, which that crate enables in its `[dev-dependencies]`: `TestSession` is in `clockwork-diagnostics`'s.

| Tier | Kind | Needs | How to run |
|---|---|---|---|
| T0 | Unit: pure logic, such as device selection and the `slangc` lookup | Nothing beyond the prerequisites | `cargo test --workspace -- --skip gpu_tests`, or `cargo xtask check` |
| T0 | Property: invariants checked over generated inputs, with `proptest` (see "Property tests") | Nothing beyond the prerequisites | As for unit tests, which they run with |
| T1 | GPU: real Vulkan work, headless, with validation required | A Vulkan 1.4 driver and the validation layer | `cargo test`, which runs every tier, or `cargo xtask check --gpu` |
| T2 | Benchmark: hot paths, measured against their budgets with `criterion` (see "Benchmarks") | Depends on the benchmark | `cargo bench -p <crate>`, on demand |

How tests are told apart:
- **T1 tests live in modules named `gpu_tests`**, so `--skip gpu_tests` leaves them out. Don't use `gpu_tests` anywhere else in a test's path, or it will be skipped too.
- **Everything else is T0**, and must pass on a machine with no Vulkan driver, as the CI runners are.

Later designs add the other tiers design 0001 lists. Design 0002 adds property tests and benchmarks.

## Writing tests

- **Name a test for the behaviour it checks,** as a sentence about the code: `rejects_devices_without_graphics`, `linux_sdk_comes_before_path`. Not `test_select_3`.
- **One behaviour per test,** so that a failure names what broke. Several assertions are fine if they check the same behaviour.
- **Document each test**: what it checks, and why it matters, if that isn't obvious.
- **Keep logic pure where you can,** taking its inputs (the platform, environment variables, a file probe, device properties) as parameters, so that it's T0 and every platform's behaviour is tested on any machine (ADR 0003). `crates/rhi/build/slangc.rs` and `xtask/src/window_systems.rs` are examples: the Linux window systems are tested on Windows, and Win32 on Linux.
- **Build paths with `Path`,** never by joining strings, and compare `Path` values, not strings, so tests pass on both operating systems.
- **Check what a command list recorded through `CommandList::recorded`,** which resolves its arenas, rather than matching its private `Command`s, so that the recording model can change without rewriting the tests (design 0002, "Keeping the recording model reversible"). Test each recording check with `#[should_panic(expected = "...")]` and its whole message.
- **Fixture files go in `testdata/`,** next to the code that uses them.
- **Fix a bug by writing a failing test first,** then the fix.

## Property tests

A property test checks an invariant over many generated inputs, rather than a few chosen ones, using [`proptest`](https://docs.rs/proptest). It suits code with a simple specification and a large input space, such as an allocator, where the cases that break it are hard to pick by hand. The first is `allocations_match_a_simulated_model`, in `crates/gpu-memory/src/tlsf.rs`, which runs random allocations and frees against a model of the live ranges.

- **Check against a model.** The clearest property is usually that the code agrees with a simple, obviously correct version of itself, such as a list of live ranges standing in for the allocator's free lists.
- **Keep generated inputs small,** a few hundred steps of small sizes, so that a run takes well under a second, and failures shrink to cases a person can read.
- **Make sure it can fail.** Break the code on purpose, once, and check that the test catches it, then undo the change.
- **Commit `proptest-regressions/`.** When a property fails, `proptest` shrinks the input to a minimal case and records it in a file there, named for the test's source file. Every later run replays the recorded cases first, so a fixed bug stays fixed. Review the file like any other test change.
- **Run more cases** with `PROPTEST_CASES`, for example `PROPTEST_CASES=10000 cargo test allocations_match_a_simulated_model`, when changing the code a property covers. The default is 256.

Property tests are T0: they run with the unit tests, and need no GPU.

## Benchmarks

Benchmarks measure hot paths against the budgets in their designs, such as design 0002's "Performance", using [`criterion`](https://docs.rs/criterion). Each is in its crate's `benches/`, a separate crate that can reach only the public API, so it measures what callers pay. The first is `crates/gpu-memory/benches/allocator.rs`, which times the TLSF suballocator's and the pool's allocations and frees.

```
cargo bench -p clockwork-gpu-memory
cargo bench -p clockwork-gpu-memory -- tlsf/allocate
```

The second runs only the benchmarks whose names contain `tlsf/allocate`. Results are printed, and kept in `target/criterion/`, so the next run reports the change from the last. To compare a branch with `main`, run the benchmarks on `main` with `-- --save-baseline main`, then on the branch with `-- --baseline main`.

- **Benchmarks are run on demand,** not by `cargo xtask check` or CI, whose shared runners are too noisy for timings. The check builds them, through clippy's `--all-targets`, so they can't stop compiling. Run them when changing the code they cover, and give the numbers, and the machine, in the PR.
- **Budgets are for the reference machine** a design names, the RTX 3090 machine for design 0002. Elsewhere, compare against `main` on the same machine rather than against the budget.
- **Measure a steady state,** not an empty structure: the allocator's benchmarks fragment the block first, and every batch frees what it allocated, so each iteration sees the same state.
- **Make the work repeatable,** with fixed seeds for anything random, so that two runs measure the same thing.
- **Time only the operation,** with `iter_custom` where setup or the inverse operation would otherwise be counted, as the allocator's benchmarks do to time allocations and frees separately.

## Warnings and errors fail tests

Any warning or error that a test causes fails it, whichever subsystem or dependency logged it (R2, R3). This works through `tracing`: every event is *attributed* to a session, and each test has its own.

### Logging levels

Failing tests on warnings works only because the levels mean something definite. Log at:

| Level | When | Examples |
|---|---|---|
| `ERROR` | An operation failed, or a bug was detected | A validation error; a failed shader compile in hot reload |
| `WARN` | Something unexpected happened that someone should act on | A validation warning; a Lua script overrunning its budget |
| `INFO` | Normal, notable events, including expected limitations | The selected device; an optional extension the device lacks |
| `DEBUG`, `TRACE` | Detail for diagnosing problems | Per-frame timings; every barrier |

Code that logs a normal condition at `WARN` is a bug. Fix it by changing the level, not by suppressing the message.

### `TestSession`

Every test that runs engine code starts with a `TestSession` (`crates/diagnostics/src/capture.rs`):

```rust
use crate::diagnostics::capture::TestSession;

#[test]
fn resolves_paths() {
    let _session = TestSession::new();
    // ...code that should log nothing at WARN or above...
} // `_session` is dropped, which fails the test if anything was
```

`TestSession::new` installs the test subscriber, once per test process, and enters a new session's span on the test's thread. When the session is dropped, at the end of the test, it panics, failing the test, if a warning or error was attributed to it unexpectedly, listing each one (the first 64, then a count). Name it `_session`, not `_`, which would drop it at once.

Logs appear in a failing test's output, at `INFO` and above. Set `RUST_LOG` to see more, for example `RUST_LOG=debug cargo test resolves_paths`. `RUST_LOG` changes only what's printed: warnings and errors fail tests whatever it says.

### Expected warnings

A test of code that should warn declares the warning before the code that triggers it, with its level, its target (matched exactly), and a matcher for the rest:

```rust
let session = TestSession::new();
session.expect(Level::WARN, "clockwork::import", |event| event.message.contains("case"));
// ...resolve a path whose case doesn't match...
```

Each expectation is met by one event, so a warning that's expected twice is declared twice. An expected event that never arrives fails the test too, naming the line that declared it. The matcher gets a `CapturedEvent` (level, target, message ID and message). It runs on whichever thread logged the event, so it mustn't log, and a matcher that panics counts as not matching.

### Rules

- **Never install another subscriber** (`tracing::subscriber::set_default`, `with_default`, or a global one) in a test. It hides that thread's events from the capture layer, so its warnings pass silently. A test that needs to check what was logged uses `expect`. The only exceptions are `crates/diagnostics/src/lib.rs`'s own tests of the layers.
- **Threads a test starts must enter the session's span,** or their events can't be attributed:

  ```rust
  let span = session.span().clone();
  std::thread::spawn(move || {
      let _entered = span.entered();
      // ...
  });
  ```

  Code called back from C on its own threads, such as the Vulkan debug callback, tags its events with an explicit `session` field (`session = id.as_u64()`) instead.
- **Unattributed banners.** A warning or error from outside any session, or from a session whose `TestSession` has been dropped, can't fail a test. It's printed to stderr, even from passing tests, with a banner:

  ```text
  === clockwork: unattributed WARN outside any session ===
    WARN clockwork_rhi::vulkan: ...
  ```

  A banner means some test isn't in a session, or a thread didn't enter the span. Find it and fix it.
- **Suppression.** A message that's harmless and can't be fixed here, such as a false positive in the validation layer, can be suppressed: add an entry to `SUPPRESSED` in `crates/diagnostics/src/lib.rs`, with the message's target, how to recognise the message, and a reason, linking an upstream issue where there is one. Recognise it by its message ID (`MessageMatch::Id`; for validation, the VUID) whenever it has one. Messages without an ID, such as a dependency's warnings through the `log` crate or the Vulkan loader's, are recognised by the start of their text (`MessageMatch::Prefix`): make the prefix long enough to identify one message, and leave out parts that vary, such as version numbers. A message through `log` has the target it's printed with, the logging module's, even though `tracing` receives it with the target `log`. A test checks that every entry has a target, a non-empty ID or prefix, and a reason. Suppressed messages are still logged at their level. Best-practices validation messages (IDs beginning `BestPractices-`) are only logged, never counted.

### How it works

`crates/diagnostics/src/lib.rs` defines sessions and the layers. A `Session` has a process-unique `SessionId`, and a span named `session` with the ID in a `session` field. An event is attributed to its own `session` field if it has one, otherwise to the nearest enclosing session span, otherwise to nothing. The test subscriber has three layers, each with its own filter: `SessionLayer`, which records each session span's ID; a fmt layer, filtered by `RUST_LOG`; and `TestCapture`, which takes every warning and error and routes it to its session's `TestSession`. The application's subscriber is the same, except that a `DiagnosticCounter` counts warnings and errors in place of `TestCapture`.

## GPU tests

T1 tests do real Vulkan work, headless, so they need a driver but no display. They live in modules named `gpu_tests`, next to the code they test: `crates/rhi/src/gpu_tests.rs` creates backends through `clockwork_rhi::create`, and `crates/rhi/src/vulkan/gpu_tests.rs` tests the Vulkan backend itself.

### Writing one

A Vulkan T1 test creates its backend with `VulkanTestBackend::headless` (`crates/rhi/src/vulkan/test_support.rs`), in a `TestSession`:

```rust
use crate::diagnostics::capture::TestSession;
use super::test_support::VulkanTestBackend; // in crates/rhi/src/vulkan/gpu_tests.rs

#[test]
fn clears_are_validated() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    // ...use backend.device, and so on...
} // `backend` is dropped, then `session`, which fails the test if anything was logged at WARN or above
```

Declare the session first, so that the backend is dropped first, and anything logged while it's destroyed still reaches the session. Tests of `rhi::create` itself pass it `test_support::backend_options(&session)`, which is how `VulkanTestBackend` is configured too.

Destroy every buffer, texture and sampler the test creates. Destroying one retires it, and it's freed once nothing translated before can use it (design 0002, "Deferred destruction"), at the next submission or `collect_garbage`, or when the backend is dropped. A test that checks memory statistics or heap slots after destroying calls `collect_garbage` first. Objects left when the backend is dropped are freed with it, but each kind left is reported with a `WARN` on `clockwork::leaks` (design 0002, "Handles"), which fails the test:

```text
WARN clockwork::leaks: 2 samplers were never destroyed, and were freed with the device: "second", "third"
```

The report names the first eight objects it finds. `leaked_objects_are_reported` in `crates/rhi/src/vulkan/gpu_tests.rs` is the only test that leaves objects on purpose, and it expects the reports.

### Validation is required

Every T1 test runs with the Khronos validation layer, and can't run without it: `VulkanTestBackend::headless` and `backend_options` always ask for `Validation::Required`, so creating the backend fails if the layer, or a requested feature, isn't available. There's no way to turn validation off for tests, because a pass without it would prove less than it claims.

Validation messages are events on the `clockwork::validation` target, attributed to the backend's session (the debug messenger passes the session's ID explicitly, since the layer can call it from any thread). So, like any other warning or error, a validation warning or error fails the test that caused it. Each event has a `message_id` field, the message's VUID or other ID, such as `VUID-VkBufferCreateInfo-size-00912` or `SYNC-HAZARD-WRITE-AFTER-WRITE`. A test that makes a mistake on purpose expects the message by its ID:

```rust
session.expect(Level::ERROR, VALIDATION_TARGET, |event| {
    event.message_id.as_deref() == Some("VUID-VkBufferCreateInfo-size-00912")
});
```

Synchronization validation doesn't see copies and fills by address (`VK_KHR_device_address_commands`), which is how the RHI copies and fills, on either the buffer side or the texture side. A missing barrier between them isn't reported, so a test that copies or fills still records the barriers the RHI's users would, and checks the result, such as `buffer_round_trip` in `crates/rhi/src/vulkan/translate/gpu_tests.rs`.

A test that a driver bug breaks on one vendor's drivers skips there with `test_support::skip_for_driver_bug`, which prints a banner like a missing capability's, and names the bug, which `TODO.md` tracks. `padded_rows_round_trip` skips on NVIDIA's drivers this way.

A false positive is suppressed by its ID, with a reason and an upstream issue (see "Rules" above). Best-practices messages (IDs beginning `BestPractices-`) are only logged, never counted. The Vulkan loader's own messages come through the same messenger, so they're on `clockwork::validation` too, with the ID `Loader Message`, which identifies none of them: suppress those by prefix.

### `CLOCKWORK_VALIDATION`

Which validation features T1 tests require: `sync` (core and synchronization validation) if it's unset. It's a comma-separated list of:

| Feature | Checks | Notes |
|---|---|---|
| `core` | API usage against the specification's valid usage rules | Implied by every other feature |
| `sync` | Hazards between commands, such as two writes with no barrier between them | The default |
| `gpu` | Inside shaders, such as out-of-bounds accesses (GPU-assisted validation) | Slow |
| `best-practices` | Advice | Logged, never counted |

`off` is rejected, since T1 tests always validate. For example, to check a change with GPU-assisted validation too:

```sh
CLOCKWORK_VALIDATION=sync,gpu cargo test
```

On Windows, in PowerShell: `$env:CLOCKWORK_VALIDATION = "sync,gpu"; cargo test`.

The features are passed to the layer as settings, through `VK_EXT_layer_settings`, using the setting names in the layer's manifest, `VkLayer_khronos_validation.json` (listed in `FEATURE_SETTINGS` in `crates/rhi/src/vulkan/validation.rs`). Every feature's setting is passed, on or off, so that what's validated doesn't depend on the layer's defaults. If a newer layer renames a setting, it reports the old name with a `VALIDATION-SETTINGS` warning, which fails every T1 test: update the table from the new manifest.

### `CLOCKWORK_DEVICE`

Restricts the tests to devices whose names contain it, ignoring case, for example `CLOCKWORK_DEVICE=4090`, or `llvmpipe` for lavapipe. If no device matches, creating the backend fails, listing every device's name. Unset, the tests use the device the backend would choose anyway: the best that meets the requirements.

### Capabilities and skips

Some tests need a capability that not every device has, such as a dedicated transfer queue or `VK_EXT_present_timing`. They declare it with `require_capabilities!`, which returns from the test early if the device lacks any of them:

```rust
let session = TestSession::new();
let backend = VulkanTestBackend::headless(&session);
require_capabilities!(backend, Capability::DedicatedTransferQueue);
```

The capabilities are listed in `Capability`, in `crates/rhi/src/vulkan/test_support.rs`, with the names used below: `present-timing`, `swapchain-maintenance`, `dedicated-compute-queue`, `dedicated-transfer-queue` and `bc-compression`. Each names an optional part of the RHI's `Capabilities` or `PresentCapabilities` (design 0002), and a device has it when the object that reports it says so, so tests see the device as the RHI's users do. What device selection requires, such as descriptor heaps, isn't listed: every selected device has it.

A skipped test passes, but prints a banner to stderr, even when the test passes, so skips are visible:

```text
=== clockwork: skipped rhi::vulkan::gpu_tests::...: llvmpipe (LLVM 19.1.7, 256 bits) lacks the dedicated-transfer-queue capability ===
```

`every_required_capability_is_present`, in `crates/rhi/src/vulkan/gpu_tests.rs`, asks about every capability, so a required one the device lacks fails a test even before any test needs it.

### `CLOCKWORK_REQUIRED_CAPABILITIES`

Skips mustn't hide failures on the machines that have the capability. `CLOCKWORK_REQUIRED_CAPABILITIES` lists the capabilities whose absence fails a test rather than skipping it: `all`, or a comma-separated list of the names above. Unset, nothing is required, and every missing capability skips. An unknown name is an error, so that a typo can't turn a failure into a skip.

`cargo xtask check --gpu` sets it to `all`, so on the reference machines a test needing a capability either runs or fails. With `--lavapipe`, it's the capabilities lavapipe has instead (see "Lavapipe").

## The application's options

The application takes command-line options (R14), listed by `cargo run -- --help`:

| Option | Effect | Default |
|---|---|---|
| `--device <SUBSTRING>` | Only consider devices whose names contain it, ignoring case | The best device |
| `--validation <FEATURES>` | `off`, or a comma-separated list of the features in "`CLOCKWORK_VALIDATION`" above. Used if the validation layer is available; otherwise a warning is logged and the application runs without it. | `sync` in debug builds, `off` in release builds |
| `--exit-after-frames <N>` | Exit cleanly after rendering N frames, at least 1 | Run until the window is closed |
| `--fail-on-warnings` | Require validation, unless it's off, and exit with an error if any warning or error was logged | Off |

At exit, the application logs a summary of the warnings and errors it counted, from every subsystem and dependency, at `INFO`:

```text
INFO clockwork: 0 errors and 2 warnings were logged
```

They're counted as tests count them (see "Warnings and errors fail tests"): suppressed and best-practices messages don't count, and `RUST_LOG` changes only what's printed. Under `--fail-on-warnings`, any count above zero makes the application exit with an error after the summary, so a run with `--fail-on-warnings` passes only if it logged nothing at `WARN` or above. It also turns validation from preferred to required, so that a missing validation layer fails the run instead of passing it unvalidated. In a release build, validation is still off unless `--validation` asks for it.

Tests can't take options, since `cargo test` can't pass any to them, so they read environment variables instead. The engine itself never reads the environment: each variable has an option to match, parsed by the same function, so the two can't drift apart.

| Test variable | Application option | Differences |
|---|---|---|
| `CLOCKWORK_DEVICE` | `--device` | None |
| `CLOCKWORK_VALIDATION` | `--validation` | Tests reject `off`, and require validation, where the application only prefers it |
| `CLOCKWORK_REQUIRED_CAPABILITIES` | None | Only tests skip on missing capabilities |

## CI

`.github/workflows/ci.yml` runs on every pull request, whatever its base branch, and on every push to `main`. A branch with no pull request isn't checked: open a draft pull request to run CI on one. The jobs run on GitHub-hosted runners:

| Job | Runner | Runs |
|---|---|---|
| `linux` | `ubuntu-24.04` | `cargo xtask check` |
| `windows` | `windows-latest` | `cargo xtask check` |
| `linux-lavapipe` | `ubuntu-24.04` | `cargo xtask lavapipe`, then `cargo xtask check --gpu --lavapipe --virtual-displays` |

Each job installs the pinned Vulkan SDK (checking the download's SHA-256), for `slangc`, and in `linux-lavapipe` for the validation layer too. The runners have no Vulkan driver, which proves that T0 tests need none. To move to a new SDK, change `VULKAN_SDK_VERSION` and both hashes at the top of the workflow; LunarG publishes the hashes at `https://sdk.lunarg.com/sdk/sha/<version>/<linux|windows>/<file>.json`.

`linux-lavapipe` installs the packages "Lavapipe" and "Virtual displays" list, `libxkbcommon-x11-0` for the X11 smoke run, and Meson from PyPI at `MESON_VERSION`, since Ubuntu's is too old. Its lavapipe build is cached, keyed on the runner's image and on `xtask/src/lavapipe.rs`, which holds the pin and the build's options, so it's rebuilt only when one of them changes. It's saved as soon as it's built, so a failing check doesn't throw it away. An uncached build takes several minutes. A package added to the build's prerequisites must be added to the job too.

The Linux jobs are pinned to an Ubuntu release, rather than `ubuntu-latest`, so that GitHub's moving `ubuntu-latest` to a newer release doesn't change what `linux-lavapipe` installs and links without a review. Moving to a newer release is a change like any other: update both jobs' `runs-on`, check that `linux-lavapipe`'s packages still exist there, and whether its Meson is still too old (the comment on `MESON_VERSION` names the release). The new image changes the lavapipe cache key, so the first run rebuilds it.

A failure in CI should reproduce with the same command on the same operating system: `cargo xtask check`, or for `linux-lavapipe`, `cargo xtask lavapipe` and then `cargo xtask check --gpu --lavapipe --virtual-displays`.
