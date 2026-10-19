# Testing

How to set up a machine, run the checks, and write tests. This guide describes only tools that exist: each slice of [design 0001](design/0001-foundations.md) adds the section for what it builds (R19).

## Prerequisites

Every machine needs:
- **Rust**, stable, with `clippy` and `rustfmt` (`rustup component add clippy rustfmt`)
- **The Vulkan SDK** from [LunarG](https://vulkan.lunarg.com/sdk/home), which provides the Slang shader compiler (`slangc`) that `build.rs` runs, and the Khronos validation layer that GPU tests need. CI uses 1.4.363.0 (see `VULKAN_SDK_VERSION` in `.github/workflows/ci.yml`).

GPU tests also need a Vulkan 1.4 driver. The checks below don't.

### Minimum versions for GPU tests

The engine targets the extensions [ADR 0002](decisions/0002-vulkan-feature-baseline.md) lists, so GPU tests need a driver that has them and a validation layer that knows them:

| What | Minimum | Why |
|---|---|---|
| Vulkan SDK | 1.4.350.0 | The first SDK whose validation layer knows `VK_KHR_device_address_commands`, which arrived in the 1.4.346 headers. An older layer can't validate code that uses it. 1.4.341.1 knows `VK_EXT_descriptor_heap` and `VK_EXT_present_timing`, but not `VK_KHR_device_address_commands`. CI uses 1.4.363.0. |
| NVIDIA driver, Windows | 617.42 | Checked on the RTX 4090 on 2026-10-19 with `vulkaninfo`: it has every ADR 0002 extension, required and optional. 610.88 lacks `VK_KHR_device_address_commands`. |
| NVIDIA driver, Linux | 615.71.09 | Checked on the RTX 3090 on 2026-10-15 with `vulkaninfo` (ADR 0002). |

`reference_device_reports_required_extensions`, in `src/rhi/vulkan/gpu_tests.rs`, fails under `cargo xtask check --gpu` on a device without the required extensions. The application logs the optional ones the selected device has and lacks, at `INFO`, when it starts.

Nothing enables these extensions yet, so until the RHI uses them, GPU tests also pass with an older SDK, such as 1.4.341.1.

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

`build.rs` and `cargo xtask check` use the same lookup (`build/slangc.rs`), in this order:
1. `SLANGC`, if set: the path to `slangc`, used as given
2. the Vulkan SDK: `$VULKAN_SDK/bin/slangc` on Linux, `%VULKAN_SDK%\Bin\slangc.exe` on Windows
3. each directory on `PATH`

If none has it, the build fails with a message naming all three. Changing `SLANGC` or `VULKAN_SDK` recompiles the shaders. Changing `PATH` doesn't, so after moving `slangc` on `PATH`, run `cargo clean` or touch `build.rs`.

## `cargo xtask check`

`cargo xtask check` runs every check a branch must pass, and is what CI runs. Run it before asking for review. It runs, stopping at the first failure:
1. **Prerequisites:** prints the `slangc` it found, where from, and its version, and the Vulkan SDK's version. A missing `slangc` is an error; a missing `VULKAN_SDK` is only reported, since only GPU tests need the SDK's validation layer. With `--gpu`, it also prints the layers and devices the Vulkan loader finds, and fails if the validation layer is missing or there's no device.
2. **Formatting:** `cargo fmt --all --check`. Run `cargo fmt --all` to fix it.
3. **Clippy**, in debug and release: `cargo clippy --workspace --all-targets -- -D warnings`, so every warning, including the lints below, fails the check.
4. **Rustdoc:** `cargo doc --workspace --no-deps --document-private-items`, with `RUSTDOCFLAGS=-D warnings`, so broken intra-doc links fail the check.
5. **Tests that need no GPU:** `cargo test --workspace -- --skip gpu_tests`. With `--gpu`, every tier instead (see "GPU tests" below).
6. **Smoke runs**, with `--gpu` only: the application in a window under each window system (see "Smoke runs" below).

Options:

| Option | Effect |
|---|---|
| `--gpu` | Run the GPU (T1) tests too, with `CLOCKWORK_REQUIRED_CAPABILITIES=all`: `cargo test --workspace` in place of step 5. Then the smoke runs. Needs a Vulkan 1.4 driver, the validation layer, and a display for each window system. |
| `--no-window` | With `--gpu`: leave out the smoke runs, on purpose, for example over SSH. |
| `--virtual-displays` | Linux, with `--gpu`: run the smoke runs on virtual displays, so they work with no desktop session (see "Virtual displays" below). |

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

On Linux, that means a desktop session with both: a Wayland session with Xwayland, or an X11 session running a Wayland compositor. Otherwise, use `--virtual-displays`. A window system with no display is an error before anything else runs, not a skip, so that a check can't pass having run fewer smoke runs than it should. Use `--no-window` to leave them all out on purpose. The choice is made by `xtask/src/window_systems.rs`.

### Virtual displays

`cargo xtask check --gpu --virtual-displays` runs the Linux smoke runs with no desktop session, as on a CI runner or over SSH. Before the other checks, it starts:
- **`Xvfb`**, a virtual X server, which chooses a free display number itself (`-displayfd`), so it can't collide with another
- **Weston**, the reference Wayland compositor, headless (`weston --backend=headless`), on a socket named for the `xtask` process

Both are stopped when the check ends, whether it passes or not. The smoke runs, and nothing else, get their `DISPLAY` and `WAYLAND_DISPLAY`, and an `XDG_RUNTIME_DIR` for Weston's socket: the session's if it's set, otherwise `target/virtual-displays/runtime`. Their logs are in `target/virtual-displays/`, and an error starting either names its log.

They need the `xvfb` and `weston` packages (on Ubuntu and Debian, `sudo apt install xvfb weston`). A missing program is reported when the check starts.

`--virtual-displays` is an error on Windows, which needs no display server, and with `--no-window`. The configuration is in `xtask/src/virtual_displays.rs`.

To reproduce a failing smoke run, run the same command, unsetting the same variables. On Linux, for example, `env -u WAYLAND_DISPLAY -u WAYLAND_SOCKET cargo run -- --exit-after-frames 300 --fail-on-warnings` for X11.

`cargo xtask` is an alias (in `.cargo/config.toml`) for running the `xtask` crate, the repository's developer tooling. The repository is a Cargo workspace of the engine and `xtask`, with the engine as its only default member, so plain `cargo build`, `cargo run` and `cargo test` act on the engine alone.

### Lints

`Cargo.toml` sets these lints for the whole workspace (R13):
- `clippy::undocumented_unsafe_blocks`: every `unsafe` block has a `// SAFETY:` comment directly above it, saying why it's sound. A comment covering several blocks isn't enough: give each its own, even if it says "as above".
- `clippy::missing_docs_in_private_items`: every item is documented, including private ones, struct fields and enum variants.
- `unsafe_op_in_unsafe_fn`: unsafe operations inside an `unsafe fn` still need an `unsafe` block, with its own `// SAFETY:` comment.

Clippy doesn't apply `missing_docs_in_private_items` to test code (`#[cfg(test)]` modules and `#[test]` functions), but the rule is the same there: document test modules, tests and test helpers, saying what each test checks and why it matters. Review checks this.

See "Documentation and lints" in design 0001 for what each kind of doc comment must say.

## Test tiers

Tests are unit tests inside the crate they test (ADR 0001). There's no `tests/` directory.

| Tier | Kind | Needs | How to run |
|---|---|---|---|
| T0 | Unit: pure logic, such as device selection and the `slangc` lookup | Nothing beyond the prerequisites | `cargo test --workspace -- --skip gpu_tests`, or `cargo xtask check` |
| T1 | GPU: real Vulkan work, headless, with validation required | A Vulkan 1.4 driver and the validation layer | `cargo test`, which runs every tier, or `cargo xtask check --gpu` |

How tests are told apart:
- **T1 tests live in modules named `gpu_tests`**, so `--skip gpu_tests` leaves them out. Don't use `gpu_tests` anywhere else in a test's path, or it will be skipped too.
- **Everything else is T0**, and must pass on a machine with no Vulkan driver, as the CI runners are.

Later designs add the other tiers design 0001 lists.

## Writing tests

- **Name a test for the behaviour it checks,** as a sentence about the code: `rejects_devices_without_graphics`, `linux_sdk_comes_before_path`. Not `test_select_3`.
- **One behaviour per test,** so that a failure names what broke. Several assertions are fine if they check the same behaviour.
- **Document each test**: what it checks, and why it matters, if that isn't obvious.
- **Keep logic pure where you can,** taking its inputs (the platform, environment variables, a file probe, device properties) as parameters, so that it's T0 and every platform's behaviour is tested on any machine (ADR 0003). `build/slangc.rs` and `xtask/src/window_systems.rs` are examples: the Linux window systems are tested on Windows, and Win32 on Linux.
- **Build paths with `Path`,** never by joining strings, and compare `Path` values, not strings, so tests pass on both operating systems.
- **Fixture files go in `testdata/`,** next to the code that uses them.
- **Fix a bug by writing a failing test first,** then the fix.

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

Every test that runs engine code starts with a `TestSession` (`src/diagnostics/capture.rs`):

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

- **Never install another subscriber** (`tracing::subscriber::set_default`, `with_default`, or a global one) in a test. It hides that thread's events from the capture layer, so its warnings pass silently. A test that needs to check what was logged uses `expect`. The only exceptions are `src/diagnostics.rs`'s own tests of the layers.
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
    WARN clockwork::rhi::vulkan: ...
  ```

  A banner means some test isn't in a session, or a thread didn't enter the span. Find it and fix it.
- **Suppression.** A message that's harmless and can't be fixed here, such as a false positive in the validation layer, can be suppressed: add an entry to `SUPPRESSED` in `src/diagnostics.rs`, with the message's target, how to recognise the message, and a reason, linking an upstream issue where there is one. Recognise it by its message ID (`MessageMatch::Id`; for validation, the VUID) whenever it has one. Messages without an ID, such as a dependency's warnings through the `log` crate or the Vulkan loader's, are recognised by the start of their text (`MessageMatch::Prefix`): make the prefix long enough to identify one message, and leave out parts that vary, such as version numbers. A test checks that every entry has a target, a non-empty ID or prefix, and a reason. Suppressed messages are still logged at their level. Best-practices validation messages (IDs beginning `BestPractices-`) are only logged, never counted.

### How it works

`src/diagnostics.rs` defines sessions and the layers. A `Session` has a process-unique `SessionId`, and a span named `session` with the ID in a `session` field. An event is attributed to its own `session` field if it has one, otherwise to the nearest enclosing session span, otherwise to nothing. The test subscriber has three layers, each with its own filter: `SessionLayer`, which records each session span's ID; a fmt layer, filtered by `RUST_LOG`; and `TestCapture`, which takes every warning and error and routes it to its session's `TestSession`. The application's subscriber is the same, except that a `DiagnosticCounter` counts warnings and errors in place of `TestCapture`.

## GPU tests

T1 tests do real Vulkan work, headless, so they need a driver but no display. They live in modules named `gpu_tests`, next to the code they test: `src/rhi/gpu_tests.rs` creates backends through `rhi::create`, and `src/rhi/vulkan/gpu_tests.rs` tests the Vulkan backend itself.

### Writing one

A Vulkan T1 test creates its backend with `VulkanTestBackend::headless` (`src/rhi/vulkan/test_support.rs`), in a `TestSession`:

```rust
use crate::diagnostics::capture::TestSession;
use super::test_support::VulkanTestBackend; // in src/rhi/vulkan/gpu_tests.rs

#[test]
fn clears_are_validated() {
    let session = TestSession::new();
    let backend = VulkanTestBackend::headless(&session);
    // ...use backend.device, and so on...
} // `backend` is dropped, then `session`, which fails the test if anything was logged at WARN or above
```

Declare the session first, so that the backend is dropped first, and anything logged while it's destroyed still reaches the session. Tests of `rhi::create` itself pass it `test_support::backend_options(&session)`, which is how `VulkanTestBackend` is configured too.

### Validation is required

Every T1 test runs with the Khronos validation layer, and can't run without it: `VulkanTestBackend::headless` and `backend_options` always ask for `Validation::Required`, so creating the backend fails if the layer, or a requested feature, isn't available. There's no way to turn validation off for tests, because a pass without it would prove less than it claims.

Validation messages are events on the `clockwork::validation` target, attributed to the backend's session (the debug messenger passes the session's ID explicitly, since the layer can call it from any thread). So, like any other warning or error, a validation warning or error fails the test that caused it. Each event has a `message_id` field, the message's VUID or other ID, such as `VUID-VkBufferCreateInfo-size-00912` or `SYNC-HAZARD-WRITE-AFTER-WRITE`. A test that makes a mistake on purpose expects the message by its ID:

```rust
session.expect(Level::ERROR, VALIDATION_TARGET, |event| {
    event.message_id.as_deref() == Some("VUID-VkBufferCreateInfo-size-00912")
});
```

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

The features are passed to the layer as settings, through `VK_EXT_layer_settings`, using the setting names in the layer's manifest, `VkLayer_khronos_validation.json` (listed in `FEATURE_SETTINGS` in `src/rhi/vulkan/validation.rs`). Every feature's setting is passed, on or off, so that what's validated doesn't depend on the layer's defaults. If a newer layer renames a setting, it reports the old name with a `VALIDATION-SETTINGS` warning, which fails every T1 test: update the table from the new manifest.

### `CLOCKWORK_DEVICE`

Restricts the tests to devices whose names contain it, ignoring case, for example `CLOCKWORK_DEVICE=4090`, or `llvmpipe` for lavapipe. If no device matches, creating the backend fails, listing every device's name. Unset, the tests use the device the backend would choose anyway: the best that meets the requirements.

### Capabilities and skips

Some tests need a capability that not every device has, such as a dedicated transfer queue or `VK_EXT_present_timing`. They declare it with `require_capabilities!`, which returns from the test early if the device lacks any of them:

```rust
let session = TestSession::new();
let backend = VulkanTestBackend::headless(&session);
require_capabilities!(backend, Capability::DescriptorHeap);
```

The capabilities are listed in `Capability`, in `src/rhi/vulkan/test_support.rs`, with the names used below: `descriptor-heap`, `device-address-commands`, `present-timing`, `swapchain-maintenance`, `dedicated-compute-queue` and `dedicated-transfer-queue`. The list is for tests only, until the RHI has its own capability query. A device has an extension's capability when it has both the extension and the extension's main feature.

A skipped test passes, but prints a banner to stderr, even when the test passes, so skips are visible:

```text
=== clockwork: skipped rhi::vulkan::gpu_tests::...: NVIDIA GeForce RTX 4090 lacks the device-address-commands capability ===
```

### `CLOCKWORK_REQUIRED_CAPABILITIES`

Skips mustn't hide failures on the machines that have the capability. `CLOCKWORK_REQUIRED_CAPABILITIES` lists the capabilities whose absence fails a test rather than skipping it: `all`, or a comma-separated list of the names above. Unset, nothing is required, and every missing capability skips. An unknown name is an error, so that a typo can't turn a failure into a skip.

`cargo xtask check --gpu` sets it to `all`, so on the reference machines a test needing a capability either runs or fails.

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
| `linux` | `ubuntu-latest` | `cargo xtask check` |
| `windows` | `windows-latest` | `cargo xtask check` |

Each job installs the pinned Vulkan SDK (checking the download's SHA-256), for `slangc`. Neither runner has a Vulkan driver, which proves that T0 tests need none. To move to a new SDK, change `VULKAN_SDK_VERSION` and both hashes at the top of the workflow; LunarG publishes the hashes at `https://sdk.lunarg.com/sdk/sha/<version>/<linux|windows>/<file>.json`.

A failure in CI should reproduce with `cargo xtask check` on the same operating system.
