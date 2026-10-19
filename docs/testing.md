# Testing

How to set up a machine, run the checks, and write tests. This guide describes only tools that exist: each slice of [design 0001](design/0001-foundations.md) adds the section for what it builds (R19).

## Prerequisites

Every machine needs:
- **Rust**, stable, with `clippy` and `rustfmt` (`rustup component add clippy rustfmt`)
- **The Vulkan SDK** from [LunarG](https://vulkan.lunarg.com/sdk/home), which provides the Slang shader compiler (`slangc`) that `build.rs` runs, and the Khronos validation layer that GPU tests need. CI uses 1.4.341.1 (see `VULKAN_SDK_VERSION` in `.github/workflows/ci.yml`).

GPU tests also need a Vulkan 1.4 driver. The checks below don't.

### Linux

1. Download the Linux tarball and extract it, for example into `~/vulkan`, giving `~/vulkan/1.4.341.1/`.
2. Set up its environment in every shell that builds the engine, for example from `~/.profile`:

   ```sh
   source ~/vulkan/1.4.341.1/setup-env.sh
   ```

   This sets `VULKAN_SDK` (to `~/vulkan/1.4.341.1/x86_64`), and adds the SDK to `PATH`, `LD_LIBRARY_PATH` and the Vulkan loader's layer search path.

### Windows

1. Run the SDK installer. The defaults are enough.
2. Open a new terminal. The installer sets `VULKAN_SDK` (for example `C:\VulkanSDK\1.4.341.1`) and adds its `Bin` directory to `PATH`, and registers the validation layer in the registry, so there's nothing else to set up.

Check out the repository with Git's defaults: `.gitattributes` keeps every text file, shaders included, with LF line endings on both operating systems, so `cargo fmt --check` and diffs agree.

### How `slangc` is found

`build.rs` and `cargo xtask check` use the same lookup (`build/slangc.rs`), in this order:
1. `SLANGC`, if set: the path to `slangc`, used as given
2. the Vulkan SDK: `$VULKAN_SDK/bin/slangc` on Linux, `%VULKAN_SDK%\Bin\slangc.exe` on Windows
3. each directory on `PATH`

If none has it, the build fails with a message naming all three. Changing `SLANGC` or `VULKAN_SDK` recompiles the shaders. Changing `PATH` doesn't, so after moving `slangc` on `PATH`, run `cargo clean` or touch `build.rs`.

## `cargo xtask check`

`cargo xtask check` runs every check a branch must pass, and is what CI runs. Run it before asking for review. It runs, stopping at the first failure:
1. **Prerequisites:** prints the `slangc` it found, where from, and its version, and the Vulkan SDK's version. A missing `slangc` is an error; a missing `VULKAN_SDK` is only reported, since only GPU tests need the SDK's validation layer.
2. **Formatting:** `cargo fmt --all --check`. Run `cargo fmt --all` to fix it.
3. **Clippy**, in debug and release: `cargo clippy --workspace --all-targets -- -D warnings`, so every warning, including the lints below, fails the check.
4. **Rustdoc:** `cargo doc --workspace --no-deps --document-private-items`, with `RUSTDOCFLAGS=-D warnings`, so broken intra-doc links fail the check.
5. **Tests that need no GPU:** `cargo test --workspace -- --skip gpu_tests`.

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
| T1 | GPU: real Vulkan work, headless | A Vulkan 1.4 driver | `cargo test`, which runs every tier |

How tests are told apart:
- **T1 tests live in modules named `gpu_tests`**, so `--skip gpu_tests` leaves them out. Don't use `gpu_tests` anywhere else in a test's path, or it will be skipped too.
- **Everything else is T0**, and must pass on a machine with no Vulkan driver, as the CI runners are.

Later slices of design 0001 add T1 tooling (`cargo xtask check --gpu`, required validation, capability skips), and later designs add the other tiers the design lists.

## Writing tests

- **Name a test for the behaviour it checks,** as a sentence about the code: `rejects_devices_without_graphics`, `linux_sdk_comes_before_path`. Not `test_select_3`.
- **One behaviour per test,** so that a failure names what broke. Several assertions are fine if they check the same behaviour.
- **Document each test**: what it checks, and why it matters, if that isn't obvious.
- **Keep logic pure where you can,** taking its inputs (the platform, environment variables, a file probe, device properties) as parameters, so that it's T0 and every platform's behaviour is tested on any machine (ADR 0003). `build/slangc.rs` is an example.
- **Build paths with `Path`,** never by joining strings, and compare `Path` values, not strings, so tests pass on both operating systems.
- **Fixture files go in `testdata/`,** next to the code that uses them.
- **Fix a bug by writing a failing test first,** then the fix.

## CI

`.github/workflows/ci.yml` runs on every push and pull request, on GitHub-hosted runners:

| Job | Runner | Runs |
|---|---|---|
| `linux` | `ubuntu-latest` | `cargo xtask check` |
| `windows` | `windows-latest` | `cargo xtask check` |

Each job installs the pinned Vulkan SDK (checking the download's SHA-256), for `slangc`. Neither runner has a Vulkan driver, which proves that T0 tests need none. To move to a new SDK, change `VULKAN_SDK_VERSION` and both hashes at the top of the workflow; LunarG publishes the hashes at `https://sdk.lunarg.com/sdk/sha/<version>/<linux|windows>/<file>.json`.

A failure in CI should reproduce with `cargo xtask check` on the same operating system.
