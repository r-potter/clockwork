# 0006: Split the engine into a crate per subsystem

- **Status:** Proposed
- **Date:** 2026-10-22
- **Supersedes:** [0001](0001-single-binary-crate.md), once accepted

## Context

[ADR 0001](0001-single-binary-crate.md) kept the engine one binary crate, because nothing needed a library: unit tests inside the crate can reach private items, and the rest was simpler without an API boundary. Its consequences said to revisit that if the dependency rules were broken more than once. Since then, other reasons have come up:

- **Benchmarks can't reach the engine.** Design 0002 sets CPU budgets for the RHI, measured with `criterion`, the first T2 benchmarks. Cargo's benchmarks in `benches/` are separate crates, which can only use a library crate's public items, so they can't call anything in a binary.
- **The dependency rules are checked by review alone.** `ARCHITECTURE.md` says which subsystems may depend on which, for example that no Vulkan type appears outside the RHI, and adds: "If the code is split into crates later, they become crate boundaries that the compiler enforces."
- **Android needs a library crate** for its `android_main` entry point (`TODO.md`, ADR 0003), so some split is coming anyway.
- **The asset importers were always going to be a crate of their own,** `crates/assets`, to keep `openusd`'s compile out of the engine's rebuild cycle (ADR 0001).

## Decision

The engine is a workspace of crates, one per subsystem, under `crates/`, each named `clockwork-<subsystem>`. The root package stays the `clockwork` binary: the platform layer and the application code above the subsystems.

**The first split moves only the code that exists now:**

| Crate | Contents | Depends on |
|---|---|---|
| `clockwork-diagnostics` | Sessions, log attribution, classification and counting (`src/diagnostics.rs`). `TestSession` and the capture layer behind a `test-support` feature. | `tracing`, `tracing-subscriber`, `tracing-log` |
| `clockwork-gpu-memory` | The pure parts of design 0002's allocator: the TLSF suballocator and the pool. No Vulkan types. | Nothing |
| `clockwork-rhi` | The RHI's API and the Vulkan backend, including memory type selection and the allocator's Vulkan shell, `build.rs` and the GPU test fixtures (`src/test_support.rs`, `VulkanTestBackend`) behind a `test-support` feature | `clockwork-diagnostics`, `clockwork-gpu-memory`, `anyhow`, `ash`, `ash-window`, `raw-window-handle`, `tracing`, and `tracing-subscriber` for `test-support`, which sets up the tests' logging |
| `clockwork` (root) | The platform layer and options (`src/main.rs`), and later the triangle | All of the above, `anyhow`, `clap`, `tracing`, `tracing-subscriber`, `winit`, and `glam`, which nothing uses yet, for the triangle's vertices and push data (design 0002, "Data structures and layout") |

The table lists each crate's dependencies, leaving out those used only by its tests, such as `proptest` and `log`.

**Rules for every crate:**
- **Dependencies follow `ARCHITECTURE.md`'s diagram,** and nothing else. `clockwork-diagnostics` sits below everything. Only `clockwork-rhi` depends on `ash`, so no Vulkan type can reach another crate.
- **A crate's public items are its API.** Keep them minimal: everything else is `pub(crate)` or private. The workspace enables rustc's `missing_docs` and `unreachable_pub` lints, alongside the existing ones, so a public item must be documented and reachable from the crate's root.
- **Fixtures shared between crates sit behind a `test-support` feature,** which crates enable only in their `[dev-dependencies]`. `#[cfg(test)]` doesn't cross crates.
- **Unit tests stay inside their crate,** and still reach private items. A crate may also have `tests/`, for tests of its public API, and `benches/`, for benchmarks with `criterion`.
- **New crates are named in the design of their subsystem.** The roadmap's later subsystems (the frame graph, input, ECS, renderer and scripting) and `crates/assets` get their crates in their designs, rather than as empty crates now.
- **`default-members`** lists the root package and every engine crate, so that plain `cargo build` and `cargo test` cover the engine, and leave out `xtask`.

**Not split:** the Vulkan backend stays inside `clockwork-rhi`, rather than a `clockwork-rhi-vulkan` crate. Design 0002 keeps Vulkan the only backend (ADR 0003), and a backend crate would need the RHI's internals, such as handle tables and command lists, to be public to reach them.

## Consequences

- **The dependency rules are enforced by the compiler.** A forbidden dependency is a missing entry in `Cargo.toml`, not a review comment.
- **Benchmarks and integration tests use plain Cargo:** `cargo bench -p clockwork-gpu-memory`, and `tests/` where they help. Benchmarks measure public APIs, so command translation and handle lookups are measured as callers pay for them, through `CommandList` and `Device`.
- **Android's library crate needs no further split.** The app becomes a binary over library crates, and Android adds a second entry point over the same crates.
- **Builds can be faster,** since crates compile in parallel and are rebuilt only when they or their dependencies change.
- **Unused public items aren't reported.** Rustc's dead-code lint ignores a library's public items. Keeping APIs minimal, with `unreachable_pub`, limits what can hide.
- **Small functions called across crates can't be inlined without `#[inline]`,** or link-time optimization. Hot paths that cross crates, if any, need one or the other. The benchmarks are how to tell.
- **Moving the code is one mechanical PR, with no change in behaviour,** made when no other branch is moving the same files: after the allocator's PRs in design 0002's `rhi-memory` slice (#34 to #38) have merged, and before the slice's benchmarks. It updates `ARCHITECTURE.md`'s source layout, the paths `xtask` uses (`build/slangc.rs`, and the source file it reads `Capability`'s names from), `docs/testing.md`, and design 0002's paths, as a revision. Benchmarks follow, in each crate's `benches/`, starting with the allocator's, which complete `rhi-memory`.
- **ADR 0001's status becomes "Superseded by 0006"** when this is accepted.
- **Revisit** if crates come to be split along lines other than subsystems, or a crate's API keeps growing to serve another crate's internals. Either means the boundary is in the wrong place.

## Alternatives considered

- **Keep one binary crate, and run benchmarks as `#[ignore]`d unit tests that drive `criterion` as a library.** It needs no move, and benchmarks can reach private items. But `cargo bench` would do nothing, `criterion`'s command-line options don't pass through the test harness, and the dependency rules stay review-only. It also leaves the Android split for later.
- **One library crate and a thin binary.** It gives `cargo bench`, but one large public API, with none of the boundaries between subsystems.
- **Including source files into `benches/` with `#[path]`.** It only works for modules with no dependencies on the rest of the engine, which leaves out almost everything worth measuring.
