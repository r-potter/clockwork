# 0001: Keep the engine a single binary crate

- **Status:** Proposed
- **Date:** 2026-10-15

## Context

A common Rust layout splits an application into a library crate and a thin binary, so that integration tests in `tests/` can link against the library. The withdrawn roadmap proposed this. It adds a second crate root and a public API boundary inside the engine, and nothing currently needs either: unit tests inside the crate can reach private items, which tests of a backend need anyway.

## Decision

The engine stays one binary crate (`src/main.rs`). Tests are unit tests within it, with shared fixtures in `src/test_support.rs`, compiled only under `cfg(test)`.

A separate crate is added only when something needs one:
- **The importers** become `crates/import`, so that `openusd`'s long compile stays out of the engine's rebuild cycle (`0006-importers`).
- **Android** needs the app in a library crate for its `android_main` entry point (see `TODO.md`).

## Consequences

- There's no `tests/` directory. End-to-end tests, such as rendering a frame headless and reading it back, are unit tests in `gpu_tests` modules.
- Developer tooling isn't part of the engine. It's an `xtask` crate in `xtask/`, run as `cargo xtask`, which makes the repository a Cargo workspace. The engine is still one binary crate: it's the workspace's root package and its only default member, so `cargo build`, `run` and `test` behave as they do without a workspace. Shell scripts were considered, but they don't run on Windows (ADR 0003).
- Dependency rules between the engine's modules are enforced by review, not by crate boundaries. Revisit this if a rule is broken more than once.
