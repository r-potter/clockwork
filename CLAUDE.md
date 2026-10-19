# clockwork

A small game engine written in Rust, rendering with Vulkan.

## How work is done

- **Designs come first.** Every roadmap step starts as a design in [`docs/design/`](docs/design/README.md), which is reviewed and accepted before implementation starts. Cross-cutting decisions are ADRs in [`docs/decisions/`](docs/decisions/README.md). Follow the accepted designs and ADRs, and when something doesn't fit, say so and record it as a revision rather than silently diverging.
- **Implementation is in slices, reviewed as small PRs.** A slice a design lists is usually split into several small, focused pull requests rather than one large one. Each PR covers a single concern, builds and passes the checks on its own, and is a few hundred lines at most, tests included. A slice's PRs form a stack: the first branch is cut from the latest `main`, and each later one from the branch before it, so they're reviewed and merged in order.
- **History is linear.** PRs are merged by rebase, so every commit on a branch lands on `main` as written. Merge commits and squash merges are turned off, and a ruleset requires linear history on `main`. Each commit must make sense on its own, with a message that explains it:
  - Address review comments with `git commit --fixup=<commit>`, so the reviewer sees exactly what changed. Once the PR is approved, fold the fixups in with `git rebase --autosquash origin/main` (no `-i` needed since Git 2.44), and push with `--force-with-lease`.
  - When a PR in a stack is merged, GitHub deletes its branch and points the next PR at `main`. Rebase the next branch with `git rebase origin/main`, which drops the commits that were just merged, then check it still passes and push with `--force-with-lease`.
- **Check before review.** Run `cargo xtask check`, and `cargo xtask check --gpu` for anything that touches the GPU. [`docs/testing.md`](docs/testing.md) describes the test tiers and how to write tests.
- **Outstanding work** is tracked in [`TODO.md`](TODO.md).

## Conventions

### Coordinate system

- Y is up.
- Right-handed.
- One world unit is one metre.

### Camera

- Cameras are described by physical parameters mirroring OpenUSD's `UsdGeomCamera`: `focalLength`, `horizontalAperture`, `verticalAperture`, `horizontalApertureOffset`, `verticalApertureOffset`, `projection`, `clippingRange`, `fStop`, `focusDistance`, `shutter:open`/`shutter:close` and `exposure`. Derive projection matrices from these rather than storing a field of view.
- Focal length and apertures are in millimetres. Clipping range and focus distance are in world units (metres).
- As in USD, a camera looks down its local -Z axis with +Y up.

### Rendering

- Shaders are written in Slang.
- Materials: support `UsdPreviewSurface` first, then move to OpenPBR via MaterialX once the pipeline works end to end.
- Use reverse depth (reverse-Z): the near plane maps to depth 1 and the far plane to 0. Clear depth to 0 and use a greater (or greater-equal) depth test.

### Code

- **Create infos:** functions that build a `Vk*CreateInfo` structure, and variables that hold one, are named `*_create_info` (`*_create_infos` for several). Vulkan has other kinds of info structure, such as begin, submit and present infos, and those keep plain `*_info` names.
- **`SAFETY` comments:** every `unsafe` block has its own `SAFETY:` comment. It can say "as above" only when a full `SAFETY:` comment comes earlier in the same scope, never on the first block.
- **Documentation:** every item is documented, including private ones. The workspace lints enforce this, except in test code, where it's a review rule.
- **Log levels are a contract:** `WARN` means someone should act, and warnings and errors fail tests and `--fail-on-warnings` runs. Fix a normal condition logged at `WARN` by changing its level. Suppress a message only when it's harmless and can't be fixed here, as `src/diagnostics.rs` describes.
- **Commit messages:** an imperative sentence as the summary, then a body explaining what changed and why.
