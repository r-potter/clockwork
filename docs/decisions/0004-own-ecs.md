# 0004: Build our own entity-component system

- **Status:** Proposed
- **Date:** 2026-10-15

## Context

Large-scale simulation is one of the things the engine showcases. Mature Rust ECS libraries exist (`bevy_ecs`, `hecs`, `flecs` through bindings), and adopting one would be faster. But the storage layout, the scheduler and the extraction for rendering are where the interesting engineering is, and where this engine's simulations would be tuned.

## Decision

We design and build the ECS ourselves: archetype storage, queries, deferred structural changes, and a parallel scheduler. The design is `0005-ecs`.

## Consequences

- The ECS is a major piece of work, with `unsafe` storage code. It must be `miri`-clean, property-tested against a simple reference model, and benchmarked, with `bevy_ecs` and `hecs` as comparison baselines in the benchmarks.
- We control the layout end to end, including what extraction for the renderer copies and when.
- Features that libraries provide for free (relations, change detection, observers) are added only when a simulation needs them.
