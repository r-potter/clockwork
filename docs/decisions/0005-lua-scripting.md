# 0005: Embed Lua through mlua for scripting

- **Status:** Proposed
- **Date:** 2026-10-15

## Context

Scripting lets behaviour and demos change without recompiling the engine. The candidates were Lua, Rhai, WebAssembly (wasmtime) and Python (pyo3). Lua is the established choice in games: small, fast, easy to sandbox, and familiar to the people the engine is shown to.

## Decision

Scripts are Lua, embedded through the `mlua` crate. The `0008-scripting` design chooses between Luau, which has a built-in sandbox and gradual types, and LuaJIT, which is faster but has a weaker sandbox and limited platform support.

## Consequences

- `mlua` builds the Lua VM from C source, adding a C toolchain to the build. The `vendored` feature avoids needing a system Lua.
- Script-defined systems must fit into the ECS scheduler, which needs access declarations for code the scheduler can't analyse. `0005-ecs` must leave room for this.
- Revisit this if scripts need to be written in several languages, which WebAssembly would allow.
