# Architecture decision records

Short records of cross-cutting decisions: choices that constrain more than one subsystem, or that a newcomer would otherwise question. Larger designs belong in [`../design/`](../design/README.md).

Records are numbered and never deleted. A decision that changes gets a new record, and the old one's status becomes *Superseded by NNNN*.

## Format

Copy this into `NNNN-<title>.md`:

```markdown
# NNNN: Decision, stated as a sentence

- **Status:** Proposed | Accepted | Superseded by NNNN
- **Date:** YYYY-MM-DD

## Context

The forces at play, and what prompted the decision.

## Decision

What we do, stated plainly.

## Consequences

What becomes easier, what becomes harder, and what would make us revisit this.
```

## Index

| Number | Decision | Status |
|---|---|---|
| 0001 | [Keep the engine a single binary crate](0001-single-binary-crate.md) | Proposed |
| 0002 | [Target a bleeding-edge Vulkan feature set with no fallbacks](0002-vulkan-feature-baseline.md) | Proposed |
