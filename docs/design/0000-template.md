# NNNN: Title

- **Status:** Proposed
- **Author:**
- **Created:** YYYY-MM-DD
- **Accepted:**
- **Depends on:** (other designs or ADRs)

<!--
Guidance for authors, to delete once the section is written:
- Write for a reviewer who knows graphics and Rust, but not this codebase.
- Prefer concrete signatures, data layouts and numbers over adjectives.
- Every requirement needs an acceptance criterion that a test can check.
- Say what is out of scope just as clearly as what is in scope.
-->

## Context

Why this design is needed now. What exists today, with file references, and what problem or limitation it addresses.

## Goals

A numbered list, so that requirements and tests can refer to them.

## Non-goals

What this design deliberately doesn't do, and where (if anywhere) that is handled instead.

## Requirements

| ID | Requirement | Acceptance criterion |
|---|---|---|
| R1 | | How we know it's met. Usually a named test. |

## Architecture

Components, their responsibilities and their relationships. Include Mermaid diagrams for structure and for data or control flow.

### Ownership and lifetimes

Who owns each object, what it borrows, when it's destroyed, and what guarantees destruction order.

### Threading

Which types are `Send` or `Sync`, which operations may run concurrently, and how that's enforced.

## API

The public interface as Rust signatures, with doc comments. Show a short example of typical use.

## Data structures and layout

In-memory layout where it matters for performance or GPU compatibility. File formats, with an example.

## Error handling

Which failures are errors returned to the caller, which are panics (bugs), and which are diagnostics. Error messages are lowercase with no trailing punctuation.

## Performance

Budgets (time, memory, allocations per frame), and how each is measured. "Not performance-critical" is a valid answer if justified.

## Alternatives considered

Each alternative, its advantages, and why it was rejected.

## Test plan

By tier, as defined in [0001, Test tiers](0001-foundations.md#test-tiers):

- **T0 unit / property / snapshot / soundness:** which invariants, what's generated, what's snapshotted.
- **T1 GPU / golden image:** which behaviour, with which validation settings, and which capabilities each test needs.
- **T2 fuzz / bench:** targets and the regressions they guard against.
- **Fixtures:** what test data is needed and where it lives.

Map each requirement to at least one test.

## Implementation slices

| Branch | Contents | Requirements covered |
|---|---|---|
| `topic-part` | | R1, R2 |

Each slice must build, pass every check, and be useful on its own.

## Risks

Each risk, its likelihood and impact, and the mitigation.

## Open questions

Questions for review. Each is resolved before acceptance, either by an answer recorded here or by moving it to a later design.

## As built

Filled in at close-out: what differs from this design, and why.

## Revision history

| Date | Change |
|---|---|
| YYYY-MM-DD | Proposed |
