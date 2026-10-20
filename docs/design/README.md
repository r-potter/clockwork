# Design documents

Every step of the roadmap starts as a design document in this directory. A design is reviewed, revised and accepted **before** any implementation of it starts.

## Process

1. **Draft.** Copy [`0000-template.md`](0000-template.md) to `NNNN-<topic>.md`, using the next free number, on a branch named `design/<topic>`. Set the status to *Proposed*.
2. **Review.** The design is reviewed on its branch and revised there. Review comments that change the design are answered by changing the document, not only in discussion, so the document stays the source of truth.
3. **Accept.** When approved, set the status to *Accepted*, add the date, and merge the branch. Update the index below and, once it exists, the roadmap table in `docs/ARCHITECTURE.md`.
4. **Implement.** Each implementation slice the design lists is reviewed as one or more small, focused pull requests, stacked so they're merged in order: the first branch is cut from `main`, and each later one from the branch before it. Each PR covers a single concern, builds and passes the checks on its own, and names the slice, design sections and test plan items it covers. PRs are merged by rebase, so `main` has a linear history (see `CLAUDE.md`).
5. **Close out.** When the last slice is merged, fill in the design's *As built* section with anything that differs from the design and why, set the status to *Implemented*, and update `TODO.md`.

A change of direction after acceptance is either:
- a **revision**: a dated entry in the design's *Revision history*, for changes that don't alter its goals; or
- a **superseding design**: a new document, with the old one's status set to *Superseded by NNNN*, for changes that do.

Small cross-cutting decisions that don't need a full design are recorded as [ADRs](../decisions/README.md).

## Statuses

| Status | Meaning |
|---|---|
| Proposed | Under review. Not to be implemented yet. |
| Accepted | Approved. Implementation may start. |
| Implemented | All slices merged. *As built* notes filled in. |
| Superseded by NNNN | Replaced. Kept for history. |
| Withdrawn | Abandoned before acceptance. Kept for history, with the reason. |

## Index

| Number | Design | Status |
|---|---|---|
| 0000 | [Template](0000-template.md) | n/a |
| 0001 | [Foundations](0001-foundations.md) | Implemented |
