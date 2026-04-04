# Documentation

This directory is organized by **audience**, **stability**, and **purpose**.

## Technical Report

**[`technical-report.md`](technical-report.md)** — the end-to-end argument for spectral triage: why spectral geometry carries merge-compatibility information, the conjunctive failure mechanism, field trial validation, and bounded scope. Start here if you want the full picture in one document. Serves both practitioners (Sections 1, 4, 5) and researchers (Sections 2, 3, 6, 7).

**[`field-trial-retrospective.md`](field-trial-retrospective.md)** — the story behind the results: what we expected, what surprised us, and what changed in the product. Companion to the technical report's §5.

## Program Snapshot

**[`docs/strategy/state-of-program-april-2026.md`](strategy/state-of-program-april-2026.md)** — canonical April 2026 decision memo covering validated, bounded, exploratory, paused, and GPU-blocked lines, plus the next proving-ground gate.

**[`docs/product_brief.md`](product_brief.md)** — one-page internal product brief: what Gradience is, what it solves now, what it does not do yet, and the next proving grounds.

## Claims Boundary

**[`docs/claims.md`](claims.md)** — authoritative source for what is validated, bounded, not yet claimed, and currently recommended for product use.

## Consolidation Backlog

**[`docs/consolidation_backlog.md`](consolidation_backlog.md)** — product-facing structural consolidation checklist derived from the research inventory.

## Start Here

- [`docs/00_start_here/README.md`](00_start_here/README.md)
- [`docs/00_start_here/project-map.md`](00_start_here/project-map.md)
- [`docs/00_start_here/stable-vs-experimental.md`](00_start_here/stable-vs-experimental.md)
- [`docs/00_start_here/current-bounded-conclusions.md`](00_start_here/current-bounded-conclusions.md)
- [`docs/00_start_here/bounded-validation-summary.md`](00_start_here/bounded-validation-summary.md)
- [`docs/00_start_here/demo-paths.md`](00_start_here/demo-paths.md)

## Product Docs Hierarchy

- [`docs/getting_started/README.md`](getting_started/README.md)
- [`docs/workflows/README.md`](workflows/README.md)
- [`docs/reference/README.md`](reference/README.md)
- [`docs/explanations/README.md`](explanations/README.md)
- [`docs/research/README.md`](research/README.md)

## Zones

1. **Getting Started** — install, quickstart, first triage example.
   - [`docs/getting_started/README.md`](getting_started/README.md)
2. **Workflows** — canonical run guides and report interpretation.
   - [`docs/workflows/README.md`](workflows/README.md)
3. **Reference** — CLI, artifact schemas, terminology/statuses.
   - [`docs/reference/README.md`](reference/README.md)
4. **Explanations** — conceptual docs and boundaries.
   - [`docs/explanations/README.md`](explanations/README.md)
5. **Research** — technical report, theory, findings, bounded/experimental status.
   - [`docs/research/README.md`](research/README.md)
6. **Architecture** — substrate layers, seams, boundaries, design decisions.
   - [`docs/architecture/README.md`](architecture/README.md)
7. **Development** — implementation/design/maintenance references.
   - [`docs/development/README.md`](development/README.md)
8. **Theory** — analytical derivations and spectral-geometry notes.
   - [`docs/theory/README.md`](theory/README.md)

## Notes

- Historical and raw research records remain in `sidecar/`, `field_trials/`, and `experiments/`.
- This docs tree is the curated front-end; it does not replace those archives.
