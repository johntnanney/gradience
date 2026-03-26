# Milestone: Inventory Preflight Packaging

**Date:** 2026-03-25

## What changed

Gradience is now explicitly packaged as an inventory preflight system for LoRA adapter merging. This milestone consolidates months of empirical work into a usable operational path.

## Additions

- **`docs/inventory-preflight.md`** — canonical workflow doc with quickstart, step-by-step guide, signal contribution table, and honest scope statement
- **`docs/examples/mixed-task-inventory-walkthrough.md`** — flagship worked example: 6 adapters, 4 tasks, 15 pairs reduced to 2 (87% search-space reduction)
- **`docs/examples/same-task-control-walkthrough.md`** — contrast case: advisory silence, confirmatory behavior, honest about lower value
- **`examples/inventory_preflight_mixed_task/`** — self-contained artifact bundle (24 files: QA, pair reports, inventory summary, neighborhoods)
- **`examples/inventory_preflight_same_task_control/`** — self-contained control bundle (9 files)
- **`assets/preflight_before_after.svg`** — visual showing 15 undifferentiated pairs → 2 safe + 13 caution + isolated

## Changes

- **README** reframed: tagline is now "Inventory preflight for LoRA adapter merging"; lead section centers search-space reduction; before/after figure added; walkthrough links prominent
- **Report language**: advisory section renamed from "TASK-RELATIONSHIP ADVISORY" to "TASK-BOUNDARY WARNING" with action line ("Do not prioritize this pair unless you have a specific reason to merge across task boundaries")
- **Regime map** updated with cross-task severity decomposition, cross-backbone replication results, and explicit established/open/closed status

## Empirical evidence behind this packaging

| Evidence | Scope |
|----------|-------|
| Same-task safety | 49 pairs, 0 material degradations, 3 blind-spot studies |
| Cross-task boundary detection | 132+ advisory checks, 0 false positives, 2 backbones |
| Cross-task severity subtypes | 56 pairs across 2 backbones, 4 severity levels identified |
| Severity grading limitation | Neither task-pair identity nor core-space shared-basis replicates across backbones |

## What is now established

- Source QA as the first decision anchor
- Pair-risk as the default structural layer
- Task-relationship advisory as stable interpretive infrastructure
- Neighborhoods as inventory compression (6+ adapters)
- Same-task safety on small encoder models

## What is explicitly closed

- Same-task rescue logic (no actionable blind spot found)
- Task-pair severity featureization (backbone-dependent)
- Core-space as a severity grading signal (correlation sign flips across backbones)

## What remains open research

- Cross-task severity grading (no reliable cross-backbone signal exists)
- Extension to larger models and decoder-only architectures
