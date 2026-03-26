# Gradience 2.0 Advanced Tier Milestone

Date: 2026-03-17

## Scope

This milestone packages two advanced, optional workflows without changing default preflight behavior:

1. Core-space shared-basis audit (`merge-audit --compute-core-space`)
2. Inventory merge neighborhoods (`suggest-neighborhoods`)

## Outcome

Advanced tier is now documented and callable through stable wrappers:

- `gradience.api.compute_core_space_diagnostic(...)`
- `gradience.api.suggest_neighborhoods(...)`

Both remain:
- optional
- diagnostic-first
- non-default
- non-overriding for default recommendation logic

## Contract Safety

No breaking changes were introduced to:
- `gradience.adapter_qa/v1`
- `gradience.merge_qa_report/v1`
- `gradience.inventory_summary/v1`

Advanced outputs stay additive:
- optional `core_space` block in `MergeQAReport`
- independent `gradience.merge_neighborhoods/v1` artifact

## Evidence Path Used

- fixture-based validation for neighborhoods (`scripts/eval_neighborhoods.py`)
- benchmark + realism pass for core-space (`scripts/run_core_space_benchmark.py`)
- internal rubric and feature-status notes under `docs/internal/`

## Operational Guidance

- Keep advanced features available in practitioner workflows.
- Keep default getting-started centered on the core preflight spine.
- Revisit promotion decisions with corpus-backed usage evidence, not one-off examples.
