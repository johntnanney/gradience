# n98 -- Aggregation Stability Check: Original Panel Freeze

**Type:** substudy setup note
**Date:** 2026-03-31
**Program:** Route2 Substudy 2 -- Aggregation-Sensitive Compatibility Stability Check
**Stage:** A
**Depends on:** n81-n85
**Status:** complete

---

## Objective

Freeze the original aggregation-sensitive panel and the exact claims under test before any perturbation rerun.

---

## Frozen baseline

- Original panel source: `sidecar/results/aggregation_sensitive_compatibility/panel_table.json`
- Original outputs source: `sidecar/results/aggregation_sensitive_compatibility/aggregation_comparison.json`
- Aggregation families held fixed:
  - worst-case
  - distributional
  - QA-dominant
  - QA-gated distributional

Panel structure remains:

- 12 total cases
- 3 merge-facing
- 3 routing-facing
- 3 triage-facing QA-blocked/mixed
- 3 triage-facing QA-clear

---

## Claims frozen for stability testing

- A1: Aggregation is the first major decision seam.
- B1: Worst-case collapses some cases that distributional keeps graded.
- B2: Distributional exposes gradation hidden by worst-case in merge-like settings.
- B3: QA-dominant is a distinct operational family.
- C1: Compact (4-6) aggregation-sensitive taxonomy is viable.
- C2: Both aggregation-invariant and aggregation-sensitive case types exist.
- D1: Route 2 workflow should treat aggregation as a design variable.

---

## Outputs

- `sidecar/results/route2_stability/aggregation/original_panel_snapshot.json`
- `sidecar/results/route2_stability/aggregation/original_claims_snapshot.json`
- `sidecar/notes/n98_aggregation_stability_original_panel.md` (this note)
