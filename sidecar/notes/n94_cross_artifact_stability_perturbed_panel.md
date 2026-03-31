# n94 — Cross-Artifact Stability Check: Perturbed Panel Construction

**Type:** substudy method note  
**Date:** 2026-03-31  
**Program:** Route2 Substudy 1 — Cross-Artifact Portability Stability Check  
**Status:** Stage B complete

---

## Objective

Construct a minimally perturbed panel that preserves original structure while introducing local case substitutions.

---

## Perturbation choices

- **LoRA:** 2 substitutions.
  - Same-task anchor swapped to a same-task near-miss substantial case.
  - Same-family case swapped to SST-2 x IMDB from targeted same-family confirmation.
- **Checkpoint delta:** 2 substitutions.
  - Same-family swapped to nearby seed variant (SST-2 s123 x Yelp).
  - Cross-task swapped to Yelp x QNLI within the same inventory.
- **LoHa:** no substitutions.
  - Fallback documented: only three same-task LoHa pilot pairs are available in-scope.

---

## Why this is still disciplined

- Panel size remains 9.
- Artifact classes remain unchanged (LoRA, LoHa, checkpoint delta).
- Relation coverage remains unchanged where originally testable.
- No new representation family introduced.
- No new field trial run introduced.

---

## Outputs

- `sidecar/results/route2_stability/cross_artifact/perturbed_panel_table.json`
- `sidecar/results/route2_stability/cross_artifact/perturbed_panel_table.md`
- `sidecar/results/route2_stability/cross_artifact/panel_diff_table.md`

---

## Stage B result

Stage B succeeds. The panel is perturbed enough to test brittleness but remains locally comparable to the original panel.
