# OA-v2 Regime-Pure Next-Run Plan (Pre-Committed)

## Current Status
- OA-v2 remains **exploratory**.
- OA-v1 remains the authoritative policy/report path.
- Latest 30-pair strict-naive gate outcome (label `oa_v2_30_40_r1`):
  - cohort design gate: pass
  - threshold/policy gate: fail

## Why One More Run Is Still Plausible
- OA-v2 improved overall rank relation vs OA-v1 in the 30-pair run.
- The intended high-overlap/low-conflict slice remained small (`n=9`), limiting stability.
- A final regime-pure rerun is justified only to test whether intended-slice coverage was the main blocker.

## Regime-Pure Design Contract
- Total cohort size target: `30–40` pairs.
- Strict naive only (`uniform_linear`, checked at merge-plan level).
- Dataset-matched source baselines.
- Intended slice definition (locked):
  - `mean_overlap >= 0.25`
  - `conflict_fraction <= 0.10`
- Intended-slice coverage target (locked):
  - aim for `>= 20` intended-slice pairs if feasible from available inventory.

## Feasibility Snapshot (Current Inventory)
- From current activation inventory:
  - pairs meeting intended slice at `mean_overlap >= 0.25` and `conflict_fraction <= 0.10`: `9`
  - pairs meeting `mean_overlap >= 0.20` and `conflict_fraction <= 0.10`: `13`
- Therefore `>=20` intended-slice coverage is **not feasible** without either:
  1. expanding the adapter inventory, or
  2. pre-registering a relaxed intended-slice threshold.

## Promotion Gate (Locked)
No threshold/policy update unless all hold on the regime-pure rerun:
1. `abs_spearman_gain >= 0.15` on intended slice (OA-v2 vs OA-v1).
2. poor-merge recall gain `>= 0.20` at matched alert rate (or stronger precision/recall tradeoff).
3. leave-one-out sign consistency `>= 0.70`.
4. interpretability decomposition preserved (`interaction_primary` + `concentration_secondary`).

## Pre-Run Decision Required (to keep design honest)
Choose exactly one path before the final rerun:
1. **Inventory expansion path**: keep intended slice at `0.25/0.10`, add enough auditable pairs to make `>=20` feasible.
2. **Threshold-locked path**: keep current inventory, pre-register a relaxed intended slice (for example `0.20/0.10`) and keep all other gate rules unchanged.

## Pre-Committed Stop Rule
- If any gate rule fails on the regime-pure rerun:
  - keep OA-v2 exploratory,
  - no threshold/policy promotion,
  - pause further OA-v2 policy-escalation work in this cycle.

## Canonical Inputs/Outputs
- Last gate report:
  - `/Users/john/code/gradience/field_trials/analytical_spectral_geometry/gate_report_oa_v2_30_40_r1.json`
- Last failure-anatomy readout:
  - `/Users/john/code/gradience/field_trials/analytical_spectral_geometry/failure_anatomy_oa_v2_30_40_r1.md`
- Next run should emit:
  - prefixed strict-naive rerun outputs under `field_trials/over_accumulation_followup/`
  - cross-check JSON/MD
  - gate report JSON/MD

## Execution Note
This document is intentionally pre-committed to prevent ad hoc threshold drift or open-ended rerun loops.
