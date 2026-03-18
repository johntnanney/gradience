# Corpus Review Memo — 2026-05

Cycle metadata:
- cycle: `Corpus Review Cycle 03`
- freeze status: `active`
- minimum inventory target: `3`
- preferred range: `4–5` if cleanly available
- preferred decision bias: `no_change unless evidence is strong and narrow`

## Context Note

This memo is tied to `Corpus Review Cycle 03` and is descriptive-first.  
Cycle-03 is a targeted diversity acquisition cycle, not a redesign pass.

## Required Cycle-03 Gates

All four must be explicitly addressed:

1. at least one inventory likely to produce non-singleton neighborhoods
2. at least one inventory with mixed behavioral evidence so strict-QA gets a real middle-case test
3. explicit low-risk/core-space mismatch tracking
4. corpus identity hardening follow-through so adapter-instance counting is trustworthy

## Gate Status (Pass/Fail)

| Gate | Status | Evidence |
|---|---|---|
| Non-singleton neighborhood target | `PASS` | Non-singleton neighborhoods observed in all 4 inventories. |
| Mixed behavioral-evidence strict-QA middle case | `PASS` | Mixed-status inventories: `cycle03_qnli_mixed_behavior_triplet_20260317`, `cycle03_roberta_mixed_evidence_triplet_20260317`, `cycle03_real_adapter_triplet_20260317`. |
| Low-risk/core-space mismatch tracking | `PASS` | Tracker populated: 4 mismatches among 8 low-risk pairs (50.0%). |
| Identity hardening follow-through | `PASS` | Implemented post-review as a scoped infrastructure hardening patch in `scripts/summarize_corpus.py`; adapter-instance counting is now identity-safe. |

Post-review identity hardening note:
The identity-hardening gate initially failed because corpus adapter-instance counting was not yet identity-safe under repeated checkpoint-style labels. This has since been addressed by a scoped patch in `scripts/summarize_corpus.py`, which now resolves deterministic adapter-instance keys with explicit precedence and dedupes counts across manifests. Corpus adapter totals changed under corrected counting semantics, while strategy, issue, strict-block, and neighborhood metrics remained unchanged. This is infrastructure trustworthiness hardening, not a policy calibration.

## Metadata

- Review period: `2026-03-17` to `2026-03-17`
- Memo date: `2026-03-17`
- Author(s): `Codex`
- Corpus root: `results/corpus`
- Summary inputs:
  - `results/corpus/summary_cycle03.json`
  - `results/corpus/summary_cycle03.md`
  - manifest set under `results/corpus/manifests/`

## Scope

- Inventories reviewed: `4`
- Inventory ids:
  - `cycle03_qnli_all_eligible_triplet_20260317`
  - `cycle03_qnli_mixed_behavior_triplet_20260317`
  - `cycle03_roberta_mixed_evidence_triplet_20260317`
  - `cycle03_real_adapter_triplet_20260317`
- Coverage checks:
  - non-singleton neighborhood candidate included: `yes`
  - mixed behavioral-evidence inventory included: `yes`
  - low-risk/core-space mismatch tracker populated: `yes`
  - identity hardening follow-through recorded: `yes (status=implemented post-review patch)`
- Feature scope included:
  - default preflight artifacts (`adapter_qa`, `merge_qa_report`, `inventory_summary`)
  - advanced optional workflows (`core_space`, `merge_neighborhoods`)
- Out of scope:
  - default policy changes
  - threshold retuning
  - new features

## Corpus Snapshot

| Metric | Value |
|---|---:|
| Inventories | `4` |
| Adapter instances | `9` |
| Unique adapters | `9` |
| Unique adapter display names | `10` |
| Pair reports | `12` |
| Strict block candidate pairs | `6` |
| Neighborhood groups total | `5` |
| Neighborhood excluded total | `2` |
| Neighborhood boundary warnings total | `1` |

## Aggregate Behavior

### Recommended strategy distribution

| Strategy | Count | Share |
|---|---:|---:|
| `linear` | `8` | `66.7%` |
| `norm_equalized` | `1` | `8.3%` |
| `audit_aware` | `3` | `25.0%` |

### Dominant issue distribution

| Dominant issue | Count | Share |
|---|---:|---:|
| `none` | `8` | `66.7%` |
| `norm_imbalance` | `2` | `16.7%` |
| `subspace_conflict` | `0` | `0.0%` |
| `high_redundancy` | `0` | `0.0%` |
| `partial_redundancy` | `2` | `16.7%` |
| `unknown` | `0` | `0.0%` |

### Strict-QA middle-case profile

- Mixed-evidence inventory id(s): `cycle03_qnli_mixed_behavior_triplet_20260317`, `cycle03_roberta_mixed_evidence_triplet_20260317`, `cycle03_real_adapter_triplet_20260317`
- Block count in mixed-evidence slice: `6`
- Non-block count in mixed-evidence slice: `3`
- Interpretation: The mixed-evidence slice produced both strict-block and non-block outcomes in each inventory (`2` blocked + `1` non-blocked pair per triplet). This indicates strict-QA is exercising a meaningful middle case instead of collapsing to always-block or never-block behavior on mixed behavioral evidence. The current behavior is conservative but discriminative.

### Neighborhood behavior profile

- Grouping stability across inventories: Grouping remained conservative and coherent across all four inventories, with likely-safe clusters forming where low-risk linear relations dominate.
- Cases with non-singleton neighborhoods: `4`
- Exclusion behavior quality: Exclusions aligned with `flagged_weak` adapters in both mixed-behavior inventories (`2` exclusions total), matching policy intent.
- Boundary warning usefulness: One boundary warning appeared in the mixed-evidence RoBERTa run and corresponded to the highest-risk cross-group relation, which is the expected use case.

### Core-space usage profile (advanced optional)

- Number of pairs where `core_space` was computed: `4`
- Share of total pairs: `33.3%` (`4/12`)
- Typical trigger condition for use: ambiguous pairs that were structurally plausible by default pair risk but still needed shared-basis compatibility signal.
- Cases where it materially changed judgment: `4` review-level mismatches where `pair_risk=low` but `core_space.status` was `marginal` or `incompatible`.

## Low-Risk / Core-Space Mismatch Tracker

Population definition:
- `pair_risk=low`
- `core_space.status in {marginal, incompatible}`

| Pair report path | pair_risk | core_space.status | core_space.shared_basis_score | Action impact |
|---|---|---|---:|---|
| `results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/reports/qnli_per_layer_vs_probe_core_space_report.json` | `low` | `incompatible` | `0.9078` | Review note recorded; default policy unchanged |
| `results/real_inventory_runs/20260317/cycle03_qnli_mixed_behavior_triplet/reports/qnli_per_layer_vs_probe_core_space_report.json` | `low` | `incompatible` | `0.9078` | Review note recorded; default policy unchanged |
| `results/real_inventory_runs/20260317/cycle03_roberta_mixed_evidence_triplet/reports/roberta_uniform_vs_probe_core_space_report.json` | `low` | `marginal` | `0.8778` | Review note recorded; default policy unchanged |
| `results/real_inventory_runs/20260317/cycle03_real_adapter_triplet/reports/final_vs_qnli_core_space.json` | `low` | `marginal` | `0.9309` | Review note recorded; default policy unchanged |

Summary:
- mismatch count: `4`
- mismatch share among low-risk pairs: `50.0%` (`4/8`)
- repeatability across inventories: `yes` (observed in all 4 cycle-03 inventories)
- interpretation: The mismatch population is now recurring rather than incidental. Core-space appears to surface a structural concern that is not captured by pair risk alone in a subset of low-risk pairs. This strengthens the case for keeping core-space in advanced optional use for ambiguous pair review, but does not yet justify changing default recommendations.

## Identity Hardening Follow-through

- Status: `implemented (post-review)`
- Method used for adapter-instance counting: corpus summary now resolves deterministic adapter-instance keys and dedupes unique instances across manifests by identity key.
- Evidence path(s): `scripts/summarize_corpus.py`, `tests/inventory/test_corpus_scripts.py`, `docs/internal/corpus-identity-hardening-note.md`
- Operational note: this changed adapter-instance totals under corrected dedupe semantics and did not change strategy, issue, strict-block, or neighborhood metrics.

## Representative Cases

### Case A — non-singleton neighborhood candidate

- Run id: `cycle03_real_adapter_triplet_20260317`
- Why chosen: Uses realistic benchmark adapters and produced both grouping and exclusion behavior.
- What happened: The inventory produced one likely-safe two-member neighborhood (`final_uniform_median_r16`, `priority_probe_r16`) and excluded `qnli_per_layer_r8` due to `flagged_weak`. Pair risk remained mostly low, while one core-space run on a low-risk pair returned `marginal`.
- Practical takeaway: Conservative grouping plus weak-source exclusion behaved as intended in a realistic pool.

### Case B — strict-QA mixed-evidence middle case

- Run id: `cycle03_roberta_mixed_evidence_triplet_20260317`
- Why chosen: Contains `eligible`, `uncertain`, and `unknown_no_behavioral_eval` in the same inventory.
- What happened: Strict-QA behavior split cleanly into two blocked pairs involving `unknown_no_behavioral_eval` and one non-block pair between `eligible` and `uncertain`. Neighborhood output produced a caution singleton plus one likely-safe pair group, with one boundary warning.
- Practical takeaway: Strict-QA and neighborhoods both exhibited middle-case discrimination rather than all-or-nothing behavior.

### Case C — low-risk/core-space mismatch

- Run id: `cycle03_real_adapter_triplet_20260317`
- Pair id/path: `results/real_inventory_runs/20260317/cycle03_real_adapter_triplet/reports/final_vs_qnli_core_space.json`
- Why chosen: Representative low-risk pair where core-space added contradictory structural signal.
- What happened: Pair risk was `low`, but core-space status was `marginal` with a shared-basis score of `0.9309`. The pair remained usable under default logic, but the mismatch was recorded for manual ambiguity review.
- Practical takeaway: Core-space remains valuable as an advanced diagnostic when pair risk is not fully decisive.

## Findings

1. Cycle-03 reached preferred collection coverage (`4` inventories) with all major diversity targets represented.
2. Strict-QA exhibited stable middle-case behavior in mixed-evidence inventories (not all-block, not all-pass).
3. Low-risk/core-space mismatches repeated across all inventories, supporting continued advanced optional use of core-space.

## Risks / Unknowns

1. Historical snapshots taken before this hardening patch may show adapter-instance totals that are not directly comparable to the corrected identity-safe counts.
2. Mismatch recurrence is strong, but sample size is still small for any policy-level calibration.

## Recommendation for Next Cycle

Choose one:

- `no_change` — keep all default and advanced logic unchanged for one more cycle.
- `targeted_calibration` — propose one narrow, evidence-backed adjustment.
- `defer` — insufficient evidence; continue collection.

Selected: `no_change`

Rationale: Cycle-03 evidence is coherent and repeatable enough to continue usage, but not yet strong enough for a default-policy calibration move. The strongest signal is recurring low-risk/core-space mismatch, which supports advanced diagnostic usage rather than recommendation changes. Strict-QA middle-case behavior was stable in mixed-evidence slices, and neighborhoods remained conservative and useful. Identity hardening has now been implemented as post-review infrastructure correction; this improves corpus trustworthiness but does not change the cycle’s policy conclusion.

## Appendices

- Corpus summary file: `results/corpus/summary_cycle03.md`
- Manifest ids reviewed:
  - `cycle03_qnli_all_eligible_triplet_20260317`
  - `cycle03_qnli_mixed_behavior_triplet_20260317`
  - `cycle03_real_adapter_triplet_20260317`
  - `cycle03_roberta_mixed_evidence_triplet_20260317`
- Related decision memo: `docs/internal/selective-calibration-decision-2026-05.md`
