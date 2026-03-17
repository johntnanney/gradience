# Corpus Review Memo — 2026-04

Cycle metadata:
- cycle: `Corpus Review Cycle 02`
- freeze status: `active`
- minimum inventory target: `3`
- preferred range: `4–5` if cleanly available
- preferred decision bias: `no_change unless evidence is strong and narrow`

## Context Note

This memo is tied to `Corpus Review Cycle 02` and is descriptive-first.  
Cycle-02 expands inventory diversity and tracks disagreement populations; it is not a redesign pass.

## Metadata

- Review period: `2026-03-17` to `2026-03-17`
- Memo date: `2026-03-17`
- Author(s): `Codex + investigator`
- Corpus root: `results/corpus`
- Summary inputs:
  - `results/corpus/summary_cycle02.json`
  - `results/corpus/summary_cycle02.md`
  - manifest set under `results/corpus/manifests/`

## Scope

- Inventories reviewed: `3`
- Inventory ids:
  - `cycle02_qnli_triplet_20260317`
  - `cycle02_roberta_sst2_triplet_20260317`
  - `cycle02_final_test_quartet_20260317`
- Coverage checks:
  - non-checkpoint adapter identities included: `yes` (run-local named adapter aliases)
  - semantically varied inventory included: `yes` (distilbert + roberta pools)
  - medium/high risk mix included: `yes`
  - non-singleton neighborhood case included: `no`
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
| Inventories | `3` |
| Adapter instances | `10` |
| Unique adapters | `10` |
| Pair reports | `12` |
| Strict block candidate pairs | `0` |
| Neighborhood groups total | `10` |
| Neighborhood excluded total | `0` |
| Neighborhood boundary warnings total | `12` |

## Aggregate Behavior

### Recommended strategy distribution

| Strategy | Count | Share |
|---|---:|---:|
| `linear` | `5` | `41.7%` |
| `norm_equalized` | `1` | `8.3%` |
| `audit_aware` | `6` | `50.0%` |
| `<other>` | `0` | `0.0%` |

### Dominant issue distribution

| Dominant issue | Count | Share |
|---|---:|---:|
| `none` | `5` | `41.7%` |
| `norm_imbalance` | `5` | `41.7%` |
| `subspace_conflict` | `0` | `0.0%` |
| `high_redundancy` | `1` | `8.3%` |
| `partial_redundancy` | `1` | `8.3%` |
| `unknown` | `0` | `0.0%` |

### Strict-QA block profile

- Block count: `0`
- Most common block cause(s): `none observed in cycle-02 slice`
- Observed risk if strict mode is disabled: `No strict-block pressure was observed in this slice; no strict policy action is justified from these runs.`

### Neighborhood behavior profile

- Grouping stability across inventories: `High. All three inventories produced consistent caution-neighborhood singleton partitions under current conservative rules.`
- Exclusion behavior quality: `No exclusions in this slice; behavior appears stable but not stress-tested against weak adapters in cycle-02 data yet.`
- Boundary warning usefulness: `Warnings were dense and directionally consistent with high/medium-risk pairs; useful for signaling cross-group caution.`
- Cases with non-singleton neighborhoods: `0`

### Core-space usage profile (advanced optional)

- Number of pairs where `core_space` was computed: `3`
- Share of total pairs: `25.0%`
- Typical trigger condition for use: `one manually selected ambiguous pair per inventory`
- Cases where it materially changed judgment: `1 (low-risk pair reported as core-space incompatible)`

## Low-Risk / Core-Space Mismatch Tracker

Population definition:
- `pair_risk=low`
- `core_space.status in {marginal, incompatible}`

| Pair report path | pair_risk | core_space.status | core_space.shared_basis_score | Action impact |
|---|---|---|---:|---|
| `results/real_inventory_runs/20260317/cycle02_qnli_triplet/reports/qnli_per_layer_vs_probe_r32_core_space_report.json` | `low` | `incompatible` | `0.908` | `review note` |

Summary:
- mismatch count: `1`
- mismatch share among low-risk pairs: `20.0%` (1/5)
- interpretation: `Core-space disagreement exists and is detectable in realistic runs. The signal is non-zero but sparse in this cycle. Evidence is not yet strong enough to justify default-policy changes.`

## Representative Cases

### Case A — useful neighborhood output

- Run id: `cycle02_final_test_quartet_20260317`
- Why chosen: `largest cycle-02 inventory with mixed pair risk and six pair reports`
- What happened: `Neighborhood output remained conservative and fully explainable, with singleton groups and dense cross-group warnings aligned with observed high/medium risks.`
- Practical takeaway: `Current neighborhood behavior is stable but still conservative-first.`

### Case B — strict-QA block pressure

- Run id: `cycle02_qnli_triplet_20260317`
- Why chosen: `representative low/medium risk inventory to check strict block behavior`
- What happened: `No strict block candidates were produced in the cycle-02 slice despite structural flags and unknown behavioral eligibility.`
- Practical takeaway: `No strict calibration action is indicated from cycle-02 data.`

### Case C — ambiguous pair with core-space

- Run id: `cycle02_qnli_triplet_20260317`
- Pair id/path: `results/real_inventory_runs/20260317/cycle02_qnli_triplet/reports/qnli_per_layer_vs_probe_r32_core_space_report.json`
- Why chosen: `low pair-risk case with optional core-space computed`
- What happened: `Pair risk remained low (`none` dominant issue), while core-space status was `incompatible` with high shared-basis score and low distortion.`
- Practical takeaway: `Keep tracking low-risk/core-space disagreement as a specific evaluation population.`

## Findings

1. Strategy/issue distributions are coherent and non-collapsed across the three inventories.
2. Core-space disagreement with low-risk pair assessment appeared in 1/5 low-risk pairs.
3. Neighborhoods remained stable and conservative, with no exclusions and all singleton groups.

## Risks / Unknowns

1. No non-singleton neighborhood formation yet in cycle-02, so clustering utility is not fully stress-tested.
2. Adapter identity quality still depends on aliasing for checkpoint-style sources; follow-up hardening remains relevant.

## Recommendation for Next Cycle

Choose one:

- `no_change` — keep all default and advanced logic unchanged for one more cycle.
- `targeted_calibration` — propose one narrow, evidence-backed adjustment.
- `defer` — insufficient evidence; continue collection.

Selected: `no_change`

Rationale: `Cycle-02 produced usable coverage and one meaningful low-risk/core-space mismatch, but not enough repeated evidence for calibration. Default and advanced behavior should remain frozen while collecting 1–2 additional diverse inventories.`

## Appendices

- Corpus summary file: `results/corpus/summary_cycle02.md`
- Manifest ids reviewed:
  - `cycle02_qnli_triplet_20260317`
  - `cycle02_roberta_sst2_triplet_20260317`
  - `cycle02_final_test_quartet_20260317`
- Related decision memo: `docs/internal/selective-calibration-decision-2026-04.md`
