# Corpus Review Memo — 2026-03

Cycle metadata:
- cycle: `Corpus Review Cycle 01`
- freeze status: `active`
- minimum inventory target: `3`
- preferred decision bias: `no_change unless evidence is strong and narrow`

## Context Note

This memo is tied to `Corpus Review Cycle 01` and is descriptive-first.  
This cycle is for observing system behavior in aggregate, not redesigning policies or expanding feature scope.

## Metadata

- Review period: `2026-03-17` to `2026-03-17`
- Memo date: `2026-03-17`
- Author(s): `codex`
- Corpus root: `results/corpus`
- Summary inputs:
  - `results/corpus/summary_cycle01.json`
  - `results/corpus/summary_cycle01.md`
  - manifest set under `results/corpus/manifests/`

## Scope

- Inventories reviewed: `3`
- Inventory ids:
  - `study17_cache_triplet_20260317`
  - `core_space_real_adapter_triplet_20260317`
  - `canonical_test2_triplet_20260317`
- Corpus coverage placeholders:
  - adapter instances: `7` (manifest-level count)
  - unique adapters: `7` (manifest-level count)
  - pair reports: `9`
  - inventories with core-space usage: `3`
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
| Adapter instances | `7` |
| Unique adapters | `7` |
| Pair reports | `9` |
| Strict block candidate pairs | `9` |
| Neighborhood groups total | `9` |
| Neighborhood excluded total | `0` |
| Neighborhood boundary warnings total | `9` |

## Aggregate Behavior

### Recommended strategy distribution

| Strategy | Count | Share |
|---|---:|---:|
| `linear` | `5` | `55.6%` |
| `norm_equalized` | `3` | `33.3%` |
| `audit_aware` | `1` | `11.1%` |

### Dominant issue distribution

| Dominant issue | Count | Share |
|---|---:|---:|
| `none` | `5` | `55.6%` |
| `high_redundancy` | `3` | `33.3%` |
| `subspace_conflict` | `1` | `11.1%` |
| `norm_imbalance` | `0` | `0.0%` |
| `partial_redundancy` | `0` | `0.0%` |
| `unknown` | `0` | `0.0%` |

### Strict-QA block profile

- Block count: `9 / 9` pair reports.
- Most common block cause(s): `unknown_no_behavioral_eval` on at least one side (in practice both sides in this batch).
- Observed risk if strict mode is disabled: structural recommendations are available, but all decisions remain unsupported by behavioral evidence.

### Neighborhood behavior profile

- Grouping stability across inventories: all three inventories produced singleton clusters only (`3 groups` each).
- Exclusion behavior quality: no exclusions were emitted (`0`), because no `flagged_weak` inputs were present.
- Boundary warning usefulness: all inventories emitted full cross-group warnings (`3` each), indicating conservative behavior but low discrimination in this sample.

### Core-space usage profile (advanced optional)

- Number of pairs where `core_space` was computed: `3`.
- Share of total pairs: `33.3%` (`3 / 9`).
- Typical trigger condition for use: one designated ambiguous pair per inventory.
- Cases where it materially changed judgment: `2` pairs showed `pair_risk=low` with `core_space.status=incompatible`; this added tension signals but did not change default recommended strategy.

## Representative Cases

### Case A — useful neighborhood output

- Run id: `canonical_test2_triplet_20260317`
- Why chosen: includes mixed pair risk (`2 low`, `1 high`) within one inventory.
- What happened: neighborhoods remained singleton and emitted `3` boundary warnings. The output was conservative and consistent with pair-level caution, but did not surface a tighter local grouping.
- Practical takeaway: neighborhood output is stable and readable, but this cycle sample did not yet show high-compatibility clustering behavior.

### Case B — strict-QA block pressure

- Run id: `study17_cache_triplet_20260317` (pattern also present in the other two runs)
- Why chosen: all adapters lacked behavioral evaluation.
- What happened: all three pair reports are strict block candidates. Structural outputs were clean, but strict gating would block operational use.
- Practical takeaway: strict-QA remains correctly conservative; current corpus needs behaviorally evaluated inventories to test nuanced block dynamics.

### Case C — ambiguous pair with core-space

- Run id: `core_space_real_adapter_triplet_20260317`
- Pair id/path: `results/real_inventory_runs/20260317/core_space_real_adapter_triplet/reports/probe_uniform_core_space_report.json`
- Why chosen: pair was structurally low risk and explicitly selected as ambiguous.
- What happened: `pair_risk=low` with `core_space.status=incompatible` and low distortion. This indicates core-space can disagree with primary pair signals in realistic runs.
- Practical takeaway: core-space is adding differentiated diagnostic signal; keep it optional and review-oriented.

## Findings

1. Default structural recommendations are dominated by low-risk `linear` behavior in this first 3-inventory batch.
2. Strict-QA pressure is absolute in this sample (`9/9` block candidates) due missing behavioral eval, which is expected but limits policy interpretation.
3. Neighborhood outputs were fully conservative singletons with universal boundaries, so utility is currently “safe but not yet discriminative” in this small set.

## Risks / Unknowns

1. Adapter counting in manifests can undercount when multiple QA artifacts share the same `adapter_name` (for example checkpoint directories all named `checkpoint-50`).
2. Core-space disagreements with low pair-risk are present, but sample size is too small to justify calibration changes yet.

## Recommendation for Next Cycle

Choose one:

- `no_change` — keep all default and advanced logic unchanged for one more cycle.
- `targeted_calibration` — propose one narrow, evidence-backed adjustment.
- `defer` — insufficient evidence; continue collection.

Selected: `no_change`

Rationale: evidence is coherent but still narrow. The current run validates pipeline reproducibility and surfaces concrete signals (strict-QA pressure, neighborhood conservatism, core-space disagreement cases) without proving a single urgent miscalibration. The correct move is to collect additional real inventories before approving any behavior change.

## Appendices

- Corpus summary file: `results/corpus/summary_cycle01.json`
- Manifest ids reviewed:
  - `study17_cache_triplet_20260317`
  - `core_space_real_adapter_triplet_20260317`
  - `canonical_test2_triplet_20260317`
- Related decision memo: `docs/internal/selective-calibration-decision-2026-03.md`
