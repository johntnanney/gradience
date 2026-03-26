# Corpus Review Memo — Template

## Metadata

- Review period: `<YYYY-MM-DD> to <YYYY-MM-DD>`
- Memo date: `<YYYY-MM-DD>`
- Author(s): `<name(s)>`
- Corpus root: `results/corpus`
- Summary inputs:
  - `results/corpus/summary.json` (or cycle-specific summary)
  - manifest set under `results/corpus/manifests/`

## Scope

- Inventories reviewed: `<N>`
- Inventory ids:
  - `<run_id_1>`
  - `<run_id_2>`
  - `<run_id_3>`
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
| Inventories | `<int>` |
| Adapter instances | `<int>` |
| Unique adapters | `<int>` |
| Pair reports | `<int>` |
| Strict block candidate pairs | `<int>` |
| Neighborhood groups total | `<int>` |
| Neighborhood excluded total | `<int>` |
| Neighborhood boundary warnings total | `<int>` |

## Aggregate Behavior

### Recommended strategy distribution

| Strategy | Count | Share |
|---|---:|---:|
| `linear` | `<int>` | `<pct>` |
| `norm_equalized` | `<int>` | `<pct>` |
| `audit_aware` | `<int>` | `<pct>` |
| `<other>` | `<int>` | `<pct>` |

### Dominant issue distribution

| Dominant issue | Count | Share |
|---|---:|---:|
| `none` | `<int>` | `<pct>` |
| `norm_imbalance` | `<int>` | `<pct>` |
| `subspace_conflict` | `<int>` | `<pct>` |
| `high_redundancy` | `<int>` | `<pct>` |
| `partial_redundancy` | `<int>` | `<pct>` |
| `unknown` | `<int>` | `<pct>` |

### Strict-QA block profile

- Block count: `<int>`
- Most common block cause(s): `<status/value>`
- Observed risk if strict mode is disabled: `<1-3 sentences>`

### Neighborhood behavior profile

- Grouping stability across inventories: `<1-3 sentences>`
- Exclusion behavior quality: `<1-3 sentences>`
- Boundary warning usefulness: `<1-3 sentences>`

### Core-space usage profile (advanced optional)

- Number of pairs where `core_space` was computed: `<int>`
- Share of total pairs: `<pct>`
- Typical trigger condition for use: `<ambiguous case criteria>`
- Cases where it materially changed judgment: `<count + short note>`

## Representative Cases

### Case A — useful neighborhood output

- Run id: `<run_id>`
- Why chosen: `<1 sentence>`
- What happened: `<2-4 sentences>`
- Practical takeaway: `<1 sentence>`

### Case B — strict-QA block pressure

- Run id: `<run_id>`
- Why chosen: `<1 sentence>`
- What happened: `<2-4 sentences>`
- Practical takeaway: `<1 sentence>`

### Case C — ambiguous pair with core-space

- Run id: `<run_id>`
- Pair id/path: `<pair report path>`
- Why chosen: `<1 sentence>`
- What happened: `<2-4 sentences>`
- Practical takeaway: `<1 sentence>`

## Findings

1. `<finding 1>`
2. `<finding 2>`
3. `<finding 3>`

## Risks / Unknowns

1. `<risk 1>`
2. `<risk 2>`

## Recommendation for Next Cycle

Choose one:

- `no_change` — keep all default and advanced logic unchanged for one more cycle.
- `targeted_calibration` — propose one narrow, evidence-backed adjustment.
- `defer` — insufficient evidence; continue collection.

Selected: `<no_change | targeted_calibration | defer>`

Rationale: `<3-6 sentences>`

## Appendices

- Corpus summary file: `<path>`
- Manifest ids reviewed:
  - `<run_id_1>`
  - `<run_id_2>`
  - `<run_id_3>`
- Related decision memo: `<path or N/A>`
