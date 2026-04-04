# n120 -- Collapse vs Contamination Behavioral Rerun

**Type:** rerun findings note  
**Date:** 2026-03-31  
**Program:** Route2 Collapse vs Contamination Replication  
**Stage:** C  
**Depends on:** n119, `sidecar/results/example_semantics/predictions/*.json`, `sidecar/scripts/analyze_example_behavior.py`  
**Status:** complete

---

## Objective

Recompute core behavioral metrics on the replication panel and test whether channel separation remains visible.

---

## Rerun method

Used the same metric definitions as the existing example-semantics pipeline:

1. observed failure rate (`1 - merged accuracy`)
2. confidence collapse (`conf_merged < 0.4` and `conf_source_a > 0.6`)
3. high-confidence wrong (`merged wrong` and `conf_merged > 0.8`)
4. confusion/neither-source rate
5. preservation rate

Slice targets use deterministic index partitions:

- even slice: `index % 2 == 0`
- odd slice: `index % 2 == 1`

---

## Per-target rerun results

| target_id | expected_channel | failure_rate | confidence_collapse_rate | high_confidence_wrong_rate | confusion_or_neither_source_rate | preservation_rate |
|---|---|---:|---:|---:|---:|---:|
| R1_FR02_case | collapse_like | 0.336 | 0.056 | 0.008 | 0.154 | 0.7906 |
| R2_FR01_even_slice | collapse_like | 0.344 | 0.068 | 0.000 | 0.160 | 0.7959 |
| R3_CT01_even_slice | contamination_like | 0.176 | 0.008 | 0.056 | 0.140 | 0.8745 |
| R4_CT01_odd_slice | contamination_like | 0.172 | 0.004 | 0.036 | 0.148 | 0.8739 |

---

## Key observations

1. Confidence channels remain separated:
   - collapse-like: high collapse (0.056-0.068), near-zero high-confidence wrong (0.000-0.008).
   - contamination-like: low collapse (0.004-0.008), elevated high-confidence wrong (0.036-0.056).
2. Neither-source rates remain near-matched across channels (~0.14-0.16), preserving the "similar novel-failure pressure, different channel" interpretation.
3. Contamination slices are internally stable (even/odd maintain the same signature shape).
4. Collapse signature persists across both nearby case replication (FR-02) and slice replication (FR-01 even).

---

## Output artifacts

- `sidecar/results/route2_stress_tests/collapse_vs_contamination/behavior_summary.json`
- `sidecar/results/route2_stress_tests/collapse_vs_contamination/channel_comparison_table.json`
- `sidecar/results/route2_stress_tests/collapse_vs_contamination/channel_comparison_table.md`
- `sidecar/figures/collapse_vs_contamination_replication_matrix.svg`
