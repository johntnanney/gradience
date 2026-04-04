# Preflight Policy: Cross-Artifact Contracts

This document defines the consistency contracts between Gradience's three artifact types. These contracts are tested by `tests/test_cross_artifact_policy.py`.

## Artifact Spine

```
AdapterQAArtifact → MergeQAReport → InventorySummary
   (per-adapter)      (per-pair)      (inventory-level)
```

## Eligibility Status Flow

The `EligibilityStatus` enum has four values:

| Status | Meaning |
|--------|---------|
| `eligible` | Adapter outperforms base model on target task |
| `uncertain` | Evidence exists but is inconclusive |
| `flagged_weak` | Adapter appears weaker than base model |
| `unknown_no_behavioral_eval` | No behavioral evaluation provided |

### How status propagates

1. **QA artifact** records the status in `eligibility.status`.
2. **Merge report** copies each adapter's status into `adapter_a.eligibility_status` / `adapter_b.eligibility_status`. If no QA artifact was provided, the value is `null`.
3. **Inventory summary** counts statuses in `adapter_status_counts` (from QA artifacts) and identifies `strict_qa_block_candidates` (from merge reports).

## Strict-QA Blocking

The `--strict-qa` flag (on `merge-audit`) and the `strict_qa_block_candidates` count (in inventory summaries) use the same blocking rule:

A pair is blocked if **either** adapter has:
- `eligibility_status == "flagged_weak"`
- `eligibility_status == "unknown_no_behavioral_eval"`
- `eligibility_status` is `null` (no QA provided)

## Strategy/Risk Alignment

Merge reports map `pair_risk` to `recommended_strategy`:

| Risk Level | Dominant Verdict | Strategy | Meaning |
|------------|-----------------|----------|---------|
| `low` | any | `linear` | Safe to merge with simple linear combination |
| `medium` | `imbalanced` | `linear` | Merge with rebalanced coefficients (down-weight dominant adapter) |
| `medium` | other | `norm_equalized` | Merge with norm equalization for redundancy/partial-overlap cases |
| `high` | any | `audit_aware` | Requires careful audit-guided merge or manual review |

> **Why not norm_equalized for imbalance?** Norm equalization amplifies the weaker
> adapter's spectrum, increasing cross-term spectral inflation when subspaces
> overlap. Rebalanced linear coefficients are more appropriate.

The `recommended_action` field is explanatory prose and does not override `recommended_strategy`.

## Inventory Aggregation Rules

- `adapter_status_counts` sums to the number of QA artifacts
- `pair_risk_counts` sums to the number of merge reports
- `recommended_strategy_counts` sums to the number of merge reports
- `strict_qa_block_candidates` is at most the number of merge reports
- Count maps only include keys with non-zero values

## Neighborhood Exclusion Alignment

`suggest-neighborhoods` uses the same policy direction as strict-QA screening:

- default exclusion includes `flagged_weak`
- `--strict-qa` exclusion includes `flagged_weak`, `unknown_no_behavioral_eval`, and missing QA status (`null`)

Neighborhood output is diagnostic inventory guidance, not a replacement for pair-level `MergeQAReport` decisions.
