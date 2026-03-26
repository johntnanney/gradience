# Core-Space Case Series Tracking

Tracks every core-space use across the real-inventory case series. Focus: the narrow class where ordinary pair_risk=low but core-space disagrees.

## Logging rule

For every case-study inventory, record whether any pair met:
- pair_risk = low
- core-space computed and returned marginal or incompatible

This is the main class of interest.

---

## Core-space usage log

### Use 1 — case_study_qnli4_realmix_20260318 (first published case study)

| Field | Value |
|-------|-------|
| Inventory ID | case_study_qnli4_realmix_20260318 |
| Pair | qnli_probe_elig × qnli_per_layer_elig |
| Ordinary pair-risk | ambiguous (low-risk pair with uncertain adapter) |
| Core-space status | — (not in target class; pair was selected as ambiguous, not strictly low-risk) |
| Did deeper result change judgment? | Yes — moved pair to caution track |
| How did it change action? | Pair deferred from active candidate set |

Note: this use preceded the formal tracking protocol. The pair was selected as "ambiguous" rather than strictly low-risk, so it is not a clean instance of the target class.

### Use 2 — roberta_mixed_evidence_4 (series wave 1)

| Field | Value |
|-------|-------|
| Inventory ID | roberta_mixed_evidence_4 |
| Pair | roberta_uniform_elig × roberta_probe_uncertain |
| Ordinary pair-risk | **low** |
| Core-space status | **marginal** (shared_basis_score=0.878) |
| Did deeper result change judgment? | Yes — low-risk demoted to caution |
| How did it change action? | Pair moved from "safe linear merge" to "proceed carefully or prefer norm-equalized" |

**Target class match: YES.** pair_risk=low, core-space=marginal.

### Use 3 — distilbert_large_pool_6 (series wave 1)

| Field | Value |
|-------|-------|
| Inventory ID | distilbert_large_pool_6 |
| Pair | final_probe_r16_ckpt50 × qnli_probe_elig |
| Ordinary pair-risk | **low** |
| Core-space status | **incompatible** (shared_basis_score=0.867) |
| Did deeper result change judgment? | Yes — low-risk flipped to incompatible |
| How did it change action? | Pair removed from active candidate set entirely |

**Target class match: YES.** pair_risk=low, core-space=incompatible. Strongest disagreement observed.

---

## Running counts

| Metric | Count |
|--------|-------|
| Total core-space uses | 3 |
| Target class instances (low-risk + core-space disagrees) | **2** |
| Core-space agreed with ordinary low-risk | **0** |
| Core-space returned marginal | 1 |
| Core-space returned incompatible | 1 |
| Times judgment changed | 3/3 |
| Times action changed | 3/3 |

### Use 4 — behaviorally_complete_5 (wave 2, Target 1)

3 cross-task low-risk pairs tested:

| Pair | Ordinary risk | Core-space | Shared basis | Changed? |
|------|-------------|------------|-------------|----------|
| uniform_elig × final_uniform_median_r16 | **low** | **incompatible** | 0.859 | yes |
| probe_elig × final_uniform_median_r16 | **low** | **incompatible** | 0.860 | yes |
| per_layer_elig × final_uniform_median_r16 | **low** | **marginal** | 0.931 | yes |

**Target class matches: 3 (all low-risk, all disagreed).**

### Use 5 — core_space_hunt_4 (wave 2, Target 4) — FULL CENSUS

All 6 pairs tested regardless of risk level:

| Pair | Task relation | Ordinary risk | Core-space | Shared basis |
|------|-------------- |-------------|------------|-------------|
| final × priority | same-group | **low** | **incompatible** | 0.824 |
| final × qprobe | cross-task | **low** | **incompatible** | 0.860 |
| final × quniform | cross-task | medium | **incompatible** | 0.859 |
| priority × qprobe | cross-task | **low** | **marginal** | 0.864 |
| priority × quniform | cross-task | **low** | **marginal** | 0.863 |
| qprobe × quniform | same-task | **low** | **incompatible** | 0.807 |

**Target class matches: 5 (all 5 low-risk pairs disagreed). This includes same-task and same-group pairs.**

### Use 6 — two_cluster_neighborhood_7 (wave 2, Target 2)

2 cross-group low-risk pairs tested:

| Pair | Ordinary risk | Core-space | Shared basis |
|------|-------------|------------|-------------|
| qnli_probe × final_probe_ckpt50 | **low** | **incompatible** | 0.867 |
| qnli_uniform × priority | **medium** | **marginal** | 0.863 |

**Target class match: 1 (low-risk pair disagreed).**

---

## Updated running counts

| Metric | Wave 1 | Wave 2 | Total |
|--------|--------|--------|-------|
| Total core-space uses | 3 | 11 | 14 |
| Target class instances (low-risk + disagrees) | 2 | 9 | **11** |
| Core-space agreed with low-risk | 0 | 0 | **0** |
| Core-space returned marginal | 1 | 4 | 5 |
| Core-space returned incompatible | 1 | 7 | 8 |
| Times judgment changed on low-risk | 2/2 | 9/9 | **11/11** |

## Observations (updated after wave 2)

1. **Core-space has disagreed with ordinary low-risk in every case tested (11/11).** This is no longer a small-sample pattern — it is a consistent structural finding across 5 inventories.

2. **Same-task and same-group pairs are not exempt.** The T4 census showed qprobe × quniform (same task, same rank) at 0.807 and final × priority (same group, same rank) at 0.824. Different rank allocation policies create different basis representations regardless of task.

3. **The shared_basis_score range is 0.807-0.931.** Most values cluster in 0.82-0.87. The one outlier (0.931, per_layer_elig × final, marginal) suggests per-layer adapters may share more basis structure than uniform/probe.

4. **Core-space may be too strict.** A 100% disagreement rate means the diagnostic does not discriminate among pairs — it flags all of them. Either the "compatible" threshold is set too high, or this adapter pool is genuinely fractured at depth.

5. **Core-space remains correctly tiered as non-default.**

## Verified adjudication update (2026-03)

The downstream merge evaluation was performed on freshly trained DistilBERT adapters (3 SST-2, 3 QNLI, all independently verified above base). Key findings:

- **Same-task merges flagged as incompatible were safe.** All 6 same-task pairs preserved accuracy within ~1.2pp of best individual, despite core-space shared_basis scores in the 0.870-0.873 range.
- **Cross-task merges degraded substantially (~8-18pp).** Consistent with core-space incompatibility, but ordinary pair-risk already separated these from same-task safe pairs.
- **Core-space overwarned on same-task seed variants** and added only modest additional discrimination inside the already-unsafe cross-task group.

**Updated interpretation:** The 11/11 structural disagreement rate is real, but structural disagreement alone does not predict behavioral harm. Core-space is structurally informative but its behaviorally useful role is narrower and more regime-dependent than the tracking table alone suggests. Its strongest supported role is in genuinely ambiguous task relationships where ordinary pair-risk is not already decisive.

See: `docs/internal/verified_adjudication_implications.md`
