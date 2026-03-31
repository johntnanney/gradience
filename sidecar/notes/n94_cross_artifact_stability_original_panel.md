# n94 — Cross-Artifact Stability Check: Original Panel Freeze

**Type:** panel freeze
**Date:** 2026-03-31
**Program:** Route 2 Stability and Replication Check, Substudy 1
**Depends on:** n76-n80 (cross-artifact compatibility program)
**Status:** Baseline frozen. Ready for perturbation.

---

## What this document is

A frozen reference for the stability check. The original cross-artifact panel (9 cases) and its conclusions (6 claims) are recorded here so that later stages compare against a fixed baseline, not a drifting memory.

---

## Original panel (9 cases)

| Case ID | Artifact class | Task pair | Relation | Evidence | Key metric |
|---------|---------------|-----------|----------|----------|------------|
| lora_same_task_sst2_pair | LoRA | SST-2 x SST-2 | same_task | behavioral_reported | merge_acc=0.876, compat=0.475 |
| lora_same_family_mnli_qnli | LoRA | MNLI x QNLI | same_family | structural_only | compat=0.431, routing_confusability=0.379 |
| lora_cross_task_sst2_agnews | LoRA | SST-2 x AG News | cross_task | behavioral_reported | merge_acc=0.842, compat=0.111 |
| loha_same_task_r4_r8 | LoHa | SST-2 x SST-2 | same_task | unknown | compat=0.102, risk=low |
| loha_same_task_r4_r16 | LoHa | SST-2 x SST-2 | same_task | unknown | compat=0.142, risk=low |
| loha_same_task_r8_r16 | LoHa | SST-2 x SST-2 | same_task | unknown | compat=0.145, risk=low |
| ckpt_same_task_sst2_seeds | checkpoint_delta | SST-2 s42 x SST-2 s123 | same_task | structural_only | compat=0.892, risk=medium |
| ckpt_same_family_sst2_yelp | checkpoint_delta | SST-2 x Yelp Polarity | same_family | structural_only | compat=0.652, risk=high |
| ckpt_cross_task_sst2_qnli | checkpoint_delta | SST-2 x QNLI | cross_task | structural_only | compat=0.626, risk=high |

## Claims under test

| Claim | Family | Original verdict | Product status |
|-------|--------|-----------------|----------------|
| A1: QA gating is a strong invariant | Strong invariant | strong | safe_to_expose |
| A2: Conservative narrowing is a strong invariant | Strong invariant | strong | safe_to_expose |
| B1: Task-relation ordering is moderately portable | Moderate invariant | moderate | safe_with_guardrail |
| B2: Same-family intermediate status is moderately portable | Moderate invariant | moderate | safe_with_guardrail |
| C1: Structural metrics are representation-local | Locality | strong_local | research_only |
| D1: Near-miss portability remains inconclusive | Thin/inconclusive | inconclusive | not_stable_enough |

## Known panel limitations

1. **LoHa is same-task only.** All three LoHa adapters are SST-2. No cross-task or same-family LoHa pairs exist.
2. **Behavioral evaluation sparse.** Only 2/9 cases (both LoRA) have behavioral merge data.
3. **Scores not comparable.** Different artifact classes use different scoring systems.
4. **Single backbone.** All cases use distilbert-base-uncased.

## Data locations

- Panel snapshot: `results/route2_stability/cross_artifact/original_panel_snapshot.json`
- Claims snapshot: `results/route2_stability/cross_artifact/original_claims_snapshot.json`
- Original program data: `results/cross_artifact_portability/`
