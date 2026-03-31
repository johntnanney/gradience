# Cross-Artifact Portability Panel

Generated: 2026-03-31

## Panel Overview

| Property | Value |
|----------|-------|
| Panel size | 9 cases |
| Artifact classes | 3 (LoRA, LoHa, checkpoint delta) |
| Backbone | distilbert-base-uncased (all cases) |
| Task relations covered | same_task (3 classes), same_family (2), cross_task (2) |

## Cases

| Case ID | Class | Task A | Task B | Relation | Compatibility | Risk | Evidence | Scenarios |
|---------|-------|--------|--------|----------|--------------|------|----------|-----------|
| lora_same_task_sst2_pair | LoRA | sst2 | sst2 | same_task | 0.475 | retained | behavioral | merge, triage |
| lora_same_family_mnli_qnli | LoRA | mnli | qnli | same_family | 0.431 | moderate | structural | merge, routing |
| lora_cross_task_sst2_agnews | LoRA | sst2 | ag_news | cross_task | 0.111 | control | behavioral | merge, triage |
| loha_same_task_r4_r8 | LoHa | sst2 | sst2 | same_task | 0.102 | low | none | triage |
| loha_same_task_r4_r16 | LoHa | sst2 | sst2 | same_task | 0.142 | low | none | triage |
| loha_same_task_r8_r16 | LoHa | sst2 | sst2 | same_task | 0.145 | low | none | triage |
| ckpt_same_task_sst2_seeds | ckpt_delta | sst2 | sst2 | same_task | 0.892 | medium | structural | triage |
| ckpt_same_family_sst2_yelp | ckpt_delta | sst2 | yelp | same_family | 0.652 | high | structural | triage |
| ckpt_cross_task_sst2_qnli | ckpt_delta | sst2 | qnli | cross_task | 0.626 | high | structural | triage |

## Known Gaps

1. **LoHa is same-task only.** All three LoHa adapters were trained on SST-2. No same-family or cross-task LoHa pairs exist in the current panel.
2. **LoHa has no behavioral evaluation.** All three LoHa pairs are `unknown_no_behavioral_eval`.
3. **Checkpoint deltas lack behavioral evaluation.** Structural and triage outputs only.
4. **LoRA routing cases lack behavioral merge evaluation.** The MNLI x QNLI pair has routing data but no merge accuracy.
5. **Compatibility scores are not directly comparable across classes.** LoRA uses merge-audit compatibility, LoHa uses shimmed merge-audit, checkpoint deltas use summary-based pairwise scoring. The scales and semantics differ.

## Task Relation Coverage Matrix

| Relation | LoRA | LoHa | Checkpoint Delta |
|----------|------|------|-----------------|
| same_task | 1 case (behavioral) | 3 cases (structural only) | 1 case (structural) |
| same_family | 1 case (structural) | - | 1 case (structural) |
| cross_task | 1 case (behavioral) | - | 1 case (structural) |

## Representation Paths

| Class | Representation | Extraction Method |
|-------|---------------|-------------------|
| LoRA | Native A/B factors | Direct from adapter state dict |
| LoHa | Materialized SVD refactored to synthetic A/B | Shim: `loha_to_lora_state_dict(mode="materialized")` |
| Checkpoint delta | Layerwise summary (Repr C) | `extract_checkpoint_delta.py` + `layer_summary_repr.py` |
