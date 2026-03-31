# Perturbed Panel Table (Substudy 1)

## Substitution summary

- LoRA: 2 substitutions, 1 unchanged case.
- LoHa: 0 substitutions (fallback; no nearby non-same-task replacements in current Ring 1 pilot).
- Checkpoint delta: 2 substitutions, 1 unchanged case.

## Cases

| case_id | class | relation | evidence | compatibility_score | pair_risk | source |
|---|---|---|---|---:|---|---|
| lora_same_task_near_miss_substantial_hate | lora | same_task | behavioral_reported_with_weak_source_caveat | 0.2115 | low | targeted_confirmation_near_miss T02 |
| lora_same_family_sst2_imdb | lora | same_family | behavioral_reported | 0.3140 | medium | targeted_confirmation_same_family T01 |
| lora_cross_task_sst2_agnews | lora | cross_task | behavioral_reported | 0.1114 | high | targeted_confirmation_same_family T01 |
| loha_same_task_r4_r8 | loha | same_task | unknown_no_behavioral_eval | 0.1018 | low | Ring 1 inventory pilot |
| loha_same_task_r4_r16 | loha | same_task | unknown_no_behavioral_eval | 0.1418 | low | Ring 1 inventory pilot |
| loha_same_task_r8_r16 | loha | same_task | unknown_no_behavioral_eval | 0.1452 | low | Ring 1 inventory pilot |
| ckpt_same_task_sst2_seeds | checkpoint_delta | same_task | structural_only | 0.8922 | medium | checkpoint_inventory_t02 pairwise |
| ckpt_same_family_sst2s123_yelp | checkpoint_delta | same_family | structural_only | 0.6410 | high | checkpoint_inventory_t02 pairwise |
| ckpt_cross_task_yelp_qnli | checkpoint_delta | cross_task | structural_only | 0.4891 | high | checkpoint_inventory_t02 pairwise |

## Coverage check

- Artifact classes preserved: LoRA, LoHa, checkpoint delta.
- Relation coverage preserved where originally testable: same_task, same_family, cross_task.
- Representation families unchanged: factor-based (LoRA/LoHa), summary-based checkpoint deltas.
