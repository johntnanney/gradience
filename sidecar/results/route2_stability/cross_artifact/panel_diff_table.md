# Panel Diff (Original -> Perturbed)

## LoRA

| original_case | perturbed_case | change_type | why reasonable |
|---|---|---|---|
| lora_same_task_sst2_pair | lora_same_task_near_miss_substantial_hate | same relation, different same-task instance | Nearby same-task case from existing targeted confirmation; stresses weak-source/near-miss behavior without leaving LoRA or CPU scope. |
| lora_same_family_mnli_qnli | lora_same_family_sst2_imdb | same relation, different family/source | Nearby same-family case from existing same-family targeted confirmation with behavioral follow-through. |
| lora_cross_task_sst2_agnews | lora_cross_task_sst2_agnews | unchanged | Keeps one fixed cross-task LoRA anchor for comparability. |

## LoHa

| original_case | perturbed_case | change_type | why fallback used |
|---|---|---|---|
| loha_same_task_r4_r8 | loha_same_task_r4_r8 | unchanged | No nearby non-same-task LoHa replacement exists in current Ring 1 pilot without expanding scope. |
| loha_same_task_r4_r16 | loha_same_task_r4_r16 | unchanged | Same reason as above. |
| loha_same_task_r8_r16 | loha_same_task_r8_r16 | unchanged | Same reason as above. |

## Checkpoint delta

| original_case | perturbed_case | change_type | why reasonable |
|---|---|---|---|
| ckpt_same_task_sst2_seeds | ckpt_same_task_sst2_seeds | unchanged | Keep same-task checkpoint anchor fixed. |
| ckpt_same_family_sst2_yelp | ckpt_same_family_sst2s123_yelp | same relation, nearby seed substitution | Same inventory, same family, same representation path; minimal substitution. |
| ckpt_cross_task_sst2_qnli | ckpt_cross_task_yelp_qnli | same relation, nearby cross-task substitution | Same inventory and base, selected from same low-compatibility region. |

## Diff summary

- Total substitutions: 4/9 cases.
- Structure preserved: yes (3 classes, 9 cases, same relation coverage where originally testable).
- Scope expansion: none.
