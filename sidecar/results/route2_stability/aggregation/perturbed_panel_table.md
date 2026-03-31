# Perturbed Aggregation Panel (Substudy 2)

## Substitution summary

- Total replacements: 3/12
- Merge-facing: 1 replacement
- Routing-facing: 1 replacement
- Triage-facing: 1 replacement

## Cases

| case_id | group | relation | qa_regime | key metric |
|---|---|---|---|---|
| mrg_safe_same_task | merge_facing | same_task | clear | compat 0.475 |
| mrg_near_miss_marginal | merge_facing | same_task | clear | compat 0.127 |
| mrg_cross_task_control | merge_facing | cross_task | clear | compat 0.111 |
| rte_same_task_confusable | routing_facing | same_task | structural_only | routing 0.481 (high) |
| mnli_rte_moderate_alt | routing_facing | same_family | structural_only | routing 0.262 (moderate) |
| qnli_rte_separable | routing_facing | same_family | structural_only | routing 0.222 (low) |
| tri_same_task_qa_blocked | triage_facing_qa_blocked | same_task | blocked | compat 0.892 |
| tri_same_family_qa_blocked | triage_facing_qa_blocked | same_family | blocked | compat 0.652 |
| tri_cross_task_qa_review_alt | triage_facing_qa_blocked | cross_task | mixed | compat 0.798 |
| tri_same_task_qa_clear | triage_facing_qa_clear | same_task | clear | compat 0.475 |
| tri_same_family_qa_clear | triage_facing_qa_clear | same_family | clear | compat 0.314 |
| tri_cross_task_qa_clear | triage_facing_qa_clear | cross_task | clear | compat 0.111 |

## Coverage check

- Same aggregation families as original: worst-case, distributional, QA-dominant, QA-gated distributional.
- Same group structure preserved.
- Same backbone and artifact families preserved.
