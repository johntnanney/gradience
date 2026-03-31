# Aggregation Panel Diff (Original -> Perturbed)

| scenario_family | original_case | perturbed_case | change_type | why reasonable |
|---|---|---|---|---|
| merge | mrg_near_miss_substantial | mrg_near_miss_marginal | same-role same-task near-miss swap | Drawn from same near-miss T02 inventory; preserves merge-facing near-miss function while perturbing severity. |
| routing | mnli_qnli_moderate | mnli_rte_moderate_alt | same-role moderate same-family routing swap | Drawn from same routing pilot; preserves moderate confusable function with nearby pair. |
| triage | tri_cross_task_qa_review | tri_cross_task_qa_review_alt | same-role mixed-QA cross-task swap | Drawn from same checkpoint T02 preflight; preserves mixed-evidence triage function with higher compatibility variant. |

## Unchanged anchors

- merge: `mrg_safe_same_task`, `mrg_cross_task_control`
- routing: `rte_same_task_confusable`, `qnli_rte_separable`
- triage QA-blocked: `tri_same_task_qa_blocked`, `tri_same_family_qa_blocked`
- triage QA-clear: `tri_same_task_qa_clear`, `tri_same_family_qa_clear`, `tri_cross_task_qa_clear`

## Diff summary

- 3 of 12 cases replaced (25%).
- One replacement per scenario family.
- No aggregation definitions changed.
