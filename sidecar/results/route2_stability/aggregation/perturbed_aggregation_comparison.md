# Perturbed Aggregation Comparison (Route2 Substudy 2)

**Source panel:** `sidecar/results/route2_stability/aggregation/perturbed_panel_table.json`
**Aggregation families:** worst-case, distributional, QA-dominant, QA-gated distributional

## Agreement distribution

| pattern | count |
|---|---:|
| full_agreement | 2 |
| partial_agreement | 8 |
| strong_divergence | 2 |

## Per-case outputs

| case_id | worst_case | distributional | qa_dominant | qa_gated_distributional | agreement_pattern |
|---|---|---|---|---|---|
| mrg_safe_same_task | merge_caution | routing_confusable | qa_clear | confusable | partial_agreement |
| mrg_near_miss_marginal | merge_caution | routing_needs_disambiguation | qa_clear | needs_disambiguation | partial_agreement |
| mrg_cross_task_control | merge_risky | routing_separable | qa_clear | separable | full_agreement |
| rte_same_task_confusable | merge_caution | routing_confusable | qa_unclear | qa_unclear | partial_agreement |
| mnli_rte_moderate_alt | merge_caution | routing_needs_disambiguation | qa_unclear | qa_unclear | partial_agreement |
| qnli_rte_separable | merge_caution | routing_separable | qa_unclear | qa_unclear | strong_divergence |
| tri_same_task_qa_blocked | merge_caution | routing_confusable | qa_blocked | qa_blocked | strong_divergence |
| tri_same_family_qa_blocked | merge_risky | routing_needs_disambiguation | qa_blocked | qa_blocked | partial_agreement |
| tri_cross_task_qa_review_alt | merge_risky | routing_separable | qa_review | qa_review | partial_agreement |
| tri_same_task_qa_clear | merge_caution | routing_confusable | qa_clear | confusable | partial_agreement |
| tri_same_family_qa_clear | merge_caution | routing_needs_disambiguation | qa_clear | needs_disambiguation | partial_agreement |
| tri_cross_task_qa_clear | merge_risky | routing_separable | qa_clear | separable | full_agreement |

## Qualitative interpretation

1. The aggregation seam remains visible: only 2/12 cases are invariant across families.
2. Worst-case still flattens routing-facing gradation that distributional preserves.
3. QA-dominant still overrides structurally positive cases in blocked/mixed triage regimes.
4. QA-gated distributional continues to preserve both evidence constraints and structural gradation.
