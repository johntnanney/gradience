# Decision-Profile Table

Generated: 2026-03-31T01:23:15.537832+00:00

| Profile ID | Name | Merge stance | Routing stance | Triage stance |
|---|---|---|---|---|
| P1_redundant_confusable | Redundant / Routing-Confusable | typically redundant; deduplicate unless constrained | consider_dedup or strong disambiguation | only actionable if source QA is clear |
| P2_overlap_needs_disambiguation | Overlap / Needs Disambiguation | merge-compatible but not automatically high-value | needs_disambiguation | review candidate if QA is not weak |
| P3_merge_ok_routing_separable | Merge-OK / Routing-Separable | can be acceptable under merge lens | easily_routed | not necessarily prioritized without quality evidence |
| P4_qa_blocked_structurally_nontrivial | QA-Blocked / Structurally Nontrivial | defer despite structural plausibility | structural separability/confusability secondary to source quality | qa_blocked dominates |
| P5_same_family_optional | Same-Family Optional | can perform near same-task in validated examples | often moderate confusability, needs disambiguation | informational caution / optional probe |
| P6_cross_task_low_value_control | Cross-Task Low-Value Control | control/risky; usually below same-task alternatives | often separable and not confusable | caution or exclusion unless strong contrary evidence |

## Representative cases

- `P1_redundant_confusable`: rte_seed_pair_confusable, tri_same_task_near_miss_sst2_pair_t02
- `P2_overlap_needs_disambiguation`: mnli_qnli_moderate_confusable, tri_same_family_review_sst2_yelp_t02
- `P3_merge_ok_routing_separable`: qnli_rte_separable
- `P4_qa_blocked_structurally_nontrivial`: tri_same_task_near_miss_sst2_pair_t02, tri_cross_task_weak_region_yelp_qnli_t02
- `P5_same_family_optional`: mrg_safe_same_task_sst2_sst2_t01, tri_same_family_review_sst2_yelp_t02
- `P6_cross_task_low_value_control`: mrg_cross_task_control_sst2_agnews_t01, tri_cross_task_weak_region_yelp_qnli_t02
