# Decision-Dependent Compatibility Panel

Generated: 2026-03-31T01:23:15.533843+00:00

Total cases: 9

| Case ID | Group | Artifact | Relation | Scenarios | Structural | Behavioral |
|---|---|---|---|---|---|---|
| mrg_safe_same_task_sst2_sst2_t01 | merge_sensitive | lora_adapter_pair | same_task | merge, triage | yes | yes |
| mrg_near_miss_substantial_hate_t02 | merge_sensitive | lora_adapter_pair | same_task | merge, triage | yes | yes |
| mrg_cross_task_control_sst2_agnews_t01 | merge_sensitive | lora_adapter_pair | cross_task | merge, triage | yes | yes |
| rte_seed_pair_confusable | routing_sensitive | lora_adapter_pair | same_task | merge, routing | yes | no |
| mnli_qnli_moderate_confusable | routing_sensitive | lora_adapter_pair | same_family | merge, routing | yes | no |
| qnli_rte_separable | routing_sensitive | lora_adapter_pair | same_family | merge, routing | yes | no |
| tri_same_task_near_miss_sst2_pair_t02 | triage_sensitive | full_checkpoint_pair | same_task | triage | yes | yes |
| tri_same_family_review_sst2_yelp_t02 | triage_sensitive | full_checkpoint_pair | same_family | triage | yes | yes |
| tri_cross_task_weak_region_yelp_qnli_t02 | triage_sensitive | full_checkpoint_pair | cross_task | triage | yes | no |

## Informative notes

- `mrg_safe_same_task_sst2_sst2_t01`: Retained same-task anchor in targeted confirmation T01.
- `mrg_near_miss_substantial_hate_t02`: Near-miss substantial case from targeted confirmation T02.
- `mrg_cross_task_control_sst2_agnews_t01`: Cross-task control in targeted confirmation T01.
- `rte_seed_pair_confusable`: Same-task pair with highest routing confusability in pilot.
- `mnli_qnli_moderate_confusable`: Same-family NLI pair with moderate routing confusability.
- `qnli_rte_separable`: Clearly separable routing pair despite same NLI family.
- `tri_same_task_near_miss_sst2_pair_t02`: No retained pairs; strongest same-task near-miss probe.
- `tri_same_family_review_sst2_yelp_t02`: Same-family checkpoint pair routed to informational caution/review.
- `tri_cross_task_weak_region_yelp_qnli_t02`: Cross-task weak/risky region in checkpoint triage inventory.
