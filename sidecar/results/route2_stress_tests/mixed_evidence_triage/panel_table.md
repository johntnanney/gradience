# Mixed-Evidence Triage Stress-Test Panel

| case_id | artifact_class | backbone | task_relation | evidence_regime | prior_route2_profile | role_in_panel |
|---|---|---|---|---|---|---|
| anchor_clear_retained_irony | lora_adapter_pair | distilbert-base-uncased | same_task | clear | aggregation_invariant_safe | anchor_retained |
| anchor_blocked_same_task_checkpoint | full_checkpoint_pair | distilbert-base-uncased | same_task | blocked | qa_dominance_override | anchor_blocked |
| anchor_blocked_cross_task_checkpoint | full_checkpoint_pair | distilbert-base-uncased | cross_task | blocked | weak_low_value_blocked | anchor_blocked |
| review_mixed_cross_task_sst2_mrpc | full_checkpoint_pair | distilbert-base-uncased | cross_task | mixed | mixed_evidence_review_high_structure | review |
| review_mixed_cross_task_yelp_mrpc | full_checkpoint_pair | distilbert-base-uncased | cross_task | mixed | mixed_evidence_review_lower_structure | review |
| optional_same_family_clear_sst2_imdb | lora_adapter_pair | distilbert-base-uncased | same_family | clear | same_family_optional_safe_like | same_family_optional |
| optional_same_family_mixed_sst2_yelp_a | full_checkpoint_pair | distilbert-base-uncased | same_family | mixed | same_family_optional_review | same_family_optional |
| optional_same_family_mixed_sst2_yelp_b | full_checkpoint_pair | distilbert-base-uncased | same_family | mixed | same_family_optional_review_weaker | same_family_optional |
