# Mixed-Evidence Triage Aggregation Comparison

## Agreement distribution

| pattern | count |
|---|---:|
| full_agreement | 0 |
| partial_agreement | 6 |
| strong_divergence | 2 |

## Per-case outputs

| case_id | worst_case | distributional | qa_dominant | final_triage_interpretation | agreement_pattern |
|---|---|---|---|---|---|
| anchor_clear_retained_irony | merge_caution | routing_confusable | qa_clear | evaluate_first | partial_agreement |
| anchor_blocked_same_task_checkpoint | merge_caution | routing_confusable | qa_blocked | blocked_high_structure | strong_divergence |
| anchor_blocked_cross_task_checkpoint | merge_risky | routing_separable | qa_blocked | blocked_low_value | strong_divergence |
| review_mixed_cross_task_sst2_mrpc | merge_caution | routing_needs_disambiguation | qa_review | review_priority_high | partial_agreement |
| review_mixed_cross_task_yelp_mrpc | merge_risky | routing_separable | qa_review | review_priority_lower | partial_agreement |
| optional_same_family_clear_sst2_imdb | merge_caution | routing_needs_disambiguation | qa_clear | review_optional_clear | partial_agreement |
| optional_same_family_mixed_sst2_yelp_a | merge_risky | routing_needs_disambiguation | qa_review | review_optional_medium | partial_agreement |
| optional_same_family_mixed_sst2_yelp_b | merge_risky | routing_needs_disambiguation | qa_review | review_optional_medium_lower | partial_agreement |

## Soft-middle interpretation

1. QA-dominant remains a coherent primary partition (`qa_clear`, `qa_review`, `qa_blocked`).
2. Same-family optional cases remain review/clear and do not collapse into blocked outcomes.
3. Secondary structural gradation remains useful inside `qa_review` for review ordering, with guardrails.
