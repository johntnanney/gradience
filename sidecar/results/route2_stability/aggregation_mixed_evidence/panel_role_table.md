# Mixed-Evidence Panel Role Table

| case_id | panel_role | role_function | reason |
|---|---|---|---|
| anchor_clear_retained_irony | anchor | clear retained reference | Provides a strong-evidence same-task baseline.
| anchor_blocked_same_task_checkpoint | blocked | QA-override anchor | High structural compatibility but both sources weak.
| anchor_blocked_cross_task_checkpoint | blocked | low-value blocked control | Cross-task plus both weak sources.
| review_mixed_cross_task_sst2_mrpc | review | review-worthy mixed case | Mixed evidence (flagged_weak + eligible), medium risk.
| review_mixed_cross_task_yelp_mrpc | review | lower-structure mixed case | Mixed evidence (uncertain + eligible), higher risk.
| optional_same_family_clear_sst2_imdb | optional | safe-like optional anchor | Same-family clear-evidence case with strong follow-through.
| optional_same_family_mixed_sst2_yelp_a | optional | optional review case A | Same-family mixed evidence (flagged_weak + uncertain).
| optional_same_family_mixed_sst2_yelp_b | optional | optional review case B | Second same-family mixed case to test middle consistency.
