# OA-v2 Gate Report

- label: `oa_v2_30_40_r1`
- strict-naive input: `field_trials/over_accumulation_followup/oa_v2_30_40_r1_strict_naive_rerun_results.json`
- cohort range target: `30-40`
- analyzed pairs: `30`
- cohort design gate pass: `True`
- threshold/policy gate pass: `False`
- promotion ready: `False`

## Rule Status
- Rule 1 abs Spearman gain >= 0.15: `False` (value=-0.06751595780557779)
- Rule 2 recall gain >= 0.20: `False` (value=-0.08333333333333334)
- Rule 3 sign consistency >= 0.70: `False` (value=0.5555555555555556)
- Rule 4 interpretability decomposition: `True`

OA-v1 remains authoritative unless threshold/policy gate passes.
