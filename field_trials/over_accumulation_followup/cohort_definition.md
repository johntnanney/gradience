# Over-Accumulation Follow-up Cohort Definition

## Scope
- This first-pass cohort is retrospective and built from existing local merge outputs with recomputed current over-accumulation diagnostics.
- Rows included: **21** pair-level evaluations from local field-trial artifacts.
- `--only-naive` filter active: **no**.

## Group Rules
- High-overlap gate: `mean_overlap >= 0.25` and `conflict_layer_fraction <= 0.10`.
- Group 1 (`group_1_high_overlap_low_proxy`): high-overlap rows with lower half over-accumulation scores.
- Group 2 (`group_2_high_overlap_elevated_proxy` or `..._official`): high-overlap rows with upper half scores, or official `watch/elevated` advisory when present.
- Group 3 (`group_3_non_overlap_pathology_anchor`): lower-overlap rows where imbalance/cross-task pathology is the primary issue.

## Important Caveats
- In this cohort, official pair-level over-accumulation advisory values may remain `none`; the Group 1/Group 2 split can therefore be a score-based proxy rather than an official advisory-band comparison.
- Source-relative deltas use evidence scores and may be `native-task-proxy` when source scores are unavailable on the pair eval dataset.

## Cohort Counts
- `exploratory_other`: 8
- `group_1_high_overlap_low_proxy`: 4
- `group_2_high_overlap_elevated_proxy`: 5
- `group_3_non_overlap_pathology_anchor`: 4
- High-overlap score split cut (`max_over_accumulation_score`): `0.3477`
