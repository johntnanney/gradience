# Checkpoint Inventory T02 — Field Note

Generated: 2026-03-31T00:59:03.302813+00:00

## 1. Inventory

- Included `5` full fine-tuned checkpoints sharing base `distilbert-base-uncased`.
- Panel shape: same-task pair (SST-2 seeds), same-family non-identical-task pair (SST-2 × Yelp), plus cross-task controls (MRPC, QNLI).
- Chosen to directly exercise the same-family branch in checkpoint inventory triage.

## 2. Gradience Stance

- Dominant driver: `source_qa`
- Inventory type: `mixed_quality`
- Exploration posture: `narrow`
- Evaluate-first: none
- QA status counts: eligible=1, uncertain=1, flagged_weak=3
- Pair relationships: same_task=1, same_family=2, cross_task=7
- Action-plan summary: QA dominates this inventory; no credible same-task candidates remain.

## 3. Follow-through Results

| Category | Checkpoint(s) | Task | Score(s) | Note |
|---|---|---|---|---|
| same_task_near_miss_probe | sst2_s42, sst2_s123 | sst2 | sst2_s42=0.5067; sst2_s123=0.5067; base=0.5167 | No retained pair remained; probing the strongest same-task near-miss candidate. |
| same_family_pair_probe | sst2_s42, yelp_s42 | same_family_cross_dataset | sst2_s42_on_sst2=0.5067; base_on_sst2=0.5167; yelp_s42_on_yelp_polarity=0.8767; base_on_yelp_polarity=0.58 | Same-family non-identical-task branch probe (task-family equivalence sanity check). |
| optional_single_checkpoint | mrpc_s42 | mrpc | mrpc_s42=0.6767; base=0.4933 | Optional same-inventory checkpoint sanity check. |
| lower_priority_control_checkpoint | qnli_s42 | qnli | qnli_s42=0.5167; base=0.49 | Lower-priority control checkpoint; used to sanity-check weak/risky region. |

## 4. Product Judgment

- Did checkpoint workflow feel useful? yes
- Did evidence gate remain central? yes
- Did reports explain triage clearly? yes (inventory summary + action plan + run-bundle packet).
- Does this feel like a real broader use case? yes, with explicit same-family branch coverage and narrow-scope caveats.
