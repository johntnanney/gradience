# Checkpoint Inventory T01 — Field Note

Generated: 2026-03-31T00:26:32.848191+00:00

## 1. Inventory

- Included `4` full fine-tuned checkpoints sharing base `distilbert-base-uncased`.
- Panel shape: 2 same-task (SST-2), 2 cross-task controls (MRPC, QNLI).
- Chosen for CPU-feasible, cached-data execution with known Ring 2 structural variation.

## 2. Gradience Stance

- Dominant driver: `source_qa`
- Inventory type: `mixed_quality`
- Exploration posture: `narrow`
- Evaluate-first: none
- QA status counts: eligible=1, uncertain=0, flagged_weak=3
- Action-plan summary: QA dominates this inventory; no credible same-task candidates remain.

## 3. Follow-through Results

| Category | Checkpoint(s) | Task | Score(s) | Note |
|---|---|---|---|---|
| same_task_near_miss_probe | sst2_s42, sst2_s123 | sst2 | sst2_s42=0.5067; sst2_s123=0.5067; base=0.5167 | No retained pair remained; probing the strongest same-task near-miss candidate. |
| optional_single_checkpoint | mrpc_s42 | mrpc | mrpc_s42=0.6767; base=0.4933 | Optional same-inventory checkpoint sanity check. |
| lower_priority_control_checkpoint | qnli_s42 | qnli | qnli_s42=0.5167; base=0.4900 | Lower-priority control checkpoint; used to sanity-check weak/risky region. |

## 4. Product Judgment

- Did checkpoint workflow feel useful? yes
- Did evidence gate remain central? yes
- Did reports explain triage clearly? yes (preflight summary + action plan + review packet).
- Does this feel like a real broader use case? yes, but still narrow and CPU-bounded.
