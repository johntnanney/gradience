# Preflight Summary — preflight_ev_123354

**Run:** run_20260328_173436
**Generated:** 2026-03-28 17:34 UTC

## Source QA

- eligible: 6
- flagged_weak: 2

## Task-boundary partition

- Same-task pairs (advisory silent): 3
- Cross-task pairs (advisory active): 25
- Total pairs: 28

## Reduced candidate set

- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- myselfmankar__distilbert-base-sst2-lora × NightPrince__peft-distilbert-sst2  (medium risk, norm_equalized)

**QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 2.**

## Provenance

Sources with behavioral evidence: 8/8

*Behavioral scores are user-reported; Gradience does not independently
verify claimed evaluation results.*

## Inventory action plan

See `inventory_action_plan.md` for the full structured plan.

## Detailed artifacts

- `qa/` — source QA artifacts
- `pair_reports/` — pairwise merge reports
- `inventory/` — inventory summary
- `neighborhoods/` — neighborhood grouping
