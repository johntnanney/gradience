# Preflight Summary — preflight_ev_090044

**Run:** run_20260329_140056
**Generated:** 2026-03-29 14:00 UTC

## Source QA

- eligible: 4

## Task-boundary partition

- Same-task pairs (advisory silent): 2
- Cross-task pairs (advisory active): 4
- Total pairs: 6

## Reduced candidate set

- dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment × wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k  (low risk, linear)
- myselfmankar__distilbert-base-sst2-lora × rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2  (medium risk, audit_aware)

**Inventory is mostly explained by task boundary. Candidate space reduced from 6 pairs to 2 (67% reduction).**

## Provenance

Sources with behavioral evidence: 4/4

*Behavioral scores are user-reported; Gradience does not independently
verify claimed evaluation results.*

## Inventory action plan

See `inventory_action_plan.md` for the full structured plan.

## Detailed artifacts

- `qa/` — source QA artifacts
- `pair_reports/` — pairwise merge reports
- `inventory/` — inventory summary
- `neighborhoods/` — neighborhood grouping
