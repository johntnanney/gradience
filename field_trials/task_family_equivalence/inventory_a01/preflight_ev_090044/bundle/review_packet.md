# Review Packet — preflight_ev_090044

**Run:** run_20260329_140056
**Generated:** 2026-03-29 14:00 UTC
**Adapters:** 4
**Pairs:** 6
**Retained candidates:** 2

## Inventory Policy Summary

*Not available for this run.*

## Source QA / Trust Snapshot

- eligible: 4

**Evidence:** 4/4 sources with behavioral evidence

*Behavioral scores are user-reported; Gradience does not independently
verify claimed evaluation results.*

## Action Plan

**Starting pairs:** 6
**Retained candidates:** 2
**Reduction:** 67%
**Cross-task excluded:** 4

**Evaluate first:**
- dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment × wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k  (low risk, linear)
- myselfmankar__distilbert-base-sst2-lora × rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2  (medium risk, audit_aware)

**Same-task safe zone:**
- dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment × wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k  (low risk, linear)
- myselfmankar__distilbert-base-sst2-lora × rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2  (medium risk, audit_aware)

**Cross-task caution zone:**
- IMDB × SST-2 region

**Summary:** Inventory is mostly explained by task boundary. Candidate space reduced from 6 pairs to 2 (67% reduction).

## Artifacts

- `preflight_summary.json` — machine-readable preflight summary
- `preflight_summary.md` — human-readable preflight summary
- `inventory_action_plan.md` — structured action plan
- `run_manifest.json` — run metadata
- `qa/` — source QA artifacts
- `pair_reports/` — pairwise merge reports
- `inventory/` — inventory summary
