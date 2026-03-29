# Review Packet — preflight_ev_123354

**Run:** run_20260328_173436
**Generated:** 2026-03-28 17:34 UTC
**Adapters:** 8
**Pairs:** 28
**Retained candidates:** 2

## Inventory Policy Summary

*Not available for this run.*

## Source QA / Trust Snapshot

- eligible: 6
- flagged_weak: 2

**Evidence:** 8/8 sources with behavioral evidence

*Behavioral scores are user-reported; Gradience does not independently
verify claimed evaluation results.*

**Excluded sources:**
- TransferGraph__Aureliano_distilbert-base-uncased-if-finetuned-lora-tweet_eval_hate: weak source — low confidence
- jmeneu__distilbert-base-uncased-lora-text-classification: weak source — low confidence

## Action Plan

**Starting pairs:** 28
**Retained candidates:** 2
**Reduction:** 93%
**Cross-task excluded:** 13

**Evaluate first:**
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- myselfmankar__distilbert-base-sst2-lora × NightPrince__peft-distilbert-sst2  (medium risk, norm_equalized)

**Same-task safe zone:**
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- myselfmankar__distilbert-base-sst2-lora × NightPrince__peft-distilbert-sst2  (medium risk, norm_equalized)

**Cross-task caution zone:**
- AG_NEWS × SST-2 region
- AG_NEWS × TWEET_EVAL/EMOTION region
- AG_NEWS × TWEET_EVAL/HATE region
- SST-2 × TWEET_EVAL/EMOTION region
- SST-2 × TWEET_EVAL/HATE region
- TWEET_EVAL/EMOTION × TWEET_EVAL/HATE region

**Summary:** QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 2.

## Artifacts

- `preflight_summary.json` — machine-readable preflight summary
- `preflight_summary.md` — human-readable preflight summary
- `inventory_action_plan.md` — structured action plan
- `run_manifest.json` — run metadata
- `qa/` — source QA artifacts
- `pair_reports/` — pairwise merge reports
- `inventory/` — inventory summary
