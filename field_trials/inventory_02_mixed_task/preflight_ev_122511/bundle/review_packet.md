# Review Packet — preflight_ev_122511

**Run:** run_20260328_172528
**Generated:** 2026-03-28 17:25 UTC
**Adapters:** 5
**Pairs:** 10
**Retained candidates:** 1

## Inventory Policy Summary

*Not available for this run.*

## Source QA / Trust Snapshot

- eligible: 4
- uncertain: 1

**Evidence:** 5/5 sources with behavioral evidence

*Behavioral scores are user-reported; Gradience does not independently
verify claimed evaluation results.*

## Action Plan

**Starting pairs:** 10
**Retained candidates:** 1
**Reduction:** 90%
**Cross-task excluded:** 9

**Evaluate first:**
- TransferGraph__roberta-base-finetuned-lora-ag_news × TransferGraph__cointegrated_roberta-base-formality-finetuned-lora-ag_news  (medium risk, norm_equalized)

**Same-task safe zone:**
- TransferGraph__roberta-base-finetuned-lora-ag_news × TransferGraph__cointegrated_roberta-base-formality-finetuned-lora-ag_news  (medium risk, norm_equalized)

**Cross-task caution zone:**
- AG_NEWS × MNLI region
- AG_NEWS × TWEET_EVAL/HATE region
- AG_NEWS × TWEET_EVAL/IRONY region
- MNLI × TWEET_EVAL/HATE region
- MNLI × TWEET_EVAL/IRONY region
- TWEET_EVAL/HATE × TWEET_EVAL/IRONY region

**Summary:** Inventory is mostly explained by task boundary. Candidate space reduced from 10 pairs to 1 (90% reduction).

## Artifacts

- `preflight_summary.json` — machine-readable preflight summary
- `preflight_summary.md` — human-readable preflight summary
- `inventory_action_plan.md` — structured action plan
- `run_manifest.json` — run metadata
- `qa/` — source QA artifacts
- `pair_reports/` — pairwise merge reports
- `inventory/` — inventory summary
