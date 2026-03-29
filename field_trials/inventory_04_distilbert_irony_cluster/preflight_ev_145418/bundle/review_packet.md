# Review Packet — preflight_ev_145418

**Run:** run_20260328_195511
**Generated:** 2026-03-28 19:55 UTC
**Adapters:** 8
**Pairs:** 28
**Retained candidates:** 5

## Inventory Policy Summary

*Not available for this run.*

## Source QA / Trust Snapshot

- eligible: 7
- flagged_weak: 1

**Evidence:** 8/8 sources with behavioral evidence

*Behavioral scores are user-reported; Gradience does not independently
verify claimed evaluation results.*

**Excluded sources:**
- TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony: weak source — low confidence

## Action Plan

**Starting pairs:** 28
**Retained candidates:** 5
**Reduction:** 82%
**Cross-task excluded:** 16

**Evaluate first:**
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion  (medium risk, norm_equalized)

**Same-task safe zone:**
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion  (medium risk, norm_equalized)
- TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (low risk, linear)

**Cross-task caution zone:**
- AG_NEWS × TWEET_EVAL/EMOTION region
- AG_NEWS × TWEET_EVAL/IRONY region
- TWEET_EVAL/EMOTION × TWEET_EVAL/IRONY region

**Summary:** QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 5.

## Artifacts

- `preflight_summary.json` — machine-readable preflight summary
- `preflight_summary.md` — human-readable preflight summary
- `inventory_action_plan.md` — structured action plan
- `run_manifest.json` — run metadata
- `qa/` — source QA artifacts
- `pair_reports/` — pairwise merge reports
- `inventory/` — inventory summary
