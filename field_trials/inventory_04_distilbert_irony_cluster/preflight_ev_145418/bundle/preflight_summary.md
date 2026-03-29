# Preflight Summary — preflight_ev_145418

**Run:** run_20260328_195511
**Generated:** 2026-03-28 19:55 UTC

## Source QA

- eligible: 7
- flagged_weak: 1

## Task-boundary partition

- Same-task pairs (advisory silent): 8
- Cross-task pairs (advisory active): 20
- Total pairs: 28

## Reduced candidate set

- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news  (low risk, linear)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (medium risk, norm_equalized)
- TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion × TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion  (medium risk, norm_equalized)

## Near-miss candidates

Structurally plausible, evidence-constrained. Optional if evaluation budget allows.

- TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony  (low risk, linear — TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony is evidence-constrained)
- TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony × TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony  (low risk, linear — TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony is evidence-constrained)
- TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony × TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony  (low risk, linear — TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony is evidence-constrained)

**QA and task boundary dominate this inventory. Candidate space reduced from 28 pairs to 5.**

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
