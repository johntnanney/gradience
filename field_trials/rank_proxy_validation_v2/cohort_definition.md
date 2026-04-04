# Rank Proxy Validation v2 Cohort Definition

## Scope
- Primary informative families: imdb, sst2
- Secondary context families: ag_news, tweet_eval
- Adapter x dataset rows: 12

## Cohort Table
| adapter_id | dataset | task_family | inclusion_status | compressibility | full_acc | quality_gap | quality_band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAJESHCHAUHAN101/distilbert-base-uncased-lora-text-classification | imdb | imdb | primary_informative | compressible | 0.8594 | 0.0052 | near_top |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | imdb | primary_informative | compressible | 0.8646 | 0.0000 | near_top |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | imdb | primary_informative | compressible | 0.8125 | 0.0521 | large_gap |
| NightPrince/peft-distilbert-sst2 | sst2 | sst2 | primary_informative | compressible | 0.7083 | 0.1979 | large_gap |
| myselfmankar/distilbert-base-sst2-lora | sst2 | sst2 | primary_informative | compressible | 0.8698 | 0.0365 | mid_gap |
| rambodazimi/distilbert-base-uncased-finetuned-LoRA-SST2 | sst2 | sst2 | primary_informative | compressible | 0.9062 | 0.0000 | near_top |
| TransferGraph/JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news | ag_news | ag_news | secondary_context | saturated | 0.8906 | 0.0000 | single_source_dataset |
| TransferGraph/cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion | tweet_eval/emotion | tweet_eval | secondary_context | saturated | 0.7240 | 0.0677 | large_gap |
| TransferGraph/distilbert-base-uncased-finetuned-lora-tweet_eval_emotion | tweet_eval/emotion | tweet_eval | secondary_context | saturated | 0.7917 | 0.0000 | near_top |
| TransferGraph/JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | tweet_eval/irony | tweet_eval | secondary_context | saturated | 0.6042 | 0.0104 | mid_gap |
| TransferGraph/neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | tweet_eval/irony | tweet_eval | secondary_context | saturated | 0.5885 | 0.0260 | mid_gap |
| TransferGraph/vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | tweet_eval/irony | tweet_eval | secondary_context | saturated | 0.6146 | 0.0000 | near_top |
