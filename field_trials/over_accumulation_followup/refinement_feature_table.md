# OA Refinement Feature Table

Full numeric feature inventory is in `refinement_feature_table.json`.

## Pair-Level Snapshot
| pair_id | cohort | eval_dataset | delta_vs_best | mean_overlap | current_max_oa | hotspot_top3_oa | top3_interaction_ace | triple_high_frac | source_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classification__x__dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment | high_tail | imdb | -0.0060 | 0.256 | 0.201 | 0.175 | 0.111 | 0.000 | 0.016 |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony__x__TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | high_tail | tweet_eval/irony | -0.0100 | 0.390 | 0.195 | 0.186 | 0.000 | 0.000 | 0.006 |
| TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion__x__TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion | high_tail | tweet_eval/emotion | -0.0540 | 0.416 | 0.210 | 0.198 | 0.000 | 0.000 | 0.078 |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__myselfmankar__distilbert-base-sst2-lora | high_tail | sst2 | -0.0220 | 0.308 | 0.380 | 0.339 | 0.259 | 0.083 | 0.038 |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k | high_tail | imdb | -0.0080 | 0.254 | 0.240 | 0.225 | 0.155 | 0.000 | 0.020 |
| myselfmankar__distilbert-base-sst2-lora__x__rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2 | high_tail | sst2 | -0.0100 | 0.468 | 0.348 | 0.347 | 0.265 | 0.167 | 0.018 |
| NightPrince__peft-distilbert-sst2__x__myselfmankar__distilbert-base-sst2-lora | lower_tail_matched | sst2 | -0.1500 | 0.254 | 0.143 | 0.117 | 0.039 | 0.000 | 0.190 |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classification__x__wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k | lower_tail_matched | imdb | 0.0040 | 0.208 | 0.184 | 0.155 | 0.092 | 0.000 | 0.004 |
| TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion__x__TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion | lower_tail_matched | tweet_eval/emotion | -0.5680 | 0.241 | 0.168 | 0.154 | 0.000 | 0.000 | 0.620 |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news__x__TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news | lower_tail_matched | ag_news | -0.0080 | 0.242 | 0.122 | 0.118 | 0.000 | 0.000 | 0.022 |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony__x__TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | lower_tail_matched | tweet_eval/irony | -0.0060 | 0.350 | 0.163 | 0.160 | 0.000 | 0.000 | 0.014 |
| TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony__x__TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | lower_tail_matched | tweet_eval/irony | -0.0180 | 0.215 | 0.136 | 0.132 | 0.000 | 0.000 | 0.020 |
