# Strict Naive Rerun Results

| pair_id | group | status | eval_dataset | merged_score | best_source | delta_vs_best | strict_naive_ok | strict_naive_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| myselfmankar__distilbert-base-sst2-lora__x__rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2 | high_tail | ok | sst2 | 0.894 | 0.902 | -0.008 | yes | ok |
| TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion__x__TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion | high_tail | ok | tweet_eval/emotion | 0.732 | 0.786 | -0.054 | yes | ok |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony__x__TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | high_tail | ok | tweet_eval/irony | 0.610 | 0.620 | -0.010 | yes | ok |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__myselfmankar__distilbert-base-sst2-lora | high_tail | ok | sst2 | 0.862 | 0.884 | -0.022 | yes | ok |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classification__x__dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment | high_tail | ok | imdb | 0.868 | 0.876 | -0.008 | yes | ok |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k | high_tail | ok | imdb | 0.868 | 0.876 | -0.008 | yes | ok |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classification__x__wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k | high_tail | ok | imdb | 0.864 | 0.860 | 0.004 | yes | ok |
| TransferGraph__cointegrated_roberta-base-formality-finetuned-lora-ag_news__x__TransferGraph__roberta-base-finetuned-lora-ag_news | lower_tail_matched | ok | ag_news | 0.912 | 0.922 | -0.010 | yes | ok |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony__x__TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | lower_tail_matched | ok | tweet_eval/irony | 0.608 | 0.614 | -0.006 | yes | ok |
| NightPrince__peft-distilbert-sst2__x__myselfmankar__distilbert-base-sst2-lora | lower_tail_matched | ok | sst2 | 0.734 | 0.884 | -0.150 | yes | ok |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news__x__TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news | lower_tail_matched | ok | ag_news | 0.892 | 0.900 | -0.008 | yes | ok |
| TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion__x__TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion | lower_tail_matched | ok | tweet_eval/emotion | 0.188 | 0.756 | -0.568 | yes | ok |
| TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony__x__TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | lower_tail_matched | ok | tweet_eval/irony | 0.602 | 0.620 | -0.018 | yes | ok |
