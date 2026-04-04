# Activation Threshold Audit

## Scope
- Deduplicated adapters audited: **36**
- Audited pairs: **189**
- Layer entries: **2308**
- Audit errors: **0**

## Pair-Level Advisory Activation
- `elevated`: 1
- `none`: 188

## Layer-Band Activation
- `high`: 7
- `low`: 2265
- `watch`: 36

## Score Quantiles
- Pair max score quantiles: {'q50': 0.0687, 'q75': 0.1204, 'q90': 0.1838, 'q95': 0.2401, 'q99': 0.38}
- Layer score quantiles: {'q50': 0.019095171689987183, 'q75': 0.0486310601234436, 'q90': 0.10769253815571848, 'q95': 0.16757805347442625, 'q99': 0.45749999999999996}

## Cutpoint Positioning
- Pair max >= 0.35: 4
- Pair max >= 0.60: 1
- Pair max >= 0.75: 0
- Layers score >= 0.35: 43
- Layers score >= 0.60: 7
- Layers band=watch: 36
- Layers band=high: 7

## Top OA Pairs
| pair_id | mean_overlap | max_oa | oa_advisory | watch_layers | high_layers | verdict | both_loadable | num_labels_equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| muneeb-ai__distilbert-base-uncased-lora-imdb-sentiment__x__muneeb-ai__distilbert-base-uncased-lora-imdb-sentiment | 1.000 | 0.642 | elevated | 33 | 7 | imbalanced | no | no |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__myselfmankar__distilbert-base-sst2-lora | 0.308 | 0.380 | none | 1 | 0 | redundant | yes | yes |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__myselfmankar__distilbert-base-sst2-lora | 0.308 | 0.380 | none | 1 | 0 | redundant | no | no |
| TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion__x__myselfmankar__distilbert-base-sst2-lora | 0.459 | 0.355 | none | 1 | 0 | redundant | yes | no |
| myselfmankar__distilbert-base-sst2-lora__x__rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2 | 0.468 | 0.348 | none | 0 | 0 | imbalanced | yes | yes |
| myselfmankar__distilbert-base-sst2-lora__x__rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2 | 0.468 | 0.348 | none | 0 | 0 | imbalanced | no | no |
| TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion__x__dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment | 0.305 | 0.326 | none | 0 | 0 | imbalanced | yes | no |
| TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion__x__myselfmankar__distilbert-base-sst2-lora | 0.269 | 0.251 | none | 0 | 0 | redundant | yes | no |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k | 0.254 | 0.240 | none | 0 | 0 | safe | yes | yes |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k | 0.254 | 0.240 | none | 0 | 0 | safe | no | no |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2 | 0.187 | 0.231 | none | 0 | 0 | imbalanced | yes | yes |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment__x__rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2 | 0.187 | 0.231 | none | 0 | 0 | imbalanced | no | no |
| TransferGraph__jaesun_distilbert-base-uncased-finetuned-cola-finetuned-lora-tweet_eval_hate__x__myselfmankar__distilbert-base-sst2-lora | 0.148 | 0.231 | none | 0 | 0 | safe | yes | yes |
| TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony__x__myselfmankar__distilbert-base-sst2-lora | 0.141 | 0.227 | none | 0 | 0 | safe | yes | yes |
| TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion__x__dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment | 0.246 | 0.215 | none | 0 | 0 | imbalanced | yes | no |
| TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion__x__TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion | 0.416 | 0.210 | none | 0 | 0 | redundant | yes | yes |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classification__x__dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment | 0.256 | 0.201 | none | 0 | 0 | safe | yes | yes |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony__x__TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony | 0.390 | 0.195 | none | 0 | 0 | redundant | yes | yes |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classification__x__wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k | 0.208 | 0.184 | none | 0 | 0 | safe | yes | yes |
| TransferGraph__cointegrated_roberta-base-formality-finetuned-lora-ag_news__x__TransferGraph__roberta-base-finetuned-lora-ag_news | 0.420 | 0.170 | none | 0 | 0 | redundant | yes | yes |
