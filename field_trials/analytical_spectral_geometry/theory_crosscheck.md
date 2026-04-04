# Analytical Spectral Theory Cross-Check

**Date**: 2026-04-03
**Pairs analyzed**: 30 (12 poor merges, threshold -0.05)
**Errors**: 0

## Spearman Correlations (metric vs merge_delta)

Higher (more negative) Spearman = metric better predicts bad merges.

| Metric | Spearman |
|--------|----------|
| v1_max_score | 0.1859 |
| v2_max_score | 0.2617 |
| theory_max_risk | 0.3820 |
| theory_max_inflation_ratio | 0.3856 |
| theory_max_cross_bound | 0.3847 |

## Poor-Merge Enrichment (top tercile alert)

| Metric | Recall | Lift | Rate (alerted) | Rate (rest) |
|--------|--------|------|----------------|-------------|
| v1 | 0.2500 | -0.1500 | 0.3000 | 0.4500 |
| v2 | 0.1667 | -0.3000 | 0.2000 | 0.5000 |
| theory_risk | 0.0833 | -0.4500 | 0.1000 | 0.5500 |
| theory_inflation | 0.0833 | -0.4500 | 0.1000 | 0.5500 |

## Theory vs Heuristic Agreement (per-layer)

| Classification | Count |
|---------------|-------|
| agrees | 182 |
| theory_higher | 50 |
| theory_lower | 110 |

## Per-Pair Summary

| Pair | Delta | Poor | V1 max | V2 max | Theory risk | Inflation ratio | NormEq reduces |
|------|-------|------|--------|--------|-------------|-----------------|---------------|
| TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplai | -0.5680 | YES | 0.168 | 0.158 | 0.111 | 1.111 | 0.0% |
| TransferGraph__jaesun_distilbert-base-uncased-finetuned-cola | -0.4500 | YES | 0.081 | 0.029 | 0.019 | 1.019 | 0.0% |
| TransferGraph__jaesun_distilbert-base-uncased-finetuned-cola | -0.4120 | YES | 0.231 | 0.063 | 0.083 | 1.083 | 8.3% |
| TransferGraph__vaariis_distilbert-base-uncased-finetuned-emo | -0.2940 | YES | 0.227 | 0.061 | 0.062 | 1.062 | 8.3% |
| TransferGraph__vaariis_distilbert-base-uncased-finetuned-emo | -0.2940 | YES | 0.104 | 0.045 | 0.056 | 1.056 | 8.3% |
| TransferGraph__vaariis_distilbert-base-uncased-finetuned-emo | -0.2660 | YES | 0.083 | 0.024 | 0.029 | 1.029 | 0.0% |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emoti | -0.2640 | YES | 0.051 | 0.027 | 0.047 | 1.047 | 0.0% |
| NightPrince__peft-distilbert-sst2__x__myselfmankar__distilbe | -0.1500 | YES | 0.143 | 0.027 | 0.002 | 1.002 | 0.0% |
| NightPrince__peft-distilbert-sst2__x__wt-golf__distilbert-ba | -0.1240 | YES | 0.040 | 0.011 | 0.001 | 1.001 | 0.0% |
| TransferGraph__Aureliano_distilbert-base-uncased-if-finetune | -0.0960 | YES | 0.121 | 0.111 | 0.182 | 1.182 | 0.0% |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classifi | -0.0620 | YES | 0.116 | 0.050 | 0.659 | 1.659 | 16.7% |
| TransferGraph__cambridgeltl_guardian_news_distilbert-base-un | -0.0540 | YES | 0.210 | 0.308 | 0.421 | 1.421 | 0.0% |
| myselfmankar__distilbert-base-sst2-lora__x__wt-golf__distilb | -0.0300 | no | 0.120 | 0.154 | 0.357 | 1.357 | 0.0% |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment | -0.0220 | no | 0.380 | 0.238 | 0.551 | 1.551 | 8.3% |
| NightPrince__peft-distilbert-sst2__x__TransferGraph__Aurelia | -0.0200 | no | 0.075 | 0.014 | 0.001 | 1.001 | 0.0% |
| TransferGraph__neibla_distilbert-base-uncased-finetuned-emot | -0.0180 | no | 0.136 | 0.197 | 0.313 | 1.313 | 0.0% |
| NightPrince__peft-distilbert-sst2__x__TransferGraph__jaesun_ | -0.0160 | no | 0.045 | 0.012 | 0.001 | 1.001 | 0.0% |
| NightPrince__peft-distilbert-sst2__x__TransferGraph__JB173_d | -0.0120 | no | 0.057 | 0.019 | 0.001 | 1.001 | 0.0% |
| TransferGraph__cointegrated_roberta-base-formality-finetuned | -0.0100 | no | 0.170 | 0.291 | 1.000 | 2.228 | 0.0% |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emoti | -0.0100 | no | 0.195 | 0.366 | 0.609 | 1.609 | 0.0% |
| NightPrince__peft-distilbert-sst2__x__TransferGraph__vaariis | -0.0100 | no | 0.070 | 0.022 | 0.002 | 1.002 | 0.0% |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classifi | -0.0080 | no | 0.201 | 0.092 | 0.514 | 1.514 | 0.0% |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment | -0.0080 | no | 0.240 | 0.229 | 1.000 | 2.274 | 0.0% |
| dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment | -0.0080 | no | 0.231 | 0.086 | 0.475 | 1.475 | 0.0% |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emoti | -0.0080 | no | 0.122 | 0.167 | 0.823 | 1.823 | 0.0% |
| myselfmankar__distilbert-base-sst2-lora__x__rambodazimi__dis | -0.0060 | no | 0.348 | 0.200 | 0.461 | 1.461 | 0.0% |
| TransferGraph__JB173_distilbert-base-uncased-finetuned-emoti | -0.0060 | no | 0.163 | 0.284 | 0.429 | 1.429 | 0.0% |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classifi | 0.0040 | no | 0.184 | 0.140 | 1.000 | 2.850 | 16.7% |
| NightPrince__peft-distilbert-sst2__x__TransferGraph__neibla_ | 0.0040 | no | 0.057 | 0.014 | 0.001 | 1.001 | 0.0% |
| RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classifi | 0.0160 | no | 0.122 | 0.081 | 0.187 | 1.187 | 0.0% |
