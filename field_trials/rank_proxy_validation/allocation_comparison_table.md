# Allocation Comparison Table

- Rows: 360

| adapter_id | dataset | budget | policy | proxy | spearman | topk_overlap | mean_abs_rank_dev | attn_share_abs_diff | policy_minus_proxy_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAJESHCHAUHAN101/distilbert-base-uncased-lora-text-classification | imdb | 0.65 | energy_90 | proxy_gradient | -1.000 | 0.000 | 3.000 | 0.000 | 0.016 |
| RAJESHCHAUHAN101/distilbert-base-uncased-lora-text-classification | imdb | 0.65 | erank | proxy_gradient | -1.000 | 0.000 | 3.000 | 0.000 | 0.016 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.65 | energy_90 | proxy_gradient | -1.000 | 0.000 | 3.000 | 0.000 | 0.005 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.65 | knee | proxy_gradient | -1.000 | 0.000 | 3.000 | 0.000 | 0.005 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.65 | oht | proxy_gradient | -1.000 | 0.000 | 3.000 | 0.000 | 0.005 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.65 | stable_rank_ceil | proxy_gradient | -1.000 | 0.000 | 3.000 | 0.000 | 0.005 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.35 | energy_90 | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.35 | knee | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.35 | erank | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.35 | oht | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.35 | stable_rank_ceil | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.50 | energy_90 | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.50 | knee | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.50 | oht | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.50 | stable_rank_ceil | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.65 | energy_90 | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.65 | knee | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.65 | oht | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | 0.65 | stable_rank_ceil | proxy_ablation | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | 0.50 | knee | proxy_gradient | -0.935 | 0.000 | 6.000 | 0.000 | 0.000 |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | 0.50 | stable_rank_ceil | proxy_gradient | -0.935 | 0.000 | 6.000 | 0.000 | 0.000 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.65 | oht | proxy_ablation | 0.855 | 1.000 | 1.000 | 0.000 | 0.000 |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | 0.50 | energy_90 | proxy_gradient | -0.829 | 0.000 | 5.833 | 0.000 | 0.000 |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | 0.50 | erank | proxy_gradient | -0.738 | 0.000 | 5.833 | 0.000 | 0.000 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.50 | energy_90 | proxy_gradient | -0.738 | 0.000 | 5.833 | 0.000 | 0.000 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.50 | knee | proxy_ablation | 0.725 | 1.000 | 1.167 | 0.000 | 0.000 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.50 | erank | proxy_ablation | 0.725 | 1.000 | 1.167 | 0.000 | 0.000 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.50 | oht | proxy_ablation | 0.725 | 1.000 | 1.167 | 0.000 | 0.000 |
| myselfmankar/distilbert-base-sst2-lora | sst2 | 0.65 | stable_rank_ceil | proxy_ablation | 0.707 | 1.000 | 2.500 | 0.000 | 0.005 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.65 | energy_90 | proxy_ablation | 0.707 | 1.000 | 1.167 | 0.000 | 0.000 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.65 | knee | proxy_ablation | 0.707 | 1.000 | 1.167 | 0.000 | 0.000 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.65 | stable_rank_ceil | proxy_ablation | 0.707 | 1.000 | 1.167 | 0.000 | 0.000 |
| myselfmankar/distilbert-base-sst2-lora | sst2 | 0.65 | stable_rank_ceil | proxy_gradient | -0.682 | 0.000 | 11.333 | 0.000 | -0.052 |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | 0.65 | knee | proxy_gradient | -0.682 | 0.000 | 5.667 | 0.000 | 0.000 |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | 0.65 | stable_rank_ceil | proxy_gradient | -0.682 | 0.000 | 5.667 | 0.000 | 0.000 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.65 | energy_90 | proxy_gradient | -0.682 | 0.333 | 5.667 | 0.000 | 0.000 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.65 | erank | proxy_gradient | -0.682 | 0.333 | 5.667 | 0.000 | 0.000 |
| myselfmankar/distilbert-base-sst2-lora | sst2 | 0.35 | oht | proxy_gradient | 0.664 | 0.667 | 2.500 | 0.000 | 0.000 |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | 0.65 | energy_90 | proxy_ablation | 0.616 | 1.000 | 1.167 | 0.000 | -0.010 |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | 0.65 | erank | proxy_ablation | 0.616 | 1.000 | 1.167 | 0.000 | -0.010 |
