# Ablation vs Gradient Gap Investigation (Informative Subset)

## Scope
- informative families: sst2, imdb
- adapters analyzed: 6
- budgets: [0.35, 0.5, 0.65]
- repeats: 3
- grad_samples=32, ablation_samples=48

## Gradient vs Ablation Outcome Concentration
| adapter_id | task_family | proxy_gradient_dvu | proxy_ablation_dvu | grad_minus_ablation |
| --- | --- | --- | --- | --- |
| myselfmankar/distilbert-base-sst2-lora | sst2 | 0.0573 | 0.0000 | 0.0573 |
| rambodazimi/distilbert-base-uncased-finetuned-LoRA-SST2 | sst2 | 0.0139 | 0.0069 | 0.0069 |
| NightPrince/peft-distilbert-sst2 | sst2 | 0.0000 | 0.0000 | 0.0000 |
| RAJESHCHAUHAN101/distilbert-base-uncased-lora-text-classification | imdb | 0.0017 | 0.0035 | -0.0017 |
| wt-golf/distilbert-base-uncased-lora-text-classification-imdb-1k | imdb | -0.0017 | 0.0000 | -0.0017 |
| dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment | imdb | 0.0000 | 0.0035 | -0.0035 |
- summary: mean grad-minus-ablation=0.0095, gradient_better=2, ablation_better=3, top2_abs_contribution_share=0.9024.

## Proxy Stability Under Resampling
- mean pairwise Spearman: gradient=0.903, ablation=0.400, grad-minus-ablation=0.503.
- mean pairwise top-k overlap: gradient=0.611, ablation=0.593, grad-minus-ablation=0.019.

## Top-k Overlap vs Within-Top-k Distribution
| task_family | budget | n | mean_topk_overlap | mean_abs_dev_shared_topk | mean_abs_dev_union_topk | mean_abs_dev_all_layers |
| --- | --- | --- | --- | --- | --- | --- |
| sst2 | 0.35 | 9 | 0.148 | 0.000 | 10.204 | 5.611 |
| sst2 | 0.50 | 9 | 0.111 | 0.000 | 9.596 | 7.352 |
| sst2 | 0.65 | 9 | 0.222 | 0.000 | 6.307 | 7.556 |
| imdb | 0.35 | 9 | 0.296 | 0.000 | 2.911 | 1.204 |
| imdb | 0.50 | 9 | 0.185 | 0.000 | 3.102 | 2.222 |
| imdb | 0.65 | 9 | 0.185 | 0.000 | 2.519 | 2.296 |

## Caution
- This is a bounded CPU follow-up; treat as directional evidence under the current small informative subset.
