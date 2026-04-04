# OA Refinement Matched Case Studies

## Selection
- {'high_overlap_region': 'mean_overlap >= 0.20 and conflict_fraction <= 0.10', 'disappointing_threshold': 'merge_delta_vs_best_source <= -0.05', 'benign_threshold': 'merge_delta_vs_best_source >= -0.02'}
- Disappointing candidates: 3
- Benign candidates: 3

## Comparison 1
- Disappointing: `TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion__x__TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion` | delta=-0.5680 | overlap=0.241 | max_oa=0.168
- Benign: `TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony__x__TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony` | delta=-0.0060 | overlap=0.350 | max_oa=0.163
- Match context: task_relation(same_task vs same_task), dataset(tweet_eval/emotion vs tweet_eval/irony), model_family(bert vs distilbert)
- Top distinguishing OA features (disappointing minus benign):
| feature | disappointing | benign | delta |
| --- | --- | --- | --- |
| concentration_oa_mass_top5_fraction | 0.4261 | 0.5930 | -0.1669 |
| concentration_oa_mass_top3_fraction | 0.2763 | 0.3810 | -0.1047 |
| concentration_oa_mass_top1_fraction | 0.1005 | 0.1296 | -0.0290 |
| raw_max_alignment | 0.5586 | 0.5441 | 0.0145 |
| interaction_max_align_x_exposure | 0.5586 | 0.5441 | 0.0145 |

## Comparison 2
- Disappointing: `NightPrince__peft-distilbert-sst2__x__myselfmankar__distilbert-base-sst2-lora` | delta=-0.1500 | overlap=0.254 | max_oa=0.143
- Benign: `RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classification__x__dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment` | delta=-0.0060 | overlap=0.256 | max_oa=0.201
- Match context: task_relation(same_task vs same_task), dataset(sst2 vs imdb), model_family(distilbert vs distilbert)
- Top distinguishing OA features (disappointing minus benign):
| feature | disappointing | benign | delta |
| --- | --- | --- | --- |
| raw_max_coefficient_exposure | 0.3436 | 1.0000 | -0.6564 |
| concentration_oa_mass_top5_fraction | 0.5510 | 0.9225 | -0.3715 |
| interaction_max_align_x_exposure | 0.0919 | 0.3979 | -0.3060 |
| concentration_oa_mass_top3_fraction | 0.3690 | 0.6596 | -0.2906 |
| concentration_ace_mass_top3_fraction | 0.4109 | 0.6342 | -0.2233 |

## Comparison 3
- Disappointing: `TransferGraph__cambridgeltl_guardian_news_distilbert-base-uncased-finetuned-lora-tweet_eval_emotion__x__TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion` | delta=-0.0540 | overlap=0.416 | max_oa=0.210
- Benign: `RAJESHCHAUHAN101__distilbert-base-uncased-lora-text-classification__x__wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k` | delta=0.0040 | overlap=0.208 | max_oa=0.184
- Match context: task_relation(same_task vs same_task), dataset(tweet_eval/emotion vs imdb), model_family(distilbert vs distilbert)
- Top distinguishing OA features (disappointing minus benign):
| feature | disappointing | benign | delta |
| --- | --- | --- | --- |
| concentration_ace_mass_top3_fraction | 0.0000 | 0.7004 | -0.7004 |
| concentration_oa_mass_top5_fraction | 0.5976 | 0.9504 | -0.3527 |
| concentration_oa_mass_top3_fraction | 0.3957 | 0.7284 | -0.3328 |
| raw_max_alignment | 0.6994 | 0.3899 | 0.3095 |
| interaction_max_align_x_exposure | 0.6994 | 0.3899 | 0.3095 |

