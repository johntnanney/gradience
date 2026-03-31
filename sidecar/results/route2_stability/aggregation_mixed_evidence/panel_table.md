# Mixed-Evidence Triage Perturbation Panel

## Panel summary

- Panel size: 8
- Evidence regimes: clear (2), mixed (4), blocked (2)
- Roles: anchor (1), blocked (2), review (2), optional (3)
- Design intent: intentionally overweight soft-middle triage cases

## Cases

| case_id | artifact_class | relation | evidence_regime | role | pair_risk | compatibility |
|---|---|---|---|---|---|---:|
| anchor_clear_retained_irony | lora_adapter_pair | same_task | clear | anchor | medium | 0.3682 |
| anchor_blocked_same_task_checkpoint | full_checkpoint_pair | same_task | blocked | blocked | medium | 0.8922 |
| anchor_blocked_cross_task_checkpoint | full_checkpoint_pair | cross_task | blocked | blocked | high | 0.6259 |
| review_mixed_cross_task_sst2_mrpc | full_checkpoint_pair | cross_task | mixed | review | medium | 0.7980 |
| review_mixed_cross_task_yelp_mrpc | full_checkpoint_pair | cross_task | mixed | review | high | 0.5843 |
| optional_same_family_clear_sst2_imdb | lora_adapter_pair | same_family | clear | optional | medium | 0.3140 |
| optional_same_family_mixed_sst2_yelp_a | full_checkpoint_pair | same_family | mixed | optional | high | 0.6523 |
| optional_same_family_mixed_sst2_yelp_b | full_checkpoint_pair | same_family | mixed | optional | high | 0.6410 |

## Why this panel is softer than the baseline aggregation panel

1. Mixed-evidence and optional/review cases are the majority (5/8).
2. Same-family optional cases are explicitly overrepresented (3/8).
3. Only one clear retained anchor is kept, plus blocked anchors for calibration.
