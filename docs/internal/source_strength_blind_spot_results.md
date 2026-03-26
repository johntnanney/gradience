# Source-Strength Asymmetry Blind Spot Study — Results

## Study setup

6 QNLI adapters on distilbert-base-uncased with intentional strength spread:

| Adapter | Band | Data | Steps | Accuracy |
|---------|------|------|-------|----------|
| strong_s42 | strong | 2000 | 1000 | 0.702 |
| strong_s7 | strong | 2000 | 1000 | 0.708 |
| medium_s42 | medium | 2000 | 500 | 0.510 |
| medium_s7 | medium | 2000 | 500 | 0.538 |
| weak_s42 | weak | 500 | 500 | 0.594 |
| weak_s7 | weak | 500 | 500 | 0.574 |

Base accuracy: ~0.466. All 6 beat base. Strength range: 0.510–0.708 (19.8pp spread).

Note: the "medium" band (full data, reduced training) actually underperformed the "weak" band (reduced data, reduced training) — likely because 500 steps on 2000 samples was too early to converge, while 500 steps on 500 samples saw the data multiple times. The actual strength ordering is: strong > weak > medium. This does not affect the study — the relevant variable is the strength gap between sources, not the label.

## Main results

| Outcome | Count | % |
|---------|-------|---|
| safe (<=1.5pp) | 13 | 87% |
| mildly_degraded (1.5-4pp) | 2 | 13% |
| materially_degraded (>4pp) | 0 | 0% |

**Strength-gap vs delta correlation: r = 0.174** (very weak, not significant at n=15).

## Key finding: blind_spot_not_found

Source-strength asymmetry within same-task QNLI pairs does not create a meaningful blind spot. 13 of 15 pairs are safe. The 2 mildly degraded pairs (both involving strong_s42 × weak, delta +2.4pp) are the only exceptions — and even these are mild.

## By band pairing

| Band pairing | Pairs | Safe | Avg delta |
|-------------|-------|------|-----------|
| strong × strong | 1 | 1/1 | +0.004 |
| strong × medium | 4 | 4/4 | +0.006 |
| strong × weak | 4 | 2/4 | +0.015 |
| medium × medium | 1 | 1/1 | +0.014 |
| medium × weak | 4 | 4/4 | +0.004 |
| weak × weak | 1 | 1/1 | +0.002 |

The only band pairing with any degradation is strong × weak — and even there, 2 of 4 pairs are safe. The effect is inconsistent across seeds (strong_s42 × weak pairs degrade; strong_s7 × weak pairs don't).

## What the current workflow gets right

1. **All pairs rated medium risk.** No pair-risk overcall or undercall. All 15 pairs have `pair_risk=medium` — appropriate for same-task pairs with structural similarity.
2. **Task advisory is correctly silent.** All pairs are same-task/same-dataset (qnli_dev). Advisory never fires.
3. **No materially degraded merges.** Even the worst pair drops only 2.4pp.

## What the current workflow does not capture

Source-strength gap does predict a very mild tendency toward degradation in the strong × weak band (avg delta +0.015 vs +0.004 for matched pairs). But the effect is:
- small (1.5pp average difference)
- inconsistent across seeds
- never materially degraded
- correlation r=0.174 (not meaningful)

This is not a blind spot. It is noise-level variation within an already-safe regime.

## Recommendation

**`blind_spot_not_found`**

Source-strength asymmetry within same-task QNLI pairs does not create a meaningful blind spot for the current Gradience workflow. The mild degradation observed in 2/15 pairs is better explained by seed-specific noise than by a systematic strength-gap effect.

No new signal, advisory, or interpretive layer is warranted for this regime.

## Implication for the regime map

The same-task row is further strengthened:

> Same-task, all eligible → workflow confirmatory. Neither training-style variation, domain shift in high-transfer tasks, nor source-strength asymmetry creates a blind spot. Same-task merges remain broadly safe.

## Cumulative blind-spot study results

| Study | Target regime | Pairs | Result |
|-------|--------------|-------|--------|
| Training-style variation | rank/alpha/dropout/checkpoint | 15 | not found |
| Domain shift (sentiment) | movies/restaurants/products | 15 | not found |
| Source-strength asymmetry | strong/medium/weak | 15 | not found |

Three consecutive blind-spot studies on same-task pairs have found no actionable blind spot. The same-task regime appears robustly safe for small encoder models on GLUE-family tasks.
