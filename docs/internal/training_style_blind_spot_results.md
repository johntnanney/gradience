# Training-Style Blind Spot Study — Results

## Study setup

6 QNLI adapters on distilbert-base-uncased, all verified above base (0.652–0.742 vs 0.466 base). Same task, same eval_dataset, varied training style:

| Adapter | Rank | Alpha | Dropout | Steps | Accuracy |
|---------|------|-------|---------|-------|----------|
| A | r8 | 8 | 0.0 | 1000 | 0.652 |
| B | r8 | 32 | 0.0 | 1000 | 0.700 |
| C | r16 | 16 | 0.0 | 1000 | 0.702 |
| D | r16 | 64 | 0.0 | 1000 | 0.742 |
| E | r16 | 64 | 0.1 | 1000 | 0.728 |
| F | r16 | 64 | 0.0 | 500 | 0.676 |

All 15 pairwise merges evaluated with uniform_linear strategy.

## Main results

| Outcome | Count | % |
|---------|-------|---|
| safe (<=1.5pp) | 9 | 60% |
| mildly_degraded (1.5-4pp) | 6 | 40% |
| materially_degraded (>4pp) | 0 | 0% |

No materially degraded merges. All degradation was mild (1.6–3.4pp). The worst merge (A×D: r8/a8 × r16/a64) dropped 3.4pp — from 0.742 to 0.708.

## Key finding: blind_spot_not_found

Same-task, same-dataset pairs remain safe enough across training-style variation that the current workflow is adequate in this regime.

The strongest degradation predictors are not training-style variables — they are **source strength mismatch** and **checkpoint maturity**.

## Analysis by contrast type

### Source strength mismatch is the main predictor

| Adapter involvement | Pairs | Safe | Degraded | Avg delta |
|---------------------|-------|------|----------|-----------|
| Pairs with A (weakest, 0.652) | 5 | 2 | 3 | +0.020 |
| Pairs without A | 10 | 7 | 3 | +0.009 |
| Pairs with F (early checkpoint, 0.676) | 5 | 2 | 3 | +0.016 |
| Pairs without F | 10 | 7 | 3 | +0.011 |

The two weakest adapters (A and F) account for most of the degradation. When both adapters are strong (B through E), merges are almost always safe.

### Training-style variables are secondary

| Contrast | Pairs | Safe | Degraded | Avg delta |
|----------|-------|------|----------|-----------|
| Same rank | 7 | 5 | 2 | 0.009 |
| Different rank | 8 | 4 | 4 | 0.016 |
| Both fully trained | 10 | 7 | 3 | 0.011 |
| One early checkpoint | 5 | 2 | 3 | 0.016 |

Rank difference and checkpoint stage show mild effects, but they are largely confounded with source strength. Dropout and alpha differences are not meaningfully predictive.

### Pair-risk captures the strong-vs-weak boundary

| Pair risk | Pairs | Safe | Degraded | Avg delta |
|-----------|-------|------|----------|-----------|
| medium | 13 | 9 | 4 | 0.011 |
| high | 2 | 0 | 2 | 0.025 |

The 2 high-risk pairs (both involving A, the weakest adapter) were both degraded. Pair-risk correctly identifies the strongest source of merge variation even within same-task pools.

## What the current workflow gets right

1. **Pair-risk separates the worst pairs.** The 2 high-risk pairs were both degraded. 9 of 13 medium-risk pairs were safe.
2. **Task advisory is correctly silent.** All 15 pairs are same-task — the advisory never fires. This is the right behavior.
3. **No materially degraded merges.** Even the worst same-task merge was only 3.4pp below the best source. Task identity dominates.

## What the current workflow does not capture

1. **Source strength mismatch within same-task pools.** Adapter A (0.652) merged with strong adapters (0.742) produces mild degradation. Pair-risk catches this when the mismatch causes norm imbalance, but not always.
2. **Checkpoint maturity.** Adapter F (early checkpoint, 0.676) produces slightly worse merges than fully trained adapters with the same hyperparameters.

These are real but small effects (1.6–3.4pp). They do not justify a new advisory or score.

## Recommendation

**`blind_spot_not_found`**

Same-task, same-dataset merges remain behaviorally safe across rank, alpha, dropout, and checkpoint-stage variation. The mild degradation that occurs is:
- driven primarily by source strength mismatch
- partly captured by existing pair-risk
- never materially degraded (>4pp)

No new signal or advisory is warranted for this regime. The current workflow is sufficient.

## Implication for the regime map

The same-task row in the regime map is confirmed:

> Same-task, all eligible → workflow mostly confirmatory, full advanced stack often overkill

Training-style diversity within a task does not create a blind spot that the current system misses.
