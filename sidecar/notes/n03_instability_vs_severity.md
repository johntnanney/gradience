# Note: Instability as a First-Class Phenomenon

## Metadata

- **Type:** implication
- **Date:** 2026-03-26
- **Related studies:** S01
- **Related panels:** P01

---

## Summary

Instability — the degree to which a pair's severity varies across seeds and backbones — appears to be a more robust and more informative descriptor of cross-task merge risk than absolute severity. The two catastrophic anchors in the evidence base are also the two most unstable pairs by every metric. This note argues that instability deserves first-class status in the sidecar's conceptual framework.

## The Severity Problem

The core finding from the two-backbone analysis is that severity does not generalize. QNLI×MRPC is catastrophic on DistilBERT (41.7%) and mild on RoBERTa (1.7%). QNLI×SST-2 moves in the opposite direction: severe on DistilBERT (11.0%), catastrophic on RoBERTa (27.2%).

This makes absolute severity a poor candidate for a portable cross-task descriptor. A severity ranking derived from one backbone would actively mislead on another.

## Instability as an Alternative

Instead of asking "how severe is this pair?" we can ask "how unstable is this pair?" — meaning, how much does its severity vary when conditions change (seeds, backbones)?

The instability analysis computes a composite score from three components:

- **Seed range** (0.4 weight): how much the worst-case delta varies across seed combinations within a backbone
- **Backbone shift** (0.3 weight): how much the worst-case delta changes between backbones
- **Coefficient of variation** (0.3 weight): relative variability of delta across conditions

## Results

### Instability ranking

| Pair | Instability | Taxonomy | Max seed range | Backbone shift |
|------|------------:|----------|---------------:|---------------:|
| QNLI × MRPC | 0.87 | backbone reversal | 28.9% | 40.0% |
| QNLI × SST-2 | 0.74 | backbone reversal | 26.2% | 16.2% |
| QNLI × RTE | 0.30 | stable asymmetric | 6.9% | 1.9% |
| MRPC × SST-2 | 0.21 | stable asymmetric | 8.0% | 2.2% |
| RTE × MRPC | 0.19 | stable asymmetric | 4.7% | 1.2% |
| RTE × SST-2 | 0.15 | stable asymmetric | 4.3% | 4.3% |

### Key observation

There is a clear gap between the two backbone-reversal pairs (instability > 0.7) and the four stable-asymmetric pairs (instability < 0.3). This gap is much cleaner than any severity-based ranking, which would disagree between backbones.

### Regime contrast

Instability also separates regimes that severity does not:

| Regime | N pairs | Max Δ | Description |
|--------|--------:|------:|-------------|
| Same-task | 8 | 2.2% | Always safe |
| Domain shift | 15 | 2.2% | Always safe (even cross-domain sentiment) |
| Source strength | 15 | 2.4% | Always safe (even strong × weak) |
| Training style | 15 | 3.4% | Always safe (even different rank/alpha) |
| Cross-task (stable) | ~16 | 15.0% | Degrades but predictably |
| Cross-task (unstable) | ~8 | 41.7% | Degrades unpredictably |

The cross-task regime splits into a stable zone (where instability < 0.3, deltas 5–15%, consistent across conditions) and an unstable zone (where instability > 0.7, deltas anywhere from 1% to 42% depending on seed/backbone).

## What Instability Reveals

### 1. Catastrophic pairs are the unstable pairs

This is not a tautology. High severity and high instability could in principle be independent. A pair could be consistently catastrophic (high severity, low instability) or erratically moderate (low severity, high instability). In fact, the data shows that high severity and high instability co-occur: the pairs that are catastrophic somewhere are also the pairs that are most variable across conditions.

This suggests that catastrophic interference has a threshold character. It requires specific conditions (seed, backbone) to trigger. When those conditions are not met, the same pair is mild.

### 2. Instability may be more portable than severity

The two backbone-reversal pairs are the most unstable on both backbones, even though their severity profiles are inverted. QNLI×MRPC is unstable on DistilBERT (range 28.9%) and would likely show high instability on a third backbone too, even if its severity there is low.

If this hypothesis holds on DeBERTa-v3, instability would be the first candidate for a cross-backbone merge descriptor.

### 3. Instability explains why severity signals fail to generalize

The failed severity-grading signals (core-space shared-basis, pair-risk, format similarity, source-strength gap) all tried to predict a pair's absolute severity. But if severity is itself unstable, then the problem is not that the signals are wrong — it is that the target variable is not well-defined across conditions. Instability reframes the problem: instead of predicting a specific delta, we can predict whether a pair is in the stable or unstable regime.

## Implications

### For the sidecar

Instability should become a first-class concept alongside severity. The taxonomy (stable mild, stable asymmetric, unstable severe, backbone reversal) is a better organizing framework for future studies than a simple severity ranking.

Workstream B (layerwise conflict) should compare stable-asymmetric pairs against backbone-reversal pairs, not just "catastrophic vs mild." The relevant contrast is not severity but stability.

### For core Gradience

Instability is **not yet promotable** to core. The instability score is computed from two backbones, and we do not yet know whether it is stable across a third. It also requires behavioral evaluation data that is expensive to generate.

However, if instability proves portable across DeBERTa-v3, it would be the strongest candidate for a future core feature. It would change the preflight output from "this is a cross-task pair, exercise caution" to "this is an unstable cross-task pair, budget extra evaluation."

That is a meaningful workflow improvement — but it requires the DeBERTa leg of S01 to confirm portability.

### For the research direction

The instability framing shifts the sidecar's central question from "what determines cross-task severity?" to "what determines cross-task stability?" This is a better question because:

1. Stability is more likely to be a real property of the pair (it persists across seeds and may persist across backbones).
2. Stability is more actionable (an unstable pair deserves more evaluation budget, regardless of its point estimate).
3. Stability may have a mechanistic explanation (threshold-dependent interference in specific subspaces) that severity alone does not.

## Decision or Recommendation

Adopt instability as a first-class concept in the sidecar's research framework. Update future study designs to stratify by stability class rather than severity class where possible. The DeBERTa leg of S01 should specifically test whether the instability ranking is preserved on a third backbone.
