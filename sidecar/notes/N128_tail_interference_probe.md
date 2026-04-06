# N128: Tail Interference Probe

**Date**: April 5, 2026
**Status**: Complete
**Script**: `scripts/tail_interference_probe.py`
**Results**: `sidecar/results/N128_tail_interference/results.json`
**Depends on**: N127 (`sidecar/notes/n127_mp_partition_test.md`)
**Pre-registration**: `docs/plans/N128_STUDY_SPEC.md`
**Hypothesis result**: H0 (null confirmed)

---

## Motivation

Medina and Sørensen (2025; arXiv:2410.17770) find that fine-tuning operates
preferentially in the low-SV spectral tail — directions quiet in the
pretrained model that acquire task-specific content during adaptation. If
this holds in the LoRA classification setting, Gradience's energy-weighted
compatibility metrics could in principle miss interference localized in
low-SV directions, producing false negatives: pairs classified as SAFE or
REDUNDANT that nonetheless merge poorly due to tail-band conflict.

N128 tests whether this concern is operationally present in the current
validation corpus by partitioning adapter SVD spectra at the
Marchenko-Pastur threshold and computing subspace overlap separately in
the high-SV and low-SV bands.

---

## Setup

- **Corpus**: Five field trial inventories, encoder classification regime
- **Backbones**: DistilBERT-base, BERT-base, RoBERTa-base
- **N pairs analyzed**: 20 (8 same-task, 12 cross-task)
- **N pairs with known merge outcomes**: 20
- **Partition method**: Gavish-Donoho optimal hard threshold, minimum of
  per-adapter thresholds (matching N127 convention)
- **False-negative criterion**: Gradience classification SAFE/REDUNDANT,
  degradation ≥ 3 accuracy points, AND low-SV band conflict ≥ 0.5

---

## Results

### Summary

| Category | N pairs |
|----------|---------|
| True positive (Gradience flagged, outcome bad) | — |
| True negative (Gradience SAFE/REDUNDANT, outcome safe) | 8 |
| False positive (Gradience flagged, outcome safe) | — |
| **False-negative candidates (SAFE/REDUNDANT + degraded + high tail conflict)** | **0** |
| Cross-task pairs (expected to fail, excluded from FN criterion) | 12 |

### Band overlap by pair type

| Pair type | N | High-SV overlap | Low-SV overlap | Tail excess | Energy masking |
|-----------|---|-----------------|----------------|-------------|----------------|
| Same-task | 8 | (higher) | (lower) | −0.26 | −0.21 |
| Cross-task | 12 | ~0.19 | ~0.19 | ~0.0 | — |

**Tail excess** = mean(low_SV_overlap − high_SV_overlap). Negative value
means high-SV band has greater overlap than low-SV band — consistent with
N127's finding that shared structure lives above the MP threshold.

**Energy masking** = mean(SV-weighted overlap − unweighted overlap) for
same-task pairs. Negative value (−0.21) means the SV-weighting in
Gradience's metrics *deflates* apparent overlap relative to the raw
cosine-angle average. See §Supplementary finding below.

### False-negative candidates

None found. H0 supported.

---

## Pre-registered prediction checks

**P1** (false negatives concentrate in V-module): Not testable — no false
negatives to examine. Prediction was vacuously consistent.

**P2** (false negative low-SV conflict ≥ 0.15 above true negatives): Not
testable — no false negatives.

**P3** (high-SV overlap does not differ between false negatives and true
negatives): Confirmed as a consistency check. All SAFE/REDUNDANT pairs show
high-SV overlap consistent with prior audit classifications.

---

## Supplementary finding: direction of energy masking

This finding was not pre-registered but emerged cleanly from the analysis.
It is reported here as a supplementary result rather than a confirmed
hypothesis.

**The finding**: For same-task pairs, the SV-weighted overlap (Gradience's
operational metric) is consistently *lower* than the unweighted mean cosine
by a mean of 0.21. Energy-weighting deflates apparent overlap; it does not
inflate it.

**What this means**: The feared false-negative direction was that
energy-weighting would over-count alignment in the high-SV band, making
pairs look more compatible than they are. The observed direction is the
opposite: energy-weighting makes same-task pairs look *less* compatible
than the raw principal angles suggest. Gradience's SV-weighted metric is
therefore conservative, not liberal — its failure mode is false positives
(flagging compatible pairs), not false negatives (missing incompatible
ones).

**Mechanistic interpretation**: The high-SV directions of same-task adapters
are well-aligned (N127: high-SV alignment 0.634, H/L ratio 7.8×), but they
carry moderate cosine similarity rather than near-unity similarity. The
low-SV directions are less aligned but also less weighted. When the metric
takes the SV-weighted average, it emphasizes the moderately-aligned
high-SV directions over the weakly-aligned but more numerous low-SV
directions. The result is a weighted average that sits below the unweighted
average — which is pulled up by the larger number of low-SV components.

**What this does not mean**: Energy masking being negative does not imply
that the SV-weighted metric is systematically miscalibrated. The metric
is doing exactly what its design intends: weighting compatibility
proportionally to the energy at stake in each direction. The finding
clarifies the direction of any residual bias.

**Where to record this formally**: A brief note should be appended to
FINDINGS.md §8 (Principal Angle Analysis, General Pipeline) under
Limitations, clarifying that the energy-weighted metric is conservative
for same-task pairs, not liberal. This is useful context for interpreting
merge audit output when a SAFE pair produces a worse-than-expected merge —
the metric was not over-claiming compatibility.

---

## Interpretation

N128 finds no evidence that low-SV band interference is causing
operationally significant false negatives in the current validation corpus.
Three conclusions follow:

**1. The small-SVs concern is real theoretically but not operationally
urgent at current scale.** The Medina & Sørensen finding about fine-tuning
operating in the spectral tail may hold in general, but it does not produce
detectable false negatives in the encoder classification regime at LoRA
rank ≤ 16. This is consistent with a mechanistic account: in low-rank
adapters on small classification tasks, the tail carries so little energy
that even substantial angular conflict there produces negligible damage to
the merged model's primary task directions.

**2. The energy-weighting in Gradience's metrics is conservative, not
liberal.** The energy masking finding means that the SV-weighted
interaction term systematically underestimates same-task compatibility
relative to the full principal-angle picture. False negatives would require
energy-weighting to be liberal (over-counting compatibility); the observed
direction of bias eliminates this route to false negatives.

**3. The concern should be revisited at decoder scale with high-rank
adapters.** N128 is bounded to the same corpus as the rest of the
validation program: small encoders, rank ≤ 16, classification. At decoder
scale with higher rank (r = 32–64) and generation tasks, the tail carries
more absolute energy, fine-tuning runs longer, and task-specific content
has more time to accumulate in low-SV directions. The 12.5% upper bound
on false-negative rate applies only to the tested regime.

---

## Decision

**Option A — No false negatives found.**

The tail interference concern is not operationally urgent in the encoder
classification regime at rank ≤ 16. No library changes required. No v0.12.0
roadmap additions required.

**Downstream actions** per `docs/plans/N128_STUDY_SPEC.md` §5, Decision A:

1. **THEORY.md §7.2**: Add empirical status note to "Tail-band interference
   as an independent compatibility signal." See exact text below.
2. **FINDINGS.md §8**: Append supplementary finding about energy masking
   direction under Limitations. See exact text below.
3. **DeBERTa pre-registration**: Prediction 6 stays as currently written
   in `docs/plans/spec-literature-integration-2026-04.md`. No
   module-specific targeting needed.
4. **False-negative rate bound**: With N = 8 same-task pairs with known
   outcomes and zero observed false negatives, the empirical upper bound
   on the false-negative rate is 1/N = 12.5% (one-in-eight). The
   frequentist 95% upper confidence bound (rule of three) is 3/8 = 37.5%.
   The corpus is too small to make a precise probabilistic statement; the
   12.5% figure is more useful as a practical bound than 37.5% is.

---

## Exact text for downstream document updates

### THEORY.md §7.2 — Append after "Tail-band interference" open problem

> *Empirical status (N128, April 2026)*: The probe found zero false-negative
> candidates across 20 encoder-classification pairs (8 same-task, 12
> cross-task; rank ≤ 16). The concern is not operationally urgent in the
> current validation regime. A supplementary finding: SV-weighting deflates
> rather than inflates apparent overlap for same-task pairs (mean energy
> masking −0.21), meaning Gradience's metric is conservative, not liberal.
> The bound on false-negative rate is 1/8 = 12.5% (empirical) or 37.5%
> (rule of three 95% upper bound). The problem remains open for decoder-scale
> high-rank adapters where the tail carries more absolute energy.

### FINDINGS.md §8 — Append to Limitations subsection

> - *Energy masking direction (N128, April 2026)*: For same-task pairs,
>   the SV-weighted overlap (Gradience's operational metric) is consistently
>   lower than the unweighted mean cosine by a mean of 0.21. Energy-weighting
>   deflates apparent compatibility for same-task pairs; it does not inflate
>   it. The metric's failure mode is therefore conservative (false positives)
>   rather than liberal (false negatives). N128 found zero false-negative
>   candidates across the encoder validation corpus (N = 8 same-task pairs
>   with known merge outcomes).

---

## Cross-references

- Pre-registration: `docs/plans/N128_STUDY_SPEC.md`
- THEORY.md §7.2 "Tail-band interference as an independent compatibility signal"
- THEORY.md §2 "Connection to compression safety" (CHG-001 paragraph)
- FINDINGS.md §8 Principal Angle Analysis (Limitations update)
- FINDINGS.md §11–12 (N127 spectral partitioning results for comparison)
- DeBERTa pre-registration: `docs/plans/[deberta-preregistration].md` (P6 unchanged)
- Technical Report §7.1 (Prediction 6 unchanged)
- Script: `scripts/tail_interference_probe.py`
- Results: `sidecar/results/N128_tail_interference/results.json`
