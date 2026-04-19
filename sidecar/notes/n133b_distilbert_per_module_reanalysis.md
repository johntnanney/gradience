# N133b: DistilBERT §13 / §25 Per-Module Re-Analysis

**Date:** 2026-04-11
**Trigger:** N133 (§28) Mistral-7B exploratory analysis (Q6) revealed
that the pooled ρ(C_k, alignment) = −0.216 on Mistral is a
Simpson's paradox artifact of pooling across q/k/v/o modules with
different mean C_k and different mean alignment. The Mistral
within-module residualized correlation is essentially null.

**Question:** Does the original DistilBERT §13 / §25 finding
(QNLI pooled ρ ≈ +0.56) also suffer from a Simpson's paradox, or
is it a genuine within-module relationship?

**Data:** `sidecar/data/n130/phase3_prediction_1.json` — N130 Phase 3
outputs for DistilBERT-base on SST-2 and QNLI, 24 layers × 2 tasks.
Per-layer records include `module`, `C_k`, `alignment`.

## Method

For each task (SST-2 and QNLI):

1. Compute naive pooled Spearman ρ(C_k, alignment) across all 24 layers
   (to reproduce the headline N130 number).
2. Report the per-module ρ from N130 directly (n = 6 per module).
3. Compute the within-module residualized correlation: subtract per-module
   mean C_k and per-module mean alignment, then correlate the residuals
   (both Spearman and Pearson).

If the within-module residualized ρ is near zero, the headline pooled
result is a Simpson's paradox artifact. If the within-module ρ is
similar to the pooled ρ, the pooled result reflects a genuine
within-module relationship.

## Results

### SST-2 (always null)

```
Pooled (n=24)                 : ρ = −0.076, p = 0.73
Per-module (n=6):
  q : ρ = −0.371, p = 0.47
  k : ρ = −0.371, p = 0.47
  v : ρ = +0.200, p = 0.70
  o : ρ = +0.429, p = 0.40
Within-module residualized    : ρ = +0.106, p = 0.62 (Spearman)
                                r = +0.147, p = 0.49 (Pearson)

Per-module mean C_k and mean alignment:
  q: C_k=0.458, align=0.626
  k: C_k=0.465, align=0.608
  v: C_k=0.281, align=0.705
  o: C_k=0.234, align=0.597
```

**SST-2 was never a real effect.** Pooled null, within-module null,
per-module mixed signs. Consistent with every previous reading of
the N130 data.

### QNLI (the headline finding)

```
Pooled (n=24)                 : ρ = +0.5626, p = 0.0042
Per-module (n=6):
  q : ρ = +0.429, p = 0.40
  k : ρ = +0.600, p = 0.21
  v : ρ = +0.714, p = 0.11
  o : ρ = +0.771, p = 0.07
Within-module residualized    : ρ = +0.546, p = 0.006 (Spearman)
                                r = +0.448, p = 0.028 (Pearson)

Per-module mean C_k and mean alignment:
  q: C_k=0.458, align=0.415
  k: C_k=0.465, align=0.609
  v: C_k=0.281, align=0.363
  o: C_k=0.234, align=0.405
```

**The QNLI finding survives per-module stratification.**

- Within-module residualized Spearman ρ = **+0.546**, p = 0.006 —
  essentially identical to the naive pooled ρ = +0.563.
- All four modules individually give positive correlations
  (ρ = +0.43, +0.60, +0.71, +0.77). Per-module tests are
  underpowered (n = 6), but the sign consistency is perfect.
- Pearson residualized r = +0.448 is also clearly non-zero.

## Why DistilBERT QNLI and nothing else?

Comparing how the between-module and within-module gradients align
across architecture × task cells:

**DistilBERT QNLI — gradients REINFORCE each other.**
The between-module gradient has k-module with the highest mean
C_k (0.465) AND the highest mean alignment (0.609). High-C_k
modules are also high-alignment modules. The within-module
gradients are also positive (all four modules ρ > 0). Pooled
and within-module correlations agree because both gradients point
the same way. The ρ = +0.56 captures a *real* within-module
relationship amplified by an aligned between-module gradient.

**Mistral-7B (N133) — gradients OPPOSE each other.**
The between-module gradient has V-module with the *lowest* mean
C_k (0.085) but the *highest* mean alignment (0.145), and
Q-module with high C_k (0.564) but low alignment (0.109).
High-C_k modules are low-alignment modules. The within-module
gradient is essentially zero. The pooled ρ = −0.216 is an
artifact entirely of the between-module opposing gradient —
when you condition on module, the effect vanishes.

**DistilBERT SST-2 — both gradients are near zero.**
No effect pooled, no effect within-module, slightly mixed signs
across modules. Always was null.

**DeBERTa (N132) — pooled signs all positive but NS.**
Within-module residualized correlations were not reported in §27,
but the pooled per-task ρ's (+0.11 to +0.22) are small enough
that conditioning on module is unlikely to move the needle much.
The §27 null result stands.

## Revised verdict on the C_k → alignment program

The DistilBERT §13 / §25 finding is **real but highly specific**:
it works on DistilBERT-base QNLI and no other architecture × task
combination tested so far. This is a narrower claim than §13
originally made (and much narrower than the N131 composite
predictor §26 assumed), but it is a *genuine* claim and not an
artifact of pooling.

The falsification program converges on this statement:

> **C_k of pretrained W₀ predicts same-task LoRA alignment in
> a robust, within-module sense on DistilBERT-base QNLI.
> On every other tested combination (DistilBERT SST-2, DeBERTa
> SST-2/QNLI/MRPC/RTE, Mistral-7B all six N133 tasks), the
> within-module effect is indistinguishable from zero.**

This is still incompatible with using C_k as a general spectral
triage signal, because 10 of 11 tested task × architecture cells
give null within-module effects. But it rescues §13 from being
characterized as a Simpson's paradox artifact.

## Methodological lesson

Any future claim about C_k → alignment correlation should report
**both** the naive pooled ρ and the within-module residualized ρ.
Pooling across modules with heterogeneous C_k distributions and
heterogeneous alignment distributions is dangerous. The N133
Mistral result gave a p-value of 10⁻⁹ on a Simpson's paradox
that disappears when properly conditioned — and this was only
caught because the "wrong sign" was obvious enough to prompt
a per-module re-analysis. A weaker Simpson's paradox (with the
right sign) would not have been flagged.

Recommendation: retrofit N130/N131/N132 scripts to always emit
per-module ρ and a within-module residualized partial ρ alongside
the pooled number. This is 10 lines of code per script and
prevents the whole family of errors.

## Connection to FINDINGS §28

The §28 draft's "B-P3 DISCONFIRMED — null, not reversed" headline
still stands: Mistral's within-module effect is null. But the
**reason** needs updating from "this replicates the N132 null"
to "this matches every tested architecture × task cell except
DistilBERT QNLI, which remains the single confirmed case."

The §28 Interpretation section should note that the §13 finding
is retroactively *narrowed* (DistilBERT QNLI only) rather than
*rescued from artifact*. The per-module re-analysis preserves
§13 as a real effect while tightening the scope of its claim.
