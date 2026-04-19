# N133 Phase 2 Exploratory Analysis

**Date:** 2026-04-11
**Inputs:** `sidecar/data/n133/{adapter_profiles,w0_properties,pair_alignment_full,pair_alignment_summary}.json`
**Status:** Phase 3 (merge eval) still running; this is an exploration of Phase 2 audit data only.

## Questions addressed

1. Full distribution of cross-task alignments — do any cross-task pairs exceed same-task?
2. Per-module asymmetry in alignment (q/k/v/o_proj)
3. Erank variation: task signal vs seed noise
4. (Bonus) Mistral W₀ C_k by module, layer-depth structure, task-pair structure

## Q1 — Cross-task alignment distribution

```
SAME-TASK (n=6)   : min=0.1057  max=0.1452  mean=0.1249  std=0.0144
CROSS-TASK (n=60) : min=0.0363  max=0.0494  mean=0.0409  std=0.0034

Gap: min(same) - max(cross) = 0.1057 - 0.0494 = +0.0563
```

**Zero overlap.** Same-task strictly above cross-task with a ~57% margin.
Cohen's d = 15.3. The nearest same-task pair (SQuAD, 0.1057) is 2.1×
the nearest cross-task pair (MNLI↔SST-2, 0.0494). B-P1 holds not by
threshold tuning but by a huge distributional gap.

Same-task ordering (weakest → strongest same-task signal):
SQuAD (0.106) < Summarization (0.108) < Code (0.126) < GSM8K (0.128) <
MNLI (0.137) < SST-2 (0.145).

Note that the two classification tasks (SST-2, MNLI) produce the cleanest
same-task signal — consistent with their low intrinsic erank and
concentrated task geometry.

## Q1b — Which cross-task pairs look most similar?

Top 5 cross-task pairs (all pairs of task-pairs, averaged over 4 seed combos):

| Task pair              | mean   | std    |
|------------------------|--------|--------|
| MNLI ↔ SST-2           | 0.0489 | 0.0004 |
| MNLI ↔ SQuAD           | 0.0444 | 0.0002 |
| MNLI ↔ Summarization   | 0.0439 | 0.0002 |
| SQuAD ↔ SST-2          | 0.0438 | 0.0001 |
| SST-2 ↔ Summarization  | 0.0429 | 0.0002 |

Bottom 5:
| Code ↔ Summarization   | 0.0376 |
| GSM8K ↔ SQuAD          | 0.0375 |
| Code ↔ GSM8K           | 0.0363 |

**Interpretation:** MNLI is a mild hub (highest 3 entries involve MNLI).
Natural-language discriminative tasks (MNLI/SST-2/SQuAD) cluster slightly.
Generative reasoning tasks (Code, GSM8K) are most isolated — even from
each other. The within-task-pair std is 0.0001–0.0007, so seed noise
is negligible; this ordering is real structure, not sampling variation.

**Per-task "hub score"** (mean cross-task alignment over the 20 pairs
containing that task):
MNLI 0.0436 > SST-2 0.0428 > SQuAD 0.0412 > Summ 0.0409 > Code 0.0387 > GSM8K 0.0379.

Range is only 0.006 — the hub effect is real but small relative to the
same/cross gap.

## Q2 — Per-module asymmetry (alignment side)

Mean alignment by LoRA module, split by same vs cross:

| Module | same_mean | cross_mean | **ratio** | t-stat |
|--------|-----------|------------|-----------|--------|
| Q      | 0.1085    | 0.0417     |  2.60×    |  56.49 |
| K      | 0.1325    | 0.0756     |  **1.75×** |  41.63 |
| V      | 0.1450    | 0.0304     |  4.77×    |  97.30 |
| O      | 0.1135    | 0.0157     |  **7.23×** | 127.03 |

**Headline finding:** The aggregate 3.06× B-P2 ratio is an average
that masks enormous per-module heterogeneity. **O-projection alone
gives 7.23× separation** (nearly matching DistilBERT's aggregate 5×).
**K-projection alone gives only 1.75× separation — below the B-P2
threshold of 2.0×.** A triage pipeline that used only K-projection
alignment would fail B-P2.

Interpretation: K-projection has a high "cross-task floor" (0.0756 —
almost 2× any other module's cross-task mean), while O-projection has
the cleanest floor (0.0157). This suggests K-projection picks up
cross-task-shared structure (syntactic / positional features),
while O-projection is where task-specific content concentrates.

V and O carry the signal; Q and K carry noise.

## Q2b — Per-module erank (adapter side)

| Module | mean erank | std    | note                          |
|--------|-----------|--------|-------------------------------|
| Q      |  7.90     |  2.07  | highest erank                 |
| O      |  7.61     |  2.45  |                               |
| K      |  6.66     |  1.85  |                               |
| V      |  5.33     |  2.09  | **lowest erank**              |

ANOVA F=114.2, p = 10⁻⁶⁶. Huge asymmetry across modules.

**Combined with Q2:** V has the LOWEST erank (5.33) but the HIGHEST
same-task alignment (0.145). V projections are compressed (small task
subspace) and that subspace is consistently used across seeds of the
same task — the archetypal "task-specific low-rank update". Q has the
HIGHEST erank but only 0.109 same-task alignment — Q is using more
directions but those directions are less stable across seeds.

This is a much richer story than simply "Q/K erank > V/O erank".

## Q3 — Erank variance: task signal vs seed noise

Per-adapter mean erank (two seeds per task):

| Task          | s42    | s123   | Δ      | Δ/mean |
|---------------|--------|--------|--------|--------|
| SST-2         |  5.139 |  5.273 | 0.134  | 2.58%  |
| MNLI          |  6.005 |  6.075 | 0.070  | 1.17%  |
| SQuAD         |  7.653 |  7.438 | 0.215  | 2.85%  |
| GSM8K         |  6.825 |  6.902 | 0.077  | 1.12%  |
| Code          |  6.671 |  6.523 | 0.148  | 2.24%  |
| Summarization |  8.899 |  9.086 | 0.187  | 2.08%  |

**Seed noise: ≤ 3% of mean erank on every task.**

Decomposed variance:
- Between-task SD (over 6 task means) = 1.190
- Within-task typical SD (from seed pairs) = 0.098
- **Between / within ratio = 12.1×** at the adapter-mean level

Per-(task, layer, module) cell decomposition:
- Within-seed pooled SD = 0.718
- Between-task per-(layer,module) SD = 1.570
- Ratio = 2.2× (smaller because layer-by-layer noise is larger)

Inference tests:
- One-way ANOVA F = 105.42, p = 9.3 × 10⁻⁹⁶
- Friedman χ² = 388.0, p = 1.1 × 10⁻⁸¹ (non-parametric, blocks by layer & module)

**Verdict:** Erank variation across tasks is massively significant and
not a seed-noise artifact. Even the weakest task separation (GSM8K vs
Code, means 6.86 vs 6.60) exceeds the seed noise band.

## Q4 — W₀ C_k distribution by module (Mistral-7B pretrained)

| Module | mean C_k | std    | min    | max    |
|--------|----------|--------|--------|--------|
| Q      | 0.5641   | 0.1957 | 0.2063 | 0.9999 |
| K      | 0.5470   | 0.1764 | 0.1568 | 0.9964 |
| V      | **0.0853** | 0.1289 | 0.0024 | 0.6919 |
| O      | 0.3155   | 0.1475 | 0.0728 | 0.7650 |

**V-projection has radically lower C_k than Q/K/O.** Mistral's V matrices
have a flat spectrum (C_k ≈ 0.085 means you need ~92% of the rank to
capture the energy, vs ~56% for Q). This is a property of the pretrained
network, not the adapters.

GQA comparison:
- Q + O (shape 4096 × 4096): mean C_k = 0.440
- K + V (shape 4096 × 1024, GQA-compressed): mean C_k = 0.316
- t = 2.80, p = 0.006

GQA compression does shift C_k, but the V vs K asymmetry (both GQA-shape)
is much larger than the GQA effect itself. V's low C_k is a
spectral-shape fact about Mistral, not a shape artifact.

**Implication for B-P3:** The C_k → alignment correlation is being
computed on a highly heterogeneous C_k distribution. V-modules contribute
most of the low-C_k samples and also most of the high-alignment samples,
inflating a spurious negative correlation. The "wrong sign" B-P3 result
may be partially driven by V-module dominating the low-C_k end.

Follow-up: recompute the B-P3 correlation **within each module
separately**. If the correlation is ≈ 0 within Q and within O, but
negative when pooled, then the overall negative ρ is Simpson's-paradox
confounding, not a genuine inverse relationship.

## Q5 — Layer-depth analysis

Mean same-task alignment by layer index:

| Layer | same_mean | cross_mean | ratio |
|-------|-----------|------------|-------|
|  0    | 0.1619    | 0.0697     | 2.32× |
|  7    | 0.1081    | 0.0378     | 2.86× |
| 15    | 0.1100    | 0.0356     | 3.09× |
| 23    | 0.1321    | 0.0418     | 3.16× |
| 31    | 0.2066    | 0.0488     | **4.24×** |

- ρ(layer, same_alignment)  = +0.540, p = 0.0014
- ρ(layer, cross_alignment) = +0.134, p = 0.46 (NS)

**Same-task alignment increases with depth; cross-task alignment does not.**
This means the triage signal-to-noise ratio *grows* with depth: layer 31
gives 4.2× separation, layer 0 only 2.3×. Deeper layers encode more
abstract task-specific representations; early layers are closer to
token-level features and therefore more cross-task-shared.

Combined with the Q2 module finding: a layer-31 O-projection alone
would likely give an enormous same/cross ratio (rough estimate:
7.23× × (4.24/3.06) ≈ 10×). This is where the signal concentrates.

## Headline results

1. **B-P1 holds by a 57% distributional gap (Cohen's d = 15.3)**, not
   by threshold tuning. Closest cross-task pair is 2.1× below closest
   same-task pair.
2. **Per-module triage signal is highly heterogeneous.** O: 7.23×,
   V: 4.77×, Q: 2.60×, **K: 1.75× (below B-P2 threshold)**. Aggregate
   3.06× is misleading; O-projection carries most of the signal.
3. **V has lowest erank but highest same-task alignment.** Classic
   "compressed task-specific update" pattern. Q has highest erank but
   lower alignment — more dimensions, less stable use of them.
4. **Erank variance is 12× task-dominated vs seed-dominated** at the
   adapter level. The N130/N132/N133 erank-varies-by-task finding is
   solid and not a sampling artifact.
5. **Triage signal grows with layer depth.** Layer 0: 2.32×, Layer 31:
   4.24×. Deep-layer O-projection is the strongest single signal
   available.
6. **V-module W₀ C_k is radically lower than Q/K/O C_k** (0.085 vs
   0.31-0.56). This suggests the negative-sign B-P3 result may be
   confounded by V-module dominating the low-C_k / high-alignment
   regime. Per-module B-P3 analysis is warranted.
7. **MNLI is a mild hub task** (highest mean cross-task alignment),
   but the hub effect is small compared to the same/cross gap.

## Q6 — B-P3 per-module analysis (Simpson's paradox confirmed)

Per-module Spearman correlation of W₀ C_k with same-task alignment:

| Module | n    | ρ(C_k, align) | p       |
|--------|------|---------------|---------|
| Q      | 192  | −0.173        | 0.016   |
| K      | 192  | −0.172        | 0.017   |
| V      | 192  | **+0.143**    | 0.047   |
| O      | 192  | −0.062        | 0.394   |
| Pooled | 768  | **−0.216**    | 10⁻⁹    |

Residualized (controlling for module identity):

- Within-module Spearman ρ = −0.066, p = 0.069 (NS)
- Within-module Pearson r = +0.106, p = 0.003 (right sign)

**Verdict: the pooled "wrong sign" result is a Simpson's paradox
artifact.** V-module has radically low mean C_k (0.085) and high
mean alignment (0.145); Q-module has high mean C_k (0.564) and low
alignment (0.109). When you pool across the four modules, the
between-module structure dominates the within-module relationship
and produces a spurious significant negative pooled coefficient.

When module is controlled for, the C_k → alignment relationship on
Mistral is **essentially null**. This matches N132 (DeBERTa) rather
than contradicting it more dramatically. Both decoder-era models
produce near-zero within-module C_k effects.

**Methodological implication for §13 (DistilBERT):** the original
ρ = 0.53 DistilBERT result should be re-verified with per-module
stratification. If it survives, it is a genuine finding. If it
also drops to null after stratification, the entire C_k →
alignment program needs to be rethought as a between-module artifact
of DistilBERT's specific spectral geometry.

## Open follow-ups

- ✅ Recompute B-P3 per module — done (Q6, Simpson's paradox confirmed).
- ✅ Generate figures — done (per-module bars, layer-depth curves,
  C_k-by-module scatter in `sidecar/data/n133/figures/`).
- Re-run DistilBERT N127/N130 B-P3 with per-module stratification
  to check whether §13's ρ = 0.53 is also a Simpson's paradox.
- Cross-reference with Phase 3 merge outcomes once Phase 3 completes —
  does O-projection alignment predict merge degradation better than
  aggregate alignment?
- Build a layer-31 O-projection-only triage variant and compare
  against the aggregate baseline.
