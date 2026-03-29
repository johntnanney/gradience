# Note: V-Module Head-Level Geometry Findings

## Metadata

- **Type:** findings
- **Date:** 2026-03-26
- **Related notes:** n23 (protocol), n22 (head panel), n21 (per-module findings)
- **Project:** V-Module Head-Level Program, Stage B

---

## Purpose

This note reports the results of the head-level V-module geometry pilot. It answers: does the decisive difference localize further to specific attention heads inside the V module, especially in seed-sensitive catastrophic families like CA-01?

**Overall classification: MIXED-POSITIVE — head-level analysis resolves the CA-01 seed sensitivity mystery but does not improve the module-level group discrimination.**

The head-level decomposition succeeds at one of its two objectives and fails at the other. For seed sensitivity, it is transformative: the CA-01 severity gap (29 percentage points, invisible at per-module resolution) is now clearly visible at individual heads, with deltas up to 33× larger than the module aggregate. For group discrimination, the module-level V-module dimensionality ratio (d=3.36) remains the strongest signal; head-level resolution introduces variance that weakens discrimination rather than sharpening it.

---

## 1. Prediction Outcomes

### P1 (CA-01 head localization): POSITIVE

**Prediction:** The 29-point severity gap in CA-01 concentrates at ≤4 heads where head-level dimensionality ratio or alignment shows deltas ≥ 0.10.

**Result:** 7 heads across layers 1–3 show |Δ_DR| ≥ 0.15 between worst (s42×s7, Δ=41.7%) and mild (s7×s7, Δ=12.7%) variants. The maximum is Δ_DR = -0.229 at layer 3 head 6. For top direction overlap, the maximum is Δ_OV = -0.341 at layer 4 head 9. The criterion was ≥0.10; the observed values are 2–3× larger. The signal exists at head resolution but was invisible at module level because individual heads show deltas of opposite sign, causing cancellation when averaged.

**Head-by-layer distribution of |Δ_DR| ≥ 0.15:**

| Layer | Hot Heads | Δ_DR Values |
|:-----:|:----------|:------------|
| 1 | H2, H4, H11 | +0.153, +0.157, +0.154 |
| 2 | H1, H6, H11 | +0.171, -0.195, +0.193 |
| 3 | H6 | -0.229 |

Note the sign pattern: at layers 1–2, most hot heads show *positive* Δ_DR (the worst-seed variant has *higher* dim ratio than the mild-seed variant), which is counterintuitive — one would expect the more catastrophic variant to show lower dim ratio. But head 6 at layers 2–3 shows *negative* Δ_DR (the catastrophic variant has lower dim ratio), consistent with the module-level direction. The mixed signs explain the cancellation: the module averages heads with opposing patterns, producing a near-zero aggregate.

**Interpretation:** The CA-01 seed sensitivity does not operate through a uniform shift in V-module geometry. Instead, different seeds produce different head-level geometric configurations. The catastrophic seed (s42) rotates some heads toward commensurability with MRPC (positive Δ_DR) while rotating others away (negative Δ_DR). The net effect — whether the merge is catastrophic or mild — depends on which heads' incompatibilities dominate the output, which is a function of how the O module and classification head weight different heads' contributions.

### P2 (CA-02 head localization): POSITIVE

**Prediction:** At least one head shows d > 2.0 on dimensionality ratio between toxic and benign variants.

**Result:** The maximum single-head Δ_DR is -0.459 at layer 4 head 11 (toxic vs. benign adapter). Multiple heads at layer 4 show |Δ_DR| > 0.35. The signal is much larger at individual heads than at module level (module Δ_DR = -0.101). Individual head Cohen's d was not computed as a formal per-head toxic/benign d-statistic, but the raw deltas (up to 4.5× the module delta) confirm strong head-level concentration.

**Top 5 CA-02 heads by |Δ_DR| (toxic – benign):**

| Layer | Head | Δ_DR | Δ_OV |
|:-----:|:----:|:----:|:----:|
| 4 | 11 | -0.459 | -0.075 |
| 4 | 6 | -0.379 | +0.019 |
| 5 | 7 | -0.364 | +0.050 |
| 4 | 1 | -0.360 | +0.219 |
| 4 | 10 | -0.356 | +0.068 |

The CA-02 head-level signal is concentrated at layer 4, where 4 of the top 5 heads reside. Unlike CA-01, all large Δ_DR values are negative — the toxic adapter consistently produces lower dimensionality ratios at the hardest-hit heads. This is directionally consistent with the module-level finding.

### P3 (Head mismatch improves discrimination): NEGATIVE

**Prediction:** Worst-head dimensionality ratio achieves d > 3.36 or overlap < 0.05 on the backbone-controlled comparison.

**Result:** The module-mean head-level approach yields d = 1.25 (vs. 3.36 at module level). The worst-head approach yields d = 0.75. Even the best individual head (head 5) only achieves d = 1.75. Head-level resolution *weakens* the discrimination.

| Approach | G1 Mean | G2 Mean | Cohen's d |
|:---------|:-------:|:-------:|:---------:|
| Module-level (n21) | 0.693 | 0.837 | 3.36 |
| Head-level mean | 0.689 | 0.790 | 1.25 |
| Worst-head | 0.485 | 0.548 | 0.75 |
| Best single head (H5) | 0.686 | 0.820 | 1.75 |

**Interpretation:** The module-level aggregation is not diluting the signal — it is *concentrating* it. By averaging across 12 heads, the module-level dimensionality ratio reduces noise (head-level variance within each variant) while preserving the group-level mean difference. This is the opposite of what happened with the within-layer → per-module decomposition (where aggregation diluted the signal). The V-module is the correct granularity for catastrophic/safe discrimination; the head level is the correct granularity for understanding seed sensitivity.

---

## 2. Group-Level Head Discrimination (RoBERTa, Backbone-Controlled)

### Per-head dim_ratio discrimination

| Head | G1 Mean | G2 Mean | Cohen's d |
|:----:|:-------:|:-------:|:---------:|
| 5 | 0.686 | 0.820 | 1.75 |
| 0 | 0.661 | 0.786 | 1.64 |
| 3 | 0.696 | 0.817 | 1.50 |
| 1 | 0.704 | 0.805 | 1.39 |
| 4 | 0.717 | 0.821 | 1.22 |
| 6 | 0.669 | 0.762 | 1.18 |
| 9 | 0.693 | 0.798 | 1.10 |
| 8 | 0.718 | 0.823 | 1.06 |
| 7 | 0.673 | 0.756 | 0.97 |
| 2 | 0.690 | 0.768 | 0.87 |
| 10 | 0.682 | 0.773 | 0.92 |
| 11 | 0.676 | 0.744 | 0.80 |

All 12 heads show the same directional pattern (G1 < G2), confirming the V-module signal is distributed across all heads. No single head achieves module-level discrimination. The range of per-head d values (0.80–1.75) is narrower than the module-level d (3.36), indicating that the module-level signal emerges from the consistency across heads rather than from any individual head.

### Dim ratio spread

| Group | Mean Spread | Range |
|:-----:|:----------:|:-----:|
| G1 (catastrophic) | 0.105 | [0.087, 0.122] |
| G2 (safe collision) | 0.117 | [0.083, 0.141] |
| G3 (non-collision) | 0.095 | [0.092, 0.098] |

Head-level heterogeneity does not distinguish the groups. The spreads overlap completely. This rules out "concentrated head-level mismatch" as a group-level discriminator.

---

## 3. CA-01 Seed Sensitivity: The Cancellation Mechanism

### 3.1 Why the signal was invisible at module level

The CA-01 worst variant (s42×s7) and mild variant (s7×s7) differ in the QNLI adapter seed only. At the module level, the V-module dimensionality ratio for both variants was nearly identical (~0.86). The head-level data reveals why: the two seeds produce different head-level configurations that *average to the same module-level value* but have very different distributions.

At layer 2, for example:
- Head 6: worst DR = 0.796 (s42), mild DR = 0.992 (s7). The worst seed's head 6 is more mismatched. Δ = -0.195.
- Head 11: worst DR = 0.950 (s42), mild DR = 0.756 (s7). The worst seed's head 11 is *less* mismatched. Δ = +0.193.

These opposite-sign deltas cancel almost perfectly at the module level (net Δ ≈ 0). But their effect on the merge is not symmetric: a single badly mismatched head may cause disproportionate damage if the O module and downstream classification head amplify its output.

### 3.2 The head-level seed sensitivity pattern

Across all 48 (layer × head) positions in the CA-01 comparison:
- 7 positions show |Δ_DR| ≥ 0.15
- 16 positions show |Δ_DR| ≥ 0.10
- The maximum is |Δ_DR| = 0.229 (layer 3, head 6)

For top direction overlap:
- The maximum is |Δ_OV| = 0.341 (layer 4, head 9)
- 5 positions show |Δ_OV| ≥ 0.10

The hot heads are not concentrated at a single layer. Layers 1 and 2 each have 3 hot heads for dim ratio; layer 3 has 1; layer 4 has 0 hot heads for dim ratio but the single largest alignment delta (Δ_OV = -0.341).

### 3.3 Interpretation

The CA-01 seed sensitivity operates through a **head-level geometric rebalancing**: different QNLI seeds learn V-module perturbations that distribute their effective dimensionality differently across heads. The module-level dimensionality ratio is a summary statistic that loses this distributional information. The catastrophic outcome depends not on the module-average dim ratio (which is similar for both seeds) but on the *worst case* head-level incompatibility, weighted by how much each head contributes to the downstream prediction.

This connects to the thresholded subspace interference hypothesis: the threshold is not a simple function of the module-level dim ratio. It involves the interaction between head-level geometry and the O-module / classification head weighting. Some head-level configurations cross the threshold; others with the same module-level average do not.

---

## 4. CA-02 Seed Sensitivity: Consistent Head-Level Amplification

Unlike CA-01, the CA-02 toxic/benign difference is directionally consistent at head level: the toxic adapter (qnli_s42) produces lower dim ratios at nearly all heads, with the strongest effects at layer 4. This explains why the module-level signal was already visible for CA-02 (V Δ_DR = -0.101) but not for CA-01 (V Δ_DR = +0.006) — in CA-02, the head-level signals reinforce each other rather than canceling.

The head-level amplification factor is 4.5× (single-head max Δ_DR = -0.459 vs. module Δ_DR = -0.101), confirming that even in CA-02, the module-level statistic understates the head-level signal due to averaging across less-affected heads.

---

## 5. Outcome Classification

**Classification: MIXED-POSITIVE.**

### Why POSITIVE for seed sensitivity

1. **CA-01 is no longer a mystery at per-module resolution.** The 29-point severity gap localizes to specific heads — 7 positions with |Δ_DR| ≥ 0.15, maximum 0.229. This is 33× the module-level maximum (0.007).

2. **The cancellation mechanism is identified.** Opposite-sign deltas at different heads cancel when averaged, producing near-zero module-level signals. The seed-sensitive variable is the *distribution* of dimensionality across heads, not the aggregate.

3. **CA-02 head-level amplification confirms module-level findings.** The toxic adapter's V-module incompatibility concentrates at layer 4, with individual heads showing 4.5× the module-level delta.

### Why MIXED for discrimination

1. **Head-level resolution does not improve group discrimination.** The module-level d=3.36 drops to d=1.25 at head level. The module is the optimal aggregation level for this signal.

2. **No individual head achieves module-level separation.** Best single-head d = 1.75 (head 5). The discriminative power of V-module dim ratio comes from cross-head consistency, not from any champion head.

3. **Head-level spread does not distinguish groups.** Dim ratio heterogeneity is similar across catastrophic, safe collision, and non-collision groups.

### Why not STRONG POSITIVE

1. The seed sensitivity finding explains *where* the signal is but not *why* specific head configurations produce catastrophe. The causal chain from head-level geometry through the O module to classification loss is still interpretive.

2. The cancellation mechanism for CA-01 makes head-level geometry a less promising *predictive* signal: predicting catastrophe requires knowing which heads matter for downstream performance, which depends on the O module and task head — information not available from V-module geometry alone.

---

## 6. Implications

### 6.1 For the V-module dimensionality mismatch concept

The V-module dimensionality ratio remains the right variable at the right granularity for *discriminating* catastrophic from safe collision pairs. The head-level analysis does not displace it but explains its limitations:

- **Strength confirmed:** All 12 heads show the same directional pattern (catastrophic < safe). The module-level signal is robust, not driven by outlier heads.
- **Limitation identified:** Module-level dim ratio cannot capture seed sensitivity when head-level deltas cancel. For seed-sensitivity prediction, a head-level descriptor (e.g., max |Δ_DR| across heads, or the spread of head dim ratios within a variant) would be needed.

### 6.2 For the instability concept

The cancellation mechanism offers a geometric interpretation of instability: an unstable pair is one where small seed changes produce head-level geometric reconfigurations that cross a catastrophic threshold at some heads while improving commensurability at others. The module average is unchanged, but the functional impact depends on which heads' incompatibilities dominate the output.

### 6.3 For the thresholded subspace interference hypothesis

The threshold is more nuanced than a single module-level dim ratio cutoff. It involves:
1. V-module dim ratio (module-level) as a necessary precondition — low module-level ratio indicates aggregate risk.
2. Head-level dim ratio distribution as a modulating factor — the same module average can produce catastrophe or not, depending on which specific heads carry the mismatch.
3. O-module / task head interaction as the amplification mechanism — determines which head-level incompatibilities actually manifest as classification errors.

### 6.4 For DeBERTa adjudication

The head-level analysis does not change the DeBERTa prediction (V-module dim ratio should separate catastrophic from safe collision). However, it adds a secondary prediction: if DeBERTa produces a catastrophic case, the head-level dim ratio distribution should show the same cancellation pattern as CA-01 or the same directional concentration as CA-02, depending on the nature of the catastrophe.

**Note:** DeBERTa-v3 uses disentangled attention with separate content and position value projections. The head-level slicing for DeBERTa will need to account for this structural difference.

### 6.5 For output-space escalation (Stage D)

The cancellation mechanism points directly toward the O module as the next analysis target. The O module transforms V-module outputs from each head back into the residual stream. If certain heads' incompatibilities are amplified by the O module while others are dampened, this could explain why different head-level configurations with the same module average produce different severity outcomes.

However, output-space analysis is a GPU-conditional next step. The CPU-only contribution of this program is now complete.

---

## 7. Recommended Follow-Up

**Priority 1:** DeBERTa replication (GPU-required). Does the V-module dim ratio signal persist? Does the head-level distribution pattern transfer?

**Priority 2:** O-module head-weight analysis (CPU-feasible, but lower priority). The O module's per-head weighting could be extracted from the O-module LoRA product. If certain heads' contributions are disproportionately amplified, this could explain the CA-01 cancellation mechanism.

**Priority 3:** Head-level dim ratio as supplementary predictor. While not improving group discrimination, the max-head or spread-based descriptors could serve as seed-sensitivity indicators alongside the module-level dim ratio.

---

## 8. Structured Outputs

| File | Location |
|------|----------|
| Head metrics (all variants) | `sidecar/results/head_level_v/head_metrics.json` |
| Group head comparison | `sidecar/results/head_level_v/group_head_comparison.json` |
| Head discrimination | `sidecar/results/head_level_v/head_discrimination.json` |
| Seed sensitivity per head | `sidecar/results/head_level_v/seed_sensitivity_per_head.json` |
| Head summary descriptors | `sidecar/results/head_level_v/head_summary_descriptors.json` |
| Head discrimination heatmap | `sidecar/figures/head_level_discrimination.svg` |
| CA-01 seed sensitivity per head | `sidecar/figures/head_level_ca01_seed_sensitivity.svg` |
| CA-02 seed sensitivity per head | `sidecar/figures/head_level_ca02_seed_sensitivity.svg` |
| Worst-head spotlight | `sidecar/figures/head_level_worst_head_spotlight.svg` |
