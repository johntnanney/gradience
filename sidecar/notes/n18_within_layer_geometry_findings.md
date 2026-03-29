# Note: Within-Layer Geometry Findings

## Metadata

- **Type:** findings
- **Date:** 2026-03-26
- **Related notes:** n17 (protocol), n16 (collision subset), n15 (per-layer findings)
- **Project:** Within-Layer Collision Program, Stage B

---

## Purpose

This note reports the results of the within-layer geometry pilot study. It answers the question: inside the collision subset, what within-layer geometric condition separates catastrophic from safe or merely unstable outcomes?

**Overall classification: MIXED, with important negative sub-results.**

The within-layer metrics reveal real structural differences, but these differences are primarily backbone-driven (DistilBERT vs. RoBERTa), not catastrophe-driven. The critical seed-sensitivity test is negative: within-layer metrics do not differentiate catastrophic from mild seed variants of the same pair. The findings constrain the mechanism but do not identify a clean within-layer predictor.

---

## 1. Group-Level Results (Critical Layers)

| Metric | Group 1 (Catastrophic) | Group 2 (Safe Collision) | Group 3 (Non-Collision) |
|--------|----------------------:|------------------------:|------------------------:|
| **Principal cosine** | 0.307 | 0.267 | 0.276 |
| **Top direction overlap** | 0.222 | 0.118 | 0.146 |
| **Dimensionality ratio** | 0.827 | 0.851 | 0.700 |
| **Directional conflict** | 0.402 | 0.428 | 0.473 |

### Reading the table

- **Principal cosine** (subspace alignment): Group 1 is slightly higher than Groups 2 and 3. The subspaces of catastrophic pairs overlap a bit more at critical layers — but the difference is modest and ranges overlap substantially.

- **Top direction overlap** (dominant direction match): Group 1 is notably higher (0.222 vs. 0.118). This is the sharpest group-level signal — catastrophic pairs' dominant singular directions align more at critical layers. But this signal is driven almost entirely by CA-01 (DistilBERT); CA-02 (RoBERTa) looks more like the safe controls.

- **Dimensionality ratio** (effective rank similarity): Group 3 is distinctly lower (0.700 vs. 0.827/0.851). Non-collision pairs have more dimensionality asymmetry. Groups 1 and 2 are similar — both collision groups use comparable effective dimensions.

- **Directional conflict** (opposing perturbations): The ordering is *reversed* from naive expectation. Group 1 has the *lowest* conflict (0.402), Group 3 the highest (0.473). Catastrophic pairs are not more "opposed" than safe pairs — if anything, they are more aligned in perturbation direction, which may be precisely the problem (see §3).

---

## 2. The Backbone Confound

The group-level results are confounded by backbone. Group 1 contains CA-01 (DistilBERT) and CA-02 (RoBERTa). Group 2 contains only RoBERTa cases. When CA-01 is separated from CA-02:

| Case | Principal cos | Top overlap | Dim ratio | Conflict |
|------|-------------:|------------:|----------:|---------:|
| CA-01 (DistilBERT) | 0.395 | 0.351 | 0.874 | 0.361 |
| CA-02 (RoBERTa) | 0.219 | 0.092 | 0.779 | 0.444 |
| SC-QMRB (RoBERTa) | 0.257 | 0.095 | 0.834 | 0.439 |
| SC-MSRB (RoBERTa) | 0.276 | 0.142 | 0.868 | 0.417 |

CA-01's within-layer profile is distinctly different from all RoBERTa cases. But CA-02's profile is *indistinguishable* from the safe RoBERTa controls (SC-QMRB and SC-MSRB). The CA-01 signal is therefore backbone-driven: DistilBERT's 6-layer architecture forces all adapters into tighter subspace overlap at every layer, producing higher principal cosines and top-direction overlap regardless of whether the outcome is catastrophic.

This is the study's most important finding: **the within-layer geometry difference between catastrophic and safe cases collapses when backbone is controlled.**

---

## 3. The Directional Conflict Reversal

Directional conflict is lowest in the catastrophic group (0.402) and highest in the non-collision group (0.473). This is the opposite of the naive prediction that catastrophic pairs would show more opposing perturbations.

The reversal has a coherent interpretation: catastrophic interference may arise not from *opposition* but from *similarity-without-identity*. When two adapters push in nearly the same direction within a shared subspace, a linear merge produces a perturbation that is quantitatively averaged but qualitatively satisfies neither task. The merge output is a smeared version of both tasks' learned features — too close to either to distinguish, too different from either to satisfy. Safe cross-task pairs, which push in more orthogonal directions, produce merge outputs that partially preserve each task's distinctive features because the perturbations occupy different subspace regions.

This interpretation connects to the victim pattern from n10: the "stronger" adapter (with more precise, concentrated features) is destroyed precisely because the merge averages its precise signal with a similar-but-not-identical signal from the other adapter. If the signals were more *opposed*, the merge might actually preserve more of each.

However, this interpretation is speculative and the conflict score ranges overlap substantially. The reversal is a trend, not a clean separation.

---

## 4. The Seed-Sensitivity Test (Negative)

### CA-01: Within-pair seed variants

| Variant | Worst Δ | Top overlap | Conflict | Principal cos |
|---------|--------:|------------:|---------:|--------------:|
| s42×s7 | 41.7% | 0.352 | 0.386 | 0.398 |
| s7×s7 | 12.7% | 0.369 | 0.385 | 0.404 |
| s42×s42 | — | 0.345 | 0.401 | 0.395 |
| s7×s42 | — | 0.339 | 0.271 | 0.383 |

The worst variant (s42×s7, Δ=41.7%) and the mildest variant (s7×s7, Δ=12.7%) have nearly identical within-layer metrics. Top overlap: 0.352 vs. 0.369. Conflict: 0.386 vs. 0.385. The 29-percentage-point gap in severity is not reflected in any of the four within-layer metrics.

**This is a clear negative result.** The within-layer geometry at the level of this analysis (concatenated Q/K/V/O matrices, SVD-based subspace comparison) does not explain seed sensitivity within CA-01. The seed-dependent variable, whatever it is, operates at a level of resolution below what these metrics capture.

### CA-02: Toxic vs. benign adapter

| Adapter | Top overlap | Conflict | Principal cos |
|---------|------------:|---------:|--------------:|
| qnli_s42 (toxic) | 0.043 | 0.494 | 0.171 |
| qnli_s7 (benign) | 0.141 | 0.395 | 0.267 |

Here a difference does appear: the toxic adapter (qnli_s42) shows *lower* top overlap and *higher* conflict than the benign adapter (qnli_s7). This is counter-intuitive under a simple "overlap causes catastrophe" model. The toxic adapter's subspace is actually more *orthogonal* to SST-2's subspace at critical layers — but with higher directional conflict in the shared portion.

One interpretation: qnli_s42 learned a perturbation that is partially orthogonal to SST-2 (low overlap) but pushes against it in the small shared subspace (high conflict). The linear merge then corrupts SST-2's features precisely in the narrow region where both adapters have stakes, while the orthogonal portion of qnli_s42's perturbation is preserved but irrelevant to SST-2's task.

This interpretation is tentative. The differences, while consistent across seed combinations with each adapter, are on a single backbone and might reflect adapter-level idiosyncrasies rather than a general mechanism.

---

## 5. The QNLI×MRPC Cross-Backbone Comparison

The highest-information comparison in the panel is QNLI×MRPC on DistilBERT (catastrophic) vs. RoBERTa (safe). The per-layer profiles (Figure 2) show:

- **Top direction overlap** is dramatically higher on DistilBERT (0.25–0.45 across layers) than on RoBERTa (0.03–0.15). This is the sharpest visual difference in the study.

- **Principal cosine** is higher on DistilBERT (0.20–0.45) than on RoBERTa (0.15–0.35), particularly in later layers.

- **Directional conflict** shows a different profile shape: on DistilBERT it is relatively uniform (0.35–0.50) while on RoBERTa it peaks in early layers and drops in later layers.

However, these differences cannot be attributed to catastrophe vs. safety because the backbones differ in layer count (6 vs. 12), hidden dimension structure, and pretraining. The DistilBERT adapter subspaces are more tightly packed by construction (fewer layers, same rank), not because the task pair is more dangerous.

---

## 6. Outcome Classification

**Classification: MIXED, trending toward NEGATIVE for the specific question asked.**

### What was found

1. **Top direction overlap separates groups at the aggregate level.** Group 1 mean (0.222) exceeds Group 2 (0.118). But this is driven by the CA-01/DistilBERT backbone effect.

2. **Directional conflict is reversed.** Catastrophic pairs show *lower* conflict, consistent with a "similar-but-not-identical perturbations" interpretation rather than an "opposing perturbations" interpretation.

3. **Dimensionality ratio separates collision from non-collision.** Groups 1 and 2 have similar effective rank ratios (~0.84); Group 3 is lower (0.70). This is a collision property, not a catastrophe property.

4. **Within-pair seed sensitivity is not explained.** CA-01's 29-point seed range is invisible in all four metrics. This is the clearest negative result.

5. **CA-02's toxic adapter shows a distinctive but counterintuitive pattern.** Lower overlap, higher conflict — the opposite of what a simple "overlap causes catastrophe" model predicts.

### Why MIXED and not NEGATIVE

The directional conflict reversal is genuinely informative. It suggests that the catastrophic mechanism involves *similarity*, not opposition — a substantive mechanistic constraint even though it doesn't yield a clean predictor. And the CA-02 toxic adapter pattern, while counterintuitive, does show a measurable difference that may reflect a real geometric distinction.

### Why MIXED and not POSITIVE

No metric cleanly separates catastrophic from safe cases when backbone is controlled. The seed-sensitivity test is negative. The most dramatic signal (top direction overlap on DistilBERT) is plausibly a backbone architecture effect, not a catastrophe marker.

---

## 7. Implications

### 7.1 Where the threshold is not

The within-layer analysis at the module-aggregate level (concatenated Q/K/V/O matrices) does not contain the threshold variable for catastrophic interference. This rules out a substantial class of mechanistic hypotheses — specifically, any hypothesis that locates the catastrophic trigger in the aggregate subspace relationship between two adapters' learned perturbations at matched layers.

### 7.2 Where the threshold might be

Three possibilities remain, in order of decreasing tractability:

1. **Per-module subspace geometry.** The current analysis concatenated all four attention modules per layer. The catastrophe-relevant interaction might be between specific modules (e.g., value matrices only, or query-key interaction). A per-module analysis is CPU-feasible with existing data.

2. **Higher-order subspace interaction.** The metrics examined pairwise subspace relationships (angles, overlap, conflict). The threshold may involve the interaction between multiple layers simultaneously — e.g., whether the same singular direction is disrupted across several consecutive layers, creating a cascading failure.

3. **Output-space / head-level incompatibility.** The threshold may not be in the body of the transformer at all but in the classification head interaction or in the effective decision boundary. This is a different kind of analysis (Workstream C).

### 7.3 The residual value of the collision model

The collision model from n15 remains valid as a *necessary precondition* for catastrophe. The within-layer analysis does not undermine it — it refines it by showing that the sufficient condition is not at the aggregate-subspace level. The full picture is now:

```
Collision (shared-layer loading) → necessary precondition
Within-layer aggregate subspace → NOT the threshold variable
Per-module or higher-order interaction → untested
```

### 7.4 DeBERTa implications

DeBERTa's disentangled attention may affect the per-module interaction in ways that standard attention does not. If the catastrophic threshold is per-module (e.g., in how the value matrices interact), DeBERTa's different attention decomposition could shift which pairs are vulnerable. This is testable once DeBERTa adapters are available.

---

## 8. Recommended Follow-Up

The highest-value CPU-only follow-up is a **per-module decomposition** of the within-layer metrics — computing the same four metrics separately for each attention component (Q, K, V, O) rather than concatenating them. If the catastrophic signal is concentrated in one module (e.g., value matrices), the concatenation may have diluted it.

This is a focused analysis that can reuse the existing infrastructure with minimal modification. It should be treated as a Stage B supplement, not a new stage.

---

## 9. Structured Outputs

| File | Location |
|------|----------|
| Geometry metrics (all variants) | `sidecar/results/within_layer_geometry/geometry_metrics.json` |
| Group comparison | `sidecar/results/within_layer_geometry/group_comparison.json` |
| Contrast panel summary | `sidecar/results/within_layer_geometry/contrast_panel.json` |
| Group comparison figure | `sidecar/figures/within_layer_group_comparison.svg` |
| CA-01 vs SC-QMRB figure | `sidecar/figures/within_layer_ca01_vs_scqmrb.svg` |
| CA-02 toxic vs benign figure | `sidecar/figures/within_layer_ca02_toxic_vs_benign.svg` |
