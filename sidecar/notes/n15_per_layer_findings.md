# Note: Per-Layer Structural Findings

## Metadata

- **Type:** findings
- **Date:** 2026-03-26
- **Related studies:** S01
- **Related notes:** n13 (artifact inventory), n14 (protocol), n12 (mining inventory), n06 (program statement)
- **Project:** Phase 3, Project G — Per-Layer Structural Analysis

---

## Purpose

This note reports the results of the per-layer structural analysis of saved LoRA adapter weights. It answers the study question: *Do catastrophic cross-task anchors show stronger per-layer concentration or sharper per-layer divergence than stable contrast pairs?*

**Overall classification: MIXED.** The per-layer structure contains signal, but the signal runs in a direction that reframes the initial hypothesis. Catastrophic pairs do not show *higher* layer-level divergence than stable pairs — they show *lower* divergence and *higher* alignment. The data is more consistent with a collision mechanism than a competition mechanism.

---

## 1. Group-Level Results

| Metric | Group A (Catastrophic) | Group B (Same-Task) | Group C (Stable Cross-Task) |
|--------|----------------------:|--------------------:|---------------------------:|
| **Pair divergence (JS)** | 0.0065 | 0.0008 | 0.0137 |
| **Alignment (Spearman ρ)** | 0.757 | 0.900 | 0.618 |
| **Concentration (Gini)** | 0.152 | 0.152 | 0.115 |

(All values are group means. Full ranges and per-variant data in the structured outputs.)

### Reading the table

- **Pair divergence** measures how differently two adapters distribute their norm mass across layers. Higher = more different layer profiles. Group A is *lower* than Group C — catastrophic pairs' adapters have more *similar* layer profiles than stable pairs.

- **Alignment** measures whether both adapters load the same layers. Higher = same layers. Group A is *higher* than Group C — catastrophic pairs' adapters converge on the same layers more than stable pairs do.

- **Concentration** measures how peaked each adapter's layer distribution is. Higher = more concentrated. Group A is moderately higher than Group C, but the ranges overlap substantially.

---

## 2. Interpretation

### 2.1 The collision pattern

The central finding is that catastrophic pairs show *lower* pair divergence and *higher* alignment than stable cross-task pairs. This is the opposite of a "competition" model where adapters pull in different layer-directions and the merge struggles to reconcile them. Instead, the data suggests a **collision** model:

- Catastrophic pairs involve adapters that concentrate their adaptation in the *same* layers.
- When these similarly-distributed adaptations are linearly merged, the merge creates destructive interference precisely in the high-norm layers where both adapters have made their largest perturbations.
- Stable cross-task pairs avoid this because their adapters distribute differently across layers — the merge has room to accommodate both.

This is conceptually intuitive: if two adapters modify the same layers in different ways, a 50/50 linear merge produces a compromise that satisfies neither task. If they modify different layers, the merge can preserve both.

### 2.2 Same-task controls behave as expected

Group B (same-task) shows the lowest pair divergence (0.0008) and highest alignment (0.900). This is correct: two seed variants of the same task should learn very similar per-layer profiles. The small divergence is noise. This validates the metric — same-task pairs should look maximally similar, and they do.

### 2.3 The collision model connects to the alignment proxy

The alignment proxy (Spearman ρ) provides the most interpretable signal. The ordering is:

```
Same-task controls (0.900) > Catastrophic anchors (0.757) > Stable cross-task (0.618)
```

Catastrophic pairs sit *between* same-task and stable cross-task on the alignment axis. This suggests they occupy a dangerous middle ground: similar enough in layer profile to interfere (unlike stable cross-task pairs, which diverge and coexist) but different enough in task content to produce conflicting gradients (unlike same-task pairs, which reinforce each other).

### 2.4 Concentration is a weaker signal

The concentration index (Gini) shows Group A slightly higher than Group C (0.152 vs. 0.115), but with substantial range overlap (Group A: [0.102, 0.218]; Group C: [0.052, 0.251]). Concentration alone does not cleanly separate the groups. The signal is in the *relationship between* adapters (divergence, alignment), not in single-adapter properties (concentration).

---

## 3. Within-Group Details

### 3.1 CA-01 (QNLI × MRPC on DistilBERT)

| Variant | JS divergence | Alignment | Note |
|---------|-------------:|----------:|------|
| s42×s42 | 0.0008 | 0.943 | Near-identical layer profiles |
| s42×s7 | 0.0012 | 0.829 | Worst catastrophic variant (Δ=41.7%) |
| s7×s42 | 0.0016 | 0.771 | |
| s7×s7 | 0.0014 | 0.886 | Best catastrophic variant (Δ=12.7%) |

All CA-01 variants show very low JS divergence and very high alignment. On the 6-layer DistilBERT backbone, QNLI and MRPC adapters learn almost identical per-layer profiles — norm mass rises toward the later layers for both. This is the collision pattern at its clearest.

**Seed sensitivity note:** The worst variant (s42×s7, Δ=41.7%) and best variant (s7×s7, Δ=12.7%) have similar alignment values (0.829 vs. 0.886). Per-layer alignment does not explain the 29-point seed range within this pair. The catastrophe's seed sensitivity must originate in *within-layer* subspace geometry, not in layer-level norm distribution. This constrains the mechanism: the per-layer profile creates the *precondition* for collision, but the specific outcome depends on finer-grained structure.

### 3.2 CA-02 (QNLI × SST-2 on RoBERTa)

| Variant | JS divergence | Alignment | Note |
|---------|-------------:|----------:|------|
| s42×s42 | 0.011 | 0.622 | |
| s42×s7 | 0.008 | 0.636 | Worst catastrophic variant (Δ=27.2%) |
| s7×s42 | 0.016 | 0.678 | |
| s7×s7 | 0.013 | 0.692 | |

CA-02 shows higher divergence and lower alignment than CA-01. On the 12-layer RoBERTa backbone, QNLI and SST-2 have somewhat different per-layer profiles. The alignment values (~0.65) are lower than CA-01 (~0.86) but still higher than the Group C mean (0.618). This is a weaker collision signal — present but not as sharp as on DistilBERT.

The difference between CA-01 and CA-02 is consistent with the backbone-local interpretation in n11: on the shallower DistilBERT, adapters are forced into the same layers (fewer options), producing tighter collision. On the deeper RoBERTa, there is more room for layer-level differentiation, but the collision mechanism still operates.

### 3.3 Stable cross-task contrasts (Group C)

The SC-03 and SC-04 cases (RoBERTa) show the highest pair divergence in the dataset (JS = 0.029–0.032 for SC-03, the RTE × MRPC pair). This pair involves RTE adapters, which have distinctively low concentration (Gini ~0.055) — nearly uniform across layers. MRPC adapters, by contrast, have high concentration (Gini ~0.25). The divergence is driven by this asymmetry: RTE spreads evenly, MRPC peaks in specific layers, and the merge can accommodate both because they don't collide.

---

## 4. Norm Mass Profiles

The norm mass profile figure reveals backbone-level regularities:

**DistilBERT:** All adapters show a rising profile — later layers carry more norm mass. The profiles are tightly bunched, explaining why all DistilBERT pairs have low JS divergence. The differentiation is in *degree* of rise, not in *direction*. SST-2 and QNLI show the steepest rises; RTE is the flattest.

**RoBERTa:** Profiles are more varied. There is a characteristic "U-shape" for some adapters (high norms in early and late layers), while others peak in middle layers. MRPC adapters show the most peaked profiles (layers 8–10). RTE adapters are nearly flat. QNLI and SST-2 are intermediate. The greater profile diversity on RoBERTa explains why Group C divergence is higher on RoBERTa than on DistilBERT.

---

## 5. Outcome Classification

**Classification: MIXED.**

The per-layer analysis provides informative results, but they do not cleanly separate Group A from Group C on any single metric.

### What was found

1. **A collision pattern, not a competition pattern.** Catastrophic pairs show *lower* divergence and *higher* alignment than stable pairs. The mechanism appears to involve same-layer interference, not cross-layer mismatch.

2. **The alignment proxy is the most discriminating metric.** Group A mean (0.757) > Group C mean (0.618). The ordering is consistent with the collision interpretation.

3. **Concentration is a weak signal.** It trends in the expected direction but does not separate groups.

4. **Per-layer structure does not explain seed sensitivity.** Within CA-01, the 29-point seed range is not visible in per-layer metrics. The seed-dependent variable is sub-layer.

### Why MIXED and not POSITIVE

The group ranges overlap. The highest-alignment Group C variant (SC-02 s42×s42, ρ=0.886) exceeds the lowest-alignment Group A variant (CA-02 s42×s42, ρ=0.622). No metric achieves the "worst Group A exceeds best Group C" criterion defined in the protocol for a positive outcome.

### Why MIXED and not NEGATIVE

There are clear directional trends. Group A is consistently higher on alignment and lower on divergence than Group C across all pairings. The collision pattern is interpretively coherent and consistent with the thresholded subspace interference hypothesis — it provides a geometric picture of *why* certain pairs are vulnerable.

---

## 6. Implications for the Instability Program

### 6.1 Mechanistic constraint

The collision pattern constrains the thresholded subspace interference hypothesis: the "specific geometric conditions" referenced in n06 §6 include, at minimum, high per-layer alignment between adapters. Adapters that load the same layers heavily are at higher risk of destructive interference when merged.

However, per-layer alignment is necessary but not sufficient. The seed sensitivity within CA-01 demonstrates that adapters with similar layer profiles can still produce wildly different outcomes depending on their within-layer subspace geometry. The per-layer analysis identifies a precondition for catastrophe, not a deterministic predictor.

### 6.2 Structural predictor implications

The instability program's second criterion for promotion (n06 §4, criterion 2) requires a structural predictor computable from adapter weights. Per-layer alignment is a candidate signal — it can be computed from source adapters without merge evaluation. But it is not sufficient as a standalone predictor (overlap with Group C prevents reliable classification). A useful structural predictor would likely combine per-layer alignment with a within-layer subspace angle metric (the natural follow-up to this analysis).

### 6.3 DeBERTa expectations

The collision pattern generates a DeBERTa expectation: if DeBERTa's disentangled attention causes adapters to distribute their norm mass differently across layers (reducing alignment), catastrophic outcomes may be less frequent on DeBERTa than on standard-attention backbones. This is an additional testable prediction beyond the three pre-registered in n07.

---

## 7. Recommended Follow-Up

The highest-value next step is **within-layer subspace angle analysis** — computing principal angles between adapter subspaces at each layer, specifically for the high-norm layers identified by the norm mass profiles. This would test whether the collision pattern extends into subspace geometry: do catastrophic pairs' adapters not only load the same layers but also occupy overlapping subspaces within those layers?

This analysis requires SVD computation on the LoRA products at each layer, which is CPU-feasible but more computationally involved than the norm-based analysis performed here. It should be treated as a separate protocol note if pursued.

---

## 8. Structured Outputs

| File | Location |
|------|----------|
| Per-layer norms | `sidecar/results/per_layer_analysis/per_layer_norms.json` |
| Per-layer metrics | `sidecar/results/per_layer_analysis/per_layer_metrics.json` |
| Group comparison | `sidecar/results/per_layer_analysis/group_comparison.json` |
| Artifact inventory | `sidecar/results/per_layer_analysis/artifact_inventory.json` |
| Norm mass profiles figure | `sidecar/figures/norm_mass_profiles.svg` |
| Group comparison figure | `sidecar/figures/group_comparison.svg` |
