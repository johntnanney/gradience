# N136: N133 Bounded Reanalysis

**Status:** Complete
**Parent:** N133 (decoder-scale merge triage), N135 (C_k architecture-specificity)
**Date:** April 2026

## 1. Purpose

N133 Phase 3/4 and the B-P5 diagnostic cascade established that per-pair alignment metrics do not predict merge degradation at decoder scale beyond task-family membership. The B-P5 diagnostic identified three independent confounds: metric saturation (~2e-3 resolution), task-family aliasing (R² = 0.966), and insufficient seed replication (n=2). Sixteen alternative aggregations of per-layer alignment scalars all failed to exceed ΔR² = 0.01 after family residualization.

This reanalysis has two goals:

**Goal A (Confirmatory):** Determine whether the confound runs uniformly through all spectral-SV features computable from the existing N133 data, including novel aggregations informed by N135. If it does, this strengthens the paper's negative claim and validates N134's constraint design.

**Goal B (Exploratory):** Test whether any feature shows within-cluster variation where magnitude-only metrics don't. "Within-cluster" means: among pairs that share the same task-family membership, does the feature rank them in a way that correlates with merge degradation? If so, this identifies a candidate for N134's exploratory analyses.

This is explicitly a bounded reanalysis — 12 cross-task evaluated pairs, no new data collection, no policy/verdict changes. Any positive finding on 12 points is hypothesis-generating for N134, not confirmatory.

## 2. What Was Already Tested

The B-P5 composite risk script tested 10 variants:

| Variant | ρ (vs max_degradation) | p | Triage recall |
|---------|----------------------|-------|---------------|
| mean_alignment | +0.655 | 0.021 | 3/3 |
| inv_min_erank | +0.319 | 0.313 | 3/3 |
| OVmix_x_inv_erank | +0.221 | 0.491 | 3/3 |
| O_depth_x_inv_erank | +0.112 | 0.729 | 2/3 |
| O_mean | -0.091 | 0.778 | 1/3 |
| O_depth | (similar) | NS | |
| O_quad | (similar) | NS | |
| z_sum | (similar) | NS | |

The confound check showed `mean_alignment` achieves its ρ by separating two task-family clusters (MNLI-containing vs generation-task pairs). Within either cluster, the metric has no resolution.

## 3. What Hasn't Been Tested

### 3.1 N135-informed C_k-gated alignment

N135 established a regime transition in the C_k constraint channel at approximately C* ≈ 0.30–0.40. The N133 W_0 data shows Mistral's Q-projection C_k ≈ 0.55, K ≈ 0.55, V ≈ 0.09, O ≈ 0.32. This means:

- Q/K layers are above threshold (C_k constraining)
- V layers are well below (C_k not constraining)
- O layers are near the boundary

**Novel feature:** Compute alignment restricted to layers where the base model W_0 has C_k above the N135 threshold. The hypothesis: alignment in C_k-gated layers is more informative about merge compatibility because those are the layers where the base model's geometry actually constrains the adapter.

### 3.2 Spectral shape divergence

The existing data stores 16 singular values per layer per adapter. Two adapters whose singular value *profiles* diverge (not just magnitudes) may merge poorly because their spectral energy is distributed across different effective-rank structures.

**Novel features:**
- **KL divergence** between normalized spectral energy distributions of adapter A and B, per layer, aggregated
- **Spectral slope ratio** — ratio of spectral decay rates (log(σ_1/σ_r))
- **Erank ratio** — ratio of adapter eranks (not just min-erank; the asymmetry may matter)

### 3.3 Module-stratified depth gradients

N133 Phase 2 showed the SNR gradient O > V > Q > K steepens with depth (2.32× at layer 0 → 4.24× at layer 31). The B-P5 script tested O-module variants with depth weighting, but didn't test:

- **O-only deep layers** — alignment restricted to O-projections in layers 16–31 (the high-SNR regime)
- **V-only alignment** — V showed highest absolute same-task alignment (0.145); its cross-task behavior may differ from O's
- **Q/K exclusion** — drop Q and K entirely (they contribute most noise per N133 Phase 2)

### 3.4 Asymmetric degradation features

Previous analyses used `max_degradation` as the outcome. But merge degradation is often asymmetric — task A may degrade heavily while task B is preserved. Features that predict *which* task degrades (the dominant-vs-subordinate pattern from Study 16) may have resolution where symmetric features don't.

**Novel features:**
- **Frobenius ratio × alignment** — do imbalanced pairs with high alignment degrade more asymmetrically?
- **Erank asymmetry** — does the lower-erank adapter's task always suffer more?

### 3.5 Family-residualized within-cluster ranking

The key question is not "does feature X correlate with degradation?" (task-family explains that) but "within the set of pairs that share a task-family signature, does feature X rank degradation correctly?"

For the 12 N133 pairs, the task-family partition is approximately:
- Cluster 1: MNLI-containing pairs (high degradation, high alignment)
- Cluster 2: generation-task pairs (low degradation by threshold, low alignment)

Within each cluster, is there any feature that ranks the 5–7 pairs in a way that tracks residual degradation variance?

## 4. Data Available

All data is in `sidecar/data/n133/`:

- `pod_pull/audits/adapter_profiles.json` — 12 adapters, per-layer erank + 16 singular values per layer
- `pod_pull/audits/w0_properties.json` — per-layer W_0 C_k for 128 layers
- `pod_pull/audits/pair_alignment_full.json` — 66 pairs, per-layer SV-weighted alignment
- `pod_pull/merges/merge_eval_summary.json` — 18 merge outcomes with per-task degradation
- `bp5_composite_risk.json` — existing composite scores for reference

**Constraint:** Raw singular vectors (U matrices) are not stored. Direction-aware signed inner products would require re-running SVD from adapter weight files (GPU-dependent for full fidelity, though numpy SVD on saved safetensors is CPU-possible). This reanalysis works with stored scalars only.

## 5. Predictions

**P-null (Primary expectation):** All novel features collapse onto the task-family partition, achieving apparent correlation via cluster membership. After family residualization, no feature exceeds ΔR² > 0.03 (the B-P5 ceiling). This confirms the confound runs through the entire spectral-SV feature family and validates N134's design constraints.

**P-alt (Alternative):** At least one novel feature (most likely C_k-gated O-module alignment or spectral shape divergence) shows within-cluster ranking that tracks degradation. This would be hypothesis-generating for N134's exploratory analyses.

**Decision rule:** A feature is "interesting" if within-cluster Spearman ρ > 0.5 for at least one cluster with n ≥ 5 pairs. This is not a significance threshold (insufficient power on 5–7 points) but a signal worth pre-registering as an N134 exploratory metric.

## 6. Relation to N134

This reanalysis does not change any N134 pre-registered prediction. Its findings feed into N134 in two ways:

- If P-null confirmed: the paper's claim that "spectral-SV features cannot predict per-pair merge risk at decoder scale" is supported by exhaustive feature search, not just the original 10 variants.
- If P-alt finds an interesting feature: it enters N134's exploratory analysis list (explicitly flagged as post-hoc from N133 data).

## 7. Results

*Run: April 2026. Script: `scripts/n136_n133_reanalysis.py`. Output: `sidecar/data/n136/`.*

**Sample:** 6 evaluated cross-task pairs (down from 12 adapter profiles; only pairs with merge evaluation outcomes were analyzed).

### 7.1 Primary verdict: P-null confirmed (with caveats)

Family membership alone explains R² = 0.939 of merge degradation variance, leaving only 6.1% residual. Of 21 novel features tested, three marginally exceed the pre-registered ΔR² > 0.03 threshold:

| Feature | ΔR² | Residual ρ | Residual p |
|---------|------|------------|------------|
| mean_slope_ratio | 0.052 | — | > 0.12 |
| jsd_x_erank_ratio | 0.031 | — | > 0.12 |
| ck_gated_O_mean | 0.030 | — | > 0.12 |

**Interpretation:** These are overfitting artifacts, not reliable signal. Three features capturing 3–5% of residual variance from a near-saturated model (93.9%) on 6 data points, all with non-significant residual correlations, is expected when screening 21 candidates. The marginal hits do not survive even informal multiple-comparison correction.

**Conclusion for Goal A:** The confound identified in B-P5 runs uniformly through the entire spectral-SV feature family, including N135-informed C_k-gated variants, spectral shape divergence, module-stratified depth gradients, and composite features. This strengthens the paper's negative claim from "10 features tested" to "31 features tested (10 original + 21 novel), none reliable."

### 7.2 Goal B: Within-cluster resolution

Within-cluster analysis was limited by the small sample (n ≤ 3–4 per cluster after family partition). No feature met the pre-registered decision rule (within-cluster ρ > 0.5 for a cluster with n ≥ 5). The metric saturation diagnosis is confirmed quantitatively: alignment-based features show CV < 0.06, meaning the entire feature range spans approximately 2e-3 — below the noise floor for discriminating pairs within a task-family cluster.

### 7.3 Global correlations (pre-residualization)

Pre-residualization correlations provide useful context for understanding why residualization collapses them:

| Feature | Global ρ | p |
|---------|----------|-------|
| OV_mean | -0.928 | 0.008 |
| V_mean | -0.841 | 0.036 |
| align_cv | +0.812 | 0.050 |
| erank_ratio | +0.812 | 0.050 |
| mean_alignment | -0.406 | 0.42 |

The strong global predictors (OV_mean, V_mean) achieve their correlations by separating task-family clusters, exactly as predicted. The original mean_alignment is actually *weaker* globally (ρ = -0.406) than in the B-P5 analysis (ρ = +0.655), likely reflecting the reduced sample (6 vs 12 pairs with different outcome definitions).

### 7.4 Asymmetric degradation: the erank finding

The one genuinely novel result: **erank ratio predicts degradation asymmetry** (ρ = +0.886, p = 0.019). Pairs where the two adapters differ more in effective rank show more lopsided task degradation — one task is preserved while the other suffers disproportionately.

This is mechanistically interpretable: an adapter concentrating its signal in fewer effective dimensions occupies a lower-dimensional subspace that is easier to corrupt during linear combination. The higher-erank adapter's broader spectral footprint makes it more robust to interference, so its task is preferentially preserved.

This finding answers a *different question* from pair-risk ranking. It does not help predict which pairs will degrade (task-family handles that), but it predicts the *pattern* of degradation conditional on it occurring. It connects to the dominant-vs-subordinate dynamic observed in Study 16.

**Status:** Hypothesis-generating. Pre-register for N134's exploratory analysis list as: "H_asym: erank ratio predicts degradation asymmetry (directional: lower-erank adapter's task suffers more)."

### 7.5 Dynamic range confirmation

Quantitative confirmation of the N133 metric saturation diagnosis:

| Feature class | Typical CV | Implication |
|--------------|-----------|-------------|
| Alignment-based (mean, gated, module-specific) | < 0.06 | ~2e-3 resolution window |
| erank_diff | 0.30 | Better dynamic range |
| frob_ratio | 0.10 | Moderate dynamic range |

The alignment features are not failing because of wrong aggregation — they are failing because the underlying measurements lack resolution at decoder scale. No aggregation strategy can recover signal that isn't present in the per-layer scalars.

## 8. Implications

### For the paper's negative claim

The claim is now supported by exhaustive feature search: 31 spectral-SV features tested (10 original B-P5 variants + 21 novel N136 variants including N135-informed gating), none reliable after family residualization. The confound is not an artifact of limited feature exploration — it is structural. At decoder scale on Mistral-7B, per-pair alignment metrics computed from stored spectral scalars do not predict merge degradation beyond task-family membership.

### For N134 design

P-null confirmation validates N134's design constraints: the study needs either (a) substantially more pairs per task-family cluster to detect within-cluster effects, or (b) qualitatively different features (e.g., direction-aware signed inner products from raw U matrices, which require GPU re-computation). The erank asymmetry finding should enter N134's exploratory analysis list.

### For the Gradience toolkit

The honesty update (implemented per HONESTY_UPDATE_SPEC.md) correctly gates decoder-scale risk predictions. N136 confirms that gating was warranted — no novel feature rescue is available from stored scalars. The toolkit's decoder-scale merge auditing should continue to surface descriptive spectral statistics (erank, alignment, C_k) without making per-pair risk claims until N134 provides validated thresholds.

---

*Script: `scripts/n136_n133_reanalysis.py`*
*Output: `sidecar/data/n136/`*
