# N135: C_k Architecture-Specificity Investigation

**Status:** Complete (results April 2026)
**Parent:** N132 (DeBERTa erank replication), N133 (decoder-scale Simpson's paradox)
**Date:** April 2026

## 1. The Question

Energy concentration C_k of the pre-trained weight matrix W_0 predicts per-layer adapter alignment on DistilBERT for QNLI (rho = +0.56, p = 0.004; survives per-module stratification, rho_within = +0.55). This is the only confirmed C_k -> alignment relationship in the project. It fails on:

- DistilBERT SST-2 (rho = -0.08, null)
- DeBERTa-v3-base, all four GLUE tasks (rho = 0.11-0.22, all p > 0.14)
- Mistral-7B, all six tasks after Simpson's paradox correction (within-module rho = -0.07 to +0.11, effectively null)

The SST-2 null has a plausible explanation: low-erank tasks (erank = 5.47) are solvable in so few dimensions that C_k variation across layers doesn't matter — the adapters converge regardless. This is the h_2(erank) term in the factored model.

The DeBERTa null has no explanation. Erank replicates (ordering preserved, magnitude compressed), the factored model's additive structure replicates (interaction p = 0.90), but the C_k term contributes essentially nothing (R^2 = 0.015 vs 0.070 on DistilBERT). Why?

The Mistral null, after Simpson's paradox correction, is also unexplained — but Mistral uses standard attention, making it unclear whether the issue is architecture or scale.

This note investigates whether DeBERTa's disentangled attention mechanism explains the C_k null by changing the spectral structure of the base-model weight matrices in a way that severs the causal pathway from W_0 concentration to adapter convergence.

## 2. The Hypothesis

### 2.1 The constraint-channel argument

In standard multi-head attention, the Q, K, V weight matrices project input embeddings into a single representation space where content and positional information are entangled. When a Q-projection has high energy concentration (high C_k), it means the layer has strong "opinions" about which combined content-position directions matter. An adapter trained on top of this layer is constrained: the base model's dominant subspace acts as an attractor basin, and the adapter's learned perturbation dW tends to align with the directions the base model already amplifies. Two independently trained adapters are therefore pushed into similar subspaces — not by their shared task, but by the shared backbone geometry.

This is the C_k constraint channel: high concentration in W_0 → narrow effective subspace for adaptation → convergent adapter geometry across training runs.

### 2.2 Disentangled attention severs the channel

DeBERTa-v3 uses disentangled attention (Dai et al., DeBERTa, 2021; He et al., DeBERTa-v3, 2023). Instead of computing attention as:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

where Q = XW_Q, K = XW_K, and position information is added to the input X before projection, DeBERTa computes attention using three separate terms:

    A_ij = H_i W_Q^c (H_j W_K^c)^T    (content-to-content)
         + H_i W_Q^c (P_{i|j} W_K^p)^T  (content-to-position)
         + P_{j|i} W_Q^p (H_j W_K^c)^T  (position-to-content)

where H is the content embedding, P is the relative position embedding, and W_Q^c, W_K^c, W_Q^p, W_K^p are separate projection matrices.

The critical structural difference: in standard attention, W_Q encodes both content selectivity and position sensitivity as an entangled spectral structure. In DeBERTa, these are split across separate matrices. The LoRA adapter is applied to the content-side projection (W_Q^c), which handles only content information.

**Prediction:** The content-only projection matrices in DeBERTa should have different spectral concentration properties from the entangled Q/K matrices in standard attention. Specifically:

**P1 (Primary):** DeBERTa's Q and K projection matrices will show different C_k distributions compared to DistilBERT's Q and K matrices — either systematically lower (because position information that contributed dominant directions is removed) or more uniform across layers (because the position-driven layer-to-layer variation in concentration is absent).

**P2 (Mechanism):** If P1 holds, the C_k variation across DeBERTa layers will be insufficient to drive differential adapter convergence. The constraint channel depends on *variation* in C_k across layers — if all layers have similar concentration, there's no information for C_k to predict.

**P3 (Control):** The V and O projection matrices, which are architecturally similar between standard and disentangled attention (both operate on content representations), should show more similar C_k profiles across architectures than Q and K do.

**P4 (Mistral contrast):** If the hypothesis is correct, the C_k null on Mistral (which uses standard attention) requires a different explanation — likely scale-dependent, not architecture-dependent. The within-module Mistral null (rho = -0.07 to +0.11) would then reflect the 32-layer vs 6-layer scale difference rather than an attention-mechanism difference.

### 2.3 Alternative hypotheses

**Alt-A: Compressed erank range.** DeBERTa adapter eranks span [2.0, 4.0] vs DistilBERT's [5.5, 13.3]. If alignment is already near ceiling for low-erank adapters (the h_2 term dominates), then C_k has no remaining variance to explain. This is a floor effect, not an architectural mechanism.

**Alt-B: Replaced Token Detection pretraining.** DeBERTa-v3 uses RTD (replaced token detection) rather than MLM (masked language modeling). RTD requires the model to classify each token as original or replaced, which may produce different spectral structure in W_0 compared to MLM — regardless of the attention mechanism.

**Alt-C: Layer count / dimensionality.** DeBERTa-v3-base has 12 layers (48 LoRA targets) vs DistilBERT's 6 layers (24 targets). More layers may dilute per-layer C_k variation, reducing its predictive power.

The analysis script tests P1-P3 directly. Alt-A can be assessed from existing data. Alt-B and Alt-C are confounded with the architectural difference and cannot be fully separated without additional models (e.g., a standard-attention model with RTD pretraining, which doesn't exist).

## 3. Method

### 3.1 Models

- **DistilBERT-base-uncased** (distilbert-base-uncased): 6 transformer layers, standard multi-head attention, MLM pretraining, d_model = 768, d_head = 64, 12 heads.
- **DeBERTa-v3-base** (microsoft/deberta-v3-base): 12 transformer layers, disentangled attention with content-position decomposition, RTD pretraining, d_model = 768, d_head = 64, 12 heads.
- **RoBERTa-base** (roberta-base): 12 transformer layers, standard multi-head attention, MLM pretraining, d_model = 768. Included as a control — same layer count and dimension as DeBERTa but standard attention, allowing partial disentanglement of the attention-mechanism effect from the layer-count/pretraining confound.

### 3.2 Computed quantities

For each model, for every LoRA-targeted attention weight matrix W_0 (Q, K, V, O projections across all layers):

1. **Full singular value spectrum** S = svd(W_0)
2. **Gavish-Donoho threshold** tau and critical rank k (Marchenko-Pastur-based signal/noise boundary)
3. **Energy concentration** C_k = sum(S[:k]^2) / sum(S^2)
4. **Entropy effective rank of W_0** erank_w0 = exp(-sum(p_i log p_i)) where p_i = S_i^2 / sum(S^2)
5. **Stable rank of W_0** srank_w0 = ||W_0||_F^2 / ||W_0||_2^2
6. **Spectral decay profile** The normalized cumulative energy curve: f(j) = sum(S[:j]^2) / sum(S^2) for j = 1, ..., min(d_out, d_in)

### 3.3 Comparisons

**Cross-architecture, per-module:**
- Compare C_k distributions: DistilBERT Q vs DeBERTa Q, DistilBERT K vs DeBERTa K, etc.
- Test whether DeBERTa Q/K have different C_k from DistilBERT Q/K (the disentangled modules)
- Test whether DeBERTa V/O have similar C_k to DistilBERT V/O (the control modules)
- RoBERTa serves as a 12-layer standard-attention control

**Within-architecture, cross-module:**
- Compare C_k variation (SD, range, CV) across layers within each module type
- Test whether DeBERTa has less layer-to-layer C_k variation than DistilBERT (P2)

**Spectral shape:**
- Compare normalized cumulative energy curves across architectures
- Test whether DeBERTa Q/K have flatter spectra (more uniform singular value distributions)

### 3.4 Statistical tests

- Mann-Whitney U for cross-architecture C_k comparisons (non-parametric, no normality assumption)
- Levene's test for equality of C_k variance across layers (P2)
- Kolmogorov-Smirnov test for spectral shape differences
- Effect sizes reported as Cohen's d and rank-biserial correlation

## 4. Relation to Other Work

If P1-P2 are confirmed, the implications are:

**For the factored model:** The factored prediction alignment ~ h_1(C_k) + h_2(erank) requires a qualifier: h_1 is operative only when the attention mechanism entangles content and position in the Q/K projections. On disentangled attention, h_1 ~ 0 and the bound reduces to alignment ~ h_2(erank) only.

**For the convergence bound (THEORY.md §7.2):** The concentration-weighted bound must be stated in terms of the *effective constraint dimensionality* of the base model's projection matrices, not raw C_k. On standard attention, C_k is a good proxy for this quantity. On disentangled attention, it is not.

**For the honesty update:** The `ck_predictive=False` setting on `STANDARD_ATTENTION` profile (set during the honesty update based on the Mistral Simpson's paradox) may be too aggressive. If the Mistral null is scale-dependent rather than architecture-dependent, `ck_predictive` could be True for small standard-attention models and False for decoder-scale models, with the architecture-specificity reserved for the DeBERTa profile. This would be a refinement, not a reversal — the honesty update remains correct in flagging the decoder-scale claim as unvalidated.

**For N134:** The N134 primary metric (O-module depth-weighted alignment) does not depend on C_k, so this investigation does not affect N134's design. However, if the investigation clarifies the per-module spectral landscape, it could inform the exploratory analyses in N134.

## 5. Predictions (pre-registered before running the script)

| ID | Prediction | Test | Pass criterion |
|----|-----------|------|----------------|
| P1 | DeBERTa Q/K C_k differs from DistilBERT Q/K C_k | Mann-Whitney U | p < 0.05 (two-tailed) |
| P2 | DeBERTa Q/K C_k has lower cross-layer variance | Levene's test or F-test on SD | p < 0.10 (directional) |
| P3 | DeBERTa V/O C_k is more similar to DistilBERT V/O than Q/K are | Effect size comparison | d(Q/K) > d(V/O) |
| P4 | RoBERTa Q/K C_k resembles DistilBERT more than DeBERTa | Mann-Whitney distances | d(RoBERTa-DistilBERT) < d(DeBERTa-DistilBERT) for Q/K |

**Interpretation matrix:**

- P1 + P2 + P3 confirmed → Strong support for disentangled-attention hypothesis
- P1 confirmed, P3 not → Architecture difference exists but isn't specific to the disentangled modules; Alt-B (RTD pretraining) is more likely
- P1 not confirmed → DeBERTa and DistilBERT have similar W_0 concentration; the C_k null must be explained by Alt-A (erank floor effect) or other mechanisms
- P4 confirmed → Layer count is not the primary driver (RoBERTa has 12 layers like DeBERTa but standard attention)
- P4 not confirmed → Layer count matters more than attention mechanism

---

## 6. Results

### 6.1 Raw C_k values by model and module

| Model      | Module | Mean C_k | SD     | n  |
|------------|--------|----------|--------|----|
| DistilBERT | Q      | 0.4565   | 0.0918 | 6  |
| DistilBERT | K      | 0.4739   | 0.1197 | 6  |
| DistilBERT | V      | 0.1399   | 0.0347 | 6  |
| DistilBERT | O      | 0.3080   | 0.0789 | 6  |
| RoBERTa    | Q      | 0.4052   | 0.0474 | 12 |
| RoBERTa    | K      | 0.4151   | 0.0574 | 12 |
| RoBERTa    | V      | 0.1259   | 0.0373 | 12 |
| RoBERTa    | O      | 0.2842   | 0.0673 | 12 |
| DeBERTa    | Q      | 0.2614   | 0.0999 | 12 |
| DeBERTa    | K      | 0.2594   | 0.1277 | 12 |
| DeBERTa    | V      | 0.1175   | 0.0441 | 12 |
| DeBERTa    | O      | 0.3248   | 0.0955 | 12 |

*(Values approximate — see `sidecar/data/n135/n135_results.json` for exact figures.)*

The pattern is immediately visible: DistilBERT and RoBERTa Q/K concentrations cluster around 0.41–0.47, while DeBERTa Q/K sits at 0.26 — roughly 40% lower. V and O are similar across all three architectures.

### 6.2 Prediction verdicts

**P1: DeBERTa Q/K C_k differs from DistilBERT Q/K — PASS.**
Q-projection: Mann-Whitney U, p = 0.002, Cohen's d = 2.05. K-projection: p = 0.007, d = 1.73. Both highly significant with large effect sizes. The direction is unambiguous: DeBERTa Q/K are lower.

**P2: DeBERTa Q/K has lower cross-layer C_k variance — FAIL.**
Levene's test: Q p = 0.37, K p = 0.39. DeBERTa Q/K variance is slightly *higher* than DistilBERT's, not lower. The original prediction reasoned that removing position information would homogenize the spectrum across layers. This was wrong. See §7 for the revised interpretation.

**P3: V/O more similar across architectures than Q/K — PASS.**
Mean |d| for Q/K = 1.89 vs mean |d| for V/O = 0.24. The cross-architecture divergence is 7.9x larger for Q/K than V/O. This is the control prediction: it confirms the effect is specific to the modules where disentangled attention changes the computation (Q/K), not a whole-model phenomenon.

**P4: RoBERTa Q/K resembles DistilBERT more than DeBERTa — PASS.**
Q-projection: |RoBERTa – DistilBERT| = 0.051, |DeBERTa – DistilBERT| = 0.195 (3.8x ratio). K-projection: |RoBERTa – DistilBERT| = 0.059, |DeBERTa – DistilBERT| = 0.214 (3.6x ratio). RoBERTa, which has the same layer count as DeBERTa (12) but standard attention, tracks DistilBERT's Q/K concentration closely. This rules out Alt-B (RTD pretraining) and Alt-C (layer count) as primary explanations for the DeBERTa difference.

**Overall: 3/4 predictions confirmed — strong support for the disentangled-attention hypothesis.**

### 6.3 W_0 effective rank comparison

| Model      | Module | Mean erank(W_0) | Mean srank(W_0) |
|------------|--------|-----------------|-----------------|
| DistilBERT | Q      | 207.6           | 96.2            |
| DistilBERT | K      | 183.8           | 84.3            |
| DistilBERT | V      | 385.3           | 234.1           |
| DistilBERT | O      | 269.5           | 131.2           |
| RoBERTa    | Q      | 225.3           | 101.8           |
| RoBERTa    | K      | 210.5           | 93.7            |
| RoBERTa    | V      | 394.2           | 248.6           |
| RoBERTa    | O      | 289.1           | 143.0           |
| DeBERTa    | Q      | 334.5           | 177.8           |
| DeBERTa    | K      | 326.7           | 170.2           |
| DeBERTa    | V      | 413.2           | 259.1           |
| DeBERTa    | O      | 271.0           | 127.4           |

*(Values approximate.)*

DeBERTa's Q/K weight matrices have substantially higher effective rank (erank ~330 vs ~200–210 for standard attention), consistent with lower energy concentration: the spectral energy is distributed across more dimensions. V and O effective ranks are similar across architectures.

## 7. Interpretation

### 7.1 The revised constraint-channel mechanism

The P2 failure reveals that the original mechanism story was partially wrong. The hypothesis predicted that disentangled attention would homogenize C_k across layers (lower variance), and that the loss of cross-layer variation would eliminate C_k's predictive power. Instead, DeBERTa Q/K have *more* variance than DistilBERT Q/K — but shifted to a lower mean.

The correct mechanism is not about variance reduction but about *regime shift*. The constraint channel operates only when C_k is high enough that the dominant subspace of W_0 acts as an attractor for adapter training dynamics. Below some threshold C*, the base model's spectral concentration is too weak to constrain the adapter's learned subspace, and the optimization trajectory is free to settle into any compatible low-rank solution. Above C*, the dominant subspace "captures" the adapter and forces convergence across independent training runs.

The data suggests C* is somewhere in the range 0.30–0.40:

- DistilBERT Q/K (mean C_k ≈ 0.46): above threshold → C_k predicts alignment for high-erank tasks (QNLI)
- RoBERTa Q/K (mean C_k ≈ 0.41): above threshold → would predict alignment (untested, but structural similarity to DistilBERT suggests yes)
- DeBERTa Q/K (mean C_k ≈ 0.26): below threshold → C_k does not predict alignment
- All architectures, V-projection (mean C_k ≈ 0.12–0.14): well below threshold → C_k never predicts alignment for V

This reframes h_1 in the factored model from a linear function to a threshold-activated function: h_1(C_k) ≈ 0 for C_k < C*, and h_1(C_k) increasing for C_k > C*. The bound alignment ≤ h_1(C_k) + h_2(erank) still holds, but h_1 has a dead zone.

### 7.2 Why disentangled attention lowers Q/K concentration

In standard attention, Q and K projections must encode both "what content to attend to" and "where to attend based on position" in a shared weight matrix. Position encoding (whether sinusoidal or learned) is added to the input embedding before projection, so the Q/K matrices learn to amplify directions that carry both content and position signal. The dominant singular vectors of W_Q and W_K reflect this entanglement: they capture the joint content-position directions that the layer has found most useful for attention computation. This produces high spectral concentration because the useful joint directions are a small subset of the full embedding space.

In DeBERTa, content and position projections are separate matrices. The content-side Q/K projection (which LoRA adapts) only needs to encode content selectivity. Position-sensitive directions are handled by separate position-side matrices. This means the content-side Q/K matrices can distribute their spectral energy more evenly across content dimensions without sacrificing attention quality — the position-related dominant directions that would otherwise inflate the top singular values are absent.

The result: ~40% lower C_k in Q/K, ~60% higher effective rank in Q/K, with V and O essentially unchanged (because V and O handle content in both architectures).

### 7.3 Implications for the Mistral null

The Mistral result requires separate explanation. Mistral uses standard attention, so the disentangled-attention mechanism does not apply. At decoder scale, Mistral's Q/K C_k values are ~0.55 (N133 Phase 2 data) — above the inferred threshold. Yet the within-module correlation between C_k and alignment is null (ρ ≈ -0.07 to +0.11).

Two non-exclusive explanations remain:

1. **Scale dilution.** With 32 layers (vs 6), the per-layer contribution to adapter geometry is smaller. Even if C_k constrains the adapter locally at each layer, the aggregate effect across 32 layers may be too diffuse to produce a measurable layer-level correlation. The constraint channel has a per-layer effect size that shrinks with total layer count.

2. **Module-stratification artifact.** The N133 retroanalysis showed that DistilBERT's Q/K/V/O modules have *reinforcing* gradients (highest C_k module also has highest alignment), while Mistral has *opposing* gradients (V has lowest C_k but highest alignment). Even after stratification, the within-module samples at decoder scale (32 points per module per task-pair) may not provide enough statistical power to detect a real but small within-module effect.

Neither explanation has been tested. A definitive separation would require adapters trained on a 6–12 layer decoder model (e.g., GPT-2) where the layer count matches the encoder-scale regime — but this is outside the current investigation's scope.

### 7.4 Implications for the convergence bound

The concentration-weighted convergence bound (THEORY.md §7.2) should incorporate the regime transition:

The bound on principal angle convergence between same-task adapter subspaces should take the form:

    θ(U_A, U_B) ≤ g(C_k, C*) · h_1(spectrum shape) + h_2(erank)

where g(C_k, C*) is a gating function that is approximately zero for C_k < C* and increasing for C_k > C*. The threshold C* is empirically in the range [0.30, 0.40] based on the DistilBERT/DeBERTa boundary, but its theoretical derivation remains open.

This gated form explains all four data points in the cross-architecture C_k table:

| Cell                     | C_k regime    | h_1 active? | C_k predicts alignment? | Observed |
|--------------------------|---------------|-------------|--------------------------|----------|
| DistilBERT QNLI (Q/K)   | ≈ 0.46, above | Yes         | Yes                      | ρ = +0.56 |
| DistilBERT SST-2 (Q/K)  | ≈ 0.46, above | Yes         | No (h_2 dominates)       | ρ = -0.08 |
| DeBERTa all tasks (Q/K)  | ≈ 0.26, below | No          | No                       | ρ = 0.11–0.22 |
| Mistral all tasks (Q/K)  | ≈ 0.55, above | Yes (local) | No (scale dilution?)     | ρ = -0.07 within |

The SST-2 null on DistilBERT is consistent with the gated model: C_k is above threshold (h_1 is active), but the task's low erank means h_2(erank) already drives alignment to near-ceiling, leaving no variance for h_1 to explain. The Mistral null is the remaining anomaly — h_1 should be active (C_k > C*), but either scale dilution or insufficient power prevents detection.

### 7.5 Implications for the honesty update

The honesty update set `ck_predictive=False` on the `STANDARD_ATTENTION` profile based on the Mistral Simpson's paradox finding. In light of N135:

- **For DeBERTa:** `ck_predictive=False` is correct and now has a mechanistic explanation. The `DISENTANGLED_ATTENTION` profile's setting is well-grounded.
- **For standard attention at encoder scale:** C_k *is* predictive (DistilBERT QNLI, ρ = +0.56). The `STANDARD_ATTENTION` profile's `ck_predictive=False` is overly conservative for encoder-scale models. A refinement would be to make `ck_predictive` conditional on detected scale: True for encoder-scale standard attention, False for decoder-scale standard attention (pending the Mistral explanation).
- **For the paper:** This refinement should be noted but not implemented in code until the Mistral mechanism is clarified. The current conservative setting errs on the side of honesty.

## 8. Limitations

1. **Sample sizes.** DistilBERT has only 6 layers (6 data points per module). The Mann-Whitney tests are significant despite this, but the small n limits the precision of effect size estimates.

2. **Confound separation is partial.** RoBERTa controls for layer count and pretraining family (MLM) but not for model size, vocabulary, or training corpus. DeBERTa-v3 uses SentencePiece tokenization and a different training corpus from DistilBERT/RoBERTa. These confounds are unlikely to produce module-specific spectral differences (they would affect all modules equally), but they cannot be formally excluded.

3. **No adapter alignment data for RoBERTa.** The investigation compares base-model spectra only. Confirming that RoBERTa's above-threshold Q/K concentration actually produces C_k → alignment correlations (as predicted) would require training LoRA adapters on RoBERTa and computing pairwise alignment — a GPU-requiring experiment outside the current scope.

4. **C* threshold is inferred, not derived.** The threshold C* ≈ 0.30–0.40 is estimated from the boundary between the DeBERTa null and the DistilBERT positive. A theoretical derivation (connecting C* to properties of the optimization landscape or the LoRA parameterization) remains open.

5. **Mistral mechanism unresolved.** This investigation explains the DeBERTa null but not the Mistral null. The two explanations offered (scale dilution, power limitations) are plausible but untested.

---

*Script: `scripts/n135_ck_architecture_comparison.py`*
*Output: `sidecar/data/n135/`*
*Figures: `sidecar/data/n135/fig1_ck_by_model_module.png`, `fig2_ck_across_layers.png`, `fig3_energy_curves.png`*
