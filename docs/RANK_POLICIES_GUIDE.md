# Spectral Analysis Policies -- Interpretive Guide for Researchers

Gradience provides multiple **scientifically-grounded policies** for analyzing LoRA singular value spectra. Each policy embodies a different mathematical lens on the same spectral data, and the patterns of agreement and disagreement across policies reveal structural properties of learned adaptations that no single metric can capture.

## The Four Core Policies

### 1. Energy@90% (Cumulative Energy Retention) -- `energy@0.90`

**What it measures:** The minimal rank k at which 90% of the total spectral energy (sum of squared singular values) is retained.

**Formula:** Find minimal k where `sum(sigma_i^2) / sum(sigma_j^2) >= 0.90`

**What it reveals about training dynamics:**
Energy concentration reflects how the optimizer distributes learned information across spectral components. A layer that concentrates energy in few singular values has learned a low-dimensional transformation -- the optimizer found a compact solution. A layer with diffuse energy needed many spectral components, suggesting the learned mapping is intrinsically higher-dimensional or that training has not yet converged to a low-rank solution.

**Interpretation patterns:**
- **Conservative (high rank) when:** Gradual energy decay -- learned transformation uses many spectral directions, suggesting complex or distributed feature interactions
- **Aggressive (low rank) when:** Rapid energy concentration -- clear low-rank structure where the optimizer converged to a compact solution

**Assumptions:**
- 90% energy retention preserves most task-relevant information
- Energy distribution (squared singular values) correlates with functional importance
- Tail singular values contribute negligible signal to the learned transformation

**What variation across layers tells you:**
Layers with sharply different energy profiles are learning qualitatively different transformations. Attention layers often show faster energy concentration than feed-forward layers, reflecting the typically lower intrinsic dimensionality of attention mappings. Large variation in energy rank across layers of the same type may indicate uneven training dynamics or architectural bottlenecks.

**Research connections:** Energy thresholding connects to principal component analysis and the Eckart-Young-Mirsky theorem -- the truncated SVD is the optimal low-rank approximation in Frobenius norm. The choice of 90% is conventional but arbitrary; comparing results at 85%, 90%, and 95% thresholds can reveal how sharply the energy spectrum decays.

---

### 2. Knee Detection -- `knee`

**What it measures:** The elbow point in the singular value scree plot, identified via the Kneedle algorithm (maximum distance from the line connecting the first and last singular values).

**Method:** Kneedle algorithm -- finds the point of maximum curvature in the scree plot

**What it reveals about training dynamics:**
The presence or absence of a clear knee reveals whether the optimizer found a discrete separation between "important" and "unimportant" spectral components. A sharp knee suggests the learned transformation has a natural rank -- there is a clear boundary between signal and residual. A gradual curve without a distinct knee suggests a continuum of importance, where compression at any threshold involves a meaningful trade-off.

**Interpretation patterns:**
- **Conservative (high rank) when:** Gentle, gradual elbow -- no clear signal/residual boundary, suggesting the transformation has no natural low-rank decomposition
- **Aggressive (low rank) when:** Sharp elbow early in spectrum -- clear separation between dominant and residual components

**Assumptions:**
- Singular values exhibit elbow structure (not always the case)
- The point of maximum curvature indicates a meaningful signal/residual boundary
- The geometric heuristic aligns with task-relevant importance

**What variation across layers tells you:**
Layers with sharp knees at different ranks reveal heterogeneous intrinsic dimensionality across the network. If early layers show knees at rank 2-3 while later layers show knees at rank 6-8, this suggests that later layers learn higher-dimensional transformations -- potentially because they must represent more abstract, compositional features. Layers with no detectable knee may be worth special attention: they may be poorly trained, over-parameterized, or learning transformations that resist low-rank approximation.

**Research connections:** Scree plot analysis originates in factor analysis (Cattell, 1966). The Kneedle algorithm (Satopaa et al., 2011) formalizes the visual intuition of "finding the elbow." The existence of a clear knee connects to the effective rank literature and to questions about whether neural network layers learn naturally low-rank transformations.

---

### 3. Entropy Effective Rank -- `erank`

**What it measures:** The information-theoretic effective dimensionality of the singular value distribution, computed as the exponential of the Shannon entropy of the normalized singular values.

**Formula:** `exp(-sum(p_i * log(p_i)))` where `p_i = sigma_i / sum(sigma_j)` (Roy & Vetterli normalization)

**What it reveals about training dynamics:**
Entropy effective rank measures how "spread out" the spectral energy is across components, without imposing a binary signal/noise distinction. A high erank means the optimizer distributed information relatively evenly across many spectral directions -- the learned transformation uses its full capacity. A low erank means a few components dominate, suggesting the optimizer found a compact representation. Unlike energy thresholding, erank is sensitive to the shape of the entire distribution, not just where a cumulative threshold is crossed.

**Interpretation patterns:**
- **Conservative (high rank) when:** Nearly uniform singular values (high entropy) -- the transformation uses its full spectral capacity
- **Aggressive (low rank) when:** Dominated by few large singular values (low entropy) -- strongly peaked distribution

**Assumptions:**
- Spectral entropy reflects true effective dimensionality
- Uniform singular value distribution implies maximum rank is needed
- The Roy & Vetterli normalization (linear, not squared) is appropriate for the domain

**What variation across layers tells you:**
erank variation across layers maps the information-geometric landscape of the network. Layers with high erank are using their full parametric capacity and may resist compression. Layers with low erank have "wasted" capacity from a spectral perspective and are natural compression targets. A systematic pattern where erank increases or decreases through the network depth reveals how information dimensionality evolves across the architecture.

**Research connections:** Entropy effective rank (Roy & Vetterli, 2007) connects to information geometry and the concept of effective degrees of freedom. The measure is related to Renyi entropy of order 1 applied to the singular value distribution. In random matrix theory, the expected erank of a Wishart matrix provides a null model for comparison -- deviations from this null indicate structured (non-random) spectral content.

---

### 4. Optimal Hard Threshold (Experimental) -- `oht`

**What it measures:** The rank at which singular values cross the Gavish-Donoho threshold, separating statistically significant spectral components from those consistent with noise.

**Formula:** Gavish-Donoho threshold `tau = omega(beta) * median(sigma)` where `omega(beta) ~ 0.56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43`

**What it reveals about training dynamics:**
OHT applies a formal signal detection framework to the singular value spectrum. Singular values above the threshold are statistically distinguishable from what would arise in a random matrix of the same dimensions -- they represent learned structure. Singular values below the threshold are consistent with noise or negligible adaptation. This provides a principled (if assumption-heavy) answer to the question: "How many spectral components actually reflect learning?"

**Interpretation patterns:**
- **Conservative (high rank) when:** High estimated noise floor -- many components needed to exceed the noise threshold
- **Aggressive (low rank) when:** Clear signal/noise separation -- few dominant components well above the noise floor

**Assumptions:**
- The noise model (random matrix theory predictions) applies to LoRA residuals
- Signal singular values are "large" relative to the noise floor
- The matrix aspect ratio correctly parameterizes the threshold
- **Experimental caveat:** OHT was developed for denoising full-rank matrices, not for analyzing low-rank adaptations. The noise model may not hold for LoRA weight matrices.

**What variation across layers tells you:**
Layers where OHT detects very few signal components (rank 1-2) have adapted minimally in a spectral sense -- the optimizer made small, low-dimensional adjustments. Layers where OHT detects many signal components have undergone substantial, high-dimensional adaptation. If OHT consistently returns rank 1 across all layers, this may indicate that the overall adaptation magnitude is small relative to the matrix dimensions, and the noise model may be dominating.

**Research connections:** Based on Gavish & Donoho (2014), which established that 4/sqrt(3) is the asymptotically optimal threshold for singular value hard thresholding in the spiked covariance model. The adaptation to LoRA is non-trivial because LoRA weight matrices are not noisy observations of a low-rank signal in the classical sense -- they are the low-rank signal itself. Interpreting sub-threshold components as "noise" is a useful heuristic but should be validated empirically.

## Policy Disagreement as a Diagnostic

When policies disagree significantly, the pattern of disagreement is itself informative about the spectral structure of the adaptation. Rather than treating disagreement as a problem to resolve, researchers should interpret it as a window into training geometry.

### Example: energy@90=8, knee=3, erank=6, oht=2

**What each policy sees:**
- **energy@90=8:** Energy decays gradually -- 90% retention requires 8 components, indicating diffuse spectral content
- **oht=2:** Only 2 components exceed the noise threshold -- most spectral energy is in statistically indistinguishable tail components
- **knee=3:** The scree plot shows an elbow at rank 3 -- geometric structure suggests a natural decomposition point
- **erank=6:** Moderate entropy indicates the distribution is neither strongly peaked nor uniform

**What the disagreement reveals:**
This pattern -- high energy rank, low OHT rank, intermediate knee and erank -- is characteristic of adaptations with a "long tail": a few strong components carrying the core transformation, followed by many weak but non-negligible components. The gap between energy@90 and OHT quantifies the extent of this tail. The knee position suggests where qualitative structure changes. The erank captures the overall distributional shape.

**Research interpretation:** This adapter learned a predominantly low-dimensional transformation (OHT's perspective) but with distributed residual structure that collectively accounts for significant energy (energy@90's perspective). Compression to rank 2-3 would preserve the core transformation; compression to rank 6-8 would also preserve the tail structure. The empirical question -- which matters for task performance -- is exactly what Bench is designed to answer.

### Common Disagreement Patterns

| Pattern | Energy@90 | Knee | eRank | OHT | What it reveals about the adaptation |
|---------|-----------|------|-------|-----|--------------------------------------|
| **Long-tailed** | High (6-8) | Low (2-3) | Medium (4-6) | Low (1-2) | Core transformation is low-rank but residual tail carries non-trivial energy |
| **Uniform** | Medium (4-6) | Medium (4-6) | High (7-8) | Medium (3-4) | Distributed adaptation with no dominant spectral direction |
| **Clean low-rank** | Low (2-4) | Low (2-4) | Low (3-5) | Low (1-3) | Strongly structured adaptation -- all policies agree on low dimensionality |
| **Diffuse/noisy** | High (6-8) | High (5-8) | High (6-8) | High (4-6) | No clear spectral structure -- may indicate under-training or intrinsically high-dimensional task |

### When All Policies Agree

Strong agreement across policies is informative too. If all four policies converge on a similar rank, the spectral structure is unambiguous -- the adaptation has a clear intrinsic dimensionality that multiple mathematical frameworks detect independently. This is the strongest evidence for a "natural rank" of the adaptation.

### When Energy@90 and Knee Disagree

A large gap between energy@90 and knee rank is particularly diagnostic. It means the scree plot has a clear geometric break point (knee) but the energy beyond that break is substantial (energy@90). This reveals a spectrum with two qualitatively different regimes: a "head" of strong components and a "tail" of many weak-but-collectively-important components. The spectral gap between the knee and the energy threshold quantifies the information content of the tail, which is a direct measure of how much the adaptation resists low-rank compression.

## CLI Usage

### Default Policies (Recommended)
```bash
gradience audit --peft-dir ./adapter
# Uses: energy@0.90, knee, erank (balanced coverage)
```

### Custom Policy Selection
```bash
# Focus on energy distribution
gradience audit --peft-dir ./adapter --rank-policies energy@0.90,energy@0.95

# Signal detection focus
gradience audit --peft-dir ./adapter --rank-policies knee,oht

# Full spectral characterization
gradience audit --peft-dir ./adapter --rank-policies energy@0.90,knee,erank,oht
```

### CLI Output Example
```
Rank policy suggestions:
  Policy            Median   P90   Max   Don't Compress
  ----------------  ------  ----  ----  --------------
  energy@0.90            4     6     8         25%
  knee                   2     3     4          0%
  erank                  6     7     8         50%
  oht                    2     2     3          0%
```

**Reading the table:**
- **Median:** Central tendency of rank suggestions across layers -- the "typical" spectral dimensionality under this policy
- **P90:** Conservative (90th percentile) rank suggestion -- useful when worst-case preservation matters
- **Max:** Most conservative suggestion across all layers
- **Don't Compress:** Percentage of layers where the policy suggests keeping full rank -- high values indicate the policy sees most layers as not compressible

## Bench Integration

### Automatic Policy Testing
```yaml
# bench_config.yaml (policies auto-applied)
compression:
  allowed_ranks: [1, 2, 4, 8, 16]

# Creates variants: uniform_knee_p90, uniform_erank_p90, uniform_oht_p90
```

### Custom Policy Targeting
```yaml
compression:
  svd_variants:
    - name: conservative_choice
      rank_source: audit.rank_suggestions.energy_90.uniform_p90

    - name: aggressive_choice
      rank_source: audit.rank_suggestions.oht.uniform_median

    - name: balanced_choice
      rank_source: audit.rank_suggestions.knee.uniform_p90
```

### Experimental Workflow
1. **Audit:** Generate policy suggestions with `gradience audit` -- characterize the spectral landscape
2. **Hypothesize:** Each policy's rank suggestion is a testable hypothesis about what spectral content matters for task performance
3. **Test:** Bench evaluates all policy-derived configurations on real tasks
4. **Evidence:** Performance metrics validate or refute each policy's implicit assumptions
5. **Interpret:** The pattern of which policies predict task performance reveals what aspects of spectral structure are functionally relevant

## Failure Modes and Limitations

Understanding when each policy's assumptions break down is essential for responsible interpretation.

### Energy@90% limitations:
- Gradual singular value decay leads to overly conservative rank estimates -- the 90% threshold becomes arbitrary when there is no natural energy gap
- The threshold itself is conventional, not principled -- there is no theoretical reason why 90% (rather than 85% or 95%) should be the right cutoff for any particular task
- Energy (squared singular values) may not correlate with task-relevant importance -- a small singular value could correspond to a critical fine-grained feature

### Knee detection limitations:
- Fails when there is no clear elbow (uniform decay produces no geometric break point)
- Ambiguous when multiple elbows exist (complex spectral structure with several transition regimes)
- Sensitive to noise in singular values, which can produce spurious elbow detections

### Entropy effective rank limitations:
- The Roy & Vetterli normalization (linear singular values, not squared) may not match the appropriate notion of importance for all tasks
- Information-theoretic dimensionality is not the same as task-relevant dimensionality -- high entropy does not necessarily mean all spectral directions matter for performance
- Sensitive to small singular values that may be numerically insignificant but affect entropy

### OHT limitations:
- **Most significant:** LoRA weight matrices are not noisy observations of a low-rank signal -- they are learned low-rank adaptations. The random matrix theory noise model may not apply.
- Noise characteristics of LoRA residuals differ from the i.i.d. Gaussian assumptions underlying the Gavish-Donoho threshold
- When the signal/noise boundary is unclear, the threshold can be misleading

## Mathematical Details and Citations

### Energy Threshold
```
Cumulative energy: E(k) = sum_{i=1}^{k} sigma_i^2 / sum_{j=1}^{r} sigma_j^2
Rank selection: k = argmin{k : E(k) >= threshold}
```

### Optimal Hard Threshold
**Citation:** Gavish & Donoho (2014) "The Optimal Hard Threshold for Singular Values is 4/sqrt(3)"

```
Threshold: tau = omega(beta) * median(sigma)
Aspect ratio: beta = min(m,n) / max(m,n)
Cubic approximation: omega(beta) ~ 0.56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43
```

### Entropy Effective Rank
**Citation:** Roy & Vetterli (2007) "The effective rank: A measure of effective dimensionality"

```
Normalization: p_i = sigma_i / sum_j sigma_j
Shannon entropy: H = -sum_i p_i log p_i
Effective rank: erank = exp(H)
```

### Knee Detection
**Citation:** Satopaa et al. (2011) "Finding a 'Kneedle' in a Haystack"

```
Difference curve: D(i) = y(i) - x(i)
where y = normalized cumulative energy, x = normalized index
Knee point: k = argmax D(i)
```

## Interpreting Results: A Researcher's Checklist

1. **Start with disagreement.** The spread across policies is often more informative than any single policy's output. Large disagreement means the spectral structure is ambiguous and warrants careful empirical investigation.

2. **Examine layer-level variation.** If all layers show similar rank suggestions, the network has homogeneous spectral structure. If rank suggestions vary widely across layers, the architecture has heterogeneous intrinsic dimensionality -- and per-layer compression may be essential.

3. **Compare across training runs.** Rank suggestions that are stable across seeds reflect genuine spectral structure. Suggestions that vary across seeds may be artifacts of optimization stochasticity.

4. **Use Bench to arbitrate.** When spectral analysis alone cannot determine how much compression is tolerable, empirical task evaluation is the definitive test. The entire point of the Gradience methodology is to replace spectral intuition with empirical evidence.

5. **Document your reasoning.** Report which policies you examined, what disagreement patterns you observed, and why you chose the compression configuration you did. This makes your spectral analysis reproducible and your conclusions auditable.

---

## Policies as Scientific Hypotheses

Each policy embodies a **hypothesis about what matters** in singular value spectra:

- **Energy@90:** "Retaining 90% of spectral energy preserves task-relevant information"
- **Knee:** "The point of maximum curvature in the scree plot marks a natural signal boundary"
- **eRank:** "Information-theoretic entropy of the spectrum reflects functional dimensionality"
- **OHT:** "Random matrix theory provides a principled noise floor for LoRA spectra"

**No policy is universally correct.** The value of running multiple policies is not to find the "right" one, but to triangulate on the spectral structure from multiple mathematical perspectives. Disagreement between policies is not a failure -- it is data about the complexity of the learned representation.

**Use Bench to test these hypotheses against your actual tasks.**
