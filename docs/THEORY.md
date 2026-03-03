# Theoretical Foundations

This document outlines the mathematical and conceptual framework that Gradience
draws on, and identifies open questions that its measurements can help investigate.

---

## 1. Why SVD of Weight Updates Reveals Training Structure

A LoRA adapter parameterizes a weight update as $\Delta W = BA$, where
$B \in \mathbb{R}^{d_{\text{out}} \times r}$ and $A \in \mathbb{R}^{r \times d_{\text{in}}}$.
The singular value decomposition of $\Delta W$ decomposes the update into an
ordered set of orthogonal rank-1 contributions:

$$\Delta W = \sum_{i=1}^{r} \sigma_i \, u_i v_i^\top$$

Each $\sigma_i$ quantifies the magnitude of the $i$-th learned direction.
The spectrum $\{\sigma_i\}$ is an empirical fingerprint of how training
distributed representational capacity across the available dimensions.

**Key observation.** If training only needs $k \ll r$ directions, the spectrum
concentrates: the first $k$ singular values carry nearly all the Frobenius
energy $\|\Delta W\|_F^2 = \sum_i \sigma_i^2$, and the remaining $r - k$
values are noise-floor artifacts of the optimizer trajectory. Gradience's
`energy_rank_90` (the minimal $k$ such that $\sum_{i=1}^{k} \sigma_i^2 \geq 0.9 \|\Delta W\|_F^2$)
directly measures this concentration.

### Stable rank as effective dimensionality

The *stable rank* of a matrix $M$ is defined as

$$\text{srank}(M) = \frac{\|M\|_F^2}{\|M\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2}$$

Unlike the algebraic rank (which is fragile to noise), stable rank is a
continuous, perturbation-robust measure of effective dimensionality. It equals
1 when the matrix is rank-1 and equals $r$ when all singular values are equal.
Gradience reports stable rank per layer and computes *utilization* as
$\text{srank}(\Delta W) / r$ -- the fraction of allocated rank that training
actually used.

### Entropy effective rank

An information-theoretic alternative is the *entropy effective rank*:

$$\text{erank}(M) = \exp\!\Bigl(-\sum_{i} p_i \ln p_i\Bigr), \quad p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}$$

This is the exponential of the Shannon entropy of the normalized energy
distribution. It captures how uniformly energy is spread across directions,
making it sensitive to "long tail" spectra that stable rank may underweight.
Gradience's `entropy_effective` rank policy uses this quantity.


## 2. Matrix Perturbation Theory and Stability of Learned Features

The singular values of $\Delta W$ are not arbitrary numbers -- they are
constrained by perturbation theory, which tells us how much the spectrum can
change under bounded modifications to the matrix.

### Weyl's perturbation theorem

For Hermitian matrices $H$ and $H + E$, Weyl's inequality gives
$|\lambda_i(H+E) - \lambda_i(H)| \leq \|E\|_2$. An analogous result for
singular values states:

$$|\sigma_i(M + E) - \sigma_i(M)| \leq \|E\|_2$$

**Implication for Gradience.** If a rank-$k$ truncation removes directions
with singular values $\sigma_{k+1}, \ldots, \sigma_r$, the perturbation to any
downstream computation that depends on $\Delta W$ is bounded by $\sigma_{k+1}$.
Gradience's compression pipeline relies on this: when $\sigma_{k+1}$ is small,
truncation is safe.

### Singular value gaps and feature stability

A large gap $\sigma_k - \sigma_{k+1}$ means the top-$k$ subspace is
*spectrally isolated*. By the Davis-Kahan theorem, the angular perturbation
of this subspace under noise is inversely proportional to the gap. Layers
with clear spectral gaps have stable learned features; layers with smooth,
gradually decaying spectra have ambiguous subspace boundaries and are harder
to compress predictably.

Gradience's `knee_elbow` policy attempts to detect such gaps via scree plot
analysis. The `optimal_hard_threshold` policy applies random matrix theory
(the Marchenko-Pastur distribution) to distinguish signal from noise in the
spectrum.

### Connection to compression safety

The Update Dominance Ratio (UDR) $= \|\Delta W\|_2 / \|W_{\text{base}}\|_2$
quantifies how "loud" the adapter is relative to the pretrained weight. When
UDR is small, the adapter is a mild perturbation to the base model, and
matrix perturbation bounds apply tightly. When UDR is large, the adapter is
substantially restructuring the representation, and truncation may have
nonlinear downstream effects that spectral bounds alone cannot predict.


## 3. Information-Geometric Perspective

### Fisher information and parameter space geometry

The Fisher Information Matrix $F$ defines a Riemannian metric on parameter space:

$$F_{ij} = \mathbb{E}\!\left[\frac{\partial \log p(y|x,\theta)}{\partial \theta_i} \cdot \frac{\partial \log p(y|x,\theta)}{\partial \theta_j}\right]$$

The eigenspectrum of $F$ describes the curvature of the statistical model
manifold. Directions with large Fisher eigenvalues are directions where the
output distribution changes rapidly with parameter perturbation; directions
with small eigenvalues are "flat" and can be compressed without functional loss.

### Entropy effective rank as a Fisher probe

The entropy effective rank of $\Delta W$ can be interpreted as a probe into
the effective dimensionality of the learned perturbation. Under the hypothesis
that training concentrates updates along high-Fisher-eigenvalue directions,
the entropy effective rank of the adapter should correlate with the effective
dimensionality of the task-relevant subspace of the Fisher manifold.

**Open question.** Does $\text{erank}(\Delta W)$ predict $\text{erank}(F)$
restricted to the adapter subspace? Gradience's `research.fisher` module
provides the infrastructure to compute empirical Fisher spectra; correlating
these with adapter spectra is a natural next experiment.

### Condition number and training anisotropy

When $\kappa(F)$ is large, the parameter space is highly anisotropic: some
directions carry orders of magnitude more information than others. This
geometric distortion makes first-order optimization difficult (SGD step sizes
are wrong in most directions). Gradience's Fisher module tracks $\kappa(F)$
alongside weight spectra, enabling direct measurement of this relationship.


## 4. Phase Transitions in Training Dynamics

### Spectral signatures of grokking

Grokking -- delayed generalization long after memorization -- has been
observed in small-scale settings and is hypothesized to involve a phase
transition in the learned representation. Key theoretical predictions:

1. **Critical slowing down.** Near a phase transition, the autocorrelation
   time of an order parameter diverges. In training, this would manifest as
   the loss (or spectral metrics) showing increasing temporal correlation
   before the transition.

2. **Fluctuation amplification.** The variance of an order parameter diverges
   at criticality. For spectral metrics, we might see the variance of stable
   rank across batches spike before grokking.

3. **Rank phase transition.** Grokking may involve a sudden shift from a
   high-rank memorization solution to a low-rank generalizing solution.
   Gradience's spectral trajectory tracking can detect such rank collapse events.

Gradience's `research.phase_transitions` module implements detection of
critical slowing down (diverging autocorrelation time), fluctuation
amplification (diverging variance), and susceptibility measures, borrowed
from the theory of second-order phase transitions in statistical mechanics.

### Double descent in rank space

The classical double-descent phenomenon (test error rises and falls as model
complexity increases) may have a spectral analog. As training progresses,
the effective rank of $\Delta W$ may exhibit non-monotonic behavior:
initially increasing as the model explores, then decreasing as it finds
a compact representation, potentially rising again in the interpolation regime.

**Open question.** Does double descent manifest in the spectral trajectory
of individual layers? Are there layers that exhibit rank inflation while
others simultaneously compress? Gradience's per-layer temporal tracking
is designed to capture exactly this kind of heterogeneous dynamics.

### Early vs. late training dynamics

Empirically, early training often produces rapidly changing, high-rank
updates (the model is broadly exploring), while late training refines a
low-rank structure. The crossover point may be a useful diagnostic:
adapters that never transition to a low-rank regime may be undertrained
or stuck in a memorization mode.


## 5. The Hessian Connection

### Loss landscape curvature and adapter spectra

The Hessian $H = \nabla^2 \mathcal{L}$ describes the local curvature of
the loss surface. Its eigenspectrum has a well-documented structure in
deep networks: a bulk of near-zero eigenvalues (flat directions) and a
small number of large eigenvalues (sharp directions).

**Hypothesis.** The singular values of $\Delta W$ reflect the curvature
structure of the loss landscape: training concentrates updates along
directions of high curvature (large Hessian eigenvalues), producing a
low-rank $\Delta W$ whose top singular vectors align with the top Hessian
eigenvectors.

If this hypothesis holds, then:
- Low stable rank $\Rightarrow$ training found a sharp minimum (few high-curvature directions)
- High stable rank $\Rightarrow$ training is in a flat region (many directions of similar curvature)
- Spectral gaps in $\Delta W$ should correlate with spectral gaps in $H$

Gradience's `research.hessian` module provides tractable Hessian measurements
via power iteration for top eigenvalues and Hutchinson's estimator for the
trace, enabling direct investigation of this relationship.

### Computational tractability

Full Hessian computation is $O(p^2)$ in storage and $O(p^3)$ in time, which
is infeasible for models with $>10^8$ parameters. However, Hessian-vector
products are $O(p)$ via reverse-mode autodiff, and these suffice for:

- Top-$k$ eigenvalues via power iteration or Lanczos ($O(k \cdot n_{\text{iter}})$ Hvps)
- Trace estimation via Hutchinson's method ($O(n_{\text{samples}})$ Hvps)
- Spectral density estimation via stochastic Lanczos quadrature

**Open question.** Can we establish a quantitative correspondence between
the Hessian spectrum (restricted to the adapter subspace) and the adapter's
singular value spectrum? This would provide a principled justification for
spectral-based compression beyond the empirical observation that it works.

**Partial answer (March 2026 reanalysis).** Canonical correlation analysis
between Hessian-space metrics (lambda1, trace_H, gHg) and representation-space
metrics (participation ratio, anisotropy, CKA) yields CC1 = 0.661, indicating
a moderate shared signal. The two measurement systems are coupled but not
redundant. See Findings §7 and `Gradience II/reanalysis/module_e_results.json`.


## 6. Subspace Alignment and Merge Analysis

### Principal angles between adapter subspaces

Given two adapters $\Delta W_a$ and $\Delta W_b$ with column spaces
$\mathcal{U}_a$ and $\mathcal{U}_b$, the *principal angles*
$\theta_1 \leq \theta_2 \leq \cdots$ between these subspaces characterize
their geometric relationship. The cosines $\cos \theta_i$ range from 1
(perfect alignment in the $i$-th direction) to 0 (orthogonal).

Gradience computes principal angles via the SVD of $U_a^\top U_b$ (following
Bjorck and Golub's classical method) and summarizes them as:
- `mean_overlap`: average $\cos \theta_i$ across directions
- `max_overlap`: maximum single-direction alignment
- `directional_agreement`: projection cosine similarity

### Merge compatibility prediction

**Empirical finding.** High subspace overlap predicts safe merging: when
two adapters have learned similar subspaces, simple averaging preserves
the shared structure. Low overlap indicates the adapters have specialized
in complementary directions, and naive merging destroys both.

Gradience's merge audit applies this principle per-layer:
- Layers with high overlap can be merged with simple averaging
- Layers with low overlap require more sophisticated strategies (TIES,
  DARE, task-arithmetic) or should be excluded from the merge
- `v_proj` layers consistently show more overlap than `q_proj` layers,
  suggesting value projections learn more universal features while query
  projections specialize to task-specific attention patterns

### Magnitude balance and merge coefficients

Beyond subspace geometry, the relative magnitude of two adapters affects
merge quality. Gradience reports `magnitude_ratio` ($\sigma_1$ of the
larger adapter divided by $\sigma_1$ of the smaller) and `frobenius_ratio`
as scale diagnostics. Large imbalances suggest one adapter will dominate
the merge unless coefficients are adjusted.


## 7. Open Theoretical Questions

1. **Spectrum universality.** Do adapters trained on the same task with
   different random seeds converge to the same spectral shape (up to
   rotation)? Gradience's multi-seed protocol provides the data to test
   this, but a theoretical framework explaining *why* (or why not) is missing.

2. **Spectral scaling laws.** How does the entropy effective rank of
   adapters scale with model size, dataset size, and training compute?
   Scaling laws for loss are well-established; spectral scaling laws
   would connect capacity allocation to compute budgets.

3. **Generalization bounds from spectra.** Can spectral properties of
   $\Delta W$ provide tighter generalization bounds than parameter-count-based
   bounds? The stable rank is a natural candidate for a complexity measure
   that is smaller than the nominal rank.

4. **Hessian-spectrum alignment.** Quantifying the correspondence between
   adapter singular vectors and Hessian eigenvectors would provide a
   principled theory of why spectral compression works.

5. **Cross-architecture geometry.** Do different architectures (attention
   vs. MLP, transformer vs. SSM) produce qualitatively different spectral
   geometries, or are there universal patterns? Gradience's architecture-agnostic
   audit can answer this empirically once sufficient data is collected.

6. **Phase transition detection.** Can spectral observables serve as
   reliable order parameters for detecting training phase transitions
   (grokking, mode collapse, catastrophic forgetting) before they
   manifest in the loss curve?

   **Partial answer (March 2026 reanalysis).** Hessian trace detects
   changepoints ~300 steps before loss in a single-run telemetry stream.
   One candidate phase transition near step 58,450 was identified via
   susceptibility clustering and trajectory tortuosity. However, critical
   slowing down in loss *precedes* that of geometric metrics in the same
   data, complicating the picture. Replication across runs is needed.
   See Findings §7.

   **Update (Study 12, March 2026).** DFA exponents of spectral
   complexity differ significantly across five hyperparameter regimes
   (F = 116.86, p ≈ 10⁻²³; n=49 runs, 10 seeds per regime). High
   learning rate produces α ≈ 1.57 while other regimes cluster at
   α ≈ 1.90--2.07. This confirms that long-range temporal correlations
   in spectral observables are regime-dependent, not a generic SGD
   property. Whether DFA exponents can serve as real-time anomaly
   detectors (flagging deviation from expected regime dynamics) remains
   an open engineering question. See Findings §8.
