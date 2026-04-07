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

*A qualification is necessary for the post-fine-tuning setting. The Weyl
perturbation argument supports truncation safety for the **pretrained** model,
where small singular values reflect dimensions along which the matrix carries
negligible signal. Fine-tuning may change this: random matrix theory analysis
of pretrained Transformer weight matrices finds that fine-tuning operates
preferentially in the low-SV spectral tail — the directions that are quiet in
the pretrained model but that acquire task-specific content during adaptation
(Medina & Sørensen, 2025; arXiv:2410.17770). If this holds in the LoRA
setting, the safety bound $|\sigma_i(M+E) - \sigma_i(M)| \leq \|E\|_2$ still
holds formally, but what it bounds has changed: the perturbation now has
content rather than just noise. Two regimes should therefore be distinguished:
(a) **compression of the pretrained model** — small-SV truncation is safe,
as the tail carries no signal yet; and (b) **compatibility assessment of two
fine-tuned adapters** — interference in low-SV directions may be
task-relevant, even if it does not dominate the merged spectrum's energy.
Gradience's compression pipeline operates in regime (a); its merge triage
operates in regime (b). Whether this distinction is operationally consequential
— whether low-SV band interference produces merge failures that high-SV
analysis misses — is an open empirical question addressed by the tail
interference probe (§7.2, "Tail-band interference as an independent
compatibility signal"; see also `scripts/tail_interference_probe.py`).*


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

### Curvature telemetry as a phase-transition probe

The theoretical predictions above — critical slowing down, fluctuation
amplification, rank phase transitions — describe phenomena that
second-order geometric signals should detect before first-order signals
(loss, accuracy) respond. Recent curvature telemetry results provide the
first empirical evidence for this claim in a fine-tuning context: Hessian
energy (sum of squared eigenvalues, $\sum \lambda^2$) leads validation accuracy by
3--6 updates during LoRA fine-tuning, with walk-forward forecasters
using curvature features alone reducing RMSE by ~36% versus a persistence
baseline (see FINDINGS.md §16a). This lead-lag relationship is precisely
the kind of leading indicator hypothesized above — a spectral observable
that detects geometric events (curvature collapse into flatter basins)
before they manifest as performance changes.

The three-act gradient alignment structure observed on Mistral-7B
(FINDINGS.md §17) — explore (low alignment, high curvature), lock-on
(alignment at edge-of-stability, $R_q \approx 1.06$), destabilize
(alignment drops, variance increases) — maps directly onto the curvature
telemetry paper's narrative: high-curvature exploration followed by
flat-basin consolidation followed by accuracy improvement. The curvature
telemetry result provides the micro-scale temporal evidence; the three-act
structure provides the macro-scale phase portrait. Together they suggest
that the phase-transition detection framework outlined above is not
merely theoretical but captures real dynamics observable in LoRA
fine-tuning trajectories.


## 5. The Hessian Connection

### 5.1 Static correspondence: weight spectra and loss curvature

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

Full Hessian computation is $O(p^2)$ in storage and $O(p^3)$ in time, which
is infeasible for models with $>10^8$ parameters. However, Hessian-vector
products are $O(p)$ via reverse-mode autodiff, and these suffice for:

- Top-$k$ eigenvalues via power iteration or Lanczos ($O(k \cdot n_{\text{iter}})$ Hvps)
- Trace estimation via Hutchinson's method ($O(n_{\text{samples}})$ Hvps)
- Spectral density estimation via stochastic Lanczos quadrature

**Partial answer (March 2026 reanalysis).** Canonical correlation analysis
between Hessian-space metrics (lambda1, trace_H, gHg) and representation-space
metrics (participation ratio, anisotropy, CKA) yields CC1 = 0.661, indicating
a moderate shared signal. The two measurement systems are coupled but not
redundant — they share a dominant axis of variation but capture different
aspects of the parameter geometry. See FINDINGS.md §16 and
`Gradience II/reanalysis/module_e_results.json`. This establishes the
*cross-sectional* evidence: at any given snapshot, Hessian geometry and
adapter geometry are correlated.

### 5.2 Dynamic correspondence: curvature telemetry

The cross-sectional correlation in §5.1 leaves open whether the
Hessian-adapter relationship is merely coincidental or reflects a
genuine dynamic coupling. The curvature telemetry results (FINDINGS.md
§16a) provide the *temporal* evidence: Hessian energy $\sum \lambda^2$ leads
validation accuracy by 3--6 updates during LoRA fine-tuning, and
walk-forward forecasters using only curvature features reduce
short-horizon accuracy RMSE by ~36% versus a persistence baseline.
The lead-lag relationship is validated with AR(1) pre-whitening,
effective sample size correction, contiguous block-bootstrap confidence
intervals, and surrogate-null tests (phase randomization and circular
rotation).

This result transforms the Hessian connection from a static correlation
into a dynamic predictive relationship. The conceptual mechanism:
Hessian energy rising signals the optimizer entering high-curvature
regions (exploration, the loss landscape is sharply curved along the
update directions). Hessian energy collapsing signals escape to flatter
basins (consolidation, the optimizer has found a low-curvature region
where the representation can stabilize). Accuracy improving shortly
after signals the representation locking in within the new basin.

The connection to the spectral partitioning results (§6, FINDINGS.md
§14) is suggestive: the N127 checkpoint progression showing high-SV
alignment rising from 0.244 to 0.608 and plateauing around step 150
is the *SVD-side* view of the same process that curvature telemetry
captures from the *Hessian side*. The curvature collapse events
identified in the telemetry paper are plausibly the moments when the
spectral partition sharpens — when high-SV directions lock in and
low-SV directions differentiate. This is a testable prediction (see
§7.2, "Curvature-partition correspondence").

### 5.3 Toward a unified spectral measurement framework

Both the post-training SVD audit and the during-training Hessian
telemetry are measuring the same underlying object — the spectral
geometry of the parameter-loss landscape — from different temporal
vantage points and via different mathematical decompositions (singular
values of weight matrices vs. eigenvalues of the loss Hessian). The
static correspondence (§5.1) shows the two views share signal. The
dynamic correspondence (§5.2) shows that one view *predicts* the
other across time. Together they suggest that a complete diagnostic
framework should use Hessian telemetry during training to forecast
when the representation is stabilizing, then switch to SVD audit
after training to characterize what the stabilized representation
looks like and whether it is compatible with other adapters.

This conceptual unification motivates a methodological commitment that
distinguishes the program from pure ML systems work: *spectral metrics
are measurement instruments, and the question of whether they can be
trusted requires the same psychometric analysis that any measurement
instrument demands.* In classical test theory, no score is reported
without its reliability coefficient and standard error of measurement.
The same discipline should apply to spectral diagnostics:

- **Test-retest reliability (cross-seed ICC).** Does $\sum \lambda^2$ at step
  50 agree across 5 random seeds? If ICC < 0.7, the metric is too
  noisy to trust at that granularity.
- **Standard error of measurement (SEM).** Given the ICC, how much
  does a single $\sum \lambda^2$ reading vary due to measurement noise versus
  genuine geometric change?
- **Minimal detectable change (MDC).** How large must a shift in
  $\sum \lambda^2$ be before we are confident it reflects a real geometric
  event, not noise?
- **Information function analog.** At what range of curvature values
  is $\sum \lambda^2$ most discriminating? (Analogous to where an IRT item's
  information function peaks.)

No existing ML diagnostic toolkit asks these questions. The reliability
analysis applies equally to SVD-based audit metrics (stable rank,
energy concentration, subspace overlap) and to Hessian-based telemetry
metrics ($\sum \lambda^2$, $\lambda_1$, trace). Both families of metrics require
empirical reliability characterization before their signals can be
trusted for operational decisions — and the infrastructure for
computing that characterization (cross-seed ICC, block-bootstrap CIs,
surrogate-null tests) is shared between the two.

The practical implication for Gradience v1.0: the `stats.reliability`
module treats spectral metrics as instruments with quantifiable
psychometric properties, and the `spectral` layer provides the shared
computation infrastructure that both the audit pipeline and the
telemetry recorder consume.


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


### Spectral partitioning: shared vs. task-specific directions

Recent evidence from multi-task LoRA training (Tian, Ledent, & Sun, 2026;
ICLR 2026, arXiv:2603.01526) indicates that the singular value spectrum of
LoRA adapters partitions into functionally distinct bands during training:

- **High-SV band** (top 20% by singular value magnitude): 89% inter-task
  alignment across 16 instruction-following tasks on LLaMA-2-7B,
  concentrating 54% of total singular value mass. These directions encode
  structure shared across tasks.

- **Low-SV band** (bottom 50%): only 3% inter-task alignment, encoding
  task-specific features.

This partitioning has direct theoretical implications for merge analysis.
The interaction term in the merged spectrum (Technical Report §2.3) is
weighted by singular values: $z = \text{sign}(\delta) \cdot \cos(\theta)
\cdot \cos(\phi)$, where $\theta$ and $\phi$ are principal angles between
singular subspaces. If high-SV directions are shared, then the principal
angles in the high-energy band will be small for same-task adapters,
making the energy-weighted interaction constructive. Large angles in the
low-SV band contribute minimally because those directions carry little
energy. This explains *why* energy-rank concentration is predictive of
merge compatibility: the metric naturally emphasizes the shared directions
where conflict would matter most.

**Perturbation-theoretic explanation.** The Davis-Kahan theorem (§2 above)
provides a candidate mechanism. If the pre-trained weight matrix $W_0$ has
a spectrally isolated dominant subspace (large gap $\sigma_k - \sigma_{k+1}$),
then any bounded perturbation $\Delta W$ of rank $r \leq k$ will have its
dominant singular directions attracted toward $W_0$'s dominant subspace —
the angular perturbation is bounded by $\|\Delta W\|_2 / (\sigma_k - \sigma_{k+1})$.
Different tasks trained on the same backbone experience the same pre-trained
spectral constraint, so their dominant update directions converge. Low-energy
directions, which interact with the spectrally flat region of $W_0$'s
spectrum (where gaps are small), have no such convergence pressure and are
free to specialize.

This account makes two testable predictions: (a) layers with larger spectral
gaps (or, more precisely, greater spectral mass concentration) in $W_0$ should
show higher inter-adapter alignment in the high-SV band, and (b) the
Marchenko-Pastur bulk edge (used in Gradience's `optimal_hard_threshold`
policy) should approximate the boundary between shared and task-specific
spectral bands.

**Empirical results (N127, April 2026).** Both predictions have been tested
on Gradience's independently trained adapter corpus (DistilBERT-base,
rank 16, SST-2 and QNLI tasks). Using the Gavish-Donoho optimal hard
threshold as the partition point:

- **The partitioning is present and task-dependent.** Same-task adapter pairs
  show 7.8× higher SV-weighted alignment in the high-SV band than the low-SV
  band (high-SV alignment = 0.634). Cross-task pairs drop to 2.5× (high-SV
  alignment = 0.133), with the difference highly significant (t = 23.4,
  p ≈ 10⁻⁴⁶). The high-SV directions encode task-specific shared structure,
  not generic backbone geometry.

- **Convergence is monotonic with a plateau.** Tracking alignment across
  training steps 50–200, high-SV alignment rises from 0.244 to 0.608 and
  plateaus around step 150. Spectral energy concentration sharpens
  simultaneously (56% → 86%). Low-SV alignment barely changes (0.060 → 0.076).
  This matches the attractor picture: the pre-trained spectral structure
  governs the dominant directions early; later training fills in task-specific
  detail in the residual dimensions.

- **Energy concentration, not raw gap, predicts alignment.** The naive
  Davis-Kahan operationalization (σ₁−σ₂) shows no significant correlation
  with per-layer high-SV alignment (r = 0.038, p = 0.86 for SST-2). But
  energy concentration in $W_0$ — the fraction of spectral mass in the top-k
  subspace — does predict alignment significantly for QNLI adapters
  (Spearman r = 0.53–0.58, p < 0.01). This refines the theoretical
  prediction: the relevant quantity for a formal convergence bound is not the
  adjacent-SV gap but the degree of low-dimensional concentration in $W_0$.

These results constitute a converging-operations argument with Tian et al.:
training-side gradient analysis and post-hoc SVD-based audit arrive at the
same spectral partition through independent methodological pipelines.
The magnitude is weaker without shared gradients (7.8× vs 30×), consistent
with the absence of co-training reinforcement.

**Cross-architecture replication.** The same-task/cross-task spectral
separation is not confined to the small-encoder regime. Earlier Gradience
research on Mistral-7B (Gradience Series Post 3) found same-task subspace
overlap of 0.473 vs cross-task overlap of 0.200 across 36 adapter pairs
(chat, GSM8K, code tasks; rank 8; 3 seeds) — a 2.4× separation
(t = 12.985, p < 0.0001). This used unweighted mean overlap rather than
the SV-weighted metric from N127, so the magnitudes are not directly
comparable, but the structural pattern — same-task pairs clustering
substantially higher on overlap — replicates at decoder scale. The
Mistral-7B study additionally showed that subspace overlap predicts merge
dominance at r = 0.846 and rank-orders merge quality at Spearman ρ = 0.710,
confirming that the spectral observables are not only structurally meaningful
but operationally predictive on 7B-parameter decoders.

The remaining open theoretical work is to formalize the concentration-weighted
convergence bound (see §7.2, "Concentration-weighted convergence bound").

*Limitation note (N128, April 2026)*: The SV-weighted overlap metric used
in Gradience's merge pipeline is conservative for same-task pairs: it
deflates apparent compatibility relative to the unweighted mean cosine by
a mean of 0.21 (energy masking direction finding). The metric's residual
bias favors false positives (flagging compatible pairs), not false negatives
(missing incompatible ones). See FINDINGS.md §8 Limitations for the full
statement.


## 7. Theoretical Questions: Status and Agenda

This section classifies the research program's theoretical questions by
their empirical status as of April 2026. Questions in §7.1 have received
substantial empirical answers — the phenomenon is confirmed, even if the
formal theory remains incomplete. Questions in §7.2 are genuinely open:
neither confirmed nor refuted, and without empirical traction.


### 7.1 Empirically Resolved or Substantially Constrained

These questions have moved beyond "open" status. The empirical results
constrain what a theory must explain and, in some cases, identify the
specific formal object that a proof should target.

**Cross-architecture geometry.** *Original question:* Do different
architectures produce qualitatively different spectral geometries, or
are there universal patterns? *Status: answered in the affirmative —
spectral profiles are architecture-agnostic in practice.* Study 14
(29 adapters, 8 base models, 2B–8B parameters) found consistent
spectral profiles across Llama-2/3, Mistral-7B, Gemma, and Phi,
including median 50% compression potential and negative rank-utilization
correlation. Post 7 expanded this to 86 adapters across 22 architectures
and 12 task categories; core findings were unchanged (mean utilization
0.172, compression potential 50%). The attention/MLP utilization gap
reported in encoder-only experiments (Finding §5 in FINDINGS.md) does
not replicate at scale — at n=86, attention and MLP utilization are
essentially identical (0.167 vs 0.166). The remaining theoretical
interest is quantitative: *why* does the spectral profile converge
across architectures? A candidate explanation is that the rank-deficiency
is dominated by the optimization dynamics (SGD on a low-rank
parameterization) rather than the model architecture, but this has
not been formalized.

**Spectral partitioning convergence.** *Original question:* Does
independent fine-tuning (not co-training) produce the same high-SV shared
/ low-SV task-specific partitioning observed by Tian et al. (2026)?
*Status: empirically confirmed.* N127 (April 2026, DistilBERT-base,
rank 16, SST-2 and QNLI) showed that independently trained same-task
adapter pairs exhibit 7.8× higher SV-weighted alignment in the high-SV
band than the low-SV band, cross-task pairs show 2.5× (t = 23.4,
p ≈ 10⁻⁴⁶ for the difference), and alignment rises monotonically during
training, plateauing around step 150 with spectral energy concentration
increasing from 56% to 86% (FINDINGS.md §§11–14). Cross-architecture
support comes from Mistral-7B (Post 3): same-task overlap 0.473 vs.
cross-task 0.200, 2.4× separation, t = 12.985. The phenomenon is
confirmed on two encoder backbones. A third replication on DeBERTa-v3-base
(N07, FINDINGS.md §23) showed same-task compatibility 0.449 vs. cross-task
0.160 (t = 15.87, p < 10⁻¹⁵), extending the partitioning to a backbone
with a distinct pretraining objective (replaced token detection vs. MLM).
The stable rank convergence to 1.2–1.6 at nominal rank 16 is quantitatively
consistent with N127's energy concentration finding. What remains is the
formal convergence bound (see §7.2).

A critical refinement: the naive Davis-Kahan operationalization (adjacent
spectral gap σ₁−σ₂ of the pre-trained matrix W₀) does not predict
per-layer alignment (r = 0.038, p = 0.86). But W₀ *energy concentration*
— fraction of spectral mass in the top-k subspace — does predict
alignment for QNLI adapters (r = 0.53–0.58, p < 0.01). This pins down
the mathematical object: a formal bound should be stated in terms of
spectral mass concentration, not adjacent-SV gaps. The bound must also
account for task-dependence (the W₀ concentration → alignment
relationship holds for QNLI but not SST-2).

**Phase transition detection.** *Original question:* Can spectral
observables serve as reliable order parameters for detecting training
phase transitions before they manifest in the loss curve? *Status:
substantially confirmed, with methodological refinement.* Two
complementary lines of evidence now support spectral observables as
leading indicators.

First, macro-scale detection: Hessian trace detects changepoints
approximately 300 steps before loss in a single-run telemetry stream
via CUSUM changepoint analysis (FINDINGS.md §16). One candidate phase
transition near step 58,450 was identified via susceptibility clustering
and trajectory tortuosity. However, critical slowing down in loss
*precedes* that of geometric metrics in the same data, complicating the
"geometry detects transitions first" narrative for CSD specifically.

Second, micro-scale forecasting: the curvature telemetry paper
(FINDINGS.md §16a) provides a cleaner and statistically more rigorous
result. Hessian energy $\sum \lambda^2$ leads validation accuracy by 3--6
updates, and walk-forward forecasters using only curvature features
reduce short-horizon accuracy RMSE by ~36% versus a persistence
baseline. The methodology — AR(1) pre-whitening, effective sample size
correction, contiguous block-bootstrap CIs, and surrogate-null tests —
validates the lead-lag relationship against the primary threats to
causal inference in time series (shared autocorrelation, spurious
cross-correlation from trending). This is the strongest evidence to
date for the "spectral observables as leading indicators" hypothesis,
providing actionable *forecasting*, not just retrospective *detection*.

The CSD complication remains but is now contextualized: CSD and CCF-based
lead-lag analysis are different detection methods asking different
questions. CSD asks whether the system approaches criticality; the CCF
analysis asks whether curvature dynamics *predict* performance dynamics.
The latter holds robustly.

Third, cross-architecture replication: the N07 DeBERTa adjudication
(FINDINGS.md §23) replicated the curvature lead-lag on DeBERTa-v3-base
(185M parameters) using stochastic Hutchinson estimation rather than
deterministic finite-difference probes. Median optimal lag = 3 intervals
(150 steps) across all 8 adapters, consistent with §16a's 3–6 update
lead. Phase transitions were detected in all 4 large adapters via rolling
variance analysis (regime shift magnitude 57.8%). This extends the
lead-lag finding from a single architecture/estimator combination
(GPT-2/deterministic) to a second (DeBERTa/stochastic), strengthening
the case that curvature-as-leading-indicator is a general property of
LoRA fine-tuning dynamics.

DFA exponents of spectral complexity differ significantly across five
hyperparameter regimes (F = 116.86, p ≈ 10⁻²³; n=49 runs, 10 seeds
per regime; Study 12, FINDINGS.md §18). High learning rate produces
α ≈ 1.57 while other regimes cluster at α ≈ 1.90–2.07. This confirms
that long-range temporal correlations in spectral observables are
regime-dependent, not a generic SGD property. The remaining questions
are: (a) whether DFA exponents can serve as real-time anomaly detectors,
(b) whether the three-act gradient alignment structure observed on
Mistral-7B (FINDINGS.md §17) generalizes across seeds and tasks, and
(c) replication of the phase transition candidate.

**Structural-behavioral separation.** *Not originally listed as a
theoretical question, but resolved empirically and theoretically
significant.* Study 16 (5 Llama-2-7B pairs, Frobenius ratios up to
19.7×; FINDINGS.md §9) demonstrated that structural compatibility is
necessary but not sufficient for merge quality. Study 17 (FINDINGS.md
§10) showed that pre-merge spectral compression does not meaningfully
improve outcomes. These results constrain the theory: any formal account
of merge success must include a behavioral component that is not
reducible to spectral geometry. The implication is that a complete
theory of merge compatibility has at least two independent factors —
subspace geometry and source adapter quality — and the spectral
observables capture only the first.


### 7.2 Genuinely Open

These questions have no substantial empirical traction. They represent
the theoretical frontier of the research program.

**Concentration-weighted convergence bound.** This is the most precisely
defined open problem. The empirical results from §7.1 (spectral
partitioning convergence) specify exactly what needs to be proved: that
the principal angle between dominant subspaces of independently trained
LoRA adapters on the same backbone converges to a small value, bounded
as a function of the spectral mass concentration in W₀'s top-k
subspace. The naive Davis-Kahan bound (using the adjacent spectral gap)
fails empirically; the bound must use a concentration-weighted metric
instead. Key constraints: the bound must be task-dependent (it holds
for QNLI but not SST-2 on the same backbone), must account for plateau
behavior (convergence saturates around step 150 in the small-encoder
regime), and should predict the ~2.5× residual cross-task H/L ratio as
a consequence of backbone-level shared structure. This is a pen-and-paper
problem in matrix perturbation theory, likely requiring a modified
Davis-Kahan argument that replaces the gap condition with a spectral
mass condition. See the discussion in the spectral partitioning
subsection above (§6) and THEORY.md §2 for the perturbation-theoretic
setup.

**Tail-band interference as an independent compatibility signal.** The
energy-weighted interaction bound in the Technical Report (§2.3) and the
spectral partitioning results (§6, N127) both assume that high-SV directions
are the primary locus of compatibility-determining interaction between adapter
pairs. This assumption rests on the weighting structure of the cross-term:
the interaction $z_i = \text{sign}(\delta_i) \cdot \cos(\theta_i) \cdot
\cos(\phi_i)$ is weighted by singular value magnitudes, so low-SV directions
contribute negligibly to the merged spectrum regardless of their angular
relationship. This is formally correct as a claim about spectral energy in
the result, but it sidesteps a different question: whether the low-SV band
carries task-relevant information that is damaged by cross-adapter
interference, even if the damage does not appear in the dominant singular
values of the merged output.

Medina and Sørensen (2025) find that fine-tuning refines model behavior
primarily in low-SV spectral regions — directions quiet before adaptation
that acquire task-specific content during it. If this holds in the LoRA
setting, two adapters could be spectrally compatible in the high-SV band
(same-task, low principal angles, Gradience would classify SAFE) while
carrying conflicting task-specific modifications in the low-SV band. Whether
this produces observable merge degradation is unknown.

The open problem has two parts. First, an **empirical test**: for adapter
pairs in the Gradience validation corpus with known merge outcomes, compute
subspace overlap separately in the high-SV and low-SV bands (partitioned at
the Marchenko-Pastur threshold) and look for cases where high-SV compatibility
is favorable but low-SV conflict is high. If no such cases exist in the
current corpus, the concern is not operationally urgent at current scales.
If they exist and correlate with merge degradation, a tail-aware compatibility
metric is warranted. This test is specified fully in CHG-005
(`scripts/tail_interference_probe.py`). Second, a **theoretical question**:
under what conditions does interference in the low-SV band produce measurable
behavioral degradation, given that the merged spectrum's dominant directions
are unaffected? A candidate answer involves the readout layer's sensitivity to
tail-band noise, which would connect this problem to the conjunctive failure
model's readout-gate condition.

**Empirical handle**: Run CHG-005 on the existing validation corpus before
the DeBERTa GPU session. If CHG-005 finds false negatives, add a sixth
prediction to the DeBERTa study (see CHG-004) targeting tail-band partition
stability. If CHG-005 finds none, the problem remains open but not urgent.

*Empirical status (N128, April 2026)*: The probe found zero false-negative
candidates across 20 encoder-classification pairs (8 same-task, 12
cross-task; rank ≤ 16). The concern is not operationally urgent in the
current validation regime. A supplementary finding: SV-weighting deflates
rather than inflates apparent overlap for same-task pairs (mean energy
masking −0.21), meaning Gradience's metric is conservative, not liberal.
The bound on false-negative rate is 1/8 = 12.5% (empirical) or 37.5%
(rule of three 95% upper bound). The problem remains open for decoder-scale
high-rank adapters where the tail carries more absolute energy.

**Spectrum universality.** Do adapters trained on the same task with
different random seeds converge to the same spectral *shape* (up to
rotation)? Gradience's multi-seed protocol provides the data to test
this — cross-seed spectral stability is high for aggregate statistics
like stable rank (CV < 0.1; FINDINGS.md §4) — but the question about
*shape* convergence (the full ordered spectrum {σᵢ}, not just summary
statistics) has not been tested, and the theoretical framework explaining
why the shape should be seed-invariant is absent. A positive answer
would imply that the spectral profile is a property of the (task,
architecture) pair, not the optimization trajectory. A negative answer
would bound the precision of spectral triage.

**Spectral scaling laws.** How does the entropy effective rank of
adapters scale with model size, dataset size, and training compute?
Scaling laws for loss are well-established (Kaplan et al., Hoffmann
et al.); spectral scaling laws would connect capacity allocation to
compute budgets. The Post 7 audit (86 adapters, 22 architectures)
provides a cross-sectional dataset, but the confounds (different
training recipes, datasets, durations) make scaling analysis difficult.
A controlled study varying model size while holding other factors
constant would be needed. This question is empirically accessible but
resource-intensive.

**Generalization bounds from spectra.** Can spectral properties of
ΔW provide tighter generalization bounds than parameter-count-based
bounds? The stable rank is a natural candidate for a complexity measure
that is smaller than the nominal rank and tracks the actual capacity
used. PAC-Bayes bounds using spectral norms exist in the literature
(Neyshabur et al., 2018; Arora et al., 2018) but have not been
specialized to the low-rank LoRA parameterization where the spectral
structure is particularly clean.

**Hessian-spectrum alignment.** Quantifying the correspondence between
adapter singular vectors and Hessian eigenvectors would provide a
principled theory of why spectral compression works — specifically,
why truncating low-energy singular directions preserves task performance.
The canonical correlation between Hessian-space and representation-space
metrics is moderate (CC1 = 0.661; FINDINGS.md §16), suggesting shared
signal but not redundancy. A formal alignment result — proving that the
top-k singular directions of ΔW approximate the top-k eigendirections
of the task Hessian restricted to the LoRA subspace — would ground
compression safety in optimization theory rather than empirical
observation.

**Curvature-partition correspondence.** Does curvature collapse (as
detected by Hessian telemetry) coincide with spectral partition
sharpening (as measured by the Marchenko-Pastur partition)? The
curvature telemetry paper (FINDINGS.md §16a) shows that $\sum \lambda^2$
collapse precedes accuracy improvement. The N127 checkpoint progression
(FINDINGS.md §14) shows that high-SV alignment rises and plateaus
during training. The open question is whether these are the same event
observed through different instruments — the Hessian eigenspectrum
collapsing into a flatter basin at the same moment that the SVD-based
spectral partition sharpens.

If they are the same event, the curvature telemetry signal becomes an
*online* proxy for the spectral partition quality that the audit
pipeline measures *offline*. A practitioner could monitor $\sum \lambda^2$
during training and infer, in real time, whether the adapter's spectral
structure has stabilized enough for reliable post-hoc audit — rather
than waiting until training completes to discover that the partition
never crystallized.

*Partial empirical traction (N07, April 2026).* The DeBERTa adjudication
study (FINDINGS.md §23) instrumented training with both curvature
telemetry (Hutchinson trace + power iteration every 50 steps) and
structural SVD snapshots (stable rank, energy_rank_90 at same cadence).
Two results provide indirect support: (1) curvature lead-lag replicates
cross-architecture (median lag 3 intervals, consistent with §16a's 3–6
update lead on GPT-2), confirming that the Hessian temporal dynamics are
not architecture-specific; (2) phase transitions detected in Hessian
dynamics (4/4 large adapters) coincide temporally with the training
regime where structural metrics stabilize (energy_rank_90 plateauing in
the second third of training, regime shift magnitude 57.8%). However,
the direct test — cross-correlating curvature collapse events with
SV-weighted alignment jumps at matched checkpoints — requires denser
Hessian sampling and MP-partitioned alignment computation at each
snapshot. The N07 data provides curvature and *aggregate* structural
metrics but not *per-direction* alignment against the MP partition at
each step. A follow-up study with finer-grained dual instrumentation
(both Hessian eigenvectors and per-direction SVD alignment at each
checkpoint) would resolve this correspondence definitively.

The prediction remains: curvature collapse events (defined as
$\sum \lambda^2$ dropping below its running mean by $>1\sigma$) should
coincide with step-wise increases in high-SV alignment within a window
of $\pm 3$ snapshots. The N07 results are consistent with this but do
not yet test it directly.
