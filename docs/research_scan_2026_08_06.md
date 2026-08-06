# Research Scan — August 6, 2026

**Scope.** Automated scan for new empirical research and technical documentation
highly relevant to the Gradience research program. Focused on spectral analysis
of LoRA adapters, phase transition detection, random matrix theory applied to
training dynamics, and merge compatibility analysis.

**Method.** Searched recent (2026) arXiv publications across Gradience's core
research areas; filtered for direct relevance to the program's theoretical
framework and open questions; cross-referenced against `RESEARCH_INVENTORY.md`
and `docs/RESEARCH_INVENTORY.md` to exclude already-tracked work.

---

## Article 1: The Intruder Threshold — A Spectral Law for LoRA Fine-Tuning

**Authors:** Peng Xie et al.
**Date:** July 26, 2026
**arXiv:** [2607.23711](https://arxiv.org/abs/2607.23711)

### Summary

This paper derives a closed-form spectral law for when LoRA fine-tuning creates
"intruder dimensions" — new leading singular vectors of the updated weight
matrix W + Delta W that are nearly orthogonal to all pretrained singular
vectors and drive catastrophic forgetting. The critical update strength is
computed per-layer from the measured spectrum of the pretrained weight matrix
alone via the rectangular spiked-deformation transform, with no fitted
parameters.

### Key results

- In a pre-specified study spanning four dense Transformer families, a
  state-space model, a mixture-of-experts model, and an encoder-decoder
  (18 adapters, 9,840 layer scans), the law localizes the empirical
  intruder threshold within a factor of two on 82% of layers.
- Separates intruder-bearing from intruder-free layers at deployment with
  mean AUC 0.89.
- Predicted-supercritical layers carry up to 240x more forgetting than
  subcritical ones.
- A projection-free per-layer budget rule derived from the threshold cuts
  forgetting by 62% on the most fragile model.

### Relevance to Gradience

**Critical — direct theoretical upgrade path.**

This paper provides precisely the kind of random-matrix-theoretic grounding
that Gradience's `optimal_hard_threshold` rank policy already uses (via the
Marchenko-Pastur / Gavish-Donoho framework) but extends it in a direction
Gradience has not yet pursued: predicting when the LoRA update itself is
strong enough to create *new* spectral structure in the combined W + Delta W
matrix, rather than merely measuring the adapter's internal spectrum.

Specific integration points:

1. **Compression safety.** THEORY.md §2 ("Connection to compression safety")
   discusses the UDR = ||Delta W||_2 / ||W_base||_2 as a heuristic for when
   truncation may have nonlinear downstream effects. The Intruder Threshold
   provides the *exact* spectral condition where this regime change occurs —
   a theoretically grounded replacement for the UDR heuristic.

2. **Merge compatibility.** The intruder threshold per layer could serve as a
   per-layer safety bound for merge audits: when two adapters' combined
   Delta W exceeds the intruder threshold at a layer, the merge is likely
   to produce catastrophic interference regardless of subspace alignment.
   This addresses THEORY.md §2's open question about "whether low-SV band
   interference produces merge failures that high-SV analysis misses" — if
   the combined update creates intruder dimensions, the failure mechanism
   is qualitatively different from subspace misalignment.

3. **ROADMAP.md: "Spectral scaling laws"** — the intruder threshold is a
   per-layer scalar derived from the pretrained spectrum; studying how it
   scales across architectures and model sizes would directly address the
   spectral scaling laws open question.

4. **Measurement discipline.** The no-free-parameter derivation from the
   spiked-deformation transform is exactly the kind of theoretically
   grounded threshold that avoids the fitted-threshold critique. The 82%
   localization rate and AUC 0.89 provide concrete reliability numbers.

**Criticality rating: HIGH.** The intruder threshold is a natural complement
to Gradience's existing Marchenko-Pastur-based partition. Integrating it
would add a forgetting-risk diagnostic to the audit pipeline with no
training required — only the pretrained weight spectrum is needed.

---

## Article 2: Spectral Phase Transitions and Trainability in Neural Network Learning Dynamics

**Authors:** Chanju Park et al.
**Date:** June 26, 2026
**arXiv:** [2606.28486](https://arxiv.org/abs/2606.28486)

### Summary

This paper formulates neural network training as the stochastic evolution of
an initially random matrix ensemble driven by SGD updates. It shows that
training induces a Baik-Ben Arous-Péché (BBP) transition where isolated
eigenvalues detach from the random bulk distribution, providing a dynamical
framework for representation formation. The paper derives a phase diagram of
trainability governed by step size and initial weight variance in a solvable
linear teacher-student model, where spectral evolution is analytically
tractable.

### Key results

- Demonstrates that the BBP phase transition — originally from spiked random
  matrix theory — governs when meaningful learned features emerge during
  training: below the transition, the learned weight matrix is indistinguishable
  from random noise; above it, isolated eigenvalues encode task-relevant
  directions.
- The phase diagram shows trainability depends on the ratio of learning rate
  to initialization scale, with a critical boundary separating the trainable
  from the untrainable regime.
- Spectral evolution is analytically tractable in the teacher-student setting,
  providing closed-form predictions for when and how signal eigenvalues emerge
  from the bulk.

### Relevance to Gradience

**High — formalizes Gradience's core empirical observation.**

Gradience's central empirical finding is that LoRA adapters use ~1/6 of
allocated rank (utilization ~0.17, n=86 adapters). The BBP framework provides
a *theoretical explanation*: the adapter's effective dimensionality is
determined by how many signal eigenvalues clear the BBP transition threshold.
Directions that remain below the threshold are noise-floor artifacts of the
optimizer trajectory — exactly the interpretation Gradience adopts in
THEORY.md §1 but without the formal random-matrix-theoretic grounding.

Specific integration points:

1. **THEORY.md §1 ("Why SVD of Weight Updates Reveals Training Structure").**
   The current text says "the remaining r - k values are noise-floor artifacts
   of the optimizer trajectory." The BBP framework makes this precise: below
   the BBP threshold, eigenvalues belong to the random bulk; above it, they
   are signal. This replaces a qualitative observation with a quantitative
   prediction.

2. **THEORY.md §4 ("Phase Transitions in Training Dynamics") and the Spectral
   Edge Thesis.** Gradience already tracks Xu (2026, arXiv:2603.28964) as a
   candidate formal framework for phase transitions. The BBP framework from
   Park et al. is a complementary (and arguably more foundational) approach:
   while Xu tracks the spectral gap of the Gram matrix of parameter *updates*,
   Park et al. track the eigenvalue evolution of the weight matrices
   themselves. The two frameworks address related but distinct objects, and
   both are relevant to Gradience's phase-transition detection in
   `research/phase_transitions.py`.

3. **ROADMAP.md: "Phase transition detection in spectral observables."** The
   BBP transition provides a specific, mathematically precise phase transition
   to detect: the moment when a new signal eigenvalue detaches from the bulk.
   This is more specific than the current heuristic detection (autocorrelation
   time, variance ratio) and could inform a more principled detection method.

4. **Rank policy validation.** The BBP threshold could serve as a rank policy:
   the signal rank equals the number of eigenvalues above the BBP threshold.
   This would be a theoretically grounded alternative to the empirical
   `energy_threshold(0.90)` and `optimal_hard_threshold` policies.

**Criticality rating: MEDIUM-HIGH.** The theoretical framework is highly
relevant, but the current results are in a solvable linear teacher-student
model — applicability to the nonlinear, high-dimensional LoRA setting is
plausible but unproven. Worth tracking as a theoretical reference and
potential foundation for rank policy development, but not immediately
actionable for the Gradience pipeline without empirical validation at
decoder scale.

---

## Article 3: Predicting Mergeability of Parameter-Efficient Fine-Tuning Updates

**Authors:** Lin Tang, Wei Zhang, Jing Li, Hongyu Chen, Ming Zhao, Yuxuan Wang
**Date:** June 17, 2026
**arXiv:** [2606.19549](https://arxiv.org/abs/2606.19549)

### Summary

This paper formalizes adapter mergeability as the degree to which an adapter
preserves its single-task utility after merging, and shows that mergeability
can be forecast from signals measured in the first few percent of training.
The paper packages these signals into MergeProbe, a lightweight predictor
that estimates pairwise and set-level retention and turns the estimate into a
concrete decision: merge directly, reweight, prune, or route.

### Key results

- Mergeability can be predicted from early-training signals: how the low-rank
  updates and their gradients align across tasks and how much they disturb
  shared representations.
- MergeProbe is a lightweight predictor that estimates pairwise and set-level
  retention from these early signals.
- The predictor turns estimates into concrete operational decisions: merge
  directly, reweight, prune, or route.
- Early prediction avoids the cost of fully training both adapters before
  discovering they are incompatible.

### Relevance to Gradience

**High — parallel research program with complementary methodology.**

This paper addresses the same core problem as Gradience's merge triage
pipeline: predicting whether adapter pairs will merge successfully before
committing to expensive behavioral evaluation. The key methodological
difference is temporal: Gradience operates *post-training* (spectral audit
of completed adapters), while MergeProbe operates *during training* (signals
from the first few percent of training).

Specific integration points:

1. **Complementary temporal coverage.** Gradience's merge-audit operates on
   finished adapters. MergeProbe operates during training. Together they could
   provide a two-stage triage: MergeProbe flags likely-incompatible pairs
   early (saving training compute), and Gradience's spectral audit provides
   the definitive post-training assessment.

2. **Validation opportunity.** Do MergeProbe's early-training predictions
   correlate with Gradience's post-training spectral compatibility metrics
   (subspace overlap, magnitude ratio, SV-weighted alignment)? If yes,
   this validates both approaches. If no, it identifies where the two
   measurement systems diverge and which captures information the other
   misses.

3. **ROADMAP.md: "Spectral trajectory analysis during training."** MergeProbe's
   finding that mergeability is predictable from early training signals is
   directly relevant to this open question. If early spectral trajectories
   predict final merge compatibility, Gradience's telemetry hooks could
   provide the same early-warning capability.

4. **Measurement discipline.** The paper's claim that mergeability is
   predictable from early signals is an empirical claim that could benefit
   from the same measurement-discipline analysis Gradience applies (cross-seed
   reliability, SEM, MDC). What is the test-retest reliability of MergeProbe
   predictions across random seeds?

**Criticality rating: MEDIUM-HIGH.** A parallel research program attacking
the same problem from a complementary angle. Most valuable as a cross-validation
target for Gradience's post-training triage and as a source of insights about
what signals matter for merge compatibility. Not immediately integrable into
the Gradience pipeline (different temporal regime), but the findings about
which geometric signals predict mergeability should inform Gradience's
feature selection.

---

## Additional papers noted (already tracked or lower priority)

- **"Spectral Geometry of LoRA Adapters Encodes Training Objective and Predicts
  Harmful Compliance"** (arXiv:2604.08844) — already tracked in
  `docs/RESEARCH_INVENTORY.md` as "High — already integrated."

- **"Crowded in B-Space: Calibrating Shared Directions for LoRA Merging"**
  (arXiv:2604.16826) — already tracked in `RESEARCH_INVENTORY.md` as a
  candidate replication target.

- **"SDS-LoRA: Overcoming Anisotropic Gradient Scaling in Low-Rank Adaptation"**
  (arXiv:2606.16454) — relevant to understanding optimizer dynamics in LoRA
  (anisotropic gradient scaling driven by singular values), but addresses
  a training-time optimization problem rather than the post-training
  measurement questions Gradience focuses on.

- **"Post-Optimization Adaptive Rank Allocation for LoRA" (PARA)**
  (arXiv:2604.27796) — uses SVD-based rank pruning with global thresholds,
  operationally similar to Gradience's `energy_threshold` policy. Validates
  the approach but does not add new theoretical or empirical insight.

---

## Recommendations

1. **Integrate the Intruder Threshold (Article 1) into the external literature
   register** with status "High — candidate integration target." The
   spiked-deformation transform threshold is a natural extension of the
   existing Marchenko-Pastur partition and could add a forgetting-risk
   diagnostic to the audit pipeline.

2. **Track the BBP framework (Article 2) as a theoretical reference** with
   status "Medium — tracking." It provides the formal grounding for
   Gradience's core empirical observation about low rank utilization, but
   the linear teacher-student setting limits immediate applicability.

3. **Track MergeProbe (Article 3) as a parallel program** with status
   "Medium-High — cross-validation target." Most valuable for validating
   whether early-training signals and post-training spectral signals converge
   on the same compatibility judgments.
