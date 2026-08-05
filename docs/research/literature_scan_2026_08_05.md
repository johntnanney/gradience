# Literature Scan — 2026-08-05

Automated scan of recent preprints relevant to the Gradience research
program. Three papers identified as high-relevance; none currently appear
in the project's `RESEARCH_INVENTORY.md`.

---

## 1. The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning

- **arXiv:** [2607.23711](https://arxiv.org/abs/2607.23711) (July 2026)
- **Authors:** Peng Xie et al.

### Summary

Derives a closed-form, parameter-free per-layer critical update strength

    s* = theta_bar / (gamma * sigma_1(BA))

from the rectangular spiked-deformation transform (a generalization of the
Baik-Ben Arous-Peche transition to rectangular matrices). When a LoRA
update exceeds this threshold, it creates "intruder dimensions" -- new
leading singular vectors of the updated weight matrix that are nearly
orthogonal to all pretrained singular vectors. These intruder dimensions
are identified as the primary driver of catastrophic forgetting.

### Empirical validation

Pre-specified study spanning 4 dense Transformer families, a state-space
model (SSM), a mixture-of-experts model, and an encoder-decoder
architecture (18 adapters, 9,840 layer scans). The law localizes the
empirical intruder threshold within a factor of two on 82% of layers and
separates intruder-bearing from intruder-free layers at deployment time
with mean AUC 0.89. Predicted-supercritical layers carry up to 240x more
forgetting than subcritical ones. A projection-free per-layer budget rule
derived from the threshold cuts forgetting by 62% on the most fragile
model.

### Relevance to Gradience

**Critical -- immediate integration candidate.**

This paper provides the missing theoretical bridge between several
Gradience research threads:

1. **Spectral analysis.** The intruder threshold is computed from SVD
   quantities Gradience already measures (singular values of BA, spectral
   properties of the pretrained weight matrix). The threshold could be
   added as a new metric alongside energy_rank_90, stable_rank, and
   entropy_effective_rank.

2. **Phase transitions.** The intruder emergence IS a spectral phase
   transition -- the BBP-family transition Gradience's `phase_transitions`
   module is designed to detect. This provides the formal connection to the
   Spectral Edge Thesis (arXiv:2603.28964, already tracked in
   RESEARCH_INVENTORY.md) but applied specifically to LoRA rather than
   general training dynamics.

3. **Rank policies.** The per-layer critical threshold directly informs
   rank selection: the optimal rank should keep the update subcritical.
   This could be implemented as a new rank policy
   (`intruder_threshold_ceil`) alongside the existing `energy_threshold`,
   `knee_elbow`, and `optimal_hard_threshold` policies.

4. **Merge compatibility.** Intruder dimensions predict catastrophic
   forgetting, which is the same failure mode Gradience's merge pipeline
   diagnoses. The threshold provides a principled, per-layer measure of
   merge fragility that complements the existing subspace-overlap and
   magnitude-ratio metrics.

5. **Adapter QA.** The intruder count / supercriticality ratio could
   become a new field in the `gradience.adapter_qa/v1` schema, offering a
   more targeted structural signal than the current energy_rank_90_median.

**Criticality: HIGH.** Unifies BBP theory with practical LoRA diagnostics
in a way directly implementable within Gradience's existing SVD pipeline.
The closed-form threshold requires no gradient computation, consistent
with Gradience's training-free CPU-audit regime. The 9,840-layer
validation across 7 architecture families is unusually broad empirical
coverage. Recommended action: add to RESEARCH_INVENTORY.md at "High"
priority, prototype the threshold computation against existing audit data,
assess whether it discriminates between Gradience's existing
SAFE/CONFLICTING/IMBALANCED verdict categories.

---

## 2. HiP-LoRA: Budgeted Spectral Plasticity for Robust Low-Rank Adaptation

- **arXiv:** [2604.17751](https://arxiv.org/abs/2604.17751) (April 2026)
- **Authors:** (See paper)

### Summary

HiP-LoRA is a spectrum-aware LoRA variant that operates in the spectral
basis of the pretrained weight matrix. It decomposes updates into two
channels:

- **Principal channel:** updates within the dominant singular subspace of
  the pretrained weights (high-SV directions). A singular-value-weighted
  stability budget constrains how much these directions can be perturbed.
- **Residual channel:** a standard low-rank update in the orthogonal
  complement (low-SV directions), where task-specific learning is
  concentrated.

The stability budget on the principal channel continuously balances
pretrained behavior preservation with task-specific plasticity. The SVD
of the pretrained weights is cached (one-time cost), and updates are
projected into principal vs. residual components throughout training.

### Key findings

Experiments on Llama-3.1-8B show HiP-LoRA drastically reduces pretraining
degradation and multi-adapter "MergeFail" events compared to standard
LoRA, LoRA+, rsLoRA, PiSSA, and DoRA under matched parameter budgets.
Particularly strong results in interference-sensitive tasks: continual
tuning, knowledge editing, and multi-adapter merging.

### Relevance to Gradience

**High -- conceptual validation and potential diagnostic integration.**

1. **Spectral partitioning validation.** HiP-LoRA's principal/residual
   decomposition is the training-time implementation of the same spectral
   partitioning Gradience discovered empirically: high-SV directions
   encode task-shared structure (89% inter-task alignment per FINDINGS.md
   sections 11-14), low-SV directions encode task-specific features.
   HiP-LoRA's success confirms this partitioning is not just descriptive
   but causally operative.

2. **"Spectral interference" as failure mode.** HiP-LoRA's central
   diagnosis -- that standard LoRA concentrates energy on leading
   pretrained singular directions, causing forgetting and merge fragility
   -- directly explains why Gradience's CONFLICTING verdict layers show
   high subspace overlap with directional disagreement. The update
   dominance ratio (UDR) from THEORY.md section 2 may be the
   post-hoc diagnostic for the interference HiP-LoRA prevents.

3. **Merge robustness baseline.** The MergeFail metric and experimental
   protocol could be adopted as a validation target for Gradience's merge
   audit: if spectral audit correctly predicts which adapter pairs will
   MergeFail under HiP-LoRA vs. standard LoRA, it validates the audit's
   discriminative power.

**Criticality: MEDIUM-HIGH.** The conceptual alignment is strong, but
HiP-LoRA is a training method, not a post-hoc diagnostic. Integration
would be indirect: (a) validate that Gradience's spectral partitioning
metrics predict HiP-LoRA's effectiveness, (b) borrow the MergeFail
protocol for validation experiments. The principal/residual decomposition
could also inform a new "spectral interference index" diagnostic. No
immediate code integration path, but important theoretical confirmation.

---

## 3. Spectral Phase Transitions and Trainability in Neural Network Learning Dynamics

- **arXiv:** [2606.28486](https://arxiv.org/abs/2606.28486) (June 2026)
- **Authors:** Chanju Park, Dario Bocchi, Francesco D'Amico, Biagio Lucini, Gert Aarts

### Summary

Formulates neural network training as the stochastic evolution of an
initially random matrix ensemble driven by SGD updates. Demonstrates that
training induces a Baik-Ben Arous-Peche (BBP) transition where isolated
eigenvalues detach from the random bulk distribution, providing a
dynamical framework for representation formation.

The key contribution is an analytically solvable linear teacher-student
model where spectral evolution is fully tractable. From this model, the
authors derive a phase diagram of trainability governed by two parameters:
step size (learning rate) and initial weight variance. The BBP threshold
defines the boundary between a "trainable" regime (where the signal
eigenvalue successfully detaches) and an "untrainable" regime (where it
remains trapped in the bulk).

### Key findings

- The BBP transition provides a principled notion of "representation
  emergence" -- the exact step at which a learned feature becomes
  spectrally distinguishable from noise.
- The phase diagram reveals a sharp boundary in (step size, init variance)
  space: too-large init variance prevents the signal eigenvalue from
  detaching regardless of step size, while too-small step size slows
  detachment below practical training horizons.
- The analytical predictions are validated on multi-layer networks beyond
  the teacher-student setting.

### Relevance to Gradience

**Medium -- theoretical deepening of phase transition framework.**

1. **Phase transition formalization.** Gradience's `phase_transitions.py`
   module detects phase transitions empirically (autocorrelation time,
   variance ratio, rank jumps). This paper provides the theoretical
   substrate: the BBP threshold defines WHEN a learned representation
   should become detectable, and thus when Gradience's spectral metrics
   should show a transition. This is the dynamical complement to the
   Spectral Edge Thesis (arXiv:2603.28964), which focuses on spectral gap
   dynamics rather than eigenvalue detachment.

2. **Connection to the Intruder Threshold.** The BBP transition is the
   same mathematical object underlying the Intruder Threshold paper
   (2607.23711 above), but studied in the forward direction (representation
   emergence) rather than the destructive direction (intruder creation).
   Together, the two papers bracket the BBP transition from both sides:
   too little signal = no learning; too much signal = intruder catastrophe.

3. **Trainability diagnostics.** The phase diagram could inform
   Gradience's `finetune/` module: estimating whether a given
   (learning rate, init scheme) will produce a detectable spectral
   transition within the planned training horizon. This is predictive
   rather than diagnostic, but the spectral framework is shared.

**Criticality: MEDIUM.** Theoretically important for grounding
Gradience's empirical phase transition observations in rigorous random
matrix theory, but the linear teacher-student setting is a significant
abstraction from practical LoRA fine-tuning. The practical connection is
indirect: Gradience would need to validate that BBP-predicted transition
points correspond to observed effective-rank jumps in real LoRA training
runs. Most useful as theoretical scaffolding for interpreting existing
grokking and phase transition findings (experiments/grokking_long/
FINDING_rank_order_parameter.txt) rather than as an immediate integration
target. Recommended action: add to RESEARCH_INVENTORY.md at "Medium"
priority alongside the Spectral Edge Thesis.

---

## Summary Table

| Paper | arXiv | Relevance | Criticality | Recommended Action |
|-------|-------|-----------|-------------|--------------------|
| The Intruder Threshold | 2607.23711 | Unifies BBP theory with LoRA spectral diagnostics; closed-form per-layer threshold directly computable from existing SVD pipeline | **HIGH** | Add to RESEARCH_INVENTORY at High priority; prototype threshold computation on existing audit data |
| HiP-LoRA | 2604.17751 | Validates spectral partitioning causally; merge robustness protocol applicable to audit validation | **MEDIUM-HIGH** | Add to RESEARCH_INVENTORY; assess MergeFail protocol for validation experiments |
| Spectral Phase Transitions | 2606.28486 | Rigorous BBP formalization; complements Spectral Edge Thesis and Intruder Threshold | **MEDIUM** | Add to RESEARCH_INVENTORY alongside Spectral Edge Thesis for theoretical context |

### Also noted (not in top 3)

- **SpectralLoRA** ([2604.10649](https://arxiv.org/abs/2604.10649)):
  SVD-DCT correlation analysis shows strong Pearson correlation between
  SVD energy concentration and DCT compressibility. Relevant to rank
  policy work but less novel for Gradience's existing framework.

- **SSR-Merge** ([2606.10617](https://arxiv.org/abs/2606.10617), ICML
  2026): Subspace signal routing for LoRA merging via decorrelation and
  directional routing. Relevant to merge pipeline but requires calibration
  data (violates training-free regime) and focused on diffusion models.

- **SAD-LoRA** ([2607.04306](https://arxiv.org/abs/2607.04306)): Spectral
  alignment for knowledge distillation via LoRA. Adjacent but focused on
  distillation rather than post-hoc analysis.
