# Research Scan — 2026-09-04

Automated scan for new empirical research and technical documentation relevant to the Gradience spectral analysis framework.

## 1. The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning

- **Authors:** Peng Xie (TU Munich, Campus Heilbronn)
- **Date:** July 26, 2026
- **ArXiv:** [2607.23711](https://arxiv.org/abs/2607.23711)

### Summary

Derives a per-layer critical update strength `s* = θ̄ / (γ · σ₁(BA))` from the rectangular spiked-deformation transform (a BBP-type random matrix theory result), predicting exactly when LoRA fine-tuning creates "intruder dimensions" — new leading singular vectors of the updated weight `W + BA` that are nearly orthogonal to all pretrained singular vectors and drive catastrophic forgetting. The law is computed entirely from the measured spectrum of `W` with no fitted parameters. Validated on 4 dense Transformer families, a state-space model, a mixture-of-experts model, and an encoder-decoder (18 adapters, 9,840 layer scans): localizes the empirical threshold within 2x on 82% of layers, separates intruder-bearing from intruder-free layers with mean AUC 0.89.

### Relevance to Gradience

**Direct and deep.** This paper provides a parameter-free, per-layer spectral law grounded in the same random matrix theory family (Marchenko-Pastur / BBP transitions) that Gradience already uses for its `optimal_hard_threshold` rank policy. Specifically:

- **Spectral audit pipeline:** The intruder threshold `s*` could be integrated as a new diagnostic in `vnext/audit/lora_audit.py`, flagging layers where the LoRA update magnitude approaches or exceeds the critical strength. This is a natural complement to the existing energy-rank and stable-rank metrics.
- **Merge compatibility:** Intruder dimensions are precisely the kind of structural pathology that Gradience's merge pipeline diagnoses empirically (V-module pathology in the conjunctive failure model). A per-layer intruder prediction could strengthen the verdict decision tree in `vnext/merge/verdicts.py`.
- **Rank policies:** The critical strength formula provides a theoretically grounded upper bound on safe update magnitude per layer, which could inform rank allocation policies.
- **Open questions:** Connects to Gradience's open work on spectral scaling laws and the curvature-partition correspondence — the BBP transition is the formal mechanism underlying the spectral partition sharpening Gradience observes empirically.
- **Validation overlap:** Uses the same RMT framework as Xu's Spectral Edge Thesis (arXiv:2603.28964), already cited in Gradience's THEORY.md.

### Criticality Rating: CRITICAL

This is the strongest candidate for integration. The intruder threshold provides a missing theoretical piece that could turn Gradience's empirical spectral audit into a theory-backed diagnostic with predictive power for catastrophic forgetting risk. The parameter-free nature and broad validation (9,840 layer scans) make it immediately actionable.

---

## 2. Tight Sample Complexity for Low-Rank Adaptation: Matching Bounds and Rank Selection

- **Authors:** (Multiple, see paper)
- **Date:** July 30, 2026
- **ArXiv:** [2607.27680](https://arxiv.org/abs/2607.27680)

### Summary

Establishes the first matching upper and lower bounds for LoRA generalization: `Θ(rd/n)` excess risk for rank-r adaptation over n samples with dimension d. Key results:

1. **Upper bound** via local Rademacher complexity: `Õ(rd/n)` for the empirical risk minimizer over rank-r LoRA.
2. **Lower bound** via Fano-type packing: `Ω(rd/n)` — proves the upper bound is minimax-optimal.
3. **Rank selection theory:** For constrained ERM, the optimal rank equals the intrinsic rank `r*`, and over-ranking strictly hurts. For adaptive estimators (nuclear-norm-then-truncate), over-ranking is harmless and the rate saturates at `Θ̃(r*d/n)`.

Validated on synthetic trace regression and real LoRA fine-tuning (DistilBERT, RoBERTa on SST-2 and MRPC).

### Relevance to Gradience

**Directly answers a stated open research question.** Gradience's ROADMAP.md lists "generalization bounds from spectra" and "PAC-Bayes bounds for the LoRA parameterization" as open questions. This paper resolves the core complexity-theoretic version:

- **Theoretical foundation for rank utilization finding:** Gradience's central empirical discovery — mean rank utilization of 0.172, median 50% of allocated rank capturing 90% of spectral energy — is now backed by a formal result: over-ranking strictly increases excess risk (for constrained ERM). The intrinsic rank `r*` is what matters, not the nominal rank, which is exactly what Gradience's energy-rank and stable-rank metrics measure.
- **Rank policy validation:** The nuclear-norm-then-truncate result validates the family of approaches Gradience already implements (energy threshold, optimal hard threshold policies). These are robust to over-specification of rank.
- **Quantitative risk framework:** The `Θ(rd/n)` scaling law provides a sample-complexity angle to complement Gradience's spectral metrics, potentially enabling sample-aware rank recommendations.
- **Validation on same architectures:** Uses DistilBERT and RoBERTa on SST-2 — the same encoder-scale / task combinations that Gradience's field trials validated on.

### Criticality Rating: HIGH

While less immediately actionable than the Intruder Threshold (it's a theoretical result, not a new diagnostic), it provides the formal generalization-theoretic foundation that Gradience's empirical program has been building toward. Could be cited as theoretical justification for the entire rank-suggestion framework and referenced in the technical report.

---

## 3. Spectral Phase Transitions and Trainability in Neural Network Learning Dynamics

- **Authors:** Chanju Park, Dario Bocchi, Francesco D'Amico, Biagio Lucini, Gert Aarts
- **Date:** June 26, 2026
- **ArXiv:** [2606.28486](https://arxiv.org/abs/2606.28486)

### Summary

Formulates neural network training as the stochastic evolution of an initially random matrix ensemble driven by SGD. Derives analytically that training induces a Baik-Ben Arous-Péché (BBP) transition — isolated eigenvalues detach from the random bulk distribution — providing a first-principles dynamical framework for representation formation. Demonstrates this in a solvable linear teacher-student model where spectral evolution is analytically tractable. Obtains a phase diagram of trainability governed by step size (learning rate) and initial weight variance.

### Relevance to Gradience

**Provides theoretical foundation for existing capability.** Gradience's research module (`gradience/research/phase_transitions.py`) already implements phase transition detection via critical slowing down, fluctuation amplification, and susceptibility metrics. This paper provides the first-principles derivation of *why* those signals work:

- **BBP transition as mechanism:** The detachment of isolated eigenvalues from the bulk is the formal mechanism underlying Gradience's empirical observation of the "three-act training structure" (Explore → Lock-on → Destabilize). The Lock-on phase corresponds to eigenvalue emergence above the BBP threshold.
- **Curvature-partition correspondence:** Gradience's open question about whether Hessian collapse and spectral partition sharpening are the same event is directly illuminated by this framework — both would be manifestations of the BBP transition viewed through different spectral lenses (Hessian vs. weight matrix).
- **Spectral Edge Thesis connection:** Strengthens the theoretical basis for Xu's Spectral Edge Thesis (arXiv:2603.28964, already cited in Gradience) by providing the dynamical mechanism. The rolling-window Gram matrix spectral gap that Xu identifies is the empirical signature of the BBP transition Park et al. derive analytically.
- **Trainability phase diagram:** The learning-rate / initialization phase diagram could inform Gradience's finetune alert system (`gradience/finetune/`), providing theoretically grounded thresholds for training instability detection.

### Criticality Rating: MEDIUM-HIGH

The theoretical contribution is significant but less immediately actionable than the other two papers. The linear teacher-student model is a simplification — extending to the LoRA setting requires additional work. However, it provides the theoretical spine connecting several of Gradience's existing capabilities and open questions. Most valuable as a citation and conceptual framework rather than a direct implementation target.

---

## Honorable Mentions

Papers that are relevant but less directly impactful:

- **"Norm-Bounded Low-Rank Adaptation"** (NB-LoRA, arXiv:2501.19050, Jan 2025): Parameterization admitting explicit singular value bounds. Relevant to Gradience's spectral control but more of an engineering contribution than a theoretical one.
- **"How Much Rank Does LoRA Need? Rank-Error Bounds for Transformer Attention"** (arXiv:2608.26052, Aug 2026): Task-dependent rank-error bounds for attention layers specifically. Complementary to paper #2 but narrower in scope.
- **"Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging"** (OSRM, arXiv:2505.22934, ACL 2025): Proposes constraining LoRA subspaces pre-training to reduce merge interference. Validates Gradience's subspace-overlap diagnostic but proposes a training-time solution rather than a post-hoc analysis tool.
