# Research Scan — 2026-10-14

Automated scan for new empirical research and technical documentation highly relevant to the Gradience spectral-analysis framework. Papers below are **not yet tracked** in `RESEARCH_INVENTORY.md` or referenced in `docs/THEORY.md`.

---

## 1. "At-Grok Is Not Converged: A Measurement-Validity Audit for Grokking Representation Metrics"

**Authors:** Truong Xuan Khanh
**Date:** July 2026
**Link:** [arXiv:2607.06639](https://arxiv.org/abs/2607.06639)

### Summary

This paper performs a psychometric-style validity audit on the spectral representation metrics commonly used to characterize grokking — exactly the kind of measurement-instrument analysis that Gradience's THEORY.md §6 calls for but notes no other program in the modular-adaptation space has attempted.

**Key findings:**

- Effective rank measured at the grokking transition **overstates the converged value by 3–5× on MLPs and 1.3–1.5× on transformers**. Reading the metric at the accuracy transition, as most grokking studies do, gives a systematically biased picture of representation structure.
- Representation compression **trails** the accuracy transition by roughly the time-to-grok (~10,000+ steps), rather than coinciding with it. This matters for any study treating spectral compression as an order parameter for grokking.
- A one-variable ablation (adding LayerNorm) changed the fraction of compression completed by the grok step from 0.87 to 0.25, demonstrating that architectural choices modulate when spectral metrics become informative.
- The paper releases an audit toolkit that separates onset from compression, flags right-censoring, excludes boundary cells that never fully generalize, and checks that the reference floor has plateaued — directly operationalizing the SEM/MDC measurement framework Gradience advocates.

### Relevance to Gradience

**Direct and fundamental.** This paper validates and extends Gradience's most distinctive contribution: the measurement-instrument stance toward spectral diagnostics. Gradience's THEORY.md §6 explicitly calls out the need for test-retest reliability, standard error of measurement, and minimal detectable change for spectral metrics. This paper delivers the first external implementation of that program, applied to grokking representation metrics. The finding that at-grok measurements are systematically biased has immediate implications for:

- Gradience's `PhaseTransitionTracker` and `GrokDetector` modules (`gradience/research/phase_transitions.py`), which may need convergence-checking before reporting spectral values
- The grokking experiments in `experiments/grokking/`, where effective rank trajectories are interpreted as order parameters
- The broader reliability claims in FINDINGS.md Part D about spectral observables as phase transition indicators

### Criticality: **HIGH**

This is the most directly relevant paper found. It independently validates Gradience's core methodological thesis while providing concrete evidence that the community's standard measurement practice produces biased estimates. The audit toolkit could be adapted for Gradience's own measurement pipeline. Should be tracked in `RESEARCH_INVENTORY.md` and its findings considered for integration into the phase-transition detection framework.

---

## 2. "Rank-Accuracy Trade-off for LoRA: A Gradient-Flow Analysis"

**Authors:** Michael Rushka, Diego Klabjan
**Date:** February 2026
**Link:** [arXiv:2602.10212](https://arxiv.org/abs/2602.10212)

### Summary

This paper provides a formal dynamical-systems proof of a claim central to Gradience's spectral framework: under gradient flow with spectral initialization, the singular values of the LoRA product BA converge to the **top-r singular values of W₀**, achieving the Eckart–Young–Mirsky optimal rank-r approximation.

**Key contributions:**

- Reduces the LoRA training dynamics to scalar ODEs for the singular values of the product Y(t)X(t), proving convergence to the top-r singular values of the pretrained weight matrix.
- Shows that the gradient-flow equations are identical whether LoRA's A and B parameters are updated simultaneously or sequentially — a result that simplifies analysis of LoRA training dynamics.
- Derives an explicit rank-accuracy trade-off: the residual error of rank-r LoRA is determined by σ_{r+1}(W₀), directly linking compression potential to the pretrained weight spectrum's tail decay.
- Builds on Xu et al. (AISTATS 2025), which showed that misalignment between the singular spaces of the pretrained model and the target matrix affects final error.

### Relevance to Gradience

**Theoretical grounding for core empirical findings.** This paper provides the formal justification for several phenomena Gradience has documented empirically:

- The finding that LoRA adapters typically use ~1/6 of allocated rank (FINDINGS.md Part A) follows directly from the gradient-flow convergence: if W₀'s spectrum decays rapidly, the LoRA product will concentrate on the top few singular values regardless of nominal rank.
- The explicit σ_{r+1} error bound informs Gradience's rank selection policies (`vnext/audit/rank_policies.py`), particularly the energy@0.90 and Gavish-Donoho policies, by providing a principled target for where to truncate.
- The spectral initialization connection addresses the open question in THEORY.md §7 about why adapter spectra converge to a characteristic shape — they're attracted to the top eigenspace of W₀.
- The result strengthens the theoretical basis for the concentration-weighted convergence bound Gradience is pursuing (THEORY.md §7.5, listed as genuinely open): the gradient-flow dynamics naturally weight convergence by spectral mass.

### Criticality: **HIGH**

This provides the missing theoretical underpinning for Gradience's core spectral observations. The explicit residual-error formula (σ_{r+1}) could be integrated into rank suggestion logic. The gradient-flow convergence proof also constrains the open question about spectrum universality. However, caveat: the proof assumes spectral initialization and gradient flow (no stochasticity), so empirical validation on SGD-trained adapters is needed before fully relying on its conclusions. Should be tracked and its implications assessed against the empirical record.

---

## 3. "Spectral Entropy Collapse as a Phase Transition in Delayed Generalisation: An Interventional and Predictive Framework for Grokking"

**Authors:** Truong Xuan Khanh et al.
**Date:** April 2026
**Link:** [arXiv:2604.13123](https://arxiv.org/abs/2604.13123)

### Summary

This paper identifies the normalized spectral entropy of the representation covariance matrix as a phase-transition order parameter for grokking, complementing Gradience's existing use of effective rank and autocorrelation time.

**Key findings:**

- Proposes a two-phase account of grokking: a **norm-expansion phase** followed by **spectral entropy collapse**, where entropy crossing a task-specific critical threshold (~0.609 in their experiments) precedes generalization.
- Across three modular-arithmetic tasks and 10 seeds, the entropy threshold was crossed in every run, approximately 1,020 steps before generalization onset — providing a predictive window.
- An **interventional experiment** (representation-mixing to prevent entropy collapse) successfully delayed grokking, with a norm-matched control ruling out parameter norm as the driving mechanism. This is one of the few causal (not just correlational) results in the grokking literature.
- The authors note their causal claims fall short of a full Pearl/Rubin framework.

### Relevance to Gradience

**Complementary to existing phase-transition detection.** Gradience's `PhaseTransitionTracker` already uses three independent spectral metrics (effective rank, autocorrelation time, variance ratio) for phase-transition detection (experiments/grokking/). This paper adds a fourth candidate — spectral entropy of the representation covariance — with the advantage of having an interventional validation (not just correlation).

- The ~1,020-step predictive window could inform the Hessian-leads-accuracy finding in FINDINGS.md §16a, providing a representation-side counterpart to the curvature-side 3–6 update lead.
- The norm-expansion → entropy-collapse sequence may correspond to the explore → lock-on → destabilize three-act structure documented on Mistral-7B (FINDINGS.md Part D).
- However, the measurement-validity audit (Paper #1 above) by the same lead author cautions that spectral measurements at the grokking transition may be systematically biased — the two papers should be read together.

### Criticality: **MEDIUM-HIGH**

This adds a validated spectral observable (with causal evidence) to Gradience's phase-transition toolkit. The interventional design is stronger than the purely observational evidence Gradience currently relies on. However, the same-author measurement-validity audit partially undermines the predictive precision claimed here, and the modular-arithmetic setting is narrow. Worth tracking and evaluating for integration into the phase-transition framework, but the measurement-validity paper (Paper #1) takes priority.

---

## Methodology Note

Search covered arXiv preprints and published proceedings from late 2025 through October 2026. Papers already tracked in `RESEARCH_INVENTORY.md` or referenced in `docs/THEORY.md` were excluded (including Xu 2026/Spectral Edge Thesis, KnOTS/ICLR 2025, Task Singular Vectors/CVPR 2025, Akbar et al./ICML 2025, Medina & Sorensen 2025, Tian et al./ICLR 2026, Badirli et al. 2025, and Tang et al./Crowded in B-Space 2026). All papers listed are unreviewed arXiv preprints unless otherwise noted; findings should be verified before integration.
