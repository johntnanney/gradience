# Literature Scan: 2026-11-07

**Scope:** Recent (2025-2026) empirical research and technical papers relevant to Gradience's spectral analysis framework for LoRA fine-tuning dynamics.

**Method:** Extended web search across LoRA spectral analysis, merge compatibility, phase transitions, rank dynamics, and training geometry.

---

## Paper 1: LoRA-Lens — Training Induces Spectral Compression in Low-Rank Adapters

- **Venue:** ICML 2026
- **Link:** https://icml.cc/virtual/2026/72398

### Summary

LoRA-Lens introduces a "capacity utilization" diagnostic that measures the fraction of an adapter's rank budget carrying meaningful spectral energy. Across five models from three families, the authors observe that training compresses adapter spectra: capacity utilization drops from 1.0 to 0.43-0.83. The V-projection compresses 2-3x more than the Q-projection. Crucially, the authors show this compression does not reliably indicate redundancy: V's `lora_B` parameters are 30-350x more Fisher-sensitive than Q's, consistent with the OV circuit's role as the content channel. They propose FisherLoRA, which keeps V at full rank and reduces Q, cutting Q/V adapter parameters by 38% with worst-case degradation of 0.006.

### Relevance to Gradience

**Critical — directly validates and extends core metrics.**

- Gradience's `energy_rank_90` and stable rank metrics are conceptually equivalent to LoRA-Lens's "capacity utilization." This paper provides independent large-scale validation of the spectral compression phenomenon Gradience measures.
- The V-vs-Q asymmetry finding (2-3x compression difference) directly informs Gradience's per-layer rank suggestion policies. Current policies treat Q and V projections symmetrically; this evidence suggests projection-type-aware rank allocation.
- The Fisher sensitivity finding (V's lora_B 30-350x more sensitive than Q) bridges Gradience's two instrument classes: SVD-based spectral audits and Hessian/Fisher curvature telemetry. This is direct evidence for the open research question about curvature-partition correspondence (whether Hessian structure correlates with spectral partition sharpening).
- The "compression does not mean redundancy" finding is a caution for Gradience's compression validation pipeline in `bench/`: spectral energy concentration alone may not be a safe signal for rank reduction.

### Integration Priority: HIGH

Immediate implications for rank suggestion policies (projection-type awareness) and theoretical framework (curvature-partition correspondence). The Fisher-spectral bridge result should be cited in the technical report's open frontiers section.

---

## Paper 2: Spectral Phase Transitions and Trainability in Neural Network Learning Dynamics

- **Authors:** Park, Bocchi, D'Amico, Lucini, Aarts
- **Venue:** arXiv preprint, June 2026 (2606.28486); cross-listed cond-mat.dis-nn, cs.LG, hep-lat
- **Link:** https://arxiv.org/abs/2606.28486

### Summary

This paper frames SGD training as a stochastic evolution of an initially random weight-matrix ensemble, where gradient updates reshape the spectral bulk. The authors formalize this as a Baik-Ben Arous-Péché (BBP) transition: isolated eigenvalues separate from the random bulk during training, marking the onset of learned representations. They derive a trainability phase diagram controlled by step size and initial weight variance in a solvable linear teacher-student model, then extend numerical simulations to nonlinear settings showing spectral alignment emerging during training.

### Relevance to Gradience

**High — provides theoretical grounding for phase transition detection.**

- Gradience already uses Marchenko-Pastur random matrix theory for signal/noise partitioning of LoRA spectra. This paper adds a dynamical dimension: the BBP transition describes *when* and *how* eigenvalues separate from the bulk during training, which is exactly what Gradience's phase transition detection aims to capture.
- The trainability phase diagram (step size vs. initial variance) could inform Gradience's Hessian-based leading indicator framework. If the BBP transition has a predictable onset point, this gives a theoretical basis for the 3-6 update leading indicator window Gradience has observed empirically.
- The paper's formalism connects statistical mechanics (which Gradience references in its phase transition framework: critical slowing down, susceptibility, order parameters) with random matrix theory (which Gradience uses operationally). This bridges two theoretical pillars that are currently somewhat separate in Gradience's framework.
- **Limitation:** The analytical results are for linear teacher-student models. The nonlinear extensions are numerical, and no LoRA-specific experiments are reported. The connection to fine-tuning (as opposed to training from scratch) would need to be established.

### Integration Priority: MEDIUM-HIGH

The BBP transition formalism should be reviewed for compatibility with Gradience's existing MP-based signal/noise partitioning. If the dynamical extension holds for fine-tuning spectra, it would provide theoretical grounding for the empirical leading-indicator results. Worth citing in the theory document's phase transition section, with a note that LoRA-specific validation is needed.

---

## Paper 3: Not All Ranks Are Equal — Budget-Aware LoRA Merging Across Tasks

- **Authors:** Amballa et al.
- **Venue:** arXiv preprint, September 2026 (2609.22237)
- **Link:** https://arxiv.org/pdf/2609.22237

### Summary

This paper argues that uniform rank allocation across layers and tasks is a primary source of the performance gap between merged LoRA adapters and per-task LoRAs. The authors propose "Net Utility," a data-free metric for non-uniform rank allocation that distributes rank budget based on per-layer importance. They report +2.1% improvement on vision tasks and +2.2% on language tasks over uniform-budget baselines, with a maximum gain of +3.8 over DARE at rank 64. The method is data-free and works as a post-hoc allocation step on top of existing merge strategies.

### Relevance to Gradience

**Medium-High — validates non-uniform rank allocation for merging and offers a benchmark comparison point.**

- Gradience's merge pipeline (`merge_audit` -> `diagnose_pair` -> `recommend_merge`) already considers per-layer compatibility, but the merge recommendation currently targets strategy selection (linear, ties, dare_ties, etc.) rather than per-layer rank allocation. Net Utility's data-free approach is complementary: Gradience's spectral diagnostics (subspace overlap, principal angles) could serve as an alternative or combined signal for rank allocation during merging.
- The paper's finding that uniform-rank merging is suboptimal validates the design intuition behind Gradience's per-layer `layer_diagnoses` and `layer_verdicts` — different layers have different compatibility profiles and should be treated differently.
- **Benchmark opportunity:** Net Utility is data-free and reported on standard benchmarks. Gradience's spectral-based rank suggestion could be evaluated on the same benchmarks for a direct comparison, which would quantify the value of spectral information over simpler utility heuristics.
- The +3.8 gain over DARE specifically is relevant since Gradience's merge pipeline includes `dare_ties` and `dare_linear` as strategy recommendations.

### Integration Priority: MEDIUM

The core claim (non-uniform rank allocation improves merging) aligns with Gradience's architecture but does not introduce new theoretical machinery. The main value is as a benchmark comparison target and as validation for the per-layer approach Gradience already takes. Consider evaluating Gradience's spectral rank suggestions against Net Utility on the same benchmarks.

---

## Notable Mentions (Lower Priority)

**SpectralLoRA** (arXiv 2604.10649): Uses 2D DCT (not SVD) to analyze LoRA update frequency structure. Finds 33% of DCT coefficients capture 90% of spectral energy. Interesting as an alternative spectral decomposition, but results are limited to BERT-base/RoBERTa-base on GLUE. The SVD-DCT correlation (r=0.906) suggests both approaches capture similar structure. Lower priority because Gradience's SVD-based framework is more general and better established.

**What Does the Rank Buy?** (arXiv 2609.32002): Argues rank controls adaptation cost rather than capacity, with implicit low-nuclear-norm bias regardless of rank. Relevant to theoretical framing but limited details available from preprint stage.

**KnOTS / Model Merging Survey** (ICLR 2025 / arXiv 2603.09938): KnOTS uses SVD to map LoRA updates into shared space for merging. The 2026 survey frames LoRA merging as raising questions about rank allocation and subspace compatibility — language closely aligned with Gradience's framework. Useful as related work context.
