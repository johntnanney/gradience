# Literature Scan — October 31, 2026

**Scope**: New empirical research and technical documentation highly relevant
to the Gradience research program, published since the last RESEARCH_INVENTORY
update (April 2026).

**Method**: Extended web search across arXiv, ACL Anthology, PMLR, and
secondary aggregators. Search terms targeted Gradience's five core research
strands: spectral analysis of LoRA adapters, merge compatibility prediction,
Hessian/curvature training dynamics, spectral partitioning theory, and
random matrix theory for fine-tuning.

---

## Article 1: A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs

| Field | Value |
|-------|-------|
| Authors | Dayal Singh Kalra et al. (Meta/FAIR) |
| arXiv | [2601.16979](https://arxiv.org/abs/2601.16979) |
| Venue | Likely NeurIPS 2026 (per third-party notes; verify) |
| Code | [facebookresearch/scalable-curvature](https://github.com/facebookresearch/scalable-curvature) |
| Date | January 2026 |

### Summary

Proposes *critical sharpness* (λ_c = 2/η_c, where η_c is the smallest
learning rate that causes the training loss to increase in the next step)
as a computationally efficient proxy for Hessian sharpness, requiring fewer
than 10 forward passes given the update direction Δθ. Also introduces
*relative critical sharpness*, which quantifies the curvature of one loss
landscape while optimizing another — specifically designed to analyze the
pre-training → fine-tuning transition and guide data mixing strategies.
Demonstrated at scale up to 7B parameters (OLMo-2 models). Captures
well-documented Hessian phenomena including progressive sharpening and
Edge of Stability.

### Relevance to Gradience

**Direct.** Gradience's curvature telemetry pipeline (FINDINGS.md §16a)
relies on Hessian eigenvalue computation via power iteration and
Hutchinson estimation. This is computationally expensive and has only been
validated at GPT-2 small (124M) and DeBERTa-v3-base (185M) scale. Critical
sharpness offers a dramatically cheaper alternative that:

1. **Scales the curvature lead-lag finding.** The §16a result (Hessian
   energy leads validation accuracy by 3–6 updates) is Gradience's strongest
   during-training diagnostic. If critical sharpness captures the same
   lead-lag signal, it can be measured during LoRA fine-tuning of 7B+
   models — currently infeasible with full Hessian estimation.

2. **Operationalizes relative curvature.** The "relative critical sharpness"
   concept directly addresses the pre-training → fine-tuning transition,
   which is the setting where Gradience's curvature telemetry operates.
   It could provide a principled measure of how much the fine-tuning
   landscape diverges from the pre-training landscape — a signal currently
   absent from the telemetry pipeline.

3. **Connects to the unified measurement framework.** THEORY.md §5.3
   envisions switching from Hessian telemetry during training to SVD audit
   after training. Critical sharpness lowers the cost of the during-training
   half, making the unified framework practical at production scale.

**Caveats.** The paper focuses on pre-training dynamics, not LoRA
fine-tuning specifically. Whether critical sharpness captures the same
information as Hessian eigenvalue spectra in the low-rank adaptation
setting requires a dedicated replication. The reduction from a full
eigenspectrum to a single scalar (η_c) may lose the multi-modal structure
that Gradience's Hessian telemetry leverages (e.g., the spectral gap
dynamics in §16a).

### Integration Priority

**High — replicate before integrating.** The natural test: compute critical
sharpness alongside the existing Hessian telemetry in a re-run of the
N07 Experiment B protocol, and check whether the CCF lead-lag signal
transfers. If it does, critical sharpness becomes the production-grade
curvature instrument for Gradience's during-training monitoring; if it
doesn't, the scalar reduction is too lossy for Gradience's purposes.
Code is public, which lowers the replication barrier.

---

## Article 2: Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation

| Field | Value |
|-------|-------|
| Authors | Hao Gu, Mao-Lin Luo, Zi-Hao Zhou, Han-Chen Zhang, Min-Ling Zhang, Tong Wei |
| arXiv | [2602.00722](https://arxiv.org/abs/2602.00722) |
| Venue | ICML 2026 (PMLR vol. 306, pp. 37137–37157) |
| Code | EBLoRA (linked from arXiv) |
| Date | January 2026 |

### Summary

Demonstrates that LoRA updates have uneven singular value spectra: a few
directions carry most of the adaptation energy. The paper's central claim
is that these dominant directions are the ones most exposed to interference
from subsequent tasks in continual learning. The authors formalize this as
a *spectral imbalance* problem: the magnitude of each rank-one component
(set by its singular value) determines both its adaptation contribution
and its vulnerability to catastrophic forgetting. Their fix separates the
update's magnitude from its direction, solving a constrained problem on a
restricted Stiefel manifold via projected first-order optimization.
Reported improvements in both backward and forward forgetting relative to
continual learning baselines.

### Relevance to Gradience

**Direct and theoretically significant.** This paper independently validates
and extends three core Gradience findings:

1. **Spectral concentration is the norm, not the exception.** Gu et al.'s
   observation that "a few directions carry most of the adaptation energy"
   replicates Gradience's §§1–3 findings (mean utilization ~0.17, median
   50% compression potential) from a completely different research program.
   The ICML peer review adds credibility to what was previously Gradience's
   solo empirical claim at scale.

2. **Dominant directions are interference-prone.** This directly connects
   to Gradience's spectral partitioning findings (§§11–14). Gradience
   found that high-SV directions are *shared* across same-task adapters
   (7.8× alignment ratio); Gu et al. show that these same high-SV
   directions are the primary locus of *inter-task interference*. Together,
   the two results tighten the theoretical picture: high-SV directions
   are where compatibility and conflict both live. This has direct
   implications for the concentration-weighted convergence bound
   (THEORY.md §7.2 OQ1) — any formal bound must account for the dual
   role of dominant directions.

3. **Magnitude-direction separation for merge analysis.** The Stiefel
   manifold formulation — separating *what direction* the update moves
   in from *how much energy* it carries — is a natural framework for
   analyzing Gradience's magnitude_ratio and frobenius_ratio diagnostics.
   Gradience currently treats magnitude and direction as correlated but
   operationally separate signals; Gu et al. provide a geometric framework
   that formalizes this separation.

**Caveats.** The paper addresses continual learning (sequential tasks),
not simultaneous merging. Whether the spectral imbalance mechanism applies
to the pairwise merge setting (where both adapters are applied at once)
requires additional analysis. The Stiefel manifold method is a training-time
intervention, which is orthogonal to Gradience's training-free CPU audit
regime.

### Integration Priority

**High — cite and integrate theoretically.** The paper should be cited in
THEORY.md §6 (spectral partitioning) as independent ICML-published evidence
for the dual role of dominant singular directions. The magnitude-direction
separation framework warrants a dedicated section in the theoretical
foundations. No infrastructure investment needed — the value is conceptual,
not computational. The ICML venue gives this the strongest peer-review
provenance of any paper in the external references.

---

## Article 3: Demystifying Mergeability: Interpretable Properties to Predict Model Merging Success

| Field | Value |
|-------|-------|
| Authors | Luca Zhou, Bo Zhao, Rose Yu, Emanuele Rodolà |
| arXiv | [2601.22285](https://arxiv.org/abs/2601.22285) (v9, June 2026) |
| Venue | Preprint (9 revisions suggest active development) |
| Date | January 2026 (v1); June 2026 (v9) |

### Summary

Uses L1-regularized linear models over 28 interpretable pairwise metrics
(in five families: gradient-based, weight-based, activation-based,
Fisher-based, task-specific) to predict post-merge normalized accuracy
across five merging methods. Evaluated on 190 task pairs from 20 fine-tuned
CLIP ViT-B/16 encoders. Central finding: mergeability is not a fixed
property of a model pair but depends on the specific merging algorithm.
Different merge methods rely on different model properties for success.
Gradient alignment metrics consistently emerge as the most fundamental
method-agnostic signal.

### Relevance to Gradience

**Direct competitor and potential validator.** This paper occupies the same
cell as Gradience on Axis 4 (Purpose: prediction/triage) of the conceptual
map (THEORY.md §8):

1. **Method-dependent mergeability validates audit-aware strategy
   selection.** Gradience's recommendation engine selects merge strategies
   per-pair (linear, norm-equalized, audit-aware) based on structural
   diagnosis. Zhou et al.'s finding that different algorithms need
   different predictors is independent evidence that strategy-aware
   triage is the right architecture — a single "merge score" is
   insufficient.

2. **Gradient alignment as the fundamental signal.** The finding that
   gradient-based metrics dominate connects to Gradience's curvature
   telemetry program (§16a). If gradient alignment is the deepest
   predictor, then Gradience's Hessian-based curvature signals (which
   are second-order derivatives of the loss, closely related to gradient
   structure) may be measuring the right object from a different vantage
   point. This strengthens the case for curvature telemetry as a
   *during-training* mergeability predictor.

3. **Metric family comparison.** The 28-metric, 5-family framework
   provides a systematic baseline for evaluating Gradience's metric
   choices. Gradience uses primarily weight-delta metrics (stable rank,
   energy rank, subspace overlap, SV-weighted alignment) with optional
   Fisher diagnostics. Zhou et al.'s finding on which metric families
   matter for which merge methods could inform which Gradience diagnostics
   to emphasize in different strategy recommendations.

**Caveats.** Vision transformers (CLIP ViT-B/16), not NLP models. 9
revisions suggest the paper is still evolving. The evaluation uses
full fine-tuning, not LoRA specifically. The L1-regularized linear model
may miss nonlinear interactions that Gradience's conjunctive failure
model captures. No peer-reviewed venue yet.

### Integration Priority

**Medium — track and cite selectively.** The paper's framework is useful
as a methodological reference but the substrate difference (vision vs.
NLP, full FT vs. LoRA) limits direct comparison. The method-dependence
finding should be cited in the technical report's related work (§8) as
independent evidence for strategy-aware triage. The gradient-alignment
finding should be noted in THEORY.md §5.2 as convergent evidence linking
gradient structure to mergeability. No infrastructure investment warranted
until the paper stabilizes (it's at v9) and/or until a LoRA-specific
replication appears. The companion paper "Predicting Mergeability of PEFT
Updates" (arXiv 2606.19549, Tang et al., June 2026) — which claims
early-training mergeability prediction specifically for LoRA — may be the
more directly actionable reference but has a narrower evaluation base.

---

## Additional Papers Noted (Not Top-3, But Worth Tracking)

| Paper | arXiv | Relevance | Priority |
|-------|-------|-----------|----------|
| From SGD to Spectra: A Theory of Neural Network Weight Dynamics (Olsen et al., 2025) | [2507.12709](https://arxiv.org/abs/2507.12709) | SDE framework for weight spectral evolution; Dyson Brownian motion for squared SVs. Candidate theoretical foundation for spectral universality question (THEORY.md §7.2). | Low — theoretical; no LoRA-specific validation |
| Can Spectral-Clipping Enable Better Learning While Forgetting Less for LoRA? (Wi & Park, ACL 2026) | [2608.12332](https://arxiv.org/abs/2608.12332) | Large-SV pretrained components reusable; small-SV components task-specific. Aligns with spectral partitioning findings but from a different angle (which SVs of W₀ to clip vs. which SVs of ΔW partition). | Low — confirmatory for existing findings |
| Predicting Mergeability of PEFT Updates (Tang et al., 2026) | [2606.19549](https://arxiv.org/abs/2606.19549) | Early-training mergeability prediction for LoRA; uses Fisher, activation, gradient signals. Narrow eval (single benchmark). | Medium — watch for expanded evaluation |
| Optimizer-Dependent Hessian Eigenvector Dynamics (arXiv 2606.30226, ICML HiLD Workshop 2026) | [2606.30226](https://arxiv.org/abs/2606.30226) | SGD stabilizes Hessian eigenvectors, Adam keeps them mixing. Connects to Xu's optimizer-dependent k*. MLP-only. | Low — note only |
| Pruning via Marchenko-Pastur (arXiv 2606.02608, 2026) | [2606.02608](https://arxiv.org/abs/2606.02608) | Uses MP bulk edge for pruning budget allocation. Different application but same mathematical machinery as Gradience's optimal_hard_threshold policy. | Low — note only |

---

## Summary of Recommended Actions

1. **Kalra et al. (critical sharpness)**: Design a replication experiment
   pairing critical sharpness with existing Hessian telemetry on N07
   Experiment B data. If the lead-lag signal transfers, adopt as the
   production curvature instrument.

2. **Gu et al. (spectral imbalance)**: Cite in THEORY.md §6 and §7.2.
   Add to RESEARCH_INVENTORY.md §9 at High priority (peer-reviewed ICML).
   No infrastructure work needed.

3. **Zhou et al. (demystifying mergeability)**: Add to RESEARCH_INVENTORY.md
   §9 at Medium priority. Cite selectively in technical report §8. Monitor
   for stabilization and LoRA-specific follow-ups.
