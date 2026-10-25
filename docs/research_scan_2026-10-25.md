# Gradience Research Scan — October 25, 2026

**Purpose:** Identify new empirical research highly relevant to the Gradience
research program. Three papers selected from a broad search across spectral
LoRA analysis, adapter merging theory, and phase transition dynamics.

---

## 1. "What Does the Rank Buy? A Spectral and Distributional Analysis of Low-Rank Adaptation"

**arXiv:** [2609.32002](https://arxiv.org/abs/2609.32002)
**Authors:** Babak Barazandeh (Cribl AI Research Lab)
**Date:** September 2026

### Summary

This paper challenges the widely held assumption that LoRA rank r acts as a
capacity control. Under per-factor Frobenius norm budgets (the regime weight
decay induces), the authors show that every complexity and displacement
functional they analyze is maximized by a rank-one update — meaning the rank
cap never binds. The reachable model class is identical for every r >= 1, and
Rademacher complexity does not depend on r.

Rank instead matters in two places: (a) statistically, a joint budget on the
product BA restores a data-dependent, rank-sensitive complexity bound, but
only for well-spread feature distributions; (b) spectrally, rank sets the
*price of adaptation* — canceling the leading singular directions of the
pretrained weight requires both sufficient rank and sufficient norm budget.

### Relevance to Gradience

This is directly relevant to Gradience's central empirical finding: adapters
universally use only ~1/6 of allocated rank (mean utilization 0.172 at n=86;
FINDINGS.md §2-3). The paper provides a formal explanation: if rank is not
a capacity knob but rather governs *which spectral modifications are
reachable*, then low utilization is not waste — it is the expected outcome
of optimization under norm-constrained dynamics. The rank-one optimality
result explains why stable rank concentrates at 1.2-1.6 (N07 DeBERTa
finding) regardless of nominal rank.

Specific connections:
- **THEORY.md §1 (stable rank as effective dimensionality):** The paper's
  framework reinterprets what stable rank measures — not capacity usage but
  the optimizer's chosen spectral footprint under budget constraints.
- **THEORY.md §7.2 (spectral scaling laws):** If rank doesn't control
  capacity, the open question about how entropy effective rank scales with
  model size needs reframing — the scaling may be governed by norm budgets
  (weight decay strength) rather than rank allocation.
- **FINDINGS.md §3 (rank-utilization correlation):** The attenuation of the
  rank-utilization correlation from r=-0.578 (n=29) to r=-0.191 (n=86)
  is consistent with this theory — if rank doesn't bind, the correlation
  is an artifact of confounded training recipes, not a structural law.

### Integration Criticality: **HIGH**

This paper could reshape THEORY.md §1 and §7.2. The rank-as-reachability
framing is a cleaner theoretical foundation for Gradience's spectral
measurements than the current implicit "rank as capacity" framing. The
rank-one optimality result directly explains the stable-rank concentration
observed across 86 adapters and three architectures.

**Caveat:** Single-author preprint, not yet peer-reviewed. The theoretical
results rest on per-factor norm budget assumptions that may not hold for all
training recipes (e.g., no weight decay, or LoRA alpha scaling). Verify that
the norm-budget regime matches the training configurations in Gradience's
validation corpus before integrating.

---

## 2. "Beyond Uniform Subspaces: Spectrum-Aware and Depth-Adaptive Fusion for Multi-Task Model Merging" (SADA-Merging)

**arXiv:** [2609.24612](https://arxiv.org/abs/2609.24612)
**Authors:** Ruxi Gu, Zilei Wang, Wei Wang
**Date:** September 2026

### Summary

SADA-Merging is a spectrum-aware merging framework that allocates task-specific
subspace capacity according to spectral entropy, estimates task-wise plasticity
from spectral concentration to control spectral transformation, and introduces
depth-dependent anchoring to compensate for projection-induced distortion.
It operates on task updates (weight deltas) and is applicable to both full
fine-tuning and LoRA settings.

Key empirical finding for Gradience: **LoRA task vectors exhibit markedly
weaker spectral heterogeneity than their full fine-tuning counterparts.**
Spectral entropy still differentiates rank needs across tasks, but the
dynamic range is compressed in the LoRA setting, limiting the method's
marginal gains there.

### Relevance to Gradience

This paper operates in Gradience's Axis 4 "Execution" cell (improving merge
results) rather than Gradience's "Triage" cell, but the theoretical and
empirical overlap is substantial:

- **Spectral entropy as a merge-relevant observable:** SADA-Merging
  independently validates that spectral entropy — which Gradience already
  computes as `entropy_effective` rank (THEORY.md §1) — is the right
  quantity for adaptive merge strategy selection. This is external
  confirmation that Gradience's spectral measurement infrastructure
  captures merge-relevant structure, not just descriptive statistics.

- **LoRA's weak spectral heterogeneity:** This finding directly connects
  to Gradience's utilization results (mean 0.172, FINDINGS.md §3). If LoRA
  adapters have compressed spectral profiles, the triage problem is
  *easier* (fewer degrees of freedom to conflict) but the *execution*
  problem is harder (less spectral signal to route). This explains the
  empirical observation that Gradience's triage works well at encoder
  scale but has less margin at decoder scale.

- **Depth-dependent anchoring:** The finding that deeper layers need
  stronger anchoring during merging connects to Gradience's layer-level
  diagnosis (v_proj vs q_proj patterns in FINDINGS.md §8) and the
  spectral partitioning convergence result (FINDINGS.md §14), where
  layer-level alignment dynamics differ systematically by depth.

- **THEORY.md §8 (Conceptual Map):** This paper adds a new entry to
  Axis 1=Weight deltas, Axis 4=Execution. It is the first execution-cell
  paper to use spectral entropy as the primary allocation signal, making
  it the closest methodological neighbor to Gradience in the merge
  execution space.

### Integration Criticality: **MEDIUM-HIGH**

The paper validates Gradience's spectral measurement choices and provides
a bridge between triage (Gradience's purpose) and execution (SADA-Merging's
purpose). The LoRA weak-heterogeneity finding should be noted in THEORY.md §6
as an independent empirical constraint on merge triage precision. The spectral
entropy validation strengthens the case for `entropy_effective` as a
first-class merge-triage metric.

**Caveat:** Primarily evaluated on ViT models, not language models. The LoRA
finding is in the appendix and may be less thoroughly validated than the main
FFT results. No overlap with Gradience's encoder or decoder validation corpus.

---

## 3. "Can Spectral-Clipping Enable Better Learning While Forgetting Less for Low-Rank Adaptation?" (SCLoRA)

**arXiv:** [2608.12332](https://arxiv.org/abs/2608.12332)
**Authors:** Hyowon Wi, Noseong Park
**Venue:** ACL 2026 Long Paper (peer-reviewed)
**Date:** August 2026

### Summary

This paper establishes a theoretical connection between uncontrolled singular
value growth in LoRA adapters and catastrophic forgetting of pretrained
knowledge. The authors prove that when singular values of the LoRA update
grow without constraint, the adapted model drifts from the pretrained
representation, causing forgetting. Their method (SCLoRA) injects
parameterized singular components with spectral clipping into the pretrained
model, aware of the pretrained model's spectral distribution.

The paper positions itself relative to PiSSA (trains on principal components)
and MiLoRA (trains on minor components), framing all SVD-based LoRA variants
as methods that choose *which singular directions to update*. SCLoRA adds
*how much* each direction is allowed to change.

### Relevance to Gradience

This paper provides a mechanistic link between spectral structure and a
behavioral outcome (forgetting) that is directly relevant to merge triage:

- **THEORY.md §7.2 (tail-band interference):** Gradience's open question
  asks whether interference in the low-SV band produces measurable
  behavioral degradation. SCLoRA's framework provides the theoretical
  bridge: if SV growth causes forgetting, then merging two adapters whose
  SVs grew in conflicting directions should produce *mutual forgetting* —
  each adapter's modifications are partially erased by the other's. The
  spectral partition (high-SV shared, low-SV task-specific) from N127
  predicts which directions would conflict. SCLoRA's theory formalizes
  *why* that conflict matters: because SV magnitude tracks the degree of
  departure from the pretrained representation, and merging conflicting
  departures causes regression.

- **FINDINGS.md §9 (structural != sufficient):** SCLoRA's insight that
  *magnitude of spectral change* matters independently of *subspace
  geometry* provides theoretical grounding for Study 16's finding. Two
  adapters can occupy compatible subspaces (high overlap) but have
  incompatible SV magnitudes — one has grown its SVs far from the
  pretrained model while the other hasn't. The merge would be spectrally
  "safe" but behaviorally poor, which is exactly the false-negative
  scenario Study 16 exposed.

- **Merge QA Report (vnext/merge/qa_report.py):** The SV-growth-as-
  forgetting link suggests a concrete enhancement: comparing the
  per-direction SV growth profiles of two adapters relative to the
  pretrained weight could provide a forgetting-risk signal independent
  of the existing subspace-overlap metric.

- **THEORY.md §2 (Weyl perturbation and compression safety):** The
  qualification about fine-tuning changing what the perturbation bound
  *bounds* (noise vs. content) is exactly SCLoRA's point, now formalized
  with a forgetting-specific argument. The paper supports the two-regime
  distinction Gradience already draws (compression of pretrained vs.
  compatibility of fine-tuned).

### Integration Criticality: **MEDIUM-HIGH**

This is the strongest candidate for theoretical integration because it is
peer-reviewed (ACL 2026) and provides the missing mechanistic link between
spectral observables and behavioral outcomes that THEORY.md §7.2 identifies
as the key open problem. It doesn't solve the tail-band interference question
directly, but it provides the formal framework in which that question can be
stated precisely.

**Caveat:** The forgetting theory may apply more cleanly to single-adapter
training than to pairwise merge interactions. The mapping from "SV growth
causes forgetting" to "SV conflict in merged adapter causes mutual forgetting"
requires additional theoretical work that the paper does not provide. The
method (SCLoRA) itself is a training-time intervention, orthogonal to
Gradience's training-free audit regime.

---

## Summary Table

| Paper | arXiv | Integration Priority | Key Connection |
|-------|-------|---------------------|----------------|
| What Does the Rank Buy? | 2609.32002 | **High** | Reframes rank as reachability, not capacity — explains universal low utilization |
| SADA-Merging | 2609.24612 | **Medium-High** | Independently validates spectral entropy for merge allocation; LoRA weak-heterogeneity finding |
| SCLoRA | 2608.12332 | **Medium-High** | Formalizes SV-growth-to-forgetting link; supports tail-band interference theory |

### Recommended Next Steps

1. **"What Does the Rank Buy?"**: Add to RESEARCH_INVENTORY.md §9 at Medium
   priority (preprint, needs replication). Check whether the norm-budget
   assumptions match the training recipes in the Gradience validation corpus.
   If they do, draft a revision of THEORY.md §1 that distinguishes "rank as
   capacity" (the current implicit framing) from "rank as reachability"
   (the new framing). This is a pen-and-paper assessment, not an experiment.

2. **SADA-Merging**: Add to RESEARCH_INVENTORY.md §9 at Low-Medium priority
   (ViT-only evaluation, appendix-only LoRA results). Note the LoRA
   weak-heterogeneity finding in THEORY.md §6 as an independent empirical
   constraint. No infrastructure work needed.

3. **SCLoRA**: Add to RESEARCH_INVENTORY.md §9 at Medium priority (peer-
   reviewed but mapping to merge setting is non-trivial). Consider whether
   the SV-growth framework can sharpen the tail-band interference probe
   (CHG-005) by adding a per-direction growth comparison to the existing
   subspace-overlap metric.
