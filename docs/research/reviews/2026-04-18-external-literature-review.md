# External Literature Review — 2026-04-18

**Audience:** researcher, maintainer
**Status:** standalone review (single-pass, compiled from one session)
**Purpose:** identify recently published empirical or technical work most relevant to the Gradience research program, and rate integration criticality
**Scope:** external publications (arXiv, Nature/npj AI, ICLR/ICCV 2025–2026); does not re-survey internal Gradience findings
**See also:** [`research-overview.md`](../research-overview.md), [`../../THEORY.md`](../../THEORY.md)

---

## 1. Summary

This review compiles three externally published papers from 2025–2026 that are
most relevant to Gradience's theoretical and technical framework. Each entry
gives a short summary, maps the paper onto Gradience's existing modules and
open research questions (THEORY.md §7), and assigns an integration criticality
rating.

| # | Paper | Core Overlap | Criticality |
|---|-------|--------------|-------------|
| 1 | Spectral Geometry of LoRA Adapters (arXiv:2604.08844) | Identical spectral feature set; behavioral prediction from geometry | **HIGH** |
| 2 | Preference-Aligned LoRA Merging / TARA-Merging (arXiv:2603.26299) | Subspace coverage and anisotropy as formal merge-quality axes | **MEDIUM-HIGH** |
| 3 | Phase Transitions in LLM Compression (npj AI 2026) | Empirical criticality thresholds for low-rank decomposition | **MEDIUM** |

---

## 2. Candidate 1 — Spectral Geometry of LoRA Adapters Encodes Training Objective and Predicts Harmful Compliance

- **Authors:** Roi Paul
- **Date:** 2026-04-10
- **Venue:** arXiv preprint
- **Link:** https://arxiv.org/abs/2604.08844

### Summary

The paper tests whether low-rank spectral summaries of LoRA weight deltas can
(a) identify the fine-tuning objective that produced an adapter and (b) predict
downstream behavioral harm. In a pre-registered experiment on
Llama-3.2-3B-Instruct, the authors manufactured 38 LoRA adapters across four
categories (healthy SFT, DPO on inverted harmlessness preferences, DPO on
inverted helpfulness preferences, and activation-steering-derived adapters)
and extracted per-layer spectral features: norms, stable rank, singular-value
entropy, effective rank, and singular-vector cosine alignment to a healthy
centroid.

Headline results:

- Within a single training method (DPO), a logistic regression classifier
  achieved AUC ≈ 1.00 on binary drift detection and on all six pairwise
  objective comparisons.
- Near-perfect ordinal severity ranking (ρ ≥ 0.956).
- DPO-inverted-harmlessness adapters showed elevated harmful compliance on
  HEx-PHI prompts (mean ASR 0.266 vs. healthy 0.112, Δ = +0.154), with
  near-perfect dose-response (ρ = 0.986).
- Cross-method monitoring (DPO vs. activation steering) requires per-method
  calibration — a single calibrated classifier does not generalize across
  manufacturing regimes.

### Relevance to Gradience

This is the most directly relevant external work I identified. Its feature
vocabulary is effectively a subset of Gradience's: stable rank, entropy
effective rank, per-layer singular-value norms, and subspace alignment are
all exposed in `gradience.vnext.audit` and summarized in the
`AdapterQAArtifact` schema.

Three specific intersections:

1. **Structural → behavioral bridge (THEORY.md §7.1).** Gradience's Study 16
   and Study 17 established that spectral structure is necessary but not
   sufficient for merge quality. This paper demonstrates a concrete regime in
   which spectral geometry *is* sufficient for behavioral prediction — when
   measurement is made relative to a calibrated "healthy centroid". That is a
   methodological innovation Gradience's QA artifact could adopt: rather than
   reporting raw spectral metrics, include per-adapter distance from a
   reference centroid defined by a known-healthy pool. This could tighten the
   `eligible` / `flagged_weak` / `uncertain` distinction in
   `vnext/audit/qa_artifact.py`.

2. **Decoder-side validation.** Gradience's partitioning evidence is strongest
   on encoders (DistilBERT, DeBERTa); the Mistral-7B evidence is weaker and
   was obtained under unweighted overlap. This paper replicates the
   "spectral-features-predict-something" claim on Llama-3.2-3B. It does not
   adjudicate Gradience's own decoder-scale claims, but it is independent
   evidence that the feature family generalizes to decoder architectures at
   the 3B scale.

3. **Per-method calibration.** The cross-method calibration finding
   (one-size-fits-all classifiers fail across DPO vs. activation-steering)
   maps onto Gradience's existing policy that per-family spectral profiles
   differ. It supports keeping the current per-family baseline strategy and
   cautions against promoting a single global centroid.

### Integration Criticality: HIGH

- Cite in THEORY.md §7.1 under "Structural-behavioral separation" as an
  independent demonstration that the separation is regime-dependent.
- Evaluate centroid-relative spectral distance as an additional QA field in
  `AdapterQAArtifact` (additive-only, schema-preserving).
- Relevant to the existing decoder-only spectral fingerprinting plan
  (`docs/plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`).

---

## 3. Candidate 2 — Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy (TARA-Merging)

- **Authors:** multi-author
- **Date:** 2026-03-27
- **Venue:** arXiv preprint (also on OpenReview)
- **Link:** https://arxiv.org/abs/2603.26299

### Summary

The paper addresses the geometry of multi-LoRA merging through two formal
axes:

- **Subspace coverage:** how broadly LoRA update directions cover the
  representational space.
- **Directional anisotropy:** imbalance of influence across those directions.

It proposes TARA-Merging (Task-Rank Anisotropy Alignment), which aligns
merging weights via a preference-weighted cross-entropy pseudo-loss while
preserving task-relevant LoRA subspaces via direction-wise reweighting.
Evaluated across 8 vision and 6 NLI benchmarks, TARA-Merging outperforms
vanilla and LoRA-aware baselines.

### Relevance to Gradience

The paper formalizes two geometric concepts Gradience already measures
operationally:

1. **Subspace coverage** ≈ Gradience's principal-angle analysis and
   `mean_overlap` / `directional_agreement` metrics (THEORY.md §6).
2. **Directional anisotropy** ≈ what Gradience measures via stable rank,
   utilization, and magnitude ratio.

Two specific intersections:

1. **Analytical spectral geometry plan.** Gradience's
   [`analytical spectral geometry of merge operations`](../../plans/2026-04-03-analytical-spectral-geometry-of-merge-operations-plan.md)
   plan aims to derive closed-form and semi-analytical bounds for how merge
   strategies transform singular-value structure. The TARA framework provides
   an external formalization of the same geometric quantities (coverage and
   anisotropy) that the plan treats analytically. Reading this paper before
   deriving TIES and DARE bounds (phases 4–5 of the plan) may shortcut the
   formalization.

2. **Verdict tree refinement.** Gradience's merge verdict logic
   (`vnext/merge/`) currently treats subspace overlap and magnitude ratio as
   discrete classification inputs. TARA demonstrates that treating them as
   continuous optimization targets in a preference-weighted loss produces
   measurable gains. This does not replace Gradience's triage role (TARA
   assumes merging will happen; Gradience decides *whether* to merge), but
   the direction-wise reweighting formulation could refine the
   `norm_equalized` strategy recommendation in `MergeQAReport`.

### Integration Criticality: MEDIUM-HIGH

- Cite in THEORY.md §6 and in the analytical spectral geometry plan as the
  external formalization to compare against.
- Do *not* adopt TARA-Merging as a strategy — Gradience is a triage system,
  not a merge optimizer — but consider whether its direction-wise reweighting
  suggests an additional `MergeQAReport.recommended_strategy` value.

---

## 4. Candidate 3 — Phase Transitions in Large Language Model Compression

- **Authors:** Ma, Z., Li, Z., Zhang, L., et al.
- **Date:** 2026
- **Venue:** npj Artificial Intelligence, Vol. 2, Article 21
- **Link:** https://www.nature.com/articles/s44387-026-00072-8

### Summary

The paper demonstrates that LLMs exhibit **Model Phase Transitions** — sharp
performance collapses beyond critical compression thresholds. Across 30+
pruning, quantization, and low-rank decomposition techniques, the authors
identify critical thresholds:

- Structured pruning fails at 30–45% sparsity.
- Unstructured pruning fails at 55–65%.
- **Low-rank decomposition fails at 17–30% retention.**
- Quantization fails below 3-bit precision.

Structural, numerical, and algebraic redundancy are characterized as
orthogonal sources, enabling a criticality-aware compression framework that
reaches near-lossless compression to 10% of original size.

### Relevance to Gradience

Two intersections:

1. **Compression pipeline calibration.** Gradience's Post 7 audit found a
   median compression potential of ~50% across 86 adapters, which sits well
   within the safe regime identified by this paper (low-rank decomposition
   starts failing at 17–30% retention, i.e. 70–83% compression). The paper's
   *transition point* characterization is new calibration data for where
   `energy_rank_90`-driven truncation should issue warnings, and for what the
   bench compression validation suite should treat as a catastrophic-failure
   threshold.

2. **Phase transition framework (THEORY.md §4).** Gradience's phase
   transition machinery (`research.phase_transitions`) is motivated by
   statistical-mechanics signatures (critical slowing down, fluctuation
   amplification). This paper provides external empirical evidence that
   compression-induced phase transitions are real, sharp, and characterizable
   — which strengthens the framework's motivation even though it does not
   introduce new mathematical machinery.

The paper is less relevant to the merge pipeline directly and does not
resolve any open question in THEORY.md §7.2.

### Integration Criticality: MEDIUM

- Cite in THEORY.md §4 as external validation of the phase-transition
  framework.
- Use the 17–30% low-rank retention threshold to set warning bands in the
  bench compression validation suite.
- No API or schema changes implied.

---

## 5. Consolidated recommendations

1. **Paper 1 (Spectral Geometry of LoRA Adapters)** is the highest-value
   integration target. It directly addresses the structural-behavioral
   separation identified in THEORY.md §7.1 and offers a concrete
   methodological innovation (centroid-relative spectral distance) that could
   sharpen `AdapterQAArtifact` eligibility logic. Recommend reading in full
   and evaluating the centroid-relative approach as an additive QA field.

2. **Paper 2 (TARA-Merging)** is a useful formalization to read before
   tackling phases 4–5 of the analytical spectral geometry plan. Not a
   strategy replacement, but a reference point.

3. **Paper 3 (Phase Transitions in LLM Compression)** provides calibration
   data and external validation for the phase-transition framework, but
   implies no architectural changes.

All three papers should be added to the reference list in THEORY.md.

---

## 6. Papers reviewed but not promoted

- **SpectralLoRA (arXiv:2604.10649)** — DCT-domain spectral analysis of LoRA
  updates on BERT/RoBERTa; complementary to SVD-based analysis but
  encoder-only and methodologically orthogonal (frequency domain rather than
  singular-value domain). Interesting as future reading but not a priority
  integration target.
- **Revisiting LoRA through the Lens of Parameter Redundancy
  (ACL 2025)** — parameter-redundancy framing; partially redundant with
  Gradience's stable-rank/utilization analysis. No new machinery.
- **Model Zoos for Benchmarking Phase Transitions (OpenReview)** — relevant
  to THEORY.md §4 but phenomenology-first rather than mechanism-first; lower
  fit with Gradience's measurement-instrument framing.

---

## 7. Sources

- [Spectral Geometry of LoRA Adapters Encodes Training Objective and Predicts Harmful Compliance (arXiv:2604.08844)](https://arxiv.org/abs/2604.08844)
- [Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Directional Anisotropy (arXiv:2603.26299)](https://arxiv.org/abs/2603.26299)
- [Phase Transitions in Large Language Model Compression (npj AI 2026)](https://www.nature.com/articles/s44387-026-00072-8)
- [SpectralLoRA: Is Low-Frequency Structure Sufficient for LoRA Adaptation? (arXiv:2604.10649)](https://arxiv.org/abs/2604.10649)
- [Model Merging in LLMs — Awesome List (ACM Computing Surveys 2026)](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications)
