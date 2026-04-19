# Empirical Findings

An authoritative compendium of results obtained with Gradience, organized by
research strand. Each finding states a claim, supporting evidence, known
limitations, and reproducibility notes.

**Scope.** This document covers results from both Gradience I (the published
library, v0.11.0) and Gradience II (GPU-scale decoder studies, March 2026).
Where a finding has been superseded by a later study, the supersession is
noted and the original retained for the historical record.

**Conventions.** All p-values are two-sided unless noted. Effect sizes use
Cohen's d for paired comparisons and Pearson r for correlations unless
otherwise specified. Confidence intervals are 95% unless noted. "Utilization"
always means stable_rank / nominal_rank.

**Last substantive revision:** April 4, 2026.

---

# Part A — Spectral Structure of Individual Adapters

This strand establishes what the spectral profile of a single LoRA adapter
looks like: how much of the allocated rank is used, how spectral energy is
distributed, and what compression this implies.

---

## 1. Spectral Compression of Mistral-7B on GSM8K

### Claim

LoRA adapters trained at rank r = 64 for Mistral-7B on GSM8K use
approximately 4--8 effective dimensions per layer, as measured by stable
rank and energy rank at 90%. A 50% parameter reduction (truncating to
r = 32 or lower, per layer) is validated with less than 2.5% accuracy
degradation across 3 independent seeds.

### Evidence

- **Spectral audit.** `gradience audit` on the rank-64 adapter shows
  `energy_rank_90` values of 3--8 across attention and MLP layers,
  with utilization ratios (srank / r) consistently below 0.15.
  The singular value spectra exhibit a sharp knee: the first 4--6 values
  carry >90% of the Frobenius energy, and the remaining values decay
  into a noise floor.

- **Multi-seed validation.** `gradience-bench` ran the full compression
  protocol across 3 random seeds. The benchmark measures accuracy on the
  GSM8K test split after applying per-layer rank truncation suggested by
  the `energy_threshold(0.90)` policy. Results: mean baseline accuracy
  (rank-64) reported across seeds; mean compressed accuracy (per-layer
  truncation) within 2.5% of baseline; 95% confidence intervals computed
  with sample standard deviation (ddof=1); Cohen's d effect size reported
  to quantify practical significance.

- **Parameter savings.** The per-layer truncation reduces total adapter
  parameters by approximately 50%, with some layers truncated to rank 3--4
  and others retained at higher rank (12--16) where the spectral structure
  demands it.

### Limitations

- Single architecture (Mistral-7B) and single task (GSM8K). Generalization
  to other architectures and tasks is plausible but unverified at this
  point. See §2 and §3 for subsequent broader validation.
- GSM8K is a relatively narrow math reasoning benchmark. Tasks with broader
  knowledge requirements may need more effective dimensions.
- Accuracy degradation is measured on the GSM8K test set only; downstream
  effects on other capabilities of the base model are not assessed.

### Reproducibility

Requires `gradience[bench]` installation and GPU access. Configuration
files in `gradience/bench/configs/` specify the exact experimental setup.
Multi-seed protocol uses `gradience.bench.multi_seed` with 3 seeds by default.


---

## 2. Architecture-Agnostic Spectral Profiles (Study 14, March 2026)

**Supersession note.** The n=29 sample reported here was later expanded to
n=86 in Post 7 (§3). Where results differ, the larger sample is
authoritative. Both are retained because the smaller sample includes
statistics not recomputed at scale.

### Claim

Low rank utilization is a pervasive property of publicly available LoRA
adapters, not an artifact of a particular architecture or task. Across
29 adapters spanning 8 base models and 5 task categories, mean utilization
is 0.166, median compression potential is 50%, and higher nominal rank
predicts lower utilization.

### Evidence

- **Sample.** 29 successfully audited adapters (of 30 discovered on
  HuggingFace Hub, 1 excluded at 500 MB file limit). 8 base models:
  Llama-2-7B, Llama-3-8B, Mistral-7B, Gemma-2B, Gemma-7B, Phi-2.
  5 task categories: chat, classification, data, general, text-generation.
  Nominal ranks 4--64. All computation CPU-only, float64.

- **Utilization.** Mean 0.166, median 0.159, std 0.104, range [0.000, 0.389].
  One zero-utilization outlier (likely an uninitialized checkpoint).

- **Module-type breakdown** (4,558 layers total):

  | Type      | N layers | Mean util | Median util | Mean srank |
  |-----------|----------|-----------|-------------|------------|
  | Attention | 2,908    | 0.168     | 0.135       | 2.39       |
  | MLP       | 1,618    | 0.187     | 0.149       | 2.52       |
  | Other     | 32       | 0.095     | 0.092       | 1.52       |

- **Rank--utilization correlation.** r = -0.578 (weak negative). Higher
  nominal rank tends toward lower utilization. Utilization by rank:

  | Rank | N  | Mean util | Median |
  |------|----|-----------|--------|
  | 4    | 2  | 0.344     | 0.344  |
  | 6    | 1  | 0.305     | 0.305  |
  | 8    | 8  | 0.216     | 0.194  |
  | 16   | 11 | 0.141     | 0.129  |
  | 32   | 4  | 0.094     | 0.089  |
  | 64   | 3  | 0.056     | 0.053  |

- **Compression potential.** Median adapter needs 50% of nominal rank to
  capture 90% spectral energy. p90 energy rank: 69%.

### Limitations

- Study overweights popular adapters (discovered by HuggingFace download
  count). Private, specialized, and less-downloaded adapters underrepresented.
- No downstream task evaluation of compression impact (spectral measurement
  only; behavioral validation in §1 is for Mistral-7B/GSM8K only).
- The rank--utilization correlation (r = -0.578) was substantially
  attenuated at n=86 (see §3), suggesting it was overstated in the
  smaller sample.

### Reproducibility

Scripts and full results in `Gradience II/results/study14_broader_benchmarks/`.
Gradience v0.11.0, CPU-only, float64.


---

## 3. Scaling to 86 Adapters, 22 Architectures (Post 7, March 2026)

### Claim

The core spectral findings from Study 14 hold at nearly 3× the sample
size. Across 86 adapters, mean utilization remains ~0.17, median
compression potential remains 50%, and module type (attention vs. MLP)
does not meaningfully predict utilization. The rank--utilization
correlation weakens substantially (r = -0.578 to r = -0.191), indicating
the original estimate was inflated by the small sample.

### Evidence

- **Sample.** 86 successfully audited adapters (100 discovered, 14 excluded
  at file size limit). 11+ base architectures (adds Llama-3.2-3B,
  CodeLlama-7B, Phi-3-mini, Phi-3.5-mini to Study 14's set).
  12 task categories (adds medical, code, math, summarization, QA,
  translation, legal to original set). Nominal ranks 4--128.

- **Utilization.** Mean 0.172, median 0.154, std 0.142, range [0.00, 1.00].
  75th percentile: 0.184. Only 10% of adapters exceed utilization 0.32.
  Typical adapter uses approximately 1/6 of allocated rank.

- **Module-type breakdown** (12,496 layers total):

  | Type      | N layers | Mean util |
  |-----------|----------|-----------|
  | Attention | 7,776    | 0.167     |
  | MLP       | 4,400    | 0.166     |
  | Other     | 320      | —         |

  The attention/MLP gap from Study 14 (0.168 vs. 0.187) disappears at
  scale. Module type does not meaningfully predict utilization.

- **Rank--utilization correlation.** r = -0.191 (was -0.578 at n=29).
  The relationship remains directionally negative but is substantially
  weaker than the small-sample estimate suggested. Median utilization
  by rank still shows monotonic decline (r=4: 0.345; r=128: 0.024).

- **Compression potential.** Median energy rank at 90% threshold: 50% of
  nominal rank (unchanged from n=29). 90th percentile: 73%. Consistent
  with the controlled Mistral-7B/GSM8K bench experiments (§1).

- **High-utilization outliers.** Rank-64 BERT-medium experiments: 0.41--0.45
  utilization. Rank-64 OpenWebMath adapter (20B tokens): 0.62 utilization
  (highest non-trivial). Hypothesis: scale × data volume drives higher
  utilization.

### Limitations

- Sample still overweights popular adapters. 31/86 adapters could not be
  auto-matched to a known base model architecture.
- Rank coverage uneven (r=8: n=21; r=16: n=32; r=4: n=3; r=128: n=1).
- No downstream task evaluation of compression impact.
- The attenuation of the rank--utilization correlation from n=29 to n=86
  is itself informative: it means quantitative claims about correlation
  strength require larger samples than initially assumed.

### Reproducibility

Primary data in `Gradience II/results/study14_broader_benchmarks/`.
Extended analysis reported in Post 7
(`Gradience II/docs/blog_series/SERIES_POST_7_GRADIENCE_011.md`).


---

## 4. Cross-Seed Stability of Spectral Metrics

### Claim

Stable rank is highly consistent across training seeds for the same
architecture-task pair. Layer-specific utilization ratios show more
variance, particularly in attention layers.

### Evidence

- **Multi-seed audit.** Running `gradience audit` on 3 independently
  trained Mistral-7B/GSM8K adapters (same hyperparameters, different
  random seeds) shows: stable rank with low coefficient of variation
  across seeds (typically <0.1); energy rank at 90% similarly stable,
  with occasional ±1 variation; utilization with moderate variance in
  specific attention layers, coefficients of variation up to 0.2--0.3
  in some q_proj and k_proj layers.

- **Interpretation.** Stable rank and energy rank capture the "shape" of
  the spectral distribution, which appears to be determined primarily by
  the architecture and task, not the random initialization. Utilization
  variability reflects seed-dependent differences in how much of the
  allocated rank a particular layer's optimizer trajectory happens to use.

### Limitations

- Three seeds is the minimum for meaningful variance estimation. The
  95% confidence intervals are wide with n = 3.
- Only one architecture-task pair tested. Cross-architecture or cross-task
  stability of this observation is unknown.
- The distinction between "metric stability" and "learned representation
  stability" is important: stable rank may be consistent even if the
  actual learned subspaces (principal vectors) differ across seeds.

### Reproducibility

Use `gradience-bench` multi-seed mode with ≥3 seeds. Compare per-layer
`stable_rank` and `utilization` across seeds using the structured JSONL output.


---

## 5. Attention vs. MLP Spectral Structure

**Supersession note.** The attention/MLP utilization gap reported here
(~0.08--0.12 vs. ~0.10--0.18) was observed on Mistral-7B/GSM8K. At
n=86 (§3), the gap vanishes (0.167 vs. 0.166). The spectral *shape*
difference (dominant σ₁ in attention) has not been tested at scale.

### Claim

Attention layers and MLP layers exhibit systematically different spectral
profiles. Attention layers (especially q_proj and k_proj) tend to
have lower utilization and sharper spectral concentration than MLP layers
(gate_proj, up_proj, down_proj).

### Evidence

- **Module-type aggregation.** Gradience's audit groups layers by inferred
  module type. In the Mistral-7B/GSM8K experiments, attention layers
  show mean utilization of ~0.08--0.12, while MLP layers show ~0.10--0.18.
- **Spectral shape.** Attention layers more frequently exhibit a dominant
  first singular value (σ₁ / σ₂ > 3) compared to MLP layers,
  which tend toward smoother spectral decay.

### Limitations

- Observed on a single architecture. The utilization gap does not replicate
  at n=86 (§3). The spectral shape difference (dominant first SV) has
  not been tested at scale and may also be architecture-specific.
- The functional interpretation (why attention specializes more sharply)
  is speculative.

### Reproducibility

Examine the `module_type` field in `gradience audit --json` output and
aggregate metrics by type.


---

## 6. Rank Policy Disagreement Patterns

### Claim

When the `energy_threshold(0.90)` and `knee_elbow` rank policies disagree
substantially (suggested ranks differing by more than 2×), the
underlying singular value spectrum has a specific structure: a gradual
decay without a sharp knee, or a long tail of moderate-magnitude singular
values that carry significant cumulative energy despite being individually
small.

### Evidence

- **Disagreement analysis.** `gradience audit --policies` runs all five
  rank policies (`energy_threshold`, `entropy_effective`,
  `optimal_hard_threshold`, `knee_elbow`, `stable_rank_ceil`) and the
  `policy_analysis` module computes per-layer disagreement metrics:
  policy spread (max k minus min k) and importance scores.

- **Spectral structure of disagreement cases.** Layers where
  `energy_threshold` suggests k = 6 but `knee_elbow` suggests k = 2
  typically have a smooth, concave scree plot with no sharp drop-off.
  The energy policy accumulates contributions from the gradual tail;
  the knee policy finds an early inflection that misses this tail energy.

- **Conservative policy comparison.** `optimal_hard_threshold` (based on
  random matrix theory) and `stable_rank_ceil` tend to agree more closely
  with each other than with energy or knee policies, suggesting they
  capture a similar notion of "signal vs. noise" that differs from
  cumulative energy capture.

### Limitations

- Disagreement analysis is descriptive, not prescriptive: it identifies
  *where* policies diverge but does not determine which policy is "correct"
  for downstream performance.
- The relationship between policy disagreement and actual compression safety
  (measured by benchmark accuracy) is not yet systematically validated.

### Reproducibility

Run `gradience audit <adapter_dir> --policies --json` and examine the
`rank_suggestions` field per layer. The `policy_analysis.compute_layer_importance_scores`
function implements the disagreement metrics.


---

# Part B — Merge Compatibility and Adapter Interaction

This strand characterizes what happens when two adapters are combined:
how spectral geometry predicts merge outcomes, and what the decision
boundaries are.

---

## 7. Subspace Overlap Predicts Merge Dominance (Post 3, March 2026)

### Claim

Subspace overlap between adapter pairs, measured via principal angles,
strongly predicts merge dominance on Mistral-7B. The correlation between
mean_overlap and merge dominance index is r = 0.846 (p < 0.0001, n=27
cross-task pairs). Same-task pairs show 2.4× higher overlap than
cross-task pairs, confirming the metric captures genuine subspace
structure rather than noise.

### Evidence

- **Design.** Mistral-7B, 3 tasks (chat/instruction, math/GSM8K, code),
  3 seeds each, yielding 12 adapters. 27 cross-task merge pairs + 9
  same-task calibration pairs. Merge method: linear averaging at weights
  [0.5/0.5], [0.7/0.3], [0.3/0.7].

- **Dominance prediction.**
  - mean_overlap vs. dominance (D): r = 0.846, p < 0.0001
  - compatibility_score vs. D: r = 0.846, p < 0.0001
  - frob_bounded_ratio vs. D: r = 0.838, p < 0.0001

- **Retention prediction (Q_min).** Spearman rank correlations:
  - compatibility_score: ρ = 0.710, p < 0.0001
  - frob_bounded_ratio: ρ = 0.642, p = 0.0003
  - mean_overlap: ρ = 0.638, p = 0.0003
  Note: 11/27 merges achieved Q_min = 1.000 (ceiling effect attenuates
  the correlation).

- **Pair difficulty ranking.** The audit correctly ranks pair difficulty
  without access to task labels: chat+code (compatibility 0.286--0.290,
  highest retention), chat+math (0.187--0.190, moderate), math+code
  (0.175--0.180, lowest). Rankings derived from singular values, not
  task semantics.

- **Same-task vs. cross-task calibration.** Same-task pairs (e.g., chat
  seed 1 vs. chat seed 2): mean overlap 0.473. Cross-task pairs: mean
  overlap 0.200. Separation: 2.4×, t = 12.985, p < 0.0001.

### Limitations

- Only linear averaging tested. TIES, DARE, and other robust merge
  methods untested. The overlap--dominance relationship may differ under
  non-linear merge strategies.
- Compatibility scores cluster in a narrow band (0.17--0.29), limiting
  raw R² and leaving open whether the relationship holds at extremes.
- Tested exclusively on the authors' own adapters. Wild Hub adapters
  with unknown training provenance pending.
- Optimal merge weights are not yet prescribed; the system detects
  problems but does not yet solve them.

### Reproducibility

Requires Mistral-7B adapters from `Gradience II/` and the merge-audit
pipeline. Full analysis in `Gradience II/docs/blog_series/SERIES_POST_3.md`
and `Gradience II/docs/implementation/merge_experiment_report.md`.


---

## 8. Merge Compatibility via Principal Angle Analysis (General Pipeline)

### Claim

Subspace overlap between adapter pairs, measured via principal angles,
predicts merge compatibility in the general case. High overlap
(mean_overlap > 0.5) indicates safe simple-averaging merges; low overlap
indicates risk of destructive interference.

### Evidence

- **Per-layer analysis.** `gradience merge-audit` computes `SubspaceMetrics`
  for each layer pair, including `mean_overlap`, `directional_agreement`,
  and `magnitude_ratio`. Layers with `mean_overlap > 0.5` and
  `directional_agreement > 0.3` merge cleanly under simple averaging.

- **Module-type patterns.** Across tested adapter pairs, `v_proj` layers
  consistently show higher subspace overlap than `q_proj` layers. This
  pattern is consistent with the hypothesis that value projections learn
  more universal features (shared across tasks), while query projections
  specialize to task-specific attention patterns.

- **TIES merging interaction.** Applying TIES merging (trim, elect sign,
  disjoint merge) on layers with already-low subspace overlap further
  degrades performance, because TIES's sign election amplifies conflicts
  between orthogonal subspaces. Simple averaging on high-overlap layers
  combined with exclusion or task-arithmetic on low-overlap layers performs
  better.

### Limitations

- Tested on a limited number of adapter pairs. The overlap thresholds
  (0.5 for safe merging) are empirical observations, not theoretically
  derived boundaries.
- The interaction between magnitude balance and subspace overlap is not
  fully characterized. Two adapters with high overlap but extreme magnitude
  ratio may still merge poorly.
- Principal angle analysis captures linear subspace geometry but not
  nonlinear interactions that may arise in deep networks.
- *Energy masking direction (N128, April 2026)*: For same-task pairs,
  the SV-weighted overlap (Gradience's operational metric) is consistently
  lower than the unweighted mean cosine by a mean of 0.21. Energy-weighting
  deflates apparent compatibility for same-task pairs; it does not inflate
  it. The metric's failure mode is therefore conservative (false positives)
  rather than liberal (false negatives). N128 found zero false-negative
  candidates across the encoder validation corpus (N = 8 same-task pairs
  with known merge outcomes).

### Reproducibility

Requires two adapter directories and `gradience merge-audit`. The
`SubspaceMetrics` dataclass in `gradience.vnext.merge.spectral_compat`
defines all computed quantities.


---

## 9. Structural Compatibility Necessary but Not Sufficient (Study 16, March 2026)

### Claim

Spectral overlap is necessary but not sufficient for merge quality.
Structurally compatible merges can still fail behaviorally when one or
both source adapters are weak. This finding drove the introduction of
eligibility gating into the Gradience recommendation engine.

### Evidence

- **Design.** 5 Llama-2-7B adapter pairs with end-to-end merge ablation.
  Frobenius norm ratios up to 19.7× across pairs. Merge conditions:
  raw linear averaging, norm-equalized averaging, and per-layer
  strategy selection.

- **Key observation.** Pair 06 (one weak source adapter) merged with
  favorable structural metrics (moderate overlap, low conflict) but
  produced a behaviorally disappointing merge. Structural diagnosis alone
  would have marked this pair as safe.

- **Norm-equalized merging.** Geometric-mean Frobenius norm rescaling
  before linear averaging removes scale imbalance as a confound and
  often matches or beats per-layer strategies. Promoted to first-class
  strategy in v0.11.

- **Architecture consequence.** Recommendation pipeline redesigned into
  two stages: Stage A (Diagnosis) extracts structural facts (overlap,
  conflict dimensions, magnitude ratios) and outputs LayerDiagnosis and
  PairDiagnosis. Stage B (Policy) translates diagnosis to strategy
  and coefficients, modulated by source eligibility.

- **Eligibility gating.** `EligibilityStatus` enum introduced: ELIGIBLE,
  UNCERTAIN, FLAGGED_WEAK, UNKNOWN. `classify_eligibility()` compares
  adapter to base model on user-reported behavioral metrics. Default
  margin: 0.0 (any improvement over base = ELIGIBLE).

- **Test coverage.** 396 merge tests total, 23 specifically encoding
  Study 16 conclusions. Flagged-weak-never-silent invariant tested
  across all code paths.

### Limitations

- Eligibility classification thresholds are not empirically calibrated
  beyond the Study 16 observations. Broader calibration on diverse
  corpora is a planned future task (see `docs/plans/`).
- Norm-equalized vs. audit-aware merge strategies not compared by
  automated benchmark; the claim that norm-equalized "often matches"
  rests on Study 16 observations, not systematic ablation.
- No N-way merge support; all analysis is pairwise.

### Reproducibility

Engine integration notes in `docs/ENGINE_NOTES_STUDY16.md`.
Test suite: `tests/merge/` (396 tests).


---

## 10. Compression Is Not a Core Workflow (Study 17, March 2026)

### Claim

Pre-merge spectral compression does not meaningfully improve merge
outcomes. At 95% cumulative energy retention, compression is effectively
a no-op. At more aggressive thresholds (90%, 80%), compression is
behaviorally low-cost but produces only small and inconsistent
improvements. Compression survived but did not win.

### Evidence

- **Study 17A.** Tested whether 95% cumulative-energy compression
  improves merge outcomes. Result: no meaningful structural improvement.
  95% threshold retains too much of the spectrum to make a difference.

- **Study 17B.** Tested full_normeq, comp90_normeq, comp80_normeq on
  pair_03 and pair_04. Aggressive compression was behaviorally low-cost,
  with some worst-side improvements, but effect sizes were small and
  no large transformative gains were observed.

- **Product decision.** Compression demoted from core workflow to
  experimental/gated/advanced feature. The validated Gradience spine
  became: (1) single-adapter QA, (2) source eligibility judgment,
  (3) pairwise merge-risk reporting, (4) strict-QA blocking,
  (5) inventory summary/aggregation.

### Limitations

- Tested on a small number of adapter pairs (pair_03, pair_04).
  Compression might have larger effects on pairs with very different
  spectral profiles (e.g., one highly compressed, one broadly utilized).
- Study 17A tested only one threshold (95%). The space between 80%
  and 95% was sampled sparsely.
- The decision to demote compression is a product judgment, not a
  universal claim. Compression may be valuable in contexts Gradience
  does not currently address.

### Reproducibility

Decision document: `docs/study17-compression-conclusion.md`.
Study data in `Gradience II/` (when consolidated: `results/study17_compression/`).


---

# Part C — Spectral Partitioning and Subspace Geometry

This strand addresses the deeper geometric question: why do
independently trained adapters share certain spectral directions
and diverge in others? The findings here connect post-hoc spectral
audits to training dynamics and pre-trained weight structure.

---

## 11. Marchenko-Pastur Spectral Partitioning (N127, April 2026)

### Claim

The Gavish-Donoho optimal hard threshold partitions adapter spectra
into high-SV and low-SV bands with differential inter-adapter alignment.
High-SV directions show 1.8× greater alignment than low-SV directions
when measured with an SV-weighted metric (d ≈ 1.15, p ≈ 10⁻⁵). The
partition is task-dependent: same-task pairs show 7.8× high/low ratio
versus 2.5× for cross-task pairs.

### Evidence

- **Design.** 3 independently trained adapters on DistilBERT-base-uncased
  (SEQ_CLS, seeds 42/123/456, rank 16, checkpoint-50). 3 same-seed-pair
  comparisons across 24 layers (6 transformer layers × 4 modules: Q, K,
  V, O). Three rank policies tested: probe_r16, uniform_median_r16,
  uniform_p90_r16.

- **SV-weighted alignment (Tian et al. metric).** Σᵢ Σⱼ σᵢ·σⱼ·|cos(uᵢ,uⱼ)|
  / (Σσᵢ · Σσⱼ):

  | Band          | Alignment   | Std         | Energy fraction |
  |---------------|-------------|-------------|-----------------|
  | High-SV       | 0.098--0.102| 0.040--0.046| 49--50%         |
  | Low-SV        | 0.054--0.056| 0.016       | 50--51%         |
  | Full spectrum  | 0.061       | 0.015--0.017| 100%            |

  High/Low ratio: 1.8× (consistent across all 3 policies).
  Paired t-test: t ≈ 5.5, p ≈ 1×10⁻⁵, Cohen's d ≈ 1.15.

- **Metric divergence.** Unweighted mean cosine shows the opposite pattern:
  high-SV band 0.110--0.114, low-SV band 0.207. This inversion (high/low
  ratio 0.5×, t ≈ -9.8, p ≈ 10⁻⁹, d ≈ -2.0) is not contradictory but
  reflects dimensionality confounding: the MP threshold selects only
  1--3 directions (mean k = 1.3 of 16) carrying ~50% of energy. The
  SV-weighted metric is the correct normalization.

- **Module-type breakdown (SV-weighted):**

  | Module | High-SV | Low-SV | Ratio |
  |--------|---------|--------|-------|
  | Q      | 0.121   | 0.059  | 2.0×  |
  | K      | 0.140   | 0.074  | 1.9×  |
  | V      | 0.088   | 0.043  | 2.0×  |
  | O      | 0.061   | 0.048  | 1.3×  |

  All module types show high > low. V-module strongest separation.
  O-module weakest.

### Limitations

- Small encoder architecture (DistilBERT-base). Replication on
  decoder-only models at scale is needed.
- Only 3 adapter pairs; variance estimates are accordingly uncertain.
- The unweighted/weighted metric divergence, while explicable, means
  the conclusion depends on which normalization is used. The argument
  for SV-weighting is principled (magnitude matters for merge) but
  not universally accepted.

### Reproducibility

Scripts: `scripts/mp_partition_extensions.py`. Data: `sidecar/results/mp_partition_test/`.
Research note: `sidecar/notes/n127_mp_partition_test.md`.


---

## 12. Spectral Partitioning Is Task-Dependent (N127 Extension 2, April 2026)

### Claim

The high-SV alignment ratio is strongly modulated by task relationship.
Same-task adapter pairs (SST-2 × SST-2, different seeds) show 7.8×
high-SV/low-SV alignment ratio; cross-task pairs (SST-2 × QNLI, etc.)
show 2.5×. The same-task/cross-task separation in high-SV alignment
is the single strongest effect observed in the spectral partitioning
experiments.

### Evidence

  | Pair type   | N  | High-SV | Low-SV | Full  | H/L ratio |
  |-------------|-----|---------|--------|-------|-----------|
  | Same-task   | 1   | 0.634   | 0.081  | 0.324 | 7.8×      |
  | Cross-task  | 4   | 0.133   | 0.054  | 0.089 | 2.5×      |

- Same vs. cross (high-SV): t = 23.4, p = 3.4×10⁻⁴⁶.
- Same vs. cross (low-SV): t = 9.2, p = 1.5×10⁻¹⁵.

  High-SV alignment drops 4.8× from same-task to cross-task pairs.
  Low-SV alignment also drops, but less dramatically (1.5×), consistent
  with the interpretation that low-SV directions are less task-specific
  but not fully random.

### Limitations

- Only 1 same-task and 4 cross-task comparisons. The 7.8× vs 2.5×
  ratio should be treated as an order-of-magnitude estimate.
- Same encoder architecture and scale as §11. Task-dependence at
  decoder scale is supported by Post 3 (§7, same-task/cross-task
  separation 2.4×) but has not been tested with the full MP
  partition methodology at that scale.

### Reproducibility

Same scripts and data as §11 (Extension 2 in `scripts/mp_partition_extensions.py`).


---

## 13. Pre-Trained Spectral Concentration Predicts Alignment (N127 Extension 1, April 2026)

### Claim

The pre-trained weight matrix W₀'s spectral energy concentration
(not raw spectral gap) predicts adapter high-SV alignment strength.
This is task-dependent: the relationship holds for QNLI adapter pairs
(r = 0.53--0.58, p < 0.01) but not for SST-2 pairs (r ≈ 0.01--0.04,
p > 0.40).

### Evidence

- **Method.** For each of the 24 layers, compute W₀ energy concentration
  (energy_top_1, energy_top_2: fraction of Frobenius energy in the top
  1 or 2 singular values). Correlate against high-SV alignment from
  the adapter pair audit.

- **QNLI pairs:**
  - energy_top_1 vs. high-SV alignment: r = 0.531, p = 0.008
  - energy_top_2 vs. high-SV alignment: r = 0.577, p = 0.003

- **SST-2 pairs:** No significant correlations (r ≈ 0.01--0.04, p > 0.40).

- **Naive gap metric failure.** The raw spectral gap (σ₁ - σ₂ of W₀)
  does not predict alignment for either task. Energy concentration is
  the relevant quantity, consistent with a Davis-Kahan-style perturbation
  argument where concentration (not gap) controls the stability of the
  dominant subspace under additive perturbation.

### Limitations

- Two tasks, one architecture. The task-dependence (QNLI yes, SST-2 no)
  is unexplained and may indicate that the relationship requires a
  minimum level of spectral structure in W₀ that happens to be present
  for QNLI layers but not SST-2 layers.
- Moderate correlations (r ≈ 0.5--0.6) with n = 24 layers. These are
  meaningful but not strong enough for layer-level prediction.
- The link to Davis-Kahan is conceptual, not formal. A rigorous
  concentration-weighted perturbation bound remains an open
  theoretical problem (see THEORY.md §7, OQ6).

### Reproducibility

Extension 1 in `scripts/mp_partition_extensions.py`.


---

## 14. Spectral Partitioning Converges During Training (N127 Extension 3, April 2026)

### Claim

High-SV alignment increases monotonically during training (steps 50--200)
and plateaus around step 150. The spectrum simultaneously sharpens, with
high-SV energy fraction rising from 56% to 86%. Low-SV alignment
remains approximately flat throughout.

### Evidence

- **Extended checkpoint progression** (uniform_median_r16 policy, steps
  50--200):

  | Step | High-SV | Low-SV | H/L ratio | E_high |
  |------|---------|--------|-----------|--------|
  | 50   | 0.244   | 0.060  | 4.05×     | 56.1%  |
  | 100  | 0.547   | 0.070  | 7.76×     | 73.1%  |
  | 150  | 0.607   | 0.075  | 8.11×     | 83.9%  |
  | 200  | 0.608   | 0.076  | 8.01×     | 85.7%  |

- **Spearman correlations.** High-SV alignment vs. step: r = 1.000,
  p < 0.001 (monotonic). H/L ratio vs. step: r = 0.800, p = 0.200
  (directionally positive but not significant at n = 4 checkpoints).

- **Plateau behavior.** High-SV alignment rises 2.5× from step 50 to
  step 150 (0.244 → 0.607), then stabilizes. Energy concentration
  continues to increase slightly (83.9% → 85.7%) but the alignment
  metric saturates.

### Limitations

- Only 4 checkpoints. The plateau at step 150 could be an artifact of
  sparse sampling; denser checkpoint coverage would reveal whether the
  convergence is smooth or step-wise.
- Small encoder scale only. Whether the plateau step is
  architecture-dependent or general is an open question.
- The plateau could reflect training convergence (no further learning)
  rather than a geometric saturation property. Distinguishing these
  requires training-loss analysis at each checkpoint.
- The plateau in high-SV alignment at step ~150 may correspond to the
  curvature collapse events identified in the curvature telemetry paper
  (§16a). If these are the same phenomenon observed through different
  instruments (Hessian eigenvalues vs. MP-partitioned SVD alignment),
  the curvature telemetry signal would serve as an online proxy for
  spectral partition quality. This correspondence has not been tested
  (see THEORY.md §7.2, "Curvature-partition correspondence").

### Reproducibility

Extension 3 in `scripts/mp_partition_extensions.py`. Results in
`sidecar/results/mp_partition_test/extension_results.json`.


---

# Part D — Training Dynamics and Curvature Telemetry

This strand establishes the *during-training* face of Gradience's spectral
measurement framework. Where Parts A--C characterize finished adapters and
their interactions, Part D asks: can the same spectral lens, applied to
loss-surface geometry during training, provide leading indicators of model
learning? The curvature telemetry results (§16a) show that it can — Hessian
energy forecasts validation accuracy 3--6 updates ahead with ~36% RMSE
improvement over a persistence baseline. The findings below trace the
development of this result: from regime classification (§15) through
Hessian telemetry detection (§16) and curvature telemetry forecasting
(§16a) to the three-act gradient alignment structure (§17) and DFA scaling
exponents (§18).

---

## 15. Regime Classification via Early Geometric Features (Reanalysis, March 2026)

### Claim (revised)

Geometric features extracted from the first 200 training steps achieve
~67% five-class regime classification accuracy via Leave-One-Seed-Out
cross-validation, significantly above chance (permutation p = 0.0001).
Loss-only features achieve ~40% (p = 0.0009). The gap between the two
approaches is not statistically significant at the current sample size
(McNemar's p = 0.289). Geometric features carry 7.4× more mutual
information about training regimes than loss.

**Note.** An earlier, informal claim of "100% geometric accuracy vs. 65%
for loss" appeared in project blog posts. This figure is not reproduced
on the available five-class data. It likely derives from a binary
classification problem (baseline vs. low-weight-decay), where geometry
does achieve perfect separation with margins of 0.81--0.99. See the
full reanalysis report in `Gradience II/reanalysis/REANALYSIS_REPORT.md`.

### Evidence

- **Permutation test (n=10,000).** Observed geometry accuracy 66.7%
  (10/15); null mean 13.0%, null 99th percentile 46.7%. p = 0.0001.

- **Bootstrap confidence interval (B=5,000).** Mean 47.4%, std 14.1%,
  95% CI [20.0%, 73.3%]. The wide CI reflects the n=15 constraint.

- **Feature ablation.** No single geometric feature exceeds 40% accuracy.
  Best pair: `weight_norm_mean` + `grad_to_weight_ratio_mean` (66.7%).
  Removing `cos_grad_weight_mean` *improves* accuracy to 80.0%,
  suggesting this feature introduces noise at this sample size.

- **Information theory.** KSG mutual information of geometric features
  with regime labels: 4.96 nats (joint). Loss MI: 0.67 nats.
  Ratio: 7.4×. Slight synergy (negative redundancy of -0.17 nats).

- **Minimum description length.** BIC favors loss-only (35.25) over
  geometry (67.92) due to the 35-parameter vs. 10-parameter complexity
  penalty at n=15. Geometry achieves lower NLL but cannot justify its
  parameter cost at this sample size.

### Limitations

- Five regimes (baseline, high_lr, high_wd, low_lr, low_wd) with only
  3 seeds each (n=15 total). This is the minimum viable sample for LOSO.
  Doubling to 6 seeds would substantially narrow the bootstrap CI.

- The `early_spectral_complexity_mean` feature — theoretically the most
  important geometric quantity — was never computed. Classification
  results are based on 6 weight/gradient features only.

- The McNemar test is underpowered at n=15. The test cannot distinguish
  67% from 40% accuracy at conventional significance levels.

### Reproducibility

Full protocol, scripts, and module-level JSON results in
`Gradience II/reanalysis/`. Protocol document:
`Gradience_Reanalysis_Protocol.md`.


---

## 16. Hessian Telemetry and Phase Transition Candidates (Reanalysis, March 2026)

### Claim

Geometric metrics (Hessian trace, top eigenvalue) detect training regime
transitions approximately 300 steps before loss metrics in a single-run
telemetry stream (600 records, steps 1--60,000). The Hessian trace is the
single earliest-responding metric. One candidate phase transition was
identified near step 58,450.

### Evidence

- **Changepoint detection.** Earliest CUSUM changepoint: trace_H at
  step 1,900; lambda1 at step 8,800; train_loss at step 12,400.
  Geometry leads loss by 300--10,500 steps depending on metric.

- **Phase transition candidate.** Susceptibility peaks in both
  `sharpness_ratio` and `gHg_ratio` cluster near step 58,450.
  Trajectory tortuosity spikes in the same region (max tortuosity
  30,595 at window size 20 near step 59,000).

- **Critical slowing down.** Autocorrelation time of lambda1 and
  trace_H peaks at step 50,500. However, loss autocorrelation peaks
  earlier (step 3,500), complicating the "geometry detects transitions
  first" narrative for CSD specifically.

- **Canonical correlation.** First canonical correlation between
  Hessian-space metrics (lambda1, trace_H, gHg, sharpness) and
  representation-space metrics (participation ratio, anisotropy, CKA,
  rqi_star): CC1 = 0.661. The two measurement systems share a
  meaningful common signal but are not redundant.

### Limitations

- Single training run. All time-series dynamics are conditional on the
  specific hyperparameters and model of that run.

- The canonical correlation analysis compares metrics across different
  runs (telemetry.jsonl vs. telemetry10.csv), making alignment
  correlational rather than causal.

- The phase transition candidate at step 58,450 has not been replicated.
  It may be an artifact of the specific run.

### Reproducibility

Analysis scripts: `Gradience II/reanalysis/module_b_timeseries.py`,
`module_d_phase_transitions.py`, `module_e_cross_strand.py`.
Data: `Gradience II/results/telemetry.jsonl` and `telemetry10.csv`.

### Relationship to §16a

The curvature telemetry paper (§16a below) provides a more rigorous
statistical framework for the lead-lag claim reported here. This section
(§16) used CUSUM changepoint detection on a single run to establish that
geometric metrics detect transitions ~300 steps before loss. §16a uses
CCF with pre-whitening, block-bootstrap CIs, and surrogate-null tests on
multiple runs to establish that curvature features *forecast* accuracy
with quantified skill. The two findings are complementary: §16 provides
macro-scale detection (changepoints over 60,000 steps), §16a provides
micro-scale forecasting (walk-forward prediction over ~180 updates).


---

## 16a. Curvature Telemetry: Hessian Energy as a Leading Indicator of Validation Accuracy (April 2026)

### Claim

Hessian energy ($\sum \lambda^2$) leads validation accuracy by 3--6 updates
during LoRA fine-tuning, and walk-forward forecasters using only curvature
features reduce short-horizon accuracy RMSE by ~36% versus a persistence
baseline. The lead-lag relationship is validated with AR(1) pre-whitening,
effective sample size correction, contiguous block-bootstrap CIs, and
surrogate-null tests.

### Evidence

- **Model and setup.** GPT-2 small (124M parameters) with LoRA (~0.59M
  trainable), AdamW optimizer, moderately aggressive learning rate.
  Tasks: toy arithmetic (synthetic chain-of-thought), GSM8K-lite.
  Snapshot cadence: every 5--6 updates, ~180 updates total per run.
  Deterministic finite-difference Hessian estimators (fixed probe
  directions, not stochastic Hutchinson).

- **Cross-correlation function.** CCF peak at negative lags ($-2$ to
  $-6$); representative run: pre-whitened peak at lag $-5$. Curvature
  changes precede accuracy changes, not the reverse.

- **Curvature dynamics.** First sustained accuracy jump preceded by
  ~1.4M-unit drop in $\sum \lambda^2$. The pattern: Hessian energy rises
  during exploration (optimizer in high-curvature regions), collapses
  during consolidation (escape to flatter basins), accuracy improves
  shortly after (representation stabilizes in new basin).

- **Walk-forward forecasting.** Expanding-window ridge regression with
  leakage-free walk-forward evaluation. RMSE ~0.0042 vs. 0.0065
  persistence baseline (~36% improvement). The improvement is robust
  to the specific ridge parameter (cross-validated).

- **Statistical validation.** Surrogate-null tests (phase randomization
  and circular rotation, 1000 surrogates each) reject the null
  hypothesis that the observed lead-lag is an artifact of shared
  autocorrelation structure. Block-bootstrap CIs for the CCF peak
  coefficient exclude zero.

### Connection to existing findings

- **§16 (Hessian Telemetry).** The curvature telemetry paper provides
  the *forecasting* complement to §16's *detection* results. §16 shows
  that geometric metrics detect changepoints ~300 steps before loss;
  §16a shows that $\sum \lambda^2$ can *forecast* near-future accuracy with
  quantified skill, not just detect that something changed.

- **§17 (Three-Act Structure).** The curvature telemetry paper's core
  dynamic — high curvature during exploration, collapse during
  consolidation, accuracy improvement after collapse — is the
  micro-level version of the three-act structure (explore, lock-on,
  destabilize) observed at macro scale on Mistral-7B. The two results
  describe the same phenomenon at different temporal resolutions and
  model scales.

- **§14 (Spectral Partitioning Converges During Training).** The
  curvature collapse events are hypothesized to coincide with the
  moments when high-SV alignment sharpens — the spectral partition
  crystallizing as the optimizer settles into a flatter basin. This is
  an open prediction, not yet tested (see THEORY.md §7.2,
  "Curvature-partition correspondence").

### Limitations

- GPT-2 small + LoRA only; scaling to larger models may require
  stochastic Hessian estimators (Hutchinson or stochastic Lanczos)
  rather than the deterministic finite-difference approach used here.
- Short runs (~180 updates); long pretraining dynamics may differ.
- Finite-difference estimators assume local quadraticity of the loss
  surface; strong non-convexity could distort the estimates.
- Learning rate and snapshot cadence shape the CCF; mitigated by
  partialling out LR and detrending, but residual confounds are
  possible.

### Reproducibility

Manuscript, code, and 601-record telemetry dataset available.
Deterministic finite-difference approach ensures exact replication with
same probe directions and random seeds.


---

## 17. Three-Act Gradient Alignment and Representation Compression (Post 5, March 2026)

### Claim

A single Mistral-7B fine-tuning run exhibits a three-act structure
in gradient--Hessian alignment (Rayleigh quotient Rq = gᵀHg / (gᵀg · λ₁))
and an expand-then-compress trajectory in representation space
(participation ratio). These dynamics are consistent with the
edge-of-stability literature.

### Evidence

- **Data source.** Single run: Mistral-7B, LoRA rank 64, GSM8K, 60,000
  steps. Hessian stream: 601 measurements. Representation stream: 1,200
  measurements.

- **Gradient alignment.**
  - Mean Rq over 60k steps: 0.90
  - Act I (steps 1--12,000): mean 0.79, low variance (exploration)
  - Act II (steps 12,000--36,000): mean 1.06 (locked onto dominant
    direction, edge-of-stability)
  - Act III (steps 36,000--60,000): mean 0.78, variance increases
    (destabilization)

- **Representation dynamics.**
  - Expansion (steps 100--24,000): PR climbs from 41 to mid-60s
  - Compression (steps 24,000--72,000): negative slope 4× steeper
    than expansion
  - Plateau (steps 72,000--120,000): PR stabilizes around 64
  - PR--anisotropy coupling: ρ = -0.74, p ≈ 10⁻²⁰⁸

- **CKA convergence.** Power-law model (CKA ~ t^α): α = -0.40, R² = 0.29.
  Early convergence approximately 12× faster than late. Scale-free
  dynamics suggest criticality but the fit explains less than 35%
  of variance.

- **DFA exponents.**

  | Metric             | DFA α | Interpretation        |
  |--------------------|-------|-----------------------|
  | λ₁                 | 0.97  | Near-1/f, scale-free  |
  | gHg                | 0.92  | Near-1/f              |
  | Train loss         | 1.01  | 1/f noise             |
  | λ₁/trace_H (order) | 0.68  | Long-range correlated |
  | trace_H            | 1.24  | Superdiffusive drift  |

### Limitations

- Single run, single scale, single task. The three-act structure has not
  been validated across seeds or regimes.
- CKA power-law fit explains less than 35% of variance.
- DFA interpretation is provisional: exponents > 1.0 indicate trending
  (non-stationary) series, complicating the "long-range correlation"
  interpretation.

### Reproducibility

Full analysis in `Gradience II/docs/blog_series/SERIES_POST_5_SPECTRAL_MICROSCOPE.md`.
Data: `Gradience II/results/telemetry/`.


---

## 18. Regime-Dependent DFA Exponents (Study 12, March 2026)

### Claim

Training regimes produce different long-range temporal correlation
structures in spectral complexity time series. Detrended Fluctuation
Analysis (DFA) exponents for spectral complexity differ significantly
across five hyperparameter regimes (one-way ANOVA F = 116.86,
p ≈ 7.7 × 10⁻²³). High learning rate produces markedly lower
persistence (α ≈ 1.574) than low learning rate (α ≈ 2.073), with
baseline, high weight decay, and low weight decay clustering in between
(α ≈ 1.90--1.92).

Separately, the regime classification results from §15 replicate at
n=49 with stable but moderate accuracy: spectral complexity alone
achieves 83.7%, 6 geometric features 79.6%, loss only 73.5% (all
p = 0.0002 by permutation test). No feature set is significantly
better than any other by McNemar's test.

### Evidence

- **Design.** 5 regimes × 10 seeds = 50 runs (49 usable; 1 empty
  telemetry file). NanoGPT 6-layer, Shakespeare character-level.
  Same telemetry pipeline as Study 11.

- **Classification (LOSO centroid classifier).**
  - Loss only: 73.5% (36/49), permutation p = 0.0002
  - 6 geometric features: 79.6% (39/49), p = 0.0002
  - Spectral complexity only: 83.7% (41/49), p = 0.0002
  - McNemar: spectral vs. loss p = 0.30; geometry_7 vs. loss p = 0.58

- **DFA exponents (spectral complexity).**
  - low_lr:  α = 2.073 ± 0.025 (n=10)
  - high_wd: α = 1.917 ± 0.026 (n=10)
  - baseline: α = 1.905 ± 0.058 (n=9)
  - low_wd:  α = 1.896 ± 0.048 (n=10)
  - high_lr: α = 1.574 ± 0.077 (n=10)
  - One-way ANOVA: F = 116.86, p = 7.71 × 10⁻²³

- **DFA exponents (gradient norm).**
  - high_lr: α = 1.053 ± 0.097 (highest)
  - high_wd: α = 0.687 ± 0.042 (lowest)

### Interpretation

The DFA result is distinct from the classification result in kind.
Classification asks whether features differ in central tendency across
regimes. DFA asks whether the *temporal correlation structure* of a
feature differs. These are orthogonal: identical means can coexist
with different dynamics, and vice versa.

The finding that high learning rate specifically disrupts temporal
persistence of spectral complexity (α ≈ 1.57 vs. ≈ 1.90--2.07 for
other regimes) suggests that learning rate modulates the stochastic
character of Hessian spectral evolution.

### Limitations

- Single architecture (NanoGPT, 6 layers) and single task (Shakespeare
  character-level). Universality across models and tasks is untested.

- DFA exponents > 1.0 indicate non-stationary (trending) series.
  Standard DFA interpretation (0.5 = white noise, 1.0 = 1/f) applies
  to stationary series. The between-regime comparison remains valid
  but the absolute values should not be interpreted as "long-range
  correlation" in the classical sense.

### Reproducibility

Launcher: `Gradience II/reanalysis/study12_replication/run_study12_replication.sh`
Analysis: `Gradience II/reanalysis/study12_replication/analyze_study12.py`
Protocol: `Gradience II/reanalysis/study12_replication/STUDY12_PROTOCOL.md`
Results: `Gradience II/analysis/study12/study12_results.json`


---

# Part E — Product Decisions Derived from Empirical Findings

This section records cases where empirical results directly drove
architectural or product decisions in the Gradience library. These
are not findings in themselves but document the research-to-engineering
translation path.

---

## 19. Recommendation Engine Design (Post 8)

Post 8 documents the construction of Gradience's merge recommendation
engine as a structural-to-behavioral validation pipeline. The engine
implements a two-stage architecture (structural diagnosis → policy
modulation by eligibility) derived from Study 16's finding that
structural compatibility alone is insufficient (§9). The recommendation
engine does not independently validate behavioral claims; it gates
recommendations on user-reported evidence. See
`Gradience II/docs/blog_series/` (Post 8, when consolidated:
`SERIES_POST_8_FINAL.md`).


---

## 20. Core Workflow Selection (Studies 16--17)

The combination of Study 16 (§9, structural ≠ sufficient) and Study 17
(§10, compression does not win) determined the library's core workflow:
adapter QA → eligibility screening → pairwise merge-risk → inventory
preflight. Compression was retained as an experimental feature only.
This is documented in `docs/study17-compression-conclusion.md`.


---

## 21. External Literature Convergence (2025–2026)

### Claim

Four independent research programs, published in 2025–2026 and working from
different vantages than Gradience's, have produced results that constitute
external confirmation of four core Gradience claims: (a) subspace misalignment
between LoRA adapters predicts merge failure; (b) per-layer singular vector
geometry is the operative predictor rather than aggregate statistics; (c)
the cross-term in the merged adapter spectrum is the mathematical object
governing merge outcome; and (d) structural compatibility is necessary but
not sufficient for merge quality.

### Evidence

- **Subspace misalignment predicts failure (KnOTS; Stoica et al., ICLR 2025;
  arXiv:2410.19735).** SVD-based alignment of LoRA task-updates before merging
  improves outcomes by up to 4.3% across vision and language benchmarks. The
  core diagnostic is that LoRA fine-tuned models exhibit significantly lower
  inter-adapter alignment (CKA) than fully fine-tuned counterparts, and that
  improving this alignment is the mechanism of improvement. Task-vector
  orthogonality alone does not reliably predict merge difficulty — consistent
  with Gradience's conjunctive failure model, which shows readout orthogonality
  alone explains nothing. KnOTS and Gradience address complementary problems:
  KnOTS improves merges for pairs that proceed; Gradience decides which pairs
  should proceed. The alignment deficit they diagnose is the same structural
  condition Gradience's pairwise audit measures.

- **Per-layer principal angles are the operative predictor (Task Singular
  Vectors; Gargiulo et al., CVPR 2025; arXiv:2412.00081).** Merge interference
  is proportional to the cosine of the angle between singular vectors of
  different task matrices — formally equivalent to Gradience's principal-angle
  compatibility metrics. Independent compression experiments confirm that task
  matrices are low-rank and that dominant singular directions carry nearly all
  task-relevant content, replicating §§1–3 above on different data and
  architectures.

- **Cross-term formalization (Akbar et al., ICML 2025 Workshop;
  OpenReview:t9FrMviTaP).** Formal proof that direct LoRA merging (combining
  A and B matrices separately) introduces an interfering cross-term that
  degrades performance, while multiplied merging avoids it via linear mode
  connectivity. The cross-term they identify is algebraically the same
  interaction quantity $z = \text{sign}(\delta) \cdot \cos(\theta) \cdot
  \cos(\phi)$ derived in the Technical Report §2.3. Two independent
  theoretical programs arrive at the same mathematical object from opposite
  directions — one predicting which pairs to avoid, one improving how to merge
  the rest.

- **Structural ≠ behavioral quality at ecosystem scale (Badirli et al.,
  arXiv:2602.12323).** A survey of public LoRA adapter reuse on HuggingFace
  Hub finds that structural compatibility is not sufficient for merge quality,
  and that source adapter behavioral quality is an independent determinant.
  This is an independent ecosystem-scale replication of Study 16's
  structural-behavioral separation (§9). The paper additionally documents that
  Hub adapter quality is often poor or poorly characterized, confirming the
  evidence gate design (§19) at scale: behavioral screening before merging
  is not a conservative edge case but a routine necessity.

### Limitations

- These are not direct replications of Gradience's experiments. Architectural
  differences, task domains, adapter training procedures, merge methods, and
  evaluation criteria are all distinct. The convergence is structural — the
  same geometric claims and the same mathematical objects — not numerical.
- KnOTS and TSV operate on vision transformers and larger NLP models; the
  cross-term paper uses image classification. No paper directly studies
  the small-encoder classification regime that constitutes Gradience's primary
  validation corpus.
- The conjunctive failure mechanism (V-module pathology + readout incompatibility)
  has not been independently investigated. External convergence is on the
  spectral-geometry machinery, not on the specific causal model.

### Reproducibility

Citations with arXiv identifiers in the Technical Report reference section.
Mathematical correspondences can be verified by comparing: Technical Report
§2.3 interaction term with Akbar et al. cross-term derivation; Gradience's
`mean_overlap` metric with Gargiulo et al.'s cosine-angle interference
measure. Code: `gstoica27/KnOTS`, `AntoAndGar/task_singular_vectors`.


---

## 22. Portfolio Rank Collapse (N129, April 2026)

### Claim

Pairwise triage is necessary but not sufficient for portfolio-scale merging.
Additive combination of retained adapters via Task Arithmetic produces
spectral concentration (rank collapse) that grows linearly with pool size,
with onset at k=2–3 in the field trial corpus — below the previously assumed
safe threshold of k=5.

### Evidence

OLS slope β₁ of skewness ratio ρ = σ₁/mean(σ), normalized by per-inventory
k=1 baseline, vs pool size k. One-sided t-test for β₁ > 0 (H_null: no
spectral collapse). Mean(σ) computed over effective (non-noise) singular
values only (noise floor: 1e-10 × σ₁).

| Inventory | Backbone | |R| | β₁ | p-value | k_collapse_rho (2×baseline) |
|-----------|----------|-----|------|---------|------------------------------|
| inv_02 | roberta-base | 2 | 0.380 | <0.001 | extrap 3.6 |
| inv_03 | distilbert | 4 | 0.923 | 0.024 | 2 |
| inv_04 | distilbert | 4 | 0.284 | <0.001 | extrap 4.5 |
| inv_05 | bert-base | 6 | 0.349 | <0.001 | 3 |

Mean β₁ = 0.48 ± 0.15. H_null rejected in all inventories. β₁ < 1.0 in
all cases (prediction P1 confirmed), consistent with triage selecting
spectrally compatible subsets relative to the Skorobogat et al. (2025)
theoretical rate of ~1.0 for random pools. Mixed-task pools collapse faster
than same-task (P2 confirmed). k_collapse ≤ 5 in ≥ 2 inventories (P3
disconfirmed). Triage selection (H_selection) did not measurably slow
collapse relative to full pools.

inv_03 β₁ = 0.923 is driven by rank heterogeneity (mixing rank-1 and
rank-16 adapters); homogeneous rank-1 pools show β₁ = 0.28–0.35.

### Limitations

- Small inventories: largest retained pool has |R|=6. Linear model fit on
  2–6 points. Confidence intervals on β₁ are wide.
- No behavioral validation: collapse curves measure spectral structure only.
  No merge-then-evaluate results for k > 2 pools.
- Encoder-only regime: DistilBERT, BERT-base, RoBERTa-base. Decoder-scale
  models with higher-rank adapters may show different dynamics.
- Adapter discovery gaps: some retained adapters in inv_04 could not be
  matched to adapter_cache entries.
- The k_collapse metric uses ρ (skewness ratio), not ε (energy fraction).
  For rank-1 adapters, ε crosses 0.80 at k=2 trivially (baseline ε = 1.0
  by construction). See N129 sidecar note, "Metric clarification" section.

### Reproducibility

Script: `scripts/portfolio_rank_collapse_probe.py`. Results:
`sidecar/results/N129_rank_collapse/results.json`. Requires field trial
inventories in `field_trials/inventory_0{2,3,4,5}/`. RNG seed: 42.
Pre-registration: inline in conversation (April 5, 2026). Study note:
`sidecar/notes/N129_rank_collapse_probe.md`.


---

## 23. DeBERTa Adjudication: Cross-Architecture Replication of Spectral Triage (N07, April 2026)

### Claim

The spectral triage pipeline's core claims — spectral partitioning,
task-discriminating compatibility scores, norm imbalance diagnosis, curvature
lead-lag, and phase transition detection — replicate on DeBERTa-v3-base, a
third encoder architecture with a distinct pretraining objective (replaced
token detection vs. masked language modeling). Seven pre-registered predictions
(A–G) were tested on 8 adapters (4 GLUE tasks × 2 seeds), 28 merge-audit
pairs, and 52 merge evaluations. All seven are supported, six with strong
statistical significance.

### Evidence

- **Design.** DeBERTa-v3-base (185M parameters) with LoRA rank 16 targeting
  query_proj, key_proj, value_proj, and attention.output.dense (96 LoRA
  tensors, 1.18M trainable / 185.6M total = 0.64%). Tasks: QNLI, SST-2,
  RTE, MRPC. Seeds: 42, 7. Training: 3 epochs, lr=2e-4, AdamW, bf16,
  warmup 6%. Curvature sidecar (Hutchinson trace + power iteration)
  and structural SVD snapshots every 50 steps.

- **Individual adapter quality.** All 8 adapters well above 70% threshold:

  | Adapter | Accuracy | Training samples |
  |---------|----------|-----------------|
  | deberta_qnli_s42 | 94.47% | 104,743 |
  | deberta_qnli_s7 | 94.33% | 104,743 |
  | deberta_sst2_s42 | 95.30% | 67,349 |
  | deberta_sst2_s7 | 95.76% | 67,349 |
  | deberta_rte_s42 | 81.23% | 2,490 |
  | deberta_rte_s7 | 80.14% | 2,490 |
  | deberta_mrpc_s42 | 87.75% | 3,668 |
  | deberta_mrpc_s7 | 86.52% | 3,668 |

- **Prediction A: Same-task pairs show higher spectral redundancy.**
  *Supported (p < 10⁻¹⁵).* Same-task compatibility score mean 0.449 vs.
  cross-task 0.160 (t = 15.87). Redundancy fraction: same-task 42.7% vs.
  cross-task 1.4% (Mann-Whitney p < 0.001). This replicates §12's
  task-dependent partitioning on a third backbone.

- **Prediction B: Cross-task pairs show distinct geometric signatures.**
  *Supported (p < 10⁻⁶).* ANOVA F = 19.82 across 6 cross-task pair types.
  Different task combinations produce statistically distinguishable
  compatibility profiles, confirming that the merge-audit pipeline reads
  genuine task-geometric structure, not noise.

- **Prediction C: Spectral risk level predicts merge quality degradation.**
  *Supported — acting on the diagnostic improves outcomes.* Under naive
  0.5/0.5 linear merge (Phase 3), all risk levels show similar
  degradation (low: −0.371, medium: −0.403, high: −0.396; Spearman
  r = −0.009, p = 0.95). This is the expected baseline: the risk
  classification is designed to select *strategies*, not predict
  *magnitude* under a fixed strategy.

  Phase 3b tested strategy-aware merging — selecting norm-equalized merge
  for norm_imbalance pairs and TIES for redundancy pairs. Results:

  | Strategy | n | Mean improvement over baseline | Positive |
  |----------|---|-------------------------------|----------|
  | TIES (redundancy) | 16 | +0.087 | 9/16 (56%) |
  | Norm-equalized (imbalance) | 32 | +0.024 | 13/32 (41%) |
  | Linear (no issue) | 4 | −0.072 | 2/4 |

  TIES showed the largest gains (mean +8.7 percentage points over naive
  linear), with individual improvements up to +38.8pp (qnli_s7 × sst2_s42
  on SST-2: 89.7% strategic vs 50.9% baseline). Norm equalization
  provided modest gains (+2.4pp mean). The diagnosis-to-strategy mapping
  improves outcomes in 24/52 evaluations (46%), with mean overall
  improvement of +3.6pp. The improvements are concentrated in the cases
  where the diagnosis correctly identified an actionable geometric
  problem (redundancy → TIES deduplication).

- **Prediction D: Curvature leads validation accuracy.**
  *Supported.* Across all 8 adapters, median optimal cross-correlation
  lag = 3 snapshot intervals (150 training steps). This replicates the
  §16a finding (GPT-2 small, 3–6 update lead) on a different architecture,
  model scale, and training task, using stochastic Hutchinson estimators
  rather than deterministic finite-difference probes. The lead-lag
  relationship is not an artifact of the specific Hessian estimation
  method.

- **Prediction E: Spectral partitioning — between-task rank differences
  exceed within-task variance.**
  *Supported.* Between-task stable rank differences (mean 0.225) are 46×
  larger than within-task variance (0.005). All adapters converge to very
  low effective rank: stable rank 1.2–1.6 out of nominal rank 16,
  energy_rank_90 of 1.75–3.0. This replicates §11's spectral compression
  finding and extends it to DeBERTa, confirming that extreme low-rank
  convergence is not backbone-specific.

- **Prediction F: Norm imbalance correlates with dataset size.**
  *Supported (r = 0.994, p < 10⁻⁷).* Near-perfect log-log correlation
  between training set size and total adapter Frobenius norm. QNLI
  adapters (105K samples): norm ~162; SST-2 (67K): ~132; MRPC (3.7K):
  ~31; RTE (2.5K): ~33. This establishes the mechanistic origin of
  Gradience's norm_imbalance diagnosis: adapters trained on larger
  datasets accumulate proportionally larger weight magnitudes, and the
  merge pipeline correctly identifies this as the dominant geometric
  issue in 16 of 28 pairs.

- **Prediction G: Phase transitions detectable in curvature dynamics.**
  *Supported (4/4 sufficiently-long adapters).* Rolling variance of
  Hessian energy changes shows spikes exceeding 3× median in all adapters
  with ≥10 curvature snapshots (QNLI and SST-2). Mean regime shift
  magnitude 57.8% between training thirds. This replicates the phase
  transition detection capability of §16 on DeBERTa.

### Connection to existing findings

- **§§11–14 (Spectral Partitioning, N127).** N07 replicates the core
  spectral partitioning findings on a third backbone (DeBERTa-v3 vs.
  N127's DistilBERT-base). The same-task/cross-task compatibility
  separation (Prediction A, 2.8× ratio) is quantitatively consistent
  with N127 Extension 2's 3.1× ratio (same-task 7.8× vs. cross-task
  2.5× in H/L alignment). The extreme low-rank convergence (stable rank
  1.2–1.6 at r=16) matches N127's observation of concentrated spectral
  energy. The partitioning generalizes across pretraining objectives.

- **§16a (Curvature Telemetry).** N07 is the first cross-architecture
  replication of the curvature lead-lag finding, extending it from GPT-2
  small (124M, deterministic Hessian) to DeBERTa-v3-base (185M,
  stochastic Hutchinson). The median lag of 3 intervals (150 steps at
  50-step cadence) is consistent with §16a's 3–6 update lead (at 5–6
  step cadence). The relationship is robust to estimator choice.

- **§16 (Hessian Telemetry).** The phase transition detection
  (Prediction G) extends §16's changepoint detection to DeBERTa, using
  a different detection method (rolling variance vs. CUSUM). Both
  identify regime shifts in Hessian dynamics during training.

- **§9 (Study 16, Structural-Behavioral Separation).** Prediction C's
  null result under naive linear merge is consistent with §9's finding
  that structural compatibility is necessary but not sufficient. The
  risk classification identifies the *type* of geometric problem (norm
  imbalance, redundancy) but the practitioner must act on it by selecting
  the appropriate merge strategy.

- **Technical Report §7.1.** N07 resolves the "GPU-blocked" status of the
  DeBERTa adjudication. The sixth prediction (spectral partitioning
  remains task-discriminating on DeBERTa) is confirmed: Prediction A
  shows same-task compatibility 0.449 vs. cross-task 0.160 despite
  DeBERTa's distinct pretraining objective. The triage pipeline's
  energy-weighted compatibility metrics generalize to replaced token
  detection pretraining.

### Limitations

- DeBERTa-v3-base is a 185M-parameter encoder. The results extend the
  validated backbone count from 2 to 3 but remain in the encoder regime.
  Decoder-only validation at this level of rigor is pending.
- Prediction C's strategy-aware results show improvement in 46% of
  evaluations but also some regressions (notably TIES on same-task pairs
  where deduplication removes useful shared directions). The
  diagnosis-to-strategy mapping is a heuristic, not guaranteed optimal.
- Curvature estimation uses stochastic Hutchinson (30 random vectors)
  and power iteration (3 eigenvalues). The lead-lag result is robust but
  the curvature estimates are noisier than §16a's deterministic approach.
- The N07 predictions (A–G) were formulated after preliminary results
  from N127 and §16a. While they test genuinely new claims (cross-backbone
  generalization, stochastic Hessian estimation, DeBERTa-specific geometry),
  they are not fully blind predictions in the pre-registration sense.
- Merge evaluation uses the task-specific classifier head from one adapter
  in each pair. Cross-task merged accuracy depends on how the averaged
  LoRA backbone interacts with the unadapted classifier head.
- RTE (n=277 validation) and MRPC (n=408 validation) have small eval sets;
  accuracy estimates on merged models have wider confidence intervals for
  these tasks.

### Reproducibility

Scripts: `scripts/n07_deberta/train_deberta.py` (Phase 1),
`run_phase2.py` (Phase 2), `run_phase3.py` (Phase 3),
`run_phase4.py` (Phase 4). Results archived in
`scripts/n07_deberta/phase4_analysis.json`. RunPod A100-SXM4-80GB.
DeBERTa-v3-base from `microsoft/deberta-v3-base` via HuggingFace.
PEFT LoRA with target_modules `["query_proj", "key_proj", "value_proj",
"attention.output.dense"]`. Seeds 42 and 7. GLUE datasets via
HuggingFace `datasets`. Full adapter weights and telemetry at
`/workspace/n07/` on the RunPod instance.


## 24. Per-Module Decomposition and Curvature-Partition Correspondence (N07 Experiments A/B, April 2026)

### Claim

Two follow-up experiments on the N07 DeBERTa-v3 adapters test deeper
structural predictions from Tech Report §7.1 and THEORY.md §7.2.
Experiment A decomposes merge compatibility per attention module (Q/K/V/O)
and per head, testing whether V-module geometry is uniquely predictive of
merge outcome. Experiment B trains with dense dual-instrument telemetry
(structural SVD every 10 steps, Hessian curvature every 50 steps) to test
whether curvature collapse events temporally precede spectral alignment
sharpening.

### Evidence

**Experiment A: Per-Module Decomposition** (153s, 28 pairs, 8 adapters)

Tested four predictions from Tech Report §7.1:

| Prediction | Result | Statistic | Interpretation |
|---|---|---|---|
| P2: V-module DR separates catastrophic/safe | **Not supported** | Cohen's d = 0.02, p = 0.96 | No module separates outcome classes |
| P3: Instability transfers to DeBERTa | **Not supported** (marginal) | p = 0.146, direction correct | Cross-task σ = 0.112 > same-task σ = 0.067 |
| P4: Readout attractor structure | **Feature-set switching** all 4 tasks | Mean row cosine ≈ 0.0, PC overlap < 0.05 | Cross-seed classifier weights are near-orthogonal |
| P5: Head-level modulation | **Not supported** | 4/10 pairs confirmed | Threshold-dependent, partial signal |

Key per-module spectral profiles (stable rank averaged across 12 layers):

| Module | Small-dataset (MRPC/RTE) | Large-dataset (QNLI/SST-2) |
|---|---|---|
| Q | 1.55–2.24 | 1.18–1.53 |
| K | 1.81–2.11 | 1.22–1.53 |
| V | 1.16–1.43 | 1.15–1.55 |
| O | 1.09–1.24 | 1.14–1.37 |

The V-module consistently has the **lowest stable rank** among Q/K/V/O
(most concentrated spectral energy), but this concentration is uniform
across catastrophic and safe merge pairs — it does not predict merge
outcome. The dimensionality ratio (DR) distributions for catastrophic
(mean 0.622) and safe (mean 0.624) pairs overlap completely.

**P4 finding (readout attractors)** is a strong informative null:
DeBERTa-v3 cross-seed classifier weights are near-orthogonal for all four
tasks (mean cosine similarity < 0.03, PC overlap < 0.05), classifying
unambiguously as **feature-set switching** rather than rotational degeneracy.
This means different random seeds discover essentially unrelated readout
directions that achieve comparable accuracy, suggesting a high-dimensional
feature manifold with many viable readout projections.

**Experiment B: Curvature-Partition Correspondence** (205s, MRPC task, seeds 42 & 123)

Trained with dense dual-instrument telemetry: structural SVD snapshots
every 10 training steps and Hessian top eigenvalue estimation every 50 steps,
using QR-accelerated low-rank SVD and LoRA-restricted Hessian-vector products.

| Prediction | Result | Statistic | Interpretation |
|---|---|---|---|
| CP-1: Curvature precedes alignment | **Supported** | 41/96 modules show negative lag | Median lag -2 to -4 steps where present |
| CP-2: MP boundary tracks transitions | **Supported** | 73/96 significant (p < 0.05) | Strong Spearman correlations (up to r = -0.89) |
| CP-3: Negative lag dominates globally | **Not supported** | 43% negative, mean lag +0.51 | Near-chance distribution |
| CP-4: Cross-seed replication | **Not supported** | Jaccard = 0.08 | Module-level timing is seed-dependent |
| CP-5: V-module strongest | **Not supported** | V = 0.508 vs K = 0.519 | All module types show similar correlation magnitude |

CP-1 and CP-2 provide partial evidence for the curvature-partition
correspondence hypothesis: curvature changes do correlate with spectral
structure changes, and the MP boundary meaningfully tracks alignment
evolution. However, the temporal ordering is **not consistently
curvature-first** across all modules (CP-3), the effect is
**seed-dependent** at the module level (CP-4), and shows **no V-module
specificity** (CP-5).

The most striking result is CP-2: 76% of (module, seed) combinations show
a statistically significant Spearman correlation between the count of
singular values above the Marchenko-Pastur threshold and the module's
stable rank evolution during training. This confirms that the MP boundary
is an empirically useful spectral partition criterion, not just a
theoretical convenience.

### Connection to existing findings

- **§11–14 (spectral partitioning)**: Experiment A confirms V-modules have
  the lowest stable rank across all DeBERTa adapters but — counter to §7.1
  predictions — this does not make them uniquely predictive of merge outcome.
  The per-module DR is uniformly non-discriminative, suggesting merge
  degradation is a collective phenomenon not localized to specific modules.
- **§16a (curvature telemetry)**: Experiment B extends §16a's curvature-leads-
  accuracy finding to the curvature-partition correspondence: curvature
  changes do correlate with alignment changes (CP-2) and sometimes precede
  them (CP-1), but the temporal relationship is less clean than the
  curvature-accuracy lead-lag reported in §16a.
- **§22 (portfolio collapse)**: P4's feature-set switching result — cross-seed
  classifiers are orthogonal — is consistent with §22's observation that
  LoRA portfolios occupy distinct regions of weight space that can collapse
  during merging.

### Limitations

- Experiment A's P2 null result may reflect DeBERTa's specific architecture
  (disentangled attention) rather than a general failure of the V-module
  hypothesis. Testing on DistilBERT/RoBERTa is needed.
- Experiment B used only MRPC (n=3,668 train). Larger datasets with more
  training steps would provide more curvature snapshots for cross-correlation.
- Curvature estimation used 5 power iterations and 3 Hutchinson samples
  (reduced from original spec for computational tractability). Higher-fidelity
  estimates may reveal cleaner temporal structure.
- CP-4's low Jaccard (0.08) suggests the curvature-alignment coupling is
  fundamentally stochastic at the individual-module level, even if
  statistically present in aggregate.

### Reproducibility

Scripts: `scripts/n07_deberta/experiment_a_per_module.py` (Experiment A),
`experiment_b_curvature_partition.py` (Experiment B). Results archived in
`scripts/n07_deberta/experiment_a_results/` and
`scripts/n07_deberta/experiment_b_results/`. RunPod A100-SXM4-80GB.
Experiment A uses existing N07 adapters (no training). Experiment B
trains fresh adapters with dense telemetry (seeds 42, 123).


---

# Appendix: Cross-Reference Index

| Finding | Primary source | Repository | Key statistic |
|---------|---------------|------------|---------------|
| §1  Mistral compression      | Bench experiments     | Gradience I | 50% compression, <2.5% degradation |
| §2  Study 14 (n=29)          | Post 2, Study 14      | Gradience II | Mean util 0.166, 8 architectures |
| §3  Post 7 (n=86)            | Post 7                | Gradience II | Mean util 0.172, 22 architectures |
| §4  Cross-seed stability     | Multi-seed bench      | Gradience I | CV < 0.1 for stable rank |
| §5  Attn vs MLP              | Bench experiments     | Gradience I | Gap vanishes at n=86 |
| §6  Policy disagreement      | Audit pipeline        | Gradience I | Descriptive |
| §7  Overlap → dominance      | Post 3                | Gradience II | r = 0.846, n=27 pairs |
| §8  Principal angle merging  | Merge pipeline        | Gradience I | Threshold 0.5 empirical |
| §9  Study 16 (eligibility)   | Study 16              | Both | Frobenius ratios up to 19.7× |
| §10 Study 17 (compression)   | Study 17A/B           | Both | Null result |
| §11 MP partition             | N127                  | Gradience I | d ≈ 1.15, H/L 1.8× |
| §12 Task-dependent partition | N127 Ext 2            | Gradience I | Same 7.8× vs cross 2.5× |
| §13 W₀ energy → alignment   | N127 Ext 1            | Gradience I | r = 0.53--0.58 (QNLI) |
| §14 Partition convergence    | N127 Ext 3            | Gradience I | Plateau at step 150 |
| §15 Regime classification    | Reanalysis            | Gradience II | 67% accuracy, p=0.0001 |
| §16 Hessian telemetry        | Reanalysis            | Gradience II | Geometry leads loss by 300 steps |
| §16a Curvature telemetry     | Paper 1 (April 2026)  | Gradience II | $\sum \lambda^2$ leads accuracy by 3--6 steps, 36% RMSE improvement |
| §17 Three-act structure      | Post 5                | Gradience II | Rq mean 0.90, three acts |
| §18 DFA exponents            | Study 12              | Gradience II | F=116.86, p≈10⁻²³ |
| §19 Engine design            | Post 8                | Gradience II | Architecture document |
| §20 Workflow selection        | Studies 16--17        | Both | Product decision |
| §21 External convergence | Literature survey (April 2026) | External | See §21 citations |
| §22 Portfolio collapse    | N129 (April 2026)              | Gradience I | β₁ = 0.48, k_collapse 2–3 |
| §23 DeBERTa adjudication | N07 (April 2026)               | Gradience I | 7/7 predictions confirmed, r=0.994 size→norm |
| §24 Per-module / curvature | N07 Exp A/B (April 2026)      | Gradience I | V-module DR d=0.02, MP boundary 73/96 significant |
| §25 Spectral susceptibility | N130 (April 2026)             | Gradience I | Γ_k ≤ C_k as predictor; erank QNLI 13.3 vs SST-2 5.5, d=2.05 |
| §26 Composite predictor     | N131 (April 2026)             | Gradience I | C_k + erank ADDITIVE (ΔR²=0.11, p=0.016); interaction NS |
| §27 DeBERTa erank replication | N132 (April 2026)           | Gradience I | Task erank varies (F=13.4, p<10⁻⁸); C_k→alignment does NOT replicate |


## 25. Spectral Susceptibility (Γ_k) Validation (N130, April 2026)

### Claim

Spectral susceptibility Γ_k — a gap-sensitive measure incorporating
nearest-neighbor singular value spacing — does **not** outperform energy
concentration C_k as a predictor of per-layer alignment. The result is a
**negative finding** that strengthens the case for concentration-based
metrics over gap-based alternatives.

### Background

The N127 extension results (§13) established that W₀ energy concentration
C_k correlates with per-layer alignment for QNLI (r = 0.53–0.58, p < 0.01)
but not SST-2 (r ≈ 0.01). The naive Davis-Kahan gap (σ₁−σ₂) had already
failed (r = 0.038, p = 0.86). This experiment tested whether a more
sophisticated gap-based measure — spectral susceptibility
Γ_k = (1/E_k) Σᵢ σᵢ²/δᵢ² where δᵢ is the nearest-neighbor SV gap — could
outperform C_k by capturing sensitivity to perturbation rather than just
energy distribution.

### Evidence

**Prediction 1: Static correlation (1/Γ_k vs C_k as alignment predictors).**

| Task  | ρ(C_k, align) | p(C_k)  | ρ(1/Γ_k, align) | p(1/Γ_k) | |ρ_Γ| − |ρ_C| | Decision    |
|-------|---------------|---------|-----------------|----------|--------------|-------------|
| SST-2 | −0.076        | 0.725   | −0.088          | 0.683    | +0.012       | EQUIVALENT  |
| QNLI  | +0.563        | 0.004   | −0.071          | 0.741    | −0.491       | EQUIVALENT* |

*Bootstrap 95% CI for the difference: SST-2 [−0.370, +0.386], QNLI [−0.708, +0.112].
Neither CI excludes zero, so the decision is EQUIVALENT by the pre-registered criterion
(|diff| ≥ 0.10 and CI excludes zero required for CONFIRMED/DISCONFIRMED). However, the
QNLI result is striking: C_k achieves r = 0.56 (p = 0.004) while 1/Γ_k is near zero.
The gap-based measure is not just equivalent — it is substantively worse for the one task
where a predictor exists.*

Per-module breakdown (QNLI): V-modules show the largest C_k advantage
(ρ_C = +0.71, ρ_Γ = −0.14, diff = −0.57), consistent with V-modules
being the primary locus of task-specific alignment (§24).

**Prediction 2: Dynamic trajectory (ε_t · √Γ_k(t) decreases during training).**

| Metric              | Value     | Decision     |
|---------------------|-----------|--------------|
| Layers analyzed     | 72        |              |
| Fraction decreasing | 0/72 (0%) | DISCONFIRMED |
| Mean Spearman ρ     | +1.000    |              |
| ε increasing        | 72/72     |              |
| √Γ decreasing       | 35/72     |              |

The product ε_t · √Γ_k(t) increases monotonically in all 72 (seed × layer)
trajectories because Frobenius norm growth completely dominates the √Γ_k
decrease. While √Γ_k does decrease in about half of layers (the adapted
weight matrix becomes more spectrally stable), the perturbation magnitude
grows much faster.

**Prediction 3: Task asymmetry (QNLI effective rank > SST-2).**

| Metric        | SST-2 | QNLI  | Statistic          |
|---------------|-------|-------|--------------------|
| Mean erank    | 5.47  | 13.32 | t = 10.06          |
|               |       |       | p < 10⁻⁹           |
|               |       |       | Cohen's d = 2.05   |

All 24 layers show QNLI > SST-2, consistent with QNLI adapters requiring
higher-dimensional representations for the more complex task. This confirms
an independent prediction but is orthogonal to the Γ_k question — effective
rank is a property of Δ_W, not of W₀'s spectral gaps.

### Connection to existing findings

- **§13 (W₀ energy → alignment)**: N130 confirms that C_k remains the best
  available static predictor. The gap-based alternative (Γ_k) adds no
  information.
- **§11–14 (MP partition)**: The Gavish-Donoho threshold used for k
  selection in Γ_k computation is the same MP partition from N127, ensuring
  methodological consistency.
- **§24 (curvature-partition)**: The dynamic result (P2 DISCONFIRMED)
  parallels Experiment B's finding that curvature and structure are coupled
  but not in a simple lead-lag relationship.
- **THEORY.md §7.2**: Strengthens the argument that a formal convergence
  bound should be stated in concentration-weighted (C_k) terms, not
  gap-based (Davis-Kahan or Γ_k) terms. The gap approach is now
  doubly excluded: simple gaps (§13) and regularized susceptibility
  (§25) both fail.

### Limitations

- Single backbone (DistilBERT-base). Γ_k may behave differently for
  larger models with smoother spectra (fewer near-degenerate gaps).
- Two tasks only (SST-2, QNLI). The gap-sensitivity may matter for
  tasks with intermediate spectral structure.
- The Gavish-Donoho threshold assumes MP-distributed noise SVs, which
  is only approximately satisfied for pre-trained weight matrices.
- Γ_k regularization (ε_reg = 10⁻⁶ · σ₁) prevented numerical overflow
  but 0 gaps were actually regularized, so this was not a factor.

### Reproducibility

Script: `scripts/n130_gamma_k_validation.py`. CPU-only, ~24 seconds.
Input data: N127 results (`sidecar/results/mp_partition_test/extension_results.json`),
SST-2 checkpoints (`bench_runs/uniform_r16_seed{42,123,456}/`),
QNLI checkpoint (`bench_runs/qnli_test/probe_r32/checkpoint-50/`).
Output: `sidecar/data/n130/`.


## 26. Composite Predictor Validation: C_k × f(erank) (N131, April 2026)

### Claim

C_k and adapter effective rank (erank) contribute **additively** — not
interactively — to predicting per-layer alignment. Adding erank to C_k
improves prediction (ΔR² = +0.11, F p = 0.016), but the interaction
C_k × erank is not significant (p = 0.33). The composite multiplicative
predictor C_k × f(erank) does **not** outperform C_k alone — in fact,
all composite products perform worse than C_k by itself.

### Background

N130 (§25) established two facts: (1) C_k predicts alignment for QNLI
(ρ = 0.56, p = 0.004) but not SST-2 (ρ ≈ −0.08); (2) effective rank
differs massively between tasks (d = 2.05). The hypothesis was that C_k's
predictive power is *moderated* by erank: the multiplicative composite
C_k × erank would unify the two tasks into a single predictive relationship.

### Evidence

**Prediction 1: Hierarchical regression (ADDITIVE, not CONFIRMED).**

| Model                      | R²     | ΔR² vs M1 | Key p-value       |
|----------------------------|--------|-----------|-------------------|
| M1: C_k                   | 0.070  | —         |                   |
| M2: C_k + erank           | 0.184  | +0.114    | F p = 0.016       |
| M3: C_k + erank + C_k×er  | 0.201  | +0.131    | interaction p = 0.33 |

The interaction term is not significant (p = 0.33; with clustered SEs by
transformer layer: p = 0.30). Erank adds significant explanatory power
as a main effect (M2 vs M1: F = 6.27, p = 0.016), but the moderation
hypothesis is rejected. The formal bound factors as
alignment ≤ h₁(C_k) + h₂(erank), not alignment ≤ g(C_k × erank).

Crucially, the composite *products* perform worse than C_k alone:

| Predictor            | Spearman ρ | p     | OLS R² | AIC    |
|----------------------|-----------|-------|--------|--------|
| C_k alone            | +0.258    | 0.077 | 0.070  | −27.6  |
| C_k × erank          | +0.043    | 0.773 | 0.001  | −24.1  |
| C_k × log(erank)     | +0.080    | 0.587 | 0.003  | −24.3  |
| C_k × I(erank > 9.4) | −0.085    | 0.568 | 0.003  | −24.2  |

All composites have near-zero correlation with alignment (|ρ| < 0.09) and
higher AIC than C_k alone (ΔAIC = +3.3). The multiplication destroys
C_k's signal rather than amplifying it.

**Prediction 2: Within-task replication (CONFIRMED).**

| Task  | ρ(C_k, alignment) | p     |
|-------|-------------------|-------|
| SST-2 | −0.076            | 0.725 |
| QNLI  | +0.563            | 0.004 |

Exact replication of N130 values. C_k predicts alignment within QNLI
but not within SST-2. The task asymmetry is robust.

**Prediction 3: Functional form (UNDERDETERMINED).**

All three composite forms (linear, log, threshold) fit equally poorly
(ΔAIC range = 0.1 across forms). The functional form cannot be
discriminated because the composite predictor itself is invalid.

### Interpretation

The additive result has a clear interpretation. The task asymmetry
observed in N130 does not operate through a layer-level interaction
between C_k and erank. Instead:

- **C_k** captures a within-QNLI effect: layers with higher W₀ spectral
  concentration produce higher alignment, but only for a task (QNLI) that
  is complex enough to engage the full spectral structure.
- **Erank** captures a between-task effect: QNLI adapters use more
  dimensions than SST-2 adapters. This shifts alignment levels (QNLI
  mean = 0.448, SST-2 mean = 0.634) but does not change C_k's slope.

The fact that SST-2 has *higher* mean alignment despite *lower* erank is
notable: simpler tasks concentrate their adaptation into fewer directions
that align well across seeds, while complex tasks spread adaptation
across more directions with lower per-direction consistency.

### Connection to existing findings

- **§25 (Γ_k validation)**: N131 completes the N130 follow-up program.
  The composite predictor that N130 motivated does not work as a product,
  but C_k and erank are independently informative.
- **§13 (W₀ energy → alignment)**: The within-task replication confirms
  C_k's QNLI-specific predictive power. The bound should be stated in
  terms of C_k (concentration) with erank as an independent moderator
  at the task level, not the layer level.
- **THEORY.md §7.2**: The formal convergence bound should factor as
  alignment ≤ h₁(C_k) + h₂(erank), where h₁ is the concentration term
  (significant for complex tasks) and h₂ is the task-dimensionality
  term (shifts the baseline). This is simpler than the originally
  hypothesized joint function.

### Limitations

- n = 48 (24 layers × 2 tasks). The interaction test has limited power
  (post-hoc power ≈ 0.15 for the observed effect size). A true interaction
  might exist but be undetectable at this sample size.
- Two tasks only. The additive vs interactive distinction requires more
  tasks spanning a wider erank range to resolve definitively.
- Layers within a model are not independent. Clustered SEs by transformer
  layer (6 clusters) do not change the qualitative result.
- The direction of the alignment–erank relationship (higher erank →
  *lower* alignment) was unexpected and should be confirmed on other
  backbones.

### Reproducibility

Script: `scripts/n131_composite_predictor.py`. CPU-only, <1 second.
Input data: N130 outputs (`sidecar/data/n130/`), N127 results
(`sidecar/results/mp_partition_test/extension_results.json`).
Output: `sidecar/data/n131/` (CSV, summary JSON, 4 diagnostic figures).


## 27. DeBERTa Effective Rank Replication (N132, April 2026)

### Claim

Task dimensionality (effective rank) varies systematically across GLUE
tasks on DeBERTa-v3-base, replicating the N130 finding. However, the
C_k → alignment predictive relationship does **not** replicate on
DeBERTa: no task shows significant C_k–alignment correlation. This is a
**partial replication** with a significant negative result.

### Background

N130 (§25) found that effective rank differs massively between tasks on
DistilBERT (QNLI = 13.3, SST-2 = 5.5, d = 2.05) and that C_k predicts
alignment for QNLI (ρ = 0.56, p = 0.004) but not SST-2. N132 tests
whether these findings generalize to a second architecture (DeBERTa-v3-base,
12 layers, disentangled attention) using the existing N07 adapter data
(4 GLUE tasks × 2 seeds, rank 16).

### Evidence

**A-P1: Task dimensionality varies across GLUE tasks (CONFIRMED).**

| Task  | Mean erank | SD    |
|-------|-----------|-------|
| SST-2 | 2.03      | 1.01  |
| QNLI  | 2.49      | 1.18  |
| MRPC  | 3.14      | 2.69  |
| RTE   | 3.95      | 3.22  |

Per-layer ANOVA: F = 13.42, p = 2.4 × 10⁻⁸. Maximum pairwise
Cohen's d = 11.67 (RTE vs SST-2). The task ordering SST2 < QNLI <
MRPC < RTE is plausible: binary sentiment (SST-2) requires fewest
dimensions; textual entailment with small training data (RTE) requires
the most adaptation flexibility.

Note: DeBERTa eranks (2.0–4.0) are much lower than DistilBERT eranks
(5.5–13.3) for the same tasks at the same nominal rank (16). This likely
reflects DeBERTa's disentangled attention and replaced token detection
pretraining, which produce a more efficient adaptation geometry.

**A-P2: C_k predicts alignment for high-erank tasks (DISCONFIRMED).**

| Task  | ρ(C_k, overlap) | p     |
|-------|----------------|-------|
| SST-2 | +0.126          | 0.393 |
| QNLI  | +0.106          | 0.474 |
| MRPC  | +0.216          | 0.140 |
| RTE   | +0.191          | 0.195 |

No task achieves significance. The highest-erank task (RTE, ρ = 0.19)
does not reach the ρ ≥ 0.30 threshold. The C_k → alignment relationship
that holds for DistilBERT/QNLI does not generalize to DeBERTa.

The hierarchical regression (N131 replication) confirms the negative
result: M1 (C_k) R² = 0.015, M2 (C_k + erank) R² = 0.148,
M3 (interaction) R² = 0.148. Erank adds power (as a between-task
mean shift), but C_k contributes essentially nothing (R² = 0.015 vs
0.070 on DistilBERT), and the interaction is completely absent
(p = 0.90).

**A-P3: Cross-architecture erank ordering preserved (CONFIRMED).**

| Task  | DistilBERT erank | DeBERTa erank |
|-------|-----------------|---------------|
| SST-2 | 5.47             | 2.03          |
| QNLI  | 13.32            | 2.49          |

The rank ordering SST-2 < QNLI is preserved across architectures.
However, the magnitude ratio is compressed: DistilBERT QNLI/SST-2 =
2.43× vs DeBERTa QNLI/SST-2 = 1.23×. DeBERTa adapts more
efficiently, compressing all tasks into a narrower erank range.

### Interpretation

The C_k → alignment finding (§13, §25, §26) is **architecture-specific**,
not universal. It holds for DistilBERT-base (6-layer, standard attention,
MLM pretraining) but not for DeBERTa-v3-base (12-layer, disentangled
attention, RTD pretraining). Two candidate explanations:

1. **Spectral geometry differs.** DeBERTa's disentangled attention
   separates content and position projections, producing a different
   W₀ spectral structure. The C_k values have a different relationship
   to the adapter subspace because the pre-trained geometry is organized
   differently.

2. **Adapter eranks are too compressed.** On DeBERTa, all tasks have
   erank in [2.0, 4.0] — much narrower than DistilBERT's [5.5, 13.3].
   With less variation in task dimensionality, there may be insufficient
   dynamic range for C_k to differentiate. The moderation effect exists
   in principle but requires a wider erank spread to manifest.

This result has important implications for the convergence bound program:
a bound stated purely in terms of C_k will not generalize across
architectures. Either C_k needs to be redefined in an
architecture-invariant way, or the bound must account for
backbone-specific spectral geometry.

### Connection to existing findings

- **§25 (Γ_k validation)**: N132 replicates the erank asymmetry (A-P1,
  A-P3) but not the C_k predictive relationship (A-P2). The task-ordering
  finding is architecture-general; the C_k finding is not.
- **§26 (composite predictor)**: The N131 additive result
  (C_k + erank factors) is vacuously confirmed on DeBERTa because
  C_k contributes nothing — the "additive" model reduces to erank alone.
- **§13 (W₀ energy → alignment)**: The ρ = 0.53–0.58 correlation is
  bounded to DistilBERT. DeBERTa shows ρ = 0.11–0.22 (all NS).
- **§23 (DeBERTa adjudication)**: N07's 7/7 prediction successes were
  about merge diagnostics and risk classification, not about C_k–alignment
  prediction. N132 does not contradict N07's results.
- **THEORY.md §7.2**: The concentration-weighted convergence bound must
  be qualified: C_k may be the right variable for the DistilBERT regime
  but not universally. The bound may need to condition on backbone
  spectral structure.

### Limitations

- Only 2 seeds per task → 1 same-task pair per task, limiting
  within-task alignment estimation.
- The "overlap" metric from N07 differs from the SV-weighted alignment
  used in N127/N130. The metrics capture similar structure but are not
  directly comparable.
- DeBERTa adapters have low absolute erank (2–4), reducing the variance
  available for detecting C_k moderation.
- 4 tasks on one architecture. More tasks and architectures needed to
  determine whether the C_k finding is specific to DistilBERT or to
  standard-attention architectures more broadly.

### Reproducibility

Script: `scripts/n132_deberta_erank.py`. CPU-only, ~5 seconds.
Input data: N07 experiment_a_results (`scripts/n07_deberta/experiment_a_results/`),
DeBERTa-v3-base pretrained weights (HuggingFace).
Output: `sidecar/data/n132/` (CSV files, summary JSON, 2 diagnostic figures).

## 28. Decoder-Scale Controlled Merge Triage (N133, April 2026)

> **Status:** complete. All six behavioral predictions (B-P1 … B-P6)
> and the follow-up N133b per-module re-analysis of DistilBERT are
> locked. The B-P5 "partial confirmation" in the raw Phase 4 summary
> is revised to an explicit null after the B-P5 diagnostic showed the
> apparent triage recall was a task-family and metric-range confound
> (see `sidecar/notes/n133_bp5_diagnostic.md`). The N134 follow-up
> spec (`sidecar/notes/n134_spec.md`) is the confirmatory design
> that addresses the confounds catalogued here.

### Claim

At decoder scale (Mistral-7B-v0.3, 32-layer, standard attention),
spectral alignment cleanly separates same-task from cross-task adapter
pairs (B-P1, B-P2 strongly confirmed), and adapter effective rank varies
systematically by task (B-P4, first half). Two encoder-era findings —
C_k predicts alignment (§13, §25, §26) and no module-type asymmetry —
**do not replicate** at decoder scale: the within-module C_k → alignment
correlation is null (B-P3 disconfirmed after Simpson's-paradox
correction), and there is a significant Q/K vs V/O asymmetry in both
erank and same-task alignment (B-P6 disconfirmed). The pre-specified
alignment-based triage (B-P5) **does not work** at decoder scale — not
because the geometric signal is absent, but because the original 3/3
recall in the raw Phase 4 output was driven by the interaction of a
task-family confound (MNLI-heavy pairs vs generation-heavy pairs) with
a metric-range confound (four of six source baselines saturated at
ceilings or floors). N133 is the first decoder-scale test of the
spectral triage pipeline, the first decoder-scale replication of
N132's negative C_k result, and the first experiment to flag per-module
stratification as a blocker for cross-architecture C_k claims.

### Background

Prior N127–N132 work established the spectral audit pipeline on encoder
models: DistilBERT-base (6 layers, standard attention, MLM pretraining)
and DeBERTa-v3-base (12 layers, disentangled attention, RTD pretraining).
The pipeline measures per-layer effective rank (erank), SV-weighted
alignment between adapter pairs, and pretrained-weight energy
concentration (C_k). N131 (§26) proposed a composite predictor
C_k × f(erank); N132 (§27) showed that the C_k component fails to
replicate on DeBERTa, leaving erank as the dominant signal.

N133 tests whether the same pipeline transfers to a 7B decoder — the
scale at which LoRA is most commonly deployed — and whether the N132
negative result on C_k replicates under architecture × scale change.

**Experimental design.** 6 tasks × 2 seeds = 12 LoRA adapters on
Mistral-7B-v0.3 with r = 16 targeting `q_proj`, `k_proj`, `v_proj`,
`o_proj`. Tasks span discriminative (SST-2, MNLI), extractive (SQuAD),
generative (GSM8K, code via CodeAlpaca-20k, summarization). All 12
adapters achieved the expected source accuracy (SST-2 0.98–0.99,
MNLI 0.97, SQuAD 1.00, GSM8K 0.23–0.32, code 1.00, summarization 1.00).

Phase 2 computes per-adapter spectral profiles, W₀ properties for the
128 attention projections (32 layers × 4 modules), and SV-weighted
alignment for all 66 adapter pairs. Phase 3 evaluates 18 priority merges
(6 same-task + top-12 cross-task by alignment). Phase 4 tests the six
behavioral predictions B-P1 … B-P6.

### Evidence

**B-P1: Task-boundary detection — zero false positives (CONFIRMED).**

| Pair class | n  | Mean alignment |
|------------|----|----------------|
| Same-task  |  6 | 0.1249         |
| Cross-task | 60 | 0.0409         |

Midpoint threshold τ = 0.0829. 0 / 6 same-task pairs fall below τ;
0 / 60 cross-task pairs rise above τ. The alignment signal perfectly
discriminates the task boundary on Mistral-7B at this experimental
budget. This is the third architecture (after DistilBERT and DeBERTa)
on which the pipeline shows clean task-boundary separation.

**B-P2: Spectral separation ≥ 2.0× (CONFIRMED).**

Same / cross alignment ratio = **3.06×**, t = 35.78, p < 10⁻⁶.
The separation is smaller than DistilBERT's ~5× (N130) but larger
than the 2.0× prediction threshold, and the per-pair standard
deviation is low enough that the two distributions do not overlap.

**B-P3: C_k predicts alignment (DISCONFIRMED — null, not reversed).**

The naive pooled correlation (ignoring module identity) is ρ = −0.216,
p < 10⁻⁸, n = 768 — i.e. significantly *negative*. Per-task pooled
correlations all point in the same negative direction, all but
summarization significantly so:

| Task          | ρ(C_k, alignment) | p      | n   |
|---------------|-------------------|--------|-----|
| SST-2         | −0.357            | <0.001 | 128 |
| MNLI          | −0.304            |  0.001 | 128 |
| SQuAD         | −0.260            |  0.003 | 128 |
| GSM8K         | −0.397            | <0.001 | 128 |
| Code          | −0.277            |  0.002 | 128 |
| Summarization | −0.022            |  0.807 | 128 |
| **Pooled**    | **−0.216**        | **<0.001** | **768** |

**However, the naive pooled signal is a Simpson's paradox artifact.**
Mistral's attention modules have radically different mean C_k and
different mean alignment from each other:

| Module | mean W₀ C_k | mean same-task align | within-module ρ(C_k,align) | p |
|--------|-------------|----------------------|---------------------------|---|
| Q      | 0.564       | 0.1085               | −0.173                    | 0.016 |
| K      | 0.547       | 0.1325               | −0.172                    | 0.017 |
| V      | **0.085**   | **0.1450**           | **+0.143**                | 0.047 |
| O      | 0.316       | 0.1135               | −0.062                    | 0.394 |

V-projection has radically lower W₀ C_k than Q/K/O (0.085 vs 0.32–0.56)
*and* the highest same-task alignment (0.145). When C_k and alignment
are pooled across modules, the negative between-module trend
(low-C_k V-modules tend to be high-alignment; high-C_k Q-modules
tend to be lower-alignment) dominates the pooled Spearman coefficient.
Within each module, the story is completely different.

Controlling for module identity via partial correlation:

- Within-module residualized Spearman ρ = −0.066 (p = 0.069, NS)
- Within-module residualized Pearson r = +0.106 (p = 0.003, right sign)

**The corrected within-module finding is essentially null**, not
reversed. The naive "wrong sign" reading is a pooling artifact.

The finding is a clean replication-failure of the §13 / §25 / §26
composite-predictor program's C_k component on Mistral, not an
unprincipled sign flip. B-P3 is disconfirmed by null effect under
proper module control.

**Retroactive re-analysis of DistilBERT §13 (N133b).** Motivated by
the Mistral Simpson's paradox, the original N130 DistilBERT data was
re-analyzed with per-module stratification to check whether its
QNLI pooled ρ = +0.56 was also a pooling artifact. The QNLI finding
**survives**: per-module correlations are consistently positive
(q +0.43, k +0.60, v +0.71, o +0.77 — all n = 6, all individually
underpowered but directionally unanimous), and the within-module
residualized Spearman ρ = +0.546 (p = 0.006), essentially identical
to the naive pooled +0.563. DistilBERT QNLI is a *real* within-module
effect, not a Simpson's paradox. (DistilBERT SST-2 was always
null: pooled ρ = −0.076, within-module residualized ρ = +0.106 NS.)

The explanation is that DistilBERT QNLI has two reinforcing
gradients: the k-module has both the highest mean C_k (0.465) and
the highest mean alignment (0.609), so pooling amplifies the
within-module signal instead of confounding it. Mistral has the
opposite pattern — V-module has the lowest mean C_k (0.085) but
the highest mean alignment (0.145) — and the opposing gradients
produce a spurious pooled negative.

The revised cross-architecture picture is therefore:

| Cell                  | Pooled ρ | Within-module ρ | Verdict |
|-----------------------|---------|-----------------|---------|
| DistilBERT QNLI       | **+0.56** | **+0.55**      | **real, robust** |
| DistilBERT SST-2      |  −0.08  |  +0.11          | null (always) |
| DeBERTa all 4 tasks   | +0.11…+0.22 | (within-mod not computed, but pooled NS) | null |
| Mistral all 6 tasks (N133) | **−0.22 pooled** | **−0.07 / +0.11** | **null (Simpson's paradox)** |

The §13 / §25 / §26 program is narrowed, not falsified: C_k → alignment
is a real, within-module relationship on DistilBERT-base QNLI, and
essentially nothing else tested so far. It remains unusable as a
general spectral triage signal, because 10 / 11 architecture × task
cells give null within-module effects — but the original DistilBERT
QNLI finding is preserved as a genuine effect.

**B-P4: Erank varies by task and moderates C_k (PARTIAL).**

| Task          | Mean erank | SD   | n |
|---------------|-----------|------|---|
| SST-2         |  5.21     | 0.07 | 2 |
| MNLI          |  6.04     | 0.04 | 2 |
| SQuAD         |  7.55     | 0.11 | 2 |
| GSM8K         |  6.86     | 0.04 | 2 |
| Code          |  6.60     | 0.07 | 2 |
| Summarization |  8.99     | 0.09 | 2 |

Between-adapter ANOVA: F = 308.53, p < 10⁻⁶. Per-layer ANOVA
(n = 1536 layer-adapter observations): F = 105.42, p = 9.3 × 10⁻⁹⁶.
Erank varies dramatically and reliably across tasks, replicating the
N130 (DistilBERT) and N132 (DeBERTa) erank-asymmetry findings on a
third architecture. The task ordering SST-2 < MNLI < Code ≈ GSM8K <
SQuAD < Summarization is consistent with intuition: binary sentiment
at the low end, long-form generation at the high end.

Hierarchical regression (N131 replication): M1 (C_k) R² = 0.016;
M3 (C_k + erank + C_k × erank) R² = 0.609; interaction p = 0.261.
The jump from M1 to M3 is almost entirely driven by the erank main
effect — erank varies across tasks and layers and carries most of the
predictive signal, but the interaction with C_k is not significant.

The prediction is downgraded to PARTIAL because erank variation
(the first half of B-P4) is strongly supported, but the C_k × erank
interaction (the second half) is absent — consistent with the overall
breakdown of the C_k signal at decoder scale.

**B-P5: Alignment-based triage eliminates ≥ 70% of cross-task pairs
with zero good merges missed (NULL — confound cascade).**

The raw Phase 4 summary reports B-P5 at 70.0% elimination with 3/3
good merges retained, which on its face would place this prediction
at CONFIRMED. A three-stage post-hoc diagnostic
(`sidecar/notes/n133_bp5_diagnostic.md`) revealed that every layer
of this apparent success was a confound, and the pre-specified
alignment triage does not work at decoder scale once the confounds
are controlled. B-P5 is reported as **NULL**, not confirmed.

*Stage 1 — sign inversion.* The original B-P5 script retained the
*top* 30% cross-task pairs by alignment. Under the B-P1/B-P2 semantics
established in the same experiment, *high* alignment means more shared
direction and therefore *more* expected interference, not less. The
published 3/3 recall therefore came from selecting the pairs most
likely to collide, not least. Re-running with the corrected sign
(retain *lowest* alignment as safest) preserves 3/3 good merges on
the 12 evaluated cross-task pairs, so we initially promoted the
corrected-sign version to a working triage. Spearman ρ(mean_alignment,
max_degradation) on the 12 pairs was +0.655 (p = 0.021): alignment
was a real statistical predictor of merge damage with the expected
sign.

*Stage 2 — task-cluster saturation.* Within that apparent success,
four of the 12 evaluated cross-task pairs (all the code × GSM8K
combinations) tied at exactly mean_alignment = 0.0363. The pre-specified
triage had zero within-cluster resolution on these four pairs and was
splitting them 2-good / 2-bad essentially by tiebreak. Looking at
the "good" labels directly (threshold max_degradation < 0.10), the
three retained good merges were `code–summarization`, `code–squad`,
and `squad–summarization` — three pairs whose combined source
baselines are all saturated at metric ceilings (code 1.00, squad 1.00,
summarization 1.00). Degradation for these pairs is bounded below by
ceiling floor effects on the *source* metric, independently of any
merge-geometry property. The three missed good merges (`mnli–squad`,
`mnli–sst2`, `mnli–summarization`) share the opposite property:
MNLI and SST-2 have room-for-degradation baselines (0.97 and 0.98)
and were therefore the only candidates that could produce a
max_degradation < 0.10 outside the ceiling clusters. Alignment was
not separating good from bad merges; it was separating
ceiling-saturated from non-saturated task pairs.

*Stage 3 — full-60-pair ranking.* Applying the corrected-sign
alignment triage to the full 60 cross-task pairs makes the confound
explicit. Of the 18 safest (lowest-alignment) cross-task pairs, 0
involve MNLI and 12 involve GSM8K; 100% of the 20 GSM8K cross-task
pairs land in the safest-30 bucket, versus 30% of MNLI pairs.
"Lowest alignment = safest merge" at full scale is indistinguishable
from a binary classifier that prefers pairs containing a generation
task with a ceiling-saturated metric. The 60-pair triage is a
task-family classifier, not a geometric merge-risk model.

*Stage 4 — composite risk reconstruction fails.* We attempted to
salvage a decoder-scale triage by searching ten post-hoc composite
scores built from features already local: O-module mean alignment,
depth-weighted O-module alignment (linear and quadratic), O+V
depth-weighted mixes, inv_min_erank, and cross-products of O-depth
with inv_min_erank, together with a z-summed O-depth + inv_min_erank.
On the same 12 evaluated cross-task pairs the only candidate that
matched the corrected-sign mean_alignment baseline at 3/3 recall was
inv_min_erank (ρ = +0.319, p = 0.313; 3/3 good preserved). However,
a confound check on inv_min_erank dropping the two erank outliers
(summarization erank 8.99, SST-2 erank 5.21) collapsed the signal:
on the remaining compressed-erank subset {mnli, code, gsm8k, squad},
Spearman ρ(inv_min_erank, max_deg) = +0.055 (p = 0.899), and the
in-subset triage retained 0/3 good merges. inv_min_erank was also
operating as a task-family classifier; once the binary erank partition
is removed, nothing remains. Every O-only variant (with or without
depth weighting) dropped from 3/3 to 1/3 retained goods, disconfirming
the O-projection triage branch that Phase 2 per-module SNR results
had suggested. No composite of the locally available features
reproduces the B-P5 claim after confound control.

The within-task Phase 2 finding that O-projection gives 7.23×
same/cross separation is not contradicted: it is a statement about
the source-adapter geometry, not about per-pair merge outcomes.
Scoring merges with O-projection alignment alone was strictly worse
than the pooled-module baseline for the actual triage task, because
the per-pair noise in a single-module score dominates the per-pair
signal.

*Summary.* B-P5 as pre-specified does not hold at decoder scale on
this experimental budget. The geometric signal (B-P1, B-P2) is real
but is not the binding constraint on observed merge outcomes on this
task set; the binding constraints are task-family identity and
metric-range saturation. Any decoder-scale merge-triage predictor
will need a task set whose source baselines are confined to a narrow
dynamic range and whose task-family structure does not collapse to a
binary partition. The N134 spec (`sidecar/notes/n134_spec.md`) is
the confirmatory design that addresses both confounds explicitly.

**Observation B-P5.a: The mean_alignment metric has structural
resolution ≈ 2 × 10⁻³ and cannot resolve pairs inside that band.**

A standalone tied-pair analysis (`scripts/n133_tied_pairs_analysis.py`,
`sidecar/data/n133/tied_pairs_analysis.json`) quantifies the tiebreak
problem exposed by the B-P5 diagnostic. The 60 cross-task pairs are
compressed into a total alignment range of 0.01315 ([0.0363, 0.0494]).
At the four-decimal precision used in every reported table,
**38 of 60 cross-task pairs (63%) fall into at least one tie cluster
of size ≥ 2**, with the largest cluster containing 11 pairs. Zero
pairs are bit-equal in IEEE-754 float32: the smallest neighbor gap
is 4.3 × 10⁻⁷, which is 115× the float32 ULP at this magnitude, so
**the ties are structural in the metric, not a floating-point
artifact.** Per-pair layer-level SEM is median 2.15 × 10⁻³ —
substantially larger than any printable tie cluster — meaning two
pair means within ~2 SEM of each other are statistically
indistinguishable at the per-layer sample the audit actually took.

The tie problem is binding on the B-P5 conclusion: three of the
twelve evaluated cross-task pairs sit inside the single largest tie
cluster (the four code × GSM8K pairs tied at α = 0.03625, of which
all four were evaluated), and inside that tie the merge-outcome
split is 2 good / 2 bad with a max_degradation std of 0.051 on a
scale where the full-sample std is 0.213. Across the three
evaluated-containing tie clusters the mean within-tie max_deg std
is 0.124 (ratio 0.58 to the full-sample std). In other words,
even if mean_alignment were perfectly calibrated across clusters,
the metric has no within-tie resolution and cannot account for more
than roughly (1 − 0.58²) ≈ 66% of the outcome variance, and in
practice explains far less once across-cluster calibration is
imperfect. This is a property of the measurement at rank-16 on
32-layer 4-module LoRA adapters, not of the particular task set.

**Observation B-P5.b: Family-residualized baseline — task-family
identity alone explains R² = 0.97 of the evaluated merge outcomes,
leaving ΔR² < 0.02 for every candidate metric.**

A family-residualized partial-correlation analysis
(`scripts/n133_family_residualized_baseline.py`,
`sidecar/data/n133/family_residualized_baseline.json`) tested two
task-family schemes:

- `FAMILY_A` — the coarse binary partition suggested by the metric-range
  / erank confound (`gen_ceiling` = {code, summarization, squad} vs
  `disc_headroom` = {mnli, sst2, gsm8k}).
- `FAMILY_B` — a finer six-family scheme (sentiment, nli, extractive_qa,
  math_gen, code_gen, nl_gen).

Under `FAMILY_B`, a plain OLS of max_degradation on task-family-pair
dummies yields **R² = 0.9658** on the 12 evaluated cross-task pairs.
Task-family identity is a near-complete predictor of merge outcome
at this experimental budget, and every geometric metric we tested
collapses to a null increment:

| Metric         | Raw ρ  | Partial ρ (FAMILY_B) | ΔR² (FAMILY_B) |
|----------------|--------|----------------------|-----------------|
| mean_alignment | +0.655 | +0.690               | **+0.004**      |
| inv_min_erank  | +0.319 | +0.248               | **+0.001**      |
| O_mean         | −0.091 | −0.414               | **+0.006**      |
| O_depth        | −0.140 | −0.487               | **+0.009**      |
| O_quad         | −0.182 | −0.487               | **+0.009**      |
| V_depth        | +0.315 | −0.210               | **+0.001**      |
| OVmix_depth    | +0.074 | −0.658               | **+0.015**      |

The O-projection-depth metric actually **flips sign** between raw
and partial correlations (+ → −), demonstrating that its original
raw correlation was being carried by which family the pair belonged
to, not by any within-family relationship.

Under the coarser `FAMILY_A` binary scheme, family-only R² is only
0.0019 and several metrics show ΔR² ≥ 0.10 — but every one passes
with the **wrong sign** (e.g. `O_depth` ρ_partial = −0.539 when
B-P5 predicted high alignment → high degradation). The FAMILY_A
"passes" are the diagnostic signature of a score acting as a
family classifier under a family model too coarse to fully
decondition it; FAMILY_B is the stricter test and rejects the same
metrics completely.

The analysis is also the pre-registered reference implementation of
the N134 H1 decision rule: a candidate score must achieve
Spearman partial ρ ≥ 0.50 *and* ΔR² ≥ 0.10 over the family-pair
baseline on the N134 task set. Applied to N133 as a negative control,
the rule correctly rejects every metric we have (FAMILY_A passes
are rejected by the sign constraint; FAMILY_B passes do not exist).
The rule is therefore known to be well-calibrated against this
class of confound before any N134 data arrives.

**Observation B-P5.c: No aggregation of per-layer alignment beats the
family baseline — negative evidence that a faithful KnOTS / TSV
comparison on this specific N133 sample would escape the confound.**

A faithful head-to-head against KnOTS (Stoica et al. 2024) and TSV
(Gargiulo et al. 2024) cannot be run from the local artifact bundle
because both methods require U / V matrices from the adapter
`.safetensors` files, and the N133 audit JSONs persist only the
per-layer singular values plus the scalar SV-weighted alignment
number (not the underlying orthonormal subspaces from which that
alignment was computed). A faithful comparison is scheduled alongside
N134 where the U / V access is trivial.

What can be done locally is a non-faithful but informative proxy:
KnOTS and TSV are themselves aggregations of per-layer subspace-overlap
quantities, and Gradience's `mean_alignment` is one such aggregation
over the same underlying values. If *any* alternative aggregation of
the per-layer alignment scalars beats the family baseline, there is a
concrete escape route the N133 data already licenses; if none of them
do, the "try a different aggregation" escape route is closed on this
sample. `scripts/n133_knots_tsv_proxy.py` sweeps 16 aggregations:

- plain mean (baseline), max, p90, L₂-normalised mean, L∞ norm
- top-k-layer means at k = 16, 32
- deep-half and deep-quarter-layer means
- per-module (Q, K, V, O) means and per-module maxima
- QK-only and VO-only group means

Under the same FAMILY_B residualization as B-P5.b:

| Aggregation         | Raw ρ  | Partial ρ | ΔR²    | H1  |
|---------------------|--------|-----------|--------|-----|
| mean_alignment      | +0.655 | +0.690    | +0.004 | fail |
| max_alignment       | +0.389 | +0.595    | +0.005 | fail |
| p90_alignment       | +0.648 | +0.641    | +0.009 | fail |
| topk16_mean         | +0.613 | +0.431    | +0.004 | fail |
| topk32_mean         | +0.634 | +0.625    | +0.006 | fail |
| deep_half_mean      | +0.630 | +0.459    | +0.004 | fail |
| deep_quarter_mean   | +0.630 | +0.421    | +0.007 | fail |
| l2_alignment        | +0.571 | +0.533    | +0.005 | fail |
| l_inf_alignment     | +0.389 | +0.595    | +0.005 | fail |
| O_mean              | −0.091 | −0.414    | +0.006 | fail |
| O_max               | +0.074 | +0.133    | +0.000 | fail |
| V_mean              | +0.252 | +0.116    | +0.000 | fail |
| Q_mean              | +0.375 | +0.280    | +0.001 | fail |
| K_mean              | +0.644 | +0.364    | +0.004 | fail |
| QK_mean             | +0.634 | +0.613    | +0.006 | fail |
| VO_mean             | +0.214 | −0.288    | +0.007 | fail |

**Zero of the 16 aggregations pass the pre-registered N134 H1 rule.**
The maximum ΔR² observed over the family-pair baseline is +0.009
(p90_alignment), an order of magnitude below the 0.10 threshold.
This is negative evidence that a faithful KnOTS or TSV comparison
on this specific N133 sample would improve over mean_alignment
under the same metric-range + task-family confound. It is *not*
evidence about KnOTS/TSV performance in general, and does not
substitute for a faithful head-to-head on an N134-style unconfounded
task set (which the N134 spec schedules in §6 alongside H1).

**B-P6: No module-type asymmetry (DISCONFIRMED).**

The original B-P6 (attention vs MLP) is not directly testable here
because LoRA targets only q/k/v/o_proj at this experimental budget.
We report two within-attention module asymmetries instead, both of
which disconfirm "no asymmetry":

*Erank by module* (adapter side, mean over all layers and tasks):

| Module | Mean erank | n   |
|--------|-----------|-----|
| Q      |  7.90     | 384 |
| O      |  7.61     | 384 |
| K      |  6.66     | 384 |
| V      |  5.33     | 384 |

ANOVA F = 114.2, p = 10⁻⁶⁶. Q/K pooled mean erank (7.28) vs V/O
pooled mean erank (6.47): t = 6.86, p < 10⁻⁶. The 12.5% mean gap
is reliable and not a seed artifact.

*Alignment by module* (same-task pairs):

| Module | Mean alignment | Note |
|--------|----------------|------|
| V      | 0.1450 | highest same-task signal |
| K      | 0.1325 |                          |
| O      | 0.1135 |                          |
| Q      | 0.1085 |                          |

V-projection has the *lowest* erank but the *highest* same-task
alignment — the archetypal "compressed task-specific subspace"
pattern. Q-projection has the highest erank but the weakest
alignment — more directions used but less stable across seeds.
This decoupling of erank and alignment across modules is itself
a decoder-scale finding with no encoder analog.

**Bonus: Per-module triage signal-to-noise is highly heterogeneous.**

The B-P2 aggregate 3.06× same/cross ratio masks enormous per-module
variation when computed module-by-module:

| Module | same_mean | cross_mean | **ratio** | t     |
|--------|-----------|------------|-----------|-------|
| O      | 0.1135    | 0.0157     | **7.23×** | 127.0 |
| V      | 0.1450    | 0.0304     |  4.77×    |  97.3 |
| Q      | 0.1085    | 0.0417     |  2.60×    |  56.5 |
| K      | 0.1325    | 0.0756     |  **1.75×** |  41.6 |

**O-projection alone gives 7.23× separation**, nearly matching
DistilBERT's aggregate signal. **K-projection alone gives only
1.75× separation — below the B-P2 threshold of 2.0×.** A spectral
triage pipeline that used only K-projection alignment would fail
B-P2; one that used only O-projection would outperform the
aggregate baseline by more than 2×.

The per-module pattern suggests K-projection picks up cross-task-shared
structure (likely syntactic or positional features), while
O-projection isolates task-specific downstream routing. V-projection
is intermediate: compressed and task-specific, but not as clean a
triage signal as O.

**Bonus: Triage signal grows with layer depth.**

Same-task alignment rises from 0.162 at layer 0 to 0.207 at layer 31
(ρ = +0.540, p = 0.0014, n = 32), while cross-task alignment is
roughly flat (ρ = +0.134, p = 0.46). The same/cross ratio therefore
grows from 2.32× at layer 0 to 4.24× at layer 31. Deep-layer
O-projection is plausibly the strongest single triage signal
available — a natural candidate for a minimal-cost spectral
triage variant.

### Interpretation

N133 separates the N127–N132 findings into two groups:

1. **Architecture-general findings.** Task-boundary detection via
   spectral alignment (B-P1), same/cross separation ≥ 2× (B-P2), and
   per-task erank variation (first half of B-P4) replicate cleanly on
   DistilBERT (6-layer encoder), DeBERTa (12-layer encoder), and now
   Mistral-7B (32-layer decoder, 7B parameters). These appear to be
   properties of LoRA as a fitting procedure under task-conditional
   data distributions, not of any specific backbone.

2. **Architecture-specific findings.** C_k → alignment prediction
   (§13, §25, §26, first confirmed on DistilBERT) does not replicate
   on either DeBERTa (N132, NS) or Mistral (N133, null within-module
   after Simpson's-paradox correction). The absence of module-type
   asymmetry also does not replicate: Mistral shows a significant
   Q/K vs V/O erank gap and a still larger alignment asymmetry
   (O-projection gives 7.23× same/cross separation vs K-projection's
   1.75×). Any theoretical bound stated in terms of C_k alone will
   not be cross-architecture, and any module-uniform assumption
   will not hold at decoder scale.

The Mistral data also uncovers two methodological points that
should be retroactively checked against the DistilBERT results:

- **Per-module stratification matters.** Mistral's V-projection has
  mean W₀ C_k ≈ 0.085 vs 0.32–0.56 for Q/K/O, *and* V carries the
  highest same-task alignment. Pooling C_k and alignment across
  these four modules produces a spurious negative between-module
  trend that survives significance testing (ρ = −0.22, p < 10⁻⁸)
  despite the within-module correlation being essentially zero.
  The original DistilBERT §13 finding (ρ = 0.53–0.58) used a single
  module family and did not face this confound, but any
  cross-architecture claim about C_k must condition on module.

- **The triage signal is not uniform across modules.** O-projection
  alone gives 7.23× same/cross separation on Mistral — nearly
  matching DistilBERT's aggregate 5× — while K-projection gives
  only 1.75× (below the B-P2 threshold). An optimal spectral
  triage at decoder scale should weight layers/modules by their
  individual signal-to-noise ratio rather than average uniformly.

Follow-up N134 (specified in `sidecar/notes/n134_spec.md`) is the
confirmatory design for decoder-scale merge-risk prediction. It is
pre-registered against the four N133 confounds catalogued above:
(C1) source-metric dynamic range — task set restricted to a narrow
accuracy band so degradation is not bounded below by ceiling/floor
effects; (C2) task-family non-partition — no binary split on
surface task type, pilot-measured erank compression, and a minimum
number of distinct families so the alignment score cannot act as a
family classifier; (C3) within-task variance — ≥ 3 seeds per task so
same-task merge noise is estimable; (C4) no post-hoc fitting — the
primary H1 score (O-module depth-weighted alignment) is pre-specified
with a published threshold (ρ ≥ 0.50, ΔR² ≥ 0.10 over a task-family
baseline) and any deviation is reported as a null. The retroactive
N133b per-module re-analysis of DistilBERT is already folded into
B-P3 above.

### Connection to existing findings

- **§13 (W₀ energy → alignment)**: Retroactively narrowed, not
  falsified. The ρ = 0.56 QNLI effect survives per-module
  stratification (within-module residualized ρ = +0.546, p = 0.006,
  see the N133b re-analysis summarized above). The effect is
  genuine on DistilBERT-base QNLI and absent on every other
  architecture × task combination tested so far. On Mistral, an
  apparently-significant pooled negative effect turns out to be
  a Simpson's paradox artifact — evidence that any future C_k
  claims must be stratified by module.
- **§25 (Γ_k validation)**: Γ_k was validated on the same DistilBERT
  regime where C_k is predictive. The N133 result raises the
  question of whether Γ_k transfers to decoder-scale architectures;
  this was not tested here.
- **§26 (composite predictor)**: The N131 composite C_k × f(erank)
  reduces to erank alone on both DeBERTa and Mistral, because the
  C_k component contributes essentially nothing within modules.
  Erank remains the robust architecture-independent signal.
- **§27 (DeBERTa erank replication)**: N133 is a clean second
  replication of §27's null C_k result. Both decoder-era models
  (DeBERTa-v3-base, Mistral-7B-v0.3) produce within-module
  C_k → alignment correlations indistinguishable from zero. The
  consistency between the two is notable given their very different
  architectures and training setups.
- **§23 (DeBERTa adjudication)**: N07's merge-diagnostic predictions
  were about decision branches and verdict ordering, not C_k. N133
  does not contradict N07.
- **THEORY.md §7.2**: The concentration-weighted convergence bound
  must now be qualified as *DistilBERT-specific* or reformulated
  in terms of erank. The "C_k is the right variable" hypothesis
  is effectively falsified for transfer across architectures, even
  after correcting the pooling error.

### Limitations

- **2 seeds per task** → only 1 same-task pair per task, so the
  same-task alignment distribution is estimated from 6 observations.
  The strong B-P1/B-P2 result is robust to this because the
  same/cross gap is huge (3.06×), but finer discrimination
  (e.g. within-task outliers) is not possible at this budget.
- **Single 7B model.** Mistral-7B is one decoder; the sign flip on
  B-P3 could be specific to Mistral's pretraining data mixture
  rather than decoder architectures generally. A TinyLlama and /
  or Llama-3-8B replication is the natural next step.
- **Attention-only LoRA target.** B-P6's attn-vs-MLP comparison
  is not directly testable at this budget. The Q/K vs V/O finding
  is the closest available signal.
- **GSM8K source accuracy is low** (0.23–0.32) and four of the
  remaining five task baselines saturate at or near 1.00 (SST-2 0.98,
  MNLI 0.97, SQuAD 1.00, code 1.00, summarization 1.00). This
  metric-range spread is the central reason B-P5 collapses: the
  "good merge" label (max_degradation < 0.10) is bounded below by
  ceiling/floor effects rather than by merge geometry. See B-P5
  above and `sidecar/notes/n133_bp5_diagnostic.md`.
- **Task-family near-binary partition.** At six tasks, the N133 task
  set is effectively partitioned into a high-erank / generation /
  ceiling-saturated cluster (code, summarization, SQuAD) and a
  lower-erank / discriminative / headroom cluster (SST-2, MNLI,
  GSM8K). Any per-pair score that happens to track this split —
  alignment, inv_min_erank, O-module depth weighting — will be
  indistinguishable from a binary task-family classifier on the 60
  cross-task pairs. N134's task-set constraint (C2) addresses this
  directly.
- **Only 12 evaluated cross-task merges.** All post-hoc composite
  risk scores are validated on a 12-pair sample, so no claim about
  decoder-scale merge-risk prediction can be statistically licensed
  from the current dataset. N133 supports B-P1/B-P2/B-P4 (which are
  adapter-side measurements with n = 12 adapters and n = 66 pairs)
  and disconfirms B-P3/B-P5/B-P6; it does not establish a positive
  decoder-scale merge-triage signal.

### Reproducibility

Scripts: `scripts/n133_train_adapters.py`,
`scripts/n133_spectral_audit.py`, `scripts/n133_merge_eval.py`,
`scripts/n133_analysis.py`. Single H100 GPU, ~6 hours end-to-end
(4 h training, 1 h spectral audit, ~1 h merge evaluation, <1 min
analysis). Resume-friendly: every phase skips its output JSON
if present. B-P5 diagnostic scripts (CPU-only, local):
`scripts/n133_bp5_composite_risk.py` (10 composite risk variants
vs max_degradation on 12 evaluated pairs),
`scripts/n133_bp5_confound_check.py` (inv_min_erank on
compressed-erank subset + corrected-sign alignment triage on all
60 cross-task pairs),
`scripts/n133_tied_pairs_analysis.py` (tied-pair resolution
analysis underlying observation B-P5.a; output
`sidecar/data/n133/tied_pairs_analysis.json`),
`scripts/n133_family_residualized_baseline.py` (family-residualized
partial correlations underlying B-P5.b and the pre-registered
reference implementation of the N134 H1 decision rule; output
`sidecar/data/n133/family_residualized_baseline.json`),
`scripts/n133_knots_tsv_proxy.py` (16-aggregation sweep underlying
B-P5.c and the KnOTS/TSV proxy caveat; output
`sidecar/data/n133/knots_tsv_proxy.json`). Diagnostic note:
`sidecar/notes/n133_bp5_diagnostic.md`. N134 spec:
`sidecar/notes/n134_spec.md`.

Input: Mistral-7B-v0.3 (HuggingFace), 6 datasets
(SST-2, MNLI, SQuAD v1.1, GSM8K, CodeAlpaca-20k, XSum).
Output: `sidecar/data/n133/` (12 adapters, 4 audit JSONs, 18 merge
JSONs + summary, analysis summary.json, 3 diagnostic figures).
