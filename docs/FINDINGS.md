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

### Reproducibility

Extension 3 in `scripts/mp_partition_extensions.py`. Results in
`sidecar/results/mp_partition_test/extension_results.json`.


---

# Part D — Training Dynamics and Telemetry

This strand addresses what happens during training: can spectral
and geometric observables detect training regimes, transitions,
and anomalies in real time?

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
| §17 Three-act structure      | Post 5                | Gradience II | Rq mean 0.90, three acts |
| §18 DFA exponents            | Study 12              | Gradience II | F=116.86, p≈10⁻²³ |
| §19 Engine design            | Post 8                | Gradience II | Architecture document |
| §20 Workflow selection        | Studies 16--17        | Both | Product decision |
