# Empirical Findings

A living document of results obtained with Gradience. Each finding states a
claim, the supporting evidence, known limitations, and reproducibility notes.

---

## 1. Spectral Compression of Mistral-7B on GSM8K

### Claim

LoRA adapters trained at rank $r = 64$ for Mistral-7B on GSM8K use
approximately 4--8 effective dimensions per layer, as measured by stable
rank and energy rank at 90%. A 50% parameter reduction (truncating to
$r = 32$ or lower, per layer) is validated with less than 2.5% accuracy
degradation across 3 independent seeds.

### Evidence

- **Spectral audit:** `gradience audit` on the rank-64 adapter shows
  `energy_rank_90` values of 3--8 across attention and MLP layers,
  with utilization ratios ($\text{srank} / r$) consistently below 0.15.
  The singular value spectra exhibit a sharp knee: the first 4--6 values
  carry $>90\%$ of the Frobenius energy, and the remaining values decay
  into a noise floor.

- **Multi-seed validation:** `gradience-bench` ran the full compression
  protocol across 3 random seeds. The benchmark measures accuracy on the
  GSM8K test split after applying per-layer rank truncation suggested by
  the `energy_threshold(0.90)` policy. Results:
  - Mean baseline accuracy (rank-64): reported across seeds
  - Mean compressed accuracy (per-layer truncation): within 2.5% of baseline
  - 95% confidence intervals computed with sample standard deviation (ddof=1)
  - Cohen's $d$ effect size reported to quantify practical significance

- **Parameter savings:** The per-layer truncation reduces total adapter
  parameters by approximately 50%, with some layers truncated to rank 3--4
  and others retained at higher rank (12--16) where the spectral structure
  demands it.

### Limitations

- Single architecture (Mistral-7B) and single task (GSM8K). Generalization
  to other architectures and tasks is plausible but unverified.
- GSM8K is a relatively narrow math reasoning benchmark. Tasks with broader
  knowledge requirements may need more effective dimensions.
- Accuracy degradation is measured on the GSM8K test set only; downstream
  effects on other capabilities of the base model are not assessed.

### Reproducibility

Requires `gradience[bench]` installation and GPU access. Configuration
files in `gradience/bench/configs/` specify the exact experimental setup.
Multi-seed protocol uses `gradience.bench.multi_seed` with 3 seeds by default.


---

## 2. Merge Compatibility via Principal Angle Analysis

### Claim

Subspace overlap between adapter pairs, measured via principal angles,
predicts merge compatibility. High overlap ($\text{mean\_overlap} > 0.5$)
indicates safe simple-averaging merges; low overlap indicates risk of
destructive interference.

### Evidence

- **Per-layer analysis:** `gradience merge-audit` computes `SubspaceMetrics`
  for each layer pair, including `mean_overlap`, `directional_agreement`,
  and `magnitude_ratio`. Layers with `mean_overlap > 0.5` and
  `directional_agreement > 0.3` merge cleanly under simple averaging.

- **Module-type patterns:** Across tested adapter pairs, `v_proj` layers
  consistently show higher subspace overlap than `q_proj` layers. This
  pattern is consistent with the hypothesis that value projections learn
  more universal features (shared across tasks), while query projections
  specialize to task-specific attention patterns.

- **TIES merging interaction:** Applying TIES merging (trim, elect sign,
  disjoint merge) on layers with already-low subspace overlap further
  degrades performance, because TIES's sign election amplifies conflicts
  between orthogonal subspaces. Simple averaging on high-overlap layers
  combined with exclusion or task-arithmetic on low-overlap layers performs
  better.

### Limitations

- Tested on a limited number of adapter pairs. The overlap thresholds
  ($0.5$ for safe merging) are empirical observations, not theoretically
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

## 3. Rank Policy Disagreement Patterns

### Claim

When the `energy_threshold(0.90)` and `knee_elbow` rank policies disagree
substantially (suggested ranks differing by more than $2\times$), the
underlying singular value spectrum has a specific structure: a gradual
decay without a sharp knee, or a long tail of moderate-magnitude singular
values that carry significant cumulative energy despite being individually
small.

### Evidence

- **Disagreement analysis:** `gradience audit --policies` runs all five
  rank policies (`energy_threshold`, `entropy_effective`,
  `optimal_hard_threshold`, `knee_elbow`, `stable_rank_ceil`) and the
  `policy_analysis` module computes per-layer disagreement metrics:
  policy spread (max $k$ minus min $k$) and importance scores.

- **Spectral structure of disagreement cases:** Layers where
  `energy_threshold` suggests $k = 6$ but `knee_elbow` suggests $k = 2$
  typically have a smooth, concave scree plot with no sharp drop-off.
  The energy policy accumulates contributions from the gradual tail;
  the knee policy finds an early inflection that misses this tail energy.

- **Conservative policy comparison:** `optimal_hard_threshold` (based on
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

## 4. Cross-Seed Stability of Spectral Metrics

### Claim

Stable rank is highly consistent across training seeds for the same
architecture-task pair. Layer-specific utilization ratios show more
variance, particularly in attention layers.

### Evidence

- **Multi-seed audit:** Running `gradience audit` on 3 independently
  trained Mistral-7B/GSM8K adapters (same hyperparameters, different
  random seeds) shows:
  - Stable rank: low coefficient of variation across seeds (typically $<0.1$)
  - Energy rank at 90%: similarly stable, with occasional $\pm 1$ variation
  - Utilization: moderate variance in specific attention layers, with
    coefficients of variation up to $0.2$--$0.3$ in some `q_proj` and
    `k_proj` layers

- **Interpretation:** Stable rank and energy rank capture the "shape" of
  the spectral distribution, which appears to be determined primarily by
  the architecture and task, not the random initialization. Utilization
  variability reflects seed-dependent differences in how much of the
  allocated rank a particular layer's optimizer trajectory happens to use.

### Limitations

- Three seeds is the minimum for meaningful variance estimation. The
  95% confidence intervals are wide with $n = 3$.
- Only one architecture-task pair tested. Cross-architecture or cross-task
  stability of this observation is unknown.
- The distinction between "metric stability" and "learned representation
  stability" is important: stable rank may be consistent even if the
  actual learned subspaces (principal vectors) differ across seeds.

### Reproducibility

Use `gradience-bench` multi-seed mode with $\geq 3$ seeds. Compare per-layer
`stable_rank` and `utilization` across seeds using the structured JSONL output.


---

## 5. Attention vs. MLP Spectral Structure

### Claim

Attention layers and MLP layers exhibit systematically different spectral
profiles. Attention layers (especially `q_proj` and `k_proj`) tend to
have lower utilization and sharper spectral concentration than MLP layers
(`gate_proj`, `up_proj`, `down_proj`).

### Evidence

- **Module-type aggregation:** Gradience's audit groups layers by inferred
  module type (`attn`, `mlp`, `other`). In the Mistral-7B/GSM8K experiments,
  attention layers show mean utilization of $\sim 0.08$--$0.12$, while
  MLP layers show $\sim 0.10$--$0.18$.
- **Spectral shape:** Attention layers more frequently exhibit a dominant
  first singular value ($\sigma_1 / \sigma_2 > 3$) compared to MLP layers,
  which tend toward smoother spectral decay.

### Limitations

- Observed on a single architecture. Architectures with different MLP
  structures (e.g., gated vs. standard) may show different patterns.
- The functional interpretation (why attention specializes more sharply)
  is speculative.

### Reproducibility

Examine the `module_type` field in `gradience audit --json` output and
aggregate metrics by type.
