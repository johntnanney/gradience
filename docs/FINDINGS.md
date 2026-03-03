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


---

## 6. Regime Classification via Early Geometric Features (Reanalysis, March 2026)

### Claim (revised)

Geometric features extracted from the first 200 training steps achieve
~67% five-class regime classification accuracy via Leave-One-Seed-Out
cross-validation, significantly above chance (permutation p = 0.0001).
Loss-only features achieve ~40% (p = 0.0009). The gap between the two
approaches is not statistically significant at the current sample size
(McNemar's p = 0.289). Geometric features carry 7.4x more mutual
information about training regimes than loss.

**Note.** An earlier, informal claim of "100% geometric accuracy vs. 65%
for loss" appeared in project blog posts. This figure is not reproduced
on the available five-class data. It likely derives from a binary
classification problem (baseline vs. low-weight-decay), where geometry
does achieve perfect separation with margins of 0.81--0.99. See the
full reanalysis report in `Gradience II/reanalysis/REANALYSIS_REPORT.md`.

### Evidence

- **Permutation test (n=10,000).** Observed geometry accuracy 66.7%
  (10/15); null mean 13.0%, null 99th percentile 46.7%.
  p = 0.0001.

- **Bootstrap confidence interval (B=5,000).** Mean 47.4%, std 14.1%,
  95% CI [20.0%, 73.3%]. The wide CI reflects the n=15 constraint.

- **Feature ablation.** No single geometric feature exceeds 40% accuracy.
  Best pair: `weight_norm_mean` + `grad_to_weight_ratio_mean` (66.7%).
  Removing `cos_grad_weight_mean` *improves* accuracy to 80.0%,
  suggesting this feature introduces noise at this sample size.

- **Information theory.** KSG mutual information of geometric features
  with regime labels: 4.96 nats (joint). Loss MI: 0.67 nats.
  Ratio: 7.4x. Slight synergy (negative redundancy of -0.17 nats).

- **Minimum description length.** BIC favors loss-only (35.25) over
  geometry (67.92) due to the 35-parameter vs. 10-parameter complexity
  penalty at n=15. Geometry achieves lower NLL but cannot justify its
  parameter cost at this sample size.

### Limitations

- Five regimes (baseline, high_lr, high_wd, low_lr, low_wd) with only
  3 seeds each (n=15 total). This is the minimum viable sample for LOSO.
  Doubling to 6 seeds would substantially narrow the bootstrap CI.

- The `early_spectral_complexity_mean` feature -- theoretically the most
  important geometric quantity -- was never computed. Classification
  results are based on 6 weight/gradient features only.

- The McNemar test is underpowered at n=15. The test cannot distinguish
  67% from 40% accuracy at conventional significance levels.

### Reproducibility

Full protocol, scripts, and module-level JSON results are in
`Gradience II/reanalysis/`. The protocol document is
`Gradience_Reanalysis_Protocol.md`.


---

## 7. Hessian Telemetry and Phase Transition Candidates (Reanalysis, March 2026)

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

## 8. Replication and Regime-Dependent DFA Exponents (Study 12, March 2026)

### Claim

Training regimes produce different long-range temporal correlation
structures in spectral complexity time series. Detrended Fluctuation
Analysis (DFA) exponents for spectral complexity differ significantly
across five hyperparameter regimes (one-way ANOVA F = 116.86,
p ≈ 7.7 × 10⁻²³). High learning rate produces markedly lower
persistence (α ≈ 1.574) than low learning rate (α ≈ 2.073), with
baseline, high weight decay, and low weight decay clustering in between
(α ≈ 1.90--1.92).

Separately, the regime classification results from §6 replicate at
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
  - 7 geometric features: 79.6% (39/49), p = 0.0002
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
  - Within-regime standard deviations are small relative to
    between-regime gaps.

### Interpretation

The DFA result is distinct from the classification result in kind.
Classification asks whether features differ in central tendency across
regimes. DFA asks whether the *temporal correlation structure* of a
feature differs. These are orthogonal: identical means can coexist
with different dynamics, and vice versa.

The finding that high learning rate specifically disrupts temporal
persistence of spectral complexity (α ≈ 1.57 vs. ≈ 1.90--2.07 for
other regimes) suggests that learning rate modulates the stochastic
character of Hessian spectral evolution. This is consistent with
larger per-step perturbations breaking smooth trending behaviour.

### Limitations

- Single architecture (NanoGPT, 6 layers) and single task (Shakespeare
  character-level). Universality across models and tasks is untested.

- DFA exponents > 1.0 indicate non-stationary (trending) series.
  Standard DFA interpretation (0.5 = white noise, 1.0 = 1/f) applies
  to stationary series. The between-regime comparison remains valid
  (different trends have different exponents), but the absolute values
  should not be interpreted as "long-range correlation" in the
  classical sense.

- The regime perturbations are simple (single-parameter changes from
  baseline). More complex hyperparameter interactions are unexplored.

### Reproducibility

Launcher: `Gradience II/reanalysis/study12_replication/run_study12_replication.sh`
Analysis: `Gradience II/reanalysis/study12_replication/analyze_study12.py`
Protocol: `Gradience II/reanalysis/study12_replication/STUDY12_PROTOCOL.md`
Results: `Gradience II/analysis/study12/study12_results.json`
