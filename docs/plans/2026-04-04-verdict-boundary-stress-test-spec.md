# Verdict Boundary Stress Test — Spec

**Date:** April 4, 2026
**Status:** proposed
**Depends on:** N127 (MP partition extensions), merge pipeline (`vnext/merge/`)
**Addresses:** Operational calibration of triage logic against spectral partition findings


## Motivation

The merge pipeline routes adapter pairs through verdict branches — SAFE,
REDUNDANT, CONFLICTING, IMBALANCED — based on spectral observables computed
at each layer. The thresholds governing these branches were established
empirically on a small-encoder corpus (DistilBERT-base, rank 4–32,
SST-2/QNLI tasks). They have been validated in the sense that the verdicts
correlate with merge outcomes in the tested regime, but they have not been
stress-tested against the spectral partition findings from N127.

N127 established that same-task and cross-task adapter pairs have
fundamentally different spectral profiles in the high-SV band:

- Same-task pairs: 0.634 high-SV alignment, 7.8× H/L ratio
- Cross-task pairs: 0.133 high-SV alignment, 2.5× H/L ratio

This raises a concrete operational question: **are the verdict branches
sensitive to this difference in the way they should be?** Specifically:

1. Do same-task SAFE verdicts and cross-task SAFE verdicts have
   distinguishable spectral profiles?
2. Do the existing thresholds draw the boundaries in the right place,
   or do they conflate structurally different situations under the
   same verdict label?
3. Would task-relationship-aware thresholds improve triage quality?


## Scope

This work is a diagnostic audit of the existing pipeline, not a redesign.
The deliverable is an empirical characterization of how the current verdict
branches interact with the same-task/cross-task distinction. If the
characterization reveals that the boundaries are well-calibrated, we
document that and stop. If it reveals systematic misalignment, we propose
threshold adjustments with evidence — but do not implement them in this
phase.


## Background: Current Verdict Logic

Reference: `gradience/vnext/merge/verdicts.py`, function `assess_layer()`.

The verdict tree is a six-branch priority cascade. The spectral observables
that govern routing are:

| Observable | Source | Role in verdict logic |
|---|---|---|
| `mean_overlap` | Mean of principal angle cosines (cos θᵢ) | Primary geometry discriminator |
| `directional_agreement` | Projection cosine similarity | Distinguishes REDUNDANT from CONFLICTING |
| `frobenius_ratio` | ‖ΔW_larger‖_F / ‖ΔW_smaller‖_F | Triggers IMBALANCED |
| `magnitude_ratio` | σ₁(larger) / σ₁(smaller) | Secondary scale check |
| `effective_rank` | Energy-threshold rank (90%) | Compression and rank recommendations |

Default thresholds (`VerdictThresholds`):

| Threshold | Value | Controls |
|---|---|---|
| `low_overlap` | 0.20 | Below → orthogonal (SAFE, Branch 1) |
| `high_overlap` | 0.50 | Above → significant overlap zone |
| `aligned` | 0.50 | Above → REDUNDANT (Branch 2) |
| `conflicting` | −0.30 | Below → CONFLICTING (Branch 3) |
| `imbalanced` | 5.0 | Frobenius ratio above → IMBALANCED |
| `imbalanced_frob` | 5.0 | Same, Frobenius-specific |

Branch priority: IMBALANCED(low-overlap) → SAFE(orthogonal) →
REDUNDANT → CONFLICTING → IMBALANCED(high-overlap) → SAFE(default).

Two additional presets exist (`conservative`, `permissive`) that shift
all thresholds toward stricter or looser detection.


## What the N127 Findings Predict

The spectral partition results generate specific predictions about how
the verdict observables should behave for same-task vs cross-task pairs:

**Prediction 1: mean_overlap should be systematically higher for
same-task pairs.** If high-SV alignment is 0.634 (same-task) vs 0.133
(cross-task), and the principal angle cosines that feed `mean_overlap`
are computed over the same subspaces, then same-task pairs should cluster
higher on the overlap axis. The question is whether the 0.20 and 0.50
thresholds fall in the right place relative to these two distributions.

**Prediction 2: directional_agreement should be more strongly positive
for same-task pairs.** Same-task adapters optimizing the same loss
surface should produce updates that point in similar directions, pushing
directional agreement toward the REDUNDANT branch. Cross-task adapters
may show weaker or even negative directional agreement.

**Prediction 3: the SAFE(default) branch may absorb structurally
different cases.** Branch 5 catches everything that falls through the
earlier branches. If same-task pairs with moderate overlap land in the
same branch as cross-task pairs with moderate overlap, the pipeline is
treating geometrically distinct situations identically. The default
0.5 confidence assigned to Branch 5 verdicts reflects this ambiguity
but does not resolve it.

**Prediction 4: Frobenius ratios should be relatively task-independent.**
Magnitude imbalance is a property of training dynamics (learning rate,
number of steps, data volume), not task identity per se. We would not
expect the IMBALANCED branches to be strongly task-sensitive.


## Experiment Design

### Phase 1: Profile Collection

**Goal:** Collect per-layer spectral profiles and verdict assignments for
a controlled set of adapter pairs spanning the same-task / cross-task
distinction.

**Adapter pairs (from adjudication study verified adapters):**

Same-task pairs (SST-2):
- sst2_r16_s42 × sst2_r16_s123
- sst2_r16_s42 × sst2_r16_s123_v2

Same-task pairs (QNLI):
- qnli_r16_s42 × qnli_r16_s123
- qnli_r16_s42 × qnli_r16_s123_v2

Cross-task pairs:
- sst2_r16_s42 × qnli_r16_s42
- sst2_r16_s42 × qnli_r16_s123
- sst2_r16_s123 × qnli_r16_s42
- sst2_r16_s123 × qnli_r16_s123

Cross-rank pairs (same task):
- sst2_r16_s42 × sst2_r8_s42
- qnli_r16_s42 × qnli_r8_s42

**For each pair, collect:**
- Full `MergeAuditReport` via `merge_audit()` (default thresholds)
- Per-layer `SubspaceMetrics` (all fields)
- Per-layer `LayerVerdict` including branch assignment
- Aggregate verdict distribution
- Re-run with `conservative` and `permissive` threshold presets

**Additionally, for each pair, compute from `mp_partition_extensions.py`:**
- High-SV band alignment (above MP threshold)
- Low-SV band alignment (below MP threshold)
- H/L ratio
- Per-layer energy concentration

This produces paired observations: the verdict pipeline's view of each
pair alongside the spectral partition's view of the same pair.


### Phase 2: Distributional Analysis

**Goal:** Characterize how same-task and cross-task pairs distribute
across the verdict observable space, and where the current thresholds
fall relative to those distributions.

**Analysis 1: Observable distributions by task relationship.**

For each observable (`mean_overlap`, `directional_agreement`,
`frobenius_ratio`, `magnitude_ratio`, `effective_rank`), compute:
- Distribution (mean, std, range) for same-task layers
- Distribution for cross-task layers
- Separation statistic (t-test, Cohen's d, or Kolmogorov-Smirnov)
- Where the current threshold falls relative to both distributions
  (percentile within each)

This answers: *do same-task and cross-task pairs occupy different
regions of the observable space, and do the thresholds respect that
difference?*

**Analysis 2: Verdict distribution by task relationship.**

Tabulate:
- Verdict counts (SAFE/REDUNDANT/CONFLICTING/IMBALANCED) per layer
  for same-task pairs
- Same for cross-task pairs
- Chi-square or Fisher's exact test for distributional difference
- Confusion-matrix framing: which verdicts are shared across task
  types, which are task-type-specific?

This answers: *does the pipeline produce different verdict distributions
for same-task vs cross-task pairs, and is the difference in the
right direction?*

**Analysis 3: Branch 5 (SAFE default) decomposition.**

For layers that land in Branch 5 specifically:
- Compare spectral profiles of same-task Branch-5 layers vs
  cross-task Branch-5 layers
- Test whether these are distinguishable by the N127 spectral
  partition metrics (high-SV alignment, H/L ratio)
- If distinguishable: the pipeline is conflating structurally
  different cases under the same label

This answers: *is the catch-all SAFE branch hiding a meaningful
distinction?*

**Analysis 4: Threshold sensitivity.**

For each threshold, compute:
- How many layers change verdict between `default`, `conservative`,
  and `permissive` presets?
- Are the threshold-sensitive layers disproportionately same-task or
  cross-task?
- At what threshold values would same-task and cross-task distributions
  be optimally separated for each observable?

This answers: *how fragile are the current boundaries, and would
task-aware thresholds be substantially different?*


### Phase 3: Correlation with Spectral Partition

**Goal:** Directly test whether the N127 spectral partition metrics
predict verdict assignments better than, or complementarily to, the
existing observables.

**Analysis 5: Partition metrics as verdict predictors.**

For each layer across all pairs:
- Compute logistic regression: verdict ~ high_SV_alignment + H/L_ratio
- Compare with: verdict ~ mean_overlap + directional_agreement
- Test whether adding partition metrics to the existing observables
  improves prediction (likelihood ratio test)

This is not proposing that the pipeline should use logistic regression.
It is testing whether the spectral partition captures verdict-relevant
information that the current observables miss.

**Analysis 6: Energy-weighted overlap vs unweighted overlap.**

The current `mean_overlap` is an unweighted mean of principal angle
cosines. The N127 SV-weighted alignment metric weights by singular
values. Compare:
- Correlation between `mean_overlap` and verdict assignment
- Correlation between SV-weighted alignment and verdict assignment
- Are there layers where the two metrics disagree, and which better
  predicts merge outcome?

This tests a specific potential improvement: should `mean_overlap` in
the pipeline be energy-weighted?


## Implementation

### Script: `scripts/verdict_boundary_stress_test.py`

```
Dependencies:
- gradience.vnext.merge (merge_audit, VerdictThresholds)
- gradience.vnext.audit (lora_audit)
- scripts/mp_partition_test (load_adapter_weights, compute_layer_svd,
  mp_threshold, sv_weighted_alignment)
- numpy, scipy.stats
- json (output)

Inputs:
- Adapter paths (adjudication study verified adapters)
- Threshold presets (default, conservative, permissive)

Outputs:
- sidecar/results/verdict_boundary_stress_test/
  - profiles.json     — per-layer metrics for all pairs
  - analysis.json     — distributional statistics
  - report.txt        — formatted summary
```

### Estimated Effort

Phase 1 (profile collection): ~2 hours. Mostly scripting to run
`merge_audit()` on each pair and extract structured results alongside
the MP partition metrics.

Phase 2 (distributional analysis): ~2 hours. Statistical analysis over
the collected profiles. Straightforward once Phase 1 data exists.

Phase 3 (correlation analysis): ~1 hour. Logistic regression and
comparison metrics on the Phase 1 data.

Total: roughly one working session, assuming no pipeline bugs.


## Decision Criteria

The stress test produces one of three outcomes:

**Outcome A: Boundaries are well-calibrated.** Same-task and cross-task
pairs produce different verdict distributions, the thresholds separate
them appropriately, and Branch 5 does not conflate structurally distinct
cases. In this case, we document the result as validation evidence and
make no changes.

**Outcome B: Boundaries are adequate but could be sharpened.** The
distributions differ but the thresholds are not optimally placed, or
Branch 5 contains distinguishable subpopulations. In this case, we
propose specific threshold adjustments (with evidence) and optionally
a task-relationship-aware modifier — but defer implementation to a
separate PR with its own test coverage.

**Outcome C: Boundaries are systematically miscalibrated.** The current
thresholds conflate same-task and cross-task pairs in ways that produce
operationally wrong recommendations (e.g., same-task pairs getting
CONFLICTING verdicts, or cross-task pairs getting SAFE verdicts with
high confidence). In this case, we flag the issue as a priority fix
and design the corrective threshold logic.

In all three outcomes, the work produces a concrete empirical record
that either supports or challenges the current calibration. That record
has value regardless of whether thresholds change.


## What This Does Not Address

- **Decoder-only validation.** All analysis runs on DistilBERT-base
  adapters. Generalizing to 7B+ models requires GPU access and is a
  separate effort.
- **New verdict branches.** We are testing existing branches, not
  proposing new ones. If the analysis suggests that task relationship
  should be a first-class input to the verdict tree, that is a design
  decision for a later phase.
- **Merge outcome ground truth.** This analysis characterizes spectral
  profiles and verdict assignments. It does not re-run actual merges
  to measure task performance degradation. The connection between
  verdicts and merge outcomes rests on the existing validation work.
- **Over-accumulation scoring.** The over-accumulation module has its
  own threshold structure. It is adjacent to this work but out of scope.


## Relationship to Existing Work

This spec operationalizes one of the implications identified in the N127
consolidated conclusions:

> The 7.8× same-task vs 2.5× cross-task ratio suggests that the
> `task_relationship` field in merge QA reports may have spectral
> grounding — same-task pairs have fundamentally more aligned high-SV
> structure. This could eventually support spectral confidence scoring
> for merge recommendations.

The stress test is the empirical step between that observation and any
pipeline change. It asks whether the existing machinery already handles
the distinction well, or whether the N127 findings reveal a gap.
