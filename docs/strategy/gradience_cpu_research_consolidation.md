# Gradience CPU Research Consolidation

**Date**: 2026-04-03
**Scope**: All CPU-only research lines through completion of analytical spectral geometry study
**Purpose**: Canonical packet for what is established, bounded, exploratory, negative, and what requires GPU

---

## 1. Stable Bounded Findings

These are operationally usable results with frozen policies. They have been
validated, bounded, and are not expected to change without new substrate
(decoder models, GPU compute).

### 1a. Rank-Proxy Validation

**Status**: Bounded positive, frozen policy

Spectral rank policies are competitive fixed-budget compression guides in
compressible encoder families (sst2, imdb on distilbert). Tested across
12 adapters, 9 methods, 3 budget levels (0.35/0.50/0.65), 324 compression
evaluations.

| Proxy | Role | Evidence |
|-------|------|----------|
| `proxy_gradient` | Primary operational comparator | Best mean compression outcome (delta_vs_uniform +0.006), 2x stability over ablation (Spearman 0.90 vs 0.40) |
| `oht` | Lead spectral policy | Best spectral (delta_vs_uniform +0.004-0.010), structurally aligned with ablation patterns |
| `proxy_ablation` (attenuate) | Companion explanatory | Useful with tie-aware diagnostics; less stable than gradient |
| `rank_reduction` | Paused | Persistent degenerate behavior at retain_ratio 0.75/0.85; not reliable |

**Boundary**: CPU-only, shared-base small encoders, classification tasks,
compressible families. Not a universal policy dominance claim.

**Artifacts**: `field_trials/rank_proxy_validation/`, `field_trials/rank_proxy_validation_v2/`,
`docs/00_start_here/bounded-validation-summary.md`

### 1b. Ablation Proxy Resolution

**Status**: Resolved bounded

Tie-aware reliability cleanup confirmed gradient dominance. Gradient is
substantially more stable under resampling. Attenuate remains useful as
structural companion evidence. Rank-reduction soft-ablation showed
persistent low-information behavior.

**Decision**: Use gradient for operational comparison, attenuate for
structural evidence, avoid rank-reduction unless tightly scoped.

**Artifact**: `docs/strategy/ablation_proxy_resolution_summary.md`

### 1c. Collapse vs Contamination Distinction

**Status**: Replicated with guardrails

Behavioral distinction between collapse-like and contamination-like
failure modes replicates on modest panel. Collapse targets show
uncertainty-dominant signatures (high confidence-collapse, near-zero
high-confidence-wrong). Contamination targets show confident-wrong-dominant
signatures. Confidence-channel metrics are the key discriminators.

**Boundary**: Bounded to merge-facing decision context, tested
case family/backbone. No universal cross-context claim.

**Artifact**: `docs/strategy/collapse_vs_contamination_summary.md`,
`sidecar/results/route2_stress_tests/collapse_vs_contamination/`

---

## 2. Secondary Bounded Findings

Useful results with clear scope limits. They provide operational value
but are not front-line decision tools.

### 2a. Phase Probes (HTSR Tail Exponent, Edge-Gap)

**Status**: Bounded keep, secondary observables

Both probes show bounded regime sensitivity but lack robustness for
front-line summary metrics. HTSR alpha estimates are noisy on small
adapter matrices. Edge-gap captures spectral shape information not
in stable rank alone but adds marginal discriminative power.

**Decision**: Retain as secondary research observables in the
hierarchical stack (below stable rank, effective rank, DFA).
Do not promote to primary metrics or product surface.

**Artifacts**: `docs/strategy/phase_probe_addon_summary.md`,
`docs/design/htsr_tail_estimator.md`, `docs/design/spectral_edge_gap_probe.md`

### 2b. Merge-Aware Training Monitor

**Status**: Bounded keep, internal diagnostic only

Optional HuggingFace callback that compares adapter state against a fixed
reference adapter during training. Emits periodic compatibility snapshots
and conservative run-end trend labels.

Validated in demo: callback infrastructure works, telemetry shape correct,
4 snapshots emitted at configured cadence. Run-level trend label was
`inconclusive` in simulation — needs real training runs to assess
interpretability.

**Reference-choice guidance** (from tiny synthetic study):
- Same-task reference: most interpretable (overlap/score trends coherent)
- Same-family reference: partially readable, mixed trajectories
- Cross-task reference: not interpretable under current trend rules

**Boundary**: Does not support optimizer feedback, auto-stop, or
"improved training" claims. Internal research tool only.

**Artifacts**: `gradience/vnext/integrations/merge_aware_monitor.py`,
`field_trials/merge_aware_monitor_demo/`,
`field_trials/merge_aware_monitor_reference_choice/`,
`docs/strategy/merge_aware_monitor_summary.md`

---

## 3. Exploratory Lines

Signal exists but is not strong enough or stable enough for policy use.

### 3a. Over-Accumulation Diagnostic

**Status**: Keep exploratory, not policy-ready

The over-accumulation diagnostic estimates when high-overlap adapter pairs
may amplify shared directions under naive linear merge. The structural
signal exists (the heuristic formula has interpretable components), but
it does not predict merge quality.

**OA-v1** (heuristic): `alignment × (0.7·concentration + 0.3·coeff_exposure)`
- Remains the authoritative production advisory path
- Advisory activates correctly on high-overlap pairs
- Does not predict which high-overlap merges will fail

**OA-v2** (interaction-first): Uses full rank-r geometry from right-space
principal angles and effective singular values.
- 30-pair strict-naive gate: cohort design PASS, threshold/policy FAIL
- Spearman gain vs v1: -0.068 (needed ≥0.15)
- Recall gain: -0.083 (needed ≥0.20)
- Sign consistency: 0.556 (needed ≥0.70)

Neither v1 nor v2 meets promotion criteria. Both remain diagnostic-only.

**Artifacts**: `gradience/vnext/merge/over_accumulation.py`,
`field_trials/over_accumulation_followup/`,
`docs/strategy/over_accumulation_summary.md`

---

## 4. Null / Negative / Paused Lines

### 4a. Analytical Spectral Geometry (Negative Completion)

**Status**: Negative completion — informative

Derived closed-form over-accumulation conditions for linear merge. The
math is correct: the cross-term that drives spectral inflation is
`2αβ Σ_i s_{a,i} s_{b,i} cos(θ_i) cos(φ_i) sign(δ)`. Validated on
synthetic matrices (26 tests pass, Weyl bounds hold 100%, Frobenius
error < 0.05).

**Empirical cross-check (30 pairs): all metrics have wrong-sign Spearman.**

| Metric | Spearman (vs merge delta) |
|--------|--------------------------|
| OA-v1 max | +0.186 (wrong sign) |
| OA-v2 max | +0.262 (wrong sign) |
| Theory risk | +0.382 (wrong sign) |
| Theory inflation ratio | +0.386 (wrong sign) |

Higher predicted spectral inflation correlates with *better* merges.
Root cause: the worst merges are cross-task pairs with low overlap
(task mismatch, not spectral inflation). The best merges are same-task
pairs with high overlap (aligned features merge beneficially).

**Key finding**: Spectral metrics characterize geometry; task
relationship predicts quality. The merge pipeline should use both,
not substitute one for the other.

**What this invalidates**:
- Over-accumulation score as merge quality predictor
- Threshold tuning on spectral metrics alone
- Phases 3-5 of the spec (TIES/DARE/strategy-selection bounds) —
  not pursued because the empirical cross-check showed the underlying
  question is wrong

**What this validates**:
- The verdict tree's branch structure as geometric classification
- Norm equalization's theoretical basis (reduces cross-term when ρ_F >> 1)
- The spectral audit as a valid geometric characterization tool

**Artifacts**: `gradience/vnext/merge/spectral_theory.py` (13 dataclasses,
11 functions), `field_trials/analytical_spectral_geometry/`,
`docs/theory/linear_merge_spectral_analysis.md`,
`docs/theory/over_accumulation_theory.md`

### 4b. Rank-Reduction Expansion (Paused)

**Status**: Paused indefinitely

Rank-reduction soft-ablation (retain_ratio 0.75, 0.85) showed persistent
degenerate/low-information behavior across tested regimes. Not worth
further investment without a specific hypothesis for why it would work
in a new regime.

### 4c. OA-v2 Promotion (Blocked)

**Status**: Blocked behind gate criteria

OA-v2 cannot be promoted to policy until all four gate rules pass.
Three of four rules failed on the 30-pair cohort. The analytical
spectral geometry study further weakened the case by showing the
underlying spectral inflation measure has wrong-sign correlation
with merge quality.

---

## 5. Decoder-Side Ecological Evidence

### 5a. Public Ecosystem Spectral Census

**Status**: Pilot-plus gate PASS, partial success

CPU-only spectral fingerprinting of 49 public decoder LoRA adapters
across 3 architecture families (llama: 18, mistral: 15, qwen: 16).

**Key findings**:

| Result | Value |
|--------|-------|
| Pipeline viability | 98% (49/50 audited) |
| Architecture eta-sq (mean) | 0.20 |
| Task eta-sq (mean) | 0.09 |
| Dominant factor | Architecture |
| Strongest architecture signals | utilization (0.39), entropy_erank (0.29), energy_rank_90 (0.28) |
| kNN architecture purity | 0.81 (random: 0.34) |
| kNN task purity | 0.77 (random: 0.56) |
| Confound R² (nominal rank) | 0.65 — residualization required |
| Module-type asymmetry (attn < MLP) | 11% — does NOT replicate encoder pattern |

**Task diversity gap**: chat_instruct dominates (35/49). Only 2 task
categories have ≥5 adapters. This limits task-vs-architecture conclusions.

**Pilot-plus gate**: All 4 conditions passed (viability ≥70%, 3 families
with ≥5, residualized signal >0.05 for 4 metrics, 7 subtypes with ≥10
layers). Recommendation: proceed to core cohort.

**Guardrails**: Observational findings from found artifacts, not causal
claims. Confound assessment required before interpreting effects. Census
does not replace controlled GPU-return study.

**Artifacts**: `field_trials/public_ecosystem_census/`,
`scripts/ecosystem_census.py`

---

## 6. What Next Requires Controlled GPU Work

The CPU-only research phase is complete. All lines have reached usable
decision states (established, bounded, exploratory, or negative). The
remaining high-leverage questions require decoder-side controlled
experiments with GPU compute.

### 6a. Decoder-Only Spectral Fingerprinting (Highest Priority)

The census established that architecture-level spectral signatures exist
in public decoder adapters, but the ecological data cannot control for
confounds (nominal rank R²=0.65, task imbalance, adapter quality
variation). A controlled study would:

- Train matched adapter pairs on the same tasks with the same
  hyperparameters across architectures
- Measure whether spectral fingerprints (stable rank, effective rank,
  utilization, entropy) are architecture-intrinsic or training-determined
- Validate whether the census's architecture > task finding holds
  under controlled conditions

**Spec**: `docs/plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`

### 6b. Task-Conditioned Merge Quality Prediction

The analytical spectral geometry negative result shows that spectral
metrics alone cannot predict merge quality — task relationship is the
dominant factor. A GPU study could:

- Build same-task and cross-task adapter pairs with known behavioral
  evaluations
- Test whether spectral metrics conditioned on task relationship
  (same-task pairs only) do predict merge quality
- Determine the minimum behavioral information needed alongside
  spectral metrics for reliable merge quality prediction

### 6c. Decoder Merge Empirical Validation

All merge pipeline validation (verdict tree, strategies, over-accumulation)
has been on small encoder models (distilbert, bert-base). Decoder-scale
validation would:

- Run merge audit + execution on 7B-class adapter pairs
- Validate whether verdict thresholds transfer from encoder to decoder
- Test norm-equalized merge effectiveness at decoder scale

---

## Summary Table

| Line | Status | Tier | Key Number |
|------|--------|------|------------|
| Rank-proxy validation | Bounded positive | Stable operational | gradient Spearman 0.90 |
| Ablation proxy | Resolved bounded | Frozen policy | gradient > attenuate > rank-reduction |
| Phase probes (HTSR/edge-gap) | Bounded keep | Secondary observable | Below stable rank in hierarchy |
| Merge-aware monitor | Bounded keep | Internal diagnostic | Same-task ref most interpretable |
| Over-accumulation v1 | Keep exploratory | Diagnostic only | 30-pair gate: 1/4 rules pass |
| Over-accumulation v2 | Blocked | Not promotable | Spearman gain -0.068, recall gain -0.083 |
| Analytical spectral geometry | Negative completion | Theory-closed | Spearman +0.38 (wrong sign) |
| Collapse vs contamination | Replicated | Bounded explanatory | Confidence channels discriminate |
| Public ecosystem census | Pilot-plus PASS | Observational | Arch eta-sq 0.20 > task 0.09 |
| Rank-reduction | Paused | Not pursued | Degenerate behavior |

## Posture

**Freeze and maintain.** The CPU phase is complete. No new CPU-only
research branches should be opened. The existing results are consolidated,
bounded, and documented. The next high-leverage work is decoder-side
controlled GPU experiments.

**Communication rule**: Lead with stable findings, use bounded findings
operationally, acknowledge exploratory lines as diagnostic-only, cite
negative results as scope boundaries. Do not claim spectral metrics
predict merge quality — they characterize geometry. Task relationship
is the quality signal.
