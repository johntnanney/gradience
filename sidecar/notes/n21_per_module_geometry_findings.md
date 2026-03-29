# Note: Per-Module Geometry Findings

## Metadata

- **Type:** findings
- **Date:** 2026-03-26
- **Related notes:** n20 (protocol), n19 (subset), n18 (within-layer findings)
- **Project:** Per-Module Geometry Program, Stage B

---

## Purpose

This note reports the results of the per-module geometry pilot. It answers: does the decisive difference between catastrophic and safe collision cases appear at the per-module level, concentrated in specific attention components?

**Overall classification: POSITIVE — the V module (value matrices) provides the cleanest separation in the entire sidecar evidence base.**

The per-module decomposition succeeds where the aggregate within-layer analysis failed. The value projection module shows a catastrophic/safe collision separation with Cohen's d = 3.36 and zero range overlap on dimensionality ratio — meaning no catastrophic variant's V-module dimensionality ratio overlaps with any safe collision variant's. This is the first metric in the sidecar program to cleanly separate these groups when backbone is controlled.

---

## 1. Group-Level Per-Module Summary (Critical Layers)

| Module | Metric | Group 1 (Catastrophic) | Group 2 (Safe Collision) | Group 3 (Non-Collision) |
|:------:|--------|:----------------------:|:------------------------:|:-----------------------:|
| **Q** | Principal cos | 0.245 | 0.246 | 0.198 |
| | Top overlap | 0.217 | 0.196 | 0.132 |
| | Dim ratio | 0.824 | 0.787 | 0.647 |
| | Conflict | 0.449 | 0.452 | 0.467 |
| **K** | Principal cos | 0.361 | 0.307 | 0.373 |
| | Top overlap | 0.310 | 0.244 | 0.214 |
| | Dim ratio | 0.812 | 0.845 | 0.719 |
| | Conflict | 0.441 | 0.439 | 0.458 |
| **V** | Principal cos | 0.281 | 0.271 | 0.189 |
| | Top overlap | 0.281 | 0.245 | 0.140 |
| | Dim ratio | **0.783** | **0.837** | 0.710 |
| | Conflict | 0.441 | 0.434 | 0.497 |
| **O** | Principal cos | 0.312 | 0.273 | 0.335 |
| | Top overlap | 0.285 | 0.228 | 0.254 |
| | Dim ratio | 0.811 | 0.825 | 0.697 |
| | Conflict | 0.464 | 0.491 | 0.511 |

### Reading the table

The group-level summary includes all 24 variants (8 per group), mixing DistilBERT and RoBERTa. The backbone confound identified in n18 is therefore present. The backbone-controlled discrimination analysis (§2) is the authoritative comparison.

At the group level, the most visible pattern is that Group 3 (non-collision) consistently shows lower dimensionality ratios across all modules, confirming the n18 finding that non-collision pairs have more asymmetric effective ranks. The Group 1 vs. Group 2 contrast is subtler and requires backbone control.

---

## 2. Module Discrimination: Catastrophic vs. Safe Collision (RoBERTa Only)

This is the study's primary analysis. Restricting to RoBERTa cases (CA-02 vs. SC-QMRB + SC-MSRB) removes the backbone confound.

| Module | Metric | G1 Mean | G2 Mean | Cohen's d | Range Overlap |
|:------:|--------|:-------:|:-------:|:---------:|:-------------:|
| **V** | **Dim ratio** | **0.693** | **0.837** | **3.36** | **0.000** |
| V | Top overlap | 0.133 | 0.245 | 2.00 | 0.113 |
| V | Principal cos | 0.155 | 0.271 | 1.78 | 0.319 |
| V | Conflict | 0.475 | 0.434 | 1.47 | 0.497 |
| K | Dim ratio | 0.765 | 0.845 | 1.39 | 0.259 |
| K | Top overlap | 0.195 | 0.244 | 1.08 | 0.423 |
| Q | Dim ratio | 0.715 | 0.787 | 0.93 | 0.435 |
| Q | Top overlap | 0.164 | 0.196 | 0.91 | 0.421 |
| O | Top overlap | 0.140 | 0.228 | 0.81 | 0.496 |
| Q | Principal cos | 0.215 | 0.246 | 0.77 | 0.254 |

(Table sorted by Cohen's d. Only top 10 of 16 module × metric combinations shown.)

### Key findings

**V module dominates discrimination.** All four of the V module's metrics appear among the top 6 discriminators. The V-module dimensionality ratio (d = 3.36, overlap = 0.000) is the single strongest signal in the sidecar program. For context, the best signal from the aggregate within-layer analysis (n18) was top direction overlap with d ≈ 0.8, heavily confounded by backbone.

**Dimensionality ratio is the key metric.** Across all four modules, dimensionality ratio shows consistent Group 1 < Group 2 directionality: catastrophic pairs have more asymmetric effective ranks within each module. This pattern is strongest in V (d = 3.36), followed by K (d = 1.39), then Q (d = 0.93).

**The O module is the weakest discriminator.** The output projection shows the smallest group differences. This is consistent with the interpretation that the catastrophic mechanism operates in the attention body (V/K), not in the residual-stream projection.

**Interpretation:** Catastrophic collision pairs show more dimensionality asymmetry in their value (and to a lesser extent key) projections at critical layers. One adapter's V-module perturbation occupies a substantially different number of effective dimensions than the other's. When these are linearly merged, the lower-rank perturbation is either swamped by or destructively averaged with the higher-rank one. Safe collision pairs have more balanced V-module dimensionality — their perturbations are commensurable even though their subspaces overlap.

---

## 3. Seed Sensitivity Analysis

### 3.1 CA-01 (DistilBERT): NEGATIVE per module

| Module | Δcos | Δoverlap | Δdim_ratio | Δconflict |
|:------:|:----:|:--------:|:----------:|:---------:|
| Q | -0.039 | -0.022 | -0.017 | -0.001 |
| K | -0.011 | -0.015 | -0.042 | -0.001 |
| V | +0.034 | -0.008 | +0.006 | +0.006 |
| O | -0.034 | +0.003 | +0.070 | -0.002 |

(Worst variant s42×s7 minus mild variant s7×s7.)

**All deltas are < 0.07.** The 29-percentage-point severity gap (41.7% vs. 12.7%) is invisible at per-module resolution, just as it was at the aggregate level (n18 §4). The seed-dependent variable in CA-01 operates below per-module granularity — likely at the level of individual attention heads, specific weight directions, or output-space interaction.

### 3.2 CA-02 (RoBERTa): POSITIVE per module — O and V modules concentrate the signal

| Module | Δcos | Δoverlap | Δdim_ratio | Δconflict |
|:------:|:----:|:--------:|:----------:|:---------:|
| Q | -0.034 | -0.080 | -0.201 | +0.068 |
| K | +0.020 | -0.055 | -0.142 | +0.080 |
| V | **-0.151** | **-0.113** | -0.101 | +0.044 |
| O | **-0.313** | **-0.241** | -0.074 | +0.004 |

(Toxic adapter qnli_s42 mean minus benign adapter qnli_s7 mean.)

The toxic adapter (qnli_s42) shows dramatically different per-module geometry than the benign adapter (qnli_s7):

- **O module:** Δcos = -0.31, Δoverlap = -0.24. The toxic adapter's output projection subspace is nearly orthogonal to SST-2's (cos = 0.055, overlap = 0.020), while the benign adapter's is moderately aligned (cos = 0.368, overlap = 0.261). This is the largest per-module delta in the study.

- **V module:** Δcos = -0.15, Δoverlap = -0.11. The toxic adapter's value projection is substantially less aligned with SST-2's than the benign adapter's.

- **Dimensionality ratio drops across all modules** for the toxic adapter, with Q showing the largest drop (Δ = -0.20). The toxic adapter's perturbations are more dimensionally asymmetric with SST-2's across all modules.

- **Conflict increases for Q and K** but not for V or O. The toxic adapter shows more directional opposition in query and key subspaces but not in value or output.

**Interpretation:** The CA-02 toxic adapter (qnli_s42) learned perturbations that are geometrically incompatible with SST-2 at the module level, particularly in the output projection (where the merge feeds back into the residual stream) and the value projection (where the merge determines what information is passed through attention). The benign adapter (qnli_s7) learned perturbations that are more aligned and dimensionally commensurable with SST-2's, allowing a linear merge to partially preserve both tasks' features.

This was completely invisible in the aggregate within-layer analysis (n18 §4 reported the CA-02 toxic adapter pattern as "counterintuitive" — lower aggregate overlap, higher aggregate conflict). The per-module decomposition resolves the mystery: the aggregate was averaging across four modules with very different patterns, producing a misleading summary.

---

## 4. The V-Module Dimensionality Ratio Signal

This section documents the study's strongest finding in detail.

### What the signal is

Catastrophic collision cases (CA-02 on RoBERTa) have V-module dimensionality ratios of 0.64–0.74 at critical layers. Safe collision controls (SC-QMRB, SC-MSRB on RoBERTa) have V-module dimensionality ratios of 0.79–0.89. The ranges do not overlap.

### What it means geometrically

Dimensionality ratio = min(eff_rank_a, eff_rank_b) / max(eff_rank_a, eff_rank_b). A ratio of 0.65 means one adapter's V-module perturbation occupies ~35% fewer effective dimensions than the other's. A ratio of 0.85 means the perturbations occupy roughly comparable dimensionality.

In catastrophic cases, the two adapters' value projections are "structurally incommensurable" — one concentrates its learned transformation in a tight subspace while the other spreads it more broadly. A linear merge of such perturbations cannot preserve the narrow adapter's precision; the merge averages a low-rank signal with a higher-rank one, smearing the concentrated features.

In safe collision cases, the value projections have similar effective dimensionality. Even though the subspaces may overlap (they are in the collision regime, after all), the perturbations are structurally commensurable — they "speak the same language" in terms of dimensionality, even if they say different things.

### Caveats

1. **Two-backbone result.** The discrimination analysis controls for backbone (RoBERTa only), but the sample is small (4 catastrophic variants vs. 8 safe collision variants). DeBERTa confirmation is needed before promotion.

2. **One catastrophic case.** CA-02 is the only catastrophic case on RoBERTa in the contrast panel. The signal may be specific to QNLI×SST-2 rather than generalizing to all catastrophic collision pairs.

3. **Dimensionality ratio is a necessary condition, not a sufficient one.** Group 3 (non-collision) also shows low dimensionality ratios (V mean = 0.710). The V-module signal distinguishes catastrophic collision from safe collision; it does not distinguish catastrophic from non-collision.

---

## 5. Outcome Classification

**Classification: POSITIVE.**

### Why POSITIVE

1. **The V-module dimensionality ratio achieves zero range overlap.** No other metric in the sidecar program has achieved this on the backbone-controlled comparison.

2. **The V-module concentrates the signal.** Three of the top four discriminators are V-module metrics (d = 3.36, 2.00, 1.78). This confirms the per-module decomposition hypothesis: the catastrophe-relevant signal was being diluted by aggregation across modules with different patterns.

3. **CA-02 seed sensitivity is now partly explained.** The toxic adapter's O and V modules show large geometry differences from the benign adapter — differences that were invisible in the aggregate analysis.

### Why not STRONG POSITIVE

1. CA-01 seed sensitivity remains unexplained at per-module resolution.
2. The signal comes from a single catastrophic case on RoBERTa (CA-02).
3. The mechanism is correlational — we have identified *what* is different (V-module dimensionality asymmetry) but not *why* it causes catastrophe (the causal chain from V-module geometry to performance degradation is interpretive).

---

## 6. Implications

### 6.1 For the instability concept

The V-module dimensionality ratio is a candidate for a more fine-grained instability predictor. If DeBERTa confirms that catastrophic collision pairs show lower V-module dimensionality ratios, this could become the first predictive signal (as opposed to the current instability measure, which is a retrospective descriptor requiring multi-seed evaluation).

### 6.2 For the collision model

The collision model (n15) identified shared-layer loading as a necessary precondition. The per-module analysis refines this: within the collision regime, it is the value projection's dimensionality structure that distinguishes catastrophic from safe outcomes. The updated mechanistic picture:

```
Collision (shared-layer loading) → necessary precondition
V-module dimensionality asymmetry → strongest correlate of catastrophic outcome
O-module alignment collapse → strongest correlate of toxic adapter identity (CA-02)
Aggregate subspace geometry → too coarse (dilutes module-specific signals)
CA-01 seed sensitivity → below per-module resolution
```

### 6.3 For output-space escalation (Stage D)

The finding that the O module (output projection) concentrates the toxic adapter signal in CA-02 provides a natural bridge to Workstream C (output-space analysis). The output projection feeds directly into the residual stream and ultimately into the classification head. If the O-module geometry is where the "merge damage" enters the output, then output-space incompatibility analysis could explain why some V-module asymmetries produce catastrophic results and others do not.

However, Stage D is no longer the immediate priority. The V-module signal is strong enough to justify DeBERTa replication (Stage C or DeBERTa S01 leg) as the next step.

### 6.4 For core Gradience (speculative)

If the V-module dimensionality ratio proves portable across backbones, it could become a computable pair-level warning signal:

1. Extract V-module W matrices at high-norm layers
2. Compute effective rank for each adapter's V-module perturbation
3. If the dimensionality ratio is below a threshold (e.g., < 0.75), flag the pair for caution

This would be a qualitative advance over the current binary (same-task safe / cross-task caution) boundary. But this is speculative — the signal needs DeBERTa confirmation and larger-sample validation.

---

## 7. Recommended Follow-Up

**Priority 1:** DeBERTa replication. Does the V-module dimensionality ratio signal persist on a third backbone? DeBERTa-v3's disentangled attention architecture separates content and position information in a way that may affect V-module geometry differently. This is the adjudication test.

**Priority 2:** Per-module seed sensitivity for CA-01 at sub-module resolution. The CA-01 seed effect is invisible at per-module granularity. The remaining candidates are: (a) attention-head-level decomposition within the V module, (b) output-space interaction analysis. Both require more code but are CPU-feasible.

**Priority 3:** Validation on non-panel pairs. The current result comes from a 6-case contrast panel. Expanding to additional cross-task pairs (QNLI×RTE, RTE×SST-2) would test whether V-module dimensionality ratio generalizes beyond the panel.

---

## 8. Structured Outputs

| File | Location |
|------|----------|
| Module metrics (all variants) | `sidecar/results/per_module_geometry/module_metrics.json` |
| Group module comparison | `sidecar/results/per_module_geometry/group_module_comparison.json` |
| Module discrimination | `sidecar/results/per_module_geometry/module_discrimination.json` |
| Seed sensitivity per module | `sidecar/results/per_module_geometry/seed_sensitivity_per_module.json` |
| Group comparison figure | `sidecar/figures/per_module_group_comparison.svg` |
| Discrimination heatmap | `sidecar/figures/per_module_discrimination.svg` |
| CA-02 seed sensitivity figure | `sidecar/figures/per_module_ca02_seed_sensitivity.svg` |
| V-module spotlight figure | `sidecar/figures/per_module_v_spotlight.svg` |
