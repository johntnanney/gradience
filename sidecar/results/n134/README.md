# N134 Results — Decoder-Scale Controlled Merge Triage

Pre-registered confirmatory study; pre-registration: `sidecar/notes/n134_spec.md` (v3.1).
Pipeline: `scripts/n134/` on GPU pod (RTX 6000 Ada, Mistral-7B-v0.3).

## Primary result: H1 NOT CONFIRMED

Hypothesis: O-module depth-weighted spectral alignment (S_H1) predicts
post-merge degradation among cross-task pairs, beyond FAMILY_B membership.

| Criterion | Threshold | Observed | Pass |
|-----------|-----------|----------|------|
| Raw Spearman ρ sign | positive | −0.180 | ✗ (wrong sign) |
| Partial Spearman ρ (\| FAMILY_B) | ≥ +0.50, p<0.05 | −0.533 (p=1.6e−4) | ✗ |
| ΔR² over FAMILY_B baseline | ≥ 0.10 | +0.003 | ✗ |

FAMILY_B alone: R² = 0.881. Adding S_H1: R² = 0.884. Block-bootstrap
(5000 resamples, family-pair blocked): partial ρ ∈ [−0.825, −0.131];
ΔR² ∈ [0.00002, 0.023]. N = 45 evaluated cross-task pairs.

## Confirmatory replications (all PASS)

| Claim | Result |
|-------|--------|
| B-P1 task-boundary detection | 0/24 same below midpoint; 0/252 cross above |
| B-P2 same/cross separation | ratio 2.28× (thr 2.0); Welch t=15.9, p=6e−14 |
| B-P4 erank ANOVA across tasks | F=165.6, p=1.0e−13 |

## Interpretation

Per spec §8: task-boundary detection generalizes to decoder scale; per-pair
risk prediction via O-module depth-weighted alignment does not. FAMILY_B
membership explains ≈88% of the variance in max post-merge degradation,
leaving little room for spectral residual signal at this scale. The product
surface narrows from "risk regression within family" to "task-boundary
triage."

## Phase 4b exploratory (non-evidential)

- Per-module alignment mean: Q=0.041, K=0.062, V=0.038, O=0.023.
  V+O/Q+K ratio = 0.56 — V/O are *less* aligned than Q/K, opposite to
  the prior Gradience assumption that O is the critical module.
- Layer-depth ratio (same/cross alignment): 1.66 at layer 0, 2.58 at
  layer 31; linear slope +0.031 per layer, r=0.919, p=1e−13.
  Deeper layers separate same-task from cross-task pairs more sharply.
- All 10 N133 composite variants fall short of H1 thresholds
  (max |ρ_partial| = 0.479 for O_deep_mean; max ΔR² = 0.013 for
  erank_ratio, which is also the only variant with the right sign).

## Phase 5: four-method scheduled comparison (exploratory context)

Triage protocol: rank the 45 cross-task pairs by each method's risk
score, retain the 22 lowest-risk pairs, report mean max_degradation
in the retained set. Block-bootstrap (5000 resamples, family-pair
blocked) for CIs. Random baseline (mean over all 45): 3.14%.

| Method | ρ(score, max_deg) | p | Retained deg | Improvement | Bootstrap 95% CI |
|---|---|---|---|---|---|
| Gradience S_H1 | −0.180 | 0.236 | 4.27% | −1.13 pp | [0.84%, 8.61%] |
| KnOTS | +0.183 | 0.230 | **2.18%** | **+0.96 pp** | [−1.18%, 4.75%] |
| TSV | −0.214 | 0.159 | 4.59% | −1.45 pp | [1.09%, 8.68%] |
| SVC | −0.275 | 0.068 | 4.89% | −1.74 pp | [1.34%, 9.07%] |

**Calibration finding.** No off-the-shelf spectral per-pair triage method
posts a statistically significant rank correlation at this N (all
p > 0.05). KnOTS is the only method with positive improvement over
random (+0.96 pp), but its 95% CI crosses zero. The N134 primary null
is not specific to S_H1 — it characterizes the regime (N=45 cross-task
pairs, decoder scale, FAMILY_B explains 88% of variance in
degradation, leaving little residual signal for any spectral
per-pair method to pick up).

Three of four methods (Gradience, TSV, SVC) produce negatively-signed
rank correlations; KnOTS alone has the right sign. This is consistent
with the decoder-scale "family baseline dominates" picture: at this
scale, per-pair spectral quantities contain enough noise that the
sign of their residual correlation with degradation is unstable.

Plumbing note. Phase 5 required writing an on-demand synthesizer
(`LazyPairV21` in `scripts/n134/08_compare_methods.py`) that
reconstructs per-pair v2.1 dicts from the per-adapter `.npz` sidecars
written by 03_spectral_audit. The method math (KnOTS/TSV/SVC
`*_score_from_v21` functions) was unchanged.

## Artifacts

- `analysis_h1.json` — pre-registered H1 decision + bootstrap CIs +
  replications + per-pair table.
- `analysis_secondary.json` — exploratory per-module, depth-trend,
  and composite-evaluation results.
- `method_comparison.json` — four-method scheduled comparison
  (Gradience S_H1 vs KnOTS vs TSV vs SVC).
- `pair_sample.json` — 69-pair deterministic sample (seed=134;
  24 same-task + 45 cross-task).
- `figures/h1_scatter.png`, `h1_bootstrap.png`, `h1_replications.png`
- `figures/secondary_*.png`
- `figures/method_comparison_bar.png`, `method_scatter_grid.png`
- `audit/` — per-pair alignment JSONs (rank-16 SVD summaries).
- `merges/merge_eval_summary.json` — 69-pair merge + eval outputs.
- `merges/retry_ledger.json` — idempotent restart state.

Large binary sidecars (per-adapter `*.npz` with full U/S/V factors)
remain on the pod.
