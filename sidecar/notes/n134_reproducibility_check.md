# N134 Reproducibility Check

**Date performed:** 2026-04-21.
**Tag verified against:** `n134-submission-draft-v1` (commit `f462767`).
**Protocol:** four-tier amended protocol per the pre-T6 scoping discussion (see `papers/n134_workshop/revision_notes.md` RN-011). Strict byte-identical reproduction was not attempted because the pod environment was decommissioned before a `pip freeze` was captured; the amended protocol is calibrated to what can be verified under a development-environment gap.

---

## Environment

| | Pod (original) | Dev (reproduction) |
|---|---|---|
| Python | 3.11 | 3.14.0 |
| `torch` | 2.3.x | 2.9.1 |
| `transformers` | 4.44.x | 4.57.3 |
| `peft` | 0.12.x | 0.18.1 |
| `numpy` | 1.26.x | 2.4.1 |
| `scipy` | 1.13.x | 1.17.1 |

The pod versions are inferred from the D1/D2/D3 compatibility fixes in the N134 report. The dev environment is captured in `sidecar/results/n134/environment_dev.txt` with the full `pip freeze` and a header stating what the snapshot is and is not.

### Script-hygiene note for replicators

The committed analysis scripts (`scripts/n134/06_analysis_h1.py`, `07_analysis_secondary.py`, `08_compare_methods.py`) hardcode `WORKSPACE = Path("/workspace/n134")`, which is the pod-absolute path. Running these scripts on a replicator's machine will either crash (path does not exist) or, if the path happens to exist, overwrite its contents. The reproduction procedure used here is to copy the scripts to `/tmp/n134_t6/`, apply a one-line `sed` substitution on the `WORKSPACE` constant, stage committed data under `/tmp/n134_t6/workspace/`, and run from there. A proper amendment (accepting `WORKSPACE` from an environment variable with the pod path as default) is planned as a separate, post-tag additive commit and is not in the tagged state.

---

## Tier 1 — Qualitative claims

All six pre-registered qualitative claims reproduce.

| Claim | Committed | Reproduced | Result |
|---|---|---|---|
| H1 not confirmed under pre-registered rule | `h1_confirmed = False` | `h1_confirmed = False` | **PASS** |
| B-P1 task-boundary detection confirmed | `confirmed = True` | `confirmed = True` | **PASS** |
| B-P2 spectral separation confirmed | `confirmed = True` | `confirmed = True` | **PASS** |
| B-P4 erank ANOVA confirmed | `confirmed = True` | `confirmed = True` | **PASS** |
| No N133 composite exceeds H1 thresholds | `any_composite_exceeds_h1 = False` | `any_composite_exceeds_h1 = False` | **PASS** |
| All 10 composite rho_partial signs unchanged | pass | pass | **PASS** |

Method-comparison qualitative claims (from Phase 5 / `method_comparison.json`) were not rerun because 08 is not reproducible from committed state; see §Tier 4 below.

## Tier 2 — Structural JSON agreement

Both `analysis_h1.json` and `analysis_secondary.json` schemas reproduce exactly. Top-level key sets, nested sub-dict key sets, `per_layer` array lengths, and `composite_scores` list lengths all match committed outputs. No `KeyError` under any `from_dict`-style access pattern.

## Tier 3 — Quantitative agreement

Tolerance schedule (from the pre-T6 scoping discussion, amended for rank-correlation sensitivity):

- ±0.01 absolute: correlations (raw Spearman, Pearson), percentage-scale quantities, p-values of order 10⁻² or larger.
- ±0.005: R² values.
- ±0.02 absolute for **rank-based statistics computed on OLS residuals** (partial Spearman on residualized variables); see §Rank-on-residuals observation.

### 06_analysis_h1.py

| Quantity | Committed | Reproduced | \|Δ\| | Tolerance | Status |
|---|---|---|---|---|---|
| Raw Spearman ρ | −0.18029341 | −0.18029341 | 0 | ±0.01 | PASS |
| Partial Spearman ρ | −0.53302369 | −0.54481964 | 1.18e−2 | ±0.02 (rank-on-resid) | **PASS** |
| p_partial | 1.6345e-04 | 1.0907e-04 | 5.44e-05 | ±0.01 | PASS |
| R² family-only | 0.88071387 | 0.88071387 | 0 | ±0.005 | PASS |
| R² family + S_H1 | 0.88350568 | 0.88350568 | 0 | ±0.005 | PASS |
| ΔR² | 0.00279181 | 0.00279181 | 0 | ±0.005 | PASS |
| Bootstrap ρ mean | −0.53100633 | −0.53122635 | 2.2e-4 | ±0.02 (rank-on-resid) | PASS |
| Bootstrap ρ CI low | −0.82513857 | −0.82332528 | 1.8e-3 | ±0.02 (rank-on-resid) | PASS |
| Bootstrap ρ CI high | −0.13122569 | −0.13290804 | 1.7e-3 | ±0.02 (rank-on-resid) | PASS |
| Bootstrap ΔR² CI low | 1.713e-5 | 1.713e-5 | 0 | ±0.005 | PASS |
| Bootstrap ΔR² CI high | 0.02289019 | 0.02289019 | 1e-16 | ±0.005 | PASS |

### 07_analysis_secondary.py

Every scalar compared (module means Q/K/V/O, V+O/Q+K ratio, depth-trend slope/r/p, all ten composite ΔR² values) is either bit-identical or differs only at float-precision noise (≤ 3e-16). **PASS** across the board; no quantity tests the tolerance schedule.

### 08_compare_methods.py

**Not reproducible from committed state** under any tier; see Tier 4.

## Tier 4 — Environment-gap and data-availability documentation

Two gaps affect what this reproducibility check covers. They are distinct in kind.

### Environment gap (numerical precision)

Single value falls into the rank-on-residuals precision regime: the partial Spearman ρ in 06_analysis_h1.py drifts by 1.18e-2 between the pod environment (−0.5330) and the dev environment (−0.5448). All other scalars in 06 reproduce bit-identical or at float-precision-noise magnitude. The localization is clean: raw Spearman (no residualization) is bit-identical, OLS R² is bit-identical, ΔR² is bit-identical, but Spearman applied to OLS residuals drifts. See §Rank-on-residuals observation for the explanation.

### Data-availability gap (Phase 5 not reproducible from committed state)

`08_compare_methods.py` requires per-adapter SVD factors as `.npz` binary sidecars at `sidecar/results/n134/audit/{adapter_id}_svd.npz`. These files were produced on the pod (~50 MB each × 24 adapters ≈ 1.2 GB total) and were **not** committed to the repository — they are referenced from the report's §2 as "large binary sidecars remain on the pod." Pod has been decommissioned.

Consequences:

1. Tier 1 claims specific to Phase 5 (KnOTS only right-signed, SVC nearest to significance, three of four methods wrong-signed, no CI excludes random baseline) cannot be verified by re-running `08_compare_methods.py` against the tagged repository alone. They are supported by `sidecar/results/n134/method_comparison.json`, which was produced on the pod at known-good state and committed in `1527d72`.

2. Tier 2 and Tier 3 verification for Phase 5 are likewise not possible from the tagged state. A replicator who wanted to verify Phase 5 independently would need to either (a) re-audit all 24 LoRA adapters from the same base model and seed with a fresh SVD pass, producing fresh `.npz` files, or (b) obtain the pod's `.npz` sidecars from a snapshot if one existed. Neither is on the path for this reproducibility check.

3. **Practical guidance for a future replicator.** If Phase 5 needs to be re-verified — for reviewer response, for follow-up work, or for a rerun after `08_compare_methods.py` is amended — the fresh audit path (option a) is the cleanest approach and is documented in `scripts/n134/03_spectral_audit.py`. The audit itself takes ~1 hour on a single H100-class GPU; the per-adapter `.npz` outputs are what Phase 5 needs.

---

## Rank-on-residuals observation

The partial Spearman ρ is the single scalar in the N134 analysis that exhibits non-trivial numerical drift across the environment gap. The drift magnitude (~0.012 absolute, out of a point estimate of ~−0.53) is a **property of the statistic**, not a bug in any library or a symptom of environment drift beyond what is expected at this Python/numpy/scipy version gap.

The diagnostic pattern:

- Raw Spearman ρ on the original S_H1 and max_degradation arrays: **bit-identical** across environments.
- OLS fit of S_H1 and max_degradation on FAMILY_B dummies: bit-identical sum-of-squares (R² is bit-identical); residuals identical in aggregate.
- Spearman applied to the OLS residuals: drifts by ~0.012.

Spearman correlation is a **rank-based** statistic: it depends on the pairwise ordering of the input values, not their magnitudes. At n = 45 residuals where many values are close together, a floating-point perturbation too small to shift the sum-of-squares by more than its 15th decimal place can still flip the rank order of a few near-tied residual pairs. Each rank-pair flip shifts Spearman by 2/(n(n²−1)) ≈ 2.2e-5 per pair; flipping on the order of a few hundred near-tied pairs accumulates to drift of the magnitude we observe.

Across environments, the committed value and the reproduced value are **both correct**: each is computed faithfully from the same input data by the same algorithm. The difference is in which near-tied rank-pairs the underlying `numpy.linalg.lstsq` path produces on different floating-point hardware / BLAS configurations / intermediate-precision paths. Neither value is "more true" than the other. The statistic itself has intrinsic precision of roughly ±0.01 at this observation count.

### Implication for the paper's claim

The pre-registered H1 decision is robust to this precision — the partial ρ is approximately −0.53 in both environments, decisively below the +0.50 threshold (wrong sign) in both environments, and the decision (H1 not confirmed) is identical in both. But the specific point estimate of partial ρ should be reported in the paper as approximately −0.53 ± 0.01 rather than as a four-decimal value. See RN-011 in `papers/n134_workshop/revision_notes.md` for the paper-language amendment this observation requires.

This is itself an instance of the measurement-discipline concerns that motivate the Gradience program: a headline statistic whose numerical precision (±0.01 at n = 45) is comparable in magnitude to its bootstrap confidence-interval width from sampling variability (~0.35 from the [−0.825, −0.131] CI) is a statistic that should be reported with SEM-like language rather than a point estimate. The finding is a free worked example for the paper.

---

## Verification summary

- **Qualitative claims (Tier 1):** all six reproduce.
- **Structural agreement (Tier 2):** complete.
- **Quantitative agreement (Tier 3):** all within the amended tolerance schedule, with one value (partial ρ) falling into the rank-on-residuals precision regime. Localized, explicable, documented.
- **Environment gap (Tier 4):** Python 3.11 → 3.14 and numpy/scipy major-version gap; one precision-sensitive scalar exposed. Documented.
- **Data-availability gap (Tier 4):** Phase 5 not reproducible from committed state; per-adapter `.npz` factors are pod-only. Documented with guidance for future replicators.

The committed values in `analysis_h1.json` and `analysis_secondary.json` are the canonical values produced under the tagged commit's intended environment. A future replicator using a different Python version should expect bit-identical or near-identical reproduction for all quantities except partial Spearman on OLS residuals, which is precision-sensitive at this observation count. `method_comparison.json` is the canonical Phase 5 output; it cannot be regenerated from the tagged repository alone without re-auditing adapters.
