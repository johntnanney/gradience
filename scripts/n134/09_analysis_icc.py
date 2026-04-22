"""N134 cross-seed ICC for S_H1 as an instrument.

Closes the [TODO] in app:reliability of papers/n134_workshop/draft_v2_thesis_b.tex
per the spec at sidecar/notes/n134_icc_spec.md.

Design choices (per spec §3):
  - ICC(2,1), absolute agreement, single measurement.
  - Shrout-Fleiss F-distribution CI as primary; block-bootstrap over tasks
    (5000 resamples, seed 134) as secondary.
  - SEM = SD_pooled * sqrt(1 - ICC), reported alongside.
  - Instrument output: raw compute_s_h1(pair) imported from 06_analysis_h1.py.
    No re-implementation of the O-module depth-weighting.

Shrout-Fleiss ICC(2,1) and its CI, on a complete n_tasks x n_raters panel:

  Let MS_R = between-tasks MS (rows)
      MS_C = between-raters MS (columns)
      MS_E = residual MS
      n    = n_tasks (rows)
      k    = n_raters (columns) — here, within-task seed-pairs

  ICC(2,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E + (k/n)*(MS_C - MS_E))

  F-distribution CI (absolute-agreement, single-measurement):
    Let F_L = F* / F_{1-alpha/2, df1=n-1, df2=v}
        F_U = F* * F_{1-alpha/2, df1=v, df2=n-1}
    where F* = MS_R / MS_E
          v  = Satterthwaite df on MS_C + MS_E combination
                = [k*ICC*MS_C + (n(1+(k-1)*ICC) - k*ICC)*MS_E]^2
                  / ((k*ICC*MS_C)^2/(k-1) + ((n(1+(k-1)*ICC)-k*ICC)*MS_E)^2/((n-1)*(k-1)))
  (Shrout & Fleiss 1979, eq. 6; McGraw & Wong 1996 has the absolute-agreement
  variant we use here.)

Usage:
  python scripts/n134/09_analysis_icc.py

Override workspace via N134_WORKSPACE env var (same pattern as 06/07/08).

Output:
  {WORKSPACE}/analysis_icc.json   (schema per spec §5)
  stdout                          (one-paragraph prose summary for paper)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Import compute_s_h1 from the H1 analysis script.  Adding scripts/n134/ to
# sys.path lets us use the 06 module without repackaging it.  Filename
# starts with a digit so we import via importlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util as _ilu  # noqa: E402
_spec_06 = _ilu.spec_from_file_location(
    "n134_06_analysis_h1",
    str(Path(__file__).resolve().parent / "06_analysis_h1.py"),
)
_m06 = _ilu.module_from_spec(_spec_06)
_spec_06.loader.exec_module(_m06)
compute_s_h1 = _m06.compute_s_h1


# ---------------------------------------------------------------------------
# Paths (env-var override, same pattern as 06/07/08)
# ---------------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("N134_WORKSPACE", "/workspace/n134"))
AUDIT_DIR = WORKSPACE / "audit"


# ---------------------------------------------------------------------------
# Panel ordering (per spec §2)
# ---------------------------------------------------------------------------

TASK_ORDER = [
    "arc_challenge", "boolq", "commonsenseqa", "hellaswag",
    "openbookqa", "piqa", "siqa", "winogrande",
]
SEED_PAIR_ORDER = [(42, 123), (42, 456), (123, 456)]
N_TASKS = len(TASK_ORDER)           # 8
N_RATERS = len(SEED_PAIR_ORDER)     # 3
N_TOTAL = N_TASKS * N_RATERS        # 24


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def _pair_key(task: str, seed_a: int, seed_b: int) -> tuple[str, str]:
    """Return (committed-order key, alphabetical key) to handle either."""
    committed = f"{task}_s{seed_a}_vs_{task}_s{seed_b}"
    reversed_ = f"{task}_s{seed_b}_vs_{task}_s{seed_a}"
    return committed, reversed_


def build_panel(pair_full: dict) -> np.ndarray:
    """8x3 matrix of S_H1 values, row=task, col=seed_pair (per SEED_PAIR_ORDER)."""
    panel = np.full((N_TASKS, N_RATERS), np.nan, dtype=np.float64)
    for i, task in enumerate(TASK_ORDER):
        for j, (sa, sb) in enumerate(SEED_PAIR_ORDER):
            k1, k2 = _pair_key(task, sa, sb)
            pair = pair_full.get(k1) or pair_full.get(k2)
            if pair is None:
                raise KeyError(
                    f"Missing same-task pair for {task} seeds ({sa},{sb}); "
                    f"tried keys {k1!r} and {k2!r}"
                )
            if not pair.get("is_same_task"):
                raise ValueError(
                    f"Pair {k1} found but is_same_task is not true."
                )
            panel[i, j] = compute_s_h1(pair)
    return panel


# ---------------------------------------------------------------------------
# Shrout-Fleiss ICC(2,1) with absolute-agreement F-distribution CI
# ---------------------------------------------------------------------------

def shrout_fleiss_icc21(panel: np.ndarray, alpha: float = 0.05) -> dict:
    """Compute ICC(2,1) absolute-agreement, single-measurement, with F-CI.

    Panel: (n_tasks, n_raters) complete float matrix.
    """
    n, k = panel.shape
    grand_mean = panel.mean()
    row_means = panel.mean(axis=1)
    col_means = panel.mean(axis=0)

    # Sum of squares
    ss_total = ((panel - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols  # residual (two-way additive model)

    # Mean squares
    ms_rows = ss_rows / (n - 1)            # MS_R (between tasks)
    ms_cols = ss_cols / (k - 1)            # MS_C (between seed-pairs)
    ms_err = ss_err / ((n - 1) * (k - 1))  # MS_E (residual)

    # ICC(2,1), absolute agreement, single measurement
    icc = (ms_rows - ms_err) / (
        ms_rows + (k - 1) * ms_err + (k / n) * (ms_cols - ms_err)
    )

    # Shrout-Fleiss F-distribution CI for ICC(2,1) absolute agreement.
    # See Shrout & Fleiss 1979 eq. 6; McGraw & Wong 1996 Table 7 (ICC(A,1)).
    # The CI uses a Satterthwaite-approximate F ratio on the combination of
    # MS_C and MS_E.
    a = (k * icc) / (n * (1 - icc))
    b = 1 + (k * icc * (n - 1)) / (n * (1 - icc))
    v_num = (a * ms_cols + b * ms_err) ** 2
    v_den = (
        (a * ms_cols) ** 2 / (k - 1)
        + (b * ms_err) ** 2 / ((n - 1) * (k - 1))
    )
    v = v_num / v_den  # Satterthwaite df
    f_star = ms_rows / ms_err

    f_crit_upper = stats.f.ppf(1 - alpha / 2, n - 1, v)
    f_crit_lower = stats.f.ppf(1 - alpha / 2, v, n - 1)
    f_lower = f_star / f_crit_upper
    f_upper = f_star * f_crit_lower

    ci_low = (f_lower - 1) / (
        f_lower + (k - 1) + (k / n) * (ms_cols / ms_err - 1)
    )
    ci_high = (f_upper - 1) / (
        f_upper + (k - 1) + (k / n) * (ms_cols / ms_err - 1)
    )

    # SD_pooled over all 24 cells, then SEM = SD * sqrt(1 - ICC)
    sd_pooled = float(panel.std(ddof=1))
    sem = sd_pooled * float(np.sqrt(max(0.0, 1.0 - icc)))

    return {
        "icc": float(icc),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ms_rows": float(ms_rows),
        "ms_cols": float(ms_cols),
        "ms_err": float(ms_err),
        "sd_pooled": sd_pooled,
        "sem": float(sem),
        "f_star": float(f_star),
        "df_satterthwaite": float(v),
    }


# ---------------------------------------------------------------------------
# Block bootstrap over tasks
# ---------------------------------------------------------------------------

def bootstrap_icc(
    panel: np.ndarray,
    n_resamples: int = 5000,
    seed: int = 134,
    alpha: float = 0.05,
) -> dict:
    """Block-bootstrap CI for ICC(2,1): resample tasks (rows) with replacement."""
    rng = np.random.default_rng(seed)
    n, _ = panel.shape
    iccs = []
    failures = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled = panel[idx]
        try:
            result = shrout_fleiss_icc21(resampled, alpha=alpha)
            iccs.append(result["icc"])
        except (ZeroDivisionError, ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
    iccs = np.array(iccs, dtype=np.float64)
    return {
        "ci_low": float(np.percentile(iccs, 100 * alpha / 2)),
        "ci_high": float(np.percentile(iccs, 100 * (1 - alpha / 2))),
        "n_valid": len(iccs),
        "n_failures": failures,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 74)
    print("  N134 cross-seed ICC for S_H1 (per sidecar/notes/n134_icc_spec.md)")
    print("=" * 74)

    pair_full_path = AUDIT_DIR / "pair_alignment_full.json"
    pair_full = json.loads(pair_full_path.read_text())

    # Sanity: how many same-task pairs?
    same_task = {k: v for k, v in pair_full.items() if v.get("is_same_task")}
    print(f"\n  Same-task pairs in audit: {len(same_task)} (expected {N_TOTAL})")
    assert len(same_task) == N_TOTAL, (
        f"Panel incomplete: got {len(same_task)} same-task pairs, expected {N_TOTAL}"
    )

    panel = build_panel(pair_full)
    assert not np.isnan(panel).any(), "Panel has NaN cells after build"

    print(f"\n  Panel ({N_TASKS} tasks x {N_RATERS} seed-pairs):")
    print(f"  {'task':<16s}  " + "  ".join(
        f"({a},{b})".rjust(10) for a, b in SEED_PAIR_ORDER
    ))
    for i, task in enumerate(TASK_ORDER):
        row = "  ".join(f"{v:>10.6f}" for v in panel[i])
        print(f"  {task:<16s}  {row}")

    print(f"\n  Panel grand mean: {panel.mean():.6f}")
    print(f"  Panel SD (pooled): {panel.std(ddof=1):.6f}")
    print(f"  Panel range: [{panel.min():.6f}, {panel.max():.6f}]")

    # ---- Primary: Shrout-Fleiss F-distribution CI ----
    print("\n" + "-" * 74)
    print("  ICC(2,1) — Shrout-Fleiss F-distribution CI (primary)")
    print("-" * 74)
    primary = shrout_fleiss_icc21(panel, alpha=0.05)
    print(f"  ICC(2,1):   {primary['icc']:+.4f}")
    print(f"  95% CI:     [{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}]")
    print(f"  F*:         {primary['f_star']:.4f}")
    print(f"  Satterthwaite df: {primary['df_satterthwaite']:.2f}")
    print(f"  MS_rows:    {primary['ms_rows']:.6e}")
    print(f"  MS_cols:    {primary['ms_cols']:.6e}")
    print(f"  MS_err:     {primary['ms_err']:.6e}")
    print(f"  SD_pooled:  {primary['sd_pooled']:.6f}")
    print(f"  SEM:        {primary['sem']:.6f}")

    # ---- Secondary: block bootstrap ----
    print("\n" + "-" * 74)
    print("  Block-bootstrap over tasks (secondary, 5000 resamples, seed=134)")
    print("-" * 74)
    boot = bootstrap_icc(panel, n_resamples=5000, seed=134, alpha=0.05)
    print(f"  95% CI:     [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
    print(f"  Valid samples: {boot['n_valid']}/5000 (failures: {boot['n_failures']})")

    # ---- Comparison rule (spec §3.2) ----
    lower_disagreement = abs(primary['ci_low'] - boot['ci_low'])
    upper_disagreement = abs(primary['ci_high'] - boot['ci_high'])
    max_disagreement = max(lower_disagreement, upper_disagreement)
    ci_agreement = "within_0.05" if max_disagreement <= 0.05 else "divergent"
    print(f"\n  CI agreement: {ci_agreement} "
          f"(max bound difference {max_disagreement:.4f})")

    # ---- Sanity checks (spec §6) ----
    sanity = {
        "panel_complete": int(not np.isnan(panel).any()) == 1,
        "icc_in_range": 0.0 <= primary['icc'] <= 1.0,
        "sem_less_than_sd_pooled": primary['sem'] < primary['sd_pooled'],
        "ci_agreement": ci_agreement,
    }
    # The magnitude-expectation check from §6 is advisory not gating:
    if primary['icc'] < 0.5:
        print(f"\n  [ADVISORY] ICC < 0.5. Spec §6 flags this as unexpected; "
              f"examine panel before paper-reporting.")

    print("\n  Sanity:")
    for k, v in sanity.items():
        print(f"    {k}: {v}")

    # ---- Emit JSON (schema per spec §5) ----
    output = {
        "instrument": "S_H1",
        "instrument_source":
            "scripts/n134/06_analysis_h1.py:compute_s_h1",
        "spec": "sidecar/notes/n134_icc_spec.md",
        "design": {
            "icc_form": "ICC(2,1)",
            "agreement_type": "absolute",
            "measurement_type": "single",
            "n_tasks": N_TASKS,
            "n_seed_pairs_per_task": N_RATERS,
            "n_total_observations": N_TOTAL,
            "task_ordering": TASK_ORDER,
            "seed_pair_ordering": [list(p) for p in SEED_PAIR_ORDER],
        },
        "results": {
            "icc": primary["icc"],
            "ci_shrout_fleiss_95": [primary["ci_low"], primary["ci_high"]],
            "ci_bootstrap_95": [boot["ci_low"], boot["ci_high"]],
            "bootstrap_n_resamples": 5000,
            "bootstrap_seed": 134,
            "bootstrap_n_valid": boot["n_valid"],
            "sem": primary["sem"],
            "sd_pooled": primary["sd_pooled"],
            "ms_between_tasks": primary["ms_rows"],
            "ms_between_seed_pairs": primary["ms_cols"],
            "ms_residual": primary["ms_err"],
            "f_star": primary["f_star"],
            "df_satterthwaite": primary["df_satterthwaite"],
        },
        "panel": {
            task: {
                f"seed_pair_{a}_{b}": float(panel[i, j])
                for j, (a, b) in enumerate(SEED_PAIR_ORDER)
            }
            for i, task in enumerate(TASK_ORDER)
        },
        "sanity_checks": sanity,
    }
    out_path = WORKSPACE / "analysis_icc.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved {out_path}")

    # ---- Prose summary (spec §8, for copy-paste) ----
    def _desc(icc_val: float) -> str:
        if icc_val > 0.9:
            return "excellent"
        if icc_val > 0.75:
            return "good"
        if icc_val > 0.50:
            return "moderate"
        return "poor"
    conv = _desc(primary['icc'])

    print("\n" + "=" * 74)
    print("  PROSE SUMMARY (for paper copy-paste; see spec §8 templates)")
    print("=" * 74)
    print(
        f"\n  Cross-seed ICC(2,1) for S_H1 (24 same-task pairs; absolute\n"
        f"  agreement, single measurement): ICC = {primary['icc']:.3f},\n"
        f"  95%% CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}] "
        f"(Shrout-Fleiss).\n"
        f"  Block-bootstrap over tasks 95%% CI "
        f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}];\n"
        f"  {ci_agreement}. SEM = {primary['sem']:.4f}.\n"
        f"  Conventional descriptor: {conv}.\n"
    )


if __name__ == "__main__":
    main()
