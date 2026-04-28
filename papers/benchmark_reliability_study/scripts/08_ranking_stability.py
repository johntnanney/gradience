"""08_ranking_stability.py — SPEC_CPU_v0_2 §8.9, §9.4

Compute ranking-stability diagnostics on condition-level data:

    1. Kendall's tau across condition pairs (per benchmark).
    2. Pairwise reversal fraction across conditions (H3 test).
    3. Bootstrap pairwise win probability p(A > B).

Plus a multi-panel figure summarising the per-benchmark distribution of
pairwise win probabilities.

Usage:
    python scripts/08_ranking_stability.py \\
      --condition-level runs/normalized/condition_level_primary.csv \\
      --analysis-config configs/analysis_config.yaml \\
      --out-dir analysis/ranking_stability/

Optional:
    --figures-dir figures/
        Directory for the ranking_stability_by_benchmark.png figure.
        Defaults to <out-dir>/../../figures/ (i.e. paper-root/figures/).

Exit codes per SPEC §11.1:
    0 — success
    1 — generic failure (implementation error)

This script is pure analysis — it does not perform data-contract validation,
so there is no exit code 3 path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

from gradience_study.config import load_config  # noqa: E402


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [08_ranking_stability] {msg}", file=sys.stderr)


# -- Output schemas ----------------------------------------------------------


REVERSAL_FIELDS: list[str] = [
    "benchmark_id",
    "model_a",
    "model_b",
    "overall_mean_diff",
    "condition_reversal_fraction",
    "n_conditions",
    "exceeds_h3_threshold",
]

WIN_PROB_FIELDS: list[str] = [
    "benchmark_id",
    "model_a",
    "model_b",
    "p_a_gt_b_mean",
    "p_a_gt_b_ci_lower",
    "p_a_gt_b_ci_upper",
    "n_bootstrap",
]

KENDALL_FIELDS: list[str] = [
    "benchmark_id",
    "tau_mean",
    "tau_median",
    "tau_ci_lower",
    "tau_ci_upper",
    "n_condition_pairs",
]


# -- Pivot helper ------------------------------------------------------------


def pivot_condition_scores(
    condition_level: pd.DataFrame,
    benchmark_id: str,
) -> pd.DataFrame:
    """Pivot one benchmark's condition rows: rows=cell_id, cols=model_id.

    A "cell" is a (subject, prompt, seed, scoring_rule) tuple — the
    measurement-design cell that all 3 models are evaluated under for
    apples-to-apples ranking comparison. The original implementation
    pivoted on `condition_id`, but our condition_ids include `model_id`
    so each row had accuracy for only one model and the downstream
    `dropna(how="any")` removed every row (D-22 fix; the old behavior
    yielded 0 condition pairs for kendall tau and 0 reversal candidates
    on the canonical run).

    Aggregates duplicate (cell_id, model_id) pairs by mean (defensive;
    in practice cell_id is unique within a (benchmark, model) cell).
    NaN columns/rows are preserved — they may indicate missing
    conditions for some models and are handled downstream by per-pair
    selection of the rows where both models have a score.
    """
    sub = condition_level[condition_level["benchmark_id"] == benchmark_id]
    if sub.empty:
        return pd.DataFrame()
    # Build a cell_id that excludes model_id. Subject_id is optional
    # (only MMLU has subjects); fillna with empty so the join is stable.
    sub = sub.assign(
        cell_id=(
            sub["subject_id"].fillna("").astype(str) + "|"
            + sub["prompt_id"].astype(str) + "|"
            + sub["seed_id"].fillna("0shot").astype(str) + "|"
            + sub["scoring_rule_id"].astype(str)
        )
    )
    pivot = sub.pivot_table(
        index="cell_id",
        columns="model_id",
        values="accuracy",
        aggfunc="mean",
    )
    return pivot


# -- Kendall tau across condition pairs --------------------------------------


def kendall_tau_across_condition_pairs(
    pivot: pd.DataFrame,
) -> tuple[float, float, float, float, int]:
    """Mean / median / 2.5–97.5 percentile of pairwise condition Kendall tau.

    For every pair of conditions c_i, c_j, treat each row as a vector of
    model accuracies and compute Kendall's tau between the two vectors.
    Returns (mean, median, ci_lower, ci_upper, n_pairs). NaN-filled or
    constant rows yield NaN tau and are skipped.
    """
    rows = pivot.dropna(how="any")
    n = len(rows)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    taus: list[float] = []
    arr = rows.to_numpy()
    for i, j in combinations(range(n), 2):
        a = arr[i]
        b = arr[j]
        # Constant vectors give Kendall tau = NaN; skip those.
        if np.all(a == a[0]) or np.all(b == b[0]):
            continue
        tau, _ = scipy_stats.kendalltau(a, b)
        if np.isnan(tau):
            continue
        taus.append(float(tau))

    if not taus:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    arr_t = np.asarray(taus, dtype=float)
    return (
        float(np.mean(arr_t)),
        float(np.median(arr_t)),
        float(np.percentile(arr_t, 2.5)),
        float(np.percentile(arr_t, 97.5)),
        int(len(arr_t)),
    )


# -- Pairwise reversal fraction (§9.4) ---------------------------------------


def pairwise_reversal_fraction(
    pivot: pd.DataFrame,
    h3_threshold: float,
) -> list[dict[str, Any]]:
    """Compute reversal fraction for every model pair within a benchmark."""
    rows: list[dict[str, Any]] = []
    models = list(pivot.columns)
    for m_a, m_b in combinations(models, 2):
        cell = pivot[[m_a, m_b]].dropna(how="any")
        if cell.empty:
            continue
        a = cell[m_a].to_numpy(dtype=float)
        b = cell[m_b].to_numpy(dtype=float)
        n_conds = int(len(cell))
        diff_overall = float(a.mean() - b.mean())
        sign_overall = np.sign(diff_overall)
        per_cond_diff = a - b
        sign_per_cond = np.sign(per_cond_diff)
        # If the overall mean diff is exactly zero, every condition where the
        # within-condition diff is non-zero is a "reversal" relative to a
        # neutral overall ranking. We still report the count.
        n_reversals = int(np.sum(sign_per_cond != sign_overall))
        reversal_fraction = float(n_reversals) / float(n_conds) if n_conds else 0.0

        rows.append(
            {
                "model_a": m_a,
                "model_b": m_b,
                "overall_mean_diff": diff_overall,
                "condition_reversal_fraction": reversal_fraction,
                "n_conditions": n_conds,
                "exceeds_h3_threshold": bool(reversal_fraction > h3_threshold),
            }
        )
    return rows


# -- Bootstrap pairwise win probability --------------------------------------


def bootstrap_pairwise_win_prob(
    pivot: pd.DataFrame,
    n_resamples: int,
    seed: int,
    ci_lower_pct: float,
    ci_upper_pct: float,
) -> list[dict[str, Any]]:
    """For every model pair, bootstrap p(A > B) over conditions resampled
    with replacement. Returns a list of dict rows.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    models = list(pivot.columns)

    for m_a, m_b in combinations(models, 2):
        cell = pivot[[m_a, m_b]].dropna(how="any")
        if cell.empty:
            continue
        a = cell[m_a].to_numpy(dtype=float)
        b = cell[m_b].to_numpy(dtype=float)
        n = len(a)

        boot_probs = np.empty(n_resamples, dtype=float)
        for k in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            boot_probs[k] = float(np.mean(a[idx] > b[idx]))

        rows.append(
            {
                "model_a": m_a,
                "model_b": m_b,
                "p_a_gt_b_mean": float(np.mean(boot_probs)),
                "p_a_gt_b_ci_lower": float(np.percentile(boot_probs, ci_lower_pct)),
                "p_a_gt_b_ci_upper": float(np.percentile(boot_probs, ci_upper_pct)),
                "n_bootstrap": int(n_resamples),
            }
        )
    return rows


# -- CSV writers -------------------------------------------------------------


def _format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return repr(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)
        for r in rows:
            writer.writerow([_format_value(r.get(c)) for c in fields])


# -- Figure ------------------------------------------------------------------


def make_ranking_stability_figure(
    win_prob_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Multi-panel figure: per-benchmark distribution of pairwise p(A > B)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not win_prob_rows:
        # Emit an empty placeholder figure so downstream consumers don't 404.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No ranking-stability data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    df = pd.DataFrame(win_prob_rows)
    benchmarks = sorted(df["benchmark_id"].unique())
    n = len(benchmarks)
    ncols = min(3, max(1, n))
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    sns.set_style("whitegrid")

    for idx, b in enumerate(benchmarks):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = df[df["benchmark_id"] == b]
        ax.hist(sub["p_a_gt_b_mean"], bins=12, color="steelblue", edgecolor="black", alpha=0.85)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="p=0.5")
        ax.set_title(b, fontsize=10)
        ax.set_xlabel("bootstrap p(A > B)")
        ax.set_ylabel("# model pairs")
        ax.set_xlim(0.0, 1.0)
        ax.legend(fontsize=8, loc="upper right")

    # Hide unused axes.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_axis_off()

    fig.suptitle("Pairwise win probability across conditions, by benchmark", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -- Main pipeline -----------------------------------------------------------


def run(
    condition_level_path: Path,
    analysis_config_path: Path,
    out_dir: Path,
    figures_dir: Path | None,
) -> int:
    if not condition_level_path.exists():
        _log("ERROR", f"Condition-level CSV not found: {condition_level_path}")
        return 1

    # Load analysis config. The config-loader expects a study_config.yaml
    # path so its include resolution works. We support two modes here:
    #   (a) --analysis-config points at the study_config.yaml — preferred.
    #   (b) --analysis-config points at the analysis_config.yaml directly —
    #       we read it as raw YAML and build only the bits we need.
    bootstrap_n: int
    bootstrap_seed: int
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    h3_threshold: float
    try:
        merged = load_config(analysis_config_path)
        bootstrap_n = merged.analysis_config.bootstrap.n_resamples
        bootstrap_seed = merged.analysis_config.bootstrap.random_seed
        bootstrap_ci_lower = merged.analysis_config.bootstrap.ci_lower_percentile
        bootstrap_ci_upper = merged.analysis_config.bootstrap.ci_upper_percentile
        h3_threshold = merged.analysis_config.h3_ranking_reversal_threshold
    except Exception:  # noqa: BLE001
        # Fallback: treat the path as a bare analysis_config.yaml.
        import yaml

        with open(analysis_config_path, "r", encoding="utf-8") as f:
            ac = yaml.safe_load(f)
        boot = ac["bootstrap"]
        bootstrap_n = int(boot["n_resamples"])
        bootstrap_seed = int(boot["random_seed"])
        bootstrap_ci_lower = float(boot["ci_lower_percentile"])
        bootstrap_ci_upper = float(boot["ci_upper_percentile"])
        h3_threshold = float(ac.get("h3_ranking_reversal_threshold", 0.10))

    df = pd.read_csv(condition_level_path)
    _log(
        "INFO",
        f"Loaded {len(df)} condition rows from {condition_level_path} "
        f"({df['benchmark_id'].nunique() if not df.empty else 0} benchmarks).",
    )

    benchmarks = sorted(df["benchmark_id"].dropna().unique())

    reversal_rows: list[dict[str, Any]] = []
    win_prob_rows: list[dict[str, Any]] = []
    kendall_rows: list[dict[str, Any]] = []

    for benchmark_id in benchmarks:
        pivot = pivot_condition_scores(df, benchmark_id)
        if pivot.empty or pivot.shape[1] < 2:
            _log(
                "WARNING",
                f"Benchmark {benchmark_id!r}: fewer than 2 models with data; skipping.",
            )
            continue

        # 1. Kendall tau across condition pairs.
        tau_mean, tau_median, tau_lo, tau_hi, n_pairs = (
            kendall_tau_across_condition_pairs(pivot)
        )
        kendall_rows.append(
            {
                "benchmark_id": benchmark_id,
                "tau_mean": tau_mean,
                "tau_median": tau_median,
                "tau_ci_lower": tau_lo,
                "tau_ci_upper": tau_hi,
                "n_condition_pairs": n_pairs,
            }
        )

        # 2. Pairwise reversal fraction.
        rev_rows = pairwise_reversal_fraction(pivot, h3_threshold)
        for r in rev_rows:
            r["benchmark_id"] = benchmark_id
            reversal_rows.append(r)

        # 3. Bootstrap pairwise win probability.
        wp_rows = bootstrap_pairwise_win_prob(
            pivot,
            n_resamples=bootstrap_n,
            seed=bootstrap_seed,
            ci_lower_pct=bootstrap_ci_lower,
            ci_upper_pct=bootstrap_ci_upper,
        )
        for r in wp_rows:
            r["benchmark_id"] = benchmark_id
            win_prob_rows.append(r)

    # Write CSVs.
    write_csv(out_dir / "ranking_reversals.csv", reversal_rows, REVERSAL_FIELDS)
    write_csv(
        out_dir / "pairwise_win_probabilities.csv", win_prob_rows, WIN_PROB_FIELDS
    )
    write_csv(out_dir / "kendall_tau_by_benchmark.csv", kendall_rows, KENDALL_FIELDS)

    _log(
        "INFO",
        f"Wrote ranking_reversals.csv ({len(reversal_rows)} rows), "
        f"pairwise_win_probabilities.csv ({len(win_prob_rows)} rows), "
        f"kendall_tau_by_benchmark.csv ({len(kendall_rows)} rows) "
        f"to {out_dir}",
    )

    # H3 test result (pre-reg §3.3): for at least N benchmarks, at least
    # one model pair has a condition-reversal fraction exceeding the H3
    # threshold. Aggregating from the per-pair reversal_rows.
    benchmarks_with_exceeding_pair = sorted({
        r["benchmark_id"] for r in reversal_rows
        if r.get("exceeds_h3_threshold")
    })
    benchmarks_with_no_exceeding_pair = sorted(
        set(benchmarks) - set(benchmarks_with_exceeding_pair)
    )
    n_required_for_h3 = 3  # pre-reg §3.3: "at least three of the five..."
    h3_result = {
        "h3_hypothesis": (
            "for at least N primary benchmarks, at least one model pair "
            "has a condition-reversal fraction exceeding the H3 threshold "
            "(indicating that single-occasion ranking is unstable)"
        ),
        "threshold": float(h3_threshold),
        "n_required": n_required_for_h3,
        "n_benchmarks_with_exceeding_pair": len(benchmarks_with_exceeding_pair),
        "benchmarks_with_exceeding_pair": benchmarks_with_exceeding_pair,
        "benchmarks_with_no_exceeding_pair": benchmarks_with_no_exceeding_pair,
        "h3_confirmed": (
            len(benchmarks_with_exceeding_pair) >= n_required_for_h3
        ),
    }
    h3_path = out_dir / "h3_test.json"
    h3_path.write_text(json.dumps(h3_result, indent=2, sort_keys=True) + "\n")
    _log(
        "INFO",
        f"H3: {h3_result['n_benchmarks_with_exceeding_pair']}/"
        f"{h3_result['n_required']} benchmarks have at least one pair "
        f"exceeding {h3_threshold}; confirmed={h3_result['h3_confirmed']}",
    )

    # Figure.
    if figures_dir is None:
        figures_dir = out_dir.parent.parent / "figures"
    figure_path = figures_dir / "ranking_stability_by_benchmark.png"
    make_ranking_stability_figure(win_prob_rows, figure_path)
    _log("INFO", f"Wrote figure to {figure_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ranking-stability analysis (SPEC_CPU_v0_2 §8.9, §9.4)."
    )
    parser.add_argument(
        "--condition-level",
        required=True,
        type=Path,
        help="Path to condition-level CSV from 05_make_condition_scores.py.",
    )
    parser.add_argument(
        "--analysis-config",
        required=True,
        type=Path,
        help=(
            "Path to study_config.yaml (preferred — full config) or to a "
            "bare analysis_config.yaml. Bootstrap and H3 threshold are read "
            "from the analysis_config block."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for analysis CSVs (created if missing).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for the PNG figure. Defaults to <out-dir>/../../figures.",
    )
    args = parser.parse_args()

    try:
        return run(
            args.condition_level,
            args.analysis_config,
            args.out_dir,
            args.figures_dir,
        )
    except Exception as e:  # noqa: BLE001
        _log("CRITICAL", f"Unhandled exception: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
