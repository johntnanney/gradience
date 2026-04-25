"""09_mmlu_subject_decomp.py — SPEC_CPU_v0_2 §8.10

Decompose MMLU-panel item-level variance by subject:

    1. Model x subject accuracy matrix (rows=models, cols=subjects),
       averaged over all conditions (prompts, seeds, scoring rules) with
       row/column marginals.
    2. Variance components at the item level: model main, subject main,
       prompt main, model x subject interaction, prompt x subject
       interaction, residual.
       Mixed-effects fit via statsmodels MixedLM; ANOVA fallback on
       condition-level subject accuracies if mixed-effects fails.
    3. H4 check: model x subject interaction proportion >=
       analysis_config.h4_mmlu_interaction_threshold (default 0.10).

Plus a seaborn heatmap of the model x subject accuracy matrix.

Usage:
    python scripts/09_mmlu_subject_decomp.py \\
      --item-level runs/normalized/item_level_primary.parquet \\
      --analysis-config configs/analysis_config.yaml \\
      --out-dir analysis/mmlu_subjects/

Optional:
    --figures-dir figures/
        Directory for the heatmap PNG. Defaults to
        <out-dir>/../../figures/.
    --benchmark-id mmlu_panel
        Override the benchmark filter (default 'mmlu_panel').

Exit codes per SPEC §11.1:
    0 — success
    1 — generic failure (implementation error)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from gradience_study.config import load_config  # noqa: E402


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [09_mmlu_subject_decomp] {msg}", file=sys.stderr)


# -- Variance-component schema ----------------------------------------------


VC_FIELDS: list[str] = ["source", "variance", "proportion", "h4_relevant"]

# Sources reported regardless of fit method, in canonical order.
VC_SOURCE_ORDER: list[str] = [
    "model_main",
    "subject_main",
    "prompt_main",
    "model_x_subject",
    "prompt_x_subject",
    "residual",
]


# -- Accuracy matrix ---------------------------------------------------------


def build_accuracy_matrix(item_level: pd.DataFrame) -> pd.DataFrame:
    """Pivot item-level rows to a model x subject mean accuracy matrix
    averaged over all conditions (prompts, seeds, scoring rules).

    Returns a DataFrame whose index is model_id (with a final row labelled
    `__subject_marginal__`) and whose columns are sorted subject_ids
    (with a final column `model_marginal_mean`).
    """
    # Per-(model, subject) mean accuracy over all items and conditions.
    grouped = (
        item_level.groupby(["model_id", "subject_id"], dropna=True)["is_correct"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot(index="model_id", columns="subject_id", values="is_correct")
    # Sort for deterministic output.
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    # Row marginal: per-model mean across subjects (uniform-weighted).
    pivot["model_marginal_mean"] = pivot.iloc[:, : len(pivot.columns)].mean(axis=1)
    # Column marginal row: per-subject mean across models.
    subject_marginals = pivot.iloc[:, :-1].mean(axis=0)
    overall_mean = float(pivot.iloc[:, :-1].to_numpy().mean())
    marginal_row = pd.Series(
        {**subject_marginals.to_dict(), "model_marginal_mean": overall_mean},
        name="__subject_marginal__",
    )
    pivot = pd.concat([pivot, marginal_row.to_frame().T])
    return pivot


def write_accuracy_matrix(matrix: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write with model_id as a real column, not just the index.
    df = matrix.reset_index().rename(columns={"index": "model_id"})
    if "model_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "model_id"})
    df.to_csv(out_path, index=False, lineterminator="\n")


# -- Variance components: mixed-effects --------------------------------------


def _crossed_random_var(
    df: pd.DataFrame,
    formula_fixed: str,
    re_formula: str | None,
    vc_formula: dict[str, str],
) -> dict[str, float]:
    """Fit a binomial-link mixed model with crossed random effects.

    Used for the Level-1 attempt. Returns a dict of variance components
    keyed by source name. Raises on convergence failure.
    """
    import statsmodels.formula.api as smf  # noqa: PLC0415

    model = smf.mixedlm(
        formula=formula_fixed,
        data=df,
        groups=np.ones(len(df)),  # single dummy group; vc_formula carries structure
        re_formula=re_formula,
        vc_formula=vc_formula,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=True, method="lbfgs", maxiter=200)
    if not result.converged:
        raise RuntimeError("MixedLM did not converge")
    out: dict[str, float] = {}
    # Variance components live in result.cov_re for re_formula, plus
    # result.vcomp for vc_formula entries (in the same order as vc_formula).
    if vc_formula:
        for name, comp in zip(vc_formula.keys(), result.vcomp):
            out[name] = float(comp)
    out["residual"] = float(result.scale)
    return out


def variance_components_mixed_effects(
    item_level: pd.DataFrame,
) -> dict[str, float]:
    """Attempt the full Level-1 mixed-effects decomposition for MMLU.

    Random effects (variance components):
        - subject (subject_main)
        - prompt (prompt_main)
        - model:subject (model_x_subject)
        - prompt:subject (prompt_x_subject)
    Fixed effects:
        - model

    Residual is the trailing residual variance from the MixedLM fit.

    Note: the model main effect is treated as a fixed effect (so its
    'variance' as reported here is the variance explained by the model
    fixed-effect contrasts), to match the spec's allowance that "fixed
    is simpler and adequate for 3 models."
    """
    df = item_level.copy()
    # Numerify outcome.
    df["y"] = df["is_correct"].astype(int)
    # Make sure categorical fields are strings.
    for col in ("model_id", "subject_id", "prompt_id"):
        df[col] = df[col].astype(str)

    # Fixed-effect model variance: variance of fitted values from the
    # model main-effect-only fit.
    model_main_var = float(
        df.groupby("model_id")["y"].mean().pipe(lambda s: s.var(ddof=0))
    )

    vc_formula = {
        "subject_main": "0 + C(subject_id)",
        "prompt_main": "0 + C(prompt_id)",
        "model_x_subject": "0 + C(model_id):C(subject_id)",
        "prompt_x_subject": "0 + C(prompt_id):C(subject_id)",
    }

    components = _crossed_random_var(
        df=df,
        formula_fixed="y ~ C(model_id)",
        re_formula="0",
        vc_formula=vc_formula,
    )
    components["model_main"] = model_main_var
    return components


# -- Variance components: ANOVA fallback -------------------------------------


def variance_components_anova_fallback(
    item_level: pd.DataFrame,
) -> dict[str, float]:
    """Method-of-moments / Type-III sum-of-squares variance components
    estimated from condition-level subject accuracies.

    Aggregates items to the (model, subject, prompt) cell level (mean
    accuracy across all seeds and scoring rules), then partitions sums of
    squares into main effects and interactions. This is a balanced-anova
    approximation: cells with missing combinations are dropped.
    """
    cell = (
        item_level.groupby(["model_id", "subject_id", "prompt_id"], dropna=True)[
            "is_correct"
        ]
        .mean()
        .reset_index(name="acc")
    )
    if cell.empty:
        return {k: 0.0 for k in VC_SOURCE_ORDER}

    grand_mean = float(cell["acc"].mean())

    def _ms(by: list[str]) -> float:
        means = cell.groupby(by)["acc"].mean()
        return float(((means - grand_mean) ** 2).mean())

    ss_model = _ms(["model_id"])
    ss_subject = _ms(["subject_id"])
    ss_prompt = _ms(["prompt_id"])

    # Interactions: variance of cell means after subtracting the additive
    # main-effect contributions.
    model_means = cell.groupby("model_id")["acc"].mean()
    subject_means = cell.groupby("subject_id")["acc"].mean()
    prompt_means = cell.groupby("prompt_id")["acc"].mean()

    def _additive_pred(row: pd.Series) -> float:
        return float(
            grand_mean
            + (model_means[row["model_id"]] - grand_mean)
            + (subject_means[row["subject_id"]] - grand_mean)
            + (prompt_means[row["prompt_id"]] - grand_mean)
        )

    cell["additive"] = cell.apply(_additive_pred, axis=1)
    cell["resid_full"] = cell["acc"] - cell["additive"]

    # Model x subject interaction component: variance of the (model, subject)
    # cell mean minus model main minus subject main.
    ms_means = cell.groupby(["model_id", "subject_id"])["acc"].mean()
    ms_int = []
    for (m, s), v in ms_means.items():
        ms_int.append(
            v - grand_mean - (model_means[m] - grand_mean) - (subject_means[s] - grand_mean)
        )
    ss_model_x_subject = float(np.var(np.asarray(ms_int, dtype=float), ddof=0))

    ps_means = cell.groupby(["prompt_id", "subject_id"])["acc"].mean()
    ps_int = []
    for (p, s), v in ps_means.items():
        ps_int.append(
            v - grand_mean - (prompt_means[p] - grand_mean) - (subject_means[s] - grand_mean)
        )
    ss_prompt_x_subject = float(np.var(np.asarray(ps_int, dtype=float), ddof=0))

    # Residual: variance of cell-level residuals after additive main-effect fit
    # minus the two-way interaction contributions.
    residual_var = float(np.var(cell["resid_full"].to_numpy(dtype=float), ddof=0))
    # The simple form above absorbs interaction variance into resid_full;
    # subtract the explicit interaction estimates so the components partition.
    residual_var = max(0.0, residual_var - ss_model_x_subject - ss_prompt_x_subject)

    return {
        "model_main": ss_model,
        "subject_main": ss_subject,
        "prompt_main": ss_prompt,
        "model_x_subject": ss_model_x_subject,
        "prompt_x_subject": ss_prompt_x_subject,
        "residual": residual_var,
    }


# -- Variance components: dispatcher ----------------------------------------


def compute_variance_components(
    item_level: pd.DataFrame,
) -> tuple[dict[str, float], str]:
    """Run the mixed-effects Level-1 fit; fall back to ANOVA on convergence
    failure or any modelling exception.

    Returns:
        (components, fallback_used) where fallback_used is one of
        "mixed_effects" or "anova".
    """
    try:
        components = variance_components_mixed_effects(item_level)
        return components, "mixed_effects"
    except Exception as e:  # noqa: BLE001
        _log(
            "WARNING",
            f"Mixed-effects fit failed ({type(e).__name__}: {e}); "
            f"falling back to ANOVA on condition-level subject accuracies.",
        )
        components = variance_components_anova_fallback(item_level)
        return components, "anova"


def normalize_to_proportions(
    components: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (variances, proportions). Proportions sum to 1 (guarded
    against zero-total)."""
    var: dict[str, float] = {k: max(0.0, float(components.get(k, 0.0))) for k in VC_SOURCE_ORDER}
    total = sum(var.values())
    if total <= 0:
        prop = {k: 0.0 for k in VC_SOURCE_ORDER}
    else:
        prop = {k: var[k] / total for k in VC_SOURCE_ORDER}
    return var, prop


def write_variance_components_csv(
    out_path: Path,
    variances: dict[str, float],
    proportions: dict[str, float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(VC_FIELDS)
        for src in VC_SOURCE_ORDER:
            writer.writerow(
                [
                    src,
                    repr(variances[src]),
                    repr(proportions[src]),
                    "true" if src == "model_x_subject" else "false",
                ]
            )


# -- Heatmap ----------------------------------------------------------------


def make_heatmap(matrix: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of the model x subject accuracy matrix (excluding marginals)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if matrix.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "No MMLU subject data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    # Strip marginal row and column for the heatmap proper.
    core = matrix.drop(index="__subject_marginal__", errors="ignore")
    if "model_marginal_mean" in core.columns:
        core = core.drop(columns="model_marginal_mean")

    fig, ax = plt.subplots(figsize=(max(6, 1.0 * core.shape[1]), max(3, 0.6 * core.shape[0])))
    sns.heatmap(
        core,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "mean accuracy"},
        ax=ax,
    )
    ax.set_title("MMLU model x subject mean accuracy")
    ax.set_xlabel("subject")
    ax.set_ylabel("model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -- Main pipeline ----------------------------------------------------------


def _load_h4_threshold(analysis_config_path: Path) -> float:
    """Read h4_mmlu_interaction_threshold from study_config.yaml or bare
    analysis_config.yaml.
    """
    try:
        merged = load_config(analysis_config_path)
        return float(merged.analysis_config.h4_mmlu_interaction_threshold)
    except Exception:  # noqa: BLE001
        import yaml  # noqa: PLC0415

        with open(analysis_config_path, "r", encoding="utf-8") as f:
            ac = yaml.safe_load(f)
        return float(ac.get("h4_mmlu_interaction_threshold", 0.10))


def run(
    item_level_path: Path,
    analysis_config_path: Path,
    out_dir: Path,
    figures_dir: Path | None,
    benchmark_id: str,
) -> int:
    if not item_level_path.exists():
        _log("ERROR", f"Item-level parquet not found: {item_level_path}")
        return 1

    h4_threshold = _load_h4_threshold(analysis_config_path)

    df = pd.read_parquet(item_level_path)
    _log(
        "INFO",
        f"Loaded {len(df)} item rows from {item_level_path}; "
        f"filtering to benchmark_id={benchmark_id!r}.",
    )
    df = df[df["benchmark_id"] == benchmark_id].copy()

    if df.empty:
        _log("ERROR", f"No items for benchmark_id={benchmark_id!r}.")
        return 1

    # Drop rows without a subject_id; MMLU items must have one.
    df = df.dropna(subset=["subject_id"])
    if df.empty:
        _log("ERROR", f"All MMLU items have null subject_id for {benchmark_id!r}.")
        return 1

    # 1. Accuracy matrix.
    matrix = build_accuracy_matrix(df)
    write_accuracy_matrix(matrix, out_dir / "mmlu_subject_accuracy_matrix.csv")
    _log(
        "INFO",
        f"Wrote model x subject matrix ({matrix.shape[0]-1} models, "
        f"{matrix.shape[1]-1} subjects).",
    )

    # 2. Variance components.
    components, fallback_used = compute_variance_components(df)
    variances, proportions = normalize_to_proportions(components)
    write_variance_components_csv(
        out_dir / "mmlu_subject_variance_components.csv", variances, proportions
    )

    interaction_prop = float(proportions["model_x_subject"])
    h4_confirmed = bool(interaction_prop >= h4_threshold)
    _log(
        "INFO",
        f"Variance components ({fallback_used} method): "
        f"model_x_subject proportion = {interaction_prop:.4f} "
        f"(H4 threshold {h4_threshold}); H4 confirmed: {h4_confirmed}",
    )

    # 3. H4 test JSON.
    h4_obj = {
        "h4_hypothesis": f"model x subject interaction >= {h4_threshold}",
        "threshold": h4_threshold,
        "observed_proportion": interaction_prop,
        "h4_confirmed": h4_confirmed,
        "fallback_used": fallback_used,
    }
    h4_path = out_dir / "h4_test.json"
    h4_path.parent.mkdir(parents=True, exist_ok=True)
    with open(h4_path, "w", encoding="utf-8") as f:
        json.dump(h4_obj, f, indent=2, sort_keys=True)
        f.write("\n")

    # 4. Figure.
    if figures_dir is None:
        figures_dir = out_dir.parent.parent / "figures"
    figure_path = figures_dir / "mmlu_model_subject_heatmap.png"
    make_heatmap(matrix, figure_path)
    _log("INFO", f"Wrote heatmap to {figure_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MMLU subject-level variance decomposition (SPEC §8.10)."
    )
    parser.add_argument(
        "--item-level",
        required=True,
        type=Path,
        help="Path to item-level parquet from 04_normalize_outputs.py.",
    )
    parser.add_argument(
        "--analysis-config",
        required=True,
        type=Path,
        help="Path to study_config.yaml or bare analysis_config.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for analysis output (created if missing).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for the heatmap PNG. Defaults to <out-dir>/../../figures.",
    )
    parser.add_argument(
        "--benchmark-id",
        type=str,
        default="mmlu_panel",
        help="Benchmark to filter on. Default: mmlu_panel.",
    )
    args = parser.parse_args()

    try:
        return run(
            args.item_level,
            args.analysis_config,
            args.out_dir,
            args.figures_dir,
            args.benchmark_id,
        )
    except Exception as e:  # noqa: BLE001
        _log("CRITICAL", f"Unhandled exception: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
