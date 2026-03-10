#!/usr/bin/env python3
"""
Study 16 Perplexity Analysis.

Correlates spectral merge quality metrics (Q_min, D, cosine retention)
with downstream perplexity measurements.  Framed as a small-N case study
with effect-size emphasis, not a correlation study with inferential
statistics.

Analysis hierarchy:
    Primary:    Within-pair naive vs recommended comparisons (effect sizes)
    Secondary:  Pooled rank-order association (Spearman, bootstrap CIs)
    Exploratory: Imbalanced vs balanced pair behaviour, norm-equalized
                 ablation isolating magnitude correction from full engine

Key metrics:
    - Loss delta:          merged_loss - source_loss
                           (positive = merge degraded; primary analytic quantity)
    - Perplexity ratio:    source_ppl / merged_ppl
                           (>1 = merged better; <1 = merged worse; secondary)
    - Normalized retention: (base_loss - merged_loss) / (base_loss - source_loss)
                           (1.0 = full source gain preserved; 0.0 = no better
                            than base; <0 = worse than base)

Inputs:
    - results/study16_perplexity/perplexity_results.json
    - results/study16b_merge_ablation/study16_merge_ablation.json

Outputs:
    - results/study16_perplexity/validation_report.json
    - results/study16_perplexity/validation_report.md

Usage:
    python scripts/study16_analyze_perplexity.py \\
        --perplexity-dir results/study16_perplexity \\
        --spectral-dir results/study16b_merge_ablation \\
        --output-dir results/study16_perplexity
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_perplexity_results(path: Path) -> tuple[dict, list[dict]]:
    """Load perplexity_results.json.  Returns (metadata, results)."""
    with open(path) as f:
        data = json.load(f)
    return data.get("metadata", {}), data["results"]


def load_spectral_results(path: Path) -> dict:
    """Load Study 16b spectral results.  Returns the full JSON."""
    with open(path) as f:
        return json.load(f)


def get_spectral_for_pair(spectral_data: dict, pair_id: str) -> dict | None:
    """Look up spectral comparison-table entry for a pair."""
    for row in spectral_data.get("comparison_table", []):
        if pair_id in row.get("pair_id", ""):
            return row
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Pair definitions  (must match the eval script)
# ═══════════════════════════════════════════════════════════════════════════

PAIR_INFO = {
    "pair_01": {
        "label_a": "metamath-r16",
        "label_b": "openwebmath-r16",
        "task_a": "math",
        "task_b": "math",
        "eval_set_a": "gsm8k_math",
        "eval_set_b": "gsm8k_math",
        "same_domain": True,
    },
    "pair_02": {
        "label_a": "metamath-r16",
        "label_b": "magicoder-r16",
        "task_a": "math",
        "task_b": "code",
        "eval_set_a": "gsm8k_math",
        "eval_set_b": "mbpp_code",
        "same_domain": False,
    },
    "pair_03": {
        "label_a": "magicoder-r16",
        "label_b": "btgenbot-r8",
        "task_a": "code",
        "task_b": "chat",
        "eval_set_a": "mbpp_code",
        "eval_set_b": "oasst2_chat",
        "same_domain": False,
    },
    "pair_04": {
        "label_a": "openwebmath-r64",
        "label_b": "btgenbot-r8",
        "task_a": "math",
        "task_b": "chat",
        "eval_set_a": "gsm8k_math",
        "eval_set_b": "oasst2_chat",
        "same_domain": False,
    },
    "pair_06": {
        "label_a": "catsubcat-r16",
        "label_b": "btgenbot-r8",
        "task_a": "chat",
        "task_b": "chat",
        "eval_set_a": "oasst2_chat",
        "eval_set_b": "oasst2_chat",
        "same_domain": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════


def lookup_result(results: list[dict], adapter_label: str, eval_set_id: str) -> dict | None:
    """Find the result for a given adapter and eval set."""
    for r in results:
        if r["adapter_label"] == adapter_label and r["eval_set_id"] == eval_set_id:
            loss = r.get("token_mean_loss", r.get("mean_loss"))
            if loss is not None and not (isinstance(loss, float) and np.isnan(loss)):
                return r
    return None


def lookup_loss(results: list[dict], adapter_label: str, eval_set_id: str) -> float | None:
    """Get token_mean_loss for an adapter on an eval set."""
    r = lookup_result(results, adapter_label, eval_set_id)
    if r is None:
        return None
    return r.get("token_mean_loss", r.get("mean_loss"))


def lookup_ppl(results: list[dict], adapter_label: str, eval_set_id: str) -> float | None:
    """Get perplexity for an adapter on an eval set."""
    r = lookup_result(results, adapter_label, eval_set_id)
    if r is None:
        return None
    return r.get("perplexity")


def compute_loss_delta(source_loss: float, merged_loss: float) -> float:
    """Loss delta: merged - source.
    Positive = merge made it worse.  Primary analytic quantity.
    """
    return merged_loss - source_loss


def compute_ppl_ratio(source_ppl: float, merged_ppl: float) -> float:
    """Perplexity ratio: source / merged.
    > 1.0 = merged is better (lower ppl)
    = 1.0 = no change
    < 1.0 = merged is worse (higher ppl)
    """
    if merged_ppl <= 0:
        return float("nan")
    return source_ppl / merged_ppl


def compute_normalized_retention(
    base_loss: float,
    source_loss: float,
    merged_loss: float,
) -> float:
    """Normalized retention: how much of source-adapter-over-base gain is kept.

    = (base_loss - merged_loss) / (base_loss - source_loss)

    1.0 = merged preserves all source gain over base
    0.0 = merged is no better than base
    < 0  = merged is worse than base
    > 1  = merged exceeds source (rare but possible)
    """
    denom = base_loss - source_loss
    if abs(denom) < 1e-8:
        return float("nan")  # source adapter didn't help vs base
    return (base_loss - merged_loss) / denom


def compute_ppl_dominance(loss_task_a: float, loss_task_b: float) -> float:
    """Behavioral dominance: asymmetry in cross-task loss deltas.
    Analogous to spectral D.  Uses log-ratio for scale invariance.
    """
    if loss_task_a <= 0 or loss_task_b <= 0:
        return float("nan")
    # Convert losses to ppls for log-ratio
    ppl_a = np.exp(loss_task_a)
    ppl_b = np.exp(loss_task_b)
    return abs(np.log(ppl_a) - np.log(ppl_b))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation and p-value."""
    from scipy import stats

    if len(x) < 3:
        return (float("nan"), float("nan"))
    rho, p = stats.spearmanr(x, y)
    return (float(rho), float(p))


def bootstrap_ci(
    x: np.ndarray, y: np.ndarray, func=None, n_boot: int = 5000, ci: float = 0.95
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a correlation.
    Returns (estimate, lower, upper).
    """
    from scipy import stats

    if func is None:

        def func(a, b):
            return stats.spearmanr(a, b)[0]

    estimate = func(x, y)
    n = len(x)
    rng = np.random.RandomState(42)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        with contextlib.suppress(Exception):
            boot_vals.append(func(x[idx], y[idx]))

    if not boot_vals:
        return (estimate, float("nan"), float("nan"))

    boot_vals = np.array(boot_vals)
    alpha = 1.0 - ci
    lower = float(np.percentile(boot_vals, 100 * alpha / 2))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return (estimate, lower, upper)


# ═══════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════


def build_analysis(
    ppl_results: list[dict],
    spectral_data: dict,
) -> dict:
    """Build the full validation analysis."""
    analysis: dict[str, Any] = {}

    # ── 1. Baselines ──────────────────────────────────────────────────
    base_losses = {}
    source_losses = {}
    source_ppls = {}

    for r in ppl_results:
        es_id = r.get("eval_set_id", r.get("eval_task", ""))
        loss = r.get("token_mean_loss", r.get("mean_loss"))
        ppl = r.get("perplexity")

        if r["adapter_type"] == "base":
            base_losses[es_id] = loss
        elif r["adapter_type"] == "source":
            key = (r["adapter_label"], es_id)
            source_losses[key] = loss
            source_ppls[key] = ppl

    analysis["base_model"] = {es: {"loss": l, "ppl": float(np.exp(l)) if l else None} for es, l in base_losses.items()}
    analysis["source_adapters"] = {
        f"{label}@{es}": {"loss": loss, "ppl": source_ppls.get((label, es))}
        for (label, es), loss in source_losses.items()
    }

    # ── 2. Per-pair within-condition analysis (PRIMARY) ───────────────
    pair_analyses = []

    for pair_id, info in PAIR_INFO.items():
        spectral = get_spectral_for_pair(spectral_data, pair_id)
        if spectral is None:
            continue

        es_a = info["eval_set_a"]
        es_b = info["eval_set_b"]

        # Source baselines
        source_loss_a = source_losses.get((info["label_a"], es_a))
        source_loss_b = source_losses.get((info["label_b"], es_b))
        source_ppl_a = source_ppls.get((info["label_a"], es_a))
        source_ppl_b = source_ppls.get((info["label_b"], es_b))
        base_loss_a = base_losses.get(es_a)
        base_loss_b = base_losses.get(es_b)

        pair_entry = {
            "pair_id": pair_id,
            "labels": f"{info['label_a']} × {info['label_b']}",
            "same_domain": info["same_domain"],
            "source_loss_a": source_loss_a,
            "source_loss_b": source_loss_b,
            "base_loss_a": base_loss_a,
            "base_loss_b": base_loss_b,
            "conditions": {},
        }

        for cond in ["naive", "norm_equalized", "recommended"]:
            merged_label = f"{pair_id}_{cond}"

            merged_loss_a = lookup_loss(ppl_results, merged_label, es_a)
            merged_loss_b = lookup_loss(ppl_results, merged_label, es_b)
            merged_ppl_a = lookup_ppl(ppl_results, merged_label, es_a)
            merged_ppl_b = lookup_ppl(ppl_results, merged_label, es_b)

            if merged_loss_a is None and merged_loss_b is None:
                continue  # this condition not available (e.g., norm_eq for balanced pair)

            # Spectral metrics for this condition
            prefix = "naive" if cond == "naive" else "rec"
            if cond == "norm_equalized":
                prefix = None  # no spectral data for norm-eq

            s_Q_min = spectral.get(f"{prefix}_Q_min") if prefix else None
            s_D = spectral.get(f"{prefix}_D") if prefix else None

            cond_entry: dict[str, Any] = {
                "spectral_Q_min": s_Q_min,
                "spectral_D": s_D,
            }

            # Per-side metrics
            for side, m_loss, m_ppl, s_loss, s_ppl, b_loss in [
                ("a", merged_loss_a, merged_ppl_a, source_loss_a, source_ppl_a, base_loss_a),
                ("b", merged_loss_b, merged_ppl_b, source_loss_b, source_ppl_b, base_loss_b),
            ]:
                if m_loss is not None and s_loss is not None:
                    cond_entry[f"merged_loss_{side}"] = m_loss
                    cond_entry[f"merged_ppl_{side}"] = m_ppl
                    cond_entry[f"loss_delta_{side}"] = round(compute_loss_delta(s_loss, m_loss), 6)
                    cond_entry[f"ppl_ratio_{side}"] = (
                        round(compute_ppl_ratio(s_ppl, m_ppl), 4) if (s_ppl and m_ppl) else None
                    )

                    if b_loss is not None:
                        cond_entry[f"normalized_retention_{side}"] = round(
                            compute_normalized_retention(b_loss, s_loss, m_loss), 4
                        )

            # Worst-side metrics
            deltas = [
                cond_entry.get(f"loss_delta_{s}") for s in ["a", "b"] if cond_entry.get(f"loss_delta_{s}") is not None
            ]
            retentions = [
                cond_entry.get(f"normalized_retention_{s}")
                for s in ["a", "b"]
                if cond_entry.get(f"normalized_retention_{s}") is not None
            ]

            if deltas:
                cond_entry["worst_loss_delta"] = max(deltas)  # worst = largest increase
            if retentions:
                cond_entry["worst_normalized_retention"] = min(retentions)

            pair_entry["conditions"][cond] = cond_entry

        pair_analyses.append(pair_entry)

    analysis["pair_analyses"] = pair_analyses

    # ── 3. Paired comparison: naive vs recommended (PRIMARY) ──────────
    paired_comparisons = []

    for pa in pair_analyses:
        naive = pa["conditions"].get("naive")
        rec = pa["conditions"].get("recommended")
        norm_eq = pa["conditions"].get("norm_equalized")

        if not naive or not rec:
            continue

        comp: dict[str, Any] = {
            "pair_id": pa["pair_id"],
            "labels": pa["labels"],
        }

        # Effect sizes: recommended vs naive
        for side in ["a", "b"]:
            naive_loss = naive.get(f"merged_loss_{side}")
            rec_loss = rec.get(f"merged_loss_{side}")
            if naive_loss is not None and rec_loss is not None:
                comp[f"naive_loss_{side}"] = naive_loss
                comp[f"rec_loss_{side}"] = rec_loss
                comp[f"delta_loss_{side}"] = round(naive_loss - rec_loss, 6)  # positive = rec better

            naive_ret = naive.get(f"normalized_retention_{side}")
            rec_ret = rec.get(f"normalized_retention_{side}")
            if naive_ret is not None and rec_ret is not None:
                comp[f"naive_retention_{side}"] = naive_ret
                comp[f"rec_retention_{side}"] = rec_ret
                comp[f"delta_retention_{side}"] = round(rec_ret - naive_ret, 4)

        # Norm-equalized comparison (ablation: is magnitude correction enough?)
        if norm_eq:
            for side in ["a", "b"]:
                ne_loss = norm_eq.get(f"merged_loss_{side}")
                rec_loss = rec.get(f"merged_loss_{side}")
                if ne_loss is not None and rec_loss is not None:
                    comp[f"norm_eq_loss_{side}"] = ne_loss
                    comp[f"delta_normeq_vs_rec_{side}"] = round(ne_loss - rec_loss, 6)

        # Spectral deltas for reference
        spectral = get_spectral_for_pair(spectral_data, pa["pair_id"])
        if spectral:
            comp["spectral_delta_Q_min"] = spectral.get("delta_Q_min")
            comp["spectral_delta_D"] = spectral.get("delta_D")

        paired_comparisons.append(comp)

    analysis["paired_comparisons"] = paired_comparisons

    # ── 4. Rank-order association (SECONDARY) ─────────────────────────
    # Pool all merged conditions that have both spectral and ppl metrics
    corr_points = []
    for pa in pair_analyses:
        for cond_name, cond in pa["conditions"].items():
            q_min = cond.get("spectral_Q_min")
            worst_ret = cond.get("worst_normalized_retention")
            d = cond.get("spectral_D")
            if q_min is not None and worst_ret is not None:
                corr_points.append(
                    {
                        "pair_id": pa["pair_id"],
                        "condition": cond_name,
                        "Q_min": q_min,
                        "D": d,
                        "worst_retention": worst_ret,
                        "worst_loss_delta": cond.get("worst_loss_delta"),
                    }
                )

    associations = []

    if len(corr_points) >= 3:
        x_qmin = np.array([p["Q_min"] for p in corr_points])
        y_ret = np.array([p["worst_retention"] for p in corr_points])

        rho, p = spearman_rho(x_qmin, y_ret)
        est, lo, hi = bootstrap_ci(x_qmin, y_ret)
        associations.append(
            {
                "description": "Q_min vs worst normalized retention",
                "spearman_rho": round(rho, 4) if not np.isnan(rho) else None,
                "p_value": round(p, 4) if not np.isnan(p) else None,
                "bootstrap_95_ci": [round(lo, 4), round(hi, 4)],
                "n": len(corr_points),
                "note": "small-N; interpret as directional evidence, not inference",
            }
        )

        # D vs worst loss delta
        d_vals = np.array([p["D"] for p in corr_points if p["D"] is not None])
        ld_vals = np.array(
            [p["worst_loss_delta"] for p in corr_points if p["D"] is not None and p["worst_loss_delta"] is not None]
        )
        if len(d_vals) >= 3 and len(d_vals) == len(ld_vals):
            rho2, p2 = spearman_rho(d_vals, ld_vals)
            est2, lo2, hi2 = bootstrap_ci(d_vals, ld_vals)
            associations.append(
                {
                    "description": "D vs worst loss delta",
                    "spearman_rho": round(rho2, 4) if not np.isnan(rho2) else None,
                    "p_value": round(p2, 4) if not np.isnan(p2) else None,
                    "bootstrap_95_ci": [round(lo2, 4), round(hi2, 4)],
                    "n": len(d_vals),
                    "note": "positive rho = higher D → larger loss degradation (expected)",
                }
            )

    analysis["rank_order_associations"] = associations
    analysis["correlation_points"] = corr_points

    # ── 5. Pre-registered interpretation ──────────────────────────────
    analysis["interpretation_framework"] = {
        "strong_validation": (
            "Q_min strongly tracks normalized retention AND D tracks "
            "catastrophic loss degradation. Gradience can claim proxy-level "
            "decision support for merge quality."
        ),
        "partial_validation": (
            "D tracks catastrophic failures but Q_min weakly tracks full "
            "retention. Gradience becomes a failure-prevention / dominance-"
            "mitigation tool, not a full quality predictor."
        ),
        "proxy_breakdown": (
            "Neither Q_min nor D predict perplexity outcomes. Gradience "
            "remains a structural audit layer, not a behavioral predictor. "
            "Important negative result."
        ),
    }

    return analysis


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════


def generate_markdown_report(analysis: dict) -> str:
    """Generate a markdown validation report."""
    lines = []
    lines.append("# Study 16 — Perplexity Validation Report\n")
    lines.append("*Framed as a small-N case study with effect-size emphasis.*\n")

    # Base model
    lines.append("## Base Model Baselines\n")
    lines.append("| Eval Set | Loss | Perplexity |")
    lines.append("|----------|------|-----------|")
    for es, info in sorted(analysis.get("base_model", {}).items()):
        loss = info.get("loss")
        ppl = info.get("ppl")
        lines.append(f"| {es} | {loss:.4f} | {ppl:.2f} |" if loss else f"| {es} | — | — |")
    lines.append("")

    # Source adapters
    lines.append("## Source Adapter Baselines\n")
    lines.append("| Adapter @ EvalSet | Loss | Perplexity |")
    lines.append("|-------------------|------|-----------|")
    for key, info in sorted(analysis.get("source_adapters", {}).items()):
        loss = info.get("loss")
        ppl = info.get("ppl")
        lines.append(f"| {key} | {loss:.4f} | {ppl:.2f} |" if loss else f"| {key} | — | — |")
    lines.append("")

    # Paired comparisons (PRIMARY)
    lines.append("## Primary Analysis: Naive vs Recommended (Effect Sizes)\n")
    for comp in analysis.get("paired_comparisons", []):
        lines.append(f"### {comp['pair_id']} — {comp['labels']}\n")
        for side in ["a", "b"]:
            nl = comp.get(f"naive_loss_{side}")
            rl = comp.get(f"rec_loss_{side}")
            dl = comp.get(f"delta_loss_{side}")
            nr = comp.get(f"naive_retention_{side}")
            rr = comp.get(f"rec_retention_{side}")
            dr = comp.get(f"delta_retention_{side}")
            if nl is not None and rl is not None:
                lines.append(
                    f"**Side {side}:** naive loss = {nl:.4f}, "
                    f"rec loss = {rl:.4f}, "
                    f"Δ = {dl:+.4f} (positive = rec better)"
                )
                if nr is not None:
                    lines.append(f"  normalized retention: naive = {nr:.4f}, rec = {rr:.4f}, Δ = {dr:+.4f}")

            # Norm-equalized ablation
            nel = comp.get(f"norm_eq_loss_{side}")
            dne = comp.get(f"delta_normeq_vs_rec_{side}")
            if nel is not None:
                lines.append(f"  norm-equalized loss = {nel:.4f}, gap vs recommended = {dne:+.4f}")
        lines.append("")

        sq = comp.get("spectral_delta_Q_min")
        if sq is not None:
            lines.append(f"Spectral δQ_min = {sq:+.4f}\n")

    # Rank-order associations (SECONDARY)
    lines.append("## Secondary Analysis: Rank-Order Associations\n")
    lines.append("*Interpret as directional evidence, not inferential statistics.*\n")
    for assoc in analysis.get("rank_order_associations", []):
        rho = assoc.get("spearman_rho")
        ci = assoc.get("bootstrap_95_ci", [None, None])
        lines.append(f"**{assoc['description']}** (n={assoc['n']})")
        if rho is not None:
            lines.append(f"  Spearman ρ = {rho:.4f}, 95% bootstrap CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
        if assoc.get("note"):
            lines.append(f"  *{assoc['note']}*")
        lines.append("")

    # Interpretation framework
    lines.append("## Pre-Registered Interpretation Framework\n")
    framework = analysis.get("interpretation_framework", {})
    for outcome, desc in framework.items():
        lines.append(f"**{outcome.replace('_', ' ').title()}:** {desc}\n")

    lines.append("## Conclusion\n")
    lines.append("*To be written after reviewing results.*\n")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study 16 — Perplexity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--perplexity-dir",
        type=Path,
        default=Path("results/study16_perplexity"),
        help="Directory containing perplexity_results.json",
    )
    parser.add_argument(
        "--spectral-dir",
        type=Path,
        default=Path("results/study16b_merge_ablation"),
        help="Directory containing study16_merge_ablation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as perplexity-dir)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.perplexity_dir

    # Load data
    ppl_path = args.perplexity_dir / "perplexity_results.json"
    spectral_path = args.spectral_dir / "study16_merge_ablation.json"

    if not ppl_path.exists():
        print(f"Error: {ppl_path} not found. Run study16_perplexity_validation.py first.")
        sys.exit(1)
    if not spectral_path.exists():
        print(f"Error: {spectral_path} not found.")
        sys.exit(1)

    print("Loading perplexity results ...")
    metadata, ppl_results = load_perplexity_results(ppl_path)
    print(f"  {len(ppl_results)} evaluation results loaded")
    print(f"  Scoring method: {metadata.get('scoring', 'unknown')}")

    print("Loading spectral results ...")
    spectral_data = load_spectral_results(spectral_path)
    n_pairs = len(spectral_data.get("comparison_table", []))
    print(f"  {n_pairs} pairs in spectral comparison table")

    # Build analysis
    print("\nRunning analysis ...")
    analysis = build_analysis(ppl_results, spectral_data)

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_report.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  JSON report: {json_path}")

    # Save markdown
    md_report = generate_markdown_report(analysis)
    md_path = output_dir / "validation_report.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"  Markdown report: {md_path}")

    # Print paired comparison summary
    print(f"\n{'=' * 70}")
    print("  PAIRED COMPARISONS (naive vs recommended)")
    print(f"{'=' * 70}")
    for comp in analysis.get("paired_comparisons", []):
        pid = comp["pair_id"]
        sq = comp.get("spectral_delta_Q_min", 0)
        print(f"\n  {pid} ({comp['labels']})")
        for side in ["a", "b"]:
            dl = comp.get(f"delta_loss_{side}")
            dr = comp.get(f"delta_retention_{side}")
            if dl is not None:
                print(
                    f"    Side {side}: Δloss = {dl:+.4f}  Δretention = {dr:+.4f}"
                    if dr
                    else f"    Side {side}: Δloss = {dl:+.4f}"
                )
        if sq:
            print(f"    Spectral δQ_min = {sq:+.4f}")
    print(f"\n{'=' * 70}")

    # Associations
    print("\n  RANK-ORDER ASSOCIATIONS")
    print(f"{'─' * 70}")
    for assoc in analysis.get("rank_order_associations", []):
        rho = assoc.get("spearman_rho")
        ci = assoc.get("bootstrap_95_ci", [None, None])
        if rho is not None:
            print(f"  {assoc['description']}: ρ={rho:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (n={assoc['n']})")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
