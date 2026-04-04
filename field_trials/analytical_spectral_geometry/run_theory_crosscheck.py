#!/usr/bin/env python3
"""Cross-check analytical spectral theory predictions against field trial merge outcomes.

Reads the strict-naive rerun results (30 pairs with actual merge outcomes),
re-runs merge_audit to get per-layer SubspaceMetrics, then computes:
  1. OA-v1 heuristic scores (existing)
  2. OA-v2 interaction scores (existing)
  3. Analytical theory scores (NEW: over_accumulation_exact_condition)

Compares all three against the actual merge_delta_vs_best_source outcome.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gradience.vnext.merge import merge_audit
from gradience.vnext.merge.over_accumulation import estimate_over_accumulation, estimate_over_accumulation_v2
from gradience.vnext.merge.spectral_compat import SubspaceMetrics
from gradience.vnext.merge.spectral_theory import (
    norm_equalized_over_accumulation_analysis,
    over_accumulation_exact_condition,
)

INPUT_JSON = Path("field_trials/over_accumulation_followup/oa_v2_30_40_r1_strict_naive_rerun_results.json")
OUTPUT_JSON = Path("field_trials/analytical_spectral_geometry/theory_crosscheck.json")
OUTPUT_MD = Path("field_trials/analytical_spectral_geometry/theory_crosscheck.md")
POOR_THRESHOLD = -0.05


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None

    def _rank(vals: list[float]) -> list[float]:
        ordered = sorted((v, i) for i, v in enumerate(vals))
        out = [0.0] * len(vals)
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                j += 1
            r = ((i + j - 1) / 2.0) + 1.0
            for k in range(i, j):
                out[ordered[k][1]] = r
            i = j
        return out

    rx = _rank(x)
    ry = _rank(y)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx)
    den_y = sum((b - my) ** 2 for b in ry)
    den = math.sqrt(den_x * den_y)
    if den <= 0:
        return None
    return num / den


def _topk_mean(vals: list[float], k: int) -> float:
    if not vals:
        return 0.0
    top = sorted(vals, reverse=True)[: max(1, min(k, len(vals)))]
    return sum(top) / len(top)


def _to_subspace_metrics(metrics_dict: dict[str, Any]) -> SubspaceMetrics:
    d = dict(metrics_dict)
    for key in (
        "principal_angle_cosines",
        "right_principal_angle_cosines",
        "effective_singular_values_a",
        "effective_singular_values_b",
    ):
        if key in d and isinstance(d[key], list):
            d[key] = tuple(d[key])
    return SubspaceMetrics(**d)


def _poor_enrichment(rows: list[dict[str, Any]], metric_key: str, poor_key: str = "is_poor") -> dict[str, Any]:
    n = len(rows)
    if n < 3:
        return {"n_alerted": 0, "poor_rate_alerted": None, "poor_rate_rest": None, "lift": None, "recall": None}
    k = max(1, n // 3)
    idx = sorted(range(n), key=lambda i: float(rows[i][metric_key]), reverse=True)
    alerted = set(idx[:k])
    poor_total = sum(1 for r in rows if bool(r[poor_key]))
    poor_alerted = sum(1 for i in alerted if bool(rows[i][poor_key]))
    poor_rest = sum(1 for i in range(n) if i not in alerted and bool(rows[i][poor_key]))
    rate_alerted = poor_alerted / len(alerted) if alerted else 0.0
    rate_rest = poor_rest / (n - len(alerted)) if (n - len(alerted)) else 0.0
    recall = poor_alerted / poor_total if poor_total else 0.0
    return {
        "n_alerted": len(alerted),
        "poor_rate_alerted": round(rate_alerted, 4),
        "poor_rate_rest": round(rate_rest, 4),
        "lift": round(rate_alerted - rate_rest, 4),
        "recall": round(recall, 4),
    }


def main() -> int:
    rows_in = json.loads(INPUT_JSON.read_text())
    pair_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, row in enumerate(rows_in):
        if str(row.get("status")) != "ok":
            continue
        strict_ok = bool((row.get("strict_naive_check") or {}).get("ok"))
        if not strict_ok:
            continue
        a_path = row.get("source_a_eval", {}).get("adapter_path")
        b_path = row.get("source_b_eval", {}).get("adapter_path")
        if not a_path or not b_path:
            continue
        if not Path(a_path).exists() or not Path(b_path).exists():
            errors.append(f"[{i}] adapter path missing: {a_path} or {b_path}")
            continue

        pair_id = row.get("pair_id", f"pair_{i}")
        best_source = max(
            _safe_float(row.get("source_a_eval", {}).get("adapter_score")),
            _safe_float(row.get("source_b_eval", {}).get("adapter_score")),
        )
        merged_score = _safe_float(row.get("merged_eval", {}).get("merged_accuracy"))
        delta = merged_score - best_source if math.isfinite(merged_score) and math.isfinite(best_source) else math.nan

        try:
            report = merge_audit(str(a_path), str(b_path), verbose=False)
            layer_dicts = list(report.layer_verdicts or [])

            # --- OA-v1 scores ---
            v1_scores = [float(ld.get("over_accumulation_score", 0.0)) for ld in layer_dicts]

            # --- OA-v2 scores ---
            v2_scores: list[float] = []
            v2_interaction: list[float] = []

            # --- Analytical theory scores ---
            theory_risk_scores: list[float] = []
            theory_inflation_ratios: list[float] = []
            theory_cross_bounds: list[float] = []
            theory_inflation_possible: list[bool] = []
            theory_drivers: list[str] = []
            theory_vs_heuristic: list[str] = []

            # --- Norm-equalized analysis ---
            normeq_reduction_factors: list[float] = []
            normeq_reduces: list[bool] = []

            for ld in layer_dicts:
                m = _to_subspace_metrics(ld.get("metrics", {}))

                # OA-v2
                est_v2 = estimate_over_accumulation_v2(m, assumed_coefficients=(0.5, 0.5))
                v2_scores.append(float(est_v2.score))
                v2_interaction.append(float(est_v2.factors.interaction_primary))

                # Analytical theory
                theory = over_accumulation_exact_condition(m, coefficients=(0.5, 0.5))
                theory_risk_scores.append(theory.analytical_risk_score)
                theory_inflation_ratios.append(theory.inflation_ratio_upper)
                theory_cross_bounds.append(theory.cross_term_spectral_bound)
                theory_inflation_possible.append(theory.is_inflation_possible)
                theory_drivers.append(theory.inflation_driven_by)
                theory_vs_heuristic.append(theory.analytical_vs_heuristic_alignment)

                # Norm-equalized comparison
                neq = norm_equalized_over_accumulation_analysis(m, coefficients=(0.5, 0.5))
                normeq_reduction_factors.append(neq.reduction_factor)
                normeq_reduces.append(neq.inflation_reduced)

            pair_rows.append({
                "pair_id": pair_id,
                "merge_delta_vs_best_source": round(delta, 6) if math.isfinite(delta) else None,
                "is_poor": delta <= POOR_THRESHOLD if math.isfinite(delta) else False,
                "best_source_score": round(best_source, 4) if math.isfinite(best_source) else None,
                "merged_score": round(merged_score, 4) if math.isfinite(merged_score) else None,
                "n_layers": len(layer_dicts),
                # V1
                "oa_v1_max_score": round(max(v1_scores), 4) if v1_scores else 0.0,
                "oa_v1_top3_mean": round(_topk_mean(v1_scores, 3), 4) if v1_scores else 0.0,
                # V2
                "oa_v2_max_score": round(max(v2_scores), 4) if v2_scores else 0.0,
                "oa_v2_top3_interaction": round(_topk_mean(v2_interaction, 3), 4) if v2_interaction else 0.0,
                # Analytical theory
                "theory_max_risk": round(max(theory_risk_scores), 4) if theory_risk_scores else 0.0,
                "theory_top3_risk": round(_topk_mean(theory_risk_scores, 3), 4) if theory_risk_scores else 0.0,
                "theory_max_inflation_ratio": round(max(theory_inflation_ratios), 4) if theory_inflation_ratios else 1.0,
                "theory_mean_inflation_ratio": round(
                    sum(theory_inflation_ratios) / len(theory_inflation_ratios), 4
                ) if theory_inflation_ratios else 1.0,
                "theory_max_cross_bound": round(max(theory_cross_bounds), 4) if theory_cross_bounds else 0.0,
                "theory_inflation_possible_fraction": round(
                    sum(theory_inflation_possible) / len(theory_inflation_possible), 4
                ) if theory_inflation_possible else 0.0,
                "theory_driver_counts": {
                    d: theory_drivers.count(d) for d in set(theory_drivers)
                } if theory_drivers else {},
                "theory_vs_heuristic_counts": {
                    a: theory_vs_heuristic.count(a) for a in set(theory_vs_heuristic)
                } if theory_vs_heuristic else {},
                # Norm-equalized
                "normeq_mean_reduction_factor": round(
                    sum(normeq_reduction_factors) / len(normeq_reduction_factors), 4
                ) if normeq_reduction_factors else 1.0,
                "normeq_reduces_fraction": round(
                    sum(normeq_reduces) / len(normeq_reduces), 4
                ) if normeq_reduces else 0.0,
            })
            print(f"  [{len(pair_rows)}/{len(rows_in)}] {pair_id}: delta={delta:.4f}, "
                  f"v1={max(v1_scores):.3f}, v2={max(v2_scores):.3f}, "
                  f"theory={max(theory_risk_scores):.3f}")

        except Exception as e:
            errors.append(f"[{i}] {pair_id}: {e}")

    # === Analysis ===
    y = [float(r["merge_delta_vs_best_source"]) for r in pair_rows if r["merge_delta_vs_best_source"] is not None]
    x_v1 = [float(r["oa_v1_max_score"]) for r in pair_rows if r["merge_delta_vs_best_source"] is not None]
    x_v2 = [float(r["oa_v2_max_score"]) for r in pair_rows if r["merge_delta_vs_best_source"] is not None]
    x_theory = [float(r["theory_max_risk"]) for r in pair_rows if r["merge_delta_vs_best_source"] is not None]
    x_theory_inflation = [
        float(r["theory_max_inflation_ratio"]) for r in pair_rows if r["merge_delta_vs_best_source"] is not None
    ]
    x_theory_cross = [
        float(r["theory_max_cross_bound"]) for r in pair_rows if r["merge_delta_vs_best_source"] is not None
    ]

    spearman_results = {
        "v1_max_score": _spearman(x_v1, y),
        "v2_max_score": _spearman(x_v2, y),
        "theory_max_risk": _spearman(x_theory, y),
        "theory_max_inflation_ratio": _spearman(x_theory_inflation, y),
        "theory_max_cross_bound": _spearman(x_theory_cross, y),
    }

    enrichment = {
        "v1": _poor_enrichment(pair_rows, "oa_v1_max_score"),
        "v2": _poor_enrichment(pair_rows, "oa_v2_max_score"),
        "theory_risk": _poor_enrichment(pair_rows, "theory_max_risk"),
        "theory_inflation": _poor_enrichment(pair_rows, "theory_max_inflation_ratio"),
    }

    # Theory vs heuristic agreement summary
    agreement_summary: dict[str, int] = {}
    for r in pair_rows:
        for label, count in r.get("theory_vs_heuristic_counts", {}).items():
            agreement_summary[label] = agreement_summary.get(label, 0) + count

    payload = {
        "status": "theory_crosscheck_completed",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_pairs": len(pair_rows),
        "n_errors": len(errors),
        "errors": errors,
        "analysis": {
            "spearman_correlations": {k: round(v, 4) if v is not None else None for k, v in spearman_results.items()},
            "poor_merge_enrichment": enrichment,
            "theory_vs_heuristic_agreement": agreement_summary,
        },
        "pair_rows": pair_rows,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))

    # === Markdown report ===
    def _fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    n_poor = sum(1 for r in pair_rows if r.get("is_poor"))

    md_lines = [
        "# Analytical Spectral Theory Cross-Check",
        "",
        f"**Date**: {datetime.now(UTC).strftime('%Y-%m-%d')}",
        f"**Pairs analyzed**: {len(pair_rows)} ({n_poor} poor merges, threshold {POOR_THRESHOLD})",
        f"**Errors**: {len(errors)}",
        "",
        "## Spearman Correlations (metric vs merge_delta)",
        "",
        "Higher (more negative) Spearman = metric better predicts bad merges.",
        "",
        "| Metric | Spearman |",
        "|--------|----------|",
    ]
    for name, val in spearman_results.items():
        md_lines.append(f"| {name} | {_fmt(val)} |")

    md_lines += [
        "",
        "## Poor-Merge Enrichment (top tercile alert)",
        "",
        "| Metric | Recall | Lift | Rate (alerted) | Rate (rest) |",
        "|--------|--------|------|----------------|-------------|",
    ]
    for name, e in enrichment.items():
        md_lines.append(
            f"| {name} | {_fmt(e.get('recall'))} | {_fmt(e.get('lift'))} "
            f"| {_fmt(e.get('poor_rate_alerted'))} | {_fmt(e.get('poor_rate_rest'))} |"
        )

    md_lines += [
        "",
        "## Theory vs Heuristic Agreement (per-layer)",
        "",
        "| Classification | Count |",
        "|---------------|-------|",
    ]
    for label, count in sorted(agreement_summary.items()):
        md_lines.append(f"| {label} | {count} |")

    # Per-pair details
    md_lines += [
        "",
        "## Per-Pair Summary",
        "",
        "| Pair | Delta | Poor | V1 max | V2 max | Theory risk | Inflation ratio | NormEq reduces |",
        "|------|-------|------|--------|--------|-------------|-----------------|---------------|",
    ]
    for r in sorted(pair_rows, key=lambda x: x.get("merge_delta_vs_best_source") or 0):
        short_id = r["pair_id"][:60]
        delta_str = f"{r['merge_delta_vs_best_source']:.4f}" if r["merge_delta_vs_best_source"] is not None else "N/A"
        md_lines.append(
            f"| {short_id} | {delta_str} | {'YES' if r['is_poor'] else 'no'} "
            f"| {r['oa_v1_max_score']:.3f} | {r['oa_v2_max_score']:.3f} "
            f"| {r['theory_max_risk']:.3f} | {r['theory_max_inflation_ratio']:.3f} "
            f"| {r['normeq_reduces_fraction']:.1%} |"
        )

    OUTPUT_MD.write_text("\n".join(md_lines) + "\n")

    # === Print summary ===
    print(f"\n{'='*60}")
    print(f"Theory Cross-Check: {len(pair_rows)} pairs, {len(errors)} errors")
    print(f"{'='*60}")
    print(f"\nSpearman correlations (metric vs merge_delta):")
    for name, val in spearman_results.items():
        print(f"  {name:30s}: {_fmt(val)}")
    print(f"\nPoor-merge enrichment (top tercile recall):")
    for name, e in enrichment.items():
        print(f"  {name:20s}: recall={_fmt(e.get('recall'))}, lift={_fmt(e.get('lift'))}")
    print(f"\nTheory vs heuristic agreement: {agreement_summary}")
    print(f"\nOutputs: {OUTPUT_JSON}, {OUTPUT_MD}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
