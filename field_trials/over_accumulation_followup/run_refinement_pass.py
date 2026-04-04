#!/usr/bin/env python3
"""Over-accumulation diagnostic refinement pass (CPU-only, artifact-driven).

This script executes the bounded refinement workflow:
Stage A: freeze current diagnostic baseline
Stage B: build subfactor/interaction/hotspot feature table
Stage C: run refinement analyses against strict-naive outcomes
Stage D: generate matched case studies
Stage E: assess refined candidate quality
Stage F: write refine/keep/pause decision memo + strategy summary
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FOLLOWUP_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_STRATEGY_DIR = REPO_ROOT / "docs" / "strategy"


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if _is_finite(v)]
    if not clean:
        return None
    return statistics.mean(clean)


def _median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if _is_finite(v)]
    if not clean:
        return None
    return statistics.median(clean)


def _pvariance(values: list[float]) -> float | None:
    clean = [float(v) for v in values if _is_finite(v)]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    return statistics.pvariance(clean)


def _topk_mean(values: list[float], k: int) -> float | None:
    clean = [float(v) for v in values if _is_finite(v)]
    if not clean:
        return None
    top = sorted(clean, reverse=True)[: max(1, min(k, len(clean)))]
    return statistics.mean(top)


def _topk_mass_fraction(values: list[float], k: int) -> float | None:
    clean = [max(0.0, float(v)) for v in values if _is_finite(v)]
    if not clean:
        return None
    total = sum(clean)
    if total <= 0:
        return 0.0
    top = sorted(clean, reverse=True)[: max(1, min(k, len(clean)))]
    return sum(top) / total


def _fraction(values: list[float], threshold: float) -> float | None:
    clean = [float(v) for v in values if _is_finite(v)]
    if not clean:
        return None
    return sum(1 for v in clean if v >= threshold) / len(clean)


def _rank(values: list[float]) -> list[float]:
    ordered = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rv = ((i + j - 1) / 2.0) + 1.0
        for k in range(i, j):
            ranks[ordered[k][1]] = rv
        i = j
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx = _rank(x)
    ry = _rank(y)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx)
    den_y = sum((b - my) ** 2 for b in ry)
    den = math.sqrt(den_x * den_y)
    if den <= 0.0:
        return None
    return num / den


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None or not _is_finite(float(v)):
        return "n/a"
    return f"{float(v):.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _is_attention_layer(layer_name: str, module_type: str) -> bool:
    token = f"{layer_name} {module_type}".lower()
    attention_markers = ("attention", "q_lin", "v_lin", "query", "value")
    return any(m in token for m in attention_markers)


def _feature_family(name: str) -> str:
    if name.startswith("current_"):
        return "current_pair"
    if name.startswith("raw_"):
        return "raw_subfactor"
    if name.startswith("interaction_"):
        return "interaction"
    if name.startswith("hotspot_"):
        return "hotspot"
    if name.startswith("burden_"):
        return "burden"
    if name.startswith("concentration_"):
        return "concentration_of_risk"
    if name.startswith("module_"):
        return "module_summary"
    if name.startswith("control_"):
        return "control"
    return "other"


def _interpretability_weight(family: str) -> float:
    if family in {"raw_subfactor", "current_pair", "hotspot"}:
        return 1.0
    if family in {"interaction", "burden", "module_summary"}:
        return 0.9
    if family in {"concentration_of_risk"}:
        return 0.85
    if family == "control":
        return 0.75
    return 0.8


def _median_split_stats(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) != len(y) or len(x) < 4:
        return {
            "n_high": 0,
            "n_low": 0,
            "median_x": None,
            "mean_delta_high": None,
            "mean_delta_low": None,
            "delta_gap_high_minus_low": None,
        }
    median_x = statistics.median(x)
    high = [yy for xx, yy in zip(x, y) if xx >= median_x]
    low = [yy for xx, yy in zip(x, y) if xx < median_x]
    if not high or not low:
        return {
            "n_high": len(high),
            "n_low": len(low),
            "median_x": median_x,
            "mean_delta_high": None,
            "mean_delta_low": None,
            "delta_gap_high_minus_low": None,
        }
    mh = statistics.mean(high)
    ml = statistics.mean(low)
    return {
        "n_high": len(high),
        "n_low": len(low),
        "median_x": median_x,
        "mean_delta_high": mh,
        "mean_delta_low": ml,
        "delta_gap_high_minus_low": mh - ml,
    }


def _tercile_split_stats(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) != len(y) or len(x) < 6:
        return {
            "n_top": 0,
            "n_bottom": 0,
            "top_mean_delta": None,
            "bottom_mean_delta": None,
            "top_minus_bottom": None,
        }
    n = len(x)
    idx_sorted = sorted(range(n), key=lambda i: x[i])
    k = max(2, n // 3)
    bot_idx = idx_sorted[:k]
    top_idx = idx_sorted[-k:]
    bot = [y[i] for i in bot_idx]
    top = [y[i] for i in top_idx]
    tb = statistics.mean(top) - statistics.mean(bot)
    return {
        "n_top": len(top),
        "n_bottom": len(bot),
        "top_mean_delta": statistics.mean(top),
        "bottom_mean_delta": statistics.mean(bot),
        "top_minus_bottom": tb,
    }


def _loo_spearman(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) != len(y) or len(x) < 5:
        return {
            "n_loo": 0,
            "median_spearman": None,
            "median_abs_spearman": None,
            "min_spearman": None,
            "max_spearman": None,
            "sign_consistency_vs_full": None,
        }
    full = _spearman(x, y)
    vals: list[float] = []
    for i in range(len(x)):
        xx = x[:i] + x[i + 1 :]
        yy = y[:i] + y[i + 1 :]
        rho = _spearman(xx, yy)
        if rho is not None:
            vals.append(rho)
    if not vals:
        return {
            "n_loo": 0,
            "median_spearman": None,
            "median_abs_spearman": None,
            "min_spearman": None,
            "max_spearman": None,
            "sign_consistency_vs_full": None,
        }
    full_sign = 0
    if full is not None:
        if full > 0:
            full_sign = 1
        elif full < 0:
            full_sign = -1
    sign_consistency = None
    if full_sign != 0:
        same = sum(1 for v in vals if (1 if v > 0 else -1 if v < 0 else 0) == full_sign)
        sign_consistency = same / len(vals)
    return {
        "n_loo": len(vals),
        "median_spearman": statistics.median(vals),
        "median_abs_spearman": statistics.median([abs(v) for v in vals]),
        "min_spearman": min(vals),
        "max_spearman": max(vals),
        "sign_consistency_vs_full": sign_consistency,
    }


def build_baseline_snapshot(
    activation: dict[str, Any],
    cohort: dict[str, Any],
    strict_analysis: dict[str, Any],
    deep_summary: dict[str, Any],
    cutpoint_sweep: dict[str, Any],
) -> dict[str, Any]:
    cohort_rows = list(cohort.get("cohort_rows", []))
    task_rel_counts: dict[str, int] = defaultdict(int)
    for row in cohort_rows:
        task_rel_counts[str(row.get("task_relationship", "unknown"))] += 1

    cutpoint_035 = None
    for row in cutpoint_sweep.get("threshold_grid", []):
        if _safe_float(row.get("threshold"), default=-1.0) == 0.35:
            cutpoint_035 = row
            break

    baseline = {
        "scope": "strict-naive targeted high-overlap refinement baseline",
        "current_oa_definition_snapshot": {
            "cutpoint_summary": activation.get("cutpoint_summary", {}),
            "pair_advisory_counts": activation.get("pair_advisory_counts", {}),
            "layer_band_counts": activation.get("layer_band_counts", {}),
        },
        "activation_audit_snapshot": {
            "audited_pair_count": activation.get("audit_scope", {}).get("audited_pair_count"),
            "layer_entry_count": activation.get("audit_scope", {}).get("layer_entry_count"),
            "pair_max_score_quantiles": activation.get("pair_max_score_quantiles"),
            "layer_score_quantiles": activation.get("layer_score_quantiles"),
            "pairs_with_max_score_ge_0_35": activation.get("cutpoint_summary", {}).get("pairs_with_max_score_ge_0_35"),
            "pairs_with_max_score_ge_0_60": activation.get("cutpoint_summary", {}).get("pairs_with_max_score_ge_0_60"),
            "pairs_with_max_score_ge_0_75": activation.get("cutpoint_summary", {}).get("pairs_with_max_score_ge_0_75"),
        },
        "targeted_strict_naive_cohort_snapshot": {
            "cohort_count": cohort.get("cohort_count"),
            "selection_parameters": cohort.get("selection_parameters"),
            "candidate_count": cohort.get("candidate_count"),
            "preferred_candidate_count": cohort.get("preferred_candidate_count"),
            "working_pool_count": cohort.get("working_pool_count"),
            "task_relationship_counts": dict(task_rel_counts),
        },
        "first_pass_findings_snapshot": {
            "summary": {
                "rerun_ok": deep_summary.get("rerun_ok"),
                "rerun_error": deep_summary.get("rerun_error"),
                "high_tail_mean_delta_vs_best": deep_summary.get("high_tail_mean_delta_vs_best"),
                "low_tail_mean_delta_vs_best": deep_summary.get("low_tail_mean_delta_vs_best"),
                "spearman_delta_vs_max_oa": deep_summary.get("spearman_delta_vs_max_oa"),
            },
            "strict_analysis": strict_analysis,
            "threshold_0_35_snapshot": cutpoint_035,
        },
        "current_interpretation": {
            "pair_activation_is_sparse": True,
            "pair_level_explanatory_power_is_weak": True,
            "concept_not_yet_falsified": True,
            "note": (
                "Current OA pair-level behavior appears sparse and blunt in strict-naive outcomes. "
                "Refinement should test subfactors/interactions/hotspots before line escalation."
            ),
        },
    }
    return baseline


def write_baseline_md(path: Path, baseline: dict[str, Any]) -> None:
    cdef = baseline["current_oa_definition_snapshot"]
    audit = baseline["activation_audit_snapshot"]
    cohort = baseline["targeted_strict_naive_cohort_snapshot"]
    fp = baseline["first_pass_findings_snapshot"]["summary"]
    th = baseline["first_pass_findings_snapshot"].get("threshold_0_35_snapshot") or {}

    lines = [
        "# Over-Accumulation Refinement Pass Baseline",
        "",
        "## Current OA Diagnostic State",
        f"- Pair advisory counts: {cdef.get('pair_advisory_counts', {})}",
        f"- Layer band counts: {cdef.get('layer_band_counts', {})}",
        f"- Cutpoint summary: {cdef.get('cutpoint_summary', {})}",
        "",
        "## Activation Audit Snapshot",
        f"- Audited pairs: {audit.get('audited_pair_count')}",
        f"- Layer entries: {audit.get('layer_entry_count')}",
        f"- Pair max score quantiles: {audit.get('pair_max_score_quantiles')}",
        f"- Layer score quantiles: {audit.get('layer_score_quantiles')}",
        (
            f"- Pair max score cutpoints: >=0.35={audit.get('pairs_with_max_score_ge_0_35')}, "
            f">=0.60={audit.get('pairs_with_max_score_ge_0_60')}, "
            f">=0.75={audit.get('pairs_with_max_score_ge_0_75')}"
        ),
        "",
        "## Strict-Naive Cohort Snapshot",
        f"- Cohort count: {cohort.get('cohort_count')}",
        f"- Selection parameters: {cohort.get('selection_parameters')}",
        (
            f"- Candidate pools: total={cohort.get('candidate_count')}, "
            f"preferred={cohort.get('preferred_candidate_count')}, "
            f"working={cohort.get('working_pool_count')}"
        ),
        f"- Task relationship mix: {cohort.get('task_relationship_counts')}",
        "",
        "## First-Pass + Deeper Findings Snapshot",
        f"- Strict reruns: ok={fp.get('rerun_ok')}, error={fp.get('rerun_error')}",
        (
            f"- Mean delta vs best: high-tail={_fmt(fp.get('high_tail_mean_delta_vs_best'), 4)}, "
            f"lower-tail={_fmt(fp.get('low_tail_mean_delta_vs_best'), 4)}"
        ),
        f"- Spearman(delta, pair max OA): {_fmt(fp.get('spearman_delta_vs_max_oa'), 4)}",
        (
            f"- Threshold 0.35 activation snapshot: pairs={th.get('activated_pair_count')}, "
            f"layers={th.get('activated_layer_count')}, rerun_n={th.get('rerun_activated_count')}"
        ),
        "",
        "## Current Interpretation",
        "- Pair-level activation is sparse under current thresholds.",
        "- Pair-level OA signal appears weak/unstable under strict-naive outcomes.",
        "- The line is not yet falsified but remains low-confidence without a sharper diagnostic signal.",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_refinement_feature_rows(
    activation: dict[str, Any], reruns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    pair_map = {str(row["pair_id"]): row for row in activation.get("all_pairs", [])}
    layers_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for layer in activation.get("all_layers", []):
        layers_by_pair[str(layer.get("pair_id", ""))].append(layer)

    out_rows: list[dict[str, Any]] = []
    for rr in reruns:
        if rr.get("status") != "ok":
            continue
        pair_id = str(rr["pair_id"])
        pair = pair_map.get(pair_id)
        if pair is None:
            continue
        layers = layers_by_pair.get(pair_id, [])

        scores = [_safe_float(l.get("score")) for l in layers]
        align = [_safe_float(l.get("alignment")) for l in layers]
        conc = [_safe_float(l.get("concentration")) for l in layers]
        coeff = [_safe_float(l.get("coefficient_exposure")) for l in layers]

        ac = [a * c for a, c in zip(align, conc)]
        ae = [a * e for a, e in zip(align, coeff)]
        ce = [c * e for c, e in zip(conc, coeff)]
        ace = [a * c * e for a, c, e in zip(align, conc, coeff)]

        attention_layers = [l for l in layers if _is_attention_layer(str(l.get("layer_name", "")), str(l.get("module_type", "")))]
        non_attention_layers = [l for l in layers if l not in attention_layers]
        attention_scores = [_safe_float(l.get("score")) for l in attention_layers]
        non_attention_scores = [_safe_float(l.get("score")) for l in non_attention_layers]

        merged_score = _safe_float((rr.get("merged_eval") or {}).get("merged_accuracy"))
        source_a = _safe_float((rr.get("source_a_eval") or {}).get("adapter_score"))
        source_b = _safe_float((rr.get("source_b_eval") or {}).get("adapter_score"))
        best_source = _safe_float(rr.get("best_source_score"))
        worst_source = _safe_float(rr.get("worst_source_score"))

        row = {
            "pair_id": pair_id,
            "cohort_group": rr.get("cohort_group"),
            "adapter_a_id": rr.get("adapter_a_id"),
            "adapter_b_id": rr.get("adapter_b_id"),
            "eval_dataset": rr.get("eval_dataset"),
            "model_family": str(pair.get("family")),
            "task_relationship": pair.get("task_relationship"),
            "dataset_family_pair": pair.get("dataset_family_pair"),
            "mean_overlap": _safe_float(pair.get("mean_overlap")),
            "conflict_fraction": _safe_float(pair.get("conflict_fraction")),
            "imbalance_fraction": _safe_float(pair.get("imbalance_fraction")),
            "overall_verdict": pair.get("overall_verdict"),
            "merge_delta_vs_best_source": _safe_float(rr.get("merge_delta_vs_best_source")),
            "merge_delta_vs_average_source": _safe_float(rr.get("merge_delta_vs_average_source")),
            "merge_delta_vs_worst_source": _safe_float(rr.get("merge_delta_vs_worst_source")),
            "merged_score": merged_score,
            "best_source_score": best_source,
            "worst_source_score": worst_source,
            "source_a_score": source_a,
            "source_b_score": source_b,
            "control_source_score_gap": (best_source - worst_source) if _is_finite(best_source) and _is_finite(worst_source) else None,
            "control_source_score_mean": _mean([source_a, source_b]),
            "control_source_score_var": _pvariance([source_a, source_b]),
            "current_pair_over_accumulation_advisory": pair.get("over_accumulation_advisory"),
            "current_max_over_accumulation_score": _safe_float(pair.get("max_over_accumulation_score")),
            "current_high_risk_layer_count": _safe_float(pair.get("high_risk_layer_count")),
            "current_watch_layer_count": _safe_float(pair.get("watch_layer_count")),
            "current_triple_high_layer_count": _safe_float(pair.get("triple_high_layer_count")),
            "current_triple_high_layer_fraction": _safe_float(pair.get("triple_high_layer_fraction")),
            "raw_mean_alignment": _mean(align),
            "raw_max_alignment": max([v for v in align if _is_finite(v)], default=None),
            "raw_mean_concentration": _mean(conc),
            "raw_max_concentration": max([v for v in conc if _is_finite(v)], default=None),
            "raw_mean_coefficient_exposure": _mean(coeff),
            "raw_max_coefficient_exposure": max([v for v in coeff if _is_finite(v)], default=None),
            "interaction_mean_align_x_concentration": _mean(ac),
            "interaction_max_align_x_concentration": max([v for v in ac if _is_finite(v)], default=None),
            "interaction_top3_align_x_concentration_mean": _topk_mean(ac, 3),
            "interaction_mean_align_x_exposure": _mean(ae),
            "interaction_max_align_x_exposure": max([v for v in ae if _is_finite(v)], default=None),
            "interaction_top3_align_x_exposure_mean": _topk_mean(ae, 3),
            "interaction_mean_concentration_x_exposure": _mean(ce),
            "interaction_max_concentration_x_exposure": max([v for v in ce if _is_finite(v)], default=None),
            "interaction_top3_concentration_x_exposure_mean": _topk_mean(ce, 3),
            "interaction_mean_align_x_concentration_x_exposure": _mean(ace),
            "interaction_max_align_x_concentration_x_exposure": max([v for v in ace if _is_finite(v)], default=None),
            "interaction_top3_align_x_concentration_x_exposure_mean": _topk_mean(ace, 3),
            "control_layer_count": len(layers),
            "hotspot_max_layer_oa_score": max([v for v in scores if _is_finite(v)], default=None),
            "hotspot_top1_layer_oa_mean": _topk_mean(scores, 1),
            "hotspot_top3_layer_oa_mean": _topk_mean(scores, 3),
            "hotspot_top5_layer_oa_mean": _topk_mean(scores, 5),
            "hotspot_top3_interaction_ace_mean": _topk_mean(ace, 3),
            "hotspot_top5_interaction_ace_mean": _topk_mean(ace, 5),
            "burden_frac_layers_score_ge_0_10": _fraction(scores, 0.10),
            "burden_frac_layers_score_ge_0_20": _fraction(scores, 0.20),
            "burden_frac_layers_score_ge_0_35": _fraction(scores, 0.35),
            "burden_frac_layers_score_ge_0_60": _fraction(scores, 0.60),
            "burden_frac_layers_align_ge_0_45_and_conc_ge_0_45": (
                sum(1 for a, c in zip(align, conc) if _is_finite(a) and _is_finite(c) and a >= 0.45 and c >= 0.45) / len(layers)
                if layers
                else None
            ),
            "burden_frac_layers_align_ge_0_45_and_exposure_ge_0_90": (
                sum(1 for a, e in zip(align, coeff) if _is_finite(a) and _is_finite(e) and a >= 0.45 and e >= 0.90)
                / len(layers)
                if layers
                else None
            ),
            "burden_frac_layers_conc_ge_0_45_and_exposure_ge_0_90": (
                sum(1 for c, e in zip(conc, coeff) if _is_finite(c) and _is_finite(e) and c >= 0.45 and e >= 0.90)
                / len(layers)
                if layers
                else None
            ),
            "burden_frac_layers_triple_high": (
                sum(
                    1
                    for a, c, e in zip(align, conc, coeff)
                    if _is_finite(a) and _is_finite(c) and _is_finite(e) and a >= 0.45 and c >= 0.45 and e >= 0.90
                )
                / len(layers)
                if layers
                else None
            ),
            "concentration_oa_mass_top1_fraction": _topk_mass_fraction(scores, 1),
            "concentration_oa_mass_top3_fraction": _topk_mass_fraction(scores, 3),
            "concentration_oa_mass_top5_fraction": _topk_mass_fraction(scores, 5),
            "concentration_ace_mass_top1_fraction": _topk_mass_fraction(ace, 1),
            "concentration_ace_mass_top3_fraction": _topk_mass_fraction(ace, 3),
            "concentration_ace_mass_top5_fraction": _topk_mass_fraction(ace, 5),
            "module_attention_layer_fraction": (len(attention_layers) / len(layers)) if layers else None,
            "module_attention_score_mean": _mean(attention_scores),
            "module_attention_top3_oa_mean": _topk_mean(attention_scores, 3),
            "module_attention_frac_score_ge_0_20": _fraction(attention_scores, 0.20),
            "module_non_attention_score_mean": _mean(non_attention_scores),
            "module_non_attention_top3_oa_mean": _topk_mean(non_attention_scores, 3),
            "module_non_attention_frac_score_ge_0_20": _fraction(non_attention_scores, 0.20),
        }
        out_rows.append(row)

    out_rows.sort(key=lambda r: (str(r.get("cohort_group")), str(r.get("pair_id"))))
    return out_rows


def write_feature_table_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# OA Refinement Feature Table",
        "",
        "Full numeric feature inventory is in `refinement_feature_table.json`.",
        "",
        "## Pair-Level Snapshot",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row.get("pair_id")),
                str(row.get("cohort_group")),
                str(row.get("eval_dataset")),
                _fmt(_safe_float(row.get("merge_delta_vs_best_source")), 4),
                _fmt(_safe_float(row.get("mean_overlap")), 3),
                _fmt(_safe_float(row.get("current_max_over_accumulation_score")), 3),
                _fmt(_safe_float(row.get("hotspot_top3_layer_oa_mean")), 3),
                _fmt(_safe_float(row.get("interaction_top3_align_x_concentration_x_exposure_mean")), 3),
                _fmt(_safe_float(row.get("burden_frac_layers_triple_high")), 3),
                _fmt(_safe_float(row.get("control_source_score_gap")), 3),
            ]
        )
    lines.append(
        _md_table(
            [
                "pair_id",
                "cohort",
                "eval_dataset",
                "delta_vs_best",
                "mean_overlap",
                "current_max_oa",
                "hotspot_top3_oa",
                "top3_interaction_ace",
                "triple_high_frac",
                "source_gap",
            ],
            table_rows,
        )
    )
    path.write_text("\n".join(lines) + "\n")


def analyze_refinement_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_key = "merge_delta_vs_best_source"
    y = [float(row[outcome_key]) for row in rows]

    excluded = {
        "pair_id",
        "cohort_group",
        "adapter_a_id",
        "adapter_b_id",
        "eval_dataset",
        "model_family",
        "task_relationship",
        "dataset_family_pair",
        "overall_verdict",
        "current_pair_over_accumulation_advisory",
    }

    allowed_families = {
        "current_pair",
        "raw_subfactor",
        "interaction",
        "hotspot",
        "burden",
        "concentration_of_risk",
        "module_summary",
        "control",
    }

    numeric_features: list[str] = []
    for key in rows[0].keys():
        if key in excluded or key == outcome_key:
            continue
        fam = _feature_family(key)
        if fam not in allowed_families:
            continue
        vals = [_safe_float(r.get(key)) for r in rows]
        clean = [v for v in vals if _is_finite(v)]
        if len(clean) < 4:
            continue
        if len(set(round(v, 10) for v in clean)) < 2:
            continue
        numeric_features.append(key)
    numeric_features.sort()

    metric_rows: list[dict[str, Any]] = []
    for feature in numeric_features:
        pairs_xy = []
        for row in rows:
            xv = _safe_float(row.get(feature))
            yv = _safe_float(row.get(outcome_key))
            if _is_finite(xv) and _is_finite(yv):
                pairs_xy.append((xv, yv))
        if len(pairs_xy) < 4:
            continue
        xvals = [p[0] for p in pairs_xy]
        yvals = [p[1] for p in pairs_xy]
        rho = _spearman(xvals, yvals)
        med = _median_split_stats(xvals, yvals)
        ter = _tercile_split_stats(xvals, yvals)
        loo = _loo_spearman(xvals, yvals)
        family = _feature_family(feature)
        abs_rho = abs(rho) if rho is not None else 0.0
        sign_cons = loo.get("sign_consistency_vs_full")
        if sign_cons is None:
            sign_cons = 0.5
        rank_score = abs_rho * (0.5 + 0.5 * sign_cons) * _interpretability_weight(family)
        metric_rows.append(
            {
                "feature": feature,
                "family": family,
                "n": len(xvals),
                "spearman_delta_vs_best": rho,
                "abs_spearman": abs_rho,
                "median_split": med,
                "tercile_split": ter,
                "loo": loo,
                "rank_score": rank_score,
                "risk_direction_consistent": (rho is not None and rho < 0.0),
            }
        )

    baseline_features = [
        "current_max_over_accumulation_score",
        "current_high_risk_layer_count",
        "current_watch_layer_count",
    ]
    baseline_rows = [r for r in metric_rows if r["feature"] in baseline_features]
    best_baseline_abs = max([r["abs_spearman"] for r in baseline_rows], default=0.0)

    oa_families = {
        "current_pair",
        "raw_subfactor",
        "interaction",
        "hotspot",
        "burden",
        "concentration_of_risk",
        "module_summary",
    }
    oa_rows = [r for r in metric_rows if r["family"] in oa_families]
    control_rows = [r for r in metric_rows if r["family"] == "control"]
    oa_rows_sorted = sorted(oa_rows, key=lambda r: (r["rank_score"], r["abs_spearman"]), reverse=True)
    control_rows_sorted = sorted(control_rows, key=lambda r: (r["rank_score"], r["abs_spearman"]), reverse=True)

    candidates_exceed_baseline = [
        r
        for r in oa_rows_sorted
        if r["feature"] not in baseline_features and r["abs_spearman"] > best_baseline_abs + 1e-12
    ]

    restricted_rows = [
        row
        for row in rows
        if _safe_float(row.get("mean_overlap"), default=0.0) >= 0.20
        and _safe_float(row.get("conflict_fraction"), default=1.0) <= 0.10
    ]
    restricted_top_checks: list[dict[str, Any]] = []
    if len(restricted_rows) >= 4:
        y_sub = [_safe_float(r.get(outcome_key)) for r in restricted_rows]
        top_features = [r["feature"] for r in oa_rows_sorted[:10]]
        for feat in top_features:
            x_sub = [_safe_float(r.get(feat)) for r in restricted_rows]
            pairs_sub = [(x, y) for x, y in zip(x_sub, y_sub) if _is_finite(x) and _is_finite(y)]
            if len(pairs_sub) < 4:
                continue
            rho_sub = _spearman([p[0] for p in pairs_sub], [p[1] for p in pairs_sub])
            restricted_top_checks.append({"feature": feat, "spearman_delta_vs_best": rho_sub, "n": len(pairs_sub)})

    top_oa = oa_rows_sorted[:15]
    top_control = control_rows_sorted[:5]

    return {
        "outcome_metric": outcome_key,
        "n_pairs": len(rows),
        "candidate_feature_count": len(metric_rows),
        "oa_feature_count": len(oa_rows),
        "control_feature_count": len(control_rows),
        "baseline_features": baseline_features,
        "baseline_rows": baseline_rows,
        "best_baseline_abs_spearman": best_baseline_abs,
        "top_ranked_oa_features": top_oa,
        "top_ranked_control_features": top_control,
        "candidates_exceeding_baseline": candidates_exceed_baseline,
        "restricted_subset": {
            "n_pairs": len(restricted_rows),
            "criteria": "mean_overlap >= 0.20 and conflict_fraction <= 0.10",
            "top_feature_checks": restricted_top_checks,
        },
        "all_feature_metrics": metric_rows,
    }


def write_refinement_analysis_md(path: Path, analysis: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# OA Refinement Analysis")
    lines.append("")
    lines.append(f"- Outcome metric: `{analysis['outcome_metric']}`")
    lines.append(f"- Pair count: {analysis['n_pairs']}")
    lines.append(
        f"- Features tested: total={analysis['candidate_feature_count']}, "
        f"oa={analysis['oa_feature_count']}, control={analysis['control_feature_count']}"
    )
    lines.append("")
    lines.append("## Baseline Comparison")
    base_rows = []
    for row in analysis.get("baseline_rows", []):
        ms = row.get("median_split", {})
        base_rows.append(
            [
                row["feature"],
                _fmt(row.get("spearman_delta_vs_best"), 4),
                _fmt(ms.get("delta_gap_high_minus_low"), 4),
                _fmt(row.get("loo", {}).get("sign_consistency_vs_full"), 3),
            ]
        )
    if base_rows:
        lines.append(
            _md_table(
                ["feature", "spearman", "median_split_gap", "loo_sign_consistency"],
                base_rows,
            )
        )
    lines.append(
        f"- Best baseline |Spearman|: {_fmt(analysis.get('best_baseline_abs_spearman'), 4)}"
    )
    lines.append("")
    lines.append("## Top OA Candidates")
    cand_rows = []
    for row in analysis.get("top_ranked_oa_features", [])[:12]:
        ms = row.get("median_split", {})
        cand_rows.append(
            [
                row["feature"],
                row["family"],
                _fmt(row.get("spearman_delta_vs_best"), 4),
                _fmt(ms.get("delta_gap_high_minus_low"), 4),
                _fmt(row.get("loo", {}).get("median_abs_spearman"), 4),
                _fmt(row.get("loo", {}).get("sign_consistency_vs_full"), 3),
                _fmt(row.get("rank_score"), 4),
            ]
        )
    lines.append(
        _md_table(
            [
                "feature",
                "family",
                "spearman",
                "median_split_gap",
                "loo_median_abs",
                "loo_sign_consistency",
                "rank_score",
            ],
            cand_rows,
        )
    )
    lines.append("")
    lines.append("## Features Exceeding Current Baseline")
    exceed_rows = analysis.get("candidates_exceeding_baseline", [])
    if exceed_rows:
        rows = []
        for row in exceed_rows[:12]:
            rows.append(
                [
                    row["feature"],
                    row["family"],
                    _fmt(row.get("spearman_delta_vs_best"), 4),
                    _fmt(row.get("abs_spearman"), 4),
                ]
            )
        lines.append(_md_table(["feature", "family", "spearman", "|spearman|"], rows))
    else:
        lines.append("- No OA candidates exceeded best baseline |Spearman| in this pass.")
    lines.append("")
    lines.append("## Restricted High-Overlap/Low-Conflict Check")
    rs = analysis.get("restricted_subset", {})
    lines.append(f"- Criteria: {rs.get('criteria')}")
    lines.append(f"- n={rs.get('n_pairs')}")
    if rs.get("top_feature_checks"):
        rs_rows = []
        for row in rs["top_feature_checks"][:10]:
            rs_rows.append([row["feature"], str(row["n"]), _fmt(row.get("spearman_delta_vs_best"), 4)])
        lines.append(_md_table(["feature", "n", "spearman"], rs_rows))
    lines.append("")
    lines.append("## Control Feature Context")
    ctrl = analysis.get("top_ranked_control_features", [])
    if ctrl:
        ctrl_rows = []
        for row in ctrl[:5]:
            ctrl_rows.append(
                [
                    row["feature"],
                    _fmt(row.get("spearman_delta_vs_best"), 4),
                    _fmt(row.get("abs_spearman"), 4),
                ]
            )
        lines.append(_md_table(["feature", "spearman", "|spearman|"], ctrl_rows))
    else:
        lines.append("- No control features were available.")
    path.write_text("\n".join(lines) + "\n")


def build_case_studies(rows: list[dict[str, Any]], analysis: dict[str, Any]) -> dict[str, Any]:
    in_region = [
        r
        for r in rows
        if _safe_float(r.get("mean_overlap"), default=0.0) >= 0.20
        and _safe_float(r.get("conflict_fraction"), default=1.0) <= 0.10
    ]
    disappointing = sorted([r for r in in_region if _safe_float(r.get("merge_delta_vs_best_source")) <= -0.05], key=lambda r: r["merge_delta_vs_best_source"])
    benign = sorted([r for r in in_region if _safe_float(r.get("merge_delta_vs_best_source")) >= -0.02], key=lambda r: r["merge_delta_vs_best_source"], reverse=True)

    disappointing = disappointing[:3]
    benign = benign[:3]

    top_features = [r["feature"] for r in analysis.get("top_ranked_oa_features", [])[:8]]
    if not top_features:
        top_features = [
            "current_max_over_accumulation_score",
            "hotspot_top3_layer_oa_mean",
            "interaction_top3_align_x_concentration_x_exposure_mean",
            "burden_frac_layers_triple_high",
        ]

    def _match_score(a: dict[str, Any], b: dict[str, Any]) -> float:
        score = 0.0
        if a.get("eval_dataset") == b.get("eval_dataset"):
            score += 4.0
        if a.get("task_relationship") == b.get("task_relationship"):
            score += 2.0
        if a.get("dataset_family_pair") == b.get("dataset_family_pair"):
            score += 2.0
        if a.get("model_family") == b.get("model_family"):
            score += 1.0
        ov_a = _safe_float(a.get("mean_overlap"), default=0.0)
        ov_b = _safe_float(b.get("mean_overlap"), default=0.0)
        score -= abs(ov_a - ov_b)
        return score

    matches: list[dict[str, Any]] = []
    used_benign: set[str] = set()
    for d in disappointing:
        candidates = [b for b in benign if b["pair_id"] not in used_benign]
        if not candidates:
            candidates = benign
        if not candidates:
            continue
        best = sorted(candidates, key=lambda b: _match_score(d, b), reverse=True)[0]
        used_benign.add(best["pair_id"])

        diffs = []
        for feat in top_features:
            dv = _safe_float(d.get(feat))
            bv = _safe_float(best.get(feat))
            if not _is_finite(dv) or not _is_finite(bv):
                continue
            diffs.append({"feature": feat, "disappointing_minus_benign": dv - bv, "disappointing": dv, "benign": bv})
        diffs = sorted(diffs, key=lambda x: abs(x["disappointing_minus_benign"]), reverse=True)[:5]

        matches.append(
            {
                "disappointing_pair": d,
                "benign_pair": best,
                "top_feature_differences": diffs,
            }
        )

    return {
        "selection_criteria": {
            "high_overlap_region": "mean_overlap >= 0.20 and conflict_fraction <= 0.10",
            "disappointing_threshold": "merge_delta_vs_best_source <= -0.05",
            "benign_threshold": "merge_delta_vs_best_source >= -0.02",
        },
        "disappointing_candidates": disappointing,
        "benign_candidates": benign,
        "matched_comparisons": matches,
    }


def write_case_studies_md(path: Path, cases: dict[str, Any]) -> None:
    lines = []
    lines.append("# OA Refinement Matched Case Studies")
    lines.append("")
    lines.append("## Selection")
    lines.append(f"- {cases.get('selection_criteria')}")
    lines.append(f"- Disappointing candidates: {len(cases.get('disappointing_candidates', []))}")
    lines.append(f"- Benign candidates: {len(cases.get('benign_candidates', []))}")
    lines.append("")

    comparisons = cases.get("matched_comparisons", [])
    if not comparisons:
        lines.append("No matched comparisons were available under current thresholds and cohort outcomes.")
        path.write_text("\n".join(lines) + "\n")
        return

    for i, comp in enumerate(comparisons, start=1):
        d = comp["disappointing_pair"]
        b = comp["benign_pair"]
        lines.append(f"## Comparison {i}")
        lines.append(
            f"- Disappointing: `{d['pair_id']}` | delta={_fmt(d.get('merge_delta_vs_best_source'), 4)} | "
            f"overlap={_fmt(d.get('mean_overlap'), 3)} | max_oa={_fmt(d.get('current_max_over_accumulation_score'), 3)}"
        )
        lines.append(
            f"- Benign: `{b['pair_id']}` | delta={_fmt(b.get('merge_delta_vs_best_source'), 4)} | "
            f"overlap={_fmt(b.get('mean_overlap'), 3)} | max_oa={_fmt(b.get('current_max_over_accumulation_score'), 3)}"
        )
        lines.append(
            f"- Match context: task_relation({d.get('task_relationship')} vs {b.get('task_relationship')}), "
            f"dataset({d.get('eval_dataset')} vs {b.get('eval_dataset')}), model_family({d.get('model_family')} vs {b.get('model_family')})"
        )
        lines.append("- Top distinguishing OA features (disappointing minus benign):")
        diff_rows = []
        for diff in comp.get("top_feature_differences", []):
            diff_rows.append(
                [
                    diff["feature"],
                    _fmt(diff["disappointing"], 4),
                    _fmt(diff["benign"], 4),
                    _fmt(diff["disappointing_minus_benign"], 4),
                ]
            )
        if diff_rows:
            lines.append(_md_table(["feature", "disappointing", "benign", "delta"], diff_rows))
        else:
            lines.append("- No valid feature differences for this pair.")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def assess_refined_candidate(analysis: dict[str, Any], cases: dict[str, Any]) -> dict[str, Any]:
    baseline_abs = _safe_float(analysis.get("best_baseline_abs_spearman"), default=0.0)
    top_oa = analysis.get("top_ranked_oa_features", [])
    baseline_features = set(analysis.get("baseline_features", []))

    refined_candidates = [r for r in top_oa if r["feature"] not in baseline_features]
    best_refined = refined_candidates[0] if refined_candidates else (top_oa[0] if top_oa else None)

    decision = "pause"
    rationale: list[str] = []
    evidence_strength = "weak"

    top_control = (analysis.get("top_ranked_control_features") or [None])[0]

    if best_refined is not None:
        rho = _safe_float(best_refined.get("spearman_delta_vs_best"), default=0.0)
        abs_rho = abs(rho)
        loo = best_refined.get("loo", {})
        sign_cons = _safe_float(loo.get("sign_consistency_vs_full"), default=0.0)
        loo_med_abs = _safe_float(loo.get("median_abs_spearman"), default=0.0)
        med_gap = _safe_float(best_refined.get("median_split", {}).get("delta_gap_high_minus_low"), default=math.nan)
        improvement = abs_rho - baseline_abs
        risk_consistent = bool(best_refined.get("risk_direction_consistent"))
        outlier_sensitive = abs(rho - _safe_float(loo.get("median_spearman"), default=rho)) > 0.15

        rationale.append(
            f"Best refined candidate `{best_refined['feature']}` has Spearman={_fmt(rho,4)} "
            f"(improvement over best baseline |rho|={_fmt(improvement,4)})."
        )
        rationale.append(
            f"Stability: LOO median |rho|={_fmt(loo_med_abs,4)}, sign consistency={_fmt(sign_cons,3)}, "
            f"outlier_sensitive={outlier_sensitive}."
        )
        rationale.append(
            f"Median split gap (high minus low) on delta_vs_best={_fmt(med_gap,4)}; "
            f"risk-direction-consistent={risk_consistent}."
        )

        if (
            improvement >= 0.12
            and risk_consistent
            and sign_cons >= 0.70
            and loo_med_abs >= 0.20
            and _is_finite(med_gap)
            and med_gap <= -0.03
            and not outlier_sensitive
        ):
            decision = "refine"
            evidence_strength = "moderate"
            rationale.append("Candidate exceeds baseline with direction/stability good enough for a bounded OA v2 diagnostic.")
        elif (
            improvement >= 0.05
            and (abs_rho >= 0.25 or loo_med_abs >= 0.15)
            and len(cases.get("matched_comparisons", [])) >= 2
        ):
            decision = "keep_exploratory"
            evidence_strength = "suggestive"
            rationale.append(
                "Refined candidates show some directional signal, but stability/coverage is insufficient for confident promotion."
            )
        else:
            decision = "pause"
            evidence_strength = "weak"
            rationale.append("No refined candidate achieved enough explanatory strength and stability to justify continued emphasis.")

        if top_control is not None:
            control_abs = _safe_float(top_control.get("abs_spearman"), default=0.0)
            if control_abs > abs_rho + 0.15:
                rationale.append(
                    f"Control feature `{top_control.get('feature')}` has materially stronger |Spearman|={_fmt(control_abs,4)}, "
                    "indicating non-OA confounding remains substantial."
                )
                if decision == "refine":
                    decision = "keep_exploratory"
                    evidence_strength = "suggestive"
                    rationale.append("Decision downgraded from refine to keep_exploratory due confound dominance.")
    else:
        rationale.append("No analyzable refined feature candidates were available.")

    return {
        "decision": decision,
        "evidence_strength": evidence_strength,
        "best_refined_candidate": best_refined,
        "best_baseline_abs_spearman": baseline_abs,
        "matched_case_count": len(cases.get("matched_comparisons", [])),
        "rationale": rationale,
        "options": {
            "refine": "Adopt a candidate OA v2 feature in next implementation pass.",
            "keep_exploratory": "Retain OA line as low-confidence exploratory; collect stronger evidence before promotion.",
            "pause": "De-emphasize OA line until better data/theory appears.",
        },
    }


def write_candidate_assessment_md(path: Path, assessment: dict[str, Any]) -> None:
    lines = []
    lines.append("# OA Refined Candidate Assessment")
    lines.append("")
    lines.append(f"- Decision: **{assessment.get('decision')}**")
    lines.append(f"- Evidence strength: **{assessment.get('evidence_strength')}**")
    lines.append(
        f"- Matched comparisons available: {assessment.get('matched_case_count')}"
    )
    lines.append(
        f"- Best baseline |Spearman|: {_fmt(assessment.get('best_baseline_abs_spearman'), 4)}"
    )
    lines.append("")
    brc = assessment.get("best_refined_candidate")
    if brc:
        lines.append("## Best Refined Candidate")
        lines.append(f"- Feature: `{brc.get('feature')}` ({brc.get('family')})")
        lines.append(f"- Spearman(delta_vs_best): {_fmt(brc.get('spearman_delta_vs_best'), 4)}")
        lines.append(
            f"- LOO sign consistency: {_fmt(brc.get('loo', {}).get('sign_consistency_vs_full'), 3)}"
        )
        lines.append(f"- LOO median |Spearman|: {_fmt(brc.get('loo', {}).get('median_abs_spearman'), 4)}")
        lines.append(
            f"- Median split gap: {_fmt(brc.get('median_split', {}).get('delta_gap_high_minus_low'), 4)}"
        )
        lines.append("")
    lines.append("## Rationale")
    for item in assessment.get("rationale", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Decision Options")
    for key, val in assessment.get("options", {}).items():
        lines.append(f"- `{key}`: {val}")
    path.write_text("\n".join(lines) + "\n")


def write_refinement_decision_memo(
    path: Path,
    baseline: dict[str, Any],
    analysis: dict[str, Any],
    assessment: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Over-Accumulation Refinement Decision Memo")
    lines.append("")
    lines.append("## 1) What Was Tested")
    lines.append(
        "- We decomposed OA into raw subfactors, interactions, hotspot metrics, risk-concentration metrics, and optional module summaries."
    )
    lines.append(
        "- We tested rank relationships and coarse split behavior against strict-naive `merge_delta_vs_best_source`."
    )
    lines.append(
        "- We compared all refined OA candidates against current pair-level OA baselines."
    )
    lines.append("")
    lines.append("## 2) What Improved (If Anything)")
    top = (analysis.get("top_ranked_oa_features") or [None])[0]
    if top is not None:
        lines.append(
            f"- Top OA candidate: `{top['feature']}` with Spearman={_fmt(top.get('spearman_delta_vs_best'), 4)} "
            f"and LOO sign consistency={_fmt(top.get('loo', {}).get('sign_consistency_vs_full'), 3)}."
        )
        lines.append(
            f"- Best baseline |Spearman| was {_fmt(analysis.get('best_baseline_abs_spearman'), 4)}."
        )
    else:
        lines.append("- No meaningful OA candidate outperformed baseline.")
    lines.append("")
    lines.append("## 3) What Remained Weak")
    lines.append("- Pair-level activation remains sparse under current thresholds.")
    lines.append("- Small cohort size limits confidence and increases outlier sensitivity.")
    lines.append("- Directional stability of candidate features is still limited.")
    lines.append("- The strongest OA concentration signal points in a non-risk direction in this cohort.")
    lines.append("- Control covariates (especially source score gap) remain stronger than OA-derived features.")
    lines.append("")
    lines.append("## 4) Decision")
    lines.append(f"- **{assessment.get('decision')}**")
    lines.append(f"- Evidence strength: **{assessment.get('evidence_strength')}**")
    lines.append("")
    lines.append("## 5) If Refine, What Exactly Changes")
    if assessment.get("decision") == "refine":
        brc = assessment.get("best_refined_candidate") or {}
        lines.append(
            f"- Promote `{brc.get('feature')}` as OA-v2 candidate signal, replacing broad pair averaging emphasis."
        )
        lines.append("- Keep current taxonomy unchanged; expose candidate as additive diagnostic line.")
        lines.append("- Re-run strict-naive validation with fixed candidate definition before any policy changes.")
    else:
        lines.append("- Not selected in this pass.")
        lines.append("- If selected later: promote one risk-direction-consistent hotspot/interaction feature as OA-v2 candidate.")
        lines.append("- Keep current taxonomy unchanged and re-run strict-naive validation before policy updates.")
    lines.append("")
    lines.append("## 5b) Chosen Path Next Steps")
    if assessment.get("decision") == "keep_exploratory":
        lines.append("- Keep OA as low-confidence exploratory signal.")
        lines.append("- Retain current diagnostic but avoid stronger prominence claims.")
        lines.append("- Build a larger high-overlap strict-naive cohort with stronger top-tail coverage.")
        lines.append("- Re-run this exact decomposition/interactions/hotspot analysis before any OA-v2 implementation.")
    elif assessment.get("decision") == "pause":
        lines.append("- Pause OA emphasis in user-facing strategy guidance.")
        lines.append("- Preserve artifacts for future revisit, but avoid further implementation investment now.")
    else:
        lines.append("- Proceed with a bounded OA-v2 implementation and immediate follow-up validation.")
    lines.append("")
    lines.append("## 6) If Pause, What Would Revive The Line")
    lines.append("- Broader strict-naive high-overlap cohort with stronger top-tail score coverage.")
    lines.append("- Cleaner matched cases with lower source-quality confounding.")
    lines.append("- Improved theoretical framing for benign alignment vs harmful concentration interactions.")
    path.write_text("\n".join(lines) + "\n")


def write_strategy_summary(path: Path, assessment: dict[str, Any], analysis: dict[str, Any]) -> None:
    lines = []
    lines.append("# Over-Accumulation Refinement Summary")
    lines.append("")
    lines.append("## Status")
    lines.append(f"- Decision: **{assessment.get('decision')}**")
    lines.append(f"- Evidence strength: **{assessment.get('evidence_strength')}**")
    lines.append("")
    lines.append("## Key Evidence")
    lines.append(
        f"- Best baseline |Spearman|: {_fmt(analysis.get('best_baseline_abs_spearman'), 4)}"
    )
    top = (analysis.get("top_ranked_oa_features") or [None])[0]
    if top is not None:
        lines.append(
            f"- Top OA candidate: `{top['feature']}` | Spearman={_fmt(top.get('spearman_delta_vs_best'),4)} | "
            f"LOO sign consistency={_fmt(top.get('loo', {}).get('sign_consistency_vs_full'), 3)}"
        )
    ctrl = (analysis.get("top_ranked_control_features") or [None])[0]
    if ctrl is not None:
        lines.append(
            f"- Top control feature: `{ctrl['feature']}` | Spearman={_fmt(ctrl.get('spearman_delta_vs_best'),4)}"
        )
    lines.append("- Current pass remains diagnostic-only and CPU-only.")
    lines.append("")
    lines.append("## Interpretation")
    if assessment.get("decision") == "refine":
        lines.append("- A narrower OA candidate appears strong enough for a bounded OA-v2 implementation attempt.")
    elif assessment.get("decision") == "keep_exploratory":
        lines.append("- OA remains plausible but not validated strongly enough for stronger policy prominence.")
        lines.append("- The strongest OA concentration signal is directionally non-risk in this cohort.")
        lines.append("- Non-OA controls remain stronger than OA-derived features.")
    else:
        lines.append("- No robust refined OA signal was recovered; line should be de-emphasized until stronger evidence appears.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outcome",
        default="merge_delta_vs_best_source",
        help="Outcome metric to use (currently informational; analysis is fixed to merge_delta_vs_best_source).",
    )
    args = parser.parse_args()

    _ = args.outcome
    DOCS_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    activation = _json(FOLLOWUP_DIR / "activation_threshold_audit.json")
    cohort = _json(FOLLOWUP_DIR / "targeted_high_overlap_cohort.json")
    reruns = _json(FOLLOWUP_DIR / "strict_naive_rerun_results.json")
    strict_analysis = _json(FOLLOWUP_DIR / "strict_naive_explanatory_analysis.json")
    deep_summary = _json(FOLLOWUP_DIR / "deeper_diagnostic_summary.json")
    cutpoint_sweep = _json(FOLLOWUP_DIR / "cutpoint_sweep.json")

    # Stage A
    baseline = build_baseline_snapshot(activation, cohort, strict_analysis, deep_summary, cutpoint_sweep)
    _write_json(FOLLOWUP_DIR / "refinement_pass_baseline.json", baseline)
    write_baseline_md(FOLLOWUP_DIR / "refinement_pass_baseline.md", baseline)

    # Stage B
    feature_rows = build_refinement_feature_rows(activation, reruns)
    _write_json(FOLLOWUP_DIR / "refinement_feature_table.json", feature_rows)
    write_feature_table_md(FOLLOWUP_DIR / "refinement_feature_table.md", feature_rows)

    # Stage C
    analysis = analyze_refinement_features(feature_rows)
    _write_json(FOLLOWUP_DIR / "refinement_analysis.json", analysis)
    write_refinement_analysis_md(FOLLOWUP_DIR / "refinement_analysis.md", analysis)

    # Stage D
    cases = build_case_studies(feature_rows, analysis)
    _write_json(FOLLOWUP_DIR / "refinement_case_studies.json", cases)
    write_case_studies_md(FOLLOWUP_DIR / "refinement_case_studies.md", cases)

    # Stage E
    assessment = assess_refined_candidate(analysis, cases)
    _write_json(FOLLOWUP_DIR / "refined_candidate_assessment.json", assessment)
    write_candidate_assessment_md(FOLLOWUP_DIR / "refined_candidate_assessment.md", assessment)

    # Stage F
    write_refinement_decision_memo(
        FOLLOWUP_DIR / "refinement_decision_memo.md",
        baseline,
        analysis,
        assessment,
    )
    write_strategy_summary(
        DOCS_STRATEGY_DIR / "over_accumulation_refinement_summary.md",
        assessment,
        analysis,
    )

    summary = {
        "decision": assessment.get("decision"),
        "evidence_strength": assessment.get("evidence_strength"),
        "n_feature_rows": len(feature_rows),
        "n_candidates": analysis.get("candidate_feature_count"),
        "best_baseline_abs_spearman": analysis.get("best_baseline_abs_spearman"),
        "best_oa_feature": (analysis.get("top_ranked_oa_features") or [{}])[0].get("feature"),
        "best_oa_feature_spearman": (analysis.get("top_ranked_oa_features") or [{}])[0].get(
            "spearman_delta_vs_best"
        ),
    }
    _write_json(FOLLOWUP_DIR / "refinement_decision_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
