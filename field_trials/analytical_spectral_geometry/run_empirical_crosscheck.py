#!/usr/bin/env python3
"""Strict-naive empirical cross-check for OA-v1 vs OA-v2 (analysis-only)."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gradience.vnext.merge import merge_audit
from gradience.vnext.merge.over_accumulation import estimate_over_accumulation_v2
from gradience.vnext.merge.spectral_compat import SubspaceMetrics


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _topk_mean(vals: list[float], k: int) -> float:
    if not vals:
        return 0.0
    top = sorted(vals, reverse=True)[: max(1, min(k, len(vals)))]
    return _mean(top)


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
        "poor_rate_alerted": rate_alerted,
        "poor_rate_rest": rate_rest,
        "lift": rate_alerted - rate_rest,
        "recall": recall,
    }


def _sign_consistency_loo(x: list[float], y: list[float]) -> dict[str, Any]:
    full = _spearman(x, y)
    if full is None or len(x) < 5:
        return {"full_spearman": full, "n_loo": 0, "sign_consistency": None}
    full_sign = 1 if full > 0 else -1 if full < 0 else 0
    vals = []
    for i in range(len(x)):
        xx = x[:i] + x[i + 1 :]
        yy = y[:i] + y[i + 1 :]
        rho = _spearman(xx, yy)
        if rho is not None:
            vals.append(rho)
    if not vals or full_sign == 0:
        return {"full_spearman": full, "n_loo": len(vals), "sign_consistency": None}
    same = sum(1 for v in vals if (1 if v > 0 else -1 if v < 0 else 0) == full_sign)
    return {"full_spearman": full, "n_loo": len(vals), "sign_consistency": same / len(vals)}


def _sign_consistency_bootstrap(x: list[float], y: list[float], trials: int = 400, seed: int = 7) -> dict[str, Any]:
    full = _spearman(x, y)
    if full is None or len(x) < 5:
        return {"full_spearman": full, "n_boot": 0, "sign_consistency": None}
    rng = random.Random(seed)
    n = len(x)
    full_sign = 1 if full > 0 else -1 if full < 0 else 0
    vals = []
    for _ in range(trials):
        idx = [rng.randrange(n) for _ in range(n)]
        xx = [x[i] for i in idx]
        yy = [y[i] for i in idx]
        rho = _spearman(xx, yy)
        if rho is not None:
            vals.append(rho)
    if not vals or full_sign == 0:
        return {"full_spearman": full, "n_boot": len(vals), "sign_consistency": None}
    same = sum(1 for v in vals if (1 if v > 0 else -1 if v < 0 else 0) == full_sign)
    return {"full_spearman": full, "n_boot": len(vals), "sign_consistency": same / len(vals)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--strict-naive-json",
        type=Path,
        default=Path("field_trials/over_accumulation_followup/strict_naive_rerun_results.json"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("field_trials/analytical_spectral_geometry/empirical_crosscheck.json"),
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=Path("field_trials/analytical_spectral_geometry/empirical_crosscheck.md"),
    )
    p.add_argument("--poor-threshold", type=float, default=-0.05)
    p.add_argument("--subset-overlap-min", type=float, default=0.25)
    p.add_argument("--subset-conflict-max", type=float, default=0.10)
    p.add_argument("--cohort-min-pairs", type=int, default=30)
    p.add_argument("--cohort-max-pairs", type=int, default=40)
    p.add_argument("--label", type=str, default="strict_naive_30_40")
    p.add_argument("--gate-report-json", type=Path, default=None)
    p.add_argument("--gate-report-md", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows_in = json.loads(args.strict_naive_json.read_text())
    pair_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped_not_strict = 0

    for row in rows_in:
        if str(row.get("status")) != "ok":
            continue
        strict_ok = bool((row.get("strict_naive_check") or {}).get("ok"))
        if not strict_ok:
            skipped_not_strict += 1
            continue
        a_path = row.get("source_a_eval", {}).get("adapter_path")
        b_path = row.get("source_b_eval", {}).get("adapter_path")
        if not a_path or not b_path:
            continue
        try:
            report = merge_audit(str(a_path), str(b_path), verbose=False)
            agg = report.aggregate
            layer_dicts = list(report.layer_verdicts or [])
            n_layers = max(len(layer_dicts), 1)

            v1_scores = [float(ld.get("over_accumulation_score", 0.0)) for ld in layer_dicts]
            v2_scores: list[float] = []
            v2_interaction_primary: list[float] = []
            for ld in layer_dicts:
                m = _to_subspace_metrics(ld.get("metrics", {}))
                est_v2 = estimate_over_accumulation_v2(m, assumed_coefficients=(0.5, 0.5))
                v2_scores.append(float(est_v2.score))
                v2_interaction_primary.append(float(est_v2.factors.interaction_primary))

            pair_rows.append(
                {
                    "pair_id": row.get("pair_id"),
                    "merge_delta_vs_best_source": float(row.get("merge_delta_vs_best_source", math.nan)),
                    "source_score_gap": float(row.get("best_source_score", math.nan))
                    - float(row.get("worst_source_score", math.nan)),
                    "mean_overlap": float(getattr(agg, "mean_overlap", 0.0)),
                    "conflict_fraction": float(getattr(agg, "n_conflicting", 0) / n_layers),
                    "imbalance_fraction": float(getattr(agg, "n_imbalanced", 0) / n_layers),
                    "oa_v1_max_score": max(v1_scores) if v1_scores else 0.0,
                    "oa_v1_top3_score_mean": _topk_mean(v1_scores, 3),
                    "oa_v2_max_score": max(v2_scores) if v2_scores else 0.0,
                    "oa_v2_top3_interaction": _topk_mean(v2_interaction_primary, 3),
                    "is_poor": float(row.get("merge_delta_vs_best_source", 0.0)) <= float(args.poor_threshold),
                }
            )
        except Exception as e:  # pragma: no cover - artifact capture
            errors.append({"pair_id": row.get("pair_id"), "error": str(e)})

    y = [float(r["merge_delta_vs_best_source"]) for r in pair_rows]
    x_v1 = [float(r["oa_v1_max_score"]) for r in pair_rows]
    x_v2 = [float(r["oa_v2_max_score"]) for r in pair_rows]
    x_v2_top3 = [float(r["oa_v2_top3_interaction"]) for r in pair_rows]

    overall = {
        "n_pairs": len(pair_rows),
        "spearman_delta_vs_oa_v1_max": _spearman(x_v1, y),
        "spearman_delta_vs_oa_v2_max": _spearman(x_v2, y),
        "spearman_delta_vs_oa_v2_top3_interaction": _spearman(x_v2_top3, y),
    }

    subset = [
        r
        for r in pair_rows
        if float(r["mean_overlap"]) >= float(args.subset_overlap_min)
        and float(r["conflict_fraction"]) <= float(args.subset_conflict_max)
    ]
    y_s = [float(r["merge_delta_vs_best_source"]) for r in subset]
    x1_s = [float(r["oa_v1_max_score"]) for r in subset]
    x2_s = [float(r["oa_v2_max_score"]) for r in subset]
    x2t_s = [float(r["oa_v2_top3_interaction"]) for r in subset]
    subset_stats = {
        "n_pairs": len(subset),
        "spearman_delta_vs_oa_v1_max": _spearman(x1_s, y_s),
        "spearman_delta_vs_oa_v2_max": _spearman(x2_s, y_s),
        "spearman_delta_vs_oa_v2_top3_interaction": _spearman(x2t_s, y_s),
    }

    enrich_v1 = _poor_enrichment(pair_rows, "oa_v1_max_score")
    enrich_v2 = _poor_enrichment(pair_rows, "oa_v2_max_score")
    recall_gain = None
    if enrich_v1.get("recall") is not None and enrich_v2.get("recall") is not None:
        recall_gain = float(enrich_v2["recall"]) - float(enrich_v1["recall"])

    loo = _sign_consistency_loo(x2_s, y_s)
    boot = _sign_consistency_bootstrap(x2_s, y_s)

    interpretability_ok = all(
        {"oa_v2_max_score", "oa_v2_top3_interaction"}.issubset(set(r.keys())) for r in pair_rows
    )
    abs_gain = None
    if subset_stats["spearman_delta_vs_oa_v1_max"] is not None and subset_stats["spearman_delta_vs_oa_v2_max"] is not None:
        abs_gain = abs(float(subset_stats["spearman_delta_vs_oa_v2_max"])) - abs(
            float(subset_stats["spearman_delta_vs_oa_v1_max"])
        )

    gate = {
        "rule_1_abs_spearman_gain_ge_0_15": bool(abs_gain is not None and abs_gain >= 0.15),
        "rule_1_abs_spearman_gain_value": abs_gain,
        "rule_2_recall_gain_ge_0_20": bool(recall_gain is not None and recall_gain >= 0.20),
        "rule_2_recall_gain_value": recall_gain,
        "rule_3_sign_consistency_ge_0_70": bool(
            loo.get("sign_consistency") is not None and float(loo["sign_consistency"]) >= 0.70
        ),
        "rule_3_sign_consistency_value": loo.get("sign_consistency"),
        "rule_4_interpretability_decomposition_ok": bool(interpretability_ok),
    }
    gate["all_pass"] = all(
        [
            gate["rule_1_abs_spearman_gain_ge_0_15"],
            gate["rule_2_recall_gain_ge_0_20"],
            gate["rule_3_sign_consistency_ge_0_70"],
            gate["rule_4_interpretability_decomposition_ok"],
        ]
    )
    cohort_design_gate = {
        "cohort_min_pairs": int(args.cohort_min_pairs),
        "cohort_max_pairs": int(args.cohort_max_pairs),
        "n_pairs_in_crosscheck": len(pair_rows),
        "in_target_range": bool(int(args.cohort_min_pairs) <= len(pair_rows) <= int(args.cohort_max_pairs)),
        "skipped_non_strict_rows": skipped_not_strict,
    }

    payload = {
        "status": "empirical_crosscheck_completed",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "strict_naive_cohort_only",
        "params": {
            "poor_threshold": args.poor_threshold,
            "subset_overlap_min": args.subset_overlap_min,
            "subset_conflict_max": args.subset_conflict_max,
        },
        "n_ok_pairs": len(pair_rows),
        "n_errors": len(errors),
        "errors": errors,
        "pair_rows": pair_rows,
        "analysis": {
            "cohort_design_gate": cohort_design_gate,
            "overall": overall,
            "high_overlap_low_conflict_subset": subset_stats,
            "poor_merge_enrichment": {
                "v1_top_tercile": enrich_v1,
                "v2_top_tercile": enrich_v2,
                "recall_gain_v2_minus_v1": recall_gain,
            },
            "sign_consistency": {
                "loo": loo,
                "bootstrap": boot,
            },
            "threshold_policy_gate": gate,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))

    gate_report_json = args.gate_report_json or args.out_json.with_name(f"{args.out_json.stem}_gate_report.json")
    gate_report_md = args.gate_report_md or args.out_md.with_name(f"{args.out_md.stem}_gate_report.md")
    gate_report = {
        "status": "oa_v2_gate_report_generated",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": args.label,
        "input_strict_naive_json": str(args.strict_naive_json),
        "cohort_design_gate": cohort_design_gate,
        "threshold_policy_gate": gate,
        "analysis_snapshot": {
            "overall_spearman_delta_vs_v1": overall["spearman_delta_vs_oa_v1_max"],
            "overall_spearman_delta_vs_v2": overall["spearman_delta_vs_oa_v2_max"],
            "subset_spearman_delta_vs_v1": subset_stats["spearman_delta_vs_oa_v1_max"],
            "subset_spearman_delta_vs_v2": subset_stats["spearman_delta_vs_oa_v2_max"],
            "recall_gain_v2_minus_v1": recall_gain,
            "sign_consistency_loo": loo.get("sign_consistency"),
        },
        "promotion_ready": bool(cohort_design_gate["in_target_range"] and gate["all_pass"]),
    }
    gate_report_json.parent.mkdir(parents=True, exist_ok=True)
    gate_report_json.write_text(json.dumps(gate_report, indent=2))

    md = f"""# OA-v2 Empirical Cross-Check (Strict-Naive Cohort)

- pairs analyzed: `{len(pair_rows)}`
- errors: `{len(errors)}`
- cohort design gate (30-40): `{cohort_design_gate['in_target_range']}` (n={len(pair_rows)})
- overall Spearman (delta vs OA-v1 max): `{overall['spearman_delta_vs_oa_v1_max']}`
- overall Spearman (delta vs OA-v2 max): `{overall['spearman_delta_vs_oa_v2_max']}`
- subset Spearman (delta vs OA-v1 max): `{subset_stats['spearman_delta_vs_oa_v1_max']}`
- subset Spearman (delta vs OA-v2 max): `{subset_stats['spearman_delta_vs_oa_v2_max']}`
- gate pass: `{gate['all_pass']}`

OA-v1 remains authoritative unless all gate conditions pass.
"""
    args.out_md.write_text(md)
    gate_report_md.write_text(
        "\n".join(
            [
                "# OA-v2 Gate Report",
                "",
                f"- label: `{args.label}`",
                f"- strict-naive input: `{args.strict_naive_json}`",
                f"- cohort range target: `{args.cohort_min_pairs}-{args.cohort_max_pairs}`",
                f"- analyzed pairs: `{len(pair_rows)}`",
                f"- cohort design gate pass: `{cohort_design_gate['in_target_range']}`",
                f"- threshold/policy gate pass: `{gate['all_pass']}`",
                f"- promotion ready: `{gate_report['promotion_ready']}`",
                "",
                "## Rule Status",
                f"- Rule 1 abs Spearman gain >= 0.15: `{gate['rule_1_abs_spearman_gain_ge_0_15']}` "
                f"(value={gate['rule_1_abs_spearman_gain_value']})",
                f"- Rule 2 recall gain >= 0.20: `{gate['rule_2_recall_gain_ge_0_20']}` "
                f"(value={gate['rule_2_recall_gain_value']})",
                f"- Rule 3 sign consistency >= 0.70: `{gate['rule_3_sign_consistency_ge_0_70']}` "
                f"(value={gate['rule_3_sign_consistency_value']})",
                f"- Rule 4 interpretability decomposition: `{gate['rule_4_interpretability_decomposition_ok']}`",
                "",
                "OA-v1 remains authoritative unless threshold/policy gate passes.",
                "",
            ]
        )
    )

    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "gate_report_json": str(gate_report_json),
                "n_pairs": len(pair_rows),
                "cohort_design_gate_pass": cohort_design_gate["in_target_range"],
                "gate_pass": gate["all_pass"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
