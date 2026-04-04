#!/usr/bin/env python3
"""Direction-aware compatibility reanalysis on strict-naive merge cohort.

This is a bounded CPU-only follow-up that tests whether directional structure
(top-k and band-partitioned singular alignment/conflict) adds explanatory value
beyond coarse merge-audit summaries.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gradience.vnext.merge import merge_audit
from gradience.vnext.merge.spectral_compat import SubspaceMetrics

STRICT_RESULTS = Path("field_trials/over_accumulation_followup/oa_v2_30_40_r1_strict_naive_rerun_results.json")
OUT_DIR = Path("sidecar/results/direction_aware_compatibility")
NOTES_DIR = Path("sidecar/notes")
COARSE_ENERGY_THRESHOLD = 0.90
DIRECTIONAL_ENERGY_THRESHOLD = 0.999

BASELINE_MD = NOTES_DIR / "n133_direction_aware_compatibility_baseline.md"
COHORT_JSON = OUT_DIR / "cohort_manifest.json"
TABLE_JSON = OUT_DIR / "directional_metric_table.json"
ANALYSIS_MD = OUT_DIR / "comparative_analysis.md"
CASE_MD = OUT_DIR / "case_studies.md"
VERDICT_MD = NOTES_DIR / "n134_direction_aware_compatibility_verdict.md"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    family: str  # coarse | directional
    risk_high: bool


METRICS: list[MetricSpec] = [
    MetricSpec("coarse_oa_v1_max", "OA-v1 max", "coarse", True),
    MetricSpec("coarse_mean_overlap", "Mean overlap", "coarse", False),
    MetricSpec("coarse_mean_agreement", "Mean directional agreement", "coarse", False),
    MetricSpec("coarse_conflict_fraction", "Conflict fraction", "coarse", True),
    MetricSpec("coarse_imbalance_fraction", "Imbalance fraction", "coarse", True),
    MetricSpec("dir_align_top1_top3", "Top-1 directional alignment (top3-mean)", "directional", False),
    MetricSpec("dir_align_top4_top3", "Top-4 directional alignment (top3-mean)", "directional", False),
    MetricSpec("dir_align_head_top3", "Head-band alignment (top3-mean)", "directional", False),
    MetricSpec("dir_align_middle_top3", "Middle-band alignment (top3-mean)", "directional", False),
    MetricSpec("dir_risk_middle_disagreement_max", "Middle-band disagreement (max)", "directional", True),
    MetricSpec("dir_risk_weighted_disagreement_max", "Weighted disagreement (max)", "directional", True),
    MetricSpec("dir_gap_head_minus_middle_max", "Head-minus-middle alignment gap (max)", "directional", True),
    MetricSpec("dir_risk_conflict_concentration_max", "Conflict concentration top3-fraction (max)", "directional", True),
    MetricSpec("dir_risk_signed_middle_conflict_max", "Signed middle-conflict burden (max)", "directional", True),
    MetricSpec("dir_risk_middle_disagreement_top3", "Middle-band disagreement (top3-mean)", "directional", True),
]


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def _mean(vals: list[float]) -> float | None:
    clean = [v for v in vals if math.isfinite(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _topk_mean(vals: list[float], k: int) -> float | None:
    clean = [v for v in vals if math.isfinite(v)]
    if not clean:
        return None
    top = sorted(clean, reverse=True)[: max(1, min(k, len(clean)))]
    return sum(top) / len(top)


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
    denx = sum((a - mx) ** 2 for a in rx)
    deny = sum((b - my) ** 2 for b in ry)
    den = math.sqrt(denx * deny)
    if den <= 0.0:
        return None
    return num / den


def _zscore(vals: list[float]) -> list[float]:
    if not vals:
        return []
    m = statistics.mean(vals)
    s = statistics.pstdev(vals)
    if s <= 1e-12:
        return [0.0 for _ in vals]
    return [(v - m) / s for v in vals]


def _terciles(vals: list[float]) -> tuple[float, float]:
    clean = sorted(v for v in vals if math.isfinite(v))
    if not clean:
        return (math.nan, math.nan)
    n = len(clean)
    i1 = max(0, min(n - 1, int(round((n - 1) * (1 / 3)))))
    i2 = max(0, min(n - 1, int(round((n - 1) * (2 / 3)))))
    return (clean[i1], clean[i2])


def _outcome_bin(delta: float) -> str:
    if not math.isfinite(delta):
        return "unknown"
    if delta <= -0.10:
        return "catastrophic"
    if delta <= -0.05:
        return "fragile"
    if delta <= -0.02:
        return "near_miss"
    return "retained"


def _to_subspace_metrics(metrics_dict: dict[str, Any]) -> SubspaceMetrics:
    d = dict(metrics_dict)
    for k in (
        "principal_angle_cosines",
        "right_principal_angle_cosines",
        "effective_singular_values_a",
        "effective_singular_values_b",
    ):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return SubspaceMetrics(**d)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return math.nan
    wsum = sum(weights)
    if wsum <= 1e-12:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / wsum


def _direction_metrics(m: SubspaceMetrics) -> dict[str, float | int | bool]:
    left = [_clamp01(v) for v in (m.principal_angle_cosines or ())]
    right = [_clamp01(v) for v in (m.right_principal_angle_cosines or ())]
    if not right:
        right = left[:]

    sa = [max(0.0, float(v)) for v in (m.effective_singular_values_a or ())]
    sb = [max(0.0, float(v)) for v in (m.effective_singular_values_b or ())]

    if not sa or not sb:
        # fallback if old payload lacks singular spectra
        s_len = min(len(left), len(right))
        sa = [1.0] * s_len
        sb = [1.0] * s_len

    k = min(len(left), len(right), len(sa), len(sb))
    if k <= 0:
        fallback = _clamp01(0.5 * m.mean_overlap + 0.5 * m.max_overlap)
        neg_gate = _clamp01(max(-m.directional_agreement, 0.0))
        return {
            "k": 0,
            "degraded": True,
            "align_top1": fallback,
            "align_top2_mean": fallback,
            "align_top4_mean": fallback,
            "align_weighted": fallback,
            "align_head": fallback,
            "align_middle": fallback,
            "align_tail": fallback,
            "risk_middle_disagreement": 1.0 - fallback,
            "risk_weighted_disagreement": 1.0 - fallback,
            "gap_head_minus_middle": 0.0,
            "risk_conflict_concentration": 1.0,
            "risk_signed_middle_conflict": neg_gate * (1.0 - fallback),
        }

    pair = [left[i] * right[i] for i in range(k)]
    w_raw = [sa[i] * sb[i] for i in range(k)]
    wsum = sum(w_raw)
    if wsum <= 1e-12:
        w = [1.0 / k] * k
        degraded = True
    else:
        w = [x / wsum for x in w_raw]
        degraded = False

    # band by index proportion
    head_end = max(1, math.ceil(0.2 * k))
    mid_end = max(head_end + 1, math.ceil(0.5 * k)) if k > 1 else head_end
    mid_end = min(mid_end, k)

    head_pair = pair[:head_end]
    mid_pair = pair[head_end:mid_end] if mid_end > head_end else []
    tail_pair = pair[mid_end:] if mid_end < k else []

    head_w = w[:head_end]
    mid_w = w[head_end:mid_end] if mid_end > head_end else []
    tail_w = w[mid_end:] if mid_end < k else []

    align_head = _weighted_mean(head_pair, head_w)
    align_middle = _weighted_mean(mid_pair, mid_w) if mid_pair else align_head
    align_tail = _weighted_mean(tail_pair, tail_w) if tail_pair else align_middle

    conflict_mass = [w[i] * (1.0 - pair[i]) for i in range(k)]
    cm_total = sum(conflict_mass)
    if cm_total <= 1e-12:
        conflict_top3_frac = 0.0
    else:
        conflict_top3_frac = sum(sorted(conflict_mass, reverse=True)[: min(3, len(conflict_mass))]) / cm_total

    neg_gate = _clamp01(max(-m.directional_agreement, 0.0))

    return {
        "k": k,
        "degraded": degraded,
        "align_top1": pair[0],
        "align_top2_mean": sum(pair[: min(2, k)]) / min(2, k),
        "align_top4_mean": sum(pair[: min(4, k)]) / min(4, k),
        "align_weighted": _weighted_mean(pair, w),
        "align_head": align_head,
        "align_middle": align_middle,
        "align_tail": align_tail,
        "risk_middle_disagreement": 1.0 - align_middle,
        "risk_weighted_disagreement": 1.0 - _weighted_mean(pair, w),
        "gap_head_minus_middle": max(0.0, align_head - align_middle),
        "risk_conflict_concentration": conflict_top3_frac,
        "risk_signed_middle_conflict": neg_gate * (1.0 - align_middle),
    }


def _poor_enrichment(rows: list[dict[str, Any]], metric_key: str, risk_high: bool) -> dict[str, float | int | None]:
    n = len(rows)
    if n < 3:
        return {
            "n_alerted": 0,
            "poor_rate_alerted": None,
            "poor_rate_rest": None,
            "lift": None,
            "recall": None,
        }

    k = max(1, n // 3)
    idx = sorted(range(n), key=lambda i: float(rows[i][metric_key]), reverse=risk_high)
    alerted = set(idx[:k])

    poor_total = sum(1 for r in rows if bool(r["is_poor"]))
    poor_alerted = sum(1 for i in alerted if bool(rows[i]["is_poor"]))
    poor_rest = sum(1 for i in range(n) if i not in alerted and bool(rows[i]["is_poor"]))

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


def main() -> int:
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    rows_in = json.loads(STRICT_RESULTS.read_text())
    rows = [r for r in rows_in if r.get("status") == "ok" and (r.get("strict_naive_check") or {}).get("ok")]

    pair_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for r in rows:
        pair_id = str(r.get("pair_id"))
        a_path = (r.get("source_a_eval") or {}).get("adapter_path")
        b_path = (r.get("source_b_eval") or {}).get("adapter_path")
        if not a_path or not b_path:
            errors.append({"pair_id": pair_id, "error": "missing_adapter_path"})
            continue

        try:
            report_coarse = merge_audit(
                str(a_path),
                str(b_path),
                energy_threshold=COARSE_ENERGY_THRESHOLD,
                verbose=False,
            )
            report_dir = merge_audit(
                str(a_path),
                str(b_path),
                energy_threshold=DIRECTIONAL_ENERGY_THRESHOLD,
                verbose=False,
            )
        except Exception as e:  # pragma: no cover
            errors.append({"pair_id": pair_id, "error": str(e)})
            continue

        agg = report_coarse.aggregate
        layers = list(report_dir.layer_verdicts or [])
        n_layers = max(1, len(layers))

        layer_metric_rows: list[dict[str, Any]] = []
        for ld in layers:
            m = _to_subspace_metrics(ld.get("metrics", {}))
            dm = _direction_metrics(m)
            row = {
                "pair_id": pair_id,
                "layer_name": ld.get("layer_name"),
                "module_type": ld.get("module_type"),
                "verdict": ld.get("verdict"),
                "over_accumulation_score": _safe_float(ld.get("over_accumulation_score"), 0.0),
                "mean_overlap": _safe_float(m.mean_overlap, 0.0),
                "max_overlap": _safe_float(m.max_overlap, 0.0),
                "directional_agreement": _safe_float(m.directional_agreement, 0.0),
                "stable_rank_a": _safe_float(m.stable_rank_a, 1.0),
                "stable_rank_b": _safe_float(m.stable_rank_b, 1.0),
                "effective_rank_a": int(m.effective_rank_a),
                "effective_rank_b": int(m.effective_rank_b),
                "k": int(dm["k"]),
                "degraded": bool(dm["degraded"]),
                "align_top1": float(dm["align_top1"]),
                "align_top2_mean": float(dm["align_top2_mean"]),
                "align_top4_mean": float(dm["align_top4_mean"]),
                "align_weighted": float(dm["align_weighted"]),
                "align_head": float(dm["align_head"]),
                "align_middle": float(dm["align_middle"]),
                "align_tail": float(dm["align_tail"]),
                "risk_middle_disagreement": float(dm["risk_middle_disagreement"]),
                "risk_weighted_disagreement": float(dm["risk_weighted_disagreement"]),
                "gap_head_minus_middle": float(dm["gap_head_minus_middle"]),
                "risk_conflict_concentration": float(dm["risk_conflict_concentration"]),
                "risk_signed_middle_conflict": float(dm["risk_signed_middle_conflict"]),
            }
            layer_metric_rows.append(row)
            layer_rows.append(row)

        def _vals(key: str) -> list[float]:
            return [float(x[key]) for x in layer_metric_rows if math.isfinite(_safe_float(x.get(key)))]

        delta = _safe_float(r.get("merge_delta_vs_best_source"))
        pair_row = {
            "pair_id": pair_id,
            "cohort_group": r.get("cohort_group"),
            "eval_dataset": r.get("eval_dataset"),
            "base_model_id": r.get("base_model_id"),
            "adapter_a_name": r.get("adapter_a_name"),
            "adapter_b_name": r.get("adapter_b_name"),
            "merge_delta_vs_best_source": delta,
            "is_poor": bool(delta <= -0.05),
            "outcome_bin": _outcome_bin(delta),
            "source_gap": _safe_float(r.get("best_source_score")) - _safe_float(r.get("worst_source_score")),
            "n_layers": len(layer_metric_rows),
            "degraded_layer_fraction": (
                sum(1 for x in layer_metric_rows if x.get("degraded")) / len(layer_metric_rows) if layer_metric_rows else 1.0
            ),
            "coarse_oa_v1_max": _safe_float(getattr(agg, "max_over_accumulation_score", 0.0), 0.0),
            "coarse_mean_overlap": _safe_float(getattr(agg, "mean_overlap", 0.0), 0.0),
            "coarse_mean_agreement": _safe_float(getattr(agg, "mean_agreement", 0.0), 0.0),
            "coarse_conflict_fraction": _safe_float(getattr(agg, "n_conflicting", 0), 0.0) / n_layers,
            "coarse_imbalance_fraction": _safe_float(getattr(agg, "n_imbalanced", 0), 0.0) / n_layers,
            "dir_align_top1_top3": _topk_mean(_vals("align_top1"), 3),
            "dir_align_top4_top3": _topk_mean(_vals("align_top4_mean"), 3),
            "dir_align_head_top3": _topk_mean(_vals("align_head"), 3),
            "dir_align_middle_top3": _topk_mean(_vals("align_middle"), 3),
            "dir_risk_middle_disagreement_max": max(_vals("risk_middle_disagreement"), default=math.nan),
            "dir_risk_weighted_disagreement_max": max(_vals("risk_weighted_disagreement"), default=math.nan),
            "dir_gap_head_minus_middle_max": max(_vals("gap_head_minus_middle"), default=math.nan),
            "dir_risk_conflict_concentration_max": max(_vals("risk_conflict_concentration"), default=math.nan),
            "dir_risk_signed_middle_conflict_max": max(_vals("risk_signed_middle_conflict"), default=math.nan),
            "dir_risk_middle_disagreement_top3": _topk_mean(_vals("risk_middle_disagreement"), 3),
        }

        pair_rows.append(pair_row)

    # Cohort manifest
    pair_rows_sorted = sorted(pair_rows, key=lambda x: (x["merge_delta_vs_best_source"], x["pair_id"]))
    bins = Counter(x["outcome_bin"] for x in pair_rows_sorted)
    datasets = Counter(x["eval_dataset"] for x in pair_rows_sorted)

    retained = [x["pair_id"] for x in pair_rows_sorted if x["outcome_bin"] == "retained"]
    fragile_or_worse = [x["pair_id"] for x in pair_rows_sorted if x["outcome_bin"] in {"fragile", "catastrophic"}]

    manifest = {
        "generated_at_utc": now_iso,
        "source": str(STRICT_RESULTS),
        "n_pairs": len(pair_rows_sorted),
        "cohort_groups": dict(Counter(x["cohort_group"] for x in pair_rows_sorted)),
        "outcome_bins": dict(bins),
        "datasets": dict(datasets),
        "selected_subsets": {
            "retained_subset_pair_ids": retained[:10],
            "fragile_or_catastrophic_subset_pair_ids": fragile_or_worse[:10],
            "near_miss_subset_pair_ids": [x["pair_id"] for x in pair_rows_sorted if x["outcome_bin"] == "near_miss"][:10],
        },
        "pairs": pair_rows_sorted,
    }

    # Comparative analysis
    intended = [
        x
        for x in pair_rows_sorted
        if _safe_float(x.get("coarse_mean_overlap"), 0.0) >= 0.25
        and _safe_float(x.get("coarse_conflict_fraction"), 1.0) <= 0.10
    ]

    def _metric_analysis(slice_rows: list[dict[str, Any]], spec: MetricSpec) -> dict[str, Any]:
        usable = [r for r in slice_rows if math.isfinite(_safe_float(r.get(spec.key)))]
        x = [float(r[spec.key]) for r in usable]
        y = [float(r["merge_delta_vs_best_source"]) for r in usable]
        spearman = _spearman(x, y)
        enrich = _poor_enrichment(usable, spec.key, risk_high=spec.risk_high)

        catastrophic = [float(r[spec.key]) for r in usable if r["outcome_bin"] == "catastrophic"]
        retained_vals = [float(r[spec.key]) for r in usable if r["outcome_bin"] == "retained"]

        effect = None
        if catastrophic and retained_vals:
            mc = statistics.mean(catastrophic)
            mr = statistics.mean(retained_vals)
            # orient by risk direction for readability
            effect = (mc - mr) if spec.risk_high else (mr - mc)

        return {
            "metric": spec.key,
            "label": spec.label,
            "family": spec.family,
            "risk_high": spec.risk_high,
            "n": len(usable),
            "spearman_vs_delta": spearman,
            "poor_enrichment": enrich,
            "catastrophic_vs_retained_effect_oriented": effect,
        }

    analysis_all = [_metric_analysis(pair_rows_sorted, s) for s in METRICS]
    analysis_intended = [_metric_analysis(intended, s) for s in METRICS]

    # ranking helper
    for row in analysis_all:
        sp = row["spearman_vs_delta"]
        row["abs_spearman"] = abs(sp) if sp is not None else None
    for row in analysis_intended:
        sp = row["spearman_vs_delta"]
        row["abs_spearman"] = abs(sp) if sp is not None else None

    top_directional_intended = sorted(
        [r for r in analysis_intended if r["family"] == "directional" and r["abs_spearman"] is not None],
        key=lambda r: (r["abs_spearman"], r["poor_enrichment"].get("lift") or -999),
        reverse=True,
    )[:5]

    top_coarse_intended = sorted(
        [r for r in analysis_intended if r["family"] == "coarse" and r["abs_spearman"] is not None],
        key=lambda r: (r["abs_spearman"], r["poor_enrichment"].get("lift") or -999),
        reverse=True,
    )[:5]

    # Case selection: coarse-ambiguous with divergent outcomes
    oa_vals = [float(r["coarse_oa_v1_max"]) for r in pair_rows_sorted if math.isfinite(_safe_float(r["coarse_oa_v1_max"]))]
    t1, t2 = _terciles(oa_vals)
    ambiguous = [r for r in pair_rows_sorted if t1 <= float(r["coarse_oa_v1_max"]) <= t2]

    poor_ambiguous = sorted(
        [r for r in ambiguous if r["merge_delta_vs_best_source"] <= -0.05],
        key=lambda r: float(r["dir_risk_middle_disagreement_max"]),
        reverse=True,
    )
    retained_ambiguous = sorted(
        [r for r in ambiguous if r["merge_delta_vs_best_source"] >= -0.02],
        key=lambda r: float(r["dir_risk_middle_disagreement_max"]),
    )

    case_pairs = poor_ambiguous[:2] + retained_ambiguous[:2]

    case_details: list[dict[str, Any]] = []
    for cp in case_pairs:
        pid = cp["pair_id"]
        lrows = [lr for lr in layer_rows if lr["pair_id"] == pid]
        hotspots = sorted(lrows, key=lambda x: float(x["risk_middle_disagreement"]), reverse=True)[:3]
        case_details.append(
            {
                "pair_id": pid,
                "eval_dataset": cp["eval_dataset"],
                "outcome_delta": cp["merge_delta_vs_best_source"],
                "outcome_bin": cp["outcome_bin"],
                "coarse_oa_v1_max": cp["coarse_oa_v1_max"],
                "coarse_mean_overlap": cp["coarse_mean_overlap"],
                "coarse_conflict_fraction": cp["coarse_conflict_fraction"],
                "dir_risk_middle_disagreement_max": cp["dir_risk_middle_disagreement_max"],
                "dir_gap_head_minus_middle_max": cp["dir_gap_head_minus_middle_max"],
                "hotspot_layers": [
                    {
                        "layer_name": h["layer_name"],
                        "module_type": h["module_type"],
                        "risk_middle_disagreement": h["risk_middle_disagreement"],
                        "align_head": h["align_head"],
                        "align_middle": h["align_middle"],
                        "directional_agreement": h["directional_agreement"],
                    }
                    for h in hotspots
                ],
            }
        )

    # Write JSON table artifact
    table = {
        "generated_at_utc": now_iso,
        "source": str(STRICT_RESULTS),
        "coarse_energy_threshold": COARSE_ENERGY_THRESHOLD,
        "directional_energy_threshold": DIRECTIONAL_ENERGY_THRESHOLD,
        "n_pairs": len(pair_rows_sorted),
        "n_layers": len(layer_rows),
        "n_errors": len(errors),
        "errors": errors,
        "metrics_catalog": [s.__dict__ for s in METRICS],
        "pair_rows": pair_rows_sorted,
        "layer_rows": layer_rows,
        "analysis": {
            "all_pairs": analysis_all,
            "intended_high_overlap_low_conflict": analysis_intended,
            "top_directional_intended": top_directional_intended,
            "top_coarse_intended": top_coarse_intended,
        },
        "case_selection": {
            "coarse_oa_v1_terciles": {"lower": t1, "upper": t2},
            "n_ambiguous_pairs": len(ambiguous),
            "selected_cases": case_details,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    COHORT_JSON.write_text(json.dumps(manifest, indent=2))
    TABLE_JSON.write_text(json.dumps(table, indent=2))

    # Baseline note (n133)
    lines = [
        "# N133 — Direction-Aware Compatibility Baseline",
        "",
        f"- Generated: `{now_iso}`",
        f"- Source cohort: `{STRICT_RESULTS}`",
        f"- Strict-naive pairs: `{len(pair_rows_sorted)}`",
        f"- Coarse audit threshold: `{COARSE_ENERGY_THRESHOLD}`",
        f"- Direction-aware threshold: `{DIRECTIONAL_ENERGY_THRESHOLD}`",
        "",
        "## Frozen Baseline",
        f"- Outcome bins: `{dict(bins)}`",
        f"- Dataset mix: `{dict(datasets)}`",
        f"- Cohort groups: `{dict(Counter(x['cohort_group'] for x in pair_rows_sorted))}`",
        "",
        "## Study Focus",
        "- Test whether direction-aware structure (top-k + band-partitioned) explains variance beyond coarse summaries.",
        "- Keep this as bounded reanalysis only (no policy/verdict replacement).",
        "",
        "## Candidate Metric Families",
        "- Top-band directional alignment (`top1`, `top4`, head-band).",
        "- Mid-spectrum disagreement and head-minus-middle gaps.",
        "- Conflict concentration in small directional subsets.",
    ]
    BASELINE_MD.write_text("\n".join(lines) + "\n")

    # Comparative markdown
    def _fmt(x: Any, nd: int = 4) -> str:
        if x is None:
            return "n/a"
        try:
            xf = float(x)
        except Exception:
            return str(x)
        if not math.isfinite(xf):
            return "n/a"
        return f"{xf:.{nd}f}"

    md = [
        "# Direction-Aware Compatibility Comparative Analysis",
        "",
        f"- Generated: `{now_iso}`",
        f"- Pairs analyzed: `{len(pair_rows_sorted)}`",
        f"- Intended high-overlap/low-conflict slice: `{len(intended)}`",
        f"- Coarse threshold: `{COARSE_ENERGY_THRESHOLD}` | Direction-aware threshold: `{DIRECTIONAL_ENERGY_THRESHOLD}`",
        "",
        "## Top Metrics In Intended Slice",
        "",
        "### Directional",
        "| Metric | |rho| | rho(delta) | Poor-lift | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in top_directional_intended:
        pe = r["poor_enrichment"]
        md.append(
            f"| {r['label']} | {_fmt(r['abs_spearman'])} | {_fmt(r['spearman_vs_delta'])} | "
            f"{_fmt(pe.get('lift'))} | {_fmt(pe.get('recall'))} |"
        )

    md += [
        "",
        "### Coarse",
        "| Metric | |rho| | rho(delta) | Poor-lift | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in top_coarse_intended:
        pe = r["poor_enrichment"]
        md.append(
            f"| {r['label']} | {_fmt(r['abs_spearman'])} | {_fmt(r['spearman_vs_delta'])} | "
            f"{_fmt(pe.get('lift'))} | {_fmt(pe.get('recall'))} |"
        )

    md += [
        "",
        "## Snapshot Interpretation",
        "- Compare directional and coarse rankings in the intended slice first.",
        "- Treat small-N shifts as bounded; this pass is explanatory, not policy-authoritative.",
        "",
        "## Artifacts",
        f"- Cohort manifest: `{COHORT_JSON}`",
        f"- Full metric table: `{TABLE_JSON}`",
    ]
    ANALYSIS_MD.write_text("\n".join(md) + "\n")

    # Case studies markdown
    case_md = [
        "# Direction-Aware Compatibility Case Studies",
        "",
        "Selection rule: coarse OA-v1 in middle tercile, then contrasting outcomes by directional middle-disagreement.",
        "",
    ]
    for c in case_details:
        case_md += [
            f"## `{c['pair_id']}`",
            f"- Dataset: `{c['eval_dataset']}`",
            f"- Outcome: `{c['outcome_bin']}` (delta vs best: `{_fmt(c['outcome_delta'])}`)",
            f"- Coarse: OA-v1 max `{_fmt(c['coarse_oa_v1_max'])}`, mean overlap `{_fmt(c['coarse_mean_overlap'])}`, conflict frac `{_fmt(c['coarse_conflict_fraction'])}`",
            f"- Directional: mid-disagreement max `{_fmt(c['dir_risk_middle_disagreement_max'])}`, head-mid gap max `{_fmt(c['dir_gap_head_minus_middle_max'])}`",
            "- Hotspot layers:",
        ]
        for h in c["hotspot_layers"]:
            case_md.append(
                f"  - `{h['layer_name']}` ({h['module_type']}): mid-disagreement `{_fmt(h['risk_middle_disagreement'])}`, "
                f"head `{_fmt(h['align_head'])}`, middle `{_fmt(h['align_middle'])}`, dir-agreement `{_fmt(h['directional_agreement'])}`"
            )
        case_md.append("")

    CASE_MD.write_text("\n".join(case_md) + "\n")

    # Verdict note (n134)
    top_dir = top_directional_intended[0] if top_directional_intended else None
    top_coarse = top_coarse_intended[0] if top_coarse_intended else None

    verdict_lines = [
        "# N134 — Direction-Aware Compatibility Verdict",
        "",
        f"- Generated: `{now_iso}`",
        f"- Cohort: `{len(pair_rows_sorted)}` strict-naive pairs",
        "",
        "## Result",
    ]
    if top_dir and top_coarse:
        verdict_lines.append(
            f"- Best directional metric in intended slice: `{top_dir['metric']}` (|rho|={_fmt(top_dir['abs_spearman'])}, lift={_fmt(top_dir['poor_enrichment'].get('lift'))})."
        )
        verdict_lines.append(
            f"- Best coarse metric in intended slice: `{top_coarse['metric']}` (|rho|={_fmt(top_coarse['abs_spearman'])}, lift={_fmt(top_coarse['poor_enrichment'].get('lift'))})."
        )
        if (top_dir.get("abs_spearman") or 0.0) > (top_coarse.get("abs_spearman") or 0.0):
            verdict_lines.append("- Preliminary interpretation: bounded directional refinement signal is present.")
        else:
            verdict_lines.append("- Preliminary interpretation: coarse summaries remain primary in this slice.")
    else:
        verdict_lines.append("- Insufficient data to rank directional vs coarse metrics.")

    verdict_lines += [
        "",
        "## Implications",
        "- Keep top-energy/coarse summaries as default.",
        "- Retain direction-aware metrics as bounded explanatory companions for fragile/ambiguous cases.",
        "- No policy or threshold changes from this pass alone.",
    ]
    VERDICT_MD.write_text("\n".join(verdict_lines) + "\n")

    print(f"Wrote: {COHORT_JSON}")
    print(f"Wrote: {TABLE_JSON}")
    print(f"Wrote: {ANALYSIS_MD}")
    print(f"Wrote: {CASE_MD}")
    print(f"Wrote: {BASELINE_MD}")
    print(f"Wrote: {VERDICT_MD}")
    print(f"Pairs: {len(pair_rows_sorted)}, Layers: {len(layer_rows)}, Errors: {len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
