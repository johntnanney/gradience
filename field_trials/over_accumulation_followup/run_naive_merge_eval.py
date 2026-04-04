#!/usr/bin/env python3
"""Over-Accumulation Follow-up: bounded retrospective validation pipeline.

This script builds the Stage A-F artifacts for the over-accumulation empirical
validation follow-up from existing local field-trial outputs.

Design intent:
- Reuse existing merge/eval artifacts when fresh benchmark runs are not feasible.
- Recompute *current* merge-audit diagnostics (including over-accumulation) for
  the exact adapter pairs in the historical cohort.
- Keep outputs explicit about scope, caveats, and whether rows are strict
  naive-linear equivalents.

Outputs written under:
  field_trials/over_accumulation_followup/
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_TRIALS_DIR = REPO_ROOT / "field_trials"
FOLLOWUP_DIR = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gradience.vnext.merge import merge_audit


PREFERRED_EVAL_PATH_HINTS: tuple[tuple[str, int], ...] = (
    ("/field_trials/phase2_eval_130608/", 50),
    ("/field_trials/targeted_confirmation_same_family/eval_t01/", 45),
    ("/field_trials/targeted_confirmation_near_miss/eval_t02/", 40),
    ("/field_trials/task_family_equivalence/eval_090340/", 35),
    ("/field_trials/task_family_equivalence/eval_090227/", 30),
)

POOR_DELTA_THRESHOLD = -0.05


@dataclass
class PairRecord:
    pair_id: str
    eval_result_path: Path
    merge_plan_path: Path
    merge_result_path: Path | None
    adapter_a_dir: Path
    adapter_b_dir: Path
    eval_dataset: str
    merged_score: float
    strategy_name: str
    role: str
    notes: str
    n_eval_samples: int | None
    naive_linear_equivalent: bool
    strategy_is_uniform_linear: bool
    coefficients_unique: list[list[float]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _path_priority(path: Path) -> int:
    s = path.as_posix()
    for needle, priority in PREFERRED_EVAL_PATH_HINTS:
        if needle in s:
            return priority
    return 0


def _normalize_repo_path(raw: str | Path) -> Path:
    s = str(raw)
    if not s:
        return Path(s)
    if s.startswith(str(REPO_ROOT)):
        return Path(s)
    idx = s.find("/field_trials/")
    if idx != -1:
        return REPO_ROOT / s[idx + 1 :]
    idx = s.find("field_trials/")
    if idx != -1:
        return REPO_ROOT / s[idx:]
    p = Path(s)
    if p.exists():
        return p
    return p


def _dataset_family(dataset: str | None) -> str:
    ds = (dataset or "").strip()
    if ds in {"sst2", "imdb", "yelp_polarity", "amazon_polarity"}:
        return "sentiment_binary"
    if ds in {"ag_news"}:
        return "topic_classification"
    if ds in {"mnli"}:
        return "nli"
    if ds.startswith("tweet_eval/"):
        return "tweet_eval"
    return ds or "unknown"


def _task_relationship(dataset_a: str | None, dataset_b: str | None) -> str:
    if dataset_a and dataset_b and dataset_a == dataset_b:
        return "same_task"
    if _dataset_family(dataset_a) == _dataset_family(dataset_b):
        return "same_family"
    return "cross_task"


def _is_naive_linear_equivalent(plan: dict[str, Any]) -> bool:
    layer_cfgs = list(plan.get("layer_configs") or [])
    if not layer_cfgs:
        return False
    for cfg in layer_cfgs:
        st = str(cfg.get("strategy", "")).strip().lower()
        coeff = cfg.get("coefficients") or []
        if len(coeff) != 2:
            return False
        if abs(_safe_float(coeff[0]) - 0.5) > 1e-6 or abs(_safe_float(coeff[1]) - 0.5) > 1e-6:
            return False
        if st not in {"linear", "uniform_linear"}:
            return False
    return True


def _unique_coefficients(plan: dict[str, Any]) -> list[list[float]]:
    seen: set[tuple[float, float]] = set()
    out: list[list[float]] = []
    for cfg in list(plan.get("layer_configs") or []):
        coeff = cfg.get("coefficients") or []
        if len(coeff) != 2:
            continue
        a = round(_safe_float(coeff[0]), 6)
        b = round(_safe_float(coeff[1]), 6)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append([a, b])
    return out


def _find_evidence(adapter_dir: Path) -> Path | None:
    inventory_dir = adapter_dir.parent.parent
    candidate = inventory_dir / "evidence" / f"{adapter_dir.name}.json"
    if candidate.exists():
        return candidate
    return None


def _read_evidence(adapter_dir: Path) -> dict[str, Any] | None:
    path = _find_evidence(adapter_dir)
    if path is None:
        return None
    try:
        return _json(path)
    except Exception:
        return None


def _read_adapter_base_model(adapter_dir: Path) -> str | None:
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        d = _json(cfg)
    except Exception:
        return None
    return d.get("base_model_name_or_path")


def _collect_eval_records() -> list[PairRecord]:
    files = sorted(FIELD_TRIALS_DIR.rglob("eval_result.json"))

    chosen: dict[str, Path] = {}
    chosen_rank: dict[str, tuple[int, str]] = {}

    for path in files:
        if "/field_trials/over_accumulation_followup/" in path.as_posix():
            continue
        try:
            d = _json(path)
        except Exception:
            continue
        pair_id = str(d.get("pair_name") or "").strip()
        if not pair_id:
            continue
        rank = (_path_priority(path), path.as_posix())
        prev = chosen_rank.get(pair_id)
        if prev is None or rank > prev:
            chosen[pair_id] = path
            chosen_rank[pair_id] = rank

    records: list[PairRecord] = []
    for pair_id, eval_path in sorted(chosen.items()):
        eval_d = _json(eval_path)
        merge_plan = eval_path.parent / "merged_adapter" / "merge_plan.json"
        if not merge_plan.exists():
            continue
        plan_d = _json(merge_plan)

        merge_result = eval_path.parent / "merged_adapter" / "merge_result.json"
        merge_result_path = merge_result if merge_result.exists() else None

        a_dir = _normalize_repo_path(plan_d.get("adapter_a_dir", ""))
        b_dir = _normalize_repo_path(plan_d.get("adapter_b_dir", ""))

        strategy_name = str(plan_d.get("strategy_name") or eval_d.get("strategy") or "unknown")
        naive_equiv = _is_naive_linear_equivalent(plan_d)

        records.append(
            PairRecord(
                pair_id=pair_id,
                eval_result_path=eval_path,
                merge_plan_path=merge_plan,
                merge_result_path=merge_result_path,
                adapter_a_dir=a_dir,
                adapter_b_dir=b_dir,
                eval_dataset=str(eval_d.get("eval_dataset") or ""),
                merged_score=_safe_float(eval_d.get("merged_accuracy")),
                strategy_name=strategy_name,
                role=str(eval_d.get("role") or "unknown"),
                notes=str(eval_d.get("notes") or ""),
                n_eval_samples=_safe_int(eval_d.get("n_eval_samples"), default=0) or None,
                naive_linear_equivalent=naive_equiv,
                strategy_is_uniform_linear=(strategy_name == "uniform_linear"),
                coefficients_unique=_unique_coefficients(plan_d),
            )
        )
    return records


def _rank(values: list[float]) -> list[float]:
    ordered = sorted((v, i) for i, v in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_val = ((i + j - 1) / 2.0) + 1.0
        for k in range(i, j):
            out[ordered[k][1]] = rank_val
        i = j
    return out


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
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


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _top_hotspots(layer_verdicts: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for lv in layer_verdicts:
        score = _safe_float(lv.get("over_accumulation_score"), 0.0)
        layer_name = str(lv.get("layer_name") or lv.get("module_type") or "unknown_layer")
        scored.append(
            {
                "layer_name": layer_name,
                "module_type": str(lv.get("module_type") or ""),
                "score": score,
                "band": str(lv.get("over_accumulation_band") or "low"),
                "factors": lv.get("over_accumulation_factors") or {},
                "verdict": str(lv.get("verdict") or ""),
            }
        )
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:k]


def _build_rows(records: list[PairRecord], only_naive: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if only_naive and not rec.naive_linear_equivalent:
            continue

        ev_a = _read_evidence(rec.adapter_a_dir)
        ev_b = _read_evidence(rec.adapter_b_dir)

        src_a_score = _safe_float((ev_a or {}).get("adapter_score"), default=math.nan)
        src_b_score = _safe_float((ev_b or {}).get("adapter_score"), default=math.nan)
        src_a_dataset = (ev_a or {}).get("eval_dataset")
        src_b_dataset = (ev_b or {}).get("eval_dataset")

        src_a_score = None if math.isnan(src_a_score) else src_a_score
        src_b_score = None if math.isnan(src_b_score) else src_b_score

        valid_src = [s for s in (src_a_score, src_b_score) if s is not None]
        best_src = max(valid_src) if valid_src else None
        worst_src = min(valid_src) if valid_src else None
        avg_src = statistics.mean(valid_src) if valid_src else None

        delta_best = rec.merged_score - best_src if best_src is not None else None
        delta_worst = rec.merged_score - worst_src if worst_src is not None else None
        delta_avg = rec.merged_score - avg_src if avg_src is not None else None

        report = merge_audit(str(rec.adapter_a_dir), str(rec.adapter_b_dir), verbose=False)
        agg = report.aggregate
        n_layers = max(len(report.layer_verdicts), 1)

        layer_hotspots = _top_hotspots(list(report.layer_verdicts), k=3)

        task_rel = _task_relationship(src_a_dataset, src_b_dataset)

        row = {
            "pair_id": rec.pair_id,
            "adapter_a": rec.adapter_a_dir.as_posix(),
            "adapter_b": rec.adapter_b_dir.as_posix(),
            "adapter_a_name": rec.adapter_a_dir.name,
            "adapter_b_name": rec.adapter_b_dir.name,
            "base_model": (
                (ev_a or {}).get("base_model")
                or (ev_b or {}).get("base_model")
                or _read_adapter_base_model(rec.adapter_a_dir)
                or _read_adapter_base_model(rec.adapter_b_dir)
                or "unknown"
            ),
            "task_relationship": task_rel,
            "eval_dataset": rec.eval_dataset,
            "merged_score": rec.merged_score,
            "source_a_score": src_a_score,
            "source_b_score": src_b_score,
            "source_a_dataset": src_a_dataset,
            "source_b_dataset": src_b_dataset,
            "source_a_matches_eval_dataset": bool(src_a_dataset == rec.eval_dataset),
            "source_b_matches_eval_dataset": bool(src_b_dataset == rec.eval_dataset),
            "source_score_method": (
                "eval-matched"
                if (src_a_dataset == rec.eval_dataset and src_b_dataset == rec.eval_dataset)
                else "native-task-proxy"
            ),
            "best_source_score": best_src,
            "worst_source_score": worst_src,
            "avg_source_score": avg_src,
            "merge_delta_vs_best_source": delta_best,
            "merge_delta_vs_worst_source": delta_worst,
            "merge_delta_vs_average_source": delta_avg,
            "current_merge_verdict": str(agg.overall_verdict),
            "compatibility_score": _safe_float(agg.compatibility_score),
            "overlap_statistics": {
                "mean_overlap": _safe_float(agg.mean_overlap),
                "median_overlap": _safe_float(agg.median_overlap),
                "max_overlap": _safe_float(agg.max_overlap),
                "mean_agreement": _safe_float(agg.mean_agreement),
            },
            "conflict_imbalance_summary": {
                "n_layers": n_layers,
                "n_conflicting": _safe_int(agg.n_conflicting),
                "n_imbalanced": _safe_int(agg.n_imbalanced),
                "n_redundant": _safe_int(agg.n_redundant),
                "n_safe": _safe_int(agg.n_safe),
                "conflict_layer_fraction": _safe_float(agg.n_conflicting) / n_layers,
                "imbalance_layer_fraction": _safe_float(agg.n_imbalanced) / n_layers,
            },
            "over_accumulation_advisory": str(agg.over_accumulation_advisory),
            "over_accumulation_summary": str(agg.over_accumulation_summary),
            "high_risk_layer_count": _safe_int(agg.high_risk_layer_count),
            "watch_layer_count": _safe_int(agg.watch_layer_count),
            "max_over_accumulation_score": _safe_float(agg.max_over_accumulation_score),
            "layer_hotspots": layer_hotspots,
            "role": rec.role,
            "historical_strategy": rec.strategy_name,
            "naive_linear_equivalent": rec.naive_linear_equivalent,
            "coefficients_unique": rec.coefficients_unique,
            "eval_result_path": rec.eval_result_path.as_posix(),
            "merge_plan_path": rec.merge_plan_path.as_posix(),
            "merge_result_path": rec.merge_result_path.as_posix() if rec.merge_result_path else None,
            "n_eval_samples": rec.n_eval_samples,
            "notes": rec.notes,
        }
        rows.append(row)
    return rows


def _assign_cohorts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    high_overlap = [
        r
        for r in rows
        if r["overlap_statistics"]["mean_overlap"] >= 0.25
        and r["conflict_imbalance_summary"]["conflict_layer_fraction"] <= 0.10
    ]
    high_overlap_sorted = sorted(high_overlap, key=lambda r: r["max_over_accumulation_score"])

    split_cut = None
    if high_overlap_sorted:
        split_cut = high_overlap_sorted[len(high_overlap_sorted) // 2]["max_over_accumulation_score"]

    for r in rows:
        ov = r["overlap_statistics"]["mean_overlap"]
        conf_frac = r["conflict_imbalance_summary"]["conflict_layer_fraction"]
        imb_frac = r["conflict_imbalance_summary"]["imbalance_layer_fraction"]
        adv = r["over_accumulation_advisory"]
        score = r["max_over_accumulation_score"]
        verdict = r["current_merge_verdict"]

        is_high_overlap = ov >= 0.25 and conf_frac <= 0.10
        is_anchor = (ov < 0.15 and (verdict in {"imbalanced", "conflicting"} or imb_frac >= 0.35))

        if is_high_overlap:
            if adv in {"watch", "elevated"}:
                cohort = "group_2_high_overlap_elevated_official"
                rationale = (
                    "High overlap with non-conflict profile and official over-accumulation advisory "
                    f"'{adv}'."
                )
            elif split_cut is not None and score >= split_cut:
                cohort = "group_2_high_overlap_elevated_proxy"
                rationale = (
                    "High overlap and top-half over-accumulation score inside the high-overlap subset. "
                    "Used as a provisional elevated-risk split because official advisory did not trigger."
                )
            else:
                cohort = "group_1_high_overlap_low_proxy"
                rationale = (
                    "High overlap with low-to-mid over-accumulation score inside the high-overlap subset."
                )
        elif is_anchor:
            cohort = "group_3_non_overlap_pathology_anchor"
            rationale = (
                "Lower-overlap pair where imbalance/cross-task pathology is more salient than "
                "over-accumulation."
            )
        else:
            cohort = "exploratory_other"
            rationale = "Outside primary high-overlap comparison and outside designated anchor slice."

        r["cohort_assignment"] = cohort
        r["cohort_rationale"] = rationale

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["cohort_assignment"]] = counts.get(r["cohort_assignment"], 0) + 1

    return {
        "high_overlap_count": len(high_overlap),
        "high_overlap_proxy_split_cut": split_cut,
        "cohort_counts": counts,
    }


def _analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    g1 = [r for r in rows if r["cohort_assignment"] == "group_1_high_overlap_low_proxy"]
    g2 = [r for r in rows if r["cohort_assignment"].startswith("group_2_high_overlap")]
    g3 = [r for r in rows if r["cohort_assignment"] == "group_3_non_overlap_pathology_anchor"]
    high = g1 + g2

    def _outcome(rs: list[dict[str, Any]]) -> dict[str, Any]:
        deltas_best = [r["merge_delta_vs_best_source"] for r in rs if r["merge_delta_vs_best_source"] is not None]
        deltas_avg = [
            r["merge_delta_vs_average_source"] for r in rs if r["merge_delta_vs_average_source"] is not None
        ]
        poor = [r for r in rs if (r["merge_delta_vs_best_source"] is not None and r["merge_delta_vs_best_source"] <= POOR_DELTA_THRESHOLD)]
        return {
            "n_pairs": len(rs),
            "mean_delta_vs_best": _mean(deltas_best),
            "median_delta_vs_best": _median(deltas_best),
            "mean_delta_vs_avg": _mean(deltas_avg),
            "median_delta_vs_avg": _median(deltas_avg),
            "poor_merge_count": len(poor),
            "poor_merge_threshold": POOR_DELTA_THRESHOLD,
            "poor_merge_pairs": [r["pair_id"] for r in poor],
        }

    advisory_counts: dict[str, int] = {}
    for r in high:
        adv = r["over_accumulation_advisory"]
        advisory_counts[adv] = advisory_counts.get(adv, 0) + 1

    deltas_high = [r["merge_delta_vs_best_source"] for r in high if r["merge_delta_vs_best_source"] is not None]
    oa_scores_high = [r["max_over_accumulation_score"] for r in high if r["merge_delta_vs_best_source"] is not None]
    high_risk_counts = [r["high_risk_layer_count"] for r in high if r["merge_delta_vs_best_source"] is not None]

    overlaps = [r["overlap_statistics"]["mean_overlap"] for r in high if r["merge_delta_vs_best_source"] is not None]
    conflict_fracs = [
        r["conflict_imbalance_summary"]["conflict_layer_fraction"]
        for r in high
        if r["merge_delta_vs_best_source"] is not None
    ]
    imbalance_fracs = [
        r["conflict_imbalance_summary"]["imbalance_layer_fraction"]
        for r in high
        if r["merge_delta_vs_best_source"] is not None
    ]

    contrast = {
        "spearman_delta_vs_max_over_accumulation_score": _spearman(oa_scores_high, deltas_high),
        "spearman_delta_vs_high_risk_layer_count": _spearman(high_risk_counts, deltas_high),
        "spearman_delta_vs_mean_overlap": _spearman(overlaps, deltas_high),
        "spearman_delta_vs_conflict_fraction": _spearman(conflict_fracs, deltas_high),
        "spearman_delta_vs_imbalance_fraction": _spearman(imbalance_fracs, deltas_high),
    }

    # Optional compact OLS with numpy when available.
    ols = None
    try:
        import numpy as np

        if len(high) >= 6:
            y = np.array(
                [r["merge_delta_vs_best_source"] for r in high if r["merge_delta_vs_best_source"] is not None],
                dtype=float,
            )
            X_rows: list[list[float]] = []
            for r in high:
                if r["merge_delta_vs_best_source"] is None:
                    continue
                X_rows.append(
                    [
                        1.0,
                        float(r["max_over_accumulation_score"]),
                        float(r["overlap_statistics"]["mean_overlap"]),
                        float(r["conflict_imbalance_summary"]["conflict_layer_fraction"]),
                        float(r["conflict_imbalance_summary"]["imbalance_layer_fraction"]),
                    ]
                )
            X = np.array(X_rows, dtype=float)
            beta, residuals, rank, singular_vals = np.linalg.lstsq(X, y, rcond=None)
            ols = {
                "model": "delta_vs_best ~ max_over_accumulation + mean_overlap + conflict_frac + imbalance_frac",
                "coefficients": {
                    "intercept": float(beta[0]),
                    "max_over_accumulation_score": float(beta[1]),
                    "mean_overlap": float(beta[2]),
                    "conflict_fraction": float(beta[3]),
                    "imbalance_fraction": float(beta[4]),
                },
                "n_samples": int(X.shape[0]),
                "rank": int(rank),
                "residual_sum_squares": (
                    float(residuals[0]) if hasattr(residuals, "__len__") and len(residuals) else None
                ),
                "singular_values": [float(v) for v in singular_vals.tolist()],
            }
    except Exception:
        ols = None

    return {
        "cohort_comparison": {
            "group_1_high_overlap_low_proxy": _outcome(g1),
            "group_2_high_overlap_elevated_proxy_or_official": _outcome(g2),
            "group_3_non_overlap_pathology_anchor": _outcome(g3),
            "high_overlap_union": _outcome(high),
        },
        "official_advisory_distribution_high_overlap": advisory_counts,
        "continuous_relationships_high_overlap": {
            "spearman_delta_vs_max_over_accumulation_score": _spearman(oa_scores_high, deltas_high),
            "spearman_delta_vs_high_risk_layer_count": _spearman(high_risk_counts, deltas_high),
            "n_pairs": len(high),
        },
        "factor_contrast_high_overlap": contrast,
        "compact_ols_high_overlap": ols,
        "notes": [
            "Official over_accumulation_advisory may be degenerate (all 'none') in this cohort.",
            "Proxy split is only for bounded first-pass analysis; it does not modify production advisory semantics.",
            "Delta vs source uses native-task evidence proxy when source eval scores are not dataset-matched.",
        ],
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _write_cohort_definition(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]], only_naive: bool) -> None:
    lines = []
    lines.append("# Over-Accumulation Follow-up Cohort Definition")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        "- This first-pass cohort is retrospective and built from existing local merge outputs with recomputed current over-accumulation diagnostics."
    )
    lines.append(
        f"- Rows included: **{len(rows)}** pair-level evaluations from local field-trial artifacts."
    )
    lines.append(
        f"- `--only-naive` filter active: **{'yes' if only_naive else 'no'}**."
    )
    lines.append("")
    lines.append("## Group Rules")
    lines.append("- High-overlap gate: `mean_overlap >= 0.25` and `conflict_layer_fraction <= 0.10`.")
    lines.append(
        "- Group 1 (`group_1_high_overlap_low_proxy`): high-overlap rows with lower half over-accumulation scores."
    )
    lines.append(
        "- Group 2 (`group_2_high_overlap_elevated_proxy` or `..._official`): high-overlap rows with upper half scores, or official `watch/elevated` advisory when present."
    )
    lines.append(
        "- Group 3 (`group_3_non_overlap_pathology_anchor`): lower-overlap rows where imbalance/cross-task pathology is the primary issue."
    )
    lines.append("")
    lines.append("## Important Caveats")
    lines.append(
        "- In this cohort, official pair-level over-accumulation advisory values may remain `none`; the Group 1/Group 2 split can therefore be a score-based proxy rather than an official advisory-band comparison."
    )
    lines.append(
        "- Source-relative deltas use evidence scores and may be `native-task-proxy` when source scores are unavailable on the pair eval dataset."
    )
    lines.append("")
    lines.append("## Cohort Counts")
    counts = meta.get("cohort_counts", {})
    for key in sorted(counts):
        lines.append(f"- `{key}`: {counts[key]}")
    split_cut = meta.get("high_overlap_proxy_split_cut")
    if split_cut is not None:
        lines.append(f"- High-overlap score split cut (`max_over_accumulation_score`): `{split_cut:.4f}`")
    path.write_text("\n".join(lines) + "\n")


def _write_cohort_table_md(path: Path, rows: list[dict[str, Any]]) -> None:
    md_rows = []
    for r in rows:
        md_rows.append(
            [
                r["pair_id"],
                r["task_relationship"],
                r["cohort_assignment"],
                r["current_merge_verdict"],
                _fmt(r["overlap_statistics"]["mean_overlap"], 3),
                _fmt(r["max_over_accumulation_score"], 3),
                r["over_accumulation_advisory"],
                str(r["high_risk_layer_count"]),
                _fmt(r["merge_delta_vs_best_source"], 3),
                "yes" if r["naive_linear_equivalent"] else "no",
            ]
        )
    table = _md_table(
        [
            "pair_id",
            "task_relationship",
            "cohort",
            "verdict",
            "mean_overlap",
            "max_oa_score",
            "oa_advisory",
            "high_risk_layers",
            "delta_vs_best",
            "naive_linear_equivalent",
        ],
        md_rows,
    )
    path.write_text("# Cohort Table\n\n" + table + "\n")


def _write_eval_results_md(path: Path, rows: list[dict[str, Any]]) -> None:
    md_rows = []
    for r in rows:
        md_rows.append(
            [
                r["pair_id"],
                r["eval_dataset"],
                _fmt(r["merged_score"], 3),
                _fmt(r["best_source_score"], 3),
                _fmt(r["merge_delta_vs_best_source"], 3),
                _fmt(r["worst_source_score"], 3),
                _fmt(r["merge_delta_vs_worst_source"], 3),
                _fmt(r["merge_delta_vs_average_source"], 3),
                r["source_score_method"],
                r["historical_strategy"],
            ]
        )
    table = _md_table(
        [
            "pair_id",
            "eval_dataset",
            "merged_score",
            "best_source",
            "delta_vs_best",
            "worst_source",
            "delta_vs_worst",
            "delta_vs_avg",
            "source_score_method",
            "historical_strategy",
        ],
        md_rows,
    )
    path.write_text("# Eval Results\n\n" + table + "\n")


def _write_explanatory_analysis_md(path: Path, analysis: dict[str, Any]) -> None:
    c = analysis["cohort_comparison"]
    lines = []
    lines.append("# Explanatory Analysis")
    lines.append("")
    lines.append("## Group Comparison")
    lines.append(
        f"- Group 1 count: {c['group_1_high_overlap_low_proxy']['n_pairs']} | "
        f"mean delta vs best: {_fmt(c['group_1_high_overlap_low_proxy']['mean_delta_vs_best'], 4)}"
    )
    lines.append(
        f"- Group 2 count: {c['group_2_high_overlap_elevated_proxy_or_official']['n_pairs']} | "
        f"mean delta vs best: {_fmt(c['group_2_high_overlap_elevated_proxy_or_official']['mean_delta_vs_best'], 4)}"
    )
    lines.append(
        f"- Group 3 anchors count: {c['group_3_non_overlap_pathology_anchor']['n_pairs']} | "
        f"mean delta vs best: {_fmt(c['group_3_non_overlap_pathology_anchor']['mean_delta_vs_best'], 4)}"
    )
    lines.append("")
    lines.append("## Advisory-Band Check")
    lines.append(
        f"- Official advisory distribution in high-overlap subset: {analysis['official_advisory_distribution_high_overlap']}"
    )
    lines.append(
        "- If the distribution is degenerate (all `none`), explanatory comparisons use the bounded proxy split only."
    )
    lines.append("")
    lines.append("## Continuous Relationships")
    cr = analysis["continuous_relationships_high_overlap"]
    lines.append(
        f"- Spearman(delta_vs_best, max_over_accumulation_score): {_fmt(cr['spearman_delta_vs_max_over_accumulation_score'], 4)}"
    )
    lines.append(
        f"- Spearman(delta_vs_best, high_risk_layer_count): {_fmt(cr['spearman_delta_vs_high_risk_layer_count'], 4)}"
    )
    lines.append(f"- High-overlap pairs analyzed: {cr['n_pairs']}")
    lines.append("")
    lines.append("## Contrast vs Existing Factors")
    fc = analysis["factor_contrast_high_overlap"]
    lines.append(
        f"- Spearman(delta_vs_best, mean_overlap): {_fmt(fc['spearman_delta_vs_mean_overlap'], 4)}"
    )
    lines.append(
        f"- Spearman(delta_vs_best, conflict_fraction): {_fmt(fc['spearman_delta_vs_conflict_fraction'], 4)}"
    )
    lines.append(
        f"- Spearman(delta_vs_best, imbalance_fraction): {_fmt(fc['spearman_delta_vs_imbalance_fraction'], 4)}"
    )
    if analysis.get("compact_ols_high_overlap"):
        ols = analysis["compact_ols_high_overlap"]
        lines.append("")
        lines.append("## Compact OLS (High-overlap subset)")
        lines.append(f"- Model: `{ols['model']}`")
        lines.append(f"- n={ols['n_samples']}, rank={ols['rank']}")
        coefs = ols["coefficients"]
        lines.append(
            f"- Coefficients: intercept={_fmt(coefs['intercept'], 6)}, "
            f"max_oa={_fmt(coefs['max_over_accumulation_score'], 6)}, "
            f"overlap={_fmt(coefs['mean_overlap'], 6)}, "
            f"conf={_fmt(coefs['conflict_fraction'], 6)}, "
            f"imb={_fmt(coefs['imbalance_fraction'], 6)}"
        )
    lines.append("")
    lines.append("## Notes")
    for note in analysis.get("notes", []):
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n")


def _write_layer_hotspot_notes(path: Path, rows: list[dict[str, Any]]) -> None:
    group1 = [r for r in rows if r["cohort_assignment"] == "group_1_high_overlap_low_proxy"]
    group2 = [r for r in rows if r["cohort_assignment"].startswith("group_2_high_overlap")]
    anchors = [r for r in rows if r["cohort_assignment"] == "group_3_non_overlap_pathology_anchor"]

    case_low = sorted(group1, key=lambda r: r["max_over_accumulation_score"])[0] if group1 else None
    case_high = sorted(group2, key=lambda r: r["max_over_accumulation_score"], reverse=True)[0] if group2 else None
    case_anchor = sorted(anchors, key=lambda r: r["merge_delta_vs_best_source"] or 0.0)[0] if anchors else None

    def _case_block(title: str, r: dict[str, Any] | None) -> list[str]:
        if r is None:
            return [f"### {title}", "", "- Not available in this cohort.", ""]
        lines = [f"### {title}", ""]
        lines.append(f"- Pair: `{r['pair_id']}`")
        lines.append(f"- Cohort: `{r['cohort_assignment']}`")
        lines.append(
            f"- Outcome: merged={_fmt(r['merged_score'], 4)}, "
            f"delta_vs_best={_fmt(r['merge_delta_vs_best_source'], 4)}"
        )
        lines.append(
            f"- Structure: mean_overlap={_fmt(r['overlap_statistics']['mean_overlap'], 4)}, "
            f"max_oa_score={_fmt(r['max_over_accumulation_score'], 4)}, "
            f"advisory={r['over_accumulation_advisory']}"
        )
        lines.append("- Top layer hotspots:")
        for hs in r.get("layer_hotspots", []):
            lines.append(
                f"  - `{hs['layer_name']}` score={_fmt(hs['score'], 4)} band={hs['band']} "
                f"(alignment={_fmt(_safe_float((hs.get('factors') or {}).get('alignment')), 3)}, "
                f"concentration={_fmt(_safe_float((hs.get('factors') or {}).get('concentration')), 3)}, "
                f"coefficient={_fmt(_safe_float((hs.get('factors') or {}).get('coefficient_exposure')), 3)})"
            )
        lines.append("")
        return lines

    lines = ["# Layer Hotspot Notes", ""]
    lines += _case_block("Case 1 — High-overlap low-score reference", case_low)
    lines += _case_block("Case 2 — High-overlap high-score proxy-elevated", case_high)
    lines += _case_block("Case 3 — Non-overlap pathology anchor", case_anchor)
    lines.append("## Interpretation")
    lines.append(
        "- In this first pass, hotspot ranking is interpretable, but official pair-level advisory may remain non-triggered."
    )
    lines.append(
        "- Hotspot notes are diagnostic context, not causal proof."
    )
    path.write_text("\n".join(lines) + "\n")


def _write_validation_memo(path: Path, rows: list[dict[str, Any]], analysis: dict[str, Any], only_naive: bool) -> None:
    c = analysis["cohort_comparison"]
    g1 = c["group_1_high_overlap_low_proxy"]
    g2 = c["group_2_high_overlap_elevated_proxy_or_official"]
    high = c["high_overlap_union"]
    lines = []
    lines.append("# Over-Accumulation Diagnostic Validation Memo (First Pass)")
    lines.append("")
    lines.append("## 1. What Was Tested")
    lines.append(
        f"- Cohort size: **{len(rows)}** pair evaluations from local field-trial artifacts."
    )
    lines.append(
        f"- Naive-only filter active: **{'yes' if only_naive else 'no'}**."
    )
    lines.append(
        "- Over-accumulation metrics were recomputed with current Gradience merge-audit code for each pair."
    )
    lines.append(
        "- Outcome metric focus: `merge_delta_vs_best_source` with evidence-based source baselines."
    )
    lines.append("")
    lines.append("## 2. What the Advisory Explained")
    lines.append(
        f"- Group 1 (high-overlap lower-score): n={g1['n_pairs']}, "
        f"mean delta vs best={_fmt(g1['mean_delta_vs_best'], 4)}."
    )
    lines.append(
        f"- Group 2 (high-overlap higher-score / official elevated when present): n={g2['n_pairs']}, "
        f"mean delta vs best={_fmt(g2['mean_delta_vs_best'], 4)}."
    )
    lines.append(
        f"- High-overlap union: n={high['n_pairs']}, "
        f"Spearman(delta, max_oa)={_fmt(analysis['continuous_relationships_high_overlap']['spearman_delta_vs_max_over_accumulation_score'], 4)}."
    )
    lines.append(
        f"- Official advisory distribution in high-overlap subset: {analysis['official_advisory_distribution_high_overlap']}."
    )
    lines.append("")
    lines.append("## 3. What Remained Bounded")
    lines.append("- Task scope: classification-focused field-trial adapters.")
    lines.append("- Artifact scope: LoRA/low-rank PEFT pairs already present in repo.")
    lines.append("- Merge condition scope: retrospective; includes mixed historical strategies unless rerun with strict naive baseline.")
    lines.append(
        "- Explanatory status: correlational only; this pass does not establish causal mechanism."
    )
    lines.append("")
    lines.append("## 4. What This Means for Gradience")
    official_counts = analysis["official_advisory_distribution_high_overlap"]
    if official_counts.get("watch", 0) + official_counts.get("elevated", 0) == 0:
        lines.append(
            "- Official advisory did not activate in this cohort, so direct `none vs watch/elevated` validation remains pending."
        )
        lines.append(
            "- The score-proxy split provides bounded exploratory signal but is not strong enough to upgrade policy confidence by itself."
        )
    else:
        lines.append(
            "- Official advisory activated for part of high-overlap cohort; first-pass comparisons indicate bounded additive explanatory value."
        )
    lines.append("")
    lines.append("## 5. Whether to Proceed to Execution-Side Study")
    lines.append(
        "- Recommendation: **deeper diagnostic study first**, with strict naive reruns for a targeted high-overlap cohort and dataset-matched source baselines."
    )
    lines.append(
        "- Execution-side calibrated merge comparison (e.g., SVC-style) is reasonable only after strict-naive evidence is refreshed under current runtime constraints."
    )
    path.write_text("\n".join(lines) + "\n")


def _write_strategy_summary(path: Path, rows: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    c = analysis["cohort_comparison"]
    lines = []
    lines.append("# Over-Accumulation Follow-up Summary")
    lines.append("")
    lines.append("## Outcome")
    lines.append(
        f"- Evaluated {len(rows)} local pair outcomes with recomputed current over-accumulation diagnostics."
    )
    lines.append(
        f"- High-overlap comparison: Group 1 n={c['group_1_high_overlap_low_proxy']['n_pairs']} vs "
        f"Group 2 n={c['group_2_high_overlap_elevated_proxy_or_official']['n_pairs']}."
    )
    lines.append(
        f"- Mean delta vs best: Group 1={_fmt(c['group_1_high_overlap_low_proxy']['mean_delta_vs_best'], 4)}, "
        f"Group 2={_fmt(c['group_2_high_overlap_elevated_proxy_or_official']['mean_delta_vs_best'], 4)}."
    )
    lines.append(
        f"- Spearman(delta, max_oa) in high-overlap subset: "
        f"{_fmt(analysis['continuous_relationships_high_overlap']['spearman_delta_vs_max_over_accumulation_score'], 4)}."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- This first pass is useful for bounded diagnostics, but it is not a strong stand-alone validation of official advisory-band behavior if official advisories remain non-triggered."
    )
    lines.append(
        "- Keep the over-accumulation layer as a cautious additive signal; avoid elevating it to policy-driving status until strict-naive follow-up reruns are available."
    )
    lines.append("")
    lines.append("## Next Step")
    lines.append(
        "- Run a strict naive-only rerun for a targeted high-overlap subset (with dataset-matched source baselines), then repeat the same explanatory analysis."
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-naive",
        action="store_true",
        help="Restrict cohort to merge plans that are naive-linear equivalents (all layers linear 0.5/0.5).",
    )
    args = parser.parse_args()

    FOLLOWUP_DIR.mkdir(parents=True, exist_ok=True)

    records = _collect_eval_records()
    rows = _build_rows(records, only_naive=args.only_naive)
    cohort_meta = _assign_cohorts(rows)
    analysis = _analysis(rows)

    # Stage A outputs
    (FOLLOWUP_DIR / "cohort_table.json").write_text(json.dumps(rows, indent=2))
    _write_cohort_table_md(FOLLOWUP_DIR / "cohort_table.md", rows)
    _write_cohort_definition(FOLLOWUP_DIR / "cohort_definition.md", cohort_meta, rows, args.only_naive)

    # Stage B outputs
    merge_manifest = [
        {
            "pair_id": r["pair_id"],
            "eval_result_path": r["eval_result_path"],
            "merge_plan_path": r["merge_plan_path"],
            "merge_result_path": r["merge_result_path"],
            "historical_strategy": r["historical_strategy"],
            "naive_linear_equivalent": r["naive_linear_equivalent"],
            "coefficients_unique": r["coefficients_unique"],
            "merge_condition_note": (
                "naive-linear-equivalent historical output"
                if r["naive_linear_equivalent"]
                else "historical output (non-naive-equivalent strategy observed)"
            ),
        }
        for r in rows
    ]
    (FOLLOWUP_DIR / "merge_outputs_manifest.json").write_text(json.dumps(merge_manifest, indent=2))

    # Stage C outputs
    eval_rows = [
        {
            "pair_id": r["pair_id"],
            "eval_dataset": r["eval_dataset"],
            "merged_score": r["merged_score"],
            "source_a_score": r["source_a_score"],
            "source_b_score": r["source_b_score"],
            "best_source_score": r["best_source_score"],
            "worst_source_score": r["worst_source_score"],
            "avg_source_score": r["avg_source_score"],
            "merge_delta_vs_best_source": r["merge_delta_vs_best_source"],
            "merge_delta_vs_worst_source": r["merge_delta_vs_worst_source"],
            "merge_delta_vs_average_source": r["merge_delta_vs_average_source"],
            "source_score_method": r["source_score_method"],
            "n_eval_samples": r["n_eval_samples"],
            "historical_strategy": r["historical_strategy"],
            "naive_linear_equivalent": r["naive_linear_equivalent"],
        }
        for r in rows
    ]
    (FOLLOWUP_DIR / "eval_results.json").write_text(json.dumps(eval_rows, indent=2))
    _write_eval_results_md(FOLLOWUP_DIR / "eval_results.md", rows)

    # Stage D outputs
    (FOLLOWUP_DIR / "explanatory_analysis.json").write_text(json.dumps(analysis, indent=2))
    _write_explanatory_analysis_md(FOLLOWUP_DIR / "explanatory_analysis.md", analysis)

    # Stage E output
    _write_layer_hotspot_notes(FOLLOWUP_DIR / "layer_hotspot_notes.md", rows)

    # Stage F outputs
    _write_validation_memo(FOLLOWUP_DIR / "validation_memo.md", rows, analysis, args.only_naive)
    strategy_doc = REPO_ROOT / "docs" / "strategy" / "over_accumulation_summary.md"
    _write_strategy_summary(strategy_doc, rows, analysis)

    # Optional summary json
    summary = {
        "n_records": len(records),
        "n_rows_in_study": len(rows),
        "only_naive": args.only_naive,
        "cohort_counts": cohort_meta.get("cohort_counts", {}),
        "high_overlap_proxy_split_cut": cohort_meta.get("high_overlap_proxy_split_cut"),
        "official_advisory_distribution_high_overlap": analysis.get(
            "official_advisory_distribution_high_overlap", {}
        ),
        "spearman_delta_vs_max_oa_high_overlap": analysis["continuous_relationships_high_overlap"][
            "spearman_delta_vs_max_over_accumulation_score"
        ],
    }
    (FOLLOWUP_DIR / "validation_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
