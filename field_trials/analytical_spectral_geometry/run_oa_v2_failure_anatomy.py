#!/usr/bin/env python3
"""Failure-anatomy analysis for OA-v2 30-40 strict-naive cross-check artifacts.

This is a no-rerun diagnostic slice that explains why the promotion gate failed
for the completed OA-v2 30-pair run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_XCHECK = Path("field_trials/analytical_spectral_geometry/empirical_crosscheck_oa_v2_30_40_r1.json")
DEFAULT_STRICT = Path("field_trials/over_accumulation_followup/oa_v2_30_40_r1_strict_naive_rerun_results.json")
DEFAULT_OUT_JSON = Path("field_trials/analytical_spectral_geometry/failure_anatomy_oa_v2_30_40_r1.json")
DEFAULT_OUT_MD = Path("field_trials/analytical_spectral_geometry/failure_anatomy_oa_v2_30_40_r1.md")


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(vals: list[float]) -> float | None:
    clean = [v for v in vals if not math.isnan(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


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
            rank_v = ((i + j - 1) / 2.0) + 1.0
            for k in range(i, j):
                out[ordered[k][1]] = rank_v
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
    if den <= 0.0:
        return None
    return num / den


def _dataset_family(dataset: str | None) -> str:
    ds = (dataset or "").strip().lower()
    if ds in {"sst2", "imdb", "yelp_polarity", "amazon_polarity"}:
        return "sentiment_binary"
    if ds in {"ag_news"}:
        return "topic_classification"
    if ds in {"mnli"}:
        return "nli"
    if ds.startswith("tweet_eval/"):
        return "tweet_eval"
    return ds or "unknown"


def _backbone_family(base_model_id: str | None) -> str:
    b = (base_model_id or "").lower()
    if "distilbert" in b:
        return "distilbert"
    if "roberta" in b:
        return "roberta"
    if "bert" in b:
        return "bert"
    return b or "unknown"


def _source_gap_band(gap: float) -> str:
    if math.isnan(gap):
        return "unknown"
    if gap <= 0.05:
        return "near_top"
    if gap <= 0.15:
        return "mid_gap"
    return "large_gap"


def _rank_mismatch_band(ratio: float) -> str:
    if math.isnan(ratio):
        return "unknown"
    if ratio <= 1.5:
        return "matched"
    if ratio <= 3.0:
        return "moderate_mismatch"
    return "large_mismatch"


def _load_rank(adapter_path: str | None) -> float:
    if not adapter_path:
        return math.nan
    cfg = Path(adapter_path) / "adapter_config.json"
    try:
        d = json.loads(cfg.read_text())
        return _safe_float(d.get("r"), default=math.nan)
    except Exception:
        return math.nan


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


@dataclass
class Row:
    pair_id: str
    delta: float
    oa_v1: float
    oa_v2: float
    oa_v2_top3: float
    mean_overlap: float
    conflict_fraction: float
    source_gap: float
    task_family: str
    backbone_family: str
    source_gap_band: str
    rank_a: float
    rank_b: float
    rank_ratio: float
    rank_mismatch_band: str
    is_poor: bool
    in_intended_slice: bool
    cohort_group: str


def _build_rows(xcheck_payload: dict[str, Any], strict_rows: list[dict[str, Any]]) -> list[Row]:
    strict_by_id = {str(r.get("pair_id")): r for r in strict_rows}

    params = xcheck_payload.get("params", {})
    overlap_min = _safe_float(params.get("subset_overlap_min"), default=0.25)
    conflict_max = _safe_float(params.get("subset_conflict_max"), default=0.10)

    out: list[Row] = []
    for r in xcheck_payload.get("pair_rows", []):
        pair_id = str(r.get("pair_id"))
        strict = strict_by_id.get(pair_id, {})
        eval_dataset = strict.get("eval_dataset")
        base_model_id = strict.get("base_model_id")

        rank_a = _load_rank((strict.get("source_a_eval") or {}).get("adapter_path"))
        rank_b = _load_rank((strict.get("source_b_eval") or {}).get("adapter_path"))
        if not math.isnan(rank_a) and not math.isnan(rank_b) and min(rank_a, rank_b) > 0:
            rank_ratio = max(rank_a, rank_b) / min(rank_a, rank_b)
        else:
            rank_ratio = math.nan

        mean_overlap = _safe_float(r.get("mean_overlap"))
        conflict_fraction = _safe_float(r.get("conflict_fraction"))
        source_gap = _safe_float(r.get("source_score_gap"))
        in_slice = bool(mean_overlap >= overlap_min and conflict_fraction <= conflict_max)

        out.append(
            Row(
                pair_id=pair_id,
                delta=_safe_float(r.get("merge_delta_vs_best_source")),
                oa_v1=_safe_float(r.get("oa_v1_max_score")),
                oa_v2=_safe_float(r.get("oa_v2_max_score")),
                oa_v2_top3=_safe_float(r.get("oa_v2_top3_interaction")),
                mean_overlap=mean_overlap,
                conflict_fraction=conflict_fraction,
                source_gap=source_gap,
                task_family=_dataset_family(eval_dataset),
                backbone_family=_backbone_family(base_model_id),
                source_gap_band=_source_gap_band(source_gap),
                rank_a=rank_a,
                rank_b=rank_b,
                rank_ratio=rank_ratio,
                rank_mismatch_band=_rank_mismatch_band(rank_ratio),
                is_poor=bool(r.get("is_poor", False)),
                in_intended_slice=in_slice,
                cohort_group=str(strict.get("cohort_group", "unknown")),
            )
        )
    return out


def _slice_summary(rows: list[Row]) -> dict[str, Any]:
    x1 = [r.oa_v1 for r in rows]
    x2 = [r.oa_v2 for r in rows]
    y = [r.delta for r in rows]
    poor_rate = _mean([1.0 if r.is_poor else 0.0 for r in rows])
    return {
        "n": len(rows),
        "mean_delta_vs_best": _mean(y),
        "mean_oa_v1": _mean(x1),
        "mean_oa_v2": _mean(x2),
        "spearman_delta_vs_oa_v1": _spearman(x1, y),
        "spearman_delta_vs_oa_v2": _spearman(x2, y),
        "spearman_gain_v2_minus_v1": (
            (_spearman(x2, y) or 0.0) - (_spearman(x1, y) or 0.0)
            if (_spearman(x2, y) is not None and _spearman(x1, y) is not None)
            else None
        ),
        "poor_rate": poor_rate,
        "mean_source_gap": _mean([r.source_gap for r in rows]),
    }


def _group_summary(rows: list[Row], attr: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Row]] = {}
    for r in rows:
        key = str(getattr(r, attr))
        groups.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    for key, g in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        s = _slice_summary(g)
        s["group"] = key
        out.append(s)
    return out


def _cohort_mixture(rows: list[Row]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.cohort_group] = out.get(r.cohort_group, 0) + 1
    return out


def _build_findings(all_rows: list[Row], intended_rows: list[Row], xcheck_payload: dict[str, Any]) -> list[str]:
    overall = _slice_summary(all_rows)
    intended = _slice_summary(intended_rows)

    findings: list[str] = []
    findings.append(
        "OA-v2 improves overall rank correlation but does not hold directionally in the intended high-overlap/low-conflict slice."
    )
    findings.append(
        f"Overall Spearman(delta, OA-v1)={_fmt(overall['spearman_delta_vs_oa_v1'])}, "
        f"OA-v2={_fmt(overall['spearman_delta_vs_oa_v2'])}; intended-slice Spearman OA-v1={_fmt(intended['spearman_delta_vs_oa_v1'])}, "
        f"OA-v2={_fmt(intended['spearman_delta_vs_oa_v2'])}."
    )
    findings.append(
        f"Intended slice coverage is {len(intended_rows)}/{len(all_rows)} pairs, which remains a key stability limiter."
    )

    gate = (xcheck_payload.get("analysis") or {}).get("threshold_policy_gate", {})
    findings.append(
        "Promotion gate remains failed: "
        f"rule1={gate.get('rule_1_abs_spearman_gain_ge_0_15')}, "
        f"rule2={gate.get('rule_2_recall_gain_ge_0_20')}, "
        f"rule3={gate.get('rule_3_sign_consistency_ge_0_70')}, "
        f"rule4={gate.get('rule_4_interpretability_decomposition_ok')}."
    )
    return findings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crosscheck-json", type=Path, default=DEFAULT_XCHECK)
    p.add_argument("--strict-naive-json", type=Path, default=DEFAULT_STRICT)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    xcheck_payload = json.loads(args.crosscheck_json.read_text())
    strict_rows = json.loads(args.strict_naive_json.read_text())

    all_rows = _build_rows(xcheck_payload, strict_rows)
    intended_rows = [r for r in all_rows if r.in_intended_slice]

    overall = _slice_summary(all_rows)
    intended = _slice_summary(intended_rows)

    breakdown = {
        "task_family": _group_summary(all_rows, "task_family"),
        "backbone_family": _group_summary(all_rows, "backbone_family"),
        "source_gap_band": _group_summary(all_rows, "source_gap_band"),
        "rank_mismatch_band": _group_summary(all_rows, "rank_mismatch_band"),
        "cohort_group": _group_summary(all_rows, "cohort_group"),
    }

    payload = {
        "status": "oa_v2_failure_anatomy_completed",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": {
            "crosscheck_json": str(args.crosscheck_json),
            "strict_naive_json": str(args.strict_naive_json),
        },
        "n_pairs": len(all_rows),
        "n_intended_slice": len(intended_rows),
        "cohort_group_mix": _cohort_mixture(all_rows),
        "summary": {
            "overall": overall,
            "intended_high_overlap_low_conflict_slice": intended,
        },
        "breakdown": breakdown,
        "findings": _build_findings(all_rows, intended_rows, xcheck_payload),
        "gate_snapshot": (xcheck_payload.get("analysis") or {}).get("threshold_policy_gate", {}),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))

    lines: list[str] = []
    lines.append("# OA-v2 Failure Anatomy (30-pair Strict-Naive Run)")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Cross-check source: `{args.crosscheck_json}`")
    lines.append(f"- Strict-naive source: `{args.strict_naive_json}`")
    lines.append(f"- Pairs analyzed: `{len(all_rows)}`")
    lines.append(f"- Intended slice pairs (high-overlap/low-conflict): `{len(intended_rows)}`")
    lines.append("")
    lines.append("## Core Readout")
    lines.append(
        f"- Overall Spearman(delta, OA-v1)={_fmt(overall['spearman_delta_vs_oa_v1'])}, "
        f"OA-v2={_fmt(overall['spearman_delta_vs_oa_v2'])}."
    )
    lines.append(
        f"- Intended slice Spearman(delta, OA-v1)={_fmt(intended['spearman_delta_vs_oa_v1'])}, "
        f"OA-v2={_fmt(intended['spearman_delta_vs_oa_v2'])}."
    )
    lines.append(
        f"- Intended slice mean delta vs best: `{_fmt(intended['mean_delta_vs_best'])}`; "
        f"overall mean delta vs best: `{_fmt(overall['mean_delta_vs_best'])}`."
    )
    lines.append("")
    lines.append("## Findings")
    for f in payload["findings"]:
        lines.append(f"- {f}")
    lines.append("")

    def _table(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| group | n | mean_delta_vs_best | rho_v1 | rho_v2 | rho_gain_v2-v1 | poor_rate | mean_source_gap |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                "| "
                + f"{r.get('group','')} | {r.get('n',0)} | "
                + f"{_fmt(r.get('mean_delta_vs_best'))} | {_fmt(r.get('spearman_delta_vs_oa_v1'))} | "
                + f"{_fmt(r.get('spearman_delta_vs_oa_v2'))} | {_fmt(r.get('spearman_gain_v2_minus_v1'))} | "
                + f"{_fmt(r.get('poor_rate'))} | {_fmt(r.get('mean_source_gap'))} |"
            )
        lines.append("")

    _table("Task-Family Stratification", breakdown["task_family"])
    _table("Backbone-Family Stratification", breakdown["backbone_family"])
    _table("Source-Gap Band Stratification", breakdown["source_gap_band"])
    _table("Rank-Mismatch Band Stratification", breakdown["rank_mismatch_band"])

    lines.append("## Gate Snapshot")
    gate = payload["gate_snapshot"]
    lines.append(f"- Rule 1 abs Spearman gain >= 0.15: `{gate.get('rule_1_abs_spearman_gain_ge_0_15')}`")
    lines.append(f"- Rule 2 recall gain >= 0.20: `{gate.get('rule_2_recall_gain_ge_0_20')}`")
    lines.append(f"- Rule 3 sign consistency >= 0.70: `{gate.get('rule_3_sign_consistency_ge_0_70')}`")
    lines.append(f"- Rule 4 interpretability decomposition: `{gate.get('rule_4_interpretability_decomposition_ok')}`")
    lines.append(f"- All-pass: `{gate.get('all_pass')}`")
    lines.append("")
    lines.append("## Status")
    lines.append("- OA-v2 remains exploratory; no threshold/policy promotion from this run.")
    lines.append("- Next-run candidate should increase intended-slice coverage before any further gate decision.")
    lines.append("")

    args.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "n_pairs": len(all_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
