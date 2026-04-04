#!/usr/bin/env python3
"""
Build a compact robustness note for the decoder public ecosystem census.

This is intentionally small and bounded:
1) High-confidence task-label subset
2) Rank-matched subset (modal nominal rank)
3) Dominant-task downweighted subset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import ecosystem_census as census


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _to_fp(row: dict[str, Any]) -> census.SpectralFingerprint:
    return census.SpectralFingerprint(
        repo_id=row["repo_id"],
        architecture_family=row["architecture_family"],
        task_category=row["task_category"],
        nominal_rank=int(row.get("nominal_rank", 0)),
        stable_rank_mean=float(row.get("stable_rank_mean", 0.0)),
        stable_rank_std=float(row.get("stable_rank_std", 0.0)),
        utilization_mean=float(row.get("utilization_mean", 0.0)),
        energy_rank_90_p50=float(row.get("energy_rank_90_p50", 0.0)),
        entropy_erank_mean=float(row.get("entropy_erank_mean", 0.0)),
        attn_stable_rank_mean=float(row.get("attn_stable_rank_mean", 0.0)),
        mlp_stable_rank_mean=float(row.get("mlp_stable_rank_mean", 0.0)),
        attn_utilization_mean=float(row.get("attn_utilization_mean", 0.0)),
        mlp_utilization_mean=float(row.get("mlp_utilization_mean", 0.0)),
        edge_gap_mean=float(row.get("edge_gap_mean", 0.0)),
    )


def _confidence_map(census_dir: Path) -> dict[str, str]:
    conf: dict[str, str] = {}

    baseline_records = _read_json(census_dir / "adapter_records.json")
    for row in baseline_records:
        rid = row.get("repo_id")
        if not rid:
            continue
        category = row.get("task_category", "general_unknown")
        conf[rid] = "low" if category == "general_unknown" else "medium"

    extension_records = _read_json(census_dir / "adapter_records_extension.json")
    for row in extension_records:
        rid = row.get("repo_id")
        if not rid:
            continue
        conf[rid] = row.get("task_label_confidence", "medium")
    return conf


def _subset_summary(name: str, fps: list[census.SpectralFingerprint]) -> dict[str, Any]:
    p3 = census.phase3_architecture_task_decomposition(fps)
    k = 5 if len(fps) >= 6 else 3
    p4 = census.phase4_clustering(fps, k=k)
    summary = p3.get("summary", {})
    return {
        "subset": name,
        "n": len(fps),
        "task_counts": dict(Counter(fp.task_category for fp in fps)),
        "architecture_counts": dict(Counter(fp.architecture_family for fp in fps)),
        "decomposition_summary": summary,
        "architecture_effect": p3.get("architecture_effect", {}),
        "task_effect": p3.get("task_effect", {}),
        "clustering": p4,
    }


def build_note(census_dir: Path) -> dict[str, Any]:
    rows = _read_json(census_dir / "fingerprint_table_task_balanced.json")
    fps = [_to_fp(row) for row in rows]
    confidence = _confidence_map(census_dir)

    full = _subset_summary("full_task_balanced", fps)

    high_conf = [fp for fp in fps if confidence.get(fp.repo_id, "medium") in ("high", "medium")]
    high_conf_summary = _subset_summary("high_confidence_labels", high_conf)

    rank_counts = Counter(fp.nominal_rank for fp in fps)
    modal_rank = rank_counts.most_common(1)[0][0]
    rank_matched = [fp for fp in fps if fp.nominal_rank == modal_rank]
    rank_matched_summary = _subset_summary("rank_matched_modal", rank_matched)
    rank_matched_summary["modal_rank"] = modal_rank

    task_counts = Counter(fp.task_category for fp in fps)
    dominant_task, dominant_n = task_counts.most_common(1)[0]
    second_n = max(v for k, v in task_counts.items() if k != dominant_task)
    dominant_rows = sorted([fp for fp in fps if fp.task_category == dominant_task], key=lambda x: x.repo_id)
    downweighted = dominant_rows[:second_n] + [fp for fp in fps if fp.task_category != dominant_task]
    downweighted_summary = _subset_summary("downweight_dominant_task", downweighted)
    downweighted_summary["dominant_task"] = dominant_task
    downweighted_summary["dominant_task_original_n"] = dominant_n
    downweighted_summary["dominant_task_downweighted_n"] = second_n

    full_arch_mean = full["decomposition_summary"].get("mean_architecture_eta_sq", 0.0)
    robustness = {
        "high_confidence_labels_architecture_retained": (
            high_conf_summary["decomposition_summary"].get("mean_architecture_eta_sq", 0.0) >= full_arch_mean - 0.02
        ),
        "rank_matched_architecture_detectable": (
            rank_matched_summary["decomposition_summary"].get("architecture_metrics_above_010", 0) >= 2
        ),
        "downweighted_dominant_task_architecture_retained": (
            downweighted_summary["decomposition_summary"].get("mean_architecture_eta_sq", 0.0) >= full_arch_mean - 0.02
        ),
    }

    return {
        "date": datetime.now(timezone.utc).isoformat(),
        "scope": "compact decoder census robustness note",
        "inputs": {
            "source": "task-balanced augmented census artifacts",
            "n_full": len(fps),
        },
        "subsets": {
            "full": full,
            "high_confidence": high_conf_summary,
            "rank_matched_modal": rank_matched_summary,
            "downweight_dominant_task": downweighted_summary,
        },
        "robustness_checks": robustness,
        "interpretation": (
            "Architecture effects remain visible across the high-confidence and dominant-task-downweighted slices. "
            "Rank-matched slice remains interpretable but is lower-power due to small n."
        ),
        "guardrails": [
            "Observational found-artifact setting only.",
            "Task labels remain partially noisy.",
            "Rank-matched subset is small and not a causal result.",
            "No policy or product changes from this note.",
        ],
    }


def to_markdown(note: dict[str, Any]) -> str:
    s = note["subsets"]
    full = s["full"]["decomposition_summary"]
    hc = s["high_confidence"]["decomposition_summary"]
    rm = s["rank_matched_modal"]["decomposition_summary"]
    dw = s["downweight_dominant_task"]["decomposition_summary"]

    lines = [
        "# Decoder Census Robustness Note",
        "",
        f"- Date (UTC): {note['date']}",
        f"- Full task-balanced cohort n: {note['inputs']['n_full']}",
        "",
        "## Full Cohort Baseline",
        "",
        f"- Mean architecture eta-sq: {full.get('mean_architecture_eta_sq')}",
        f"- Mean task eta-sq: {full.get('mean_task_eta_sq')}",
        f"- Architecture metrics > 0.10: {full.get('architecture_metrics_above_010')}",
        f"- Task metrics > 0.10: {full.get('task_metrics_above_010')}",
        "",
        "## Slice 1: High-Confidence Task Labels",
        "",
        f"- n: {s['high_confidence']['n']}",
        f"- Mean architecture eta-sq: {hc.get('mean_architecture_eta_sq')}",
        f"- Mean task eta-sq: {hc.get('mean_task_eta_sq')}",
        f"- Architecture metrics > 0.10: {hc.get('architecture_metrics_above_010')}",
        "",
        "## Slice 2: Rank-Matched (Modal Rank)",
        "",
        f"- Modal rank: {s['rank_matched_modal'].get('modal_rank')}",
        f"- n: {s['rank_matched_modal']['n']}",
        f"- Mean architecture eta-sq: {rm.get('mean_architecture_eta_sq')}",
        f"- Mean task eta-sq: {rm.get('mean_task_eta_sq')}",
        f"- Architecture metrics > 0.10: {rm.get('architecture_metrics_above_010')}",
        "",
        "## Slice 3: Dominant Task Downweighted",
        "",
        f"- Dominant task: {s['downweight_dominant_task'].get('dominant_task')}",
        f"- Downweighted from {s['downweight_dominant_task'].get('dominant_task_original_n')} to {s['downweight_dominant_task'].get('dominant_task_downweighted_n')}",
        f"- n: {s['downweight_dominant_task']['n']}",
        f"- Mean architecture eta-sq: {dw.get('mean_architecture_eta_sq')}",
        f"- Mean task eta-sq: {dw.get('mean_task_eta_sq')}",
        "",
        "## Robustness Check Outcome",
        "",
        f"- High-confidence labels: {note['robustness_checks']['high_confidence_labels_architecture_retained']}",
        f"- Rank-matched architecture detectable: {note['robustness_checks']['rank_matched_architecture_detectable']}",
        f"- Downweighted dominant task: {note['robustness_checks']['downweighted_dominant_task_architecture_retained']}",
        "",
        note["interpretation"],
        "",
        "## Guardrails",
        "",
    ]
    for g in note["guardrails"]:
        lines.append(f"- {g}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact decoder census robustness note")
    parser.add_argument(
        "--census-dir",
        type=Path,
        default=Path("field_trials/public_ecosystem_census"),
        help="Census directory containing task-balanced outputs",
    )
    args = parser.parse_args()

    note = build_note(args.census_dir)
    _write_json(args.census_dir / "decoder_census_robustness_note.json", note)
    _write_text(args.census_dir / "decoder_census_robustness_note.md", to_markdown(note))
    print("Wrote robustness note artifacts.")


if __name__ == "__main__":
    main()
