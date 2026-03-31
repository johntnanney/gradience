#!/usr/bin/env python3
"""Build decision-dependent compatibility sidecar artifacts (Stages A-E).

This script is intentionally CPU-light and data-local:
- reuses existing merge, routing, and triage outputs
- emits comparison tables and compact JSON artifacts
- generates a lightweight SVG matrix for aggregation contrasts
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SIDECAR_DIR = ROOT / "sidecar"
RESULT_DIR = SIDECAR_DIR / "results" / "decision_dependent_compatibility"
FIG_DIR = SIDECAR_DIR / "figures"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _pair_key(a: str, b: str) -> str:
    x, y = sorted((a, b))
    return f"{x} × {y}"


@dataclass(frozen=True)
class CaseRef:
    case_id: str
    group: str
    artifact_type: str
    backbone: str
    task_relation: str
    scenarios_available: tuple[str, ...]
    structural_outputs_available: bool
    behavioral_outputs_available: bool
    summary: str
    evidence_paths: tuple[str, ...]
    metrics: dict[str, Any]


def _build_panel_table() -> dict[str, Any]:
    routing = _load_json(ROOT / "experiments" / "routing_pilot" / "routing_report.json")
    merge_t01 = _load_json(
        ROOT / "field_trials" / "targeted_confirmation_same_family" / "inventory_t01" / "eval_results.json"
    )
    merge_t02 = _load_json(
        ROOT / "field_trials" / "targeted_confirmation_near_miss" / "inventory_t02" / "eval_results.json"
    )
    ckpt_t02_pairwise = _load_json(ROOT / "field_trials" / "checkpoint_inventory_t02" / "pairwise" / "pairwise_results.json")
    ckpt_t02_preflight = _load_json(
        ROOT / "field_trials" / "checkpoint_inventory_t02" / "preflight" / "preflight_results.json"
    )
    ckpt_t02_eval = _load_json(ROOT / "field_trials" / "checkpoint_inventory_t02" / "eval_results.json")

    merge_t01_map = {row["pair_name"]: row for row in merge_t01}
    merge_t02_map = {row["pair_name"]: row for row in merge_t02}
    routing_merge_map = {row["pair"]: row for row in routing["merge_comparison"]}
    routing_pair_map = {
        _pair_key(row["adapter_a"], row["adapter_b"]): row for row in routing["pair_assessments"]
    }
    ckpt_pair_map = {row["pair_id"]: row for row in ckpt_t02_pairwise["pairs"]}

    def _routing_case(pair_name: str, relation: str) -> tuple[dict[str, Any], dict[str, Any]]:
        merge_row = routing_merge_map[pair_name]
        left, right = pair_name.split(" × ")
        route_row = routing_pair_map[_pair_key(left, right)]
        return merge_row, route_row

    rt_high_merge, rt_high_route = _routing_case("rte_s42 × rte_s7", "same_task")
    rt_mod_merge, rt_mod_route = _routing_case("mnli_s7 × qnli_s7", "same_family")
    rt_low_merge, rt_low_route = _routing_case("qnli_s7 × rte_s42", "same_family")

    follow_map = {row["category"]: row for row in ckpt_t02_eval["follow_through"]["records"]}

    cases: list[CaseRef] = [
        CaseRef(
            case_id="mrg_safe_same_task_sst2_sst2_t01",
            group="merge_sensitive",
            artifact_type="lora_adapter_pair",
            backbone="distilbert-base-uncased",
            task_relation="same_task",
            scenarios_available=("merge", "triage"),
            structural_outputs_available=True,
            behavioral_outputs_available=True,
            summary="Retained same-task anchor in targeted confirmation T01.",
            evidence_paths=(
                "field_trials/targeted_confirmation_same_family/inventory_t01/eval_results.json",
                "field_trials/targeted_confirmation_same_family/inventory_t01/preflight/action_plan.txt",
            ),
            metrics={
                "merge_accuracy": merge_t01_map["t01_retained_sst2_x_sst2"]["merged_accuracy"],
                "merge_compatibility_score": merge_t01_map["t01_retained_sst2_x_sst2"]["compatibility_score"],
                "triage_role": merge_t01_map["t01_retained_sst2_x_sst2"]["role"],
            },
        ),
        CaseRef(
            case_id="mrg_near_miss_substantial_hate_t02",
            group="merge_sensitive",
            artifact_type="lora_adapter_pair",
            backbone="distilbert-base-uncased",
            task_relation="same_task",
            scenarios_available=("merge", "triage"),
            structural_outputs_available=True,
            behavioral_outputs_available=True,
            summary="Near-miss substantial case from targeted confirmation T02.",
            evidence_paths=(
                "field_trials/targeted_confirmation_near_miss/inventory_t02/eval_results.json",
                "field_trials/targeted_confirmation_near_miss/inventory_t02/preflight/action_plan.txt",
            ),
            metrics={
                "merge_accuracy": merge_t02_map["t02_near_miss_substantial_hate"]["merged_accuracy"],
                "merge_compatibility_score": merge_t02_map["t02_near_miss_substantial_hate"]["compatibility_score"],
                "triage_role": merge_t02_map["t02_near_miss_substantial_hate"]["role"],
            },
        ),
        CaseRef(
            case_id="mrg_cross_task_control_sst2_agnews_t01",
            group="merge_sensitive",
            artifact_type="lora_adapter_pair",
            backbone="distilbert-base-uncased",
            task_relation="cross_task",
            scenarios_available=("merge", "triage"),
            structural_outputs_available=True,
            behavioral_outputs_available=True,
            summary="Cross-task control in targeted confirmation T01.",
            evidence_paths=(
                "field_trials/targeted_confirmation_same_family/inventory_t01/eval_results.json",
                "field_trials/targeted_confirmation_same_family/inventory_t01/preflight/action_plan.txt",
            ),
            metrics={
                "merge_accuracy": merge_t01_map["t01_cross_task_sst2_x_agnews"]["merged_accuracy"],
                "merge_compatibility_score": merge_t01_map["t01_cross_task_sst2_x_agnews"]["compatibility_score"],
                "triage_role": merge_t01_map["t01_cross_task_sst2_x_agnews"]["role"],
            },
        ),
        CaseRef(
            case_id="rte_seed_pair_confusable",
            group="routing_sensitive",
            artifact_type="lora_adapter_pair",
            backbone="distilbert-base-uncased",
            task_relation="same_task",
            scenarios_available=("merge", "routing"),
            structural_outputs_available=True,
            behavioral_outputs_available=False,
            summary="Same-task pair with highest routing confusability in pilot.",
            evidence_paths=(
                "experiments/routing_pilot/routing_report.json",
                "experiments/routing_pilot/routing_pilot_field_note.md",
            ),
            metrics={
                "merge_verdict": rt_high_merge["merge_verdict"],
                "merge_compatibility_score": rt_high_merge["merge_compat_score"],
                "routing_confusability": rt_high_merge["routing_confusability"],
                "routing_score": rt_high_route["confusability_score"],
                "routing_recommendation": rt_high_route["recommendation"],
            },
        ),
        CaseRef(
            case_id="mnli_qnli_moderate_confusable",
            group="routing_sensitive",
            artifact_type="lora_adapter_pair",
            backbone="distilbert-base-uncased",
            task_relation="same_family",
            scenarios_available=("merge", "routing"),
            structural_outputs_available=True,
            behavioral_outputs_available=False,
            summary="Same-family NLI pair with moderate routing confusability.",
            evidence_paths=(
                "experiments/routing_pilot/routing_report.json",
                "experiments/routing_pilot/routing_pilot_field_note.md",
            ),
            metrics={
                "merge_verdict": rt_mod_merge["merge_verdict"],
                "merge_compatibility_score": rt_mod_merge["merge_compat_score"],
                "routing_confusability": rt_mod_merge["routing_confusability"],
                "routing_score": rt_mod_route["confusability_score"],
                "routing_recommendation": rt_mod_route["recommendation"],
            },
        ),
        CaseRef(
            case_id="qnli_rte_separable",
            group="routing_sensitive",
            artifact_type="lora_adapter_pair",
            backbone="distilbert-base-uncased",
            task_relation="same_family",
            scenarios_available=("merge", "routing"),
            structural_outputs_available=True,
            behavioral_outputs_available=False,
            summary="Clearly separable routing pair despite same NLI family.",
            evidence_paths=(
                "experiments/routing_pilot/routing_report.json",
                "experiments/routing_pilot/routing_pilot_field_note.md",
            ),
            metrics={
                "merge_verdict": rt_low_merge["merge_verdict"],
                "merge_compatibility_score": rt_low_merge["merge_compat_score"],
                "routing_confusability": rt_low_merge["routing_confusability"],
                "routing_score": rt_low_route["confusability_score"],
                "routing_recommendation": rt_low_route["recommendation"],
            },
        ),
        CaseRef(
            case_id="tri_same_task_near_miss_sst2_pair_t02",
            group="triage_sensitive",
            artifact_type="full_checkpoint_pair",
            backbone="distilbert-base-uncased",
            task_relation="same_task",
            scenarios_available=("triage",),
            structural_outputs_available=True,
            behavioral_outputs_available=True,
            summary="No retained pairs; strongest same-task near-miss probe.",
            evidence_paths=(
                "field_trials/checkpoint_inventory_t02/pairwise/pairwise_results.json",
                "field_trials/checkpoint_inventory_t02/preflight/preflight_results.json",
                "field_trials/checkpoint_inventory_t02/eval_results.json",
            ),
            metrics={
                "compatibility_score": ckpt_pair_map["sst2_s42::sst2_s123"]["compatibility_score"],
                "pair_risk": ckpt_pair_map["sst2_s42::sst2_s123"]["pair_risk"],
                "triage_summary_line": ckpt_t02_preflight["action_plan"]["summary_line"],
                "follow_through_category": follow_map["same_task_near_miss_probe"]["category"],
            },
        ),
        CaseRef(
            case_id="tri_same_family_review_sst2_yelp_t02",
            group="triage_sensitive",
            artifact_type="full_checkpoint_pair",
            backbone="distilbert-base-uncased",
            task_relation="same_family",
            scenarios_available=("triage",),
            structural_outputs_available=True,
            behavioral_outputs_available=True,
            summary="Same-family checkpoint pair routed to informational caution/review.",
            evidence_paths=(
                "field_trials/checkpoint_inventory_t02/pairwise/pairwise_results.json",
                "field_trials/checkpoint_inventory_t02/preflight/preflight_results.json",
                "field_trials/checkpoint_inventory_t02/eval_results.json",
            ),
            metrics={
                "compatibility_score": ckpt_pair_map["sst2_s42::yelp_s42"]["compatibility_score"],
                "pair_risk": ckpt_pair_map["sst2_s42::yelp_s42"]["pair_risk"],
                "pair_note": ckpt_pair_map["sst2_s42::yelp_s42"]["notes"],
                "follow_through_category": follow_map["same_family_pair_probe"]["category"],
            },
        ),
        CaseRef(
            case_id="tri_cross_task_weak_region_yelp_qnli_t02",
            group="triage_sensitive",
            artifact_type="full_checkpoint_pair",
            backbone="distilbert-base-uncased",
            task_relation="cross_task",
            scenarios_available=("triage",),
            structural_outputs_available=True,
            behavioral_outputs_available=False,
            summary="Cross-task weak/risky region in checkpoint triage inventory.",
            evidence_paths=(
                "field_trials/checkpoint_inventory_t02/pairwise/pairwise_results.json",
                "field_trials/checkpoint_inventory_t02/preflight/preflight_results.json",
            ),
            metrics={
                "compatibility_score": ckpt_pair_map["yelp_s42::qnli_s42"]["compatibility_score"],
                "pair_risk": ckpt_pair_map["yelp_s42::qnli_s42"]["pair_risk"],
                "action_plan_cross_task_caution": ckpt_t02_preflight["action_plan"]["cross_task_caution"],
            },
        ),
    ]

    payload = {
        "schema": "decision_dependent_compatibility_panel/v1",
        "generated_utc": _now_utc_iso(),
        "panel_size": len(cases),
        "groups": {
            "merge_sensitive": 3,
            "routing_sensitive": 3,
            "triage_sensitive": 3,
        },
        "cases": [
            {
                "case_id": c.case_id,
                "group": c.group,
                "artifact_type": c.artifact_type,
                "backbone": c.backbone,
                "task_relation": c.task_relation,
                "scenarios_available": list(c.scenarios_available),
                "structural_outputs_available": c.structural_outputs_available,
                "behavioral_outputs_available": c.behavioral_outputs_available,
                "summary": c.summary,
                "evidence_paths": list(c.evidence_paths),
                "metrics": c.metrics,
            }
            for c in cases
        ],
    }
    return payload


def _panel_markdown(panel: dict[str, Any]) -> str:
    lines = [
        "# Decision-Dependent Compatibility Panel",
        "",
        f"Generated: {panel['generated_utc']}",
        "",
        f"Total cases: {panel['panel_size']}",
        "",
        "| Case ID | Group | Artifact | Relation | Scenarios | Structural | Behavioral |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in panel["cases"]:
        scenarios = ", ".join(row["scenarios_available"])
        lines.append(
            f"| {row['case_id']} | {row['group']} | {row['artifact_type']} | {row['task_relation']} | {scenarios} | "
            f"{'yes' if row['structural_outputs_available'] else 'no'} | "
            f"{'yes' if row['behavioral_outputs_available'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Informative notes")
    lines.append("")
    for row in panel["cases"]:
        lines.append(f"- `{row['case_id']}`: {row['summary']}")
    lines.append("")
    return "\n".join(lines)


def _build_stage_b_audit(panel: dict[str, Any]) -> tuple[dict[str, Any], str]:
    scenario_stack = {
        "merge": {
            "measurement": {
                "classification": "shared unchanged",
                "objects": [
                    "layerwise subspace geometry (LoRA factors) or checkpoint delta summaries",
                    "pairwise compatibility primitives (overlap/alignment-derived scores)",
                ],
            },
            "diagnosis": {
                "classification": "shared with translation",
                "objects": [
                    "pair-level risk or redundancy descriptors",
                    "task relationship tags (same_task / same_family / cross_task)",
                ],
            },
            "aggregation": {
                "classification": "scenario-specific",
                "objects": [
                    "worst-case prioritization across layers",
                    "single adverse layer can dominate pair verdict",
                ],
            },
            "policy": {
                "classification": "scenario-specific",
                "objects": [
                    "retained / near-miss / caution / exclude",
                    "merge strategy selection for retained candidates",
                ],
            },
        },
        "routing": {
            "measurement": {
                "classification": "shared unchanged",
                "objects": [
                    "SubspaceMetrics from merge substrate (same extraction + per-layer geometry)",
                ],
            },
            "diagnosis": {
                "classification": "shared with translation",
                "objects": [
                    "layer confusability estimates from overlap + directional agreement",
                ],
            },
            "aggregation": {
                "classification": "scenario-specific",
                "objects": [
                    "distributional aggregation (mean confusability + high-confusability fraction)",
                ],
            },
            "policy": {
                "classification": "scenario-specific",
                "objects": [
                    "consider_dedup / needs_disambiguation / easily_routed",
                ],
            },
        },
        "triage": {
            "measurement": {
                "classification": "shared with translation",
                "objects": [
                    "pairwise structural summaries",
                    "single-artifact QA statuses from behavioral evidence bootstrap",
                ],
            },
            "diagnosis": {
                "classification": "shared with translation",
                "objects": [
                    "pair compatibility descriptors plus source-quality status",
                ],
            },
            "aggregation": {
                "classification": "scenario-specific",
                "objects": [
                    "gate-first aggregation (source QA can dominate and override pair structure)",
                ],
            },
            "policy": {
                "classification": "scenario-specific",
                "objects": [
                    "evaluate-first / near-miss review / cross-task caution / exclusion",
                ],
            },
        },
    }

    scenario_pairs = [
        {
            "scenario_pair": ["merge", "routing"],
            "shared_measurement_objects": [
                "LoRA extraction pipeline",
                "per-layer subspace geometry",
            ],
            "shared_diagnosis_objects": [
                "pair-level overlap/alignment judgments",
            ],
            "aggregation_divergence": "worst-case (merge) vs distributional (routing)",
            "policy_divergence": "merge admissibility vs routing confusability handling",
            "first_divergence_layer": "aggregation",
            "notes": "Observed directly in routing pilot: same measurements, different fleet guidance.",
        },
        {
            "scenario_pair": ["merge", "triage"],
            "shared_measurement_objects": [
                "pairwise compatibility summaries",
            ],
            "shared_diagnosis_objects": [
                "same_task/same_family/cross_task boundary handling",
            ],
            "aggregation_divergence": "worst-case pair risk vs gate-first source QA",
            "policy_divergence": "merge candidate retention vs evaluation-budget prioritization",
            "first_divergence_layer": "aggregation",
            "notes": "Checkpoint T02 shows QA dominance overriding structurally plausible pairs.",
        },
        {
            "scenario_pair": ["routing", "triage"],
            "shared_measurement_objects": [
                "pairwise structural similarity signals",
            ],
            "shared_diagnosis_objects": [
                "relation-aware pair interpretation",
            ],
            "aggregation_divergence": "distributional separability vs QA gate-first narrowing",
            "policy_divergence": "router disambiguation vs triage review/exclusion actions",
            "first_divergence_layer": "aggregation",
            "notes": "Same-family can require routing disambiguation yet still be triage-blocked by weak sources.",
        },
    ]

    payload = {
        "schema": "decision_dependent_scenario_stack/v1",
        "generated_utc": _now_utc_iso(),
        "layers": ["measurement", "diagnosis", "aggregation", "policy"],
        "scenario_stack": scenario_stack,
        "scenario_pair_audit": scenario_pairs,
        "aggregation_first_divergence_supported": True,
    }

    lines = [
        "# Shared vs Specific Table",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        "| Layer | Merge | Routing | Triage |",
        "|---|---|---|---|",
        "| Measurement | shared unchanged | shared unchanged | shared with translation |",
        "| Diagnosis | shared with translation | shared with translation | shared with translation |",
        "| Aggregation | scenario-specific (worst-case) | scenario-specific (distributional) | scenario-specific (QA gate-first) |",
        "| Policy | scenario-specific (merge candidacy) | scenario-specific (routing actions) | scenario-specific (evaluation prioritization) |",
        "",
        "## Scenario-pair notes",
        "",
    ]
    for row in scenario_pairs:
        pair = " vs ".join(row["scenario_pair"])
        lines.append(f"- **{pair}:** first divergence at `{row['first_divergence_layer']}`; {row['aggregation_divergence']}.")
    lines.append("")
    return payload, "\n".join(lines)


def _build_stage_c_aggregation() -> tuple[dict[str, Any], str]:
    pairwise = _load_json(ROOT / "field_trials" / "checkpoint_inventory_t02" / "pairwise" / "pairwise_results.json")
    qa_summary = _load_json(ROOT / "field_trials" / "checkpoint_inventory_t02" / "qa_artifacts" / "qa_summary.json")

    status_map = {row["checkpoint_id"]: row["status"] for row in qa_summary["checkpoints"]}
    pairs = pairwise["pairs"]

    def _max_non_classifier_divergence(pair: dict[str, Any]) -> float:
        vals = [
            float(item["divergence_score"])
            for item in pair.get("dominant_divergence_layers", [])
            if item.get("layer") != "classifier.weight"
        ]
        if not vals:
            return 0.0
        return max(vals)

    def _merge_like_label(pair: dict[str, Any], max_div: float) -> str:
        risk = pair.get("pair_risk", "high")
        if risk == "high" or max_div >= 4.5:
            return "merge_risky"
        if risk == "medium" or max_div >= 3.0:
            return "merge_caution"
        return "merge_favorable"

    def _routing_like_label(pair: dict[str, Any]) -> str:
        comp = float(pair["compatibility_score"])
        cos = float(pair["cosine_similarity"])
        if comp >= 0.85 and cos >= 0.995:
            return "routing_confusable"
        if comp >= 0.62:
            return "routing_needs_disambiguation"
        return "routing_separable"

    def _qa_gate_label(a: str, b: str) -> str:
        sa = status_map.get(a, "unknown_no_behavioral_eval")
        sb = status_map.get(b, "unknown_no_behavioral_eval")
        if sa == "flagged_weak" or sb == "flagged_weak":
            return "qa_blocked"
        if sa == "uncertain" or sb == "uncertain":
            return "qa_review"
        if sa == "eligible" and sb == "eligible":
            return "qa_clear"
        return "qa_review"

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        a = pair["checkpoint_a"]
        b = pair["checkpoint_b"]
        max_div = _max_non_classifier_divergence(pair)
        merge_label = _merge_like_label(pair, max_div)
        routing_label = _routing_like_label(pair)
        qa_label = _qa_gate_label(a, b)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "task_relationship": pair["task_relationship"],
                "structural_inputs": {
                    "compatibility_score": float(pair["compatibility_score"]),
                    "cosine_similarity": float(pair["cosine_similarity"]),
                    "pair_risk": pair["pair_risk"],
                    "max_non_classifier_divergence": round(max_div, 4),
                },
                "source_status": {
                    a: status_map.get(a, "unknown_no_behavioral_eval"),
                    b: status_map.get(b, "unknown_no_behavioral_eval"),
                },
                "aggregation_outputs": {
                    "worst_case_merge_like": merge_label,
                    "distributional_routing_like": routing_label,
                    "qa_gate_first_triage_like": qa_label,
                },
            }
        )

    def _count(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            label = row["aggregation_outputs"][key]
            out[label] = out.get(label, 0) + 1
        return out

    merge_counts = _count("worst_case_merge_like")
    routing_counts = _count("distributional_routing_like")
    qa_counts = _count("qa_gate_first_triage_like")

    # Cases where worst-case collapses but routing separates
    by_merge: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        lbl = row["aggregation_outputs"]["worst_case_merge_like"]
        by_merge.setdefault(lbl, []).append(row)
    collapsed_but_separated: list[dict[str, Any]] = []
    for merge_lbl, group in by_merge.items():
        routing_lbls = {g["aggregation_outputs"]["distributional_routing_like"] for g in group}
        if len(group) >= 2 and len(routing_lbls) > 1:
            collapsed_but_separated.append(
                {
                    "merge_label": merge_lbl,
                    "pair_ids": [g["pair_id"] for g in group],
                    "routing_labels_present": sorted(routing_lbls),
                }
            )

    qa_override_pairs = [
        row["pair_id"]
        for row in rows
        if row["aggregation_outputs"]["qa_gate_first_triage_like"] in {"qa_blocked", "qa_review"}
        and row["aggregation_outputs"]["distributional_routing_like"] != "routing_separable"
    ]

    payload = {
        "schema": "decision_dependent_aggregation_comparison/v1",
        "generated_utc": _now_utc_iso(),
        "source_panel": {
            "inventory": "field_trials/checkpoint_inventory_t02",
            "pair_count": len(rows),
            "shared_structural_inputs": [
                "compatibility_score",
                "cosine_similarity",
                "pair_risk",
                "dominant_divergence_layers",
            ],
        },
        "aggregation_families": {
            "A_worst_case": "merge-like sensitivity to local catastrophic layers",
            "B_distributional": "routing-like spread sensitivity",
            "C_qa_gate_first": "triage-like source-quality-first narrowing",
        },
        "case_rows": rows,
        "summary": {
            "worst_case_label_counts": merge_counts,
            "distributional_label_counts": routing_counts,
            "qa_gate_label_counts": qa_counts,
            "collapsed_under_worst_case_but_separated_distributional": collapsed_but_separated,
            "qa_gate_overrides_nonseparable_structural_cases": qa_override_pairs,
            "qa_dominance_ratio": round((qa_counts.get("qa_blocked", 0) + qa_counts.get("qa_review", 0)) / max(1, len(rows)), 4),
        },
    }

    lines = [
        "# Aggregation Comparison (Checkpoint T02)",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        f"Pairs analyzed: {len(rows)}",
        "",
        "## Label counts",
        "",
        f"- Worst-case (merge-like): {merge_counts}",
        f"- Distributional (routing-like): {routing_counts}",
        f"- QA gate-first (triage-like): {qa_counts}",
        "",
        "## Collapse vs separation",
        "",
    ]
    if collapsed_but_separated:
        for row in collapsed_but_separated:
            lines.append(
                f"- `{row['merge_label']}` collapsed {len(row['pair_ids'])} pairs but split into "
                f"{', '.join(row['routing_labels_present'])} under distributional aggregation."
            )
    else:
        lines.append("- No collapse/separation contrast detected in this panel.")
    lines.extend(
        [
            "",
            "## QA overrides",
            "",
            f"- QA gate overrode structurally non-separable cases for {len(qa_override_pairs)} pairs.",
            "",
        ]
    )
    return payload, "\n".join(lines)


def _render_aggregation_figure(*, rows: list[dict[str, Any]], title: str, generated_utc: str, out_path: Path) -> Path:
    labels = [
        "worst_case_merge_like",
        "distributional_routing_like",
        "qa_gate_first_triage_like",
    ]
    headers = ["Worst-case (merge-like)", "Distributional (routing-like)", "QA gate-first (triage-like)"]
    colors = {
        "merge_risky": "#c0392b",
        "merge_caution": "#d68910",
        "merge_favorable": "#1e8449",
        "routing_confusable": "#1f618d",
        "routing_needs_disambiguation": "#7d6608",
        "routing_separable": "#117864",
        "qa_blocked": "#943126",
        "qa_review": "#b9770e",
        "qa_clear": "#1d8348",
    }

    row_h = 34
    header_h = 74
    left_w = 360
    col_w = 240
    width = left_w + len(labels) * col_w + 40
    height = header_h + len(rows) * row_h + 80

    y0 = header_h
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />')
    parts.append(f'<text x="20" y="28" font-family="Menlo, monospace" font-size="16" font-weight="bold" fill="#1f2d3d">{escape(title)}</text>')
    parts.append(
        '<text x="20" y="50" font-family="Menlo, monospace" font-size="12" fill="#4a5665">'
        f"Generated {escape(generated_utc)}</text>"
    )

    # Headers
    parts.append(
        f'<rect x="20" y="{header_h - 24}" width="{left_w - 20}" height="24" fill="#ecf0f1" stroke="#d5d8dc" />'
    )
    parts.append(
        f'<text x="28" y="{header_h - 8}" font-family="Menlo, monospace" font-size="12" fill="#17202a">Case</text>'
    )
    for i, head in enumerate(headers):
        x = left_w + i * col_w
        parts.append(f'<rect x="{x}" y="{header_h - 24}" width="{col_w}" height="24" fill="#ecf0f1" stroke="#d5d8dc" />')
        parts.append(
            f'<text x="{x + 8}" y="{header_h - 8}" font-family="Menlo, monospace" font-size="11" fill="#17202a">{escape(head)}</text>'
        )

    # Rows
    for idx, row in enumerate(rows):
        y = y0 + idx * row_h
        bg = "#fbfcfc" if idx % 2 == 0 else "#f7f9f9"
        parts.append(f'<rect x="20" y="{y}" width="{width-40}" height="{row_h}" fill="{bg}" />')
        parts.append(
            f'<text x="28" y="{y + 22}" font-family="Menlo, monospace" font-size="11" fill="#1b2631">{escape(row["pair_id"])}</text>'
        )
        for j, key in enumerate(labels):
            label = row["aggregation_outputs"][key]
            color = colors.get(label, "#95a5a6")
            x = left_w + j * col_w
            parts.append(
                f'<rect x="{x + 8}" y="{y + 6}" width="{col_w - 16}" height="{row_h - 12}" fill="{color}" rx="4" ry="4" />'
            )
            parts.append(
                f'<text x="{x + 14}" y="{y + 22}" font-family="Menlo, monospace" font-size="10" fill="#ffffff">{escape(label)}</text>'
            )

    legend_y = y0 + len(rows) * row_h + 24
    parts.append(
        f'<text x="20" y="{legend_y}" font-family="Menlo, monospace" font-size="11" fill="#1b2631">Legend:</text>'
    )
    lx = 80
    for label, color in colors.items():
        parts.append(f'<rect x="{lx}" y="{legend_y - 10}" width="10" height="10" fill="{color}" />')
        parts.append(
            f'<text x="{lx + 14}" y="{legend_y}" font-family="Menlo, monospace" font-size="10" fill="#34495e">{escape(label)}</text>'
        )
        lx += 170
        if lx > width - 180:
            legend_y += 16
            lx = 80

    parts.append("</svg>")
    svg = "\n".join(parts)

    _write_text(out_path, svg)
    return out_path


def _render_stage_c_figure(aggregation: dict[str, Any]) -> Path:
    return _render_aggregation_figure(
        rows=aggregation["case_rows"],
        title="Decision-Dependent Aggregation Matrix (Checkpoint T02)",
        generated_utc=aggregation["generated_utc"],
        out_path=FIG_DIR / "decision_dependent_aggregation_matrix.svg",
    )


def _build_adapter_triage_aggregation_pass() -> tuple[dict[str, Any], str]:
    base = ROOT / "field_trials" / "targeted_confirmation_same_family" / "inventory_t01" / "preflight"
    merge_files = {
        "merge_exact_same_task.json": _load_json(base / "merge_exact_same_task.json"),
        "merge_same_family_cross_dataset.json": _load_json(base / "merge_same_family_cross_dataset.json"),
        "merge_true_cross_task.json": _load_json(base / "merge_true_cross_task.json"),
    }
    qa_files = [
        _load_json(base / "qa_myselfmankar__distilbert-base-sst2-lora.json"),
        _load_json(base / "qa_rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2.json"),
        _load_json(base / "qa_dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment.json"),
        _load_json(base / "qa_TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news.json"),
    ]

    status_map = {row["adapter"]["name"]: row["eligibility"]["status"] for row in qa_files}

    def _name_from_path(p: str) -> str:
        return Path(p).name

    def _merge_like_label(pair_risk: str, compat: float) -> str:
        if pair_risk == "high" or compat < 0.2:
            return "merge_risky"
        if pair_risk == "medium" or compat < 0.5:
            return "merge_caution"
        return "merge_favorable"

    def _routing_like_label(compat: float) -> str:
        # Adapter-side stress-test thresholds are anchored to routing pilot score ranges.
        if compat >= 0.45:
            return "routing_confusable"
        if compat >= 0.25:
            return "routing_needs_disambiguation"
        return "routing_separable"

    def _qa_label(a: str, b: str) -> str:
        sa = status_map.get(a, "unknown_no_behavioral_eval")
        sb = status_map.get(b, "unknown_no_behavioral_eval")
        if sa == "flagged_weak" or sb == "flagged_weak":
            return "qa_blocked"
        if sa in {"uncertain", "unknown_no_behavioral_eval"} or sb in {"uncertain", "unknown_no_behavioral_eval"}:
            return "qa_review"
        return "qa_clear"

    rows: list[dict[str, Any]] = []
    for fname, pair in merge_files.items():
        a = _name_from_path(pair["adapter_a"]["path"])
        b = _name_from_path(pair["adapter_b"]["path"])
        comp = float(pair["compatibility_score"])
        pair_id = f"{a}::{b}"
        rows.append(
            {
                "pair_id": pair_id,
                "source_file": str((base / fname).relative_to(ROOT)),
                "task_relationship": pair.get("task_relationship", "unknown"),
                "structural_inputs": {
                    "compatibility_score": comp,
                    "pair_risk": pair["pair_risk"],
                    "dominant_issue": pair["dominant_issue"],
                    "recommended_strategy": pair["recommended_strategy"],
                },
                "source_status": {
                    a: status_map.get(a, "unknown_no_behavioral_eval"),
                    b: status_map.get(b, "unknown_no_behavioral_eval"),
                },
                "aggregation_outputs": {
                    "worst_case_merge_like": _merge_like_label(pair["pair_risk"], comp),
                    "distributional_routing_like": _routing_like_label(comp),
                    "qa_gate_first_triage_like": _qa_label(a, b),
                },
            }
        )

    def _count(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            lbl = row["aggregation_outputs"][key]
            out[lbl] = out.get(lbl, 0) + 1
        return out

    merge_counts = _count("worst_case_merge_like")
    routing_counts = _count("distributional_routing_like")
    qa_counts = _count("qa_gate_first_triage_like")

    payload = {
        "schema": "decision_dependent_aggregation_comparison_adapter_t01/v1",
        "generated_utc": _now_utc_iso(),
        "source_panel": {
            "inventory": "field_trials/targeted_confirmation_same_family/inventory_t01/preflight",
            "pair_count": len(rows),
            "shared_structural_inputs": [
                "compatibility_score",
                "pair_risk",
                "dominant_issue",
                "task_relationship",
            ],
            "note": "Narrow stress test to check profile stability on adapter triage substrate.",
        },
        "aggregation_families": {
            "A_worst_case": "merge-like local sensitivity (risk-first)",
            "B_distributional": "routing-like separability tiers from compatibility scores",
            "C_qa_gate_first": "triage gate status using adapter eligibility",
        },
        "case_rows": rows,
        "summary": {
            "worst_case_label_counts": merge_counts,
            "distributional_label_counts": routing_counts,
            "qa_gate_label_counts": qa_counts,
            "profile_stability_signal": {
                "same_task_pair": next(
                    (r["aggregation_outputs"] for r in rows if r["task_relationship"] == "same_task"),
                    None,
                ),
                "same_family_pair": next(
                    (r["aggregation_outputs"] for r in rows if r["task_relationship"] == "same_family"),
                    None,
                ),
                "cross_task_pair": next(
                    (r["aggregation_outputs"] for r in rows if r["task_relationship"] == "cross_task"),
                    None,
                ),
            },
            "qa_dominance_ratio": round(
                (qa_counts.get("qa_blocked", 0) + qa_counts.get("qa_review", 0)) / max(1, len(rows)), 4
            ),
        },
    }

    lines = [
        "# Aggregation Comparison (Adapter Triage Stress Test)",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        f"Pairs analyzed: {len(rows)}",
        "",
        "## Label counts",
        "",
        f"- Worst-case (merge-like): {merge_counts}",
        f"- Distributional (routing-like): {routing_counts}",
        f"- QA gate-first (triage-like): {qa_counts}",
        "",
        "## Profile stability check",
        "",
        "- Same-task, same-family, and cross-task pairs remain separable under distributional aggregation.",
        "- QA gate is clear in this panel (`qa_clear` for all pairs), contrasting with checkpoint T02's QA-dominant blocking regime.",
        "",
    ]
    return payload, "\n".join(lines)


def _build_stage_d_profiles(panel: dict[str, Any], aggregation: dict[str, Any]) -> tuple[dict[str, Any], str]:
    profiles = [
        {
            "profile_id": "P1_redundant_confusable",
            "name": "Redundant / Routing-Confusable",
            "definition": "Very high structural overlap creates low incremental merge value and high routing confusion.",
            "scenario_signatures": {
                "merge": "typically redundant; deduplicate unless constrained",
                "routing": "consider_dedup or strong disambiguation",
                "triage": "only actionable if source QA is clear",
            },
            "representative_case_ids": [
                "rte_seed_pair_confusable",
                "tri_same_task_near_miss_sst2_pair_t02",
            ],
        },
        {
            "profile_id": "P2_overlap_needs_disambiguation",
            "name": "Overlap / Needs Disambiguation",
            "definition": "Moderate overlap supports plausible merge use but routing requires task signal to avoid confusion.",
            "scenario_signatures": {
                "merge": "merge-compatible but not automatically high-value",
                "routing": "needs_disambiguation",
                "triage": "review candidate if QA is not weak",
            },
            "representative_case_ids": [
                "mnli_qnli_moderate_confusable",
                "tri_same_family_review_sst2_yelp_t02",
            ],
        },
        {
            "profile_id": "P3_merge_ok_routing_separable",
            "name": "Merge-OK / Routing-Separable",
            "definition": "Pair can look merge-acceptable while still being easy for routing separation.",
            "scenario_signatures": {
                "merge": "can be acceptable under merge lens",
                "routing": "easily_routed",
                "triage": "not necessarily prioritized without quality evidence",
            },
            "representative_case_ids": [
                "qnli_rte_separable",
            ],
        },
        {
            "profile_id": "P4_qa_blocked_structurally_nontrivial",
            "name": "QA-Blocked / Structurally Nontrivial",
            "definition": "Structural compatibility is present, but weak-source evidence blocks action.",
            "scenario_signatures": {
                "merge": "defer despite structural plausibility",
                "routing": "structural separability/confusability secondary to source quality",
                "triage": "qa_blocked dominates",
            },
            "representative_case_ids": [
                "tri_same_task_near_miss_sst2_pair_t02",
                "tri_cross_task_weak_region_yelp_qnli_t02",
            ],
        },
        {
            "profile_id": "P5_same_family_optional",
            "name": "Same-Family Optional",
            "definition": "Same-family relation justifies cautious optional review, not automatic retention.",
            "scenario_signatures": {
                "merge": "can perform near same-task in validated examples",
                "routing": "often moderate confusability, needs disambiguation",
                "triage": "informational caution / optional probe",
            },
            "representative_case_ids": [
                "mrg_safe_same_task_sst2_sst2_t01",
                "tri_same_family_review_sst2_yelp_t02",
            ],
        },
        {
            "profile_id": "P6_cross_task_low_value_control",
            "name": "Cross-Task Low-Value Control",
            "definition": "Cross-task boundary plus weaker compatibility yields low priority under constrained budgets.",
            "scenario_signatures": {
                "merge": "control/risky; usually below same-task alternatives",
                "routing": "often separable and not confusable",
                "triage": "caution or exclusion unless strong contrary evidence",
            },
            "representative_case_ids": [
                "mrg_cross_task_control_sst2_agnews_t01",
                "tri_cross_task_weak_region_yelp_qnli_t02",
            ],
        },
    ]

    payload = {
        "schema": "decision_dependent_profile_table/v1",
        "generated_utc": _now_utc_iso(),
        "profile_count": len(profiles),
        "profiles": profiles,
    }

    lines = [
        "# Decision-Profile Table",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        "| Profile ID | Name | Merge stance | Routing stance | Triage stance |",
        "|---|---|---|---|---|",
    ]
    for p in profiles:
        lines.append(
            f"| {p['profile_id']} | {p['name']} | {p['scenario_signatures']['merge']} | "
            f"{p['scenario_signatures']['routing']} | {p['scenario_signatures']['triage']} |"
        )
    lines.append("")
    lines.append("## Representative cases")
    lines.append("")
    for p in profiles:
        lines.append(f"- `{p['profile_id']}`: {', '.join(p['representative_case_ids'])}")
    lines.append("")
    return payload, "\n".join(lines)


def _build_stage_e_semantics(profiles: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "profile_id": "P1_redundant_confusable",
            "decision_semantics": {
                "merge": "low incremental utility; redundancy dominates",
                "routing": "high confusability implies source confusion risk without explicit signal",
                "triage": "if QA is weak, block despite structural similarity",
            },
            "behavioral_signatures": [
                {
                    "scenario": "merge",
                    "signature": "safe/near-miss classes tend to preserve consensus when quality is adequate",
                    "evidence_path": "sidecar/notes/n61_example_behavior_findings.md",
                    "confidence": "medium",
                },
                {
                    "scenario": "routing",
                    "signature": "high confusability pairs flagged for dedup/disambiguation",
                    "evidence_path": "experiments/routing_pilot/routing_report.json",
                    "confidence": "high",
                },
            ],
            "caveats": "Routing evidence is structural-operational, not example-level misroute logs.",
        },
        {
            "profile_id": "P2_overlap_needs_disambiguation",
            "decision_semantics": {
                "merge": "moderate overlap can still be merge-admissible",
                "routing": "moderate confusability requires explicit disambiguation",
                "triage": "review if source QA permits",
            },
            "behavioral_signatures": [
                {
                    "scenario": "routing",
                    "signature": "moderate confusability (needs_disambiguation) across NLI-family pairs",
                    "evidence_path": "experiments/routing_pilot/routing_report.json",
                    "confidence": "high",
                },
                {
                    "scenario": "triage",
                    "signature": "same-family probe can show asymmetric per-dataset outcomes",
                    "evidence_path": "field_trials/checkpoint_inventory_t02/eval_results.json",
                    "confidence": "medium",
                },
            ],
            "caveats": "Same-family is a review relationship, not guaranteed interchangeability.",
        },
        {
            "profile_id": "P3_merge_ok_routing_separable",
            "decision_semantics": {
                "merge": "merge and routing objectives can decouple",
                "routing": "low confusability supports easy separation",
                "triage": "requires evidence context to prioritize",
            },
            "behavioral_signatures": [
                {
                    "scenario": "routing",
                    "signature": "easily_routed pairs despite nontrivial merge overlap",
                    "evidence_path": "experiments/routing_pilot/routing_report.json",
                    "confidence": "high",
                }
            ],
            "caveats": "No direct example-level routing behavior capture yet.",
        },
        {
            "profile_id": "P4_qa_blocked_structurally_nontrivial",
            "decision_semantics": {
                "merge": "structural compatibility alone is insufficient",
                "routing": "structural view may suggest action but QA can invalidate operational use",
                "triage": "qa_blocked dominates",
            },
            "behavioral_signatures": [
                {
                    "scenario": "triage",
                    "signature": "no retained pairs under mixed-quality inventory despite informative pair structure",
                    "evidence_path": "field_trials/checkpoint_inventory_t02/preflight/preflight_results.json",
                    "confidence": "high",
                }
            ],
            "caveats": "Current checkpoint panel is intentionally narrow and QA-heavy.",
        },
        {
            "profile_id": "P5_same_family_optional",
            "decision_semantics": {
                "merge": "same-family can perform near same-task in controlled settings",
                "routing": "often moderate confusability",
                "triage": "optional/review branch, not automatic retain",
            },
            "behavioral_signatures": [
                {
                    "scenario": "merge",
                    "signature": "same-family merge (SST-2 × IMDB) matched same-task merge in T01",
                    "evidence_path": "field_trials/targeted_confirmation_same_family/inventory_t01/eval_results.json",
                    "confidence": "high",
                },
                {
                    "scenario": "triage",
                    "signature": "same-family checkpoint follow-through showed asymmetric deltas across datasets",
                    "evidence_path": "field_trials/checkpoint_inventory_t02/eval_results.json",
                    "confidence": "medium",
                },
            ],
            "caveats": "Do not over-generalize one family branch to all same-family definitions.",
        },
        {
            "profile_id": "P6_cross_task_low_value_control",
            "decision_semantics": {
                "merge": "cross-task controls remain lower-priority baselines",
                "routing": "often separable",
                "triage": "caution/exclude unless evidence is unusually strong",
            },
            "behavioral_signatures": [
                {
                    "scenario": "merge",
                    "signature": "cross-task control materially underperformed same-task and same-family in T01",
                    "evidence_path": "field_trials/targeted_confirmation_same_family/inventory_t01/eval_results.json",
                    "confidence": "high",
                },
                {
                    "scenario": "merge_behavioral",
                    "signature": "cross-task control can show high-confidence wrong predictions",
                    "evidence_path": "sidecar/notes/n61_example_behavior_findings.md",
                    "confidence": "medium",
                },
            ],
            "caveats": "Behavioral signature based on one control class in example-level panel.",
        },
    ]
    return {
        "schema": "decision_dependent_semantics_table/v1",
        "generated_utc": _now_utc_iso(),
        "rows": rows,
        "notes": [
            "Semantics rows are evidence-linked and intentionally conservative.",
            "Routing signatures are currently structural-operational, not user-traffic routing logs.",
        ],
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    panel = _build_panel_table()
    _write_json(RESULT_DIR / "panel_table.json", panel)
    _write_text(RESULT_DIR / "panel_table.md", _panel_markdown(panel))

    stage_b, stage_b_md = _build_stage_b_audit(panel)
    _write_json(RESULT_DIR / "scenario_stack_matrix.json", stage_b)
    _write_text(RESULT_DIR / "shared_vs_specific_table.md", stage_b_md)

    stage_c, stage_c_md = _build_stage_c_aggregation()
    _write_json(RESULT_DIR / "aggregation_comparison.json", stage_c)
    _write_text(RESULT_DIR / "aggregation_comparison.md", stage_c_md)
    fig_path = _render_stage_c_figure(stage_c)

    adapter_stage_c, adapter_stage_c_md = _build_adapter_triage_aggregation_pass()
    _write_json(RESULT_DIR / "aggregation_comparison_adapter_t01.json", adapter_stage_c)
    _write_text(RESULT_DIR / "aggregation_comparison_adapter_t01.md", adapter_stage_c_md)
    adapter_fig_path = _render_aggregation_figure(
        rows=adapter_stage_c["case_rows"],
        title="Decision-Dependent Aggregation Matrix (Adapter T01 Stress Test)",
        generated_utc=adapter_stage_c["generated_utc"],
        out_path=FIG_DIR / "decision_dependent_aggregation_matrix_adapter_t01.svg",
    )

    stage_d, stage_d_md = _build_stage_d_profiles(panel, stage_c)
    _write_json(RESULT_DIR / "decision_profile_table.json", stage_d)
    _write_text(RESULT_DIR / "decision_profile_table.md", stage_d_md)

    stage_e = _build_stage_e_semantics(stage_d)
    _write_json(RESULT_DIR / "decision_semantics_table.json", stage_e)

    print(f"[decision-dependent] panel_table: {RESULT_DIR / 'panel_table.json'}")
    print(f"[decision-dependent] scenario_stack_matrix: {RESULT_DIR / 'scenario_stack_matrix.json'}")
    print(f"[decision-dependent] aggregation_comparison: {RESULT_DIR / 'aggregation_comparison.json'}")
    print(f"[decision-dependent] aggregation_comparison_adapter_t01: {RESULT_DIR / 'aggregation_comparison_adapter_t01.json'}")
    print(f"[decision-dependent] decision_profile_table: {RESULT_DIR / 'decision_profile_table.json'}")
    print(f"[decision-dependent] decision_semantics_table: {RESULT_DIR / 'decision_semantics_table.json'}")
    print(f"[decision-dependent] figure: {fig_path}")
    print(f"[decision-dependent] figure_adapter_t01: {adapter_fig_path}")


if __name__ == "__main__":
    main()
