#!/usr/bin/env python3
"""Checkpoint inventory field trial T01 (CPU-only, small encoder).

Runs an end-to-end trial for full fine-tuned checkpoints as the inventory unit:
1. Build manifest
2. Bootstrap behavioral evidence
3. Generate single-checkpoint QA artifacts
4. Generate pairwise checkpoint comparison outputs
5. Emit preflight-style bundle (inventory summary/action plan/run bundle)
6. Run tiny follow-through evaluation subset
7. Write field note + trial memo
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.run_bundle import emit_run_bundle
from gradience.vnext.inventory.summary import (
    build_action_plan,
    build_inventory_summary,
    derive_inventory_policy_summary,
    format_action_plan,
    format_inventory_summary,
)
from gradience.vnext.merge.eligibility import ConfidenceLevel, EligibilityStatus, classify_eligibility
from gradience.vnext.merge.qa_report import AdapterSummary, MergeQAReport
from gradience.vnext.merge.task_families import family_label

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIAL_DIR = Path(__file__).resolve().parent

STAGE_B_RESULTS_DEFAULT = REPO_ROOT / "experiments" / "ring2_checkpoint_delta" / "stage_b_representation_c_results.json"
BASE_MODEL_DEFAULT = "distilbert-base-uncased"
INVENTORY_ID = "checkpoint_inventory_t01"
RUN_ID = "run_001"


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    checkpoint_rel_path: str
    task: str
    dataset_name: str
    dataset_config: str
    split: str
    text_columns: tuple[str, ...]
    label_column: str
    metric_name: str
    metric_direction: str
    provenance_notes: str
    expected_trial_role: tuple[str, ...]


PANEL: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        checkpoint_id="sst2_s42",
        checkpoint_rel_path="experiments/ring2_checkpoint_delta/checkpoints/sst2_s42",
        task="sst2",
        dataset_name="glue",
        dataset_config="sst2",
        split="validation",
        text_columns=("sentence",),
        label_column="label",
        metric_name="accuracy",
        metric_direction="higher_is_better",
        provenance_notes="Ring 2 local fine-tune checkpoint (seed 42).",
        expected_trial_role=("same_task", "retained_candidate"),
    ),
    CheckpointSpec(
        checkpoint_id="sst2_s123",
        checkpoint_rel_path="experiments/ring2_checkpoint_delta/checkpoints/sst2_s123",
        task="sst2",
        dataset_name="glue",
        dataset_config="sst2",
        split="validation",
        text_columns=("sentence",),
        label_column="label",
        metric_name="accuracy",
        metric_direction="higher_is_better",
        provenance_notes="Ring 2 local fine-tune checkpoint (seed 123).",
        expected_trial_role=("same_task", "retained_candidate"),
    ),
    CheckpointSpec(
        checkpoint_id="mrpc_s42",
        checkpoint_rel_path="experiments/ring2_checkpoint_delta/checkpoints/mrpc_s42",
        task="mrpc",
        dataset_name="glue",
        dataset_config="mrpc",
        split="validation",
        text_columns=("sentence1", "sentence2"),
        label_column="label",
        metric_name="accuracy",
        metric_direction="higher_is_better",
        provenance_notes="Ring 2 local fine-tune checkpoint (seed 42).",
        expected_trial_role=("cross_task", "optional"),
    ),
    CheckpointSpec(
        checkpoint_id="qnli_s42",
        checkpoint_rel_path="experiments/ring2_checkpoint_delta/checkpoints/qnli_s42",
        task="qnli",
        dataset_name="glue",
        dataset_config="qnli",
        split="validation",
        text_columns=("question", "sentence"),
        label_column="label",
        metric_name="accuracy",
        metric_direction="higher_is_better",
        provenance_notes="Ring 2 local fine-tune checkpoint (seed 42); structurally diffuse in Stage B.",
        expected_trial_role=("cross_task", "marginal"),
    ),
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _find_glue_arrow_cache(dataset_config: str) -> Path:
    root = Path.home() / ".cache" / "huggingface" / "datasets" / "glue" / dataset_config / "0.0.0"
    if not root.exists():
        raise FileNotFoundError(f"No cache root found for GLUE/{dataset_config}: {root}")

    candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        validation_path = candidate / "glue-validation.arrow"
        if validation_path.exists():
            return validation_path
    raise FileNotFoundError(f"No glue-validation.arrow found for GLUE/{dataset_config}")


def _load_validation_dataset(dataset_config: str) -> Dataset:
    validation_path = _find_glue_arrow_cache(dataset_config)
    return Dataset.from_file(str(validation_path))


def _predict_labels(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, Any]],
    text_columns: tuple[str, ...],
    *,
    batch_size: int = 32,
    max_length: int = 256,
) -> list[int]:
    preds: list[int] = []
    model.eval()

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        if len(text_columns) == 1:
            texts = [str(row[text_columns[0]]) for row in batch]
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            texts_a = [str(row[text_columns[0]]) for row in batch]
            texts_b = [str(row[text_columns[1]]) for row in batch]
            inputs = tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        with torch.no_grad():
            logits = model(**inputs).logits
            preds.extend(torch.argmax(logits, dim=-1).tolist())
    return preds


def _accuracy(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(labels, preds))
    return correct / len(labels)


def _sample_rows(spec: CheckpointSpec, sample_size: int, seed: int) -> list[dict[str, Any]]:
    ds = _load_validation_dataset(spec.dataset_config)
    n = min(sample_size, len(ds))
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    return [ds[i] for i in indices[:n]]


def _evaluate_single_model(
    *,
    model_path: str,
    tokenizer_path: str,
    rows: list[dict[str, Any]],
    text_columns: tuple[str, ...],
    label_column: str,
    num_labels: int = 2,
    local_files_only: bool = True,
    init_seed: int | None = None,
) -> float:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_files_only)
    if init_seed is not None:
        torch.manual_seed(int(init_seed))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        local_files_only=local_files_only,
    )
    labels = [int(row[label_column]) for row in rows]
    preds = _predict_labels(model, tokenizer, rows, text_columns)
    return _accuracy(labels, preds)


def _build_manifest(*, base_model: str, stage_b: dict[str, Any]) -> dict[str, Any]:
    panel_map = {item["checkpoint_id"]: item for item in stage_b["checkpoint_panel"]}
    checkpoints: list[dict[str, Any]] = []

    for spec in PANEL:
        stage_b_item = panel_map[spec.checkpoint_id]
        rel_path = str(stage_b_item["path"])
        abs_path = str((REPO_ROOT / rel_path).resolve())
        tf = family_label(spec.task, spec.task)
        checkpoints.append(
            {
                "checkpoint_id": spec.checkpoint_id,
                "path": rel_path,
                "path_abs": abs_path,
                "shared_base_model": base_model,
                "task_label": spec.task,
                "task_family": tf if tf is not None else "unmapped",
                "dataset": spec.task,
                "metric_direction": spec.metric_direction,
                "provenance_notes": spec.provenance_notes,
                "expected_trial_role": list(spec.expected_trial_role),
            }
        )

    return {
        "schema": "checkpoint_inventory_manifest/v1",
        "inventory_id": INVENTORY_ID,
        "timestamp_utc": _now_utc_iso(),
        "base_model": base_model,
        "checkpoint_count": len(checkpoints),
        "trial_scope": {
            "cpu_only": True,
            "model_family": "distilbert encoder",
            "tasks": sorted({spec.task for spec in PANEL}),
            "notes": "Single shared base; classification-only panel.",
        },
        "checkpoints": checkpoints,
    }


def _bootstrap_evidence(
    *,
    base_model: str,
    stage_b: dict[str, Any],
    sample_size: int,
    seed: int,
    margin: float,
    base_head_seed: int,
) -> dict[str, Any]:
    panel_map = {item["checkpoint_id"]: item for item in stage_b["checkpoint_panel"]}
    base_score_cache: dict[str, float] = {}
    rows_cache: dict[str, list[dict[str, Any]]] = {}
    results: list[dict[str, Any]] = []

    for spec in PANEL:
        if spec.task not in rows_cache:
            rows_cache[spec.task] = _sample_rows(spec, sample_size=sample_size, seed=seed)
        rows = rows_cache[spec.task]

        checkpoint_rel = str(panel_map[spec.checkpoint_id]["path"])
        checkpoint_abs = str((REPO_ROOT / checkpoint_rel).resolve())

        checkpoint_score = _evaluate_single_model(
            model_path=checkpoint_abs,
            tokenizer_path=checkpoint_abs,
            rows=rows,
            text_columns=spec.text_columns,
            label_column=spec.label_column,
            num_labels=2,
        )

        if spec.task not in base_score_cache:
            base_score_cache[spec.task] = _evaluate_single_model(
                model_path=base_model,
                tokenizer_path=base_model,
                rows=rows,
                text_columns=spec.text_columns,
                label_column=spec.label_column,
                num_labels=2,
                init_seed=base_head_seed,
            )
        base_score = base_score_cache[spec.task]

        qa = classify_eligibility(
            adapter_path=checkpoint_rel,
            adapter_metric=checkpoint_score,
            base_metric=base_score,
            metric_name=spec.metric_name,
            lower_is_better=False,
            eval_dataset=spec.task,
            margin=margin,
        )

        delta = checkpoint_score - base_score
        results.append(
            {
                "checkpoint_id": spec.checkpoint_id,
                "task": spec.task,
                "dataset": spec.dataset_name,
                "dataset_config": spec.dataset_config,
                "split": spec.split,
                "sample_size": len(rows),
                "sample_seed": seed,
                "metric_name": spec.metric_name,
                "metric_direction": spec.metric_direction,
                "checkpoint_score": round(float(checkpoint_score), 4),
                "base_score": round(float(base_score), 4),
                "delta_vs_base": round(float(delta), 4),
                "evidence_status_candidate": qa.status.value,
                "evidence_notes": qa.notes,
                "path": checkpoint_rel,
            }
        )

    status_counts = {"eligible": 0, "uncertain": 0, "flagged_weak": 0, "unknown_no_behavioral_eval": 0}
    for row in results:
        status_counts[row["evidence_status_candidate"]] += 1

    return {
        "schema": "checkpoint_inventory_evidence_bootstrap/v1",
        "inventory_id": INVENTORY_ID,
        "timestamp_utc": _now_utc_iso(),
        "base_model": base_model,
        "sample_size": sample_size,
        "sample_seed": seed,
        "margin": margin,
        "checkpoints": results,
        "summary": {
            "checkpoint_count": len(results),
            "status_counts": status_counts,
            "mean_delta_vs_base": round(sum(r["delta_vs_base"] for r in results) / len(results), 4),
            "evidence_gate_dominated_candidate": status_counts["eligible"] < len(results),
        },
    }


def _confidence_from_status(status: EligibilityStatus, delta: float, has_structural_flags: bool) -> ConfidenceLevel:
    abs_delta = abs(delta)
    if status == EligibilityStatus.ELIGIBLE:
        if abs_delta >= 0.03 and not has_structural_flags:
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.MEDIUM
    if status == EligibilityStatus.FLAGGED_WEAK:
        return ConfidenceLevel.MEDIUM if abs_delta >= 0.03 else ConfidenceLevel.LOW
    return ConfidenceLevel.LOW


def _build_qa_artifacts(
    *,
    base_model: str,
    stage_b: dict[str, Any],
    evidence: dict[str, Any],
    margin: float,
) -> tuple[list[AdapterQAArtifact], dict[str, Any]]:
    structural_rows = {
        row["checkpoint_id"]: row for row in stage_b["single_artifact_audit"]["checkpoints"]
    }
    evidence_rows = {row["checkpoint_id"]: row for row in evidence["checkpoints"]}

    artifacts: list[AdapterQAArtifact] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in PANEL:
        s_row = structural_rows[spec.checkpoint_id]
        e_row = evidence_rows[spec.checkpoint_id]

        q = classify_eligibility(
            adapter_path=e_row["path"],
            adapter_metric=float(e_row["checkpoint_score"]),
            base_metric=float(e_row["base_score"]),
            metric_name=e_row["metric_name"],
            lower_is_better=False,
            eval_dataset=spec.task,
            margin=margin,
        )
        status = q.status
        structural_review = s_row.get("audit_status") == "review"
        structural_flags = list(s_row.get("flags", []))

        if structural_review and status == EligibilityStatus.ELIGIBLE:
            status = EligibilityStatus.UNCERTAIN

        delta = float(e_row["delta_vs_base"])
        confidence = _confidence_from_status(status, delta=delta, has_structural_flags=bool(structural_flags))

        reasons = [q.notes or "Behavioral evidence classified with margin."]
        if structural_review:
            reasons.append("Structural summary marked this checkpoint for review.")
        if structural_flags:
            reasons.append("Structural flags: " + ", ".join(structural_flags))

        base_score = float(e_row["base_score"])
        checkpoint_score = float(e_row["checkpoint_score"])
        headroom = max(1e-8, 1.0 - base_score)
        margin_confidence = max(0.0, min(1.0, (checkpoint_score - base_score) / headroom))

        artifact = AdapterQAArtifact(
            adapter_name=spec.checkpoint_id,
            adapter_path=str((REPO_ROOT / e_row["path"]).resolve()),
            base_model=base_model,
            rank_nominal=0,
            n_layers=int(s_row.get("n_layers", 0)),
            utilization_mean=float(s_row.get("mean_energy_at_8", 0.0)),
            utilization_median=float(s_row.get("mean_energy_at_8", 0.0)),
            stable_rank_mean=float(s_row.get("mean_stable_rank", 0.0)),
            energy_rank_90_p50=None,
            rank_waste_ratio=max(0.0, 1.0 - min(1.0, float(s_row.get("mean_energy_at_8", 0.0)))),
            structural_flags=structural_flags,
            eval_available=True,
            eval_dataset=spec.task,
            metric_name=e_row["metric_name"],
            adapter_score=checkpoint_score,
            base_score=base_score,
            lower_is_better=False,
            beats_base=checkpoint_score > base_score,
            margin_confidence=margin_confidence,
            status=status,
            confidence=confidence,
            reasons=reasons,
            notes=[
                "Checkpoint QA derived from Ring 2 summary representation + bootstrap evidence.",
            ],
        )
        artifacts.append(artifact)
        summary_rows.append(
            {
                "checkpoint_id": spec.checkpoint_id,
                "status": status.value,
                "confidence": confidence.value,
                "delta_vs_base": round(delta, 4),
                "structural_flags": structural_flags,
            }
        )

    status_counts = {"eligible": 0, "uncertain": 0, "flagged_weak": 0, "unknown_no_behavioral_eval": 0}
    for a in artifacts:
        status_counts[a.status.value] += 1

    summary = {
        "schema": "checkpoint_inventory_qa_summary/v1",
        "inventory_id": INVENTORY_ID,
        "timestamp_utc": _now_utc_iso(),
        "status_counts": status_counts,
        "checkpoints": summary_rows,
    }
    return artifacts, summary


def _task_relationship(task_a: str, task_b: str) -> tuple[str, str | None]:
    if task_a == task_b:
        return "same_task", None
    fam = family_label(task_a, task_b)
    if fam is not None:
        return "same_family", fam
    return "cross_task", None


def _build_pairwise_results(stage_b: dict[str, Any]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for row in stage_b["pairwise_comparison"]["pairs"]:
        rel, fam = _task_relationship(str(row["task_a"]), str(row["task_b"]))
        pairs.append(
            {
                "pair_id": row["pair_id"],
                "checkpoint_a": row["checkpoint_a"],
                "checkpoint_b": row["checkpoint_b"],
                "task_a": row["task_a"],
                "task_b": row["task_b"],
                "task_relationship": rel,
                "task_family": fam,
                "compatibility_score": round(float(row.get("compatibility_score", 0.0)), 4),
                "pair_risk": row.get("pair_risk", "medium"),
                "mean_abs_energy_at_8_delta": round(float(row.get("mean_abs_energy_at_8_delta", 0.0)), 4),
                "mean_abs_stable_rank_delta": round(float(row.get("mean_abs_stable_rank_delta", 0.0)), 4),
                "mean_abs_effective_rank_delta": round(float(row.get("mean_abs_effective_rank_delta", 0.0)), 4),
                "dominant_divergence_layers": list(row.get("dominant_divergence_layers", [])),
                "notes": row.get("notes", ""),
            }
        )

    pair_count = len(pairs)
    low = sum(1 for p in pairs if p["pair_risk"] == "low")
    med = sum(1 for p in pairs if p["pair_risk"] == "medium")
    high = sum(1 for p in pairs if p["pair_risk"] == "high")
    same_task_count = sum(1 for p in pairs if p["task_relationship"] == "same_task")
    same_family_count = sum(1 for p in pairs if p["task_relationship"] == "same_family")
    cross_task_count = sum(1 for p in pairs if p["task_relationship"] == "cross_task")

    return {
        "schema": "checkpoint_inventory_pairwise/v1",
        "inventory_id": INVENTORY_ID,
        "timestamp_utc": _now_utc_iso(),
        "pairs": pairs,
        "summary": {
            "pair_count": pair_count,
            "risk_counts": {"low": low, "medium": med, "high": high},
            "relationship_counts": {
                "same_task": same_task_count,
                "same_family": same_family_count,
                "cross_task": cross_task_count,
            },
            "mean_compatibility": round(sum(p["compatibility_score"] for p in pairs) / max(1, pair_count), 4),
        },
    }


def _dominant_issue_from_pair(pair: dict[str, Any]) -> tuple[str, str]:
    comp = float(pair.get("compatibility_score", 0.0))
    mean_energy_delta = float(pair.get("mean_abs_energy_at_8_delta", 0.0))
    mean_stable_delta = float(pair.get("mean_abs_stable_rank_delta", 0.0))
    mean_effective_delta = float(pair.get("mean_abs_effective_rank_delta", 0.0))

    if comp >= 0.86 and mean_energy_delta < 0.03:
        return "high_redundancy", "Layer summary profiles are very close; pair appears structurally redundant."
    if comp >= 0.78:
        return "partial_redundancy", "Pair is structurally similar with non-trivial divergence in selected layers."
    if mean_stable_delta > 1.6 or mean_effective_delta > 24.0:
        return "subspace_conflict", "Large rank-profile deltas suggest subspace-level conflict risk."
    return "norm_imbalance", "Compatibility drop is dominated by energy/scale imbalance across layers."


def _strategy_for_pair_risk(pair_risk: str) -> tuple[str, str]:
    if pair_risk == "high":
        return "audit_aware", "Defer; inspect divergence layers and require stronger evidence first."
    if pair_risk == "medium":
        return "norm_equalized", "Plausible with caution; use conservative weighting and targeted validation."
    return "linear", "Structurally favorable; eligible for exploratory evaluation."


def _verdict_distribution_for_risk(pair_risk: str) -> dict[str, int]:
    if pair_risk == "high":
        return {"conflicting": 1, "safe": 0, "redundant": 0, "review": 1}
    if pair_risk == "medium":
        return {"conflicting": 0, "safe": 0, "redundant": 0, "review": 1}
    return {"conflicting": 0, "safe": 1, "redundant": 0, "review": 0}


def _build_merge_reports(
    *,
    qa_artifacts: list[AdapterQAArtifact],
    pairwise: dict[str, Any],
    base_model: str,
) -> list[MergeQAReport]:
    qa_by_name = {a.adapter_name: a for a in qa_artifacts}
    reports: list[MergeQAReport] = []

    for row in pairwise["pairs"]:
        qa_a = qa_by_name[row["checkpoint_a"]]
        qa_b = qa_by_name[row["checkpoint_b"]]

        pair_risk = str(row.get("pair_risk", "medium"))
        dominant_issue, detail = _dominant_issue_from_pair(row)
        strategy, action = _strategy_for_pair_risk(pair_risk)
        confidence = ConfidenceLevel.HIGH if pair_risk == "low" else ConfidenceLevel.MEDIUM

        task_rel = str(row.get("task_relationship", "cross_task"))
        if task_rel == "cross_task":
            advisory = "Cross-task advisory: require stronger behavioral evidence before promotion."
        elif task_rel == "same_family":
            fam = row.get("task_family", "shared_family")
            advisory = f"Same-family advisory ({fam}): plausible, but still validate behavior."
        else:
            advisory = None

        report = MergeQAReport(
            adapter_a=AdapterSummary(
                path=qa_a.adapter_path,
                rank=0,
                alpha=0.0,
                n_layers=qa_a.n_layers,
                base_model=base_model,
                eligibility_status=qa_a.status.value,
            ),
            adapter_b=AdapterSummary(
                path=qa_b.adapter_path,
                rank=0,
                alpha=0.0,
                n_layers=qa_b.n_layers,
                base_model=base_model,
                eligibility_status=qa_b.status.value,
            ),
            pair_risk=pair_risk,
            dominant_issue=dominant_issue,
            dominant_issue_detail=detail,
            recommended_action=action,
            recommended_strategy=strategy,
            confidence=confidence,
            confidence_note="Derived from summary-based checkpoint deltas; behavior-first confirmation recommended.",
            caveats=(
                "Checkpoint inventory trial path (not merge execution).",
                "Representation C structural summaries were used for pair scoring.",
            ),
            verdict_distribution=_verdict_distribution_for_risk(pair_risk),
            compatibility_score=float(row.get("compatibility_score", 0.0)),
            task_relationship_advisory=advisory,
            task_relationship=task_rel,
        )
        reports.append(report)

    return reports


def _action_plan_to_dict(plan: Any) -> dict[str, Any]:
    return {
        "exclude": list(plan.exclude),
        "same_task_priority": list(plan.same_task_priority),
        "cross_task_caution": list(plan.cross_task_caution),
        "evaluate_first": list(plan.evaluate_first),
        "summary_line": plan.summary_line,
        "total_pairs": plan.total_pairs,
        "retained_count": plan.retained_count,
        "cross_task_count": plan.cross_task_count,
        "behavioral_evidence_count": plan.behavioral_evidence_count,
        "total_source_count": plan.total_source_count,
        "retained_pair_detail": [list(x) for x in plan.retained_pair_detail],
        "near_miss_candidates": list(plan.near_miss_candidates),
        "near_miss_detail": [list(x) for x in plan.near_miss_detail],
        "near_miss_severity": [list(x) for x in plan.near_miss_severity],
    }


def _run_follow_through(
    *,
    base_model: str,
    pairwise: dict[str, Any],
    action_plan: Any,
    sample_size: int,
    seed: int,
    stage_b: dict[str, Any],
    base_head_seed: int,
) -> dict[str, Any]:
    panel_map = {item["checkpoint_id"]: item for item in stage_b["checkpoint_panel"]}
    spec_by_id = {spec.checkpoint_id: spec for spec in PANEL}
    base_score_cache: dict[str, float] = {}

    def eval_checkpoint(checkpoint_id: str) -> tuple[float, float]:
        spec = spec_by_id[checkpoint_id]
        rows = _sample_rows(spec, sample_size=sample_size, seed=seed)
        checkpoint_path = str((REPO_ROOT / panel_map[checkpoint_id]["path"]).resolve())
        score = _evaluate_single_model(
            model_path=checkpoint_path,
            tokenizer_path=checkpoint_path,
            rows=rows,
            text_columns=spec.text_columns,
            label_column=spec.label_column,
            num_labels=2,
        )
        if spec.task not in base_score_cache:
            base_score_cache[spec.task] = _evaluate_single_model(
                model_path=base_model,
                tokenizer_path=base_model,
                rows=rows,
                text_columns=spec.text_columns,
                label_column=spec.label_column,
                num_labels=2,
                init_seed=base_head_seed,
            )
        base_score = base_score_cache[spec.task]
        return float(score), float(base_score)

    pair_choice = None
    pair_category = "retained_evaluate_first_pair"
    pair_interpretation = "Evaluate-first same-task pair remains the strongest immediate comparison target."
    if action_plan.evaluate_first:
        pair_choice = action_plan.evaluate_first[0]
    else:
        for row in pairwise["pairs"]:
            if row["task_relationship"] == "same_task":
                pair_choice = f"{row['checkpoint_a']} × {row['checkpoint_b']}"
                pair_category = "same_task_near_miss_probe"
                pair_interpretation = "No retained pair remained; probing the strongest same-task near-miss candidate."
                break

    records: list[dict[str, Any]] = []

    if pair_choice:
        left, right = [x.strip() for x in pair_choice.split("×")]
        a_score, base_score = eval_checkpoint(left)
        b_score, _ = eval_checkpoint(right)
        strongest = max(a_score, b_score)
        records.append(
            {
                "category": pair_category,
                "checkpoints": [left, right],
                "task": spec_by_id[left].task,
                "sample_size": sample_size,
                "sample_seed": seed,
                "scores": {
                    left: round(a_score, 4),
                    right: round(b_score, 4),
                    "base": round(base_score, 4),
                },
                "strongest_source_score": round(strongest, 4),
                "delta_vs_strongest_source": {
                    left: round(a_score - strongest, 4),
                    right: round(b_score - strongest, 4),
                },
                "interpretation": pair_interpretation,
            }
        )

    for category, checkpoint_id in (
        ("optional_single_checkpoint", "mrpc_s42"),
        ("lower_priority_control_checkpoint", "qnli_s42"),
    ):
        score, base_score = eval_checkpoint(checkpoint_id)
        records.append(
            {
                "category": category,
                "checkpoints": [checkpoint_id],
                "task": spec_by_id[checkpoint_id].task,
                "sample_size": sample_size,
                "sample_seed": seed,
                "scores": {
                    checkpoint_id: round(score, 4),
                    "base": round(base_score, 4),
                },
                "delta_vs_strongest_source": {checkpoint_id: 0.0},
                "delta_vs_base": round(score - base_score, 4),
                "interpretation": (
                    "Optional same-inventory checkpoint sanity check."
                    if category == "optional_single_checkpoint"
                    else "Lower-priority control checkpoint; used to sanity-check weak/risky region."
                ),
            }
        )

    return {
        "schema": "checkpoint_inventory_follow_through/v1",
        "inventory_id": INVENTORY_ID,
        "timestamp_utc": _now_utc_iso(),
        "records": records,
        "summary": {
            "evaluation_count": len(records),
            "retained_or_evaluate_first_count": sum(1 for r in records if "retained" in r["category"]),
            "control_count": sum(1 for r in records if "control" in r["category"]),
        },
    }


def _format_follow_through_table(records: list[dict[str, Any]]) -> str:
    lines = ["| Category | Checkpoint(s) | Task | Score(s) | Note |", "|---|---|---|---|---|"]
    for row in records:
        checkpoints = ", ".join(row["checkpoints"])
        score_bits = []
        for key, value in row["scores"].items():
            score_bits.append(f"{key}={value:.4f}")
        scores = "; ".join(score_bits)
        lines.append(
            f"| {row['category']} | {checkpoints} | {row['task']} | {scores} | {row['interpretation']} |"
        )
    return "\n".join(lines)


def _derive_usefulness_rating(
    *,
    status_counts: dict[str, int],
    pair_count: int,
    retained_count: int,
    policy_summary: dict[str, str],
) -> dict[str, str]:
    qa_useful = "high" if len([k for k, v in status_counts.items() if v > 0]) >= 3 else "medium"
    pair_useful = "high" if pair_count > 0 and retained_count < pair_count else "medium"
    action_useful = "high" if retained_count <= max(2, math.ceil(pair_count * 0.4)) else "medium"
    report_clarity = "high"
    broader = "high" if policy_summary.get("exploration_posture") in {"narrow", "moderate"} else "medium"
    return {
        "qa_usefulness": qa_useful,
        "pairwise_comparison_usefulness": pair_useful,
        "action_plan_usefulness": action_useful,
        "report_clarity": report_clarity,
        "broader_use_case_plausibility": broader,
    }


def _build_field_note(
    *,
    manifest: dict[str, Any],
    policy_summary: dict[str, str],
    action_plan: Any,
    qa_summary: dict[str, Any],
    follow_through: dict[str, Any],
) -> str:
    status_counts = qa_summary["status_counts"]
    return (
        "# Checkpoint Inventory T01 — Field Note\n\n"
        f"Generated: {_now_utc_iso()}\n\n"
        "## 1. Inventory\n\n"
        f"- Included `{manifest['checkpoint_count']}` full fine-tuned checkpoints sharing base `{manifest['base_model']}`.\n"
        "- Panel shape: 2 same-task (SST-2), 2 cross-task controls (MRPC, QNLI).\n"
        "- Chosen for CPU-feasible, cached-data execution with known Ring 2 structural variation.\n\n"
        "## 2. Gradience Stance\n\n"
        f"- Dominant driver: `{policy_summary['dominant_driver']}`\n"
        f"- Inventory type: `{policy_summary['inventory_type']}`\n"
        f"- Exploration posture: `{policy_summary['exploration_posture']}`\n"
        f"- Evaluate-first: {', '.join(action_plan.evaluate_first) if action_plan.evaluate_first else 'none'}\n"
        f"- QA status counts: eligible={status_counts.get('eligible', 0)}, uncertain={status_counts.get('uncertain', 0)}, "
        f"flagged_weak={status_counts.get('flagged_weak', 0)}\n"
        f"- Action-plan summary: {action_plan.summary_line}\n\n"
        "## 3. Follow-through Results\n\n"
        f"{_format_follow_through_table(follow_through['records'])}\n\n"
        "## 4. Product Judgment\n\n"
        f"- Did checkpoint workflow feel useful? {'yes' if action_plan.retained_count < action_plan.total_pairs else 'partially'}\n"
        f"- Did evidence gate remain central? {'yes' if policy_summary['dominant_driver'] == 'source_qa' else 'partially'}\n"
        "- Did reports explain triage clearly? yes (preflight summary + action plan + review packet).\n"
        "- Does this feel like a real broader use case? yes, but still narrow and CPU-bounded.\n"
    )


def _build_trial_memo(
    *,
    policy_summary: dict[str, str],
    action_plan: Any,
    qa_summary: dict[str, Any],
    follow_through: dict[str, Any],
    usefulness: dict[str, str],
    measurement_schema: dict[str, Any],
) -> str:
    if action_plan.retained_count > 0:
        subset_line = "- Yes, in bounded form. The workflow produced legible triage artifacts and a clear evaluation-first subset."
    else:
        subset_line = (
            "- Yes, in bounded form. The workflow produced legible triage artifacts and a clear near-miss review subset "
            "even when the retained set was empty."
        )

    return (
        "# Checkpoint Inventory T01 — Trial Memo\n\n"
        f"Generated: {_now_utc_iso()}\n\n"
        "## 1. What transferred unchanged from adapter workflows?\n\n"
        "- Evidence bootstrap as the practical gate.\n"
        "- QA eligibility framing (eligible / uncertain / flagged_weak).\n"
        "- Pairwise compatibility-driven narrowing.\n"
        "- Inventory summary + action-plan + preflight packet reporting pattern.\n\n"
        "## 2. What was checkpoint-specific?\n\n"
        "- Artifact unit is full fine-tuned checkpoints rather than adapters.\n"
        "- Structural path uses summary-based checkpoint deltas (Representation C) rather than factor extraction.\n"
        "- No merge execution path was exercised.\n\n"
        "## 3. Did the workflow feel naturally broader than merge preflight?\n\n"
        "- Yes. The trial operated as checkpoint inventory triage and prioritization, not merge strategy selection.\n"
        f"- Candidate space narrowed from {action_plan.total_pairs} to {action_plan.retained_count} without merge execution.\n\n"
        "## 4. What broke or felt forced?\n\n"
        "- Same-family non-identical-task pair was absent in this panel, so that branch was not stress-tested.\n"
        "- Scope remains narrow (single backbone, small panel, CPU-only).\n\n"
        "## 5. Is checkpoint inventory triage a credible external use case?\n\n"
        f"{subset_line}\n"
        f"- Dominant driver remained `{policy_summary['dominant_driver']}`, reinforcing trust-aware behavior.\n\n"
        "## 6. What should happen next?\n\n"
        "- Run one additional checkpoint inventory with a same-family non-identical-task pair.\n"
        "- Keep merge execution out of scope until broader checkpoint evidence is stronger.\n"
        "- Preserve CPU-first constraints for comparability with this trial.\n\n"
        "## Measurement Schema Snapshot\n\n"
        "```json\n"
        f"{json.dumps(measurement_schema, indent=2)}\n"
        "```\n\n"
        "## Follow-through Snapshot\n\n"
        f"{_format_follow_through_table(follow_through['records'])}\n\n"
        "## Workflow Usefulness Ratings\n\n"
        f"- QA usefulness: {usefulness['qa_usefulness']}\n"
        f"- Pairwise comparison usefulness: {usefulness['pairwise_comparison_usefulness']}\n"
        f"- Action plan usefulness: {usefulness['action_plan_usefulness']}\n"
        f"- Report clarity: {usefulness['report_clarity']}\n"
        f"- Broader-use-case plausibility: {usefulness['broader_use_case_plausibility']}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint inventory field trial T01")
    parser.add_argument("--stage-b-results", default=str(STAGE_B_RESULTS_DEFAULT))
    parser.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    parser.add_argument("--evidence-sample-size", type=int, default=200)
    parser.add_argument("--evidence-seed", type=int, default=42)
    parser.add_argument("--follow-through-sample-size", type=int, default=300)
    parser.add_argument("--follow-through-seed", type=int, default=123)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--base-head-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_b = _load_json(Path(args.stage_b_results))

    manifest_path = TRIAL_DIR / "manifest.json"
    evidence_path = TRIAL_DIR / "evidence" / "bootstrap_results.json"
    qa_dir = TRIAL_DIR / "qa_artifacts"
    pairwise_path = TRIAL_DIR / "pairwise" / "pairwise_results.json"
    preflight_dir = TRIAL_DIR / "preflight"
    eval_results_path = TRIAL_DIR / "eval_results.json"
    field_note_path = TRIAL_DIR / "field_note.md"
    trial_memo_path = TRIAL_DIR / "trial_memo.md"

    manifest = _build_manifest(base_model=args.base_model, stage_b=stage_b)
    _write_json(manifest_path, manifest)

    evidence = _bootstrap_evidence(
        base_model=args.base_model,
        stage_b=stage_b,
        sample_size=args.evidence_sample_size,
        seed=args.evidence_seed,
        margin=args.margin,
        base_head_seed=args.base_head_seed,
    )
    _write_json(evidence_path, evidence)

    qa_artifacts, qa_summary = _build_qa_artifacts(
        base_model=args.base_model,
        stage_b=stage_b,
        evidence=evidence,
        margin=args.margin,
    )

    qa_dir.mkdir(parents=True, exist_ok=True)
    for artifact in qa_artifacts:
        _write_json(qa_dir / f"{artifact.adapter_name}.json", artifact.to_dict())
    _write_json(qa_dir / "qa_summary.json", qa_summary)

    pairwise = _build_pairwise_results(stage_b)
    _write_json(pairwise_path, pairwise)

    merge_reports = _build_merge_reports(
        qa_artifacts=qa_artifacts,
        pairwise=pairwise,
        base_model=args.base_model,
    )

    inventory_summary = build_inventory_summary(qa_artifacts, merge_reports)
    action_plan = build_action_plan(qa_artifacts, merge_reports)
    policy_summary = derive_inventory_policy_summary(inventory_summary, action_plan)

    preflight_qa_dir = preflight_dir / "qa"
    preflight_pair_dir = preflight_dir / "pair_reports"
    preflight_inventory_dir = preflight_dir / "inventory"
    preflight_run_dir = preflight_dir / RUN_ID
    preflight_qa_dir.mkdir(parents=True, exist_ok=True)
    preflight_pair_dir.mkdir(parents=True, exist_ok=True)
    preflight_inventory_dir.mkdir(parents=True, exist_ok=True)

    for artifact in qa_artifacts:
        _write_json(preflight_qa_dir / f"{artifact.adapter_name}.json", artifact.to_dict())
    for report in merge_reports:
        name_a = Path(report.adapter_a.path).name
        name_b = Path(report.adapter_b.path).name
        _write_json(preflight_pair_dir / f"{name_a}__{name_b}.json", report.to_dict())

    _write_json(preflight_inventory_dir / "inventory_summary.json", inventory_summary.to_dict())
    (preflight_inventory_dir / "inventory_summary.md").write_text(
        format_inventory_summary(inventory_summary, policy_summary=policy_summary) + "\n"
    )
    _write_json(preflight_inventory_dir / "inventory_action_plan.json", _action_plan_to_dict(action_plan))
    (preflight_inventory_dir / "inventory_action_plan.md").write_text(format_action_plan(action_plan) + "\n")

    emit_run_bundle(
        inventory_id=INVENTORY_ID,
        run_id=RUN_ID,
        run_dir=preflight_run_dir,
        qa_artifacts=qa_artifacts,
        merge_reports=merge_reports,
        action_plan=action_plan,
        base_model=args.base_model,
        policy_summary=policy_summary,
    )

    preflight_results = {
        "schema": "checkpoint_inventory_preflight/v1",
        "timestamp_utc": _now_utc_iso(),
        "inventory_id": INVENTORY_ID,
        "run_id": RUN_ID,
        "policy_summary": policy_summary,
        "inventory_summary": inventory_summary.to_dict(),
        "action_plan": _action_plan_to_dict(action_plan),
        "paths": {
            "qa_dir": str(preflight_qa_dir),
            "pair_reports_dir": str(preflight_pair_dir),
            "inventory_dir": str(preflight_inventory_dir),
            "run_bundle_dir": str(preflight_run_dir),
        },
    }
    _write_json(preflight_dir / "preflight_results.json", preflight_results)

    follow_through = _run_follow_through(
        base_model=args.base_model,
        pairwise=pairwise,
        action_plan=action_plan,
        sample_size=args.follow_through_sample_size,
        seed=args.follow_through_seed,
        stage_b=stage_b,
        base_head_seed=args.base_head_seed,
    )

    usefulness = _derive_usefulness_rating(
        status_counts=qa_summary["status_counts"],
        pair_count=pairwise["summary"]["pair_count"],
        retained_count=int(action_plan.retained_count),
        policy_summary=policy_summary,
    )

    measurement_schema = {
        "product_behavior": {
            "inventory_type": policy_summary["inventory_type"],
            "dominant_driver": policy_summary["dominant_driver"],
            "exploration_posture": policy_summary["exploration_posture"],
            "checkpoint_count": manifest["checkpoint_count"],
            "pair_count": pairwise["summary"]["pair_count"],
            "retained_count": int(action_plan.retained_count),
            "candidate_reduction": round(
                1.0 - (int(action_plan.retained_count) / max(1, int(action_plan.total_pairs))),
                4,
            ),
        },
        "evidence_behavior": {
            "eligible_count": qa_summary["status_counts"]["eligible"],
            "uncertain_count": qa_summary["status_counts"]["uncertain"],
            "weak_count": qa_summary["status_counts"]["flagged_weak"],
            "evidence_gate_dominated_decisions": policy_summary["dominant_driver"] == "source_qa",
        },
        "follow_through_behavior": follow_through["summary"],
        "workflow_usefulness": usefulness,
    }

    eval_results = {
        "schema": "checkpoint_inventory_eval_results/v1",
        "timestamp_utc": _now_utc_iso(),
        "inventory_id": INVENTORY_ID,
        "follow_through": follow_through,
        "measurement_schema": measurement_schema,
    }
    _write_json(eval_results_path, eval_results)

    field_note_path.write_text(
        _build_field_note(
            manifest=manifest,
            policy_summary=policy_summary,
            action_plan=action_plan,
            qa_summary=qa_summary,
            follow_through=follow_through,
        )
    )
    trial_memo_path.write_text(
        _build_trial_memo(
            policy_summary=policy_summary,
            action_plan=action_plan,
            qa_summary=qa_summary,
            follow_through=follow_through,
            usefulness=usefulness,
            measurement_schema=measurement_schema,
        )
    )

    print(f"[t01] manifest: {manifest_path}")
    print(f"[t01] evidence: {evidence_path}")
    print(f"[t01] qa_dir: {qa_dir}")
    print(f"[t01] pairwise: {pairwise_path}")
    print(f"[t01] preflight: {preflight_dir}")
    print(f"[t01] eval_results: {eval_results_path}")
    print(f"[t01] field_note: {field_note_path}")
    print(f"[t01] trial_memo: {trial_memo_path}")
    print(
        f"[t01] policy={policy_summary['inventory_type']} / "
        f"{policy_summary['dominant_driver']} / {policy_summary['exploration_posture']}"
    )


if __name__ == "__main__":
    main()
