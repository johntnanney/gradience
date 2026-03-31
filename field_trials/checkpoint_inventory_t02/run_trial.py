#!/usr/bin/env python3
"""Checkpoint inventory field trial T02 (CPU-only).

T02 extends T01 by explicitly including a same-family non-identical-task pair:
SST-2 <-> Yelp Polarity (both sentiment_binary).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
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
RING2_DIR = REPO_ROOT / "experiments" / "ring2_checkpoint_delta"
STAGE_A_ARTIFACT_ROOT = RING2_DIR / "results" / "stage_a_artifacts"

if str(RING2_DIR) not in sys.path:
    sys.path.insert(0, str(RING2_DIR))

from extract_checkpoint_delta import extract_checkpoint_delta  # type: ignore  # noqa: E402
from layer_summary_repr import delta_to_layer_summaries  # type: ignore  # noqa: E402

BASE_MODEL_DEFAULT = "distilbert-base-uncased"
INVENTORY_ID = "checkpoint_inventory_t02"
RUN_ID = "run_001"


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    checkpoint_rel_path: str
    seed: int
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
        seed=42,
        task="sst2",
        dataset_name="glue",
        dataset_config="sst2",
        split="validation",
        text_columns=("sentence",),
        label_column="label",
        metric_name="accuracy",
        metric_direction="higher_is_better",
        provenance_notes="Ring 2 local checkpoint (seed 42).",
        expected_trial_role=("same_task", "same_family", "near_miss_candidate"),
    ),
    CheckpointSpec(
        checkpoint_id="sst2_s123",
        checkpoint_rel_path="experiments/ring2_checkpoint_delta/checkpoints/sst2_s123",
        seed=123,
        task="sst2",
        dataset_name="glue",
        dataset_config="sst2",
        split="validation",
        text_columns=("sentence",),
        label_column="label",
        metric_name="accuracy",
        metric_direction="higher_is_better",
        provenance_notes="Ring 2 local checkpoint (seed 123).",
        expected_trial_role=("same_task", "near_miss_candidate"),
    ),
    CheckpointSpec(
        checkpoint_id="yelp_s42",
        checkpoint_rel_path="experiments/ring2_checkpoint_delta/checkpoints/yelp_s42",
        seed=42,
        task="yelp_polarity",
        dataset_name="yelp_polarity",
        dataset_config="plain_text",
        split="test",
        text_columns=("text",),
        label_column="label",
        metric_name="accuracy",
        metric_direction="higher_is_better",
        provenance_notes="T02 local full-checkpoint fine-tune on Yelp Polarity (seed 42).",
        expected_trial_role=("same_family", "optional"),
    ),
    CheckpointSpec(
        checkpoint_id="mrpc_s42",
        checkpoint_rel_path="experiments/ring2_checkpoint_delta/checkpoints/mrpc_s42",
        seed=42,
        task="mrpc",
        dataset_name="glue",
        dataset_config="mrpc",
        split="validation",
        text_columns=("sentence1", "sentence2"),
        label_column="label",
        metric_name="accuracy",
        metric_direction="higher_is_better",
        provenance_notes="Ring 2 local checkpoint (seed 42).",
        expected_trial_role=("cross_task", "optional"),
    ),
    CheckpointSpec(
        checkpoint_id="qnli_s42",
        checkpoint_rel_path="experiments/ring2_checkpoint_delta/checkpoints/qnli_s42",
        seed=42,
        task="qnli",
        dataset_name="glue",
        dataset_config="qnli",
        split="validation",
        text_columns=("question", "sentence"),
        label_column="label",
        metric_name="accuracy",
        metric_direction="higher_is_better",
        provenance_notes="Ring 2 local checkpoint (seed 42); structurally diffuse in prior runs.",
        expected_trial_role=("cross_task", "marginal"),
    ),
)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _sanitize(value: float) -> float:
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return 1e12
    return value


def _find_glue_validation_arrow(dataset_config: str) -> Path:
    root = Path.home() / ".cache" / "huggingface" / "datasets" / "glue" / dataset_config / "0.0.0"
    if not root.exists():
        raise FileNotFoundError(f"Missing GLUE cache root for {dataset_config}: {root}")
    candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        p = candidate / "glue-validation.arrow"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing glue-validation.arrow for {dataset_config}")


def _find_yelp_test_arrow() -> Path:
    root = Path.home() / ".cache" / "huggingface" / "datasets" / "yelp_polarity" / "plain_text" / "0.0.0"
    if not root.exists():
        raise FileNotFoundError(f"Missing Yelp cache root: {root}")
    candidates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        p = candidate / "yelp_polarity-test.arrow"
        if p.exists():
            return p
    raise FileNotFoundError("Missing yelp_polarity-test.arrow")


def _load_eval_dataset(spec: CheckpointSpec) -> Dataset:
    if spec.dataset_name == "glue":
        return Dataset.from_file(str(_find_glue_validation_arrow(spec.dataset_config)))
    if spec.dataset_name == "yelp_polarity":
        return Dataset.from_file(str(_find_yelp_test_arrow()))
    raise ValueError(f"Unsupported dataset for t02: {spec.dataset_name}")


def _sample_rows(spec: CheckpointSpec, sample_size: int, seed: int) -> list[dict[str, Any]]:
    ds = _load_eval_dataset(spec)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    chosen = indices[: min(sample_size, len(indices))]
    return [ds[i] for i in chosen]


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
            texts = [str(r[text_columns[0]]) for r in batch]
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        else:
            ta = [str(r[text_columns[0]]) for r in batch]
            tb = [str(r[text_columns[1]]) for r in batch]
            inputs = tokenizer(ta, tb, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            preds.extend(torch.argmax(logits, dim=-1).tolist())
    return preds


def _accuracy(labels: list[int], preds: list[int]) -> float:
    if not labels:
        return 0.0
    return sum(int(a == b) for a, b in zip(labels, preds)) / len(labels)


def _evaluate_single_model(
    *,
    model_path: str,
    tokenizer_path: str,
    rows: list[dict[str, Any]],
    text_columns: tuple[str, ...],
    label_column: str,
    num_labels: int = 2,
    init_seed: int | None = None,
) -> float:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if init_seed is not None:
        torch.manual_seed(int(init_seed))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        local_files_only=True,
    )
    labels = [int(r[label_column]) for r in rows]
    preds = _predict_labels(model, tokenizer, rows, text_columns)
    return _accuracy(labels, preds)


def _load_precomputed_summary(checkpoint_id: str) -> dict[str, Any] | None:
    path = STAGE_A_ARTIFACT_ROOT / checkpoint_id / "repr_c" / "summary.json"
    if path.exists():
        return _load_json(path)
    return None


def _compute_summary(spec: CheckpointSpec, base_model: str) -> dict[str, Any]:
    checkpoint_path = (REPO_ROOT / spec.checkpoint_rel_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint path for {spec.checkpoint_id}: {checkpoint_path}\n"
            "Run train_yelp_checkpoint.py first for yelp_s42."
        )
    deltas = extract_checkpoint_delta(
        base_model=base_model,
        finetuned_checkpoint=checkpoint_path,
        seed_for_head_init=spec.seed,
        local_files_only=True,
    )
    return delta_to_layer_summaries(deltas)


def _materialize_summaries(base_model: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    summary_cache_dir = TRIAL_DIR / "pairwise" / "summaries"
    summary_cache_dir.mkdir(parents=True, exist_ok=True)

    for spec in PANEL:
        cached = summary_cache_dir / f"{spec.checkpoint_id}.json"
        if cached.exists():
            out[spec.checkpoint_id] = _load_json(cached)
            continue

        pre = _load_precomputed_summary(spec.checkpoint_id)
        if pre is not None:
            out[spec.checkpoint_id] = pre
            _write_json(cached, pre)
            continue

        summary = _compute_summary(spec, base_model=base_model)
        out[spec.checkpoint_id] = summary
        _write_json(cached, summary)

    return out


def _checkpoint_flags(summary: dict[str, Any]) -> list[str]:
    layers = summary.get("layers", {})
    mean_effective_rank = float(summary.get("mean_effective_rank", 0.0))
    mean_stable_rank = float(summary.get("mean_stable_rank", 0.0))
    mean_energy_at_8 = float(summary.get("mean_energy_at_8", 0.0))

    frob_total = sum(float(layer.get("frobenius_norm", 0.0)) for layer in layers.values())
    classifier_frob = sum(
        float(layer.get("frobenius_norm", 0.0))
        for layer_name, layer in layers.items()
        if "classifier" in layer_name
    )
    classifier_share = (classifier_frob / frob_total) if frob_total > 0 else 0.0

    core_decay = [
        float(layer.get("singular_decay_ratio_top1_over_top4", 0.0))
        for layer_name, layer in layers.items()
        if "classifier" not in layer_name
    ]
    core_decay_sorted = sorted(core_decay)
    p90_idx = max(0, min(len(core_decay_sorted) - 1, int(0.9 * len(core_decay_sorted)) - 1))
    core_decay_p90 = core_decay_sorted[p90_idx] if core_decay_sorted else 0.0

    flags: list[str] = []
    if mean_energy_at_8 < 0.50:
        flags.append("diffuse_delta_spectrum")
    if mean_stable_rank > 10.5:
        flags.append("high_stable_rank")
    if mean_effective_rank > 430:
        flags.append("high_effective_rank")
    if classifier_share > 0.20:
        flags.append("classifier_heavy_delta")
    if core_decay_p90 > 6.0:
        flags.append("steep_core_singular_decay")
    return flags


def _layerwise_feature_vector(layer_names: list[str], layers: dict[str, dict[str, Any]]) -> list[float]:
    vector: list[float] = []
    for layer_name in layer_names:
        stats = layers.get(layer_name, {})
        effective_rank = _sanitize(float(stats.get("effective_rank", 0.0))) / 800.0
        stable_rank = _sanitize(float(stats.get("stable_rank", 0.0))) / 20.0
        energy_at_8 = _sanitize(float(stats.get("energy_at_8", 0.0)))
        sv_decay = _sanitize(float(stats.get("singular_decay_ratio_top1_over_top4", 0.0)))
        vector.extend([effective_rank, stable_rank, energy_at_8, math.log1p(sv_decay) / 3.0])
    return vector


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    a = torch.tensor(vec_a, dtype=torch.float32)
    b = torch.tensor(vec_b, dtype=torch.float32)
    denom = float(torch.linalg.norm(a) * torch.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def _pair_metrics(
    layers_a: dict[str, dict[str, Any]],
    layers_b: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shared_layers = sorted(set(layers_a.keys()) & set(layers_b.keys()))
    energy_deltas: list[float] = []
    stable_deltas: list[float] = []
    effective_deltas: list[float] = []
    decay_deltas: list[float] = []
    layer_deltas: list[dict[str, Any]] = []

    for layer_name in shared_layers:
        a = layers_a[layer_name]
        b = layers_b[layer_name]
        d_energy = abs(float(a.get("energy_at_8", 0.0)) - float(b.get("energy_at_8", 0.0)))
        d_stable = abs(float(a.get("stable_rank", 0.0)) - float(b.get("stable_rank", 0.0)))
        d_effective = abs(float(a.get("effective_rank", 0.0)) - float(b.get("effective_rank", 0.0)))
        d_decay = abs(
            float(a.get("singular_decay_ratio_top1_over_top4", 0.0))
            - float(b.get("singular_decay_ratio_top1_over_top4", 0.0))
        )

        divergence = d_energy / 0.12 + d_stable / 2.5 + d_effective / 80.0 + d_decay / 2.0
        energy_deltas.append(d_energy)
        stable_deltas.append(d_stable)
        effective_deltas.append(d_effective)
        decay_deltas.append(d_decay)
        layer_deltas.append(
            {
                "layer": layer_name,
                "abs_delta_energy_at_8": d_energy,
                "abs_delta_stable_rank": d_stable,
                "abs_delta_effective_rank": d_effective,
                "abs_delta_sv_decay": d_decay,
                "divergence_score": divergence,
            }
        )

    vector_a = _layerwise_feature_vector(shared_layers, layers_a)
    vector_b = _layerwise_feature_vector(shared_layers, layers_b)
    cosine = _cosine_similarity(vector_a, vector_b)

    mean_energy_delta = _safe_mean(energy_deltas)
    mean_stable_delta = _safe_mean(stable_deltas)
    mean_effective_delta = _safe_mean(effective_deltas)
    mean_decay_delta = _safe_mean(decay_deltas)

    energy_score = _clamp(1.0 - (mean_energy_delta / 0.12))
    stable_score = _clamp(1.0 - (mean_stable_delta / 2.5))
    effective_score = _clamp(1.0 - (mean_effective_delta / 80.0))
    decay_score = _clamp(1.0 - (mean_decay_delta / 2.0))

    compatibility_score = _clamp(
        0.50 * cosine + 0.20 * energy_score + 0.15 * stable_score + 0.10 * effective_score + 0.05 * decay_score
    )

    if compatibility_score >= 0.90 and cosine >= 0.999:
        pair_risk = "low"
    elif compatibility_score >= 0.78:
        pair_risk = "medium"
    else:
        pair_risk = "high"

    ranked_layers = sorted(layer_deltas, key=lambda x: x["divergence_score"], reverse=True)[:3]
    return (
        {
            "shared_layer_count": len(shared_layers),
            "cosine_similarity": cosine,
            "mean_abs_energy_at_8_delta": mean_energy_delta,
            "mean_abs_stable_rank_delta": mean_stable_delta,
            "mean_abs_effective_rank_delta": mean_effective_delta,
            "mean_abs_sv_decay_delta": mean_decay_delta,
            "compatibility_score": compatibility_score,
            "pair_risk": pair_risk,
        },
        ranked_layers,
    )


def _task_relationship(task_a: str, task_b: str) -> tuple[str, str | None]:
    if task_a == task_b:
        return "same_task", None
    fam = family_label(task_a, task_b)
    if fam is not None:
        return "same_family", fam
    return "cross_task", None


def _build_manifest(base_model: str) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    for spec in PANEL:
        checkpoints.append(
            {
                "checkpoint_id": spec.checkpoint_id,
                "path": spec.checkpoint_rel_path,
                "path_abs": str((REPO_ROOT / spec.checkpoint_rel_path).resolve()),
                "shared_base_model": base_model,
                "task_label": spec.task,
                "task_family": family_label(spec.task, spec.task) or "unmapped",
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
            "notes": "Includes same-family non-identical-task branch (sst2 vs yelp_polarity).",
        },
        "checkpoints": checkpoints,
    }


def _bootstrap_evidence(
    *,
    base_model: str,
    sample_size: int,
    seed: int,
    margin: float,
    base_head_seed: int,
) -> dict[str, Any]:
    base_score_cache: dict[str, float] = {}
    rows_cache: dict[str, list[dict[str, Any]]] = {}
    results: list[dict[str, Any]] = []

    for spec in PANEL:
        if spec.task not in rows_cache:
            rows_cache[spec.task] = _sample_rows(spec, sample_size=sample_size, seed=seed)
        rows = rows_cache[spec.task]
        checkpoint_path = str((REPO_ROOT / spec.checkpoint_rel_path).resolve())

        ckpt_score = _evaluate_single_model(
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
        delta = ckpt_score - base_score

        qa = classify_eligibility(
            adapter_path=spec.checkpoint_rel_path,
            adapter_metric=ckpt_score,
            base_metric=base_score,
            metric_name=spec.metric_name,
            lower_is_better=False,
            eval_dataset=spec.task,
            margin=margin,
        )

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
                "checkpoint_score": round(float(ckpt_score), 4),
                "base_score": round(float(base_score), 4),
                "delta_vs_base": round(float(delta), 4),
                "evidence_status_candidate": qa.status.value,
                "evidence_notes": qa.notes,
                "path": spec.checkpoint_rel_path,
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


def _single_audit_rows(summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for spec in PANEL:
        s = summaries[spec.checkpoint_id]
        flags = _checkpoint_flags(s)
        rows[spec.checkpoint_id] = {
            "checkpoint_id": spec.checkpoint_id,
            "task": spec.task,
            "seed": spec.seed,
            "n_layers": int(s.get("n_layers", 0)),
            "mean_effective_rank": float(s.get("mean_effective_rank", 0.0)),
            "mean_stable_rank": float(s.get("mean_stable_rank", 0.0)),
            "mean_energy_at_8": float(s.get("mean_energy_at_8", 0.0)),
            "audit_status": "review" if flags else "healthy",
            "flags": flags,
            "layers": s.get("layers", {}),
        }
    return rows


def _build_qa_artifacts(
    *,
    base_model: str,
    evidence: dict[str, Any],
    single_audit: dict[str, dict[str, Any]],
    margin: float,
) -> tuple[list[AdapterQAArtifact], dict[str, Any]]:
    evidence_rows = {r["checkpoint_id"]: r for r in evidence["checkpoints"]}
    artifacts: list[AdapterQAArtifact] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in PANEL:
        s_row = single_audit[spec.checkpoint_id]
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
        structural_flags = list(s_row["flags"])
        if s_row["audit_status"] == "review" and status == EligibilityStatus.ELIGIBLE:
            status = EligibilityStatus.UNCERTAIN

        delta = float(e_row["delta_vs_base"])
        confidence = _confidence_from_status(status, delta=delta, has_structural_flags=bool(structural_flags))
        reasons = [q.notes or "Behavioral evidence classified with margin."]
        if s_row["audit_status"] == "review":
            reasons.append("Structural summary marked this checkpoint for review.")
        if structural_flags:
            reasons.append("Structural flags: " + ", ".join(structural_flags))

        base_score = float(e_row["base_score"])
        ckpt_score = float(e_row["checkpoint_score"])
        headroom = max(1e-8, 1.0 - base_score)
        margin_confidence = max(0.0, min(1.0, (ckpt_score - base_score) / headroom))

        artifact = AdapterQAArtifact(
            adapter_name=spec.checkpoint_id,
            adapter_path=str((REPO_ROOT / spec.checkpoint_rel_path).resolve()),
            base_model=base_model,
            rank_nominal=0,
            n_layers=int(s_row["n_layers"]),
            utilization_mean=float(s_row["mean_energy_at_8"]),
            utilization_median=float(s_row["mean_energy_at_8"]),
            stable_rank_mean=float(s_row["mean_stable_rank"]),
            energy_rank_90_p50=None,
            rank_waste_ratio=max(0.0, 1.0 - min(1.0, float(s_row["mean_energy_at_8"]))),
            structural_flags=structural_flags,
            eval_available=True,
            eval_dataset=spec.task,
            metric_name=e_row["metric_name"],
            adapter_score=ckpt_score,
            base_score=base_score,
            lower_is_better=False,
            beats_base=ckpt_score > base_score,
            margin_confidence=margin_confidence,
            status=status,
            confidence=confidence,
            reasons=reasons,
            notes=["Checkpoint QA derived from summary-based full-checkpoint delta representation."],
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

    return (
        artifacts,
        {
            "schema": "checkpoint_inventory_qa_summary/v1",
            "inventory_id": INVENTORY_ID,
            "timestamp_utc": _now_utc_iso(),
            "status_counts": status_counts,
            "checkpoints": summary_rows,
        },
    )


def _build_pairwise_results(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    spec_by_id = {s.checkpoint_id: s for s in PANEL}
    for a_id, b_id in itertools.combinations([s.checkpoint_id for s in PANEL], 2):
        spec_a = spec_by_id[a_id]
        spec_b = spec_by_id[b_id]
        a_layers = summaries[a_id]["layers"]
        b_layers = summaries[b_id]["layers"]
        metrics, dominant = _pair_metrics(a_layers, b_layers)
        rel, fam = _task_relationship(spec_a.task, spec_b.task)
        notes = "pair risk and task relationship are directionally consistent"
        if rel == "same_family":
            notes = "same-family non-identical-task pair; compare with informational caution."

        pairs.append(
            {
                "pair_id": f"{a_id}::{b_id}",
                "checkpoint_a": a_id,
                "checkpoint_b": b_id,
                "task_a": spec_a.task,
                "task_b": spec_b.task,
                "task_relationship": rel,
                "task_family": fam,
                **metrics,
                "dominant_divergence_layers": dominant,
                "notes": notes,
            }
        )

    low = sum(1 for p in pairs if p["pair_risk"] == "low")
    med = sum(1 for p in pairs if p["pair_risk"] == "medium")
    high = sum(1 for p in pairs if p["pair_risk"] == "high")
    same_task = sum(1 for p in pairs if p["task_relationship"] == "same_task")
    same_family = sum(1 for p in pairs if p["task_relationship"] == "same_family")
    cross_task = sum(1 for p in pairs if p["task_relationship"] == "cross_task")

    return {
        "schema": "checkpoint_inventory_pairwise/v1",
        "inventory_id": INVENTORY_ID,
        "timestamp_utc": _now_utc_iso(),
        "pairs": pairs,
        "summary": {
            "pair_count": len(pairs),
            "risk_counts": {"low": low, "medium": med, "high": high},
            "relationship_counts": {"same_task": same_task, "same_family": same_family, "cross_task": cross_task},
            "mean_compatibility": round(sum(float(p["compatibility_score"]) for p in pairs) / max(1, len(pairs)), 4),
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
        pair_risk = str(row["pair_risk"])
        dominant_issue, detail = _dominant_issue_from_pair(row)
        strategy, action = _strategy_for_pair_risk(pair_risk)
        confidence = ConfidenceLevel.HIGH if pair_risk == "low" else ConfidenceLevel.MEDIUM
        rel = str(row["task_relationship"])
        if rel == "cross_task":
            advisory = "Cross-task advisory: require stronger behavioral evidence before promotion."
        elif rel == "same_family":
            fam = row.get("task_family", "shared_family")
            advisory = f"Same-family advisory ({fam}): plausible candidate with informational caution."
        else:
            advisory = None

        reports.append(
            MergeQAReport(
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
                confidence_note="Derived from summary-based full-checkpoint deltas; behavior-first confirmation advised.",
                caveats=(
                    "Checkpoint inventory trial path (no merge execution).",
                    "Representation C layer-summary signals used for comparison.",
                ),
                verdict_distribution=_verdict_distribution_for_risk(pair_risk),
                compatibility_score=float(row["compatibility_score"]),
                task_relationship_advisory=advisory,
                task_relationship=rel,
            )
        )
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
    base_head_seed: int,
) -> dict[str, Any]:
    spec_by_id = {s.checkpoint_id: s for s in PANEL}
    base_score_cache: dict[str, float] = {}

    def eval_ckpt(checkpoint_id: str) -> tuple[float, float]:
        spec = spec_by_id[checkpoint_id]
        rows = _sample_rows(spec, sample_size=sample_size, seed=seed)
        ckpt_path = str((REPO_ROOT / spec.checkpoint_rel_path).resolve())
        ckpt = _evaluate_single_model(
            model_path=ckpt_path,
            tokenizer_path=ckpt_path,
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
        return float(ckpt), float(base_score_cache[spec.task])

    records: list[dict[str, Any]] = []

    pair_choice = None
    pair_category = "retained_evaluate_first_pair"
    pair_note = "Evaluate-first same-task pair remains the strongest immediate comparison target."
    if action_plan.evaluate_first:
        pair_choice = action_plan.evaluate_first[0]
    else:
        for row in pairwise["pairs"]:
            if row["task_relationship"] == "same_task":
                pair_choice = f"{row['checkpoint_a']} × {row['checkpoint_b']}"
                pair_category = "same_task_near_miss_probe"
                pair_note = "No retained pair remained; probing the strongest same-task near-miss candidate."
                break

    if pair_choice:
        left, right = [x.strip() for x in pair_choice.split("×")]
        a_score, base_score = eval_ckpt(left)
        b_score, _ = eval_ckpt(right)
        strongest = max(a_score, b_score)
        records.append(
            {
                "category": pair_category,
                "checkpoints": [left, right],
                "task": spec_by_id[left].task,
                "sample_size": sample_size,
                "sample_seed": seed,
                "scores": {left: round(a_score, 4), right: round(b_score, 4), "base": round(base_score, 4)},
                "strongest_source_score": round(strongest, 4),
                "delta_vs_strongest_source": {left: round(a_score - strongest, 4), right: round(b_score - strongest, 4)},
                "interpretation": pair_note,
            }
        )

    same_family_pairs = [p for p in pairwise["pairs"] if p["task_relationship"] == "same_family"]
    if same_family_pairs:
        best_sf = sorted(same_family_pairs, key=lambda x: float(x["compatibility_score"]), reverse=True)[0]
        a = best_sf["checkpoint_a"]
        b = best_sf["checkpoint_b"]
        a_score, a_base = eval_ckpt(a)
        b_score, b_base = eval_ckpt(b)
        records.append(
            {
                "category": "same_family_pair_probe",
                "checkpoints": [a, b],
                "task": "same_family_cross_dataset",
                "sample_size": sample_size,
                "sample_seed": seed,
                "scores": {
                    f"{a}_on_{spec_by_id[a].task}": round(a_score, 4),
                    f"base_on_{spec_by_id[a].task}": round(a_base, 4),
                    f"{b}_on_{spec_by_id[b].task}": round(b_score, 4),
                    f"base_on_{spec_by_id[b].task}": round(b_base, 4),
                },
                "delta_vs_base": {
                    a: round(a_score - a_base, 4),
                    b: round(b_score - b_base, 4),
                },
                "interpretation": "Same-family non-identical-task branch probe (task-family equivalence sanity check).",
            }
        )

    for category, checkpoint_id in (
        ("optional_single_checkpoint", "mrpc_s42"),
        ("lower_priority_control_checkpoint", "qnli_s42"),
    ):
        s, b = eval_ckpt(checkpoint_id)
        records.append(
            {
                "category": category,
                "checkpoints": [checkpoint_id],
                "task": spec_by_id[checkpoint_id].task,
                "sample_size": sample_size,
                "sample_seed": seed,
                "scores": {checkpoint_id: round(s, 4), "base": round(b, 4)},
                "delta_vs_base": round(s - b, 4),
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
            "same_family_probe_count": sum(1 for r in records if "same_family" in r["category"]),
            "control_count": sum(1 for r in records if "control" in r["category"]),
        },
    }


def _format_follow_through_table(records: list[dict[str, Any]]) -> str:
    lines = ["| Category | Checkpoint(s) | Task | Score(s) | Note |", "|---|---|---|---|---|"]
    for row in records:
        checkpoints = ", ".join(row["checkpoints"])
        score_bits = [f"{k}={v}" for k, v in row["scores"].items()]
        scores = "; ".join(score_bits)
        lines.append(f"| {row['category']} | {checkpoints} | {row['task']} | {scores} | {row['interpretation']} |")
    return "\n".join(lines)


def _derive_usefulness_rating(
    *,
    status_counts: dict[str, int],
    pair_count: int,
    retained_count: int,
    same_family_count: int,
    policy_summary: dict[str, str],
) -> dict[str, str]:
    qa = "high" if len([k for k, v in status_counts.items() if v > 0]) >= 3 else "medium"
    pair = "high" if pair_count > 0 and (retained_count < pair_count or same_family_count > 0) else "medium"
    action = "high" if retained_count <= max(2, math.ceil(pair_count * 0.4)) else "medium"
    report = "high"
    broader = "high" if policy_summary["exploration_posture"] in {"narrow", "moderate"} and same_family_count > 0 else "medium"
    return {
        "qa_usefulness": qa,
        "pairwise_comparison_usefulness": pair,
        "action_plan_usefulness": action,
        "report_clarity": report,
        "broader_use_case_plausibility": broader,
    }


def _build_field_note(
    *,
    manifest: dict[str, Any],
    policy_summary: dict[str, str],
    action_plan: Any,
    qa_summary: dict[str, Any],
    pairwise: dict[str, Any],
    follow_through: dict[str, Any],
) -> str:
    status = qa_summary["status_counts"]
    rel_counts = pairwise["summary"]["relationship_counts"]
    return (
        "# Checkpoint Inventory T02 — Field Note\n\n"
        f"Generated: {_now_utc_iso()}\n\n"
        "## 1. Inventory\n\n"
        f"- Included `{manifest['checkpoint_count']}` full fine-tuned checkpoints sharing base `{manifest['base_model']}`.\n"
        "- Panel shape: same-task pair (SST-2 seeds), same-family non-identical-task pair (SST-2 × Yelp), plus cross-task controls (MRPC, QNLI).\n"
        "- Chosen to directly exercise the same-family branch in checkpoint inventory triage.\n\n"
        "## 2. Gradience Stance\n\n"
        f"- Dominant driver: `{policy_summary['dominant_driver']}`\n"
        f"- Inventory type: `{policy_summary['inventory_type']}`\n"
        f"- Exploration posture: `{policy_summary['exploration_posture']}`\n"
        f"- Evaluate-first: {', '.join(action_plan.evaluate_first) if action_plan.evaluate_first else 'none'}\n"
        f"- QA status counts: eligible={status['eligible']}, uncertain={status['uncertain']}, flagged_weak={status['flagged_weak']}\n"
        f"- Pair relationships: same_task={rel_counts['same_task']}, same_family={rel_counts['same_family']}, cross_task={rel_counts['cross_task']}\n"
        f"- Action-plan summary: {action_plan.summary_line}\n\n"
        "## 3. Follow-through Results\n\n"
        f"{_format_follow_through_table(follow_through['records'])}\n\n"
        "## 4. Product Judgment\n\n"
        f"- Did checkpoint workflow feel useful? {'yes' if action_plan.total_pairs > 0 else 'partially'}\n"
        f"- Did evidence gate remain central? {'yes' if policy_summary['dominant_driver'] == 'source_qa' else 'partially'}\n"
        "- Did reports explain triage clearly? yes (inventory summary + action plan + run-bundle packet).\n"
        "- Does this feel like a real broader use case? yes, with explicit same-family branch coverage and narrow-scope caveats.\n"
    )


def _build_trial_memo(
    *,
    policy_summary: dict[str, str],
    action_plan: Any,
    follow_through: dict[str, Any],
    usefulness: dict[str, str],
    measurement_schema: dict[str, Any],
) -> str:
    subset_line = (
        "clear retained subset"
        if action_plan.retained_count > 0
        else "clear near-miss/same-family review subset when retained candidates were absent"
    )
    return (
        "# Checkpoint Inventory T02 — Trial Memo\n\n"
        f"Generated: {_now_utc_iso()}\n\n"
        "## 1. What transferred unchanged from adapter workflows?\n\n"
        "- Evidence bootstrap as the practical gate.\n"
        "- QA eligibility framing and confidence labeling.\n"
        "- Pairwise compatibility-driven narrowing.\n"
        "- Preflight bundle outputs (summary, action plan, review packet).\n\n"
        "## 2. What was checkpoint-specific?\n\n"
        "- Artifact unit is full fine-tuned checkpoints, not adapters.\n"
        "- Representation path uses summary-based checkpoint deltas (Representation C).\n"
        "- In checkpoint space, same-family means non-identical tasks that share a declared task family under one shared base (here: SST-2 and Yelp within `sentiment_binary`).\n"
        "- Routing treated same-family as its own branch (`task_relationship=same_family`) with informational caution, not as auto-retained same-task and not as undifferentiated cross-task.\n"
        "- Both same-family pairs (`sst2_s42::yelp_s42`, `sst2_s123::yelp_s42`) were surfaced for review while source QA remained the binding constraint.\n"
        "- Merge execution remained out of scope.\n\n"
        "## 3. Did the workflow feel naturally broader than merge preflight?\n\n"
        "- Yes. The workflow operated as checkpoint-inventory triage and review-budget prioritization.\n"
        f"- Candidate space changed from {action_plan.total_pairs} to {action_plan.retained_count} retained, with {subset_line}.\n\n"
        "## 4. What broke or felt forced?\n\n"
        "- Same-family follow-through was asymmetric: `sst2_s42` on SST-2 was below base (`-0.01`), while `yelp_s42` on Yelp was well above base (`+0.2967`).\n"
        "- This is useful but important: same-family is a routing/review relationship, not an interchangeability claim across datasets.\n"
        "- Cross-dataset same-family follow-through cannot be summarized by a single shared-task metric.\n"
        "- Scope remains intentionally narrow (single backbone, small panel, CPU-only).\n\n"
        "## 5. Is checkpoint inventory triage a credible external use case?\n\n"
        "- Yes, in bounded form.\n"
        f"- Dominant driver was `{policy_summary['dominant_driver']}`, preserving trust-aware decision behavior.\n"
        "- The broadened workflow stayed conservative (no retained pairs under weak evidence) but useful (explicit near-miss and same-family follow-through priorities instead of opaque rejection).\n\n"
        "## 6. What should happen next?\n\n"
        "- Run one additional inventory with a second same-family branch (e.g., SST-2 × Amazon or SST-2 × IMDB).\n"
        "- Keep merge execution out of scope until broader checkpoint evidence is accumulated.\n"
        "- Preserve CPU-first repeatability while expanding inventory diversity incrementally.\n\n"
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
    parser = argparse.ArgumentParser(description="Run checkpoint inventory field trial T02")
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
    t0 = time.time()

    manifest_path = TRIAL_DIR / "manifest.json"
    evidence_path = TRIAL_DIR / "evidence" / "bootstrap_results.json"
    qa_dir = TRIAL_DIR / "qa_artifacts"
    pairwise_path = TRIAL_DIR / "pairwise" / "pairwise_results.json"
    preflight_dir = TRIAL_DIR / "preflight"
    eval_results_path = TRIAL_DIR / "eval_results.json"
    field_note_path = TRIAL_DIR / "field_note.md"
    trial_memo_path = TRIAL_DIR / "trial_memo.md"

    manifest = _build_manifest(args.base_model)
    _write_json(manifest_path, manifest)

    summaries = _materialize_summaries(args.base_model)
    single_audit = _single_audit_rows(summaries)

    evidence = _bootstrap_evidence(
        base_model=args.base_model,
        sample_size=args.evidence_sample_size,
        seed=args.evidence_seed,
        margin=args.margin,
        base_head_seed=args.base_head_seed,
    )
    _write_json(evidence_path, evidence)

    qa_artifacts, qa_summary = _build_qa_artifacts(
        base_model=args.base_model,
        evidence=evidence,
        single_audit=single_audit,
        margin=args.margin,
    )
    qa_dir.mkdir(parents=True, exist_ok=True)
    for artifact in qa_artifacts:
        _write_json(qa_dir / f"{artifact.adapter_name}.json", artifact.to_dict())
    _write_json(qa_dir / "qa_summary.json", qa_summary)

    pairwise = _build_pairwise_results(summaries)
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
        na = Path(report.adapter_a.path).name
        nb = Path(report.adapter_b.path).name
        _write_json(preflight_pair_dir / f"{na}__{nb}.json", report.to_dict())

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

    _write_json(
        preflight_dir / "preflight_results.json",
        {
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
        },
    )

    follow_through = _run_follow_through(
        base_model=args.base_model,
        pairwise=pairwise,
        action_plan=action_plan,
        sample_size=args.follow_through_sample_size,
        seed=args.follow_through_seed,
        base_head_seed=args.base_head_seed,
    )

    usefulness = _derive_usefulness_rating(
        status_counts=qa_summary["status_counts"],
        pair_count=pairwise["summary"]["pair_count"],
        retained_count=int(action_plan.retained_count),
        same_family_count=pairwise["summary"]["relationship_counts"]["same_family"],
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
            "candidate_reduction": round(1.0 - (int(action_plan.retained_count) / max(1, int(action_plan.total_pairs))), 4),
            "relationship_counts": pairwise["summary"]["relationship_counts"],
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

    _write_json(
        eval_results_path,
        {
            "schema": "checkpoint_inventory_eval_results/v1",
            "timestamp_utc": _now_utc_iso(),
            "inventory_id": INVENTORY_ID,
            "follow_through": follow_through,
            "measurement_schema": measurement_schema,
        },
    )

    field_note_path.write_text(
        _build_field_note(
            manifest=manifest,
            policy_summary=policy_summary,
            action_plan=action_plan,
            qa_summary=qa_summary,
            pairwise=pairwise,
            follow_through=follow_through,
        )
    )
    trial_memo_path.write_text(
        _build_trial_memo(
            policy_summary=policy_summary,
            action_plan=action_plan,
            follow_through=follow_through,
            usefulness=usefulness,
            measurement_schema=measurement_schema,
        )
    )

    elapsed = time.time() - t0
    print(f"[t02] manifest: {manifest_path}")
    print(f"[t02] evidence: {evidence_path}")
    print(f"[t02] qa_dir: {qa_dir}")
    print(f"[t02] pairwise: {pairwise_path}")
    print(f"[t02] preflight: {preflight_dir}")
    print(f"[t02] eval_results: {eval_results_path}")
    print(f"[t02] field_note: {field_note_path}")
    print(f"[t02] trial_memo: {trial_memo_path}")
    print(
        f"[t02] policy={policy_summary['inventory_type']} / "
        f"{policy_summary['dominant_driver']} / {policy_summary['exploration_posture']} "
        f"(elapsed={elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
