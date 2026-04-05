#!/usr/bin/env python3
"""Shared helpers for decoder triage study scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CohortItem:
    adapter_id: str
    base_model: str
    task_type: str  # SEQ_CLS | CAUSAL_LM
    task_family: str  # classification | instruction
    dataset: str
    dataset_subset: str | None
    split: str
    lora_rank: int
    lora_alpha: int
    seed: int
    max_steps: int
    lr: float
    target_modules: tuple[str, ...]
    train_samples: int
    eval_samples: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    warmup_ratio: float
    max_seq_length: int
    metric_name: str
    higher_is_better: bool

    @property
    def task_key(self) -> str:
        subset = self.dataset_subset or "default"
        return f"{self.task_type}:{self.dataset}:{subset}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def parse_cohort_item(raw: dict[str, Any]) -> CohortItem:
    return CohortItem(
        adapter_id=str(raw["adapter_id"]),
        base_model=str(raw["base_model"]),
        task_type=str(raw["task_type"]).upper(),
        task_family=str(raw["task_family"]).lower(),
        dataset=str(raw["dataset"]),
        dataset_subset=(str(raw["dataset_subset"]) if raw.get("dataset_subset") is not None else None),
        split=str(raw.get("split", "train")),
        lora_rank=int(raw["lora_rank"]),
        lora_alpha=int(raw["lora_alpha"]),
        seed=int(raw["seed"]),
        max_steps=int(raw["max_steps"]),
        lr=float(raw["lr"]),
        target_modules=tuple(str(x) for x in raw["target_modules"]),
        train_samples=int(raw["train_samples"]),
        eval_samples=int(raw["eval_samples"]),
        per_device_train_batch_size=int(raw["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(raw["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(raw["gradient_accumulation_steps"]),
        weight_decay=float(raw["weight_decay"]),
        warmup_ratio=float(raw["warmup_ratio"]),
        max_seq_length=int(raw.get("max_seq_length", 512)),
        metric_name=str(raw["metric_name"]),
        higher_is_better=bool(raw["higher_is_better"]),
    )


def load_cohort_manifest(path: Path) -> list[CohortItem]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Cohort manifest must be a list: {path}")
    items = [parse_cohort_item(x) for x in raw]
    ids = [x.adapter_id for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate adapter_id entries in cohort manifest.")
    return items


def adapter_dir(adapter_root: Path, adapter_id: str) -> Path:
    return adapter_root / adapter_id


def training_manifest_path(adapter_root: Path, adapter_id: str) -> Path:
    return adapter_dir(adapter_root, adapter_id) / "training_manifest.json"


def normalize_plan_strategy(strategy: str | None) -> str:
    if not strategy:
        return "audit_aware"
    raw = strategy.strip().lower()
    mapping = {
        "linear": "uniform_linear",
        "uniform_linear": "uniform_linear",
        "norm_equalized": "norm_equalized",
        "audit_aware": "audit_aware",
        "ties": "overlap_ties",
        "overlap_ties": "overlap_ties",
        "dare_linear": "dare_linear",
        "dare_ties": "dare_ties",
    }
    return mapping.get(raw, "audit_aware")


def pair_label(adapter_a: str, adapter_b: str) -> str:
    return f"{adapter_a} × {adapter_b}"


def parse_pair_label(label: str) -> tuple[str, str]:
    parts = [p.strip() for p in label.split("×")]
    if len(parts) != 2:
        raise ValueError(f"Bad pair label (expected 'A × B'): {label}")
    return parts[0], parts[1]


def sanitize_pair_id(label: str) -> str:
    return (
        label.replace(" × ", "__x__")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def delta_directional(*, merged_score: float, source_score: float, higher_is_better: bool) -> float:
    return (merged_score - source_score) if higher_is_better else (source_score - merged_score)

