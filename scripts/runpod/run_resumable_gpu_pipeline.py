#!/usr/bin/env python3
"""RunPod A100 resumable experiment pipeline.

Phases (independently resumable):
1) Train adapters sequentially.
2) Run QA + pairwise merge audit + inventory preflight.
3) Merge/evaluate retained, near-miss, and stratified control pairs.

This script is intentionally additive and does not replace existing study scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TASK_CONFIG: dict[str, dict[str, Any]] = {
    "qnli": {
        "subset": "qnli",
        "field_a": "question",
        "field_b": "sentence",
        "num_labels": 2,
        "val_split": "validation",
    },
    "rte": {
        "subset": "rte",
        "field_a": "sentence1",
        "field_b": "sentence2",
        "num_labels": 2,
        "val_split": "validation",
    },
    "mrpc": {
        "subset": "mrpc",
        "field_a": "sentence1",
        "field_b": "sentence2",
        "num_labels": 2,
        "val_split": "validation",
    },
    "sst2": {
        "subset": "sst2",
        "field_a": "sentence",
        "field_b": None,
        "num_labels": 2,
        "val_split": "validation",
    },
    "mnli": {
        "subset": "mnli",
        "field_a": "premise",
        "field_b": "hypothesis",
        "num_labels": 3,
        "val_split": "validation_matched",
    },
}


CLASSIFIER_KEYS = (
    "base_model.model.classifier.weight",
    "base_model.model.classifier.bias",
    "base_model.model.pre_classifier.weight",
    "base_model.model.pre_classifier.bias",
)


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    task: str
    seed: int
    train_samples: int
    eval_samples: int
    max_steps: int
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    warmup_ratio: float
    weight_decay: float
    max_length: int
    target_modules: tuple[str, ...]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _config_hash(cfg: dict[str, Any]) -> str:
    encoded = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run(cmd: list[str], *, quiet: bool = False) -> None:
    if quiet:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, check=True)


def _load_pipeline_config(path: Path) -> dict[str, Any]:
    cfg = _load_json(path)
    required = ("run_name", "output_root", "base_model", "training_defaults", "adapters")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    if not cfg["adapters"]:
        raise ValueError("Config 'adapters' must not be empty.")
    return cfg


def _materialize_adapter_specs(cfg: dict[str, Any]) -> list[AdapterSpec]:
    defaults = cfg["training_defaults"]
    specs: list[AdapterSpec] = []
    for raw in cfg["adapters"]:
        task = raw["task"].lower()
        if task not in TASK_CONFIG:
            raise ValueError(f"Unsupported task '{task}'. Supported: {sorted(TASK_CONFIG)}")
        target_modules = raw.get("target_modules", defaults.get("target_modules", ["query_proj", "value_proj"]))
        specs.append(
            AdapterSpec(
                name=raw["name"],
                task=task,
                seed=int(raw["seed"]),
                train_samples=int(raw.get("train_samples", defaults.get("train_samples", 2000))),
                eval_samples=int(raw.get("eval_samples", defaults.get("eval_samples", 500))),
                max_steps=int(raw.get("max_steps", defaults.get("max_steps", 1000))),
                learning_rate=float(raw.get("learning_rate", defaults.get("learning_rate", 5e-5))),
                train_batch_size=int(raw.get("train_batch_size", defaults.get("train_batch_size", 8))),
                eval_batch_size=int(raw.get("eval_batch_size", defaults.get("eval_batch_size", 32))),
                lora_r=int(raw.get("lora_r", defaults.get("lora_r", 16))),
                lora_alpha=int(raw.get("lora_alpha", defaults.get("lora_alpha", 16))),
                lora_dropout=float(raw.get("lora_dropout", defaults.get("lora_dropout", 0.0))),
                warmup_ratio=float(raw.get("warmup_ratio", defaults.get("warmup_ratio", 0.1))),
                weight_decay=float(raw.get("weight_decay", defaults.get("weight_decay", 0.01))),
                max_length=int(raw.get("max_length", defaults.get("max_length", 256))),
                target_modules=tuple(str(x) for x in target_modules),
            )
        )
    return specs


def _state_path(output_root: Path) -> Path:
    return output_root / "pipeline_state.json"


def _default_state(cfg_hash: str, run_name: str) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "config_hash": cfg_hash,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "phases": {
            "train": {"status": "pending", "completed": 0, "total": 0},
            "preflight": {"status": "pending"},
            "merge_eval": {"status": "pending", "completed_pairs": 0},
        },
    }


def _load_or_init_state(output_root: Path, cfg_hash: str, run_name: str) -> dict[str, Any]:
    path = _state_path(output_root)
    if not path.exists():
        state = _default_state(cfg_hash, run_name)
        _write_json(path, state)
        return state
    state = _load_json(path)
    old = state.get("config_hash")
    if old and old != cfg_hash:
        raise RuntimeError(
            "Config hash mismatch with existing pipeline_state.json. "
            "Use a fresh output_root for a different cohort/config."
        )
    return state


def _save_state(output_root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now_iso()
    _write_json(_state_path(output_root), state)


def _adapter_dir(output_root: Path, name: str) -> Path:
    return output_root / "adapters" / name


def _adapter_manifest_path(output_root: Path, name: str) -> Path:
    return _adapter_dir(output_root, name) / "training_manifest.json"


def _train_single_adapter(spec: AdapterSpec, base_model: str, output_root: Path) -> dict[str, Any]:
    adapter_dir = _adapter_dir(output_root, spec.name)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _adapter_manifest_path(output_root, spec.name)
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if manifest.get("training_complete", False):
            print(f"[train] {spec.name}: already complete, skipping")
            return manifest

    task_cfg = TASK_CONFIG[spec.task]

    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(spec.seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    ds = load_dataset("glue", task_cfg["subset"])
    train_ds = ds["train"].select(range(min(spec.train_samples, len(ds["train"]))))
    val_full = ds[task_cfg["val_split"]]
    val_ds = val_full.select(range(min(spec.eval_samples, len(val_full))))

    def preprocess(examples: dict[str, Any]) -> dict[str, Any]:
        if task_cfg["field_b"] is None:
            out = tokenizer(
                examples[task_cfg["field_a"]],
                truncation=True,
                padding=True,
                max_length=spec.max_length,
            )
        else:
            out = tokenizer(
                examples[task_cfg["field_a"]],
                examples[task_cfg["field_b"]],
                truncation=True,
                padding=True,
                max_length=spec.max_length,
            )
        out["labels"] = examples["label"]
        return out

    train_tok = train_ds.map(preprocess, batched=True)
    val_tok = val_ds.map(preprocess, batched=True)
    collator = DataCollatorWithPadding(tokenizer)

    base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=task_cfg["num_labels"])
    base_trainer = Trainer(
        model=base,
        args=TrainingArguments(output_dir=str(adapter_dir / "_base_eval"), per_device_eval_batch_size=spec.eval_batch_size, report_to=[]),
        eval_dataset=val_tok,
        data_collator=collator,
    )
    base_preds = base_trainer.predict(val_tok)
    base_acc = float((np.argmax(base_preds.predictions, axis=1) == base_preds.label_ids).mean())

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=task_cfg["num_labels"])
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=list(spec.target_modules),
        modules_to_save=["classifier", "pooler", "pre_classifier"],
    )
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "checkpoints"),
        max_steps=spec.max_steps,
        per_device_train_batch_size=spec.train_batch_size,
        per_device_eval_batch_size=spec.eval_batch_size,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        warmup_ratio=spec.warmup_ratio,
        eval_strategy="steps",
        eval_steps=max(100, min(200, spec.max_steps)),
        save_strategy="no",
        logging_steps=max(50, min(100, spec.max_steps)),
        seed=spec.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
    )
    trainer.train()
    preds = trainer.predict(val_tok)
    adapter_acc = float((np.argmax(preds.predictions, axis=1) == preds.label_ids).mean())
    margin = adapter_acc - base_acc

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    manifest = {
        "schema": "gradience.runpod_training_manifest/v1",
        "name": spec.name,
        "task": spec.task,
        "seed": spec.seed,
        "base_model": base_model,
        "num_labels": task_cfg["num_labels"],
        "train_samples": len(train_tok),
        "eval_samples": len(val_tok),
        "max_steps": spec.max_steps,
        "learning_rate": spec.learning_rate,
        "lora_r": spec.lora_r,
        "lora_alpha": spec.lora_alpha,
        "lora_dropout": spec.lora_dropout,
        "target_modules": list(spec.target_modules),
        "base_accuracy": round(base_acc, 6),
        "adapter_accuracy": round(adapter_acc, 6),
        "margin": round(margin, 6),
        "verified": margin > 0.0,
        "training_complete": True,
        "completed_at": _now_iso(),
        "adapter_dir": str(adapter_dir),
    }
    _write_json(manifest_path, manifest)
    print(
        f"[train] {spec.name}: done "
        f"(adapter={adapter_acc:.4f}, base={base_acc:.4f}, margin={margin:+.4f})"
    )
    return manifest


def _collect_training_manifests(output_root: Path, specs: list[AdapterSpec]) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for spec in specs:
        path = _adapter_manifest_path(output_root, spec.name)
        if not path.exists():
            raise RuntimeError(f"Missing training manifest for {spec.name}: {path}")
        manifest = _load_json(path)
        if not manifest.get("training_complete", False):
            raise RuntimeError(f"Incomplete training manifest for {spec.name}: {path}")
        manifests[spec.name] = manifest
    return manifests


def _phase_train(cfg: dict[str, Any], output_root: Path, state: dict[str, Any], specs: list[AdapterSpec]) -> None:
    base_model = cfg["base_model"]
    phase = state["phases"]["train"]
    phase["status"] = "running"
    phase["total"] = len(specs)
    _save_state(output_root, state)

    complete = 0
    for spec in specs:
        _train_single_adapter(spec, base_model, output_root)
        complete += 1
        phase["completed"] = complete
        _save_state(output_root, state)

    summary = {
        "schema": "gradience.runpod_training_summary/v1",
        "run_name": cfg["run_name"],
        "base_model": base_model,
        "adapter_count": len(specs),
        "verified_count": sum(
            1
            for spec in specs
            if _load_json(_adapter_manifest_path(output_root, spec.name)).get("verified", False)
        ),
        "completed_at": _now_iso(),
        "adapters": [_load_json(_adapter_manifest_path(output_root, spec.name)) for spec in specs],
    }
    _write_json(output_root / "training_summary.json", summary)
    phase["status"] = "done"
    _save_state(output_root, state)


def _phase_preflight(cfg: dict[str, Any], output_root: Path, state: dict[str, Any], specs: list[AdapterSpec]) -> None:
    phase_cfg = cfg.get("preflight", {})
    phase = state["phases"]["preflight"]
    phase["status"] = "running"
    _save_state(output_root, state)

    manifests = _collect_training_manifests(output_root, specs)
    preflight_dir = output_root / "preflight"
    qa_dir = preflight_dir / "qa"
    report_dir = preflight_dir / "pair_reports"
    inventory_dir = preflight_dir / "inventory"
    neighborhoods_dir = preflight_dir / "neighborhoods"
    for d in (qa_dir, report_dir, inventory_dir, neighborhoods_dir):
        d.mkdir(parents=True, exist_ok=True)

    qa_margin = float(phase_cfg.get("qa_margin", 0.0))
    strict_qa = bool(phase_cfg.get("strict_qa", False))
    quiet_subprocess = bool(phase_cfg.get("quiet_subprocess", False))

    # Step 2a: QA artifacts
    for spec in specs:
        manifest = manifests[spec.name]
        qa_path = qa_dir / f"{spec.name}_qa.json"
        if qa_path.exists():
            continue
        cmd = [
            "python3",
            "-m",
            "gradience",
            "audit-adapter",
            "--peft-dir",
            manifest["adapter_dir"],
            "--eval-dataset",
            spec.task,
            "--metric-name",
            "accuracy",
            "--adapter-score",
            str(manifest["adapter_accuracy"]),
            "--base-score",
            str(manifest["base_accuracy"]),
            "--higher-is-better",
            "--margin",
            str(qa_margin),
            "--out",
            str(qa_path),
        ]
        _run(cmd, quiet=quiet_subprocess)
        print(f"[preflight] QA: {spec.name}")

    # Step 2b: pairwise merge reports
    for a, b in itertools.combinations(specs, 2):
        out_json = report_dir / f"{a.name}_vs_{b.name}.json"
        if out_json.exists():
            continue
        cmd = [
            "python3",
            "-m",
            "gradience",
            "merge-audit",
            "--adapter-a",
            str(_adapter_dir(output_root, a.name)),
            "--adapter-b",
            str(_adapter_dir(output_root, b.name)),
            "--source-a-qa",
            str(qa_dir / f"{a.name}_qa.json"),
            "--source-b-qa",
            str(qa_dir / f"{b.name}_qa.json"),
            "--qa-report",
            "--emit-report",
            str(out_json),
        ]
        if strict_qa:
            cmd.append("--strict-qa")
        _run(cmd, quiet=quiet_subprocess)
        print(f"[preflight] pair: {a.name} × {b.name}")

    inv_json = inventory_dir / "inventory_summary.json"
    if not inv_json.exists():
        _run(
            [
                "python3",
                "-m",
                "gradience",
                "summarize-inventory",
                "--qa-dir",
                str(qa_dir),
                "--report-dir",
                str(report_dir),
                "--emit-report",
                str(inv_json),
            ],
            quiet=quiet_subprocess,
        )
        print("[preflight] inventory summary")

    neighborhoods_json = neighborhoods_dir / "neighborhoods.json"
    if not neighborhoods_json.exists():
        _run(
            [
                "python3",
                "-m",
                "gradience",
                "suggest-neighborhoods",
                "--qa-dir",
                str(qa_dir),
                "--report-dir",
                str(report_dir),
                "--emit-report",
                str(neighborhoods_json),
            ],
            quiet=quiet_subprocess,
        )
        print("[preflight] neighborhoods")

    # Step 2c: durable action-plan JSON for phase 3
    from gradience.api import _load_schema_artifacts
    from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
    from gradience.vnext.inventory.summary import build_action_plan, format_action_plan
    from gradience.vnext.merge.qa_report import MergeQAReport

    qa_artifacts = _load_schema_artifacts(
        files=sorted(qa_dir.glob("*.json")),
        schema_prefix="gradience.adapter_qa/",
        from_dict=AdapterQAArtifact.from_dict,
        strict_input=False,
        malformed_label="QA artifact",
    )
    merge_reports = _load_schema_artifacts(
        files=sorted(report_dir.glob("*.json")),
        schema_prefix="gradience.merge_qa_report/",
        from_dict=MergeQAReport.from_dict,
        strict_input=False,
        malformed_label="merge report",
    )
    action_plan = build_action_plan(qa_artifacts, merge_reports)

    ap_json = {
        "schema": "gradience.inventory_action_plan/v1",
        "generated_at": _now_iso(),
        "exclude": list(action_plan.exclude),
        "same_task_priority": list(action_plan.same_task_priority),
        "cross_task_caution": list(action_plan.cross_task_caution),
        "evaluate_first": list(action_plan.evaluate_first),
        "summary_line": action_plan.summary_line,
        "total_pairs": action_plan.total_pairs,
        "retained_count": action_plan.retained_count,
        "cross_task_count": action_plan.cross_task_count,
        "behavioral_evidence_count": action_plan.behavioral_evidence_count,
        "total_source_count": action_plan.total_source_count,
        "retained_pair_detail": [list(x) for x in action_plan.retained_pair_detail],
        "near_miss_candidates": list(action_plan.near_miss_candidates),
        "near_miss_detail": [list(x) for x in action_plan.near_miss_detail],
        "near_miss_severity": [list(x) for x in action_plan.near_miss_severity],
    }
    _write_json(preflight_dir / "inventory_action_plan.json", ap_json)
    (preflight_dir / "inventory_action_plan.txt").write_text(format_action_plan(action_plan))

    phase["status"] = "done"
    _save_state(output_root, state)


def _patch_classifier_into_adapter(
    merged_adapter_dir: Path,
    source_adapter_dir: Path,
    target_num_labels: int,
    tmp_out: Path,
) -> Path:
    from safetensors.torch import load_file, save_file

    tmp_adapter_dir = tmp_out
    if tmp_adapter_dir.exists():
        shutil.rmtree(tmp_adapter_dir)
    tmp_adapter_dir.mkdir(parents=True, exist_ok=True)

    merged_st = load_file(str(merged_adapter_dir / "adapter_model.safetensors"))
    source_st = load_file(str(source_adapter_dir / "adapter_model.safetensors"))

    patched = dict(merged_st)
    for key in CLASSIFIER_KEYS:
        if key in source_st:
            patched[key] = source_st[key]

    save_file(patched, str(tmp_adapter_dir / "adapter_model.safetensors"))

    cfg = _load_json(merged_adapter_dir / "adapter_config.json")
    cfg["num_labels"] = int(target_num_labels)
    _write_json(tmp_adapter_dir / "adapter_config.json", cfg)
    return tmp_adapter_dir


def _evaluate_adapter_on_task(
    *,
    adapter_dir: Path,
    base_model: str,
    task: str,
    eval_samples: int,
    source_for_classifier: Path | None = None,
) -> float:
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    task_cfg = TASK_CONFIG[task]
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    with tempfile.TemporaryDirectory() as td:
        adapter_to_load = adapter_dir
        if source_for_classifier is not None:
            adapter_to_load = _patch_classifier_into_adapter(
                merged_adapter_dir=adapter_dir,
                source_adapter_dir=source_for_classifier,
                target_num_labels=int(task_cfg["num_labels"]),
                tmp_out=Path(td) / "adapter_patched",
            )

        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=int(task_cfg["num_labels"]))
        model = PeftModel.from_pretrained(model, str(adapter_to_load))
        model = model.merge_and_unload()
        model.eval()

        ds = load_dataset("glue", task_cfg["subset"], split=task_cfg["val_split"])
        ds = ds.select(range(min(eval_samples, len(ds))))

        def preprocess(examples: dict[str, Any]) -> dict[str, Any]:
            if task_cfg["field_b"] is None:
                out = tokenizer(
                    examples[task_cfg["field_a"]],
                    truncation=True,
                    padding=True,
                    max_length=256,
                )
            else:
                out = tokenizer(
                    examples[task_cfg["field_a"]],
                    examples[task_cfg["field_b"]],
                    truncation=True,
                    padding=True,
                    max_length=256,
                )
            out["labels"] = examples["label"]
            return out

        tok_ds = ds.map(preprocess, batched=True)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=str(Path(td) / "eval"), per_device_eval_batch_size=32, report_to=[]),
            eval_dataset=tok_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        preds = trainer.predict(tok_ds)
        acc = float((np.argmax(preds.predictions, axis=1) == preds.label_ids).mean())
        return acc


def _parse_pair_label(pair_label: str) -> tuple[str, str]:
    parts = [p.strip() for p in pair_label.split("×")]
    if len(parts) != 2:
        raise ValueError(f"Cannot parse pair label '{pair_label}'. Expected 'A × B'.")
    return parts[0], parts[1]


def _sanitize_pair_id(label: str) -> str:
    return (
        label.replace(" × ", "__x__")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _normalize_plan_strategy(strategy: str | None) -> str:
    if not strategy:
        return "audit_aware"
    raw = str(strategy).strip().lower()
    mapping = {
        "uniform_linear": "uniform_linear",
        "linear": "uniform_linear",
        "audit_aware": "audit_aware",
        "norm_equalized": "norm_equalized",
        "overlap_ties": "overlap_ties",
        "ties": "overlap_ties",
        "dare_linear": "dare_linear",
        "dare_ties": "dare_ties",
    }
    return mapping.get(raw, "audit_aware")


def _select_stratified_controls(
    cross_task_labels: list[str],
    pair_risk_map: dict[str, str],
    sample_size: int,
    seed: int,
) -> list[str]:
    if sample_size <= 0 or not cross_task_labels:
        return []
    rng = random.Random(seed)
    by_risk: dict[str, list[str]] = {"high": [], "medium": [], "low": [], "unknown": []}
    for label in cross_task_labels:
        risk = pair_risk_map.get(label, "unknown")
        if risk not in by_risk:
            risk = "unknown"
        by_risk[risk].append(label)
    for v in by_risk.values():
        rng.shuffle(v)

    active = [k for k, v in by_risk.items() if v]
    if not active:
        return []

    per_bucket = max(1, sample_size // len(active))
    picked: list[str] = []

    for bucket in active:
        take = min(per_bucket, len(by_risk[bucket]))
        picked.extend(by_risk[bucket][:take])
        by_risk[bucket] = by_risk[bucket][take:]

    if len(picked) < sample_size:
        leftovers: list[str] = []
        for bucket in ("high", "medium", "low", "unknown"):
            leftovers.extend(by_risk[bucket])
        rng.shuffle(leftovers)
        picked.extend(leftovers[: max(0, sample_size - len(picked))])

    return picked[:sample_size]


def _phase_merge_eval(cfg: dict[str, Any], output_root: Path, state: dict[str, Any], specs: list[AdapterSpec]) -> None:
    phase_cfg = cfg.get("merge_eval", {})
    phase = state["phases"]["merge_eval"]
    phase["status"] = "running"
    _save_state(output_root, state)

    manifests = _collect_training_manifests(output_root, specs)
    by_name = {m["name"]: m for m in manifests.values()}

    preflight_dir = output_root / "preflight"
    action_plan = _load_json(preflight_dir / "inventory_action_plan.json")
    pair_reports_dir = preflight_dir / "pair_reports"
    merge_eval_dir = output_root / "merge_eval"
    merge_eval_dir.mkdir(parents=True, exist_ok=True)
    eval_records_dir = merge_eval_dir / "pair_results"
    eval_records_dir.mkdir(parents=True, exist_ok=True)

    # Build pair metadata maps from merge reports.
    pair_risk_map: dict[str, str] = {}
    pair_strategy_map: dict[str, str] = {}
    cross_task_labels: list[str] = []
    for p in sorted(pair_reports_dir.glob("*.json")):
        report = _load_json(p)
        a_path = report.get("adapter_a", {}).get("path", "")
        b_path = report.get("adapter_b", {}).get("path", "")
        a_name = Path(a_path).name if a_path else "?"
        b_name = Path(b_path).name if b_path else "?"
        label = f"{a_name} × {b_name}"
        pair_risk_map[label] = report.get("pair_risk", "unknown")
        pair_strategy_map[label] = report.get("recommended_strategy", "uniform_linear")
        advisory = report.get("task_relationship_advisory")
        task_rel = report.get("task_relationship")
        if advisory is not None and task_rel == "cross_task":
            cross_task_labels.append(label)

    retained = list(action_plan.get("evaluate_first", []))[: int(phase_cfg.get("retained_limit", 8))]
    near_miss = list(action_plan.get("near_miss_candidates", []))[: int(phase_cfg.get("near_miss_limit", 8))]
    selected_set = set(retained) | set(near_miss)
    controls = _select_stratified_controls(
        cross_task_labels=[x for x in cross_task_labels if x not in selected_set],
        pair_risk_map=pair_risk_map,
        sample_size=int(phase_cfg.get("control_sample_size", 8)),
        seed=int(phase_cfg.get("control_seed", 13)),
    )

    retained_detail = {row[0]: row for row in action_plan.get("retained_pair_detail", [])}
    near_miss_detail = {row[0]: row for row in action_plan.get("near_miss_detail", [])}

    queue: list[dict[str, Any]] = []
    for label in retained:
        row = retained_detail.get(label)
        strategy = _normalize_plan_strategy(row[2] if row and len(row) >= 3 else pair_strategy_map.get(label))
        queue.append({"label": label, "category": "retained", "strategy": strategy})
    for label in near_miss:
        row = near_miss_detail.get(label)
        strategy = _normalize_plan_strategy(row[2] if row and len(row) >= 3 else pair_strategy_map.get(label))
        queue.append({"label": label, "category": "near_miss", "strategy": strategy})
    for label in controls:
        queue.append(
            {
                "label": label,
                "category": "control",
                "strategy": _normalize_plan_strategy(pair_strategy_map.get(label, "audit_aware")),
            }
        )

    eval_manifest = {
        "schema": "gradience.runpod_merge_eval_manifest/v1",
        "generated_at": _now_iso(),
        "retained": retained,
        "near_miss": near_miss,
        "control": controls,
        "total_selected": len(queue),
    }
    _write_json(merge_eval_dir / "selection_manifest.json", eval_manifest)

    from gradience.vnext.merge import execute_merge, merge_audit, plan_from_audit

    max_eval_samples = int(phase_cfg.get("max_eval_samples", 500))
    completed = 0
    all_results: list[dict[str, Any]] = []

    for item in queue:
        label = item["label"]
        pair_id = _sanitize_pair_id(label)
        result_path = eval_records_dir / f"{pair_id}.json"
        if result_path.exists():
            all_results.append(_load_json(result_path))
            completed += 1
            phase["completed_pairs"] = completed
            _save_state(output_root, state)
            continue

        a_name, b_name = _parse_pair_label(label)
        if a_name not in by_name or b_name not in by_name:
            raise RuntimeError(f"Pair references unknown adapters: {label}")
        a = by_name[a_name]
        b = by_name[b_name]
        strategy = item["strategy"]

        pair_out = merge_eval_dir / "merged_adapters" / pair_id
        pair_out.mkdir(parents=True, exist_ok=True)
        merged_dir = pair_out / "merged_adapter"

        # Compute fresh report for this selected pair and execute merge.
        report = merge_audit(a["adapter_dir"], b["adapter_dir"])
        plan = plan_from_audit(strategy, report, a["adapter_dir"], b["adapter_dir"])
        merge_result = execute_merge(plan, str(merged_dir))

        task_a = a["task"]
        task_b = b["task"]
        same_task = task_a == task_b

        # Evaluate with source-specific classifier heads.
        merged_a = _evaluate_adapter_on_task(
            adapter_dir=merged_dir,
            base_model=cfg["base_model"],
            task=task_a,
            eval_samples=max_eval_samples,
            source_for_classifier=Path(a["adapter_dir"]),
        )
        if same_task:
            merged_b = merged_a
        else:
            merged_b = _evaluate_adapter_on_task(
                adapter_dir=merged_dir,
                base_model=cfg["base_model"],
                task=task_b,
                eval_samples=max_eval_samples,
                source_for_classifier=Path(b["adapter_dir"]),
            )

        delta_a = merged_a - float(a["adapter_accuracy"])
        delta_b = merged_b - float(b["adapter_accuracy"])
        delta_vs_best_source = min(delta_a, delta_b) if not same_task else delta_a
        recon = getattr(merge_result, "mean_reconstruction_error", None)
        recon_val = float(recon) if recon is not None else None

        row = {
            "schema": "gradience.runpod_merge_eval_result/v1",
            "pair_label": label,
            "pair_id": pair_id,
            "category": item["category"],
            "strategy": strategy,
            "pair_risk": pair_risk_map.get(label, "unknown"),
            "adapter_a": a_name,
            "adapter_b": b_name,
            "task_a": task_a,
            "task_b": task_b,
            "same_task": same_task,
            "source_a_score": float(a["adapter_accuracy"]),
            "source_b_score": float(b["adapter_accuracy"]),
            "merged_on_task_a": round(float(merged_a), 6),
            "merged_on_task_b": round(float(merged_b), 6),
            "delta_vs_source_a": round(float(delta_a), 6),
            "delta_vs_source_b": round(float(delta_b), 6),
            "delta_vs_best_source": round(float(delta_vs_best_source), 6),
            "mean_reconstruction_error": recon_val,
            "completed_at": _now_iso(),
        }
        _write_json(result_path, row)
        all_results.append(row)
        completed += 1
        phase["completed_pairs"] = completed
        _save_state(output_root, state)
        print(
            f"[merge_eval] {item['category']} {label} "
            f"Δbest={delta_vs_best_source:+.4f}"
        )

    # Aggregate summary
    categories = ("retained", "near_miss", "control")
    summary_rows: dict[str, Any] = {}
    for cat in categories:
        rows = [r for r in all_results if r.get("category") == cat]
        if not rows:
            continue
        deltas = [float(r["delta_vs_best_source"]) for r in rows]
        summary_rows[cat] = {
            "count": len(rows),
            "avg_delta_vs_best_source": round(float(np.mean(deltas)), 6),
            "median_delta_vs_best_source": round(float(np.median(deltas)), 6),
            "min_delta_vs_best_source": round(float(np.min(deltas)), 6),
            "max_delta_vs_best_source": round(float(np.max(deltas)), 6),
        }

    summary = {
        "schema": "gradience.runpod_merge_eval_summary/v1",
        "generated_at": _now_iso(),
        "selection_manifest": eval_manifest,
        "category_summary": summary_rows,
        "results": all_results,
    }
    _write_json(merge_eval_dir / "merge_eval_summary.json", summary)

    phase["status"] = "done"
    _save_state(output_root, state)


def _phase_order(phase: str) -> list[str]:
    if phase == "all":
        return ["train", "preflight", "merge_eval"]
    return [phase]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable RunPod GPU pipeline (train → preflight → merge_eval)")
    parser.add_argument("--config", required=True, help="Path to pipeline JSON config")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "train", "preflight", "merge_eval"],
        help="Phase to run (default: all)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override config output_root",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_pipeline_config(cfg_path)
    output_root = Path(args.output_root or cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    specs = _materialize_adapter_specs(cfg)

    cfg_hash = _config_hash(cfg)
    state = _load_or_init_state(output_root, cfg_hash, cfg["run_name"])

    meta = {
        "schema": "gradience.runpod_pipeline_meta/v1",
        "run_name": cfg["run_name"],
        "base_model": cfg["base_model"],
        "config_path": str(cfg_path),
        "config_hash": cfg_hash,
        "output_root": str(output_root),
        "adapter_count": len(specs),
        "generated_at": _now_iso(),
    }
    _write_json(output_root / "pipeline_meta.json", meta)

    for phase in _phase_order(args.phase):
        if phase == "train":
            _phase_train(cfg, output_root, state, specs)
        elif phase == "preflight":
            _phase_preflight(cfg, output_root, state, specs)
        elif phase == "merge_eval":
            _phase_merge_eval(cfg, output_root, state, specs)
        else:
            raise RuntimeError(f"Unsupported phase: {phase}")

    print(f"\nPipeline complete for phase='{args.phase}'.")
    print(f"Output root: {output_root}")
    print(f"State: {_state_path(output_root)}")


if __name__ == "__main__":
    main()
