#!/usr/bin/env python3
"""Train the controlled decoder triage cohort (resumable, manifest-driven).

Supports mixed task types:
- SEQ_CLS (GLUE-style classification adapters)
- CAUSAL_LM (instruction adapters scored by eval_loss)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import (
    CohortItem,
    adapter_dir,
    load_cohort_manifest,
    load_json,
    training_manifest_path,
    write_json,
)


GLUE_FIELD_MAP: dict[str, tuple[str, str | None]] = {
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
}


def _timestamp() -> str:
    import datetime as _dt

    return _dt.datetime.utcnow().isoformat() + "Z"


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_exp(x: float, cap: float = 20.0) -> float:
    return float(math.exp(min(max(x, -cap), cap)))


def _format_instruction_example(example: dict[str, Any], dataset_name: str) -> str:
    # Alpaca-style
    if "instruction" in example and ("output" in example or "response" in example):
        instruction = str(example.get("instruction", "")).strip()
        inp = str(example.get("input", "")).strip()
        out = str(example.get("output", example.get("response", ""))).strip()
        if inp:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        return f"### Instruction:\n{instruction}\n\n### Response:\n{out}"

    # Message-list style
    if "messages" in example and isinstance(example["messages"], list):
        parts: list[str] = []
        for msg in example["messages"]:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user")).strip()
                content = str(msg.get("content", "")).strip()
                if content:
                    parts.append(f"{role}: {content}")
            elif isinstance(msg, str) and msg.strip():
                parts.append(msg.strip())
        if parts:
            return "\n".join(parts)

    # Common text fields
    for key in ("text", "prompt", "content"):
        if key in example and str(example[key]).strip():
            if "response" in example and str(example["response"]).strip():
                return f"{example[key]}\n{example['response']}"
            if "output" in example and str(example["output"]).strip():
                return f"{example[key]}\n{example['output']}"
            return str(example[key])

    # Fallback: concatenate string-like values
    parts = [str(v).strip() for v in example.values() if isinstance(v, str) and str(v).strip()]
    if parts:
        return "\n".join(parts[:4])

    # Last resort
    return f"[unhandled_example::{dataset_name}]"


def _find_midpoint_checkpoint(checkpoints_dir: Path, midpoint_step: int) -> str | None:
    if not checkpoints_dir.exists():
        return None
    checkpoints = sorted(checkpoints_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return 0

    chosen = min(checkpoints, key=lambda p: abs(_step(p) - midpoint_step))
    return str(chosen)


def _train_seq_cls(item: CohortItem, out_dir: Path, log_dir: Path) -> dict[str, Any]:
    import numpy as np
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

    subset = item.dataset_subset or ""
    if subset not in GLUE_FIELD_MAP:
        raise ValueError(f"Unsupported SEQ_CLS dataset subset for GLUE mapping: {subset}")
    field_a, field_b = GLUE_FIELD_MAP[subset]

    set_seed(item.seed)

    tok = AutoTokenizer.from_pretrained(item.base_model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(item.dataset, subset)
    train_split = item.split if item.split in ds else "train"
    val_split = "validation_matched" if subset == "mnli" else "validation"
    if val_split not in ds:
        val_split = train_split

    train_ds = ds[train_split].shuffle(seed=item.seed).select(range(min(item.train_samples, len(ds[train_split]))))
    val_ds = ds[val_split].shuffle(seed=item.seed).select(range(min(item.eval_samples, len(ds[val_split]))))

    def preprocess(examples: dict[str, Any]) -> dict[str, Any]:
        if field_b is None:
            out = tok(examples[field_a], truncation=True, padding=True, max_length=item.max_seq_length)
        else:
            out = tok(examples[field_a], examples[field_b], truncation=True, padding=True, max_length=item.max_seq_length)
        out["labels"] = examples["label"]
        return out

    train_tok = train_ds.map(preprocess, batched=True)
    val_tok = val_ds.map(preprocess, batched=True)
    collator = DataCollatorWithPadding(tok)

    num_labels = len(set(val_ds["label"])) if "label" in val_ds.column_names else 2

    # Base model baseline
    base = AutoModelForSequenceClassification.from_pretrained(item.base_model, num_labels=num_labels)
    base_trainer = Trainer(
        model=base,
        args=TrainingArguments(
            output_dir=str(log_dir / item.adapter_id / "base_eval"),
            per_device_eval_batch_size=item.per_device_eval_batch_size,
            report_to=[],
        ),
        eval_dataset=val_tok,
        data_collator=collator,
    )
    base_preds = base_trainer.predict(val_tok)
    base_acc = float((np.argmax(base_preds.predictions, axis=1) == base_preds.label_ids).mean())

    model = AutoModelForSequenceClassification.from_pretrained(item.base_model, num_labels=num_labels)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=item.lora_rank,
        lora_alpha=item.lora_alpha,
        lora_dropout=0.05,
        target_modules=list(item.target_modules),
        bias="none",
        modules_to_save=["score", "classifier", "pre_classifier"],
    )
    model = get_peft_model(model, lora)

    ckpt_dir = out_dir / "checkpoints"
    save_steps = max(50, item.max_steps // 2)
    train_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        max_steps=item.max_steps,
        learning_rate=item.lr,
        per_device_train_batch_size=item.per_device_train_batch_size,
        per_device_eval_batch_size=item.per_device_eval_batch_size,
        gradient_accumulation_steps=item.gradient_accumulation_steps,
        warmup_ratio=item.warmup_ratio,
        weight_decay=item.weight_decay,
        eval_strategy="steps",
        eval_steps=max(100, min(200, item.max_steps)),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        logging_steps=max(25, min(100, item.max_steps)),
        seed=item.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
    )
    trainer.train()
    preds = trainer.predict(val_tok)
    adapter_acc = float((np.argmax(preds.predictions, axis=1) == preds.label_ids).mean())

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    midpoint = _find_midpoint_checkpoint(ckpt_dir, midpoint_step=item.max_steps // 2)
    return {
        "metric_name": "accuracy",
        "higher_is_better": True,
        "base_score": round(base_acc, 6),
        "adapter_score": round(adapter_acc, 6),
        "directional_delta": round(adapter_acc - base_acc, 6),
        "midpoint_checkpoint": midpoint,
        "num_labels": int(num_labels),
    }


def _train_causal_lm(item: CohortItem, out_dir: Path, log_dir: Path) -> dict[str, Any]:
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(item.seed)

    tok = AutoTokenizer.from_pretrained(item.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if item.dataset_subset:
        ds = load_dataset(item.dataset, item.dataset_subset)
    else:
        ds = load_dataset(item.dataset)

    split_name = item.split if item.split in ds else "train"
    raw_train = ds[split_name].shuffle(seed=item.seed)
    raw_train = raw_train.select(range(min(item.train_samples, len(raw_train))))

    # Prefer a native validation split when available.
    if "validation" in ds:
        raw_eval = ds["validation"].shuffle(seed=item.seed).select(range(min(item.eval_samples, len(ds["validation"]))))
    else:
        frac = max(0.05, min(0.2, item.eval_samples / max(1, len(raw_train))))
        split = raw_train.train_test_split(test_size=frac, seed=item.seed)
        raw_train = split["train"]
        raw_eval = split["test"].select(range(min(item.eval_samples, len(split["test"]))))

    def tokenize_fn(example: dict[str, Any]) -> dict[str, Any]:
        text = _format_instruction_example(example, dataset_name=item.dataset)
        enc = tok(text, truncation=True, max_length=item.max_seq_length, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_tok = raw_train.map(tokenize_fn, remove_columns=raw_train.column_names)
    eval_tok = raw_eval.map(tokenize_fn, remove_columns=raw_eval.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Base model baseline loss
    base = AutoModelForCausalLM.from_pretrained(item.base_model, torch_dtype="auto", device_map="auto")
    base_eval = Trainer(
        model=base,
        args=TrainingArguments(
            output_dir=str(log_dir / item.adapter_id / "base_eval"),
            per_device_eval_batch_size=item.per_device_eval_batch_size,
            report_to=[],
        ),
        eval_dataset=eval_tok,
        data_collator=collator,
    )
    base_metrics = base_eval.evaluate()
    base_loss = _as_float(base_metrics.get("eval_loss"), default=999.0)

    model = AutoModelForCausalLM.from_pretrained(item.base_model, torch_dtype="auto", device_map="auto")
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=item.lora_rank,
        lora_alpha=item.lora_alpha,
        lora_dropout=0.05,
        target_modules=list(item.target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora)

    ckpt_dir = out_dir / "checkpoints"
    save_steps = max(100, item.max_steps // 2)
    train_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        max_steps=item.max_steps,
        learning_rate=item.lr,
        per_device_train_batch_size=item.per_device_train_batch_size,
        per_device_eval_batch_size=item.per_device_eval_batch_size,
        gradient_accumulation_steps=item.gradient_accumulation_steps,
        warmup_ratio=item.warmup_ratio,
        weight_decay=item.weight_decay,
        eval_strategy="steps",
        eval_steps=max(100, min(250, item.max_steps)),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        logging_steps=max(25, min(100, item.max_steps)),
        seed=item.seed,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )
    trainer.train()
    adapter_eval = trainer.evaluate()
    adapter_loss = _as_float(adapter_eval.get("eval_loss"), default=999.0)

    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))

    midpoint = _find_midpoint_checkpoint(ckpt_dir, midpoint_step=item.max_steps // 2)
    return {
        "metric_name": "eval_loss",
        "higher_is_better": False,
        "base_score": round(base_loss, 6),
        "adapter_score": round(adapter_loss, 6),
        "directional_delta": round(base_loss - adapter_loss, 6),
        "base_perplexity": round(_safe_exp(base_loss), 6),
        "adapter_perplexity": round(_safe_exp(adapter_loss), 6),
        "midpoint_checkpoint": midpoint,
    }


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = load_json(path)
    if data.get("training_complete"):
        return data
    return None


def train_item(item: CohortItem, adapter_root: Path, log_root: Path) -> dict[str, Any]:
    out_dir = adapter_dir(adapter_root, item.adapter_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    manifest_path = training_manifest_path(adapter_root, item.adapter_id)
    existing = _load_existing_manifest(manifest_path)
    if existing is not None:
        print(f"[train] {item.adapter_id}: already complete, skipping")
        return existing

    print(f"[train] {item.adapter_id} ({item.task_type}, r={item.lora_rank}, seed={item.seed})")
    t0 = time.monotonic()
    if item.task_type == "SEQ_CLS":
        metrics = _train_seq_cls(item, out_dir=out_dir, log_dir=log_root)
    elif item.task_type == "CAUSAL_LM":
        metrics = _train_causal_lm(item, out_dir=out_dir, log_dir=log_root)
    else:
        raise ValueError(f"Unsupported task_type: {item.task_type}")
    elapsed = time.monotonic() - t0

    record = {
        "schema": "gradience.decoder_triage.training_manifest/v1",
        "adapter_id": item.adapter_id,
        "base_model": item.base_model,
        "task_type": item.task_type,
        "task_family": item.task_family,
        "dataset": item.dataset,
        "dataset_subset": item.dataset_subset,
        "split": item.split,
        "seed": item.seed,
        "lora_rank": item.lora_rank,
        "lora_alpha": item.lora_alpha,
        "target_modules": list(item.target_modules),
        "metric_name": metrics["metric_name"],
        "higher_is_better": bool(metrics["higher_is_better"]),
        "base_score": metrics["base_score"],
        "adapter_score": metrics["adapter_score"],
        "directional_delta": metrics["directional_delta"],
        "verified": bool(metrics["directional_delta"] > 0),
        "midpoint_checkpoint": metrics.get("midpoint_checkpoint"),
        "extra_metrics": {k: v for k, v in metrics.items() if k not in {"metric_name", "higher_is_better", "base_score", "adapter_score", "directional_delta", "midpoint_checkpoint"}},
        "max_steps": item.max_steps,
        "lr": item.lr,
        "train_samples": item.train_samples,
        "eval_samples": item.eval_samples,
        "max_seq_length": item.max_seq_length,
        "per_device_train_batch_size": item.per_device_train_batch_size,
        "per_device_eval_batch_size": item.per_device_eval_batch_size,
        "gradient_accumulation_steps": item.gradient_accumulation_steps,
        "adapter_dir": str(out_dir),
        "training_complete": True,
        "completed_at": _timestamp(),
        "elapsed_seconds": round(elapsed, 2),
    }
    write_json(manifest_path, record)
    print(
        f"[train] {item.adapter_id}: done "
        f"({record['metric_name']} adapter={record['adapter_score']:.4f}, "
        f"base={record['base_score']:.4f}, Δdir={record['directional_delta']:+.4f})"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decoder triage study cohort.")
    parser.add_argument(
        "--manifest",
        default="scripts/decoder_triage_study/cohort_manifest.json",
        help="Path to cohort manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output adapter directory (e.g., /workspace/experiments/decoder_merge_triage/adapters)",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Training log directory",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Output path for aggregate adapter manifest JSON (default: <output-dir>/../adapter_manifest.json)",
    )
    parser.add_argument(
        "--max-adapters",
        type=int,
        default=0,
        help="Optional cap for debug/smoke runs (0 = all)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    adapter_root = Path(args.output_dir).resolve()
    log_root = Path(args.log_dir).resolve()
    agg_out = (
        Path(args.manifest_out).resolve()
        if args.manifest_out
        else (adapter_root.parent / "adapter_manifest.json").resolve()
    )

    items = load_cohort_manifest(manifest_path)
    if args.max_adapters > 0:
        items = items[: args.max_adapters]

    adapter_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    start = time.monotonic()
    for item in items:
        results.append(train_item(item, adapter_root=adapter_root, log_root=log_root))

    summary = {
        "schema": "gradience.decoder_triage.adapter_manifest/v1",
        "generated_at": _timestamp(),
        "cohort_manifest": str(manifest_path),
        "output_dir": str(adapter_root),
        "log_dir": str(log_root),
        "adapter_count": len(results),
        "verified_count": sum(1 for r in results if r.get("verified", False)),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "adapters": results,
    }
    write_json(agg_out, summary)
    print(f"\nWrote aggregate adapter manifest: {agg_out}")


if __name__ == "__main__":
    main()
