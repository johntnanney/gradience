#!/usr/bin/env python3
"""
N07 DeBERTa Adjudication — Phase 1: Train 8 Adapters

Trains 8 DeBERTa-v3-base LoRA adapters per the N07 pre-registered protocol:
  4 tasks (QNLI, RTE, MRPC, SST-2) x 2 seeds (42, 7)
  LoRA r=16, alpha=16, dropout=0.1, targets=Q/K/V/O
  3 epochs, lr=2e-4, batch_size=16, warmup=6%, weight_decay=0.01

Each adapter gets:
  - GradienceCallback telemetry (run.jsonl)
  - StructuralMetricsCallback (SVD snapshots every 50 steps)
  - CurvatureSidecarCallback (Hessian snapshots every 50 steps)

Usage:
    python scripts/n07_deberta/train_deberta.py --output /workspace/n07

    # Smoke test (2 steps, 1 seed, 1 task):
    python scripts/n07_deberta/train_deberta.py --output /workspace/n07 --smoke

    # Single task:
    python scripts/n07_deberta/train_deberta.py --output /workspace/n07 --task qnli

Outputs:
    /workspace/n07/adapters/deberta_{task}_s{seed}/
        adapter_model.safetensors
        adapter_config.json
        run.jsonl              (training telemetry + curvature + structural)
        source_score.json      (single-task eval accuracy)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# N07 Protocol Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/deberta-v3-base"

TASKS = {
    "qnli": {
        "dataset": "glue",
        "subset": "qnli",
        "text_cols": ["question", "sentence"],
        "label_col": "label",
        "num_labels": 2,
    },
    "rte": {
        "dataset": "glue",
        "subset": "rte",
        "text_cols": ["sentence1", "sentence2"],
        "label_col": "label",
        "num_labels": 2,
    },
    "mrpc": {
        "dataset": "glue",
        "subset": "mrpc",
        "text_cols": ["sentence1", "sentence2"],
        "label_col": "label",
        "num_labels": 2,
    },
    "sst2": {
        "dataset": "glue",
        "subset": "sst2",
        "text_cols": ["sentence"],
        "label_col": "label",
        "num_labels": 2,
    },
}

SEEDS = [42, 7]

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["query_proj", "key_proj", "value_proj", "dense"],
    # DeBERTa-v3 attention projections:
    #   query_proj, key_proj, value_proj = Q, K, V
    #   dense = O (output projection in self-attention)
    # Verify these names on first run — if DeBERTa-v3's module names differ,
    # the LoRA injection will silently skip unmatched names.
    "bias": "none",
    "task_type": "SEQ_CLS",
}

TRAINING_ARGS = {
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "logging_steps": 10,
    "save_strategy": "no",       # save manually at end
    "fp16": True,                # A100/A10G; override to bf16 if needed
    "dataloader_num_workers": 2,
    "remove_unused_columns": True,
    "report_to": "none",         # we use GradienceCallback, not wandb
}

# Smoke test overrides
SMOKE_OVERRIDES = {
    "max_steps": 2,
    "eval_steps": 2,
    "logging_steps": 1,
    "num_train_epochs": 1,
}


def get_adapter_id(task: str, seed: int) -> str:
    return f"deberta_{task}_s{seed}"


def train_one_adapter(
    task: str,
    seed: int,
    output_dir: Path,
    smoke: bool = False,
    curvature_every: int = 50,
    structural_every: int = 50,
) -> dict:
    """Train a single DeBERTa LoRA adapter with full telemetry.

    Returns dict with adapter_id, accuracy, output_path, training_time.
    """
    import evaluate
    import numpy as np
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig

    adapter_id = get_adapter_id(task, seed)
    adapter_dir = output_dir / "adapters" / adapter_id

    # Skip if already trained
    if (adapter_dir / "adapter_config.json").exists():
        print(f"  SKIP {adapter_id} (already exists)")
        with open(adapter_dir / "source_score.json") as f:
            return json.load(f)

    print(f"\n{'=' * 60}")
    print(f"  Training {adapter_id}")
    print(f"{'=' * 60}")

    task_cfg = TASKS[task]
    t0 = time.time()

    # --- Load model + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=task_cfg["num_labels"],
    )

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)

    # Verify LoRA was injected into the expected modules
    lora_modules = [n for n, _ in model.named_parameters() if "lora_" in n]
    print(f"  LoRA params: {len(lora_modules)} tensors")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    # --- Sanity check: all 4 projection types should have LoRA ---
    expected_modules = {"query_proj", "key_proj", "value_proj", "dense"}
    found_modules = set()
    for name in lora_modules:
        for mod in expected_modules:
            if mod in name:
                found_modules.add(mod)
    missing = expected_modules - found_modules
    if missing:
        print(f"  WARNING: LoRA not injected into: {missing}")
        print(f"  Available modules in model:")
        for name, _ in model.base_model.model.named_modules():
            if "attention" in name.lower() or "proj" in name.lower():
                print(f"    {name}")
        print("  You may need to update LORA_CONFIG['target_modules'].")
        print("  Continuing anyway — but per-module analysis will be incomplete.")

    # --- Load + tokenize dataset ---
    dataset = load_dataset(task_cfg["dataset"], task_cfg["subset"])

    def tokenize(examples):
        text_cols = task_cfg["text_cols"]
        if len(text_cols) == 2:
            return tokenizer(
                examples[text_cols[0]],
                examples[text_cols[1]],
                truncation=True,
                max_length=256,
                padding="max_length",
            )
        else:
            return tokenizer(
                examples[text_cols[0]],
                truncation=True,
                max_length=128,
                padding="max_length",
            )

    tokenized = dataset.map(tokenize, batched=True)

    # Use validation split for eval (GLUE tasks have validation but no public test)
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    if smoke:
        train_dataset = train_dataset.select(range(min(32, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(16, len(eval_dataset))))

    # --- Metrics ---
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=-1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    # --- Training args ---
    training_args_dict = {**TRAINING_ARGS}
    if smoke:
        training_args_dict.update(SMOKE_OVERRIDES)

    training_args_dict["seed"] = seed
    training_args_dict["output_dir"] = str(adapter_dir / "trainer_output")

    args = TrainingArguments(**training_args_dict)

    # --- Callbacks ---
    gradience_config = GradienceCallbackConfig(
        output_dir=str(adapter_dir),
        filename="run.jsonl",
        dataset_name=f"glue/{task_cfg['subset']}",
    )
    gradience_cb = GradienceCallback(gradience_config)

    # Structural metrics (SVD on LoRA weights)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "m1_experiment"))
    from phase1_train_telemetry import _make_structural_callback
    structural_cb = _make_structural_callback(gradience_cb, structural_every_n=structural_every)

    # Curvature sidecar (Hessian spectrum)
    from curvature_sidecar import make_batch_capture_training_step, make_curvature_callback
    curvature_cb = make_curvature_callback(
        gradience_callback=gradience_cb,
        snapshot_every_n=curvature_every,
        n_top_eigenvalues=3,
        n_trace_samples=10,
        power_iterations=15,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[gradience_cb, structural_cb, curvature_cb],
    )

    # Patch training_step to capture batches for curvature estimation
    make_batch_capture_training_step(trainer, curvature_cb)

    # --- Train ---
    torch.manual_seed(seed)
    trainer.train()

    # --- Final eval ---
    eval_results = trainer.evaluate()
    accuracy = eval_results.get("eval_accuracy", 0.0)
    training_time = time.time() - t0

    print(f"\n  {adapter_id}: accuracy={accuracy:.4f}, time={training_time:.0f}s")

    # --- Save adapter ---
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # --- Save source score ---
    result = {
        "adapter_id": adapter_id,
        "task": task,
        "seed": seed,
        "accuracy": accuracy,
        "eval_results": eval_results,
        "output_path": str(adapter_dir),
        "training_time_s": round(training_time, 1),
        "n_trainable_params": n_trainable,
        "lora_modules_found": sorted(found_modules),
        "lora_modules_missing": sorted(missing) if missing else [],
    }
    with open(adapter_dir / "source_score.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="N07 DeBERTa Training")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (2 steps, 1 seed)")
    parser.add_argument("--task", type=str, choices=list(TASKS.keys()), help="Single task only")
    parser.add_argument("--curvature-every", type=int, default=50, help="Curvature snapshot interval")
    parser.add_argument("--structural-every", type=int, default=50, help="Structural snapshot interval")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [args.task] if args.task else list(TASKS.keys())
    seeds = [42] if args.smoke else SEEDS

    print(f"N07 DeBERTa Training")
    print(f"  Output: {output_dir}")
    print(f"  Tasks: {tasks}")
    print(f"  Seeds: {seeds}")
    print(f"  Smoke: {args.smoke}")
    print(f"  Curvature every: {args.curvature_every} steps")
    print(f"  Structural every: {args.structural_every} steps")
    print()

    results = []
    for task in tasks:
        for seed in seeds:
            result = train_one_adapter(
                task=task,
                seed=seed,
                output_dir=output_dir,
                smoke=args.smoke,
                curvature_every=args.curvature_every,
                structural_every=args.structural_every,
            )
            results.append(result)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Training Complete: {len(results)} adapters")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['adapter_id']}: accuracy={r['accuracy']:.4f} ({r['training_time_s']:.0f}s)")

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary: {summary_path}")

    # --- Validation gate (N07 protocol) ---
    # Flag any adapter that fails to reach reasonable accuracy
    for r in results:
        if r["accuracy"] < 0.70:
            print(f"\n  WARNING: {r['adapter_id']} accuracy {r['accuracy']:.4f} < 0.70")
            print(f"  The N07 protocol says: 'investigate before proceeding'")

    if any(r.get("lora_modules_missing") for r in results):
        print(f"\n  WARNING: Some adapters have missing LoRA modules.")
        print(f"  Check LORA_CONFIG['target_modules'] against DeBERTa-v3 architecture.")
        print(f"  Per-module analysis (Prediction D) requires all four: Q/K/V/O.")


if __name__ == "__main__":
    main()
