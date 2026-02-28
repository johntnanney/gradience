#!/usr/bin/env python3
"""
Phase 1 (Telemetry-Enhanced): Train adapters with GradienceCallback + eval.

Identical to phase1_train.py but adds:
  1. GradienceCallback — logs run.jsonl with loss, lr, grad_norm per step
  2. Eval split — holds out 10% of training data for validation
  3. eval_strategy="steps", eval_steps=25 — yields ~48 eval events per 1200-step run
  4. logging_steps=10 — finer-grained grad_norm telemetry between evals

The resulting run.jsonl files can be fed directly into
`gradience.analysis.extract_timeseries` for lead-lag analysis.

Usage:
    python scripts/m1_experiment/phase1_train_telemetry.py \
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test (5 steps, 1 seed):
    python scripts/m1_experiment/phase1_train_telemetry.py \
        --config scripts/m1_experiment/m1_config.yaml --smoke

    # Single task:
    python scripts/m1_experiment/phase1_train_telemetry.py \
        --config scripts/m1_experiment/m1_config.yaml --task math
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


def load_config(config_path: str, smoke: bool = False) -> dict:
    """Load and optionally apply smoke test overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["training"]["max_steps"] = smoke_cfg.get("max_steps", 5)
        for task in config["adapters"].values():
            task["max_train_samples"] = smoke_cfg.get("max_train_samples", 50)
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])

    return config


def train_single_adapter(
    base_model: str,
    task_name: str,
    task_config: dict,
    training_config: dict,
    seed: int,
    output_dir: Path,
    device: str = "cuda",
    eval_steps: int = 25,
    logging_steps: int = 10,
    eval_fraction: float = 0.10,
) -> Path:
    """Train one LoRA adapter with GradienceCallback + eval."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # Gradience telemetry callback
    from gradience.vnext.integrations.hf import (
        GradienceCallback,
        GradienceCallbackConfig,
    )

    # Import task formatter (sibling module)
    sys.path.insert(0, str(Path(__file__).parent))
    from task_configs import get_formatter

    adapter_dir = output_dir / task_name / f"seed_{seed}"

    # Skip if already trained
    if (adapter_dir / "adapter_config.json").exists():
        print(f"  [SKIP] {task_name}/seed_{seed} -- already exists")
        return adapter_dir

    print(f"\n  Training {task_name}/seed_{seed}...")
    start = time.monotonic()

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(training_config.get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=training_config["rank"],
        lora_alpha=training_config["alpha"],
        lora_dropout=0.0,
        target_modules=training_config["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load dataset
    ds_name = task_config["dataset"]
    ds_subset = task_config.get("subset", None)
    if ds_subset:
        ds = load_dataset(ds_name, ds_subset, split="train")
    else:
        ds = load_dataset(ds_name, split="train")

    # Subsample
    max_samples = task_config.get("max_train_samples", 10000)
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    print(f"    Dataset: {ds_name}, {len(ds)} examples (before split)")

    # Format + tokenize
    formatter = get_formatter(task_name)

    def tokenize_fn(example):
        text = formatter(example)
        enc = tokenizer(text, truncation=True, max_length=512, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names)

    # ---- NEW: Train/eval split ----
    n_eval = max(1, int(len(tokenized) * eval_fraction))
    n_train = len(tokenized) - n_eval
    split = tokenized.train_test_split(
        test_size=n_eval,
        seed=seed,
        shuffle=True,
    )
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"    Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- Gradience telemetry setup ----
    telemetry_dir = adapter_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    gradience_config = GradienceCallbackConfig(
        output_dir=str(telemetry_dir),
        filename="run.jsonl",
        dataset_name=ds_name,
        notes=f"m1_lead_lag | task={task_name} seed={seed}",
    )
    gradience_callback = GradienceCallback(config=gradience_config)

    # Training arguments (enhanced for lead-lag)
    train_dir = adapter_dir / "training_logs"
    training_args = TrainingArguments(
        output_dir=str(train_dir),
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation"],
        learning_rate=training_config["learning_rate"],
        max_steps=training_config["max_steps"],
        # ---- Telemetry-specific settings ----
        logging_steps=logging_steps,       # grad_norm every 10 steps (was 50)
        eval_strategy="steps",             # periodic eval
        eval_steps=eval_steps,             # eval every 25 steps (~48 evals per 1200 steps)
        # ---- Unchanged ----
        save_strategy="no",
        bf16=(training_config.get("torch_dtype") == "bfloat16"),
        fp16=(training_config.get("torch_dtype") == "float16"),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        # Disable find_unused_parameters warning with gradient checkpointing
        ddp_find_unused_parameters=False,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[gradience_callback],
    )
    trainer.train()

    # Save adapter
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))

    elapsed = time.monotonic() - start
    print(f"    Saved to: {adapter_dir} ({elapsed / 60:.1f} min)")
    print(f"    Telemetry: {telemetry_dir / 'run.jsonl'}")

    # Cleanup GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return adapter_dir


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 1 (Telemetry-Enhanced): Train adapters")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (5 steps, 1 seed)")
    parser.add_argument("--task", type=str, default=None,
                        help="Train a single task (e.g. 'math'). Default: all tasks")
    parser.add_argument("--eval-steps", type=int, default=25,
                        help="Eval every N steps (default: 25)")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Log grad_norm every N steps (default: 10)")
    parser.add_argument("--eval-fraction", type=float, default=0.10,
                        help="Fraction of data to hold out for eval (default: 0.10)")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    base_model = config["experiment"]["base_model"]
    seeds = config["experiment"]["seeds"]
    training_config = config["training"]

    # Filter tasks if --task specified
    adapter_configs = config["adapters"]
    if args.task:
        if args.task not in adapter_configs:
            print(f"Error: unknown task '{args.task}'. Available: {sorted(adapter_configs.keys())}")
            sys.exit(1)
        adapter_configs = {args.task: adapter_configs[args.task]}

    total_start = time.monotonic()
    n_total = len(adapter_configs) * len(seeds)
    n_done = 0

    max_steps = training_config["max_steps"]
    expected_evals = max_steps // args.eval_steps

    print(f"Phase 1 (Telemetry-Enhanced): Training {n_total} adapters")
    print(f"  Base model: {base_model}")
    print(f"  Seeds: {seeds}")
    print(f"  Tasks: {list(adapter_configs.keys())}")
    print(f"  Max steps: {max_steps}")
    print(f"  Eval every: {args.eval_steps} steps (~{expected_evals} eval events)")
    print(f"  Log every: {args.logging_steps} steps")
    print(f"  Eval fraction: {args.eval_fraction:.0%}")

    for task_name, task_config in adapter_configs.items():
        for seed in seeds:
            n_done += 1
            print(f"\n[{n_done}/{n_total}] {task_name}/seed_{seed}")
            train_single_adapter(
                base_model=base_model,
                task_name=task_name,
                task_config=task_config,
                training_config=training_config,
                seed=seed,
                output_dir=adapters_dir,
                device=config["runtime"]["device"],
                eval_steps=args.eval_steps,
                logging_steps=args.logging_steps,
                eval_fraction=args.eval_fraction,
            )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 1 complete: {n_total} adapters in {elapsed / 3600:.1f} hours")
    print(f"\nTelemetry files:")
    for task_name in adapter_configs:
        for seed in seeds:
            tpath = adapters_dir / task_name / f"seed_{seed}" / "telemetry" / "run.jsonl"
            print(f"  {tpath}")


if __name__ == "__main__":
    main()
