#!/usr/bin/env python3
"""
Phase 1: Train 12 adapters (4 tasks x 3 seeds) on Mistral-7B.

For each (task, seed) pair:
  1. Load Mistral-7B base model
  2. Attach LoRA (r=32, alpha=32, q/k/v/o_proj)
  3. Fine-tune with HF Trainer for 1200 steps
  4. Save PEFT adapter

Skips if adapter_config.json already exists in the output directory.

Usage:
    python scripts/m1_experiment/phase1_train.py \
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test (5 steps, 1 seed):
    python scripts/m1_experiment/phase1_train.py \
        --config scripts/m1_experiment/m1_config.yaml --smoke
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
) -> Path:
    """Train one LoRA adapter for a (task, seed) pair."""
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
    ds_subset = task_config.get("subset")
    if ds_subset:
        ds = load_dataset(ds_name, ds_subset, split="train")
    else:
        ds = load_dataset(ds_name, split="train")

    # Subsample
    max_samples = task_config.get("max_train_samples", 10000)
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    print(f"    Dataset: {ds_name}, {len(ds)} examples")

    # Format + tokenize
    formatter = get_formatter(task_name)

    def tokenize_fn(example):
        text = formatter(example)
        enc = tokenizer(text, truncation=True, max_length=512, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names)

    # Collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    train_dir = adapter_dir / "training_logs"
    training_args = TrainingArguments(
        output_dir=str(train_dir),
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation"],
        learning_rate=training_config["learning_rate"],
        max_steps=training_config["max_steps"],
        logging_steps=50,
        save_strategy="no",
        bf16=(training_config.get("torch_dtype") == "bfloat16"),
        fp16=(training_config.get("torch_dtype") == "float16"),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    # Save adapter
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))

    elapsed = time.monotonic() - start
    print(f"    Saved to: {adapter_dir} ({elapsed / 60:.1f} min)")

    # Cleanup GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return adapter_dir


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 1: Train adapters")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (5 steps, 1 seed)")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    base_model = config["experiment"]["base_model"]
    seeds = config["experiment"]["seeds"]
    training_config = config["training"]

    total_start = time.monotonic()
    n_total = len(config["adapters"]) * len(seeds)
    n_done = 0

    print(f"Phase 1: Training {n_total} adapters")
    print(f"  Base model: {base_model}")
    print(f"  Seeds: {seeds}")
    print(f"  Tasks: {list(config['adapters'].keys())}")

    for task_name, task_config in config["adapters"].items():
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
            )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 1 complete: {n_total} adapters in {elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
