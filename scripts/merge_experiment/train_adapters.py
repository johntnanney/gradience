#!/usr/bin/env python3
"""
Train LoRA adapters for MNLI and QNLI — fallback when Predibase templates fail.

Trains two rank-8 LoRA adapters on Mistral-7B using word-label prompts.
Each adapter targets q_proj and v_proj (matching Predibase config) so that
merge-audit results are directly comparable.

Estimated time: ~45 minutes total on A40 (two adapters x ~20 min each).

Usage::

    python train_adapters.py \\
        --base-model mistralai/Mistral-7B-v0.1 \\
        --output-dir /workspace/merge_experiment/trained_adapters \\
        --epochs 2 --batch-size 16 --lr 2e-4 --max-seq-len 256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    print(f"\nMissing dependencies: {e}")
    print('Install with: pip install "gradience[bench]"')
    sys.exit(1)

# Sibling import for prompt templates
sys.path.insert(0, str(Path(__file__).parent))
from prompt_templates import format_mnli_prompt, format_qnli_prompt  # noqa: E402

# Label maps — training data uses word labels (custom template mode)
MNLI_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
QNLI_LABEL_MAP = {0: "entailment", 1: "not_entailment"}


# ---------------------------------------------------------------------------
# Training data formatting
# ---------------------------------------------------------------------------


def format_training_example(example: dict, dataset_name: str) -> str:
    """Format a dataset example into prompt + completion for SFT."""
    if dataset_name == "mnli":
        prompt = format_mnli_prompt(
            example["premise"],
            example["hypothesis"],
            few_shot=False,
            template_mode="custom",
        )
        label = MNLI_LABEL_MAP[example["label"]]
    else:
        prompt = format_qnli_prompt(
            example["question"],
            example["sentence"],
            few_shot=False,
            template_mode="custom",
        )
        label = QNLI_LABEL_MAP[example["label"]]
    return f"{prompt} {label}"


# ---------------------------------------------------------------------------
# Single adapter training
# ---------------------------------------------------------------------------


def train_single_adapter(
    args: argparse.Namespace,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """Train a single LoRA adapter on the given task."""
    print(f"\n{'=' * 60}")
    print(f"  Training {dataset_name.upper()} adapter")
    print(f"{'=' * 60}")
    start = time.monotonic()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"  Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    # Apply LoRA — matches Predibase config for comparability
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load dataset
    print(f"  Loading dataset: glue/{dataset_name}")
    if dataset_name == "mnli":
        ds = load_dataset("glue", "mnli", split="train")
    else:
        ds = load_dataset("glue", "qnli", split="train")

    # Subsample for speed — 10k examples is sufficient for rank-8
    max_train = args.max_train_samples
    if len(ds) > max_train:
        ds = ds.select(range(max_train))
    print(f"  Training on {len(ds)} examples")

    # Tokenize
    def tokenize_fn(example):
        text = format_training_example(example, dataset_name)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names)

    # Data collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    grad_accum = max(1, 16 // args.batch_size)
    training_args = TrainingArguments(
        output_dir=str(output_dir / f"{dataset_name}_training"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
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
    adapter_dir = output_dir / dataset_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))

    elapsed = time.monotonic() - start
    print(f"  Saved to: {adapter_dir}")
    print(f"  Training time: {elapsed / 60:.1f} minutes")

    # Cleanup GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return adapter_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train fallback LoRA adapters for MNLI and QNLI"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model HuggingFace identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save trained adapters",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
        help="Max training samples per task (default: 10000)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.monotonic()

    mnli_dir = train_single_adapter(args, "mnli", output_dir)
    qnli_dir = train_single_adapter(args, "qnli", output_dir)

    total_time = time.monotonic() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Both adapters trained in {total_time / 60:.1f} minutes")
    print(f"  MNLI adapter: {mnli_dir}")
    print(f"  QNLI adapter: {qnli_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
