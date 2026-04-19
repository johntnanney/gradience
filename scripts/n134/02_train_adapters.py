"""N134 Phase 1: Train 24 LoRA adapters on Mistral-7B-v0.3.

8 tasks x 3 seeds = 24 adapters.  Seed 42 adapters already exist from
the pilot run and are copied/symlinked from /workspace/n134/pilot/.
Only seeds 123 and 456 (16 new adapters) are trained.

Tasks (multiple-choice QA):
    arc_challenge, hellaswag, winogrande, openbookqa,
    commonsenseqa, piqa, siqa, boolq

LoRA: r=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj], dropout=0.05
Training: AdamW lr=2e-4, warmup 6%, bf16, cosine schedule

Usage:
    python3 02_train_adapters.py                          # train all 24
    python3 02_train_adapters.py --task arc_challenge      # one task, all seeds
    python3 02_train_adapters.py --seed 123                # all tasks, one seed
    python3 02_train_adapters.py --smoke                   # quick smoke test

Output: /workspace/n134/adapters/{task}_s{seed}/
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
CACHE_DIR = "/workspace/hf_cache"
OUTPUT_ROOT = Path("/workspace/n134/adapters")
PILOT_ROOT = Path("/workspace/n134/pilot")
EVAL_ROOT = Path("/workspace/n134/evals")

TASKS = [
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "openbookqa",
    "commonsenseqa",
    "piqa",
    "siqa",
    "boolq",
]

SEEDS = [42, 123, 456]
PILOT_SEED = 42  # seed whose adapters already exist from pilot

MAX_SEQ_LEN = 512
SMOKE_MAX_STEPS = 20
SMOKE_EVAL_SAMPLES = 20

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "task_type": TaskType.CAUSAL_LM,
}

# ---------------------------------------------------------------------------
# Task configuration registry
# ---------------------------------------------------------------------------

TASK_CONFIGS: dict[str, dict] = {
    "arc_challenge": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "train_split": "train",
        "val_split": "validation",
    },
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "config": None,
        "train_split": "train",
        "val_split": "validation",
    },
    "winogrande": {
        "dataset": "winogrande",
        "config": "winogrande_xl",
        "train_split": "train",
        "val_split": "validation",
    },
    "openbookqa": {
        "dataset": "openbookqa",
        "config": None,
        "train_split": "train",
        "val_split": "validation",
    },
    "commonsenseqa": {
        "dataset": "tau/commonsense_qa",
        "config": None,
        "train_split": "train",
        "val_split": "validation",
    },
    "piqa": {
        "dataset": "piqa",
        "config": None,
        "train_split": "train",
        "val_split": "validation",
    },
    "siqa": {
        "dataset": "social_i_qa",
        "config": None,
        "train_split": "train",
        "val_split": "validation",
    },
    "boolq": {
        "dataset": "boolq",
        "config": None,
        "train_split": "train",
        "val_split": "validation",
    },
}


# ---------------------------------------------------------------------------
# Dataset formatting functions (multiple-choice QA text format)
# ---------------------------------------------------------------------------

def _choices_text(labels: list[str], texts: list[str]) -> str:
    """Format choices as (A) text (B) text ..."""
    return " ".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))


def format_arc_challenge(example: dict) -> dict:
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = example["answerKey"]
    text = (
        f"Question: {example['question']}\n"
        f"Choices: {_choices_text(labels, texts)}\n"
        f"Answer: {answer_key}"
    )
    return {"text": text, "answer": answer_key, "choices_labels": labels}


def format_hellaswag(example: dict) -> dict:
    endings = example["endings"]
    labels = ["A", "B", "C", "D"]
    answer_idx = int(example["label"])
    answer_key = labels[answer_idx]
    text = (
        f"Context: {example['ctx']}\n"
        f"Choices: {_choices_text(labels, endings)}\n"
        f"Answer: {answer_key}"
    )
    return {"text": text, "answer": answer_key, "choices_labels": labels}


def format_winogrande(example: dict) -> dict:
    labels = ["1", "2"]
    texts = [example["option1"], example["option2"]]
    answer_key = example["answer"]
    text = (
        f"Sentence: {example['sentence']}\n"
        f"Choices: {_choices_text(labels, texts)}\n"
        f"Answer: {answer_key}"
    )
    return {"text": text, "answer": answer_key, "choices_labels": labels}


def format_openbookqa(example: dict) -> dict:
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = example["answerKey"]
    text = (
        f"Question: {example['question_stem']}\n"
        f"Choices: {_choices_text(labels, texts)}\n"
        f"Answer: {answer_key}"
    )
    return {"text": text, "answer": answer_key, "choices_labels": labels}


def format_commonsenseqa(example: dict) -> dict:
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = example["answerKey"]
    text = (
        f"Question: {example['question']}\n"
        f"Choices: {_choices_text(labels, texts)}\n"
        f"Answer: {answer_key}"
    )
    return {"text": text, "answer": answer_key, "choices_labels": labels}


def format_piqa(example: dict) -> dict:
    labels = ["A", "B"]
    texts = [example["sol1"], example["sol2"]]
    answer_idx = int(example["label"])
    answer_key = labels[answer_idx]
    text = (
        f"Goal: {example['goal']}\n"
        f"Choices: {_choices_text(labels, texts)}\n"
        f"Answer: {answer_key}"
    )
    return {"text": text, "answer": answer_key, "choices_labels": labels}


def format_siqa(example: dict) -> dict:
    labels = ["A", "B", "C"]
    texts = [example["answerA"], example["answerB"], example["answerC"]]
    answer_idx = int(example["label"]) - 1  # siqa labels are 1-indexed
    answer_key = labels[answer_idx]
    text = (
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        f"Choices: {_choices_text(labels, texts)}\n"
        f"Answer: {answer_key}"
    )
    return {"text": text, "answer": answer_key, "choices_labels": labels}


def format_boolq(example: dict) -> dict:
    labels = ["Yes", "No"]
    answer_key = "Yes" if example["answer"] else "No"
    text = (
        f"Passage: {example['passage'][:800]}\n"
        f"Question: {example['question']}\n"
        f"Answer: {answer_key}"
    )
    return {"text": text, "answer": answer_key, "choices_labels": labels}


FORMATTERS: dict[str, callable] = {
    "arc_challenge": format_arc_challenge,
    "hellaswag": format_hellaswag,
    "winogrande": format_winogrande,
    "openbookqa": format_openbookqa,
    "commonsenseqa": format_commonsenseqa,
    "piqa": format_piqa,
    "siqa": format_siqa,
    "boolq": format_boolq,
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_task_dataset(task: str, smoke: bool = False):
    """Load and format dataset for a given task."""
    cfg = TASK_CONFIGS[task]
    max_train = 5000 if not smoke else 100
    max_eval = 500 if not smoke else SMOKE_EVAL_SAMPLES

    load_kwargs: dict = {"cache_dir": CACHE_DIR}
    if cfg["config"] is not None:
        load_kwargs["name"] = cfg["config"]

    ds = load_dataset(cfg["dataset"], **load_kwargs)

    train_raw = ds[cfg["train_split"]]
    val_raw = ds[cfg["val_split"]]

    train_raw = train_raw.select(range(min(max_train, len(train_raw))))
    val_raw = val_raw.select(range(min(max_eval, len(val_raw))))

    formatter = FORMATTERS[task]
    # Keep only the "text" column for training
    train = train_raw.map(formatter, remove_columns=train_raw.column_names)
    val = val_raw.map(formatter, remove_columns=val_raw.column_names)

    return train, val


def tokenize_dataset(dataset, tokenizer, max_length: int = MAX_SEQ_LEN):
    """Tokenize for causal LM training (input = output)."""

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c != "text"],
    ).remove_columns(["text"] if "text" in dataset.column_names else [])


# ---------------------------------------------------------------------------
# Pilot adapter copy
# ---------------------------------------------------------------------------

def copy_pilot_adapters(tasks: list[str]) -> None:
    """Copy or symlink seed-42 adapters from pilot directory."""
    for task in tasks:
        adapter_name = f"{task}_s{PILOT_SEED}"
        dest = OUTPUT_ROOT / adapter_name
        src = PILOT_ROOT / adapter_name

        if (dest / "adapter_model.safetensors").exists():
            print(f"  SKIP pilot copy {adapter_name} -- already exists at {dest}")
            continue

        if not src.exists():
            print(f"  WARN pilot source {src} not found -- will train seed {PILOT_SEED} from scratch")
            continue

        dest.mkdir(parents=True, exist_ok=True)
        # Copy all files from pilot adapter directory
        for item in src.iterdir():
            dest_item = dest / item.name
            if not dest_item.exists():
                if item.is_file():
                    shutil.copy2(str(item), str(dest_item))
                elif item.is_dir():
                    shutil.copytree(str(item), str(dest_item))
        print(f"  Copied pilot adapter {adapter_name}: {src} -> {dest}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_adapter(task: str, seed: int, tokenizer, smoke: bool = False) -> Path:
    """Train one LoRA adapter. Idempotent: skips if safetensors exists."""
    adapter_name = f"{task}_s{seed}"
    output_dir = OUTPUT_ROOT / adapter_name

    if (output_dir / "adapter_model.safetensors").exists():
        print(f"\n  SKIP {adapter_name} -- already trained")
        return output_dir

    print(f"\n{'=' * 60}")
    print(f"  Training: {adapter_name}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Load dataset
    print(f"  Loading {task} dataset...")
    train_ds, val_ds = load_task_dataset(task, smoke=smoke)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Tokenize
    print("  Tokenizing...")
    train_tok = tokenize_dataset(train_ds, tokenizer)

    # Load model fresh for each adapter.
    # See comment in 00_pilot_train.py: avoid device_map="auto" which can
    # offload layers to meta/CPU under residual-VRAM fragmentation, causing
    # Trainer(...) to fail at _move_model_to_device. Load to CPU then .to(cuda).
    print("  Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model = model.to("cuda")

    # Apply LoRA
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args
    max_steps = SMOKE_MAX_STEPS if smoke else 1000
    num_epochs = 1 if smoke else 3

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=4,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        data_collator=data_collator,
    )

    # Train
    print(f"  Training (max_steps={max_steps}, epochs={num_epochs})...")
    trainer.train()

    # Save adapter only
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    elapsed = time.time() - t0
    print(f"  Done: {adapter_name} ({elapsed:.0f}s)")

    # Record metadata (accuracy filled in by evaluation step)
    meta = {
        "task": task,
        "seed": seed,
        "elapsed_s": elapsed,
        "train_samples": len(train_ds),
        "max_steps": max_steps,
        "smoke": smoke,
        "global_step": trainer.state.global_step,
        "train_loss": (
            float(trainer.state.log_history[-1].get("train_loss", 0))
            if trainer.state.log_history
            else None
        ),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Clean up GPU memory
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# ---------------------------------------------------------------------------
# Source evaluation (validation-set accuracy)
# ---------------------------------------------------------------------------

def evaluate_adapter(task: str, seed: int, tokenizer, smoke: bool = False) -> dict:
    """Evaluate adapter on validation set via generative multiple-choice."""
    adapter_name = f"{task}_s{seed}"
    adapter_dir = OUTPUT_ROOT / adapter_name
    eval_file = EVAL_ROOT / f"{adapter_name}_source_eval.json"

    if eval_file.exists():
        print(f"  SKIP eval {adapter_name} -- already done")
        return json.loads(eval_file.read_text())

    if not (adapter_dir / "adapter_model.safetensors").exists():
        print(f"  SKIP eval {adapter_name} -- adapter not found")
        return {}

    print(f"\n  Evaluating {adapter_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    # Load eval data
    _, val_ds = load_task_dataset(task, smoke=smoke)
    n_eval = min(200, len(val_ds)) if not smoke else min(SMOKE_EVAL_SAMPLES, len(val_ds))

    correct = 0
    total = 0

    for i in range(n_eval):
        example = val_ds[i]
        text = example["text"]
        expected_answer = example["answer"]

        # Split into prompt and expected answer -- remove the answer from text
        if "Answer: " in text:
            prompt = text.rsplit("Answer: ", 1)[0] + "Answer: "
        else:
            continue

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Check if the expected answer label appears at the start of generation
        gen_start = generated[:20].strip().upper()
        expected_upper = str(expected_answer).strip().upper()
        correct += int(gen_start.startswith(expected_upper))
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"    {adapter_name}: {correct}/{total} = {accuracy:.3f}")

    result = {
        "task": task,
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    eval_file.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_file, "w") as f:
        json.dump(result, f, indent=2)

    # Update training_meta.json with final_val_accuracy
    meta_path = adapter_dir / "training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["final_val_accuracy"] = accuracy
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Accuracy verification
# ---------------------------------------------------------------------------

def verify_accuracies(tasks: list[str], seeds: list[int]) -> None:
    """Warn if any adapter accuracy falls outside [0.70, 0.90]."""
    print(f"\n{'=' * 60}")
    print("  Accuracy Verification")
    print(f"{'=' * 60}")

    warnings = []
    for task in tasks:
        for seed in seeds:
            adapter_name = f"{task}_s{seed}"
            eval_file = EVAL_ROOT / f"{adapter_name}_source_eval.json"
            if not eval_file.exists():
                warnings.append(f"  MISSING: {adapter_name} -- no eval file")
                continue
            result = json.loads(eval_file.read_text())
            acc = result.get("accuracy", 0.0)
            status = "OK" if 0.70 <= acc <= 0.90 else "WARN"
            msg = f"  {status}: {adapter_name} accuracy = {acc:.3f}"
            print(msg)
            if status == "WARN":
                warnings.append(msg)

    if warnings:
        print(f"\n  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    {w}")
    else:
        print("\n  All adapters in [0.70, 0.90] range.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="N134: Train 24 LoRA adapters (8 tasks x 3 seeds)")
    parser.add_argument("--task", type=str, default=None, help="Single task to train")
    parser.add_argument("--seed", type=int, default=None, help="Single seed")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--skip-pilot-copy", action="store_true", help="Skip copying pilot adapters")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    seeds = [args.seed] if args.seed is not None else SEEDS

    # Validate task names
    for t in tasks:
        if t not in TASK_CONFIGS:
            raise ValueError(f"Unknown task: {t}. Valid tasks: {list(TASK_CONFIGS.keys())}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Copy pilot (seed 42) adapters before training
    if not args.eval_only and not args.skip_pilot_copy and PILOT_SEED in seeds:
        print(f"\n{'=' * 60}")
        print("  Copying pilot (seed 42) adapters")
        print(f"{'=' * 60}")
        copy_pilot_adapters(tasks)

    if not args.eval_only:
        print(f"\n{'=' * 60}")
        print(f"  N134 Phase 1: Training {len(tasks) * len(seeds)} adapters")
        print(f"  Tasks: {tasks}")
        print(f"  Seeds: {seeds}")
        print(f"  Mode: {'SMOKE' if args.smoke else 'FULL'}")
        print(f"{'=' * 60}")

        t0 = time.time()
        for task in tasks:
            for seed in seeds:
                train_adapter(task, seed, tokenizer, smoke=args.smoke)
        elapsed = time.time() - t0
        print(f"\n  Total training time: {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

    # Source evaluation
    print(f"\n{'=' * 60}")
    print("  Source Evaluation")
    print(f"{'=' * 60}")
    for task in tasks:
        for seed in seeds:
            evaluate_adapter(task, seed, tokenizer, smoke=args.smoke)

    # Final accuracy verification
    verify_accuracies(tasks, seeds)


if __name__ == "__main__":
    main()
