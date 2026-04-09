"""N133 Phase 1: Train 12 LoRA adapters on Mistral-7B-v0.3.

6 tasks × 2 seeds = 12 adapters. Each adapter targets q_proj, k_proj,
v_proj, o_proj with rank 16. Training uses 3 epochs or 1000 steps
(whichever first), lr=2e-4, cosine schedule, effective batch size 16.

Usage:
    python3 n133_train_adapters.py                    # train all 12
    python3 n133_train_adapters.py --task sst2 --seed 42  # train one
    python3 n133_train_adapters.py --smoke             # quick smoke test

Output: /workspace/n133/adapters/{task}_s{seed}/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
CACHE_DIR = "/workspace/hf_cache"
OUTPUT_ROOT = Path("/workspace/n133/adapters")
EVAL_ROOT = Path("/workspace/n133/evals")

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "task_type": TaskType.CAUSAL_LM,
}

TASKS = ["sst2", "mnli", "squad", "gsm8k", "code", "summarization"]
SEEDS = [42, 123]

MAX_SEQ_LEN = 512  # keep manageable for training speed
SMOKE_MAX_STEPS = 20
SMOKE_EVAL_SAMPLES = 20


# ---------------------------------------------------------------------------
# Dataset loading and formatting
# ---------------------------------------------------------------------------

def format_sst2(example):
    """SST-2: sentiment classification."""
    label = "positive" if example["label"] == 1 else "negative"
    text = f"Classify the sentiment of this sentence as positive or negative.\n\nSentence: {example['sentence']}\nSentiment: {label}"
    return {"text": text}


def format_mnli(example):
    """MNLI: natural language inference."""
    labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
    label = labels[example["label"]]
    text = (f"Determine the relationship between the premise and hypothesis.\n\n"
            f"Premise: {example['premise']}\n"
            f"Hypothesis: {example['hypothesis']}\n"
            f"Relationship: {label}")
    return {"text": text}


def format_squad(example):
    """SQuAD: extractive QA."""
    answers = example["answers"]
    answer_text = answers["text"][0] if answers["text"] else "unanswerable"
    text = (f"Answer the question based on the context.\n\n"
            f"Context: {example['context'][:800]}\n"
            f"Question: {example['question']}\n"
            f"Answer: {answer_text}")
    return {"text": text}


def format_gsm8k(example):
    """GSM8K: math reasoning."""
    text = f"Solve the following math problem step by step.\n\nProblem: {example['question']}\nSolution: {example['answer']}"
    return {"text": text}


def format_code(example):
    """CodeAlpaca: code generation."""
    inp = example.get("input", "")
    prompt = example["instruction"]
    if inp and inp.strip():
        prompt = f"{prompt}\n\nInput: {inp}"
    text = f"Write code to solve the following task.\n\nTask: {prompt}\nCode:\n{example['output']}"
    return {"text": text}


def format_summarization(example):
    """CNN/DailyMail: summarization."""
    article = example["article"][:1500]  # truncate long articles
    text = f"Summarize the following article.\n\nArticle: {article}\n\nSummary: {example['highlights']}"
    return {"text": text}


FORMATTERS = {
    "sst2": format_sst2,
    "mnli": format_mnli,
    "squad": format_squad,
    "gsm8k": format_gsm8k,
    "code": format_code,
    "summarization": format_summarization,
}


def load_task_dataset(task: str, smoke: bool = False):
    """Load and format dataset for a given task."""
    max_train = 5000 if not smoke else 100
    max_eval = 500 if not smoke else SMOKE_EVAL_SAMPLES

    if task == "sst2":
        ds = load_dataset("glue", "sst2", cache_dir=CACHE_DIR)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))
    elif task == "mnli":
        ds = load_dataset("glue", "mnli", cache_dir=CACHE_DIR)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation_matched"].select(range(min(max_eval, len(ds["validation_matched"]))))
    elif task == "squad":
        ds = load_dataset("rajpurkar/squad_v2", cache_dir=CACHE_DIR)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))
    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", cache_dir=CACHE_DIR)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["test"].select(range(min(max_eval, len(ds["test"]))))
    elif task == "code":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", cache_dir=CACHE_DIR)
        # No standard val split — use last 500 of train
        full = ds["train"]
        n = len(full)
        train = full.select(range(min(max_train, n - 500)))
        val = full.select(range(max(0, n - 500), min(n, n - 500 + max_eval)))
    elif task == "summarization":
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", cache_dir=CACHE_DIR)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))
    else:
        raise ValueError(f"Unknown task: {task}")

    formatter = FORMATTERS[task]
    train = train.map(formatter, remove_columns=train.column_names)
    val = val.map(formatter, remove_columns=val.column_names)

    return train, val


def tokenize_dataset(dataset, tokenizer, max_length=MAX_SEQ_LEN):
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

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_adapter(task: str, seed: int, tokenizer, smoke: bool = False):
    """Train one LoRA adapter."""
    adapter_name = f"{task}_s{seed}"
    output_dir = OUTPUT_ROOT / adapter_name
    if (output_dir / "adapter_model.safetensors").exists():
        print(f"\n  SKIP {adapter_name} — already trained")
        return output_dir

    print(f"\n{'='*60}")
    print(f"  Training: {adapter_name}")
    print(f"{'='*60}")

    t0 = time.time()

    # Load dataset
    print(f"  Loading {task} dataset...")
    train_ds, val_ds = load_task_dataset(task, smoke=smoke)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Tokenize
    print("  Tokenizing...")
    train_tok = tokenize_dataset(train_ds, tokenizer)
    _val_tok = tokenize_dataset(val_ds, tokenizer)  # noqa: F841

    # Load model fresh for each adapter
    print("  Loading model...")
    # Try flash_attention_2, fall back to sdpa
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )

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
        warmup_ratio=0.05,
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

    # Record metadata
    meta = {
        "task": task,
        "seed": seed,
        "elapsed_s": elapsed,
        "train_samples": len(train_ds),
        "max_steps": max_steps,
        "smoke": smoke,
        "global_step": trainer.state.global_step,
        "train_loss": float(trainer.state.log_history[-1].get("train_loss", 0)) if trainer.state.log_history else None,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return output_dir


# ---------------------------------------------------------------------------
# Source evaluation (quick check that adapter learned something)
# ---------------------------------------------------------------------------

def evaluate_adapter(task: str, seed: int, tokenizer, smoke: bool = False):
    """Quick generative evaluation of a trained adapter."""
    adapter_name = f"{task}_s{seed}"
    adapter_dir = OUTPUT_ROOT / adapter_name
    eval_file = EVAL_ROOT / f"{adapter_name}_source_eval.json"

    if eval_file.exists():
        print(f"  SKIP eval {adapter_name} — already done")
        return json.loads(eval_file.read_text())

    print(f"\n  Evaluating {adapter_name}...")

    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    # Load eval data
    _, val_ds = load_task_dataset(task, smoke=smoke)
    n_eval = min(100, len(val_ds)) if not smoke else min(SMOKE_EVAL_SAMPLES, len(val_ds))

    correct = 0
    total = 0

    for i in range(n_eval):
        example = val_ds[i]
        text = example["text"]

        # Split into prompt and expected answer
        if task in ("sst2", "mnli"):
            # Last line is "Label: X" — use everything before as prompt
            lines = text.rsplit("\n", 1)
            prompt = lines[0] + "\n"
            expected = lines[1].strip() if len(lines) > 1 else ""
        elif task == "squad":
            parts = text.rsplit("Answer: ", 1)
            prompt = parts[0] + "Answer: "
            expected = parts[1].strip() if len(parts) > 1 else ""
        elif task == "gsm8k":
            parts = text.rsplit("Solution: ", 1)
            prompt = parts[0] + "Solution: "
            expected = parts[1].strip() if len(parts) > 1 else ""
        elif task == "code":
            parts = text.rsplit("Code:\n", 1)
            prompt = parts[0] + "Code:\n"
            expected = parts[1].strip() if len(parts) > 1 else ""
        elif task == "summarization":
            parts = text.rsplit("Summary: ", 1)
            prompt = parts[0] + "Summary: "
            expected = parts[1].strip() if len(parts) > 1 else ""
        else:
            continue

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50 if task in ("sst2", "mnli") else 100,
                do_sample=False,
                temperature=1.0,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

        # Simple match
        if task in ("sst2", "mnli"):
            # Check if the expected label word appears
            expected_label = expected.split(":")[-1].strip().lower()
            correct += int(expected_label in generated.lower())
        elif task == "gsm8k":
            # Extract final number
            import re
            expected_nums = re.findall(r"[\d,]+", expected.replace(",", ""))
            generated_nums = re.findall(r"[\d,]+", generated.replace(",", ""))
            if expected_nums and generated_nums:
                correct += int(expected_nums[-1] == generated_nums[-1])
        else:
            # For squad/code/summarization, just check non-empty generation
            correct += int(len(generated) > 5)

        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"    {adapter_name}: {correct}/{total} = {accuracy:.3f}")

    result = {
        "task": task, "seed": seed,
        "accuracy": accuracy, "correct": correct, "total": total,
    }
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_file, "w") as f:
        json.dump(result, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help="Single task to train")
    parser.add_argument("--seed", type=int, default=None, help="Single seed")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    seeds = [args.seed] if args.seed is not None else SEEDS

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not args.eval_only:
        print(f"\n{'='*60}")
        print(f"  N133 Phase 1: Training {len(tasks) * len(seeds)} adapters")
        print(f"  Tasks: {tasks}")
        print(f"  Seeds: {seeds}")
        print(f"  Mode: {'SMOKE' if args.smoke else 'FULL'}")
        print(f"{'='*60}")

        t0 = time.time()
        for task in tasks:
            for seed in seeds:
                train_adapter(task, seed, tokenizer, smoke=args.smoke)
        elapsed = time.time() - t0
        print(f"\n  Total training time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    # Quick source evaluation
    print(f"\n{'='*60}")
    print("  Source Evaluation")
    print(f"{'='*60}")
    for task in tasks:
        for seed in seeds:
            adapter_dir = OUTPUT_ROOT / f"{task}_s{seed}"
            if (adapter_dir / "adapter_model.safetensors").exists():
                evaluate_adapter(task, seed, tokenizer, smoke=args.smoke)


if __name__ == "__main__":
    main()
