"""N134 Phase 0: Single-seed pilot training for 8 candidate tasks.

Trains one LoRA adapter per task on Mistral-7B-v0.3 at seed=42, then
evaluates on the held-out validation set to measure post-fine-tune
accuracy.  The pilot determines which tasks satisfy the C1 dynamic-range
constraint ([0.70, 0.90]) before the full N134 task set is locked.

Tasks:
    1. ARC-Challenge   (allenai/ai2_arc, ARC-Challenge)
    2. HellaSwag        (Rowan/hellaswag, 5k subsample)
    3. WinoGrande       (winogrande, winogrande_xl)
    4. OpenBookQA        (openbookqa, main)
    5. CommonsenseQA     (tau/commonsense_qa)
    6. PIQA              (piqa)
    7. SIQA              (social_i_qa)
    8. BoolQ             (boolq)

LoRA config: r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj],
dropout=0.05.  Training: AdamW lr=2e-4, warmup=6%, bf16, cosine schedule.

Usage:
    python3 scripts/n134/00_pilot_train.py                  # train + eval all 8
    python3 scripts/n134/00_pilot_train.py --task arc       # single task
    python3 scripts/n134/00_pilot_train.py --smoke          # quick smoke test
    python3 scripts/n134/00_pilot_train.py --eval-only      # skip training

Output: /workspace/n134/pilot/{task}_s42/
"""

from __future__ import annotations

import argparse
import gc
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
OUTPUT_ROOT = Path("/workspace/n134/pilot")
EVAL_ROOT = Path("/workspace/n134/pilot/evals")

SEED = 42

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "task_type": TaskType.CAUSAL_LM,
}

TASKS = ["arc", "hellaswag", "winogrande", "openbookqa", "commonsenseqa", "piqa", "siqa", "boolq"]

MAX_SEQ_LEN = 512
MAX_TRAIN_SAMPLES = 5000
MAX_EVAL_SAMPLES = 500
HELLASWAG_SUBSAMPLE = 5000

SMOKE_MAX_STEPS = 20
SMOKE_TRAIN_SAMPLES = 50
SMOKE_EVAL_SAMPLES = 20


# ---------------------------------------------------------------------------
# Dataset loading and formatting
# ---------------------------------------------------------------------------

CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def format_arc(example):
    """ARC-Challenge: multiple-choice science questions."""
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    options = "\n".join(f"{lab}) {txt}" for lab, txt in zip(labels, texts))
    answer_key = example["answerKey"]
    text = f"Question: {example['question']}\n{options}\nAnswer: {answer_key}"
    return {"text": text}


def format_hellaswag(example):
    """HellaSwag: sentence completion."""
    ctx = example["ctx"]
    endings = example["endings"]
    options = "\n".join(f"{CHOICE_LABELS[i]}) {end}" for i, end in enumerate(endings))
    correct_idx = int(example["label"])
    answer = CHOICE_LABELS[correct_idx]
    text = f"Complete the sentence.\n{ctx}\n{options}\nAnswer: {answer}"
    return {"text": text}


def format_winogrande(example):
    """WinoGrande: pronoun resolution as binary choice."""
    sentence = example["sentence"]
    option1 = example["option1"]
    option2 = example["option2"]
    answer = example["answer"]  # "1" or "2"
    text = (
        f"Fill in the blank with option 1 or option 2.\n"
        f"{sentence}\n"
        f"1) {option1}\n"
        f"2) {option2}\n"
        f"Answer: {answer}"
    )
    return {"text": text}


def format_openbookqa(example):
    """OpenBookQA: multiple-choice science questions with fact support."""
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    options = "\n".join(f"{lab}) {txt}" for lab, txt in zip(labels, texts))
    answer_key = example["answerKey"]
    text = f"Question: {example['question_stem']}\n{options}\nAnswer: {answer_key}"
    return {"text": text}


def format_commonsenseqa(example):
    """CommonsenseQA: 5-way multiple-choice commonsense questions."""
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]
    options = "\n".join(f"{lab}) {txt}" for lab, txt in zip(labels, texts))
    answer_key = example["answerKey"]
    text = f"Question: {example['question']}\n{options}\nAnswer: {answer_key}"
    return {"text": text}


def format_piqa(example):
    """PIQA: physical intuition, binary choice."""
    goal = example["goal"]
    sol1 = example["sol1"]
    sol2 = example["sol2"]
    correct = CHOICE_LABELS[int(example["label"])]
    text = (
        f"Which solution is better?\n"
        f"Goal: {goal}\n"
        f"A) {sol1}\n"
        f"B) {sol2}\n"
        f"Answer: {correct}"
    )
    return {"text": text}


def format_siqa(example):
    """SIQA: social interaction QA, 3-way choice."""
    context = example["context"]
    question = example["question"]
    answer_a = example["answerA"]
    answer_b = example["answerB"]
    answer_c = example["answerC"]
    correct_idx = int(example["label"]) - 1  # 1-indexed in dataset
    correct = CHOICE_LABELS[correct_idx]
    text = (
        f"What happens next?\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"A) {answer_a}\n"
        f"B) {answer_b}\n"
        f"C) {answer_c}\n"
        f"Answer: {correct}"
    )
    return {"text": text}


def format_boolq(example):
    """BoolQ: yes/no reading comprehension."""
    passage = example["passage"]
    question = example["question"]
    answer = "yes" if example["answer"] else "no"
    text = (
        f"Answer yes or no.\n"
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    )
    return {"text": text}


FORMATTERS = {
    "arc": format_arc,
    "hellaswag": format_hellaswag,
    "winogrande": format_winogrande,
    "openbookqa": format_openbookqa,
    "commonsenseqa": format_commonsenseqa,
    "piqa": format_piqa,
    "siqa": format_siqa,
    "boolq": format_boolq,
}


def load_task_dataset(task: str, smoke: bool = False):
    """Load and format dataset for a given task.

    Returns (train_split, val_split) after formatting.
    """
    max_train = SMOKE_TRAIN_SAMPLES if smoke else MAX_TRAIN_SAMPLES
    max_eval = SMOKE_EVAL_SAMPLES if smoke else MAX_EVAL_SAMPLES

    if task == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=CACHE_DIR)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))

    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", cache_dir=CACHE_DIR)
        # Seeded 5k subsample from training set
        full_train = ds["train"].shuffle(seed=SEED)
        n_sub = min(HELLASWAG_SUBSAMPLE, len(full_train))
        subsample = full_train.select(range(n_sub))
        train = subsample.select(range(min(max_train, n_sub)))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))

    elif task == "winogrande":
        ds = load_dataset("winogrande", "winogrande_xl", cache_dir=CACHE_DIR)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))

    elif task == "openbookqa":
        ds = load_dataset("openbookqa", "main", cache_dir=CACHE_DIR)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))

    elif task == "commonsenseqa":
        ds = load_dataset("tau/commonsense_qa", cache_dir=CACHE_DIR)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))

    elif task == "piqa":
        # piqa, social_i_qa, boolq have custom loader scripts; newer
        # datasets library requires explicit consent via trust_remote_code
        ds = load_dataset("piqa", cache_dir=CACHE_DIR, trust_remote_code=True)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))

    elif task == "siqa":
        ds = load_dataset("social_i_qa", cache_dir=CACHE_DIR, trust_remote_code=True)
        train = ds["train"].select(range(min(max_train, len(ds["train"]))))
        val = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))

    elif task == "boolq":
        ds = load_dataset("boolq", cache_dir=CACHE_DIR, trust_remote_code=True)
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
# Answer extraction helpers
# ---------------------------------------------------------------------------

def _extract_expected_answer(task: str, text: str) -> tuple[str, str]:
    """Split formatted text into (prompt, expected_answer).

    Returns the prompt (everything up to and including 'Answer: ') and the
    expected answer string.
    """
    parts = text.rsplit("Answer: ", 1)
    prompt = parts[0] + "Answer: "
    expected = parts[1].strip() if len(parts) > 1 else ""
    return prompt, expected


def _check_answer(task: str, generated: str, expected: str) -> bool:
    """Check if the generated answer matches the expected answer."""
    gen_clean = generated.strip().split("\n")[0].strip()
    exp_clean = expected.strip()

    if task == "boolq":
        # Match yes/no
        gen_lower = gen_clean.lower()
        return exp_clean.lower() in gen_lower[:10]
    elif task == "winogrande":
        # Match 1 or 2
        return exp_clean in gen_clean[:5]
    else:
        # MC tasks: match the choice label (single character)
        return exp_clean.upper() in gen_clean[:5].upper()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_adapter(task: str, tokenizer, smoke: bool = False):
    """Train one LoRA adapter for the given task at seed=42."""
    adapter_name = f"{task}_s{SEED}"
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
    _val_tok = tokenize_dataset(val_ds, tokenizer)  # noqa: F841

    # Load model fresh for each adapter.
    #
    # Avoid device_map="auto" — accelerate's auto-placement uses a VRAM
    # heuristic that can see fragmented residual allocations from prior
    # tasks and decide to offload some layers to CPU/meta. This produces
    # a "Some parameters are on the meta device because they were
    # offloaded to the cpu" warning and causes Trainer(...) to fail at
    # `model.to(device)` with "Cannot copy out of meta tensor; no data!".
    # Instead, load into CPU memory (default) then move explicitly to
    # GPU — a 7B bf16 model is ~14 GB, comfortable in 48 GB VRAM.
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
        seed=SEED,
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
        "seed": SEED,
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

    # Clean up GPU memory. gc.collect() before empty_cache() ensures any
    # Python-side references are actually dropped before CUDA releases
    # the backing memory; without it, fragmentation can accumulate across
    # tasks and trigger accelerate to offload later loads to CPU/meta.
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# ---------------------------------------------------------------------------
# Source evaluation
# ---------------------------------------------------------------------------

def evaluate_adapter(task: str, tokenizer, smoke: bool = False) -> dict:
    """Evaluate a trained adapter on the held-out validation set.

    Measures accuracy by generating the answer token(s) and comparing
    to the expected answer from the formatted text.
    """
    adapter_name = f"{task}_s{SEED}"
    adapter_dir = OUTPUT_ROOT / adapter_name
    eval_file = EVAL_ROOT / f"{adapter_name}_source_eval.json"

    if eval_file.exists():
        print(f"  SKIP eval {adapter_name} -- already done")
        return json.loads(eval_file.read_text())

    print(f"\n  Evaluating {adapter_name}...")

    from peft import PeftModel

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
    n_eval = min(SMOKE_EVAL_SAMPLES, len(val_ds)) if smoke else min(MAX_EVAL_SAMPLES, len(val_ds))

    correct = 0
    total = 0

    for i in range(n_eval):
        example = val_ds[i]
        text = example["text"]
        prompt, expected = _extract_expected_answer(task, text)

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

        if _check_answer(task, generated, expected):
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"    {adapter_name}: {correct}/{total} = {accuracy:.3f}")

    result = {
        "task": task,
        "seed": SEED,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "smoke": smoke,
    }

    eval_file.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_file, "w") as f:
        json.dump(result, f, indent=2)

    # Save into adapter directory as well
    with open(adapter_dir / "training_meta.json") as f:
        meta = json.load(f)
    meta["final_val_accuracy"] = accuracy
    with open(adapter_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="N134 Phase 0: Pilot training for 8 candidate tasks")
    parser.add_argument("--task", type=str, default=None, help="Single task to train (default: all)")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode (20 steps, 50 samples)")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation, skip training")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not args.eval_only:
        print(f"\n{'=' * 60}")
        print(f"  N134 Phase 0: Pilot Training ({len(tasks)} tasks, seed={SEED})")
        print(f"  Tasks: {tasks}")
        print(f"  Mode: {'SMOKE' if args.smoke else 'FULL'}")
        print(f"{'=' * 60}")

        t0 = time.time()
        for task in tasks:
            train_adapter(task, tokenizer, smoke=args.smoke)
        elapsed = time.time() - t0
        print(f"\n  Total training time: {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

    # Source evaluation
    print(f"\n{'=' * 60}")
    print("  Source Evaluation")
    print(f"{'=' * 60}")

    results = []
    for task in tasks:
        adapter_dir = OUTPUT_ROOT / f"{task}_s{SEED}"
        if (adapter_dir / "adapter_model.safetensors").exists():
            result = evaluate_adapter(task, tokenizer, smoke=args.smoke)
            results.append(result)

    # Print summary table
    if results:
        print(f"\n{'=' * 60}")
        print("  Pilot Accuracy Summary")
        print(f"{'=' * 60}")
        print(f"  {'Task':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
        print(f"  {'-' * 50}")
        for r in results:
            print(f"  {r['task']:<20} {r['accuracy']:>10.3f} {r['correct']:>10} {r['total']:>10}")


if __name__ == "__main__":
    main()
