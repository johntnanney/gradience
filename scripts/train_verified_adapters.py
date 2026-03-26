"""Train 6 verified LoRA adapters for the ambiguous-relationship adjudication regime.

Adapter design — 6 adapters across 3 NLI-family tasks, 2 seeds each:

  QNLI (2-class)      RTE (2-class)        MNLI (3-class)
  ──────────────       ──────────────       ──────────────
  qnli_s42: seed=42   rte_s42: seed=42     mnli_s42: seed=42
  qnli_s7:  seed=7    rte_s7:  seed=7      mnli_s7:  seed=7

This gives us:
 - Same-task seed pairs (qnli_s42 x qnli_s7, etc.): controls
 - Adjacent NLI-family pairs (qnli x rte, qnli x mnli, rte x mnli): ambiguous middle
 - 15 total pairs, spanning same-task through adjacent-task relationships

Each adapter is verified to outperform base model on its training task by >= 3pp.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

OUTPUT_ROOT = Path("results/adjudication_study/ambiguous_relationship_regime/adapters")
BASE_MODEL = "distilbert-base-uncased"
MAX_STEPS = 1000
TRAIN_SAMPLES = 2000
LR = 5e-5

ADAPTERS = [
    {"name": "qnli_s42", "task": "qnli", "seed": 42},
    {"name": "qnli_s7", "task": "qnli", "seed": 7},
    {"name": "rte_s42", "task": "rte", "seed": 42},
    {"name": "rte_s7", "task": "rte", "seed": 7},
    {"name": "mnli_s42", "task": "mnli", "seed": 42},
    {"name": "mnli_s7", "task": "mnli", "seed": 7},
]

TASK_META = {
    "qnli": {"num_labels": 2, "field_a": "question", "field_b": "sentence", "val_split": "validation", "eval_n": 500},
    "rte": {"num_labels": 2, "field_a": "sentence1", "field_b": "sentence2", "val_split": "validation", "eval_n": 277},
    "mnli": {"num_labels": 3, "field_a": "premise", "field_b": "hypothesis", "val_split": "validation_matched", "eval_n": 500},
}


def train_one_adapter(spec: dict) -> dict:
    """Train a single adapter and return verified eval results."""
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

    name = spec["name"]
    task = spec["task"]
    seed = spec["seed"]
    meta = TASK_META[task]

    print(f"\n{'='*60}")
    print(f"Training: {name}  (task={task}, seed={seed})")
    print(f"{'='*60}")

    adapter_dir = OUTPUT_ROOT / name
    manifest_path = adapter_dir / "training_manifest.json"

    # Skip if already trained and verified
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("verified", False):
            print(f"  Already trained and verified (accuracy={manifest['eval_accuracy']:.3f}). Skipping.")
            return manifest

    set_seed(seed)

    # Load dataset
    dataset = load_dataset("glue", task)
    train_ds = dataset["train"].select(range(min(TRAIN_SAMPLES, len(dataset["train"]))))
    val_ds = dataset[meta["val_split"]].select(range(min(meta["eval_n"], len(dataset[meta["val_split"]]))))

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=meta["num_labels"]
    )

    # Base accuracy
    def preprocess(examples):
        result = tokenizer(
            examples[meta["field_a"]], examples[meta["field_b"]],
            truncation=True, padding=True, max_length=128,
        )
        result["labels"] = examples["label"]
        return result

    val_tok = val_ds.map(preprocess, batched=True)
    train_tok = train_ds.map(preprocess, batched=True)

    base_trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/base_eval", per_device_eval_batch_size=32, report_to=[]),
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    base_preds = base_trainer.predict(val_tok)
    base_acc = float((np.argmax(base_preds.predictions, axis=1) == base_preds.label_ids).mean())
    print(f"  Base accuracy: {base_acc:.3f}")

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Train
    adapter_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "checkpoints"),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="no",
        logging_steps=200,
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()

    # Final eval
    predictions = trainer.predict(val_tok)
    pred_classes = np.argmax(predictions.predictions, axis=1)
    accuracy = float((pred_classes == predictions.label_ids).mean())
    margin = accuracy - base_acc
    verified = margin >= 0.03
    print(f"  Adapter accuracy: {accuracy:.3f}")
    print(f"  Margin: {margin:+.3f}  {'PASS' if verified else 'FAIL'}")

    # Save adapter
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"  Saved to {adapter_dir}")

    manifest = {
        "name": name,
        "task": task,
        "seed": seed,
        "base_model": BASE_MODEL,
        "num_labels": meta["num_labels"],
        "lora_r": 16,
        "lora_alpha": 16,
        "max_steps": MAX_STEPS,
        "train_samples": TRAIN_SAMPLES,
        "eval_samples": len(val_tok),
        "eval_accuracy": accuracy,
        "base_accuracy": base_acc,
        "margin": margin,
        "verified": verified,
        "adapter_dir": str(adapter_dir),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest


def main():
    print("=" * 60)
    print("AMBIGUOUS RELATIONSHIP REGIME — ADAPTER TRAINING")
    print(f"Base model: {BASE_MODEL}")
    print(f"Tasks: QNLI, RTE, MNLI (NLI family)")
    print(f"Output: {OUTPUT_ROOT}")
    print("=" * 60)

    results = []
    for spec in ADAPTERS:
        result = train_one_adapter(spec)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Name':<15s} {'Task':<6s} {'Acc':<8s} {'Base':<8s} {'Margin':<8s} {'Pass'}")
    print(f"{'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    for r in results:
        print(
            f"{r['name']:<15s} {r['task']:<6s} "
            f"{r['eval_accuracy']:.3f}   {r['base_accuracy']:.3f}   "
            f"{r['margin']:+.3f}   {'YES' if r['verified'] else 'NO'}"
        )

    verified_count = sum(1 for r in results if r["verified"])
    print(f"\nVerified: {verified_count}/{len(results)}")

    summary = {"base_model": BASE_MODEL, "adapters": results, "verified_count": verified_count}
    summary_path = OUTPUT_ROOT / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
