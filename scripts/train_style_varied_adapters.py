"""Train 6 QNLI adapters on distilbert-base-uncased with varied training styles.

Adapter matrix:
  A: r8,  alpha=8,  dropout=0.0, full training (1000 steps)
  B: r8,  alpha=32, dropout=0.0, full training (1000 steps)
  C: r16, alpha=16, dropout=0.0, full training (1000 steps)
  D: r16, alpha=64, dropout=0.0, full training (1000 steps)
  E: r16, alpha=64, dropout=0.1, full training (1000 steps)
  F: r16, alpha=64, dropout=0.0, early checkpoint (500 steps)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_MODEL = "distilbert-base-uncased"
TASK = "qnli"
TRAIN_SAMPLES = 2000
EVAL_SAMPLES = 500
SEED = 42

OUT_ROOT = Path("results/training_style_blind_spot/sources")

ADAPTERS = [
    {"name": "A_r8_a8_d0_full",    "r": 8,  "alpha": 8,  "dropout": 0.0, "max_steps": 1000},
    {"name": "B_r8_a32_d0_full",   "r": 8,  "alpha": 32, "dropout": 0.0, "max_steps": 1000},
    {"name": "C_r16_a16_d0_full",  "r": 16, "alpha": 16, "dropout": 0.0, "max_steps": 1000},
    {"name": "D_r16_a64_d0_full",  "r": 16, "alpha": 64, "dropout": 0.0, "max_steps": 1000},
    {"name": "E_r16_a64_d1_full",  "r": 16, "alpha": 64, "dropout": 0.1, "max_steps": 1000},
    {"name": "F_r16_a64_d0_early", "r": 16, "alpha": 64, "dropout": 0.0, "max_steps": 500},
]


def train_one(cfg: dict) -> dict:
    import numpy as np
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    name = cfg["name"]
    out_dir = OUT_ROOT / name
    if (out_dir / "adapter_config.json").exists():
        print(f"  {name}: already exists, skipping training")
        # Load existing eval
        summary_path = OUT_ROOT / "training_summary.json"
        if summary_path.exists():
            summaries = json.load(open(summary_path))
            for s in summaries.get("adapters", []):
                if s["name"] == name:
                    return s
        return {"name": name, "skipped": True}

    print(f"  Training {name}: r={cfg['r']}, alpha={cfg['alpha']}, dropout={cfg['dropout']}, steps={cfg['max_steps']}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["classifier", "pre_classifier"],
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("glue", TASK)
    train_ds = dataset["train"].select(range(min(TRAIN_SAMPLES, len(dataset["train"]))))
    eval_ds = dataset["validation"].select(range(min(EVAL_SAMPLES, len(dataset["validation"]))))

    def preprocess(examples):
        result = tokenizer(examples["question"], examples["sentence"], truncation=True, padding=True, max_length=128)
        result["labels"] = examples["label"]
        return result

    train_tok = train_ds.map(preprocess, batched=True)
    eval_tok = eval_ds.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        logging_steps=100,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()

    # Evaluate
    predictions = trainer.predict(eval_tok)
    pred_classes = np.argmax(predictions.predictions, axis=1)
    eval_accuracy = float((pred_classes == predictions.label_ids).mean())

    # Base model accuracy
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    base_trainer = Trainer(
        model=base_model,
        args=TrainingArguments(output_dir="/tmp/base_eval", per_device_eval_batch_size=32, report_to=[]),
        eval_dataset=eval_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    base_preds = base_trainer.predict(eval_tok)
    base_acc = float((np.argmax(base_preds.predictions, axis=1) == base_preds.label_ids).mean())

    # Save adapter
    model.save_pretrained(str(out_dir))

    result = {
        "name": name,
        "r": cfg["r"],
        "alpha": cfg["alpha"],
        "dropout": cfg["dropout"],
        "max_steps": cfg["max_steps"],
        "eval_accuracy": eval_accuracy,
        "base_accuracy": base_acc,
        "margin": eval_accuracy - base_acc,
        "beats_base": eval_accuracy > base_acc,
        "verified": eval_accuracy > base_acc + 0.03,
        "adapter_dir": str(out_dir),
    }

    print(f"    accuracy={eval_accuracy:.3f}, base={base_acc:.3f}, margin={result['margin']:.3f}, verified={result['verified']}")
    return result


def main():
    results = []
    for cfg in ADAPTERS:
        result = train_one(cfg)
        results.append(result)

    # Save summary
    summary = {
        "base_model": BASE_MODEL,
        "task": TASK,
        "seed": SEED,
        "adapters": results,
        "verified_count": sum(1 for r in results if r.get("verified", False)),
    }

    summary_path = OUT_ROOT / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary: {summary['verified_count']}/{len(results)} verified")
    for r in results:
        v = "✓" if r.get("verified") else "✗"
        print(f"  {v} {r['name']}: acc={r.get('eval_accuracy', '?')}, margin={r.get('margin', '?')}")


if __name__ == "__main__":
    main()
