"""Train 6 QNLI adapters on distilbert-base-uncased with intentional strength spread.

Strength bands:
  Strong (2):  full data (2000), full training (1000 steps)
  Medium (2):  reduced duration (500 steps)
  Weak-but-credible (2): reduced data (500 samples), reduced duration (500 steps)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

BASE_MODEL = "distilbert-base-uncased"
EVAL_SAMPLES = 500

OUT_ROOT = Path("results/source_strength_blind_spot/sources")

ADAPTERS = [
    # Strong band: full data, full training
    {"name": "strong_s42", "band": "strong", "seed": 42, "train_samples": 2000, "max_steps": 1000,
     "r": 16, "alpha": 16, "dropout": 0.0},
    {"name": "strong_s7", "band": "strong", "seed": 7, "train_samples": 2000, "max_steps": 1000,
     "r": 16, "alpha": 16, "dropout": 0.0},
    # Medium band: full data, reduced training
    {"name": "medium_s42", "band": "medium", "seed": 42, "train_samples": 2000, "max_steps": 500,
     "r": 16, "alpha": 16, "dropout": 0.0},
    {"name": "medium_s7", "band": "medium", "seed": 7, "train_samples": 2000, "max_steps": 500,
     "r": 16, "alpha": 16, "dropout": 0.0},
    # Weak-but-credible band: reduced data, reduced training
    {"name": "weak_s42", "band": "weak", "seed": 42, "train_samples": 500, "max_steps": 500,
     "r": 16, "alpha": 16, "dropout": 0.0},
    {"name": "weak_s7", "band": "weak", "seed": 7, "train_samples": 500, "max_steps": 500,
     "r": 16, "alpha": 16, "dropout": 0.0},
]


def train_one(cfg: dict) -> dict:
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
        print(f"  {name}: already exists, skipping")
        summary_path = OUT_ROOT / "training_summary.json"
        if summary_path.exists():
            for s in json.load(open(summary_path)).get("adapters", []):
                if s["name"] == name:
                    return s
        return {"name": name, "skipped": True}

    print(f"  Training {name}: band={cfg['band']}, data={cfg['train_samples']}, steps={cfg['max_steps']}, seed={cfg['seed']}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    lora_config = LoraConfig(
        r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        task_type=TaskType.SEQ_CLS, modules_to_save=["classifier", "pre_classifier"],
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("glue", "qnli")
    train_ds = dataset["train"].select(range(min(cfg["train_samples"], len(dataset["train"]))))
    eval_ds = dataset["validation"].select(range(EVAL_SAMPLES))

    def preprocess(examples):
        result = tokenizer(examples["question"], examples["sentence"], truncation=True, padding=True, max_length=128)
        result["labels"] = examples["label"]
        return result

    train_tok = train_ds.map(preprocess, batched=True)
    eval_tok = eval_ds.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=8, per_device_eval_batch_size=32,
        learning_rate=5e-5, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="steps", eval_steps=100, save_strategy="no",
        logging_steps=100, seed=cfg["seed"], report_to=[],
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_tok, eval_dataset=eval_tok,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    predictions = trainer.predict(eval_tok)
    pred_classes = np.argmax(predictions.predictions, axis=1)
    eval_accuracy = float((pred_classes == predictions.label_ids).mean())

    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    base_trainer = Trainer(
        model=base_model,
        args=TrainingArguments(output_dir="/tmp/base_eval", per_device_eval_batch_size=32, report_to=[]),
        eval_dataset=eval_tok, data_collator=DataCollatorWithPadding(tokenizer),
    )
    base_preds = base_trainer.predict(eval_tok)
    base_acc = float((np.argmax(base_preds.predictions, axis=1) == base_preds.label_ids).mean())

    model.save_pretrained(str(out_dir))

    result = {
        "name": name, "band": cfg["band"], "seed": cfg["seed"],
        "train_samples": cfg["train_samples"], "max_steps": cfg["max_steps"],
        "eval_accuracy": eval_accuracy, "base_accuracy": base_acc,
        "margin": eval_accuracy - base_acc,
        "beats_base": eval_accuracy > base_acc,
        "adapter_dir": str(out_dir),
    }
    print(f"    accuracy={eval_accuracy:.3f}, base={base_acc:.3f}, margin={result['margin']:.3f}, band={cfg['band']}")
    return result


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in ADAPTERS:
        results.append(train_one(cfg))

    summary = {
        "base_model": BASE_MODEL, "task": "qnli", "adapters": results,
        "verified_count": sum(1 for r in results if r.get("beats_base", False)),
    }
    json.dump(summary, open(OUT_ROOT / "training_summary.json", "w"), indent=2)

    print(f"\nSummary: {summary['verified_count']}/{len(results)} beat base")
    for r in results:
        v = "Y" if r.get("beats_base") else "N"
        print(f"  [{v}] {r['name']}: acc={r.get('eval_accuracy', '?')}, band={r.get('band', '?')}")


if __name__ == "__main__":
    main()
