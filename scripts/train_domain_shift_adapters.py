"""Train 6 binary sentiment adapters across 3 domains on distilbert-base-uncased.

Matrix: 3 domains x 2 seeds
  - SST-2 (movie reviews)
  - Yelp polarity (restaurant/business reviews)
  - Amazon polarity (product reviews)

All binary sentiment. Same task, different domain.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

BASE_MODEL = "distilbert-base-uncased"
LORA_R = 16
LORA_ALPHA = 16
TRAIN_SAMPLES = 2000
EVAL_SAMPLES = 500
MAX_STEPS = 1000

OUT_ROOT = Path("results/domain_shift_blind_spot/sources")

ADAPTERS = [
    {"name": "sst2_s42",   "domain": "sst2",   "seed": 42},
    {"name": "sst2_s7",    "domain": "sst2",   "seed": 7},
    {"name": "yelp_s42",   "domain": "yelp",   "seed": 42},
    {"name": "yelp_s7",    "domain": "yelp",   "seed": 7},
    {"name": "amazon_s42", "domain": "amazon", "seed": 42},
    {"name": "amazon_s7",  "domain": "amazon", "seed": 7},
]


def load_domain_data(domain: str, split: str = "train", max_samples: int | None = None):
    """Load binary sentiment data from a given domain."""
    from datasets import load_dataset

    if domain == "sst2":
        ds = load_dataset("glue", "sst2", split="validation" if split == "eval" else split)
        # SST-2: columns = sentence, label
        ds = ds.rename_column("sentence", "text")
    elif domain == "yelp":
        split_name = "test" if split == "eval" else "train"
        ds = load_dataset("yelp_polarity", split=split_name)
        # already has "text" column
    elif domain == "amazon":
        split_name = "test" if split == "eval" else "train"
        ds = load_dataset("amazon_polarity", split=split_name)
        ds = ds.rename_column("content", "text")
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Keep only text and label
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("text", "label")])

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    return ds


def train_one(cfg: dict) -> dict:
    import torch
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
            summaries = json.load(open(summary_path))
            for s in summaries.get("adapters", []):
                if s["name"] == name:
                    return s
        return {"name": name, "skipped": True}

    print(f"  Training {name}: domain={cfg['domain']}, seed={cfg['seed']}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["classifier", "pre_classifier"],
    )
    model = get_peft_model(model, lora_config)

    train_ds = load_domain_data(cfg["domain"], "train", TRAIN_SAMPLES)
    eval_ds = load_domain_data(cfg["domain"], "eval", EVAL_SAMPLES)

    def preprocess(examples):
        result = tokenizer(examples["text"], truncation=True, padding=True, max_length=128)
        result["labels"] = examples["label"]
        return result

    train_tok = train_ds.map(preprocess, batched=True)
    eval_tok = eval_ds.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="no",
        logging_steps=200,
        seed=cfg["seed"],
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

    # Evaluate on own domain
    predictions = trainer.predict(eval_tok)
    pred_classes = np.argmax(predictions.predictions, axis=1)
    own_accuracy = float((pred_classes == predictions.label_ids).mean())

    # Base model accuracy on own domain
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

    # Cross-domain evaluation
    cross_domain_scores = {}
    all_domains = ["sst2", "yelp", "amazon"]
    from peft import PeftModel

    for d in all_domains:
        if d == cfg["domain"]:
            cross_domain_scores[d] = own_accuracy
            continue
        cross_eval = load_domain_data(d, "eval", EVAL_SAMPLES)
        cross_tok = cross_eval.map(preprocess, batched=True)

        # Reload for cross-domain eval
        fresh_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
        fresh_model = PeftModel.from_pretrained(fresh_model, str(out_dir))
        fresh_model = fresh_model.merge_and_unload()
        fresh_model.eval()

        cross_trainer = Trainer(
            model=fresh_model,
            args=TrainingArguments(output_dir="/tmp/cross_eval", per_device_eval_batch_size=32, report_to=[]),
            eval_dataset=cross_tok,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        cross_preds = cross_trainer.predict(cross_tok)
        cross_acc = float((np.argmax(cross_preds.predictions, axis=1) == cross_preds.label_ids).mean())
        cross_domain_scores[d] = cross_acc

    result = {
        "name": name,
        "domain": cfg["domain"],
        "seed": cfg["seed"],
        "own_accuracy": own_accuracy,
        "base_accuracy": base_acc,
        "margin": own_accuracy - base_acc,
        "beats_base": own_accuracy > base_acc,
        "verified": own_accuracy > base_acc + 0.03,
        "cross_domain_scores": cross_domain_scores,
        "adapter_dir": str(out_dir),
    }

    print(f"    own={own_accuracy:.3f}, base={base_acc:.3f}, margin={result['margin']:.3f}")
    print(f"    cross-domain: {cross_domain_scores}")
    return result


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in ADAPTERS:
        result = train_one(cfg)
        results.append(result)

    summary = {
        "base_model": BASE_MODEL,
        "task": "binary_sentiment",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "adapters": results,
        "verified_count": sum(1 for r in results if r.get("verified", False)),
    }

    summary_path = OUT_ROOT / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary: {summary['verified_count']}/{len(results)} verified")
    for r in results:
        v = "Y" if r.get("verified") else "N"
        print(f"  [{v}] {r['name']}: own={r.get('own_accuracy', '?'):.3f}, margin={r.get('margin', '?'):.3f}")


if __name__ == "__main__":
    main()
