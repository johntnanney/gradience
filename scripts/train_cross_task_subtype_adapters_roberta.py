"""Train 8 roberta-base adapters for cross-task generalization study: 4 tasks x 2 seeds."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

BASE_MODEL = "roberta-base"
LORA_R = 16
LORA_ALPHA = 16
MAX_STEPS = 1000
TRAIN_SAMPLES = 2000

OUT_ROOT = Path("results/task_pair_severity_generalization_study_01/roberta/sources")

ADAPTERS = [
    {"name": "qnli_s42", "task": "qnli", "seed": 42},
    {"name": "qnli_s7",  "task": "qnli", "seed": 7},
    {"name": "rte_s42",  "task": "rte",  "seed": 42},
    {"name": "rte_s7",   "task": "rte",  "seed": 7},
    {"name": "mrpc_s42", "task": "mrpc", "seed": 42},
    {"name": "mrpc_s7",  "task": "mrpc", "seed": 7},
    {"name": "sst2_s42", "task": "sst2", "seed": 42},
    {"name": "sst2_s7",  "task": "sst2", "seed": 7},
]

TASK_CONFIG = {
    "qnli": {"subset": "qnli", "fields": ("question", "sentence"), "labels": 2, "eval_n": 500},
    "rte":  {"subset": "rte",  "fields": ("sentence1", "sentence2"), "labels": 2, "eval_n": 277},
    "mrpc": {"subset": "mrpc", "fields": ("sentence1", "sentence2"), "labels": 2, "eval_n": 408},
    "sst2": {"subset": "sst2", "fields": ("sentence",), "labels": 2, "eval_n": 500},
}

# RoBERTa target modules
TARGET_MODULES = ["query", "key", "value", "dense"]


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
    task = cfg["task"]
    tc = TASK_CONFIG[task]
    out_dir = OUT_ROOT / name

    if (out_dir / "adapter_config.json").exists():
        print(f"  {name}: already exists, skipping")
        summary_path = OUT_ROOT / "training_summary.json"
        if summary_path.exists():
            for s in json.load(open(summary_path)).get("adapters", []):
                if s["name"] == name:
                    return s
        return {"name": name, "skipped": True}

    print(f"  Training {name}: task={task}, seed={cfg['seed']}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=tc["labels"])

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        task_type=TaskType.SEQ_CLS, modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("glue", tc["subset"])
    train_ds = dataset["train"].select(range(min(TRAIN_SAMPLES, len(dataset["train"]))))
    eval_ds = dataset["validation"].select(range(min(tc["eval_n"], len(dataset["validation"]))))

    fields = tc["fields"]

    def preprocess(examples):
        if len(fields) == 2:
            result = tokenizer(examples[fields[0]], examples[fields[1]], truncation=True, padding=True, max_length=128)
        else:
            result = tokenizer(examples[fields[0]], truncation=True, padding=True, max_length=128)
        result["labels"] = examples["label"]
        return result

    train_tok = train_ds.map(preprocess, batched=True)
    eval_tok = eval_ds.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=8, per_device_eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="steps", eval_steps=200, save_strategy="no",
        logging_steps=200, seed=cfg["seed"], report_to=[],
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

    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=tc["labels"])
    base_trainer = Trainer(
        model=base_model,
        args=TrainingArguments(output_dir="/tmp/base_eval", per_device_eval_batch_size=32, report_to=[]),
        eval_dataset=eval_tok, data_collator=DataCollatorWithPadding(tokenizer),
    )
    base_preds = base_trainer.predict(eval_tok)
    base_acc = float((np.argmax(base_preds.predictions, axis=1) == base_preds.label_ids).mean())

    model.save_pretrained(str(out_dir))

    result = {
        "name": name, "task": task, "seed": cfg["seed"],
        "backbone": BASE_MODEL,
        "eval_accuracy": eval_accuracy, "base_accuracy": base_acc,
        "margin": eval_accuracy - base_acc,
        "beats_base": eval_accuracy > base_acc,
        "verified": eval_accuracy > base_acc + 0.03,
        "eval_dataset": f"{task}_dev",
        "adapter_dir": str(out_dir),
    }
    print(f"    accuracy={eval_accuracy:.3f}, base={base_acc:.3f}, margin={result['margin']:.3f}")
    return result


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in ADAPTERS:
        results.append(train_one(cfg))

    summary = {
        "base_model": BASE_MODEL, "adapters": results,
        "verified_count": sum(1 for r in results if r.get("verified", False)),
    }
    json.dump(summary, open(OUT_ROOT / "training_summary.json", "w"), indent=2)

    print(f"\nSummary: {summary['verified_count']}/{len(results)} verified")
    for r in results:
        v = "Y" if r.get("verified") else "N"
        print(f"  [{v}] {r['name']}: acc={r.get('eval_accuracy', '?')}, task={r.get('task', '?')}")


if __name__ == "__main__":
    main()
