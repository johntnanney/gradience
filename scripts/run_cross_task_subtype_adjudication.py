"""Cross-task subtype adjudication: merge all 28 pairs, evaluate on both tasks."""
from __future__ import annotations

import itertools
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

BASE_MODEL = "distilbert-base-uncased"
SRC = Path("results/cross_task_subtype_study_01/sources")
QA = Path("results/cross_task_subtype_study_01/qa")
PAIRS = Path("results/cross_task_subtype_study_01/pairs")

TASK_CONFIG = {
    "qnli": {"subset": "qnli", "fields": ("question", "sentence"), "labels": 2, "eval_n": 500},
    "rte":  {"subset": "rte",  "fields": ("sentence1", "sentence2"), "labels": 2, "eval_n": 277},
    "mrpc": {"subset": "mrpc", "fields": ("sentence1", "sentence2"), "labels": 2, "eval_n": 408},
    "sst2": {"subset": "sst2", "fields": ("sentence",), "labels": 2, "eval_n": 500},
}

_DISTANCE_RAW = {
    "qnli_rte": "related", "qnli_mrpc": "related", "mrpc_rte": "related",
    "qnli_sst2": "distant", "rte_sst2": "distant", "mrpc_sst2": "distant",
}
DISTANCE_MAP = {}
for k, v in _DISTANCE_RAW.items():
    a, b = k.split("_")
    DISTANCE_MAP[(a, b)] = v
    DISTANCE_MAP[(b, a)] = v


def get_distance(task_a, task_b):
    if task_a == task_b:
        return "same_task"
    return DISTANCE_MAP.get((task_a, task_b), "unknown")


def evaluate_on_task(model_or_dir, task, tokenizer=None):
    """Evaluate a model/adapter on a specific task."""
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tc = TASK_CONFIG[task]

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if isinstance(model_or_dir, (str, Path)):
        adapter_dir = str(model_or_dir)
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=tc["labels"])
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    else:
        model = model_or_dir

    model.eval()

    ds = load_dataset("glue", tc["subset"], split="validation")
    ds = ds.select(range(min(tc["eval_n"], len(ds))))

    fields = tc["fields"]

    def preprocess(examples):
        if len(fields) == 2:
            result = tokenizer(examples[fields[0]], examples[fields[1]], truncation=True, padding=True, max_length=128)
        else:
            result = tokenizer(examples[fields[0]], truncation=True, padding=True, max_length=128)
        result["labels"] = examples["label"]
        return result

    tok_ds = ds.map(preprocess, batched=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/eval", per_device_eval_batch_size=32, report_to=[]),
        eval_dataset=tok_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    preds = trainer.predict(tok_ds)
    return float((np.argmax(preds.predictions, axis=1) == preds.label_ids).mean())


def patch_classifier(merged_dir, source_a, source_b):
    """Average classifier heads from both sources into merged adapter."""
    from safetensors.torch import load_file, save_file

    merged_path = Path(merged_dir) / "adapter_model.safetensors"
    merged_st = load_file(str(merged_path))
    if "base_model.model.classifier.weight" in merged_st:
        return

    st_a = load_file(str(Path(source_a) / "adapter_model.safetensors"))
    st_b = load_file(str(Path(source_b) / "adapter_model.safetensors"))

    for key in ["base_model.model.classifier.weight", "base_model.model.classifier.bias",
                 "base_model.model.pre_classifier.weight", "base_model.model.pre_classifier.bias"]:
        if key in st_a and key in st_b:
            merged_st[key] = (st_a[key] + st_b[key]) / 2.0
        elif key in st_a:
            merged_st[key] = st_a[key]
        elif key in st_b:
            merged_st[key] = st_b[key]

    save_file(merged_st, str(merged_path))


def main():
    from gradience.vnext.merge import execute_merge, merge_audit, plan_from_audit
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    summary = json.load(open(SRC / "training_summary.json"))
    adapters = {a["name"]: a for a in summary["adapters"]}
    names = list(adapters.keys())

    results = []

    for a_name, b_name in itertools.combinations(names, 2):
        a = adapters[a_name]
        b = adapters[b_name]
        task_a = a["task"]
        task_b = b["task"]
        distance = get_distance(task_a, task_b)

        pair_id = f"{a_name}_vs_{b_name}"
        pair_dir = PAIRS / pair_id
        eval_dir = pair_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_file = eval_dir / "eval_results.json"
        if eval_file.exists():
            print(f"  {pair_id}: exists")
            result = json.load(open(eval_file))
            results.append(result)
            continue

        print(f"\n  {pair_id} ({distance})")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                merge_dir = Path(tmpdir) / "merged"
                audit_dir = Path(tmpdir) / "audit"

                report = merge_audit(a["adapter_dir"], b["adapter_dir"], output_dir=str(audit_dir))
                plan = plan_from_audit("uniform_linear", report, a["adapter_dir"], b["adapter_dir"])
                merge_result = execute_merge(plan, str(merge_dir))
                patch_classifier(str(merge_dir), a["adapter_dir"], b["adapter_dir"])

                print(f"    Merged (recon_err={merge_result.mean_reconstruction_error:.3f})")

                # Evaluate on task A
                merged_on_a = evaluate_on_task(str(merge_dir), task_a, tokenizer)
                print(f"    On {task_a}: {merged_on_a:.3f}")

                # Evaluate on task B (only if different)
                if task_a != task_b:
                    merged_on_b = evaluate_on_task(str(merge_dir), task_b, tokenizer)
                    print(f"    On {task_b}: {merged_on_b:.3f}")
                else:
                    merged_on_b = merged_on_a

                # Load pair report for structural data
                report_file = pair_dir / "report" / "merge_qa_report.json"
                pr = json.load(open(report_file)) if report_file.exists() else {}

                result = {
                    "pair_id": pair_id,
                    "source_a": a_name, "source_b": b_name,
                    "task_a": task_a, "task_b": task_b,
                    "distance": distance,
                    "source_a_score": a["eval_accuracy"],
                    "source_b_score": b["eval_accuracy"],
                    "merged_on_task_a": merged_on_a,
                    "merged_on_task_b": merged_on_b,
                    "best_source_on_a": a["eval_accuracy"],
                    "best_source_on_b": b["eval_accuracy"],
                    "delta_task_a": a["eval_accuracy"] - merged_on_a,
                    "delta_task_b": b["eval_accuracy"] - merged_on_b,
                    "pair_risk": pr.get("pair_risk"),
                    "dominant_issue": pr.get("dominant_issue"),
                    "advisory": pr.get("task_relationship_advisory") is not None,
                    "recon_error": merge_result.mean_reconstruction_error,
                }
                results.append(result)
                json.dump(result, open(eval_file, "w"), indent=2)

        except Exception as e:
            print(f"    ERROR: {e}")
            result = {"pair_id": pair_id, "error": str(e), "distance": distance}
            results.append(result)

    # Write full results
    out_file = PAIRS / "adjudication_results.json"
    json.dump(results, open(out_file, "w"), indent=2)

    # Print summary
    print(f"\n{'='*90}")
    print("CROSS-TASK SUBTYPE ADJUDICATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Pair':<30s} {'Dist':<10s} {'Risk':<7s} {'Adv':<5s} {'MrgA':<7s} {'MrgB':<7s} {'dA':<7s} {'dB':<7s}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['pair_id']:<30s} {r['distance']:<10s} ERROR")
            continue
        adv_s = "YES" if r["advisory"] else "no"
        print(f"{r['pair_id']:<30s} {r['distance']:<10s} {r['pair_risk']:<7s} {adv_s:<5s} "
              f"{r['merged_on_task_a']:.3f}   {r['merged_on_task_b']:.3f}   "
              f"{r['delta_task_a']:+.3f}   {r['delta_task_b']:+.3f}")

    # Subtype analysis
    print(f"\n{'='*60}")
    print("BY DISTANCE CATEGORY")
    for dist in ["same_task", "related", "distant"]:
        pairs = [r for r in results if r.get("distance") == dist and "error" not in r]
        if not pairs:
            continue
        deltas_a = [r["delta_task_a"] for r in pairs]
        deltas_b = [r["delta_task_b"] for r in pairs]
        max_delta = max(max(abs(d) for d in deltas_a), max(abs(d) for d in deltas_b))
        safe = sum(1 for r in pairs if abs(r["delta_task_a"]) <= 0.015 and abs(r["delta_task_b"]) <= 0.015)
        print(f"\n  {dist}: {len(pairs)} pairs")
        print(f"    Mean delta A: {np.mean(deltas_a):+.3f}")
        print(f"    Mean delta B: {np.mean(deltas_b):+.3f}")
        print(f"    Max abs delta: {max_delta:.3f}")
        print(f"    Safe (<=1.5pp both): {safe}/{len(pairs)}")


if __name__ == "__main__":
    main()
