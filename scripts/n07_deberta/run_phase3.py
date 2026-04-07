#!/usr/bin/env python3
"""N07 Phase 3: Merge adapter pairs and evaluate on source tasks.

For each of the 28 adapter pairs:
  - Merge LoRA backbone weights (linear average)
  - Evaluate on each source task using that task's classifier head
  - Record accuracy degradation vs. individual adapter performance

Usage:
    python3 scripts/n07_deberta/run_phase3.py --base-dir /workspace/n07
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "microsoft/deberta-v3-base"

TASKS = {
    "qnli": {
        "dataset": "glue",
        "subset": "qnli",
        "text_fields": ("question", "sentence"),
        "label_field": "label",
        "split": "validation",
    },
    "rte": {
        "dataset": "glue",
        "subset": "rte",
        "text_fields": ("sentence1", "sentence2"),
        "label_field": "label",
        "split": "validation",
    },
    "mrpc": {
        "dataset": "glue",
        "subset": "mrpc",
        "text_fields": ("sentence1", "sentence2"),
        "label_field": "label",
        "split": "validation",  # use test for final, validation for now
    },
    "sst2": {
        "dataset": "glue",
        "subset": "sst2",
        "text_fields": ("sentence",),
        "label_field": "label",
        "split": "validation",
    },
}

# Cache for loaded eval datasets
_eval_cache: dict[str, object] = {}


def load_eval_dataset(task: str, tokenizer: AutoTokenizer, max_samples: int = 0):
    """Load and tokenize an evaluation dataset. Cached across calls."""
    cache_key = f"{task}_{max_samples}"
    if cache_key in _eval_cache:
        return _eval_cache[cache_key]

    cfg = TASKS[task]
    ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"])

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    text_fields = cfg["text_fields"]
    if len(text_fields) == 2:
        def tokenize_fn(examples):
            return tokenizer(
                examples[text_fields[0]],
                examples[text_fields[1]],
                padding="max_length",
                truncation=True,
                max_length=256,
            )
    else:
        def tokenize_fn(examples):
            return tokenizer(
                examples[text_fields[0]],
                padding="max_length",
                truncation=True,
                max_length=128,
            )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds.column_names if c != cfg["label_field"]])
    ds = ds.rename_column(cfg["label_field"], "labels")
    ds.set_format("torch")
    _eval_cache[cache_key] = ds
    return ds


def evaluate_model(model, dataset, batch_size: int = 64) -> float:
    """Evaluate a model on a dataset, return accuracy."""
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def _build_merged_adapter(
    adapter_a_path: Path,
    adapter_b_path: Path,
    eval_adapter_path: Path,
    merge_dir: Path,
) -> Path:
    """Create a merged adapter by averaging LoRA weights, using eval adapter's classifier.

    Returns path to the temporary merged adapter directory.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    import shutil

    merge_dir.mkdir(parents=True, exist_ok=True)

    # Load both adapter weight files
    with safe_open(str(adapter_a_path / "adapter_model.safetensors"), framework="pt") as fa:
        keys_a = list(fa.keys())
        weights_a = {k: fa.get_tensor(k) for k in keys_a}
    with safe_open(str(adapter_b_path / "adapter_model.safetensors"), framework="pt") as fb:
        keys_b = list(fb.keys())
        weights_b = {k: fb.get_tensor(k) for k in keys_b}

    # Load eval adapter's classifier head
    with safe_open(str(eval_adapter_path / "adapter_model.safetensors"), framework="pt") as fe:
        classifier_w = fe.get_tensor("base_model.model.classifier.weight")
        classifier_b = fe.get_tensor("base_model.model.classifier.bias")

    # Average LoRA weights, use eval adapter's classifier
    merged = {}
    lora_keys = [k for k in keys_a if "lora" in k]
    for k in lora_keys:
        if k in weights_b:
            merged[k] = (weights_a[k] + weights_b[k]) / 2.0
        else:
            merged[k] = weights_a[k]

    # Set classifier from eval adapter
    merged["base_model.model.classifier.weight"] = classifier_w
    merged["base_model.model.classifier.bias"] = classifier_b

    # Save merged weights
    save_file(merged, str(merge_dir / "adapter_model.safetensors"))

    # Copy adapter config from eval adapter (same LoRA structure)
    shutil.copy2(eval_adapter_path / "adapter_config.json", merge_dir / "adapter_config.json")

    return merge_dir


def merge_and_evaluate(
    adapter_a_path: Path,
    adapter_b_path: Path,
    eval_task: str,
    eval_adapter_path: Path,
    tokenizer: AutoTokenizer,
    merge_dir: Path,
    max_samples: int = 0,
    device: str = "cuda",
) -> float:
    """Merge two adapters (linear average of LoRA weights) and evaluate.

    The classifier head comes from eval_adapter_path (the adapter trained on eval_task).
    The LoRA backbone is the average of adapter_a and adapter_b.
    """
    # Build merged adapter on disk
    merged_path = _build_merged_adapter(adapter_a_path, adapter_b_path, eval_adapter_path, merge_dir)

    # Load base model + merged adapter
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, str(merged_path))

    # Merge LoRA into base weights for faster inference
    model = model.merge_and_unload()

    model = model.to(device)

    # Load eval dataset
    ds = load_eval_dataset(eval_task, tokenizer, max_samples=max_samples)

    accuracy = evaluate_model(model, ds)

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return accuracy


def get_task_from_adapter_id(adapter_id: str) -> str:
    """Extract task name from adapter ID like deberta_qnli_s42."""
    parts = adapter_id.split("_")
    # deberta_TASK_sSEED
    return parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="N07 Phase 3: Merge + Evaluate")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Limit eval to 50 samples")
    parser.add_argument("--device", default="cuda", help="Device for evaluation")
    args = parser.parse_args()

    base_dir = args.base_dir
    adapters_dir = base_dir / "adapters"
    merges_dir = base_dir / "merges"
    merges_dir.mkdir(exist_ok=True)

    max_samples = 50 if args.smoke else 0

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load individual adapter scores for comparison
    individual_scores: dict[str, float] = {}
    for adapter_dir in sorted(adapters_dir.iterdir()):
        if not adapter_dir.is_dir():
            continue
        score_file = adapter_dir / "source_score.json"
        if score_file.exists():
            data = json.loads(score_file.read_text())
            individual_scores[adapter_dir.name] = data["accuracy"]

    # Load merge-audit risk data
    reports_dir = base_dir / "reports"
    pair_risks: dict[str, dict] = {}
    for report_file in sorted(reports_dir.glob("*.json")):
        data = json.loads(report_file.read_text())
        pair_risks[report_file.stem] = {
            "pair_risk": data.get("pair_risk", "?"),
            "recommended_strategy": data.get("recommended_strategy", "?"),
            "dominant_issue": data.get("dominant_issue", "?"),
        }

    # Generate all pairs
    adapter_dirs = sorted([d for d in adapters_dir.iterdir() if d.is_dir()])
    pairs = list(itertools.combinations(adapter_dirs, 2))

    print(f"N07 Phase 3: Merge + Evaluate")
    print(f"  Base dir: {base_dir}")
    print(f"  Adapters: {len(adapter_dirs)}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Smoke: {args.smoke}")
    print()

    results = []
    t0 = time.time()

    for i, (adapter_a, adapter_b) in enumerate(pairs, 1):
        a_id = adapter_a.name
        b_id = adapter_b.name
        pair_label = f"{a_id}_vs_{b_id}"
        task_a = get_task_from_adapter_id(a_id)
        task_b = get_task_from_adapter_id(b_id)

        risk_info = pair_risks.get(pair_label, {})

        print(f"{'=' * 60}")
        print(f"  [{i}/{len(pairs)}] {pair_label}")
        print(f"  Risk: {risk_info.get('pair_risk', '?')} | "
              f"Strategy: {risk_info.get('recommended_strategy', '?')} | "
              f"Issue: {risk_info.get('dominant_issue', '?')}")
        print(f"{'=' * 60}")

        # Determine which tasks to evaluate on
        eval_tasks = sorted(set([task_a, task_b]))

        for eval_task in eval_tasks:
            # Find the adapter that was trained on this task
            # (for classifier head). Use adapter_a if both are same task.
            if task_a == eval_task:
                eval_adapter = adapter_a
                source_id = a_id
            else:
                eval_adapter = adapter_b
                source_id = b_id

            source_accuracy = individual_scores.get(source_id, 0.0)

            merge_dir = merges_dir / f"{pair_label}_{eval_task}"

            t1 = time.time()
            try:
                merged_accuracy = merge_and_evaluate(
                    adapter_a_path=adapter_a,
                    adapter_b_path=adapter_b,
                    eval_task=eval_task,
                    eval_adapter_path=eval_adapter,
                    tokenizer=tokenizer,
                    merge_dir=merge_dir,
                    max_samples=max_samples,
                    device=args.device,
                )
                elapsed = time.time() - t1
                degradation = merged_accuracy - source_accuracy

                print(f"  {eval_task}: merged={merged_accuracy:.4f} "
                      f"source={source_accuracy:.4f} "
                      f"Δ={degradation:+.4f} ({elapsed:.1f}s)")

                results.append({
                    "pair": pair_label,
                    "adapter_a": a_id,
                    "adapter_b": b_id,
                    "task_a": task_a,
                    "task_b": task_b,
                    "eval_task": eval_task,
                    "eval_adapter": source_id,
                    "source_accuracy": source_accuracy,
                    "merged_accuracy": merged_accuracy,
                    "degradation": degradation,
                    "pair_risk": risk_info.get("pair_risk", "?"),
                    "recommended_strategy": risk_info.get("recommended_strategy", "?"),
                    "dominant_issue": risk_info.get("dominant_issue", "?"),
                    "eval_time_s": elapsed,
                })
            except Exception as e:
                print(f"  {eval_task}: ERROR — {e}")
                results.append({
                    "pair": pair_label,
                    "adapter_a": a_id,
                    "adapter_b": b_id,
                    "eval_task": eval_task,
                    "error": str(e),
                    "pair_risk": risk_info.get("pair_risk", "?"),
                })

    total_time = time.time() - t0

    # Save results
    results_file = merges_dir / "phase3_results.json"
    results_file.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Phase 3 Complete")
    print(f"{'=' * 60}")
    print(f"  Total evaluations: {len(results)}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Results: {results_file}")

    # Summary by risk level
    valid = [r for r in results if "error" not in r]
    if valid:
        print(f"\n  Degradation by risk level:")
        for risk in ["low", "medium", "high"]:
            subset = [r for r in valid if r["pair_risk"] == risk]
            if subset:
                degs = [r["degradation"] for r in subset]
                mean_deg = np.mean(degs)
                print(f"    {risk:<8}: mean Δ={mean_deg:+.4f} "
                      f"(n={len(subset)}, "
                      f"range=[{min(degs):+.4f}, {max(degs):+.4f}])")

        # Same-task vs cross-task
        same = [r for r in valid if r["task_a"] == r["task_b"]]
        cross = [r for r in valid if r["task_a"] != r["task_b"]]
        if same:
            print(f"\n  Same-task pairs: mean Δ={np.mean([r['degradation'] for r in same]):+.4f} (n={len(same)})")
        if cross:
            print(f"  Cross-task pairs: mean Δ={np.mean([r['degradation'] for r in cross]):+.4f} (n={len(cross)})")


if __name__ == "__main__":
    main()
