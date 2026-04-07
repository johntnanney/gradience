#!/usr/bin/env python3
"""N07 Phase 3b: Strategy-aware merge evaluation.

Contrasts naive linear merge (Phase 3) with audit-informed strategies:
  - norm_imbalance → norm-equalized merge (scale by inverse Frobenius norm ratio)
  - partial_redundancy / high_redundancy → TIES-like merge (keep top-k magnitude directions)
  - none → linear merge (same as Phase 3 baseline)

Usage:
    python3 scripts/n07_deberta/run_phase3b.py --base-dir /workspace/n07
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shutil


MODEL_NAME = "microsoft/deberta-v3-base"

TASKS = {
    "qnli": {"dataset": "glue", "subset": "qnli", "text_fields": ("question", "sentence"), "split": "validation"},
    "rte": {"dataset": "glue", "subset": "rte", "text_fields": ("sentence1", "sentence2"), "split": "validation"},
    "mrpc": {"dataset": "glue", "subset": "mrpc", "text_fields": ("sentence1", "sentence2"), "split": "validation"},
    "sst2": {"dataset": "glue", "subset": "sst2", "text_fields": ("sentence",), "split": "validation"},
}

_eval_cache: dict[str, object] = {}


def load_eval_dataset(task: str, tokenizer, max_samples: int = 0):
    cache_key = f"{task}_{max_samples}"
    if cache_key in _eval_cache:
        return _eval_cache[cache_key]
    cfg = TASKS[task]
    ds = load_dataset(cfg["dataset"], cfg["subset"], split=cfg["split"])
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    tf = cfg["text_fields"]
    if len(tf) == 2:
        ds = ds.map(lambda x: tokenizer(x[tf[0]], x[tf[1]], padding="max_length", truncation=True, max_length=256),
                     batched=True, remove_columns=[c for c in ds.column_names if c != "label"])
    else:
        ds = ds.map(lambda x: tokenizer(x[tf[0]], padding="max_length", truncation=True, max_length=128),
                     batched=True, remove_columns=[c for c in ds.column_names if c != "label"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    _eval_cache[cache_key] = ds
    return ds


def evaluate_model(model, dataset, batch_size: int = 64) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dl:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"].to(device)).sum().item()
            total += batch["labels"].size(0)
    return correct / total if total > 0 else 0.0


def load_adapter_weights(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path / "adapter_model.safetensors"), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def get_lora_keys(weights: dict) -> list[str]:
    return [k for k in weights if "lora" in k]


def merge_linear(weights_a: dict, weights_b: dict, eval_weights: dict) -> dict:
    """Simple 0.5/0.5 average of LoRA weights."""
    merged = {}
    for k in get_lora_keys(weights_a):
        if k in weights_b:
            merged[k] = (weights_a[k] + weights_b[k]) / 2.0
        else:
            merged[k] = weights_a[k]
    merged["base_model.model.classifier.weight"] = eval_weights["base_model.model.classifier.weight"]
    merged["base_model.model.classifier.bias"] = eval_weights["base_model.model.classifier.bias"]
    return merged


def merge_norm_equalized(weights_a: dict, weights_b: dict, eval_weights: dict) -> dict:
    """Norm-equalized merge: scale each adapter's LoRA weights by inverse of its total Frobenius norm."""
    lora_keys = get_lora_keys(weights_a)

    # Compute total Frobenius norm per adapter
    norm_a = math.sqrt(sum(weights_a[k].float().pow(2).sum().item() for k in lora_keys))
    norm_b = math.sqrt(sum(weights_b[k].float().pow(2).sum().item() for k in lora_keys if k in weights_b))

    # Scale factors: equalize norms then average
    if norm_a > 0 and norm_b > 0:
        # Scale to geometric mean
        geom_mean = math.sqrt(norm_a * norm_b)
        scale_a = geom_mean / norm_a
        scale_b = geom_mean / norm_b
    else:
        scale_a = scale_b = 1.0

    merged = {}
    for k in lora_keys:
        if k in weights_b:
            merged[k] = (weights_a[k] * scale_a + weights_b[k] * scale_b) / 2.0
        else:
            merged[k] = weights_a[k] * scale_a
    merged["base_model.model.classifier.weight"] = eval_weights["base_model.model.classifier.weight"]
    merged["base_model.model.classifier.bias"] = eval_weights["base_model.model.classifier.bias"]
    return merged


def merge_ties(weights_a: dict, weights_b: dict, eval_weights: dict, density: float = 0.5) -> dict:
    """TIES-like merge: keep only top-k% magnitude parameters, resolve sign conflicts, average."""
    lora_keys = get_lora_keys(weights_a)
    merged = {}

    for k in lora_keys:
        wa = weights_a[k].float()
        if k not in weights_b:
            merged[k] = wa
            continue
        wb = weights_b[k].float()

        # Step 1: Trim — zero out small-magnitude elements per adapter
        threshold_a = torch.quantile(wa.abs().flatten(), 1.0 - density)
        threshold_b = torch.quantile(wb.abs().flatten(), 1.0 - density)
        trimmed_a = wa * (wa.abs() >= threshold_a)
        trimmed_b = wb * (wb.abs() >= threshold_b)

        # Step 2: Resolve sign conflicts — keep the sign with larger total magnitude
        sign_agree = (trimmed_a * trimmed_b) >= 0  # True where signs agree or one is zero
        # Where signs conflict, keep the one with larger absolute value
        magnitude_a_wins = trimmed_a.abs() >= trimmed_b.abs()
        resolved_a = torch.where(sign_agree | magnitude_a_wins, trimmed_a, torch.zeros_like(trimmed_a))
        resolved_b = torch.where(sign_agree | ~magnitude_a_wins, trimmed_b, torch.zeros_like(trimmed_b))

        # Step 3: Disjoint merge — average non-zero contributions
        n_nonzero = (resolved_a != 0).float() + (resolved_b != 0).float()
        n_nonzero = n_nonzero.clamp(min=1)
        merged[k] = ((resolved_a + resolved_b) / n_nonzero).to(weights_a[k].dtype)

    merged["base_model.model.classifier.weight"] = eval_weights["base_model.model.classifier.weight"]
    merged["base_model.model.classifier.bias"] = eval_weights["base_model.model.classifier.bias"]
    return merged


def select_strategy(dominant_issue: str) -> tuple[str, callable]:
    """Select merge strategy based on audit-diagnosed dominant issue."""
    if dominant_issue == "norm_imbalance":
        return "norm_equalized", merge_norm_equalized
    elif dominant_issue in ("partial_redundancy", "high_redundancy"):
        return "ties", merge_ties
    else:
        return "linear", merge_linear


def save_and_evaluate(merged_weights: dict, eval_task: str, merge_dir: Path,
                      adapter_config_path: Path, tokenizer, device: str, max_samples: int) -> float:
    merge_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged_weights, str(merge_dir / "adapter_model.safetensors"))
    shutil.copy2(adapter_config_path, merge_dir / "adapter_config.json")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(merge_dir))
    model = model.merge_and_unload()
    model = model.to(device)

    ds = load_eval_dataset(eval_task, tokenizer, max_samples=max_samples)
    accuracy = evaluate_model(model, ds)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return accuracy


def get_task(adapter_id: str) -> str:
    return adapter_id.split("_")[1]


def main():
    parser = argparse.ArgumentParser(description="N07 Phase 3b: Strategy-aware merge evaluation")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base_dir = args.base_dir
    adapters_dir = base_dir / "adapters"
    merges_dir = base_dir / "merges_strategic"
    merges_dir.mkdir(exist_ok=True)
    max_samples = 50 if args.smoke else 0

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load individual scores
    individual_scores = {}
    for d in sorted(adapters_dir.iterdir()):
        if not d.is_dir():
            continue
        sf = d / "source_score.json"
        if sf.exists():
            individual_scores[d.name] = json.loads(sf.read_text())["accuracy"]

    # Load merge reports
    reports_dir = base_dir / "reports"
    pair_reports = {}
    for f in sorted(reports_dir.glob("*.json")):
        pair_reports[f.stem] = json.loads(f.read_text())

    # Load Phase 3 baseline results
    baseline_results = {}
    p3_file = base_dir / "merges" / "phase3_results.json"
    if p3_file.exists():
        for r in json.loads(p3_file.read_text()):
            if "error" not in r:
                key = (r["pair"], r["eval_task"])
                baseline_results[key] = r["merged_accuracy"]

    adapter_dirs = sorted([d for d in adapters_dir.iterdir() if d.is_dir()])
    pairs = list(itertools.combinations(adapter_dirs, 2))

    print(f"N07 Phase 3b: Strategy-Aware Merge Evaluation")
    print(f"  Pairs: {len(pairs)}, Smoke: {args.smoke}")
    print()

    results = []
    t0 = time.time()

    for i, (adapter_a, adapter_b) in enumerate(pairs, 1):
        a_id = adapter_a.name
        b_id = adapter_b.name
        pair_label = f"{a_id}_vs_{b_id}"
        task_a = get_task(a_id)
        task_b = get_task(b_id)

        report = pair_reports.get(pair_label, {})
        dominant_issue = report.get("dominant_issue", "none")
        pair_risk = report.get("pair_risk", "?")
        strategy_name, merge_fn = select_strategy(dominant_issue)

        print(f"[{i}/{len(pairs)}] {pair_label}")
        print(f"  Issue: {dominant_issue} → Strategy: {strategy_name} | Risk: {pair_risk}")

        weights_a = load_adapter_weights(adapter_a)
        weights_b = load_adapter_weights(adapter_b)

        eval_tasks = sorted(set([task_a, task_b]))
        for eval_task in eval_tasks:
            eval_adapter = adapter_a if task_a == eval_task else adapter_b
            source_id = a_id if task_a == eval_task else b_id
            eval_weights = load_adapter_weights(eval_adapter)
            source_acc = individual_scores.get(source_id, 0.0)
            baseline_acc = baseline_results.get((pair_label, eval_task))

            # Run strategy-aware merge
            merged_weights = merge_fn(weights_a, weights_b, eval_weights)
            merge_dir = merges_dir / f"{pair_label}_{eval_task}_{strategy_name}"

            t1 = time.time()
            try:
                acc = save_and_evaluate(merged_weights, eval_task, merge_dir,
                                        eval_adapter / "adapter_config.json",
                                        tokenizer, args.device, max_samples)
                elapsed = time.time() - t1
                deg = acc - source_acc
                improvement = (acc - baseline_acc) if baseline_acc is not None else None

                print(f"  {eval_task}: strategic={acc:.4f} baseline={baseline_acc or 0:.4f} "
                      f"source={source_acc:.4f} Δ={deg:+.4f} "
                      f"improvement={improvement:+.4f}" if improvement is not None else "")

                results.append({
                    "pair": pair_label, "adapter_a": a_id, "adapter_b": b_id,
                    "task_a": task_a, "task_b": task_b, "eval_task": eval_task,
                    "strategy": strategy_name, "dominant_issue": dominant_issue,
                    "pair_risk": pair_risk,
                    "strategic_accuracy": acc, "baseline_accuracy": baseline_acc,
                    "source_accuracy": source_acc,
                    "degradation_strategic": deg,
                    "degradation_baseline": (baseline_acc - source_acc) if baseline_acc else None,
                    "improvement_over_baseline": improvement,
                    "eval_time_s": elapsed,
                })
            except Exception as e:
                print(f"  {eval_task}: ERROR — {e}")
                results.append({"pair": pair_label, "eval_task": eval_task, "error": str(e)})

    total_time = time.time() - t0

    # Save
    out_file = merges_dir / "phase3b_results.json"
    out_file.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Phase 3b Complete — {len(results)} evaluations in {total_time:.0f}s")
    print(f"{'=' * 70}")

    valid = [r for r in results if "error" not in r and r.get("improvement_over_baseline") is not None]
    if valid:
        improvements = [r["improvement_over_baseline"] for r in valid]
        print(f"\n  Overall: mean improvement over baseline = {np.mean(improvements):+.4f}")
        print(f"  Positive improvements: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")

        for strategy in ["norm_equalized", "ties", "linear"]:
            subset = [r for r in valid if r["strategy"] == strategy]
            if subset:
                imps = [r["improvement_over_baseline"] for r in subset]
                print(f"\n  {strategy}:")
                print(f"    mean improvement: {np.mean(imps):+.4f}")
                print(f"    positive: {sum(1 for x in imps if x > 0)}/{len(imps)}")
                degs = [r["degradation_strategic"] for r in subset]
                print(f"    mean degradation: {np.mean(degs):+.4f} (vs baseline {np.mean([r['degradation_baseline'] for r in subset if r['degradation_baseline']]):.4f})")

        for risk in ["low", "medium", "high"]:
            subset = [r for r in valid if r["pair_risk"] == risk]
            if subset:
                imps = [r["improvement_over_baseline"] for r in subset]
                print(f"\n  Risk {risk}: mean improvement = {np.mean(imps):+.4f} (n={len(subset)})")

    print(f"\n  Results: {out_file}")


if __name__ == "__main__":
    main()
