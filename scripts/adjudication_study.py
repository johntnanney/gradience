"""Core-space adjudication study.

For a panel of pairs where pair_risk=low but core-space disagrees,
merge the adapters and evaluate downstream to determine who is right.

Workflow per pair:
1. merge_audit (reuse existing report data)
2. plan_from_audit with uniform_linear strategy
3. execute_merge to produce PEFT-compatible adapter
4. Load distilbert-base-uncased + merged adapter
5. Evaluate on QNLI dev subset
6. Compare with individual adapter scores
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Panel definition
# ---------------------------------------------------------------------------

PANEL = [
    {
        "pair_id": "qprobe_x_quniform",
        "label": "same-task, incompatible",
        "adapter_a": "results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/adapters/qnli_probe_elig",
        "adapter_b": "results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/adapters/qnli_uniform_elig",
        "core_space_status": "incompatible",
        "shared_basis_score": 0.807,
        "individual_a_score": 0.87,
        "individual_b_score": 0.87,
    },
    {
        "pair_id": "final_x_priority",
        "label": "same-group, incompatible",
        "adapter_a": "results/core_space_benchmark/real_adapters/final_uniform_median_r16",
        "adapter_b": "results/core_space_benchmark/real_adapters/priority_probe_r16",
        "core_space_status": "incompatible",
        "shared_basis_score": 0.824,
        "individual_a_score": None,  # generic fine-tuning, no QNLI-specific score
        "individual_b_score": None,
    },
    {
        "pair_id": "uniform_x_final",
        "label": "cross-task, incompatible",
        "adapter_a": "results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/adapters/qnli_uniform_elig",
        "adapter_b": "results/core_space_benchmark/real_adapters/final_uniform_median_r16",
        "core_space_status": "incompatible",
        "shared_basis_score": 0.859,
        "individual_a_score": 0.87,
        "individual_b_score": None,
    },
    {
        "pair_id": "probe_x_final",
        "label": "cross-task, incompatible",
        "adapter_a": "results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/adapters/qnli_probe_elig",
        "adapter_b": "results/core_space_benchmark/real_adapters/final_uniform_median_r16",
        "core_space_status": "incompatible",
        "shared_basis_score": 0.860,
        "individual_a_score": 0.87,
        "individual_b_score": None,
    },
    {
        "pair_id": "priority_x_qprobe",
        "label": "cross-task, marginal",
        "adapter_a": "results/core_space_benchmark/real_adapters/priority_probe_r16",
        "adapter_b": "results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/adapters/qnli_probe_elig",
        "core_space_status": "marginal",
        "shared_basis_score": 0.864,
        "individual_a_score": None,
        "individual_b_score": 0.87,
    },
    {
        "pair_id": "perlayer_x_final",
        "label": "cross-task, marginal (highest basis)",
        "adapter_a": "results/real_inventory_runs/20260317/cycle03_qnli_all_eligible_triplet/adapters/qnli_per_layer_elig",
        "adapter_b": "results/core_space_benchmark/real_adapters/final_uniform_median_r16",
        "core_space_status": "marginal",
        "shared_basis_score": 0.931,
        "individual_a_score": 0.87,
        "individual_b_score": None,
    },
]

BASE_MODEL = "distilbert-base-uncased"
EVAL_SAMPLES = 500
MERGE_STRATEGY = "uniform_linear"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    pair_id: str
    label: str
    core_space_status: str
    shared_basis_score: float
    merged_accuracy: float | None
    individual_a_score: float | None
    individual_b_score: float | None
    best_individual: float | None
    degradation: float | None  # best_individual - merged
    merge_error: float | None  # reconstruction error from merge
    eval_error: str | None


def _patch_merged_adapter_with_classifier(
    merged_dir: str,
    source_adapter_a: str,
    source_adapter_b: str,
) -> None:
    """Copy classifier head weights from source adapters into the merged adapter.

    The merge pipeline only handles LoRA attention layers. For SEQ_CLS adapters,
    the classifier head (pre_classifier + classifier) must be carried over.
    Strategy: average the classifier weights from both source adapters.
    """
    from safetensors.torch import load_file, save_file

    merged_path = Path(merged_dir) / "adapter_model.safetensors"
    merged_st = load_file(str(merged_path))

    # Check if classifier weights are already present
    if "base_model.model.classifier.weight" in merged_st:
        return

    st_a = load_file(str(Path(source_adapter_a) / "adapter_model.safetensors"))
    st_b = load_file(str(Path(source_adapter_b) / "adapter_model.safetensors"))

    cls_keys = [
        "base_model.model.classifier.weight",
        "base_model.model.classifier.bias",
        "base_model.model.pre_classifier.weight",
        "base_model.model.pre_classifier.bias",
    ]

    for key in cls_keys:
        if key in st_a and key in st_b:
            merged_st[key] = (st_a[key] + st_b[key]) / 2.0
        elif key in st_a:
            merged_st[key] = st_a[key]
        elif key in st_b:
            merged_st[key] = st_b[key]

    save_file(merged_st, str(merged_path))


def evaluate_adapter_on_qnli(
    base_model_name: str,
    adapter_dir: str | None,
    max_samples: int = 500,
) -> float:
    """Load base model + optional adapter, evaluate on QNLI dev using Trainer."""
    import numpy as np
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        torch_dtype=torch.float32,
    )

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()

    model.eval()

    dataset = load_dataset("glue", "qnli", split="validation")
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    def preprocess(examples):
        result = tokenizer(
            examples["question"],
            examples["sentence"],
            truncation=True,
            padding=True,
            max_length=128,
        )
        result["labels"] = examples["label"]
        return result

    tokenized = dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir="/tmp/adjudication_eval",
        per_device_eval_batch_size=32,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    predictions = trainer.predict(tokenized)
    pred_classes = np.argmax(predictions.predictions, axis=1)
    accuracy = float((pred_classes == predictions.label_ids).mean())

    return accuracy


def run_adjudication(output_dir: str = "results/adjudication_study") -> list[EvalResult]:
    """Run the full adjudication panel."""
    from gradience.vnext.merge import execute_merge, merge_audit, plan_from_audit

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 0: Get base model accuracy (no adapter)
    print("Evaluating base model (no adapter)...")
    base_accuracy = evaluate_adapter_on_qnli(BASE_MODEL, None, EVAL_SAMPLES)
    print(f"  Base model accuracy: {base_accuracy:.3f}")

    results = []

    for entry in PANEL:
        pair_id = entry["pair_id"]
        print(f"\n{'='*60}")
        print(f"Pair: {pair_id} ({entry['label']})")
        print(f"  Core-space: {entry['core_space_status']} (basis={entry['shared_basis_score']:.3f})")

        eval_error = None
        merged_accuracy = None
        merge_error_val = None

        try:
            # Step 1: Merge
            print("  Merging...")
            with tempfile.TemporaryDirectory() as tmpdir:
                merge_dir = Path(tmpdir) / "merged"
                audit_dir = Path(tmpdir) / "audit"

                report = merge_audit(
                    entry["adapter_a"],
                    entry["adapter_b"],
                    output_dir=str(audit_dir),
                )

                plan = plan_from_audit(
                    MERGE_STRATEGY,
                    report,
                    entry["adapter_a"],
                    entry["adapter_b"],
                )

                merge_result = execute_merge(plan, str(merge_dir))
                merge_error_val = merge_result.mean_reconstruction_error
                print(f"  Merge done (reconstruction error: {merge_error_val:.4f})")

                # Step 1b: Patch classifier head into merged adapter
                _patch_merged_adapter_with_classifier(
                    str(merge_dir),
                    entry["adapter_a"],
                    entry["adapter_b"],
                )

                # Step 2: Evaluate merged adapter
                print(f"  Evaluating merged adapter on QNLI ({EVAL_SAMPLES} samples)...")
                merged_accuracy = evaluate_adapter_on_qnli(
                    BASE_MODEL,
                    str(merge_dir),
                    EVAL_SAMPLES,
                )
                print(f"  Merged accuracy: {merged_accuracy:.3f}")

        except Exception as e:
            eval_error = str(e)
            print(f"  ERROR: {e}")

        best_ind = None
        degradation = None
        scores = [s for s in [entry["individual_a_score"], entry["individual_b_score"]] if s is not None]
        if scores:
            best_ind = max(scores)
        if best_ind is not None and merged_accuracy is not None:
            degradation = best_ind - merged_accuracy

        result = EvalResult(
            pair_id=pair_id,
            label=entry["label"],
            core_space_status=entry["core_space_status"],
            shared_basis_score=entry["shared_basis_score"],
            merged_accuracy=merged_accuracy,
            individual_a_score=entry["individual_a_score"],
            individual_b_score=entry["individual_b_score"],
            best_individual=best_ind,
            degradation=degradation,
            merge_error=merge_error_val,
            eval_error=eval_error,
        )
        results.append(result)

    # Write results
    results_data = {
        "base_model": BASE_MODEL,
        "base_accuracy": base_accuracy,
        "eval_samples": EVAL_SAMPLES,
        "merge_strategy": MERGE_STRATEGY,
        "panel": [asdict(r) for r in results],
    }

    results_path = out / "adjudication_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults written to {results_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("ADJUDICATION SUMMARY")
    print(f"Base model accuracy: {base_accuracy:.3f}")
    print(f"{'='*80}")
    print(f"{'Pair':<25s} {'CS status':<14s} {'Basis':<7s} {'Merged':<8s} {'Best ind':<9s} {'Degrad':<8s}")
    print(f"{'-'*25} {'-'*14} {'-'*7} {'-'*8} {'-'*9} {'-'*8}")
    for r in results:
        merged_s = f"{r.merged_accuracy:.3f}" if r.merged_accuracy is not None else "ERROR"
        best_s = f"{r.best_individual:.3f}" if r.best_individual is not None else "n/a"
        deg_s = f"{r.degradation:+.3f}" if r.degradation is not None else "n/a"
        print(f"{r.pair_id:<25s} {r.core_space_status:<14s} {r.shared_basis_score:<7.3f} {merged_s:<8s} {best_s:<9s} {deg_s:<8s}")

    return results


if __name__ == "__main__":
    run_adjudication()
