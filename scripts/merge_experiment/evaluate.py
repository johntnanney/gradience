"""
NLI evaluation using prompted causal LM generation.

Evaluates a base model (optionally with a PEFT LoRA adapter) on MNLI or QNLI
by formatting each example as a prompt, generating a response, and parsing
the predicted label.

Requires: transformers, peft, datasets (install with `pip install "gradience[bench]"`)

Usage::

    python scripts/merge_experiment/evaluate.py \\
        --base-model mistralai/Mistral-7B-v0.1 \\
        --adapter-dir ./merged_adapter \\
        --dataset mnli \\
        --max-samples 500 \\
        --output results/eval_mnli.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"\nMissing dependencies: {e}")
    print('Install with: pip install "gradience[bench]"')
    sys.exit(1)

from prompt_templates import (
    format_mnli_prompt,
    format_qnli_prompt,
    parse_nli_response,
    parse_nli_response_numeric,
)

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

MNLI_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
QNLI_LABEL_MAP = {0: "entailment", 1: "not_entailment"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    base_model_name: str,
    adapter_dir: str | None = None,
    device: str = "cuda",
    dtype: str = "float16",
) -> tuple:
    """Load a causal LM with optional PEFT adapter.

    Parameters
    ----------
    base_model_name : HuggingFace model identifier
    adapter_dir : optional path to PEFT adapter directory
    device : "cuda" or "cpu"
    dtype : "float16" or "bfloat16"

    Returns
    -------
    (model, tokenizer) tuple
    """
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    print(f"  Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
    )

    if adapter_dir is not None:
        print(f"  Loading adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

    if device != "cuda":
        model = model.to(device)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_nli(
    base_model_name: str,
    adapter_dir: str | None,
    dataset_name: str,
    split: str = "validation",
    max_samples: int = 500,
    device: str = "cuda",
    dtype: str = "float16",
    max_new_tokens: int = 5,
    template_mode: str = "custom",
) -> dict[str, Any]:
    """Evaluate an adapter (or base model) on NLI via prompted generation.

    Parameters
    ----------
    base_model_name : HuggingFace model identifier
    adapter_dir : path to PEFT adapter directory (None for base model only)
    dataset_name : "mnli" or "qnli"
    split : dataset split ("validation" for MNLI, "validation" for QNLI)
    max_samples : maximum number of examples to evaluate
    device : "cuda" or "cpu"
    dtype : "float16" or "bfloat16"
    max_new_tokens : max tokens to generate per example
    template_mode : "predibase" for numeric output, "custom" for word labels

    Returns
    -------
    Dict with accuracy, per-example predictions, and metadata.
    """
    # Predibase outputs single digits — fewer tokens needed
    if template_mode == "predibase" and max_new_tokens > 3:
        max_new_tokens = 3

    # Load model
    model, tokenizer = load_model_and_tokenizer(base_model_name, adapter_dir, device, dtype)

    # Load dataset
    if dataset_name == "mnli":
        ds = load_dataset("glue", "mnli", split="validation_matched")
        label_map = MNLI_LABEL_MAP
    elif dataset_name == "qnli":
        ds = load_dataset("glue", "qnli", split=split)
        label_map = QNLI_LABEL_MAP
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnli' or 'qnli'.")

    # Subsample
    if max_samples < len(ds):
        ds = ds.select(range(max_samples))

    print(f"  Evaluating {len(ds)} examples from {dataset_name}...")

    # Run evaluation
    predictions: list[dict[str, Any]] = []
    correct = 0
    start_time = time.monotonic()

    for i, example in enumerate(ds):
        # Format prompt (mode-aware)
        if dataset_name == "mnli":
            prompt = format_mnli_prompt(
                example["premise"],
                example["hypothesis"],
                template_mode=template_mode,
            )
            gold_label = label_map[example["label"]]
        else:  # qnli
            prompt = format_qnli_prompt(
                example["question"],
                example["sentence"],
                template_mode=template_mode,
            )
            gold_label = label_map[example["label"]]

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (exclude prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse label (mode-aware)
        if template_mode == "predibase":
            predicted_label = parse_nli_response_numeric(response, task=dataset_name)
        else:
            predicted_label = parse_nli_response(response)
        is_correct = predicted_label == gold_label
        if is_correct:
            correct += 1

        predictions.append(
            {
                "index": i,
                "gold": gold_label,
                "predicted": predicted_label,
                "correct": is_correct,
                "raw_response": response.strip()[:100],  # truncate for storage
            }
        )

        if (i + 1) % 50 == 0:
            elapsed = time.monotonic() - start_time
            acc_so_far = correct / (i + 1)
            print(f"    [{i + 1}/{len(ds)}] acc={acc_so_far:.3f} ({elapsed:.0f}s elapsed)")

    total_time = time.monotonic() - start_time
    accuracy = correct / len(ds) if len(ds) > 0 else 0.0

    # Count label distribution
    pred_dist = {}
    for p in predictions:
        pred_dist[p["predicted"]] = pred_dist.get(p["predicted"], 0) + 1

    result = {
        "dataset": dataset_name,
        "split": split,
        "n_samples": len(ds),
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total_time_seconds": round(total_time, 1),
        "base_model": base_model_name,
        "adapter_dir": adapter_dir,
        "template_mode": template_mode,
        "prediction_distribution": pred_dist,
        "predictions": predictions,
    }

    print(f"  Result: accuracy={accuracy:.4f} ({correct}/{len(ds)}) in {total_time:.1f}s")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Quick validation (for template discovery timebox)
# ---------------------------------------------------------------------------


def quick_validate(
    base_model_name: str,
    adapter_dir: str,
    dataset_name: str,
    template_mode: str = "predibase",
    n_samples: int = 100,
    device: str = "cuda",
    threshold: float = 0.70,
) -> tuple:
    """Run a fast validation and return (accuracy, passed).

    Parameters
    ----------
    base_model_name : HuggingFace model identifier
    adapter_dir : path to PEFT adapter directory
    dataset_name : "mnli" or "qnli"
    template_mode : prompt template mode
    n_samples : number of validation samples
    device : "cuda" or "cpu"
    threshold : minimum accuracy to pass

    Returns
    -------
    (accuracy, passed) where passed = accuracy >= threshold
    """
    result = evaluate_nli(
        base_model_name=base_model_name,
        adapter_dir=adapter_dir,
        dataset_name=dataset_name,
        max_samples=n_samples,
        device=device,
        template_mode=template_mode,
    )
    accuracy = result["accuracy"]
    return accuracy, accuracy >= threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model (with optional adapter) on MNLI or QNLI")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=None,
        help="Path to PEFT adapter directory (omit for base model only)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnli", "qnli"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples to evaluate (default: 500)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--template-mode",
        type=str,
        default="custom",
        choices=["predibase", "custom"],
        help="Prompt template mode (default: custom)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    result = evaluate_nli(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_dir,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        device=args.device,
        template_mode=args.template_mode,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
