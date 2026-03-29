#!/usr/bin/env python3
"""Evidence bootstrap — run lightweight CPU evals to generate behavioral evidence.

For each adapter in a field-trial manifest, this script:
1. Downloads the adapter (if not cached)
2. Loads the base model + adapter
3. Runs inference on a small validation slice
4. Computes accuracy (and F1 for binary tasks)
5. Runs the same eval on the base model alone (no adapter)
6. Writes a JSON evidence file per adapter

The evidence files can then be fed into Gradience's audit_adapter()
as behavioral scores.

Usage:
    python field_trials/evidence_bootstrap.py inventory_01_same_task_control
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("bootstrap")

FIELD_TRIALS_DIR = Path(__file__).resolve().parent

# Maximum samples per eval — CPU-friendly budget
MAX_SAMPLES = 500

# ---------------------------------------------------------------------------
# Dataset registry: maps (dataset, config) → loading instructions
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict] = {
    "imdb": {
        "hf_name": "imdb",
        "hf_config": None,
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
        "metric": "accuracy",
    },
    "sst2": {
        "hf_name": "sst2",
        "hf_config": None,
        "split": "validation",
        "text_col": "sentence",
        "label_col": "label",
        "num_labels": 2,
        "metric": "accuracy",
    },
    "tweet_eval/emotion": {
        "hf_name": "tweet_eval",
        "hf_config": "emotion",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 4,
        "metric": "accuracy",
    },
    "tweet_eval/hate": {
        "hf_name": "tweet_eval",
        "hf_config": "hate",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
        "metric": "accuracy",
    },
    "tweet_eval/irony": {
        "hf_name": "tweet_eval",
        "hf_config": "irony",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
        "metric": "accuracy",
    },
    "ag_news": {
        "hf_name": "ag_news",
        "hf_config": None,
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 4,
        "metric": "accuracy",
    },
    "mnli": {
        "hf_name": "nyu-mll/multi_nli",
        "hf_config": None,
        "split": "validation_matched",
        "text_col": "premise",
        "text_col_b": "hypothesis",
        "label_col": "label",
        "num_labels": 3,
        "metric": "accuracy",
    },
}

# Map manifest dataset names to registry keys
DATASET_ALIASES: dict[str, str] = {
    "unknown": "imdb",  # best guess for generic text-classification on distilbert
}


def resolve_dataset(manifest_dataset: str) -> str:
    """Resolve a manifest dataset name to a registry key."""
    if manifest_dataset in DATASET_REGISTRY:
        return manifest_dataset
    return DATASET_ALIASES.get(manifest_dataset, manifest_dataset)


def load_eval_data(ds_key: str, max_samples: int = MAX_SAMPLES) -> tuple:
    """Load a small eval slice. Returns (texts, labels, ds_info)."""
    info = DATASET_REGISTRY[ds_key]
    slice_spec = f"{info['split']}[:{max_samples}]"
    ds = load_dataset(info["hf_name"], info["hf_config"], split=slice_spec)

    texts = ds[info["text_col"]]
    if "text_col_b" in info:
        texts_b = ds[info["text_col_b"]]
        texts = list(zip(texts, texts_b))
    labels = ds[info["label_col"]]
    return texts, labels, info


def resolve_true_base_model(peft_config: PeftConfig) -> str:
    """Resolve the actual base model ID for loading.

    Some adapters (especially TransferGraph) point to intermediate fine-tuned
    models as their base. We need to check whether the base_model is a
    well-known canonical model or a derivative.
    """
    base = peft_config.base_model_name_or_path
    canonical_bases = {
        "distilbert-base-uncased",
        "distilbert/distilbert-base-uncased",
        "roberta-base",
        "FacebookAI/roberta-base",
        "bert-base-uncased",
        "google-bert/bert-base-uncased",
    }

    # If it's already canonical, use it directly
    for cb in canonical_bases:
        if base == cb or base.endswith("/" + cb.split("/")[-1]):
            return base

    # For transfer-chain adapters (e.g., JB173/distilbert-base-uncased-finetuned-emotion),
    # try to load the actual intermediate model. If it fails, fall back to canonical.
    return base


def run_eval(
    adapter_path: Path,
    ds_key: str,
    base_model_id: str | None = None,
    max_samples: int = MAX_SAMPLES,
) -> dict:
    """Run a lightweight eval. Returns evidence dict."""
    texts, labels, ds_info = load_eval_data(ds_key, max_samples)
    n_samples = len(labels)

    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    model_id = base_model_id or resolve_true_base_model(peft_config)
    num_labels = ds_info["num_labels"]

    log.info("  loading base model: %s (num_labels=%d)", model_id, num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # --- Base model eval ---
    log.info("  evaluating base model...")
    t0 = time.time()
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels
        )
        base_model.eval()
        base_preds = _batch_predict(base_model, tokenizer, texts, batch_size=32)
        base_acc = accuracy_score(labels, base_preds)
        base_time = time.time() - t0
        log.info("  base accuracy: %.4f (%.1fs)", base_acc, base_time)
    except Exception as e:
        log.warning("  base eval failed: %s — using 0.0", e)
        base_acc = 0.0
        base_preds = []

    # --- Adapter model eval ---
    log.info("  evaluating adapter...")
    t0 = time.time()
    try:
        adapter_model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels
        )
        adapter_model = PeftModel.from_pretrained(adapter_model, str(adapter_path))
        adapter_model.eval()
        adapter_preds = _batch_predict(adapter_model, tokenizer, texts, batch_size=32)
        adapter_acc = accuracy_score(labels, adapter_preds)
        adapter_time = time.time() - t0
        log.info("  adapter accuracy: %.4f (%.1fs)", adapter_acc, adapter_time)
    except Exception as e:
        log.error("  adapter eval FAILED: %s", e)
        return {"error": str(e), "adapter_path": str(adapter_path)}

    # --- Compute metrics ---
    beats_base = adapter_acc > base_acc
    delta = adapter_acc - base_acc

    result = {
        "adapter_path": str(adapter_path),
        "base_model": model_id,
        "eval_dataset": ds_key,
        "eval_split": ds_info["split"],
        "n_samples": n_samples,
        "metric_name": "accuracy",
        "lower_is_better": False,
        "adapter_score": round(adapter_acc, 4),
        "base_score": round(base_acc, 4),
        "beats_base": beats_base,
        "delta": round(delta, 4),
    }

    # Add F1 for binary tasks
    if num_labels == 2 and adapter_preds:
        adapter_f1 = f1_score(labels, adapter_preds, average="binary")
        result["adapter_f1"] = round(adapter_f1, 4)
        if base_preds:
            base_f1 = f1_score(labels, base_preds, average="binary")
            result["base_f1"] = round(base_f1, 4)

    return result


def _batch_predict(
    model: torch.nn.Module,
    tokenizer,
    texts: list,
    batch_size: int = 32,
) -> list[int]:
    """Run batched inference on CPU."""
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Handle single text or text pairs
        if isinstance(batch[0], tuple):
            texts_a, texts_b = zip(*batch)
            inputs = tokenizer(
                list(texts_a), list(texts_b),
                padding=True, truncation=True, max_length=256, return_tensors="pt",
            )
        else:
            inputs = tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt",
            )

        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).tolist()
            preds.extend(batch_preds)
    return preds


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <inventory_dir_name>")
        sys.exit(1)

    inv_name = sys.argv[1]
    inv_dir = FIELD_TRIALS_DIR / inv_name
    manifest = json.loads((inv_dir / "manifest.json").read_text())

    log.info("=== Evidence Bootstrap: %s ===", manifest["inventory_id"])

    adapter_cache = inv_dir / "adapter_cache"
    evidence_dir = inv_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    results = []
    for entry in manifest["adapters"]:
        adapter_id = entry["adapter_id"]
        local_name = adapter_id.replace("/", "__")
        adapter_path = adapter_cache / local_name
        evidence_path = evidence_dir / f"{local_name}.json"

        if evidence_path.exists():
            log.info("cached  %s", adapter_id)
            results.append(json.loads(evidence_path.read_text()))
            continue

        # Download if needed
        if not adapter_path.exists():
            from huggingface_hub import snapshot_download
            log.info("downloading  %s", adapter_id)
            snapshot_download(repo_id=adapter_id, local_dir=str(adapter_path))

        ds_key = resolve_dataset(entry.get("dataset", "unknown"))
        log.info("evaluating  %s  on  %s", adapter_id, ds_key)
        t0 = time.time()

        evidence = run_eval(adapter_path, ds_key, base_model_id=entry.get("base_model"))
        evidence["adapter_id"] = adapter_id
        evidence["manifest_task"] = entry.get("task", "unknown")
        evidence["elapsed_seconds"] = round(time.time() - t0, 1)

        evidence_path.write_text(json.dumps(evidence, indent=2))
        results.append(evidence)
        log.info("  total: %.1fs", time.time() - t0)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Adapter':25s} | {'Dataset':20s} | {'Adapter':8s} | {'Base':8s} | {'Delta':8s} | {'Beats?'}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r.get('adapter_id','?')[:25]:25s} | ERROR: {r['error'][:50]}")
            continue
        name = r["adapter_id"].split("/")[-1][:25]
        ds = r.get("eval_dataset", "?")[:20]
        a_score = r.get("adapter_score", 0)
        b_score = r.get("base_score", 0)
        delta = r.get("delta", 0)
        beats = "✓" if r.get("beats_base") else "✗"
        print(f"{name:25s} | {ds:20s} | {a_score:8.4f} | {b_score:8.4f} | {delta:+8.4f} | {beats}")
    print("=" * 90)

    log.info("=== Evidence bootstrap complete ===")


if __name__ == "__main__":
    main()
