#!/usr/bin/env python3
"""Phase 2 evaluation — merge retained & control pairs, evaluate merged adapters.

For each pair:
1. Run merge_audit() to get the compatibility report
2. Plan the merge using the recommended strategy (from preflight)
3. Execute the merge to produce a PEFT-compatible merged adapter
4. Evaluate the merged adapter on the task dataset
5. Compare to source adapter scores (from evidence bootstrap)

Usage:
    python field_trials/run_phase2_eval.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import Gradience merge pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gradience.vnext.merge import merge_audit, load_adapter
from gradience.vnext.merge.plan import plan_from_audit
from gradience.vnext.merge.executor import execute_merge

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("phase2")

FIELD_TRIALS_DIR = Path(__file__).resolve().parent
MAX_SAMPLES = 500


# ---------------------------------------------------------------------------
# Dataset registry (reused from evidence_bootstrap.py)
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, dict] = {
    "imdb": {
        "hf_name": "imdb", "hf_config": None, "split": "test",
        "text_col": "text", "label_col": "label", "num_labels": 2,
    },
    "sst2": {
        "hf_name": "sst2", "hf_config": None, "split": "validation",
        "text_col": "sentence", "label_col": "label", "num_labels": 2,
    },
    "tweet_eval/emotion": {
        "hf_name": "tweet_eval", "hf_config": "emotion", "split": "test",
        "text_col": "text", "label_col": "label", "num_labels": 4,
    },
    "ag_news": {
        "hf_name": "ag_news", "hf_config": None, "split": "test",
        "text_col": "text", "label_col": "label", "num_labels": 4,
    },
    "tweet_eval/hate": {
        "hf_name": "tweet_eval", "hf_config": "hate", "split": "test",
        "text_col": "text", "label_col": "label", "num_labels": 2,
    },
    "tweet_eval/irony": {
        "hf_name": "tweet_eval", "hf_config": "irony", "split": "test",
        "text_col": "text", "label_col": "label", "num_labels": 2,
    },
    "mnli": {
        "hf_name": "nyu-mll/multi_nli", "hf_config": None, "split": "validation_matched",
        "text_col": "premise", "text_col_b": "hypothesis", "label_col": "label", "num_labels": 3,
    },
    "yelp_polarity": {
        "hf_name": "yelp_polarity", "hf_config": None, "split": "test",
        "text_col": "text", "label_col": "label", "num_labels": 2,
    },
    "amazon_polarity": {
        "hf_name": "amazon_polarity", "hf_config": None, "split": "test",
        "text_col": "content", "label_col": "label", "num_labels": 2,
    },
}


def load_eval_data(ds_key: str, max_samples: int = MAX_SAMPLES):
    """Load a small eval slice.

    Uses shuffling with a fixed seed to avoid label-sorted splits.
    """
    info = DATASET_REGISTRY[ds_key]
    ds = load_dataset(info["hf_name"], info["hf_config"], split=info["split"])
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    texts = ds[info["text_col"]]
    if "text_col_b" in info:
        texts_b = ds[info["text_col_b"]]
        texts = list(zip(texts, texts_b))
    labels = ds[info["label_col"]]
    return texts, labels, info


def batch_predict(model, tokenizer, texts, batch_size=32) -> list[int]:
    """Run batched inference on CPU."""
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
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


def patch_merged_adapter(merged_dir: Path, source_adapter_dir: Path) -> None:
    """Copy classifier head weights from source adapter into merged adapter.

    Gradience's merge only produces LoRA weights. But the source adapters
    often include classifier/pre_classifier weights in their safetensors.
    PEFT expects these keys when loading. We copy them from adapter A.
    """
    from safetensors.torch import load_file, save_file

    merged_path = merged_dir / "adapter_model.safetensors"
    merged_weights = load_file(str(merged_path))

    # Find source adapter weights file
    source_path = source_adapter_dir / "adapter_model.safetensors"
    if not source_path.exists():
        source_path = source_adapter_dir / "adapter_model.bin"
        if source_path.exists():
            source_weights = torch.load(str(source_path), map_location="cpu", weights_only=True)
        else:
            return  # no source weights to copy
    else:
        source_weights = load_file(str(source_path))

    # Copy non-LoRA keys (classifier, pre_classifier, etc.)
    added = []
    for key, tensor in source_weights.items():
        if "lora_" not in key and key not in merged_weights:
            merged_weights[key] = tensor
            added.append(key)

    if added:
        save_file(merged_weights, str(merged_path))
        log.info("  patched merged adapter: added %d classifier keys from source A", len(added))


def eval_merged_adapter(
    merged_dir: Path,
    base_model_id: str,
    ds_key: str,
    num_labels: int,
    source_adapter_dir: Path | None = None,
) -> dict:
    """Evaluate a merged adapter on a dataset."""
    texts, labels, ds_info = load_eval_data(ds_key)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    t0 = time.time()

    # Patch merged adapter with classifier head from source adapter A
    if source_adapter_dir is not None:
        patch_merged_adapter(merged_dir, source_adapter_dir)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, num_labels=num_labels,
    )
    model = PeftModel.from_pretrained(base_model, str(merged_dir))
    model.eval()

    preds = batch_predict(model, tokenizer, texts)
    elapsed = time.time() - t0
    acc = accuracy_score(labels, preds)

    # Clean up
    del model, base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "accuracy": round(acc, 4),
        "n_samples": len(labels),
        "eval_time_s": round(elapsed, 1),
    }


@dataclass
class PairSpec:
    """Specification for a pair to merge and evaluate."""
    name: str
    adapter_a_dir: Path
    adapter_b_dir: Path
    strategy: str
    eval_dataset: str  # which dataset to evaluate on
    base_model_id: str
    num_labels: int
    role: str  # "retained", "control", "near-miss"
    notes: str = ""


def get_evidence_score(evidence_dir: Path, adapter_local_name: str) -> float | None:
    """Look up adapter score from evidence bootstrap."""
    epath = evidence_dir / f"{adapter_local_name}.json"
    if epath.exists():
        ev = json.loads(epath.read_text())
        return ev.get("adapter_score")
    return None


def run_pair(pair: PairSpec, output_base: Path) -> dict:
    """Merge a pair and evaluate the result."""
    log.info("=== %s [%s] ===", pair.name, pair.role)

    merge_dir = output_base / pair.name
    merge_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: merge_audit
    log.info("  merge_audit...")
    t0 = time.time()
    report = merge_audit(
        str(pair.adapter_a_dir),
        str(pair.adapter_b_dir),
        verbose=False,
    )
    audit_time = time.time() - t0
    log.info("  audit done (%.1fs) — verdict=%s, score=%.3f",
             audit_time, report.aggregate.overall_verdict, report.aggregate.compatibility_score)

    # Step 2: plan
    log.info("  planning with strategy=%s...", pair.strategy)
    plan = plan_from_audit(
        pair.strategy,
        report,
        str(pair.adapter_a_dir),
        str(pair.adapter_b_dir),
    )

    # Step 3: execute merge
    merged_adapter_dir = merge_dir / "merged_adapter"
    log.info("  executing merge...")
    t0 = time.time()
    merge_result = execute_merge(plan, merged_adapter_dir, verbose=True)
    merge_time = time.time() - t0
    log.info("  merge done (%.1fs) — mean_recon_error=%.4f",
             merge_time, merge_result.mean_reconstruction_error)

    # Step 4: evaluate merged adapter
    # Use adapter A's classifier head (the merged adapter only has LoRA weights)
    log.info("  evaluating merged adapter on %s...", pair.eval_dataset)
    eval_result = eval_merged_adapter(
        merged_adapter_dir,
        pair.base_model_id,
        pair.eval_dataset,
        pair.num_labels,
        source_adapter_dir=pair.adapter_a_dir,
    )
    log.info("  merged accuracy: %.4f (%.1fs)", eval_result["accuracy"], eval_result["eval_time_s"])

    result = {
        "pair_name": pair.name,
        "role": pair.role,
        "strategy": pair.strategy,
        "eval_dataset": pair.eval_dataset,
        "compatibility": str(report.aggregate.overall_verdict),
        "compatibility_score": round(report.aggregate.compatibility_score, 3),
        "mean_reconstruction_error": round(merge_result.mean_reconstruction_error, 4),
        "merged_accuracy": eval_result["accuracy"],
        "n_eval_samples": eval_result["n_samples"],
        "audit_time_s": round(audit_time, 1),
        "merge_time_s": round(merge_time, 1),
        "eval_time_s": eval_result["eval_time_s"],
        "notes": pair.notes,
    }

    # Save per-pair result
    (merge_dir / "eval_result.json").write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Pair definitions
# ---------------------------------------------------------------------------

def build_pilot2_pairs() -> list[PairSpec]:
    """Pilot 2: mixed-task, roberta-base."""
    cache = FIELD_TRIALS_DIR / "inventory_02_mixed_task" / "adapter_cache"

    pairs = [
        # RETAINED: AG News × AG News-formality (same-task, medium risk, norm_equalized)
        PairSpec(
            name="p2_retained_agnews",
            adapter_a_dir=cache / "TransferGraph__roberta-base-finetuned-lora-ag_news",
            adapter_b_dir=cache / "TransferGraph__cointegrated_roberta-base-formality-finetuned-lora-ag_news",
            strategy="norm_equalized",
            eval_dataset="ag_news",
            base_model_id="FacebookAI/roberta-base",
            num_labels=4,
            role="retained",
            notes="Same-task pair. Gradience recommended: norm_equalized, medium risk, partial_redundancy.",
        ),
        # CONTROL 1: AG News × MNLI (cross-task, high risk, audit_aware)
        PairSpec(
            name="p2_control_agnews_x_mnli",
            adapter_a_dir=cache / "TransferGraph__roberta-base-finetuned-lora-ag_news",
            adapter_b_dir=cache / "yuuhan__roberta-base-mnli-lora",
            strategy="audit_aware",
            eval_dataset="ag_news",
            base_model_id="FacebookAI/roberta-base",
            num_labels=4,
            role="control",
            notes="Cross-task, high risk, norm_imbalance. Eval on AG News (adapter A's task).",
        ),
        # CONTROL 2: hate × irony (cross-task, LOW risk — the interesting one)
        PairSpec(
            name="p2_control_hate_x_irony",
            adapter_a_dir=cache / "TransferGraph__roberta-base-finetuned-lora-tweet_eval_hate",
            adapter_b_dir=cache / "TransferGraph__rmihaylov_roberta-base-sentiment-bg-finetuned-lora-tweet_eval_irony",
            strategy="uniform_linear",
            eval_dataset="tweet_eval/hate",
            base_model_id="FacebookAI/roberta-base",
            num_labels=2,
            role="control",
            notes="Cross-task but structurally compatible (low risk, no issues). Eval on hate (adapter A's task). Irony adapter is uncertain.",
        ),
    ]
    return pairs


def build_pilot3_pairs() -> list[PairSpec]:
    """Pilot 3: large mixed-task, distilbert."""
    cache = FIELD_TRIALS_DIR / "inventory_03_large_mixed_task" / "adapter_cache"

    pairs = [
        # RETAINED 1: AG News pair (JB173 × phailyoor) — low risk, linear
        PairSpec(
            name="p3_retained_agnews",
            adapter_a_dir=cache / "TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news",
            adapter_b_dir=cache / "TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-ag_news",
            strategy="uniform_linear",
            eval_dataset="ag_news",
            base_model_id="distilbert/distilbert-base-uncased",
            num_labels=4,
            role="retained",
            notes="Same-task, low risk, no issues, linear merge. Both r=1/alpha=1.",
        ),
        # RETAINED 2: SST-2 pair (myselfmankar × NightPrince) — medium risk, norm_equalized
        PairSpec(
            name="p3_retained_sst2",
            adapter_a_dir=cache / "myselfmankar__distilbert-base-sst2-lora",
            adapter_b_dir=cache / "NightPrince__peft-distilbert-sst2",
            strategy="norm_equalized",
            eval_dataset="sst2",
            base_model_id="distilbert/distilbert-base-uncased",
            num_labels=2,
            role="retained",
            notes="Same-task, medium risk, norm_imbalance (r=16 vs r=8). norm_equalized recommended.",
        ),
        # NEAR-MISS: hate pair (jaesun × Aureliano) — low risk structurally, but Aureliano is flagged_weak
        PairSpec(
            name="p3_nearmiss_hate",
            adapter_a_dir=cache / "TransferGraph__jaesun_distilbert-base-uncased-finetuned-cola-finetuned-lora-tweet_eval_hate",
            adapter_b_dir=cache / "TransferGraph__Aureliano_distilbert-base-uncased-if-finetuned-lora-tweet_eval_hate",
            strategy="uniform_linear",
            eval_dataset="tweet_eval/hate",
            base_model_id="distilbert/distilbert-base-uncased",
            num_labels=2,
            role="near-miss",
            notes="Same-task, low structural risk, but Aureliano flagged_weak (underperforms base). Tests whether evidence gate is right to exclude.",
        ),
        # CONTROL: AG News × emotion (cross-task, high risk)
        PairSpec(
            name="p3_control_agnews_x_emotion",
            adapter_a_dir=cache / "TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news",
            adapter_b_dir=cache / "TransferGraph__distilbert-base-uncased-finetuned-lora-tweet_eval_emotion",
            strategy="audit_aware",
            eval_dataset="ag_news",
            base_model_id="distilbert/distilbert-base-uncased",
            num_labels=4,
            role="control",
            notes="Cross-task, high risk. Eval on AG News. Should degrade if Gradience's caution is correct.",
        ),
        # CONTROL: SST-2 × AG News (cross-task, high risk with rank mismatch)
        PairSpec(
            name="p3_control_sst2_x_agnews",
            adapter_a_dir=cache / "myselfmankar__distilbert-base-sst2-lora",
            adapter_b_dir=cache / "TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news",
            strategy="audit_aware",
            eval_dataset="sst2",
            base_model_id="distilbert/distilbert-base-uncased",
            num_labels=2,
            role="control",
            notes="Cross-task, high risk, r=16 vs r=1 mismatch. Eval on SST-2. Tests worst-case cross-task merge.",
        ),
    ]
    return pairs


def print_summary(results: list[dict], pilot: str, evidence_dir: Path, cache_dir: Path):
    """Print a summary comparison table."""
    print(f"\n{'='*100}")
    print(f"Phase 2 Evaluation — {pilot}")
    print(f"{'='*100}")
    print(f"{'Pair':35s} | {'Role':10s} | {'Strategy':16s} | {'Merged':8s} | {'Notes'}")
    print(f"{'-'*100}")
    for r in results:
        name = r["pair_name"]
        role = r["role"]
        if "error" in r:
            print(f"{name:35s} | {role:10s} | ERROR: {r['error'][:60]}")
            continue
        strat = r["strategy"]
        merged_acc = r["merged_accuracy"]
        notes = r.get("notes", "")[:40]
        print(f"{name:35s} | {role:10s} | {strat:16s} | {merged_acc:8.4f} | {notes}")
    print(f"{'='*100}\n")


def main():
    timestamp = time.strftime("%H%M%S")
    output_base = FIELD_TRIALS_DIR / f"phase2_eval_{timestamp}"
    output_base.mkdir(exist_ok=True)
    log.info("Output directory: %s", output_base)

    all_results = {}

    # --- Pilot 2 ---
    log.info("========== PILOT 2 ==========")
    p2_pairs = build_pilot2_pairs()
    p2_results = []
    for pair in p2_pairs:
        try:
            result = run_pair(pair, output_base)
            p2_results.append(result)
        except Exception as e:
            log.error("FAILED: %s — %s", pair.name, e)
            p2_results.append({"pair_name": pair.name, "role": pair.role, "error": str(e)})

    all_results["pilot_2"] = p2_results
    p2_evidence = FIELD_TRIALS_DIR / "inventory_02_mixed_task" / "evidence"
    p2_cache = FIELD_TRIALS_DIR / "inventory_02_mixed_task" / "adapter_cache"
    print_summary(p2_results, "Pilot 2 (roberta-base, mixed-task)", p2_evidence, p2_cache)

    # --- Pilot 3 ---
    log.info("========== PILOT 3 ==========")
    p3_pairs = build_pilot3_pairs()
    p3_results = []
    for pair in p3_pairs:
        try:
            result = run_pair(pair, output_base)
            p3_results.append(result)
        except Exception as e:
            log.error("FAILED: %s — %s", pair.name, e)
            p3_results.append({"pair_name": pair.name, "role": pair.role, "error": str(e)})

    all_results["pilot_3"] = p3_results
    p3_evidence = FIELD_TRIALS_DIR / "inventory_03_large_mixed_task" / "evidence"
    p3_cache = FIELD_TRIALS_DIR / "inventory_03_large_mixed_task" / "adapter_cache"
    print_summary(p3_results, "Pilot 3 (distilbert, large mixed-task)", p3_evidence, p3_cache)

    # --- Save all results ---
    results_path = output_base / "phase2_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    log.info("All results saved to %s", results_path)


if __name__ == "__main__":
    main()
