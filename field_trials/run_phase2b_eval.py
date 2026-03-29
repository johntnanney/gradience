#!/usr/bin/env python3
"""Phase 2b: Near-miss confirmation pass.

Merges and evaluates retained, near-miss, and control pairs across
inventories 04 (distilbert irony cluster) and 05 (bert hate+emotion).

Uses the same merge→patch→eval pipeline as Phase 2.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("phase2b")

FIELD_TRIALS_DIR = Path(__file__).resolve().parent
MAX_SAMPLES = 500


# ---------------------------------------------------------------------------
# Dataset registry (reused from evidence_bootstrap)
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "tweet_eval/irony": {
        "hf_name": "tweet_eval", "hf_config": "irony",
        "split": "test", "text_col": "text", "label_col": "label",
        "num_labels": 2,
    },
    "tweet_eval/emotion": {
        "hf_name": "tweet_eval", "hf_config": "emotion",
        "split": "test", "text_col": "text", "label_col": "label",
        "num_labels": 4,
    },
    "tweet_eval/hate": {
        "hf_name": "tweet_eval", "hf_config": "hate",
        "split": "test", "text_col": "text", "label_col": "label",
        "num_labels": 2,
    },
    "ag_news": {
        "hf_name": "ag_news", "hf_config": None,
        "split": "test", "text_col": "text", "label_col": "label",
        "num_labels": 4,
    },
}


@dataclass
class PairSpec:
    label: str
    adapter_a: Path
    adapter_b: Path
    category: str  # "retained", "near_miss", "control"
    strategy: str
    eval_dataset: str
    base_model: str
    source_a_score: float
    source_b_score: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Merge helpers (same as Phase 2)
# ---------------------------------------------------------------------------

def merge_pair(adapter_a: Path, adapter_b: Path, strategy: str, out_dir: Path) -> Path:
    """Run Gradience merge pipeline: audit → plan → execute."""
    from gradience.vnext.merge import merge_audit, plan_from_audit, execute_merge

    log.info("  merge_audit...")
    report = merge_audit(adapter_a, adapter_b)

    log.info("  plan (strategy=%s)...", strategy)
    plan = plan_from_audit(strategy, report, str(adapter_a), str(adapter_b))

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = out_dir / "merged_adapter"
    merged_dir.mkdir(exist_ok=True)

    log.info("  execute_merge → %s", merged_dir)
    execute_merge(plan, merged_dir)

    return merged_dir


def patch_merged_adapter(merged_dir: Path, source_dir: Path) -> None:
    """Copy non-LoRA keys (classifier head) from source into merged adapter."""
    merged_sf = merged_dir / "adapter_model.safetensors"
    source_sf = source_dir / "adapter_model.safetensors"

    if not source_sf.exists() or not merged_sf.exists():
        log.warning("  cannot patch — missing safetensors")
        return

    # Load merged tensors
    merged_tensors = {}
    with safe_open(str(merged_sf), framework="pt") as f:
        for k in f.keys():
            merged_tensors[k] = f.get_tensor(k)

    # Copy non-LoRA keys from source
    patched = 0
    with safe_open(str(source_sf), framework="pt") as f:
        for k in f.keys():
            if "lora" not in k.lower() and k not in merged_tensors:
                merged_tensors[k] = f.get_tensor(k)
                patched += 1

    if patched:
        save_file(merged_tensors, str(merged_sf))
        log.info("  patched %d classifier keys from source", patched)

    # Copy adapter_config.json from source
    src_cfg = source_dir / "adapter_config.json"
    dst_cfg = merged_dir / "adapter_config.json"
    if src_cfg.exists() and not dst_cfg.exists():
        import shutil
        shutil.copy2(src_cfg, dst_cfg)


def eval_merged_adapter(
    merged_dir: Path,
    base_model_id: str,
    ds_key: str,
    max_samples: int = MAX_SAMPLES,
) -> float:
    """Load and evaluate a merged adapter. Returns accuracy."""
    info = DATASET_REGISTRY[ds_key]
    ds = load_dataset(info["hf_name"], info["hf_config"],
                      split=f"{info['split']}[:{max_samples}]")
    texts = ds[info["text_col"]]
    labels = ds[info["label_col"]]

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, num_labels=info["num_labels"]
    )
    model = PeftModel.from_pretrained(model, str(merged_dir))
    model.eval()

    preds = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            preds.extend(torch.argmax(outputs.logits, dim=-1).tolist())

    return round(accuracy_score(labels, preds), 4)


# ---------------------------------------------------------------------------
# Pair definitions
# ---------------------------------------------------------------------------

def build_pairs() -> list[PairSpec]:
    """Build Phase 2b pair list."""
    inv04 = FIELD_TRIALS_DIR / "inventory_04_distilbert_irony_cluster" / "adapter_cache"
    inv05 = FIELD_TRIALS_DIR / "inventory_05_bert_hate_emotion" / "adapter_cache"

    # Short aliases for readability
    def d(name: str) -> Path:
        return inv04 / name

    def b(name: str) -> Path:
        return inv05 / name

    pairs = []

    # ===== RETAINED PAIRS =====

    # R1: Inv04 — JB173 irony × neibla irony (medium, norm_equalized)
    pairs.append(PairSpec(
        label="irony_JB173_x_neibla",
        adapter_a=d("TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony"),
        adapter_b=d("TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony"),
        category="retained",
        strategy="norm_equalized",
        eval_dataset="tweet_eval/irony",
        base_model="distilbert/distilbert-base-uncased",
        source_a_score=0.632, source_b_score=0.620,
        notes="Both eligible. Irony same-task retained pair.",
    ))

    # R2: Inv04 — vaariis irony × neibla irony (low, linear)
    pairs.append(PairSpec(
        label="irony_vaariis_x_neibla",
        adapter_a=d("TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony"),
        adapter_b=d("TransferGraph__neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony"),
        category="retained",
        strategy="uniform_linear",
        eval_dataset="tweet_eval/irony",
        base_model="distilbert/distilbert-base-uncased",
        source_a_score=0.640, source_b_score=0.620,
        notes="Both eligible. Irony same-task, cleanest structural profile.",
    ))

    # R3: Inv05 — TG base hate × HateXplain hate (low, linear)
    pairs.append(PairSpec(
        label="bert_hate_TGbase_x_hatexplain",
        adapter_a=b("TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate"),
        adapter_b=b("TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate"),
        category="retained",
        strategy="uniform_linear",
        eval_dataset="tweet_eval/hate",
        base_model="google-bert/bert-base-uncased",
        source_a_score=0.514, source_b_score=0.588,
        notes="Both eligible. Hate same-task on new backbone (bert-base-uncased).",
    ))

    # R4: Inv05 — TG base emotion × fabriceyhc emotion (low, linear)
    pairs.append(PairSpec(
        label="bert_emotion_TGbase_x_fabriceyhc",
        adapter_a=b("TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion"),
        adapter_b=b("TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion"),
        category="retained",
        strategy="uniform_linear",
        eval_dataset="tweet_eval/emotion",
        base_model="google-bert/bert-base-uncased",
        source_a_score=0.752, source_b_score=0.204,
        notes="Both eligible, but fabriceyhc is very weak (0.204 on 4-class). Tests retained pair with marginal partner.",
    ))

    # ===== NEAR-MISS PAIRS =====

    # NM1: Inv04 — JB173 irony × phailyoor irony (low, linear — phailyoor flagged_weak)
    pairs.append(PairSpec(
        label="irony_JB173_x_phailyoor_NM",
        adapter_a=d("TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony"),
        adapter_b=d("TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony"),
        category="near_miss",
        strategy="uniform_linear",
        eval_dataset="tweet_eval/irony",
        base_model="distilbert/distilbert-base-uncased",
        source_a_score=0.632, source_b_score=0.618,
        notes="phailyoor flagged_weak (delta -0.004). Structurally low risk.",
    ))

    # NM2: Inv04 — vaariis irony × phailyoor irony (low, linear)
    pairs.append(PairSpec(
        label="irony_vaariis_x_phailyoor_NM",
        adapter_a=d("TransferGraph__vaariis_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony"),
        adapter_b=d("TransferGraph__phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony"),
        category="near_miss",
        strategy="uniform_linear",
        eval_dataset="tweet_eval/irony",
        base_model="distilbert/distilbert-base-uncased",
        source_a_score=0.640, source_b_score=0.618,
        notes="phailyoor flagged_weak. Structurally low risk.",
    ))

    # NM3: Inv05 — TG base hate × aviator-neural hate (low, linear — aviator flagged_weak)
    pairs.append(PairSpec(
        label="bert_hate_TGbase_x_aviator_NM",
        adapter_a=b("TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_hate"),
        adapter_b=b("TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate"),
        category="near_miss",
        strategy="uniform_linear",
        eval_dataset="tweet_eval/hate",
        base_model="google-bert/bert-base-uncased",
        source_a_score=0.514, source_b_score=0.574,
        notes="aviator flagged_weak (delta -0.002). Structurally low risk. New backbone.",
    ))

    # NM4: Inv05 — HateXplain hate × aviator-neural hate (low, linear)
    pairs.append(PairSpec(
        label="bert_hate_hatexplain_x_aviator_NM",
        adapter_a=b("TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_hate"),
        adapter_b=b("TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate"),
        category="near_miss",
        strategy="uniform_linear",
        eval_dataset="tweet_eval/hate",
        base_model="google-bert/bert-base-uncased",
        source_a_score=0.588, source_b_score=0.574,
        notes="aviator flagged_weak. Structurally low risk. New backbone.",
    ))

    # NM5: Inv05 — TG base emotion × HateXplain emotion (medium, norm_equalized — HateXplain flagged_weak)
    pairs.append(PairSpec(
        label="bert_emotion_TGbase_x_hatexplain_NM",
        adapter_a=b("TransferGraph__bert-base-uncased-finetuned-lora-tweet_eval_emotion"),
        adapter_b=b("TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion"),
        category="near_miss",
        strategy="norm_equalized",
        eval_dataset="tweet_eval/emotion",
        base_model="google-bert/bert-base-uncased",
        source_a_score=0.752, source_b_score=0.136,
        notes="HateXplain emotion flagged_weak (delta -0.150). Medium risk. Tests deeply weak source.",
    ))

    # NM6: Inv05 — fabriceyhc emotion × HateXplain emotion (low, linear)
    pairs.append(PairSpec(
        label="bert_emotion_fabriceyhc_x_hatexplain_NM",
        adapter_a=b("TransferGraph__fabriceyhc_bert-base-uncased-imdb-finetuned-lora-tweet_eval_emotion"),
        adapter_b=b("TransferGraph__Hate-speech-CNERG_bert-base-uncased-hatexplain-rationale-two-finetuned-lora-tweet_eval_emotion"),
        category="near_miss",
        strategy="uniform_linear",
        eval_dataset="tweet_eval/emotion",
        base_model="google-bert/bert-base-uncased",
        source_a_score=0.204, source_b_score=0.136,
        notes="HateXplain flagged_weak. Both sources are weak (one eligible, one not). Double-weak near-miss.",
    ))

    # ===== CONTROL: cross-task excluded pair =====

    # C1: Inv05 — TG base ag_news × aviator-neural hate (cross-task, aviator weak)
    pairs.append(PairSpec(
        label="bert_cross_agnews_x_aviator_hate_CTRL",
        adapter_a=b("TransferGraph__bert-base-uncased-finetuned-lora-ag_news"),
        adapter_b=b("TransferGraph__aviator-neural_bert-base-uncased-sst2-finetuned-lora-tweet_eval_hate"),
        category="control",
        strategy="audit_aware",
        eval_dataset="ag_news",
        base_model="google-bert/bert-base-uncased",
        source_a_score=0.922, source_b_score=0.574,
        notes="Cross-task (ag_news × hate). One source weak. Should be worst category.",
    ))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%H%M%S")
    out_root = FIELD_TRIALS_DIR / f"phase2b_eval_{ts}"
    out_root.mkdir(exist_ok=True)

    pairs = build_pairs()
    log.info("=== Phase 2b: %d pairs ===", len(pairs))

    results = []
    for i, spec in enumerate(pairs):
        log.info("\n[%d/%d] %s  (%s, %s)", i + 1, len(pairs), spec.label, spec.category, spec.strategy)
        pair_dir = out_root / spec.label
        t0 = time.time()

        try:
            # Step 1: Merge
            merged_dir = merge_pair(spec.adapter_a, spec.adapter_b, spec.strategy, pair_dir)

            # Step 2: Patch classifier head from source A
            patch_merged_adapter(merged_dir, spec.adapter_a)

            # Step 3: Evaluate
            log.info("  evaluating on %s...", spec.eval_dataset)
            merged_acc = eval_merged_adapter(
                merged_dir, spec.base_model, spec.eval_dataset
            )
            elapsed = time.time() - t0

            best_source = max(spec.source_a_score, spec.source_b_score)
            avg_source = (spec.source_a_score + spec.source_b_score) / 2
            delta_best = round(merged_acc - best_source, 4)
            delta_avg = round(merged_acc - avg_source, 4)

            result = {
                "label": spec.label,
                "category": spec.category,
                "strategy": spec.strategy,
                "eval_dataset": spec.eval_dataset,
                "base_model": spec.base_model,
                "source_a_score": spec.source_a_score,
                "source_b_score": spec.source_b_score,
                "best_source": best_source,
                "avg_source": round(avg_source, 4),
                "merged_score": merged_acc,
                "delta_vs_best": delta_best,
                "delta_vs_avg": delta_avg,
                "notes": spec.notes,
                "elapsed_seconds": round(elapsed, 1),
            }
            results.append(result)
            log.info("  merged=%.4f  best_src=%.4f  delta=%+.4f  (%.1fs)",
                     merged_acc, best_source, delta_best, elapsed)

        except Exception as e:
            log.error("  FAILED: %s", e)
            results.append({
                "label": spec.label,
                "category": spec.category,
                "error": str(e),
            })

    # Save results
    results_path = out_root / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info("\nResults saved to %s", results_path)

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Label':45s} | {'Cat':10s} | {'Strategy':16s} | {'Merged':8s} | {'BestSrc':8s} | {'Δ best':8s} | {'Δ avg':8s}")
    print("-" * 120)
    for r in results:
        if "error" in r:
            print(f"{r['label']:45s} | {r['category']:10s} | ERROR: {r['error'][:50]}")
            continue
        print(f"{r['label']:45s} | {r['category']:10s} | {r['strategy']:16s} | "
              f"{r['merged_score']:8.4f} | {r['best_source']:8.4f} | "
              f"{r['delta_vs_best']:+8.4f} | {r['delta_vs_avg']:+8.4f}")
    print("=" * 120)

    # Category summaries
    for cat in ["retained", "near_miss", "control"]:
        cat_results = [r for r in results if r.get("category") == cat and "error" not in r]
        if not cat_results:
            continue
        n = len(cat_results)
        avg_delta_best = sum(r["delta_vs_best"] for r in cat_results) / n
        avg_delta_avg = sum(r["delta_vs_avg"] for r in cat_results) / n
        improvers = sum(1 for r in cat_results if r["delta_vs_best"] > 0)
        print(f"\n{cat.upper()} ({n} pairs): avg Δ vs best = {avg_delta_best:+.4f}, "
              f"avg Δ vs avg = {avg_delta_avg:+.4f}, improvers = {improvers}/{n}")

    log.info("=== Phase 2b complete ===")


if __name__ == "__main__":
    main()
