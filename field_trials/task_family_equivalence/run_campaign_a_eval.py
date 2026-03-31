#!/usr/bin/env python3
"""Campaign A evaluation — merge and evaluate selected pairs for task-family equivalence.

Evaluates:
1. Same-task retained: SST-2 x SST-2 (myselfmankar x rambodazimi), eval on sst2
2. Same-family cross-dataset (KEY): myselfmankar SST-2 x dipanjanS IMDB, eval on sst2 AND imdb
3. Same-family cross-dataset (KEY): rambodazimi SST-2 x wt-golf IMDB, eval on sst2 AND imdb
4. Same-dataset: IMDB x IMDB (dipanjanS x wt-golf), eval on imdb

Usage:
    python field_trials/task_family_equivalence/run_campaign_a_eval.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_phase2_eval import PairSpec, run_pair, load_eval_data, get_evidence_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("campaign_a")

INVENTORY_DIR = Path(__file__).resolve().parent / "inventory_a01"
ADAPTER_CACHE = INVENTORY_DIR / "adapter_cache"
EVIDENCE_DIR = INVENTORY_DIR / "evidence"


def build_campaign_a_pairs() -> list[PairSpec]:
    """Define pairs for Campaign A evaluation."""
    pairs = []

    # --- 1. Same-task retained: SST-2 x SST-2 ---
    pairs.append(PairSpec(
        name="ca_retained_sst2_x_sst2",
        adapter_a_dir=ADAPTER_CACHE / "myselfmankar__distilbert-base-sst2-lora",
        adapter_b_dir=ADAPTER_CACHE / "rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2",
        strategy="audit_aware",
        eval_dataset="sst2",
        base_model_id="distilbert-base-uncased",
        num_labels=2,
        role="retained",
        notes="Same-task baseline. SST-2 x SST-2 (r=16 x r=16). Medium risk, norm_imbalance.",
    ))

    # --- 2. Same-family cross-dataset (KEY): myselfmankar SST-2 x dipanjanS IMDB ---
    # Eval on SST-2 (adapter A's task)
    pairs.append(PairSpec(
        name="ca_family_sst2xIMDB_eval_sst2",
        adapter_a_dir=ADAPTER_CACHE / "myselfmankar__distilbert-base-sst2-lora",
        adapter_b_dir=ADAPTER_CACHE / "dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment",
        strategy="audit_aware",
        eval_dataset="sst2",
        base_model_id="distilbert-base-uncased",
        num_labels=2,
        role="family_test",
        notes="KEY TEST. SST-2 x IMDB, eval on SST-2. Does cross-dataset same-family merge preserve SST-2 performance?",
    ))
    # Eval on IMDB (adapter B's task)
    pairs.append(PairSpec(
        name="ca_family_sst2xIMDB_eval_imdb",
        adapter_a_dir=ADAPTER_CACHE / "myselfmankar__distilbert-base-sst2-lora",
        adapter_b_dir=ADAPTER_CACHE / "dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment",
        strategy="audit_aware",
        eval_dataset="imdb",
        base_model_id="distilbert-base-uncased",
        num_labels=2,
        role="family_test",
        notes="KEY TEST. SST-2 x IMDB, eval on IMDB. Does cross-dataset same-family merge preserve IMDB performance?",
    ))

    # --- 3. Same-family cross-dataset (KEY): rambodazimi SST-2 x wt-golf IMDB ---
    # Eval on SST-2
    pairs.append(PairSpec(
        name="ca_family_sst2xIMDB_v2_eval_sst2",
        adapter_a_dir=ADAPTER_CACHE / "rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2",
        adapter_b_dir=ADAPTER_CACHE / "wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k",
        strategy="uniform_linear",
        eval_dataset="sst2",
        base_model_id="distilbert-base-uncased",
        num_labels=2,
        role="family_test",
        notes="KEY TEST v2. SST-2 x IMDB (low risk pair), eval on SST-2.",
    ))
    # Eval on IMDB
    pairs.append(PairSpec(
        name="ca_family_sst2xIMDB_v2_eval_imdb",
        adapter_a_dir=ADAPTER_CACHE / "rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2",
        adapter_b_dir=ADAPTER_CACHE / "wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k",
        strategy="uniform_linear",
        eval_dataset="imdb",
        base_model_id="distilbert-base-uncased",
        num_labels=2,
        role="family_test",
        notes="KEY TEST v2. SST-2 x IMDB (low risk pair), eval on IMDB.",
    ))

    # --- 4. Same-dataset: IMDB x IMDB ---
    pairs.append(PairSpec(
        name="ca_same_imdb_x_imdb",
        adapter_a_dir=ADAPTER_CACHE / "dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment",
        adapter_b_dir=ADAPTER_CACHE / "wt-golf__distilbert-base-uncased-lora-text-classification-imdb-1k",
        strategy="uniform_linear",
        eval_dataset="imdb",
        base_model_id="distilbert-base-uncased",
        num_labels=2,
        role="retained",
        notes="Same-dataset (IMDB x IMDB). Low risk, linear. Second retained baseline.",
    ))

    return pairs


def print_campaign_a_summary(results: list[dict]):
    """Print Campaign A summary with source scores for comparison."""
    print(f"\n{'='*120}")
    print("Campaign A — Task-Family Equivalence Evaluation")
    print(f"{'='*120}")
    print(f"{'Pair':40s} | {'Role':12s} | {'Eval On':8s} | {'Strategy':16s} | {'Merged':8s} | {'Notes'}")
    print(f"{'-'*120}")
    for r in results:
        name = r["pair_name"]
        role = r["role"]
        if "error" in r:
            print(f"{name:40s} | {role:12s} | ERROR: {r['error'][:60]}")
            continue
        strat = r["strategy"]
        merged_acc = r["merged_accuracy"]
        ds = r["eval_dataset"]
        notes = r.get("notes", "")[:40]
        print(f"{name:40s} | {role:12s} | {ds:8s} | {strat:16s} | {merged_acc:8.4f} | {notes}")
    print(f"{'='*120}")

    # Source adapter scores for reference
    print(f"\nSource adapter scores (from evidence bootstrap):")
    print(f"  myselfmankar SST-2:   sst2 = 0.884")
    print(f"  rambodazimi  SST-2:   sst2 = 0.902")
    print(f"  dipanjanS    IMDB:    imdb = 0.876")
    print(f"  wt-golf      IMDB:    imdb = 0.856")
    print()


def main():
    timestamp = time.strftime("%H%M%S")
    output_dir = INVENTORY_DIR.parent / f"eval_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    log.info("Output directory: %s", output_dir)

    pairs = build_campaign_a_pairs()
    results = []

    for pair in pairs:
        try:
            result = run_pair(pair, output_dir)
            results.append(result)
        except Exception as e:
            log.error("FAILED: %s — %s", pair.name, e)
            import traceback
            traceback.print_exc()
            results.append({"pair_name": pair.name, "role": pair.role, "error": str(e),
                            "eval_dataset": pair.eval_dataset, "strategy": pair.strategy})

    # Save all results
    results_path = output_dir / "campaign_a_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", results_path)

    print_campaign_a_summary(results)


if __name__ == "__main__":
    main()
