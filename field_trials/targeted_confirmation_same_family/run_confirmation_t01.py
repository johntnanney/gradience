#!/usr/bin/env python3
"""Targeted Confirmation Run 1 — Same-Family Routing.

Confirms that the new same-family routing distinguishes:
- exact same-task (retained)
- same-family cross-dataset (same_task_priority via same_family)
- true cross-task (cross_task_caution)

Workflow: download adapters -> bootstrap evidence -> audit -> merge-audit ->
          build action plan -> lock stance -> evaluate 3 merges.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence_bootstrap import run_eval
from run_phase2_eval import PairSpec, run_pair

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("confirmation_t01")

INVENTORY_DIR = Path(__file__).resolve().parent / "inventory_t01"
EVIDENCE_DIR = INVENTORY_DIR / "evidence"
PREFLIGHT_DIR = INVENTORY_DIR / "preflight"
ADAPTER_CACHE = INVENTORY_DIR / "adapter_cache"

BASE_MODEL = "distilbert-base-uncased"

ADAPTERS = [
    {"adapter_id": "myselfmankar/distilbert-base-sst2-lora",
     "dataset": "sst2",
     "local_name": "myselfmankar__distilbert-base-sst2-lora"},
    {"adapter_id": "rambodazimi/distilbert-base-uncased-finetuned-LoRA-SST2",
     "dataset": "sst2",
     "local_name": "rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2"},
    {"adapter_id": "dipanjanS/distilbert-lora-finetuned-unmerged-imdb-sentiment",
     "dataset": "imdb",
     "local_name": "dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment"},
    {"adapter_id": "TransferGraph/JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news",
     "dataset": "ag_news",
     "local_name": "TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news"},
]


def download_adapters():
    """Download adapters to local cache."""
    from huggingface_hub import snapshot_download
    for a in ADAPTERS:
        local_dir = ADAPTER_CACHE / a["local_name"]
        if local_dir.exists() and any(local_dir.iterdir()):
            log.info("  cached: %s", a["adapter_id"])
            continue
        log.info("  downloading: %s", a["adapter_id"])
        snapshot_download(a["adapter_id"], local_dir=str(local_dir))


def bootstrap_evidence():
    """Run evidence bootstrap for each adapter."""
    for a in ADAPTERS:
        evidence_path = EVIDENCE_DIR / f"{a['local_name']}.json"
        if evidence_path.exists():
            existing = json.loads(evidence_path.read_text())
            if existing.get("adapter_score") is not None:
                log.info("  evidence cached: %s (score=%.3f)", a["adapter_id"], existing["adapter_score"])
                continue

        log.info("  bootstrapping: %s on %s", a["adapter_id"], a["dataset"])
        adapter_dir = ADAPTER_CACHE / a["local_name"]
        result = run_eval(adapter_dir, a["dataset"], base_model_id=BASE_MODEL)
        result["adapter_id"] = a["adapter_id"]
        evidence_path.write_text(json.dumps(result, indent=2))
        log.info("  -> score=%.3f, base=%.3f, delta=%.3f",
                 result["adapter_score"], result["base_score"], result["delta"])


def run_gradience_preflight():
    """Run Gradience audits and build action plan. Returns (qa_artifacts, qa_reports, action_plan)."""
    from gradience.api import audit_adapter
    from gradience.vnext.merge import merge_audit
    from gradience.vnext.merge.qa_report import build_qa_report, format_qa_report
    from gradience.vnext.inventory.summary import build_action_plan, format_action_plan

    # Audit each adapter
    qa_artifacts = []
    qa_by_local = {}
    for a in ADAPTERS:
        adapter_dir = ADAPTER_CACHE / a["local_name"]
        evidence = json.loads((EVIDENCE_DIR / f"{a['local_name']}.json").read_text())

        log.info("  auditing: %s", a["adapter_id"])
        artifact = audit_adapter(
            peft_dir=str(adapter_dir),
            eval_dataset=a["dataset"],
            adapter_score=evidence["adapter_score"],
            base_score=evidence["base_score"],
            lower_is_better=False,
            metric_name="accuracy",
        )
        qa_artifacts.append(artifact)
        qa_by_local[a["local_name"]] = artifact

        qa_path = PREFLIGHT_DIR / f"qa_{a['local_name']}.json"
        qa_path.write_text(json.dumps(artifact.to_dict(), indent=2))
        log.info("    status=%s", artifact.status.value)

    # Merge-audit each pair
    pair_configs = [
        ("myselfmankar__distilbert-base-sst2-lora",
         "rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2",
         "exact_same_task"),
        ("myselfmankar__distilbert-base-sst2-lora",
         "dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment",
         "same_family_cross_dataset"),
        ("myselfmankar__distilbert-base-sst2-lora",
         "TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news",
         "true_cross_task"),
    ]

    qa_report_objects = []
    for a_local, b_local, pair_type in pair_configs:
        a_dir = str(ADAPTER_CACHE / a_local)
        b_dir = str(ADAPTER_CACHE / b_local)

        log.info("  merge_audit: %s (%s)", pair_type, f"{a_local.split('__')[-1][:15]} x {b_local.split('__')[-1][:15]}")
        report = merge_audit(
            a_dir, b_dir, verbose=False,
            source_qa_a=qa_by_local[a_local].to_qa_result(),
            source_qa_b=qa_by_local[b_local].to_qa_result(),
        )

        qa_report = build_qa_report(report)
        qa_report_objects.append(qa_report)

        # Save JSON + formatted text
        (PREFLIGHT_DIR / f"merge_{pair_type}.json").write_text(json.dumps(qa_report.to_dict(), indent=2))
        (PREFLIGHT_DIR / f"merge_{pair_type}_report.txt").write_text(format_qa_report(qa_report))

        log.info("    risk=%s, strategy=%s, task_relationship=%s",
                 qa_report.pair_risk, qa_report.recommended_strategy, qa_report.task_relationship)

    # Build action plan
    log.info("  building action plan...")
    action_plan = build_action_plan(qa_artifacts, qa_report_objects)
    formatted = format_action_plan(action_plan)
    (PREFLIGHT_DIR / "action_plan.txt").write_text(formatted)
    log.info("\n%s", formatted)

    # Save stance lock
    stance = {
        "locked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "same_task_priority": list(action_plan.same_task_priority),
        "cross_task_caution": list(action_plan.cross_task_caution),
        "exclude": list(action_plan.exclude),
        "near_miss_candidates": list(action_plan.near_miss_candidates),
        "near_miss_severity": [list(s) for s in action_plan.near_miss_severity],
        "retained_pair_detail": [list(d) for d in action_plan.retained_pair_detail],
        "summary_line": action_plan.summary_line,
    }
    (PREFLIGHT_DIR / "stance_lock.json").write_text(json.dumps(stance, indent=2))

    return qa_artifacts, qa_report_objects, action_plan


def evaluate_merges():
    """Evaluate the 3 planned merges."""
    pairs = [
        PairSpec(
            name="t01_retained_sst2_x_sst2",
            adapter_a_dir=ADAPTER_CACHE / "myselfmankar__distilbert-base-sst2-lora",
            adapter_b_dir=ADAPTER_CACHE / "rambodazimi__distilbert-base-uncased-finetuned-LoRA-SST2",
            strategy="audit_aware",
            eval_dataset="sst2",
            base_model_id=BASE_MODEL,
            num_labels=2,
            role="retained",
            notes="Exact same-task baseline (SST-2 x SST-2).",
        ),
        PairSpec(
            name="t01_same_family_sst2_x_imdb",
            adapter_a_dir=ADAPTER_CACHE / "myselfmankar__distilbert-base-sst2-lora",
            adapter_b_dir=ADAPTER_CACHE / "dipanjanS__distilbert-lora-finetuned-unmerged-imdb-sentiment",
            strategy="audit_aware",
            eval_dataset="sst2",
            base_model_id=BASE_MODEL,
            num_labels=2,
            role="same_family",
            notes="KEY: Same-family cross-dataset (SST-2 x IMDB), eval on SST-2.",
        ),
        PairSpec(
            name="t01_cross_task_sst2_x_agnews",
            adapter_a_dir=ADAPTER_CACHE / "myselfmankar__distilbert-base-sst2-lora",
            adapter_b_dir=ADAPTER_CACHE / "TransferGraph__JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-ag_news",
            strategy="audit_aware",
            eval_dataset="sst2",
            base_model_id=BASE_MODEL,
            num_labels=2,
            role="cross_task_control",
            notes="True cross-task control (SST-2 x ag_news), eval on SST-2.",
        ),
    ]

    output_dir = INVENTORY_DIR.parent / "eval_t01"
    output_dir.mkdir(exist_ok=True)

    results = []
    for pair in pairs:
        try:
            result = run_pair(pair, output_dir)
            results.append(result)
            log.info("  %s: merged_acc=%.4f", pair.name, result["merged_accuracy"])
        except Exception as e:
            log.error("FAILED: %s -- %s", pair.name, e)
            import traceback
            traceback.print_exc()
            results.append({"pair_name": pair.name, "role": pair.role, "error": str(e)})

    (INVENTORY_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
    return results


def print_summary(results):
    """Print compact summary table."""
    print(f"\n{'='*100}")
    print("Targeted Confirmation T01 -- Same-Family Routing")
    print(f"{'='*100}")
    print(f"{'Pair':45s} | {'Role':18s} | {'Strategy':16s} | {'Merged Acc':>10s}")
    print(f"{'-'*100}")
    for r in results:
        if "error" in r:
            print(f"{r['pair_name']:45s} | {r['role']:18s} | ERROR: {r['error'][:40]}")
        else:
            print(f"{r['pair_name']:45s} | {r['role']:18s} | {r['strategy']:16s} | {r['merged_accuracy']:10.4f}")
    print(f"\nSource: myselfmankar SST-2=0.884, rambodazimi SST-2=0.902, dipanjanS IMDB=0.876, JB173 ag_news=0.912")
    print(f"{'='*100}")


def main():
    log.info("=== Targeted Confirmation T01: Same-Family Routing ===")
    download_adapters()
    bootstrap_evidence()
    qa_artifacts, qa_reports, action_plan = run_gradience_preflight()
    results = evaluate_merges()
    print_summary(results)


if __name__ == "__main__":
    main()
