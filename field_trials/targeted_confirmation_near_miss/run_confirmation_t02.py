#!/usr/bin/env python3
"""Targeted Confirmation Run 2 -- Near-Miss Severity Ordering.

Confirms that near-miss severity ordering makes the optional-evaluation
section more practically useful: marginal near-miss at top, substantial at bottom.

Workflow: download adapters -> bootstrap evidence -> audit -> merge-audit ->
          build action plan -> lock stance -> evaluate 4 merges.
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
log = logging.getLogger("confirmation_t02")

INVENTORY_DIR = Path(__file__).resolve().parent / "inventory_t02"
EVIDENCE_DIR = INVENTORY_DIR / "evidence"
PREFLIGHT_DIR = INVENTORY_DIR / "preflight"
ADAPTER_CACHE = INVENTORY_DIR / "adapter_cache"

ADAPTERS = [
    {"adapter_id": "TransferGraph/JB173_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony",
     "dataset": "tweet_eval/irony",
     "local_name": "TransferGraph__JB173_irony"},
    {"adapter_id": "TransferGraph/neibla_distilbert-base-uncased-finetuned-emotion-finetuned-lora-tweet_eval_irony",
     "dataset": "tweet_eval/irony",
     "local_name": "TransferGraph__neibla_irony"},
    {"adapter_id": "TransferGraph/phailyoor_distilbert-base-uncased-finetuned-yahd-finetuned-lora-tweet_eval_irony",
     "dataset": "tweet_eval/irony",
     "local_name": "TransferGraph__phailyoor_irony"},
    {"adapter_id": "TransferGraph/jaesun_distilbert-base-uncased-finetuned-cola-finetuned-lora-tweet_eval_hate",
     "dataset": "tweet_eval/hate",
     "local_name": "TransferGraph__jaesun_hate"},
    {"adapter_id": "TransferGraph/Aureliano_distilbert-base-uncased-if-finetuned-lora-tweet_eval_hate",
     "dataset": "tweet_eval/hate",
     "local_name": "TransferGraph__Aureliano_hate"},
]

# Base model lookup: each adapter's true base model (for num_labels inference)
BASE_MODELS = {
    "tweet_eval/irony": ("distilbert-base-uncased", 2),
    "tweet_eval/hate": ("distilbert-base-uncased", 2),
}


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
                log.info("  evidence cached: %s (score=%.3f, delta=%.3f)",
                         a["adapter_id"], existing["adapter_score"], existing.get("delta", 0))
                continue

        log.info("  bootstrapping: %s on %s", a["adapter_id"], a["dataset"])
        adapter_dir = ADAPTER_CACHE / a["local_name"]
        base_model_id = BASE_MODELS[a["dataset"]][0]
        result = run_eval(adapter_dir, a["dataset"], base_model_id=base_model_id)
        result["adapter_id"] = a["adapter_id"]
        evidence_path.write_text(json.dumps(result, indent=2))
        log.info("  -> score=%.3f, base=%.3f, delta=%.3f",
                 result["adapter_score"], result["base_score"], result["delta"])


def run_gradience_preflight():
    """Run Gradience audits and build action plan."""
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
        log.info("    status=%s (delta=%.3f)", artifact.status.value,
                 (evidence["adapter_score"] - evidence["base_score"]))

    # Merge-audit planned pairs
    pair_configs = [
        # Retained: JB173_irony x neibla_irony
        ("TransferGraph__JB173_irony", "TransferGraph__neibla_irony", "retained_irony"),
        # Near-miss marginal: JB173_irony x phailyoor_irony (weak, delta -0.004)
        ("TransferGraph__JB173_irony", "TransferGraph__phailyoor_irony", "near_miss_marginal"),
        # Near-miss substantial: jaesun_hate x Aureliano_hate (weak, delta -0.15)
        ("TransferGraph__jaesun_hate", "TransferGraph__Aureliano_hate", "near_miss_substantial"),
        # Excluded control: JB173_irony x Aureliano_hate (cross-task + weak)
        ("TransferGraph__JB173_irony", "TransferGraph__Aureliano_hate", "excluded_control"),
    ]

    qa_report_objects = []
    for a_local, b_local, pair_type in pair_configs:
        a_dir = str(ADAPTER_CACHE / a_local)
        b_dir = str(ADAPTER_CACHE / b_local)

        log.info("  merge_audit: %s", pair_type)
        report = merge_audit(
            a_dir, b_dir, verbose=False,
            source_qa_a=qa_by_local[a_local].to_qa_result(),
            source_qa_b=qa_by_local[b_local].to_qa_result(),
        )
        qa_report = build_qa_report(report)
        qa_report_objects.append(qa_report)

        (PREFLIGHT_DIR / f"merge_{pair_type}.json").write_text(json.dumps(qa_report.to_dict(), indent=2))
        (PREFLIGHT_DIR / f"merge_{pair_type}_report.txt").write_text(format_qa_report(qa_report))

        task_rel = qa_report.task_relationship
        log.info("    risk=%s, strategy=%s, task_rel=%s",
                 qa_report.pair_risk, qa_report.recommended_strategy, task_rel)

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
        "near_miss_detail": [list(d) for d in action_plan.near_miss_detail],
        "retained_pair_detail": [list(d) for d in action_plan.retained_pair_detail],
        "summary_line": action_plan.summary_line,
    }
    (PREFLIGHT_DIR / "stance_lock.json").write_text(json.dumps(stance, indent=2))

    return qa_artifacts, qa_report_objects, action_plan


def evaluate_merges():
    """Evaluate the 4 planned merges."""
    pairs = [
        PairSpec(
            name="t02_retained_irony_x_irony",
            adapter_a_dir=ADAPTER_CACHE / "TransferGraph__JB173_irony",
            adapter_b_dir=ADAPTER_CACHE / "TransferGraph__neibla_irony",
            strategy="audit_aware",
            eval_dataset="tweet_eval/irony",
            base_model_id="distilbert-base-uncased",
            num_labels=2,
            role="retained",
            notes="Retained same-task (irony x irony, both eligible).",
        ),
        PairSpec(
            name="t02_near_miss_marginal_irony",
            adapter_a_dir=ADAPTER_CACHE / "TransferGraph__JB173_irony",
            adapter_b_dir=ADAPTER_CACHE / "TransferGraph__phailyoor_irony",
            strategy="audit_aware",
            eval_dataset="tweet_eval/irony",
            base_model_id="distilbert-base-uncased",
            num_labels=2,
            role="near_miss_marginal",
            notes="Near-miss marginal: phailyoor delta=-0.004 (barely weak).",
        ),
        PairSpec(
            name="t02_near_miss_substantial_hate",
            adapter_a_dir=ADAPTER_CACHE / "TransferGraph__jaesun_hate",
            adapter_b_dir=ADAPTER_CACHE / "TransferGraph__Aureliano_hate",
            strategy="audit_aware",
            eval_dataset="tweet_eval/hate",
            base_model_id="distilbert-base-uncased",
            num_labels=2,
            role="near_miss_substantial",
            notes="Near-miss substantial: Aureliano delta=-0.15 (deeply weak).",
        ),
        PairSpec(
            name="t02_excluded_irony_x_hate",
            adapter_a_dir=ADAPTER_CACHE / "TransferGraph__JB173_irony",
            adapter_b_dir=ADAPTER_CACHE / "TransferGraph__Aureliano_hate",
            strategy="audit_aware",
            eval_dataset="tweet_eval/irony",
            base_model_id="distilbert-base-uncased",
            num_labels=2,
            role="excluded_control",
            notes="Excluded: cross-task (irony x hate) + weak source. Eval on irony.",
        ),
    ]

    output_dir = INVENTORY_DIR.parent / "eval_t02"
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
    """Print compact summary."""
    print(f"\n{'='*100}")
    print("Targeted Confirmation T02 -- Near-Miss Severity Ordering")
    print(f"{'='*100}")
    print(f"{'Pair':45s} | {'Role':22s} | {'Strategy':16s} | {'Merged Acc':>10s}")
    print(f"{'-'*100}")
    for r in results:
        if "error" in r:
            print(f"{r['pair_name']:45s} | {r['role']:22s} | ERROR: {r['error'][:40]}")
        else:
            print(f"{r['pair_name']:45s} | {r['role']:22s} | {r['strategy']:16s} | {r['merged_accuracy']:10.4f}")
    print(f"{'='*100}")


def main():
    log.info("=== Targeted Confirmation T02: Near-Miss Severity Ordering ===")
    download_adapters()
    bootstrap_evidence()
    qa_artifacts, qa_reports, action_plan = run_gradience_preflight()
    results = evaluate_merges()
    print_summary(results)


if __name__ == "__main__":
    main()
