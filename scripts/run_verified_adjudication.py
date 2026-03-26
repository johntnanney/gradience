"""Ambiguous-relationship adjudication study.

For each pair in the panel:
1. Run merge_audit with core-space
2. Merge with uniform_linear
3. Evaluate merged adapter on both source tasks
4. Compare with individual adapter scores

Cross-task merges with different num_labels use each source's classifier
head when evaluating on that source's task.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

BASE_MODEL = "distilbert-base-uncased"
ADAPTER_ROOT = Path("results/adjudication_study/ambiguous_relationship_regime/adapters")
OUTPUT_ROOT = Path("results/adjudication_study/ambiguous_relationship_regime")

TASK_META = {
    "qnli": {"num_labels": 2, "field_a": "question", "field_b": "sentence", "val_split": "validation", "eval_n": 500},
    "rte": {"num_labels": 2, "field_a": "sentence1", "field_b": "sentence2", "val_split": "validation", "eval_n": 277},
    "mnli": {"num_labels": 3, "field_a": "premise", "field_b": "hypothesis", "val_split": "validation_matched", "eval_n": 500},
}

# 8-pair panel
PANEL = [
    # Group 1: Same-task controls
    {"pair_id": "qnli_s42_x_qnli_s7", "group": "same-task", "a": "qnli_s42", "b": "qnli_s7", "task_a": "qnli", "task_b": "qnli"},
    {"pair_id": "rte_s42_x_rte_s7", "group": "same-task", "a": "rte_s42", "b": "rte_s7", "task_a": "rte", "task_b": "rte"},
    # Group 2: Adjacent-task ambiguous
    {"pair_id": "qnli_s42_x_rte_s42", "group": "adjacent", "a": "qnli_s42", "b": "rte_s42", "task_a": "qnli", "task_b": "rte"},
    {"pair_id": "qnli_s42_x_mnli_s42", "group": "adjacent", "a": "qnli_s42", "b": "mnli_s42", "task_a": "qnli", "task_b": "mnli"},
    {"pair_id": "rte_s42_x_mnli_s42", "group": "adjacent", "a": "rte_s42", "b": "mnli_s42", "task_a": "rte", "task_b": "mnli"},
    {"pair_id": "qnli_s7_x_rte_s7", "group": "adjacent", "a": "qnli_s7", "b": "rte_s7", "task_a": "qnli", "task_b": "rte"},
    # Group 3: Contrast
    {"pair_id": "mnli_s42_x_mnli_s7", "group": "same-task", "a": "mnli_s42", "b": "mnli_s7", "task_a": "mnli", "task_b": "mnli"},
    {"pair_id": "rte_s7_x_mnli_s7", "group": "adjacent-diff-labels", "a": "rte_s7", "b": "mnli_s7", "task_a": "rte", "task_b": "mnli"},
]


def evaluate_on_task(adapter_dir: str | None, task: str, classifier_source: str | None = None) -> float:
    """Evaluate an adapter on a specific task."""
    from datasets import load_dataset
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    meta = TASK_META[task]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if adapter_dir is not None and classifier_source is not None:
        # Cross-task eval: swap classifier head from source adapter
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_adapter = Path(tmpdir) / "adapter"
            tmp_adapter.mkdir()

            merged_st = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
            source_st = load_file(str(Path(classifier_source) / "adapter_model.safetensors"))

            cls_keys = [
                "base_model.model.classifier.weight",
                "base_model.model.classifier.bias",
                "base_model.model.pre_classifier.weight",
                "base_model.model.pre_classifier.bias",
            ]
            for key in cls_keys:
                if key in source_st:
                    merged_st[key] = source_st[key]

            save_file(merged_st, str(tmp_adapter / "adapter_model.safetensors"))
            # Need config with correct num_labels
            cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
            (tmp_adapter / "adapter_config.json").write_text(json.dumps(cfg))

            model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=meta["num_labels"])
            model = PeftModel.from_pretrained(model, str(tmp_adapter))
            model = model.merge_and_unload()
    elif adapter_dir is not None:
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=meta["num_labels"])
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=meta["num_labels"])

    model.eval()

    ds = load_dataset("glue", task, split=meta["val_split"])
    if meta["eval_n"] < len(ds):
        ds = ds.select(range(meta["eval_n"]))

    def preprocess(examples):
        result = tokenizer(
            examples[meta["field_a"]], examples[meta["field_b"]],
            truncation=True, padding=True, max_length=128,
        )
        result["labels"] = examples["label"]
        return result

    tok_ds = ds.map(preprocess, batched=True)
    args = TrainingArguments(output_dir="/tmp/eval", per_device_eval_batch_size=32, report_to=[])
    trainer = Trainer(model=model, args=args, eval_dataset=tok_ds, data_collator=DataCollatorWithPadding(tokenizer))
    preds = trainer.predict(tok_ds)
    return float((np.argmax(preds.predictions, axis=1) == preds.label_ids).mean())


def patch_merged_classifier(merged_dir: str, source_a: str, source_b: str) -> None:
    """Average classifier heads from both sources into the merged adapter (same-task only)."""
    from safetensors.torch import load_file, save_file

    merged_path = Path(merged_dir) / "adapter_model.safetensors"
    merged_st = load_file(str(merged_path))

    if "base_model.model.classifier.weight" in merged_st:
        return

    st_a = load_file(str(Path(source_a) / "adapter_model.safetensors"))
    st_b = load_file(str(Path(source_b) / "adapter_model.safetensors"))

    cls_keys = [
        "base_model.model.classifier.weight",
        "base_model.model.classifier.bias",
        "base_model.model.pre_classifier.weight",
        "base_model.model.pre_classifier.bias",
    ]

    for key in cls_keys:
        if key in st_a and key in st_b and st_a[key].shape == st_b[key].shape:
            merged_st[key] = (st_a[key] + st_b[key]) / 2.0
        elif key in st_a:
            merged_st[key] = st_a[key]
        elif key in st_b:
            merged_st[key] = st_b[key]

    save_file(merged_st, str(merged_path))


@dataclass
class PairResult:
    pair_id: str
    group: str
    task_a: str
    task_b: str
    same_task: bool
    individual_a_on_task_a: float
    individual_b_on_task_b: float
    pair_risk: str
    dominant_issue: str
    compatibility_score: float
    core_space_status: str | None
    shared_basis_score: float | None
    merged_on_task_a: float | None
    merged_on_task_b: float | None
    degradation_task_a: float | None
    degradation_task_b: float | None
    reconstruction_error: float | None
    error: str | None


def run_pair(entry: dict, individual_scores: dict) -> PairResult:
    """Run merge audit, merge, and evaluate for one pair."""
    from gradience.vnext.merge import diagnose_pair, execute_merge, merge_audit, plan_from_audit, recommend_merge

    pair_id = entry["pair_id"]
    adapter_a = str(ADAPTER_ROOT / entry["a"])
    adapter_b = str(ADAPTER_ROOT / entry["b"])
    task_a = entry["task_a"]
    task_b = entry["task_b"]
    same_task = task_a == task_b

    print(f"\n{'='*60}")
    print(f"Pair: {pair_id} ({entry['group']})")

    ind_a = individual_scores[entry["a"]]
    ind_b = individual_scores[entry["b"]]

    pair_risk = ""
    dominant_issue = ""
    compat_score = 0.0
    cs_status = None
    cs_basis = None
    merged_a = None
    merged_b = None
    recon_error = None
    error = None

    try:
        # Merge audit + core-space
        report = merge_audit(adapter_a, adapter_b, compute_core_space=True)
        ov = report.aggregate.overall_verdict
        pair_risk = ov.value if hasattr(ov, "value") else str(ov)
        compat_score = report.aggregate.compatibility_score
        if report.core_space:
            cs_status = report.core_space.status
            cs_basis = report.core_space.shared_basis_score_mean
        print(f"  risk={pair_risk}, compat={compat_score:.3f}, cs={cs_status} ({cs_basis})")

        try:
            diagnosis = diagnose_pair(report)
            rec = recommend_merge(diagnosis)
            dominant_issue = rec.dominant_issue if hasattr(rec, "dominant_issue") else "unknown"
        except Exception:
            dominant_issue = "unknown"

        # Merge
        with tempfile.TemporaryDirectory() as tmpdir:
            merge_dir = Path(tmpdir) / "merged"
            plan = plan_from_audit("uniform_linear", report, adapter_a, adapter_b)
            result = execute_merge(plan, str(merge_dir))
            recon_error = result.mean_reconstruction_error
            print(f"  merge recon_error={recon_error:.4f}")

            if same_task:
                patch_merged_classifier(str(merge_dir), adapter_a, adapter_b)
                merged_a = evaluate_on_task(str(merge_dir), task_a)
                merged_b = merged_a
                print(f"  merged on {task_a}: {merged_a:.3f} (individual: {ind_a:.3f})")
            else:
                # Don't patch — use source-specific classifiers for each eval
                merged_a = evaluate_on_task(str(merge_dir), task_a, classifier_source=adapter_a)
                merged_b = evaluate_on_task(str(merge_dir), task_b, classifier_source=adapter_b)
                print(f"  merged on {task_a}: {merged_a:.3f} (individual: {ind_a:.3f})")
                print(f"  merged on {task_b}: {merged_b:.3f} (individual: {ind_b:.3f})")

    except Exception as e:
        error = str(e)
        print(f"  ERROR: {e}")

    deg_a = (ind_a - merged_a) if (merged_a is not None) else None
    deg_b = (ind_b - merged_b) if (merged_b is not None) else None

    return PairResult(
        pair_id=pair_id, group=entry["group"], task_a=task_a, task_b=task_b, same_task=same_task,
        individual_a_on_task_a=ind_a, individual_b_on_task_b=ind_b,
        pair_risk=pair_risk, dominant_issue=dominant_issue, compatibility_score=compat_score,
        core_space_status=cs_status, shared_basis_score=cs_basis,
        merged_on_task_a=merged_a, merged_on_task_b=merged_b,
        degradation_task_a=deg_a, degradation_task_b=deg_b,
        reconstruction_error=recon_error, error=error,
    )


def main():
    print("=" * 60)
    print("AMBIGUOUS RELATIONSHIP ADJUDICATION")
    print("=" * 60)

    # Load individual scores
    individual_scores = {}
    for entry in PANEL:
        for name in [entry["a"], entry["b"]]:
            if name not in individual_scores:
                manifest = json.loads((ADAPTER_ROOT / name / "training_manifest.json").read_text())
                individual_scores[name] = manifest["eval_accuracy"]

    print("\nIndividual scores:")
    for name, score in sorted(individual_scores.items()):
        print(f"  {name}: {score:.3f}")

    results = []
    for entry in PANEL:
        result = run_pair(entry, individual_scores)
        results.append(result)

    # Save
    out_path = OUTPUT_ROOT / "adjudication_results.json"
    with open(out_path, "w") as f:
        json.dump({"base_model": BASE_MODEL, "individual_scores": individual_scores, "pairs": [asdict(r) for r in results]}, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Summary
    print(f"\n{'='*110}")
    print("ADJUDICATION SUMMARY")
    print(f"{'='*110}")
    print(f"{'Pair':<32s} {'Group':<18s} {'Risk':<8s} {'CS':<12s} {'Basis':<7s} {'Mrgd-A':<8s} {'Mrgd-B':<8s} {'Deg-A':<8s} {'Deg-B':<8s}")
    print(f"{'-'*32} {'-'*18} {'-'*8} {'-'*12} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        ma = f"{r.merged_on_task_a:.3f}" if r.merged_on_task_a is not None else "ERR"
        mb = f"{r.merged_on_task_b:.3f}" if r.merged_on_task_b is not None else "ERR"
        da = f"{r.degradation_task_a:+.3f}" if r.degradation_task_a is not None else "n/a"
        db = f"{r.degradation_task_b:+.3f}" if r.degradation_task_b is not None else "n/a"
        cs = r.core_space_status or "none"
        bs = f"{r.shared_basis_score:.3f}" if r.shared_basis_score is not None else "n/a"
        print(f"{r.pair_id:<32s} {r.group:<18s} {r.pair_risk:<8s} {cs:<12s} {bs:<7s} {ma:<8s} {mb:<8s} {da:<8s} {db:<8s}")

    # Classification
    print(f"\nDEGRADATION CLASSIFICATION")
    for r in results:
        if r.error:
            continue
        parts = []
        if r.degradation_task_a is not None:
            d = r.degradation_task_a
            cat = "safe" if d <= 0.02 else ("CAUTION" if d <= 0.05 else "DEGRADED")
            parts.append(f"{r.task_a}: {cat} ({d:+.3f})")
        if r.degradation_task_b is not None and not r.same_task:
            d = r.degradation_task_b
            cat = "safe" if d <= 0.02 else ("CAUTION" if d <= 0.05 else "DEGRADED")
            parts.append(f"{r.task_b}: {cat} ({d:+.3f})")
        cs_note = f" [cs:{r.core_space_status} {r.shared_basis_score:.3f}]" if r.core_space_status else ""
        print(f"  {r.pair_id}: {'; '.join(parts)}{cs_note}")


if __name__ == "__main__":
    main()
