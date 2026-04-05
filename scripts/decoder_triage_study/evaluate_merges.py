#!/usr/bin/env python3
"""Merge/evaluate retained, near-miss, and stratified control pairs."""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _common import (
    CohortItem,
    adapter_dir,
    delta_directional,
    load_cohort_manifest,
    load_json,
    normalize_plan_strategy,
    pair_label,
    parse_pair_label,
    sanitize_pair_id,
    training_manifest_path,
    write_json,
)


GLUE_FIELD_MAP: dict[str, tuple[str, str | None]] = {
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
}

CLASSIFIER_KEYS = (
    "base_model.model.classifier.weight",
    "base_model.model.classifier.bias",
    "base_model.model.pre_classifier.weight",
    "base_model.model.pre_classifier.bias",
    "score.weight",
    "score.bias",
)


def _format_instruction_example(example: dict[str, Any], dataset_name: str) -> str:
    if "instruction" in example and ("output" in example or "response" in example):
        instruction = str(example.get("instruction", "")).strip()
        inp = str(example.get("input", "")).strip()
        out = str(example.get("output", example.get("response", ""))).strip()
        if inp:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        return f"### Instruction:\n{instruction}\n\n### Response:\n{out}"

    if "messages" in example and isinstance(example["messages"], list):
        parts = []
        for msg in example["messages"]:
            if isinstance(msg, dict):
                role = str(msg.get("role", "user")).strip()
                content = str(msg.get("content", "")).strip()
                if content:
                    parts.append(f"{role}: {content}")
            elif isinstance(msg, str) and msg.strip():
                parts.append(msg.strip())
        if parts:
            return "\n".join(parts)

    for key in ("text", "prompt", "content"):
        if key in example and str(example[key]).strip():
            if "response" in example and str(example["response"]).strip():
                return f"{example[key]}\n{example['response']}"
            if "output" in example and str(example["output"]).strip():
                return f"{example[key]}\n{example['output']}"
            return str(example[key])

    parts = [str(v).strip() for v in example.values() if isinstance(v, str) and str(v).strip()]
    return "\n".join(parts[:4]) if parts else f"[unhandled_example::{dataset_name}]"


def _load_training_records(adapter_root: Path, items: list[CohortItem]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in items:
        p = training_manifest_path(adapter_root, item.adapter_id)
        if not p.exists():
            raise FileNotFoundError(f"Missing training manifest for {item.adapter_id}: {p}")
        rec = load_json(p)
        if not rec.get("training_complete", False):
            raise RuntimeError(f"Incomplete training manifest for {item.adapter_id}: {p}")
        records[item.adapter_id] = rec
    return records


def _patch_classifier_for_eval(merged_dir: Path, source_dir: Path, num_labels: int, temp_dir: Path) -> Path:
    from safetensors.torch import load_file, save_file

    patched_dir = temp_dir / "patched_adapter"
    patched_dir.mkdir(parents=True, exist_ok=True)

    merged_sf = merged_dir / "adapter_model.safetensors"
    source_sf = source_dir / "adapter_model.safetensors"
    if not merged_sf.exists():
        raise FileNotFoundError(f"Merged adapter weights missing: {merged_sf}")
    merged_state = load_file(str(merged_sf))
    patched = dict(merged_state)
    if source_sf.exists():
        source_state = load_file(str(source_sf))
        for key in CLASSIFIER_KEYS:
            if key in source_state:
                patched[key] = source_state[key]

    save_file(patched, str(patched_dir / "adapter_model.safetensors"))
    cfg = load_json(merged_dir / "adapter_config.json")
    cfg["num_labels"] = int(num_labels)
    write_json(patched_dir / "adapter_config.json", cfg)
    return patched_dir


def _eval_seq_cls(merged_dir: Path, source_dir: Path, item: CohortItem, max_eval_samples: int) -> float:
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

    subset = item.dataset_subset or ""
    field_a, field_b = GLUE_FIELD_MAP[subset]
    ds = load_dataset(item.dataset, subset)
    split = "validation_matched" if subset == "mnli" and "validation_matched" in ds else "validation"
    if split not in ds:
        split = item.split if item.split in ds else "train"
    val_ds = ds[split].select(range(min(max_eval_samples, len(ds[split]))))
    num_labels = len(set(val_ds["label"])) if "label" in val_ds.column_names else 2

    tok = AutoTokenizer.from_pretrained(item.base_model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    def preprocess(examples: dict[str, Any]) -> dict[str, Any]:
        if field_b is None:
            out = tok(examples[field_a], truncation=True, padding=True, max_length=item.max_seq_length)
        else:
            out = tok(examples[field_a], examples[field_b], truncation=True, padding=True, max_length=item.max_seq_length)
        out["labels"] = examples["label"]
        return out

    val_tok = val_ds.map(preprocess, batched=True)
    collator = DataCollatorWithPadding(tok)

    with tempfile.TemporaryDirectory() as td:
        patched_dir = _patch_classifier_for_eval(
            merged_dir=merged_dir,
            source_dir=source_dir,
            num_labels=num_labels,
            temp_dir=Path(td),
        )
        model = AutoModelForSequenceClassification.from_pretrained(item.base_model, num_labels=num_labels)
        model = PeftModel.from_pretrained(model, str(patched_dir))
        model = model.merge_and_unload()
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=str(Path(td) / "eval"), per_device_eval_batch_size=8, report_to=[]),
            eval_dataset=val_tok,
            data_collator=collator,
        )
        preds = trainer.predict(val_tok)
        acc = float((np.argmax(preds.predictions, axis=1) == preds.label_ids).mean())
    return acc


def _eval_causal_loss(merged_dir: Path, item: CohortItem, max_eval_samples: int) -> float:
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if item.dataset_subset:
        ds = load_dataset(item.dataset, item.dataset_subset)
    else:
        ds = load_dataset(item.dataset)

    if "validation" in ds:
        eval_raw = ds["validation"]
    else:
        split_name = item.split if item.split in ds else "train"
        eval_raw = ds[split_name].shuffle(seed=item.seed)
    eval_raw = eval_raw.select(range(min(max_eval_samples, len(eval_raw))))

    tok = AutoTokenizer.from_pretrained(item.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize_fn(example: dict[str, Any]) -> dict[str, Any]:
        text = _format_instruction_example(example, dataset_name=item.dataset)
        enc = tok(text, truncation=True, max_length=item.max_seq_length, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    eval_tok = eval_raw.map(tokenize_fn, remove_columns=eval_raw.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(item.base_model, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, str(merged_dir))
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(merged_dir / "_eval_tmp"), per_device_eval_batch_size=4, report_to=[]),
        eval_dataset=eval_tok,
        data_collator=collator,
    )
    metrics = trainer.evaluate()
    return float(metrics.get("eval_loss", 999.0))


def _parse_pair_report(p: Path) -> dict[str, Any]:
    r = load_json(p)
    a_name = Path(r["adapter_a"]["path"]).name
    b_name = Path(r["adapter_b"]["path"]).name
    label = pair_label(a_name, b_name)
    return {
        "label": label,
        "a": a_name,
        "b": b_name,
        "pair_risk": r.get("pair_risk", "unknown"),
        "recommended_strategy": normalize_plan_strategy(r.get("recommended_strategy")),
        "task_relationship": r.get("task_relationship"),
        "task_relationship_advisory": r.get("task_relationship_advisory"),
    }


def _select_controls(
    candidates: list[str],
    pair_risk: dict[str, str],
    n: int,
    seed: int,
) -> list[str]:
    if n <= 0 or not candidates:
        return []
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {"high": [], "medium": [], "low": [], "unknown": []}
    for label in candidates:
        risk = pair_risk.get(label, "unknown")
        if risk not in buckets:
            risk = "unknown"
        buckets[risk].append(label)
    for v in buckets.values():
        rng.shuffle(v)

    active = [k for k, v in buckets.items() if v]
    if not active:
        return []
    per = max(1, n // len(active))
    picked: list[str] = []
    for key in active:
        take = min(per, len(buckets[key]))
        picked.extend(buckets[key][:take])
        buckets[key] = buckets[key][take:]
    if len(picked) < n:
        rem: list[str] = []
        for key in ("high", "medium", "low", "unknown"):
            rem.extend(buckets[key])
        rng.shuffle(rem)
        picked.extend(rem[: max(0, n - len(picked))])
    return picked[:n]


def _build_merge_plan(
    *,
    action_plan_path: Path,
    pair_reports_dir: Path,
    item_by_id: dict[str, CohortItem],
    control_sample_size: int,
    control_seed: int,
) -> list[dict[str, Any]]:
    action = load_json(action_plan_path)

    pair_reports = [_parse_pair_report(p) for p in sorted(pair_reports_dir.glob("*.json"))]
    pair_risk_map = {x["label"]: x["pair_risk"] for x in pair_reports}
    strategy_map = {x["label"]: x["recommended_strategy"] for x in pair_reports}

    retained = list(action.get("evaluate_first", []))
    near_miss = list(action.get("near_miss_candidates", []))
    selected = set(retained) | set(near_miss)

    cross_candidates: list[str] = []
    for x in pair_reports:
        a, b = x["a"], x["b"]
        if a not in item_by_id or b not in item_by_id:
            continue
        if item_by_id[a].task_family != item_by_id[b].task_family and x["label"] not in selected:
            cross_candidates.append(x["label"])
    controls = _select_controls(cross_candidates, pair_risk=pair_risk_map, n=control_sample_size, seed=control_seed)

    retained_detail = {row[0]: row for row in action.get("retained_pair_detail", [])}
    near_miss_detail = {row[0]: row for row in action.get("near_miss_detail", [])}
    plan: list[dict[str, Any]] = []
    for label in retained:
        row = retained_detail.get(label, [])
        strategy = normalize_plan_strategy(row[2] if len(row) >= 3 else strategy_map.get(label))
        plan.append({"pair_label": label, "category": "retained", "strategy": strategy, "note": "action_plan.evaluate_first"})
    for label in near_miss:
        row = near_miss_detail.get(label, [])
        strategy = normalize_plan_strategy(row[2] if len(row) >= 3 else strategy_map.get(label))
        reason = row[3] if len(row) >= 4 else "near_miss"
        plan.append({"pair_label": label, "category": "near_miss", "strategy": strategy, "note": reason})
    for label in controls:
        plan.append({"pair_label": label, "category": "control", "strategy": normalize_plan_strategy(strategy_map.get(label)), "note": "stratified_cross_task_control"})
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate merge cohort for decoder triage study.")
    parser.add_argument(
        "--manifest",
        default="scripts/decoder_triage_study/cohort_manifest.json",
        help="Path to cohort manifest JSON",
    )
    parser.add_argument("--adapter-dir", required=True, help="Adapter root directory")
    parser.add_argument("--pipeline-dir", required=True, help="Pipeline output directory from run_pipeline.py")
    parser.add_argument("--output-dir", required=True, help="Merge evaluation output directory")
    parser.add_argument("--merge-plan", default=None, help="Optional explicit merge plan JSON")
    parser.add_argument("--control-sample-size", type=int, default=6, help="Number of stratified cross-task controls")
    parser.add_argument("--control-seed", type=int, default=17, help="Random seed for control sampling")
    parser.add_argument("--max-eval-samples", type=int, default=1000, help="Evaluation sample cap per task")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    adapter_root = Path(args.adapter_dir).resolve()
    pipeline_dir = Path(args.pipeline_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_results_dir = out_dir / "pair_results"
    merged_root = out_dir / "merged_adapters"
    pair_results_dir.mkdir(parents=True, exist_ok=True)
    merged_root.mkdir(parents=True, exist_ok=True)

    items = load_cohort_manifest(manifest_path)
    item_by_id = {x.adapter_id: x for x in items}
    train_records = _load_training_records(adapter_root, items)

    # Build or load merge plan
    if args.merge_plan:
        merge_plan_path = Path(args.merge_plan).resolve()
        merge_plan = load_json(merge_plan_path).get("pairs", [])
    else:
        merge_plan = _build_merge_plan(
            action_plan_path=pipeline_dir / "inventory_action_plan.json",
            pair_reports_dir=pipeline_dir / "pair_reports",
            item_by_id=item_by_id,
            control_sample_size=args.control_sample_size,
            control_seed=args.control_seed,
        )
        merge_plan_path = out_dir / "merge_plan.json"
        write_json(merge_plan_path, {"pairs": merge_plan})

    from gradience.vnext.merge import execute_merge, merge_audit, plan_from_audit

    results: list[dict[str, Any]] = []
    for row in merge_plan:
        label = row["pair_label"]
        category = row["category"]
        strategy = normalize_plan_strategy(row["strategy"])
        a_id, b_id = parse_pair_label(label)
        pair_id = sanitize_pair_id(label)
        result_path = pair_results_dir / f"{pair_id}.json"
        if result_path.exists():
            results.append(load_json(result_path))
            print(f"[eval] skip existing: {label}")
            continue

        if a_id not in item_by_id or b_id not in item_by_id:
            raise KeyError(f"Pair references unknown adapter IDs: {label}")
        a_item, b_item = item_by_id[a_id], item_by_id[b_id]
        a_rec, b_rec = train_records[a_id], train_records[b_id]

        merged_dir = merged_root / pair_id / "merged_adapter"
        merged_dir.mkdir(parents=True, exist_ok=True)

        report = merge_audit(str(adapter_dir(adapter_root, a_id)), str(adapter_dir(adapter_root, b_id)))
        plan = plan_from_audit(strategy, report, str(adapter_dir(adapter_root, a_id)), str(adapter_dir(adapter_root, b_id)))
        merge_result = execute_merge(plan, str(merged_dir))

        # Evaluate on task A
        if a_item.task_type == "SEQ_CLS":
            merged_a = _eval_seq_cls(
                merged_dir=merged_dir,
                source_dir=adapter_dir(adapter_root, a_id),
                item=a_item,
                max_eval_samples=args.max_eval_samples,
            )
        else:
            merged_a = _eval_causal_loss(merged_dir=merged_dir, item=a_item, max_eval_samples=args.max_eval_samples)

        # Evaluate on task B
        if b_item.task_type == "SEQ_CLS":
            merged_b = _eval_seq_cls(
                merged_dir=merged_dir,
                source_dir=adapter_dir(adapter_root, b_id),
                item=b_item,
                max_eval_samples=args.max_eval_samples,
            )
        else:
            merged_b = _eval_causal_loss(merged_dir=merged_dir, item=b_item, max_eval_samples=args.max_eval_samples)

        delta_a = delta_directional(
            merged_score=merged_a,
            source_score=float(a_rec["adapter_score"]),
            higher_is_better=bool(a_rec["higher_is_better"]),
        )
        delta_b = delta_directional(
            merged_score=merged_b,
            source_score=float(b_rec["adapter_score"]),
            higher_is_better=bool(b_rec["higher_is_better"]),
        )
        # Conservative pair-level score: require both sides to hold.
        delta_best = min(delta_a, delta_b)

        result = {
            "schema": "gradience.decoder_triage.merge_eval_result/v1",
            "pair_label": label,
            "pair_id": pair_id,
            "category": category,
            "strategy": strategy,
            "note": row.get("note"),
            "adapter_a": a_id,
            "adapter_b": b_id,
            "task_a": a_item.task_key,
            "task_b": b_item.task_key,
            "task_family_a": a_item.task_family,
            "task_family_b": b_item.task_family,
            "metric_name_a": a_rec["metric_name"],
            "metric_name_b": b_rec["metric_name"],
            "higher_is_better_a": bool(a_rec["higher_is_better"]),
            "higher_is_better_b": bool(b_rec["higher_is_better"]),
            "source_score_a": float(a_rec["adapter_score"]),
            "source_score_b": float(b_rec["adapter_score"]),
            "merged_score_a": float(merged_a),
            "merged_score_b": float(merged_b),
            "delta_directional_a": float(delta_a),
            "delta_directional_b": float(delta_b),
            "delta_vs_best_source_directional": float(delta_best),
            "mean_reconstruction_error": float(getattr(merge_result, "mean_reconstruction_error", 0.0)),
        }
        write_json(result_path, result)
        results.append(result)
        print(f"[eval] {category} {label} Δbest_dir={delta_best:+.4f}")

    # Summary rollup
    category_summary: dict[str, Any] = {}
    for cat in ("retained", "near_miss", "control"):
        rows = [r for r in results if r.get("category") == cat]
        if not rows:
            continue
        vals = [float(r["delta_vs_best_source_directional"]) for r in rows]
        category_summary[cat] = {
            "count": len(rows),
            "avg_delta_directional": float(sum(vals) / len(vals)),
            "min_delta_directional": float(min(vals)),
            "max_delta_directional": float(max(vals)),
        }

    out_payload = {
        "schema": "gradience.decoder_triage.merge_results/v1",
        "merge_plan_path": str(merge_plan_path),
        "pair_count": len(results),
        "category_summary": category_summary,
        "results": results,
    }
    write_json(out_dir / "merge_results.json", out_payload)

    # Small markdown table for quick inspection
    lines = []
    lines.append("# Merge Results")
    lines.append("")
    lines.append("| Pair | Category | Strategy | Δdir A | Δdir B | Δdir best |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['pair_label']} | {r['category']} | {r['strategy']} | "
            f"{r['delta_directional_a']:+.4f} | {r['delta_directional_b']:+.4f} | "
            f"{r['delta_vs_best_source_directional']:+.4f} |"
        )
    lines.append("")
    lines.append("## Category summary")
    lines.append("")
    for cat, s in category_summary.items():
        lines.append(
            f"- {cat}: n={s['count']}, avg Δdir={s['avg_delta_directional']:+.4f}, "
            f"range=[{s['min_delta_directional']:+.4f}, {s['max_delta_directional']:+.4f}]"
        )
    (out_dir / "merge_results.md").write_text("\n".join(lines))

    print(f"\nWrote merge results: {out_dir / 'merge_results.json'}")


if __name__ == "__main__":
    main()
