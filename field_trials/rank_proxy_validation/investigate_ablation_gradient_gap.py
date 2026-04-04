#!/usr/bin/env python3
"""Investigate why ablation-aligned policies can underperform gradient proxy outcomes.

Focused on informative (compressible) task families only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from field_trials.rank_proxy_validation import run_cpu_rank_proxy_study as rp


STUDY_DIR = Path(__file__).resolve().parent


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    x = _safe_float(value, default=float("nan"))
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _pairwise_spearman(vectors: list[list[float]]) -> float | None:
    if len(vectors) < 2:
        return None
    vals: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            rho = rp.spearman(vectors[i], vectors[j])
            if rho is not None:
                vals.append(float(rho))
    if not vals:
        return None
    return statistics.mean(vals)


def _topk_from_scores(layer_names: list[str], scores: list[float], k: int) -> set[str]:
    pairs = sorted(zip(layer_names, scores), key=lambda x: x[1], reverse=True)
    return {name for name, _ in pairs[:k]}


def _pairwise_topk_overlap(layer_names: list[str], vectors: list[list[float]], k: int) -> float | None:
    if len(vectors) < 2:
        return None
    vals: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a = _topk_from_scores(layer_names, vectors[i], k)
            b = _topk_from_scores(layer_names, vectors[j], k)
            denom = max(1, min(k, len(a), len(b)))
            vals.append(len(a.intersection(b)) / denom)
    if not vals:
        return None
    return statistics.mean(vals)


def _sample_subset(texts: list[Any], labels: list[int], n: int, rng: random.Random) -> tuple[list[Any], list[int]]:
    n = min(n, len(texts))
    if n <= 0:
        return [], []
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    keep = idx[:n]
    return [texts[i] for i in keep], [labels[i] for i in keep]


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=str, default="0.35,0.50,0.65")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--grad-samples", type=int, default=32)
    parser.add_argument("--ablation-samples", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--allow-online", action="store_true")
    parser.add_argument("--max-adapters", type=int, default=0, help="0 means no cap")
    args = parser.parse_args()

    if not args.allow_online:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    budgets = sorted({max(0.05, min(1.0, b)) for b in budgets})

    quality = json.loads((STUDY_DIR / "source_quality_gap_control_slice.json").read_text())
    informative_families = set(quality["analysis_scope"]["informative_task_families"])
    informative_adapter_rows = quality["primary_informative_subset"]["adapter_quality_table"]
    informative_adapter_ids = [str(row["adapter_id"]) for row in informative_adapter_rows]

    records = rp.collect_adapter_records()
    selected: list[rp.AdapterRecord] = [records[aid] for aid in informative_adapter_ids if aid in records]
    if args.max_adapters > 0:
        selected = selected[: args.max_adapters]

    compression_rows_all = json.loads((STUDY_DIR / "compression_evaluation_table.json").read_text())
    compression_rows_inf = [
        row for row in compression_rows_all if rp.task_family_for_dataset(str(row.get("dataset") or "")) in informative_families
    ]

    # Gradient-vs-ablation performance concentration analysis from existing compression outputs.
    adapter_perf_rows: list[dict[str, Any]] = []
    per_adapter_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in compression_rows_inf:
        per_adapter_group[(str(row["adapter_id"]), str(row["dataset"]))].append(row)
    for (aid, dataset), rows in sorted(per_adapter_group.items()):
        grad = [r for r in rows if str(r["method"]) == "proxy_gradient"]
        abl = [r for r in rows if str(r["method"]) == "proxy_ablation"]
        if not grad or not abl:
            continue
        grad_dvu = statistics.mean([_safe_float(r["delta_vs_uniform"]) for r in grad])
        abl_dvu = statistics.mean([_safe_float(r["delta_vs_uniform"]) for r in abl])
        adapter_perf_rows.append(
            {
                "adapter_id": aid,
                "dataset": dataset,
                "task_family": rp.task_family_for_dataset(dataset),
                "proxy_gradient_mean_delta_vs_uniform": grad_dvu,
                "proxy_ablation_mean_delta_vs_uniform": abl_dvu,
                "gradient_minus_ablation_mean_delta_vs_uniform": grad_dvu - abl_dvu,
            }
        )

    adapter_perf_rows.sort(key=lambda x: x["gradient_minus_ablation_mean_delta_vs_uniform"], reverse=True)
    diffs = [row["gradient_minus_ablation_mean_delta_vs_uniform"] for row in adapter_perf_rows]
    abs_total = sum(abs(x) for x in diffs) if diffs else 0.0
    top2_abs_share = (
        sum(sorted([abs(x) for x in diffs], reverse=True)[:2]) / abs_total if abs_total > 0 else None
    )

    by_family_perf: dict[str, list[float]] = defaultdict(list)
    for row in adapter_perf_rows:
        by_family_perf[row["task_family"]].append(row["gradient_minus_ablation_mean_delta_vs_uniform"])
    family_perf_rows = []
    for fam, vals in sorted(by_family_perf.items(), key=lambda x: rp._task_family_sort_key(x[0])):
        family_perf_rows.append(
            {
                "task_family": fam,
                "n_adapters": len(vals),
                "mean_gradient_minus_ablation_delta_vs_uniform": statistics.mean(vals),
                "var_gradient_minus_ablation_delta_vs_uniform": statistics.pvariance(vals) if len(vals) > 1 else 0.0,
            }
        )

    # Resampling stability + top-k distribution investigation.
    stability_rows: list[dict[str, Any]] = []
    topk_dist_rows: list[dict[str, Any]] = []

    for adapter in selected:
        try:
            texts, labels = rp.load_eval_data(
                adapter.dataset, max_samples=args.max_eval_samples, allow_online=args.allow_online
            )
        except Exception:
            continue

        tokenizer = AutoTokenizer.from_pretrained(adapter.base_model, local_files_only=(not args.allow_online))
        base_model = AutoModelForSequenceClassification.from_pretrained(
            adapter.base_model,
            num_labels=adapter.num_labels,
            local_files_only=(not args.allow_online),
        )
        model = PeftModel.from_pretrained(base_model, str(adapter.adapter_dir))
        model.eval()

        _, weights_path, _ = rp.find_peft_files(adapter.adapter_dir)
        if weights_path is None:
            continue
        adapter_sd = rp.load_adapter_state_dict(weights_path)
        model_param_keys = {k for k, _ in model.named_parameters()}
        layers = rp._extract_layer_meta(adapter, adapter_sd, model_param_keys)
        if not layers:
            continue

        layer_names = [layer.layer_name for layer in layers]
        k_top = max(1, len(layers) // 4)

        grad_vectors: list[list[float]] = []
        abl_vectors: list[list[float]] = []

        for rep in range(args.repeats):
            rng = random.Random(args.seed + rep * 10007 + hash(adapter.adapter_id))
            g_texts, g_labels = _sample_subset(texts, labels, args.grad_samples, rng)
            a_texts, a_labels = _sample_subset(texts, labels, args.ablation_samples, rng)

            grad_scores = rp.compute_gradient_proxy_scores(
                model,
                tokenizer,
                g_texts,
                g_labels,
                layers,
                grad_samples=len(g_texts),
                batch_size=args.grad_batch_size,
                max_length=args.max_length,
            )
            abl_scores = rp.compute_ablation_proxy_scores(
                model,
                tokenizer,
                a_texts,
                a_labels,
                layers,
                ablation_samples=len(a_texts),
                batch_size=args.eval_batch_size,
                max_length=args.max_length,
            )

            grad_vec = [_safe_float(grad_scores.get(name), 0.0) for name in layer_names]
            abl_vec = [_safe_float(abl_scores.get(name), 0.0) for name in layer_names]
            grad_vectors.append(grad_vec)
            abl_vectors.append(abl_vec)

            for budget in budgets:
                alloc_g = rp.allocate_ranks_greedy(layers, grad_scores, budget)
                alloc_a = rp.allocate_ranks_greedy(layers, abl_scores, budget)
                topg = {x[0] for x in sorted(alloc_g.items(), key=lambda x: x[1], reverse=True)[:k_top]}
                topa = {x[0] for x in sorted(alloc_a.items(), key=lambda x: x[1], reverse=True)[:k_top]}
                shared = sorted(topg.intersection(topa))
                union = sorted(topg.union(topa))
                shared_abs = (
                    [_safe_float(abs(alloc_g[n] - alloc_a[n])) for n in shared] if shared else []
                )
                union_abs = [_safe_float(abs(alloc_g[n] - alloc_a[n])) for n in union] if union else []
                topk_dist_rows.append(
                    {
                        "adapter_id": adapter.adapter_id,
                        "dataset": adapter.dataset,
                        "task_family": rp.task_family_for_dataset(adapter.dataset),
                        "repeat": rep,
                        "budget_ratio": budget,
                        "topk_overlap": rp.topk_overlap(alloc_g, alloc_a, k_top),
                        "shared_topk_count": len(shared),
                        "mean_abs_rank_dev_shared_topk": _mean_or_none(shared_abs),
                        "mean_abs_rank_dev_union_topk": _mean_or_none(union_abs),
                        "mean_abs_rank_dev_all_layers": statistics.mean(
                            [abs(alloc_g[n] - alloc_a[n]) for n in layer_names]
                        ),
                    }
                )

        grad_pairwise_spearman = _pairwise_spearman(grad_vectors)
        abl_pairwise_spearman = _pairwise_spearman(abl_vectors)
        grad_pairwise_topk = _pairwise_topk_overlap(layer_names, grad_vectors, k_top)
        abl_pairwise_topk = _pairwise_topk_overlap(layer_names, abl_vectors, k_top)

        stability_rows.append(
            {
                "adapter_id": adapter.adapter_id,
                "dataset": adapter.dataset,
                "task_family": rp.task_family_for_dataset(adapter.dataset),
                "layer_count": len(layer_names),
                "k_top": k_top,
                "repeats": args.repeats,
                "gradient_pairwise_spearman": grad_pairwise_spearman,
                "ablation_pairwise_spearman": abl_pairwise_spearman,
                "gradient_pairwise_topk_overlap": grad_pairwise_topk,
                "ablation_pairwise_topk_overlap": abl_pairwise_topk,
                "gradient_minus_ablation_pairwise_spearman": (
                    (grad_pairwise_spearman - abl_pairwise_spearman)
                    if grad_pairwise_spearman is not None and abl_pairwise_spearman is not None
                    else None
                ),
                "gradient_minus_ablation_pairwise_topk_overlap": (
                    (grad_pairwise_topk - abl_pairwise_topk)
                    if grad_pairwise_topk is not None and abl_pairwise_topk is not None
                    else None
                ),
            }
        )

    # Aggregate stability summaries.
    def _summarize_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
        g_s = [_safe_float(r["gradient_pairwise_spearman"]) for r in rows if r.get("gradient_pairwise_spearman") is not None]
        a_s = [_safe_float(r["ablation_pairwise_spearman"]) for r in rows if r.get("ablation_pairwise_spearman") is not None]
        g_t = [_safe_float(r["gradient_pairwise_topk_overlap"]) for r in rows if r.get("gradient_pairwise_topk_overlap") is not None]
        a_t = [_safe_float(r["ablation_pairwise_topk_overlap"]) for r in rows if r.get("ablation_pairwise_topk_overlap") is not None]
        return {
            "n_adapters": len(rows),
            "mean_gradient_pairwise_spearman": _mean_or_none(g_s),
            "mean_ablation_pairwise_spearman": _mean_or_none(a_s),
            "mean_gradient_pairwise_topk_overlap": _mean_or_none(g_t),
            "mean_ablation_pairwise_topk_overlap": _mean_or_none(a_t),
            "mean_gradient_minus_ablation_pairwise_spearman": (
                (_mean_or_none(g_s) - _mean_or_none(a_s)) if g_s and a_s else None
            ),
            "mean_gradient_minus_ablation_pairwise_topk_overlap": (
                (_mean_or_none(g_t) - _mean_or_none(a_t)) if g_t and a_t else None
            ),
        }

    stability_summary_all = _summarize_stability(stability_rows)
    stability_summary_by_family = []
    by_family_stability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        by_family_stability[row["task_family"]].append(row)
    for fam, vals in sorted(by_family_stability.items(), key=lambda x: rp._task_family_sort_key(x[0])):
        rec = _summarize_stability(vals)
        rec["task_family"] = fam
        stability_summary_by_family.append(rec)

    topk_summary_by_family_budget = []
    by_fb: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in topk_dist_rows:
        by_fb[(row["task_family"], float(row["budget_ratio"]))].append(row)
    for (fam, budget), vals in sorted(by_fb.items(), key=lambda x: (rp._task_family_sort_key(x[0][0]), x[0][1])):
        topk_summary_by_family_budget.append(
            {
                "task_family": fam,
                "budget_ratio": budget,
                "n": len(vals),
                "mean_topk_overlap": statistics.mean([_safe_float(v["topk_overlap"]) for v in vals]),
                "mean_abs_rank_dev_shared_topk": _mean_or_none(
                    [_safe_float(v["mean_abs_rank_dev_shared_topk"]) for v in vals if v["mean_abs_rank_dev_shared_topk"] is not None]
                ),
                "mean_abs_rank_dev_union_topk": _mean_or_none(
                    [_safe_float(v["mean_abs_rank_dev_union_topk"]) for v in vals if v["mean_abs_rank_dev_union_topk"] is not None]
                ),
                "mean_abs_rank_dev_all_layers": statistics.mean(
                    [_safe_float(v["mean_abs_rank_dev_all_layers"]) for v in vals]
                ),
            }
        )

    output = {
        "scope": {
            "informative_task_families": sorted(informative_families, key=rp._task_family_sort_key),
            "adapter_count": len(selected),
            "budgets": budgets,
            "repeats": args.repeats,
            "grad_samples": args.grad_samples,
            "ablation_samples": args.ablation_samples,
            "max_eval_samples": args.max_eval_samples,
        },
        "gradient_vs_ablation_concentration": {
            "by_adapter": adapter_perf_rows,
            "by_task_family": family_perf_rows,
            "summary": {
                "n_adapters": len(adapter_perf_rows),
                "mean_gradient_minus_ablation_delta_vs_uniform": _mean_or_none(diffs),
                "variance_gradient_minus_ablation_delta_vs_uniform": statistics.pvariance(diffs) if len(diffs) > 1 else 0.0,
                "gradient_better_adapter_count": sum(1 for x in diffs if x > 0),
                "ablation_better_adapter_count": sum(1 for x in diffs if x < 0),
                "top2_abs_contribution_share": top2_abs_share,
            },
        },
        "proxy_stability_resampling": {
            "by_adapter": stability_rows,
            "summary_all": stability_summary_all,
            "summary_by_task_family": stability_summary_by_family,
        },
        "topk_distribution_difference": {
            "by_repeat_budget": topk_dist_rows,
            "summary_by_task_family_budget": topk_summary_by_family_budget,
        },
        "interpretation_hint": {
            "question_1_ablation_noise": "Compare mean_gradient_minus_ablation_pairwise_spearman and topk overlap in proxy_stability_resampling.summary_all.",
            "question_2_win_concentration": "See gradient_vs_ablation_concentration.summary and by_adapter.",
            "question_3_topk_distribution": "See topk_distribution_difference.summary_by_task_family_budget.",
        },
    }

    (STUDY_DIR / "ablation_gradient_gap_investigation.json").write_text(json.dumps(output, indent=2))

    md = ["# Ablation vs Gradient Gap Investigation (Informative Subset)", ""]
    md.append("## Scope")
    md.append(f"- informative families: {', '.join(output['scope']['informative_task_families'])}")
    md.append(f"- adapters analyzed: {output['scope']['adapter_count']}")
    md.append(f"- budgets: {output['scope']['budgets']}")
    md.append(f"- repeats: {output['scope']['repeats']}")
    md.append(f"- grad_samples={output['scope']['grad_samples']}, ablation_samples={output['scope']['ablation_samples']}")
    md.append("")

    md.append("## Gradient vs Ablation Outcome Concentration")
    conc_rows = []
    for row in adapter_perf_rows:
        conc_rows.append(
            [
                row["adapter_id"],
                row["task_family"],
                _fmt(row["proxy_gradient_mean_delta_vs_uniform"], 4),
                _fmt(row["proxy_ablation_mean_delta_vs_uniform"], 4),
                _fmt(row["gradient_minus_ablation_mean_delta_vs_uniform"], 4),
            ]
        )
    md.append(
        _md_table(
            ["adapter_id", "task_family", "proxy_gradient_dvu", "proxy_ablation_dvu", "grad_minus_ablation"],
            conc_rows,
        )
    )
    s = output["gradient_vs_ablation_concentration"]["summary"]
    md.append(
        f"- summary: mean grad-minus-ablation={_fmt(s['mean_gradient_minus_ablation_delta_vs_uniform'], 4)}, "
        f"gradient_better={s['gradient_better_adapter_count']}, ablation_better={s['ablation_better_adapter_count']}, "
        f"top2_abs_contribution_share={_fmt(s['top2_abs_contribution_share'], 4)}."
    )
    md.append("")

    md.append("## Proxy Stability Under Resampling")
    st = output["proxy_stability_resampling"]["summary_all"]
    md.append(
        f"- mean pairwise Spearman: gradient={_fmt(st['mean_gradient_pairwise_spearman'], 3)}, "
        f"ablation={_fmt(st['mean_ablation_pairwise_spearman'], 3)}, "
        f"grad-minus-ablation={_fmt(st['mean_gradient_minus_ablation_pairwise_spearman'], 3)}."
    )
    md.append(
        f"- mean pairwise top-k overlap: gradient={_fmt(st['mean_gradient_pairwise_topk_overlap'], 3)}, "
        f"ablation={_fmt(st['mean_ablation_pairwise_topk_overlap'], 3)}, "
        f"grad-minus-ablation={_fmt(st['mean_gradient_minus_ablation_pairwise_topk_overlap'], 3)}."
    )
    md.append("")

    md.append("## Top-k Overlap vs Within-Top-k Distribution")
    topk_rows = []
    for row in topk_summary_by_family_budget:
        topk_rows.append(
            [
                row["task_family"],
                _fmt(row["budget_ratio"], 2),
                str(row["n"]),
                _fmt(row["mean_topk_overlap"], 3),
                _fmt(row["mean_abs_rank_dev_shared_topk"], 3),
                _fmt(row["mean_abs_rank_dev_union_topk"], 3),
                _fmt(row["mean_abs_rank_dev_all_layers"], 3),
            ]
        )
    md.append(
        _md_table(
            [
                "task_family",
                "budget",
                "n",
                "mean_topk_overlap",
                "mean_abs_dev_shared_topk",
                "mean_abs_dev_union_topk",
                "mean_abs_dev_all_layers",
            ],
            topk_rows,
        )
    )
    md.append("")
    md.append("## Caution")
    md.append("- This is a bounded CPU follow-up; treat as directional evidence under the current small informative subset.")

    (STUDY_DIR / "ablation_gradient_gap_investigation.md").write_text("\n".join(md) + "\n")

    print(json.dumps(output["scope"], indent=2))


if __name__ == "__main__":
    main()
