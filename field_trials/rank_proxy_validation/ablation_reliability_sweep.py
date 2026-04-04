#!/usr/bin/env python3
"""Ablation reliability sweep on informative (compressible) families only.

Focus:
1) Sample-budget sweep for ablation stability.
2) Fixed-panel consistency checks (dataset-matched).
3) Soft ablation variants (attenuation / optional rank reduction).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
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


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _infer_current_rank(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.ndim != 2 or b.ndim != 2:
        return 1
    if a.shape[0] == b.shape[1]:
        return int(a.shape[0])
    if a.shape[1] == b.shape[0]:
        return int(a.shape[1])
    return max(1, int(min(a.shape[0], a.shape[1], b.shape[0], b.shape[1])))


def _pair_count(n: int) -> int:
    return max(0, (n * (n - 1)) // 2)


def _cmp(a: float, b: float, atol: float = 1e-12) -> int:
    d = a - b
    if abs(d) <= atol:
        return 0
    return 1 if d > 0 else -1


def _kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    c = 0
    d = 0
    t_x = 0
    t_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = _cmp(x[i], x[j])
            dy = _cmp(y[i], y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                t_x += 1
                continue
            if dy == 0:
                t_y += 1
                continue
            if dx == dy:
                c += 1
            else:
                d += 1
    den = math.sqrt(float(c + d + t_x) * float(c + d + t_y))
    if den <= 0:
        return None
    return float(c - d) / den


def _goodman_kruskal_gamma(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    c = 0
    d = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = _cmp(x[i], x[j])
            dy = _cmp(y[i], y[j])
            if dx == 0 or dy == 0:
                continue
            if dx == dy:
                c += 1
            else:
                d += 1
    den = c + d
    if den <= 0:
        return None
    return float(c - d) / float(den)


def _pairwise_metric(
    vectors: list[list[float]],
    fn,
) -> tuple[float | None, int, int]:
    n = len(vectors)
    total_pairs = _pair_count(n)
    if total_pairs <= 0:
        return None, 0, 0
    vals: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = fn(vectors[i], vectors[j])
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v))
    if not vals:
        return None, 0, total_pairs
    return statistics.mean(vals), len(vals), total_pairs


def _topk_from_scores(layer_names: list[str], scores: list[float], k: int) -> set[str]:
    ranked = sorted(zip(layer_names, scores), key=lambda x: x[1], reverse=True)
    return {name for name, _ in ranked[:k]}


def _topk_overlap_from_vectors(layer_names: list[str], a_vec: list[float], b_vec: list[float], k: int) -> float:
    a = _topk_from_scores(layer_names, a_vec, k)
    b = _topk_from_scores(layer_names, b_vec, k)
    denom = max(1, min(k, len(a), len(b)))
    return float(len(a.intersection(b))) / float(denom)


def _pairwise_topk_overlap(
    layer_names: list[str],
    vectors: list[list[float]],
    k: int,
) -> tuple[float | None, int, int]:
    return _pairwise_metric(
        vectors,
        lambda a, b: _topk_overlap_from_vectors(layer_names, a, b, k),
    )


def _vector_information_profile(
    v: list[float],
    *,
    low_info_max_unique_values: int,
    low_info_min_nonzero_fraction: float,
    high_tie_pair_fraction_threshold: float,
) -> dict[str, Any]:
    n = len(v)
    if n <= 0:
        return {
            "unique_value_count": 0,
            "nonzero_fraction": 0.0,
            "tie_pair_fraction": 1.0,
            "is_flat": True,
            "is_low_information": True,
            "is_high_tie": True,
        }
    cnt = Counter(v)
    unique_count = len(cnt)
    nonzero = sum(1 for x in v if abs(float(x)) > 1e-12)
    nonzero_fraction = float(nonzero) / float(n)
    denom = _pair_count(n)
    tie_pairs = sum(c * (c - 1) // 2 for c in cnt.values())
    tie_pair_fraction = (float(tie_pairs) / float(denom)) if denom > 0 else 1.0
    is_flat = unique_count <= 1
    is_low_information = (
        is_flat
        or unique_count <= low_info_max_unique_values
        or nonzero_fraction <= low_info_min_nonzero_fraction
    )
    is_high_tie = tie_pair_fraction >= high_tie_pair_fraction_threshold
    return {
        "unique_value_count": unique_count,
        "nonzero_fraction": nonzero_fraction,
        "tie_pair_fraction": tie_pair_fraction,
        "is_flat": is_flat,
        "is_low_information": is_low_information,
        "is_high_tie": is_high_tie,
    }


def _vectorize_scores(layer_names: list[str], score_map: dict[str, float]) -> list[float]:
    return [_safe_float(score_map.get(name), 0.0) for name in layer_names]


def _sample_indices(n_total: int, n_keep: int, seed_text: str) -> list[int]:
    n_keep = min(n_keep, n_total)
    if n_keep <= 0:
        return []
    idx = list(range(n_total))
    rng = random.Random(_stable_int(seed_text))
    rng.shuffle(idx)
    return idx[:n_keep]


def _panel_texts_labels(texts: list[Any], labels: list[int], idx: list[int]) -> tuple[list[Any], list[int]]:
    return [texts[i] for i in idx], [labels[i] for i in idx]


def compute_ablation_scores_mode(
    model: torch.nn.Module,
    tokenizer,
    texts_panel: list[Any],
    labels_panel: list[int],
    layers: list[rp.LayerMeta],
    mode: str,
    attenuation_factor: float,
    rank_retain_ratio: float,
    eval_batch_size: int,
    max_length: int,
) -> dict[str, float]:
    if not texts_panel:
        return {layer.layer_name: 0.0 for layer in layers}

    params = dict(model.named_parameters())
    baseline = rp.evaluate_accuracy(
        model,
        tokenizer,
        texts_panel,
        labels_panel,
        batch_size=eval_batch_size,
        max_length=max_length,
    )

    scores: dict[str, float] = {}
    with torch.no_grad():
        for layer in layers:
            pa = params[layer.a_model_key]
            pb = params[layer.b_model_key]
            backup_a = pa.detach().clone()
            backup_b = pb.detach().clone()

            if mode == "hard_zero":
                pa.zero_()
                pb.zero_()
            elif mode == "attenuate":
                # Softer than zeroing: scale B only to reduce LoRA contribution linearly.
                pb.mul_(attenuation_factor)
            elif mode == "rank_reduction":
                current_r = _infer_current_rank(backup_a, backup_b)
                target_r = max(1, int(math.floor(current_r * rank_retain_ratio)))
                if target_r >= current_r and current_r > 1:
                    target_r = current_r - 1
                a_new, b_new = rp._truncate_and_pad_pair(backup_a, backup_b, target_r)
                pa.copy_(a_new.to(dtype=pa.dtype, device=pa.device))
                pb.copy_(b_new.to(dtype=pb.dtype, device=pb.device))
            else:
                raise ValueError(f"Unsupported ablation mode: {mode}")

            ablated = rp.evaluate_accuracy(
                model,
                tokenizer,
                texts_panel,
                labels_panel,
                batch_size=eval_batch_size,
                max_length=max_length,
            )
            pa.copy_(backup_a)
            pb.copy_(backup_b)
            scores[layer.layer_name] = max(0.0, baseline - ablated)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=str, default="0.35,0.50,0.65")
    parser.add_argument("--ablation-samples-grid", type=str, default="48,72,96")
    parser.add_argument("--fixed-panels", type=int, default=3)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--max-eval-samples", type=int, default=96)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--allow-online", action="store_true")
    parser.add_argument("--max-adapters", type=int, default=0, help="0 means no cap")
    parser.add_argument("--ablation-modes", type=str, default="hard_zero,attenuate")
    parser.add_argument("--attenuation-factor", type=float, default=0.5)
    parser.add_argument("--rank-retain-ratio", type=float, default=0.5)
    parser.add_argument("--low-info-max-unique-values", type=int, default=2)
    parser.add_argument("--low-info-min-nonzero-fraction", type=float, default=0.2)
    parser.add_argument("--high-tie-pair-fraction-threshold", type=float, default=0.8)
    args = parser.parse_args()

    if not args.allow_online:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    budgets = sorted({max(0.05, min(1.0, b)) for b in budgets})
    ablation_grid = [int(x.strip()) for x in args.ablation_samples_grid.split(",") if x.strip()]
    ablation_grid = sorted({max(8, x) for x in ablation_grid})
    modes = [x.strip() for x in args.ablation_modes.split(",") if x.strip()]

    quality = json.loads((STUDY_DIR / "source_quality_gap_control_slice.json").read_text())
    informative_families = set(quality["analysis_scope"]["informative_task_families"])
    informative_adapter_rows = quality["primary_informative_subset"]["adapter_quality_table"]
    informative_adapter_ids = [str(row["adapter_id"]) for row in informative_adapter_rows]

    records = rp.collect_adapter_records()
    selected: list[rp.AdapterRecord] = [records[aid] for aid in informative_adapter_ids if aid in records]
    if args.max_adapters > 0:
        selected = selected[: args.max_adapters]

    stability_rows: list[dict[str, Any]] = []
    policy_agreement_rows: list[dict[str, Any]] = []
    panel_cache: dict[tuple[str, int, int, int, str], list[list[int]]] = {}
    saturated_grid_records: list[dict[str, Any]] = []

    for adapter in selected:
        try:
            texts, labels = rp.load_eval_data(
                adapter.dataset,
                max_samples=args.max_eval_samples,
                allow_online=args.allow_online,
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
        oht_scores = {layer.layer_name: _safe_float(layer.policy_scores.get("oht"), 0.0) for layer in layers}
        oht_alloc_by_budget = {b: rp.allocate_ranks_greedy(layers, oht_scores, b) for b in budgets}

        n_total = len(texts)
        dataset_key = str(adapter.dataset)
        family = rp.task_family_for_dataset(dataset_key)

        effective_grid = [n for n in ablation_grid if n < n_total]
        saturated = [n for n in ablation_grid if n >= n_total]
        if saturated:
            saturated_grid_records.append(
                {
                    "adapter_id": adapter.adapter_id,
                    "dataset": dataset_key,
                    "task_family": family,
                    "n_total_eval_samples": n_total,
                    "requested_ablation_samples": saturated,
                }
            )
        if not effective_grid:
            continue

        for n_ablate in effective_grid:
            fixed_key_base = (dataset_key, n_total, n_ablate, args.fixed_panels, f"seed:{args.seed}")
            if fixed_key_base not in panel_cache:
                fixed_panels_idx: list[list[int]] = []
                for p in range(args.fixed_panels):
                    idx = _sample_indices(
                        n_total,
                        n_ablate,
                        f"fixed|{dataset_key}|{n_total}|{n_ablate}|panel:{p}|seed:{args.seed}",
                    )
                    fixed_panels_idx.append(idx)
                panel_cache[fixed_key_base] = fixed_panels_idx
            fixed_panels_idx = panel_cache[fixed_key_base]

            random_panels_idx: list[list[int]] = []
            for r in range(args.random_repeats):
                idx = _sample_indices(
                    n_total,
                    n_ablate,
                    f"random|{adapter.adapter_id}|{dataset_key}|{n_total}|{n_ablate}|rep:{r}|seed:{args.seed}",
                )
                random_panels_idx.append(idx)

            panel_sets = {
                "fixed": fixed_panels_idx,
                "random": random_panels_idx,
            }

            for mode in modes:
                for panel_type, idx_sets in panel_sets.items():
                    vectors: list[list[float]] = []
                    allocs_by_budget: dict[float, list[dict[str, int]]] = {b: [] for b in budgets}
                    for i, idx in enumerate(idx_sets):
                        panel_texts, panel_labels = _panel_texts_labels(texts, labels, idx)
                        if not panel_texts:
                            continue
                        score_map = compute_ablation_scores_mode(
                            model=model,
                            tokenizer=tokenizer,
                            texts_panel=panel_texts,
                            labels_panel=panel_labels,
                            layers=layers,
                            mode=mode,
                            attenuation_factor=args.attenuation_factor,
                            rank_retain_ratio=args.rank_retain_ratio,
                            eval_batch_size=args.eval_batch_size,
                            max_length=args.max_length,
                        )
                        vec = _vectorize_scores(layer_names, score_map)
                        vectors.append(vec)
                        for b in budgets:
                            alloc = rp.allocate_ranks_greedy(layers, score_map, b)
                            allocs_by_budget[b].append(alloc)

                    if len(vectors) >= 2:
                        k_top_q25 = max(1, len(layers) // 4)
                        k_top_q50 = max(1, len(layers) // 2)
                        sp_mean, sp_valid_pairs, total_pairs = _pairwise_metric(vectors, rp.spearman)
                        kd_mean, kd_valid_pairs, _ = _pairwise_metric(vectors, _kendall_tau_b)
                        gm_mean, gm_valid_pairs, _ = _pairwise_metric(vectors, _goodman_kruskal_gamma)
                        tk25_mean, tk25_valid_pairs, _ = _pairwise_topk_overlap(layer_names, vectors, k_top_q25)
                        tk50_mean, tk50_valid_pairs, _ = _pairwise_topk_overlap(layer_names, vectors, k_top_q50)
                        diagnostics = [
                            _vector_information_profile(
                                v,
                                low_info_max_unique_values=args.low_info_max_unique_values,
                                low_info_min_nonzero_fraction=args.low_info_min_nonzero_fraction,
                                high_tie_pair_fraction_threshold=args.high_tie_pair_fraction_threshold,
                            )
                            for v in vectors
                        ]
                        flat_count = sum(1 for d in diagnostics if d["is_flat"])
                        low_info_count = sum(1 for d in diagnostics if d["is_low_information"])
                        high_tie_count = sum(1 for d in diagnostics if d["is_high_tie"])
                        stability_rows.append(
                            {
                                "adapter_id": adapter.adapter_id,
                                "dataset": dataset_key,
                                "task_family": family,
                                "ablation_mode": mode,
                                "panel_type": panel_type,
                                "ablation_samples": n_ablate,
                                "n_panels": len(vectors),
                                "pairwise_total_pairs": total_pairs,
                                "pairwise_spearman": sp_mean,
                                "pairwise_spearman_valid_pairs": sp_valid_pairs,
                                "pairwise_kendall_tau_b": kd_mean,
                                "pairwise_kendall_valid_pairs": kd_valid_pairs,
                                "pairwise_goodman_kruskal_gamma": gm_mean,
                                "pairwise_gamma_valid_pairs": gm_valid_pairs,
                                "pairwise_topk_overlap_q25": tk25_mean,
                                "pairwise_topk_q25_valid_pairs": tk25_valid_pairs,
                                "pairwise_topk_overlap_q50": tk50_mean,
                                "pairwise_topk_q50_valid_pairs": tk50_valid_pairs,
                                # Backward-compatible alias.
                                "pairwise_topk_overlap": tk25_mean,
                                "flat_vector_count": flat_count,
                                "low_information_vector_count": low_info_count,
                                "high_tie_vector_count": high_tie_count,
                                "mean_unique_value_count": statistics.mean(d["unique_value_count"] for d in diagnostics),
                                "mean_nonzero_fraction": statistics.mean(d["nonzero_fraction"] for d in diagnostics),
                                "mean_tie_pair_fraction": statistics.mean(d["tie_pair_fraction"] for d in diagnostics),
                            }
                        )

                    for b in budgets:
                        allocs = allocs_by_budget[b]
                        if len(allocs) < 1:
                            continue
                        spearman_vals: list[float] = []
                        kendall_vals: list[float] = []
                        gamma_vals: list[float] = []
                        topk_vals: list[float] = []
                        oht_alloc = oht_alloc_by_budget[b]
                        oht_vec = [float(oht_alloc[name]) for name in layer_names]
                        k_top_alloc = max(1, len(layers) // 4)
                        for alloc in allocs:
                            proxy_vec = [float(alloc[name]) for name in layer_names]
                            rho = rp.spearman(proxy_vec, oht_vec)
                            if rho is not None:
                                spearman_vals.append(float(rho))
                            tau = _kendall_tau_b(proxy_vec, oht_vec)
                            if tau is not None:
                                kendall_vals.append(float(tau))
                            gamma = _goodman_kruskal_gamma(proxy_vec, oht_vec)
                            if gamma is not None:
                                gamma_vals.append(float(gamma))
                            topk_vals.append(rp.topk_overlap(alloc, oht_alloc, k_top_alloc))
                        policy_agreement_rows.append(
                            {
                                "adapter_id": adapter.adapter_id,
                                "dataset": dataset_key,
                                "task_family": family,
                                "ablation_mode": mode,
                                "panel_type": panel_type,
                                "ablation_samples": n_ablate,
                                "budget_ratio": b,
                                "n_panels": len(allocs),
                                "spearman_valid_panels": len(spearman_vals),
                                "kendall_valid_panels": len(kendall_vals),
                                "gamma_valid_panels": len(gamma_vals),
                                "mean_alloc_spearman_vs_oht": (
                                    statistics.mean(spearman_vals) if spearman_vals else None
                                ),
                                "mean_alloc_kendall_tau_b_vs_oht": (
                                    statistics.mean(kendall_vals) if kendall_vals else None
                                ),
                                "mean_alloc_goodman_kruskal_gamma_vs_oht": (
                                    statistics.mean(gamma_vals) if gamma_vals else None
                                ),
                                "mean_alloc_topk_overlap_vs_oht": (
                                    statistics.mean(topk_vals) if topk_vals else None
                                ),
                            }
                        )

    # Aggregate summaries.
    stability_summary: list[dict[str, Any]] = []
    grouped_stability: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        grouped_stability[(row["ablation_mode"], row["panel_type"], int(row["ablation_samples"]))].append(row)
    for (mode, panel_type, n_ablate), vals in sorted(grouped_stability.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        sp = [_safe_float(v["pairwise_spearman"]) for v in vals if v["pairwise_spearman"] is not None]
        kd = [_safe_float(v["pairwise_kendall_tau_b"]) for v in vals if v["pairwise_kendall_tau_b"] is not None]
        gm = [
            _safe_float(v["pairwise_goodman_kruskal_gamma"])
            for v in vals
            if v["pairwise_goodman_kruskal_gamma"] is not None
        ]
        tk25 = [_safe_float(v["pairwise_topk_overlap_q25"]) for v in vals if v["pairwise_topk_overlap_q25"] is not None]
        tk50 = [_safe_float(v["pairwise_topk_overlap_q50"]) for v in vals if v["pairwise_topk_overlap_q50"] is not None]
        total_pairs = sum(int(v.get("pairwise_total_pairs") or 0) for v in vals)
        sp_valid_pairs = sum(int(v.get("pairwise_spearman_valid_pairs") or 0) for v in vals)
        kd_valid_pairs = sum(int(v.get("pairwise_kendall_valid_pairs") or 0) for v in vals)
        gm_valid_pairs = sum(int(v.get("pairwise_gamma_valid_pairs") or 0) for v in vals)
        tk25_valid_pairs = sum(int(v.get("pairwise_topk_q25_valid_pairs") or 0) for v in vals)
        tk50_valid_pairs = sum(int(v.get("pairwise_topk_q50_valid_pairs") or 0) for v in vals)
        total_vectors = sum(int(v.get("n_panels") or 0) for v in vals)
        flat_vectors = sum(int(v.get("flat_vector_count") or 0) for v in vals)
        low_info_vectors = sum(int(v.get("low_information_vector_count") or 0) for v in vals)
        high_tie_vectors = sum(int(v.get("high_tie_vector_count") or 0) for v in vals)
        stability_summary.append(
            {
                "ablation_mode": mode,
                "panel_type": panel_type,
                "ablation_samples": n_ablate,
                "n_adapters": len(vals),
                "mean_pairwise_spearman": _mean_or_none(sp),
                "mean_pairwise_kendall_tau_b": _mean_or_none(kd),
                "mean_pairwise_goodman_kruskal_gamma": _mean_or_none(gm),
                "mean_pairwise_topk_overlap_q25": _mean_or_none(tk25),
                "mean_pairwise_topk_overlap_q50": _mean_or_none(tk50),
                # Backward-compatible alias.
                "mean_pairwise_topk_overlap": _mean_or_none(tk25),
                "pairwise_total_pairs": total_pairs,
                "pairwise_spearman_valid_pairs": sp_valid_pairs,
                "pairwise_kendall_valid_pairs": kd_valid_pairs,
                "pairwise_gamma_valid_pairs": gm_valid_pairs,
                "pairwise_topk_q25_valid_pairs": tk25_valid_pairs,
                "pairwise_topk_q50_valid_pairs": tk50_valid_pairs,
                "pairwise_spearman_valid_pair_fraction": (
                    float(sp_valid_pairs) / float(total_pairs) if total_pairs > 0 else None
                ),
                "pairwise_kendall_valid_pair_fraction": (
                    float(kd_valid_pairs) / float(total_pairs) if total_pairs > 0 else None
                ),
                "pairwise_gamma_valid_pair_fraction": (
                    float(gm_valid_pairs) / float(total_pairs) if total_pairs > 0 else None
                ),
                "flat_vector_count": flat_vectors,
                "low_information_vector_count": low_info_vectors,
                "high_tie_vector_count": high_tie_vectors,
                "total_vectors": total_vectors,
                "flat_vector_fraction": (float(flat_vectors) / float(total_vectors) if total_vectors > 0 else None),
                "low_information_vector_fraction": (
                    float(low_info_vectors) / float(total_vectors) if total_vectors > 0 else None
                ),
                "high_tie_vector_fraction": (
                    float(high_tie_vectors) / float(total_vectors) if total_vectors > 0 else None
                ),
                "mean_unique_value_count": _mean_or_none(
                    [_safe_float(v["mean_unique_value_count"]) for v in vals]
                ),
                "mean_nonzero_fraction": _mean_or_none(
                    [_safe_float(v["mean_nonzero_fraction"]) for v in vals]
                ),
                "mean_tie_pair_fraction": _mean_or_none(
                    [_safe_float(v["mean_tie_pair_fraction"]) for v in vals]
                ),
            }
        )

    policy_agreement_summary: list[dict[str, Any]] = []
    grouped_policy: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in policy_agreement_rows:
        grouped_policy[
            (row["ablation_mode"], row["panel_type"], int(row["ablation_samples"]), float(row["budget_ratio"]))
        ].append(row)
    for (mode, panel_type, n_ablate, budget), vals in sorted(
        grouped_policy.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])
    ):
        sp = [_safe_float(v["mean_alloc_spearman_vs_oht"]) for v in vals if v["mean_alloc_spearman_vs_oht"] is not None]
        kd = [
            _safe_float(v["mean_alloc_kendall_tau_b_vs_oht"])
            for v in vals
            if v["mean_alloc_kendall_tau_b_vs_oht"] is not None
        ]
        gm = [
            _safe_float(v["mean_alloc_goodman_kruskal_gamma_vs_oht"])
            for v in vals
            if v["mean_alloc_goodman_kruskal_gamma_vs_oht"] is not None
        ]
        tk = [_safe_float(v["mean_alloc_topk_overlap_vs_oht"]) for v in vals if v["mean_alloc_topk_overlap_vs_oht"] is not None]
        total_panels = sum(int(v.get("n_panels") or 0) for v in vals)
        sp_valid = sum(int(v.get("spearman_valid_panels") or 0) for v in vals)
        kd_valid = sum(int(v.get("kendall_valid_panels") or 0) for v in vals)
        gm_valid = sum(int(v.get("gamma_valid_panels") or 0) for v in vals)
        policy_agreement_summary.append(
            {
                "ablation_mode": mode,
                "panel_type": panel_type,
                "ablation_samples": n_ablate,
                "budget_ratio": budget,
                "n_adapters": len(vals),
                "mean_alloc_spearman_vs_oht": _mean_or_none(sp),
                "mean_alloc_kendall_tau_b_vs_oht": _mean_or_none(kd),
                "mean_alloc_goodman_kruskal_gamma_vs_oht": _mean_or_none(gm),
                "mean_alloc_topk_overlap_vs_oht": _mean_or_none(tk),
                "total_panels": total_panels,
                "spearman_valid_panels": sp_valid,
                "kendall_valid_panels": kd_valid,
                "gamma_valid_panels": gm_valid,
                "spearman_valid_panel_fraction": (float(sp_valid) / float(total_panels) if total_panels > 0 else None),
                "kendall_valid_panel_fraction": (float(kd_valid) / float(total_panels) if total_panels > 0 else None),
                "gamma_valid_panel_fraction": (float(gm_valid) / float(total_panels) if total_panels > 0 else None),
            }
        )

    # Material improvement deltas (max sample - min sample) per mode/panel and mode/panel/budget.
    min_n = min(ablation_grid) if ablation_grid else None
    max_n = max(ablation_grid) if ablation_grid else None
    stability_delta_rows: list[dict[str, Any]] = []
    if min_n is not None and max_n is not None and min_n != max_n:
        by_mode_panel: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in stability_summary:
            by_mode_panel[(row["ablation_mode"], row["panel_type"])][int(row["ablation_samples"])] = row
        for key, mapping in sorted(by_mode_panel.items()):
            if min_n not in mapping or max_n not in mapping:
                continue
            low = mapping[min_n]
            high = mapping[max_n]
            stability_delta_rows.append(
                {
                    "ablation_mode": key[0],
                    "panel_type": key[1],
                    "min_ablation_samples": min_n,
                    "max_ablation_samples": max_n,
                    "delta_pairwise_spearman": _safe_float(high["mean_pairwise_spearman"]) - _safe_float(low["mean_pairwise_spearman"]),
                    "delta_pairwise_kendall_tau_b": _safe_float(high["mean_pairwise_kendall_tau_b"]) - _safe_float(low["mean_pairwise_kendall_tau_b"]),
                    "delta_pairwise_goodman_kruskal_gamma": (
                        _safe_float(high["mean_pairwise_goodman_kruskal_gamma"])
                        - _safe_float(low["mean_pairwise_goodman_kruskal_gamma"])
                    ),
                    "delta_pairwise_topk_overlap_q25": (
                        _safe_float(high["mean_pairwise_topk_overlap_q25"])
                        - _safe_float(low["mean_pairwise_topk_overlap_q25"])
                    ),
                    "delta_pairwise_topk_overlap_q50": (
                        _safe_float(high["mean_pairwise_topk_overlap_q50"])
                        - _safe_float(low["mean_pairwise_topk_overlap_q50"])
                    ),
                    # Backward-compatible alias.
                    "delta_pairwise_topk_overlap": (
                        _safe_float(high["mean_pairwise_topk_overlap_q25"])
                        - _safe_float(low["mean_pairwise_topk_overlap_q25"])
                    ),
                }
            )

    policy_delta_rows: list[dict[str, Any]] = []
    if min_n is not None and max_n is not None and min_n != max_n:
        by_key: dict[tuple[str, str, float], dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in policy_agreement_summary:
            by_key[(row["ablation_mode"], row["panel_type"], float(row["budget_ratio"]))][
                int(row["ablation_samples"])
            ] = row
        for key, mapping in sorted(by_key.items()):
            if min_n not in mapping or max_n not in mapping:
                continue
            low = mapping[min_n]
            high = mapping[max_n]
            policy_delta_rows.append(
                {
                    "ablation_mode": key[0],
                    "panel_type": key[1],
                    "budget_ratio": key[2],
                    "min_ablation_samples": min_n,
                    "max_ablation_samples": max_n,
                    "delta_alloc_spearman_vs_oht": (
                        _safe_float(high["mean_alloc_spearman_vs_oht"]) - _safe_float(low["mean_alloc_spearman_vs_oht"])
                    ),
                    "delta_alloc_kendall_tau_b_vs_oht": (
                        _safe_float(high["mean_alloc_kendall_tau_b_vs_oht"])
                        - _safe_float(low["mean_alloc_kendall_tau_b_vs_oht"])
                    ),
                    "delta_alloc_goodman_kruskal_gamma_vs_oht": (
                        _safe_float(high["mean_alloc_goodman_kruskal_gamma_vs_oht"])
                        - _safe_float(low["mean_alloc_goodman_kruskal_gamma_vs_oht"])
                    ),
                    "delta_alloc_topk_overlap_vs_oht": (
                        _safe_float(high["mean_alloc_topk_overlap_vs_oht"]) - _safe_float(low["mean_alloc_topk_overlap_vs_oht"])
                    ),
                }
            )

    output = {
        "scope": {
            "informative_task_families": sorted(informative_families, key=rp._task_family_sort_key),
            "adapter_count": len(selected),
            "ablation_modes": modes,
            "ablation_samples_grid": ablation_grid,
            "fixed_panels": args.fixed_panels,
            "random_repeats": args.random_repeats,
            "budgets": budgets,
            "max_eval_samples": args.max_eval_samples,
            "low_info_max_unique_values": args.low_info_max_unique_values,
            "low_info_min_nonzero_fraction": args.low_info_min_nonzero_fraction,
            "high_tie_pair_fraction_threshold": args.high_tie_pair_fraction_threshold,
        },
        "saturated_ablation_grid_records": saturated_grid_records,
        "stability_by_adapter": stability_rows,
        "stability_summary": stability_summary,
        "policy_agreement_by_adapter": policy_agreement_rows,
        "policy_agreement_summary": policy_agreement_summary,
        "stability_delta_max_minus_min_samples": stability_delta_rows,
        "policy_agreement_delta_max_minus_min_samples": policy_delta_rows,
        "interpretation_hint": {
            "budget_effect_test": "Use stability_delta_max_minus_min_samples to test if larger ablation panels materially improve stability.",
            "fixed_vs_random_test": "Compare panel_type=fixed vs panel_type=random in stability_summary.",
            "soft_vs_hard_test": "Compare ablation_mode=attenuate (and optionally rank_reduction) against hard_zero.",
            "tie_aware_metrics": "Use pairwise_kendall_tau_b and pairwise_goodman_kruskal_gamma when Spearman validity is sparse.",
            "low_information_flags": "Use flat_vector_fraction, low_information_vector_fraction, and high_tie_vector_fraction to detect low-information ablation panels.",
        },
    }

    (STUDY_DIR / "ablation_reliability_sweep.json").write_text(json.dumps(output, indent=2))

    md = ["# Ablation Reliability Sweep (Informative Subset)", ""]
    md.append("## Scope")
    md.append(f"- informative families: {', '.join(output['scope']['informative_task_families'])}")
    md.append(f"- adapters analyzed: {output['scope']['adapter_count']}")
    md.append(f"- ablation modes: {', '.join(output['scope']['ablation_modes'])}")
    md.append(f"- ablation sample grid: {output['scope']['ablation_samples_grid']}")
    md.append(f"- fixed panels per setting: {output['scope']['fixed_panels']}")
    md.append(f"- random repeats per setting: {output['scope']['random_repeats']}")
    md.append(f"- budgets: {output['scope']['budgets']}")
    md.append(
        "- low-info flags:"
        f" max_unique<={output['scope']['low_info_max_unique_values']},"
        f" min_nonzero_fraction<={output['scope']['low_info_min_nonzero_fraction']},"
        f" high_tie_pair_fraction>={output['scope']['high_tie_pair_fraction_threshold']}"
    )
    if saturated_grid_records:
        md.append(f"- saturated grid entries skipped (requested ablation_samples >= eval samples): {len(saturated_grid_records)} adapter-level records")
    md.append("")

    md.append("## Stability Summary")
    st_rows = []
    for row in stability_summary:
        st_rows.append(
            [
                row["ablation_mode"],
                row["panel_type"],
                str(row["ablation_samples"]),
                str(row["n_adapters"]),
                _fmt(row["mean_pairwise_spearman"], 3),
                _fmt(row["mean_pairwise_kendall_tau_b"], 3),
                _fmt(row["mean_pairwise_goodman_kruskal_gamma"], 3),
                _fmt(row["mean_pairwise_topk_overlap_q25"], 3),
                _fmt(row["mean_pairwise_topk_overlap_q50"], 3),
                _fmt(row["pairwise_spearman_valid_pair_fraction"], 3),
                _fmt(row["flat_vector_fraction"], 3),
                _fmt(row["low_information_vector_fraction"], 3),
                _fmt(row["high_tie_vector_fraction"], 3),
            ]
        )
    md.append(
        _md_table(
            [
                "mode",
                "panel_type",
                "ablation_samples",
                "n_adapters",
                "mean_spearman",
                "mean_kendall_tau_b",
                "mean_gamma",
                "mean_topk_q25",
                "mean_topk_q50",
                "spearman_valid_pair_fraction",
                "flat_vector_fraction",
                "low_info_vector_fraction",
                "high_tie_vector_fraction",
            ],
            st_rows,
        )
    )
    md.append("")

    md.append("## Stability Improvement (Max vs Min Sample)")
    delta_rows = []
    for row in stability_delta_rows:
        delta_rows.append(
            [
                row["ablation_mode"],
                row["panel_type"],
                f"{row['min_ablation_samples']}->{row['max_ablation_samples']}",
                _fmt(row["delta_pairwise_spearman"], 3),
                _fmt(row["delta_pairwise_kendall_tau_b"], 3),
                _fmt(row["delta_pairwise_goodman_kruskal_gamma"], 3),
                _fmt(row["delta_pairwise_topk_overlap_q25"], 3),
                _fmt(row["delta_pairwise_topk_overlap_q50"], 3),
            ]
        )
    md.append(
        _md_table(
            [
                "mode",
                "panel_type",
                "sample_range",
                "delta_spearman",
                "delta_kendall_tau_b",
                "delta_gamma",
                "delta_topk_q25",
                "delta_topk_q50",
            ],
            delta_rows,
        )
    )
    md.append("")

    md.append("## Policy Agreement vs OHT Summary")
    pa_rows = []
    for row in policy_agreement_summary:
        pa_rows.append(
            [
                row["ablation_mode"],
                row["panel_type"],
                str(row["ablation_samples"]),
                _fmt(row["budget_ratio"], 2),
                str(row["n_adapters"]),
                _fmt(row["mean_alloc_spearman_vs_oht"], 3),
                _fmt(row["mean_alloc_kendall_tau_b_vs_oht"], 3),
                _fmt(row["mean_alloc_goodman_kruskal_gamma_vs_oht"], 3),
                _fmt(row["mean_alloc_topk_overlap_vs_oht"], 3),
                _fmt(row["spearman_valid_panel_fraction"], 3),
            ]
        )
    md.append(
        _md_table(
            [
                "mode",
                "panel_type",
                "ablation_samples",
                "budget",
                "n_adapters",
                "mean_alloc_spearman_vs_oht",
                "mean_alloc_kendall_vs_oht",
                "mean_alloc_gamma_vs_oht",
                "mean_alloc_topk_overlap_vs_oht",
                "spearman_valid_panel_fraction",
            ],
            pa_rows,
        )
    )
    md.append("")

    md.append("## Policy Agreement Change (Max vs Min Sample)")
    pd_rows = []
    for row in policy_delta_rows:
        pd_rows.append(
            [
                row["ablation_mode"],
                row["panel_type"],
                _fmt(row["budget_ratio"], 2),
                f"{row['min_ablation_samples']}->{row['max_ablation_samples']}",
                _fmt(row["delta_alloc_spearman_vs_oht"], 3),
                _fmt(row["delta_alloc_kendall_tau_b_vs_oht"], 3),
                _fmt(row["delta_alloc_goodman_kruskal_gamma_vs_oht"], 3),
                _fmt(row["delta_alloc_topk_overlap_vs_oht"], 3),
            ]
        )
    md.append(
        _md_table(
            [
                "mode",
                "panel_type",
                "budget",
                "sample_range",
                "delta_alloc_spearman_vs_oht",
                "delta_alloc_kendall_vs_oht",
                "delta_alloc_gamma_vs_oht",
                "delta_alloc_topk_vs_oht",
            ],
            pd_rows,
        )
    )
    md.append("")
    md.append("## Caution")
    md.append("- This sweep is informative-subset-only and CPU-bounded; treat as directional evidence.")

    (STUDY_DIR / "ablation_reliability_sweep.md").write_text("\n".join(md) + "\n")
    print(json.dumps(output["scope"], indent=2))


if __name__ == "__main__":
    main()
