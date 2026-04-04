#!/usr/bin/env python3
"""CPU-only rank-policy vs proxy validation study.

Compares Gradience spectral rank suggestions against lightweight offline
importance proxies (gradient norm, ablation sensitivity) under fixed
compression budgets for small encoder adapters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import DownloadConfig, load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gradience.vnext.audit.lora_audit import (
    audit_lora_peft_dir,
    find_peft_files,
    load_adapter_state_dict,
)
from gradience.vnext.svd_truncate import _compute_svd_truncation


REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_TRIALS_DIR = REPO_ROOT / "field_trials"
STUDY_DIR = Path(__file__).resolve().parent
STRATEGY_DOC = REPO_ROOT / "docs" / "strategy" / "rank_proxy_validation_summary.md"

STRICT_RERUN_RESULTS = FIELD_TRIALS_DIR / "over_accumulation_followup" / "strict_naive_rerun_results.json"

POLICY_NAMES = ["energy_90", "knee", "erank", "oht", "stable_rank_ceil"]
PROXY_NAMES = ["proxy_gradient", "proxy_ablation"]
BASELINE_NAMES = ["baseline_uniform", "baseline_random"]
ALL_METHODS = POLICY_NAMES + PROXY_NAMES + BASELINE_NAMES
PRIORITY_TASK_FAMILIES = ["sst2", "imdb", "tweet_eval", "ag_news"]
SOURCE_QUALITY_BANDS = ["near_top", "mid_gap", "large_gap", "single_source_dataset"]

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "imdb": {
        "hf_name": "imdb",
        "hf_config": None,
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
    },
    "sst2": {
        "hf_name": "sst2",
        "hf_config": None,
        "split": "validation",
        "text_col": "sentence",
        "label_col": "label",
        "num_labels": 2,
    },
    "tweet_eval/emotion": {
        "hf_name": "tweet_eval",
        "hf_config": "emotion",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 4,
    },
    "tweet_eval/hate": {
        "hf_name": "tweet_eval",
        "hf_config": "hate",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
    },
    "tweet_eval/irony": {
        "hf_name": "tweet_eval",
        "hf_config": "irony",
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
    },
    "ag_news": {
        "hf_name": "ag_news",
        "hf_config": None,
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 4,
    },
    "mnli": {
        "hf_name": "nyu-mll/multi_nli",
        "hf_config": None,
        "split": "validation_matched",
        "text_col": "premise",
        "text_col_b": "hypothesis",
        "label_col": "label",
        "num_labels": 3,
    },
    "yelp_polarity": {
        "hf_name": "yelp_polarity",
        "hf_config": None,
        "split": "test",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
    },
    "amazon_polarity": {
        "hf_name": "amazon_polarity",
        "hf_config": None,
        "split": "test",
        "text_col": "content",
        "label_col": "label",
        "num_labels": 2,
    },
}


@dataclass(frozen=True)
class AdapterRecord:
    adapter_id: str
    local_name: str
    adapter_dir: Path
    dataset: str
    base_model: str
    model_family: str
    num_labels: int


@dataclass(frozen=True)
class LayerMeta:
    layer_name: str
    module_type: str
    a_sd_key: str
    b_sd_key: str
    a_model_key: str
    b_model_key: str
    max_r: int
    rank_unit_cost: int
    policy_scores: dict[str, float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _model_family(base_model: str | None) -> str:
    b = (base_model or "").lower()
    if "distilbert" in b:
        return "distilbert"
    if "roberta" in b:
        return "roberta"
    if "bert" in b and "roberta" not in b and "distilbert" not in b:
        return "bert"
    return b or "unknown"


def _inventory_dirs() -> list[Path]:
    dirs: list[Path] = []
    dirs.extend(p for p in FIELD_TRIALS_DIR.glob("inventory_*") if (p / "adapter_cache").exists() and (p / "evidence").exists())
    dirs.extend(
        p
        for p in FIELD_TRIALS_DIR.glob("targeted_confirmation_*/inventory_*")
        if (p / "adapter_cache").exists() and (p / "evidence").exists()
    )
    dirs.extend(
        p
        for p in FIELD_TRIALS_DIR.glob("task_family_equivalence/inventory_*")
        if (p / "adapter_cache").exists() and (p / "evidence").exists()
    )
    seen: set[str] = set()
    out: list[Path] = []
    for d in sorted(dirs):
        key = d.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def collect_adapter_records() -> dict[str, AdapterRecord]:
    records: dict[str, AdapterRecord] = {}
    for inv in _inventory_dirs():
        ev_dir = inv / "evidence"
        cache_dir = inv / "adapter_cache"
        for ev_path in sorted(ev_dir.glob("*.json")):
            try:
                ev = _json(ev_path)
            except Exception:
                continue
            adapter_id = str(ev.get("adapter_id") or ev_path.stem)
            dataset = ev.get("eval_dataset")
            base_model = ev.get("base_model")
            if not isinstance(dataset, str) or dataset not in DATASET_REGISTRY:
                continue
            if not isinstance(base_model, str):
                continue
            local_name = ev_path.stem
            adapter_dir = cache_dir / local_name
            if not adapter_dir.exists():
                continue
            num_labels = DATASET_REGISTRY[dataset]["num_labels"]
            candidate = AdapterRecord(
                adapter_id=adapter_id,
                local_name=local_name,
                adapter_dir=adapter_dir,
                dataset=dataset,
                base_model=base_model,
                model_family=_model_family(base_model),
                num_labels=num_labels,
            )
            if adapter_id not in records:
                records[adapter_id] = candidate
    return records


def select_pilot_adapters(records: dict[str, AdapterRecord], max_adapters: int) -> list[AdapterRecord]:
    reruns = _json(STRICT_RERUN_RESULTS)
    ordered_ids: list[str] = []
    for row in reruns:
        for key in ("adapter_a_id", "adapter_b_id"):
            aid = row.get(key)
            if isinstance(aid, str) and aid not in ordered_ids:
                ordered_ids.append(aid)

    candidates = [records[aid] for aid in ordered_ids if aid in records]
    distilbert_candidates = [r for r in candidates if r.model_family == "distilbert"]
    pool = distilbert_candidates if distilbert_candidates else candidates

    selected: list[AdapterRecord] = []
    seen_dataset: set[str] = set()
    for rec in pool:
        if rec.dataset in seen_dataset:
            continue
        selected.append(rec)
        seen_dataset.add(rec.dataset)
        if len(selected) >= max_adapters:
            return selected

    if len(selected) < max_adapters:
        for rec in pool:
            if rec in selected:
                continue
            selected.append(rec)
            if len(selected) >= max_adapters:
                break

    return selected


def load_eval_data(dataset_key: str, max_samples: int, allow_online: bool) -> tuple[list[Any], list[int]]:
    info = DATASET_REGISTRY[dataset_key]
    if allow_online:
        ds = load_dataset(info["hf_name"], info["hf_config"], split=info["split"])
    else:
        dcfg = DownloadConfig(local_files_only=True)
        ds = load_dataset(
            info["hf_name"],
            info["hf_config"],
            split=info["split"],
            download_config=dcfg,
            download_mode="reuse_cache_if_exists",
        )
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    texts = ds[info["text_col"]]
    if "text_col_b" in info:
        texts = list(zip(texts, ds[info["text_col_b"]]))
    labels = [int(x) for x in ds[info["label_col"]]]
    return list(texts), labels


def _tokenize_batch(tokenizer, batch: list[Any], max_length: int) -> dict[str, torch.Tensor]:
    if isinstance(batch[0], tuple):
        texts_a, texts_b = zip(*batch)
        return tokenizer(
            list(texts_a),
            list(texts_b),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    return tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def evaluate_accuracy(
    model: torch.nn.Module,
    tokenizer,
    texts: list[Any],
    labels: list[int],
    batch_size: int,
    max_length: int,
) -> float:
    model.eval()
    preds: list[int] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = _tokenize_batch(tokenizer, batch, max_length=max_length)
        with torch.no_grad():
            logits = model(**inputs).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    if not labels:
        return 0.0
    correct = sum(1 for p, y in zip(preds, labels) if int(p) == int(y))
    return correct / len(labels)


def _model_key_from_sd_key(sd_key: str, model_param_keys: set[str]) -> str | None:
    if sd_key in model_param_keys:
        return sd_key
    if ".lora_A.weight" in sd_key:
        candidate = sd_key.replace(".lora_A.weight", ".lora_A.default.weight")
        if candidate in model_param_keys:
            return candidate
    if ".lora_B.weight" in sd_key:
        candidate = sd_key.replace(".lora_B.weight", ".lora_B.default.weight")
        if candidate in model_param_keys:
            return candidate
    return None


def _rank_unit_cost(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    if a.ndim != 2 or b.ndim != 2:
        return (1, 1)
    if a.shape[0] == b.shape[1]:
        # Standard orientation: A(r, d_in), B(d_out, r)
        return int(a.shape[1] + b.shape[0]), int(a.shape[0])
    if a.shape[1] == b.shape[0]:
        # Transposed orientation
        return int(a.shape[0] + b.shape[1]), int(a.shape[1])
    # Fallback
    r_guess = int(min(max(a.shape), max(b.shape)))
    return int(max(1, a.numel() + b.numel()) // max(1, r_guess)), int(max(1, r_guess))


def _extract_layer_meta(
    adapter: AdapterRecord,
    adapter_sd: dict[str, torch.Tensor],
    model_param_keys: set[str],
) -> list[LayerMeta]:
    audit = audit_lora_peft_dir(
        adapter.adapter_dir,
        compute_udr=False,
        rank_policies=["knee_elbow", "entropy_effective", "optimal_hard_threshold", "stable_rank_ceil"],
    )
    layers: list[LayerMeta] = []
    for layer in audit.layers:
        a_sd = layer.a_key
        b_sd = layer.b_key
        if a_sd not in adapter_sd or b_sd not in adapter_sd:
            continue
        a_model = _model_key_from_sd_key(a_sd, model_param_keys)
        b_model = _model_key_from_sd_key(b_sd, model_param_keys)
        if a_model is None or b_model is None:
            continue
        a_tensor = adapter_sd[a_sd]
        b_tensor = adapter_sd[b_sd]
        cost, max_r = _rank_unit_cost(a_tensor, b_tensor)
        rs = layer.rank_suggestions or {}
        policy_scores = {
            "energy_90": _safe_float((rs.get("energy_90") or {}).get("k"), default=float(layer.energy_rank_90)),
            "knee": _safe_float((rs.get("knee") or {}).get("k"), default=float(layer.energy_rank_90)),
            "erank": _safe_float((rs.get("erank") or {}).get("k"), default=float(layer.effective_rank)),
            "oht": _safe_float((rs.get("oht") or {}).get("k"), default=float(layer.energy_rank_90)),
            "stable_rank_ceil": _safe_float(
                (rs.get("stable_rank_ceil") or {}).get("k"),
                default=max(1.0, float(math.ceil(layer.stable_rank))),
            ),
        }
        layers.append(
            LayerMeta(
                layer_name=layer.name,
                module_type=layer.module_type,
                a_sd_key=a_sd,
                b_sd_key=b_sd,
                a_model_key=a_model,
                b_model_key=b_model,
                max_r=max_r,
                rank_unit_cost=max(1, cost),
                policy_scores=policy_scores,
            )
        )
    return layers


def compute_gradient_proxy_scores(
    model: torch.nn.Module,
    tokenizer,
    texts: list[Any],
    labels: list[int],
    layers: list[LayerMeta],
    grad_samples: int,
    batch_size: int,
    max_length: int,
) -> dict[str, float]:
    model.eval()
    params = dict(model.named_parameters())
    # Some adapters are loaded in inference mode; explicitly enable grads on LoRA tensors.
    for layer in layers:
        if layer.a_model_key in params:
            params[layer.a_model_key].requires_grad_(True)
        if layer.b_model_key in params:
            params[layer.b_model_key].requires_grad_(True)
    model.zero_grad(set_to_none=True)
    n = min(grad_samples, len(texts))
    if n == 0:
        return {layer.layer_name: 0.0 for layer in layers}
    with torch.enable_grad():
        for i in range(0, n, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]
            inputs = _tokenize_batch(tokenizer, batch_texts, max_length=max_length)
            y = torch.tensor(batch_labels, dtype=torch.long)
            outputs = model(**inputs, labels=y)
            loss = outputs.loss
            loss.backward()

    scores: dict[str, float] = {}
    for layer in layers:
        ga = params[layer.a_model_key].grad
        gb = params[layer.b_model_key].grad
        norm_sq = 0.0
        if ga is not None:
            norm_sq += float(torch.sum(ga.detach().float() ** 2).cpu().item())
        if gb is not None:
            norm_sq += float(torch.sum(gb.detach().float() ** 2).cpu().item())
        scores[layer.layer_name] = math.sqrt(max(norm_sq, 0.0))
    model.zero_grad(set_to_none=True)
    return scores


def _truncate_and_pad_pair(A: torch.Tensor, B: torch.Tensor, target_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    A_cpu = A.detach().cpu()
    B_cpu = B.detach().cpu()
    transposed = False

    if A_cpu.ndim != 2 or B_cpu.ndim != 2:
        return A_cpu.clone(), B_cpu.clone()

    if A_cpu.shape[0] == B_cpu.shape[1]:
        A_std = A_cpu
        B_std = B_cpu
    elif A_cpu.shape[1] == B_cpu.shape[0]:
        A_std = A_cpu.T.contiguous()
        B_std = B_cpu.T.contiguous()
        transposed = True
    else:
        return A_cpu.clone(), B_cpu.clone()

    r = int(A_std.shape[0])
    k = max(1, min(int(target_rank), r))
    if k >= r:
        return A_cpu.clone(), B_cpu.clone()

    A_new, B_new, _ = _compute_svd_truncation(A_std, B_std, k)

    A_pad = torch.zeros_like(A_std)
    B_pad = torch.zeros_like(B_std)
    A_pad[:k, :] = A_new.to(dtype=A_pad.dtype)
    B_pad[:, :k] = B_new.to(dtype=B_pad.dtype)

    if transposed:
        return A_pad.T.contiguous(), B_pad.T.contiguous()
    return A_pad, B_pad


def apply_allocation_to_model(
    model: torch.nn.Module,
    adapter_sd: dict[str, torch.Tensor],
    layers: list[LayerMeta],
    allocation: dict[str, int],
) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for layer in layers:
            target_r = int(allocation[layer.layer_name])
            A_orig = adapter_sd[layer.a_sd_key]
            B_orig = adapter_sd[layer.b_sd_key]
            A_new, B_new = _truncate_and_pad_pair(A_orig, B_orig, target_r)
            params[layer.a_model_key].copy_(A_new.to(dtype=params[layer.a_model_key].dtype, device=params[layer.a_model_key].device))
            params[layer.b_model_key].copy_(B_new.to(dtype=params[layer.b_model_key].dtype, device=params[layer.b_model_key].device))


def compute_ablation_proxy_scores(
    model: torch.nn.Module,
    tokenizer,
    texts: list[Any],
    labels: list[int],
    layers: list[LayerMeta],
    ablation_samples: int,
    batch_size: int,
    max_length: int,
) -> dict[str, float]:
    n = min(ablation_samples, len(texts))
    if n == 0:
        return {layer.layer_name: 0.0 for layer in layers}

    texts_sub = texts[:n]
    labels_sub = labels[:n]
    params = dict(model.named_parameters())
    baseline = evaluate_accuracy(model, tokenizer, texts_sub, labels_sub, batch_size=batch_size, max_length=max_length)

    scores: dict[str, float] = {}
    with torch.no_grad():
        for layer in layers:
            pa = params[layer.a_model_key]
            pb = params[layer.b_model_key]
            backup_a = pa.detach().clone()
            backup_b = pb.detach().clone()
            pa.zero_()
            pb.zero_()
            ablated = evaluate_accuracy(model, tokenizer, texts_sub, labels_sub, batch_size=batch_size, max_length=max_length)
            pa.copy_(backup_a)
            pb.copy_(backup_b)
            scores[layer.layer_name] = max(0.0, baseline - ablated)
    return scores


def allocate_ranks_greedy(
    layers: list[LayerMeta],
    score_map: dict[str, float],
    budget_ratio: float,
) -> dict[str, int]:
    rank_unit_cost = {l.layer_name: l.rank_unit_cost for l in layers}
    max_r = {l.layer_name: l.max_r for l in layers}
    layer_names = [l.layer_name for l in layers]

    max_params = sum(rank_unit_cost[name] * max_r[name] for name in layer_names)
    min_params = sum(rank_unit_cost[name] for name in layer_names)  # min rank=1
    target_params = int(round(max_params * budget_ratio))
    target_params = max(min_params, min(target_params, max_params))

    alloc = {name: 1 for name in layer_names}
    used = min_params

    while used < target_params:
        best_name = None
        best_value = -1.0
        for name in layer_names:
            if alloc[name] >= max_r[name]:
                continue
            c = rank_unit_cost[name]
            if used + c > target_params:
                continue
            score = max(0.0, _safe_float(score_map.get(name), default=0.0))
            value = score / max(1e-9, float(c))
            if value > best_value:
                best_value = value
                best_name = name
        if best_name is None:
            break
        alloc[best_name] += 1
        used += rank_unit_cost[best_name]

    return alloc


def _rank(values: list[float]) -> list[float]:
    ordered = sorted((v, i) for i, v in enumerate(values))
    out = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rv = ((i + j - 1) / 2.0) + 1.0
        for k in range(i, j):
            out[ordered[k][1]] = rv
        i = j
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx = _rank(x)
    ry = _rank(y)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx)
    den_y = sum((b - my) ** 2 for b in ry)
    den = math.sqrt(den_x * den_y)
    if den <= 0:
        return None
    return num / den


def topk_overlap(a: dict[str, int], b: dict[str, int], k: int) -> float:
    if not a or not b or k <= 0:
        return 0.0
    a_top = {x[0] for x in sorted(a.items(), key=lambda x: x[1], reverse=True)[:k]}
    b_top = {x[0] for x in sorted(b.items(), key=lambda x: x[1], reverse=True)[:k]}
    denom = max(1, min(k, len(a_top), len(b_top)))
    return len(a_top.intersection(b_top)) / denom


def module_budget_splits(layers: list[LayerMeta], alloc: dict[str, int]) -> tuple[float, float]:
    total = 0.0
    attn = 0.0
    mlp = 0.0
    for layer in layers:
        p = layer.rank_unit_cost * alloc[layer.layer_name]
        total += p
        if layer.module_type == "attn":
            attn += p
        elif layer.module_type == "mlp":
            mlp += p
    if total <= 0:
        return 0.0, 0.0
    return attn / total, mlp / total


def task_family_for_dataset(dataset: str) -> str:
    d = (dataset or "").strip()
    if d.startswith("tweet_eval/"):
        return "tweet_eval"
    if d in {"sst2", "imdb", "ag_news"}:
        return d
    return f"other:{d}" if d else "other:unknown"


def _task_family_sort_key(family: str) -> tuple[int, int, str]:
    if family in PRIORITY_TASK_FAMILIES:
        return (0, PRIORITY_TASK_FAMILIES.index(family), family)
    return (1, 0, family)


def source_quality_gap_band(gap: float, dataset_count: int) -> str:
    if dataset_count < 2:
        return "single_source_dataset"
    g = max(0.0, float(gap))
    if g <= 0.01:
        return "near_top"
    if g <= 0.05:
        return "mid_gap"
    return "large_gap"


def _source_quality_band_sort_key(band: str) -> tuple[int, str]:
    if band in SOURCE_QUALITY_BANDS:
        return (SOURCE_QUALITY_BANDS.index(band), band)
    return (len(SOURCE_QUALITY_BANDS), band)


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.allow_online:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    STRATEGY_DOC.parent.mkdir(parents=True, exist_ok=True)

    records = collect_adapter_records()
    selected = select_pilot_adapters(records, max_adapters=args.max_adapters)

    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    budgets = sorted({max(0.05, min(1.0, b)) for b in budgets})

    allocation_rows: list[dict[str, Any]] = []
    compression_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for adapter in selected:
        try:
            texts, labels = load_eval_data(adapter.dataset, max_samples=args.max_eval_samples, allow_online=args.allow_online)
        except Exception as e:
            skipped.append({"adapter_id": adapter.adapter_id, "reason": f"dataset_load_failed: {e}"})
            continue

        model_id = adapter.base_model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=(not args.allow_online))
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                num_labels=adapter.num_labels,
                local_files_only=(not args.allow_online),
            )
            model = PeftModel.from_pretrained(base_model, str(adapter.adapter_dir))
            model.eval()
        except Exception as e:
            skipped.append({"adapter_id": adapter.adapter_id, "reason": f"model_load_failed: {e}"})
            continue

        try:
            _, weights_path, _ = find_peft_files(adapter.adapter_dir)
            if weights_path is None:
                skipped.append({"adapter_id": adapter.adapter_id, "reason": "weights_not_found"})
                continue
            adapter_sd = load_adapter_state_dict(weights_path)
        except Exception as e:
            skipped.append({"adapter_id": adapter.adapter_id, "reason": f"adapter_state_load_failed: {e}"})
            continue

        model_param_keys = {k for k, _ in model.named_parameters()}
        layers = _extract_layer_meta(adapter, adapter_sd, model_param_keys)
        if not layers:
            skipped.append({"adapter_id": adapter.adapter_id, "reason": "no_layer_meta"})
            continue

        full_acc = evaluate_accuracy(model, tokenizer, texts, labels, batch_size=args.eval_batch_size, max_length=args.max_length)

        grad_scores = compute_gradient_proxy_scores(
            model,
            tokenizer,
            texts,
            labels,
            layers,
            grad_samples=args.grad_samples,
            batch_size=args.grad_batch_size,
            max_length=args.max_length,
        )
        ablation_scores = compute_ablation_proxy_scores(
            model,
            tokenizer,
            texts,
            labels,
            layers,
            ablation_samples=args.ablation_samples,
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
        )

        scores_by_method: dict[str, dict[str, float]] = {}
        for policy in POLICY_NAMES:
            scores_by_method[policy] = {layer.layer_name: _safe_float(layer.policy_scores.get(policy), default=0.0) for layer in layers}
        scores_by_method["proxy_gradient"] = grad_scores
        scores_by_method["proxy_ablation"] = ablation_scores
        scores_by_method["baseline_uniform"] = {layer.layer_name: 1.0 for layer in layers}

        per_budget_allocs: dict[tuple[str, float], dict[str, int]] = {}
        per_budget_acc: dict[tuple[str, float], float] = {}

        rng = random.Random(args.seed + hash(adapter.adapter_id))
        for budget in budgets:
            random_scores = {layer.layer_name: rng.random() for layer in layers}
            scores_by_method["baseline_random"] = random_scores

            for method in ALL_METHODS:
                alloc = allocate_ranks_greedy(layers, scores_by_method[method], budget_ratio=budget)
                per_budget_allocs[(method, budget)] = alloc
                apply_allocation_to_model(model, adapter_sd, layers, alloc)
                acc = evaluate_accuracy(
                    model,
                    tokenizer,
                    texts,
                    labels,
                    batch_size=args.eval_batch_size,
                    max_length=args.max_length,
                )
                per_budget_acc[(method, budget)] = acc

        # Restore original adapter weights before moving on.
        apply_allocation_to_model(model, adapter_sd, layers, {layer.layer_name: layer.max_r for layer in layers})

        # Compression rows
        for budget in budgets:
            uniform_acc = per_budget_acc[("baseline_uniform", budget)]
            for method in ALL_METHODS:
                acc = per_budget_acc[(method, budget)]
                alloc = per_budget_allocs[(method, budget)]
                total_alloc_params = sum(layer.rank_unit_cost * alloc[layer.layer_name] for layer in layers)
                total_max_params = sum(layer.rank_unit_cost * layer.max_r for layer in layers)
                compression_rows.append(
                    {
                        "adapter_id": adapter.adapter_id,
                        "dataset": adapter.dataset,
                        "base_model": adapter.base_model,
                        "method": method,
                        "budget_ratio": budget,
                        "full_adapter_accuracy": full_acc,
                        "compressed_accuracy": acc,
                        "delta_vs_full_adapter": acc - full_acc,
                        "delta_vs_uniform": acc - uniform_acc,
                        "allocated_params": total_alloc_params,
                        "max_params": total_max_params,
                        "realized_budget_ratio": (total_alloc_params / total_max_params) if total_max_params > 0 else None,
                    }
                )

        # Allocation agreement rows (Gradience vs proxies)
        k_top = max(1, len(layers) // 4)
        for budget in budgets:
            for policy in POLICY_NAMES:
                alloc_policy = per_budget_allocs[(policy, budget)]
                for proxy in PROXY_NAMES:
                    alloc_proxy = per_budget_allocs[(proxy, budget)]
                    names = [layer.layer_name for layer in layers]
                    vec_p = [float(alloc_policy[n]) for n in names]
                    vec_x = [float(alloc_proxy[n]) for n in names]
                    rho = spearman(vec_p, vec_x)
                    tk = topk_overlap(alloc_policy, alloc_proxy, k=k_top)
                    attn_p, mlp_p = module_budget_splits(layers, alloc_policy)
                    attn_x, mlp_x = module_budget_splits(layers, alloc_proxy)
                    abs_dev = [abs(alloc_policy[n] - alloc_proxy[n]) for n in names]
                    allocation_rows.append(
                        {
                            "adapter_id": adapter.adapter_id,
                            "dataset": adapter.dataset,
                            "budget_ratio": budget,
                            "policy_method": policy,
                            "proxy_method": proxy,
                            "spearman_rank_correlation": rho,
                            "topk_overlap": tk,
                            "topk_k": k_top,
                            "attn_budget_share_policy": attn_p,
                            "attn_budget_share_proxy": attn_x,
                            "mlp_budget_share_policy": mlp_p,
                            "mlp_budget_share_proxy": mlp_x,
                            "attn_share_abs_diff": abs(attn_p - attn_x),
                            "mlp_share_abs_diff": abs(mlp_p - mlp_x),
                            "mean_abs_rank_deviation": statistics.mean(abs_dev) if abs_dev else None,
                            "max_abs_rank_deviation": max(abs_dev) if abs_dev else None,
                            "policy_vs_proxy_acc_delta": per_budget_acc[(policy, budget)] - per_budget_acc[(proxy, budget)],
                        }
                    )

    # Summaries
    compression_by_method: dict[str, dict[str, float | int]] = {}
    for method in ALL_METHODS:
        rows = [r for r in compression_rows if r["method"] == method]
        if not rows:
            continue
        compression_by_method[method] = {
            "n": len(rows),
            "mean_delta_vs_full": statistics.mean([float(r["delta_vs_full_adapter"]) for r in rows]),
            "mean_delta_vs_uniform": statistics.mean([float(r["delta_vs_uniform"]) for r in rows]),
            "mean_realized_budget_ratio": statistics.mean([float(r["realized_budget_ratio"]) for r in rows if r["realized_budget_ratio"] is not None]),
        }

    summary = {
        "selected_adapter_count": len(selected),
        "selected_adapters": [a.adapter_id for a in selected],
        "budget_ratios": budgets,
        "methods": ALL_METHODS,
        "allocation_rows": len(allocation_rows),
        "compression_rows": len(compression_rows),
        "compression_by_method": compression_by_method,
        "skipped": skipped,
    }

    return {
        "summary": summary,
        "selected_adapters": [a.__dict__ for a in selected],
        "allocation_rows": allocation_rows,
        "compression_rows": compression_rows,
    }


def write_outputs(results: dict[str, Any]) -> None:
    summary = results["summary"]
    allocation_rows = results["allocation_rows"]
    compression_rows = results["compression_rows"]

    _write_json(STUDY_DIR / "allocation_comparison_table.json", allocation_rows)
    _write_json(STUDY_DIR / "compression_evaluation_table.json", compression_rows)
    _write_json(STUDY_DIR / "study_summary.json", summary)

    # Task-family stratified summaries
    family_coverage: dict[str, dict[str, Any]] = {}
    for row in compression_rows:
        family = task_family_for_dataset(str(row.get("dataset") or ""))
        info = family_coverage.setdefault(
            family,
            {
                "task_family": family,
                "datasets": set(),
                "adapters": set(),
                "rows": 0,
                "realized_budgets": [],
            },
        )
        info["datasets"].add(str(row.get("dataset") or ""))
        info["adapters"].add(str(row.get("adapter_id") or ""))
        info["rows"] += 1
        rb = row.get("realized_budget_ratio")
        if rb is not None:
            info["realized_budgets"].append(_safe_float(rb, 0.0))
    family_coverage_rows = []
    for family, info in sorted(family_coverage.items(), key=lambda x: _task_family_sort_key(x[0])):
        realized = info["realized_budgets"]
        min_rb = min(realized) if realized else None
        max_rb = max(realized) if realized else None
        mean_rb = statistics.mean(realized) if realized else None
        has_effective_compression = bool(realized and min_rb is not None and min_rb < 0.99)
        family_coverage_rows.append(
            {
                "task_family": family,
                "dataset_count": len(info["datasets"]),
                "adapter_count": len(info["adapters"]),
                "row_count": int(info["rows"]),
                "datasets": sorted(info["datasets"]),
                "min_realized_budget_ratio": min_rb,
                "max_realized_budget_ratio": max_rb,
                "mean_realized_budget_ratio": mean_rb,
                "has_effective_compression": has_effective_compression,
            }
        )
    informative_families = {
        row["task_family"] for row in family_coverage_rows if bool(row.get("has_effective_compression"))
    }
    non_informative_families = {
        row["task_family"] for row in family_coverage_rows if not bool(row.get("has_effective_compression"))
    }

    comp_by_family_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    comp_by_family_budget_method: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in compression_rows:
        family = task_family_for_dataset(str(row.get("dataset") or ""))
        method = str(row.get("method") or "")
        budget = float(row.get("budget_ratio") or 0.0)
        comp_by_family_method[(family, method)].append(row)
        comp_by_family_budget_method[(family, budget, method)].append(row)

    comp_family_method_rows: list[dict[str, Any]] = []
    for (family, method), vals in sorted(comp_by_family_method.items(), key=lambda x: (_task_family_sort_key(x[0][0]), x[0][1])):
        comp_family_method_rows.append(
            {
                "task_family": family,
                "method": method,
                "n": len(vals),
                "mean_delta_vs_full_adapter": statistics.mean([_safe_float(v.get("delta_vs_full_adapter"), 0.0) for v in vals]),
                "mean_delta_vs_uniform": statistics.mean([_safe_float(v.get("delta_vs_uniform"), 0.0) for v in vals]),
                "mean_compressed_accuracy": statistics.mean([_safe_float(v.get("compressed_accuracy"), 0.0) for v in vals]),
                "mean_realized_budget_ratio": statistics.mean([_safe_float(v.get("realized_budget_ratio"), 0.0) for v in vals]),
            }
        )

    comp_family_budget_method_rows: list[dict[str, Any]] = []
    for (family, budget, method), vals in sorted(
        comp_by_family_budget_method.items(),
        key=lambda x: (_task_family_sort_key(x[0][0]), x[0][1], x[0][2]),
    ):
        comp_family_budget_method_rows.append(
            {
                "task_family": family,
                "budget_ratio": budget,
                "method": method,
                "n": len(vals),
                "mean_delta_vs_full_adapter": statistics.mean([_safe_float(v.get("delta_vs_full_adapter"), 0.0) for v in vals]),
                "mean_delta_vs_uniform": statistics.mean([_safe_float(v.get("delta_vs_uniform"), 0.0) for v in vals]),
                "mean_compressed_accuracy": statistics.mean([_safe_float(v.get("compressed_accuracy"), 0.0) for v in vals]),
            }
        )

    alloc_by_family_pair: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    alloc_by_family_budget_pair: dict[tuple[str, float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in allocation_rows:
        family = task_family_for_dataset(str(row.get("dataset") or ""))
        budget = float(row.get("budget_ratio") or 0.0)
        policy = str(row.get("policy_method") or "")
        proxy = str(row.get("proxy_method") or "")
        alloc_by_family_pair[(family, policy, proxy)].append(row)
        alloc_by_family_budget_pair[(family, budget, policy, proxy)].append(row)

    alloc_family_pair_rows: list[dict[str, Any]] = []
    for (family, policy, proxy), vals in sorted(
        alloc_by_family_pair.items(),
        key=lambda x: (_task_family_sort_key(x[0][0]), x[0][1], x[0][2]),
    ):
        alloc_family_pair_rows.append(
            {
                "task_family": family,
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(vals),
                "mean_spearman_rank_correlation": statistics.mean(
                    [_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]
                ),
                "mean_topk_overlap": statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]),
                "mean_abs_rank_deviation": statistics.mean(
                    [_safe_float(v.get("mean_abs_rank_deviation"), 0.0) for v in vals]
                ),
                "mean_policy_vs_proxy_acc_delta": statistics.mean(
                    [_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]
                ),
            }
        )

    alloc_family_budget_pair_rows: list[dict[str, Any]] = []
    for (family, budget, policy, proxy), vals in sorted(
        alloc_by_family_budget_pair.items(),
        key=lambda x: (_task_family_sort_key(x[0][0]), x[0][1], x[0][2], x[0][3]),
    ):
        alloc_family_budget_pair_rows.append(
            {
                "task_family": family,
                "budget_ratio": budget,
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(vals),
                "mean_spearman_rank_correlation": statistics.mean(
                    [_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]
                ),
                "mean_topk_overlap": statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]),
                "mean_abs_rank_deviation": statistics.mean(
                    [_safe_float(v.get("mean_abs_rank_deviation"), 0.0) for v in vals]
                ),
                "mean_policy_vs_proxy_acc_delta": statistics.mean(
                    [_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]
                ),
            }
        )

    best_method_by_family: list[dict[str, Any]] = []
    for family in sorted({row["task_family"] for row in comp_family_method_rows}, key=_task_family_sort_key):
        family_rows = [row for row in comp_family_method_rows if row["task_family"] == family]
        if not family_rows:
            continue
        best = max(family_rows, key=lambda row: _safe_float(row.get("mean_delta_vs_uniform"), 0.0))
        best_method_by_family.append(
            {
                "task_family": family,
                "best_method_by_delta_vs_uniform": best["method"],
                "best_mean_delta_vs_uniform": best["mean_delta_vs_uniform"],
                "n": best["n"],
            }
        )

    stratified_payload = {
        "analysis_scope": {
            "primary_subset": "informative_task_families_only",
            "informative_task_families": sorted(informative_families, key=_task_family_sort_key),
            "non_informative_task_families": sorted(non_informative_families, key=_task_family_sort_key),
            "informative_definition": "task family has effective compression when min_realized_budget_ratio < 0.99",
        },
        "task_family_definition": {
            "sst2": "SST2 adapters",
            "imdb": "IMDB adapters",
            "tweet_eval": "All tweet_eval/* datasets grouped together",
            "ag_news": "AG News adapters",
            "other:*": "Any other represented dataset as a distinct family key",
        },
        "family_coverage": family_coverage_rows,
        "best_method_by_family": best_method_by_family,
        "compression_by_task_family_method": comp_family_method_rows,
        "compression_by_task_family_budget_method": comp_family_budget_method_rows,
        "allocation_by_task_family_policy_proxy": alloc_family_pair_rows,
        "allocation_by_task_family_budget_policy_proxy": alloc_family_budget_pair_rows,
    }
    _write_json(STUDY_DIR / "task_family_stratified_readout.json", stratified_payload)

    strat_md_lines = ["# Task-Family Stratified Readout", ""]
    strat_md_lines.append("## Scope")
    strat_md_lines.append(f"- adapters: {summary['selected_adapter_count']}")
    strat_md_lines.append(f"- budgets: {summary['budget_ratios']}")
    strat_md_lines.append(
        f"- primary informative families: {', '.join(sorted(informative_families, key=_task_family_sort_key)) if informative_families else 'none'}"
    )
    strat_md_lines.append(
        f"- non-informative (saturated) families: {', '.join(sorted(non_informative_families, key=_task_family_sort_key)) if non_informative_families else 'none'}"
    )
    strat_md_lines.append("")
    strat_md_lines.append("## Family Coverage")
    coverage_table = [
        [
            row["task_family"],
            str(row["adapter_count"]),
            str(row["dataset_count"]),
            str(row["row_count"]),
            "yes" if row["has_effective_compression"] else "no",
            _fmt(row["mean_realized_budget_ratio"], 3),
            ", ".join(row["datasets"]),
        ]
        for row in family_coverage_rows
    ]
    strat_md_lines.append(
        _md_table(
            [
                "task_family",
                "adapters",
                "dataset_count",
                "rows",
                "effective_compression",
                "mean_realized_budget",
                "datasets",
            ],
            coverage_table,
        )
    )
    strat_md_lines.append("")
    strat_md_lines.append("## Best Method by Family (mean delta vs uniform)")
    best_table = [
        [
            row["task_family"],
            row["best_method_by_delta_vs_uniform"],
            _fmt(row["best_mean_delta_vs_uniform"], 4),
            str(row["n"]),
        ]
        for row in best_method_by_family
    ]
    strat_md_lines.append(
        _md_table(["task_family", "best_method", "mean_delta_vs_uniform", "n"], best_table)
    )
    strat_md_lines.append("")
    strat_md_lines.append("## Compression by Family and Method")
    comp_table = [
        [
            row["task_family"],
            row["method"],
            str(row["n"]),
            _fmt(row["mean_delta_vs_full_adapter"], 4),
            _fmt(row["mean_delta_vs_uniform"], 4),
            _fmt(row["mean_compressed_accuracy"], 4),
            _fmt(row["mean_realized_budget_ratio"], 3),
        ]
        for row in comp_family_method_rows
    ]
    strat_md_lines.append(
        _md_table(
            [
                "task_family",
                "method",
                "n",
                "mean_delta_vs_full",
                "mean_delta_vs_uniform",
                "mean_compressed_acc",
                "mean_realized_budget",
            ],
            comp_table,
        )
    )
    strat_md_lines.append("")
    strat_md_lines.append("## Policy-Proxy Agreement by Family")
    alloc_table = [
        [
            row["task_family"],
            row["policy_method"],
            row["proxy_method"],
            str(row["n"]),
            _fmt(row["mean_spearman_rank_correlation"], 3),
            _fmt(row["mean_topk_overlap"], 3),
            _fmt(row["mean_abs_rank_deviation"], 3),
            _fmt(row["mean_policy_vs_proxy_acc_delta"], 3),
        ]
        for row in alloc_family_pair_rows
    ]
    strat_md_lines.append(
        _md_table(
            [
                "task_family",
                "policy",
                "proxy",
                "n",
                "mean_spearman",
                "mean_topk_overlap",
                "mean_abs_rank_dev",
                "mean_policy_minus_proxy_acc",
            ],
            alloc_table,
        )
    )
    (STUDY_DIR / "task_family_stratified_readout.md").write_text("\n".join(strat_md_lines) + "\n")

    # Source-quality-gap control slice
    adapter_quality_raw: dict[tuple[str, str], dict[str, Any]] = {}
    for row in compression_rows:
        key = (str(row.get("adapter_id") or ""), str(row.get("dataset") or ""))
        if key not in adapter_quality_raw:
            adapter_quality_raw[key] = {
                "adapter_id": key[0],
                "dataset": key[1],
                "task_family": task_family_for_dataset(key[1]),
                "full_adapter_accuracy": _safe_float(row.get("full_adapter_accuracy"), 0.0),
            }

    dataset_to_acc: dict[str, list[float]] = defaultdict(list)
    for info in adapter_quality_raw.values():
        dataset_to_acc[info["dataset"]].append(_safe_float(info["full_adapter_accuracy"], 0.0))
    dataset_best = {ds: max(vals) for ds, vals in dataset_to_acc.items() if vals}
    dataset_count = {ds: len(vals) for ds, vals in dataset_to_acc.items()}

    adapter_quality_rows: list[dict[str, Any]] = []
    adapter_quality_index: dict[tuple[str, str], dict[str, Any]] = {}
    for key, info in sorted(adapter_quality_raw.items(), key=lambda x: (_task_family_sort_key(x[1]["task_family"]), x[1]["dataset"], x[1]["adapter_id"])):
        ds = str(info["dataset"])
        full_acc = _safe_float(info["full_adapter_accuracy"], 0.0)
        ds_best = _safe_float(dataset_best.get(ds), 0.0)
        ds_n = int(dataset_count.get(ds, 0))
        gap = max(0.0, ds_best - full_acc)
        band = source_quality_gap_band(gap, ds_n)
        out = {
            "adapter_id": str(info["adapter_id"]),
            "dataset": ds,
            "task_family": str(info["task_family"]),
            "full_adapter_accuracy": full_acc,
            "dataset_best_full_adapter_accuracy": ds_best,
            "source_quality_gap_vs_dataset_best": gap,
            "dataset_adapter_count": ds_n,
            "source_quality_gap_band": band,
        }
        adapter_quality_rows.append(out)
        adapter_quality_index[key] = out

    band_coverage: dict[str, dict[str, Any]] = {}
    for row in adapter_quality_rows:
        band = row["source_quality_gap_band"]
        rec = band_coverage.setdefault(
            band,
            {
                "source_quality_gap_band": band,
                "adapters": set(),
                "datasets": set(),
            },
        )
        rec["adapters"].add(row["adapter_id"])
        rec["datasets"].add(row["dataset"])
    band_coverage_rows = []
    for band, rec in sorted(band_coverage.items(), key=lambda x: _source_quality_band_sort_key(x[0])):
        band_coverage_rows.append(
            {
                "source_quality_gap_band": band,
                "adapter_count": len(rec["adapters"]),
                "dataset_count": len(rec["datasets"]),
                "datasets": sorted(rec["datasets"]),
            }
        )

    adapter_quality_rows_informative = [
        row for row in adapter_quality_rows if str(row.get("task_family") or "") in informative_families
    ]
    band_coverage_informative: dict[str, dict[str, Any]] = {}
    for row in adapter_quality_rows_informative:
        band = row["source_quality_gap_band"]
        rec = band_coverage_informative.setdefault(
            band,
            {
                "source_quality_gap_band": band,
                "adapters": set(),
                "datasets": set(),
            },
        )
        rec["adapters"].add(row["adapter_id"])
        rec["datasets"].add(row["dataset"])
    band_coverage_rows_informative = []
    for band, rec in sorted(band_coverage_informative.items(), key=lambda x: _source_quality_band_sort_key(x[0])):
        band_coverage_rows_informative.append(
            {
                "source_quality_gap_band": band,
                "adapter_count": len(rec["adapters"]),
                "dataset_count": len(rec["datasets"]),
                "datasets": sorted(rec["datasets"]),
            }
        )

    comp_by_band_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    comp_by_family_band_method: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    comp_by_band_method_informative: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    comp_by_family_band_method_informative: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in compression_rows:
        key = (str(row.get("adapter_id") or ""), str(row.get("dataset") or ""))
        q = adapter_quality_index.get(key)
        if q is None:
            continue
        band = str(q["source_quality_gap_band"])
        family = str(q["task_family"])
        method = str(row.get("method") or "")
        comp_by_band_method[(band, method)].append(row)
        comp_by_family_band_method[(family, band, method)].append(row)
        if family in informative_families:
            comp_by_band_method_informative[(band, method)].append(row)
            comp_by_family_band_method_informative[(family, band, method)].append(row)

    compression_by_gap_band_method_rows: list[dict[str, Any]] = []
    for (band, method), vals in sorted(comp_by_band_method.items(), key=lambda x: (_source_quality_band_sort_key(x[0][0]), x[0][1])):
        compression_by_gap_band_method_rows.append(
            {
                "source_quality_gap_band": band,
                "method": method,
                "n": len(vals),
                "mean_delta_vs_full_adapter": statistics.mean([_safe_float(v.get("delta_vs_full_adapter"), 0.0) for v in vals]),
                "mean_delta_vs_uniform": statistics.mean([_safe_float(v.get("delta_vs_uniform"), 0.0) for v in vals]),
                "mean_compressed_accuracy": statistics.mean([_safe_float(v.get("compressed_accuracy"), 0.0) for v in vals]),
            }
        )

    compression_by_family_gap_band_method_rows: list[dict[str, Any]] = []
    for (family, band, method), vals in sorted(
        comp_by_family_band_method.items(),
        key=lambda x: (_task_family_sort_key(x[0][0]), _source_quality_band_sort_key(x[0][1]), x[0][2]),
    ):
        compression_by_family_gap_band_method_rows.append(
            {
                "task_family": family,
                "source_quality_gap_band": band,
                "method": method,
                "n": len(vals),
                "mean_delta_vs_full_adapter": statistics.mean([_safe_float(v.get("delta_vs_full_adapter"), 0.0) for v in vals]),
                "mean_delta_vs_uniform": statistics.mean([_safe_float(v.get("delta_vs_uniform"), 0.0) for v in vals]),
                "mean_compressed_accuracy": statistics.mean([_safe_float(v.get("compressed_accuracy"), 0.0) for v in vals]),
            }
        )

    compression_by_gap_band_method_rows_informative: list[dict[str, Any]] = []
    for (band, method), vals in sorted(
        comp_by_band_method_informative.items(),
        key=lambda x: (_source_quality_band_sort_key(x[0][0]), x[0][1]),
    ):
        compression_by_gap_band_method_rows_informative.append(
            {
                "source_quality_gap_band": band,
                "method": method,
                "n": len(vals),
                "mean_delta_vs_full_adapter": statistics.mean([_safe_float(v.get("delta_vs_full_adapter"), 0.0) for v in vals]),
                "mean_delta_vs_uniform": statistics.mean([_safe_float(v.get("delta_vs_uniform"), 0.0) for v in vals]),
                "mean_compressed_accuracy": statistics.mean([_safe_float(v.get("compressed_accuracy"), 0.0) for v in vals]),
            }
        )

    compression_by_family_gap_band_method_rows_informative: list[dict[str, Any]] = []
    for (family, band, method), vals in sorted(
        comp_by_family_band_method_informative.items(),
        key=lambda x: (_task_family_sort_key(x[0][0]), _source_quality_band_sort_key(x[0][1]), x[0][2]),
    ):
        compression_by_family_gap_band_method_rows_informative.append(
            {
                "task_family": family,
                "source_quality_gap_band": band,
                "method": method,
                "n": len(vals),
                "mean_delta_vs_full_adapter": statistics.mean([_safe_float(v.get("delta_vs_full_adapter"), 0.0) for v in vals]),
                "mean_delta_vs_uniform": statistics.mean([_safe_float(v.get("delta_vs_uniform"), 0.0) for v in vals]),
                "mean_compressed_accuracy": statistics.mean([_safe_float(v.get("compressed_accuracy"), 0.0) for v in vals]),
            }
        )

    alloc_by_band_pair: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    alloc_by_family_band_pair: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    alloc_by_band_pair_informative: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    alloc_by_family_band_pair_informative: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in allocation_rows:
        key = (str(row.get("adapter_id") or ""), str(row.get("dataset") or ""))
        q = adapter_quality_index.get(key)
        if q is None:
            continue
        band = str(q["source_quality_gap_band"])
        family = str(q["task_family"])
        policy = str(row.get("policy_method") or "")
        proxy = str(row.get("proxy_method") or "")
        alloc_by_band_pair[(band, policy, proxy)].append(row)
        alloc_by_family_band_pair[(family, band, policy, proxy)].append(row)
        if family in informative_families:
            alloc_by_band_pair_informative[(band, policy, proxy)].append(row)
            alloc_by_family_band_pair_informative[(family, band, policy, proxy)].append(row)

    allocation_by_gap_band_policy_proxy_rows: list[dict[str, Any]] = []
    for (band, policy, proxy), vals in sorted(
        alloc_by_band_pair.items(),
        key=lambda x: (_source_quality_band_sort_key(x[0][0]), x[0][1], x[0][2]),
    ):
        allocation_by_gap_band_policy_proxy_rows.append(
            {
                "source_quality_gap_band": band,
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(vals),
                "mean_spearman_rank_correlation": statistics.mean(
                    [_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]
                ),
                "mean_topk_overlap": statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]),
                "mean_abs_rank_deviation": statistics.mean(
                    [_safe_float(v.get("mean_abs_rank_deviation"), 0.0) for v in vals]
                ),
                "mean_policy_vs_proxy_acc_delta": statistics.mean(
                    [_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]
                ),
            }
        )

    allocation_by_family_gap_band_policy_proxy_rows: list[dict[str, Any]] = []
    for (family, band, policy, proxy), vals in sorted(
        alloc_by_family_band_pair.items(),
        key=lambda x: (_task_family_sort_key(x[0][0]), _source_quality_band_sort_key(x[0][1]), x[0][2], x[0][3]),
    ):
        allocation_by_family_gap_band_policy_proxy_rows.append(
            {
                "task_family": family,
                "source_quality_gap_band": band,
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(vals),
                "mean_spearman_rank_correlation": statistics.mean(
                    [_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]
                ),
                "mean_topk_overlap": statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]),
                "mean_abs_rank_deviation": statistics.mean(
                    [_safe_float(v.get("mean_abs_rank_deviation"), 0.0) for v in vals]
                ),
                "mean_policy_vs_proxy_acc_delta": statistics.mean(
                    [_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]
                ),
            }
        )

    allocation_by_gap_band_policy_proxy_rows_informative: list[dict[str, Any]] = []
    for (band, policy, proxy), vals in sorted(
        alloc_by_band_pair_informative.items(),
        key=lambda x: (_source_quality_band_sort_key(x[0][0]), x[0][1], x[0][2]),
    ):
        allocation_by_gap_band_policy_proxy_rows_informative.append(
            {
                "source_quality_gap_band": band,
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(vals),
                "mean_spearman_rank_correlation": statistics.mean(
                    [_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]
                ),
                "mean_topk_overlap": statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]),
                "mean_abs_rank_deviation": statistics.mean(
                    [_safe_float(v.get("mean_abs_rank_deviation"), 0.0) for v in vals]
                ),
                "mean_policy_vs_proxy_acc_delta": statistics.mean(
                    [_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]
                ),
            }
        )

    allocation_by_family_gap_band_policy_proxy_rows_informative: list[dict[str, Any]] = []
    for (family, band, policy, proxy), vals in sorted(
        alloc_by_family_band_pair_informative.items(),
        key=lambda x: (_task_family_sort_key(x[0][0]), _source_quality_band_sort_key(x[0][1]), x[0][2], x[0][3]),
    ):
        allocation_by_family_gap_band_policy_proxy_rows_informative.append(
            {
                "task_family": family,
                "source_quality_gap_band": band,
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(vals),
                "mean_spearman_rank_correlation": statistics.mean(
                    [_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]
                ),
                "mean_topk_overlap": statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]),
                "mean_abs_rank_deviation": statistics.mean(
                    [_safe_float(v.get("mean_abs_rank_deviation"), 0.0) for v in vals]
                ),
                "mean_policy_vs_proxy_acc_delta": statistics.mean(
                    [_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]
                ),
            }
        )

    best_method_by_gap_band: list[dict[str, Any]] = []
    for band in sorted({row["source_quality_gap_band"] for row in compression_by_gap_band_method_rows}, key=_source_quality_band_sort_key):
        band_rows = [row for row in compression_by_gap_band_method_rows if row["source_quality_gap_band"] == band]
        if not band_rows:
            continue
        best = max(band_rows, key=lambda row: _safe_float(row.get("mean_delta_vs_uniform"), 0.0))
        best_method_by_gap_band.append(
            {
                "source_quality_gap_band": band,
                "best_method_by_delta_vs_uniform": best["method"],
                "best_mean_delta_vs_uniform": best["mean_delta_vs_uniform"],
                "n": best["n"],
            }
        )

    best_method_by_gap_band_informative: list[dict[str, Any]] = []
    for band in sorted(
        {row["source_quality_gap_band"] for row in compression_by_gap_band_method_rows_informative},
        key=_source_quality_band_sort_key,
    ):
        band_rows = [
            row for row in compression_by_gap_band_method_rows_informative if row["source_quality_gap_band"] == band
        ]
        if not band_rows:
            continue
        best = max(band_rows, key=lambda row: _safe_float(row.get("mean_delta_vs_uniform"), 0.0))
        best_method_by_gap_band_informative.append(
            {
                "source_quality_gap_band": band,
                "best_method_by_delta_vs_uniform": best["method"],
                "best_mean_delta_vs_uniform": best["mean_delta_vs_uniform"],
                "n": best["n"],
            }
        )

    quality_payload = {
        "analysis_scope": {
            "primary_subset": "informative_task_families_only",
            "informative_task_families": sorted(informative_families, key=_task_family_sort_key),
            "non_informative_task_families": sorted(non_informative_families, key=_task_family_sort_key),
            "informative_definition": "task family has effective compression when min_realized_budget_ratio < 0.99",
        },
        "quality_gap_definition": {
            "scope": "dataset-matched source baseline",
            "metric": "source_quality_gap_vs_dataset_best = best_full_adapter_accuracy(dataset) - full_adapter_accuracy(adapter)",
            "bands": {
                "near_top": "gap <= 0.01 and dataset has >= 2 adapters",
                "mid_gap": "0.01 < gap <= 0.05 and dataset has >= 2 adapters",
                "large_gap": "gap > 0.05 and dataset has >= 2 adapters",
                "single_source_dataset": "dataset has < 2 adapters in cohort",
            },
        },
        "primary_informative_subset": {
            "band_coverage": band_coverage_rows_informative,
            "adapter_quality_table": adapter_quality_rows_informative,
            "best_method_by_gap_band": best_method_by_gap_band_informative,
            "compression_by_gap_band_method": compression_by_gap_band_method_rows_informative,
            "compression_by_family_gap_band_method": compression_by_family_gap_band_method_rows_informative,
            "allocation_by_gap_band_policy_proxy": allocation_by_gap_band_policy_proxy_rows_informative,
            "allocation_by_family_gap_band_policy_proxy": allocation_by_family_gap_band_policy_proxy_rows_informative,
        },
        "all_families_context": {
            "band_coverage": band_coverage_rows,
            "adapter_quality_table": adapter_quality_rows,
            "best_method_by_gap_band": best_method_by_gap_band,
            "compression_by_gap_band_method": compression_by_gap_band_method_rows,
            "compression_by_family_gap_band_method": compression_by_family_gap_band_method_rows,
            "allocation_by_gap_band_policy_proxy": allocation_by_gap_band_policy_proxy_rows,
            "allocation_by_family_gap_band_policy_proxy": allocation_by_family_gap_band_policy_proxy_rows,
        },
    }
    _write_json(STUDY_DIR / "source_quality_gap_control_slice.json", quality_payload)

    quality_md_lines = ["# Source-Quality-Gap Control Slice", ""]
    quality_md_lines.append("## Analysis Scope")
    quality_md_lines.append(
        f"- primary informative families: {', '.join(sorted(informative_families, key=_task_family_sort_key)) if informative_families else 'none'}"
    )
    quality_md_lines.append(
        f"- non-informative (saturated) families: {', '.join(sorted(non_informative_families, key=_task_family_sort_key)) if non_informative_families else 'none'}"
    )
    quality_md_lines.append("")
    quality_md_lines.append("## Definition")
    quality_md_lines.append("- dataset-matched source baseline: gap is measured against best full-adapter accuracy within the same dataset.")
    quality_md_lines.append("- gap metric: `best_full_adapter_accuracy(dataset) - full_adapter_accuracy(adapter)`.")
    quality_md_lines.append("- bands: `near_top` (<=0.01), `mid_gap` (0.01-0.05), `large_gap` (>0.05), `single_source_dataset` (<2 adapters in dataset).")
    quality_md_lines.append("")
    quality_md_lines.append("## Band Coverage (Informative Subset)")
    band_table = [
        [
            row["source_quality_gap_band"],
            str(row["adapter_count"]),
            str(row["dataset_count"]),
            ", ".join(row["datasets"]),
        ]
        for row in band_coverage_rows_informative
    ]
    quality_md_lines.append(
        _md_table(["gap_band", "adapters", "dataset_count", "datasets"], band_table)
    )
    quality_md_lines.append("")
    quality_md_lines.append("## Adapter Quality Table (Informative Subset)")
    adapter_q_table = [
        [
            row["adapter_id"],
            row["dataset"],
            row["task_family"],
            _fmt(row["full_adapter_accuracy"], 4),
            _fmt(row["dataset_best_full_adapter_accuracy"], 4),
            _fmt(row["source_quality_gap_vs_dataset_best"], 4),
            str(row["dataset_adapter_count"]),
            row["source_quality_gap_band"],
        ]
        for row in adapter_quality_rows_informative
    ]
    quality_md_lines.append(
        _md_table(
            [
                "adapter_id",
                "dataset",
                "task_family",
                "full_acc",
                "dataset_best_acc",
                "gap_vs_dataset_best",
                "dataset_adapter_count",
                "gap_band",
            ],
            adapter_q_table,
        )
    )
    quality_md_lines.append("")
    quality_md_lines.append("## Compression by Gap Band and Method (Informative Subset)")
    comp_band_table = [
        [
            row["source_quality_gap_band"],
            row["method"],
            str(row["n"]),
            _fmt(row["mean_delta_vs_full_adapter"], 4),
            _fmt(row["mean_delta_vs_uniform"], 4),
            _fmt(row["mean_compressed_accuracy"], 4),
        ]
        for row in compression_by_gap_band_method_rows_informative
    ]
    quality_md_lines.append(
        _md_table(
            [
                "gap_band",
                "method",
                "n",
                "mean_delta_vs_full",
                "mean_delta_vs_uniform",
                "mean_compressed_acc",
            ],
            comp_band_table,
        )
    )
    quality_md_lines.append("")
    quality_md_lines.append("## Policy-Proxy Agreement by Gap Band (Informative Subset)")
    alloc_band_table = [
        [
            row["source_quality_gap_band"],
            row["policy_method"],
            row["proxy_method"],
            str(row["n"]),
            _fmt(row["mean_spearman_rank_correlation"], 3),
            _fmt(row["mean_topk_overlap"], 3),
            _fmt(row["mean_abs_rank_deviation"], 3),
            _fmt(row["mean_policy_vs_proxy_acc_delta"], 3),
        ]
        for row in allocation_by_gap_band_policy_proxy_rows_informative
    ]
    quality_md_lines.append(
        _md_table(
            [
                "gap_band",
                "policy",
                "proxy",
                "n",
                "mean_spearman",
                "mean_topk_overlap",
                "mean_abs_rank_dev",
                "mean_policy_minus_proxy_acc",
            ],
            alloc_band_table,
        )
    )
    quality_md_lines.append("")
    quality_md_lines.append("## Non-Informative Context (All Families)")
    quality_md_lines.append(
        "- Families with realized budget near 1.0 are reported for completeness but are excluded from primary interpretation."
    )
    context_band_table = [
        [
            row["source_quality_gap_band"],
            str(row["adapter_count"]),
            str(row["dataset_count"]),
            ", ".join(row["datasets"]),
        ]
        for row in band_coverage_rows
    ]
    quality_md_lines.append(
        _md_table(["gap_band", "adapters", "dataset_count", "datasets"], context_band_table)
    )
    (STUDY_DIR / "source_quality_gap_control_slice.md").write_text("\n".join(quality_md_lines) + "\n")

    # Compressible-family-only summary (primary interpretation subset)
    compression_rows_informative = [
        row for row in compression_rows if task_family_for_dataset(str(row.get("dataset") or "")) in informative_families
    ]
    allocation_rows_informative = [
        row for row in allocation_rows if task_family_for_dataset(str(row.get("dataset") or "")) in informative_families
    ]

    method_summary_informative: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        vals = [row for row in compression_rows_informative if str(row.get("method") or "") == method]
        if not vals:
            continue
        dvu = [_safe_float(v.get("delta_vs_uniform"), 0.0) for v in vals]
        dvf = [_safe_float(v.get("delta_vs_full_adapter"), 0.0) for v in vals]
        method_summary_informative.append(
            {
                "method": method,
                "n": len(vals),
                "mean_delta_vs_uniform": statistics.mean(dvu),
                "var_delta_vs_uniform": statistics.pvariance(dvu) if len(dvu) > 1 else 0.0,
                "mean_delta_vs_full_adapter": statistics.mean(dvf),
                "var_delta_vs_full_adapter": statistics.pvariance(dvf) if len(dvf) > 1 else 0.0,
                "mean_realized_budget_ratio": statistics.mean(
                    [_safe_float(v.get("realized_budget_ratio"), 0.0) for v in vals]
                ),
            }
        )
    method_summary_informative.sort(key=lambda row: _safe_float(row.get("mean_delta_vs_uniform"), 0.0), reverse=True)

    policy_proxy_summary_informative: list[dict[str, Any]] = []
    for policy in POLICY_NAMES:
        for proxy in PROXY_NAMES:
            vals = [
                row
                for row in allocation_rows_informative
                if str(row.get("policy_method") or "") == policy and str(row.get("proxy_method") or "") == proxy
            ]
            if not vals:
                continue
            policy_proxy_summary_informative.append(
                {
                    "policy_method": policy,
                    "proxy_method": proxy,
                    "n": len(vals),
                    "mean_spearman_rank_correlation": statistics.mean(
                        [_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]
                    ),
                    "mean_topk_overlap": statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]),
                    "mean_policy_vs_proxy_acc_delta": statistics.mean(
                        [_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]
                    ),
                }
            )

    proxy_agreement_split_informative: list[dict[str, Any]] = []
    for proxy in PROXY_NAMES:
        vals = [row for row in allocation_rows_informative if str(row.get("proxy_method") or "") == proxy]
        if not vals:
            continue
        proxy_agreement_split_informative.append(
            {
                "proxy_method": proxy,
                "n": len(vals),
                "mean_spearman_rank_correlation": statistics.mean(
                    [_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]
                ),
                "mean_topk_overlap": statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]),
                "mean_policy_vs_proxy_acc_delta": statistics.mean(
                    [_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]
                ),
            }
        )
    proxy_agreement_split_informative.sort(key=lambda row: row["proxy_method"])

    compressible_payload = {
        "analysis_scope": {
            "subset": "informative_task_families_only",
            "informative_task_families": sorted(informative_families, key=_task_family_sort_key),
            "non_informative_task_families": sorted(non_informative_families, key=_task_family_sort_key),
        },
        "method_performance_summary": method_summary_informative,
        "policy_proxy_agreement_summary": policy_proxy_summary_informative,
        "proxy_agreement_split_summary": proxy_agreement_split_informative,
    }
    _write_json(STUDY_DIR / "compressible_family_summary.json", compressible_payload)

    compress_md_lines = ["# Compressible-Family-Only Summary", ""]
    compress_md_lines.append("## Scope")
    compress_md_lines.append(
        f"- informative families: {', '.join(sorted(informative_families, key=_task_family_sort_key)) if informative_families else 'none'}"
    )
    compress_md_lines.append(
        f"- non-informative families: {', '.join(sorted(non_informative_families, key=_task_family_sort_key)) if non_informative_families else 'none'}"
    )
    compress_md_lines.append(f"- compression rows (informative): {len(compression_rows_informative)}")
    compress_md_lines.append(f"- allocation rows (informative): {len(allocation_rows_informative)}")
    compress_md_lines.append("")
    compress_md_lines.append("## Method Performance")
    method_table = [
        [
            row["method"],
            str(row["n"]),
            _fmt(row["mean_delta_vs_uniform"], 4),
            _fmt(row["var_delta_vs_uniform"], 6),
            _fmt(row["mean_delta_vs_full_adapter"], 4),
            _fmt(row["var_delta_vs_full_adapter"], 6),
            _fmt(row["mean_realized_budget_ratio"], 3),
        ]
        for row in method_summary_informative
    ]
    compress_md_lines.append(
        _md_table(
            [
                "method",
                "n",
                "mean_delta_vs_uniform",
                "var_delta_vs_uniform",
                "mean_delta_vs_full",
                "var_delta_vs_full",
                "mean_realized_budget",
            ],
            method_table,
        )
    )
    compress_md_lines.append("")
    compress_md_lines.append("## Policy-Proxy Agreement")
    agreement_table = [
        [
            row["policy_method"],
            row["proxy_method"],
            str(row["n"]),
            _fmt(row["mean_spearman_rank_correlation"], 3),
            _fmt(row["mean_topk_overlap"], 3),
            _fmt(row["mean_policy_vs_proxy_acc_delta"], 3),
        ]
        for row in policy_proxy_summary_informative
    ]
    compress_md_lines.append(
        _md_table(
            [
                "policy",
                "proxy",
                "n",
                "mean_spearman",
                "mean_topk_overlap",
                "mean_policy_minus_proxy_acc",
            ],
            agreement_table,
        )
    )
    compress_md_lines.append("")
    compress_md_lines.append("## Proxy Agreement Split")
    split_table = [
        [
            row["proxy_method"],
            str(row["n"]),
            _fmt(row["mean_spearman_rank_correlation"], 3),
            _fmt(row["mean_topk_overlap"], 3),
            _fmt(row["mean_policy_vs_proxy_acc_delta"], 3),
        ]
        for row in proxy_agreement_split_informative
    ]
    compress_md_lines.append(
        _md_table(
            [
                "proxy",
                "n",
                "mean_spearman",
                "mean_topk_overlap",
                "mean_policy_minus_proxy_acc",
            ],
            split_table,
        )
    )
    (STUDY_DIR / "compressible_family_summary.md").write_text("\n".join(compress_md_lines) + "\n")

    # Source-quality-gap analysis focused on informative subset
    by_band_method_inf = {
        (str(row["source_quality_gap_band"]), str(row["method"])): row
        for row in compression_by_gap_band_method_rows_informative
    }

    def _band_method_dvu(band: str, method: str) -> float | None:
        row = by_band_method_inf.get((band, method))
        if row is None:
            return None
        return _safe_float(row.get("mean_delta_vs_uniform"), default=0.0)

    pg_near = _band_method_dvu("near_top", "proxy_gradient")
    pg_mid = _band_method_dvu("mid_gap", "proxy_gradient")
    pg_large = _band_method_dvu("large_gap", "proxy_gradient")
    oht_near = _band_method_dvu("near_top", "oht")
    oht_mid = _band_method_dvu("mid_gap", "oht")
    oht_large = _band_method_dvu("large_gap", "oht")

    proxy_gradient_concentration_check = {
        "proxy_gradient_mean_delta_vs_uniform": {
            "near_top": pg_near,
            "mid_gap": pg_mid,
            "large_gap": pg_large,
        },
        "proxy_gradient_near_minus_large": (pg_near - pg_large) if pg_near is not None and pg_large is not None else None,
        "proxy_gradient_near_minus_mid": (pg_near - pg_mid) if pg_near is not None and pg_mid is not None else None,
        "proxy_gradient_minus_oht_by_band": {
            "near_top": (pg_near - oht_near) if pg_near is not None and oht_near is not None else None,
            "mid_gap": (pg_mid - oht_mid) if pg_mid is not None and oht_mid is not None else None,
            "large_gap": (pg_large - oht_large) if pg_large is not None and oht_large is not None else None,
        },
    }

    spectral_vs_ablation_alignment_by_band: list[dict[str, Any]] = []
    spectral_vs_gradient_alignment_by_band: list[dict[str, Any]] = []
    for band in sorted(
        {str(row["source_quality_gap_band"]) for row in allocation_by_gap_band_policy_proxy_rows_informative},
        key=_source_quality_band_sort_key,
    ):
        vals_ab = [
            row
            for row in allocation_by_gap_band_policy_proxy_rows_informative
            if str(row.get("source_quality_gap_band") or "") == band and str(row.get("proxy_method") or "") == "proxy_ablation"
        ]
        if vals_ab:
            spectral_vs_ablation_alignment_by_band.append(
                {
                    "source_quality_gap_band": band,
                    "n": len(vals_ab),
                    "mean_spearman_rank_correlation": statistics.mean(
                        [_safe_float(v.get("mean_spearman_rank_correlation"), 0.0) for v in vals_ab]
                    ),
                    "mean_topk_overlap": statistics.mean(
                        [_safe_float(v.get("mean_topk_overlap"), 0.0) for v in vals_ab]
                    ),
                    "mean_policy_vs_proxy_acc_delta": statistics.mean(
                        [_safe_float(v.get("mean_policy_vs_proxy_acc_delta"), 0.0) for v in vals_ab]
                    ),
                }
            )
        vals_gr = [
            row
            for row in allocation_by_gap_band_policy_proxy_rows_informative
            if str(row.get("source_quality_gap_band") or "") == band and str(row.get("proxy_method") or "") == "proxy_gradient"
        ]
        if vals_gr:
            spectral_vs_gradient_alignment_by_band.append(
                {
                    "source_quality_gap_band": band,
                    "n": len(vals_gr),
                    "mean_spearman_rank_correlation": statistics.mean(
                        [_safe_float(v.get("mean_spearman_rank_correlation"), 0.0) for v in vals_gr]
                    ),
                    "mean_topk_overlap": statistics.mean(
                        [_safe_float(v.get("mean_topk_overlap"), 0.0) for v in vals_gr]
                    ),
                    "mean_policy_vs_proxy_acc_delta": statistics.mean(
                        [_safe_float(v.get("mean_policy_vs_proxy_acc_delta"), 0.0) for v in vals_gr]
                    ),
                }
            )

    informative_quality_analysis_payload = {
        "analysis_scope": {
            "subset": "informative_task_families_only",
            "informative_task_families": sorted(informative_families, key=_task_family_sort_key),
            "bands_present": [row["source_quality_gap_band"] for row in band_coverage_rows_informative],
        },
        "method_behavior_by_source_quality_band": compression_by_gap_band_method_rows_informative,
        "best_method_by_source_quality_band": best_method_by_gap_band_informative,
        "proxy_gradient_concentration_check": proxy_gradient_concentration_check,
        "spectral_vs_ablation_alignment_by_band": spectral_vs_ablation_alignment_by_band,
        "spectral_vs_gradient_alignment_by_band": spectral_vs_gradient_alignment_by_band,
        "caution": "This is a small-sample control slice; use as directional evidence only.",
    }
    _write_json(STUDY_DIR / "source_quality_gap_informative_analysis.json", informative_quality_analysis_payload)

    inf_gap_md = ["# Source-Quality Gap Informative-Subset Analysis", ""]
    inf_gap_md.append("## Scope")
    inf_gap_md.append(
        f"- informative families: {', '.join(sorted(informative_families, key=_task_family_sort_key)) if informative_families else 'none'}"
    )
    inf_gap_md.append(
        f"- bands present: {', '.join([row['source_quality_gap_band'] for row in band_coverage_rows_informative]) if band_coverage_rows_informative else 'none'}"
    )
    inf_gap_md.append("")
    inf_gap_md.append("## Method Behavior by Source-Quality Band")
    band_method_table = [
        [
            row["source_quality_gap_band"],
            row["method"],
            str(row["n"]),
            _fmt(row["mean_delta_vs_uniform"], 4),
            _fmt(row["mean_delta_vs_full_adapter"], 4),
        ]
        for row in compression_by_gap_band_method_rows_informative
    ]
    inf_gap_md.append(
        _md_table(
            ["gap_band", "method", "n", "mean_delta_vs_uniform", "mean_delta_vs_full"],
            band_method_table,
        )
    )
    inf_gap_md.append("")
    inf_gap_md.append("## Proxy-Gradient Concentration Check")
    pg_table = [
        ["near_top", _fmt(pg_near, 4), _fmt(oht_near, 4), _fmt((pg_near - oht_near) if pg_near is not None and oht_near is not None else None, 4)],
        ["mid_gap", _fmt(pg_mid, 4), _fmt(oht_mid, 4), _fmt((pg_mid - oht_mid) if pg_mid is not None and oht_mid is not None else None, 4)],
        ["large_gap", _fmt(pg_large, 4), _fmt(oht_large, 4), _fmt((pg_large - oht_large) if pg_large is not None and oht_large is not None else None, 4)],
    ]
    inf_gap_md.append(
        _md_table(
            ["gap_band", "proxy_gradient_dvu", "oht_dvu", "proxy_gradient_minus_oht"],
            pg_table,
        )
    )
    inf_gap_md.append(f"- proxy_gradient near_top minus large_gap: {_fmt(proxy_gradient_concentration_check['proxy_gradient_near_minus_large'], 4)}")
    inf_gap_md.append("")
    inf_gap_md.append("## Spectral-vs-Ablation Alignment by Source-Quality Band")
    ab_table = [
        [
            row["source_quality_gap_band"],
            str(row["n"]),
            _fmt(row["mean_spearman_rank_correlation"], 3),
            _fmt(row["mean_topk_overlap"], 3),
            _fmt(row["mean_policy_vs_proxy_acc_delta"], 3),
        ]
        for row in spectral_vs_ablation_alignment_by_band
    ]
    inf_gap_md.append(
        _md_table(
            ["gap_band", "n", "mean_spearman", "mean_topk_overlap", "mean_policy_minus_ablation_acc"],
            ab_table,
        )
    )
    inf_gap_md.append("")
    inf_gap_md.append("## Spectral-vs-Gradient Alignment by Source-Quality Band")
    gr_table = [
        [
            row["source_quality_gap_band"],
            str(row["n"]),
            _fmt(row["mean_spearman_rank_correlation"], 3),
            _fmt(row["mean_topk_overlap"], 3),
            _fmt(row["mean_policy_vs_proxy_acc_delta"], 3),
        ]
        for row in spectral_vs_gradient_alignment_by_band
    ]
    inf_gap_md.append(
        _md_table(
            ["gap_band", "n", "mean_spearman", "mean_topk_overlap", "mean_policy_minus_gradient_acc"],
            gr_table,
        )
    )
    inf_gap_md.append("")
    inf_gap_md.append("## Caution")
    inf_gap_md.append("- Small sample sizes in `mid_gap` and `large_gap` should be treated as directional only.")
    (STUDY_DIR / "source_quality_gap_informative_analysis.md").write_text("\n".join(inf_gap_md) + "\n")

    # Allocation comparison markdown
    alloc_md_lines = ["# Allocation Comparison Table", ""]
    alloc_md_lines.append(f"- Rows: {len(allocation_rows)}")
    alloc_md_lines.append("")
    top_alloc = sorted(
        allocation_rows,
        key=lambda r: (abs(_safe_float(r.get("spearman_rank_correlation"), 0.0)), abs(_safe_float(r.get("policy_vs_proxy_acc_delta"), 0.0))),
        reverse=True,
    )[:40]
    alloc_rows = []
    for row in top_alloc:
        alloc_rows.append(
            [
                row["adapter_id"],
                row["dataset"],
                _fmt(row["budget_ratio"], 2),
                row["policy_method"],
                row["proxy_method"],
                _fmt(row["spearman_rank_correlation"], 3),
                _fmt(row["topk_overlap"], 3),
                _fmt(row["mean_abs_rank_deviation"], 3),
                _fmt(row["attn_share_abs_diff"], 3),
                _fmt(row["policy_vs_proxy_acc_delta"], 3),
            ]
        )
    alloc_md_lines.append(
        _md_table(
            [
                "adapter_id",
                "dataset",
                "budget",
                "policy",
                "proxy",
                "spearman",
                "topk_overlap",
                "mean_abs_rank_dev",
                "attn_share_abs_diff",
                "policy_minus_proxy_acc",
            ],
            alloc_rows,
        )
    )
    (STUDY_DIR / "allocation_comparison_table.md").write_text("\n".join(alloc_md_lines) + "\n")

    # Compression evaluation markdown
    comp_md_lines = ["# Compression Evaluation Table", ""]
    comp_md_lines.append(f"- Rows: {len(compression_rows)}")
    comp_md_lines.append("")
    comp_rows = sorted(
        compression_rows,
        key=lambda r: (r["dataset"], r["budget_ratio"], r["method"]),
    )
    comp_table_rows = []
    for row in comp_rows:
        comp_table_rows.append(
            [
                row["adapter_id"],
                row["dataset"],
                _fmt(row["budget_ratio"], 2),
                row["method"],
                _fmt(row["compressed_accuracy"], 4),
                _fmt(row["delta_vs_full_adapter"], 4),
                _fmt(row["delta_vs_uniform"], 4),
                _fmt(row["realized_budget_ratio"], 3),
            ]
        )
    comp_md_lines.append(
        _md_table(
            [
                "adapter_id",
                "dataset",
                "budget",
                "method",
                "compressed_acc",
                "delta_vs_full",
                "delta_vs_uniform",
                "realized_budget",
            ],
            comp_table_rows,
        )
    )
    (STUDY_DIR / "compression_evaluation_table.md").write_text("\n".join(comp_md_lines) + "\n")

    # Disagreement memo
    disagree_lines = ["# Rank Allocation Disagreement Memo", ""]
    disagree_lines.append("## Scope")
    disagree_lines.append(f"- adapters: {summary['selected_adapter_count']}")
    disagree_lines.append(f"- budgets: {summary['budget_ratios']}")
    disagree_lines.append(
        f"- primary informative families: {', '.join(sorted(informative_families, key=_task_family_sort_key)) if informative_families else 'none'}"
    )
    disagree_lines.append(
        f"- non-informative families: {', '.join(sorted(non_informative_families, key=_task_family_sort_key)) if non_informative_families else 'none'}"
    )
    disagree_lines.append("")

    allocation_rows_informative = [
        row for row in allocation_rows if task_family_for_dataset(str(row.get("dataset") or "")) in informative_families
    ]
    compression_rows_informative = [
        row for row in compression_rows if task_family_for_dataset(str(row.get("dataset") or "")) in informative_families
    ]

    if allocation_rows_informative:
        by_pair_inf: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in allocation_rows_informative:
            by_pair_inf[(row["policy_method"], row["proxy_method"])].append(row)
        disagree_lines.append("## Policy vs Proxy Summary (Informative Families Only)")
        rows_inf = []
        for (policy, proxy), vals in sorted(by_pair_inf.items()):
            rows_inf.append(
                [
                    policy,
                    proxy,
                    str(len(vals)),
                    _fmt(statistics.mean([_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]), 3),
                    _fmt(statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]), 3),
                    _fmt(statistics.mean([_safe_float(v.get("mean_abs_rank_deviation"), 0.0) for v in vals]), 3),
                    _fmt(statistics.mean([_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]), 3),
                ]
            )
        disagree_lines.append(
            _md_table(
                [
                    "policy",
                    "proxy",
                    "n",
                    "mean_spearman",
                    "mean_topk_overlap",
                    "mean_abs_rank_dev",
                    "mean_policy_minus_proxy_acc",
                ],
                rows_inf,
            )
        )
        disagree_lines.append("")

    if compression_rows_informative:
        by_method_inf: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in compression_rows_informative:
            by_method_inf[str(row["method"])].append(row)
        disagree_lines.append("## Compression Method Summary (Informative Families Only)")
        cm_rows_inf = []
        for method, vals in sorted(by_method_inf.items()):
            cm_rows_inf.append(
                [
                    method,
                    str(len(vals)),
                    _fmt(statistics.mean([_safe_float(v.get("delta_vs_full_adapter"), 0.0) for v in vals]), 4),
                    _fmt(statistics.mean([_safe_float(v.get("delta_vs_uniform"), 0.0) for v in vals]), 4),
                    _fmt(statistics.mean([_safe_float(v.get("realized_budget_ratio"), 0.0) for v in vals]), 3),
                ]
            )
        disagree_lines.append(
            _md_table(
                [
                    "method",
                    "n",
                    "mean_delta_vs_full",
                    "mean_delta_vs_uniform",
                    "mean_realized_budget",
                ],
                cm_rows_inf,
            )
        )
        disagree_lines.append("")

    if allocation_rows:
        by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in allocation_rows:
            by_pair[(row["policy_method"], row["proxy_method"])].append(row)
        disagree_lines.append("## Policy vs Proxy Summary")
        rows = []
        for (policy, proxy), vals in sorted(by_pair.items()):
            rows.append(
                [
                    policy,
                    proxy,
                    str(len(vals)),
                    _fmt(statistics.mean([_safe_float(v.get("spearman_rank_correlation"), 0.0) for v in vals]), 3),
                    _fmt(statistics.mean([_safe_float(v.get("topk_overlap"), 0.0) for v in vals]), 3),
                    _fmt(statistics.mean([_safe_float(v.get("mean_abs_rank_deviation"), 0.0) for v in vals]), 3),
                    _fmt(statistics.mean([_safe_float(v.get("policy_vs_proxy_acc_delta"), 0.0) for v in vals]), 3),
                ]
            )
        disagree_lines.append(
            _md_table(
                [
                    "policy",
                    "proxy",
                    "n",
                    "mean_spearman",
                    "mean_topk_overlap",
                    "mean_abs_rank_dev",
                    "mean_policy_minus_proxy_acc",
                ],
                rows,
            )
        )
        disagree_lines.append("")

    # Compression method summary
    disagree_lines.append("## Compression Method Summary")
    cm_rows = []
    for method, info in sorted(summary["compression_by_method"].items()):
        cm_rows.append(
            [
                method,
                str(info["n"]),
                _fmt(info["mean_delta_vs_full"], 4),
                _fmt(info["mean_delta_vs_uniform"], 4),
                _fmt(info["mean_realized_budget_ratio"], 3),
            ]
        )
    disagree_lines.append(
        _md_table(
            [
                "method",
                "n",
                "mean_delta_vs_full",
                "mean_delta_vs_uniform",
                "mean_realized_budget",
            ],
            cm_rows,
        )
    )
    disagree_lines.append("")
    disagree_lines.append("## Interpretation")
    disagree_lines.append(
        "- This pass is a bounded CPU proxy-comparison study. It tests whether cheap spectral suggestions recover useful layerwise structure and compression behavior under fixed budgets."
    )
    disagree_lines.append(
        "- Stronger control-corrected claims should wait for a larger matched cohort with broader top-tail budget representation."
    )
    (STUDY_DIR / "disagreement_memo.md").write_text("\n".join(disagree_lines) + "\n")

    # Strategy summary note
    strategy_lines = ["# CPU Rank-Proxy Validation Summary", ""]
    strategy_lines.append("## Outcome")
    strategy_lines.append(
        "- CPU-only proxy comparison run completed for spectral rank policies, offline proxy targets, and matched-budget baselines."
    )
    strategy_lines.append(f"- adapters evaluated: {summary['selected_adapter_count']}")
    strategy_lines.append(f"- budgets: {summary['budget_ratios']}")
    strategy_lines.append("")
    strategy_lines.append("## What This Supports")
    strategy_lines.append(
        "- Whether Gradience's cheap spectral suggestions recover non-trivial layerwise allocation structure compared to gradient/ablation proxies."
    )
    strategy_lines.append(
        "- Whether matched-budget compression behavior is competitive with simple baselines in this bounded encoder regime."
    )
    strategy_lines.append("")
    strategy_lines.append("## Task-Family Readout")
    strategy_lines.append(
        f"- Primary interpretation subset: {', '.join(sorted(informative_families, key=_task_family_sort_key)) if informative_families else 'none'}."
    )
    strategy_lines.append(
        f"- Saturated/non-informative context: {', '.join(sorted(non_informative_families, key=_task_family_sort_key)) if non_informative_families else 'none'}."
    )
    for row in family_coverage_rows:
        strategy_lines.append(
            f"- `{row['task_family']}`: adapters={row['adapter_count']}, effective_compression={'yes' if row['has_effective_compression'] else 'no'}, mean_realized_budget={_fmt(row['mean_realized_budget_ratio'], 3)}."
        )
    strategy_lines.append(
        "- Best-by-family method should only be interpreted where effective compression is active."
    )
    for row in [r for r in best_method_by_family if r["task_family"] in informative_families]:
        strategy_lines.append(
            f"- `{row['task_family']}` best-by-mean-delta-vs-uniform: `{row['best_method_by_delta_vs_uniform']}` ({_fmt(row['best_mean_delta_vs_uniform'], 4)})."
        )
    strategy_lines.append("")
    strategy_lines.append("## Source-Quality Gap Control (Informative Subset)")
    strategy_lines.append(
        "- Primary slice uses dataset-matched source-quality gap bands only inside informative families."
    )
    for row in best_method_by_gap_band_informative:
        strategy_lines.append(
            f"- `{row['source_quality_gap_band']}` best-by-mean-delta-vs-uniform: `{row['best_method_by_delta_vs_uniform']}` ({_fmt(row['best_mean_delta_vs_uniform'], 4)})."
        )
    strategy_lines.append("")
    strategy_lines.append("## Guardrail")
    strategy_lines.append(
        "- Treat this as bounded advisor-oriented evidence, not adaptive-training SOTA equivalence."
    )
    strategy_lines.append(
        "- Families with realized budget near 1.0 are weak evidence for allocation-method differences."
    )
    STRATEGY_DOC.write_text("\n".join(strategy_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-adapters", type=int, default=4)
    parser.add_argument("--budgets", type=str, default="0.50,0.65")
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--grad-samples", type=int, default=64)
    parser.add_argument("--ablation-samples", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-online", action="store_true")
    args = parser.parse_args()

    results = run_study(args)
    write_outputs(results)
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
