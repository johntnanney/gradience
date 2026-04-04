#!/usr/bin/env python3
"""Build canonical v2 rank-proxy validation artifacts from existing v1 outputs.

This pass avoids heavy re-runs and repackages bounded CPU results into the
`rank_proxy_validation_v2` structure.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_DIR = REPO_ROOT / "field_trials" / "rank_proxy_validation"
V2_DIR = Path(__file__).resolve().parent
DOC_SUMMARY_PATH = REPO_ROOT / "docs" / "strategy" / "rank_proxy_bounded_validation_summary.md"

SPECTRAL_METHODS = {"oht", "knee", "energy_90", "erank", "stable_rank_ceil"}
PROXY_METHODS = {"proxy_gradient", "proxy_ablation"}
BASELINE_METHODS = {"baseline_uniform", "baseline_random"}

METHOD_ALIAS = {
    "baseline_uniform": "uniform",
    "baseline_random": "random_matched_budget",
    "proxy_ablation": "proxy_ablation_attenuate",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _fmt(v: float | None, d: int = 4) -> str:
    if v is None:
        return "n/a"
    x = float(v)
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{d}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _norm_method(method: str) -> str:
    return METHOD_ALIAS.get(method, method)


def _method_family(method: str) -> str:
    if method in SPECTRAL_METHODS:
        return "spectral"
    if method in PROXY_METHODS:
        return "proxy"
    if method in BASELINE_METHODS:
        return "baseline"
    return "other"


def _task_family(dataset: str) -> str:
    d = (dataset or "").strip()
    if d.startswith("tweet_eval/"):
        return "tweet_eval"
    if d in {"sst2", "imdb", "ag_news"}:
        return d
    return f"other:{d}" if d else "other:unknown"


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _pvar(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def _inclusion_status(task_family: str, informative_families: set[str]) -> str:
    return "primary_informative" if task_family in informative_families else "secondary_context"


def _pick_policy_share(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    for preferred in ("proxy_gradient", "proxy_ablation"):
        for row in rows:
            if str(row.get("proxy_method")) == preferred:
                return row
    return rows[0]


def _pick_proxy_share(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    for preferred in ("oht", "knee", "energy_90", "erank", "stable_rank_ceil"):
        for row in rows:
            if str(row.get("policy_method")) == preferred:
                return row
    return rows[0]


def build() -> None:
    V2_DIR.mkdir(parents=True, exist_ok=True)

    study_summary = _load_json(V1_DIR / "study_summary.json")
    compression_rows_raw: list[dict[str, Any]] = _load_json(V1_DIR / "compression_evaluation_table.json")
    allocation_rows_raw: list[dict[str, Any]] = _load_json(V1_DIR / "allocation_comparison_table.json")
    stratified = _load_json(V1_DIR / "task_family_stratified_readout.json")
    quality = _load_json(V1_DIR / "source_quality_gap_control_slice.json")
    compressible_summary = _load_json(V1_DIR / "compressible_family_summary.json")
    ablation_gap_path = V1_DIR / "ablation_gradient_gap_investigation.json"
    ablation_gap = _load_json(ablation_gap_path) if ablation_gap_path.exists() else None

    informative_families = set(stratified["analysis_scope"]["informative_task_families"])
    non_informative_families = set(stratified["analysis_scope"]["non_informative_task_families"])
    family_coverage = {row["task_family"]: row for row in stratified["family_coverage"]}

    quality_rows_all: list[dict[str, Any]] = quality["all_families_context"]["adapter_quality_table"]
    quality_map: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["adapter_id"]), str(row["dataset"])): row for row in quality_rows_all
    }

    # Build baseline maps for delta augmentations.
    random_acc: dict[tuple[str, str, float], float] = {}
    gradient_acc: dict[tuple[str, str, float], float] = {}
    ablation_acc: dict[tuple[str, str, float], float] = {}
    for row in compression_rows_raw:
        key = (str(row["adapter_id"]), str(row["dataset"]), float(row["budget_ratio"]))
        method = str(row["method"])
        acc = _safe_float(row["compressed_accuracy"])
        if method == "baseline_random":
            random_acc[key] = acc
        elif method == "proxy_gradient":
            gradient_acc[key] = acc
        elif method == "proxy_ablation":
            ablation_acc[key] = acc

    # Build share maps from allocation comparison rows.
    policy_share_candidates: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    proxy_share_candidates: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in allocation_rows_raw:
        adapter_id = str(row["adapter_id"])
        dataset = str(row["dataset"])
        budget = float(row["budget_ratio"])
        policy = str(row["policy_method"])
        proxy = str(row["proxy_method"])
        policy_share_candidates[(adapter_id, dataset, budget, policy)].append(row)
        proxy_share_candidates[(adapter_id, dataset, budget, proxy)].append(row)

    policy_share_map: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for key, rows in policy_share_candidates.items():
        picked = _pick_policy_share(rows)
        if picked is None:
            continue
        policy_share_map[key] = {
            "attn_budget_share": picked.get("attn_budget_share_policy"),
            "mlp_budget_share": picked.get("mlp_budget_share_policy"),
            "topk_k_reference": picked.get("topk_k"),
        }

    proxy_share_map: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for key, rows in proxy_share_candidates.items():
        picked = _pick_proxy_share(rows)
        if picked is None:
            continue
        proxy_share_map[key] = {
            "attn_budget_share": picked.get("attn_budget_share_proxy"),
            "mlp_budget_share": picked.get("mlp_budget_share_proxy"),
            "topk_k_reference": picked.get("topk_k"),
        }

    # Stage A: cohort definition
    adapter_records_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in compression_rows_raw:
        adapter_id = str(row["adapter_id"])
        dataset = str(row["dataset"])
        base_model = str(row["base_model"])
        tf = _task_family(dataset)
        inclusion = _inclusion_status(tf, informative_families)
        coverage = family_coverage.get(tf, {})
        compressible = bool(coverage.get("has_effective_compression", False))
        q = quality_map.get((adapter_id, dataset), {})
        adapter_records_map[(adapter_id, dataset)] = {
            "adapter_id": adapter_id,
            "dataset": dataset,
            "task_family": tf,
            "base_model": base_model,
            "compressibility_label": "compressible" if compressible else "saturated",
            "inclusion_status": inclusion,
            "full_adapter_accuracy": q.get("full_adapter_accuracy"),
            "dataset_best_full_adapter_accuracy": q.get("dataset_best_full_adapter_accuracy"),
            "source_quality_gap_vs_dataset_best": q.get("source_quality_gap_vs_dataset_best"),
            "source_quality_gap_band": q.get("source_quality_gap_band"),
            "dataset_adapter_count": q.get("dataset_adapter_count"),
            "rationale": (
                "effective compression active in this task family"
                if compressible
                else "realized budgets saturate near 1.0 in this task family"
            ),
        }
    cohort_rows = sorted(
        adapter_records_map.values(),
        key=lambda r: (
            0 if r["inclusion_status"] == "primary_informative" else 1,
            r["task_family"],
            r["dataset"],
            r["adapter_id"],
        ),
    )
    cohort_payload = {
        "study": "adaptive_rank_comparison_external_validation_target_v2",
        "analysis_scope": {
            "primary_informative_task_families": sorted(informative_families),
            "secondary_context_task_families": sorted(non_informative_families),
            "informative_definition": stratified["analysis_scope"]["informative_definition"],
        },
        "counts": {
            "adapter_dataset_rows": len(cohort_rows),
            "primary_informative_rows": sum(1 for r in cohort_rows if r["inclusion_status"] == "primary_informative"),
            "secondary_context_rows": sum(1 for r in cohort_rows if r["inclusion_status"] == "secondary_context"),
        },
        "cohort_rows": cohort_rows,
    }
    _write_json(V2_DIR / "cohort_definition.json", cohort_payload)

    cohort_md_rows = []
    for row in cohort_rows:
        cohort_md_rows.append(
            [
                row["adapter_id"],
                row["dataset"],
                row["task_family"],
                row["inclusion_status"],
                row["compressibility_label"],
                _fmt(_safe_float(row.get("full_adapter_accuracy")) if row.get("full_adapter_accuracy") is not None else None, 4),
                _fmt(
                    _safe_float(row.get("source_quality_gap_vs_dataset_best"))
                    if row.get("source_quality_gap_vs_dataset_best") is not None
                    else None,
                    4,
                ),
                str(row.get("source_quality_gap_band") or "n/a"),
            ]
        )
    cohort_md = [
        "# Rank Proxy Validation v2 Cohort Definition",
        "",
        "## Scope",
        f"- Primary informative families: {', '.join(sorted(informative_families))}",
        f"- Secondary context families: {', '.join(sorted(non_informative_families))}",
        f"- Adapter x dataset rows: {len(cohort_rows)}",
        "",
        "## Cohort Table",
        _md_table(
            [
                "adapter_id",
                "dataset",
                "task_family",
                "inclusion_status",
                "compressibility",
                "full_acc",
                "quality_gap",
                "quality_band",
            ],
            cohort_md_rows,
        ),
        "",
    ]
    (V2_DIR / "cohort_definition.md").write_text("\n".join(cohort_md))

    # Stage B: allocation table (canonicalized from persisted v1 outputs).
    allocation_table_rows: list[dict[str, Any]] = []
    for row in compression_rows_raw:
        adapter_id = str(row["adapter_id"])
        dataset = str(row["dataset"])
        budget = float(row["budget_ratio"])
        method_raw = str(row["method"])
        method_norm = _norm_method(method_raw)
        tf = _task_family(dataset)
        inclusion = _inclusion_status(tf, informative_families)
        q = quality_map.get((adapter_id, dataset), {})

        share = None
        if method_raw in SPECTRAL_METHODS:
            share = policy_share_map.get((adapter_id, dataset, budget, method_raw))
        elif method_raw in PROXY_METHODS:
            share = proxy_share_map.get((adapter_id, dataset, budget, method_raw))

        allocation_table_rows.append(
            {
                "adapter_id": adapter_id,
                "dataset": dataset,
                "task_family": tf,
                "inclusion_status": inclusion,
                "base_model": str(row["base_model"]),
                "method_original": method_raw,
                "method": method_norm,
                "method_family": _method_family(method_raw),
                "budget_ratio": budget,
                "allocated_params": int(row["allocated_params"]),
                "max_params": int(row["max_params"]),
                "realized_budget_ratio": row["realized_budget_ratio"],
                "attn_budget_share": None if share is None else share.get("attn_budget_share"),
                "mlp_budget_share": None if share is None else share.get("mlp_budget_share"),
                "topk_k_reference": None if share is None else share.get("topk_k_reference"),
                "topk_layers": None,
                "layerwise_allocation_vector": None,
                "allocation_vector_persisted": False,
                "allocation_vector_note": (
                    "v1 artifacts persisted allocation agreement and outcome deltas, "
                    "but not full layerwise vectors; regenerate via runner for exact vectors"
                ),
                "full_adapter_accuracy": row["full_adapter_accuracy"],
                "source_quality_gap_vs_dataset_best": q.get("source_quality_gap_vs_dataset_best"),
                "source_quality_gap_band": q.get("source_quality_gap_band"),
            }
        )

    allocation_table_rows.sort(
        key=lambda r: (r["task_family"], r["dataset"], r["adapter_id"], r["budget_ratio"], r["method"])
    )
    alloc_primary = [r for r in allocation_table_rows if r["inclusion_status"] == "primary_informative"]
    alloc_summary_method: list[dict[str, Any]] = []
    by_method_alloc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in alloc_primary:
        by_method_alloc[row["method"]].append(row)
    for method, rows in sorted(by_method_alloc.items()):
        alloc_summary_method.append(
            {
                "method": method,
                "n": len(rows),
                "mean_realized_budget_ratio": _mean([_safe_float(r["realized_budget_ratio"]) for r in rows]),
                "mean_attn_budget_share": _mean(
                    [_safe_float(r["attn_budget_share"]) for r in rows if r.get("attn_budget_share") is not None]
                ),
                "mean_mlp_budget_share": _mean(
                    [_safe_float(r["mlp_budget_share"]) for r in rows if r.get("mlp_budget_share") is not None]
                ),
            }
        )
    allocation_table_payload = {
        "analysis_scope": {
            "primary_informative_task_families": sorted(informative_families),
            "secondary_context_task_families": sorted(non_informative_families),
        },
        "vector_persistence_status": {
            "layerwise_vectors_persisted": False,
            "note": (
                "This canonical v2 bundle is built from persisted v1 outputs. "
                "Per-layer vectors were not stored in v1; re-run generation is required for exact vectors."
            ),
        },
        "rows": allocation_table_rows,
        "primary_summary_by_method": alloc_summary_method,
    }
    _write_json(V2_DIR / "allocation_table.json", allocation_table_payload)

    alloc_md_rows = []
    for row in alloc_summary_method:
        alloc_md_rows.append(
            [
                row["method"],
                str(row["n"]),
                _fmt(row["mean_realized_budget_ratio"], 3),
                _fmt(row["mean_attn_budget_share"], 3),
                _fmt(row["mean_mlp_budget_share"], 3),
            ]
        )
    alloc_md = [
        "# Rank Proxy Validation v2 Allocation Table",
        "",
        "## Notes",
        "- This v2 bundle canonicalizes existing v1 outputs without a heavy re-run.",
        "- Full layerwise vectors/top-k layer identities were not persisted in v1 and are left null here.",
        "",
        "## Primary Informative Summary by Method",
        _md_table(
            ["method", "n", "mean_realized_budget", "mean_attn_share", "mean_mlp_share"],
            alloc_md_rows,
        ),
        "",
    ]
    (V2_DIR / "allocation_table.md").write_text("\n".join(alloc_md))

    # Stage C: allocation comparison table (primary-first).
    allocation_cmp_rows: list[dict[str, Any]] = []
    for row in allocation_rows_raw:
        adapter_id = str(row["adapter_id"])
        dataset = str(row["dataset"])
        tf = _task_family(dataset)
        inclusion = _inclusion_status(tf, informative_families)
        q = quality_map.get((adapter_id, dataset), {})
        allocation_cmp_rows.append(
            {
                **row,
                "policy_method_original": str(row["policy_method"]),
                "proxy_method_original": str(row["proxy_method"]),
                "policy_method": _norm_method(str(row["policy_method"])),
                "proxy_method": _norm_method(str(row["proxy_method"])),
                "task_family": tf,
                "inclusion_status": inclusion,
                "source_quality_gap_band": q.get("source_quality_gap_band"),
            }
        )
    cmp_primary = [r for r in allocation_cmp_rows if r["inclusion_status"] == "primary_informative"]
    cmp_secondary = [r for r in allocation_cmp_rows if r["inclusion_status"] == "secondary_context"]

    pair_summary_primary = []
    by_pair_primary: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cmp_primary:
        by_pair_primary[(str(row["policy_method"]), str(row["proxy_method"]))].append(row)
    for (policy, proxy), rows in sorted(by_pair_primary.items()):
        pair_summary_primary.append(
            {
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(rows),
                "mean_spearman_rank_correlation": _mean(
                    [_safe_float(r["spearman_rank_correlation"]) for r in rows if r.get("spearman_rank_correlation") is not None]
                ),
                "mean_topk_overlap": _mean([_safe_float(r["topk_overlap"]) for r in rows]),
                "mean_policy_vs_proxy_acc_delta": _mean([_safe_float(r["policy_vs_proxy_acc_delta"]) for r in rows]),
            }
        )

    family_pair_summary_primary = []
    by_family_pair_primary: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cmp_primary:
        by_family_pair_primary[(str(row["task_family"]), str(row["policy_method"]), str(row["proxy_method"]))].append(row)
    for (family, policy, proxy), rows in sorted(by_family_pair_primary.items()):
        family_pair_summary_primary.append(
            {
                "task_family": family,
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(rows),
                "mean_spearman_rank_correlation": _mean(
                    [_safe_float(r["spearman_rank_correlation"]) for r in rows if r.get("spearman_rank_correlation") is not None]
                ),
                "mean_topk_overlap": _mean([_safe_float(r["topk_overlap"]) for r in rows]),
                "mean_policy_vs_proxy_acc_delta": _mean([_safe_float(r["policy_vs_proxy_acc_delta"]) for r in rows]),
            }
        )

    secondary_pair_summary = []
    by_pair_secondary: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cmp_secondary:
        by_pair_secondary[(str(row["policy_method"]), str(row["proxy_method"]))].append(row)
    for (policy, proxy), rows in sorted(by_pair_secondary.items()):
        secondary_pair_summary.append(
            {
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(rows),
                "mean_spearman_rank_correlation": _mean(
                    [_safe_float(r["spearman_rank_correlation"]) for r in rows if r.get("spearman_rank_correlation") is not None]
                ),
                "mean_topk_overlap": _mean([_safe_float(r["topk_overlap"]) for r in rows]),
            }
        )

    allocation_cmp_payload = {
        "analysis_scope": {
            "primary_informative_task_families": sorted(informative_families),
            "secondary_context_task_families": sorted(non_informative_families),
        },
        "row_counts": {
            "primary_rows": len(cmp_primary),
            "secondary_rows": len(cmp_secondary),
        },
        "rows_primary": cmp_primary,
        "summary_by_policy_proxy_primary": pair_summary_primary,
        "summary_by_task_family_policy_proxy_primary": family_pair_summary_primary,
        "secondary_context_summary_by_policy_proxy": secondary_pair_summary,
    }
    _write_json(V2_DIR / "allocation_comparison_table.json", allocation_cmp_payload)

    cmp_md_rows = []
    for row in pair_summary_primary:
        cmp_md_rows.append(
            [
                row["policy_method"],
                row["proxy_method"],
                str(row["n"]),
                _fmt(row["mean_spearman_rank_correlation"], 3),
                _fmt(row["mean_topk_overlap"], 3),
                _fmt(row["mean_policy_vs_proxy_acc_delta"], 4),
            ]
        )
    cmp_md = [
        "# Rank Proxy Validation v2 Allocation Comparison",
        "",
        "## Primary Informative Summary",
        _md_table(
            ["policy_method", "proxy_method", "n", "mean_spearman", "mean_topk_overlap", "mean_policy_minus_proxy_acc"],
            cmp_md_rows,
        ),
        "",
    ]
    (V2_DIR / "allocation_comparison_table.md").write_text("\n".join(cmp_md))

    # Task-family stratified readout.
    strat_family_method = []
    for row in stratified["compression_by_task_family_method"]:
        new_row = dict(row)
        new_row["method_original"] = str(row["method"])
        new_row["method"] = _norm_method(str(row["method"]))
        strat_family_method.append(new_row)
    strat_alloc_family = []
    for row in stratified["allocation_by_task_family_policy_proxy"]:
        new_row = dict(row)
        new_row["policy_method_original"] = str(row["policy_method"])
        new_row["proxy_method_original"] = str(row["proxy_method"])
        new_row["policy_method"] = _norm_method(str(row["policy_method"]))
        new_row["proxy_method"] = _norm_method(str(row["proxy_method"]))
        strat_alloc_family.append(new_row)

    strat_payload_v2 = {
        "analysis_scope": stratified["analysis_scope"],
        "task_family_definition": stratified["task_family_definition"],
        "family_coverage": stratified["family_coverage"],
        "primary_informative_best_method_by_family": [
            {
                **row,
                "best_method_original": row["best_method_by_delta_vs_uniform"],
                "best_method_by_delta_vs_uniform": _norm_method(str(row["best_method_by_delta_vs_uniform"])),
            }
            for row in stratified["best_method_by_family"]
            if row["task_family"] in informative_families
        ],
        "secondary_context_best_method_by_family": [
            {
                **row,
                "best_method_original": row["best_method_by_delta_vs_uniform"],
                "best_method_by_delta_vs_uniform": _norm_method(str(row["best_method_by_delta_vs_uniform"])),
            }
            for row in stratified["best_method_by_family"]
            if row["task_family"] in non_informative_families
        ],
        "compression_by_task_family_method": strat_family_method,
        "allocation_by_task_family_policy_proxy": strat_alloc_family,
    }
    _write_json(V2_DIR / "task_family_stratified_readout.json", strat_payload_v2)

    fam_md_rows = []
    for row in strat_payload_v2["primary_informative_best_method_by_family"]:
        fam_md_rows.append(
            [
                row["task_family"],
                row["best_method_by_delta_vs_uniform"],
                _fmt(row["best_mean_delta_vs_uniform"], 4),
                str(row["n"]),
            ]
        )
    fam_md = [
        "# Rank Proxy Validation v2 Task-Family Stratified Readout",
        "",
        f"- Primary informative families: {', '.join(sorted(informative_families))}",
        f"- Secondary context families: {', '.join(sorted(non_informative_families))}",
        "",
        "## Best Method by Family (Primary Informative)",
        _md_table(["task_family", "best_method", "mean_delta_vs_uniform", "n"], fam_md_rows),
        "",
    ]
    (V2_DIR / "task_family_stratified_readout.md").write_text("\n".join(fam_md))

    # Stage D: compression evaluation with additional deltas.
    compression_rows_v2: list[dict[str, Any]] = []
    for row in compression_rows_raw:
        adapter_id = str(row["adapter_id"])
        dataset = str(row["dataset"])
        budget = float(row["budget_ratio"])
        method_raw = str(row["method"])
        method_norm = _norm_method(method_raw)
        tf = _task_family(dataset)
        inclusion = _inclusion_status(tf, informative_families)
        q = quality_map.get((adapter_id, dataset), {})

        key = (adapter_id, dataset, budget)
        comp_acc = _safe_float(row["compressed_accuracy"])
        rand = random_acc.get(key)
        grad = gradient_acc.get(key)
        abl = ablation_acc.get(key)
        best_proxy = None
        if grad is not None and abl is not None:
            best_proxy = max(grad, abl)
        elif grad is not None:
            best_proxy = grad
        elif abl is not None:
            best_proxy = abl

        compression_rows_v2.append(
            {
                **row,
                "method_original": method_raw,
                "method": method_norm,
                "method_family": _method_family(method_raw),
                "task_family": tf,
                "inclusion_status": inclusion,
                "source_quality_gap_band": q.get("source_quality_gap_band"),
                "source_quality_gap_vs_dataset_best": q.get("source_quality_gap_vs_dataset_best"),
                "delta_vs_random": None if rand is None else comp_acc - rand,
                "delta_vs_gradient": None if grad is None else comp_acc - grad,
                "delta_vs_ablation": None if abl is None else comp_acc - abl,
                "delta_vs_best_proxy": None if best_proxy is None else comp_acc - best_proxy,
            }
        )

    comp_primary = [r for r in compression_rows_v2 if r["inclusion_status"] == "primary_informative"]
    comp_secondary = [r for r in compression_rows_v2 if r["inclusion_status"] == "secondary_context"]

    comp_summary_primary = []
    by_method_comp_primary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comp_primary:
        by_method_comp_primary[str(row["method"])].append(row)
    for method, rows in sorted(by_method_comp_primary.items()):
        dv_full = [_safe_float(r["delta_vs_full_adapter"]) for r in rows]
        dv_uniform = [_safe_float(r["delta_vs_uniform"]) for r in rows]
        dv_random = [_safe_float(r["delta_vs_random"]) for r in rows if r.get("delta_vs_random") is not None]
        comp_summary_primary.append(
            {
                "method": method,
                "n": len(rows),
                "mean_delta_vs_full_adapter": _mean(dv_full),
                "var_delta_vs_full_adapter": _pvar(dv_full),
                "mean_delta_vs_uniform": _mean(dv_uniform),
                "var_delta_vs_uniform": _pvar(dv_uniform),
                "mean_delta_vs_random": _mean(dv_random),
                "var_delta_vs_random": _pvar(dv_random),
                "mean_realized_budget_ratio": _mean([_safe_float(r["realized_budget_ratio"]) for r in rows]),
            }
        )
    spectral_primary = [r for r in comp_summary_primary if r["method"] in SPECTRAL_METHODS]
    lead_spectral = (
        max(spectral_primary, key=lambda r: _safe_float(r["mean_delta_vs_uniform"], -1e9))["method"]
        if spectral_primary
        else None
    )

    comp_payload = {
        "analysis_scope": {
            "primary_informative_task_families": sorted(informative_families),
            "secondary_context_task_families": sorted(non_informative_families),
            "dataset_matched_baseline": "full_adapter_accuracy and source-quality gaps are dataset-matched",
        },
        "row_counts": {"primary_rows": len(comp_primary), "secondary_rows": len(comp_secondary)},
        "rows_primary": comp_primary,
        "summary_by_method_primary": comp_summary_primary,
        "lead_spectral_policy_primary": lead_spectral,
        "secondary_context_method_summary": [
            {
                "method": method,
                "n": len(rows),
                "mean_delta_vs_uniform": _mean([_safe_float(r["delta_vs_uniform"]) for r in rows]),
            }
            for method, rows in sorted(defaultdict(list, {
                m: [r for r in comp_secondary if str(r["method"]) == m]
                for m in sorted({str(r["method"]) for r in comp_secondary})
            }).items())
            if rows
        ],
    }
    _write_json(V2_DIR / "compression_evaluation_table.json", comp_payload)

    comp_md_rows = []
    for row in comp_summary_primary:
        comp_md_rows.append(
            [
                row["method"],
                str(row["n"]),
                _fmt(row["mean_delta_vs_full_adapter"], 4),
                _fmt(row["mean_delta_vs_uniform"], 4),
                _fmt(row["mean_delta_vs_random"], 4),
                _fmt(row["mean_realized_budget_ratio"], 3),
            ]
        )
    comp_md = [
        "# Rank Proxy Validation v2 Compression Evaluation",
        "",
        "## Primary Informative Summary by Method",
        _md_table(
            [
                "method",
                "n",
                "mean_delta_vs_full",
                "mean_delta_vs_uniform",
                "mean_delta_vs_random",
                "mean_realized_budget",
            ],
            comp_md_rows,
        ),
        "",
        f"- Lead spectral policy in primary subset: `{lead_spectral}`" if lead_spectral else "- Lead spectral policy: n/a",
        "",
    ]
    (V2_DIR / "compression_evaluation_table.md").write_text("\n".join(comp_md))

    # Stage E: source-quality control slice and disagreement anatomy.
    quality_primary_rows = [
        row
        for row in quality["primary_informative_subset"]["adapter_quality_table"]
        if str(row["task_family"]) in informative_families
    ]
    band_coverage = quality["primary_informative_subset"]["band_coverage"]

    comp_by_band_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    comp_by_family_band_method: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comp_primary:
        band = str(row.get("source_quality_gap_band") or "unknown")
        method = str(row["method"])
        family = str(row["task_family"])
        comp_by_band_method[(band, method)].append(row)
        comp_by_family_band_method[(family, band, method)].append(row)

    comp_band_rows = []
    for (band, method), rows in sorted(comp_by_band_method.items()):
        comp_band_rows.append(
            {
                "source_quality_gap_band": band,
                "method": method,
                "n": len(rows),
                "mean_delta_vs_uniform": _mean([_safe_float(r["delta_vs_uniform"]) for r in rows]),
                "mean_delta_vs_full_adapter": _mean([_safe_float(r["delta_vs_full_adapter"]) for r in rows]),
                "mean_delta_vs_random": _mean(
                    [_safe_float(r["delta_vs_random"]) for r in rows if r.get("delta_vs_random") is not None]
                ),
            }
        )

    comp_family_band_rows = []
    for (family, band, method), rows in sorted(comp_by_family_band_method.items()):
        comp_family_band_rows.append(
            {
                "task_family": family,
                "source_quality_gap_band": band,
                "method": method,
                "n": len(rows),
                "mean_delta_vs_uniform": _mean([_safe_float(r["delta_vs_uniform"]) for r in rows]),
            }
        )

    best_by_band = []
    by_band_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comp_band_rows:
        by_band_group[row["source_quality_gap_band"]].append(row)
    for band, rows in sorted(by_band_group.items()):
        best = max(rows, key=lambda r: _safe_float(r["mean_delta_vs_uniform"], -1e9))
        best_by_band.append(
            {
                "source_quality_gap_band": band,
                "best_method_by_delta_vs_uniform": best["method"],
                "best_mean_delta_vs_uniform": best["mean_delta_vs_uniform"],
                "n": best["n"],
            }
        )

    alloc_by_band_pair: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cmp_primary:
        band = str(row.get("source_quality_gap_band") or "unknown")
        alloc_by_band_pair[(band, str(row["policy_method"]), str(row["proxy_method"]))].append(row)
    alloc_band_rows = []
    for (band, policy, proxy), rows in sorted(alloc_by_band_pair.items()):
        alloc_band_rows.append(
            {
                "source_quality_gap_band": band,
                "policy_method": policy,
                "proxy_method": proxy,
                "n": len(rows),
                "mean_spearman_rank_correlation": _mean(
                    [_safe_float(r["spearman_rank_correlation"]) for r in rows if r.get("spearman_rank_correlation") is not None]
                ),
                "mean_topk_overlap": _mean([_safe_float(r["topk_overlap"]) for r in rows]),
            }
        )

    source_quality_payload = {
        "analysis_scope": {
            "primary_subset": "informative_task_families_only",
            "informative_task_families": sorted(informative_families),
            "non_informative_task_families": sorted(non_informative_families),
            "informative_definition": stratified["analysis_scope"]["informative_definition"],
        },
        "quality_gap_definition": quality["quality_gap_definition"],
        "primary_informative_subset": {
            "band_coverage": band_coverage,
            "adapter_quality_table": quality_primary_rows,
            "best_method_by_gap_band": best_by_band,
            "compression_by_gap_band_method": comp_band_rows,
            "compression_by_family_gap_band_method": comp_family_band_rows,
            "allocation_by_gap_band_policy_proxy": alloc_band_rows,
        },
        "all_families_context": quality["all_families_context"],
    }
    _write_json(V2_DIR / "source_quality_gap_control_slice.json", source_quality_payload)

    sq_md_rows = []
    for row in best_by_band:
        sq_md_rows.append(
            [
                row["source_quality_gap_band"],
                row["best_method_by_delta_vs_uniform"],
                _fmt(row["best_mean_delta_vs_uniform"], 4),
                str(row["n"]),
            ]
        )
    sq_md = [
        "# Rank Proxy Validation v2 Source-Quality Gap Control Slice",
        "",
        "## Best Method by Source-Quality Band (Primary Informative)",
        _md_table(["gap_band", "best_method", "mean_delta_vs_uniform", "n"], sq_md_rows),
        "",
    ]
    (V2_DIR / "source_quality_gap_control_slice.md").write_text("\n".join(sq_md))

    # Compressible-family summary (primary).
    compressible_md_rows = []
    for row in comp_summary_primary:
        compressible_md_rows.append(
            [
                row["method"],
                str(row["n"]),
                _fmt(row["mean_delta_vs_uniform"], 4),
                _fmt(row["var_delta_vs_uniform"], 6),
                _fmt(row["mean_delta_vs_full_adapter"], 4),
            ]
        )
    comp_summary_md = [
        "# Rank Proxy Validation v2 Compressible-Family Summary",
        "",
        f"- Informative families: {', '.join(sorted(informative_families))}",
        f"- Non-informative context families: {', '.join(sorted(non_informative_families))}",
        "",
        "## Method Performance (Primary Informative)",
        _md_table(
            ["method", "n", "mean_delta_vs_uniform", "var_delta_vs_uniform", "mean_delta_vs_full"],
            compressible_md_rows,
        ),
        "",
        "## Proxy Agreement Split (Primary Informative)",
    ]
    proxy_split_rows = []
    for row in compressible_summary.get("proxy_agreement_split_summary", []):
        proxy_split_rows.append(
            [
                _norm_method(str(row["proxy_method"])),
                str(row["n"]),
                _fmt(row["mean_spearman_rank_correlation"], 3),
                _fmt(row["mean_topk_overlap"], 3),
                _fmt(row["mean_policy_vs_proxy_acc_delta"], 4),
            ]
        )
    comp_summary_md.append(
        _md_table(
            ["proxy_method", "n", "mean_spearman", "mean_topk_overlap", "mean_policy_minus_proxy_acc"],
            proxy_split_rows,
        )
    )
    comp_summary_md.append("")
    (V2_DIR / "compressible_family_summary.md").write_text("\n".join(comp_summary_md))

    # Stage E: disagreement memo.
    oht_vs_grad_band_rows = []
    for band in sorted({r["source_quality_gap_band"] for r in comp_band_rows}):
        rows_oht = [r for r in comp_band_rows if r["source_quality_gap_band"] == band and r["method"] == "oht"]
        rows_grad = [r for r in comp_band_rows if r["source_quality_gap_band"] == band and r["method"] == "proxy_gradient"]
        if not rows_oht or not rows_grad:
            continue
        oht_dvu = _safe_float(rows_oht[0]["mean_delta_vs_uniform"])
        grad_dvu = _safe_float(rows_grad[0]["mean_delta_vs_uniform"])
        oht_vs_grad_band_rows.append(
            {
                "source_quality_gap_band": band,
                "oht_mean_delta_vs_uniform": oht_dvu,
                "proxy_gradient_mean_delta_vs_uniform": grad_dvu,
                "gradient_minus_oht": grad_dvu - oht_dvu,
            }
        )

    oht_vs_proxy_family_rows = []
    for family in sorted(informative_families):
        oht_ab = [
            r
            for r in family_pair_summary_primary
            if r["task_family"] == family and r["policy_method"] == "oht" and r["proxy_method"] == "proxy_ablation_attenuate"
        ]
        oht_gr = [
            r
            for r in family_pair_summary_primary
            if r["task_family"] == family and r["policy_method"] == "oht" and r["proxy_method"] == "proxy_gradient"
        ]
        if not oht_ab or not oht_gr:
            continue
        oht_vs_proxy_family_rows.append(
            {
                "task_family": family,
                "oht_vs_ablation_mean_spearman": oht_ab[0]["mean_spearman_rank_correlation"],
                "oht_vs_ablation_mean_topk_overlap": oht_ab[0]["mean_topk_overlap"],
                "oht_vs_gradient_mean_spearman": oht_gr[0]["mean_spearman_rank_correlation"],
                "oht_vs_gradient_mean_topk_overlap": oht_gr[0]["mean_topk_overlap"],
            }
        )

    disagree_lines = [
        "# Rank Proxy Validation v2 Disagreement Memo",
        "",
        "## Scope",
        f"- Primary informative families: {', '.join(sorted(informative_families))}",
        f"- Secondary context families: {', '.join(sorted(non_informative_families))}",
        f"- Compression rows (primary): {len(comp_primary)}",
        f"- Allocation comparison rows (primary): {len(cmp_primary)}",
        "",
        "## Gradient vs OHT by Source-Quality Band",
    ]
    band_rows_md = []
    for row in oht_vs_grad_band_rows:
        band_rows_md.append(
            [
                row["source_quality_gap_band"],
                _fmt(row["oht_mean_delta_vs_uniform"], 4),
                _fmt(row["proxy_gradient_mean_delta_vs_uniform"], 4),
                _fmt(row["gradient_minus_oht"], 4),
            ]
        )
    disagree_lines.append(
        _md_table(
            ["gap_band", "oht_mean_delta_vs_uniform", "gradient_mean_delta_vs_uniform", "gradient_minus_oht"],
            band_rows_md,
        )
    )
    disagree_lines += [
        "",
        "## OHT Structural Alignment by Family",
    ]
    fam_rows_md = []
    for row in oht_vs_proxy_family_rows:
        fam_rows_md.append(
            [
                row["task_family"],
                _fmt(row["oht_vs_ablation_mean_spearman"], 3),
                _fmt(row["oht_vs_gradient_mean_spearman"], 3),
                _fmt(row["oht_vs_ablation_mean_topk_overlap"], 3),
                _fmt(row["oht_vs_gradient_mean_topk_overlap"], 3),
            ]
        )
    disagree_lines.append(
        _md_table(
            [
                "task_family",
                "oht_vs_ablation_spearman",
                "oht_vs_gradient_spearman",
                "oht_vs_ablation_topk",
                "oht_vs_gradient_topk",
            ],
            fam_rows_md,
        )
    )
    disagree_lines += [
        "",
        "## Interpretation",
        "- Structural similarity and operational superiority are distinct: OHT can align more with ablation-style structure while gradient remains stronger on mean compression outcome in this CPU-bounded setup.",
        "- Source-quality control remains necessary: near-top and mid-gap bands can show different method ordering from large-gap bands.",
        "- Saturated families remain non-informative for primary policy interpretation.",
        "",
    ]
    (V2_DIR / "disagreement_memo.md").write_text("\n".join(disagree_lines))

    # Stage F: bounded validation memo + optional summary JSON.
    lead_spectral_row = next((r for r in comp_summary_primary if r["method"] == lead_spectral), None)
    grad_row = next((r for r in comp_summary_primary if r["method"] == "proxy_gradient"), None)
    ablation_row = next((r for r in comp_summary_primary if r["method"] == "proxy_ablation_attenuate"), None)

    ablation_reliability_note = None
    if ablation_gap is not None:
        stab = ablation_gap.get("proxy_stability_resampling", {}).get("summary_all", {})
        if stab:
            ablation_reliability_note = {
                "mean_gradient_pairwise_spearman": stab.get("mean_gradient_pairwise_spearman"),
                "mean_ablation_pairwise_spearman": stab.get("mean_ablation_pairwise_spearman"),
            }

    bounded_summary_payload = {
        "study": "adaptive_rank_comparison_external_validation_target_v2",
        "primary_informative_task_families": sorted(informative_families),
        "secondary_context_task_families": sorted(non_informative_families),
        "lead_spectral_policy_primary": lead_spectral,
        "lead_spectral_mean_delta_vs_uniform": None if lead_spectral_row is None else lead_spectral_row["mean_delta_vs_uniform"],
        "proxy_gradient_mean_delta_vs_uniform_primary": None if grad_row is None else grad_row["mean_delta_vs_uniform"],
        "proxy_ablation_mean_delta_vs_uniform_primary": None if ablation_row is None else ablation_row["mean_delta_vs_uniform"],
        "policy_interpretation": {
            "operational_default_proxy": "proxy_gradient",
            "explanatory_companion_proxy": "proxy_ablation_attenuate",
            "lead_spectral_policy": lead_spectral,
        },
        "ablation_reliability_resampling_note": ablation_reliability_note,
        "guardrails": [
            "encoder-only bounded regime",
            "classification-only bounded regime",
            "compressible families only for primary claims",
            "no adaptive-training equivalence claim",
        ],
    }
    _write_json(V2_DIR / "bounded_validation_summary.json", bounded_summary_payload)

    bounded_memo_lines = [
        "# Rank Proxy Validation v2 Bounded Validation Memo",
        "",
        "## 1. What Was Tested",
        f"- Primary informative subset: {', '.join(sorted(informative_families))}.",
        f"- Secondary context subset: {', '.join(sorted(non_informative_families))}.",
        f"- Methods: {', '.join(sorted({_norm_method(str(r['method'])) for r in compression_rows_raw}))}.",
        f"- Budgets: {', '.join(str(b) for b in sorted(set(_safe_float(r['budget_ratio']) for r in compression_rows_raw)))}.",
        "- Evaluation is dataset-matched per adapter and budget, with source-quality-gap slices retained in the primary interpretation.",
        "",
        "## 2. Strongest Positive Result",
        (
            f"- Lead spectral policy in the primary informative subset is `{lead_spectral}` "
            f"(mean delta_vs_uniform={_fmt(lead_spectral_row['mean_delta_vs_uniform'], 4)})."
            if lead_spectral_row is not None
            else "- Lead spectral policy could not be identified."
        ),
        "- Spectral policies remain competitive against simple matched-budget baselines in the compressible encoder subset.",
        "- Structural agreement remains stronger against ablation-style proxy patterns than against gradient-style patterns in primary allocation-comparison summaries.",
        "",
        "## 3. What Remains Bounded",
        "- Evidence remains bounded to CPU-only encoder classification settings with primary claims restricted to compressible families.",
        "- Saturated families are retained only as secondary context and do not drive main policy interpretation.",
        "- This does not support equivalence to adaptive-training rank-allocation methods.",
        "",
        "## 4. Current Policy Interpretation",
        "- `proxy_gradient` remains the operational default comparator.",
        "- `proxy_ablation_attenuate` remains the explanatory companion comparator.",
        f"- `{lead_spectral or 'oht'}` remains the lead spectral policy in this bounded regime.",
        "- Keep the cheap-rank-advisor claim as competitive and bounded, not dominant or universal.",
        "",
        "## 5. What Would Strengthen This Line Next",
        "- Recover or regenerate full layerwise allocation vectors as first-class persisted artifacts for future reproducibility slices.",
        "- Expand compressible-family cohort size with explicit source-quality controls before broader claim escalation.",
        "- Add external recovered allocation targets from published adaptive methods before any stronger external validation claim.",
        "",
    ]
    (V2_DIR / "bounded_validation_memo.md").write_text("\n".join(bounded_memo_lines))

    # Strategy doc update for canonical v2 path.
    strategy_lines = [
        "# Rank Proxy Bounded Validation Summary",
        "",
        "## Bounded Validation Status",
        "- Status: bounded competitive signal in compressible encoder families; no broad escalation.",
        "- Primary informative families: `sst2`, `imdb`.",
        "- Secondary context families: `tweet_eval`, `ag_news` (saturated/non-informative for main interpretation).",
        "",
        "## Frozen Policy Roles",
        "- Operational default comparator: `proxy_gradient`.",
        "- Explanatory companion comparator: `proxy_ablation_attenuate`.",
        f"- Lead spectral policy in current bounded regime: `{lead_spectral or 'oht'}`.",
        "",
        "## Bounded Claim",
        "- In the compressible encoder subset, Gradience spectral rank policies are competitive fixed-budget allocation guides.",
        "- Spectral allocation structure aligns more with ablation-style structure than with gradient-style structure.",
        "- Gradient remains the stronger operational comparator under current CPU protocol.",
        "",
        "## Guardrails",
        "- Do not claim adaptive-training equivalence.",
        "- Do not treat saturated families as primary policy evidence.",
        "- Do not generalize beyond current encoder/classification bounded regime.",
        "",
        "## Canonical Artifacts",
        "- `field_trials/rank_proxy_validation_v2/cohort_definition.{md,json}`",
        "- `field_trials/rank_proxy_validation_v2/allocation_table.{md,json}`",
        "- `field_trials/rank_proxy_validation_v2/allocation_comparison_table.{md,json}`",
        "- `field_trials/rank_proxy_validation_v2/task_family_stratified_readout.{md,json}`",
        "- `field_trials/rank_proxy_validation_v2/compression_evaluation_table.{md,json}`",
        "- `field_trials/rank_proxy_validation_v2/source_quality_gap_control_slice.{md,json}`",
        "- `field_trials/rank_proxy_validation_v2/disagreement_memo.md`",
        "- `field_trials/rank_proxy_validation_v2/bounded_validation_memo.md`",
        "",
    ]
    DOC_SUMMARY_PATH.write_text("\n".join(strategy_lines))


if __name__ == "__main__":
    build()

