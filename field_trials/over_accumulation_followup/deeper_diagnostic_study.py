#!/usr/bin/env python3
"""Deeper over-accumulation diagnostic study (activation-first strict rerun pass).

Implements the requested sequence:
1) Activation/threshold audit under current code thresholds
2) Targeted high-overlap cohort selection from OA upper tail
3) Strict naive reruns (uniform linear only) with dataset-matched source baselines
4) Reassessment of predictive value on rerun outcomes
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import traceback
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_TRIALS_DIR = REPO_ROOT / "field_trials"
FOLLOWUP_DIR = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gradience.vnext.merge import merge_audit
from field_trials.evidence_bootstrap import run_eval
from field_trials.run_phase2_eval import PairSpec, run_pair


DATASET_NUM_LABELS: dict[str, int] = {
    "imdb": 2,
    "sst2": 2,
    "tweet_eval/emotion": 4,
    "tweet_eval/hate": 2,
    "tweet_eval/irony": 2,
    "ag_news": 4,
    "mnli": 3,
    "yelp_polarity": 2,
    "amazon_polarity": 2,
}

DEFAULT_OVERLAP_GATE = 0.20
DEFAULT_CONFLICT_GATE = 0.10
POOR_DELTA_THRESHOLD = -0.05
DEFAULT_N_HIGH = 6
DEFAULT_N_LOW = 6


def _task_relationship(dataset_a: str | None, dataset_b: str | None) -> str:
    if dataset_a and dataset_b and dataset_a == dataset_b:
        return "same_task"
    if _dataset_family(dataset_a) == _dataset_family(dataset_b):
        return "same_family"
    return "cross_family"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _model_family(base_model: str | None) -> str:
    b = (base_model or "").lower()
    if "distilbert" in b:
        return "distilbert"
    if "roberta" in b:
        return "roberta"
    if "bert" in b and "roberta" not in b and "distilbert" not in b:
        return "bert"
    return b or "unknown"


def _dataset_family(dataset: str | None) -> str:
    ds = (dataset or "").strip()
    if ds in {"sst2", "imdb", "yelp_polarity", "amazon_polarity"}:
        return "sentiment_binary"
    if ds in {"ag_news"}:
        return "topic_classification"
    if ds in {"mnli"}:
        return "nli"
    if ds.startswith("tweet_eval/"):
        return "tweet_eval"
    return ds or "unknown"


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None

    def _rank(vals: list[float]) -> list[float]:
        ordered = sorted((v, i) for i, v in enumerate(vals))
        out = [0.0] * len(vals)
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                j += 1
            rank_v = ((i + j - 1) / 2.0) + 1.0
            for k in range(i, j):
                out[ordered[k][1]] = rank_v
            i = j
        return out

    rx = _rank(x)
    ry = _rank(y)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = sum((a - mx) ** 2 for a in rx)
    den_y = sum((b - my) ** 2 for b in ry)
    den = math.sqrt(den_x * den_y)
    if den <= 0.0:
        return None
    return num / den


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _quantiles(values: list[float], qs: list[float]) -> dict[str, float | None]:
    clean = sorted(v for v in values if not math.isnan(float(v)))
    if not clean:
        return {f"q{int(q * 100)}": None for q in qs}
    out: dict[str, float | None] = {}
    n = len(clean)
    for q in qs:
        idx = int(q * (n - 1))
        out[f"q{int(q * 100)}"] = clean[idx]
    return out


def _strict_naive_plan_ok(plan: dict[str, Any]) -> tuple[bool, str]:
    cfgs = list(plan.get("layer_configs") or [])
    if not cfgs:
        return (False, "no_layer_configs")
    for cfg in cfgs:
        st = str(cfg.get("strategy", "")).lower().strip()
        coeff = list(cfg.get("coefficients") or [])
        if len(coeff) != 2:
            return (False, "missing_coefficients")
        if abs(_safe_float(coeff[0]) - 0.5) > 1e-6 or abs(_safe_float(coeff[1]) - 0.5) > 1e-6:
            return (False, "non_uniform_coefficients")
        if st not in {"linear", "uniform_linear"}:
            return (False, f"non_linear_strategy:{st}")
    return (True, "ok")


def _build_candidate_pools(
    activation: dict[str, Any],
    *,
    overlap_gate: float,
    conflict_gate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_pairs = list(activation["all_pairs"])
    candidates = [
        r
        for r in all_pairs
        if r["both_loadable"]
        and r["num_labels_equal"]
        and r["mean_overlap"] >= overlap_gate
        and r["conflict_fraction"] <= conflict_gate
    ]
    preferred = [r for r in candidates if r["task_relationship"] in {"same_task", "same_family"}]
    return candidates, preferred


def _resolve_target_pair_split(
    activation: dict[str, Any],
    *,
    overlap_gate: float,
    conflict_gate: float,
    default_n_high: int,
    default_n_low: int,
    target_min_pairs: int | None,
    target_max_pairs: int | None,
) -> dict[str, Any]:
    if target_min_pairs is None and target_max_pairs is None:
        return {
            "n_high": default_n_high,
            "n_low": default_n_low,
            "requested_window": None,
            "resolved_overlap_gate": overlap_gate,
            "resolved_conflict_gate": conflict_gate,
            "used_target_window_mode": False,
        }

    requested_min = int(target_min_pairs if target_min_pairs is not None else (default_n_high + default_n_low))
    requested_max = int(target_max_pairs if target_max_pairs is not None else requested_min)
    if requested_min <= 0 or requested_max <= 0:
        raise ValueError("Target pair window must be positive.")
    if requested_max < requested_min:
        raise ValueError("Target max pairs must be >= target min pairs.")

    def _selection_counts(ov: float, cf: float) -> tuple[int, int, int]:
        candidates, preferred = _build_candidate_pools(
            activation,
            overlap_gate=ov,
            conflict_gate=cf,
        )
        requested_total = min(requested_max, len(candidates))
        preferred_need = max(requested_total, int(math.ceil(requested_total / 2.0)) + 2)
        working_pool_count = len(preferred) if len(preferred) >= preferred_need else len(candidates)
        chosen_total = min(requested_total, working_pool_count)
        return (len(candidates), len(preferred), chosen_total)

    resolved_overlap = float(overlap_gate)
    resolved_conflict = float(conflict_gate)
    cand_count, pref_count, chosen_total = _selection_counts(resolved_overlap, resolved_conflict)

    # If the requested window cannot be met, relax gates minimally (overlap first, then conflict).
    if chosen_total < requested_min:
        overlap_grid = [round(x, 4) for x in [resolved_overlap - (i * 0.01) for i in range(int(resolved_overlap / 0.01) + 1)]]
        overlap_grid = [x for x in overlap_grid if x >= 0.0]
        if not overlap_grid:
            overlap_grid = [0.0]
        conflict_grid = [round(min(1.0, resolved_conflict + (i * 0.02)), 4) for i in range(0, 16)]

        found = False
        for cf in conflict_grid:
            for ov in overlap_grid:
                c_count, p_count, c_total = _selection_counts(ov, cf)
                if c_total >= requested_min:
                    resolved_overlap = ov
                    resolved_conflict = cf
                    cand_count, pref_count, chosen_total = c_count, p_count, c_total
                    found = True
                    break
            if found:
                break

    working_pool_count = chosen_total

    n_high = int(math.ceil(chosen_total / 2.0))
    n_low = int(max(0, chosen_total - n_high))

    return {
        "n_high": n_high,
        "n_low": n_low,
        "requested_window": {
            "min_pairs": requested_min,
            "max_pairs": requested_max,
            "candidate_count": cand_count,
            "preferred_candidate_count": pref_count,
            "working_pool_count": working_pool_count,
            "selected_total_pairs": chosen_total,
            "selected_in_requested_window": requested_min <= chosen_total <= requested_max,
            "resolved_overlap_gate": resolved_overlap,
            "resolved_conflict_gate": resolved_conflict,
        },
        "resolved_overlap_gate": resolved_overlap,
        "resolved_conflict_gate": resolved_conflict,
        "used_target_window_mode": True,
    }


@dataclass
class AdapterInfo:
    adapter_id: str
    local_name: str
    adapter_dir: Path
    inventory_dir: Path
    evidence_path: Path
    dataset: str | None
    adapter_score: float | None
    base_model: str | None
    num_labels: int | None
    model_family: str
    is_loadable: bool


def _inventory_dirs() -> list[Path]:
    dirs = []
    dirs.extend(
        p for p in FIELD_TRIALS_DIR.glob("inventory_*") if (p / "adapter_cache").exists() and (p / "evidence").exists()
    )
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
        s = d.as_posix()
        if s in seen:
            continue
        seen.add(s)
        out.append(d)
    return out


def collect_adapters() -> dict[str, Any]:
    all_candidates: list[AdapterInfo] = []
    for inv in _inventory_dirs():
        ev_dir = inv / "evidence"
        cache_dir = inv / "adapter_cache"
        for ev_path in sorted(ev_dir.glob("*.json")):
            try:
                ev = _json(ev_path)
            except Exception:
                continue
            local_name = ev_path.stem
            ad_dir = cache_dir / local_name
            if not ad_dir.exists():
                continue
            adapter_id = str(ev.get("adapter_id") or local_name)
            ds = ev.get("eval_dataset")
            score = ev.get("adapter_score")
            score_f = _safe_float(score, default=math.nan)
            score_f = None if math.isnan(score_f) else score_f
            num_labels = DATASET_NUM_LABELS.get(ds) if isinstance(ds, str) else None
            base_model = ev.get("base_model")
            all_candidates.append(
                AdapterInfo(
                    adapter_id=adapter_id,
                    local_name=local_name,
                    adapter_dir=ad_dir,
                    inventory_dir=inv,
                    evidence_path=ev_path,
                    dataset=ds if isinstance(ds, str) else None,
                    adapter_score=score_f,
                    base_model=base_model if isinstance(base_model, str) else None,
                    num_labels=num_labels,
                    model_family=_model_family(base_model if isinstance(base_model, str) else None),
                    is_loadable=(score_f is not None),
                )
            )

    # Dedup by adapter_id for audit stability; keep lexicographically first inventory.
    dedup: dict[str, AdapterInfo] = {}
    for a in sorted(all_candidates, key=lambda x: (x.adapter_id, x.inventory_dir.as_posix(), x.local_name)):
        if a.adapter_id not in dedup:
            dedup[a.adapter_id] = a

    adapters = list(dedup.values())
    by_family: dict[str, list[AdapterInfo]] = {}
    for a in adapters:
        by_family.setdefault(a.model_family, []).append(a)
    for fam in by_family:
        by_family[fam] = sorted(by_family[fam], key=lambda x: x.adapter_id)

    return {
        "all_candidates_count": len(all_candidates),
        "dedup_adapters": adapters,
        "by_family": by_family,
    }


def run_activation_audit(adapters_by_family: dict[str, list[AdapterInfo]]) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for fam, adapters in sorted(adapters_by_family.items()):
        for a, b in combinations(adapters, 2):
            # Use adapter IDs for global uniqueness across inventories.
            pair_id = f"{a.adapter_id.replace('/', '__')}__x__{b.adapter_id.replace('/', '__')}"
            try:
                report = merge_audit(str(a.adapter_dir), str(b.adapter_dir), verbose=False)
            except Exception as e:
                errors.append(
                    {
                        "pair_id": pair_id,
                        "family": fam,
                        "adapter_a": a.local_name,
                        "adapter_b": b.local_name,
                        "error": str(e),
                    }
                )
                continue

            agg = report.aggregate
            n_layers = max(len(report.layer_verdicts), 1)
            conf_frac = _safe_float(agg.n_conflicting) / n_layers
            imb_frac = _safe_float(agg.n_imbalanced) / n_layers
            max_score = _safe_float(agg.max_over_accumulation_score)
            if math.isnan(max_score):
                max_score = math.nan

            relationship = _task_relationship(a.dataset, b.dataset)
            fam_a = _dataset_family(a.dataset)
            fam_b = _dataset_family(b.dataset)
            fam_pair = "__".join(sorted([fam_a, fam_b]))

            layer_entries: list[dict[str, Any]] = []
            alignments: list[float] = []
            concentrations: list[float] = []
            coeff_exposures: list[float] = []
            triple_high_count = 0

            for lv in report.layer_verdicts:
                score = _safe_float(lv.get("over_accumulation_score"), default=math.nan)
                if math.isnan(score):
                    continue
                band = str(lv.get("over_accumulation_band") or "low")
                factors = lv.get("over_accumulation_factors") or {}
                alignment = _safe_float(factors.get("alignment"), default=math.nan)
                concentration = _safe_float(factors.get("concentration"), default=math.nan)
                coeff_exp = _safe_float(factors.get("coefficient_exposure"), default=math.nan)

                if not math.isnan(alignment):
                    alignments.append(alignment)
                if not math.isnan(concentration):
                    concentrations.append(concentration)
                if not math.isnan(coeff_exp):
                    coeff_exposures.append(coeff_exp)

                if (
                    not math.isnan(alignment)
                    and not math.isnan(concentration)
                    and not math.isnan(coeff_exp)
                    and alignment >= 0.45
                    and concentration >= 0.45
                    and coeff_exp >= 0.90
                ):
                    triple_high_count += 1

                layer_entries.append(
                    {
                        "pair_id": pair_id,
                        "family": fam,
                        "adapter_a": a.local_name,
                        "adapter_b": b.local_name,
                        "layer_name": str(lv.get("layer_name") or ""),
                        "module_type": str(lv.get("module_type") or ""),
                        "score": score,
                        "band": band,
                        "alignment": alignment,
                        "concentration": concentration,
                        "coefficient_exposure": coeff_exp,
                    }
                )

            pair_row = {
                "pair_id": pair_id,
                "family": fam,
                "adapter_a": a.local_name,
                "adapter_b": b.local_name,
                "adapter_a_id": a.adapter_id,
                "adapter_b_id": b.adapter_id,
                "adapter_a_dataset": a.dataset,
                "adapter_b_dataset": b.dataset,
                "adapter_a_loadable": a.is_loadable,
                "adapter_b_loadable": b.is_loadable,
                "both_loadable": bool(a.is_loadable and b.is_loadable),
                "num_labels_equal": bool(a.num_labels is not None and a.num_labels == b.num_labels),
                "adapter_a_score": a.adapter_score,
                "adapter_b_score": b.adapter_score,
                "task_relationship": relationship,
                "dataset_family_a": fam_a,
                "dataset_family_b": fam_b,
                "dataset_family_pair": fam_pair,
                "mean_overlap": _safe_float(agg.mean_overlap),
                "median_overlap": _safe_float(agg.median_overlap),
                "max_overlap": _safe_float(agg.max_overlap),
                "mean_agreement": _safe_float(agg.mean_agreement),
                "overall_verdict": str(agg.overall_verdict),
                "compatibility_score": _safe_float(agg.compatibility_score),
                "n_layers": n_layers,
                "n_conflicting": int(agg.n_conflicting),
                "n_imbalanced": int(agg.n_imbalanced),
                "conflict_fraction": conf_frac,
                "imbalance_fraction": imb_frac,
                "over_accumulation_advisory": str(agg.over_accumulation_advisory),
                "high_risk_layer_count": int(agg.high_risk_layer_count),
                "watch_layer_count": int(agg.watch_layer_count),
                "max_over_accumulation_score": max_score,
                "mean_alignment_factor": (statistics.mean(alignments) if alignments else None),
                "max_alignment_factor": (max(alignments) if alignments else None),
                "mean_concentration_factor": (statistics.mean(concentrations) if concentrations else None),
                "max_concentration_factor": (max(concentrations) if concentrations else None),
                "mean_coefficient_exposure_factor": (
                    statistics.mean(coeff_exposures) if coeff_exposures else None
                ),
                "max_coefficient_exposure_factor": (max(coeff_exposures) if coeff_exposures else None),
                "triple_high_layer_count": triple_high_count,
                "triple_high_layer_fraction": (triple_high_count / n_layers),
            }
            pair_rows.append(pair_row)
            layer_rows.extend(layer_entries)

    pair_scores = [
        r["max_over_accumulation_score"]
        for r in pair_rows
        if not math.isnan(float(r["max_over_accumulation_score"]))
    ]
    layer_scores = [r["score"] for r in layer_rows if not math.isnan(float(r["score"]))]

    pair_advisory_counts: dict[str, int] = {}
    for r in pair_rows:
        key = r["over_accumulation_advisory"]
        pair_advisory_counts[key] = pair_advisory_counts.get(key, 0) + 1

    layer_band_counts: dict[str, int] = {}
    for r in layer_rows:
        key = r["band"]
        layer_band_counts[key] = layer_band_counts.get(key, 0) + 1

    cutpoint_summary = {
        "layer_watch_rule": "alignment >= 0.45 and score >= 0.35",
        "layer_high_rule": "alignment >= 0.65 and score >= 0.60",
        "pair_watch_rule": "high_count > 0 OR (high+watch fraction) >= 0.25",
        "pair_elevated_rule": "high_count > 0 AND (high_fraction >= 0.15 OR max_score >= 0.75)",
        "pairs_with_max_score_ge_0_35": sum(1 for s in pair_scores if s >= 0.35),
        "pairs_with_max_score_ge_0_60": sum(1 for s in pair_scores if s >= 0.60),
        "pairs_with_max_score_ge_0_75": sum(1 for s in pair_scores if s >= 0.75),
        "layers_with_score_ge_0_35": sum(1 for s in layer_scores if s >= 0.35),
        "layers_with_score_ge_0_60": sum(1 for s in layer_scores if s >= 0.60),
        "layers_with_band_watch": layer_band_counts.get("watch", 0),
        "layers_with_band_high": layer_band_counts.get("high", 0),
    }

    top_pairs = sorted(
        pair_rows,
        key=lambda r: (
            r["high_risk_layer_count"],
            r["watch_layer_count"],
            r["max_over_accumulation_score"]
            if not math.isnan(float(r["max_over_accumulation_score"]))
            else -1.0,
            r["mean_overlap"],
        ),
        reverse=True,
    )[:40]

    return {
        "audit_scope": {
            "audited_pair_count": len(pair_rows),
            "layer_entry_count": len(layer_rows),
            "audit_error_count": len(errors),
        },
        "pair_advisory_counts": pair_advisory_counts,
        "layer_band_counts": layer_band_counts,
        "pair_max_score_quantiles": _quantiles(pair_scores, [0.5, 0.75, 0.9, 0.95, 0.99]),
        "layer_score_quantiles": _quantiles(layer_scores, [0.5, 0.75, 0.9, 0.95, 0.99]),
        "cutpoint_summary": cutpoint_summary,
        "top_pairs": top_pairs,
        "all_pairs": pair_rows,
        "all_layers": layer_rows,
        "errors": errors,
    }


def build_targeted_cohort(
    activation: dict[str, Any],
    *,
    overlap_gate: float,
    conflict_gate: float,
    n_high: int,
    n_low: int,
) -> dict[str, Any]:
    candidates, preferred = _build_candidate_pools(
        activation,
        overlap_gate=overlap_gate,
        conflict_gate=conflict_gate,
    )

    # Prefer same-task / same-family regime where feasible.
    working_pool = preferred if len(preferred) >= max(n_high + n_low, n_high + 2) else candidates

    rel_priority = {"same_task": 2, "same_family": 1, "cross_family": 0}

    # High-tail: maximize activation chance while preserving regime discipline.
    high_sorted = sorted(
        working_pool,
        key=lambda r: (
            1 if r["over_accumulation_advisory"] == "elevated" else 0,
            1 if r["over_accumulation_advisory"] == "watch" else 0,
            r["high_risk_layer_count"],
            r["watch_layer_count"],
            r["max_over_accumulation_score"]
            if not math.isnan(float(r["max_over_accumulation_score"]))
            else -1.0,
            rel_priority.get(r["task_relationship"], 0),
            r["mean_overlap"],
        ),
        reverse=True,
    )

    high = high_sorted[: min(n_high, len(high_sorted))]
    used_pair_ids = {r["pair_id"] for r in high}

    # Low group: strict lower-score comparators, matched to high-tail context where possible.
    low_pool = [r for r in working_pool if r["pair_id"] not in used_pair_ids]
    low: list[dict[str, Any]] = []

    def _match_key(x: dict[str, Any]) -> tuple[str, str]:
        return (x["task_relationship"], x["dataset_family_pair"])

    def _low_sort_key(x: dict[str, Any], target: dict[str, Any]) -> tuple[float, float, float, float, float]:
        k1 = _match_key(x) == _match_key(target)
        k2 = x["task_relationship"] == target["task_relationship"]
        k3 = x["family"] == target["family"]
        return (
            -float(k1),
            -float(k2),
            -float(k3),
            x["max_over_accumulation_score"]
            if not math.isnan(float(x["max_over_accumulation_score"]))
            else math.inf,
            -x["mean_overlap"],
        )

    for h in high:
        if len(low) >= n_low:
            break
        available = [r for r in low_pool if r["pair_id"] not in {x["pair_id"] for x in low}]
        if not available:
            break
        best = sorted(available, key=lambda x: _low_sort_key(x, h))[0]
        best = dict(best)
        best["matched_to_high_pair"] = h["pair_id"]
        low.append(best)

    # Fill any remaining low slots from lowest-score tail within working pool.
    if len(low) < n_low:
        taken = {r["pair_id"] for r in low}
        extra_pool = [r for r in low_pool if r["pair_id"] not in taken]
        extra_sorted = sorted(
            extra_pool,
            key=lambda r: (
                r["max_over_accumulation_score"]
                if not math.isnan(float(r["max_over_accumulation_score"]))
                else math.inf,
                r["watch_layer_count"],
                r["high_risk_layer_count"],
                -rel_priority.get(r["task_relationship"], 0),
                -r["mean_overlap"],
            ),
        )
        for r in extra_sorted:
            if len(low) >= n_low:
                break
            rr = dict(r)
            rr["matched_to_high_pair"] = None
            low.append(rr)

    # Attach groups + rationale.
    cohort_rows: list[dict[str, Any]] = []
    for r in high:
        rr = dict(r)
        rr["cohort_group"] = "high_tail"
        rr["selection_rationale"] = (
            "Upper-tail over-accumulation candidate under current thresholds "
            "(prioritizing official advisory, watch/high layer presence, max score, and same-task/family)."
        )
        cohort_rows.append(rr)
    for r in low:
        rr = dict(r)
        rr["cohort_group"] = "lower_tail_matched"
        rr["selection_rationale"] = (
            "Lower over-accumulation score comparator matched to high-tail context "
            "(task relationship / dataset family / model family) where possible."
        )
        cohort_rows.append(rr)

    # Stable ordering for reruns (high first, then low; highest overlap first within each).
    cohort_rows.sort(key=lambda r: (0 if r["cohort_group"] == "high_tail" else 1, -r["mean_overlap"]))

    return {
        "selection_parameters": {
            "overlap_gate": overlap_gate,
            "conflict_gate": conflict_gate,
            "n_high": n_high,
            "n_low": n_low,
        },
        "candidate_count": len(candidates),
        "preferred_candidate_count": len(preferred),
        "working_pool_count": len(working_pool),
        "cohort_count": len(cohort_rows),
        "cohort_rows": cohort_rows,
    }


def _base_model_for_pair(row: dict[str, Any], adapters_by_id: dict[str, AdapterInfo]) -> str:
    a = adapters_by_id.get(row["adapter_a_id"])
    b = adapters_by_id.get(row["adapter_b_id"])
    # Prefer adapter A base model; fallback to B; final fallback to distilbert.
    base = (a.base_model if a else None) or (b.base_model if b else None) or "distilbert-base-uncased"
    return base


def run_strict_naive_reruns(
    cohort: dict[str, Any],
    *,
    adapters_by_id: dict[str, AdapterInfo],
    max_samples: int,
    rerun_subdir_name: str = "strict_naive_reruns",
    results_path: Path | None = None,
) -> dict[str, Any]:
    rerun_dir = FOLLOWUP_DIR / rerun_subdir_name
    rerun_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_path or (FOLLOWUP_DIR / "strict_naive_rerun_results.json")

    results: list[dict[str, Any]] = []

    for i, row in enumerate(cohort["cohort_rows"], start=1):
        pair_id = row["pair_id"]
        a = adapters_by_id[row["adapter_a_id"]]
        b = adapters_by_id[row["adapter_b_id"]]

        # Evaluation dataset policy:
        # - same_task: that shared dataset
        # - same_family/cross_family: dataset of stronger native source score (when available)
        eval_dataset = a.dataset or b.dataset
        eval_dataset_selection = "fallback_adapter_a_or_b"
        if row.get("task_relationship") == "same_task" and a.dataset == b.dataset and a.dataset:
            eval_dataset = a.dataset
            eval_dataset_selection = "same_task_shared_dataset"
        else:
            a_score = row.get("adapter_a_score")
            b_score = row.get("adapter_b_score")
            if a.dataset and b.dataset and a_score is not None and b_score is not None:
                if _safe_float(a_score) >= _safe_float(b_score):
                    eval_dataset = a.dataset
                    eval_dataset_selection = "stronger_source_native_dataset_a"
                else:
                    eval_dataset = b.dataset
                    eval_dataset_selection = "stronger_source_native_dataset_b"
            elif a.dataset:
                eval_dataset = a.dataset
                eval_dataset_selection = "adapter_a_native_dataset"
            elif b.dataset:
                eval_dataset = b.dataset
                eval_dataset_selection = "adapter_b_native_dataset"

        if not eval_dataset or eval_dataset not in DATASET_NUM_LABELS:
            results.append(
                {
                    "pair_id": pair_id,
                    "cohort_group": row["cohort_group"],
                    "status": "error",
                    "error": f"Unsupported eval dataset for strict rerun: {eval_dataset}",
                }
            )
            continue

        base_model_id = _base_model_for_pair(row, adapters_by_id)
        num_labels = DATASET_NUM_LABELS[eval_dataset]
        run_name = f"{i:02d}_{pair_id}"

        pair_spec = PairSpec(
            name=run_name,
            adapter_a_dir=a.adapter_dir,
            adapter_b_dir=b.adapter_dir,
            strategy="uniform_linear",
            eval_dataset=eval_dataset,
            base_model_id=base_model_id,
            num_labels=num_labels,
            role=row["cohort_group"],
            notes="Strict naive rerun (uniform_linear) for over-accumulation validation.",
        )

        run_out_dir = rerun_dir
        run_record: dict[str, Any] = {
            "pair_id": pair_id,
            "run_name": run_name,
            "cohort_group": row["cohort_group"],
            "adapter_a_id": row["adapter_a_id"],
            "adapter_b_id": row["adapter_b_id"],
            "adapter_a_name": a.local_name,
            "adapter_b_name": b.local_name,
            "eval_dataset": eval_dataset,
            "eval_dataset_selection": eval_dataset_selection,
            "base_model_id": base_model_id,
            "num_labels": num_labels,
            "pre_rerun_over_accumulation_advisory": row["over_accumulation_advisory"],
            "pre_rerun_max_over_accumulation_score": row["max_over_accumulation_score"],
            "pre_rerun_watch_layer_count": row["watch_layer_count"],
            "pre_rerun_high_risk_layer_count": row["high_risk_layer_count"],
            "pre_rerun_mean_alignment_factor": row.get("mean_alignment_factor"),
            "pre_rerun_max_alignment_factor": row.get("max_alignment_factor"),
            "pre_rerun_mean_concentration_factor": row.get("mean_concentration_factor"),
            "pre_rerun_max_concentration_factor": row.get("max_concentration_factor"),
            "pre_rerun_mean_coefficient_exposure_factor": row.get("mean_coefficient_exposure_factor"),
            "pre_rerun_max_coefficient_exposure_factor": row.get("max_coefficient_exposure_factor"),
            "pre_rerun_triple_high_layer_fraction": row.get("triple_high_layer_fraction"),
            "status": "ok",
        }

        try:
            merged_eval = run_pair(pair_spec, run_out_dir)
            run_record["merged_eval"] = merged_eval

            merge_plan_path = rerun_dir / run_name / "merged_adapter" / "merge_plan.json"
            strict_ok = False
            strict_reason = "missing_merge_plan"
            if merge_plan_path.exists():
                plan = _json(merge_plan_path)
                strict_ok, strict_reason = _strict_naive_plan_ok(plan)
            run_record["strict_naive_check"] = {"ok": strict_ok, "reason": strict_reason}

            # Dataset-matched source baselines (both sources evaluated on the same eval_dataset).
            src_a = run_eval(a.adapter_dir, eval_dataset, base_model_id=base_model_id, max_samples=max_samples)
            src_b = run_eval(b.adapter_dir, eval_dataset, base_model_id=base_model_id, max_samples=max_samples)
            run_record["source_a_eval"] = src_a
            run_record["source_b_eval"] = src_b

            merged_score = _safe_float(merged_eval.get("merged_accuracy"), default=math.nan)
            a_score = _safe_float(src_a.get("adapter_score"), default=math.nan)
            b_score = _safe_float(src_b.get("adapter_score"), default=math.nan)
            valid = [v for v in (a_score, b_score) if not math.isnan(v)]
            if valid and not math.isnan(merged_score):
                best = max(valid)
                worst = min(valid)
                avg = statistics.mean(valid)
                run_record["best_source_score"] = best
                run_record["worst_source_score"] = worst
                run_record["avg_source_score"] = avg
                run_record["merge_delta_vs_best_source"] = merged_score - best
                run_record["merge_delta_vs_worst_source"] = merged_score - worst
                run_record["merge_delta_vs_average_source"] = merged_score - avg
            else:
                run_record["status"] = "error"
                run_record["error"] = "Could not compute source-relative deltas due to missing source scores."

        except Exception as e:
            run_record["status"] = "error"
            run_record["error"] = str(e)
            run_record["traceback"] = traceback.format_exc(limit=8)

        results.append(run_record)
        results_path.write_text(json.dumps(results, indent=2))

    return {"results": results}


def _variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.pvariance(values)


def _task_family_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        fam = _dataset_family(r.get("eval_dataset"))
        out[fam] = out.get(fam, 0) + 1
    return out


def analyze_rerun_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok" and r.get("merge_delta_vs_best_source") is not None]
    high = [r for r in ok if r.get("cohort_group") == "high_tail"]
    low = [r for r in ok if r.get("cohort_group") == "lower_tail_matched"]

    def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        deltas = [float(r["merge_delta_vs_best_source"]) for r in rows]
        return {
            "n": len(rows),
            "mean_delta_vs_best": statistics.mean(deltas) if deltas else None,
            "median_delta_vs_best": statistics.median(deltas) if deltas else None,
            "variance_delta_vs_best": _variance(deltas),
            "poor_count": sum(1 for d in deltas if d <= POOR_DELTA_THRESHOLD),
            "poor_threshold": POOR_DELTA_THRESHOLD,
            "task_family_mix": _task_family_mix(rows),
            "pairs": [r["pair_id"] for r in rows],
        }

    y = [float(r["merge_delta_vs_best_source"]) for r in ok]
    max_oa = [float(r.get("pre_rerun_max_over_accumulation_score", 0.0)) for r in ok]
    watch = [float(r.get("pre_rerun_watch_layer_count", 0.0)) for r in ok]
    triple = [float(r.get("pre_rerun_triple_high_layer_fraction", 0.0) or 0.0) for r in ok]
    mean_align = [float(r.get("pre_rerun_mean_alignment_factor", 0.0) or 0.0) for r in ok]
    mean_conc = [float(r.get("pre_rerun_mean_concentration_factor", 0.0) or 0.0) for r in ok]
    mean_coeff = [float(r.get("pre_rerun_mean_coefficient_exposure_factor", 0.0) or 0.0) for r in ok]
    max_align = [float(r.get("pre_rerun_max_alignment_factor", 0.0) or 0.0) for r in ok]
    max_conc = [float(r.get("pre_rerun_max_concentration_factor", 0.0) or 0.0) for r in ok]
    max_coeff = [float(r.get("pre_rerun_max_coefficient_exposure_factor", 0.0) or 0.0) for r in ok]

    return {
        "overall_ok_count": len(ok),
        "overall_error_count": len(results) - len(ok),
        "group_high_tail": _summary(high),
        "group_lower_tail_matched": _summary(low),
        "overall": _summary(ok),
        "spearman_delta_vs_max_over_accumulation_score": _spearman(max_oa, y),
        "spearman_delta_vs_watch_layer_count": _spearman(watch, y),
        "subfactor_relationships": {
            "spearman_delta_vs_mean_alignment": _spearman(mean_align, y),
            "spearman_delta_vs_mean_concentration": _spearman(mean_conc, y),
            "spearman_delta_vs_mean_coefficient_exposure": _spearman(mean_coeff, y),
            "spearman_delta_vs_max_alignment": _spearman(max_align, y),
            "spearman_delta_vs_max_concentration": _spearman(max_conc, y),
            "spearman_delta_vs_max_coefficient_exposure": _spearman(max_coeff, y),
            "spearman_delta_vs_triple_high_layer_fraction": _spearman(triple, y),
            "n": len(ok),
        },
    }


def build_cutpoint_sweep(
    activation: dict[str, Any], rerun_results: list[dict[str, Any]], thresholds: list[float] | None = None
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

    pairs = list(activation.get("all_pairs", []))
    layers = list(activation.get("all_layers", []))
    ok = [r for r in rerun_results if r.get("status") == "ok" and r.get("merge_delta_vs_best_source") is not None]

    grid_rows: list[dict[str, Any]] = []
    for th in thresholds:
        activated_pairs = [
            p
            for p in pairs
            if not math.isnan(float(p.get("max_over_accumulation_score", math.nan)))
            and float(p.get("max_over_accumulation_score")) >= th
        ]
        activated_layers = [
            l
            for l in layers
            if not math.isnan(float(l.get("score", math.nan))) and float(l.get("score")) >= th
        ]
        activated_reruns = [
            r
            for r in ok
            if float(r.get("pre_rerun_max_over_accumulation_score", 0.0)) >= th
        ]
        deltas = [float(r["merge_delta_vs_best_source"]) for r in activated_reruns]
        grid_rows.append(
            {
                "threshold": th,
                "activated_pair_count": len(activated_pairs),
                "activated_pair_fraction": (len(activated_pairs) / len(pairs)) if pairs else None,
                "activated_layer_count": len(activated_layers),
                "activated_layer_fraction": (len(activated_layers) / len(layers)) if layers else None,
                "rerun_activated_count": len(activated_reruns),
                "rerun_mean_delta_vs_best": statistics.mean(deltas) if deltas else None,
                "rerun_variance_delta_vs_best": _variance(deltas),
                "rerun_task_family_mix": _task_family_mix(activated_reruns),
            }
        )

    buckets = [
        {"name": "lt_0_15", "lo": -math.inf, "hi": 0.15},
        {"name": "0_15_to_0_25", "lo": 0.15, "hi": 0.25},
        {"name": "0_25_to_0_35", "lo": 0.25, "hi": 0.35},
        {"name": "ge_0_35", "lo": 0.35, "hi": math.inf},
    ]
    bucket_rows: list[dict[str, Any]] = []
    for b in buckets:
        rows = [
            r
            for r in ok
            if float(r.get("pre_rerun_max_over_accumulation_score", 0.0)) >= b["lo"]
            and float(r.get("pre_rerun_max_over_accumulation_score", 0.0)) < b["hi"]
        ]
        deltas = [float(r["merge_delta_vs_best_source"]) for r in rows]
        bucket_rows.append(
            {
                "bucket": b["name"],
                "n": len(rows),
                "mean_delta_vs_best": statistics.mean(deltas) if deltas else None,
                "variance_delta_vs_best": _variance(deltas),
                "task_family_mix": _task_family_mix(rows),
                "pairs": [r["pair_id"] for r in rows],
            }
        )

    return {
        "threshold_grid": grid_rows,
        "score_buckets": bucket_rows,
        "notes": [
            "Activated pair/layer rates are computed over full activation-audit inventory.",
            "Outcome stats are computed over strict-naive rerun subset only.",
        ],
    }


def _write_md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def write_activation_md(path: Path, activation: dict[str, Any], inventory_meta: dict[str, Any]) -> None:
    lines = []
    lines.append("# Activation Threshold Audit")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Deduplicated adapters audited: **{len(inventory_meta['dedup_adapters'])}**")
    lines.append(f"- Audited pairs: **{activation['audit_scope']['audited_pair_count']}**")
    lines.append(f"- Layer entries: **{activation['audit_scope']['layer_entry_count']}**")
    lines.append(f"- Audit errors: **{activation['audit_scope']['audit_error_count']}**")
    lines.append("")
    lines.append("## Pair-Level Advisory Activation")
    for k, v in sorted(activation["pair_advisory_counts"].items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Layer-Band Activation")
    for k, v in sorted(activation["layer_band_counts"].items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Score Quantiles")
    lines.append(f"- Pair max score quantiles: {activation['pair_max_score_quantiles']}")
    lines.append(f"- Layer score quantiles: {activation['layer_score_quantiles']}")
    lines.append("")
    lines.append("## Cutpoint Positioning")
    cps = activation["cutpoint_summary"]
    lines.append(f"- Pair max >= 0.35: {cps['pairs_with_max_score_ge_0_35']}")
    lines.append(f"- Pair max >= 0.60: {cps['pairs_with_max_score_ge_0_60']}")
    lines.append(f"- Pair max >= 0.75: {cps['pairs_with_max_score_ge_0_75']}")
    lines.append(f"- Layers score >= 0.35: {cps['layers_with_score_ge_0_35']}")
    lines.append(f"- Layers score >= 0.60: {cps['layers_with_score_ge_0_60']}")
    lines.append(f"- Layers band=watch: {cps['layers_with_band_watch']}")
    lines.append(f"- Layers band=high: {cps['layers_with_band_high']}")
    lines.append("")
    lines.append("## Top OA Pairs")
    top_rows = []
    for r in activation["top_pairs"][:20]:
        top_rows.append(
            [
                r["pair_id"],
                f"{r['mean_overlap']:.3f}",
                _fmt(r["max_over_accumulation_score"], 3),
                r["over_accumulation_advisory"],
                str(r["watch_layer_count"]),
                str(r["high_risk_layer_count"]),
                r["overall_verdict"],
                "yes" if r["both_loadable"] else "no",
                "yes" if r["num_labels_equal"] else "no",
            ]
        )
    lines.append(
        _write_md_table(
            [
                "pair_id",
                "mean_overlap",
                "max_oa",
                "oa_advisory",
                "watch_layers",
                "high_layers",
                "verdict",
                "both_loadable",
                "num_labels_equal",
            ],
            top_rows,
        )
    )
    path.write_text("\n".join(lines) + "\n")


def write_cohort_md(path: Path, cohort: dict[str, Any]) -> None:
    lines = []
    lines.append("# Targeted High-Overlap Cohort (Strict Naive Study)")
    lines.append("")
    lines.append("## Selection Parameters")
    lines.append(f"- {cohort['selection_parameters']}")
    lines.append(f"- Candidate pool size: {cohort['candidate_count']}")
    lines.append(f"- Preferred same-task/same-family pool size: {cohort.get('preferred_candidate_count', 0)}")
    lines.append(f"- Working pool size: {cohort.get('working_pool_count', 0)}")
    lines.append(f"- Final cohort size: {cohort['cohort_count']}")
    lines.append("")
    rows = []
    for r in cohort["cohort_rows"]:
        rows.append(
            [
                r["pair_id"],
                r["cohort_group"],
                str(r["adapter_a_dataset"]),
                str(r["adapter_b_dataset"]),
                str(r.get("task_relationship", "")),
                f"{r['mean_overlap']:.3f}",
                _fmt(r["max_over_accumulation_score"], 3),
                r["over_accumulation_advisory"],
                str(r["watch_layer_count"]),
                str(r["high_risk_layer_count"]),
                str(r.get("matched_to_high_pair") or ""),
                r["selection_rationale"],
            ]
        )
    lines.append(
        _write_md_table(
            [
                "pair_id",
                "group",
                "dataset_a",
                "dataset_b",
                "task_relationship",
                "mean_overlap",
                "max_oa",
                "oa_advisory",
                "watch_layers",
                "high_layers",
                "matched_to_high_pair",
                "rationale",
            ],
            rows,
        )
    )
    path.write_text("\n".join(lines) + "\n")


def write_rerun_md(path: Path, results: list[dict[str, Any]]) -> None:
    lines = []
    lines.append("# Strict Naive Rerun Results")
    lines.append("")
    rows = []
    for r in results:
        rows.append(
            [
                r.get("pair_id", ""),
                r.get("cohort_group", ""),
                r.get("status", ""),
                r.get("eval_dataset", ""),
                _fmt(_safe_float((r.get("merged_eval") or {}).get("merged_accuracy"), default=math.nan), 3),
                _fmt(r.get("best_source_score"), 3),
                _fmt(r.get("merge_delta_vs_best_source"), 3),
                "yes" if (r.get("strict_naive_check") or {}).get("ok") else "no",
                (r.get("strict_naive_check") or {}).get("reason", ""),
            ]
        )
    lines.append(
        _write_md_table(
            [
                "pair_id",
                "group",
                "status",
                "eval_dataset",
                "merged_score",
                "best_source",
                "delta_vs_best",
                "strict_naive_ok",
                "strict_naive_reason",
            ],
            rows,
        )
    )
    path.write_text("\n".join(lines) + "\n")


def write_analysis_md(path: Path, analysis: dict[str, Any]) -> None:
    lines = []
    lines.append("# Strict Naive Explanatory Analysis")
    lines.append("")
    lines.append(f"- Overall successful reruns: {analysis['overall_ok_count']}")
    lines.append(f"- Overall failed reruns: {analysis['overall_error_count']}")
    lines.append("")
    lines.append("## Group Comparison")
    lines.append(
        f"- High-tail n={analysis['group_high_tail']['n']}, "
        f"mean delta vs best={_fmt(analysis['group_high_tail']['mean_delta_vs_best'], 4)}, "
        f"var={_fmt(analysis['group_high_tail']['variance_delta_vs_best'], 4)}, "
        f"poor count={analysis['group_high_tail']['poor_count']}"
    )
    lines.append(
        f"- Lower-tail n={analysis['group_lower_tail_matched']['n']}, "
        f"mean delta vs best={_fmt(analysis['group_lower_tail_matched']['mean_delta_vs_best'], 4)}, "
        f"var={_fmt(analysis['group_lower_tail_matched']['variance_delta_vs_best'], 4)}, "
        f"poor count={analysis['group_lower_tail_matched']['poor_count']}"
    )
    lines.append(
        f"- Overall task-family mix: {analysis['overall'].get('task_family_mix', {})}"
    )
    lines.append("")
    lines.append("## Continuous Relationship")
    lines.append(
        f"- Spearman(delta_vs_best, pre_rerun max OA score): "
        f"{_fmt(analysis['spearman_delta_vs_max_over_accumulation_score'], 4)}"
    )
    lines.append(
        f"- Spearman(delta_vs_best, pre_rerun watch_layer_count): "
        f"{_fmt(analysis['spearman_delta_vs_watch_layer_count'], 4)}"
    )
    sf = analysis.get("subfactor_relationships", {})
    lines.append("")
    lines.append("## Benign vs Inflation-Prone Decomposition")
    lines.append(
        f"- Spearman(delta, mean alignment): {_fmt(sf.get('spearman_delta_vs_mean_alignment'), 4)}"
    )
    lines.append(
        f"- Spearman(delta, mean concentration): {_fmt(sf.get('spearman_delta_vs_mean_concentration'), 4)}"
    )
    lines.append(
        f"- Spearman(delta, mean coefficient exposure): "
        f"{_fmt(sf.get('spearman_delta_vs_mean_coefficient_exposure'), 4)}"
    )
    lines.append(
        f"- Spearman(delta, max alignment): {_fmt(sf.get('spearman_delta_vs_max_alignment'), 4)}"
    )
    lines.append(
        f"- Spearman(delta, max concentration): {_fmt(sf.get('spearman_delta_vs_max_concentration'), 4)}"
    )
    lines.append(
        f"- Spearman(delta, max coefficient exposure): "
        f"{_fmt(sf.get('spearman_delta_vs_max_coefficient_exposure'), 4)}"
    )
    lines.append(
        f"- Spearman(delta, triple-high layer fraction): "
        f"{_fmt(sf.get('spearman_delta_vs_triple_high_layer_fraction'), 4)}"
    )
    path.write_text("\n".join(lines) + "\n")


def write_cutpoint_sweep_md(path: Path, sweep: dict[str, Any]) -> None:
    lines = []
    lines.append("# Over-Accumulation Cutpoint Sweep")
    lines.append("")
    lines.append("## Threshold Grid")
    grid_rows = []
    for r in sweep.get("threshold_grid", []):
        grid_rows.append(
            [
                _fmt(r.get("threshold"), 2),
                str(r.get("activated_pair_count", "")),
                _fmt(r.get("activated_pair_fraction"), 4),
                str(r.get("activated_layer_count", "")),
                _fmt(r.get("activated_layer_fraction"), 4),
                str(r.get("rerun_activated_count", "")),
                _fmt(r.get("rerun_mean_delta_vs_best"), 4),
                _fmt(r.get("rerun_variance_delta_vs_best"), 4),
                str(r.get("rerun_task_family_mix", {})),
            ]
        )
    lines.append(
        _write_md_table(
            [
                "threshold",
                "activated_pairs",
                "pair_fraction",
                "activated_layers",
                "layer_fraction",
                "rerun_activated_n",
                "rerun_mean_delta_vs_best",
                "rerun_var_delta_vs_best",
                "rerun_task_family_mix",
            ],
            grid_rows,
        )
    )
    lines.append("")
    lines.append("## Top-Tail Outcome Buckets")
    bucket_rows = []
    for r in sweep.get("score_buckets", []):
        bucket_rows.append(
            [
                str(r.get("bucket", "")),
                str(r.get("n", "")),
                _fmt(r.get("mean_delta_vs_best"), 4),
                _fmt(r.get("variance_delta_vs_best"), 4),
                str(r.get("task_family_mix", {})),
            ]
        )
    lines.append(
        _write_md_table(
            ["bucket", "n", "mean_delta_vs_best", "var_delta_vs_best", "task_family_mix"],
            bucket_rows,
        )
    )
    if sweep.get("notes"):
        lines.append("")
        lines.append("## Notes")
        for note in sweep["notes"]:
            lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n")


def write_validation_memo(
    path: Path, activation: dict[str, Any], cohort: dict[str, Any], analysis: dict[str, Any], sweep: dict[str, Any]
) -> None:
    lines = []
    lines.append("# Deeper Diagnostic Study Memo")
    lines.append("")
    lines.append("## 1) Activation Audit Outcome")
    lines.append(
        f"- Audited pairs: {activation['audit_scope']['audited_pair_count']}, "
        f"pair advisories: {activation['pair_advisory_counts']}."
    )
    lines.append(
        f"- Layer bands: {activation['layer_band_counts']}. "
        "This quantifies whether current thresholds activate often enough for a meaningful validation slice."
    )
    lines.append("")
    lines.append("## 2) Targeted Cohort Construction")
    lines.append(
        f"- Cohort built from high-overlap/non-conflict/loadable/num-label-compatible slice: "
        f"{cohort['cohort_count']} pairs ({cohort['selection_parameters']})."
    )
    lines.append(
        "- High-tail was selected to maximize advisory activation chance under current code; "
        "lower-tail comparators were selected from the same candidate regime."
    )
    lines.append("")
    lines.append("## 3) Strict Naive Reruns")
    lines.append(
        "- Every rerun used `uniform_linear` with strict plan checks to confirm 0.5/0.5 linear layer configs."
    )
    lines.append(
        "- Source baselines were rerun on the exact evaluation dataset for each pair (dataset-matched source baselines)."
    )
    lines.append("")
    lines.append("## 4) Initial Reassessment")
    lines.append(
        f"- High-tail mean delta vs best: {_fmt(analysis['group_high_tail']['mean_delta_vs_best'], 4)} "
        f"(n={analysis['group_high_tail']['n']})."
    )
    lines.append(
        f"- Lower-tail mean delta vs best: {_fmt(analysis['group_lower_tail_matched']['mean_delta_vs_best'], 4)} "
        f"(n={analysis['group_lower_tail_matched']['n']})."
    )
    lines.append(
        f"- Spearman(delta, OA max score): {_fmt(analysis['spearman_delta_vs_max_over_accumulation_score'], 4)}."
    )
    sf = analysis.get("subfactor_relationships", {})
    lines.append(
        "- Subfactor decomposition (alignment/concentration/coefficient exposure) is reported in the explanatory "
        "analysis for benign-vs-inflation-prone overlap checks."
    )
    lines.append(
        f"- Spearman(delta, triple-high layer fraction): "
        f"{_fmt(sf.get('spearman_delta_vs_triple_high_layer_fraction'), 4)}."
    )
    top_threshold = None
    for row in sweep.get("threshold_grid", []):
        if row.get("threshold") == 0.35:
            top_threshold = row
            break
    if top_threshold:
        lines.append(
            f"- At threshold 0.35: activated pairs={top_threshold.get('activated_pair_count')}, "
            f"activated layers={top_threshold.get('activated_layer_count')}, "
            f"rerun activated n={top_threshold.get('rerun_activated_count')}."
        )
    lines.append("")
    lines.append("## 5) Interpretation Guardrails")
    lines.append("- This remains a bounded correlation study; not causal proof.")
    lines.append("- No taxonomy/policy replacement is implied from this pass alone.")
    lines.append("- Execution-side method escalation (e.g., SVC) should still depend on stronger replicated signal.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-gate", type=float, default=DEFAULT_OVERLAP_GATE)
    parser.add_argument("--conflict-gate", type=float, default=DEFAULT_CONFLICT_GATE)
    parser.add_argument("--n-high", type=int, default=DEFAULT_N_HIGH)
    parser.add_argument("--n-low", type=int, default=DEFAULT_N_LOW)
    parser.add_argument(
        "--target-min-pairs",
        type=int,
        default=None,
        help="Optional cohort-size lower bound. If set, n_high/n_low are auto-derived.",
    )
    parser.add_argument(
        "--target-max-pairs",
        type=int,
        default=None,
        help="Optional cohort-size upper bound. If set, n_high/n_low are auto-derived.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional filename prefix for generated follow-up artifacts.",
    )
    parser.add_argument(
        "--skip-reruns",
        action="store_true",
        help="Only run activation audit + cohort selection; skip strict naive reruns.",
    )
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    FOLLOWUP_DIR.mkdir(parents=True, exist_ok=True)

    def _out_path(filename: str) -> Path:
        base = FOLLOWUP_DIR / filename
        if not args.output_prefix:
            return base
        return base.with_name(f"{args.output_prefix}_{base.name}")

    # Ensure writable local HF cache for sandbox runs.
    hf_home = FOLLOWUP_DIR / ".hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home.as_posix())

    inventory_meta = collect_adapters()
    adapters = inventory_meta["dedup_adapters"]
    by_id = {a.adapter_id: a for a in adapters}

    activation = run_activation_audit(inventory_meta["by_family"])
    _out_path("activation_threshold_audit.json").write_text(json.dumps(activation, indent=2))
    write_activation_md(_out_path("activation_threshold_audit.md"), activation, inventory_meta)

    split = _resolve_target_pair_split(
        activation,
        overlap_gate=args.overlap_gate,
        conflict_gate=args.conflict_gate,
        default_n_high=args.n_high,
        default_n_low=args.n_low,
        target_min_pairs=args.target_min_pairs,
        target_max_pairs=args.target_max_pairs,
    )

    cohort = build_targeted_cohort(
        activation,
        overlap_gate=float(split.get("resolved_overlap_gate", args.overlap_gate)),
        conflict_gate=float(split.get("resolved_conflict_gate", args.conflict_gate)),
        n_high=int(split["n_high"]),
        n_low=int(split["n_low"]),
    )
    if split.get("used_target_window_mode"):
        cohort["target_window_resolution"] = split.get("requested_window")

    _out_path("targeted_high_overlap_cohort.json").write_text(json.dumps(cohort, indent=2))
    write_cohort_md(_out_path("targeted_high_overlap_cohort.md"), cohort)

    if args.skip_reruns:
        summary = {
            "mode": "activation_and_cohort_only",
            "audited_pairs": activation["audit_scope"]["audited_pair_count"],
            "pair_advisory_counts": activation["pair_advisory_counts"],
            "cohort_count": cohort["cohort_count"],
            "selection_parameters": cohort["selection_parameters"],
            "target_window_resolution": cohort.get("target_window_resolution"),
        }
        _out_path("deeper_diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    rerun_results_path = _out_path("strict_naive_rerun_results.json")
    rerun_payload = run_strict_naive_reruns(
        cohort,
        adapters_by_id=by_id,
        max_samples=args.max_samples,
        rerun_subdir_name=f"{args.output_prefix}_strict_naive_reruns" if args.output_prefix else "strict_naive_reruns",
        results_path=rerun_results_path,
    )
    rerun_results = rerun_payload["results"]
    rerun_results_path.write_text(json.dumps(rerun_results, indent=2))
    write_rerun_md(_out_path("strict_naive_rerun_results.md"), rerun_results)

    analysis = analyze_rerun_results(rerun_results)
    _out_path("strict_naive_explanatory_analysis.json").write_text(json.dumps(analysis, indent=2))
    write_analysis_md(_out_path("strict_naive_explanatory_analysis.md"), analysis)

    sweep = build_cutpoint_sweep(activation, rerun_results)
    _out_path("cutpoint_sweep.json").write_text(json.dumps(sweep, indent=2))
    write_cutpoint_sweep_md(_out_path("cutpoint_sweep.md"), sweep)

    write_validation_memo(_out_path("strict_naive_validation_memo.md"), activation, cohort, analysis, sweep)

    summary = {
        "mode": "full_deeper_diagnostic",
        "audited_pairs": activation["audit_scope"]["audited_pair_count"],
        "pair_advisory_counts": activation["pair_advisory_counts"],
        "cohort_count": cohort["cohort_count"],
        "selection_parameters": cohort["selection_parameters"],
        "target_window_resolution": cohort.get("target_window_resolution"),
        "rerun_ok": analysis["overall_ok_count"],
        "rerun_error": analysis["overall_error_count"],
        "high_tail_mean_delta_vs_best": analysis["group_high_tail"]["mean_delta_vs_best"],
        "low_tail_mean_delta_vs_best": analysis["group_lower_tail_matched"]["mean_delta_vs_best"],
        "spearman_delta_vs_max_oa": analysis["spearman_delta_vs_max_over_accumulation_score"],
        "cutpoint_sweep_thresholds": [r.get("threshold") for r in sweep.get("threshold_grid", [])],
    }
    _out_path("deeper_diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
