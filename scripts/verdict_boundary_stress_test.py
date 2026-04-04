#!/usr/bin/env python3
"""Verdict boundary stress test against MP-partition spectral findings.

CPU-only diagnostic audit that characterizes whether current merge verdict
boundaries separate same-task vs cross-task geometry as expected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2, chi2_contingency, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from gradience.vnext.merge import merge_audit
from gradience.vnext.merge.spectral_compat import SubspaceMetrics
from gradience.vnext.merge.verdicts import VerdictThresholds, assess_layer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mp_partition_test import (  # type: ignore
    compute_layer_svd,
    load_adapter_weights,
    mp_threshold,
    sv_weighted_alignment,
)

ADAPTER_ROOT = Path("results/adjudication_study/verified_adapters")
VERIFIED_RESULTS = Path("results/adjudication_study/verified_results/verified_adjudication_results.json")
OUT_DIR = Path("sidecar/results/verdict_boundary_stress_test")

PRESET_THRESHOLDS: dict[str, VerdictThresholds] = {
    "default": VerdictThresholds(),
    "conservative": VerdictThresholds.conservative(),
    "permissive": VerdictThresholds.permissive(),
}


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    adapter_a: str
    adapter_b: str
    relationship: str  # same_task | cross_task
    group: str


PAIR_SPECS: list[PairSpec] = [
    PairSpec(
        pair_id="sst2_r16_s42_x_sst2_r16_s123",
        adapter_a="sst2_r16_s42",
        adapter_b="sst2_r16_s123",
        relationship="same_task",
        group="same_task_sst2",
    ),
    PairSpec(
        pair_id="sst2_r16_s42_x_sst2_r16_s123_v2",
        adapter_a="sst2_r16_s42",
        adapter_b="sst2_r16_s123_v2",
        relationship="same_task",
        group="same_task_sst2",
    ),
    PairSpec(
        pair_id="qnli_r16_s42_x_qnli_r16_s123",
        adapter_a="qnli_r16_s42",
        adapter_b="qnli_r16_s123",
        relationship="same_task",
        group="same_task_qnli",
    ),
    PairSpec(
        pair_id="qnli_r16_s42_x_qnli_r16_s123_v2",
        adapter_a="qnli_r16_s42",
        adapter_b="qnli_r16_s123_v2",
        relationship="same_task",
        group="same_task_qnli",
    ),
    PairSpec(
        pair_id="sst2_r16_s42_x_qnli_r16_s42",
        adapter_a="sst2_r16_s42",
        adapter_b="qnli_r16_s42",
        relationship="cross_task",
        group="cross_task",
    ),
    PairSpec(
        pair_id="sst2_r16_s42_x_qnli_r16_s123",
        adapter_a="sst2_r16_s42",
        adapter_b="qnli_r16_s123",
        relationship="cross_task",
        group="cross_task",
    ),
    PairSpec(
        pair_id="sst2_r16_s123_x_qnli_r16_s42",
        adapter_a="sst2_r16_s123",
        adapter_b="qnli_r16_s42",
        relationship="cross_task",
        group="cross_task",
    ),
    PairSpec(
        pair_id="sst2_r16_s123_x_qnli_r16_s123",
        adapter_a="sst2_r16_s123",
        adapter_b="qnli_r16_s123",
        relationship="cross_task",
        group="cross_task",
    ),
    PairSpec(
        pair_id="sst2_r16_s42_x_sst2_r8_s42",
        adapter_a="sst2_r16_s42",
        adapter_b="sst2_r8_s42",
        relationship="same_task",
        group="cross_rank_same_task",
    ),
    PairSpec(
        pair_id="qnli_r16_s42_x_qnli_r8_s42",
        adapter_a="qnli_r16_s42",
        adapter_b="qnli_r8_s42",
        relationship="same_task",
        group="cross_rank_same_task",
    ),
]

BRANCH_NAME = {
    0: "imbalanced_low_overlap",
    1: "safe_orthogonal",
    2: "redundant_high_overlap_aligned",
    3: "conflicting_high_overlap_opposed",
    4: "imbalanced_remainder",
    5: "safe_default",
}

OBSERVABLES = [
    "mean_overlap",
    "directional_agreement",
    "frobenius_ratio",
    "magnitude_ratio",
    "effective_rank_mean",
]


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_layer_name(name: str) -> str:
    out = str(name)
    for prefix in ("base_model.model.", "base_model."):
        if out.startswith(prefix):
            out = out[len(prefix) :]
    return out


def _to_subspace_metrics(metrics_dict: dict[str, Any]) -> SubspaceMetrics:
    d = dict(metrics_dict)
    for key in (
        "principal_angle_cosines",
        "right_principal_angle_cosines",
        "effective_singular_values_a",
        "effective_singular_values_b",
    ):
        if key in d and isinstance(d[key], list):
            d[key] = tuple(d[key])
    return SubspaceMetrics(**d)


def _cohen_d(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if pooled <= 1e-12:
        return None
    return (ma - mb) / math.sqrt(pooled)


def _branch_id(metrics: SubspaceMetrics, thresholds: VerdictThresholds) -> int:
    # Mirrors the assess_layer priority tree exactly.
    if metrics.frobenius_ratio > thresholds.imbalanced_frob and metrics.mean_overlap < thresholds.high_overlap:
        return 0
    if metrics.mean_overlap < thresholds.low_overlap:
        return 1
    if metrics.mean_overlap > thresholds.high_overlap and metrics.directional_agreement > thresholds.aligned:
        return 2
    if metrics.mean_overlap > thresholds.high_overlap and metrics.directional_agreement < thresholds.conflicting:
        return 3
    if metrics.frobenius_ratio > thresholds.imbalanced_frob:
        return 4
    return 5


def _load_pair_outcomes() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(VERIFIED_RESULTS.read_text()) if VERIFIED_RESULTS.exists() else {"pairs": []}
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("pairs", []):
        pid = str(row.get("pair_id"))
        if not pid:
            continue
        if "degradation" in row:
            deg = _safe_float(row.get("degradation"), math.nan)
        else:
            ds = _safe_float(row.get("degradation_sst2"), math.nan)
            dq = _safe_float(row.get("degradation_qnli"), math.nan)
            vals = [v for v in (ds, dq) if math.isfinite(v)]
            deg = float(np.mean(vals)) if vals else math.nan
        out[pid] = {
            "pair_type": row.get("pair_type"),
            "degradation_value": deg,
            "safe_or_degraded": "safe" if str(row.get("pair_type")) == "same-task" else "degraded",
        }
    return out, payload


def _pair_outcome_lookup(outcomes: dict[str, dict[str, Any]], pair_id: str, a: str, b: str) -> dict[str, Any] | None:
    candidates = [
        pair_id,
        f"{a}_x_{b}",
        f"{b}_x_{a}",
    ]
    for c in candidates:
        if c in outcomes:
            return outcomes[c]
    return None


def _run_multiclass_logit(
    X_base: np.ndarray,
    X_full: np.ndarray,
    y: np.ndarray,
    seed: int = 7,
) -> dict[str, Any]:
    labels = np.unique(y)
    if len(labels) < 2:
        return {"status": "insufficient_classes"}

    clf_base = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1e6,
    )
    clf_full = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1e6,
    )
    clf_base.fit(X_base, y)
    clf_full.fit(X_full, y)

    p_base = clf_base.predict_proba(X_base)
    p_full = clf_full.predict_proba(X_full)
    ll_base = float(np.sum(np.log(np.clip(p_base[np.arange(len(y)), y], 1e-12, 1.0))))
    ll_full = float(np.sum(np.log(np.clip(p_full[np.arange(len(y)), y], 1e-12, 1.0))))
    lr_stat = 2.0 * (ll_full - ll_base)
    df = int((X_full.shape[1] - X_base.shape[1]) * (len(labels) - 1))
    lr_p = float(chi2.sf(lr_stat, df)) if df > 0 else None

    class_counts = Counter(int(v) for v in y.tolist())
    min_class = min(class_counts.values()) if class_counts else 0
    n_splits = min(5, min_class)
    cv = {}
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        acc_base: list[float] = []
        acc_full: list[float] = []
        lloss_base: list[float] = []
        lloss_full: list[float] = []
        for tr, te in skf.split(X_base, y):
            mb = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e6)
            mf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e6)
            mb.fit(X_base[tr], y[tr])
            mf.fit(X_full[tr], y[tr])
            pb = mb.predict_proba(X_base[te])
            pf = mf.predict_proba(X_full[te])
            yb = np.argmax(pb, axis=1)
            yf = np.argmax(pf, axis=1)
            acc_base.append(float(accuracy_score(y[te], yb)))
            acc_full.append(float(accuracy_score(y[te], yf)))
            lloss_base.append(float(log_loss(y[te], pb, labels=list(range(len(labels))))))
            lloss_full.append(float(log_loss(y[te], pf, labels=list(range(len(labels))))))
        cv = {
            "n_splits": n_splits,
            "base_accuracy_mean": float(np.mean(acc_base)),
            "full_accuracy_mean": float(np.mean(acc_full)),
            "base_logloss_mean": float(np.mean(lloss_base)),
            "full_logloss_mean": float(np.mean(lloss_full)),
        }
    else:
        cv = {"n_splits": n_splits}

    return {
        "status": "ok",
        "n_samples": int(len(y)),
        "n_classes": int(len(labels)),
        "train_loglik_base": ll_base,
        "train_loglik_full": ll_full,
        "lr_statistic": float(lr_stat),
        "lr_df": df,
        "lr_p_value": lr_p,
        "cv": cv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--energy-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    available_adapters = {p.name: p for p in ADAPTER_ROOT.iterdir() if p.is_dir()} if ADAPTER_ROOT.exists() else {}
    missing_pairs: list[dict[str, Any]] = []
    resolved_pairs: list[PairSpec] = []
    for p in PAIR_SPECS:
        if p.adapter_a not in available_adapters or p.adapter_b not in available_adapters:
            missing_pairs.append(
                {
                    "pair_id": p.pair_id,
                    "adapter_a": p.adapter_a,
                    "adapter_b": p.adapter_b,
                    "reason": "missing_adapter_dir",
                    "adapter_a_exists": p.adapter_a in available_adapters,
                    "adapter_b_exists": p.adapter_b in available_adapters,
                }
            )
            continue
        resolved_pairs.append(p)

    if not resolved_pairs:
        raise SystemExit("No valid pairs resolved; cannot run stress test.")

    # Load verified outcomes (for optional merge-outcome linkage)
    pair_outcomes, outcomes_payload = _load_pair_outcomes()

    # Preload adapter weights + SVD cache for MP partition metrics.
    unique_adapters = sorted({p.adapter_a for p in resolved_pairs} | {p.adapter_b for p in resolved_pairs})
    weight_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    svd_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    threshold_cache: dict[str, dict[str, tuple[float, int, float]]] = {}  # adapter -> layer -> (tau, k, energy_frac_high)

    for name in unique_adapters:
        w = load_adapter_weights(available_adapters[name])
        weight_cache[name] = w
        svd_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        thr_map: dict[str, tuple[float, int, float]] = {}
        for layer, layer_data in w.items():
            A = layer_data["A"]
            B = layer_data["B"]
            scaling = float(layer_data["scaling"])
            U, S, Vt = compute_layer_svd(A, B, scaling)
            svd_map[layer] = (U, S, Vt)
            tau, k = mp_threshold(S, shape=(B.shape[0], A.shape[1]))
            total_e = float(np.sum(S**2))
            high_e = float(np.sum(S[:k] ** 2)) if k > 0 else 0.0
            frac = (high_e / total_e) if total_e > 1e-12 else 0.0
            thr_map[layer] = (float(tau), int(k), float(frac))
        svd_cache[name] = svd_map
        threshold_cache[name] = thr_map

    profiles: list[dict[str, Any]] = []
    pair_level_summary: list[dict[str, Any]] = []
    preset_aggregate: dict[str, dict[str, Counter[str]]] = {
        preset: {"same_task": Counter(), "cross_task": Counter()} for preset in PRESET_THRESHOLDS
    }

    for pair in resolved_pairs:
        a_dir = available_adapters[pair.adapter_a]
        b_dir = available_adapters[pair.adapter_b]
        report = merge_audit(
            str(a_dir),
            str(b_dir),
            energy_threshold=float(args.energy_threshold),
            thresholds=PRESET_THRESHOLDS["default"],
            verbose=False,
        )
        layers = list(report.layer_verdicts or [])
        if not layers:
            continue

        pair_outcome = _pair_outcome_lookup(pair_outcomes, pair.pair_id, pair.adapter_a, pair.adapter_b)
        verdict_counts_per_preset: dict[str, Counter[str]] = {k: Counter() for k in PRESET_THRESHOLDS}
        branch_counts_per_preset: dict[str, Counter[str]] = {k: Counter() for k in PRESET_THRESHOLDS}

        for layer in layers:
            layer_name = str(layer.get("layer_name"))
            layer_norm = _normalize_layer_name(layer_name)
            module_type = str(layer.get("module_type"))
            metrics = _to_subspace_metrics(layer.get("metrics", {}))

            # MP partition metrics for this exact layer
            mp_bundle: dict[str, Any] = {
                "mp_available": False,
                "high_sv_alignment": math.nan,
                "low_sv_alignment": math.nan,
                "full_sv_alignment": math.nan,
                "hl_ratio": math.nan,
                "k_a": 0,
                "k_b": 0,
                "tau_a": math.nan,
                "tau_b": math.nan,
                "high_energy_frac_a": math.nan,
                "high_energy_frac_b": math.nan,
            }
            if (
                layer_norm in svd_cache[pair.adapter_a]
                and layer_norm in svd_cache[pair.adapter_b]
                and layer_norm in threshold_cache[pair.adapter_a]
                and layer_norm in threshold_cache[pair.adapter_b]
            ):
                Ua, Sa, _ = svd_cache[pair.adapter_a][layer_norm]
                Ub, Sb, _ = svd_cache[pair.adapter_b][layer_norm]
                tau_a, k_a, frac_a = threshold_cache[pair.adapter_a][layer_norm]
                tau_b, k_b, frac_b = threshold_cache[pair.adapter_b][layer_norm]

                high = sv_weighted_alignment(Ua[:, :k_a], Sa[:k_a], Ub[:, :k_b], Sb[:k_b])
                low = sv_weighted_alignment(Ua[:, k_a:], Sa[k_a:], Ub[:, k_b:], Sb[k_b:])
                full = sv_weighted_alignment(Ua, Sa, Ub, Sb)
                ratio = (high / low) if low > 1e-12 else math.inf

                mp_bundle = {
                    "mp_available": True,
                    "high_sv_alignment": float(high),
                    "low_sv_alignment": float(low),
                    "full_sv_alignment": float(full),
                    "hl_ratio": float(ratio),
                    "k_a": int(k_a),
                    "k_b": int(k_b),
                    "tau_a": float(tau_a),
                    "tau_b": float(tau_b),
                    "high_energy_frac_a": float(frac_a),
                    "high_energy_frac_b": float(frac_b),
                }

            for preset_name, thresholds in PRESET_THRESHOLDS.items():
                lv = assess_layer(layer_name, module_type, metrics, thresholds=thresholds)
                bid = _branch_id(metrics, thresholds)
                verdict = lv.verdict.value
                verdict_counts_per_preset[preset_name][verdict] += 1
                branch_counts_per_preset[preset_name][BRANCH_NAME[bid]] += 1
                preset_aggregate[preset_name][pair.relationship][verdict] += 1

                profiles.append(
                    {
                        "pair_id": pair.pair_id,
                        "adapter_a": pair.adapter_a,
                        "adapter_b": pair.adapter_b,
                        "relationship": pair.relationship,
                        "pair_group": pair.group,
                        "layer_name": layer_name,
                        "layer_name_normalized": layer_norm,
                        "module_type": module_type,
                        "preset": preset_name,
                        "verdict": verdict,
                        "confidence": float(lv.confidence),
                        "branch_id": bid,
                        "branch_name": BRANCH_NAME[bid],
                        "mean_overlap": float(metrics.mean_overlap),
                        "directional_agreement": float(metrics.directional_agreement),
                        "frobenius_ratio": float(metrics.frobenius_ratio),
                        "magnitude_ratio": float(metrics.magnitude_ratio),
                        "effective_rank_a": int(metrics.effective_rank_a),
                        "effective_rank_b": int(metrics.effective_rank_b),
                        "effective_rank_mean": float(0.5 * (metrics.effective_rank_a + metrics.effective_rank_b)),
                        "stable_rank_a": float(metrics.stable_rank_a),
                        "stable_rank_b": float(metrics.stable_rank_b),
                        "pair_outcome": pair_outcome["safe_or_degraded"] if pair_outcome else "unknown",
                        "pair_degradation_value": (
                            float(pair_outcome["degradation_value"])
                            if pair_outcome and math.isfinite(_safe_float(pair_outcome.get("degradation_value")))
                            else math.nan
                        ),
                        **mp_bundle,
                    }
                )

        pair_level_summary.append(
            {
                "pair_id": pair.pair_id,
                "relationship": pair.relationship,
                "pair_group": pair.group,
                "adapter_a": pair.adapter_a,
                "adapter_b": pair.adapter_b,
                "pair_outcome": pair_outcome["safe_or_degraded"] if pair_outcome else "unknown",
                "pair_degradation_value": (
                    float(pair_outcome["degradation_value"])
                    if pair_outcome and math.isfinite(_safe_float(pair_outcome.get("degradation_value")))
                    else None
                ),
                "verdict_counts_by_preset": {k: dict(v) for k, v in verdict_counts_per_preset.items()},
                "branch_counts_by_preset": {k: dict(v) for k, v in branch_counts_per_preset.items()},
            }
        )

    # Analysis uses default preset layer rows.
    default_rows = [r for r in profiles if r["preset"] == "default"]
    same_rows = [r for r in default_rows if r["relationship"] == "same_task"]
    cross_rows = [r for r in default_rows if r["relationship"] == "cross_task"]

    # Analysis 1: observable distributions and threshold positions.
    threshold_map = {
        "mean_overlap": [PRESET_THRESHOLDS["default"].low_overlap, PRESET_THRESHOLDS["default"].high_overlap],
        "directional_agreement": [PRESET_THRESHOLDS["default"].conflicting, PRESET_THRESHOLDS["default"].aligned],
        "frobenius_ratio": [PRESET_THRESHOLDS["default"].imbalanced_frob],
        "magnitude_ratio": [PRESET_THRESHOLDS["default"].imbalanced],
    }
    dist_analysis: dict[str, Any] = {}
    for obs in OBSERVABLES:
        s = [float(r[obs]) for r in same_rows if math.isfinite(_safe_float(r.get(obs)))]
        c = [float(r[obs]) for r in cross_rows if math.isfinite(_safe_float(r.get(obs)))]
        ks = ks_2samp(s, c) if s and c else None
        dist_analysis[obs] = {
            "same_task": {
                "n": len(s),
                "mean": float(np.mean(s)) if s else None,
                "std": float(np.std(s)) if s else None,
                "min": float(np.min(s)) if s else None,
                "max": float(np.max(s)) if s else None,
            },
            "cross_task": {
                "n": len(c),
                "mean": float(np.mean(c)) if c else None,
                "std": float(np.std(c)) if c else None,
                "min": float(np.min(c)) if c else None,
                "max": float(np.max(c)) if c else None,
            },
            "separation": {
                "cohen_d": _cohen_d(s, c),
                "ks_statistic": float(ks.statistic) if ks is not None else None,
                "ks_p_value": float(ks.pvalue) if ks is not None else None,
            },
            "threshold_positions": [],
        }
        for thr in threshold_map.get(obs, []):
            pct_same = 100.0 * (sum(1 for v in s if v <= thr) / len(s)) if s else None
            pct_cross = 100.0 * (sum(1 for v in c if v <= thr) / len(c)) if c else None
            dist_analysis[obs]["threshold_positions"].append(
                {
                    "threshold": float(thr),
                    "pct_same_below_or_equal": pct_same,
                    "pct_cross_below_or_equal": pct_cross,
                }
            )

    # Analysis 2: verdict distribution by relationship.
    verdict_order = ["safe", "redundant", "conflicting", "imbalanced"]
    contingency = np.array(
        [
            [sum(1 for r in same_rows if r["verdict"] == v) for v in verdict_order],
            [sum(1 for r in cross_rows if r["verdict"] == v) for v in verdict_order],
        ]
    )
    chi2_res = chi2_contingency(contingency) if contingency.sum() > 0 else None
    verdict_dist = {
        "same_task": {v: int(contingency[0, i]) for i, v in enumerate(verdict_order)},
        "cross_task": {v: int(contingency[1, i]) for i, v in enumerate(verdict_order)},
        "chi2": (
            {
                "statistic": float(chi2_res[0]),
                "p_value": float(chi2_res[1]),
                "dof": int(chi2_res[2]),
            }
            if chi2_res is not None
            else None
        ),
    }

    # Analysis 3: Branch-5 decomposition.
    b5 = [r for r in default_rows if int(r["branch_id"]) == 5]
    b5_same = [r for r in b5 if r["relationship"] == "same_task"]
    b5_cross = [r for r in b5 if r["relationship"] == "cross_task"]
    branch5_metrics = ["high_sv_alignment", "low_sv_alignment", "hl_ratio", "full_sv_alignment", "mean_overlap"]
    b5_analysis: dict[str, Any] = {
        "n_total": len(b5),
        "n_same_task": len(b5_same),
        "n_cross_task": len(b5_cross),
        "metrics": {},
    }
    for m in branch5_metrics:
        s = [float(r[m]) for r in b5_same if math.isfinite(_safe_float(r.get(m)))]
        c = [float(r[m]) for r in b5_cross if math.isfinite(_safe_float(r.get(m)))]
        ks = ks_2samp(s, c) if s and c else None
        b5_analysis["metrics"][m] = {
            "same_task_mean": float(np.mean(s)) if s else None,
            "cross_task_mean": float(np.mean(c)) if c else None,
            "cohen_d": _cohen_d(s, c),
            "ks_statistic": float(ks.statistic) if ks is not None else None,
            "ks_p_value": float(ks.pvalue) if ks is not None else None,
        }

    # Analysis 4: Threshold sensitivity and optimal cutpoints.
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in profiles:
        by_key[(r["pair_id"], r["layer_name"])][r["preset"]] = r

    changed_counts = {
        "default_to_conservative": {"same_task": 0, "cross_task": 0, "total": 0},
        "default_to_permissive": {"same_task": 0, "cross_task": 0, "total": 0},
    }
    transition_counts = {
        "default_to_conservative": Counter(),
        "default_to_permissive": Counter(),
    }
    for key, v in by_key.items():
        if not {"default", "conservative", "permissive"}.issubset(set(v)):
            continue
        d = v["default"]
        c = v["conservative"]
        p = v["permissive"]
        rel = d["relationship"]
        if d["verdict"] != c["verdict"]:
            changed_counts["default_to_conservative"]["total"] += 1
            changed_counts["default_to_conservative"][rel] += 1
            transition_counts["default_to_conservative"][f"{d['verdict']}->{c['verdict']}"] += 1
        if d["verdict"] != p["verdict"]:
            changed_counts["default_to_permissive"]["total"] += 1
            changed_counts["default_to_permissive"][rel] += 1
            transition_counts["default_to_permissive"][f"{d['verdict']}->{p['verdict']}"] += 1

    def _best_cutpoint(obs: str, direction: str) -> dict[str, Any]:
        vals = [float(r[obs]) for r in default_rows if math.isfinite(_safe_float(r.get(obs)))]
        y = [1 if r["relationship"] == "same_task" else 0 for r in default_rows if math.isfinite(_safe_float(r.get(obs)))]
        if len(vals) < 5 or len(set(y)) < 2:
            return {"status": "insufficient"}
        cands = sorted(set(vals))
        best = None
        for t in cands:
            if direction == "ge":
                pred = [1 if x >= t else 0 for x in vals]
            else:
                pred = [1 if x <= t else 0 for x in vals]
            tp = sum(1 for p, yy in zip(pred, y) if p == 1 and yy == 1)
            tn = sum(1 for p, yy in zip(pred, y) if p == 0 and yy == 0)
            fp = sum(1 for p, yy in zip(pred, y) if p == 1 and yy == 0)
            fn = sum(1 for p, yy in zip(pred, y) if p == 0 and yy == 1)
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0
            bal_acc = 0.5 * (tpr + tnr)
            item = {"threshold": float(t), "balanced_accuracy": float(bal_acc), "tpr": float(tpr), "tnr": float(tnr)}
            if best is None or item["balanced_accuracy"] > best["balanced_accuracy"]:
                best = item
        return {"status": "ok", "best": best}

    cutpoints = {
        "mean_overlap_ge": _best_cutpoint("mean_overlap", "ge"),
        "directional_agreement_ge": _best_cutpoint("directional_agreement", "ge"),
        "frobenius_ratio_le": _best_cutpoint("frobenius_ratio", "le"),
        "magnitude_ratio_le": _best_cutpoint("magnitude_ratio", "le"),
    }
    threshold_sensitivity = {
        "changed_counts": changed_counts,
        "transitions": {k: dict(v) for k, v in transition_counts.items()},
        "optimal_cutpoints_for_same_task_separation": cutpoints,
    }

    # Analysis 5: partition metrics as verdict predictors (multiclass logistic).
    valid_for_logit = [
        r
        for r in default_rows
        if math.isfinite(_safe_float(r.get("high_sv_alignment")))
        and math.isfinite(_safe_float(r.get("hl_ratio")))
        and math.isfinite(_safe_float(r.get("full_sv_alignment")))
    ]
    verdict_map = {v: i for i, v in enumerate(sorted({r["verdict"] for r in valid_for_logit}))}
    X_base = np.array(
        [
            [
                float(r["mean_overlap"]),
                float(r["directional_agreement"]),
                float(r["frobenius_ratio"]),
                float(r["magnitude_ratio"]),
                float(r["effective_rank_mean"]),
            ]
            for r in valid_for_logit
        ],
        dtype=np.float64,
    )
    X_part = np.array(
        [
            [
                float(r["high_sv_alignment"]),
                float(r["low_sv_alignment"]),
                float(r["full_sv_alignment"]),
                float(r["hl_ratio"]) if math.isfinite(_safe_float(r.get("hl_ratio"))) else 0.0,
                float(0.5 * (_safe_float(r.get("high_energy_frac_a"), 0.0) + _safe_float(r.get("high_energy_frac_b"), 0.0)),
                      ),
            ]
            for r in valid_for_logit
        ],
        dtype=np.float64,
    )
    y_verdict = np.array([verdict_map[r["verdict"]] for r in valid_for_logit], dtype=np.int64)
    logit_analysis = _run_multiclass_logit(X_base, np.hstack([X_base, X_part]), y_verdict, seed=int(args.seed))
    logit_analysis["verdict_class_map"] = verdict_map
    logit_analysis["n_valid_rows"] = int(len(valid_for_logit))

    # Analysis 6: weighted vs unweighted overlap.
    # Layer-level verdict discrimination.
    y_redundant = np.array([1 if r["verdict"] == "redundant" else 0 for r in valid_for_logit], dtype=np.int64)
    overlap_unw = np.array([float(r["mean_overlap"]) for r in valid_for_logit], dtype=np.float64)
    overlap_w = np.array([float(r["full_sv_alignment"]) for r in valid_for_logit], dtype=np.float64)
    layer_auc = {}
    if len(np.unique(y_redundant)) >= 2:
        layer_auc = {
            "auc_redundant_unweighted_mean_overlap": float(roc_auc_score(y_redundant, overlap_unw)),
            "auc_redundant_sv_weighted_full_alignment": float(roc_auc_score(y_redundant, overlap_w)),
        }
    else:
        layer_auc = {
            "auc_redundant_unweighted_mean_overlap": None,
            "auc_redundant_sv_weighted_full_alignment": None,
        }

    # Pair-level outcome linkage where known.
    pair_rows_known = [r for r in pair_level_summary if r.get("pair_outcome") in {"safe", "degraded"}]
    pair_metric_rows: list[dict[str, Any]] = []
    for p in pair_rows_known:
        pid = p["pair_id"]
        rows_pid = [r for r in default_rows if r["pair_id"] == pid and math.isfinite(_safe_float(r.get("full_sv_alignment")))]
        if not rows_pid:
            continue
        pair_metric_rows.append(
            {
                "pair_id": pid,
                "pair_outcome": p["pair_outcome"],
                "pair_degradation_value": p.get("pair_degradation_value"),
                "mean_overlap_unweighted": float(np.mean([r["mean_overlap"] for r in rows_pid])),
                "mean_overlap_weighted": float(np.mean([r["full_sv_alignment"] for r in rows_pid])),
                "mean_hl_ratio": float(np.mean([r["hl_ratio"] for r in rows_pid if math.isfinite(_safe_float(r.get("hl_ratio")))])),
            }
        )
    pair_outcome_link = {}
    if pair_metric_rows and len({r["pair_outcome"] for r in pair_metric_rows}) >= 2:
        y_deg = np.array([1 if r["pair_outcome"] == "degraded" else 0 for r in pair_metric_rows], dtype=np.int64)
        # degraded is predicted by low alignment -> use negative alignment scores
        x_unw = np.array([-float(r["mean_overlap_unweighted"]) for r in pair_metric_rows], dtype=np.float64)
        x_w = np.array([-float(r["mean_overlap_weighted"]) for r in pair_metric_rows], dtype=np.float64)
        pair_outcome_link = {
            "n_pairs_with_known_outcome": int(len(pair_metric_rows)),
            "auc_degraded_from_negative_unweighted_mean_overlap": float(roc_auc_score(y_deg, x_unw)),
            "auc_degraded_from_negative_sv_weighted_alignment": float(roc_auc_score(y_deg, x_w)),
        }
    else:
        pair_outcome_link = {"n_pairs_with_known_outcome": int(len(pair_metric_rows))}

    weighted_vs_unweighted = {
        "layer_level_verdict_discrimination": layer_auc,
        "pair_level_outcome_linkage": pair_outcome_link,
    }

    # Decision synthesis
    cross_safe_hi = [
        r
        for r in default_rows
        if r["relationship"] == "cross_task" and r["verdict"] == "safe" and float(r["confidence"]) >= 0.7
    ]
    same_bad = [r for r in default_rows if r["relationship"] == "same_task" and r["verdict"] in {"conflicting", "imbalanced"}]
    cross_count = max(1, len(cross_rows))
    same_count = max(1, len(same_rows))
    cross_safe_hi_rate = len(cross_safe_hi) / cross_count
    same_bad_rate = len(same_bad) / same_count

    b5_hl_p = _safe_float(
        ((b5_analysis.get("metrics") or {}).get("hl_ratio") or {}).get("ks_p_value"),
        math.nan,
    )
    changed_total = threshold_sensitivity["changed_counts"]["default_to_conservative"]["total"]
    total_layers = len(default_rows)
    changed_rate = changed_total / max(1, total_layers)

    if cross_safe_hi_rate > 0.25 or same_bad_rate > 0.25:
        decision = "Outcome C — boundaries likely miscalibrated"
    elif math.isfinite(b5_hl_p) and b5_hl_p < 0.05:
        decision = "Outcome B — boundaries adequate but sharpenable"
    elif changed_rate > 0.10:
        decision = "Outcome B — boundaries adequate but sharpenable"
    else:
        decision = "Outcome A — boundaries broadly well-calibrated"

    analysis = {
        "generated_at_utc": now,
        "input": {
            "adapter_root": str(ADAPTER_ROOT),
            "verified_outcomes_json": str(VERIFIED_RESULTS),
            "energy_threshold": float(args.energy_threshold),
            "preset_thresholds": {k: v.to_dict() for k, v in PRESET_THRESHOLDS.items()},
            "requested_pairs": [p.__dict__ for p in PAIR_SPECS],
            "resolved_pair_count": len(resolved_pairs),
            "missing_pairs": missing_pairs,
        },
        "summary_counts": {
            "n_profiles": len(profiles),
            "n_default_layers": len(default_rows),
            "n_same_task_layers": len(same_rows),
            "n_cross_task_layers": len(cross_rows),
            "preset_aggregate_verdicts": {
                preset: {rel: dict(cnt) for rel, cnt in rel_map.items()}
                for preset, rel_map in preset_aggregate.items()
            },
        },
        "analysis_1_distributions": dist_analysis,
        "analysis_2_verdict_distribution": verdict_dist,
        "analysis_3_branch5_decomposition": b5_analysis,
        "analysis_4_threshold_sensitivity": threshold_sensitivity,
        "analysis_5_partition_predicts_verdict": logit_analysis,
        "analysis_6_weighted_vs_unweighted_overlap": weighted_vs_unweighted,
        "decision": {
            "outcome": decision,
            "cross_task_safe_high_conf_rate": cross_safe_hi_rate,
            "same_task_conflicting_or_imbalanced_rate": same_bad_rate,
            "branch5_hl_ratio_ks_p_value": (None if not math.isfinite(b5_hl_p) else b5_hl_p),
            "default_to_conservative_change_rate": changed_rate,
        },
    }

    profiles_payload = {
        "generated_at_utc": now,
        "resolved_pairs": [p.__dict__ for p in resolved_pairs],
        "missing_pairs": missing_pairs,
        "pair_level_summary": pair_level_summary,
        "profiles": profiles,
    }

    profiles_json = args.out_dir / "profiles.json"
    analysis_json = args.out_dir / "analysis.json"
    report_txt = args.out_dir / "report.txt"

    profiles_json.write_text(json.dumps(profiles_payload, indent=2))
    analysis_json.write_text(json.dumps(analysis, indent=2))

    # Human-readable report.
    def _fmt(x: Any, d: int = 4) -> str:
        if x is None:
            return "n/a"
        try:
            v = float(x)
        except Exception:
            return str(x)
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.{d}f}"

    lines: list[str] = []
    lines.append("VERDICT BOUNDARY STRESS TEST")
    lines.append("=" * 72)
    lines.append(f"Generated: {now}")
    lines.append(f"Resolved pairs: {len(resolved_pairs)} / requested {len(PAIR_SPECS)}")
    if missing_pairs:
        lines.append(f"Missing pairs: {len(missing_pairs)}")
        for m in missing_pairs:
            lines.append(
                f"  - {m['pair_id']} (a_exists={m['adapter_a_exists']}, b_exists={m['adapter_b_exists']})"
            )
    lines.append("")
    lines.append("Analysis 1 — Observable separation (same-task vs cross-task)")
    for obs in OBSERVABLES:
        row = dist_analysis[obs]
        sep = row["separation"]
        lines.append(
            f"  {obs:>22}: same_mean={_fmt(row['same_task']['mean'])} "
            f"cross_mean={_fmt(row['cross_task']['mean'])} "
            f"d={_fmt(sep['cohen_d'])} ks_p={_fmt(sep['ks_p_value'])}"
        )
    lines.append("")
    lines.append("Analysis 2 — Verdict distribution")
    lines.append(f"  same-task: {verdict_dist['same_task']}")
    lines.append(f"  cross-task: {verdict_dist['cross_task']}")
    if verdict_dist["chi2"] is not None:
        lines.append(
            "  chi2="
            f"{_fmt(verdict_dist['chi2']['statistic'])}, p={_fmt(verdict_dist['chi2']['p_value'])}, "
            f"dof={verdict_dist['chi2']['dof']}"
        )
    lines.append("")
    lines.append("Analysis 3 — Branch-5 decomposition")
    lines.append(
        f"  n_total={b5_analysis['n_total']}, n_same={b5_analysis['n_same_task']}, n_cross={b5_analysis['n_cross_task']}"
    )
    for m in ("high_sv_alignment", "hl_ratio", "full_sv_alignment", "mean_overlap"):
        mr = b5_analysis["metrics"].get(m, {})
        lines.append(
            f"  {m:>22}: same_mean={_fmt(mr.get('same_task_mean'))} "
            f"cross_mean={_fmt(mr.get('cross_task_mean'))} ks_p={_fmt(mr.get('ks_p_value'))}"
        )
    lines.append("")
    lines.append("Analysis 4 — Threshold sensitivity")
    lines.append(f"  default->conservative changes: {changed_counts['default_to_conservative']}")
    lines.append(f"  default->permissive changes: {changed_counts['default_to_permissive']}")
    lines.append("")
    lines.append("Analysis 5 — Partition metrics as verdict predictors")
    lines.append(f"  status={logit_analysis.get('status')}, n={logit_analysis.get('n_valid_rows')}")
    if logit_analysis.get("status") == "ok":
        lines.append(
            f"  LR(base vs base+partition): stat={_fmt(logit_analysis.get('lr_statistic'))}, "
            f"df={logit_analysis.get('lr_df')}, p={_fmt(logit_analysis.get('lr_p_value'))}"
        )
        cv = logit_analysis.get("cv", {})
        if cv.get("n_splits", 0) >= 2:
            lines.append(
                f"  CV accuracy base={_fmt(cv.get('base_accuracy_mean'))}, full={_fmt(cv.get('full_accuracy_mean'))}; "
                f"logloss base={_fmt(cv.get('base_logloss_mean'))}, full={_fmt(cv.get('full_logloss_mean'))}"
            )
    lines.append("")
    lines.append("Analysis 6 — Weighted vs unweighted overlap")
    lv = weighted_vs_unweighted["layer_level_verdict_discrimination"]
    lines.append(
        "  AUC redundant (unweighted mean_overlap vs weighted full_alignment): "
        f"{_fmt(lv.get('auc_redundant_unweighted_mean_overlap'))} vs "
        f"{_fmt(lv.get('auc_redundant_sv_weighted_full_alignment'))}"
    )
    pv = weighted_vs_unweighted["pair_level_outcome_linkage"]
    lines.append(
        "  Pair-level degraded AUC (negative alignment): "
        f"unweighted={_fmt(pv.get('auc_degraded_from_negative_unweighted_mean_overlap'))}, "
        f"weighted={_fmt(pv.get('auc_degraded_from_negative_sv_weighted_alignment'))}"
    )
    lines.append("")
    lines.append("Decision")
    lines.append(f"  {analysis['decision']['outcome']}")
    lines.append(
        "  diagnostics: "
        f"cross_safe_hi_rate={_fmt(analysis['decision']['cross_task_safe_high_conf_rate'])}, "
        f"same_conflict_or_imbalanced_rate={_fmt(analysis['decision']['same_task_conflicting_or_imbalanced_rate'])}, "
        f"branch5_hl_p={_fmt(analysis['decision']['branch5_hl_ratio_ks_p_value'])}, "
        f"change_rate={_fmt(analysis['decision']['default_to_conservative_change_rate'])}"
    )
    lines.append("")
    lines.append(f"Outputs: {profiles_json}, {analysis_json}, {report_txt}")
    report_txt.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
