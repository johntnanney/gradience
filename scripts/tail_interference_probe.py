#!/usr/bin/env python3
"""
N128: Tail Interference Probe

Does Gradience's energy-weighted spectral compatibility miss interference
that is localized in the low-SV (below Marchenko-Pastur threshold) band?

Motivation (from THEORY.md S7.2):
    Medina & Sorensen (2025) show that fine-tuning operates primarily in
    the low-SV spectral tail.  If adapter interference is concentrated in
    these low-energy directions, Gradience's energy-weighted overlap metric
    would under-weight it, potentially producing false negatives (pair
    classified SAFE but merge outcome is degraded).

Corpus:
    - Same-task seed pairs from cross_task_subtype_study_01 (distilbert)
      and task_pair_severity_generalization_study_01 (roberta)
    - Cross-task pairs formed from the same adapters (contrast condition)
    - Merge outcomes from sidecar/results/attractor_mapping/attractor_panel_table.json

Method:
    1. Load adapter weights, compute SVD of each layer's BA product
    2. Compute Gavish-Donoho optimal hard threshold (MP threshold)
    3. Partition singular vectors into high-SV and low-SV bands
    4. Compute band-split overlap via principal angle cosines
    5. Compare energy-weighted (Gradience-style) vs unweighted (tail-sensitive)
    6. Flag false-negative candidates: classified safe + degraded outcome +
       low-band conflict >= 0.5

Reference: sidecar/notes/N128_tail_interference_probe.md
Depends on: N127 (MP partition test) for method validation
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.linalg import qr, svd
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Adapter base paths (from artifact_inventory.json)
DISTILBERT_BASE = REPO_ROOT / "results" / "cross_task_subtype_study_01" / "sources"
ROBERTA_BASE = REPO_ROOT / "results" / "task_pair_severity_generalization_study_01" / "roberta" / "sources"

# Output
OUTPUT_DIR = REPO_ROOT / "sidecar" / "results" / "N128_tail_interference"
RESULTS_JSON = OUTPUT_DIR / "results.json"

# Thresholds
FALSE_NEGATIVE_LOW_BAND_THRESHOLD = 0.5   # Low-band conflict floor for FN flag
MERGE_DELTA_DEGRADED_THRESHOLD = 0.015    # merge_delta above this = "degraded"
MERGE_DELTA_SAFE_THRESHOLD = 0.005        # merge_delta at/below this = "safe"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AdapterInfo:
    adapter_id: str
    backbone: str
    task: str
    seed: str
    path: Path


@dataclass
class PairSpec:
    pair_id: str
    adapter_a: AdapterInfo
    adapter_b: AdapterInfo
    task_relation: str          # "same_task" or "cross_task"
    merge_delta: float | None   # Known merge outcome, if available
    attractor_class: str | None  # "single_attractor" / "multi_attractor"


@dataclass
class BandOverlap:
    """Overlap metrics for one spectral band of one layer."""
    band: str                # "high_sv", "low_sv", "full"
    n_dims_a: int
    n_dims_b: int
    mean_cos: float          # Mean principal angle cosine (unweighted)
    sv_weighted: float       # SV-weighted alignment (Gradience-style)
    energy_frac_a: float
    energy_frac_b: float


@dataclass
class LayerOverlap:
    """Band-split overlap for one layer of one pair."""
    layer_name: str
    module_type: str
    shape: tuple[int, int]
    nominal_rank: int
    mp_tau_a: float
    mp_tau_b: float
    mp_k_a: int
    mp_k_b: int
    bands: dict[str, BandOverlap] = field(default_factory=dict)


@dataclass
class PairResult:
    """Aggregate result for one adapter pair."""
    pair_id: str
    task_relation: str
    backbone: str
    task_a: str
    task_b: str
    merge_delta: float | None
    attractor_class: str | None
    n_layers: int
    # Aggregate band metrics (mean across layers)
    full_mean_cos: float = 0.0
    full_sv_weighted: float = 0.0
    high_mean_cos: float = 0.0
    high_sv_weighted: float = 0.0
    low_mean_cos: float = 0.0
    low_sv_weighted: float = 0.0
    # Derived
    tail_excess: float = 0.0       # low_mean_cos - high_mean_cos (positive = tail has MORE overlap)
    energy_masking: float = 0.0    # full_sv_weighted - full_mean_cos (positive = energy weighting inflates)
    # False-negative flag
    is_false_negative_candidate: bool = False
    fn_reason: str = ""
    # Per-layer detail
    layers: list[LayerOverlap] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adapter loading (from mp_partition_test.py)
# ---------------------------------------------------------------------------

def load_adapter_weights(adapter_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load LoRA A and B matrices from a PEFT adapter directory.

    Returns dict mapping layer_key -> {"A": ndarray, "B": ndarray, "scaling": float}.
    """
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"

    if not safetensors_path.exists():
        raise FileNotFoundError(f"No safetensors at {safetensors_path}")

    with open(config_path) as f:
        config = json.load(f)

    alpha = config.get("lora_alpha", config.get("r", 8))
    r = config.get("r", 8)
    scaling = alpha / r

    weights = {}
    with safe_open(str(safetensors_path), framework="numpy") as f:
        keys = list(f.keys())
        a_keys = sorted(k for k in keys if "lora_A" in k)
        b_keys = sorted(k for k in keys if "lora_B" in k)

        for a_key in a_keys:
            layer_key = a_key.replace(".lora_A.weight", "").replace(
                ".lora_A.default.weight", ""
            )
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in keys:
                for bk in b_keys:
                    bk_layer = bk.replace(".lora_B.weight", "").replace(
                        ".lora_B.default.weight", ""
                    )
                    if bk_layer == layer_key:
                        b_key = bk
                        break

            if b_key not in keys:
                continue

            A = f.get_tensor(a_key).astype(np.float64)
            B = f.get_tensor(b_key).astype(np.float64)

            short_key = layer_key
            for prefix in ["base_model.model.", "base_model."]:
                if short_key.startswith(prefix):
                    short_key = short_key[len(prefix):]

            weights[short_key] = {"A": A, "B": B, "scaling": scaling}

    return weights


# ---------------------------------------------------------------------------
# SVD and MP threshold (from mp_partition_test.py)
# ---------------------------------------------------------------------------

def compute_layer_svd(
    A: np.ndarray, B: np.ndarray, scaling: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD of scaling * B @ A via QR decomposition.

    Returns (U, S, Vt) where:
        U: (d_out, r) left singular vectors
        S: (r,) singular values, descending
        Vt: (r, d_in) right singular vectors
    """
    Qb, Rb = qr(B, mode="reduced")
    Qa, Ra = qr(A.T, mode="reduced")
    M = scaling * (Rb @ Ra.T)
    U_small, S, Vt_small = svd(M, full_matrices=False)
    U = Qb @ U_small
    Vt = Vt_small @ Qa.T
    order = np.argsort(-S)
    return U[:, order], S[order], Vt[order, :]


def mp_threshold(S: np.ndarray, shape: tuple[int, int]) -> tuple[float, int]:
    """Compute Gavish-Donoho optimal hard threshold.

    Returns (tau, k) where tau is the threshold and k is the number of
    singular values above it.
    """
    d_out, d_in = shape
    beta = min(d_out, d_in) / max(d_out, d_in)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    median_sv = float(np.median(S))
    if median_sv < 1e-12:
        return 0.0, len(S)
    tau = omega * median_sv
    k = int(np.sum(S > tau))
    return tau, max(k, 1)


def principal_angle_cosines(U_a: np.ndarray, U_b: np.ndarray) -> np.ndarray:
    """Compute principal angle cosines between column spaces."""
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return np.array([])
    cross = U_a.T @ U_b
    cosines = svd(cross, compute_uv=False)
    return np.clip(cosines, 0.0, 1.0)


def sv_weighted_alignment(
    U_a: np.ndarray, S_a: np.ndarray,
    U_b: np.ndarray, S_b: np.ndarray,
) -> float:
    """SV-weighted alignment score (Gradience-style energy weighting)."""
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return 0.0
    cos_matrix = np.abs(U_a.T @ U_b)
    weight_matrix = np.outer(S_a, S_b)
    Z = S_a.sum() * S_b.sum()
    if Z < 1e-12:
        return 0.0
    return float(np.sum(weight_matrix * cos_matrix) / Z)


def detect_module_type(layer_name: str) -> str:
    """Detect Q/K/V/O module type from layer name."""
    name_lower = layer_name.lower()
    if "q_lin" in name_lower or ".q_proj" in name_lower or ".query" in name_lower:
        return "q"
    if "k_lin" in name_lower or ".k_proj" in name_lower or ".key" in name_lower:
        return "k"
    if "v_lin" in name_lower or ".v_proj" in name_lower or ".value" in name_lower:
        return "v"
    if "out_lin" in name_lower or ".o_proj" in name_lower or ".output" in name_lower or ".dense" in name_lower:
        return "o"
    return "other"


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

def _build_adapter_list() -> list[AdapterInfo]:
    """Build list of all available adapters from known locations."""
    adapters = []

    # Distilbert adapters
    for task in ["qnli", "mrpc", "rte", "sst2"]:
        for seed in ["s42", "s7"]:
            d = DISTILBERT_BASE / f"{task}_{seed}"
            if (d / "adapter_model.safetensors").exists():
                adapters.append(AdapterInfo(
                    adapter_id=f"distilbert_{task}_{seed}",
                    backbone="distilbert",
                    task=task.upper(),
                    seed=seed,
                    path=d,
                ))

    # RoBERTa adapters
    for task in ["qnli", "mrpc", "rte", "sst2"]:
        for seed in ["s42", "s7"]:
            d = ROBERTA_BASE / f"{task}_{seed}"
            if (d / "adapter_model.safetensors").exists():
                adapters.append(AdapterInfo(
                    adapter_id=f"roberta_{task}_{seed}",
                    backbone="roberta",
                    task=task.upper(),
                    seed=seed,
                    path=d,
                ))

    return adapters


def _load_merge_outcomes() -> dict[str, dict]:
    """Load merge outcomes from attractor panel table.

    Returns dict mapping "{backbone}_{task}" -> {merge_delta, attractor_class}.
    """
    panel_path = REPO_ROOT / "sidecar" / "results" / "attractor_mapping" / "attractor_panel_table.json"
    if not panel_path.exists():
        print(f"  Warning: attractor panel not found at {panel_path}")
        return {}

    with open(panel_path) as f:
        panel = json.load(f)

    outcomes = {}
    family_to_task = {
        "QNLI": "QNLI", "MRPC": "MRPC", "RTE": "RTE", "SST-2": "SST2",
        "SST-2 (domain)": "SST2_DOMAIN", "Yelp": "YELP", "Amazon": "AMAZON",
        "Strong QNLI": "STRONG_QNLI", "Medium QNLI": "MEDIUM_QNLI",
        "Weak QNLI": "WEAK_QNLI",
    }

    for fam in panel.get("families", []):
        task_key = family_to_task.get(fam["family"], fam["family"])
        backbone = fam["backbone"]
        key = f"{backbone}_{task_key}"
        outcomes[key] = {
            "merge_delta": fam["merge_delta"],
            "attractor_class": fam["preliminary_attractor_class"],
        }

    return outcomes


def load_corpus() -> list[PairSpec]:
    """Build the validation corpus: same-task seed pairs + cross-task contrast pairs."""
    adapters = _build_adapter_list()
    outcomes = _load_merge_outcomes()

    print(f"Found {len(adapters)} adapters")
    for a in adapters:
        print(f"  {a.adapter_id}: {a.path}")

    pairs = []

    # Group by backbone
    by_backbone: dict[str, list[AdapterInfo]] = {}
    for a in adapters:
        by_backbone.setdefault(a.backbone, []).append(a)

    for backbone, bb_adapters in by_backbone.items():
        # Same-task seed pairs
        by_task: dict[str, list[AdapterInfo]] = {}
        for a in bb_adapters:
            by_task.setdefault(a.task, []).append(a)

        for task, task_adapters in by_task.items():
            if len(task_adapters) >= 2:
                a, b = task_adapters[0], task_adapters[1]
                outcome_key = f"{backbone}_{task}"
                outcome = outcomes.get(outcome_key, {})
                pairs.append(PairSpec(
                    pair_id=f"{backbone}_{task}_seed_pair",
                    adapter_a=a,
                    adapter_b=b,
                    task_relation="same_task",
                    merge_delta=outcome.get("merge_delta"),
                    attractor_class=outcome.get("attractor_class"),
                ))

        # Cross-task pairs: all task combinations within backbone, using s42 seed
        s42_adapters = [a for a in bb_adapters if a.seed == "s42"]
        for a, b in combinations(s42_adapters, 2):
            pairs.append(PairSpec(
                pair_id=f"{backbone}_{a.task}x{b.task}_cross",
                adapter_a=a,
                adapter_b=b,
                task_relation="cross_task",
                merge_delta=None,
                attractor_class=None,
            ))

    print(f"\nCorpus: {len(pairs)} pairs")
    same = sum(1 for p in pairs if p.task_relation == "same_task")
    cross = sum(1 for p in pairs if p.task_relation == "cross_task")
    print(f"  Same-task: {same}, Cross-task: {cross}")

    return pairs


# ---------------------------------------------------------------------------
# Core analysis pipeline
# ---------------------------------------------------------------------------

def compute_svds(
    weights: dict[str, dict[str, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute SVDs for all layers of one adapter."""
    svds = {}
    for layer, w in weights.items():
        U, S, Vt = compute_layer_svd(w["A"], w["B"], w["scaling"])
        svds[layer] = (U, S, Vt)
    return svds


def partition_bands(
    U: np.ndarray, S: np.ndarray, shape: tuple[int, int],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], float, int]:
    """Partition singular vectors into high-SV and low-SV bands.

    Returns ((U_high, S_high), (U_low, S_low), tau, k).
    """
    tau, k = mp_threshold(S, shape)
    U_high, S_high = U[:, :k], S[:k]
    U_low, S_low = U[:, k:], S[k:]
    return (U_high, S_high), (U_low, S_low), tau, k


def compute_layer_overlap(
    layer_name: str,
    svd_a: tuple[np.ndarray, np.ndarray, np.ndarray],
    svd_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, int],
    nominal_rank: int,
) -> LayerOverlap:
    """Compute band-split overlap for one layer of one pair."""
    U_a, S_a, _ = svd_a
    U_b, S_b, _ = svd_b

    (U_a_hi, S_a_hi), (U_a_lo, S_a_lo), tau_a, k_a = partition_bands(U_a, S_a, shape)
    (U_b_hi, S_b_hi), (U_b_lo, S_b_lo), tau_b, k_b = partition_bands(U_b, S_b, shape)

    total_energy_a = float(np.sum(S_a**2))
    total_energy_b = float(np.sum(S_b**2))

    def _band_overlap(band_name, Ua, Sa, Ub, Sb):
        e_a = float(np.sum(Sa**2)) / total_energy_a if total_energy_a > 0 else 0.0
        e_b = float(np.sum(Sb**2)) / total_energy_b if total_energy_b > 0 else 0.0
        cos_vals = principal_angle_cosines(Ua, Ub)
        return BandOverlap(
            band=band_name,
            n_dims_a=Ua.shape[1],
            n_dims_b=Ub.shape[1],
            mean_cos=float(cos_vals.mean()) if len(cos_vals) > 0 else 0.0,
            sv_weighted=sv_weighted_alignment(Ua, Sa, Ub, Sb),
            energy_frac_a=e_a,
            energy_frac_b=e_b,
        )

    layer = LayerOverlap(
        layer_name=layer_name,
        module_type=detect_module_type(layer_name),
        shape=shape,
        nominal_rank=nominal_rank,
        mp_tau_a=tau_a,
        mp_tau_b=tau_b,
        mp_k_a=k_a,
        mp_k_b=k_b,
    )
    layer.bands["full"] = _band_overlap("full", U_a, S_a, U_b, S_b)
    layer.bands["high_sv"] = _band_overlap("high_sv", U_a_hi, S_a_hi, U_b_hi, S_b_hi)
    layer.bands["low_sv"] = _band_overlap("low_sv", U_a_lo, S_a_lo, U_b_lo, S_b_lo)

    return layer


def aggregate_pair(pair: PairSpec, layers: list[LayerOverlap]) -> PairResult:
    """Aggregate per-layer overlaps into a pair-level result."""
    n = len(layers)
    result = PairResult(
        pair_id=pair.pair_id,
        task_relation=pair.task_relation,
        backbone=pair.adapter_a.backbone,
        task_a=pair.adapter_a.task,
        task_b=pair.adapter_b.task,
        merge_delta=pair.merge_delta,
        attractor_class=pair.attractor_class,
        n_layers=n,
        layers=layers,
    )

    if n == 0:
        return result

    result.full_mean_cos = float(np.mean([l.bands["full"].mean_cos for l in layers]))
    result.full_sv_weighted = float(np.mean([l.bands["full"].sv_weighted for l in layers]))
    result.high_mean_cos = float(np.mean([l.bands["high_sv"].mean_cos for l in layers]))
    result.high_sv_weighted = float(np.mean([l.bands["high_sv"].sv_weighted for l in layers]))
    result.low_mean_cos = float(np.mean([l.bands["low_sv"].mean_cos for l in layers]))
    result.low_sv_weighted = float(np.mean([l.bands["low_sv"].sv_weighted for l in layers]))

    # Tail excess: how much more overlap exists in the low band vs high band
    result.tail_excess = result.low_mean_cos - result.high_mean_cos

    # Energy masking: difference between energy-weighted and unweighted overlap
    result.energy_masking = result.full_sv_weighted - result.full_mean_cos

    return result


def flag_false_negative(result: PairResult) -> PairResult:
    """Apply false-negative criterion to a pair result.

    A false-negative candidate is a pair where:
    1. Gradience would classify it as safe (single_attractor or small merge_delta)
    2. Merge outcome shows degradation (merge_delta above threshold)
    3. Low-band conflict is high (mean_cos >= threshold)
    """
    if result.merge_delta is None:
        return result

    classified_safe = (
        result.attractor_class == "single_attractor"
        or (result.merge_delta is not None and result.merge_delta <= MERGE_DELTA_SAFE_THRESHOLD)
    )

    outcome_degraded = result.merge_delta > MERGE_DELTA_DEGRADED_THRESHOLD

    high_low_band_conflict = result.low_mean_cos >= FALSE_NEGATIVE_LOW_BAND_THRESHOLD

    if classified_safe and outcome_degraded and high_low_band_conflict:
        result.is_false_negative_candidate = True
        result.fn_reason = (
            f"Classified safe (attractor={result.attractor_class}, "
            f"delta={result.merge_delta:.4f}) but merge degraded "
            f"(delta>{MERGE_DELTA_DEGRADED_THRESHOLD}) with high low-band "
            f"conflict ({result.low_mean_cos:.3f} >= {FALSE_NEGATIVE_LOW_BAND_THRESHOLD})"
        )
    elif classified_safe and high_low_band_conflict and not outcome_degraded:
        result.fn_reason = (
            f"Low-band conflict present ({result.low_mean_cos:.3f}) but "
            f"merge outcome safe (delta={result.merge_delta:.4f}). "
            f"Tail interference exists but is operationally benign."
        )

    return result


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _pair_to_dict(r: PairResult) -> dict:
    """Serialize a PairResult to a JSON-compatible dict."""
    return {
        "pair_id": r.pair_id,
        "task_relation": r.task_relation,
        "backbone": r.backbone,
        "task_a": r.task_a,
        "task_b": r.task_b,
        "merge_delta": r.merge_delta,
        "attractor_class": r.attractor_class,
        "n_layers": r.n_layers,
        "aggregate": {
            "full_mean_cos": round(r.full_mean_cos, 6),
            "full_sv_weighted": round(r.full_sv_weighted, 6),
            "high_mean_cos": round(r.high_mean_cos, 6),
            "high_sv_weighted": round(r.high_sv_weighted, 6),
            "low_mean_cos": round(r.low_mean_cos, 6),
            "low_sv_weighted": round(r.low_sv_weighted, 6),
            "tail_excess": round(r.tail_excess, 6),
            "energy_masking": round(r.energy_masking, 6),
        },
        "false_negative": {
            "is_candidate": r.is_false_negative_candidate,
            "reason": r.fn_reason,
        },
        "per_layer": [
            {
                "layer": l.layer_name,
                "module_type": l.module_type,
                "shape": list(l.shape),
                "mp_k_a": l.mp_k_a,
                "mp_k_b": l.mp_k_b,
                "bands": {
                    bname: {
                        "n_dims_a": b.n_dims_a,
                        "n_dims_b": b.n_dims_b,
                        "mean_cos": round(b.mean_cos, 6),
                        "sv_weighted": round(b.sv_weighted, 6),
                        "energy_frac_a": round(b.energy_frac_a, 4),
                        "energy_frac_b": round(b.energy_frac_b, 4),
                    }
                    for bname, b in l.bands.items()
                },
            }
            for l in r.layers
        ],
    }


def write_json(results: list[PairResult], path: Path) -> None:
    """Write results to JSON."""
    same_task = [r for r in results if r.task_relation == "same_task"]
    cross_task = [r for r in results if r.task_relation == "cross_task"]
    fn_candidates = [r for r in results if r.is_false_negative_candidate]

    output = {
        "schema": "sidecar.n128_tail_interference/v1",
        "study": "N128",
        "description": "Tail interference probe: does energy-weighted overlap miss low-SV band conflict?",
        "config": {
            "false_negative_low_band_threshold": FALSE_NEGATIVE_LOW_BAND_THRESHOLD,
            "merge_delta_degraded_threshold": MERGE_DELTA_DEGRADED_THRESHOLD,
            "merge_delta_safe_threshold": MERGE_DELTA_SAFE_THRESHOLD,
            "partition_method": "gavish_donoho_optimal_hard_threshold",
        },
        "summary": {
            "n_pairs_total": len(results),
            "n_same_task": len(same_task),
            "n_cross_task": len(cross_task),
            "n_false_negative_candidates": len(fn_candidates),
            "false_negative_rate": len(fn_candidates) / len(same_task) if same_task else 0.0,
            "mean_tail_excess_same_task": round(float(np.mean([r.tail_excess for r in same_task])), 6) if same_task else None,
            "mean_tail_excess_cross_task": round(float(np.mean([r.tail_excess for r in cross_task])), 6) if cross_task else None,
            "mean_energy_masking_same_task": round(float(np.mean([r.energy_masking for r in same_task])), 6) if same_task else None,
            "mean_energy_masking_cross_task": round(float(np.mean([r.energy_masking for r in cross_task])), 6) if cross_task else None,
        },
        "pairs": [_pair_to_dict(r) for r in results],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(results: list[PairResult]) -> None:
    """Print a structured console report."""
    same_task = [r for r in results if r.task_relation == "same_task"]
    cross_task = [r for r in results if r.task_relation == "cross_task"]
    fn_candidates = [r for r in results if r.is_false_negative_candidate]

    print(f"\n{'='*74}")
    print("N128: TAIL INTERFERENCE PROBE RESULTS")
    print(f"{'='*74}")

    # Section A: Summary
    print(f"\n--- A. Summary ---")
    print(f"  Total pairs:              {len(results)}")
    print(f"  Same-task seed pairs:     {len(same_task)}")
    print(f"  Cross-task contrast:      {len(cross_task)}")
    print(f"  False-negative candidates: {len(fn_candidates)}")
    if same_task:
        fn_rate = len(fn_candidates) / len(same_task)
        print(f"  False-negative rate:      {fn_rate:.1%} ({len(fn_candidates)}/{len(same_task)})")

    # Section B: Per-pair table
    print(f"\n--- B. Per-Pair Band-Split Overlap ---")
    header = (
        f"  {'Pair':<38} {'Rel':<6} {'delta':>6} "
        f"{'Hi cos':>7} {'Lo cos':>7} {'Hi svw':>7} {'Lo svw':>7} "
        f"{'TailEx':>7} {'EMask':>7} {'FN':>3}"
    )
    print(header)
    print(f"  {'_'*110}")

    for r in sorted(results, key=lambda x: (x.task_relation, x.pair_id)):
        delta_str = f"{r.merge_delta:.4f}" if r.merge_delta is not None else "   n/a"
        fn_str = " *" if r.is_false_negative_candidate else "  "
        short_id = r.pair_id if len(r.pair_id) <= 38 else r.pair_id[:35] + "..."
        print(
            f"  {short_id:<38} {r.task_relation:<6} {delta_str:>6} "
            f"{r.high_mean_cos:>7.4f} {r.low_mean_cos:>7.4f} "
            f"{r.high_sv_weighted:>7.4f} {r.low_sv_weighted:>7.4f} "
            f"{r.tail_excess:>+7.4f} {r.energy_masking:>+7.4f} {fn_str}"
        )

    # Section C: False-negative candidates
    print(f"\n--- C. False-Negative Candidates ---")
    if fn_candidates:
        for r in fn_candidates:
            print(f"\n  {r.pair_id}")
            print(f"    {r.fn_reason}")
            print(f"    Per-module low-band conflict:")
            by_mod: dict[str, list[float]] = {}
            for l in r.layers:
                by_mod.setdefault(l.module_type, []).append(l.bands["low_sv"].mean_cos)
            for mod, vals in sorted(by_mod.items()):
                print(f"      {mod}: {np.mean(vals):.4f} (n={len(vals)})")
    else:
        print("  None found.")

    # Section D: Aggregate statistics
    print(f"\n--- D. Aggregate Statistics ---")

    def _print_group(label, group):
        if not group:
            return
        print(f"\n  {label} (n={len(group)}):")
        print(f"    High-SV band:  mean_cos={np.mean([r.high_mean_cos for r in group]):.4f}  "
              f"sv_weighted={np.mean([r.high_sv_weighted for r in group]):.4f}")
        print(f"    Low-SV band:   mean_cos={np.mean([r.low_mean_cos for r in group]):.4f}  "
              f"sv_weighted={np.mean([r.low_sv_weighted for r in group]):.4f}")
        print(f"    Full spectrum:  mean_cos={np.mean([r.full_mean_cos for r in group]):.4f}  "
              f"sv_weighted={np.mean([r.full_sv_weighted for r in group]):.4f}")
        te = [r.tail_excess for r in group]
        em = [r.energy_masking for r in group]
        print(f"    Tail excess:   mean={np.mean(te):+.4f}  std={np.std(te):.4f}  "
              f"range=[{min(te):+.4f}, {max(te):+.4f}]")
        print(f"    Energy masking: mean={np.mean(em):+.4f}  std={np.std(em):.4f}  "
              f"range=[{min(em):+.4f}, {max(em):+.4f}]")

    _print_group("Same-task seed pairs", same_task)
    _print_group("Cross-task contrast pairs", cross_task)

    # Section E: Module-type breakdown (same-task only)
    if same_task:
        print(f"\n--- E. Module-Type Breakdown (same-task only) ---")
        mod_hi: dict[str, list[float]] = {}
        mod_lo: dict[str, list[float]] = {}
        for r in same_task:
            for l in r.layers:
                mod_hi.setdefault(l.module_type, []).append(l.bands["high_sv"].mean_cos)
                mod_lo.setdefault(l.module_type, []).append(l.bands["low_sv"].mean_cos)

        print(f"  {'Module':<8} {'Hi mean_cos':>12} {'Lo mean_cos':>12} {'Hi/Lo ratio':>12}")
        for mod in sorted(mod_hi.keys()):
            h = np.mean(mod_hi[mod])
            l_val = np.mean(mod_lo[mod])
            ratio = h / l_val if l_val > 1e-8 else float("inf")
            print(f"  {mod:<8} {h:>12.4f} {l_val:>12.4f} {ratio:>12.1f}x")

    # Section F: Decision
    print(f"\n--- F. Decision ---")
    if fn_candidates:
        print("  FALSE NEGATIVES FOUND. Tail interference is operationally relevant.")
        print("  Recommend: add tail-aware compatibility metric to roadmap.")
    else:
        print("  No false negatives found in the current validation corpus.")
        n_with_outcome = sum(1 for r in results if r.merge_delta is not None)
        if n_with_outcome > 0:
            upper_bound = 1.0 / n_with_outcome
            print(f"  Upper bound on FN rate: {upper_bound:.1%} (1/{n_with_outcome})")
        print("  Tail interference concern is not operationally urgent at current scale.")
        print("  Leave THEORY.md S7.2 open problem as written.")

    print(f"\n{'='*74}\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    print("N128: Tail Interference Probe")
    print(f"Repo root: {REPO_ROOT}")

    # Step 1: Load corpus
    print(f"\n{'_'*40}")
    print("Step 1: Loading corpus")
    corpus = load_corpus()
    if not corpus:
        print("ERROR: No adapter pairs found.")
        return 1

    # Step 2: Load all adapter weights and compute SVDs
    print(f"\n{'_'*40}")
    print("Step 2: Loading adapter weights and computing SVDs")

    adapter_cache: dict[str, tuple[dict, dict]] = {}

    all_adapter_ids = set()
    for pair in corpus:
        all_adapter_ids.add(pair.adapter_a.adapter_id)
        all_adapter_ids.add(pair.adapter_b.adapter_id)

    id_to_info: dict[str, AdapterInfo] = {}
    for pair in corpus:
        id_to_info[pair.adapter_a.adapter_id] = pair.adapter_a
        id_to_info[pair.adapter_b.adapter_id] = pair.adapter_b

    for aid in sorted(all_adapter_ids):
        info = id_to_info[aid]
        print(f"  Loading {aid}...", end=" ", flush=True)
        weights = load_adapter_weights(info.path)
        svds = compute_svds(weights)
        adapter_cache[aid] = (weights, svds)
        print(f"{len(weights)} layers")

    # Step 3: Compute band-split overlaps for each pair
    print(f"\n{'_'*40}")
    print("Step 3: Computing band-split overlaps")

    results: list[PairResult] = []

    for pair in corpus:
        weights_a, svds_a = adapter_cache[pair.adapter_a.adapter_id]
        weights_b, svds_b = adapter_cache[pair.adapter_b.adapter_id]

        shared_layers = sorted(set(svds_a.keys()) & set(svds_b.keys()))

        layer_results = []
        for layer_name in shared_layers:
            w_a = weights_a[layer_name]
            shape = (w_a["B"].shape[0], w_a["A"].shape[1])
            nominal_rank = w_a["A"].shape[0]

            layer_overlap = compute_layer_overlap(
                layer_name, svds_a[layer_name], svds_b[layer_name],
                shape, nominal_rank,
            )
            layer_results.append(layer_overlap)

        pair_result = aggregate_pair(pair, layer_results)
        pair_result = flag_false_negative(pair_result)
        results.append(pair_result)

        fn_marker = " [FN CANDIDATE]" if pair_result.is_false_negative_candidate else ""
        print(f"  {pair.pair_id}: hi={pair_result.high_mean_cos:.4f} "
              f"lo={pair_result.low_mean_cos:.4f} "
              f"tail_excess={pair_result.tail_excess:+.4f}{fn_marker}")

    # Step 4: Write JSON results
    print(f"\n{'_'*40}")
    print("Step 4: Writing results")
    write_json(results, RESULTS_JSON)
    print(f"  JSON: {RESULTS_JSON}")

    # Step 5: Print console report
    print_report(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
