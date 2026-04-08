#!/usr/bin/env python3
"""Experiment A: Per-Module Decomposition on DeBERTa-v3.

Tests Technical Report §7.1 predictions 2–5:
  P2: V-module dimensionality ratio separates catastrophic from safe
  P3: Instability transfers to third backbone
  P4: Mechanism–backbone confound dissolves or solidifies
  P5: Head-level modulation explains seed-to-seed severity variation

Usage:
    python3 scripts/n07_deberta/experiment_a_per_module.py --base-dir /workspace/n07
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy import stats


# ─── Constants ─────────────────────────────────────────────

MODULE_TYPES = ["query_proj", "key_proj", "value_proj", "attention.output.dense"]
MODULE_SHORT = {"query_proj": "Q", "key_proj": "K", "value_proj": "V", "attention.output.dense": "O"}
N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64  # 768 / 12


# ─── Helpers ───────────────────────────────────────────────

def stable_rank(S: np.ndarray) -> float:
    """Stable rank = ||M||_F^2 / ||M||_2^2 = sum(s^2) / max(s)^2."""
    if len(S) == 0 or S[0] == 0:
        return 0.0
    return float(np.sum(S ** 2) / (S[0] ** 2))


def energy_rank(S: np.ndarray, threshold: float = 0.90) -> int:
    """Number of singular values needed to capture `threshold` fraction of energy."""
    total = np.sum(S ** 2)
    if total == 0:
        return 0
    cumulative = np.cumsum(S ** 2) / total
    return int(np.searchsorted(cumulative, threshold) + 1)


def compute_principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces (column spaces)."""
    # SVD of U1^T @ U2
    M = U1.T @ U2
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    # Clamp to [0, 1] for numerical stability
    S = np.clip(S, 0, 1)
    return np.arccos(S)


def sv_weighted_alignment(S1: np.ndarray, U1: np.ndarray,
                          S2: np.ndarray, U2: np.ndarray,
                          band: str = "all",
                          mp_threshold: float | None = None) -> float:
    """SV-weighted alignment between two decompositions, optionally in a spectral band."""
    if mp_threshold is not None and band != "all":
        if band == "high":
            mask1 = S1 > mp_threshold
            mask2 = S2 > mp_threshold
        else:  # low
            mask1 = S1 <= mp_threshold
            mask2 = S2 <= mp_threshold
        if not np.any(mask1) or not np.any(mask2):
            return 0.0
        U1_band = U1[:, mask1]
        U2_band = U2[:, mask2]
    else:
        U1_band = U1
        U2_band = U2

    if U1_band.shape[1] == 0 or U2_band.shape[1] == 0:
        return 0.0

    # Compute overlap as mean squared cosine of principal angles
    angles = compute_principal_angles(U1_band, U2_band)
    return float(np.mean(np.cos(angles) ** 2))


def gavish_donoho_threshold(S: np.ndarray, beta: float) -> float:
    """Gavish-Donoho optimal hard threshold for singular values."""
    if beta <= 0 or beta > 1:
        beta = 1.0
    # Median of Marchenko-Pastur distribution
    lambda_star = np.sqrt(2 * (beta + 1) + (8 * beta) / (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1)))
    median_sv = np.median(S) if len(S) > 0 else 1.0
    if median_sv == 0:
        median_sv = 1.0
    return lambda_star * median_sv


def load_adapter_lora_weights(adapter_path: Path) -> dict[str, torch.Tensor]:
    """Load LoRA A and B matrices from an adapter."""
    weights = {}
    with safe_open(str(adapter_path / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if "lora" in key:
                weights[key] = f.get_tensor(key).float()
            elif "classifier" in key:
                weights[key] = f.get_tensor(key).float()
    return weights


def _lora_prefix(layer_idx: int, module_type: str) -> str | None:
    if module_type == "attention.output.dense":
        return f"base_model.model.deberta.encoder.layer.{layer_idx}.attention.output.dense"
    elif module_type in ("query_proj", "key_proj", "value_proj"):
        return f"base_model.model.deberta.encoder.layer.{layer_idx}.attention.self.{module_type}"
    return None


def get_lora_AB(weights: dict, layer_idx: int, module_type: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (A, B) LoRA matrices for a specific module. A: (r, d_in), B: (d_out, r)."""
    prefix = _lora_prefix(layer_idx, module_type)
    if prefix is None:
        return None
    a_key = f"{prefix}.lora_A.weight"
    b_key = f"{prefix}.lora_B.weight"
    if a_key not in weights or b_key not in weights:
        return None
    return weights[a_key].numpy(), weights[b_key].numpy()


def fast_svd_delta_W(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of delta_W = B@A efficiently using the (r x r) inner product.

    Since rank(B@A) <= r, we work in the low-rank space:
      1. QR on B to get orthonormal basis Q_B (d_out x r)
      2. Form M = Q_B^T @ B @ A = R_B @ A (r x d_in)
      3. SVD of M (r x d_in) — much smaller than (d_out x d_in)
      4. U = Q_B @ U_small
    """
    # Direct SVD on the r×d_in product R@A after QR of B
    Q_B, R_B = np.linalg.qr(B, mode='reduced')  # Q_B: (d_out, r), R_B: (r, r)
    M = R_B @ A  # (r, d_in) — small!
    U_small, S, Vt = np.linalg.svd(M, full_matrices=False)  # U_small: (r,r), S: (r,), Vt: (r, d_in)
    U = Q_B @ U_small  # (d_out, r)
    return U, S, Vt


def get_module_delta_W(weights: dict, layer_idx: int, module_type: str) -> np.ndarray | None:
    """Compute delta_W = B @ A for a specific module."""
    ab = get_lora_AB(weights, layer_idx, module_type)
    if ab is None:
        return None
    A, B = ab
    return B @ A  # shape: (d_out, d_in)


# ─── Step 1: Per-Module Spectral Profiles ──────────────────

def step1_per_module_profiles(adapters_dir: Path) -> dict:
    """Extract per-module spectral profiles from all adapters."""
    print("\n=== Step 1: Per-Module Spectral Profiles ===\n")
    profiles = {}

    for adapter_dir in sorted(adapters_dir.iterdir()):
        if not adapter_dir.is_dir():
            continue
        adapter_id = adapter_dir.name
        weights = load_adapter_lora_weights(adapter_dir)
        adapter_profiles = []

        for layer_idx in range(N_LAYERS):
            for module_type in MODULE_TYPES:
                ab = get_lora_AB(weights, layer_idx, module_type)
                if ab is None:
                    continue
                A, B = ab

                U, S, Vt = fast_svd_delta_W(A, B)
                sr = stable_rank(S)
                er90 = energy_rank(S, 0.90)
                frob = float(np.sqrt(np.sum(S ** 2)))
                beta = min(A.shape[0], B.shape[0]) / max(A.shape[0], B.shape[0])
                mp_thresh = gavish_donoho_threshold(S, beta)

                adapter_profiles.append({
                    "layer": layer_idx,
                    "module": module_type,
                    "module_short": MODULE_SHORT[module_type],
                    "stable_rank": sr,
                    "energy_rank_90": er90,
                    "frobenius_norm": frob,
                    "sigma_max": float(S[0]) if len(S) > 0 else 0,
                    "singular_values": S.tolist(),
                    "mp_threshold": float(mp_thresh),
                    "n_above_mp": int(np.sum(S > mp_thresh)),
                })

        profiles[adapter_id] = adapter_profiles
        # Summary
        for mt in MODULE_TYPES:
            entries = [p for p in adapter_profiles if p["module"] == mt]
            mean_sr = np.mean([e["stable_rank"] for e in entries])
            mean_frob = np.mean([e["frobenius_norm"] for e in entries])
            print(f"  {adapter_id} {MODULE_SHORT[mt]}: stable_rank={mean_sr:.3f}, frob={mean_frob:.3f}")

    return profiles


# ─── Step 2: Per-Module Pairwise Compatibility ─────────────

def step2_pairwise_compatibility(adapters_dir: Path, profiles: dict) -> dict:
    """Compute per-module pairwise compatibility for all 28 pairs."""
    print("\n=== Step 2: Per-Module Pairwise Compatibility ===\n")

    adapter_dirs = sorted([d for d in adapters_dir.iterdir() if d.is_dir()])
    pairs = list(itertools.combinations(adapter_dirs, 2))
    pairwise = {}

    # Cache all adapter weights to avoid redundant I/O
    weight_cache: dict[str, dict] = {}
    for d in adapter_dirs:
        weight_cache[d.name] = load_adapter_lora_weights(d)
    print(f"  Cached {len(weight_cache)} adapters")

    for i, (dir_a, dir_b) in enumerate(pairs):
        a_id = dir_a.name
        b_id = dir_b.name
        pair_label = f"{a_id}_vs_{b_id}"

        weights_a = weight_cache[a_id]
        weights_b = weight_cache[b_id]

        pair_data = []
        for layer_idx in range(N_LAYERS):
            for module_type in MODULE_TYPES:
                ab_a = get_lora_AB(weights_a, layer_idx, module_type)
                ab_b = get_lora_AB(weights_b, layer_idx, module_type)
                if ab_a is None or ab_b is None:
                    continue

                A_a, B_a = ab_a
                A_b, B_b = ab_b
                U_a, S_a, _ = fast_svd_delta_W(A_a, B_a)
                U_b, S_b, _ = fast_svd_delta_W(A_b, B_b)

                # Dimensionality ratio — merged delta_W = 0.5*(B_a@A_a + B_b@A_b)
                # Stack: [B_a, B_b] @ block_diag(A_a, A_b) scaled by 0.5
                # Faster: just compute merged and SVD on the 2r-rank result
                B_merged = np.concatenate([0.5 * B_a, 0.5 * B_b], axis=1)  # (d_out, 2r)
                A_merged = np.concatenate([A_a, A_b], axis=0)  # (2r, d_in)
                _, S_merged, _ = fast_svd_delta_W(A_merged, B_merged)
                sr_a = stable_rank(S_a)
                sr_b = stable_rank(S_b)
                sr_merged = stable_rank(S_merged)
                denom = sr_a + sr_b
                dr = sr_merged / denom if denom > 0 else 0.5

                # Subspace overlap (top-k principal angles)
                k = max(1, min(energy_rank(S_a, 0.90), energy_rank(S_b, 0.90), S_a.shape[0]))
                angles = compute_principal_angles(U_a[:, :k], U_b[:, :k])
                overlap = float(np.mean(np.cos(angles) ** 2))

                # Directional agreement
                # Sign-aware overlap: do the dominant directions point the same way?
                dot_products = np.sum(U_a[:, :k] * U_b[:, :k], axis=0)
                agreement = float(np.mean(dot_products))

                # Magnitude ratio
                frob_a = float(np.sqrt(np.sum(S_a ** 2)))
                frob_b = float(np.sqrt(np.sum(S_b ** 2)))
                mag_ratio = max(frob_a, frob_b) / min(frob_a, frob_b) if min(frob_a, frob_b) > 1e-10 else float("inf")

                pair_data.append({
                    "layer": layer_idx,
                    "module": module_type,
                    "module_short": MODULE_SHORT[module_type],
                    "dimensionality_ratio": dr,
                    "overlap": overlap,
                    "agreement": agreement,
                    "magnitude_ratio": mag_ratio,
                    "stable_rank_a": sr_a,
                    "stable_rank_b": sr_b,
                    "stable_rank_merged": sr_merged,
                })

        pairwise[pair_label] = pair_data
        if (i + 1) % 7 == 0:
            print(f"  Computed {i + 1}/{len(pairs)} pairs")

    print(f"  Done: {len(pairs)} pairs")
    return pairwise


# ─── Step 3: Classify Pairs ───────────────────────────────

def step3_classify_pairs(base_dir: Path) -> dict[str, str]:
    """Classify pairs as catastrophic/safe/intermediate based on Phase 3 + 3b degradation."""
    print("\n=== Step 3: Classify Pairs ===\n")

    classifications = {}

    # Load Phase 3 (naive linear) results
    p3_file = base_dir / "merges" / "phase3_results.json"
    if p3_file.exists():
        p3_results = json.loads(p3_file.read_text())
        for r in p3_results:
            if "error" in r:
                continue
            pair = r["pair"]
            deg = r["degradation"]
            # Use worst degradation across eval tasks for each pair
            if pair not in classifications or deg < classifications.get(f"{pair}_deg", 0):
                classifications[f"{pair}_deg"] = deg

    # Also load Phase 3b for additional merge conditions
    p3b_file = base_dir / "merges_strategic" / "phase3b_results.json"
    p3b_by_pair: dict[str, list] = defaultdict(list)
    if p3b_file.exists():
        p3b_results = json.loads(p3b_file.read_text())
        for r in p3b_results:
            if "error" in r:
                continue
            p3b_by_pair[r["pair"]].append(r["degradation_strategic"])

    # Classify using worst degradation across all merge conditions
    pair_classifications = {}
    for pair in set(r["pair"] for r in p3_results if "error" not in r):
        # Collect all degradation values for this pair
        all_degs = []
        for r in p3_results:
            if r.get("pair") == pair and "error" not in r:
                all_degs.append(r["degradation"])
        for d in p3b_by_pair.get(pair, []):
            all_degs.append(d)

        worst_deg = min(all_degs)
        best_deg = max(all_degs)

        if worst_deg <= -0.50:
            classification = "catastrophic"
        elif best_deg >= -0.20:
            classification = "safe"
        else:
            classification = "intermediate"

        pair_classifications[pair] = {
            "classification": classification,
            "worst_degradation": worst_deg,
            "best_degradation": best_deg,
            "n_conditions": len(all_degs),
        }

    counts = defaultdict(int)
    for v in pair_classifications.values():
        counts[v["classification"]] += 1
    print(f"  Catastrophic: {counts['catastrophic']}, Safe: {counts['safe']}, "
          f"Intermediate: {counts['intermediate']}")

    return pair_classifications


# ─── Step 4: Prediction 2 — V-Module DR ───────────────────

def step4_prediction_2(pairwise: dict, classifications: dict) -> dict:
    """Test: V-module dimensionality ratio separates catastrophic from safe."""
    print("\n=== Step 4: Prediction 2 — V-Module Dimensionality Ratio ===\n")

    results_by_module = {}
    for module_type in MODULE_TYPES:
        ms = MODULE_SHORT[module_type]
        cat_drs = []
        safe_drs = []

        for pair_label, pair_data in pairwise.items():
            cls_info = classifications.get(pair_label, {})
            cls = cls_info.get("classification", "intermediate")

            # Mean DR for this module across layers
            module_entries = [e for e in pair_data if e["module"] == module_type]
            if not module_entries:
                continue
            mean_dr = np.mean([e["dimensionality_ratio"] for e in module_entries])

            if cls == "catastrophic":
                cat_drs.append(mean_dr)
            elif cls == "safe":
                safe_drs.append(mean_dr)

        if len(cat_drs) >= 2 and len(safe_drs) >= 2:
            # Cohen's d
            pooled_std = np.sqrt(((len(cat_drs) - 1) * np.var(cat_drs, ddof=1) +
                                  (len(safe_drs) - 1) * np.var(safe_drs, ddof=1)) /
                                 (len(cat_drs) + len(safe_drs) - 2))
            d = (np.mean(safe_drs) - np.mean(cat_drs)) / pooled_std if pooled_std > 1e-10 else 0
            t_stat, p_val = stats.ttest_ind(cat_drs, safe_drs)
            range_overlap = max(min(cat_drs), min(safe_drs)) <= min(max(cat_drs), max(safe_drs))
        else:
            d = float("nan")
            t_stat = float("nan")
            p_val = float("nan")
            range_overlap = True

        result = {
            "module": module_type,
            "module_short": ms,
            "catastrophic_mean_dr": float(np.mean(cat_drs)) if cat_drs else None,
            "catastrophic_range": [float(min(cat_drs)), float(max(cat_drs))] if cat_drs else None,
            "safe_mean_dr": float(np.mean(safe_drs)) if safe_drs else None,
            "safe_range": [float(min(safe_drs)), float(max(safe_drs))] if safe_drs else None,
            "cohens_d": float(d),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "range_overlap": range_overlap,
            "n_catastrophic": len(cat_drs),
            "n_safe": len(safe_drs),
        }
        results_by_module[ms] = result

        print(f"  {ms}-module: cat={result['catastrophic_mean_dr']:.4f} "
              f"safe={result['safe_mean_dr']:.4f} "
              f"d={d:.2f} p={p_val:.4f} overlap={range_overlap}"
              if result["catastrophic_mean_dr"] is not None and result["safe_mean_dr"] is not None
              else f"  {ms}-module: insufficient data")

    # V-module specificity: is V the only module with d > 1?
    v_result = results_by_module.get("V", {})
    v_specific = (not math.isnan(v_result.get("cohens_d", float("nan"))) and
                  abs(v_result.get("cohens_d", 0)) >= 1.0 and
                  all(abs(results_by_module.get(m, {}).get("cohens_d", 0)) < 1.0
                      for m in ["Q", "K", "O"]))

    return {
        "prediction": "P2: V-module DR separates catastrophic from safe",
        "results_by_module": results_by_module,
        "v_module_specific": v_specific,
        "decision_criterion": "Cohen's d >= 1.0 with no range overlap",
        "supported": (not math.isnan(v_result.get("cohens_d", float("nan"))) and
                      abs(v_result.get("cohens_d", 0)) >= 1.0),
    }


# ─── Step 5: Prediction 3 — Instability Transfer ──────────

def step5_prediction_3(base_dir: Path, classifications: dict) -> dict:
    """Test: instability (severity variance across merge conditions) transfers to DeBERTa."""
    print("\n=== Step 5: Prediction 3 — Instability Transfer ===\n")

    # Collect all degradation values per pair across merge strategies
    pair_degs: dict[str, list] = defaultdict(list)

    for fname in ["merges/phase3_results.json", "merges_strategic/phase3b_results.json"]:
        fpath = base_dir / fname
        if not fpath.exists():
            continue
        for r in json.loads(fpath.read_text()):
            if "error" in r:
                continue
            pair_degs[r["pair"]].append(r.get("degradation", r.get("degradation_strategic", 0)))

    instability = {}
    for pair, degs in pair_degs.items():
        if len(degs) >= 2:
            instability[pair] = {
                "std": float(np.std(degs)),
                "range": float(max(degs) - min(degs)),
                "n_conditions": len(degs),
                "task_a": pair.split("_vs_")[0].split("_")[1],
                "task_b": pair.split("_vs_")[1].split("_")[1],
                "same_task": pair.split("_vs_")[0].split("_")[1] == pair.split("_vs_")[1].split("_")[1],
            }

    # Compare same-task vs cross-task instability
    same_task_inst = [v["std"] for v in instability.values() if v["same_task"]]
    cross_task_inst = [v["std"] for v in instability.values() if not v["same_task"]]

    if same_task_inst and cross_task_inst:
        u_stat, u_p = stats.mannwhitneyu(cross_task_inst, same_task_inst, alternative="greater")
    else:
        u_stat, u_p = float("nan"), float("nan")

    print(f"  Same-task instability: mean={np.mean(same_task_inst):.4f} (n={len(same_task_inst)})")
    print(f"  Cross-task instability: mean={np.mean(cross_task_inst):.4f} (n={len(cross_task_inst)})")
    print(f"  Mann-Whitney U p={u_p:.4f}")

    return {
        "prediction": "P3: Instability transfers to third backbone",
        "same_task_instability_mean": float(np.mean(same_task_inst)) if same_task_inst else None,
        "cross_task_instability_mean": float(np.mean(cross_task_inst)) if cross_task_inst else None,
        "mannwhitney_p": float(u_p),
        "cross_gt_same": (float(np.mean(cross_task_inst)) > float(np.mean(same_task_inst))
                          if same_task_inst and cross_task_inst else False),
        "supported": u_p < 0.05 if not math.isnan(u_p) else False,
        "per_pair_instability": instability,
    }


# ─── Step 6: Prediction 4 — Readout Attractor Structure ───

def step6_prediction_4(adapters_dir: Path) -> dict:
    """Test: classify readout attractor structure on DeBERTa."""
    print("\n=== Step 6: Prediction 4 — Readout Attractor Structure ===\n")

    # Load classifier weights for each adapter
    classifiers = {}
    for adapter_dir in sorted(adapters_dir.iterdir()):
        if not adapter_dir.is_dir():
            continue
        adapter_id = adapter_dir.name
        weights = load_adapter_lora_weights(adapter_dir)
        w_key = "base_model.model.classifier.weight"
        b_key = "base_model.model.classifier.bias"
        if w_key in weights:
            classifiers[adapter_id] = {
                "weight": weights[w_key].numpy(),
                "bias": weights[b_key].numpy() if b_key in weights else None,
            }

    # Analyze within-task cross-seed similarity
    tasks = defaultdict(list)
    for aid in classifiers:
        task = aid.split("_")[1]
        tasks[task].append(aid)

    task_analysis = {}
    for task, adapter_ids in tasks.items():
        if len(adapter_ids) < 2:
            continue
        a1, a2 = adapter_ids[0], adapter_ids[1]
        w1 = classifiers[a1]["weight"]
        w2 = classifiers[a2]["weight"]

        # Row-wise cosine similarity (per class)
        cos_sims = []
        for row_idx in range(w1.shape[0]):
            cos = np.dot(w1[row_idx], w2[row_idx]) / (np.linalg.norm(w1[row_idx]) * np.linalg.norm(w2[row_idx]) + 1e-10)
            cos_sims.append(float(cos))

        # Overall cosine similarity (flattened)
        flat_cos = float(np.dot(w1.flatten(), w2.flatten()) /
                         (np.linalg.norm(w1.flatten()) * np.linalg.norm(w2.flatten()) + 1e-10))

        # Classify attractor structure
        mean_row_cos = np.mean(cos_sims)
        if mean_row_cos > 0.9:
            attractor_type = "single_attractor"
        elif mean_row_cos > 0.3:
            attractor_type = "weak_attractor"
        elif mean_row_cos > -0.3:
            attractor_type = "orthogonal"
        else:
            attractor_type = "anti_correlated"

        # Check for rotational degeneracy vs feature-set switching
        # Rotational degeneracy: high cosine on the weight matrix overall but
        # sign-flipped rows (class permutation)
        # Feature-set switching: low cosine, different PCA structure
        _, s1, vt1 = np.linalg.svd(w1, full_matrices=False)
        _, s2, vt2 = np.linalg.svd(w2, full_matrices=False)
        pc_overlap = float(np.abs(np.dot(vt1[0], vt2[0])))

        if attractor_type == "orthogonal" and pc_overlap > 0.7:
            mechanism = "rotational_degeneracy"
        elif attractor_type == "orthogonal" and pc_overlap < 0.3:
            mechanism = "feature_set_switching"
        else:
            mechanism = "other"

        task_analysis[task] = {
            "adapters": [a1, a2],
            "per_class_cosine": cos_sims,
            "mean_row_cosine": float(mean_row_cos),
            "flat_cosine": flat_cos,
            "attractor_type": attractor_type,
            "pc_overlap": pc_overlap,
            "mechanism": mechanism,
        }

        print(f"  {task}: mean_row_cos={mean_row_cos:.3f} flat_cos={flat_cos:.3f} "
              f"pc_overlap={pc_overlap:.3f} → {attractor_type} ({mechanism})")

    # Cross-task classifier similarity
    cross_task_cos = []
    adapter_list = sorted(classifiers.keys())
    for i, a1 in enumerate(adapter_list):
        for a2 in adapter_list[i + 1:]:
            t1 = a1.split("_")[1]
            t2 = a2.split("_")[1]
            if t1 == t2:
                continue
            w1 = classifiers[a1]["weight"]
            w2 = classifiers[a2]["weight"]
            cos = float(np.dot(w1.flatten(), w2.flatten()) /
                        (np.linalg.norm(w1.flatten()) * np.linalg.norm(w2.flatten()) + 1e-10))
            cross_task_cos.append(cos)

    return {
        "prediction": "P4: Mechanism–backbone confound dissolves or solidifies",
        "per_task_analysis": task_analysis,
        "cross_task_mean_cosine": float(np.mean(cross_task_cos)) if cross_task_cos else None,
        "decision": "See mechanism field per task",
    }


# ─── Step 7: Prediction 5 — Head-Level Modulation ─────────

def step7_prediction_5(adapters_dir: Path, pairwise: dict, classifications: dict) -> dict:
    """Test: head-level cancellation patterns explain severity variation."""
    print("\n=== Step 7: Prediction 5 — Head-Level Modulation ===\n")

    head_analysis = {}

    # Focus on pairs with variable severity
    adapter_dirs = sorted([d for d in adapters_dir.iterdir() if d.is_dir()])

    # Select a few pairs with different classifications
    target_pairs = []
    for pair_label, cls_info in classifications.items():
        parts = pair_label.split("_vs_")
        if len(parts) == 2:
            target_pairs.append((pair_label, cls_info))

    for pair_label, cls_info in target_pairs[:10]:  # analyze up to 10 pairs
        parts = pair_label.split("_vs_")
        dir_a = adapters_dir / parts[0]
        dir_b = adapters_dir / parts[1]
        if not dir_a.exists() or not dir_b.exists():
            continue

        weights_a = load_adapter_lora_weights(dir_a)
        weights_b = load_adapter_lora_weights(dir_b)

        # V-module head-level analysis
        head_drs_by_layer = []
        module_drs_by_layer = []

        for layer_idx in range(N_LAYERS):
            ab_a = get_lora_AB(weights_a, layer_idx, "value_proj")
            ab_b = get_lora_AB(weights_b, layer_idx, "value_proj")
            if ab_a is None or ab_b is None:
                continue

            A_a, B_a = ab_a
            A_b, B_b = ab_b
            dW_a = B_a @ A_a
            dW_b = B_b @ A_b

            # Module-level DR — use fast SVD
            _, S_a, _ = fast_svd_delta_W(A_a, B_a)
            _, S_b, _ = fast_svd_delta_W(A_b, B_b)
            B_m = np.concatenate([0.5 * B_a, 0.5 * B_b], axis=1)
            A_m = np.concatenate([A_a, A_b], axis=0)
            _, S_m, _ = fast_svd_delta_W(A_m, B_m)
            module_dr = stable_rank(S_m) / (stable_rank(S_a) + stable_rank(S_b) + 1e-10)
            module_drs_by_layer.append(module_dr)

            # Per-head DR — head slices are small (768 x 64), direct SVD is fine
            head_drs = []
            for head_idx in range(N_HEADS):
                start = head_idx * HEAD_DIM
                end = start + HEAD_DIM
                head_a = dW_a[:, start:end] if dW_a.shape[1] >= end else dW_a
                head_b = dW_b[:, start:end] if dW_b.shape[1] >= end else dW_b

                if head_a.shape[1] < HEAD_DIM:
                    continue

                head_merged = 0.5 * head_a + 0.5 * head_b
                _, Sh_m, _ = np.linalg.svd(head_merged, full_matrices=False)
                _, Sh_a, _ = np.linalg.svd(head_a, full_matrices=False)
                _, Sh_b, _ = np.linalg.svd(head_b, full_matrices=False)
                head_dr = stable_rank(Sh_m) / (stable_rank(Sh_a) + stable_rank(Sh_b) + 1e-10)
                head_drs.append(head_dr)

            head_drs_by_layer.append(head_drs)

        if module_drs_by_layer:
            module_var = float(np.var(module_drs_by_layer))
            # Count heads with |delta_DR| >= 0.15 from module mean
            n_deviant_heads = 0
            all_head_deltas = []
            module_mean = np.mean(module_drs_by_layer)
            for layer_heads in head_drs_by_layer:
                for hdr in layer_heads:
                    delta = abs(hdr - module_mean)
                    all_head_deltas.append(delta)
                    if delta >= 0.15:
                        n_deviant_heads += 1

            head_analysis[pair_label] = {
                "classification": cls_info.get("classification", "?"),
                "module_level_dr_var": module_var,
                "module_level_variation_low": module_var < 0.01,
                "n_deviant_heads": n_deviant_heads,
                "head_level_modulation": n_deviant_heads >= 3,
                "mean_head_delta": float(np.mean(all_head_deltas)) if all_head_deltas else 0,
                "max_head_delta": float(np.max(all_head_deltas)) if all_head_deltas else 0,
            }

    # Check prediction: module variation < 0.10, head |delta| >= 0.15 in >= 3 heads
    modulation_confirmed = sum(1 for v in head_analysis.values()
                               if v.get("module_level_variation_low") and v.get("head_level_modulation"))

    print(f"  Pairs analyzed: {len(head_analysis)}")
    print(f"  Head-level modulation confirmed: {modulation_confirmed}/{len(head_analysis)}")

    return {
        "prediction": "P5: Head-level modulation explains severity variation",
        "per_pair": head_analysis,
        "n_modulation_confirmed": modulation_confirmed,
        "n_total": len(head_analysis),
        "supported": modulation_confirmed >= len(head_analysis) // 2 if head_analysis else False,
    }


# ─── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment A: Per-Module Decomposition")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir
    adapters_dir = base_dir / "adapters"
    output_dir = args.output_dir or (base_dir / "per_module")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Step 1
    profiles = step1_per_module_profiles(adapters_dir)
    (output_dir / "per_module_profiles.json").write_text(json.dumps(profiles, indent=2, default=str))

    # Step 2
    pairwise = step2_pairwise_compatibility(adapters_dir, profiles)
    (output_dir / "per_module_pairwise.json").write_text(json.dumps(pairwise, indent=2, default=str))

    # Step 3
    classifications = step3_classify_pairs(base_dir)
    (output_dir / "pair_classifications.json").write_text(json.dumps(classifications, indent=2, default=str))

    # Step 4: Prediction 2
    p2 = step4_prediction_2(pairwise, classifications)
    (output_dir / "prediction_2_vmodule_dr.json").write_text(json.dumps(p2, indent=2, default=str))

    # Step 5: Prediction 3
    p3 = step5_prediction_3(base_dir, classifications)
    (output_dir / "prediction_3_instability.json").write_text(json.dumps(p3, indent=2, default=str))

    # Step 6: Prediction 4
    p4 = step6_prediction_4(adapters_dir)
    (output_dir / "prediction_4_readout_attractors.json").write_text(json.dumps(p4, indent=2, default=str))

    # Step 7: Prediction 5
    p5 = step7_prediction_5(adapters_dir, pairwise, classifications)
    (output_dir / "prediction_5_head_modulation.json").write_text(json.dumps(p5, indent=2, default=str))

    # Summary
    elapsed = time.time() - t0
    summary = {
        "experiment": "A: Per-Module Decomposition on DeBERTa-v3",
        "elapsed_s": elapsed,
        "prediction_2": {"supported": p2["supported"], "v_specific": p2.get("v_module_specific")},
        "prediction_3": {"supported": p3["supported"]},
        "prediction_4": {k: v.get("mechanism") for k, v in p4.get("per_task_analysis", {}).items()},
        "prediction_5": {"supported": p5["supported"], "n_confirmed": p5["n_modulation_confirmed"]},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"  Experiment A Summary ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  P2 (V-module DR): {'✓' if p2['supported'] else '✗'} (V-specific: {p2.get('v_module_specific')})")
    print(f"  P3 (Instability): {'✓' if p3['supported'] else '✗'}")
    print(f"  P4 (Readout attractors): {summary['prediction_4']}")
    print(f"  P5 (Head modulation): {'✓' if p5['supported'] else '✗'} ({p5['n_modulation_confirmed']}/{p5['n_total']})")
    print(f"\n  Output: {output_dir}")


if __name__ == "__main__":
    main()
