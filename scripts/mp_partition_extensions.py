#!/usr/bin/env python3
"""
MP Partition Test — Three Extensions:

Extension 1: W₀ spectral gap correlation
    Compute the spectral gap of the pre-trained weight matrix at each layer
    and correlate it with the per-layer high-SV alignment from N127.
    Prediction (Davis-Kahan): larger W₀ spectral gaps → stronger high-SV
    alignment between independently trained adapters.

Extension 2: Cross-task adapter pairs
    Compare same-task alignment (SST-2 × SST-2) vs cross-task alignment
    (SST-2 × QNLI) on the same backbone.
    Prediction: the high-SV / low-SV alignment ratio should weaken or
    vanish for cross-task pairs.

Extension 3: Checkpoint progression
    Track alignment across training checkpoints (25, 50, 100, 150, 200).
    Prediction: alignment in the high-SV band should increase with
    training as adapters converge toward the pre-trained attractor.

Depends on: N127 (mp_partition_test.py)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.linalg import svd, qr
from safetensors import safe_open
from scipy import stats

# Import core functions from the main script
sys.path.insert(0, str(Path(__file__).parent))
from mp_partition_test import (
    load_adapter_weights,
    compute_layer_svd,
    mp_threshold,
    sv_weighted_alignment,
    mean_alignment,
    detect_module_type,
)


# ===================================================================
# Extension 1: W₀ spectral gap correlation
# ===================================================================

def load_base_model_weights(model_path: Path) -> dict[str, np.ndarray]:
    """Load attention weight matrices from a full model checkpoint.

    We use a fine-tuned checkpoint as a proxy for W₀. Since LoRA updates
    are small perturbations (UDR << 1 typically), the spectral gap of
    the fine-tuned weight matrix closely approximates W₀'s spectral gap.
    """
    weights = {}
    with safe_open(str(model_path), framework="numpy") as f:
        for key in f.keys():
            if "attention" in key and key.endswith(".weight"):
                # Strip to match adapter layer naming
                short_key = key
                for prefix in ["base_model.model.", "base_model."]:
                    if short_key.startswith(prefix):
                        short_key = short_key[len(prefix):]
                # Remove .weight suffix to match adapter key format
                short_key = short_key.replace(".weight", "")
                W = f.get_tensor(key).astype(np.float64)
                weights[short_key] = W
    return weights


def compute_spectral_gaps(W: np.ndarray, top_k: int = 5) -> dict:
    """Compute spectral gap information for a weight matrix.

    Returns:
        dict with gap_1_2 (σ₁-σ₂), gap_k_k1 for various k,
        relative gaps, and top singular values.
    """
    S = svd(W, compute_uv=False)
    result = {
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]) if len(S) > 1 else 0.0,
        "gap_1_2": float(S[0] - S[1]) if len(S) > 1 else float(S[0]),
        "relative_gap_1_2": float((S[0] - S[1]) / S[0]) if len(S) > 1 and S[0] > 0 else 1.0,
    }
    # Gaps at various positions
    for k in [1, 2, 3, 5, 8]:
        if len(S) > k:
            result[f"gap_{k}_{k+1}"] = float(S[k-1] - S[k])
            result[f"relative_gap_{k}_{k+1}"] = float(
                (S[k-1] - S[k]) / S[k-1]
            ) if S[k-1] > 0 else 0.0
    # Ratio of top-k energy to total
    total_energy = float(np.sum(S**2))
    for k in [1, 2, 3]:
        if len(S) >= k:
            result[f"energy_top_{k}"] = float(np.sum(S[:k]**2)) / total_energy if total_energy > 0 else 0.0
    return result


def run_spectral_gap_correlation(
    base_model_path: Path,
    adapter_dirs: list[Path],
    labels: list[str],
) -> dict:
    """Correlate W₀ spectral gaps with per-layer high-SV alignment."""
    print(f"\n{'='*70}")
    print("EXTENSION 1: W₀ SPECTRAL GAP CORRELATION")
    print(f"{'='*70}")

    # Load base model weights
    print(f"\nLoading base model from {base_model_path}")
    base_weights = load_base_model_weights(base_model_path)
    print(f"  Loaded {len(base_weights)} attention layers")

    # Load adapters and compute SVDs
    all_weights = []
    all_svds = []
    for i, d in enumerate(adapter_dirs):
        w = load_adapter_weights(d)
        all_weights.append(w)
        svds = {}
        for layer_name, layer_data in w.items():
            U, S, Vt = compute_layer_svd(
                layer_data["A"], layer_data["B"], layer_data["scaling"]
            )
            svds[layer_name] = (U, S, Vt)
        all_svds.append(svds)

    shared_layers = sorted(set.intersection(
        *[set(w.keys()) for w in all_weights],
        set(base_weights.keys())
    ))
    print(f"Shared layers (adapters ∩ base): {len(shared_layers)}")

    pairs = list(combinations(range(len(adapter_dirs)), 2))

    # Compute per-layer: spectral gap AND high-SV alignment
    results_per_layer = []

    for layer_name in shared_layers:
        W0 = base_weights[layer_name]
        gaps = compute_spectral_gaps(W0)
        module_type = detect_module_type(layer_name)
        shape = W0.shape

        # High-SV alignment for this layer (across all pairs)
        high_aligns = []
        low_aligns = []
        for idx_a, idx_b in pairs:
            U_a, S_a, _ = all_svds[idx_a][layer_name]
            U_b, S_b, _ = all_svds[idx_b][layer_name]
            _, k_a = mp_threshold(S_a, shape)
            _, k_b = mp_threshold(S_b, shape)

            h = sv_weighted_alignment(
                U_a[:, :k_a], S_a[:k_a], U_b[:, :k_b], S_b[:k_b]
            )
            l = sv_weighted_alignment(
                U_a[:, k_a:], S_a[k_a:], U_b[:, k_b:], S_b[k_b:]
            )
            high_aligns.append(h)
            low_aligns.append(l)

        results_per_layer.append({
            "layer": layer_name,
            "module": module_type,
            "gaps": gaps,
            "mean_high_align": float(np.mean(high_aligns)),
            "mean_low_align": float(np.mean(low_aligns)),
            "high_low_ratio": float(np.mean(high_aligns) / np.mean(low_aligns))
            if np.mean(low_aligns) > 1e-8 else float("inf"),
        })

    # Correlate gap metrics with alignment
    gap_metrics = ["relative_gap_1_2", "gap_1_2", "energy_top_1", "energy_top_2"]
    align_metrics = ["mean_high_align", "high_low_ratio"]

    print(f"\n{'─'*70}")
    print(f"{'Layer':<45} {'rel_gap₁₂':>9} {'high_al':>8} {'H/L':>6}")
    print(f"{'─'*70}")

    for r in results_per_layer:
        short = r["layer"][-42:] if len(r["layer"]) > 42 else r["layer"]
        rg = r["gaps"]["relative_gap_1_2"]
        ha = r["mean_high_align"]
        hl = r["high_low_ratio"]
        print(f"{short:<45} {rg:>9.4f} {ha:>8.4f} {hl:>6.1f}x")

    print(f"\n{'='*70}")
    print("CORRELATIONS: W₀ spectral gap vs adapter alignment")
    print(f"{'='*70}\n")

    correlations = {}
    for gm in gap_metrics:
        gap_values = [r["gaps"][gm] for r in results_per_layer if gm in r["gaps"]]
        if len(gap_values) < len(results_per_layer):
            continue
        for am in align_metrics:
            align_values = [r[am] for r in results_per_layer]
            if len(gap_values) != len(align_values):
                continue
            r_val, p_val = stats.spearmanr(gap_values, align_values)
            key = f"{gm} × {am}"
            correlations[key] = {"spearman_r": r_val, "p_value": p_val}
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {key:<45} r={r_val:>6.3f}  p={p_val:.3e} {sig}")

    # By module type
    print(f"\n  By module type:")
    for mt in sorted(set(r["module"] for r in results_per_layer)):
        subset = [r for r in results_per_layer if r["module"] == mt]
        if len(subset) < 3:
            continue
        gaps = [r["gaps"]["relative_gap_1_2"] for r in subset]
        aligns = [r["mean_high_align"] for r in subset]
        r_val, p_val = stats.spearmanr(gaps, aligns) if len(set(gaps)) > 1 else (0, 1)
        print(f"    {mt}: n={len(subset)}, r={r_val:.3f}, p={p_val:.3f}")

    return {
        "per_layer": results_per_layer,
        "correlations": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in correlations.items()},
    }


# ===================================================================
# Extension 2: Cross-task adapter pairs
# ===================================================================

def run_cross_task_comparison(
    same_task_dirs: list[tuple[Path, str]],
    cross_task_dirs: list[tuple[Path, str]],
    task_labels: list[str],
) -> dict:
    """Compare alignment patterns between same-task and cross-task pairs."""
    print(f"\n{'='*70}")
    print("EXTENSION 2: CROSS-TASK COMPARISON")
    print(f"{'='*70}")

    all_dirs = same_task_dirs + cross_task_dirs
    all_labels = [l for _, l in all_dirs]

    # Load all adapters
    all_weights = []
    all_svds_map = []
    for d, label in all_dirs:
        print(f"Loading {label} from {d}")
        w = load_adapter_weights(d)
        all_weights.append(w)
        svds = {}
        for layer_name, layer_data in w.items():
            U, S, Vt = compute_layer_svd(
                layer_data["A"], layer_data["B"], layer_data["scaling"]
            )
            svds[layer_name] = (U, S, Vt)
        all_svds_map.append(svds)

    shared_layers = sorted(set.intersection(*[set(w.keys()) for w in all_weights]))
    print(f"Shared layers: {len(shared_layers)}")

    n_same = len(same_task_dirs)
    n_cross = len(cross_task_dirs)

    # Classify pairs
    same_task_pairs = list(combinations(range(n_same), 2))
    cross_task_pairs = [
        (i, j)
        for i in range(n_same)
        for j in range(n_same, n_same + n_cross)
    ]

    def compute_band_alignment(pairs, pair_type):
        high_all = []
        low_all = []
        full_all = []
        for layer_name in shared_layers:
            A0 = all_weights[0][layer_name]["A"]
            B0 = all_weights[0][layer_name]["B"]
            shape = (B0.shape[0], A0.shape[1])

            for idx_a, idx_b in pairs:
                U_a, S_a, _ = all_svds_map[idx_a][layer_name]
                U_b, S_b, _ = all_svds_map[idx_b][layer_name]
                _, k_a = mp_threshold(S_a, shape)
                _, k_b = mp_threshold(S_b, shape)

                h = sv_weighted_alignment(
                    U_a[:, :k_a], S_a[:k_a], U_b[:, :k_b], S_b[:k_b]
                )
                l = sv_weighted_alignment(
                    U_a[:, k_a:], S_a[k_a:], U_b[:, k_b:], S_b[k_b:]
                )
                f = sv_weighted_alignment(U_a, S_a, U_b, S_b)
                high_all.append(h)
                low_all.append(l)
                full_all.append(f)

        return {
            "type": pair_type,
            "n_pairs": len(pairs),
            "high_sv_mean": float(np.mean(high_all)),
            "high_sv_std": float(np.std(high_all)),
            "low_sv_mean": float(np.mean(low_all)),
            "low_sv_std": float(np.std(low_all)),
            "full_mean": float(np.mean(full_all)),
            "high_low_ratio": float(np.mean(high_all) / np.mean(low_all))
            if np.mean(low_all) > 1e-8 else float("inf"),
            "high_values": high_all,
            "low_values": low_all,
        }

    same_results = compute_band_alignment(same_task_pairs, "same_task")
    cross_results = compute_band_alignment(cross_task_pairs, "cross_task")

    print(f"\n{'─'*70}")
    print(f"{'Pair Type':<15} {'N':>4} {'High-SV':>9} {'Low-SV':>9} {'Full':>9} {'H/L':>7}")
    print(f"{'─'*70}")
    for r in [same_results, cross_results]:
        print(
            f"{r['type']:<15} {r['n_pairs']:>4} "
            f"{r['high_sv_mean']:>9.4f} {r['low_sv_mean']:>9.4f} "
            f"{r['full_mean']:>9.4f} {r['high_low_ratio']:>7.1f}x"
        )
    print(f"{'─'*70}")

    # Statistical comparison
    t_high, p_high = stats.ttest_ind(
        same_results["high_values"], cross_results["high_values"]
    )
    t_low, p_low = stats.ttest_ind(
        same_results["low_values"], cross_results["low_values"]
    )
    print(f"\n  Same vs cross (high-SV): t={t_high:.3f}, p={p_high:.3e}")
    print(f"  Same vs cross (low-SV):  t={t_low:.3f}, p={p_low:.3e}")

    # Does the ratio change?
    print(f"\n  High/Low ratio:")
    print(f"    Same-task:  {same_results['high_low_ratio']:.2f}x")
    print(f"    Cross-task: {cross_results['high_low_ratio']:.2f}x")

    return {
        "same_task": {k: v for k, v in same_results.items()
                      if k not in ("high_values", "low_values")},
        "cross_task": {k: v for k, v in cross_results.items()
                       if k not in ("high_values", "low_values")},
        "comparison": {
            "high_sv_ttest_t": float(t_high), "high_sv_ttest_p": float(p_high),
            "low_sv_ttest_t": float(t_low), "low_sv_ttest_p": float(p_low),
        }
    }


# ===================================================================
# Extension 3: Checkpoint progression
# ===================================================================

def run_checkpoint_progression(
    bench_dir: Path,
    seeds: list[int],
    policy: str,
    checkpoints: list[str],
    path_builder=None,
) -> dict:
    """Track alignment across training checkpoints.

    Args:
        path_builder: optional callable(bench_dir, seed, policy, ckpt) -> Path.
            If None, uses default: bench_dir / f"seed_{seed}" / policy / ckpt.
    """
    print(f"\n{'='*70}")
    print("EXTENSION 3: CHECKPOINT PROGRESSION")
    print(f"{'='*70}")

    if path_builder is None:
        def path_builder(bd, seed, pol, ckpt):
            return bd / f"seed_{seed}" / pol / ckpt

    results_by_checkpoint = []

    for ckpt in checkpoints:
        dirs = []
        labels = []
        for seed in seeds:
            d = path_builder(bench_dir, seed, policy, ckpt)
            if d.exists() and (d / "adapter_model.safetensors").exists():
                dirs.append(d)
                labels.append(f"s{seed}")

        if len(dirs) < 2:
            print(f"  {ckpt}: only {len(dirs)} adapters found, skipping")
            continue

        print(f"\n  {ckpt}: {len(dirs)} adapters")

        # Load and compute
        all_svds = []
        all_weights_local = []
        for d in dirs:
            w = load_adapter_weights(d)
            all_weights_local.append(w)
            svds = {}
            for layer_name, layer_data in w.items():
                U, S, Vt = compute_layer_svd(
                    layer_data["A"], layer_data["B"], layer_data["scaling"]
                )
                svds[layer_name] = (U, S, Vt)
            all_svds.append(svds)

        shared_layers = sorted(set.intersection(
            *[set(w.keys()) for w in all_weights_local]
        ))
        pairs = list(combinations(range(len(dirs)), 2))

        high_all = []
        low_all = []
        full_all = []
        energy_high_all = []

        for layer_name in shared_layers:
            A0 = all_weights_local[0][layer_name]["A"]
            B0 = all_weights_local[0][layer_name]["B"]
            shape = (B0.shape[0], A0.shape[1])

            for idx_a, idx_b in pairs:
                U_a, S_a, _ = all_svds[idx_a][layer_name]
                U_b, S_b, _ = all_svds[idx_b][layer_name]
                _, k_a = mp_threshold(S_a, shape)
                _, k_b = mp_threshold(S_b, shape)

                h = sv_weighted_alignment(
                    U_a[:, :k_a], S_a[:k_a], U_b[:, :k_b], S_b[:k_b]
                )
                l = sv_weighted_alignment(
                    U_a[:, k_a:], S_a[k_a:], U_b[:, k_b:], S_b[k_b:]
                )
                f = sv_weighted_alignment(U_a, S_a, U_b, S_b)
                high_all.append(h)
                low_all.append(l)
                full_all.append(f)

                total_e = float(np.sum(S_a**2))
                e_high = float(np.sum(S_a[:k_a]**2)) / total_e if total_e > 0 else 0
                energy_high_all.append(e_high)

        step = int(ckpt.replace("checkpoint-", ""))
        entry = {
            "checkpoint": ckpt,
            "step": step,
            "n_adapters": len(dirs),
            "n_pairs": len(pairs),
            "high_sv_mean": float(np.mean(high_all)),
            "high_sv_std": float(np.std(high_all)),
            "low_sv_mean": float(np.mean(low_all)),
            "low_sv_std": float(np.std(low_all)),
            "full_mean": float(np.mean(full_all)),
            "high_low_ratio": float(np.mean(high_all) / np.mean(low_all))
            if np.mean(low_all) > 1e-8 else float("inf"),
            "energy_high_mean": float(np.mean(energy_high_all)),
        }
        results_by_checkpoint.append(entry)
        print(
            f"    high={entry['high_sv_mean']:.4f}  "
            f"low={entry['low_sv_mean']:.4f}  "
            f"ratio={entry['high_low_ratio']:.2f}x  "
            f"E_high={entry['energy_high_mean']:.1%}"
        )

    # Print progression table
    print(f"\n{'─'*70}")
    print(f"{'Step':>6} {'High-SV':>9} {'Low-SV':>9} {'H/L ratio':>10} {'E_high':>8}")
    print(f"{'─'*70}")
    for r in results_by_checkpoint:
        print(
            f"{r['step']:>6} {r['high_sv_mean']:>9.4f} "
            f"{r['low_sv_mean']:>9.4f} {r['high_low_ratio']:>10.2f}x "
            f"{r['energy_high_mean']:>8.1%}"
        )
    print(f"{'─'*70}")

    # Trend analysis
    if len(results_by_checkpoint) >= 3:
        steps = [r["step"] for r in results_by_checkpoint]
        highs = [r["high_sv_mean"] for r in results_by_checkpoint]
        lows = [r["low_sv_mean"] for r in results_by_checkpoint]
        ratios = [r["high_low_ratio"] for r in results_by_checkpoint]

        r_high, p_high = stats.spearmanr(steps, highs)
        r_low, p_low = stats.spearmanr(steps, lows)
        r_ratio, p_ratio = stats.spearmanr(steps, ratios)

        print(f"\n  Trend (Spearman correlation with step):")
        print(f"    High-SV alignment: r={r_high:.3f}, p={p_high:.3f}")
        print(f"    Low-SV alignment:  r={r_low:.3f}, p={p_low:.3f}")
        print(f"    H/L ratio:         r={r_ratio:.3f}, p={p_ratio:.3f}")

    return {"checkpoints": results_by_checkpoint}


# ===================================================================
# Main
# ===================================================================

def main():
    repo = Path("/sessions/kind-trusting-cori/mnt/gradience")
    output_dir = repo / "sidecar" / "results" / "mp_partition_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Extension 1: W₀ spectral gap -----
    # Use ring2 sst2_s42 as proxy for W₀ (fine-tuned, but spectral gaps
    # are dominated by pre-trained structure since ΔW is small)
    base_model_path = (
        repo / "experiments" / "ring2_checkpoint_delta" / "checkpoints"
        / "sst2_s42" / "model.safetensors"
    )

    # Use adjudication study adapters (same backbone, rank 16)
    adj = repo / "results" / "adjudication_study" / "verified_adapters"
    sst2_adapters = [
        adj / "sst2_r16_s42",
        adj / "sst2_r16_s123",
    ]
    sst2_labels = ["sst2_s42", "sst2_s123"]

    ext1_results = run_spectral_gap_correlation(
        base_model_path, sst2_adapters, sst2_labels
    )

    # Also run with QNLI adapters
    qnli_adapters = [
        adj / "qnli_r16_s42",
        adj / "qnli_r16_s123",
    ]
    if all(d.exists() for d in qnli_adapters):
        print("\n  --- QNLI adapters ---")
        ext1_qnli = run_spectral_gap_correlation(
            base_model_path, qnli_adapters, ["qnli_s42", "qnli_s123"]
        )
        ext1_results["qnli"] = ext1_qnli

    # ----- Extension 2: Cross-task comparison -----
    same_task = [(adj / "sst2_r16_s42", "sst2_s42"),
                 (adj / "sst2_r16_s123", "sst2_s123")]
    cross_task = [(adj / "qnli_r16_s42", "qnli_s42"),
                  (adj / "qnli_r16_s123", "qnli_s123")]

    if all(d.exists() for d, _ in same_task + cross_task):
        ext2_results = run_cross_task_comparison(
            same_task, cross_task, ["sst2", "qnli"]
        )
    else:
        print("WARNING: Missing adapters for cross-task comparison")
        ext2_results = {}

    # ----- Extension 3: Checkpoint progression -----
    # 3a: Original firstclass_multiseed_test (only has checkpoints 25, 50)
    bench = repo / "bench_runs" / "firstclass_multiseed_test"
    checkpoints_short = ["checkpoint-25", "checkpoint-50"]
    ext3_short = run_checkpoint_progression(
        bench, seeds=[42, 123, 456], policy="probe_r16",
        checkpoints=checkpoints_short,
    )

    # 3b: Longer progression from uniform_r16_seed* directories
    # Path structure: bench_runs/uniform_r16_seed{seed}/uniform_median_r16/checkpoint-{N}
    def uniform_r16_path_builder(bd, seed, pol, ckpt):
        return bd / f"uniform_r16_seed{seed}" / pol / ckpt

    bench_root = repo / "bench_runs"
    checkpoints_long = ["checkpoint-50", "checkpoint-100",
                        "checkpoint-150", "checkpoint-200"]
    print("\n\n  --- Extended progression (uniform_r16, 4 checkpoints) ---")
    ext3_long = run_checkpoint_progression(
        bench_root, seeds=[42, 123, 456], policy="uniform_median_r16",
        checkpoints=checkpoints_long,
        path_builder=uniform_r16_path_builder,
    )

    ext3_results = {
        "short_progression": ext3_short,
        "long_progression": ext3_long,
    }

    # ----- Save all results -----
    all_results = {
        "extension_1_spectral_gap": ext1_results,
        "extension_2_cross_task": ext2_results,
        "extension_3_checkpoints": ext3_results,
    }

    with open(output_dir / "extension_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {output_dir / 'extension_results.json'}")


if __name__ == "__main__":
    main()
