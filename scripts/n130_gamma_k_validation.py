#!/usr/bin/env python3
"""N130: Spectral Susceptibility (Gamma_k) Validation.

Tests whether spectral susceptibility Gamma_k outperforms energy concentration
C_k as a predictor of per-layer inter-adapter alignment, validating the
concentration-weighted convergence bound from CONVERGENCE_BOUND.md.

Three predictions:
  P1: Gamma_k outperforms C_k as per-layer alignment predictor
  P2: epsilon_t * sqrt(Gamma_k(t)) decreases during training
  P3: QNLI adapters have higher effective rank than SST-2

Usage:
    python3 scripts/n130_gamma_k_validation.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numpy.linalg import svd
from safetensors import safe_open
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "sidecar" / "results" / "mp_partition_test"
BENCH_RUNS = PROJECT_ROOT / "bench_runs"
OUTPUT_DIR = PROJECT_ROOT / "sidecar" / "data" / "n130"

# SST-2 checkpoint adapters (3 seeds x 4 checkpoints)
SST2_CHECKPOINT_DIRS = {
    seed: {
        step: BENCH_RUNS / f"uniform_r16_seed{seed}" / "uniform_median_r16" / f"checkpoint-{step}"
        for step in [50, 100, 150, 200]
    }
    for seed in [42, 123, 456]
}

# SST-2 converged adapters (checkpoint-50 from firstclass)
SST2_CONVERGED_DIRS = [
    BENCH_RUNS / f"firstclass_multiseed_test/seed_{seed}/probe_r16/checkpoint-50"
    for seed in [42, 123, 456]
]

# QNLI converged adapters (checkpoint-50)
QNLI_CONVERGED_DIR = BENCH_RUNS / "qnli_test" / "probe_r32" / "checkpoint-50"

BASE_MODEL_NAME = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Core functions (imported from N127 approach)
# ---------------------------------------------------------------------------

def load_adapter_weights(adapter_dir: Path) -> dict[str, dict]:
    """Load LoRA A/B matrices from adapter safetensors."""
    weights = {}
    path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"

    config = json.loads(config_path.read_text())
    scaling = config.get("lora_alpha", 16) / config.get("r", 16)

    with safe_open(str(path), framework="numpy") as f:
        keys = list(f.keys())
        # Group by layer
        layer_keys: dict[str, dict] = {}
        for k in keys:
            if "lora_A" in k:
                layer_name = k.replace(".lora_A.weight", "")
                for prefix in ["base_model.model.", "base_model."]:
                    if layer_name.startswith(prefix):
                        layer_name = layer_name[len(prefix):]
                if layer_name not in layer_keys:
                    layer_keys[layer_name] = {}
                layer_keys[layer_name]["A"] = f.get_tensor(k).astype(np.float64)
            elif "lora_B" in k:
                layer_name = k.replace(".lora_B.weight", "")
                for prefix in ["base_model.model.", "base_model."]:
                    if layer_name.startswith(prefix):
                        layer_name = layer_name[len(prefix):]
                if layer_name not in layer_keys:
                    layer_keys[layer_name] = {}
                layer_keys[layer_name]["B"] = f.get_tensor(k).astype(np.float64)

        for layer_name, matrices in layer_keys.items():
            if "A" in matrices and "B" in matrices:
                weights[layer_name] = {
                    "A": matrices["A"],
                    "B": matrices["B"],
                    "scaling": scaling,
                }

    return weights


def compute_layer_svd(A: np.ndarray, B: np.ndarray, scaling: float = 1.0):
    """SVD of delta_W = scaling * B @ A."""
    delta_W = scaling * B @ A
    U, S, Vt = svd(delta_W, full_matrices=False)
    return U, S, Vt


def mp_threshold(S: np.ndarray, shape: tuple[int, int]) -> tuple[float, int]:
    """Gavish-Donoho optimal hard threshold."""
    d_out, d_in = shape
    beta = min(d_out, d_in) / max(d_out, d_in)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    median_sv = float(np.median(S))
    if median_sv < 1e-12:
        return 0.0, len(S)
    tau = omega * median_sv
    k = int(np.sum(tau < S))
    return tau, max(k, 1)


def sv_weighted_alignment(
    U_a: np.ndarray, S_a: np.ndarray,
    U_b: np.ndarray, S_b: np.ndarray,
) -> float:
    """SV-weighted alignment (same as N127)."""
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return 0.0
    cos_matrix = np.abs(U_a.T @ U_b)
    weight_matrix = np.outer(S_a, S_b)
    Z = S_a.sum() * S_b.sum()
    if Z < 1e-12:
        return 0.0
    return float(np.sum(weight_matrix * cos_matrix) / Z)


def detect_module_type(layer_name: str) -> str:
    """Detect Q/K/V/O from layer name."""
    if "q_lin" in layer_name or "query" in layer_name:
        return "q"
    elif "k_lin" in layer_name or "key" in layer_name:
        return "k"
    elif "v_lin" in layer_name or "value" in layer_name:
        return "v"
    elif "out_lin" in layer_name or "output" in layer_name:
        return "o"
    return "unknown"


# ---------------------------------------------------------------------------
# Spectral susceptibility computation
# ---------------------------------------------------------------------------

def compute_energy_concentration(S: np.ndarray, k: int) -> float:
    """Energy concentration C_k = sum(S[:k]^2) / sum(S^2)."""
    total = np.sum(S**2)
    if total < 1e-20:
        return 0.0
    return float(np.sum(S[:k]**2) / total)


def compute_spectral_susceptibility(S: np.ndarray, k: int, eps_reg: float = None) -> float:
    """Spectral susceptibility Gamma_k.

    Gamma_k = (1/E_k) * sum_{i=1}^{k} sigma_i^2 / delta_i^2

    where E_k = sum_{i=1}^{k} sigma_i^2 (signal energy)
    and delta_i = min_{j != i} |sigma_i - sigma_j| (nearest-neighbor gap).

    Small Gamma_k = robust (large gaps relative to energy).
    Large Gamma_k = susceptible (small gaps, near-degenerate).
    """
    if k <= 0 or len(S) < 2:
        return float("inf")

    if eps_reg is None:
        eps_reg = 1e-6 * S[0]

    E_k = np.sum(S[:k]**2)
    if E_k < 1e-20:
        return float("inf")

    n_regularized = 0
    gamma_sum = 0.0
    for i in range(k):
        # Nearest-neighbor gap: min distance to any other SV
        gaps = np.abs(S[i] - S)
        gaps[i] = np.inf  # exclude self
        delta_i = gaps.min()

        # Regularize near-degenerate gaps
        if delta_i < eps_reg:
            delta_i = eps_reg
            n_regularized += 1

        gamma_sum += S[i]**2 / delta_i**2

    gamma_k = gamma_sum / E_k
    return float(gamma_k), n_regularized


def compute_spectral_susceptibility_safe(S: np.ndarray, k: int, eps_reg: float = None):
    """Returns (gamma_k, n_regularized)."""
    result = compute_spectral_susceptibility(S, k, eps_reg)
    if isinstance(result, tuple):
        return result
    return result, 0


# ---------------------------------------------------------------------------
# Phase 1: W0 spectral properties
# ---------------------------------------------------------------------------

def phase1_w0_properties() -> dict:
    """Compute C_k and Gamma_k for each LoRA-targeted layer of DistilBERT."""
    print("\n=== Phase 1: W0 Spectral Properties ===\n")

    from transformers import AutoModel
    model = AutoModel.from_pretrained(BASE_MODEL_NAME)

    # Extract attention weight matrices
    w0_data = {}
    for name, param in model.named_parameters():
        if "attention" in name and name.endswith(".weight"):
            W = param.detach().cpu().numpy().astype(np.float64)
            # Match the key format used by adapters and N127 data
            # AutoModel names: "transformer.layer.0.attention.q_lin.weight"
            # Adapter/N127 names: "distilbert.transformer.layer.0.attention.q_lin"
            key = name.replace(".weight", "")
            if not key.startswith("distilbert."):
                key = "distilbert." + key
            w0_data[key] = W

    print(f"  Loaded {len(w0_data)} attention weight matrices")

    results = []
    total_regularized = 0

    for layer_name, W in sorted(w0_data.items()):
        S = svd(W, compute_uv=False)
        shape = W.shape
        tau, k = mp_threshold(S, shape)
        C_k = compute_energy_concentration(S, k)
        Gamma_k, n_reg = compute_spectral_susceptibility_safe(S, k)
        total_regularized += n_reg

        module = detect_module_type(layer_name)

        results.append({
            "layer": layer_name,
            "module": module,
            "shape": list(shape),
            "k": k,
            "tau": tau,
            "C_k": C_k,
            "Gamma_k": Gamma_k,
            "inv_Gamma_k": 1.0 / Gamma_k if Gamma_k > 0 and Gamma_k != float("inf") else 0.0,
            "n_regularized": n_reg,
            "sigma_1": float(S[0]),
            "sigma_k": float(S[k-1]) if k > 0 else 0.0,
            "sigma_k1": float(S[k]) if k < len(S) else 0.0,
            "singular_values": S[:min(20, len(S))].tolist(),  # store top 20
        })

        print(f"  {layer_name[-45:]}: k={k} C_k={C_k:.4f} "
              f"Gamma_k={Gamma_k:.2f} 1/Gamma_k={1/Gamma_k:.4f}" +
              (f" ({n_reg} reg)" if n_reg > 0 else ""))

    print(f"\n  Total regularized gaps: {total_regularized}")

    del model
    return {"per_layer": results, "total_regularized": total_regularized}


# ---------------------------------------------------------------------------
# Phase 2: Per-layer alignment (load from N127 cache)
# ---------------------------------------------------------------------------

def phase2_load_alignment() -> dict:
    """Load per-layer alignment from N127 extension results."""
    print("\n=== Phase 2: Load Per-Layer Alignment ===\n")

    ext_file = RESULTS_DIR / "extension_results.json"
    ext_data = json.loads(ext_file.read_text())

    ext1 = ext_data["extension_1_spectral_gap"]
    sst2_per_layer = ext1["per_layer"]
    qnli_per_layer = ext1["qnli"]["per_layer"]

    print(f"  SST-2: {len(sst2_per_layer)} layers with alignment data")
    print(f"  QNLI:  {len(qnli_per_layer)} layers with alignment data")

    return {
        "sst2": sst2_per_layer,
        "qnli": qnli_per_layer,
    }


# ---------------------------------------------------------------------------
# Phase 3: Prediction 1 -- Static correlation
# ---------------------------------------------------------------------------

def phase3_prediction_1(w0_props: dict, alignment_data: dict) -> dict:
    """Test: Gamma_k outperforms C_k as per-layer alignment predictor."""
    print("\n=== Phase 3: Prediction 1 -- Gamma_k vs C_k ===\n")

    w0_by_layer = {r["layer"]: r for r in w0_props["per_layer"]}

    results = {}
    for task in ["sst2", "qnli"]:
        per_layer = alignment_data[task]

        # Match layers between W0 data and alignment data
        C_k_vals = []
        inv_Gamma_vals = []
        align_vals = []
        modules = []

        for entry in per_layer:
            layer_name = entry["layer"]
            if layer_name not in w0_by_layer:
                continue
            w0 = w0_by_layer[layer_name]

            C_k_vals.append(w0["C_k"])
            inv_Gamma_vals.append(w0["inv_Gamma_k"])
            align_vals.append(entry["mean_high_align"])
            modules.append(entry["module"])

        C_k_arr = np.array(C_k_vals)
        inv_Gamma_arr = np.array(inv_Gamma_vals)
        align_arr = np.array(align_vals)

        n = len(C_k_arr)
        print(f"  {task.upper()}: {n} matched layers")

        # Spearman correlations
        rho_C, p_C = stats.spearmanr(C_k_arr, align_arr)
        rho_Gamma, p_Gamma = stats.spearmanr(inv_Gamma_arr, align_arr)

        print(f"    rho(C_k, align)     = {rho_C:+.4f}  (p = {p_C:.4f})")
        print(f"    rho(1/Gamma_k, align) = {rho_Gamma:+.4f}  (p = {p_Gamma:.4f})")

        diff = abs(rho_Gamma) - abs(rho_C)
        print(f"    |rho_Gamma| - |rho_C| = {diff:+.4f}")

        # Bootstrap 95% CI for the difference
        n_boot = 10000
        boot_diffs = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            r_c, _ = stats.spearmanr(C_k_arr[idx], align_arr[idx])
            r_g, _ = stats.spearmanr(inv_Gamma_arr[idx], align_arr[idx])
            boot_diffs.append(abs(r_g) - abs(r_c))

        boot_diffs = np.array(boot_diffs)
        ci_lo = float(np.percentile(boot_diffs, 2.5))
        ci_hi = float(np.percentile(boot_diffs, 97.5))
        print(f"    Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

        # Decision
        if diff >= 0.10 and ci_lo > 0:
            decision = "CONFIRMED"
        elif diff <= -0.10 and ci_hi < 0:
            decision = "DISCONFIRMED"
        else:
            decision = "EQUIVALENT"
        print(f"    Decision: {decision}")

        # Per-module breakdown
        module_results = {}
        for m in ["q", "k", "v", "o"]:
            mask = np.array([mod == m for mod in modules])
            if mask.sum() < 3:
                continue
            r_c_m, p_c_m = stats.spearmanr(C_k_arr[mask], align_arr[mask])
            r_g_m, p_g_m = stats.spearmanr(inv_Gamma_arr[mask], align_arr[mask])
            module_results[m] = {
                "rho_C": float(r_c_m), "p_C": float(p_c_m),
                "rho_Gamma": float(r_g_m), "p_Gamma": float(p_g_m),
                "diff": float(abs(r_g_m) - abs(r_c_m)),
                "n": int(mask.sum()),
            }
            print(f"    Module {m.upper()}: rho_C={r_c_m:+.4f} rho_Gamma={r_g_m:+.4f} "
                  f"diff={abs(r_g_m)-abs(r_c_m):+.4f} (n={mask.sum()})")

        results[task] = {
            "n_layers": n,
            "rho_C": float(rho_C), "p_C": float(p_C),
            "rho_Gamma": float(rho_Gamma), "p_Gamma": float(p_Gamma),
            "diff": float(diff),
            "bootstrap_ci": [ci_lo, ci_hi],
            "decision": decision,
            "per_module": module_results,
            "per_layer": [
                {"layer": per_layer[i]["layer"], "module": modules[i],
                 "C_k": float(C_k_arr[i]), "inv_Gamma_k": float(inv_Gamma_arr[i]),
                 "alignment": float(align_arr[i])}
                for i in range(n)
            ],
        }

    # Overall decision
    decisions = [results[t]["decision"] for t in results]
    if "CONFIRMED" in decisions:
        overall = "CONFIRMED"
    elif "DISCONFIRMED" in decisions:
        overall = "DISCONFIRMED"
    else:
        overall = "EQUIVALENT"

    results["overall_decision"] = overall
    print(f"\n  P1 Overall: {overall}")

    return results


# ---------------------------------------------------------------------------
# Phase 4: Prediction 2 -- Dynamic trajectory
# ---------------------------------------------------------------------------

def phase4_prediction_2(w0_props: dict) -> dict:
    """Test: epsilon_t * sqrt(Gamma_k(t)) decreases during training."""
    print("\n=== Phase 4: Prediction 2 -- Dynamic Trajectory ===\n")

    from transformers import AutoModel
    model = AutoModel.from_pretrained(BASE_MODEL_NAME)

    # Get W0 weights (normalize to distilbert.* format to match adapter keys)
    w0_weights = {}
    for name, param in model.named_parameters():
        if "attention" in name and name.endswith(".weight"):
            key = name.replace(".weight", "")
            if not key.startswith("distilbert."):
                key = "distilbert." + key
            w0_weights[key] = param.detach().cpu().numpy().astype(np.float64)
    del model

    steps = [50, 100, 150, 200]
    all_trajectories = {}

    for seed in [42, 123, 456]:
        seed_trajectories = {}

        for step in steps:
            adapter_dir = SST2_CHECKPOINT_DIRS[seed][step]
            if not adapter_dir.exists():
                print(f"  SKIP seed={seed} step={step} (not found)")
                continue

            adapter_weights = load_adapter_weights(adapter_dir)

            for layer_name, layer_data in adapter_weights.items():
                A = layer_data["A"]
                B = layer_data["B"]
                scaling = layer_data["scaling"]

                delta_W = scaling * B @ A
                epsilon_t = float(np.sqrt(np.sum(delta_W**2)))  # Frobenius norm

                # Adapted matrix W_t = W0 + delta_W
                w0_key = layer_name
                if w0_key not in w0_weights:
                    continue
                W_t = w0_weights[w0_key] + delta_W

                # SVD of adapted matrix
                S_t = svd(W_t, compute_uv=False)
                _, k_t = mp_threshold(S_t, W_t.shape)
                Gamma_k_t, _ = compute_spectral_susceptibility_safe(S_t, k_t)
                product_t = epsilon_t * np.sqrt(Gamma_k_t) if Gamma_k_t < float("inf") else float("inf")

                key = (seed, layer_name)
                if key not in seed_trajectories:
                    seed_trajectories[key] = []
                seed_trajectories[key].append({
                    "step": step,
                    "epsilon_t": epsilon_t,
                    "Gamma_k_t": Gamma_k_t,
                    "sqrt_Gamma_k_t": float(np.sqrt(Gamma_k_t)) if Gamma_k_t < float("inf") else float("inf"),
                    "product": product_t,
                    "k_t": k_t,
                })

        all_trajectories[seed] = seed_trajectories

    # Analyze: for each layer, test if product decreases monotonically
    layer_results = {}
    monotonic_count = 0
    total_layers = 0
    all_rho_values = []

    for seed in [42, 123, 456]:
        for (s, layer_name), trajectory in all_trajectories.get(seed, {}).items():
            if s != seed:
                continue
            if len(trajectory) < 3:
                continue

            trajectory.sort(key=lambda x: x["step"])
            steps_arr = [t["step"] for t in trajectory]
            products = [t["product"] for t in trajectory]
            epsilons = [t["epsilon_t"] for t in trajectory]
            sqrt_gammas = [t["sqrt_Gamma_k_t"] for t in trajectory]

            # Check if any products are inf
            if any(p == float("inf") for p in products):
                continue

            rho, p_val = stats.spearmanr(steps_arr, products)
            is_decreasing = rho < 0

            # Count reversals (non-monotonic steps)
            reversals = sum(1 for i in range(1, len(products)) if products[i] > products[i-1])

            total_layers += 1
            if is_decreasing:
                monotonic_count += 1
            all_rho_values.append(rho)

            layer_key = f"seed{seed}_{layer_name}"
            layer_results[layer_key] = {
                "seed": seed,
                "layer": layer_name,
                "steps": steps_arr,
                "products": products,
                "epsilons": epsilons,
                "sqrt_gammas": sqrt_gammas,
                "rho_with_step": float(rho),
                "p_value": float(p_val),
                "reversals": reversals,
                "is_decreasing": is_decreasing,
            }

    fraction_decreasing = monotonic_count / total_layers if total_layers > 0 else 0
    mean_rho = float(np.mean(all_rho_values)) if all_rho_values else 0

    print(f"  Layers analyzed: {total_layers}")
    print(f"  Fraction decreasing: {monotonic_count}/{total_layers} = {fraction_decreasing:.2%}")
    print(f"  Mean Spearman rho (product vs step): {mean_rho:+.4f}")

    # Decision
    if fraction_decreasing >= 0.75:
        decision = "CONFIRMED"
    elif mean_rho < 0:
        decision = "PARTIAL"
    else:
        decision = "DISCONFIRMED"

    print(f"  Decision: {decision}")

    # Also check component trajectories
    # Does epsilon increase while sqrt(Gamma) decreases?
    eps_increasing = sum(1 for v in layer_results.values()
                         if len(v["epsilons"]) >= 2 and v["epsilons"][-1] > v["epsilons"][0])
    gamma_decreasing = sum(1 for v in layer_results.values()
                           if len(v["sqrt_gammas"]) >= 2 and
                           v["sqrt_gammas"][-1] < v["sqrt_gammas"][0] and
                           v["sqrt_gammas"][-1] != float("inf"))

    print(f"  Epsilon increases: {eps_increasing}/{total_layers}")
    print(f"  sqrt(Gamma) decreases: {gamma_decreasing}/{total_layers}")

    return {
        "fraction_decreasing": fraction_decreasing,
        "mean_rho": mean_rho,
        "total_layers": total_layers,
        "n_decreasing": monotonic_count,
        "eps_increasing": eps_increasing,
        "gamma_decreasing": gamma_decreasing,
        "decision": decision,
        "per_layer": layer_results,
    }


# ---------------------------------------------------------------------------
# Phase 5: Prediction 3 -- Task asymmetry
# ---------------------------------------------------------------------------

def phase5_prediction_3() -> dict:
    """Test: QNLI adapters have higher effective rank than SST-2."""
    print("\n=== Phase 5: Prediction 3 -- Task Asymmetry ===\n")

    def entropy_effective_rank(S: np.ndarray) -> float:
        """Entropy effective rank = exp(H) where H = -sum(p * log(p))."""
        S2 = S**2
        total = np.sum(S2)
        if total < 1e-20:
            return 0.0
        p = S2 / total
        p = p[p > 1e-20]  # avoid log(0)
        H = -np.sum(p * np.log(p))
        return float(np.exp(H))

    # Load SST-2 converged adapters
    sst2_eranks_by_layer: dict[str, list] = {}
    for adapter_dir in SST2_CONVERGED_DIRS:
        if not adapter_dir.exists():
            continue
        weights = load_adapter_weights(adapter_dir)
        for layer_name, layer_data in weights.items():
            U, S, Vt = compute_layer_svd(layer_data["A"], layer_data["B"], layer_data["scaling"])
            erank = entropy_effective_rank(S)
            if layer_name not in sst2_eranks_by_layer:
                sst2_eranks_by_layer[layer_name] = []
            sst2_eranks_by_layer[layer_name].append(erank)

    # Load QNLI converged adapters
    qnli_eranks_by_layer: dict[str, list] = {}
    if QNLI_CONVERGED_DIR.exists():
        weights = load_adapter_weights(QNLI_CONVERGED_DIR)
        for layer_name, layer_data in weights.items():
            U, S, Vt = compute_layer_svd(layer_data["A"], layer_data["B"], layer_data["scaling"])
            erank = entropy_effective_rank(S)
            if layer_name not in qnli_eranks_by_layer:
                qnli_eranks_by_layer[layer_name] = []
            qnli_eranks_by_layer[layer_name].append(erank)

    # Also use checkpoint-50 SST-2 adapters from uniform_r16 for more data
    for seed in [42, 123, 456]:
        adapter_dir = SST2_CHECKPOINT_DIRS[seed][200]  # converged
        if not adapter_dir.exists():
            continue
        weights = load_adapter_weights(adapter_dir)
        for layer_name, layer_data in weights.items():
            U, S, Vt = compute_layer_svd(layer_data["A"], layer_data["B"], layer_data["scaling"])
            erank = entropy_effective_rank(S)
            if layer_name not in sst2_eranks_by_layer:
                sst2_eranks_by_layer[layer_name] = []
            sst2_eranks_by_layer[layer_name].append(erank)

    # Compute per-layer means
    shared_layers = sorted(set(sst2_eranks_by_layer.keys()) & set(qnli_eranks_by_layer.keys()))
    print(f"  Shared layers: {len(shared_layers)}")
    print(f"  SST-2 adapters per layer: {[len(v) for v in sst2_eranks_by_layer.values()][:3]}")
    print(f"  QNLI adapters per layer: {[len(v) for v in qnli_eranks_by_layer.values()][:3]}")

    sst2_means = []
    qnli_means = []
    per_layer_data = []

    for layer in shared_layers:
        sst2_mean = np.mean(sst2_eranks_by_layer[layer])
        qnli_mean = np.mean(qnli_eranks_by_layer[layer])
        sst2_means.append(sst2_mean)
        qnli_means.append(qnli_mean)
        per_layer_data.append({
            "layer": layer,
            "module": detect_module_type(layer),
            "sst2_erank": float(sst2_mean),
            "qnli_erank": float(qnli_mean),
            "qnli_gt_sst2": qnli_mean > sst2_mean,
        })
        print(f"  {layer[-40:]}: SST-2={sst2_mean:.3f} QNLI={qnli_mean:.3f} "
              f"{'QNLI>' if qnli_mean > sst2_mean else 'SST2>'}")

    sst2_arr = np.array(sst2_means)
    qnli_arr = np.array(qnli_means)

    # Paired t-test
    if len(shared_layers) >= 2:
        t_stat, p_val = stats.ttest_rel(qnli_arr, sst2_arr)
        # Cohen's d
        diff = qnli_arr - sst2_arr
        d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0
    else:
        t_stat, p_val, d = 0, 1, 0

    mean_sst2 = float(np.mean(sst2_arr))
    mean_qnli = float(np.mean(qnli_arr))

    print(f"\n  Mean erank: SST-2={mean_sst2:.3f} QNLI={mean_qnli:.3f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"  Cohen's d: {d:.3f}")

    fraction_higher = sum(1 for d_i in per_layer_data if d_i["qnli_gt_sst2"]) / len(per_layer_data) if per_layer_data else 0

    # Decision
    if p_val < 0.05 and mean_qnli > mean_sst2:
        decision = "CONFIRMED"
    elif p_val < 0.20 and mean_qnli > mean_sst2:
        decision = "SUGGESTIVE"
    else:
        decision = "DISCONFIRMED"

    print(f"  Decision: {decision}")

    return {
        "mean_sst2_erank": mean_sst2,
        "mean_qnli_erank": mean_qnli,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(d),
        "fraction_qnli_higher": fraction_higher,
        "n_layers": len(shared_layers),
        "decision": decision,
        "per_layer": per_layer_data,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  N130: Spectral Susceptibility (Gamma_k) Validation")
    print("=" * 70)

    # Phase 1: W0 spectral properties
    w0_props = phase1_w0_properties()
    (OUTPUT_DIR / "phase1_w0_properties.json").write_text(
        json.dumps(w0_props, indent=2, default=str))

    # Phase 2: Load alignment data
    alignment_data = phase2_load_alignment()

    # Phase 3: Prediction 1
    p1_results = phase3_prediction_1(w0_props, alignment_data)
    (OUTPUT_DIR / "phase3_prediction_1.json").write_text(
        json.dumps(p1_results, indent=2, default=str))

    # Phase 4: Prediction 2
    p2_results = phase4_prediction_2(w0_props)
    (OUTPUT_DIR / "phase4_prediction_2.json").write_text(
        json.dumps(p2_results, indent=2, default=str))

    # Phase 5: Prediction 3
    p3_results = phase5_prediction_3()
    (OUTPUT_DIR / "phase5_prediction_3.json").write_text(
        json.dumps(p3_results, indent=2, default=str))

    # Summary
    elapsed = time.time() - t0

    summary = {
        "experiment": "N130: Spectral Susceptibility Validation",
        "elapsed_s": elapsed,
        "prediction_1": {
            "description": "Gamma_k outperforms C_k as alignment predictor",
            "overall_decision": p1_results["overall_decision"],
            "sst2": {"rho_C": p1_results["sst2"]["rho_C"],
                     "rho_Gamma": p1_results["sst2"]["rho_Gamma"],
                     "diff": p1_results["sst2"]["diff"],
                     "decision": p1_results["sst2"]["decision"]},
            "qnli": {"rho_C": p1_results["qnli"]["rho_C"],
                     "rho_Gamma": p1_results["qnli"]["rho_Gamma"],
                     "diff": p1_results["qnli"]["diff"],
                     "decision": p1_results["qnli"]["decision"]},
        },
        "prediction_2": {
            "description": "epsilon_t * sqrt(Gamma_k(t)) decreases during training",
            "decision": p2_results["decision"],
            "fraction_decreasing": p2_results["fraction_decreasing"],
            "mean_rho": p2_results["mean_rho"],
        },
        "prediction_3": {
            "description": "QNLI effective rank > SST-2 effective rank",
            "decision": p3_results["decision"],
            "mean_sst2": p3_results["mean_sst2_erank"],
            "mean_qnli": p3_results["mean_qnli_erank"],
            "p_value": p3_results["p_value"],
        },
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Print results table
    print(f"\n{'=' * 70}")
    print(f"  N130 Results Summary ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print()
    print("  | Prediction | Metric              | SST-2           | QNLI            | Decision     |")
    print("  |------------|---------------------|-----------------|-----------------|--------------|")
    print(f"  | P1: Gamma  | rho(1/Gamma, align) | {p1_results['sst2']['rho_Gamma']:+.4f} (p={p1_results['sst2']['p_Gamma']:.3f}) "
          f"| {p1_results['qnli']['rho_Gamma']:+.4f} (p={p1_results['qnli']['p_Gamma']:.3f}) "
          f"| {p1_results['overall_decision']:<12} |")
    print(f"  | P1: C_k    | rho(C_k, align)     | {p1_results['sst2']['rho_C']:+.4f} (p={p1_results['sst2']['p_C']:.3f}) "
          f"| {p1_results['qnli']['rho_C']:+.4f} (p={p1_results['qnli']['p_C']:.3f}) "
          f"|              |")
    print(f"  | P1: diff   | |rho_G| - |rho_C|   | {p1_results['sst2']['diff']:+.4f}          "
          f"| {p1_results['qnli']['diff']:+.4f}          "
          f"|              |")
    ci_s = p1_results["sst2"]["bootstrap_ci"]
    ci_q = p1_results["qnli"]["bootstrap_ci"]
    print(f"  | P1: CI     | bootstrap 95%       | [{ci_s[0]:+.3f},{ci_s[1]:+.3f}]     "
          f"| [{ci_q[0]:+.3f},{ci_q[1]:+.3f}]     "
          f"|              |")
    print(f"  | P2: trend  | frac decreasing     | {p2_results['fraction_decreasing']:.2%}          "
          f"|                 "
          f"| {p2_results['decision']:<12} |")
    print(f"  | P2: rho    | Spearman w/ step    | {p2_results['mean_rho']:+.4f}          "
          f"|                 "
          f"|              |")
    print(f"  | P3: erank  | QNLI vs SST-2       | {p3_results['mean_sst2_erank']:.3f}           "
          f"| {p3_results['mean_qnli_erank']:.3f}           "
          f"| {p3_results['decision']:<12} |")
    print(f"  |            | paired t, p, d       | t={p3_results['t_stat']:.2f} p={p3_results['p_value']:.4f} d={p3_results['cohens_d']:.2f}")
    print()
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
