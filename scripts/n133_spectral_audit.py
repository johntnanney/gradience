"""N133 Phase 2: Spectral audit of Mistral-7B adapters.

Computes per-adapter SVD metrics (erank, stable rank, C_k), W₀ spectral
properties, and pairwise SV-weighted alignment for all 66 adapter pairs.

Usage:
    python3 n133_spectral_audit.py

Output: /workspace/n133/audits/
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.linalg import svd
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
CACHE_DIR = "/workspace/hf_cache"
ADAPTER_ROOT = Path("/workspace/n133/adapters")
OUTPUT_DIR = Path("/workspace/n133/audits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["sst2", "mnli", "squad", "gsm8k", "code", "summarization"]
SEEDS = [42, 123]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def entropy_effective_rank(S: np.ndarray) -> float:
    """Entropy effective rank = exp(H) where H = -sum(p * log(p))."""
    S2 = S**2
    total = np.sum(S2)
    if total < 1e-20:
        return 0.0
    p = S2 / total
    p = p[p > 1e-20]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def stable_rank(S: np.ndarray) -> float:
    """Stable rank = ||A||_F^2 / ||A||_2^2 = sum(S^2) / S[0]^2."""
    if len(S) == 0 or S[0] < 1e-20:
        return 0.0
    return float(np.sum(S**2) / S[0]**2)


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


def energy_concentration(S: np.ndarray, k: int) -> float:
    """C_k = sum(S[:k]^2) / sum(S^2)."""
    total = np.sum(S**2)
    if total < 1e-20:
        return 0.0
    return float(np.sum(S[:k]**2) / total)


def sv_weighted_alignment(U_a, S_a, U_b, S_b) -> float:
    """SV-weighted alignment: A = (1/Z) sum_ij sigma_i*sigma_j*|cos(u_i, u_j)|."""
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return 0.0
    cos_matrix = np.abs(U_a.T @ U_b)
    weight_matrix = np.outer(S_a, S_b)
    Z = S_a.sum() * S_b.sum()
    if Z < 1e-12:
        return 0.0
    return float(np.sum(weight_matrix * cos_matrix) / Z)


def fast_svd_delta_W(A: np.ndarray, B: np.ndarray, scaling: float = 1.0):
    """QR-based fast SVD for low-rank delta_W = scaling * B @ A."""
    B_scaled = scaling * B
    Q_B, R_B = np.linalg.qr(B_scaled, mode="reduced")
    M = R_B @ A
    U_small, S, Vt = svd(M, full_matrices=False)
    U = Q_B @ U_small
    return U, S, Vt


def detect_module_type(layer_name: str) -> str:
    """Detect Q/K/V/O from Mistral layer name."""
    if "q_proj" in layer_name:
        return "Q"
    elif "k_proj" in layer_name:
        return "K"
    elif "v_proj" in layer_name:
        return "V"
    elif "o_proj" in layer_name:
        return "O"
    return "unknown"


def extract_layer_idx(layer_name: str) -> int:
    """Extract transformer layer index from parameter name."""
    parts = layer_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


# ---------------------------------------------------------------------------
# Load adapter weights
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
        layer_keys: dict[str, dict] = {}
        for k in keys:
            if "lora_A" in k:
                layer_name = k.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
                for prefix in ["base_model.model.", "base_model."]:
                    if layer_name.startswith(prefix):
                        layer_name = layer_name[len(prefix):]
                if layer_name not in layer_keys:
                    layer_keys[layer_name] = {}
                layer_keys[layer_name]["A"] = f.get_tensor(k).astype(np.float64)
            elif "lora_B" in k:
                layer_name = k.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
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


# ---------------------------------------------------------------------------
# Phase 2a: Per-adapter spectral profiles
# ---------------------------------------------------------------------------

def audit_adapters() -> dict:
    """Compute per-layer SVD metrics for each adapter."""
    print("\n=== Phase 2a: Per-Adapter Spectral Audit ===\n")

    all_profiles = {}

    for task in TASKS:
        for seed in SEEDS:
            adapter_name = f"{task}_s{seed}"
            adapter_dir = ADAPTER_ROOT / adapter_name

            if not (adapter_dir / "adapter_model.safetensors").exists():
                print(f"  SKIP {adapter_name} — not found")
                continue

            print(f"  Auditing {adapter_name}...")
            weights = load_adapter_weights(adapter_dir)

            profiles = []
            for layer_name in sorted(weights.keys()):
                lw = weights[layer_name]
                U, S, Vt = fast_svd_delta_W(lw["A"], lw["B"], lw["scaling"])

                layer_idx = extract_layer_idx(layer_name)
                module = detect_module_type(layer_name)

                profiles.append({
                    "layer_name": layer_name,
                    "layer": layer_idx,
                    "module": module,
                    "erank": entropy_effective_rank(S),
                    "stable_rank": stable_rank(S),
                    "frobenius_norm": float(np.sqrt(np.sum(S**2))),
                    "sigma_max": float(S[0]) if len(S) > 0 else 0.0,
                    "singular_values": S[:min(16, len(S))].tolist(),
                })

            all_profiles[adapter_name] = {
                "task": task,
                "seed": seed,
                "n_layers": len(profiles),
                "mean_erank": float(np.mean([p["erank"] for p in profiles])),
                "mean_stable_rank": float(np.mean([p["stable_rank"] for p in profiles])),
                "per_layer": profiles,
            }

            print(f"    {len(profiles)} layers, mean erank = {all_profiles[adapter_name]['mean_erank']:.2f}")

    return all_profiles


# ---------------------------------------------------------------------------
# Phase 2b: W₀ spectral properties
# ---------------------------------------------------------------------------

def audit_w0() -> dict:
    """Compute C_k for each LoRA-targeted layer of Mistral-7B W₀."""
    print("\n=== Phase 2b: Mistral W₀ Spectral Properties ===\n")

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        dtype=torch.float32,
        device_map="cpu",
    )

    w0_data = {}
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if not any(m in name for m in target_modules):
            continue
        # Only self_attn layers
        if "self_attn" not in name:
            continue

        # Strip model. prefix if present
        clean_name = name.replace(".weight", "")
        if clean_name.startswith("model."):
            clean_name = clean_name[len("model."):]

        W = param.detach().cpu().numpy().astype(np.float64)
        S = svd(W, compute_uv=False)
        shape = W.shape
        tau, k = mp_threshold(S, shape)
        C_k = energy_concentration(S, k)

        layer_idx = extract_layer_idx(clean_name)
        module = detect_module_type(clean_name)

        w0_data[clean_name] = {
            "layer": layer_idx,
            "module": module,
            "shape": list(shape),
            "k": k,
            "tau": tau,
            "C_k": C_k,
            "sigma_1": float(S[0]),
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Computed C_k for {len(w0_data)} layers")
    ck_vals = [v["C_k"] for v in w0_data.values()]
    if ck_vals:
        print(f"  C_k range: [{min(ck_vals):.3f}, {max(ck_vals):.3f}]")

    return w0_data


# ---------------------------------------------------------------------------
# Phase 2c: Pairwise SV-weighted alignment
# ---------------------------------------------------------------------------

def compute_pairwise_alignment(adapter_profiles: dict) -> dict:
    """Compute per-layer SV-weighted alignment for all pairs."""
    print("\n=== Phase 2c: Pairwise Alignment ===\n")

    # Cache adapter SVD data
    adapter_svd = {}
    for adapter_name, _profile in adapter_profiles.items():
        adapter_dir = ADAPTER_ROOT / adapter_name
        weights = load_adapter_weights(adapter_dir)
        svd_data = {}
        for layer_name, lw in weights.items():
            U, S, Vt = fast_svd_delta_W(lw["A"], lw["B"], lw["scaling"])
            svd_data[layer_name] = {"U": U, "S": S, "Vt": Vt}
        adapter_svd[adapter_name] = svd_data

    adapter_names = sorted(adapter_profiles.keys())
    pair_results = {}
    n_pairs = 0

    for a_name, b_name in combinations(adapter_names, 2):
        task_a = adapter_profiles[a_name]["task"]
        task_b = adapter_profiles[b_name]["task"]
        is_same_task = task_a == task_b

        svd_a = adapter_svd[a_name]
        svd_b = adapter_svd[b_name]

        shared_layers = sorted(set(svd_a.keys()) & set(svd_b.keys()))
        per_layer = []

        for layer_name in shared_layers:
            da = svd_a[layer_name]
            db = svd_b[layer_name]
            align = sv_weighted_alignment(da["U"], da["S"], db["U"], db["S"])

            per_layer.append({
                "layer_name": layer_name,
                "layer": extract_layer_idx(layer_name),
                "module": detect_module_type(layer_name),
                "alignment": align,
            })

        mean_align = float(np.mean([p["alignment"] for p in per_layer])) if per_layer else 0.0

        pair_key = f"{a_name}_vs_{b_name}"
        pair_results[pair_key] = {
            "adapter_a": a_name,
            "adapter_b": b_name,
            "task_a": task_a,
            "task_b": task_b,
            "is_same_task": is_same_task,
            "mean_alignment": mean_align,
            "n_layers": len(per_layer),
            "per_layer": per_layer,
        }
        n_pairs += 1

    # Summary stats
    same_task = [v for v in pair_results.values() if v["is_same_task"]]
    cross_task = [v for v in pair_results.values() if not v["is_same_task"]]

    same_mean = np.mean([v["mean_alignment"] for v in same_task]) if same_task else 0
    cross_mean = np.mean([v["mean_alignment"] for v in cross_task]) if cross_task else 0

    print(f"  Total pairs: {n_pairs}")
    print(f"  Same-task: {len(same_task)} pairs, mean alignment = {same_mean:.4f}")
    print(f"  Cross-task: {len(cross_task)} pairs, mean alignment = {cross_mean:.4f}")
    if cross_mean > 0:
        print(f"  Same/cross ratio: {same_mean/cross_mean:.2f}×")

    return pair_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  N133 Phase 2: Spectral Audit")
    print("=" * 60)

    t0 = time.time()

    # Phase 2a: Per-adapter profiles (skip if already done)
    adapter_profiles_path = OUTPUT_DIR / "adapter_profiles.json"
    if adapter_profiles_path.exists():
        print(f"\n[resume] Loading existing {adapter_profiles_path}")
        with open(adapter_profiles_path) as f:
            adapter_profiles = json.load(f)
        print(f"[resume] Loaded {len(adapter_profiles)} adapter profiles")
    else:
        adapter_profiles = audit_adapters()
        with open(adapter_profiles_path, "w") as f:
            json.dump(adapter_profiles, f, indent=2)

    # Phase 2b: W₀ properties (skip if already done)
    w0_path = OUTPUT_DIR / "w0_properties.json"
    if w0_path.exists():
        print(f"\n[resume] Loading existing {w0_path}")
        with open(w0_path) as f:
            w0_data = json.load(f)
    else:
        w0_data = audit_w0()
        with open(w0_path, "w") as f:
            json.dump(w0_data, f, indent=2)

    # Phase 2c: Pairwise alignment (skip if already done)
    pair_full_path = OUTPUT_DIR / "pair_alignment_full.json"
    if pair_full_path.exists():
        print(f"\n[resume] Skipping Phase 2c — {pair_full_path} already exists")
    else:
        pair_results = compute_pairwise_alignment(adapter_profiles)

        # Save pair results (without per-layer detail for compactness)
        pair_summary = {}
        for k, v in pair_results.items():
            pair_summary[k] = {key: val for key, val in v.items() if key != "per_layer"}
        with open(OUTPUT_DIR / "pair_alignment_summary.json", "w") as f:
            json.dump(pair_summary, f, indent=2)

        # Save full pair results
        with open(pair_full_path, "w") as f:
            json.dump(pair_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Total audit time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
