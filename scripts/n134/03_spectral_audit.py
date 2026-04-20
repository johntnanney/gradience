"""N134 Phase 2: Spectral audit with v2.1 schema (U/V factor persistence).

Computes per-adapter SVD metrics, persists U/V factors for downstream
pairwise alignment, and computes SV-weighted alignment for all C(24,2)=276
adapter pairs.

Usage:
    python3 03_spectral_audit.py                   # full audit
    python3 03_spectral_audit.py --smoke            # first 2 adapters only

Output: /workspace/n134/audit/
    {adapter_id}_v2_1.json        -- per-adapter v2.1 schema (24 files)
    adapter_profiles.json         -- summary profiles
    pair_alignment_full.json      -- all 276 pairs with per-layer detail
    pair_alignment_summary.json   -- pair summaries (no per-layer)
    w0_properties.json            -- W0 spectral properties (C_k etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.linalg import svd
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
CACHE_DIR = "/workspace/hf_cache"
ADAPTER_ROOT = Path("/workspace/n134/adapters")
OUTPUT_DIR = Path("/workspace/n134/audit")

TASKS = [
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "openbookqa",
    "commonsenseqa",
    "piqa",
    "siqa",
    "boolq",
]

SEEDS = [42, 123, 456]

N_LAYERS = 32
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ---------------------------------------------------------------------------
# Spectral utility functions
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


def stable_rank_fn(S: np.ndarray) -> float:
    """Stable rank = ||A||_F^2 / ||A||_2^2 = sum(S^2) / S[0]^2."""
    if len(S) == 0 or S[0] < 1e-20:
        return 0.0
    return float(np.sum(S**2) / S[0]**2)


def energy_rank_90(S: np.ndarray) -> int:
    """Rank at 90% cumulative energy."""
    S2 = S**2
    total = np.sum(S2)
    if total < 1e-20:
        return 0
    cumsum = np.cumsum(S2)
    threshold = 0.90 * total
    idx = int(np.searchsorted(cumsum, threshold))
    return min(idx + 1, len(S))


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


def sv_weighted_alignment(U_a: np.ndarray, S_a: np.ndarray, U_b: np.ndarray, S_b: np.ndarray) -> float:
    """SV-weighted alignment: A = (1/Z) sum_ij sigma_i*sigma_j*|cos(u_i, u_j)|."""
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return 0.0
    cos_matrix = np.abs(U_a.T @ U_b)
    weight_matrix = np.outer(S_a, S_b)
    Z = S_a.sum() * S_b.sum()
    if Z < 1e-12:
        return 0.0
    return float(np.sum(weight_matrix * cos_matrix) / Z)


# ---------------------------------------------------------------------------
# Layer name parsing
# ---------------------------------------------------------------------------

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


def detect_module_type(layer_name: str) -> str:
    """Detect Q/K/V/O from layer name."""
    for m in MODULES:
        if m in layer_name:
            return m
    return "unknown"


# ---------------------------------------------------------------------------
# Load adapter weights from safetensors
# ---------------------------------------------------------------------------

def load_adapter_weights(adapter_dir: Path) -> dict[str, dict]:
    """Load LoRA A/B matrices from adapter safetensors."""
    weights: dict[str, dict] = {}
    path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"

    config = json.loads(config_path.read_text())
    scaling = config.get("lora_alpha", 32) / config.get("r", 16)

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
# Phase 2a: Per-adapter audit with v2.1 schema (U/V persistence)
# ---------------------------------------------------------------------------

def audit_single_adapter(adapter_name: str) -> dict | None:
    """Compute v2.1 audit for one adapter. Returns None if already done."""
    output_path = OUTPUT_DIR / f"{adapter_name}_v2_1.json"
    if output_path.exists():
        print(f"  SKIP {adapter_name} -- v2.1 audit already exists")
        return None

    adapter_dir = ADAPTER_ROOT / adapter_name
    if not (adapter_dir / "adapter_model.safetensors").exists():
        print(f"  SKIP {adapter_name} -- adapter not found")
        return None

    print(f"  Auditing {adapter_name}...")

    # Parse task/seed from adapter name
    parts = adapter_name.rsplit("_s", 1)
    task = parts[0]
    seed = int(parts[1]) if len(parts) > 1 else -1

    # Load training metadata if available
    meta_path = adapter_dir / "training_meta.json"
    training_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    weights = load_adapter_weights(adapter_dir)

    # LoRA adapter rank (same for all layers — read from config)
    lora_rank = int(json.loads((adapter_dir / "adapter_config.json").read_text()).get("r", 16))

    layers = []
    for layer_name in sorted(weights.keys()):
        lw = weights[layer_name]
        A = lw["A"]          # shape (r, d_in) e.g. (16, 4096)
        B = lw["B"]          # shape (d_out, r) e.g. (4096, 16)
        scaling = lw["scaling"]

        # Fast rank-r SVD via QR. The full delta_W = scaling * B @ A has
        # shape (d_out, d_in) (e.g. 4096x4096) but rank at most r (e.g. 16).
        # Naive svd(delta_W) is O(d^3) and produces the correct rank-r
        # factors padded with zero singular values — but also wastes ~4096x
        # compute and writes 4096x4096 matrices to JSON (270MB per layer).
        # Instead: QR on B, then small SVD on the r x r product.
        B_scaled = scaling * B
        Q_B, R_B = np.linalg.qr(B_scaled, mode="reduced")    # Q:(d_out,r), R:(r,r)
        M = R_B @ A                                           # (r, d_in)
        U_small, S, Vt = np.linalg.svd(M, full_matrices=False)
        U = Q_B @ U_small                                     # (d_out, r)

        layer_idx = extract_layer_idx(layer_name)
        module = detect_module_type(layer_name)
        rank = lora_rank  # nominal rank (the LoRA r from config)

        # Store as float32 per v2.1 schema (see sidecar/data/n134/audit_v2_1.schema.json)
        U_trunc = U[:, :rank].astype(np.float32)
        Vt_trunc = Vt[:rank, :].astype(np.float32)
        S_trunc = S[:rank].astype(np.float32)

        layers.append({
            "layer_idx": layer_idx,
            "module": module,
            "rank": int(rank),
            "singular_values": S_trunc.tolist(),
            "u_factor": U_trunc.tolist(),
            "v_factor": Vt_trunc.tolist(),
            "stable_rank": stable_rank_fn(S),
            "energy_rank_90": energy_rank_90(S),
            "entropy_effective_rank": entropy_effective_rank(S),
            "frobenius_norm": float(np.sqrt(np.sum(S**2))),
        })

    result = {
        "adapter_id": adapter_name,
        "base_model": MODEL_NAME,
        "layers": layers,
        "meta": {
            "task": task,
            "seed": seed,
            "training_steps": training_meta.get("global_step", -1),
            "final_val_accuracy": training_meta.get("final_val_accuracy", None),
        },
    }

    # Write the tiny summary (scalars only, no U/V) — used by Phases 2-4.
    summary = _build_summary(result)
    summary_path = OUTPUT_DIR / f"{adapter_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Persist U/S/Vt as a binary .npz (~50 MB per adapter compressed,
    # fast load, needed by pair alignment and Phase 5).
    npz_path = OUTPUT_DIR / f"{adapter_name}_svd.npz"
    npz_data: dict = {}
    for layer in layers:
        layer_idx = layer["layer_idx"]
        module = layer["module"]
        key = f"layers.{layer_idx}.self_attn.{module}"
        npz_data[f"{key}__U"] = np.array(layer["u_factor"], dtype=np.float32)
        npz_data[f"{key}__S"] = np.array(layer["singular_values"], dtype=np.float32)
        npz_data[f"{key}__Vt"] = np.array(layer["v_factor"], dtype=np.float32)
    np.savez_compressed(str(npz_path), **npz_data)

    # The big JSON v2.1 file is no longer written by default — it's ~400 MB
    # each (JSON text overhead on 4096x16 U/V lists) and the MooseFS mount
    # on this pod throws silent quota errors after ~6 such files. The .npz
    # sidecar contains the same U/S/Vt data at ~50 MB compressed and is
    # what Phase 5 method-comparison actually reads. If a downstream
    # consumer needs JSON v2.1 specifically, reconstitute it from
    # summary.json + svd.npz via a helper (not yet written — add if needed).
    if os.environ.get("N134_WRITE_V21_JSON", "0") == "1":
        v21_path = OUTPUT_DIR / f"{adapter_name}_v2_1.json"
        with open(v21_path, "w") as f:
            json.dump(result, f, indent=2)

    n_layers = len(layers)
    mean_erank = float(np.mean([l["entropy_effective_rank"] for l in layers])) if layers else 0.0
    print(f"    {n_layers} layers, mean erank = {mean_erank:.2f}")

    return result


def _build_summary(result: dict) -> dict:
    """Extract scalars-only summary from a v2.1 audit result."""
    layers = result["layers"]
    eranks = [l["entropy_effective_rank"] for l in layers]
    sranks = [l["stable_rank"] for l in layers]
    norms = [l["frobenius_norm"] for l in layers]
    return {
        "adapter_id": result["adapter_id"],
        "meta": result["meta"],
        "n_layers": len(layers),
        "mean_erank": float(np.mean(eranks)) if eranks else 0.0,
        "mean_stable_rank": float(np.mean(sranks)) if sranks else 0.0,
        "mean_frobenius_norm": float(np.mean(norms)) if norms else 0.0,
        "per_layer": [
            {
                "layer_idx": l["layer_idx"],
                "module": l["module"],
                "erank": l["entropy_effective_rank"],
                "stable_rank": l["stable_rank"],
                "frobenius_norm": l["frobenius_norm"],
                "sigma_max": l["singular_values"][0] if l["singular_values"] else 0.0,
                "energy_rank_90": l["energy_rank_90"],
            }
            for l in layers
        ],
    }


def audit_all_adapters(adapter_names: list[str]) -> dict:
    """Compute v2.1 audit for all adapters and build summary profiles.

    On resume: reads the tiny {adapter}_summary.json (written alongside
    v2.1) instead of the ~400 MB v2.1 file. This is mandatory for memory
    safety — loading 24 × 400 MB JSONs via json.load would OOM even with
    500 GB RAM due to Python's float-object overhead (amplification of
    ~28 bytes per serialized float).
    """
    import gc

    print("\n=== Phase 2a: Per-Adapter Spectral Audit (v2.1) ===\n")

    all_profiles: dict = {}

    for adapter_name in adapter_names:
        v21_path = OUTPUT_DIR / f"{adapter_name}_v2_1.json"
        summary_path = OUTPUT_DIR / f"{adapter_name}_summary.json"

        if summary_path.exists():
            # Resume from tiny summary — no U/V loading
            print(f"  [resume] {adapter_name}_summary.json")
            with open(summary_path) as f:
                summary = json.load(f)
            all_profiles[adapter_name] = {
                "task": summary["meta"]["task"],
                "seed": summary["meta"]["seed"],
                "n_layers": summary["n_layers"],
                "mean_erank": summary["mean_erank"],
                "mean_stable_rank": summary["mean_stable_rank"],
                "mean_frobenius_norm": summary["mean_frobenius_norm"],
                "per_layer": summary["per_layer"],
            }
        elif v21_path.exists():
            # Legacy path: v2.1 exists but summary doesn't — extract
            # summary in-memory, write summary, then release.
            print(f"  [resume legacy] {adapter_name}_v2_1.json (backfilling summary)")
            with open(v21_path) as f:
                result = json.load(f)
            summary = _build_summary(result)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            del result
            gc.collect()
            all_profiles[adapter_name] = {
                "task": summary["meta"]["task"],
                "seed": summary["meta"]["seed"],
                "n_layers": summary["n_layers"],
                "mean_erank": summary["mean_erank"],
                "mean_stable_rank": summary["mean_stable_rank"],
                "mean_frobenius_norm": summary["mean_frobenius_norm"],
                "per_layer": summary["per_layer"],
            }
        else:
            result = audit_single_adapter(adapter_name)
            if result is None:
                continue
            all_profiles[adapter_name] = {
                "task": result["meta"]["task"],
                "seed": result["meta"]["seed"],
                "n_layers": len(result["layers"]),
                "mean_erank": float(np.mean([l["entropy_effective_rank"] for l in result["layers"]])),
                "mean_stable_rank": float(np.mean([l["stable_rank"] for l in result["layers"]])),
                "mean_frobenius_norm": float(np.mean([l["frobenius_norm"] for l in result["layers"]])),
                "per_layer": [
                    {
                        "layer_idx": l["layer_idx"],
                        "module": l["module"],
                        "erank": l["entropy_effective_rank"],
                        "stable_rank": l["stable_rank"],
                        "frobenius_norm": l["frobenius_norm"],
                        "sigma_max": l["singular_values"][0] if l["singular_values"] else 0.0,
                        "energy_rank_90": l["energy_rank_90"],
                    }
                    for l in result["layers"]
                ],
            }
            del result
            gc.collect()

    return all_profiles


# ---------------------------------------------------------------------------
# Phase 2b: W0 spectral properties
# ---------------------------------------------------------------------------

def audit_w0() -> dict:
    """Compute C_k for each LoRA-targeted layer of Mistral-7B W0."""
    w0_path = OUTPUT_DIR / "w0_properties.json"
    if w0_path.exists():
        print(f"\n[resume] Loading existing {w0_path}")
        with open(w0_path) as f:
            return json.load(f)

    print("\n=== Phase 2b: Mistral W0 Spectral Properties ===\n")

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    w0_data: dict = {}

    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if not any(m in name for m in MODULES):
            continue
        if "self_attn" not in name:
            continue

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

    with open(w0_path, "w") as f:
        json.dump(w0_data, f, indent=2)

    return w0_data


# ---------------------------------------------------------------------------
# Phase 2c: Pairwise SV-weighted alignment
# ---------------------------------------------------------------------------

def _load_svd_from_npz(adapter_name: str) -> dict[str, dict] | None:
    """Load cached U/S/Vt from the binary .npz (fast) or JSON fallback (slow)."""
    npz_path = OUTPUT_DIR / f"{adapter_name}_svd.npz"
    if npz_path.exists():
        with np.load(str(npz_path), allow_pickle=False) as z:
            svd_data: dict[str, dict] = {}
            # Group keys by their layer-key prefix
            keys = list(z.keys())
            layer_keys = sorted({k.rsplit("__", 1)[0] for k in keys})
            for lk in layer_keys:
                svd_data[lk] = {
                    "U": z[f"{lk}__U"].copy(),
                    "S": z[f"{lk}__S"].copy(),
                    "Vt": z[f"{lk}__Vt"].copy(),
                }
        return svd_data

    # Fallback: parse v2.1 JSON (slow, ~400 MB load). Used only if the
    # adapter was produced by an older version of this script that didn't
    # write the .npz sidecar. The caller should prefer .npz.
    v21_path = OUTPUT_DIR / f"{adapter_name}_v2_1.json"
    if not v21_path.exists():
        return None
    print(f"    [slow-path] parsing {adapter_name}_v2_1.json (no .npz sidecar)")
    with open(v21_path) as f:
        data = json.load(f)
    svd_data = {}
    for layer in data["layers"]:
        layer_idx = layer["layer_idx"]
        module = layer["module"]
        key = f"layers.{layer_idx}.self_attn.{module}"
        svd_data[key] = {
            "U": np.array(layer["u_factor"], dtype=np.float32),
            "S": np.array(layer["singular_values"], dtype=np.float32),
            "Vt": np.array(layer["v_factor"], dtype=np.float32),
        }
    # Write the .npz sidecar so this slow path only runs once per adapter
    npz_data: dict = {}
    for lk, d in svd_data.items():
        npz_data[f"{lk}__U"] = d["U"]
        npz_data[f"{lk}__S"] = d["S"]
        npz_data[f"{lk}__Vt"] = d["Vt"]
    np.savez_compressed(str(npz_path), **npz_data)
    del data
    import gc
    gc.collect()
    return svd_data


def compute_pairwise_alignment(adapter_names: list[str], adapter_profiles: dict) -> dict:
    """Compute per-layer SV-weighted alignment for all C(n,2) pairs.

    Loads all adapters' SVD data from .npz sidecars. Each adapter's
    SVD is ~5-10 MB in RAM (vs ~400 MB for JSON-parsed equivalent),
    so 24 adapters fit comfortably (~200 MB total).
    """
    pair_full_path = OUTPUT_DIR / "pair_alignment_full.json"
    if pair_full_path.exists():
        print(f"\n[resume] Skipping Phase 2c -- {pair_full_path} already exists")
        with open(pair_full_path) as f:
            return json.load(f)

    print("\n=== Phase 2c: Pairwise Alignment ===\n")

    # Load all SVD data from .npz sidecars (lightweight — ~200 MB total)
    print("  Loading SVD data from .npz sidecars...")
    adapter_svd: dict[str, dict] = {}
    for name in adapter_names:
        svd_data = _load_svd_from_npz(name)
        if svd_data is not None:
            adapter_svd[name] = svd_data
        else:
            print(f"  WARN: no SVD data for {name}, skipping")

    available = sorted(adapter_svd.keys())
    n_pairs = len(list(combinations(available, 2)))
    print(f"  Computing alignment for {n_pairs} pairs...")

    pair_results: dict = {}
    done = 0

    for a_name, b_name in combinations(available, 2):
        task_a = adapter_profiles[a_name]["task"]
        task_b = adapter_profiles[b_name]["task"]
        is_same_task = task_a == task_b

        svd_a = adapter_svd[a_name]
        svd_b = adapter_svd[b_name]

        shared_layers = sorted(set(svd_a.keys()) & set(svd_b.keys()))
        per_layer = []

        for layer_key in shared_layers:
            da = svd_a[layer_key]
            db = svd_b[layer_key]
            align = sv_weighted_alignment(da["U"], da["S"], db["U"], db["S"])

            # Parse layer_key to get layer_idx and module
            layer_idx = extract_layer_idx(layer_key)
            module = detect_module_type(layer_key)

            per_layer.append({
                "layer_key": layer_key,
                "layer_idx": layer_idx,
                "module": module,
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

        done += 1
        if done % 50 == 0:
            print(f"    {done}/{n_pairs} pairs computed...")

    # Summary stats
    same_task = [v for v in pair_results.values() if v["is_same_task"]]
    cross_task = [v for v in pair_results.values() if not v["is_same_task"]]

    same_mean = float(np.mean([v["mean_alignment"] for v in same_task])) if same_task else 0.0
    cross_mean = float(np.mean([v["mean_alignment"] for v in cross_task])) if cross_task else 0.0

    print(f"\n  Total pairs: {done}")
    print(f"  Same-task: {len(same_task)} pairs, mean alignment = {same_mean:.4f}")
    print(f"  Cross-task: {len(cross_task)} pairs, mean alignment = {cross_mean:.4f}")
    if cross_mean > 0:
        print(f"  Same/cross ratio: {same_mean / cross_mean:.2f}x")

    # Save full results
    with open(pair_full_path, "w") as f:
        json.dump(pair_results, f, indent=2)

    # Save summary (without per-layer detail)
    pair_summary: dict = {}
    for k, v in pair_results.items():
        pair_summary[k] = {key: val for key, val in v.items() if key != "per_layer"}
    with open(OUTPUT_DIR / "pair_alignment_summary.json", "w") as f:
        json.dump(pair_summary, f, indent=2)

    return pair_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="N134: Spectral audit with v2.1 schema")
    parser.add_argument("--smoke", action="store_true", help="Audit only first 2 adapters")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  N134 Phase 2: Spectral Audit (v2.1)")
    print("=" * 60)

    t0 = time.time()

    # Build adapter name list
    adapter_names = []
    for task in TASKS:
        for seed in SEEDS:
            adapter_names.append(f"{task}_s{seed}")

    if args.smoke:
        adapter_names = adapter_names[:2]
        print(f"  SMOKE mode: auditing only {adapter_names}")

    # Phase 2a: Per-adapter v2.1 audits
    adapter_profiles = audit_all_adapters(adapter_names)

    # Save adapter profiles
    profiles_path = OUTPUT_DIR / "adapter_profiles.json"
    with open(profiles_path, "w") as f:
        json.dump(adapter_profiles, f, indent=2)
    print(f"\n  Saved {len(adapter_profiles)} adapter profiles to {profiles_path}")

    # Phase 2b: W0 properties
    audit_w0()

    # Phase 2c: Pairwise alignment
    compute_pairwise_alignment(adapter_names, adapter_profiles)

    elapsed = time.time() - t0
    print(f"\n  Total audit time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
