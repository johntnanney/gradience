#!/usr/bin/env python3
"""
MP Partition Test: Does the Marchenko-Pastur threshold separate shared
from task-specific spectral structure in independently trained LoRA adapters?

Hypothesis (from Technical Report §2.3.1, THEORY.md §6):
    High-SV directions (above MP threshold) encode shared structure and
    should show high inter-adapter alignment across seeds.
    Low-SV directions (below MP threshold) encode task-specific or
    noise structure and should show low inter-adapter alignment.

Data: Same-task, same-backbone adapters from bench_runs/firstclass_multiseed_test
      (distilbert-base-uncased, SEQ_CLS, seeds 42/123/456, rank 16)

Method:
    1. Load adapter weights, compute SVD of each layer's BA product
    2. Compute Gavish-Donoho optimal hard threshold (MP threshold)
    3. Partition singular vectors into above-threshold and below-threshold bands
    4. Measure inter-adapter alignment within each band via principal angle cosines
    5. Compare alignment across bands

Reference: Tian, Ledent, & Sun (2026). Scalable Multi-Task Low-Rank Model
    Adaptation. ICLR 2026. arXiv:2603.01526.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.linalg import svd, qr
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adapter_weights(adapter_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load LoRA A and B matrices from a PEFT adapter directory.

    Returns dict mapping layer_key -> {"A": ndarray, "B": ndarray}.
    Layer key is e.g. "distilbert.transformer.layer.0.attention.q_lin".
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
        # Group by layer: keys are like
        # "base_model.model.distilbert.transformer.layer.0.attention.q_lin.lora_A.weight"
        a_keys = sorted(k for k in keys if "lora_A" in k)
        b_keys = sorted(k for k in keys if "lora_B" in k)

        for a_key in a_keys:
            # Derive layer key and find matching B
            layer_key = a_key.replace(".lora_A.weight", "").replace(
                ".lora_A.default.weight", ""
            )
            # Try both naming conventions
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in keys:
                # Try alternate naming
                for bk in b_keys:
                    bk_layer = bk.replace(".lora_B.weight", "").replace(
                        ".lora_B.default.weight", ""
                    )
                    if bk_layer == layer_key:
                        b_key = bk
                        break

            if b_key not in keys:
                print(f"  Warning: no B matrix for {a_key}, skipping")
                continue

            A = f.get_tensor(a_key).astype(np.float64)  # (r, d_in)
            B = f.get_tensor(b_key).astype(np.float64)  # (d_out, r)

            # Shorten the layer key for readability
            short_key = layer_key
            for prefix in ["base_model.model.", "base_model."]:
                if short_key.startswith(prefix):
                    short_key = short_key[len(prefix):]

            weights[short_key] = {"A": A, "B": B, "scaling": scaling}

    return weights


# ---------------------------------------------------------------------------
# SVD computation (QR-based, same method as Gradience core)
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
    # QR of B
    Qb, Rb = qr(B, mode="reduced")  # Qb: (d_out, r), Rb: (r, r)
    # QR of A^T
    Qa, Ra = qr(A.T, mode="reduced")  # Qa: (d_in, r), Ra: (r, r)
    # SVD of small r×r matrix
    M = scaling * (Rb @ Ra.T)
    U_small, S, Vt_small = svd(M, full_matrices=False)
    # Recover full singular vectors
    U = Qb @ U_small
    Vt = Vt_small @ Qa.T
    # Sort descending (should already be, but ensure)
    order = np.argsort(-S)
    return U[:, order], S[order], Vt[order, :]


# ---------------------------------------------------------------------------
# Marchenko-Pastur optimal hard threshold (Gavish-Donoho)
# ---------------------------------------------------------------------------

def mp_threshold(S: np.ndarray, shape: tuple[int, int]) -> tuple[float, int]:
    """Compute Gavish-Donoho optimal hard threshold.

    Args:
        S: singular values (descending)
        shape: (d_out, d_in) dimensions of the full weight matrix

    Returns:
        (tau, k) where tau is the threshold and k is the number of
        singular values above it.
    """
    d_out, d_in = shape
    beta = min(d_out, d_in) / max(d_out, d_in)
    # Gavish-Donoho cubic approximation for optimal omega
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    median_sv = float(np.median(S))
    if median_sv < 1e-12:
        return 0.0, len(S)
    tau = omega * median_sv
    k = int(np.sum(S > tau))
    return tau, max(k, 1)  # At least 1 above threshold


# ---------------------------------------------------------------------------
# Band-partitioned alignment measurement
# ---------------------------------------------------------------------------

def principal_angle_cosines(
    U_a: np.ndarray, U_b: np.ndarray
) -> np.ndarray:
    """Compute principal angle cosines between column spaces of U_a and U_b.

    Returns array of cosines in descending order.
    """
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return np.array([])
    cross = U_a.T @ U_b
    cosines = svd(cross, compute_uv=False)
    return np.clip(cosines, 0.0, 1.0)


def sv_weighted_alignment(
    U_a: np.ndarray, S_a: np.ndarray,
    U_b: np.ndarray, S_b: np.ndarray,
) -> float:
    """Singular-value-weighted alignment score (cf. Tian et al. 2026).

    A(B) = (1/Z) * sum_i sum_j sigma_a_i * sigma_b_j * |cos(u_a_i, u_b_j)|

    where Z = sum_i sigma_a_i * sum_j sigma_b_j (normalization).
    """
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return 0.0
    # Compute all pairwise |cos| values
    cos_matrix = np.abs(U_a.T @ U_b)  # (k_a, k_b)
    # Weight by outer product of singular values
    weight_matrix = np.outer(S_a, S_b)
    Z = S_a.sum() * S_b.sum()
    if Z < 1e-12:
        return 0.0
    return float(np.sum(weight_matrix * cos_matrix) / Z)


def mean_alignment(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """Mean principal angle cosine (unweighted)."""
    cosines = principal_angle_cosines(U_a, U_b)
    if len(cosines) == 0:
        return 0.0
    return float(cosines.mean())


@dataclass
class BandAlignment:
    """Alignment results for one spectral band of one layer."""
    band: str  # "high_sv", "low_sv", "full"
    n_dims_a: int
    n_dims_b: int
    mean_cos: float
    sv_weighted: float
    energy_frac_a: float  # Fraction of total energy in this band
    energy_frac_b: float


@dataclass
class LayerResult:
    """Results for one layer across all adapter pairs."""
    layer_name: str
    module_type: str  # "q", "k", "v", "o", "other"
    mp_threshold_values: list[float]
    mp_k_values: list[int]  # Number of SVs above threshold per adapter
    nominal_rank: int
    shape: tuple[int, int]  # (d_out, d_in)
    bands: dict[str, list[BandAlignment]] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Full experiment results."""
    description: str
    adapter_dirs: list[str]
    n_pairs: int
    layers: list[LayerResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module type detection
# ---------------------------------------------------------------------------

def detect_module_type(layer_name: str) -> str:
    """Detect Q/K/V/O module type from layer name."""
    name_lower = layer_name.lower()
    if "q_lin" in name_lower or ".q_proj" in name_lower or ".query" in name_lower:
        return "q"
    if "k_lin" in name_lower or ".k_proj" in name_lower or ".key" in name_lower:
        return "k"
    if "v_lin" in name_lower or ".v_proj" in name_lower or ".value" in name_lower:
        return "v"
    if "out_lin" in name_lower or ".o_proj" in name_lower or ".output" in name_lower:
        return "o"
    return "other"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_partition_test(
    adapter_dirs: list[Path],
    labels: list[str] | None = None,
) -> ExperimentResult:
    """Run the MP partition test on a set of same-task adapters."""

    if labels is None:
        labels = [d.name for d in adapter_dirs]

    print(f"\n{'='*70}")
    print(f"MP PARTITION TEST")
    print(f"Adapters: {len(adapter_dirs)}")
    print(f"Pairs: {len(list(combinations(range(len(adapter_dirs)), 2)))}")
    print(f"{'='*70}\n")

    # Load all adapters
    all_weights = []
    for i, d in enumerate(adapter_dirs):
        print(f"Loading adapter {labels[i]} from {d}")
        w = load_adapter_weights(d)
        all_weights.append(w)
        print(f"  Loaded {len(w)} layers")

    # Find shared layers
    shared_layers = sorted(
        set.intersection(*[set(w.keys()) for w in all_weights])
    )
    print(f"\nShared layers: {len(shared_layers)}")

    # Compute SVDs
    print("\nComputing SVDs...")
    all_svds: list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for i, w in enumerate(all_weights):
        svds = {}
        for layer in shared_layers:
            A = w[layer]["A"]
            B = w[layer]["B"]
            scaling = w[layer]["scaling"]
            U, S, Vt = compute_layer_svd(A, B, scaling)
            svds[layer] = (U, S, Vt)
        all_svds.append(svds)
        print(f"  Adapter {labels[i]}: {len(svds)} layers SVD'd")

    # Get shape info from first adapter
    sample_layer = shared_layers[0]
    sample_A = all_weights[0][sample_layer]["A"]
    sample_B = all_weights[0][sample_layer]["B"]
    d_in = sample_A.shape[1]
    d_out = sample_B.shape[0]
    r = sample_A.shape[0]

    pairs = list(combinations(range(len(adapter_dirs)), 2))
    result = ExperimentResult(
        description="MP Partition Test on distilbert-base-uncased same-task adapters",
        adapter_dirs=[str(d) for d in adapter_dirs],
        n_pairs=len(pairs),
    )

    # Process each layer
    for layer_name in shared_layers:
        module_type = detect_module_type(layer_name)
        A_shape = all_weights[0][layer_name]["A"]
        B_shape = all_weights[0][layer_name]["B"]
        shape = (B_shape.shape[0], A_shape.shape[1])
        nominal_rank = A_shape.shape[0]

        # Compute MP thresholds for each adapter at this layer
        mp_taus = []
        mp_ks = []
        for i in range(len(adapter_dirs)):
            _, S, _ = all_svds[i][layer_name]
            tau, k = mp_threshold(S, shape)
            mp_taus.append(tau)
            mp_ks.append(k)

        layer_result = LayerResult(
            layer_name=layer_name,
            module_type=module_type,
            mp_threshold_values=mp_taus,
            mp_k_values=mp_ks,
            nominal_rank=nominal_rank,
            shape=shape,
        )

        # For each pair, compute band-partitioned alignment
        for band_name in ["full", "high_sv", "low_sv"]:
            layer_result.bands[band_name] = []

        for idx_a, idx_b in pairs:
            U_a, S_a, _ = all_svds[idx_a][layer_name]
            U_b, S_b, _ = all_svds[idx_b][layer_name]
            k_a = mp_ks[idx_a]
            k_b = mp_ks[idx_b]
            total_energy_a = float(np.sum(S_a**2))
            total_energy_b = float(np.sum(S_b**2))

            # Full spectrum alignment
            full_align = BandAlignment(
                band="full",
                n_dims_a=len(S_a),
                n_dims_b=len(S_b),
                mean_cos=mean_alignment(U_a, U_b),
                sv_weighted=sv_weighted_alignment(U_a, S_a, U_b, S_b),
                energy_frac_a=1.0,
                energy_frac_b=1.0,
            )
            layer_result.bands["full"].append(full_align)

            # High-SV band (above MP threshold)
            U_a_high = U_a[:, :k_a]
            S_a_high = S_a[:k_a]
            U_b_high = U_b[:, :k_b]
            S_b_high = S_b[:k_b]
            e_a_high = float(np.sum(S_a_high**2)) / total_energy_a if total_energy_a > 0 else 0
            e_b_high = float(np.sum(S_b_high**2)) / total_energy_b if total_energy_b > 0 else 0

            high_align = BandAlignment(
                band="high_sv",
                n_dims_a=k_a,
                n_dims_b=k_b,
                mean_cos=mean_alignment(U_a_high, U_b_high),
                sv_weighted=sv_weighted_alignment(U_a_high, S_a_high, U_b_high, S_b_high),
                energy_frac_a=e_a_high,
                energy_frac_b=e_b_high,
            )
            layer_result.bands["high_sv"].append(high_align)

            # Low-SV band (below MP threshold)
            U_a_low = U_a[:, k_a:]
            S_a_low = S_a[k_a:]
            U_b_low = U_b[:, k_b:]
            S_b_low = S_b[k_b:]
            e_a_low = float(np.sum(S_a_low**2)) / total_energy_a if total_energy_a > 0 else 0
            e_b_low = float(np.sum(S_b_low**2)) / total_energy_b if total_energy_b > 0 else 0

            low_align = BandAlignment(
                band="low_sv",
                n_dims_a=len(S_a_low),
                n_dims_b=len(S_b_low),
                mean_cos=mean_alignment(U_a_low, U_b_low),
                sv_weighted=sv_weighted_alignment(U_a_low, S_a_low, U_b_low, S_b_low),
                energy_frac_a=e_a_low,
                energy_frac_b=e_b_low,
            )
            layer_result.bands["low_sv"].append(low_align)

        result.layers.append(layer_result)

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: ExperimentResult) -> str:
    """Print and return a formatted report of the experiment results."""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"\n{'='*70}")
    p("RESULTS: MP PARTITION TEST")
    p(f"{'='*70}")
    p(f"\nAdapters: {len(result.adapter_dirs)}")
    p(f"Pairs: {result.n_pairs}")
    p(f"Layers: {len(result.layers)}")

    # Aggregate by band
    all_high_mean = []
    all_low_mean = []
    all_full_mean = []
    all_high_svw = []
    all_low_svw = []
    all_full_svw = []

    # Per-module-type aggregates
    module_high: dict[str, list[float]] = {}
    module_low: dict[str, list[float]] = {}

    p(f"\n{'─'*70}")
    p(f"{'Layer':<50} {'MP k':>5} {'High':>7} {'Low':>7} {'Full':>7} {'E_high':>7}")
    p(f"{'─'*70}")

    for lr in result.layers:
        avg_k = np.mean(lr.mp_k_values)
        high_aligns = lr.bands["high_sv"]
        low_aligns = lr.bands["low_sv"]
        full_aligns = lr.bands["full"]

        h_mean = np.mean([a.sv_weighted for a in high_aligns])
        l_mean = np.mean([a.sv_weighted for a in low_aligns])
        f_mean = np.mean([a.sv_weighted for a in full_aligns])
        e_high = np.mean([a.energy_frac_a for a in high_aligns])

        all_high_svw.append(h_mean)
        all_low_svw.append(l_mean)
        all_full_svw.append(f_mean)

        h_mc = np.mean([a.mean_cos for a in high_aligns])
        l_mc = np.mean([a.mean_cos for a in low_aligns])
        f_mc = np.mean([a.mean_cos for a in full_aligns])
        all_high_mean.append(h_mc)
        all_low_mean.append(l_mc)
        all_full_mean.append(f_mc)

        mt = lr.module_type
        if mt not in module_high:
            module_high[mt] = []
            module_low[mt] = []
        module_high[mt].append(h_mean)
        module_low[mt].append(l_mean)

        # Truncate layer name for display
        short = lr.layer_name
        if len(short) > 48:
            short = "..." + short[-45:]

        p(f"{short:<50} {avg_k:>5.1f} {h_mean:>7.3f} {l_mean:>7.3f} {f_mean:>7.3f} {e_high:>7.1%}")

    p(f"{'─'*70}")

    # Grand averages
    p(f"\n{'='*70}")
    p("AGGREGATE RESULTS")
    p(f"{'='*70}")
    p(f"\n  SV-Weighted Alignment (cf. Tian et al. metric):")
    p(f"    High-SV band (above MP):  {np.mean(all_high_svw):.4f}  (std {np.std(all_high_svw):.4f})")
    p(f"    Low-SV band (below MP):   {np.mean(all_low_svw):.4f}  (std {np.std(all_low_svw):.4f})")
    p(f"    Full spectrum:             {np.mean(all_full_svw):.4f}  (std {np.std(all_full_svw):.4f})")
    ratio_svw = np.mean(all_high_svw) / np.mean(all_low_svw) if np.mean(all_low_svw) > 1e-8 else float("inf")
    p(f"    High/Low ratio:            {ratio_svw:.1f}x")

    p(f"\n  Mean Principal Angle Cosine (unweighted):")
    p(f"    High-SV band (above MP):  {np.mean(all_high_mean):.4f}  (std {np.std(all_high_mean):.4f})")
    p(f"    Low-SV band (below MP):   {np.mean(all_low_mean):.4f}  (std {np.std(all_low_mean):.4f})")
    p(f"    Full spectrum:             {np.mean(all_full_mean):.4f}  (std {np.std(all_full_mean):.4f})")
    ratio_mc = np.mean(all_high_mean) / np.mean(all_low_mean) if np.mean(all_low_mean) > 1e-8 else float("inf")
    p(f"    High/Low ratio:            {ratio_mc:.1f}x")

    # By module type
    p(f"\n  By Module Type (SV-weighted):")
    for mt in sorted(module_high.keys()):
        h = np.mean(module_high[mt])
        l = np.mean(module_low[mt])
        r = h / l if l > 1e-8 else float("inf")
        p(f"    {mt:>5}: high={h:.4f}  low={l:.4f}  ratio={r:.1f}x  (n={len(module_high[mt])} layers)")

    # Statistical test
    from scipy import stats
    if len(all_high_svw) > 2 and len(all_low_svw) > 2:
        t_stat, p_val = stats.ttest_rel(all_high_svw, all_low_svw)
        p(f"\n  Paired t-test (high vs low, SV-weighted):")
        p(f"    t = {t_stat:.3f}, p = {p_val:.2e}")
        diff = np.array(all_high_svw) - np.array(all_low_svw)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        p(f"    Cohen's d = {cohens_d:.2f}")

    if len(all_high_mean) > 2 and len(all_low_mean) > 2:
        t_stat2, p_val2 = stats.ttest_rel(all_high_mean, all_low_mean)
        p(f"\n  Paired t-test (high vs low, mean cosine):")
        p(f"    t = {t_stat2:.3f}, p = {p_val2:.2e}")
        diff2 = np.array(all_high_mean) - np.array(all_low_mean)
        cohens_d2 = np.mean(diff2) / np.std(diff2) if np.std(diff2) > 0 else 0
        p(f"    Cohen's d = {cohens_d2:.2f}")

    # MP threshold summary
    p(f"\n  MP Threshold Summary:")
    all_ks = [k for lr in result.layers for k in lr.mp_k_values]
    all_ranks = [lr.nominal_rank for lr in result.layers for _ in lr.mp_k_values]
    p(f"    Mean k above threshold: {np.mean(all_ks):.1f} / {all_ranks[0]} nominal rank")
    p(f"    k range: [{min(all_ks)}, {max(all_ks)}]")
    energy_highs = [
        a.energy_frac_a
        for lr in result.layers
        for a in lr.bands["high_sv"]
    ]
    p(f"    Mean energy in high band: {np.mean(energy_highs):.1%}")

    p(f"\n{'='*70}")

    return "\n".join(lines)


def save_results_json(result: ExperimentResult, path: Path):
    """Save results as JSON for reproducibility."""
    # Convert to serializable format
    data = {
        "description": result.description,
        "adapter_dirs": result.adapter_dirs,
        "n_pairs": result.n_pairs,
        "layers": [],
    }
    for lr in result.layers:
        layer_data = {
            "layer_name": lr.layer_name,
            "module_type": lr.module_type,
            "mp_threshold_values": lr.mp_threshold_values,
            "mp_k_values": lr.mp_k_values,
            "nominal_rank": lr.nominal_rank,
            "shape": list(lr.shape),
            "bands": {},
        }
        for band_name, aligns in lr.bands.items():
            layer_data["bands"][band_name] = [asdict(a) for a in aligns]
        data["layers"].append(layer_data)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    repo = Path("/sessions/kind-trusting-cori/mnt/gradience")
    bench = repo / "bench_runs" / "firstclass_multiseed_test"

    # Use probe_r16 (rank 16) adapters across 3 seeds at checkpoint-50
    policy = "probe_r16"
    checkpoint = "checkpoint-50"
    seeds = [42, 123, 456]

    adapter_dirs = []
    labels = []
    for seed in seeds:
        d = bench / f"seed_{seed}" / policy / checkpoint
        if d.exists():
            adapter_dirs.append(d)
            labels.append(f"seed_{seed}")
        else:
            print(f"Warning: {d} not found, trying without checkpoint...")
            d2 = bench / f"seed_{seed}" / policy
            if d2.exists() and (d2 / "adapter_model.safetensors").exists():
                adapter_dirs.append(d2)
                labels.append(f"seed_{seed}")
            else:
                print(f"  Skipping seed {seed}")

    if len(adapter_dirs) < 2:
        print("ERROR: Need at least 2 adapters. Trying alternate locations...")
        # Fallback to test_priority_output
        alt = repo / "test_priority_output"
        for seed in seeds:
            d = alt / f"seed_{seed}" / policy
            if d.exists() and (d / "adapter_model.safetensors").exists():
                adapter_dirs.append(d)
                labels.append(f"test_seed_{seed}")

    if len(adapter_dirs) < 2:
        print("FATAL: Could not find enough adapter pairs.")
        sys.exit(1)

    print(f"Found {len(adapter_dirs)} adapters: {labels}")

    # Run experiment
    result = run_partition_test(adapter_dirs, labels)

    # Print report
    report_text = print_report(result)

    # Save results
    output_dir = repo / "sidecar" / "results" / "mp_partition_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results_json(result, output_dir / "results.json")
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)
    print(f"\nResults saved to {output_dir}")

    # Also try other rank policies for comparison
    for alt_policy in ["uniform_median_r16", "uniform_p90_r16"]:
        alt_dirs = []
        alt_labels = []
        for seed in seeds:
            d = bench / f"seed_{seed}" / alt_policy / checkpoint
            if d.exists():
                alt_dirs.append(d)
                alt_labels.append(f"seed_{seed}")
        if len(alt_dirs) >= 2:
            print(f"\n\n{'#'*70}")
            print(f"ADDITIONAL RUN: {alt_policy}")
            print(f"{'#'*70}")
            alt_result = run_partition_test(alt_dirs, alt_labels)
            alt_report = print_report(alt_result)
            save_results_json(
                alt_result, output_dir / f"results_{alt_policy}.json"
            )
            with open(output_dir / f"report_{alt_policy}.txt", "w") as f:
                f.write(alt_report)


if __name__ == "__main__":
    main()
