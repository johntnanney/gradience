"""N135: C_k Architecture-Specificity Investigation.

Compares the spectral concentration (C_k) of pre-trained weight matrices
across three architectures to test whether DeBERTa's disentangled attention
explains the C_k → alignment null finding.

Models:
  - DistilBERT-base-uncased (6L, standard attention, MLM)
  - DeBERTa-v3-base (12L, disentangled attention, RTD)
  - RoBERTa-base (12L, standard attention, MLM)  [control]

CPU-only.  Downloads models from HuggingFace on first run.

Predictions (from n135_ck_architecture_specificity.md):
  P1: DeBERTa Q/K C_k differs from DistilBERT Q/K C_k
  P2: DeBERTa Q/K C_k has lower cross-layer variance
  P3: V/O are more similar across architectures than Q/K
  P4: RoBERTa Q/K C_k resembles DistilBERT more than DeBERTa

Output: sidecar/data/n135/
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy import stats

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "sidecar" / "data" / "n135"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODELS = {
    "distilbert": {
        "hf_name": "distilbert-base-uncased",
        "n_layers": 6,
        "attention_type": "standard",
        "pretraining": "MLM",
        # DistilBERT uses different param naming (no .self. prefix)
        "module_patterns": {
            "Q": "attention.q_lin",
            "K": "attention.k_lin",
            "V": "attention.v_lin",
            "O": "attention.out_lin",
        },
        "layer_pattern": "layer",  # transformer.layer.{i}
    },
    "deberta": {
        "hf_name": "microsoft/deberta-v3-base",
        "n_layers": 12,
        "attention_type": "disentangled",
        "pretraining": "RTD",
        "module_patterns": {
            "Q": "attention.self.query_proj",
            "K": "attention.self.key_proj",
            "V": "attention.self.value_proj",
            "O": "attention.output.dense",
        },
        "layer_pattern": "layer",
    },
    "roberta": {
        "hf_name": "roberta-base",
        "n_layers": 12,
        "attention_type": "standard",
        "pretraining": "MLM",
        "module_patterns": {
            "Q": "attention.self.query",
            "K": "attention.self.key",
            "V": "attention.self.value",
            "O": "attention.output.dense",
        },
        "layer_pattern": "layer",
    },
}

MODULES = ["Q", "K", "V", "O"]


# ---------------------------------------------------------------------------
# Core computations (consistent with n132)
# ---------------------------------------------------------------------------

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
    """Energy concentration C_k = sum(S[:k]^2) / sum(S^2)."""
    total = np.sum(S**2)
    if total < 1e-20:
        return 0.0
    return float(np.sum(S[:k]**2) / total)


def entropy_effective_rank(S: np.ndarray) -> float:
    """Entropy effective rank of the spectrum."""
    S2 = S**2
    total = np.sum(S2)
    if total < 1e-20:
        return 0.0
    p = S2 / total
    p = p[p > 1e-20]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def stable_rank(S: np.ndarray) -> float:
    """Stable rank = ||W||_F^2 / ||W||_2^2."""
    if len(S) == 0 or S[0] < 1e-20:
        return 0.0
    return float(np.sum(S**2) / S[0]**2)


def cumulative_energy_curve(S: np.ndarray, n_points: int = 50) -> np.ndarray:
    """Normalized cumulative energy curve, sampled at n_points."""
    S2 = S**2
    total = np.sum(S2)
    if total < 1e-20:
        return np.zeros(n_points)
    cumsum = np.cumsum(S2) / total
    # Sample at evenly spaced fractional indices
    indices = np.linspace(0, len(cumsum) - 1, n_points).astype(int)
    return cumsum[indices]


# ---------------------------------------------------------------------------
# Phase 1: Extract W_0 spectral properties for all models
# ---------------------------------------------------------------------------

def extract_model_spectra(model_key: str) -> pd.DataFrame:
    """Load a model and compute spectral properties of all attention W_0."""
    config = MODELS[model_key]
    print(f"\n  Loading {config['hf_name']}...")

    from transformers import AutoModel
    model = AutoModel.from_pretrained(config["hf_name"])

    rows = []
    for name, param in model.named_parameters():
        if not name.endswith(".weight"):
            continue

        # Match to Q/K/V/O module
        matched_module = None
        for mod_short, mod_pattern in config["module_patterns"].items():
            if mod_pattern in name:
                matched_module = mod_short
                break
        if matched_module is None:
            continue

        # Extract layer index
        parts = name.split(".")
        layer_key = config["layer_pattern"]
        if layer_key not in parts:
            continue
        layer_idx = int(parts[parts.index(layer_key) + 1])

        W = param.detach().cpu().numpy().astype(np.float64)
        S = svd(W, compute_uv=False)
        tau, k = mp_threshold(S, W.shape)
        C_k = energy_concentration(S, k)

        rows.append({
            "model": model_key,
            "attention_type": config["attention_type"],
            "pretraining": config["pretraining"],
            "n_layers": config["n_layers"],
            "layer": layer_idx,
            "layer_frac": layer_idx / (config["n_layers"] - 1) if config["n_layers"] > 1 else 0.0,
            "module": matched_module,
            "param_name": name,
            "shape_0": W.shape[0],
            "shape_1": W.shape[1],
            "k_mp": k,
            "tau_mp": tau,
            "C_k": C_k,
            "sigma_1": float(S[0]),
            "erank_w0": entropy_effective_rank(S),
            "srank_w0": stable_rank(S),
            "frob_norm": float(np.sqrt(np.sum(S**2))),
            "spectral_curve": cumulative_energy_curve(S).tolist(),
        })

    del model
    df = pd.DataFrame(rows)
    print(f"  Extracted {len(df)} layers for {model_key}")
    return df


def phase1_extract_all() -> pd.DataFrame:
    """Extract spectral properties for all three models."""
    print("\n=== Phase 1: W_0 Spectral Extraction ===")
    dfs = []
    for model_key in MODELS:
        df = extract_model_spectra(model_key)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "phase1_w0_spectra.csv", index=False)
    print(f"\n  Total: {len(combined)} weight matrices across {len(MODELS)} models")
    return combined


# ---------------------------------------------------------------------------
# Phase 2: Cross-architecture C_k comparison (P1, P3, P4)
# ---------------------------------------------------------------------------

def phase2_ck_comparison(df: pd.DataFrame) -> dict:
    """Test P1, P3, P4 via cross-architecture C_k comparisons."""
    print("\n=== Phase 2: Cross-Architecture C_k Comparison ===")
    results = {}

    # --- Per-model, per-module summary ---
    print("\n  Per-model, per-module C_k summary:")
    print(f"  {'Model':<12} {'Module':<6} {'mean C_k':>9} {'SD':>8} {'min':>8} {'max':>8} {'n':>4}")
    print("  " + "-" * 58)
    for model_key in MODELS:
        for mod in MODULES:
            subset = df[(df["model"] == model_key) & (df["module"] == mod)]
            if len(subset) == 0:
                continue
            ck = subset["C_k"]
            print(f"  {model_key:<12} {mod:<6} {ck.mean():9.4f} {ck.std():8.4f} "
                  f"{ck.min():8.4f} {ck.max():8.4f} {len(ck):4d}")
            results[f"{model_key}_{mod}_mean"] = float(ck.mean())
            results[f"{model_key}_{mod}_sd"] = float(ck.std())
            results[f"{model_key}_{mod}_n"] = int(len(ck))

    # --- P1: DeBERTa Q/K C_k vs DistilBERT Q/K C_k ---
    print("\n  P1: DeBERTa Q/K C_k differs from DistilBERT Q/K C_k")
    for mod in ["Q", "K"]:
        dist = df[(df["model"] == "distilbert") & (df["module"] == mod)]["C_k"]
        deb = df[(df["model"] == "deberta") & (df["module"] == mod)]["C_k"]
        if len(dist) == 0 or len(deb) == 0:
            print(f"    {mod}: insufficient data")
            continue
        U_stat, p_val = stats.mannwhitneyu(dist, deb, alternative="two-sided")
        # Cohen's d (approximate from means/pooled SD)
        pooled_sd = np.sqrt((dist.std()**2 + deb.std()**2) / 2)
        d = (dist.mean() - deb.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        print(f"    {mod}: DistilBERT mean={dist.mean():.4f} (n={len(dist)}), "
              f"DeBERTa mean={deb.mean():.4f} (n={len(deb)})")
        print(f"       Mann-Whitney U={U_stat:.1f}, p={p_val:.4f}, Cohen's d={d:.3f}")
        results[f"P1_{mod}_U"] = float(U_stat)
        results[f"P1_{mod}_p"] = float(p_val)
        results[f"P1_{mod}_d"] = float(d)

    # --- P3: V/O effect sizes vs Q/K effect sizes ---
    print("\n  P3: V/O more similar across architectures than Q/K")
    effect_sizes = {}
    for mod in MODULES:
        dist = df[(df["model"] == "distilbert") & (df["module"] == mod)]["C_k"]
        deb = df[(df["model"] == "deberta") & (df["module"] == mod)]["C_k"]
        if len(dist) == 0 or len(deb) == 0:
            effect_sizes[mod] = float("nan")
            continue
        pooled_sd = np.sqrt((dist.std()**2 + deb.std()**2) / 2)
        d = abs(dist.mean() - deb.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        effect_sizes[mod] = d

    qk_d = np.mean([effect_sizes.get("Q", 0), effect_sizes.get("K", 0)])
    vo_d = np.mean([effect_sizes.get("V", 0), effect_sizes.get("O", 0)])
    print(f"    Q/K mean |d| = {qk_d:.3f}  (Q={effect_sizes.get('Q', 0):.3f}, K={effect_sizes.get('K', 0):.3f})")
    print(f"    V/O mean |d| = {vo_d:.3f}  (V={effect_sizes.get('V', 0):.3f}, O={effect_sizes.get('O', 0):.3f})")
    print(f"    P3 {'SUPPORTED' if qk_d > vo_d else 'NOT SUPPORTED'}: "
          f"Q/K diverge {'more' if qk_d > vo_d else 'less'} than V/O")
    results["P3_qk_mean_d"] = float(qk_d)
    results["P3_vo_mean_d"] = float(vo_d)
    results["P3_supported"] = bool(qk_d > vo_d)

    # --- P4: RoBERTa Q/K resembles DistilBERT more than DeBERTa ---
    print("\n  P4: RoBERTa Q/K resembles DistilBERT more than DeBERTa")
    for mod in ["Q", "K"]:
        dist = df[(df["model"] == "distilbert") & (df["module"] == mod)]["C_k"]
        rob = df[(df["model"] == "roberta") & (df["module"] == mod)]["C_k"]
        deb = df[(df["model"] == "deberta") & (df["module"] == mod)]["C_k"]
        if len(dist) == 0 or len(rob) == 0 or len(deb) == 0:
            print(f"    {mod}: insufficient data")
            continue
        # Distance = |mean difference|
        d_rob_dist = abs(rob.mean() - dist.mean())
        d_deb_dist = abs(deb.mean() - dist.mean())
        print(f"    {mod}: |RoBERTa - DistilBERT| = {d_rob_dist:.4f}, "
              f"|DeBERTa - DistilBERT| = {d_deb_dist:.4f}")
        print(f"       P4 {'SUPPORTED' if d_rob_dist < d_deb_dist else 'NOT SUPPORTED'} "
              f"for {mod}")
        results[f"P4_{mod}_rob_dist_gap"] = float(d_rob_dist)
        results[f"P4_{mod}_deb_dist_gap"] = float(d_deb_dist)
        results[f"P4_{mod}_supported"] = bool(d_rob_dist < d_deb_dist)

    return results


# ---------------------------------------------------------------------------
# Phase 3: Within-architecture C_k variance (P2)
# ---------------------------------------------------------------------------

def phase3_ck_variance(df: pd.DataFrame) -> dict:
    """Test P2: DeBERTa Q/K has lower cross-layer C_k variance."""
    print("\n=== Phase 3: Cross-Layer C_k Variance (P2) ===")
    results = {}

    print(f"\n  {'Model':<12} {'Module':<6} {'C_k SD':>9} {'C_k CV':>9} {'C_k range':>10}")
    print("  " + "-" * 50)

    for model_key in MODELS:
        for mod in MODULES:
            subset = df[(df["model"] == model_key) & (df["module"] == mod)]
            if len(subset) == 0:
                continue
            ck = subset["C_k"]
            sd = ck.std()
            cv = sd / ck.mean() if ck.mean() > 0 else 0.0
            rng = ck.max() - ck.min()
            print(f"  {model_key:<12} {mod:<6} {sd:9.4f} {cv:9.4f} {rng:10.4f}")
            results[f"{model_key}_{mod}_cv"] = float(cv)
            results[f"{model_key}_{mod}_range"] = float(rng)

    # Formal Levene's test: DeBERTa Q/K vs DistilBERT Q/K
    print("\n  Levene's test for equality of C_k variance (Q/K only):")
    for mod in ["Q", "K"]:
        dist = df[(df["model"] == "distilbert") & (df["module"] == mod)]["C_k"]
        deb = df[(df["model"] == "deberta") & (df["module"] == mod)]["C_k"]
        if len(dist) < 2 or len(deb) < 2:
            continue
        stat, p_val = stats.levene(dist, deb, center="median")
        print(f"    {mod}: Levene W={stat:.3f}, p={p_val:.4f}")
        print(f"       DistilBERT SD={dist.std():.4f}, DeBERTa SD={deb.std():.4f}")
        direction = "DeBERTa lower" if deb.std() < dist.std() else "DistilBERT lower"
        print(f"       Direction: {direction}")
        results[f"P2_{mod}_levene_W"] = float(stat)
        results[f"P2_{mod}_levene_p"] = float(p_val)
        results[f"P2_{mod}_deb_lower_var"] = bool(deb.std() < dist.std())

    return results


# ---------------------------------------------------------------------------
# Phase 4: Spectral shape comparison
# ---------------------------------------------------------------------------

def phase4_spectral_shape(df: pd.DataFrame) -> dict:
    """Compare spectral decay shapes across architectures."""
    print("\n=== Phase 4: Spectral Shape Comparison ===")
    results = {}

    # Also compare erank_w0 and srank_w0 distributions
    print("\n  W_0 effective rank (erank) by model and module:")
    print(f"  {'Model':<12} {'Module':<6} {'mean erank':>11} {'mean srank':>11}")
    print("  " + "-" * 44)
    for model_key in MODELS:
        for mod in MODULES:
            subset = df[(df["model"] == model_key) & (df["module"] == mod)]
            if len(subset) == 0:
                continue
            print(f"  {model_key:<12} {mod:<6} {subset['erank_w0'].mean():11.2f} "
                  f"{subset['srank_w0'].mean():11.2f}")
            results[f"{model_key}_{mod}_erank_w0"] = float(subset["erank_w0"].mean())
            results[f"{model_key}_{mod}_srank_w0"] = float(subset["srank_w0"].mean())

    # KS test on cumulative energy curves between DistilBERT and DeBERTa
    print("\n  KS test on cumulative energy curves (DistilBERT vs DeBERTa):")
    for mod in MODULES:
        dist_curves = df[(df["model"] == "distilbert") & (df["module"] == mod)]["spectral_curve"]
        deb_curves = df[(df["model"] == "deberta") & (df["module"] == mod)]["spectral_curve"]
        if len(dist_curves) == 0 or len(deb_curves) == 0:
            continue
        # Average the curves within each model, then compare
        dist_mean = np.mean([np.array(c) for c in dist_curves], axis=0)
        deb_mean = np.mean([np.array(c) for c in deb_curves], axis=0)
        ks_stat, ks_p = stats.ks_2samp(dist_mean, deb_mean)
        print(f"    {mod}: KS={ks_stat:.4f}, p={ks_p:.4f}")
        results[f"KS_{mod}_stat"] = float(ks_stat)
        results[f"KS_{mod}_p"] = float(ks_p)

    return results


# ---------------------------------------------------------------------------
# Phase 5: Summary and verdict
# ---------------------------------------------------------------------------

def phase5_verdict(all_results: dict) -> str:
    """Interpret results against predictions."""
    print("\n=== Phase 5: Verdict ===\n")

    lines = []

    # P1
    p1_q = all_results.get("P1_Q_p", 1.0)
    p1_k = all_results.get("P1_K_p", 1.0)
    p1_pass = p1_q < 0.05 or p1_k < 0.05
    lines.append(f"P1 (Q/K C_k differs): {'PASS' if p1_pass else 'FAIL'} "
                 f"(Q p={p1_q:.4f}, K p={p1_k:.4f})")

    # P2
    p2_q_lower = all_results.get("P2_Q_deb_lower_var", False)
    p2_k_lower = all_results.get("P2_K_deb_lower_var", False)
    p2_q_p = all_results.get("P2_Q_levene_p", 1.0)
    p2_k_p = all_results.get("P2_K_levene_p", 1.0)
    p2_pass = (p2_q_lower or p2_k_lower) and (p2_q_p < 0.10 or p2_k_p < 0.10)
    lines.append(f"P2 (DeBERTa Q/K lower C_k variance): {'PASS' if p2_pass else 'FAIL'} "
                 f"(Q: lower={p2_q_lower}, p={p2_q_p:.4f}; "
                 f"K: lower={p2_k_lower}, p={p2_k_p:.4f})")

    # P3
    p3_pass = all_results.get("P3_supported", False)
    qk_d = all_results.get("P3_qk_mean_d", 0)
    vo_d = all_results.get("P3_vo_mean_d", 0)
    lines.append(f"P3 (Q/K diverge more than V/O): {'PASS' if p3_pass else 'FAIL'} "
                 f"(Q/K d={qk_d:.3f}, V/O d={vo_d:.3f})")

    # P4
    p4_q = all_results.get("P4_Q_supported", False)
    p4_k = all_results.get("P4_K_supported", False)
    p4_pass = p4_q and p4_k
    lines.append(f"P4 (RoBERTa Q/K closer to DistilBERT): {'PASS' if p4_pass else 'FAIL'} "
                 f"(Q={p4_q}, K={p4_k})")

    # Overall interpretation
    lines.append("")
    n_pass = sum([p1_pass, p2_pass, p3_pass, p4_pass])

    if n_pass >= 3:
        interp = (
            "STRONG SUPPORT for disentangled-attention hypothesis. "
            "DeBERTa's content-position decomposition changes the W_0 spectral "
            "structure in Q/K projections specifically, reducing the constraint "
            "channel through which C_k drives adapter convergence."
        )
    elif n_pass >= 2:
        interp = (
            "MODERATE SUPPORT for disentangled-attention hypothesis. "
            "Some predictions confirmed but the evidence is mixed. "
            "Alternative explanations (RTD pretraining, erank floor effect) "
            "cannot be ruled out."
        )
    elif p1_pass and not p3_pass:
        interp = (
            "ARCHITECTURE DIFFERENCE EXISTS but is NOT specific to Q/K. "
            "Alt-B (RTD pretraining) or Alt-C (layer count) more likely than "
            "the disentangled-attention mechanism."
        )
    else:
        interp = (
            "NOT SUPPORTED. DeBERTa and DistilBERT W_0 concentration profiles "
            "are similar. The C_k → alignment null on DeBERTa is likely explained "
            "by Alt-A (compressed erank range / floor effect) rather than "
            "architectural spectral differences."
        )

    lines.append(f"Overall ({n_pass}/4 predictions confirmed): {interp}")

    verdict = "\n".join(lines)
    print(verdict)
    return verdict


# ---------------------------------------------------------------------------
# Phase 6: Figures
# ---------------------------------------------------------------------------

def phase6_figures(df: pd.DataFrame) -> None:
    """Generate comparison figures."""
    print("\n=== Phase 6: Generating Figures ===")

    # Figure 1: C_k by model and module (box plot)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle("W₀ Energy Concentration (C_k) by Architecture and Module", fontsize=13)

    model_order = ["distilbert", "roberta", "deberta"]
    colors = {"distilbert": "#2196F3", "roberta": "#4CAF50", "deberta": "#FF9800"}

    for i, mod in enumerate(MODULES):
        ax = axes[i]
        data = []
        labels = []
        for model_key in model_order:
            subset = df[(df["model"] == model_key) & (df["module"] == mod)]["C_k"]
            if len(subset) > 0:
                data.append(subset.values)
                labels.append(model_key)

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, model_key in zip(bp["boxes"], labels):
            patch.set_facecolor(colors.get(model_key, "#999"))
            patch.set_alpha(0.7)

        ax.set_title(f"{mod} projection")
        ax.set_ylabel("C_k" if i == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_ck_by_model_module.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_ck_by_model_module.png")

    # Figure 2: C_k across layers (line plot) for each module
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("C_k Across Layers (Normalized Layer Position)", fontsize=13)

    for idx, mod in enumerate(MODULES):
        ax = axes[idx // 2][idx % 2]
        for model_key in model_order:
            subset = df[(df["model"] == model_key) & (df["module"] == mod)]
            if len(subset) == 0:
                continue
            subset_sorted = subset.sort_values("layer_frac")
            ax.plot(subset_sorted["layer_frac"], subset_sorted["C_k"],
                    "o-", label=model_key, color=colors[model_key], alpha=0.8,
                    markersize=4)

        ax.set_title(f"{mod} projection")
        ax.set_xlabel("Normalized layer position")
        ax.set_ylabel("C_k")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_ck_across_layers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_ck_across_layers.png")

    # Figure 3: Mean spectral energy curves by model and module
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Mean Cumulative Energy Curves (W₀)", fontsize=13)

    for idx, mod in enumerate(MODULES):
        ax = axes[idx // 2][idx % 2]
        for model_key in model_order:
            subset = df[(df["model"] == model_key) & (df["module"] == mod)]
            if len(subset) == 0:
                continue
            curves = np.array([np.array(c) for c in subset["spectral_curve"]])
            mean_curve = curves.mean(axis=0)
            x = np.linspace(0, 1, len(mean_curve))
            ax.plot(x, mean_curve, label=model_key, color=colors[model_key], linewidth=2)

        ax.set_title(f"{mod} projection")
        ax.set_xlabel("Fraction of singular values")
        ax.set_ylabel("Cumulative energy fraction")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_energy_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_energy_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("N135: C_k Architecture-Specificity Investigation")
    print("=" * 60)

    # Phase 1: Extract
    df = phase1_extract_all()

    # Phase 2: Cross-architecture comparison
    results_p2 = phase2_ck_comparison(df)

    # Phase 3: Variance comparison
    results_p3 = phase3_ck_variance(df)

    # Phase 4: Spectral shape
    results_p4 = phase4_spectral_shape(df)

    # Merge all results
    all_results = {**results_p2, **results_p3, **results_p4}

    # Phase 5: Verdict
    verdict = phase5_verdict(all_results)

    # Phase 6: Figures
    phase6_figures(df)

    # Save results
    all_results["verdict"] = verdict
    all_results["elapsed_seconds"] = time.time() - t0

    with open(OUTPUT_DIR / "n135_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    print(f"  Results saved to {OUTPUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
