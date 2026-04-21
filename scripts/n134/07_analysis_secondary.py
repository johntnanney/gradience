"""N134 secondary exploratory analysis (explicitly non-evidential).

All results in this script are exploratory. They are reported for
transparency and to guide future experiment design, not as evidence
for or against any hypothesis.

Contents:
  1. Per-module alignment heterogeneity (Q/K vs V/O asymmetry)
  2. Within-module C_k -> alignment partial correlation (N132/N133 null replication)
  3. Layer-depth trend in same/cross alignment ratio
  4. Ten N133 composite risk scores evaluated on N134 data

Inputs:
    /workspace/n134/audit/pair_alignment_full.json
    /workspace/n134/audit/pair_alignment_summary.json
    /workspace/n134/audit/adapter_profiles.json
    /workspace/n134/pair_sample.json
    /workspace/n134/merges/merge_eval_summary.json

Output:
    /workspace/n134/analysis_secondary.json
    /workspace/n134/figures/secondary_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# WORKSPACE defaults to the pod-absolute path that produced the committed
# N134 artifacts. Replicators running outside that environment can
# override via the N134_WORKSPACE environment variable without editing
# the script; the default preserves pod-era provenance.
import os  # noqa: E402

WORKSPACE = Path(os.environ.get("N134_WORKSPACE", "/workspace/n134"))
AUDIT_DIR = WORKSPACE / "audit"
MERGE_DIR = WORKSPACE / "merges"
FIG_DIR = WORKSPACE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 32

# FAMILY_B partition (same as 06_analysis_h1.py)
FAMILY_B: dict[str, str] = {
    "arc_challenge": "science_qa",
    "hellaswag": "commonsense_completion",
    "winogrande": "coreference_commonsense",
    "openbookqa": "knowledge_qa",
    "commonsenseqa": "commonsense_qa",
    "piqa": "physical_commonsense",
    "siqa": "social_commonsense",
    "boolq": "reading_comprehension",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_MODULE_ALIAS = {
    "q_proj": "Q", "k_proj": "K", "v_proj": "V", "o_proj": "O",
    "Q": "Q", "K": "K", "V": "V", "O": "O",
}


def _normalize_per_layer(pair_full: dict) -> dict:
    """Rewrite each layer entry to use short module labels (Q/K/V/O) and
    an integer "layer" key, tolerating the v2.1 audit schema
    (o_proj/layer_idx) and the original spec (O/layer).
    """
    for pair in pair_full.values():
        for layer in pair.get("per_layer", []):
            mod = layer.get("module")
            if mod in _MODULE_ALIAS:
                layer["module"] = _MODULE_ALIAS[mod]
            if "layer" not in layer and "layer_idx" in layer:
                layer["layer"] = int(layer["layer_idx"])
    return pair_full


def load_inputs() -> tuple[dict, dict, dict, dict, dict]:
    pair_full = json.loads((AUDIT_DIR / "pair_alignment_full.json").read_text())
    pair_full = _normalize_per_layer(pair_full)
    pair_summary = json.loads((AUDIT_DIR / "pair_alignment_summary.json").read_text())
    profiles = json.loads((AUDIT_DIR / "adapter_profiles.json").read_text())
    pair_sample = json.loads((WORKSPACE / "pair_sample.json").read_text())
    merges = json.loads((MERGE_DIR / "merge_eval_summary.json").read_text())
    return pair_full, pair_summary, profiles, pair_sample, merges


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def max_degradation(merge_entry: dict) -> float:
    ta, tb = merge_entry["task_a"], merge_entry["task_b"]
    degs = []
    for t in [ta, tb]:
        k = f"degradation_{t}"
        if k in merge_entry:
            degs.append(merge_entry[k])
    return max(degs) if degs else float("nan")


def build_eval_pairs(merges: dict) -> dict[str, float]:
    out = {}
    for r in merges["results"]:
        if r.get("category") == "same_task":
            continue
        deg = max_degradation(r)
        if np.isfinite(deg):
            out[r["pair"]] = deg
    return out


def family_pair_label(task_a: str, task_b: str) -> str:
    fa = FAMILY_B.get(task_a, task_a)
    fb = FAMILY_B.get(task_b, task_b)
    return "|".join(sorted([fa, fb]))


def dummy_encode(labels: list[str]) -> np.ndarray:
    uniq = sorted(set(labels))
    cols = uniq[1:]
    n = len(labels)
    X = np.ones((n, 1 + len(cols)))
    for j, c in enumerate(cols):
        for i, lab in enumerate(labels):
            X[i, 1 + j] = 1.0 if lab == c else 0.0
    return X


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(x: np.ndarray, y: np.ndarray, X_cov: np.ndarray) -> tuple[float, float]:
    resid_x = ols_residuals(x, X_cov)
    resid_y = ols_residuals(y, X_cov)
    if resid_x.std() < 1e-10 or resid_y.std() < 1e-10:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(resid_x, resid_y)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# Per-layer / per-module extraction
# ---------------------------------------------------------------------------

def module_alignments(pair: dict, module: str) -> np.ndarray:
    return np.array([
        layer["alignment"]
        for layer in pair["per_layer"]
        if layer["module"] == module
    ])


def module_alignments_by_depth(pair: dict, module: str) -> tuple[np.ndarray, np.ndarray]:
    entries = sorted(
        [(layer["layer"], layer["alignment"])
         for layer in pair["per_layer"]
         if layer["module"] == module],
        key=lambda x: x[0],
    )
    return (
        np.array([e[0] for e in entries], dtype=float),
        np.array([e[1] for e in entries], dtype=float),
    )


def module_eranks(pair: dict, module: str) -> np.ndarray:
    """Per-layer erank for a given module (from per_layer if available)."""
    return np.array([
        layer.get("erank", float("nan"))
        for layer in pair["per_layer"]
        if layer["module"] == module
    ])


def per_layer_ck(pair: dict) -> np.ndarray:
    """Per-layer C_k values if stored in pair data."""
    return np.array([
        layer.get("ck", layer.get("condition_number", float("nan")))
        for layer in pair["per_layer"]
    ])


# ---------------------------------------------------------------------------
# 1. Per-module alignment heterogeneity
# ---------------------------------------------------------------------------

def analyze_module_heterogeneity(pair_full: dict) -> dict:
    """Compare Q/K vs V/O erank and alignment asymmetry across pairs."""
    print("\n  1. Per-module alignment heterogeneity")
    print("  " + "-" * 50)

    module_means: dict[str, list[float]] = {m: [] for m in ["Q", "K", "V", "O"]}
    qk_vs_vo: list[dict] = []

    for key, pair in pair_full.items():
        row: dict[str, float] = {}
        for m in ["Q", "K", "V", "O"]:
            vals = module_alignments(pair, m)
            if len(vals) > 0:
                mean_val = float(vals.mean())
                module_means[m].append(mean_val)
                row[m] = mean_val

        if all(m in row for m in ["Q", "K", "V", "O"]):
            qk_mean = 0.5 * (row["Q"] + row["K"])
            vo_mean = 0.5 * (row["V"] + row["O"])
            qk_vs_vo.append({
                "pair": key,
                "qk_mean": qk_mean,
                "vo_mean": vo_mean,
                "ratio_vo_qk": vo_mean / (qk_mean + 1e-12),
            })

    # Summary
    for m in ["Q", "K", "V", "O"]:
        arr = np.array(module_means[m])
        print(f"    {m}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"n={len(arr)}")

    if qk_vs_vo:
        ratios = np.array([r["ratio_vo_qk"] for r in qk_vs_vo])
        print(f"\n    V+O / Q+K ratio: mean={ratios.mean():.2f}, "
              f"median={np.median(ratios):.2f}")

    result = {
        "module_grand_means": {m: float(np.mean(module_means[m]))
                               for m in ["Q", "K", "V", "O"] if module_means[m]},
        "module_grand_stds": {m: float(np.std(module_means[m]))
                              for m in ["Q", "K", "V", "O"] if module_means[m]},
        "vo_qk_ratio_mean": float(np.mean([r["ratio_vo_qk"] for r in qk_vs_vo])) if qk_vs_vo else None,
        "vo_qk_ratio_median": float(np.median([r["ratio_vo_qk"] for r in qk_vs_vo])) if qk_vs_vo else None,
        "n_pairs": len(qk_vs_vo),
    }
    return result


# ---------------------------------------------------------------------------
# 2. Within-module C_k -> alignment partial correlation
# ---------------------------------------------------------------------------

def analyze_ck_alignment(pair_full: dict) -> dict:
    """Replication of N132/N133 null: C_k does not predict alignment."""
    print("\n  2. Within-module C_k -> alignment partial correlation")
    print("  " + "-" * 50)

    # Gather per-layer (ck, alignment) tuples across all pairs
    ck_vals = []
    align_vals = []
    for pair in pair_full.values():
        for layer in pair["per_layer"]:
            ck = layer.get("ck", layer.get("condition_number", None))
            align = layer.get("alignment", None)
            if ck is not None and align is not None and np.isfinite(ck) and np.isfinite(align):
                ck_vals.append(ck)
                align_vals.append(align)

    if len(ck_vals) < 10:
        print("    Insufficient C_k data for analysis")
        return {"status": "insufficient_ck_data", "n_observations": len(ck_vals)}

    ck_arr = np.array(ck_vals)
    align_arr = np.array(align_vals)

    rho, p = stats.spearmanr(ck_arr, align_arr)
    r, pr = stats.pearsonr(ck_arr, align_arr)

    null_replicated = abs(rho) < 0.15 or p > 0.05

    print(f"    n observations: {len(ck_vals)}")
    print(f"    Spearman rho(C_k, alignment): {rho:+.4f} (p = {p:.4e})")
    print(f"    Pearson r(C_k, alignment):    {r:+.4f} (p = {pr:.4e})")
    print(f"    N132/N133 null replicated: {'YES' if null_replicated else 'NO'}")

    return {
        "n_observations": len(ck_vals),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "pearson_r": float(r),
        "pearson_p": float(pr),
        "null_replicated": null_replicated,
    }


# ---------------------------------------------------------------------------
# 3. Layer-depth trend in same/cross ratio
# ---------------------------------------------------------------------------

def analyze_depth_trend(pair_full: dict) -> dict:
    """Layer-depth trend in same-task vs cross-task alignment ratio."""
    print("\n  3. Layer-depth trend in same/cross alignment ratio")
    print("  " + "-" * 50)

    # Per-layer same vs cross alignment means
    same_by_layer: dict[int, list[float]] = {i: [] for i in range(N_LAYERS)}
    cross_by_layer: dict[int, list[float]] = {i: [] for i in range(N_LAYERS)}

    for pair in pair_full.values():
        is_same = pair.get("is_same_task", False)
        for layer in pair["per_layer"]:
            l_idx = layer["layer"]
            a = layer["alignment"]
            if 0 <= l_idx < N_LAYERS and np.isfinite(a):
                if is_same:
                    same_by_layer[l_idx].append(a)
                else:
                    cross_by_layer[l_idx].append(a)

    depths = []
    ratios = []
    same_means = []
    cross_means = []

    for l_idx in range(N_LAYERS):
        if same_by_layer[l_idx] and cross_by_layer[l_idx]:
            s_mean = float(np.mean(same_by_layer[l_idx]))
            c_mean = float(np.mean(cross_by_layer[l_idx]))
            depths.append(l_idx)
            same_means.append(s_mean)
            cross_means.append(c_mean)
            ratios.append(s_mean / (c_mean + 1e-12))

    if len(depths) < 4:
        print("    Insufficient per-layer data")
        return {"status": "insufficient_data"}

    depths_arr = np.array(depths, dtype=float)
    ratios_arr = np.array(ratios)

    # Linear trend
    slope, intercept, r_val, p_val, _ = stats.linregress(depths_arr, ratios_arr)

    print(f"    Layers with data: {len(depths)}")
    print(f"    Ratio range: [{ratios_arr.min():.2f}, {ratios_arr.max():.2f}]")
    print(f"    Linear trend: slope={slope:.4f}, r={r_val:.3f}, p={p_val:.4e}")
    print(f"    Layer 0 ratio: {ratios_arr[0]:.2f}, Layer {depths[-1]} ratio: {ratios_arr[-1]:.2f}")

    return {
        "n_layers_with_data": len(depths),
        "ratio_range": [float(ratios_arr.min()), float(ratios_arr.max())],
        "trend_slope": float(slope),
        "trend_intercept": float(intercept),
        "trend_r": float(r_val),
        "trend_p": float(p_val),
        "per_layer": [
            {"layer": int(d), "same_mean": float(sm), "cross_mean": float(cm), "ratio": float(r)}
            for d, sm, cm, r in zip(depths, same_means, cross_means, ratios)
        ],
    }


# ---------------------------------------------------------------------------
# 4. Ten N133 composite risk scores on N134 data
# ---------------------------------------------------------------------------

def compute_composite_scores(pair_full: dict, profiles: dict) -> dict[str, dict[str, float]]:
    """Compute all 10 N133 composite risk scores for each pair."""

    adapter_erank = {}
    for name, profile in profiles.items():
        adapter_erank[name] = float(np.mean([layer["erank"] for layer in profile["per_layer"]]))

    scores: dict[str, dict[str, float]] = {}

    for key, pair in pair_full.items():
        a = pair["adapter_a"]
        b = pair["adapter_b"]

        per_layer = pair["per_layer"]

        # Per-module alignments
        mod_aligns: dict[str, np.ndarray] = {}
        for m in ["Q", "K", "V", "O"]:
            mod_aligns[m] = module_alignments(pair, m)

        # Per-module depth-weighted
        mod_depth: dict[str, float] = {}
        for m in ["Q", "K", "V", "O"]:
            d, al = module_alignments_by_depth(pair, m)
            if len(d) > 0:
                w = (d + 1.0) / N_LAYERS
                mod_depth[m] = float(np.sum(w * al) / np.sum(w))
            else:
                mod_depth[m] = float("nan")

        all_alignments = np.array([layer["alignment"] for layer in per_layer])

        # Erank ratio
        ea = adapter_erank.get(a, float("nan"))
        eb = adapter_erank.get(b, float("nan"))
        erank_ratio = min(ea, eb) / (max(ea, eb) + 1e-12) if np.isfinite(ea) and np.isfinite(eb) else float("nan")

        # C_k gated: layers where C_k > 0.35
        ck_gated = []
        ck_gated_o = []
        for layer in per_layer:
            ck = layer.get("ck", layer.get("condition_number", None))
            if ck is not None and ck > 0.35:
                ck_gated.append(layer["alignment"])
                if layer["module"] == "O":
                    ck_gated_o.append(layer["alignment"])

        # JSD (spectral shape divergence) per layer if available
        jsd_vals = []
        for layer in per_layer:
            jsd = layer.get("jsd", layer.get("spectral_divergence", None))
            if jsd is not None and np.isfinite(jsd):
                jsd_vals.append(jsd)

        scores[key] = {
            # 1. mean_alignment
            "mean_alignment": float(all_alignments.mean()) if len(all_alignments) else float("nan"),
            # 2. O_only_mean
            "O_only_mean": float(mod_aligns["O"].mean()) if len(mod_aligns["O"]) else float("nan"),
            # 3. O_deep_mean (layers 16-31)
            "O_deep_mean": float(np.mean([
                layer["alignment"] for layer in per_layer
                if layer["module"] == "O" and layer["layer"] >= 16
            ])) if any(layer["module"] == "O" and layer["layer"] >= 16 for layer in per_layer) else float("nan"),
            # 4. V_mean
            "V_mean": float(mod_aligns["V"].mean()) if len(mod_aligns["V"]) else float("nan"),
            # 5. OV_mean (O+V only)
            "OV_mean": float(np.mean(
                list(mod_aligns["O"]) + list(mod_aligns["V"])
            )) if len(mod_aligns["O"]) + len(mod_aligns["V"]) > 0 else float("nan"),
            # 6. p90_align
            "p90_align": float(np.percentile(all_alignments, 90)) if len(all_alignments) else float("nan"),
            # 7. ck_gated_mean (C_k > 0.35)
            "ck_gated_mean": float(np.mean(ck_gated)) if ck_gated else float("nan"),
            # 8. ck_gated_O_mean
            "ck_gated_O_mean": float(np.mean(ck_gated_o)) if ck_gated_o else float("nan"),
            # 9. erank_ratio
            "erank_ratio": erank_ratio,
            # 10. mean_jsd (spectral shape divergence)
            "mean_jsd": float(np.mean(jsd_vals)) if jsd_vals else float("nan"),
        }

    return scores


def evaluate_composites(
    composite_scores: dict[str, dict[str, float]],
    eval_degs: dict[str, float],
    pair_full: dict,
    h1_rho: float,
    h1_dr2: float,
) -> list[dict]:
    """Evaluate each composite score against max_degradation."""
    eval_pairs = sorted(eval_degs.keys())
    y = np.array([eval_degs[p] for p in eval_pairs])

    def _lookup_in(p: str, d: dict) -> dict:
        if p in d:
            return d[p]
        parts = p.split("_vs_")
        if len(parts) == 2:
            rev = f"{parts[1]}_vs_{parts[0]}"
            if rev in d:
                return d[rev]
        raise KeyError(f"Pair not found: {p}")

    def _lookup(p: str) -> dict:
        return _lookup_in(p, pair_full)

    cell_labels = [family_pair_label(_lookup(p)["task_a"], _lookup(p)["task_b"])
                   for p in eval_pairs]
    X_fam = dummy_encode(cell_labels)
    r2_base = ols_r2(y, X_fam)

    variants = [
        "mean_alignment", "O_only_mean", "O_deep_mean", "V_mean",
        "OV_mean", "p90_align", "ck_gated_mean", "ck_gated_O_mean",
        "erank_ratio", "mean_jsd",
    ]

    results = []
    for v in variants:
        x = np.array([_lookup_in(p, composite_scores).get(v, float("nan")) for p in eval_pairs])

        if not np.all(np.isfinite(x)):
            n_finite = int(np.isfinite(x).sum())
            results.append({
                "variant": v,
                "status": "incomplete_data",
                "n_finite": n_finite,
                "n_total": len(eval_pairs),
            })
            continue

        rho_raw, p_raw = stats.spearmanr(x, y)
        rho_partial, p_partial = partial_spearman(x, y, X_fam)

        X_full = np.hstack([X_fam, x.reshape(-1, 1)])
        r2_full = ols_r2(y, X_full)
        delta_r2 = r2_full - r2_base

        exceeds_h1 = (
            np.isfinite(rho_partial)
            and rho_partial >= 0.50
            and p_partial < 0.05
            and delta_r2 >= 0.10
        )

        results.append({
            "variant": v,
            "spearman_raw": float(rho_raw),
            "spearman_raw_p": float(p_raw),
            "spearman_partial": float(rho_partial) if np.isfinite(rho_partial) else None,
            "spearman_partial_p": float(p_partial) if np.isfinite(p_partial) else None,
            "delta_r2": float(delta_r2),
            "exceeds_h1_thresholds": exceeds_h1,
        })

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_module_heterogeneity(pair_full: dict, out_path: Path) -> None:
    """Box plot of per-module alignment distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for m in ["Q", "K", "V", "O"]:
        vals = []
        for pair in pair_full.values():
            arr = module_alignments(pair, m)
            if len(arr) > 0:
                vals.append(float(arr.mean()))
        data.append(vals)
        labels.append(m)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Mean alignment per pair", fontsize=11)
    ax.set_xlabel("Module", fontsize=11)
    ax.set_title("N134: per-module alignment heterogeneity", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_depth_trend(depth_result: dict, out_path: Path) -> None:
    """Same/cross alignment ratio by layer depth."""
    if depth_result.get("status") == "insufficient_data":
        return

    per_layer = depth_result["per_layer"]
    depths = [d["layer"] for d in per_layer]
    ratios = [d["ratio"] for d in per_layer]
    same_means = [d["same_mean"] for d in per_layer]
    cross_means = [d["cross_mean"] for d in per_layer]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: same vs cross by depth
    ax = axes[0]
    ax.plot(depths, same_means, "o-", color="#27ae60", label="Same-task mean", markersize=4)
    ax.plot(depths, cross_means, "s-", color="#c0392b", label="Cross-task mean", markersize=4)
    ax.set_xlabel("Layer depth", fontsize=11)
    ax.set_ylabel("Mean alignment", fontsize=11)
    ax.set_title("Same vs cross alignment by depth", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: ratio by depth
    ax = axes[1]
    ax.plot(depths, ratios, "o-", color="#8e44ad", markersize=5)
    slope = depth_result["trend_slope"]
    intercept = depth_result["trend_intercept"]
    x_fit = np.array(depths, dtype=float)
    ax.plot(x_fit, slope * x_fit + intercept, "--", color="gray", alpha=0.7,
            label=f"slope={slope:.4f}, r={depth_result['trend_r']:.3f}")
    ax.set_xlabel("Layer depth", fontsize=11)
    ax.set_ylabel("Same/cross ratio", fontsize=11)
    ax.set_title("Same/cross alignment ratio by depth", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("N134: layer-depth trend", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_composite_comparison(composite_results: list[dict], out_path: Path) -> None:
    """Bar chart of all 10 composites: partial rho and delta-R2."""
    valid = [r for r in composite_results if r.get("status") != "incomplete_data"]
    if not valid:
        return

    names = [r["variant"] for r in valid]
    rhos = [r.get("spearman_partial", 0) or 0 for r in valid]
    dr2s = [r["delta_r2"] for r in valid]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Partial Spearman rho
    ax = axes[0]
    ax.barh(names, rhos, color="#3498db", edgecolor="white")
    ax.axvline(0.50, color="red", linestyle=":", linewidth=2, label="H1 threshold (0.50)")
    ax.set_xlabel("Partial Spearman rho", fontsize=11)
    ax.set_title("Composite scores: family-residualized partial rho vs max_degradation", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    # Delta R2
    ax = axes[1]
    colors = ["#27ae60" if r.get("exceeds_h1_thresholds") else "#95a5a6" for r in valid]
    ax.barh(names, dr2s, color=colors, edgecolor="white")
    ax.axvline(0.10, color="red", linestyle=":", linewidth=2, label="H1 threshold (0.10)")
    ax.set_xlabel("Delta R2 over family baseline", fontsize=11)
    ax.set_title("Composite scores: delta-R2 over FAMILY_B baseline", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("N134 secondary: 10 composite risk scores", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 74)
    print("  N134 secondary exploratory analysis (non-evidential)")
    print("=" * 74)

    pair_full, _pair_summary, profiles, _pair_sample, merges = load_inputs()
    eval_degs = build_eval_pairs(merges)
    eval_pairs = sorted(eval_degs.keys())
    print(f"\n  Loaded {len(pair_full)} pairs, {len(profiles)} profiles, "
          f"{len(eval_pairs)} evaluated cross-task pairs")

    result: dict = {
        "experiment": "N134 secondary exploratory analysis",
        "caveat": "All results in this file are exploratory and non-evidential.",
        "n_eval_pairs": len(eval_pairs),
    }

    # ---- 1. Module heterogeneity ----
    mod_het = analyze_module_heterogeneity(pair_full)
    result["module_heterogeneity"] = mod_het

    # ---- 2. C_k -> alignment ----
    ck_align = analyze_ck_alignment(pair_full)
    result["ck_alignment_correlation"] = ck_align

    # ---- 3. Depth trend ----
    depth_trend = analyze_depth_trend(pair_full)
    result["depth_trend"] = depth_trend

    # ---- 4. Ten composite scores ----
    print("\n  4. N133 composite risk scores on N134 data")
    print("  " + "-" * 50)

    composite_scores = compute_composite_scores(pair_full, profiles)

    # Use placeholder H1 values (actual values come from 06_analysis_h1.py)
    # These thresholds are the pre-registered decision criteria
    composite_results = evaluate_composites(
        composite_scores, eval_degs, pair_full,
        h1_rho=0.50, h1_dr2=0.10,
    )

    print(f"\n    {'Variant':22s} {'rho_raw':>8s} {'rho_partial':>12s} "
          f"{'delta_R2':>9s}   {'H1?':>6s}")
    for r in composite_results:
        if r.get("status") == "incomplete_data":
            print(f"    {r['variant']:22s}  (incomplete: {r['n_finite']}/{r['n_total']} finite)")
            continue
        rp = r["spearman_partial"]
        rp_str = f"{rp:+.3f}" if rp is not None else "  N/A"
        tag = "YES" if r.get("exceeds_h1_thresholds") else "no"
        print(f"    {r['variant']:22s} {r['spearman_raw']:+8.3f} {rp_str:>12s} "
              f"{r['delta_r2']:+9.3f}   {tag:>6s}")

    any_exceeds = any(r.get("exceeds_h1_thresholds") for r in composite_results)
    print(f"\n    Any composite exceeds H1 thresholds? {'YES' if any_exceeds else 'NO'}")
    result["composite_scores"] = composite_results
    result["any_composite_exceeds_h1"] = any_exceeds

    # ---- Figures ----
    print("\n" + "-" * 74)
    print("  FIGURES")
    print("-" * 74)

    p1 = FIG_DIR / "secondary_module_heterogeneity.png"
    plot_module_heterogeneity(pair_full, p1)
    print(f"\n  Saved {p1}")

    p2 = FIG_DIR / "secondary_depth_trend.png"
    plot_depth_trend(depth_trend, p2)
    print(f"  Saved {p2}")

    p3 = FIG_DIR / "secondary_composite_comparison.png"
    plot_composite_comparison(composite_results, p3)
    print(f"  Saved {p3}")

    # ---- Save JSON ----
    out_path = WORKSPACE / "analysis_secondary.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Saved {out_path}")

    print("\n" + "=" * 74)
    print("  N134 secondary analysis complete (all results exploratory)")
    print("=" * 74)


if __name__ == "__main__":
    main()
