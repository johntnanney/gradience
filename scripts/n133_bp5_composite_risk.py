"""N133 B-P5 follow-up: Composite merge-risk score.

Original B-P5 result (alignment-only triage) was a statistical anti-predictor
within the 12 evaluated cross-task pairs (Spearman +0.655 with max_degradation,
p=0.02). See sidecar/notes/n133_bp5_diagnostic.md.

This script builds a candidate composite merge-risk score:

    risk = f(O-module alignment, depth-weighted, min(erank_a, erank_b))

and validates it against the 12 evaluated pairs.

Inputs (all local, no GPU):
  sidecar/data/n133/pod_pull/audits/adapter_profiles.json
  sidecar/data/n133/pod_pull/audits/pair_alignment_full.json
  sidecar/data/n133/pod_pull/merges/merge_eval_summary.json

Output:
  sidecar/data/n133/bp5_composite_risk.json
  sidecar/data/n133/bp5_composite_risk_plot.png
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POD = PROJECT_ROOT / "sidecar" / "data" / "n133" / "pod_pull"
OUT = PROJECT_ROOT / "sidecar" / "data" / "n133"
OUT.mkdir(parents=True, exist_ok=True)

N_LAYERS = 32  # Mistral-7B transformer blocks


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def load_inputs():
    profiles = json.loads((POD / "audits" / "adapter_profiles.json").read_text())
    pair_full = json.loads((POD / "audits" / "pair_alignment_full.json").read_text())
    merges = json.loads((POD / "merges" / "merge_eval_summary.json").read_text())
    return profiles, pair_full, merges


def adapter_mean_erank(profile: dict) -> float:
    """Mean erank over all 128 LoRA-targeted layers."""
    return float(np.mean([layer["erank"] for layer in profile["per_layer"]]))


def module_alignments(pair: dict, module: str) -> np.ndarray:
    """Per-layer alignment for a given module ('Q','K','V','O')."""
    return np.array(
        [
            layer["alignment"]
            for layer in pair["per_layer"]
            if layer["module"] == module
        ]
    )


def module_alignments_by_depth(pair: dict, module: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (depths, alignments) for a given module, sorted by depth."""
    entries = [
        (layer["layer"], layer["alignment"])
        for layer in pair["per_layer"]
        if layer["module"] == module
    ]
    entries.sort(key=lambda x: x[0])
    d = np.array([e[0] for e in entries], dtype=float)
    a = np.array([e[1] for e in entries], dtype=float)
    return d, a


def depth_weights(depths: np.ndarray, style: str = "linear") -> np.ndarray:
    """Depth weights that emphasize deeper layers.

    Phase 2 found layer-depth SNR rising from 2.32× (layer 0) to 4.24× (layer 31).
    The linear weight (depth / (N_LAYERS - 1)) is a simple monotone choice.
    """
    if style == "linear":
        return depths / (N_LAYERS - 1)
    if style == "quadratic":
        return (depths / (N_LAYERS - 1)) ** 2
    if style == "uniform":
        return np.ones_like(depths)
    raise ValueError(f"unknown depth style: {style}")


def depth_weighted_module(pair: dict, module: str, style: str = "linear") -> float:
    """Depth-weighted mean alignment for a module."""
    d, a = module_alignments_by_depth(pair, module)
    w = depth_weights(d, style)
    w /= w.sum() + 1e-12
    return float(np.sum(w * a))


# ---------------------------------------------------------------------------
# Composite score construction
# ---------------------------------------------------------------------------

def build_features(profiles: dict, pair_full: dict) -> dict[str, dict]:
    """Build all candidate features for every pair."""
    adapter_erank = {name: adapter_mean_erank(p) for name, p in profiles.items()}

    features = {}
    for pair_key, pair in pair_full.items():
        a = pair["adapter_a"]
        b = pair["adapter_b"]

        feat = {
            "adapter_a": a,
            "adapter_b": b,
            "task_a": pair["task_a"],
            "task_b": pair["task_b"],
            "is_same_task": pair["is_same_task"],
            "mean_alignment": pair["mean_alignment"],
            "erank_a": adapter_erank[a],
            "erank_b": adapter_erank[b],
            "min_erank": min(adapter_erank[a], adapter_erank[b]),
            "mean_erank_pair": 0.5 * (adapter_erank[a] + adapter_erank[b]),
        }
        for mod in ["Q", "K", "V", "O"]:
            feat[f"{mod}_mean"] = float(module_alignments(pair, mod).mean())
            feat[f"{mod}_depth"] = depth_weighted_module(pair, mod, "linear")
            feat[f"{mod}_quad"] = depth_weighted_module(pair, mod, "quadratic")
        features[pair_key] = feat

    return features


def risk_score(feat: dict, variant: str) -> float:
    """Candidate composite scores to test."""
    if variant == "mean_alignment":
        return feat["mean_alignment"]
    if variant == "O_mean":
        return feat["O_mean"]
    if variant == "O_depth":
        return feat["O_depth"]
    if variant == "O_quad":
        return feat["O_quad"]
    if variant == "inv_min_erank":
        return 1.0 / feat["min_erank"]
    if variant == "O_depth_x_inv_erank":
        return feat["O_depth"] / feat["min_erank"]
    if variant == "O_quad_x_inv_erank":
        return feat["O_quad"] / feat["min_erank"]
    if variant == "z_sum":
        # will be filled in after z-normalizing across pairs
        raise ValueError("z_sum handled separately")
    if variant == "OVmix_depth":
        # Weighted combo of O and V (the two high-SNR modules from Phase 2)
        return 0.7 * feat["O_depth"] + 0.3 * feat["V_depth"]
    if variant == "OVmix_x_inv_erank":
        return (0.7 * feat["O_depth"] + 0.3 * feat["V_depth"]) / feat["min_erank"]
    raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def max_degradation(merge_entry: dict) -> float:
    ta, tb = merge_entry["task_a"], merge_entry["task_b"]
    degs = []
    for t in [ta, tb]:
        k = f"degradation_{t}"
        if k in merge_entry:
            degs.append(merge_entry[k])
    return max(degs) if degs else float("nan")


def evaluated_cross_task(merges: dict) -> dict[str, float]:
    """Return {pair_key: max_degradation} for evaluated cross-task pairs."""
    out = {}
    for r in merges["results"]:
        if r["category"] == "same_task":
            continue
        out[r["pair"]] = max_degradation(r)
    return out


def evaluate_variant(features, eval_degs, variant, verbose=True):
    """Compute Spearman ρ between risk score and actual max_degradation."""
    pairs = sorted(eval_degs.keys())
    scores = np.array([risk_score(features[p], variant) for p in pairs])
    degs = np.array([eval_degs[p] for p in pairs])

    rho, p_rho = stats.spearmanr(scores, degs)
    r, p_r = stats.pearsonr(scores, degs)

    if verbose:
        print(f"  {variant:28s}  ρ = {rho:+.4f} (p={p_rho:.4f})  "
              f"r = {r:+.4f} (p={p_r:.4f})")
    return {"variant": variant, "rho": float(rho), "p_rho": float(p_rho),
            "r": float(r), "p_r": float(p_r),
            "scores": scores.tolist(), "pairs": pairs, "max_deg": degs.tolist()}


def evaluate_z_sum(features, eval_degs):
    """z(O_depth) + z(1/min_erank) using stats from ALL cross-task pairs,
    not just evaluated subset (so the scoring is usable on unseen pairs)."""
    cross_feats = {k: v for k, v in features.items() if not v["is_same_task"]}

    od = np.array([v["O_depth"] for v in cross_feats.values()])
    ie = np.array([1.0 / v["min_erank"] for v in cross_feats.values()])
    od_mu, od_sd = od.mean(), od.std() + 1e-12
    ie_mu, ie_sd = ie.mean(), ie.std() + 1e-12

    pairs = sorted(eval_degs.keys())
    scores = []
    for p in pairs:
        f = features[p]
        z_od = (f["O_depth"] - od_mu) / od_sd
        z_ie = (1.0 / f["min_erank"] - ie_mu) / ie_sd
        scores.append(z_od + z_ie)
    scores = np.array(scores)
    degs = np.array([eval_degs[p] for p in pairs])
    rho, p_rho = stats.spearmanr(scores, degs)
    r, p_r = stats.pearsonr(scores, degs)
    print(f"  {'z_sum_O_depth+inv_erank':28s}  ρ = {rho:+.4f} (p={p_rho:.4f})  "
          f"r = {r:+.4f} (p={p_r:.4f})")
    return {"variant": "z_sum", "rho": float(rho), "p_rho": float(p_rho),
            "r": float(r), "p_r": float(p_r),
            "scores": scores.tolist(), "pairs": pairs, "max_deg": degs.tolist()}


# ---------------------------------------------------------------------------
# Plots and output
# ---------------------------------------------------------------------------

def make_plot(results, features, eval_degs, out_path):
    """Small-multiples: score vs max_degradation for each variant."""
    pairs = sorted(eval_degs.keys())
    degs = np.array([eval_degs[p] for p in pairs])

    # Show baseline + best few variants
    variants = ["mean_alignment", "O_depth", "inv_min_erank",
                "O_depth_x_inv_erank", "OVmix_x_inv_erank", "z_sum"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, v in zip(axes.flat, variants):
        row = next(r for r in results if r["variant"] == v)
        scores = np.array(row["scores"])
        # Color by whether it's "good" (<10% max deg) per the original B-P5 threshold
        colors = ["#27ae60" if d < 0.10 else "#c0392b" for d in degs]
        ax.scatter(scores, degs * 100, c=colors, s=80,
                   edgecolors="white", linewidth=0.7)
        ax.axhline(10, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel(v, fontsize=10)
        ax.set_ylabel("max degradation (%)", fontsize=10)
        ax.set_title(f"{v}\nρ = {row['rho']:+.3f}, p = {row['p_rho']:.3f}",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("N133 B-P5 follow-up: composite risk scores vs observed "
                 "merge degradation (n=12 cross-task pairs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("  N133 B-P5 follow-up: composite merge-risk score")
    print("=" * 70)

    profiles, pair_full, merges = load_inputs()
    print(f"\n  Loaded {len(profiles)} adapter profiles, "
          f"{len(pair_full)} pairs, {len(merges['results'])} merge evals")

    features = build_features(profiles, pair_full)
    eval_degs = evaluated_cross_task(merges)
    print(f"  Evaluating {len(eval_degs)} cross-task merges with known degradation\n")

    print("Per-adapter mean erank:")
    adapter_erank = {name: adapter_mean_erank(p) for name, p in profiles.items()}
    for n in sorted(adapter_erank):
        print(f"  {n:<30s} {adapter_erank[n]:.3f}")

    print("\nVariant comparison (Spearman & Pearson vs max_degradation, n=12):\n")
    variants = [
        "mean_alignment",      # baseline (original B-P5 ranker)
        "O_mean",
        "O_depth",
        "O_quad",
        "OVmix_depth",
        "inv_min_erank",
        "O_depth_x_inv_erank",
        "O_quad_x_inv_erank",
        "OVmix_x_inv_erank",
    ]
    results = [evaluate_variant(features, eval_degs, v) for v in variants]
    results.append(evaluate_z_sum(features, eval_degs))

    # Best (by |ρ|)
    best = max(results, key=lambda r: abs(r["rho"]))
    print(f"\nBest by |Spearman ρ|: {best['variant']} (ρ = {best['rho']:+.4f})")

    # Recall test: if we retain top 30% by risk (low risk = good merge), how many
    # of the 6 "good" merges end up in the retained set?
    print("\nTriage recall test (retain bottom 30% by risk → should be 'good'):\n")
    pairs = sorted(eval_degs.keys())
    for r in results:
        scores = np.array(r["scores"])
        degs = np.array(r["max_deg"])
        good_mask = degs < 0.10
        n_good = int(good_mask.sum())
        # Lowest-risk = retained
        order = np.argsort(scores)  # ascending
        k = 6  # triage top 6 as "safe to merge" among 12
        retained = set(pairs[i] for i in order[:k])
        retained_good = sum(1 for i in range(len(pairs)) if pairs[i] in retained and good_mask[i])
        print(f"  {r['variant']:28s}  retained_good={retained_good}/{n_good}  "
              f"(retained {k} lowest-risk, expected 'good')")

    # Save
    summary = {
        "experiment": "N133 B-P5 follow-up: composite merge-risk score",
        "n_evaluated_cross_task_pairs": len(eval_degs),
        "per_adapter_mean_erank": adapter_erank,
        "variants": [
            {
                "variant": r["variant"],
                "spearman_rho": r["rho"],
                "spearman_p": r["p_rho"],
                "pearson_r": r["r"],
                "pearson_p": r["p_r"],
            }
            for r in results
        ],
        "per_pair_features": {
            k: {kk: vv for kk, vv in v.items() if kk != "is_same_task"}
            for k, v in features.items()
            if k in eval_degs
        },
        "per_pair_max_degradation": eval_degs,
    }
    (OUT / "bp5_composite_risk.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved {OUT / 'bp5_composite_risk.json'}")

    make_plot(results, features, eval_degs, OUT / "bp5_composite_risk_plot.png")
    print(f"  Saved {OUT / 'bp5_composite_risk_plot.png'}")


if __name__ == "__main__":
    main()
