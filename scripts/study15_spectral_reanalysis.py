#!/usr/bin/env python3
"""
Study 15: Information-Theoretic Reanalysis with Spectral Complexity
====================================================================

Re-runs the Post 4 information-theoretic comparison (geometry vs. loss)
on the Study 12 dataset (n=49, 5 regimes × 10 seeds), now including
the `early_spectral_complexity_mean` feature that was missing from the
original analysis.

The original comparison (Post 4, n=15) found:
  - Geometry MI: 7.4× loss MI
  - Conditional entropy: geometry reduces uncertainty by 15%, loss by 10%
  - Classification: geometry 66.7%, loss 40% (McNemar p=0.289)

This reanalysis asks:
  1. Does the MI advantage hold at n=49?
  2. Does adding spectral complexity change the picture?
  3. Is spectral complexity alone the best single feature?

Outputs: JSON results + markdown summary.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif

EPS = 1e-12
N_PERM = 2000  # More permutations for tighter p-values at n=49

# ── Feature sets ──
GEOM_6 = [
    "early_weight_norm_mean",
    "early_weight_norm_slope",
    "early_weight_norm_curvature",
    "early_grad_norm_mean",
    "early_grad_to_weight_ratio_mean",
    "early_cos_grad_weight_mean",
]
GEOM_7 = GEOM_6 + ["early_spectral_complexity_mean"]
LOSS_ONLY = ["early_mean_train_loss"]
SPECTRAL_ONLY = ["early_spectral_complexity_mean"]


# ── Helpers ──


def zscore_fit(X):
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd < EPS] = 1.0
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_ridge(Xtr, ytr_int, n_classes, alpha=0.01):
    n, d = Xtr.shape
    Xb = np.column_stack([Xtr, np.ones(n)])
    D = d + 1
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), ytr_int] = 1.0
    R = np.eye(D)
    R[-1, -1] = 0.0
    W = np.linalg.solve(Xb.T @ Xb + alpha * R, Xb.T @ Y)
    return W


def loso_classify(X, y_int, seeds, n_classes, alpha=0.01):
    """
    Leave-One-Seed-Out classification.
    Returns (accuracy, predicted_labels, predicted_probs, true_labels).
    """
    unique_seeds = sorted(set(seeds))
    all_preds = []
    all_probs = []
    all_true = []

    for hold in unique_seeds:
        mask_tr = seeds != hold
        mask_te = ~mask_tr
        n_te = int(mask_te.sum())

        Xtr, Xte = X[mask_tr], X[mask_te]
        ytr = y_int[mask_tr]

        mu, sd = zscore_fit(Xtr)
        W = fit_ridge(zscore_apply(Xtr, mu, sd), ytr, n_classes, alpha)
        Xb = np.column_stack([zscore_apply(Xte, mu, sd), np.ones(n_te)])
        scores = Xb @ W
        probs = softmax(scores)
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds.tolist())
        all_probs.append(probs)
        all_true.extend(y_int[mask_te].tolist())

    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    all_true = np.array(all_true)
    accuracy = float(np.mean(all_preds == all_true))

    return accuracy, all_preds, all_probs, all_true


def permutation_accuracy(X, y_int, seeds, n_classes, n_perm=N_PERM, rng_seed=42):
    """Permutation test for classification accuracy."""
    rng = np.random.RandomState(rng_seed)
    null_accs = []
    for _ in range(n_perm):
        y_shuf = rng.permutation(y_int)
        acc, _, _, _ = loso_classify(X, y_shuf, seeds, n_classes)
        null_accs.append(acc)
    return np.array(null_accs)


def compute_mi(X, y_int, n_neighbors=3, n_perm=N_PERM, rng_seed=42):
    """
    Compute mutual information (per-feature and joint) with bias correction.
    Returns dict with raw, corrected MI and p-values.
    """
    rng = np.random.RandomState(rng_seed)

    # Joint MI (sum across features from sklearn)
    mi_joint_raw = float(mutual_info_classif(X, y_int, n_neighbors=n_neighbors, random_state=42).sum())

    # Permutation null
    null_mi = []
    for _ in range(n_perm):
        y_shuf = rng.permutation(y_int)
        mi_null = mutual_info_classif(X, y_shuf, n_neighbors=n_neighbors, random_state=42).sum()
        null_mi.append(mi_null)
    null_mi = np.array(null_mi)

    null_mean = float(null_mi.mean())
    mi_corrected = mi_joint_raw - null_mean
    p_value = float(np.mean(null_mi >= mi_joint_raw))

    # Per-feature MI
    per_feature = {}
    for i in range(X.shape[1]):
        x_col = X[:, [i]]
        mi_raw_i = float(mutual_info_classif(x_col, y_int, n_neighbors=n_neighbors, random_state=42)[0])

        null_i = []
        for _ in range(n_perm):
            y_shuf = rng.permutation(y_int)
            mi_n = mutual_info_classif(x_col, y_shuf, n_neighbors=n_neighbors, random_state=42)[0]
            null_i.append(mi_n)
        null_i = np.array(null_i)

        per_feature[i] = {
            "mi_raw": mi_raw_i,
            "mi_corrected": mi_raw_i - float(null_i.mean()),
            "p_value": float(np.mean(null_i >= mi_raw_i)),
        }

    return {
        "joint_mi_raw": mi_joint_raw,
        "joint_mi_corrected": mi_corrected,
        "null_mean": null_mean,
        "null_std": float(null_mi.std()),
        "p_value": p_value,
        "per_feature": per_feature,
    }


def compute_conditional_entropy(X, y_int, seeds, n_classes):
    """Compute H(regime|features) using LOSO predicted probabilities."""
    _, _, probs, true_labels = loso_classify(X, y_int, seeds, n_classes)
    n = len(true_labels)

    # H(Y) — prior entropy
    class_counts = np.bincount(true_labels, minlength=n_classes)
    p_regime = class_counts / n
    H_prior = float(entropy(p_regime, base=np.e))

    # H(Y|X) — conditional entropy from predicted probabilities
    H_cond = -np.mean(np.log(probs[np.arange(n), true_labels] + EPS))
    info_gain = H_prior - H_cond
    info_gain_frac = info_gain / H_prior if H_prior > 0 else 0.0

    return {
        "H_prior": H_prior,
        "H_conditional": float(H_cond),
        "info_gain": float(info_gain),
        "info_gain_frac": float(info_gain_frac),
    }


def compute_mdl(X, y_int, seeds, n_classes, name=""):
    """BIC / AIC model comparison."""
    n = len(y_int)
    n_features = X.shape[1]
    _, _, probs, true_labels = loso_classify(X, y_int, seeds, n_classes)

    nll = -np.sum(np.log(probs[np.arange(n), true_labels] + EPS))
    k = n_features * n_classes + n_classes
    bic = nll + (k / 2) * np.log(n)
    aic = 2 * k + 2 * nll

    return {
        "name": name,
        "n_features": n_features,
        "n_params": k,
        "nll": float(nll),
        "bic": float(bic),
        "aic": float(aic),
    }


def mcnemar_test(preds_a, preds_b, true_labels):
    """McNemar's test comparing two classifiers."""
    a_correct = preds_a == true_labels
    b_correct = preds_b == true_labels
    # Discordant pairs
    b_only = int(np.sum(~a_correct & b_correct))  # b right, a wrong
    a_only = int(np.sum(a_correct & ~b_correct))  # a right, b wrong
    n_disc = a_only + b_only

    if n_disc == 0:
        return {"chi2": 0.0, "p_value": 1.0, "a_only": a_only, "b_only": b_only}

    # McNemar with continuity correction
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n_disc
    from scipy.stats import chi2 as chi2_dist

    p_value = float(1.0 - chi2_dist.cdf(chi2, df=1))

    return {"chi2": float(chi2), "p_value": p_value, "a_only": a_only, "b_only": b_only}


# ===========================================================================
# Main analysis
# ===========================================================================


def main():
    # ── Load data ──
    csv_path = (
        Path(__file__).resolve().parent.parent / "Gradience II" / "analysis" / "study12" / "early_features_study12.csv"
    )

    # Try alternate paths
    if not csv_path.exists():
        alt = Path("/sessions/trusting-magical-allen/mnt/Gradience II/analysis/study12/early_features_study12.csv")
        if alt.exists():
            csv_path = alt
        else:
            print(f"ERROR: Cannot find early features CSV at {csv_path}")
            sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} runs from {csv_path}")

    # Verify spectral complexity is present and non-null
    assert "early_spectral_complexity_mean" in df.columns, "Missing spectral complexity column"
    n_valid = df["early_spectral_complexity_mean"].notna().sum()
    print(f"Runs with spectral complexity: {n_valid}/{len(df)}")

    labels = sorted(df["regime"].unique())
    lab2i = {lab: i for i, lab in enumerate(labels)}
    y_int = np.array([lab2i[r] for r in df["regime"]])
    seeds = df["seed"].values
    n_classes = len(labels)
    n_samples = len(df)

    print(f"Classes: {labels}")
    print(f"Distribution: {dict(Counter(df['regime']))}")

    # Build feature matrices
    X_geom6 = df[GEOM_6].values.astype(float)
    X_geom7 = df[GEOM_7].values.astype(float)
    X_loss = df[LOSS_ONLY].values.astype(float)
    X_spec = df[SPECTRAL_ONLY].values.astype(float)

    feature_sets = {
        "geometry_6": (X_geom6, GEOM_6),
        "geometry_7": (X_geom7, GEOM_7),
        "loss_only": (X_loss, LOSS_ONLY),
        "spectral_only": (X_spec, SPECTRAL_ONLY),
    }

    results = {
        "study_metadata": {
            "study_id": "study15_spectral_reanalysis",
            "title": "Information-Theoretic Reanalysis with Spectral Complexity",
            "n_samples": n_samples,
            "n_classes": n_classes,
            "n_permutations": N_PERM,
            "source_data": "Study 12 (5 regimes × 10 seeds)",
            "class_distribution": dict(Counter(df["regime"])),
        },
    }

    # ── Section 1: Classification accuracy (LOSO) ──
    print("\n" + "=" * 72)
    print("SECTION 1: LOSO CLASSIFICATION ACCURACY")
    print("=" * 72)

    class_results = {}
    all_preds = {}

    for name, (X, feats) in feature_sets.items():
        acc, preds, probs, true = loso_classify(X, y_int, seeds, n_classes)

        # Permutation test
        print(f"\n  {name}: computing permutation null ({N_PERM} permutations)...")
        null_accs = permutation_accuracy(X, y_int, seeds, n_classes)
        p_val = float(np.mean(null_accs >= acc))

        class_results[name] = {
            "accuracy": acc,
            "n_features": len(feats),
            "features": feats,
            "p_value": p_val,
            "null_mean": float(null_accs.mean()),
            "null_std": float(null_accs.std()),
        }
        all_preds[name] = (preds, true)

        print(f"  {name:>15s}: {acc:.1%} accuracy (p={p_val:.4f}, null={null_accs.mean():.1%}±{null_accs.std():.1%})")

    # McNemar tests: geometry_7 vs loss, spectral_only vs loss, geometry_7 vs geometry_6
    print("\n  McNemar tests:")
    mcnemar_pairs = [
        ("geometry_6", "loss_only"),
        ("geometry_7", "loss_only"),
        ("spectral_only", "loss_only"),
        ("geometry_7", "geometry_6"),
    ]
    mcnemar_results = {}
    for a_name, b_name in mcnemar_pairs:
        preds_a, true_a = all_preds[a_name]
        preds_b, _ = all_preds[b_name]
        mc = mcnemar_test(preds_a, preds_b, true_a)
        label = f"{a_name}_vs_{b_name}"
        mcnemar_results[label] = mc
        print(f"    {label}: χ²={mc['chi2']:.3f}, p={mc['p_value']:.4f} (a_only={mc['a_only']}, b_only={mc['b_only']})")

    results["classification"] = {
        "accuracies": class_results,
        "mcnemar_tests": mcnemar_results,
    }

    # ── Section 2: Mutual Information ──
    print("\n" + "=" * 72)
    print("SECTION 2: MUTUAL INFORMATION")
    print("=" * 72)

    mi_results = {}
    for name, (X, feats) in feature_sets.items():
        print(f"\n  {name}: computing MI ({N_PERM} permutations)...")
        mi = compute_mi(X, y_int, n_perm=N_PERM)
        mi["features"] = feats
        mi_results[name] = mi
        print(f"    Joint MI (corrected): {mi['joint_mi_corrected']:.4f} (p={mi['p_value']:.4f})")

    # MI ratios
    loss_mi = mi_results["loss_only"]["joint_mi_corrected"]
    print(f"\n  MI Ratios (vs. loss MI = {loss_mi:.4f}):")
    for name in ["geometry_6", "geometry_7", "spectral_only"]:
        geom_mi = mi_results[name]["joint_mi_corrected"]
        ratio = geom_mi / loss_mi if loss_mi > 0 else float("inf")
        mi_results[name]["mi_ratio_vs_loss"] = float(ratio)
        print(f"    {name:>15s}: {geom_mi:.4f} = {ratio:.2f}× loss MI")

    results["mutual_information"] = mi_results

    # ── Section 3: Conditional Entropy ──
    print("\n" + "=" * 72)
    print("SECTION 3: CONDITIONAL ENTROPY")
    print("=" * 72)

    ce_results = {}
    for name, (X, feats) in feature_sets.items():
        ce = compute_conditional_entropy(X, y_int, seeds, n_classes)
        ce_results[name] = ce
        print(
            f"  {name:>15s}: H(Y|X)={ce['H_conditional']:.4f}, "
            f"info_gain={ce['info_gain']:.4f} ({ce['info_gain_frac']:.1%} of prior)"
        )

    # Ratios
    loss_gain = ce_results["loss_only"]["info_gain"]
    print(f"\n  Info gain ratios (vs. loss = {loss_gain:.4f}):")
    for name in ["geometry_6", "geometry_7", "spectral_only"]:
        g = ce_results[name]["info_gain"]
        ratio = g / loss_gain if loss_gain > 0 else float("inf")
        ce_results[name]["info_gain_ratio_vs_loss"] = float(ratio)
        print(f"    {name:>15s}: {ratio:.2f}× loss info gain")

    results["conditional_entropy"] = ce_results

    # ── Section 4: MDL / BIC / AIC ──
    print("\n" + "=" * 72)
    print("SECTION 4: MODEL COMPLEXITY (BIC / AIC)")
    print("=" * 72)

    mdl_results = {}
    for name, (X, feats) in feature_sets.items():
        mdl = compute_mdl(X, y_int, seeds, n_classes, name)
        mdl_results[name] = mdl
        print(f"  {name:>15s}: BIC={mdl['bic']:.3f}, AIC={mdl['aic']:.3f} (k={mdl['n_params']}, NLL={mdl['nll']:.3f})")

    best_bic = min(mdl_results.items(), key=lambda x: x[1]["bic"])
    best_aic = min(mdl_results.items(), key=lambda x: x[1]["aic"])
    print(f"\n  Best BIC: {best_bic[0]}")
    print(f"  Best AIC: {best_aic[0]}")

    results["mdl"] = {
        "models": mdl_results,
        "best_bic": best_bic[0],
        "best_aic": best_aic[0],
    }

    # ── Summary ──
    print("\n" + "=" * 72)
    print("STUDY 15: SUMMARY")
    print("=" * 72)

    # Key comparisons
    acc_g6 = class_results["geometry_6"]["accuracy"]
    acc_g7 = class_results["geometry_7"]["accuracy"]
    acc_loss = class_results["loss_only"]["accuracy"]
    acc_spec = class_results["spectral_only"]["accuracy"]

    mi_g6 = mi_results["geometry_6"]["joint_mi_corrected"]
    mi_g7 = mi_results["geometry_7"]["joint_mi_corrected"]
    mi_loss_val = mi_results["loss_only"]["joint_mi_corrected"]
    mi_spec = mi_results["spectral_only"]["joint_mi_corrected"]

    print("\n  Classification accuracy:")
    print(f"    Loss only:          {acc_loss:.1%}")
    print(f"    Geometry (6 feat):  {acc_g6:.1%}")
    print(f"    Geometry (7 feat):  {acc_g7:.1%}")
    print(f"    Spectral only:      {acc_spec:.1%}")

    print("\n  Mutual information (corrected):")
    print(f"    Loss only:          {mi_loss_val:.4f}")
    print(f"    Geometry (6 feat):  {mi_g6:.4f} ({mi_g6 / mi_loss_val:.1f}× loss)" if mi_loss_val > 0 else "")
    print(f"    Geometry (7 feat):  {mi_g7:.4f} ({mi_g7 / mi_loss_val:.1f}× loss)" if mi_loss_val > 0 else "")
    print(f"    Spectral only:      {mi_spec:.4f} ({mi_spec / mi_loss_val:.1f}× loss)" if mi_loss_val > 0 else "")

    spectral_helps_acc = acc_g7 > acc_g6
    spectral_helps_mi = mi_g7 > mi_g6
    spectral_alone_beats_loss = acc_spec > acc_loss

    print("\n  Key questions:")
    print(
        f"    Does spectral improve classification?   {'YES' if spectral_helps_acc else 'NO'} "
        f"({acc_g7:.1%} vs {acc_g6:.1%})"
    )
    print(
        f"    Does spectral improve MI?               {'YES' if spectral_helps_mi else 'NO'} "
        f"({mi_g7:.4f} vs {mi_g6:.4f})"
    )
    print(
        f"    Spectral alone vs loss?                 {'SPECTRAL WINS' if spectral_alone_beats_loss else 'LOSS WINS'} "
        f"({acc_spec:.1%} vs {acc_loss:.1%})"
    )

    results["summary"] = {
        "spectral_improves_classification": spectral_helps_acc,
        "spectral_improves_mi": spectral_helps_mi,
        "spectral_alone_beats_loss": spectral_alone_beats_loss,
        "accuracy_delta_g7_vs_g6": float(acc_g7 - acc_g6),
        "mi_delta_g7_vs_g6": float(mi_g7 - mi_g6),
    }

    # ── Save ──
    out_dir = Path(__file__).resolve().parent.parent / "results" / "study15_spectral_reanalysis"
    # Try alternate path
    if not out_dir.parent.exists():
        out_dir = Path("/sessions/trusting-magical-allen/mnt/gradience/results/study15_spectral_reanalysis")

    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "study15_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")

    # Generate markdown summary
    md_path = out_dir / "study15_summary.md"
    md_lines = [
        "# Study 15: Information-Theoretic Reanalysis with Spectral Complexity",
        "",
        "## Overview",
        "",
        f"Re-ran the Post 4 information-theoretic comparison on Study 12 data "
        f"(n={n_samples}, {n_classes} regimes × 10 seeds), now including "
        f"`early_spectral_complexity_mean` as a 7th geometric feature.",
        "",
        "## Classification Accuracy (LOSO)",
        "",
        "| Feature Set | N Features | Accuracy | p-value |",
        "|------------|-----------|----------|---------|",
    ]
    for name in ["loss_only", "geometry_6", "geometry_7", "spectral_only"]:
        cr = class_results[name]
        md_lines.append(f"| {name} | {cr['n_features']} | {cr['accuracy']:.1%} | {cr['p_value']:.4f} |")
    md_lines += [
        "",
        "## McNemar Tests",
        "",
        "| Comparison | χ² | p-value | Significant? |",
        "|-----------|-----|---------|-------------|",
    ]
    for pair, mc in mcnemar_results.items():
        sig = "Yes" if mc["p_value"] < 0.05 else "No"
        md_lines.append(f"| {pair} | {mc['chi2']:.3f} | {mc['p_value']:.4f} | {sig} |")

    md_lines += [
        "",
        "## Mutual Information (bias-corrected)",
        "",
        "| Feature Set | MI (corrected) | MI Ratio vs Loss | p-value |",
        "|------------|---------------|-----------------|---------|",
    ]
    for name in ["loss_only", "geometry_6", "geometry_7", "spectral_only"]:
        mi = mi_results[name]
        ratio = mi.get("mi_ratio_vs_loss", 1.0) if name != "loss_only" else 1.0
        md_lines.append(f"| {name} | {mi['joint_mi_corrected']:.4f} | {ratio:.2f}× | {mi['p_value']:.4f} |")

    md_lines += [
        "",
        "## Conditional Entropy",
        "",
        "| Feature Set | H(Y|X) | Info Gain | % Prior Reduced |",
        "|------------|--------|-----------|----------------|",
    ]
    for name in ["loss_only", "geometry_6", "geometry_7", "spectral_only"]:
        ce = ce_results[name]
        md_lines.append(f"| {name} | {ce['H_conditional']:.4f} | {ce['info_gain']:.4f} | {ce['info_gain_frac']:.1%} |")

    md_lines += [
        "",
        "## Model Complexity (BIC / AIC)",
        "",
        "| Feature Set | N Params | NLL | BIC | AIC |",
        "|------------|---------|-----|-----|-----|",
    ]
    for name in ["loss_only", "geometry_6", "geometry_7", "spectral_only"]:
        m = mdl_results[name]
        md_lines.append(f"| {name} | {m['n_params']} | {m['nll']:.3f} | {m['bic']:.3f} | {m['aic']:.3f} |")
    md_lines.append(f"\nBest BIC: **{best_bic[0]}**")
    md_lines.append(f"Best AIC: **{best_aic[0]}**")

    md_lines += [
        "",
        "## Key Findings",
        "",
        f"1. Adding spectral complexity {'improved' if spectral_helps_acc else 'did not improve'} "
        f"classification accuracy ({acc_g7:.1%} with spectral vs {acc_g6:.1%} without).",
        f"2. Adding spectral complexity {'increased' if spectral_helps_mi else 'did not increase'} "
        f"mutual information ({mi_g7:.4f} vs {mi_g6:.4f}).",
        f"3. Spectral complexity alone {'outperformed' if spectral_alone_beats_loss else 'did not outperform'} "
        f"loss ({acc_spec:.1%} vs {acc_loss:.1%}).",
        "",
        "## Comparison with Post 4 (n=15)",
        "",
        "| Metric | Post 4 (n=15) | Study 15 (n=49) |",
        "|--------|---------------|-----------------|",
        f"| Geometry accuracy | 66.7% | {acc_g7:.1%} |",
        f"| Loss accuracy | 40.0% | {acc_loss:.1%} |",
        f"| MI ratio (geom/loss) | 7.4× | {mi_g7 / mi_loss_val:.1f}× |"
        if mi_loss_val > 0
        else "| MI ratio | 7.4× | N/A |",
        f"| McNemar (geom vs loss) | p=0.289 | p={mcnemar_results.get('geometry_7_vs_loss_only', {}).get('p_value', 'N/A')} |",
        "",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Summary saved to {md_path}")


if __name__ == "__main__":
    main()
