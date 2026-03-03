"""
Study 12 Analysis Pipeline
===========================

Run after study12 training is complete. Expects results in:
  results/study12_replication/<run_id>/telemetry.jsonl

Produces:
  1. early_features_study12.csv — 50-row feature matrix (7 geometric + loss)
  2. Classification results — LOSO, permutation, McNemar, MI
  3. Per-regime DFA exponents — the key Post 5 follow-up experiment
  4. Per-regime PR dynamics — information bottleneck replication
  5. Summary JSON and figures

Usage:
  cd /workspace/nanogpt_gradience   # (or wherever the study12 results live)
  python analyze_study12.py
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# ─── Configuration ───────────────────────────────────────────────────

RESULTS_ROOT = Path("results/study12_replication")
OUT_DIR = Path("analysis/study12")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EARLY_FRAC = 0.2  # First 20% of training steps for early features

REGIMES = ["baseline", "high_lr", "high_wd", "low_lr", "low_wd"]
SEEDS = list(range(1337, 1347))  # 1337–1346

# Feature column definitions (same as Study 11)
GEOMETRY_6 = [
    "early_weight_norm_mean", "early_weight_norm_slope",
    "early_weight_norm_curvature", "early_grad_norm_mean",
    "early_grad_to_weight_ratio_mean", "early_cos_grad_weight_mean",
]
GEOMETRY_7 = GEOMETRY_6 + ["early_spectral_complexity_mean"]
LOSS_ONLY = ["early_mean_train_loss"]
SPECTRAL_ONLY = ["early_spectral_complexity_mean"]


# ─── Step 1: Extract early features ─────────────────────────────────

def load_telemetry(path):
    """Load telemetry.jsonl → list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def polyfit_feature(iters, values, degree=2):
    """Fit polynomial to normalized iter range; return mean, slope, curvature."""
    iters = np.array(iters, dtype=float)
    values = np.array(values, dtype=float)
    mean = float(values.mean())

    if len(iters) < 3:
        slope = float(np.polyfit(iters, values, 1)[0]) if len(iters) >= 2 else 0.0
        return mean, slope, 0.0

    t = (iters - iters.min()) / max(iters.max() - iters.min(), 1.0)
    coeffs = np.polyfit(t, values, degree)
    curvature, slope, _ = coeffs
    return mean, float(slope), float(curvature)


def extract_early_features(run_dir, run_id, regime, seed):
    """Extract early-window features from a single run's telemetry."""
    telem_path = run_dir / "telemetry.jsonl"
    if not telem_path.exists():
        return None

    records = load_telemetry(telem_path)
    if not records:
        return None

    # Filter to early window
    max_iter = max(r.get("iter", r.get("step", 0)) for r in records)
    early_cutoff = max_iter * EARLY_FRAC

    early = [r for r in records
             if r.get("iter", r.get("step", 0)) <= early_cutoff
             and "train_loss" in r]

    if len(early) < 3:
        print(f"  WARNING: {run_id} has only {len(early)} early records")
        return None

    iters = [r.get("iter", r.get("step", 0)) for r in early]

    # Loss
    losses = [r["train_loss"] for r in early]
    early_mean_train_loss = np.mean(losses)

    # Weight norm
    wnorms = [r.get("weight_norm", 0) for r in early if "weight_norm" in r]
    if wnorms:
        wn_mean, wn_slope, wn_curv = polyfit_feature(
            iters[:len(wnorms)], wnorms)
    else:
        wn_mean, wn_slope, wn_curv = 0, 0, 0

    # Grad norm
    gnorms = [r.get("grad_norm", 0) for r in early if "grad_norm" in r]
    gn_mean = np.mean(gnorms) if gnorms else 0

    # Grad-to-weight ratio
    gw_ratios = [r.get("grad_to_weight_ratio", 0) for r in early
                 if "grad_to_weight_ratio" in r]
    gw_mean = np.mean(gw_ratios) if gw_ratios else 0

    # Cosine(grad, weight)
    cos_gw = [r.get("cos_grad_weight", 0) for r in early
              if "cos_grad_weight" in r]
    cos_mean = np.mean(cos_gw) if cos_gw else 0

    # Spectral complexity
    spec = [r.get("spectral_complexity", None) for r in early
            if r.get("spectral_complexity") is not None]
    spec_mean = np.mean(spec) if spec else ""

    return {
        "run_id": run_id,
        "regime": regime,
        "seed": seed,
        "early_steps_max": int(early_cutoff),
        "early_mean_train_loss": early_mean_train_loss,
        "early_weight_norm_mean": wn_mean,
        "early_weight_norm_slope": wn_slope,
        "early_weight_norm_curvature": wn_curv,
        "early_grad_norm_mean": gn_mean,
        "early_grad_to_weight_ratio_mean": gw_mean,
        "early_cos_grad_weight_mean": cos_mean,
        "early_spectral_complexity_mean": spec_mean,
    }


# ─── Step 2: Classification (reuses logic from add_spectral_and_reclassify.py)

def zscore(X_train):
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mu, std


def loso_accuracy(X, y, seeds_arr):
    unique_s = sorted(set(seeds_arr))
    correct = 0
    total = 0
    predictions = []
    for held_out in unique_s:
        mask_test = np.array([s == held_out for s in seeds_arr])
        mask_train = ~mask_test
        X_tr, y_tr = X[mask_train], y[mask_train]
        X_te, y_te = X[mask_test], y[mask_test]
        mu, std = zscore(X_tr)
        X_tr_z = (X_tr - mu) / std
        X_te_z = (X_te - mu) / std
        centroids = {}
        for c in sorted(set(y_tr)):
            centroids[c] = X_tr_z[y_tr == c].mean(axis=0)
        for i in range(len(X_te)):
            dists = {c: np.linalg.norm(X_te_z[i] - centroids[c]) for c in centroids}
            pred = min(dists, key=dists.get)
            predictions.append((int(y_te[i]), int(pred)))
            correct += (pred == y_te[i])
            total += 1
    return correct / total, predictions


def permutation_test(X, y, seeds_arr, n_perms=9999):
    observed, _ = loso_accuracy(X, y, seeds_arr)
    count_ge = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        y_perm = rng.permutation(y)
        perm_acc, _ = loso_accuracy(X, y_perm, seeds_arr)
        if perm_acc >= observed:
            count_ge += 1
    return observed, (count_ge + 1) / (n_perms + 1)


def mcnemar(preds_a, preds_b):
    from scipy.stats import binom
    b_right_a_wrong = sum(1 for (ta, pa), (tb, pb) in zip(preds_a, preds_b)
                          if pb == tb and pa != ta)
    a_right_b_wrong = sum(1 for (ta, pa), (tb, pb) in zip(preds_a, preds_b)
                          if pa == ta and pb != tb)
    n = b_right_a_wrong + a_right_b_wrong
    if n == 0:
        return 1.0
    k = min(b_right_a_wrong, a_right_b_wrong)
    return min(2 * binom.cdf(k, n, 0.5), 1.0)


# ─── Step 3: Per-regime DFA ─────────────────────────────────────────

def dfa_exponent(series, min_scale=4, max_scale_frac=0.25):
    """Detrended Fluctuation Analysis → scaling exponent."""
    N = len(series)
    if N < 20:
        return np.nan, 0

    # Cumulative deviation from mean
    y = np.cumsum(series - np.mean(series))

    max_scale = int(N * max_scale_frac)
    scales = np.unique(np.logspace(
        np.log10(min_scale), np.log10(max_scale), 20).astype(int))
    scales = scales[scales >= min_scale]

    fluctuations = []
    valid_scales = []
    for s in scales:
        n_segments = N // s
        if n_segments < 2:
            continue
        rms_list = []
        for i in range(n_segments):
            segment = y[i * s:(i + 1) * s]
            t = np.arange(s)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            rms_list.append(np.sqrt(np.mean((segment - trend) ** 2)))
        fluctuations.append(np.mean(rms_list))
        valid_scales.append(s)

    if len(valid_scales) < 4:
        return np.nan, 0

    log_s = np.log10(valid_scales)
    log_f = np.log10(fluctuations)
    slope, intercept = np.polyfit(log_s, log_f, 1)
    ss_res = np.sum((log_f - (slope * log_s + intercept)) ** 2)
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return slope, r2


def compute_run_dfa(telemetry_path):
    """Compute DFA exponents for key metrics in a single run."""
    records = load_telemetry(telemetry_path)
    if len(records) < 30:
        return None

    # Extract time series
    metrics = {}
    for key in ["train_loss", "weight_norm", "grad_norm", "spectral_complexity"]:
        vals = [r[key] for r in records if key in r and r[key] is not None]
        if len(vals) >= 30:
            metrics[key] = np.array(vals)

    results = {}
    for key, series in metrics.items():
        alpha, r2 = dfa_exponent(series)
        results[key] = {"alpha": alpha, "r2": r2}

    return results


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Study 12: Replication Analysis")
    print("=" * 70)

    # Step 1: Extract features
    print("\n--- Step 1: Extracting early features ---")
    rows = []
    missing = []

    for regime in REGIMES:
        for seed in SEEDS:
            # Reconstruct run_id from regime table
            lr_tags = {"baseline": "1em03", "low_wd": "1em03", "high_wd": "1em03",
                       "low_lr": "1em04", "high_lr": "1em02"}
            wd_tags = {"baseline": "0p1", "low_wd": "0", "high_wd": "1",
                       "low_lr": "0p1", "high_lr": "0p1"}
            run_id = f"{regime}_lr{lr_tags[regime]}_wd{wd_tags[regime]}_seed{seed}"
            run_dir = RESULTS_ROOT / run_id

            if not run_dir.exists():
                missing.append(run_id)
                continue

            feats = extract_early_features(run_dir, run_id, regime, seed)
            if feats:
                rows.append(feats)
                print(f"  {run_id}: OK")
            else:
                missing.append(run_id)
                print(f"  {run_id}: MISSING or too few records")

    if missing:
        print(f"\n  WARNING: {len(missing)} runs missing: {missing[:5]}...")

    if len(rows) < 10:
        print(f"\n  FATAL: Only {len(rows)} runs found. Need at least 10.")
        sys.exit(1)

    # Save features CSV
    csv_path = OUT_DIR / "early_features_study12.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {csv_path} ({len(rows)} runs)")

    # Step 2: Classification
    print("\n--- Step 2: Regime classification ---")
    unique_regimes = sorted(set(r["regime"] for r in rows))
    regime_to_idx = {r: i for i, r in enumerate(unique_regimes)}
    y = np.array([regime_to_idx[r["regime"]] for r in rows])
    seeds_arr = [int(r["seed"]) for r in rows]

    def get_X(feature_names):
        X = np.zeros((len(rows), len(feature_names)))
        for i, row in enumerate(rows):
            for j, f in enumerate(feature_names):
                val = row.get(f, "")
                X[i, j] = float(val) if val != "" else 0.0
        return X

    feature_sets = {
        "loss_only": LOSS_ONLY,
        "geometry_6": GEOMETRY_6,
        "spectral_only": SPECTRAL_ONLY,
        "geometry_7": GEOMETRY_7,
    }

    results = {"n_runs": len(rows), "n_regimes": len(unique_regimes),
               "n_seeds": len(set(seeds_arr)), "regimes": unique_regimes}

    all_preds = {}
    for name, feats in feature_sets.items():
        X = get_X(feats)
        acc, preds = loso_accuracy(X, y, seeds_arr)
        _, p_val = permutation_test(X, y, seeds_arr, n_perms=4999)
        all_preds[name] = preds

        print(f"  {name:15s}: {acc:.1%} ({int(acc*len(y))}/{len(y)})  p={p_val:.4f}")
        results[name] = {
            "accuracy": acc,
            "n_correct": int(acc * len(y)),
            "permutation_p": p_val,
        }

    # McNemar comparisons
    print("\n  Pairwise McNemar:")
    mcnemar_results = {}
    for a, b in [("geometry_7", "loss_only"), ("spectral_only", "loss_only"),
                 ("geometry_6", "geometry_7"), ("spectral_only", "geometry_6")]:
        p = mcnemar(all_preds[a], all_preds[b])
        print(f"    {a} vs {b}: p={p:.4f}")
        mcnemar_results[f"{a}_vs_{b}"] = p
    results["mcnemar"] = mcnemar_results

    # Step 3: Per-regime DFA
    print("\n--- Step 3: Per-regime DFA exponents ---")
    dfa_by_regime = defaultdict(list)

    for row in rows:
        run_id = row["run_id"]
        regime = row["regime"]
        telem_path = RESULTS_ROOT / run_id / "telemetry.jsonl"
        if telem_path.exists():
            run_dfa = compute_run_dfa(telem_path)
            if run_dfa:
                dfa_by_regime[regime].append(run_dfa)

    dfa_summary = {}
    for regime in unique_regimes:
        runs = dfa_by_regime[regime]
        if not runs:
            continue
        summary = {}
        for metric in ["train_loss", "weight_norm", "grad_norm", "spectral_complexity"]:
            alphas = [r[metric]["alpha"] for r in runs
                      if metric in r and not np.isnan(r[metric]["alpha"])]
            if alphas:
                summary[metric] = {
                    "mean_alpha": float(np.mean(alphas)),
                    "std_alpha": float(np.std(alphas)),
                    "n": len(alphas),
                }
                print(f"  {regime:12s} {metric:25s}: α = {np.mean(alphas):.3f} ± {np.std(alphas):.3f} (n={len(alphas)})")
        dfa_summary[regime] = summary

    results["dfa_by_regime"] = dfa_summary

    # Step 4: Cross-regime DFA comparison (the key test)
    print("\n--- Step 4: Cross-regime DFA comparison ---")
    if "spectral_complexity" in dfa_summary.get("baseline", {}):
        regime_alphas = {}
        for regime in unique_regimes:
            if regime in dfa_summary and "spectral_complexity" in dfa_summary[regime]:
                regime_alphas[regime] = dfa_summary[regime]["spectral_complexity"]["mean_alpha"]

        if len(regime_alphas) >= 2:
            print("  Spectral complexity DFA exponents by regime:")
            for regime, alpha in sorted(regime_alphas.items()):
                print(f"    {regime:12s}: α = {alpha:.3f}")

            # ANOVA-like test: are the regime means different?
            from scipy import stats
            groups = []
            for regime in unique_regimes:
                runs = dfa_by_regime[regime]
                alphas = [r["spectral_complexity"]["alpha"] for r in runs
                          if "spectral_complexity" in r and not np.isnan(r["spectral_complexity"]["alpha"])]
                if alphas:
                    groups.append(alphas)

            if len(groups) >= 2:
                F, p = stats.f_oneway(*groups)
                print(f"\n  One-way ANOVA: F={F:.3f}, p={p:.4f}")
                results["dfa_anova"] = {"F": float(F), "p": float(p)}

                if p < 0.05:
                    print("  → DFA exponents DIFFER across regimes (significant)")
                    print("    This is evidence that long-range correlations")
                    print("    are a property of the regime, not just SGD.")
                else:
                    print("  → DFA exponents do NOT significantly differ across regimes")
                    print("    Long-range correlations may be a generic SGD property.")

    # Save
    output_path = OUT_DIR / "study12_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
