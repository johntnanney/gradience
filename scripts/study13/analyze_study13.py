#!/usr/bin/env python3
"""
Study 13: Windowed DFA analysis of grokking telemetry.

Loads telemetry from all seeds, detects grokking transitions,
computes windowed DFA exponents, and performs statistical tests.

Usage:
    python analyze_study13.py [--results_root results/study13]
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# DFA Implementation (matching Study 12 / analyze_grokking_dfa.py)
# ═══════════════════════════════════════════════════════════════════════════


def dfa(signal, min_scale=4, max_scale_frac=0.25, n_scales=20):
    """
    Detrended Fluctuation Analysis.
    Returns: alpha, r_squared, scales, fluctuations
    """
    x = np.array(signal, dtype=float)
    x = x - np.mean(x)
    N = len(x)

    y = np.cumsum(x)

    max_scale = int(N * max_scale_frac)
    if max_scale < min_scale + 2:
        return np.nan, np.nan, [], []

    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), num=n_scales).astype(int))
    scales = scales[scales >= min_scale]

    fluctuations = []
    valid_scales = []

    for s in scales:
        n_segments = N // s
        if n_segments < 2:
            continue

        rms_values = []
        # Forward segments
        for v in range(n_segments):
            segment = y[v * s : (v + 1) * s]
            t = np.arange(s)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            rms_values.append(np.sqrt(np.mean((segment - trend) ** 2)))

        # Backward segments
        for v in range(n_segments):
            segment = y[N - (v + 1) * s : N - v * s]
            t = np.arange(s)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            rms_values.append(np.sqrt(np.mean((segment - trend) ** 2)))

        F_s = np.sqrt(np.mean(np.array(rms_values) ** 2))
        if F_s > 0:
            fluctuations.append(F_s)
            valid_scales.append(s)

    if len(valid_scales) < 3:
        return np.nan, np.nan, [], []

    log_s = np.log10(valid_scales)
    log_F = np.log10(fluctuations)

    coeffs = np.polyfit(log_s, log_F, 1)
    alpha = coeffs[0]

    predicted = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_F - predicted) ** 2)
    ss_tot = np.sum((log_F - np.mean(log_F)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return alpha, r_squared, valid_scales, fluctuations


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════


def load_seed_data(seed_dir):
    """Load telemetry and summary for one seed."""
    telemetry_path = seed_dir / "telemetry.jsonl"
    summary_path = seed_dir / "summary.json"

    if not telemetry_path.exists():
        return None

    records = []
    with open(telemetry_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    summary = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return {"records": records, "summary": summary}


def extract_hessian_records(records):
    """Filter to records that have Hessian data."""
    return [r for r in records if r.get("hessian_computed", False)]


# ═══════════════════════════════════════════════════════════════════════════
# Grokking Detection
# ═══════════════════════════════════════════════════════════════════════════


def detect_grok_step(records, threshold=0.95):
    """Find first step where val_acc exceeds threshold."""
    for r in records:
        if r.get("val_acc", 0) > threshold:
            return r["step"]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Windowed DFA
# ═══════════════════════════════════════════════════════════════════════════

CURVATURE_METRICS = ["lambda1", "trace_H", "gHg"]
CONTROL_METRICS = ["weight_norm", "grad_norm"]
ALL_METRICS = CURVATURE_METRICS + CONTROL_METRICS

WINDOW_SIZE = 150  # points
STRIDE = 30  # points
R2_THRESHOLD = 0.80
PRE_POST_MARGIN = 100  # points (= 5000 steps at 50-step intervals)


def compute_windowed_dfa(signal, steps, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Compute DFA exponents in sliding windows.

    Returns list of dicts with: center_step, alpha, r_squared
    """
    results = []
    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start : start + window_size]
        center_idx = start + window_size // 2
        center_step = steps[center_idx]

        alpha, r2, _, _ = dfa(window, min_scale=4, max_scale_frac=0.25)
        results.append(
            {
                "center_step": int(center_step),
                "center_idx": center_idx,
                "alpha": float(alpha) if not np.isnan(alpha) else None,
                "r_squared": float(r2) if not np.isnan(r2) else None,
            }
        )

    return results


def segment_windows(windowed_dfa, grok_step, steps, margin=PRE_POST_MARGIN):
    """
    Split windowed DFA results into pre-grok, transition, and post-grok.

    margin is in telemetry points (not training steps).
    """
    if grok_step is None:
        return {"pre": windowed_dfa, "transition": [], "post": []}

    # Find the index of the grok step
    grok_idx = None
    for i, s in enumerate(steps):
        if s >= grok_step:
            grok_idx = i
            break

    if grok_idx is None:
        return {"pre": windowed_dfa, "transition": [], "post": []}

    pre_cutoff_idx = grok_idx - margin
    post_cutoff_idx = grok_idx + margin

    pre = []
    transition = []
    post = []

    for w in windowed_dfa:
        if w["alpha"] is None:
            continue
        if w["r_squared"] is not None and w["r_squared"] < R2_THRESHOLD:
            continue  # Skip low-quality windows

        if w["center_idx"] < pre_cutoff_idx:
            pre.append(w)
        elif w["center_idx"] > post_cutoff_idx:
            post.append(w)
        else:
            transition.append(w)

    return {"pre": pre, "transition": transition, "post": post}


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════


def paired_t_test(alpha_pre, alpha_post):
    """Paired t-test for DFA shift (one-sided: post > pre)."""
    deltas = np.array(alpha_post) - np.array(alpha_pre)
    n = len(deltas)
    if n < 3:
        return {"t": np.nan, "p": np.nan, "n": n, "mean_delta": np.nan}

    t_stat, p_two = stats.ttest_rel(alpha_post, alpha_pre)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    # Normality check
    if n >= 8:
        shapiro_stat, shapiro_p = stats.shapiro(deltas)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    # Also compute Wilcoxon (non-parametric backup)
    try:
        wilcox_stat, wilcox_p_two = stats.wilcoxon(deltas, alternative="greater")
    except ValueError:
        wilcox_stat, wilcox_p_two = np.nan, np.nan

    return {
        "t": float(t_stat),
        "p_one_sided": float(p_one),
        "p_two_sided": float(p_two),
        "n": n,
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)),
        "cohens_d": float(np.mean(deltas) / (np.std(deltas, ddof=1) + 1e-10)),
        "shapiro_stat": float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
        "shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else None,
        "wilcoxon_stat": float(wilcox_stat) if not np.isnan(wilcox_stat) else None,
        "wilcoxon_p": float(wilcox_p_two) if not np.isnan(wilcox_p_two) else None,
    }


def early_warning_test(seed_results, n_std=2):
    """
    For each seed, find the first window where α exceeds baseline + n_std * σ.

    Returns list of lead times (T_grok - T_detect) and success rate.
    """
    lead_times = []
    successes = 0
    total = 0

    for seed_id, sr in seed_results.items():
        if sr["grok_step"] is None:
            continue

        for metric in CURVATURE_METRICS:
            segments = sr["windowed"][metric]
            pre_alphas = [w["alpha"] for w in segments["pre"]]
            if len(pre_alphas) < 3:
                continue

            baseline_mean = np.mean(pre_alphas)
            baseline_std = np.std(pre_alphas)
            threshold = baseline_mean + n_std * baseline_std

            total += 1

            # Look through ALL windows (including transition) for first exceedance
            all_windows = segments["pre"] + segments["transition"] + segments["post"]
            all_windows.sort(key=lambda w: w["center_step"])

            detected = False
            for w in all_windows:
                if w["alpha"] > threshold and w["center_step"] < sr["grok_step"]:
                    lead = sr["grok_step"] - w["center_step"]
                    lead_times.append(
                        {
                            "seed": seed_id,
                            "metric": metric,
                            "lead_steps": int(lead),
                            "detect_step": w["center_step"],
                            "grok_step": sr["grok_step"],
                        }
                    )
                    if lead >= 500:
                        successes += 1
                    detected = True
                    break

            if not detected:
                lead_times.append(
                    {
                        "seed": seed_id,
                        "metric": metric,
                        "lead_steps": 0,
                        "detect_step": None,
                        "grok_step": sr["grok_step"],
                    }
                )

    return {
        "lead_times": lead_times,
        "success_rate_500": successes / total if total > 0 else 0,
        "total_tests": total,
        "successes_500": successes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results/study13")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to save analysis output (default: results_root)"
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.output_dir) if args.output_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all seeds ─────────────────────────────────────────────────
    print("Loading seed data...")
    seed_dirs = sorted(root.glob("seed_*"))
    all_data = {}

    for sd in seed_dirs:
        seed_id = sd.name
        data = load_seed_data(sd)
        if data is None:
            print(f"  {seed_id}: NO DATA")
            continue

        hess_records = extract_hessian_records(data["records"])
        grok_step = detect_grok_step(data["records"])

        n_total = len(data["records"])
        n_hess = len(hess_records)
        grok_str = f"GROK@{grok_step}" if grok_step else "NO_GROK"
        print(f"  {seed_id}: {n_total} records ({n_hess} with Hessian), {grok_str}")

        all_data[seed_id] = {
            "records": data["records"],
            "hessian_records": hess_records,
            "summary": data["summary"],
            "grok_step": grok_step,
        }

    if not all_data:
        print("ERROR: No seed data found.")
        return

    n_grokked = sum(1 for d in all_data.values() if d["grok_step"] is not None)
    print(f"\nLoaded {len(all_data)} seeds, {n_grokked} grokked.")

    # ── Global DFA (full time series) ──────────────────────────────────
    print("\n" + "=" * 70)
    print("GLOBAL DFA EXPONENTS (full time series per seed)")
    print("=" * 70)

    global_dfa = {}
    for seed_id, sd in all_data.items():
        hess = sd["hessian_records"]
        if len(hess) < 50:
            continue

        global_dfa[seed_id] = {}
        for metric in ALL_METRICS:
            vals = [r[metric] for r in hess if metric in r]
            if len(vals) < 50:
                global_dfa[seed_id][metric] = {"alpha": None, "r2": None}
                continue
            alpha, r2, _, _ = dfa(vals)
            global_dfa[seed_id][metric] = {
                "alpha": float(alpha) if not np.isnan(alpha) else None,
                "r2": float(r2) if not np.isnan(r2) else None,
            }

    # Print summary
    for metric in ALL_METRICS:
        alphas = [global_dfa[s][metric]["alpha"] for s in global_dfa if global_dfa[s][metric]["alpha"] is not None]
        if alphas:
            print(f"  {metric:15s}: mean α = {np.mean(alphas):.3f} ± {np.std(alphas):.3f}  (n={len(alphas)})")

    # ── Windowed DFA ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("WINDOWED DFA ANALYSIS")
    print(f"  Window: {WINDOW_SIZE} pts, Stride: {STRIDE} pts, R² threshold: {R2_THRESHOLD}")
    print("=" * 70)

    seed_results = {}
    for seed_id, sd in all_data.items():
        hess = sd["hessian_records"]
        if len(hess) < WINDOW_SIZE:
            print(f"  {seed_id}: too few Hessian records ({len(hess)}), skipping")
            continue

        steps = np.array([r["step"] for r in hess])
        windowed = {}

        for metric in ALL_METRICS:
            vals = np.array([r.get(metric, np.nan) for r in hess])
            if np.any(np.isnan(vals)):
                windowed[metric] = {"pre": [], "transition": [], "post": []}
                continue

            wdfa = compute_windowed_dfa(vals, steps)
            segments = segment_windows(wdfa, sd["grok_step"], steps)
            windowed[metric] = segments

        seed_results[seed_id] = {
            "grok_step": sd["grok_step"],
            "windowed": windowed,
            "n_hessian": len(hess),
        }

        # Print per-seed summary
        grok_str = f"GROK@{sd['grok_step']}" if sd["grok_step"] else "NO_GROK"
        for metric in CURVATURE_METRICS:
            seg = windowed[metric]
            n_pre = len(seg["pre"])
            n_post = len(seg["post"])
            pre_alphas = [w["alpha"] for w in seg["pre"] if w["alpha"] is not None]
            post_alphas = [w["alpha"] for w in seg["post"] if w["alpha"] is not None]
            pre_mean = np.mean(pre_alphas) if pre_alphas else float("nan")
            post_mean = np.mean(post_alphas) if post_alphas else float("nan")
            print(
                f"  {seed_id} {metric:10s}: pre α={pre_mean:.3f} ({n_pre}w) → "
                f"post α={post_mean:.3f} ({n_post}w)  {grok_str}"
            )

    # ── Statistical Tests ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    test_results = {}

    # Test 1: Paired t-test for DFA shift
    print("\nTest 1: Paired t-test (α_post > α_pre)")
    print("-" * 50)
    for metric in ALL_METRICS:
        alpha_pre_list = []
        alpha_post_list = []

        for _seed_id, sr in seed_results.items():
            if sr["grok_step"] is None:
                continue
            seg = sr["windowed"].get(metric, {})
            pre_a = [w["alpha"] for w in seg.get("pre", []) if w["alpha"] is not None]
            post_a = [w["alpha"] for w in seg.get("post", []) if w["alpha"] is not None]

            if pre_a and post_a:
                alpha_pre_list.append(np.mean(pre_a))
                alpha_post_list.append(np.mean(post_a))

        if len(alpha_pre_list) >= 3:
            result = paired_t_test(alpha_pre_list, alpha_post_list)
            is_curvature = metric in CURVATURE_METRICS
            label = "CURVATURE" if is_curvature else "CONTROL"
            sig = (
                "***"
                if result["p_one_sided"] < 0.001
                else "**"
                if result["p_one_sided"] < 0.01
                else "*"
                if result["p_one_sided"] < 0.05
                else "ns"
            )

            print(
                f"  {metric:15s} [{label}]: Δα = {result['mean_delta']:+.4f} ± "
                f"{result['std_delta']:.4f}, t({result['n'] - 1}) = {result['t']:.3f}, "
                f"p = {result['p_one_sided']:.4g} {sig}, d = {result['cohens_d']:.3f}"
            )

            if result.get("shapiro_p") is not None and result["shapiro_p"] < 0.05:
                print(
                    f"  {'':15s}  ⚠ Non-normal (Shapiro p={result['shapiro_p']:.3f}), "
                    f"Wilcoxon p={result.get('wilcoxon_p', 'N/A')}"
                )

            test_results[f"test1_{metric}"] = result
            test_results[f"test1_{metric}"]["alpha_pre"] = alpha_pre_list
            test_results[f"test1_{metric}"]["alpha_post"] = alpha_post_list
        else:
            print(f"  {metric:15s}: insufficient data (n={len(alpha_pre_list)})")

    # Test 3: Metric consistency (Spearman between shifts)
    print("\nTest 3: Metric consistency (Spearman ρ between Δα)")
    print("-" * 50)
    metric_deltas = {}
    for metric in CURVATURE_METRICS:
        key = f"test1_{metric}"
        if key in test_results and "alpha_pre" in test_results[key]:
            pre = np.array(test_results[key]["alpha_pre"])
            post = np.array(test_results[key]["alpha_post"])
            metric_deltas[metric] = post - pre

    for i, m1 in enumerate(CURVATURE_METRICS):
        for m2 in CURVATURE_METRICS[i + 1 :]:
            if m1 in metric_deltas and m2 in metric_deltas:
                rho, p = stats.spearmanr(metric_deltas[m1], metric_deltas[m2])
                print(f"  {m1} vs {m2}: ρ = {rho:.3f}, p = {p:.4g}")
                test_results[f"test3_{m1}_vs_{m2}"] = {"rho": float(rho), "p": float(p)}

    # Test 5: Early warning
    print("\nTest 5: Early-warning lead time")
    print("-" * 50)
    ew = early_warning_test(seed_results)
    print(
        f"  Success rate (≥500 steps early): {ew['successes_500']}/{ew['total_tests']} = {ew['success_rate_500']:.1%}"
    )
    for lt in ew["lead_times"]:
        if lt["detect_step"] is not None:
            print(f"    {lt['seed']} {lt['metric']}: detected {lt['lead_steps']} steps early")
    test_results["test5_early_warning"] = ew

    # ── Save Results ───────────────────────────────────────────────────
    output = {
        "study": "study13",
        "n_seeds": len(all_data),
        "n_grokked": n_grokked,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "r2_threshold": R2_THRESHOLD,
        "pre_post_margin_pts": PRE_POST_MARGIN,
        "global_dfa": global_dfa,
        "test_results": {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, np.ndarray))} if isinstance(v, dict) else v
            for k, v in test_results.items()
        },
        "per_seed_summary": {
            seed_id: {
                "grok_step": sr["grok_step"],
                "n_hessian": sr["n_hessian"],
                "pre_post_alphas": {
                    metric: {
                        "pre_mean": float(
                            np.mean([w["alpha"] for w in sr["windowed"][metric]["pre"] if w["alpha"] is not None])
                        )
                        if sr["windowed"][metric]["pre"]
                        else None,
                        "post_mean": float(
                            np.mean([w["alpha"] for w in sr["windowed"][metric]["post"] if w["alpha"] is not None])
                        )
                        if sr["windowed"][metric]["post"]
                        else None,
                        "n_pre_windows": len(sr["windowed"][metric]["pre"]),
                        "n_post_windows": len(sr["windowed"][metric]["post"]),
                    }
                    for metric in ALL_METRICS
                },
            }
            for seed_id, sr in seed_results.items()
        },
    }

    results_path = out_dir / "study13_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Results saved to {results_path}")

    # Save detailed data for plotting
    plot_data = {
        "seed_results": seed_results,
        "global_dfa": global_dfa,
        "test_results": test_results,
        "all_data_steps": {seed_id: [r["step"] for r in sd["hessian_records"]] for seed_id, sd in all_data.items()},
        "all_data_signals": {
            seed_id: {metric: [r.get(metric) for r in sd["hessian_records"]] for metric in ALL_METRICS}
            for seed_id, sd in all_data.items()
        },
    }
    pkl_path = out_dir / "study13_windowed_alphas.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(plot_data, f)
    print(f"✓ Plot data saved to {pkl_path}")


if __name__ == "__main__":
    main()
