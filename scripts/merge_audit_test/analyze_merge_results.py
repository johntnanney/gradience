#!/usr/bin/env python3
"""
analyze_merge_results.py — Deep analysis of merge-audit outputs.

Reads the merge_audit.json files from each pair and produces:
1. Cross-pair comparison: do different task pairings produce different profiles?
2. Per-layer patterns: architectural regularities across pairs
3. Sanity checks: are metrics in expected ranges?
4. Hypothesis validation: do results match predictions?

Usage:
    python scripts/merge_audit_test/analyze_merge_results.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("/workspace/merge_test/results")


# ---------------------------------------------------------------------------
# JSON schema reference (from gradience.vnext.merge.report.to_json):
#
# {
#   "schema_version": "gradience.merge_audit/v1",
#   "timestamp": ...,
#   "gradience_version": ...,
#   "adapter_a": {"path", "rank", "alpha", "base_model", "n_layers"},
#   "adapter_b": {"path", "rank", "alpha", "base_model", "n_layers"},
#   "matching": {"n_shared", "n_only_a", "n_only_b", "shared_layers", ...},
#   "aggregate": {
#       "overall_verdict", "compatibility_score",
#       "mean_overlap", "median_overlap", "max_overlap",
#       "mean_agreement",
#       "n_safe", "n_redundant", "n_conflicting", "n_imbalanced"
#   },
#   "per_layer": [
#       {
#           "layer_name", "module_type",
#           "metrics": {
#               "principal_angle_cosines", "mean_overlap", "max_overlap",
#               "directional_agreement", "magnitude_ratio", "frobenius_ratio",
#               "frobenius_norm_a", "frobenius_norm_b",
#               "effective_rank_a", "effective_rank_b",
#               "nominal_rank_a", "nominal_rank_b",
#               "stable_rank_a", "stable_rank_b"
#           },
#           "verdict", "confidence", "recommendation",
#           "conflict_dimensions", "safe_merge_rank",
#           "suggested_strategy", "suggested_coefficients"
#       },
#       ...
#   ],
#   "recommendations": [...],
#   "issues": [...],
#   "warnings": [...],
#   "thresholds": {...}
# }
# ---------------------------------------------------------------------------


def load_audit(pair_dir: str) -> dict | None:
    """Load merge_audit.json for a given pair directory."""
    path = RESULTS_DIR / pair_dir / "merge_audit.json"
    if not path.exists():
        print(f"  [missing] {pair_dir}: no merge_audit.json")
        return None
    return json.loads(path.read_text())


def analyze_pair(pair_dir: str, audit: dict) -> dict:
    """Detailed analysis of one adapter pair."""
    print(f"\n{'=' * 70}")
    print(f"  PAIR: {pair_dir}")
    print(f"{'=' * 70}")

    a = audit["adapter_a"]
    b = audit["adapter_b"]
    agg = audit["aggregate"]
    layers = audit["per_layer"]

    print(f"\n  Adapter A: {a['path']}")
    print(f"    rank={a['rank']}, alpha={a['alpha']}, base={a['base_model']}, layers={a['n_layers']}")
    print(f"  Adapter B: {b['path']}")
    print(f"    rank={b['rank']}, alpha={b['alpha']}, base={b['base_model']}, layers={b['n_layers']}")
    print(f"  Shared: {audit['matching']['n_shared']}, "
          f"Only-A: {audit['matching']['n_only_a']}, "
          f"Only-B: {audit['matching']['n_only_b']}")

    print(f"\n  Overall verdict: {agg['overall_verdict'].upper()}")
    print(f"  Score:           {agg['compatibility_score']:.4f}")
    print(f"  Mean overlap:    {agg['mean_overlap']:.4f}")
    print(f"  Median overlap:  {agg['median_overlap']:.4f}")
    print(f"  Max overlap:     {agg['max_overlap']:.4f}")
    print(f"  Mean agreement:  {agg['mean_agreement']:.4f}")

    # Per-layer table
    print(f"\n  {'Layer':<40s} {'r_a':>4s} {'r_b':>4s} {'eff_a':>5s} {'eff_b':>5s} "
          f"{'Overlap':>8s} {'Agree':>8s} {'MagR':>6s} {'Verdict':<12s} {'Strategy':<8s}")
    print(f"  {'-'*40} {'-'*4} {'-'*4} {'-'*5} {'-'*5} "
          f"{'-'*8} {'-'*8} {'-'*6} {'-'*12} {'-'*8}")

    overlaps = []
    agreements = []
    mag_ratios = []
    verdicts_count = defaultdict(int)
    by_module_type = defaultdict(list)

    for lv in layers:
        m = lv["metrics"]
        name = lv["layer_name"]
        # Shorten: keep last meaningful parts
        parts = name.split(".")
        short = name
        for i, p in enumerate(parts):
            if p.isdigit():
                short = ".".join(parts[i:])
                short = f"L{parts[i]}.{'.'.join(parts[i+1:])}"
                break
        if len(short) > 40:
            short = "..." + short[-37:]

        overlap = m["mean_overlap"]
        agreement = m["directional_agreement"]
        mag_ratio = m["magnitude_ratio"]

        print(f"  {short:<40s} {m['nominal_rank_a']:>4d} {m['nominal_rank_b']:>4d} "
              f"{m['effective_rank_a']:>5d} {m['effective_rank_b']:>5d} "
              f"{overlap:>8.3f} {agreement:>+8.3f} {mag_ratio:>6.1f} "
              f"{lv['verdict']:<12s} {lv['suggested_strategy']:<8s}")

        overlaps.append(overlap)
        agreements.append(agreement)
        mag_ratios.append(mag_ratio)
        verdicts_count[lv["verdict"]] += 1
        by_module_type[lv["module_type"]].append(m)

    # Statistics
    print(f"\n  --- Statistics ---")
    print(f"  Overlap:   mean={statistics.mean(overlaps):.4f}, "
          f"stdev={statistics.stdev(overlaps):.4f}, "
          f"min={min(overlaps):.4f}, max={max(overlaps):.4f}")
    print(f"  Agreement: mean={statistics.mean(agreements):.4f}, "
          f"stdev={statistics.stdev(agreements):.4f}, "
          f"min={min(agreements):.4f}, max={max(agreements):.4f}")
    print(f"  Mag ratio: mean={statistics.mean(mag_ratios):.2f}, "
          f"min={min(mag_ratios):.2f}, max={max(mag_ratios):.2f}")

    print(f"\n  Verdict distribution:")
    for v, count in sorted(verdicts_count.items()):
        print(f"    {v}: {count}")

    # Module type breakdown
    print(f"\n  By module type:")
    for mtype, metrics_list in sorted(by_module_type.items()):
        type_overlaps = [m["mean_overlap"] for m in metrics_list]
        type_agreements = [m["directional_agreement"] for m in metrics_list]
        print(f"    {mtype} ({len(metrics_list)} layers): "
              f"overlap={statistics.mean(type_overlaps):.3f}, "
              f"agreement={statistics.mean(type_agreements):+.3f}")

    # Flag noteworthy layers
    print(f"\n  Noteworthy layers:")
    found_any = False
    for lv in layers:
        m = lv["metrics"]
        if m["mean_overlap"] > 0.5:
            print(f"    HIGH OVERLAP:  {lv['layer_name']} (overlap={m['mean_overlap']:.3f})")
            found_any = True
        if m["directional_agreement"] < -0.3 and m["mean_overlap"] > 0.3:
            print(f"    CONFLICT:      {lv['layer_name']} "
                  f"(agree={m['directional_agreement']:.3f}, overlap={m['mean_overlap']:.3f})")
            found_any = True
        if m["magnitude_ratio"] > 5.0:
            print(f"    IMBALANCED:    {lv['layer_name']} (mag_ratio={m['magnitude_ratio']:.1f}x)")
            found_any = True
    if not found_any:
        print(f"    (none)")

    return {
        "pair": pair_dir,
        "n_layers": len(layers),
        "verdict": agg["overall_verdict"],
        "score": agg["compatibility_score"],
        "mean_overlap": statistics.mean(overlaps),
        "stdev_overlap": statistics.stdev(overlaps) if len(overlaps) > 1 else 0,
        "mean_agreement": statistics.mean(agreements),
        "stdev_agreement": statistics.stdev(agreements) if len(agreements) > 1 else 0,
        "mean_mag_ratio": statistics.mean(mag_ratios),
        "verdicts": dict(verdicts_count),
    }


def cross_pair_comparison(summaries: list):
    """Compare metrics across all tested pairs."""
    print(f"\n\n{'=' * 70}")
    print(f"  CROSS-PAIR COMPARISON")
    print(f"{'=' * 70}")

    print(f"\n  {'Pair':<35s} {'Layers':>6s} {'Overlap':>10s} {'Agree':>10s} "
          f"{'MagR':>6s} {'Verdict':>12s}")
    print(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*10} {'-'*6} {'-'*12}")

    for s in summaries:
        print(f"  {s['pair']:<35s} {s['n_layers']:>6d} "
              f"{s['mean_overlap']:>5.3f}+{s['stdev_overlap']:.3f} "
              f"{s['mean_agreement']:>+5.3f}+{s['stdev_agreement']:.3f} "
              f"{s['mean_mag_ratio']:>6.1f} "
              f"{s['verdict']:>12s}")

    # Sort by overlap to validate hypotheses
    sorted_by_overlap = sorted(summaries, key=lambda s: s["mean_overlap"])
    print(f"\n  Pairs ranked by overlap (low -> high):")
    for i, s in enumerate(sorted_by_overlap, 1):
        print(f"    {i}. {s['pair']}: mean_overlap={s['mean_overlap']:.4f}")


def sanity_checks(summaries: list):
    """Validate metrics are in expected ranges."""
    print(f"\n\n{'=' * 70}")
    print(f"  SANITY CHECKS")
    print(f"{'=' * 70}")

    checks_passed = 0
    checks_total = 0

    for s in summaries:
        pair = s["pair"]

        # Check 1: Overlap in [0, 1]
        checks_total += 1
        if 0.0 <= s["mean_overlap"] <= 1.0:
            print(f"  [PASS] {pair}: overlap in [0, 1] ({s['mean_overlap']:.4f})")
            checks_passed += 1
        else:
            print(f"  [FAIL] {pair}: overlap out of range ({s['mean_overlap']:.4f})")

        # Check 2: Agreement in [-1, 1]
        checks_total += 1
        if -1.0 <= s["mean_agreement"] <= 1.0:
            print(f"  [PASS] {pair}: agreement in [-1, 1] ({s['mean_agreement']:.4f})")
            checks_passed += 1
        else:
            print(f"  [FAIL] {pair}: agreement out of range ({s['mean_agreement']:.4f})")

        # Check 3: Magnitude ratio >= 1.0
        checks_total += 1
        if s["mean_mag_ratio"] >= 1.0:
            print(f"  [PASS] {pair}: magnitude ratio >= 1.0 ({s['mean_mag_ratio']:.2f})")
            checks_passed += 1
        else:
            print(f"  [FAIL] {pair}: magnitude ratio < 1.0 ({s['mean_mag_ratio']:.2f})")

        # Check 4: Not all layers have identical metrics (bug indicator)
        checks_total += 1
        if s["stdev_overlap"] > 0.001:
            print(f"  [PASS] {pair}: overlap varies across layers (stdev={s['stdev_overlap']:.4f})")
            checks_passed += 1
        else:
            print(f"  [WARN] {pair}: all layers nearly identical overlap "
                  f"(stdev={s['stdev_overlap']:.6f}) — possible bug")

        # Check 5: At least 2 different verdict types across all layers
        checks_total += 1
        n_verdict_types = len(s["verdicts"])
        if n_verdict_types >= 2:
            print(f"  [PASS] {pair}: {n_verdict_types} verdict types")
            checks_passed += 1
        else:
            print(f"  [INFO] {pair}: only {n_verdict_types} verdict type(s): {list(s['verdicts'].keys())}")
            checks_passed += 1  # Not a failure, just informational

        # Check 6: Magnitude ratios in reasonable range for same-collection adapters
        checks_total += 1
        if s["mean_mag_ratio"] < 20.0:
            print(f"  [PASS] {pair}: magnitude ratio reasonable ({s['mean_mag_ratio']:.1f}x)")
            checks_passed += 1
        else:
            print(f"  [WARN] {pair}: very high magnitude ratio ({s['mean_mag_ratio']:.1f}x)")

    # Cross-pair checks
    if len(summaries) >= 2:
        checks_total += 1
        overlaps = [s["mean_overlap"] for s in summaries]
        overlap_range = max(overlaps) - min(overlaps)
        if overlap_range > 0.02:
            print(f"\n  [PASS] Cross-pair: overlap varies meaningfully across pairs "
                  f"(range={overlap_range:.4f})")
            checks_passed += 1
        else:
            print(f"\n  [WARN] Cross-pair: all pairs have nearly the same overlap "
                  f"(range={overlap_range:.4f}) — metrics may not be discriminative")

    print(f"\n  {checks_passed}/{checks_total} checks passed")

    return checks_passed, checks_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Discover result directories
    if not RESULTS_DIR.exists():
        print(f"No results directory found at {RESULTS_DIR}")
        print("Run run_merge_audits.py first.")
        sys.exit(1)

    pair_dirs = sorted([
        d.name for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (d / "merge_audit.json").exists()
    ])

    if not pair_dirs:
        print("No merge_audit.json files found in any subdirectory.")
        print("Run run_merge_audits.py first.")
        sys.exit(1)

    print(f"Found {len(pair_dirs)} result(s): {pair_dirs}")

    # Analyze each pair
    summaries = []
    for pair_dir in pair_dirs:
        audit = load_audit(pair_dir)
        if audit:
            summary = analyze_pair(pair_dir, audit)
            summaries.append(summary)

    # Cross-pair comparison
    if len(summaries) >= 2:
        cross_pair_comparison(summaries)

    # Sanity checks
    passed, total = sanity_checks(summaries)

    # Write analysis summary
    analysis_path = RESULTS_DIR / "analysis_summary.json"
    with open(analysis_path, "w") as f:
        json.dump({
            "pair_summaries": summaries,
            "sanity_checks": {"passed": passed, "total": total},
        }, f, indent=2)
    print(f"\n  Analysis saved to {analysis_path}")

    print(f"\nNext: python scripts/merge_audit_test/gpu_verification.py")
