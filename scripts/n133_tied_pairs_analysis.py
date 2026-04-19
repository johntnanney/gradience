"""N133 tied-pairs analysis (no new compute).

Question: how often does mean_alignment produce exact or near-exact ties
across the 66 N133 adapter pairs, what is the within-tie variance of merge
degradation on the evaluated subset, and is the tying a property of the
metric or of float representation?

Inputs (all local):
  sidecar/data/n133/pod_pull/audits/pair_alignment_summary.json
  sidecar/data/n133/pod_pull/audits/pair_alignment_full.json
  sidecar/data/n133/pod_pull/merges/merge_eval_summary.json

Outputs:
  sidecar/data/n133/tied_pairs_analysis.json
  stdout table for the note-worthy tie clusters
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POD = ROOT / "sidecar" / "data" / "n133" / "pod_pull"
OUT = ROOT / "sidecar" / "data" / "n133" / "tied_pairs_analysis.json"


def load():
    pair_sum = json.loads((POD / "audits" / "pair_alignment_summary.json").read_text())
    pair_full = json.loads((POD / "audits" / "pair_alignment_full.json").read_text())
    merges = json.loads((POD / "merges" / "merge_eval_summary.json").read_text())
    return pair_sum, pair_full, merges


def max_degradation(r: dict) -> float:
    ta, tb = r["task_a"], r["task_b"]
    return max(r.get(f"degradation_{ta}", 0.0), r.get(f"degradation_{tb}", 0.0))


def cluster_ties(values: dict[str, float], tol: float) -> list[list[str]]:
    """Group keys whose values fall within `tol` of each other via
    order-and-chain: sort values, break at gaps > tol, return clusters
    of size >= 2."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    clusters: list[list[str]] = []
    cur: list[str] = [items[0][0]]
    cur_val = items[0][1]
    for k, v in items[1:]:
        if abs(v - cur_val) <= tol:
            cur.append(k)
        else:
            if len(cur) >= 2:
                clusters.append(cur)
            cur = [k]
        cur_val = v
    if len(cur) >= 2:
        clusters.append(cur)
    return clusters


def float32_ulp(x: float) -> float:
    """Approximate float32 unit-in-last-place at value x."""
    return float(np.spacing(np.float32(x)))


def main():
    pair_sum, pair_full, merges = load()

    # 1. Collect alignments
    align = {k: v["mean_alignment"] for k, v in pair_sum.items()}
    cross = {k: a for k, a in align.items() if not pair_sum[k]["is_same_task"]}
    same = {k: a for k, a in align.items() if pair_sum[k]["is_same_task"]}

    print("=" * 72)
    print("  N133 tied-pairs analysis — mean_alignment resolution")
    print("=" * 72)
    print(f"\nTotal pairs:      {len(align)}  (same={len(same)}, cross={len(cross)})")
    print(f"Cross alignment range: [{min(cross.values()):.6f}, "
          f"{max(cross.values()):.6f}]")
    print(f"Same  alignment range: [{min(same.values()):.6f}, "
          f"{max(same.values()):.6f}]")

    # 2. Exact ties (bit-equal)
    exact = defaultdict(list)
    for k, v in align.items():
        exact[v].append(k)
    exact_clusters = [ks for v, ks in exact.items() if len(ks) >= 2]
    print(f"\nExact bit-equal tie clusters: {len(exact_clusters)}")
    for cl in exact_clusters:
        v = align[cl[0]]
        print(f"  value = {v:.8f}  (ulp_f32 ≈ {float32_ulp(v):.2e})")
        for k in cl:
            same_flag = "same" if pair_sum[k]["is_same_task"] else "cross"
            print(f"    {k:<45s} [{same_flag}]")

    # 3. Approximate ties at several tolerances
    # 1e-4 is roughly the printed precision in reports; 1e-6 is ~float32 ulp at 0.04
    print("\nCross-task ties at different tolerances (size >= 2):")
    for tol in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        cl = cluster_ties(cross, tol)
        n_pairs_in_ties = sum(len(c) for c in cl)
        print(f"  tol={tol:.0e}   clusters={len(cl):>2d}   "
              f"pairs_in_ties={n_pairs_in_ties:>2d}/{len(cross)}  "
              f"  largest={(max((len(c) for c in cl), default=0))}")

    # 4. Layer-level origin: if the pair has 128 per-layer numbers averaged to
    # 4 sig figs, "ties" at 1e-4 are expected by the central limit. Check.
    # For each pair, compute std of per-layer alignment and mean.
    per_layer_stats = {}
    for k, pair in pair_full.items():
        per_layer = np.array([layer["alignment"] for layer in pair["per_layer"]])
        per_layer_stats[k] = {
            "mean": float(per_layer.mean()),
            "std": float(per_layer.std()),
            "sem": float(per_layer.std() / np.sqrt(len(per_layer))),
            "n": len(per_layer),
        }

    cross_sem = np.array([per_layer_stats[k]["sem"] for k in cross])
    print(f"\nPer-pair layer-level SEM (cross-task): "
          f"median={np.median(cross_sem):.5f}  "
          f"p10={np.percentile(cross_sem, 10):.5f}  "
          f"p90={np.percentile(cross_sem, 90):.5f}")
    print("  (Two pair means within 1 SEM of each other are statistically "
          "indistinguishable at the per-layer sample.)")

    # 5. Within-tie variance of max_degradation on evaluated pairs
    eval_deg = {}
    for r in merges["results"]:
        if r["category"] == "same_task":
            continue
        eval_deg[r["pair"]] = max_degradation(r)
    print(f"\nEvaluated cross-task pairs: {len(eval_deg)}")

    # Use a tolerance informed by the per-layer SEM scale
    tol_used = 1e-4
    cross_clusters = cluster_ties(cross, tol_used)
    print(f"\nAt tol={tol_used}, cross-task has {len(cross_clusters)} tie "
          "clusters. Evaluated-subset within-cluster stats:")

    within_cluster_findings = []
    for cl in cross_clusters:
        evaluated_in_cluster = [k for k in cl if k in eval_deg]
        if len(evaluated_in_cluster) < 2:
            continue
        degs = np.array([eval_deg[k] for k in evaluated_in_cluster])
        finding = {
            "alignment": align[evaluated_in_cluster[0]],
            "members_total": len(cl),
            "members_evaluated": len(evaluated_in_cluster),
            "max_deg_min": float(degs.min()),
            "max_deg_max": float(degs.max()),
            "max_deg_range": float(degs.max() - degs.min()),
            "max_deg_std": float(degs.std()),
            "n_good": int((degs < 0.10).sum()),
            "n_bad": int((degs >= 0.10).sum()),
            "pairs": [
                {
                    "pair": k,
                    "max_deg": float(eval_deg[k]),
                    "good": bool(eval_deg[k] < 0.10),
                }
                for k in evaluated_in_cluster
            ],
        }
        within_cluster_findings.append(finding)

    for f in within_cluster_findings:
        print(f"\n  alignment = {f['alignment']:.6f}  "
              f"({f['members_evaluated']}/{f['members_total']} evaluated)")
        print(f"    max_deg range: [{f['max_deg_min']:+.3f}, "
              f"{f['max_deg_max']:+.3f}]   std = {f['max_deg_std']:.3f}")
        print(f"    good/bad split: {f['n_good']}/{f['n_bad']}")
        for p in f["pairs"]:
            tag = "GOOD" if p["good"] else "bad "
            print(f"      {p['pair']:<45s} max_deg={p['max_deg']:+.3f}  {tag}")

    # 6. Compare to what cross-cluster separation looks like. Is the
    # within-tie std comparable to the across-tie std on evaluated subset?
    eval_degs_arr = np.array(list(eval_deg.values()))
    across_std = float(eval_degs_arr.std())
    print(f"\nAcross-cluster max_deg std (all 12 evaluated): {across_std:.3f}")
    if within_cluster_findings:
        within_std_mean = float(np.mean([f["max_deg_std"] for f in within_cluster_findings]))
        print(f"Mean within-tie-cluster max_deg std:           {within_std_mean:.3f}")
        print(f"Ratio (within / across): {within_std_mean / across_std:.2f}")
        print("  (Ratio near 1.0 means tied pairs are as variable in outcome "
              "as the full sample — i.e. the metric has zero resolution "
              "within the tie.)")

    # 7. Bit-precision check: is the observed tie clustering consistent with
    # float32 rounding or with genuine metric structure?
    # For each cross-task alignment value, compute distance to nearest other
    # alignment value and compare to the float32 ULP at that magnitude.
    vals = sorted(cross.values())
    min_gaps = []
    for i, v in enumerate(vals):
        neighbors = []
        if i > 0:
            neighbors.append(abs(v - vals[i-1]))
        if i < len(vals) - 1:
            neighbors.append(abs(v - vals[i+1]))
        if neighbors:
            min_gaps.append(min(neighbors))
    min_gaps_arr = np.array(min_gaps)
    typical_ulp = float32_ulp(float(np.median(vals)))
    print(f"\nFloat-precision check (cross-task only):")
    print(f"  median min-gap-to-nearest-neighbor: {np.median(min_gaps_arr):.2e}")
    print(f"  min      min-gap-to-nearest-neighbor: {min_gaps_arr.min():.2e}")
    print(f"  float32 ulp at median alignment:     {typical_ulp:.2e}")
    ratio = float(min_gaps_arr.min() / typical_ulp) if typical_ulp > 0 else float("inf")
    print(f"  smallest gap / ulp:                  {ratio:.1f}")
    if ratio > 100:
        print("  → ties are structural in the metric, not float32 rounding.")
    else:
        print("  → ties at float-precision scale possible; check bit equality.")

    # 8. Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "N133 tied-pairs analysis",
        "n_total_pairs": len(align),
        "n_same_task": len(same),
        "n_cross_task": len(cross),
        "cross_alignment_range": [min(cross.values()), max(cross.values())],
        "exact_bit_equal_tie_clusters": [
            {
                "alignment": align[cl[0]],
                "pairs": cl,
            }
            for cl in exact_clusters
        ],
        "tolerance_sweep_cross": {
            f"{tol:.0e}": {
                "n_clusters": len(cluster_ties(cross, tol)),
                "n_pairs_in_ties": sum(len(c) for c in cluster_ties(cross, tol)),
            }
            for tol in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        },
        "within_tie_degradation_analysis": within_cluster_findings,
        "tol_used_for_eval_analysis": tol_used,
        "across_cluster_max_deg_std": across_std,
        "mean_within_cluster_max_deg_std": (
            float(np.mean([f["max_deg_std"] for f in within_cluster_findings]))
            if within_cluster_findings else None
        ),
        "float_precision": {
            "median_neighbor_gap": float(np.median(min_gaps_arr)),
            "min_neighbor_gap": float(min_gaps_arr.min()),
            "float32_ulp_at_median": typical_ulp,
            "min_gap_over_ulp": ratio,
            "interpretation": (
                "structural" if ratio > 100 else "float-precision-possible"
            ),
        },
        "per_pair_layer_stats": per_layer_stats,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Saved {OUT}")


if __name__ == "__main__":
    main()
