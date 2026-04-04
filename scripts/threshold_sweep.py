#!/usr/bin/env python3
"""Candidate Set 2 threshold sweep: local variations around current defaults.

Reuses layer profiles from the verdict boundary stress test and re-assigns
verdicts under each threshold combination.  Evaluates each against three
selection objectives:

  Obj-1: preserve low cross-task high-confidence SAFE rate
  Obj-2: avoid inflating same-task CONFLICTING/IMBALANCED rate
  Obj-3: reduce Branch-5 spectral conflation (H/L ratio KS between
         same-task and cross-task layers in the catch-all SAFE branch)

Output: sidecar/results/verdict_boundary_stress_test/sweep_results.json
        sidecar/results/verdict_boundary_stress_test/sweep_report.txt
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product as iterproduct
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROFILES_PATH = Path("sidecar/results/verdict_boundary_stress_test/profiles.json")
OUT_DIR = Path("sidecar/results/verdict_boundary_stress_test")

# ---------------------------------------------------------------------------
# Sweep grid  (Candidate Set 2 from the decision memo)
# ---------------------------------------------------------------------------

SWEEP_GRID: dict[str, list[float]] = {
    "high_overlap": [0.45, 0.50, 0.55],
    "aligned": [0.45, 0.50, 0.55],
    "conflicting": [-0.35, -0.30, -0.25],
    "imbalanced_frob": [3.0, 5.0, 7.0],
    "imbalanced": [3.0, 5.0, 7.0],
}

# low_overlap is held constant at 0.2 (the decision memo doesn't sweep it)
LOW_OVERLAP = 0.2


# ---------------------------------------------------------------------------
# Branch assignment (mirrors verdicts.py exactly)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Thresholds:
    low_overlap: float
    high_overlap: float
    aligned: float
    conflicting: float
    imbalanced: float
    imbalanced_frob: float

    def label(self) -> str:
        return (
            f"ho={self.high_overlap:.2f}_al={self.aligned:.2f}"
            f"_co={self.conflicting:.2f}_if={self.imbalanced_frob:.1f}"
            f"_im={self.imbalanced:.1f}"
        )


BRANCH_NAME = {
    0: "imbalanced_low_overlap",
    1: "safe_orthogonal",
    2: "redundant",
    3: "conflicting",
    4: "imbalanced_remainder",
    5: "safe_default",
}


def branch_id(
    mean_overlap: float,
    directional_agreement: float,
    frobenius_ratio: float,
    t: Thresholds,
) -> int:
    if frobenius_ratio > t.imbalanced_frob and mean_overlap < t.high_overlap:
        return 0
    if mean_overlap < t.low_overlap:
        return 1
    if mean_overlap > t.high_overlap and directional_agreement > t.aligned:
        return 2
    if mean_overlap > t.high_overlap and directional_agreement < t.conflicting:
        return 3
    if frobenius_ratio > t.imbalanced_frob:
        return 4
    return 5


VERDICT_FOR_BRANCH = {
    0: "imbalanced",
    1: "safe",
    2: "redundant",
    3: "conflicting",
    4: "imbalanced",
    5: "safe",
}


# ---------------------------------------------------------------------------
# Selection objective evaluation
# ---------------------------------------------------------------------------

def evaluate_objectives(
    rows: list[dict[str, Any]],
    t: Thresholds,
) -> dict[str, Any]:
    """Assign verdicts under thresholds *t* and score the three objectives."""

    same: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    b5_same: list[dict[str, Any]] = []
    b5_cross: list[dict[str, Any]] = []

    same_verdicts: Counter[str] = Counter()
    cross_verdicts: Counter[str] = Counter()

    for r in rows:
        bid = branch_id(
            r["mean_overlap"],
            r["directional_agreement"],
            r["frobenius_ratio"],
            t,
        )
        v = VERDICT_FOR_BRANCH[bid]

        if r["relationship"] == "same_task":
            same.append(r)
            same_verdicts[v] += 1
            if bid == 5:
                b5_same.append(r)
        else:
            cross.append(r)
            cross_verdicts[v] += 1
            if bid == 5:
                b5_cross.append(r)

    n_same = len(same) or 1
    n_cross = len(cross) or 1

    # Obj-1: cross-task high-confidence SAFE rate
    # "high-confidence SAFE" = branch 1 (orthogonal) or branch 5 (default)
    # with high overlap.  Simplified: fraction of cross-task getting SAFE verdict.
    cross_safe_rate = cross_verdicts["safe"] / n_cross

    # Obj-2: same-task CONFLICTING or IMBALANCED rate
    same_bad_rate = (same_verdicts["conflicting"] + same_verdicts["imbalanced"]) / n_same

    # Obj-3: Branch-5 H/L ratio separation
    hl_same = [r["hl_ratio"] for r in b5_same if math.isfinite(r.get("hl_ratio", float("nan")))]
    hl_cross = [r["hl_ratio"] for r in b5_cross if math.isfinite(r.get("hl_ratio", float("nan")))]

    if len(hl_same) >= 2 and len(hl_cross) >= 2:
        ks = ks_2samp(hl_same, hl_cross)
        b5_ks_stat = float(ks.statistic)
        b5_ks_p = float(ks.pvalue)
        b5_hl_same_mean = float(np.mean(hl_same))
        b5_hl_cross_mean = float(np.mean(hl_cross))
    else:
        b5_ks_stat = float("nan")
        b5_ks_p = float("nan")
        b5_hl_same_mean = float("nan")
        b5_hl_cross_mean = float("nan")

    b5_n = len(b5_same) + len(b5_cross)
    b5_fraction = b5_n / len(rows) if rows else 0.0

    return {
        "thresholds": t.label(),
        "high_overlap": t.high_overlap,
        "aligned": t.aligned,
        "conflicting": t.conflicting,
        "imbalanced_frob": t.imbalanced_frob,
        "imbalanced": t.imbalanced,
        # Objective scores
        "obj1_cross_safe_rate": cross_safe_rate,
        "obj2_same_bad_rate": same_bad_rate,
        "obj3_b5_ks_stat": b5_ks_stat,
        "obj3_b5_ks_p": b5_ks_p,
        "obj3_b5_hl_same_mean": b5_hl_same_mean,
        "obj3_b5_hl_cross_mean": b5_hl_cross_mean,
        "obj3_b5_n_total": b5_n,
        "obj3_b5_fraction": b5_fraction,
        # Verdict distributions
        "same_verdicts": dict(same_verdicts),
        "cross_verdicts": dict(cross_verdicts),
        "n_same": len(same),
        "n_cross": len(cross),
    }


def composite_score(result: dict[str, Any], baseline: dict[str, Any]) -> float:
    """Higher is better.

    Components:
      - Obj-1: we want cross_safe_rate LOW  (penalize increase from baseline)
      - Obj-2: we want same_bad_rate LOW    (penalize increase from baseline)
      - Obj-3: we want b5_fraction LOW      (fewer layers in catch-all branch)
               AND b5_ks_stat LOW / b5_ks_p HIGH (less separation = less conflation)

    Score = (each objective's improvement over baseline, clamped to [-1, 1])
    weighted sum.
    """
    # Obj-1: delta in cross_safe_rate (lower is better; negative delta = improvement)
    d1 = baseline["obj1_cross_safe_rate"] - result["obj1_cross_safe_rate"]

    # Obj-2: delta in same_bad_rate (lower is better; negative delta = improvement)
    d2 = baseline["obj2_same_bad_rate"] - result["obj2_same_bad_rate"]

    # Obj-3: reduction in Branch-5 traffic fraction (fewer layers in catch-all)
    d3_frac = baseline["obj3_b5_fraction"] - result["obj3_b5_fraction"]

    # Obj-3 secondary: if branch-5 still has traffic, is ks_stat lower?
    # (lower KS = less heterogeneity within branch-5)
    b5_ks_base = baseline["obj3_b5_ks_stat"]
    b5_ks_now = result["obj3_b5_ks_stat"]
    if math.isfinite(b5_ks_base) and math.isfinite(b5_ks_now):
        d3_ks = b5_ks_base - b5_ks_now
    else:
        d3_ks = 0.0

    # Weights: Obj-1 and Obj-2 are constraints (high weight),
    # Obj-3 is the optimization target.
    w1, w2, w3_frac, w3_ks = 3.0, 3.0, 2.0, 1.0

    return w1 * d1 + w2 * d2 + w3_frac * d3_frac + w3_ks * d3_ks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(
    results: list[dict[str, Any]],
    baseline: dict[str, Any],
    top_n: int = 15,
) -> str:
    lines: list[str] = []
    lines.append("CANDIDATE SET 2 — THRESHOLD SWEEP REPORT")
    lines.append("=" * 72)
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Sweep grid: {len(results)} threshold combinations")
    lines.append(f"Layer profiles: {baseline['n_same'] + baseline['n_cross']} "
                 f"(same-task: {baseline['n_same']}, cross-task: {baseline['n_cross']})")
    lines.append("")

    lines.append("BASELINE (current defaults)")
    lines.append("-" * 72)
    lines.append(f"  Thresholds: high_overlap=0.50, aligned=0.50, conflicting=-0.30, "
                 f"imbalanced_frob=5.0, imbalanced=5.0")
    lines.append(f"  Obj-1 cross-task SAFE rate:     {baseline['obj1_cross_safe_rate']:.4f}")
    lines.append(f"  Obj-2 same-task bad rate:        {baseline['obj2_same_bad_rate']:.4f}")
    lines.append(f"  Obj-3 Branch-5 fraction:         {baseline['obj3_b5_fraction']:.4f}")
    lines.append(f"  Obj-3 Branch-5 H/L KS stat:      {baseline['obj3_b5_ks_stat']:.4f} "
                 f"(p={baseline['obj3_b5_ks_p']:.2e})")
    lines.append(f"  Obj-3 Branch-5 H/L means:        same={baseline['obj3_b5_hl_same_mean']:.3f} "
                 f"cross={baseline['obj3_b5_hl_cross_mean']:.3f}")
    lines.append(f"  Same verdicts: {baseline['same_verdicts']}")
    lines.append(f"  Cross verdicts: {baseline['cross_verdicts']}")
    lines.append("")

    # Sort by composite score
    scored = [(r, composite_score(r, baseline)) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)

    lines.append(f"TOP {top_n} THRESHOLD COMBINATIONS (by composite score)")
    lines.append("-" * 72)
    lines.append(f"{'Rank':>4}  {'Score':>7}  {'Obj1 Δ':>8}  {'Obj2 Δ':>8}  "
                 f"{'B5 frac':>8}  {'B5 KS':>7}  Thresholds")
    lines.append("-" * 72)

    for i, (r, sc) in enumerate(scored[:top_n], 1):
        d1 = baseline["obj1_cross_safe_rate"] - r["obj1_cross_safe_rate"]
        d2 = baseline["obj2_same_bad_rate"] - r["obj2_same_bad_rate"]
        lines.append(
            f"{i:>4}  {sc:>+7.4f}  {d1:>+8.4f}  {d2:>+8.4f}  "
            f"{r['obj3_b5_fraction']:>8.4f}  {r['obj3_b5_ks_stat']:>7.4f}  "
            f"{r['thresholds']}"
        )

    # Also show the worst 5 to understand the danger zone
    lines.append("")
    lines.append("BOTTOM 5 (worst configurations)")
    lines.append("-" * 72)
    for i, (r, sc) in enumerate(scored[-5:], len(scored) - 4):
        d1 = baseline["obj1_cross_safe_rate"] - r["obj1_cross_safe_rate"]
        d2 = baseline["obj2_same_bad_rate"] - r["obj2_same_bad_rate"]
        lines.append(
            f"{i:>4}  {sc:>+7.4f}  {d1:>+8.4f}  {d2:>+8.4f}  "
            f"{r['obj3_b5_fraction']:>8.4f}  {r['obj3_b5_ks_stat']:>7.4f}  "
            f"{r['thresholds']}"
        )

    # Detailed breakdowns for top 5
    lines.append("")
    lines.append("DETAILED BREAKDOWN — TOP 5")
    lines.append("=" * 72)
    for i, (r, sc) in enumerate(scored[:5], 1):
        lines.append("")
        lines.append(f"#{i}  {r['thresholds']}  (score={sc:+.4f})")
        lines.append(f"  high_overlap={r['high_overlap']:.2f}  aligned={r['aligned']:.2f}  "
                     f"conflicting={r['conflicting']:.2f}  "
                     f"imbalanced_frob={r['imbalanced_frob']:.1f}  "
                     f"imbalanced={r['imbalanced']:.1f}")
        lines.append(f"  Obj-1 cross-task SAFE rate:   {r['obj1_cross_safe_rate']:.4f}  "
                     f"(baseline {baseline['obj1_cross_safe_rate']:.4f})")
        lines.append(f"  Obj-2 same-task bad rate:      {r['obj2_same_bad_rate']:.4f}  "
                     f"(baseline {baseline['obj2_same_bad_rate']:.4f})")
        lines.append(f"  Obj-3 Branch-5 fraction:       {r['obj3_b5_fraction']:.4f}  "
                     f"(baseline {baseline['obj3_b5_fraction']:.4f})")
        lines.append(f"  Obj-3 Branch-5 H/L KS:         {r['obj3_b5_ks_stat']:.4f}  "
                     f"(baseline {baseline['obj3_b5_ks_stat']:.4f})")
        lines.append(f"  Obj-3 Branch-5 H/L means:      same={r['obj3_b5_hl_same_mean']:.3f}  "
                     f"cross={r['obj3_b5_hl_cross_mean']:.3f}")
        lines.append(f"  Same verdicts:  {r['same_verdicts']}")
        lines.append(f"  Cross verdicts: {r['cross_verdicts']}")

    # Objective constraint summary
    lines.append("")
    lines.append("CONSTRAINT SATISFACTION SUMMARY")
    lines.append("-" * 72)
    n_improve_all = sum(
        1 for r, sc in scored
        if (r["obj1_cross_safe_rate"] <= baseline["obj1_cross_safe_rate"]
            and r["obj2_same_bad_rate"] <= baseline["obj2_same_bad_rate"]
            and r["obj3_b5_fraction"] <= baseline["obj3_b5_fraction"])
    )
    n_degrade_obj1 = sum(
        1 for r, _ in scored
        if r["obj1_cross_safe_rate"] > baseline["obj1_cross_safe_rate"] + 0.02
    )
    n_degrade_obj2 = sum(
        1 for r, _ in scored
        if r["obj2_same_bad_rate"] > baseline["obj2_same_bad_rate"] + 0.02
    )
    lines.append(f"  Configurations improving all three objectives:    {n_improve_all} / {len(scored)}")
    lines.append(f"  Configurations degrading Obj-1 by >2pp:           {n_degrade_obj1} / {len(scored)}")
    lines.append(f"  Configurations degrading Obj-2 by >2pp:           {n_degrade_obj2} / {len(scored)}")

    # Parameter sensitivity
    lines.append("")
    lines.append("PARAMETER SENSITIVITY (mean composite score by parameter value)")
    lines.append("-" * 72)
    for param_name, param_values in SWEEP_GRID.items():
        for val in param_values:
            matching = [sc for r, sc in scored if abs(r.get(param_name, -999) - val) < 0.001]
            if matching:
                lines.append(f"  {param_name}={val:<6.2f}  mean_score={np.mean(matching):+.4f}  "
                             f"n={len(matching)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not PROFILES_PATH.exists():
        print(f"ERROR: Profiles not found at {PROFILES_PATH}", file=sys.stderr)
        print("Run verdict_boundary_stress_test.py first.", file=sys.stderr)
        return 1

    raw = json.loads(PROFILES_PATH.read_text())
    profiles = raw if isinstance(raw, list) else raw.get("profiles", raw.get("data", []))

    # Filter to default-preset rows (these have the raw observables we re-assign)
    default_rows = [r for r in profiles if r.get("preset") == "default"]
    if not default_rows:
        print("ERROR: No default-preset rows found in profiles.", file=sys.stderr)
        return 1

    print(f"Loaded {len(default_rows)} default-preset layer profiles.")

    # Evaluate baseline (current defaults)
    baseline_t = Thresholds(
        low_overlap=LOW_OVERLAP,
        high_overlap=0.50,
        aligned=0.50,
        conflicting=-0.30,
        imbalanced=5.0,
        imbalanced_frob=5.0,
    )
    baseline = evaluate_objectives(default_rows, baseline_t)
    print(f"Baseline: Obj-1={baseline['obj1_cross_safe_rate']:.4f}, "
          f"Obj-2={baseline['obj2_same_bad_rate']:.4f}, "
          f"B5 frac={baseline['obj3_b5_fraction']:.4f}")

    # Run the sweep
    param_names = list(SWEEP_GRID.keys())
    param_values = [SWEEP_GRID[p] for p in param_names]
    combos = list(iterproduct(*param_values))
    print(f"Running sweep: {len(combos)} threshold combinations ...")

    results: list[dict[str, Any]] = []
    for combo in combos:
        kw = dict(zip(param_names, combo))
        t = Thresholds(low_overlap=LOW_OVERLAP, **kw)
        result = evaluate_objectives(default_rows, t)
        results.append(result)

    # Sort by composite score for the JSON output
    scored = [(r, composite_score(r, baseline)) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)
    for r, sc in scored:
        r["composite_score"] = sc

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sweep_json = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_combinations": len(results),
        "n_profiles": len(default_rows),
        "baseline": baseline,
        "results": [r for r, _ in scored],
    }
    (OUT_DIR / "sweep_results.json").write_text(json.dumps(sweep_json, indent=2, default=str))

    report = format_report(results, baseline)
    (OUT_DIR / "sweep_report.txt").write_text(report)

    print(f"\nSweep complete. {len(results)} configurations evaluated.")
    print(f"Outputs:")
    print(f"  {OUT_DIR / 'sweep_results.json'}")
    print(f"  {OUT_DIR / 'sweep_report.txt'}")

    # Print top 3 for quick feedback
    print(f"\nTop 3 configurations:")
    for i, (r, sc) in enumerate(scored[:3], 1):
        print(f"  #{i} score={sc:+.4f}  {r['thresholds']}")
        print(f"     Obj-1={r['obj1_cross_safe_rate']:.4f}  "
              f"Obj-2={r['obj2_same_bad_rate']:.4f}  "
              f"B5 frac={r['obj3_b5_fraction']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
