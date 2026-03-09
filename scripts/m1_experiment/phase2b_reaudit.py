#!/usr/bin/env python3
"""
Phase 2b: Re-audit all pairs with updated verdict engine.

Runs the same pairwise merge-audit as Phase 2, but:
  1. Overwrites existing audit results (--force is the default).
  2. Generates merge recommendations via recommend_merge() for each pair.
  3. Produces a verdict-distribution summary comparing old → new verdicts.
  4. Writes a combined JSON + markdown report of the full re-audit.

This script exists because the verdict decision tree changed (Frobenius
imbalance detection added as Branch 0), so all prior audit results are
stale.  Re-running on CPU is cheap (~seconds per pair) and gives us a
clean baseline before committing GPU time to the Phase 3b ablation.

Output structure:
    workspace/audits_v2/
        {pair_name}/seed_{seed}/
            merge_audit.json
            merge_audit.md
            recommendation.json
        calibration/
            {task}_seeds_{sa}_{sb}/
                merge_audit.json
                merge_audit.md
                recommendation.json
        _summary/
            reaudit_summary.json
            reaudit_summary.md
            verdict_migration.json   # old → new per-pair comparison

Usage:
    python3 scripts/m1_experiment/phase2b_reaudit.py \
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test (single seed):
    python3 scripts/m1_experiment/phase2b_reaudit.py \
        --config scripts/m1_experiment/m1_config.yaml --smoke

    # Skip recommendation generation (audit only):
    python3 scripts/m1_experiment/phase2b_reaudit.py \
        --config scripts/m1_experiment/m1_config.yaml --no-recommend

    # Compare against original Phase 2 audits:
    python3 scripts/m1_experiment/phase2b_reaudit.py \
        --config scripts/m1_experiment/m1_config.yaml --diff
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from gradience.vnext.merge import merge_audit, recommend_merge

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: str, smoke: bool = False) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])
    return config


# ---------------------------------------------------------------------------
# Data structures for summary tracking
# ---------------------------------------------------------------------------


@dataclass
class PairAuditResult:
    """Compact record of one audit for summary reporting."""

    pair_name: str
    pair_type: str  # "cross_task" or "calibration"
    seed_label: str  # "seed_42" or "seeds_42_123"
    overall_verdict: str
    compatibility_score: float
    n_layers: int
    verdict_counts: dict[str, int]  # per-layer verdict distribution
    recommendation_strategy: str | None = None
    recommendation_risk: str | None = None
    n_compress: int = 0
    audit_time_s: float = 0.0
    # For diff mode: what the old verdict was
    old_verdict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pair_name": self.pair_name,
            "pair_type": self.pair_type,
            "seed_label": self.seed_label,
            "overall_verdict": self.overall_verdict,
            "compatibility_score": round(self.compatibility_score, 4),
            "n_layers": self.n_layers,
            "verdict_counts": self.verdict_counts,
            "audit_time_s": round(self.audit_time_s, 2),
        }
        if self.recommendation_strategy is not None:
            d["recommendation"] = {
                "overall_strategy": self.recommendation_strategy,
                "overall_risk": self.recommendation_risk,
                "n_layers_needing_compression": self.n_compress,
            }
        if self.old_verdict is not None:
            d["old_verdict"] = self.old_verdict
            d["verdict_changed"] = self.old_verdict != self.overall_verdict
        return d


# ---------------------------------------------------------------------------
# Core audit + recommend
# ---------------------------------------------------------------------------


def _run_audit_and_recommend(
    adapter_a: Path,
    adapter_b: Path,
    audit_dir: Path,
    label: str,
    counter: str,
    do_recommend: bool = True,
    old_audits_dir: Path | None = None,
) -> PairAuditResult | None:
    """Run audit, optionally generate recommendation, return summary record."""

    if not adapter_a.exists():
        print(f"  [{counter}] [MISSING] {label} — adapter A not found: {adapter_a}")
        return None
    if not adapter_b.exists():
        print(f"  [{counter}] [MISSING] {label} — adapter B not found: {adapter_b}")
        return None

    print(f"  [{counter}] Auditing {label}...", end="", flush=True)
    start = time.monotonic()

    report = merge_audit(
        adapter_a_dir=adapter_a,
        adapter_b_dir=adapter_b,
        output_dir=audit_dir,
        verbose=False,
    )

    elapsed = time.monotonic() - start
    verdict = report.aggregate.get("overall_verdict", "unknown")
    score = report.aggregate.get("compatibility_score", 0.0)

    # Per-layer verdict counts
    verdict_counts: dict[str, int] = Counter()
    for lv in report.layer_verdicts:
        verdict_counts[lv["verdict"]] += 1

    print(f" {verdict.upper()} (score={score:.3f}) [{elapsed:.1f}s]", end="")

    # Recommendation
    rec_strategy = None
    rec_risk = None
    n_compress = 0

    if do_recommend:
        try:
            rec = recommend_merge(report)
            rec_strategy = rec.overall_strategy
            rec_risk = rec.overall_risk
            n_compress = rec.n_layers_needing_compression

            # Write recommendation JSON
            rec_path = audit_dir / "recommendation.json"
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rec_path, "w") as f:
                json.dump(rec.to_dict(), f, indent=2)

            print(f" → rec: {rec_strategy} (risk={rec_risk})", end="")
        except Exception as e:
            print(f" → rec: FAILED ({e})", end="")

    # Check old verdict for diff
    old_verdict = None
    if old_audits_dir is not None:
        old_json = old_audits_dir / "merge_audit.json"
        if old_json.exists():
            try:
                with open(old_json) as f:
                    old_data = json.load(f)
                old_verdict = old_data.get("aggregate", {}).get("overall_verdict")
                if old_verdict and old_verdict != verdict:
                    print(f"  [CHANGED: {old_verdict} → {verdict}]", end="")
            except (json.JSONDecodeError, KeyError):
                pass

    print()  # newline

    return PairAuditResult(
        pair_name=label.split("/")[0] if "/" in label else label,
        pair_type="calibration" if "calibration" in label else "cross_task",
        seed_label=label.split("/")[-1] if "/" in label else label,
        overall_verdict=verdict,
        compatibility_score=score,
        n_layers=len(report.layer_verdicts),
        verdict_counts=dict(verdict_counts),
        recommendation_strategy=rec_strategy,
        recommendation_risk=rec_risk,
        n_compress=n_compress,
        audit_time_s=elapsed,
        old_verdict=old_verdict,
    )


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def _write_summary(
    results: list[PairAuditResult],
    summary_dir: Path,
    has_diff: bool,
) -> None:
    """Write JSON and markdown summaries of all audits."""
    summary_dir.mkdir(parents=True, exist_ok=True)

    # --- Verdict distribution ---
    overall_verdicts = Counter(r.overall_verdict for r in results)
    layer_verdicts_total: Counter = Counter()
    for r in results:
        layer_verdicts_total.update(r.verdict_counts)

    # --- Recommendation distribution ---
    rec_strategies = Counter(r.recommendation_strategy for r in results if r.recommendation_strategy is not None)
    rec_risks = Counter(r.recommendation_risk for r in results if r.recommendation_risk is not None)

    # --- Summary JSON ---
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_audits": len(results),
        "n_cross_task": sum(1 for r in results if r.pair_type == "cross_task"),
        "n_calibration": sum(1 for r in results if r.pair_type == "calibration"),
        "overall_verdict_distribution": dict(overall_verdicts),
        "layer_verdict_distribution": dict(layer_verdicts_total),
        "recommendation_strategy_distribution": dict(rec_strategies),
        "recommendation_risk_distribution": dict(rec_risks),
        "n_needing_compression": sum(1 for r in results if r.n_compress > 0),
        "mean_audit_time_s": round(sum(r.audit_time_s for r in results) / max(len(results), 1), 2),
        "per_pair": [r.to_dict() for r in results],
    }

    with open(summary_dir / "reaudit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Verdict migration (diff) ---
    if has_diff:
        migrations: list[dict[str, Any]] = []
        for r in results:
            if r.old_verdict is not None:
                migrations.append(
                    {
                        "pair": r.pair_name,
                        "seed": r.seed_label,
                        "old_verdict": r.old_verdict,
                        "new_verdict": r.overall_verdict,
                        "changed": r.old_verdict != r.overall_verdict,
                    }
                )

        migration_summary = {
            "n_compared": len(migrations),
            "n_changed": sum(1 for m in migrations if m["changed"]),
            "n_unchanged": sum(1 for m in migrations if not m["changed"]),
            "changes": [m for m in migrations if m["changed"]],
            "all": migrations,
        }

        with open(summary_dir / "verdict_migration.json", "w") as f:
            json.dump(migration_summary, f, indent=2)

    # --- Markdown summary ---
    lines = [
        "# Phase 2b: Re-Audit Summary",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        f"Total audits: **{len(results)}** "
        f"({summary['n_cross_task']} cross-task, "
        f"{summary['n_calibration']} calibration)",
        "",
        "## Overall Verdict Distribution",
        "",
        "| Verdict | Count | % |",
        "|---------|------:|--:|",
    ]

    for v in ["safe", "redundant", "imbalanced", "conflicting"]:
        count = overall_verdicts.get(v, 0)
        pct = 100 * count / max(len(results), 1)
        lines.append(f"| {v.upper()} | {count} | {pct:.0f}% |")

    lines.extend(
        [
            "",
            "## Per-Layer Verdict Distribution",
            "",
            "| Verdict | Layers | % |",
            "|---------|-------:|--:|",
        ]
    )

    total_layers = sum(layer_verdicts_total.values())
    for v in ["safe", "redundant", "imbalanced", "conflicting"]:
        count = layer_verdicts_total.get(v, 0)
        pct = 100 * count / max(total_layers, 1)
        lines.append(f"| {v.upper()} | {count} | {pct:.0f}% |")

    if rec_strategies:
        lines.extend(
            [
                "",
                "## Recommendation Distribution",
                "",
                "| Strategy | Count |",
                "|----------|------:|",
            ]
        )
        for strat, count in rec_strategies.most_common():
            lines.append(f"| {strat} | {count} |")

        lines.extend(
            [
                "",
                "| Risk Level | Count |",
                "|------------|------:|",
            ]
        )
        for risk, count in rec_risks.most_common():
            lines.append(f"| {risk} | {count} |")

        compress_count = sum(1 for r in results if r.n_compress > 0)
        lines.extend(
            [
                "",
                f"Pairs needing pre-compression: **{compress_count}** / {len(results)}",
            ]
        )

    if has_diff:
        changed = [r for r in results if r.old_verdict is not None and r.old_verdict != r.overall_verdict]
        unchanged = [r for r in results if r.old_verdict is not None and r.old_verdict == r.overall_verdict]
        compared = [r for r in results if r.old_verdict is not None]

        lines.extend(
            [
                "",
                "## Verdict Changes (vs. Phase 2)",
                "",
                f"Compared: {len(compared)} pairs | Changed: **{len(changed)}** | Unchanged: {len(unchanged)}",
            ]
        )

        if changed:
            lines.extend(
                [
                    "",
                    "| Pair | Seed | Old | New |",
                    "|------|------|-----|-----|",
                ]
            )
            for r in changed:
                lines.append(f"| {r.pair_name} | {r.seed_label} | {r.old_verdict} | **{r.overall_verdict}** |")

    # Per-pair detail table
    lines.extend(
        [
            "",
            "## Per-Pair Results",
            "",
            "| Pair | Seed | Verdict | Score | Layers | Rec. Strategy | Risk |",
            "|------|------|---------|------:|-------:|---------------|------|",
        ]
    )
    for r in results:
        rec_s = r.recommendation_strategy or "—"
        rec_r = r.recommendation_risk or "—"
        lines.append(
            f"| {r.pair_name} | {r.seed_label} | "
            f"{r.overall_verdict} | {r.compatibility_score:.3f} | "
            f"{r.n_layers} | {rec_s} | {rec_r} |"
        )

    lines.append("")

    with open(summary_dir / "reaudit_summary.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Summary written to {summary_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 2b: Re-audit with updated verdict engine")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to m1_config.yaml",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test (single seed)",
    )
    parser.add_argument(
        "--no-recommend",
        action="store_true",
        help="Skip recommendation generation (audit only)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Compare against original Phase 2 audits and report changes",
    )
    parser.add_argument(
        "--output-suffix",
        default="v2",
        help="Suffix for output directory (default: 'v2' → audits_v2/)",
    )
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    audits_dir = workspace / f"audits_{args.output_suffix}"
    old_audits_dir = workspace / "audits" if args.diff else None

    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    calibration_enabled = config["experiment"].get("calibration_pairs", False)
    do_recommend = not args.no_recommend

    # --- Count totals ---
    pairs = list(itertools.combinations(task_names, 2))
    n_cross = len(pairs) * len(seeds)

    seed_pairs = list(itertools.combinations(seeds, 2))
    n_calibration = len(task_names) * len(seed_pairs) if calibration_enabled else 0

    n_total = n_cross + n_calibration

    print(f"Phase 2b: Re-auditing {n_total} pairs with updated verdict engine")
    print(f"  Output: {audits_dir}")
    print(f"  Cross-task pairs: {len(pairs)} x {len(seeds)} seeds = {n_cross}")
    if calibration_enabled:
        print(f"  Calibration pairs: {len(task_names)} tasks x {len(seed_pairs)} seed-pairs = {n_calibration}")
    print(f"  Seeds: {seeds}")
    print(f"  Recommendations: {'yes' if do_recommend else 'no'}")
    print(f"  Diff vs Phase 2: {'yes' if args.diff else 'no'}")
    print()

    total_start = time.monotonic()
    results: list[PairAuditResult] = []
    n_done = 0
    n_missing = 0

    # --- Cross-task audits ---
    for task_a, task_b in pairs:
        for seed in seeds:
            n_done += 1
            pair_name = f"{task_a}_{task_b}"
            audit_dir = audits_dir / pair_name / f"seed_{seed}"
            adapter_a = adapters_dir / task_a / f"seed_{seed}"
            adapter_b = adapters_dir / task_b / f"seed_{seed}"
            label = f"{pair_name}/seed_{seed}"

            old_dir = None
            if old_audits_dir:
                old_dir = old_audits_dir / pair_name / f"seed_{seed}"

            result = _run_audit_and_recommend(
                adapter_a,
                adapter_b,
                audit_dir,
                label=label,
                counter=f"{n_done}/{n_total}",
                do_recommend=do_recommend,
                old_audits_dir=old_dir,
            )
            if result:
                results.append(result)
            else:
                n_missing += 1

    # --- Same-task calibration audits ---
    if calibration_enabled:
        for task in task_names:
            for seed_a, seed_b in seed_pairs:
                n_done += 1
                cal_name = f"{task}_seeds_{seed_a}_{seed_b}"
                audit_dir = audits_dir / "calibration" / cal_name
                adapter_a = adapters_dir / task / f"seed_{seed_a}"
                adapter_b = adapters_dir / task / f"seed_{seed_b}"
                label = f"calibration/{cal_name}"

                old_dir = None
                if old_audits_dir:
                    old_dir = old_audits_dir / "calibration" / cal_name

                result = _run_audit_and_recommend(
                    adapter_a,
                    adapter_b,
                    audit_dir,
                    label=label,
                    counter=f"{n_done}/{n_total}",
                    do_recommend=do_recommend,
                    old_audits_dir=old_dir,
                )
                if result:
                    results.append(result)
                else:
                    n_missing += 1

    # --- Summary ---
    elapsed = time.monotonic() - total_start

    print(f"\n{'=' * 60}")
    print(f"Phase 2b complete: {len(results)} audits in {elapsed:.1f}s")
    if n_missing:
        print(f"  Skipped (missing adapters): {n_missing}")

    if results:
        # Quick verdict summary
        verdict_counts = Counter(r.overall_verdict for r in results)
        print("\n  Verdict distribution:")
        for v in ["safe", "redundant", "imbalanced", "conflicting"]:
            count = verdict_counts.get(v, 0)
            pct = 100 * count / len(results)
            print(f"    {v.upper():>12s}: {count:3d} ({pct:.0f}%)")

        if args.diff:
            changed = [r for r in results if r.old_verdict is not None and r.old_verdict != r.overall_verdict]
            compared = [r for r in results if r.old_verdict is not None]
            print(f"\n  Verdict changes: {len(changed)} / {len(compared)} pairs changed")
            for r in changed:
                print(f"    {r.pair_name}/{r.seed_label}: {r.old_verdict} → {r.overall_verdict}")

        # Write full summary
        summary_dir = audits_dir / "_summary"
        _write_summary(results, summary_dir, has_diff=args.diff)

    print(f"\nDone. Audits in: {audits_dir}")


if __name__ == "__main__":
    main()
