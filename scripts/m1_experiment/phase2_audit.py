#!/usr/bin/env python3
"""
Phase 2: Pairwise merge-audit.

Cross-task audits: 6 pairs x 3 seeds = 18 audits.
Calibration audits: 4 tasks x 3 seed-pairs = 12 audits (same-task, different seeds).
Total: 30 audits.

For each unique adapter pair and seed, runs gradience merge_audit()
to compute spectral compatibility metrics (principal angles, directional
agreement, magnitude balance).

Output: merge_audit.json per (pair, seed) in workspace/audits/.

Usage:
    python3 scripts/m1_experiment/phase2_audit.py \
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test:
    python3 scripts/m1_experiment/phase2_audit.py \
        --config scripts/m1_experiment/m1_config.yaml --smoke
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import yaml

from gradience.vnext.merge import merge_audit


def load_config(config_path: str, smoke: bool = False) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])
    return config


def _run_audit(
    adapter_a: Path,
    adapter_b: Path,
    audit_dir: Path,
    label: str,
    counter: str,
) -> None:
    """Run a single merge audit with progress logging."""
    # Skip if already done
    if (audit_dir / "merge_audit.json").exists():
        print(f"  [{counter}] [SKIP] {label}")
        return

    if not adapter_a.exists() or not adapter_b.exists():
        print(f"  [{counter}] [MISSING] {label}")
        return

    print(f"  [{counter}] Auditing {label}...")
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
    print(f"    Verdict: {verdict} (score={score:.3f}) [{elapsed:.1f}s]")


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 2: Pairwise merge-audit")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (1 seed)")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    audits_dir = workspace / "audits"

    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    calibration_enabled = config["experiment"].get("calibration_pairs", False)

    # --- Cross-task audits: C(4,2) = 6 unique pairs x seeds ---
    pairs = list(itertools.combinations(task_names, 2))
    n_cross = len(pairs) * len(seeds)

    # --- Calibration audits: same-task, different-seed pairs ---
    seed_pairs = list(itertools.combinations(seeds, 2))
    n_calibration = len(task_names) * len(seed_pairs) if calibration_enabled else 0

    n_total = n_cross + n_calibration

    total_start = time.monotonic()
    print(f"Phase 2: Running {n_total} pairwise merge-audits")
    print(f"  Cross-task pairs: {len(pairs)} x {len(seeds)} seeds = {n_cross}")
    if calibration_enabled:
        print(f"  Calibration pairs: {len(task_names)} tasks x {len(seed_pairs)} seed-pairs = {n_calibration}")
    print(f"  Seeds: {seeds}")

    n_done = 0

    # --- Cross-task audits ---
    for task_a, task_b in pairs:
        for seed in seeds:
            n_done += 1
            pair_name = f"{task_a}_{task_b}"
            audit_dir = audits_dir / pair_name / f"seed_{seed}"
            adapter_a = adapters_dir / task_a / f"seed_{seed}"
            adapter_b = adapters_dir / task_b / f"seed_{seed}"

            _run_audit(
                adapter_a, adapter_b, audit_dir,
                label=f"{pair_name}/seed_{seed}",
                counter=f"{n_done}/{n_total}",
            )

    # --- Same-task calibration audits ---
    if calibration_enabled:
        for task in task_names:
            for seed_a, seed_b in seed_pairs:
                n_done += 1
                cal_dir_name = f"{task}_seeds_{seed_a}_{seed_b}"
                audit_dir = audits_dir / "calibration" / cal_dir_name
                adapter_a = adapters_dir / task / f"seed_{seed_a}"
                adapter_b = adapters_dir / task / f"seed_{seed_b}"

                _run_audit(
                    adapter_a, adapter_b, audit_dir,
                    label=f"calibration/{cal_dir_name}",
                    counter=f"{n_done}/{n_total}",
                )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 2 complete: {n_total} audits in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
