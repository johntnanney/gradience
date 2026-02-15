#!/usr/bin/env python3
"""
Phase 2: Pairwise merge-audit (6 pairs x 3 seeds = 18 audits).

For each unique adapter pair and seed, runs gradience merge_audit()
to compute spectral compatibility metrics (principal angles, directional
agreement, magnitude balance).

Output: merge_audit.json per (pair, seed) in workspace/audits/.

Usage:
    python scripts/m1_experiment/phase2_audit.py \
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test:
    python scripts/m1_experiment/phase2_audit.py \
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

    # Generate all C(4,2) = 6 unique pairs
    pairs = list(itertools.combinations(task_names, 2))
    n_total = len(pairs) * len(seeds)

    total_start = time.monotonic()
    print(f"Phase 2: Running {n_total} pairwise merge-audits")
    print(f"  Pairs: {pairs}")
    print(f"  Seeds: {seeds}")

    n_done = 0
    for task_a, task_b in pairs:
        for seed in seeds:
            n_done += 1
            pair_name = f"{task_a}_{task_b}"
            audit_dir = audits_dir / pair_name / f"seed_{seed}"

            # Skip if already done
            if (audit_dir / "merge_audit.json").exists():
                print(f"  [{n_done}/{n_total}] [SKIP] {pair_name}/seed_{seed}")
                continue

            adapter_a = adapters_dir / task_a / f"seed_{seed}"
            adapter_b = adapters_dir / task_b / f"seed_{seed}"

            if not adapter_a.exists() or not adapter_b.exists():
                print(f"  [{n_done}/{n_total}] [MISSING] {pair_name}/seed_{seed}")
                continue

            print(f"  [{n_done}/{n_total}] Auditing {pair_name}/seed_{seed}...")
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

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 2 complete: {n_total} audits in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
