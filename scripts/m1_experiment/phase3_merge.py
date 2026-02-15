#!/usr/bin/env python3
"""
Phase 3: Execute merges (18 pairs x 4 methods = 72 merges).

For each (pair, seed, method): generate a merge plan from the Phase 2
audit report, then execute via the gradience merge engine.

Output: PEFT-compatible merged adapters in workspace/merges/.

Usage:
    python scripts/m1_experiment/phase3_merge.py \\
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test:
    python scripts/m1_experiment/phase3_merge.py \\
        --config scripts/m1_experiment/m1_config.yaml --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import yaml

from gradience.vnext.merge import execute_merge, plan_from_audit
from gradience.vnext.merge.report import MergeAuditReport


def load_config(config_path: str, smoke: bool = False) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])
    return config


def load_audit_report(audit_dir: Path) -> MergeAuditReport:
    """Load a merge audit report from JSON."""
    report_path = audit_dir / "merge_audit.json"
    with open(report_path) as f:
        data = json.load(f)
    return MergeAuditReport(
        adapter_a=data["adapter_a"],
        adapter_b=data["adapter_b"],
        matching=data["matching"],
        layer_verdicts=data["layer_verdicts"],
        aggregate=data["aggregate"],
    )


def get_plan_kwargs(method: str, merge_config: dict) -> dict:
    """Build plan_from_audit kwargs for a given method."""
    output_rank = merge_config["output_rank"]
    coefficients = tuple(merge_config.get("linear_coefficients", [0.5, 0.5]))

    base = {"output_rank": output_rank, "output_alpha": float(output_rank)}

    if method == "linear":
        return {**base, "coefficients": coefficients}
    elif method == "ties":
        return {**base, "trim_fraction": merge_config.get("ties_density", 0.5)}
    elif method == "dare_linear":
        return {**base, "coefficients": coefficients,
                "dare_drop_fraction": 1.0 - merge_config.get("dare_linear_density", 0.7)}
    elif method == "dare_ties":
        return {**base, "coefficients": coefficients,
                "dare_drop_fraction": 1.0 - merge_config.get("dare_ties_density", 0.5)}
    else:
        return base


# Map from m1_config method names to plan strategy names
METHOD_TO_PLAN = {
    "linear": "uniform_linear",
    "ties": "overlap_ties",
    "dare_linear": "dare_linear",
    "dare_ties": "dare_ties",
}


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 3: Execute merges")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (1 seed)")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    audits_dir = workspace / "audits"
    merges_dir = workspace / "merges"

    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    methods = config["merge"]["methods"]
    merge_config = config["merge"]

    pairs = list(itertools.combinations(task_names, 2))
    n_total = len(pairs) * len(seeds) * len(methods)

    total_start = time.monotonic()
    print(f"Phase 3: Executing {n_total} merges")
    print(f"  Pairs: {len(pairs)}, Seeds: {len(seeds)}, Methods: {methods}")

    n_done = 0
    for task_a, task_b in pairs:
        for seed in seeds:
            pair_name = f"{task_a}_{task_b}"
            audit_dir = audits_dir / pair_name / f"seed_{seed}"

            if not (audit_dir / "merge_audit.json").exists():
                print(f"  [MISSING AUDIT] {pair_name}/seed_{seed} -- skipping all methods")
                n_done += len(methods)
                continue

            report = load_audit_report(audit_dir)

            adapter_a = str(adapters_dir / task_a / f"seed_{seed}")
            adapter_b = str(adapters_dir / task_b / f"seed_{seed}")

            for method in methods:
                n_done += 1
                merge_dir = merges_dir / pair_name / f"seed_{seed}" / method

                # Skip if already done
                if (merge_dir / "adapter_config.json").exists():
                    print(f"  [{n_done}/{n_total}] [SKIP] {pair_name}/seed_{seed}/{method}")
                    continue

                print(f"  [{n_done}/{n_total}] Merging {pair_name}/seed_{seed}/{method}...")
                start = time.monotonic()

                plan_strategy = METHOD_TO_PLAN[method]
                plan_kwargs = get_plan_kwargs(method, merge_config)

                plan = plan_from_audit(
                    plan_strategy,
                    report,
                    adapter_a,
                    adapter_b,
                    **plan_kwargs,
                )

                result = execute_merge(plan, merge_dir, verbose=False)

                elapsed = time.monotonic() - start
                print(
                    f"    recon_error={result.mean_reconstruction_error:.4f} "
                    f"[{elapsed:.1f}s]"
                )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 3 complete: {n_total} merges in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
