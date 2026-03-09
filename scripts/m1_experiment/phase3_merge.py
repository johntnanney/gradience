#!/usr/bin/env python3
"""
Phase 3: Execute merges.

(18 cross-task + 12 calibration) pairs x 1 method x 3 weights = 90 merges.

For each (pair, weight): generate a uniform_linear merge plan from the
Phase 2 audit report, then execute via the gradience merge engine.

Output: PEFT-compatible merged adapters in workspace/merges/.

Usage:
    python3 scripts/m1_experiment/phase3_merge.py \\
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test:
    python3 scripts/m1_experiment/phase3_merge.py \\
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
    from gradience.vnext.merge.containers import AdapterMetadata, AggregateResult, MatchingSummary

    return MergeAuditReport(
        adapter_a=AdapterMetadata.from_dict(data["adapter_a"]),
        adapter_b=AdapterMetadata.from_dict(data["adapter_b"]),
        matching=MatchingSummary.from_dict(data["matching"]),
        layer_verdicts=data.get("per_layer", data.get("layer_verdicts", [])),
        aggregate=AggregateResult.from_dict(data["aggregate"]),
        recommendations=data.get("recommendations", []),
        issues=data.get("issues", []),
        warnings=data.get("warnings", []),
        thresholds=data.get("thresholds", {}),
    )


def _weight_label(weights: list[float]) -> str:
    """Format weight pair as directory-safe label, e.g. 'linear_w0.5_0.5'."""
    return f"linear_w{weights[0]}_{weights[1]}"


def _run_merge(
    report: MergeAuditReport,
    adapter_a: str,
    adapter_b: str,
    weights: list[float],
    merge_dir: Path,
    merge_config: dict,
    label: str,
    counter: str,
) -> None:
    """Execute a single merge with progress logging."""
    # Skip if already done
    if (merge_dir / "adapter_config.json").exists():
        print(f"  [{counter}] [SKIP] {label}")
        return

    print(f"  [{counter}] Merging {label}...")
    start = time.monotonic()

    plan = plan_from_audit(
        "uniform_linear",
        report,
        adapter_a,
        adapter_b,
        coefficients=tuple(weights),
        output_rank=merge_config["output_rank"],
        output_alpha=float(merge_config["output_rank"]),
    )

    result = execute_merge(plan, merge_dir, verbose=False)

    elapsed = time.monotonic() - start
    print(f"    recon_error={result.mean_reconstruction_error:.4f} [{elapsed:.1f}s]")


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
    merge_config = config["merge"]
    weight_grid = merge_config["weight_grid"]
    calibration_enabled = config["experiment"].get("calibration_pairs", False)

    # --- Cross-task pairs ---
    pairs = list(itertools.combinations(task_names, 2))
    n_cross = len(pairs) * len(seeds) * len(weight_grid)

    # --- Calibration pairs ---
    seed_pairs = list(itertools.combinations(seeds, 2))
    n_calibration = len(task_names) * len(seed_pairs) * len(weight_grid) if calibration_enabled else 0

    n_total = n_cross + n_calibration

    total_start = time.monotonic()
    print(f"Phase 3: Executing {n_total} merges")
    print(f"  Cross-task: {len(pairs)} pairs x {len(seeds)} seeds x {len(weight_grid)} weights = {n_cross}")
    if calibration_enabled:
        print(
            f"  Calibration: {len(task_names)} tasks x {len(seed_pairs)} seed-pairs x {len(weight_grid)} weights = {n_calibration}"
        )
    print(f"  Weight grid: {weight_grid}")

    n_done = 0

    # --- Cross-task merges ---
    for task_a, task_b in pairs:
        for seed in seeds:
            pair_name = f"{task_a}_{task_b}"
            audit_dir = audits_dir / pair_name / f"seed_{seed}"

            if not (audit_dir / "merge_audit.json").exists():
                print(f"  [MISSING AUDIT] {pair_name}/seed_{seed} -- skipping all weights")
                n_done += len(weight_grid)
                continue

            report = load_audit_report(audit_dir)
            adapter_a = str(adapters_dir / task_a / f"seed_{seed}")
            adapter_b = str(adapters_dir / task_b / f"seed_{seed}")

            for weights in weight_grid:
                n_done += 1
                wlabel = _weight_label(weights)
                merge_dir = merges_dir / pair_name / f"seed_{seed}" / wlabel

                _run_merge(
                    report,
                    adapter_a,
                    adapter_b,
                    weights,
                    merge_dir,
                    merge_config,
                    label=f"{pair_name}/seed_{seed}/{wlabel}",
                    counter=f"{n_done}/{n_total}",
                )

    # --- Calibration merges ---
    if calibration_enabled:
        for task in task_names:
            for seed_a, seed_b in seed_pairs:
                cal_dir_name = f"{task}_seeds_{seed_a}_{seed_b}"
                audit_dir = audits_dir / "calibration" / cal_dir_name

                if not (audit_dir / "merge_audit.json").exists():
                    print(f"  [MISSING AUDIT] calibration/{cal_dir_name} -- skipping all weights")
                    n_done += len(weight_grid)
                    continue

                report = load_audit_report(audit_dir)
                adapter_a = str(adapters_dir / task / f"seed_{seed_a}")
                adapter_b = str(adapters_dir / task / f"seed_{seed_b}")

                for weights in weight_grid:
                    n_done += 1
                    wlabel = _weight_label(weights)
                    merge_dir = merges_dir / "calibration" / cal_dir_name / wlabel

                    _run_merge(
                        report,
                        adapter_a,
                        adapter_b,
                        weights,
                        merge_dir,
                        merge_config,
                        label=f"calibration/{cal_dir_name}/{wlabel}",
                        counter=f"{n_done}/{n_total}",
                    )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 3 complete: {n_total} merges in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
