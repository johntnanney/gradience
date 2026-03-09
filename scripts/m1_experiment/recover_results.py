#!/usr/bin/env python3
"""
One-shot recovery script: map existing lm-eval results from subdirectory
structure to the flat JSON paths that phase4_evaluate.py and phase5_analyze.py
expect.

lm-eval writes results into directories like:
    evals/individual/__workspace__m1__adapters__chat__seed_42/results_2024-...json
    evals/merged/__workspace__m1__merges__chat_math__seed_42__linear_w0.5_0.5/results_2024-...json

This script scans for those results_*.json files, infers which eval they
belong to from the directory name, and copies them to the expected flat paths:
    evals/individual/chat_seed_42_mmlu.json
    evals/merged/chat_math_seed_42_linear_w0.5_0.5_mmlu.json

Usage:
    python scripts/m1_experiment/recover_results.py \\
        --config scripts/m1_experiment/m1_config.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _weight_label(weights: list[float]) -> str:
    return f"linear_w{weights[0]}_{weights[1]}"


def _find_results_in_dir(search_dir: Path) -> Path | None:
    """Find the most recent results_*.json in a directory tree."""
    results_files = sorted(search_dir.rglob("results_*.json"))
    if not results_files:
        return None
    return results_files[-1]


def _infer_eval_task_from_results(results_path: Path) -> str | None:
    """Read a results JSON and extract the task name from the results keys."""
    try:
        with open(results_path) as f:
            data = json.load(f)
        results = data.get("results", {})
        if results:
            # Return the first task name
            return list(results.keys())[0]
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Recover lm-eval results")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = Path(config["runtime"]["workspace"])
    evals_dir = workspace / "evals"
    adapters_dir = workspace / "adapters"
    merges_dir = workspace / "merges"

    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    weight_grid = config["merge"]["weight_grid"]
    calibration_enabled = config["experiment"].get("calibration_pairs", False)

    recovered = 0
    skipped = 0
    missing = 0

    # --- Part A: Individual adapter evals ---
    print("=== Recovering individual adapter evals ===")
    individual_dir = evals_dir / "individual"

    for task_name, task_cfg in config["adapters"].items():
        eval_task = task_cfg["eval_task"]
        for seed in seeds:
            expected_path = individual_dir / f"{task_name}_seed_{seed}_{eval_task}.json"
            if expected_path.exists():
                # Check if it's an error file we should overwrite
                try:
                    with open(expected_path) as f:
                        existing = json.load(f)
                    if not existing.get("error"):
                        print(f"  [OK]   {expected_path.name}")
                        skipped += 1
                        continue
                    print(f"  [ERR]  {expected_path.name} — has error, looking for real results")
                except Exception:
                    pass

            # Search in lm-eval subdirectories
            # Pattern: __workspace__m1__adapters__TASK__seed_SEED
            adapter_path = adapters_dir / task_name / f"seed_{seed}"
            encoded = str(adapter_path).replace("/", "__")
            # Try to find any subdir matching this adapter
            found = False
            for subdir in individual_dir.iterdir():
                if not subdir.is_dir():
                    continue
                if encoded in subdir.name or subdir.name.startswith("__"):
                    results_path = _find_results_in_dir(subdir)
                    if results_path:
                        # Verify the task matches
                        detected_task = _infer_eval_task_from_results(results_path)
                        if detected_task and eval_task not in detected_task:
                            continue

                        if args.dry_run:
                            print(f"  [COPY] {results_path} → {expected_path.name}")
                        else:
                            with open(results_path) as f:
                                data = json.load(f)
                            with open(expected_path, "w") as f:
                                json.dump(data, f, indent=2)
                            print(f"  [RECOVERED] {expected_path.name}")
                        recovered += 1
                        found = True
                        break

            if not found and not expected_path.exists():
                print(f"  [MISS] {expected_path.name}")
                missing += 1

    # --- Part B: Cross-task merged adapter evals ---
    print("\n=== Recovering cross-task merged evals ===")
    merged_eval_dir = evals_dir / "merged"
    pairs = list(itertools.combinations(task_names, 2))

    for task_a, task_b in pairs:
        pair_name = f"{task_a}_{task_b}"
        eval_task_a = config["adapters"][task_a]["eval_task"]
        eval_task_b = config["adapters"][task_b]["eval_task"]
        eval_tasks_for_pair = list(dict.fromkeys([eval_task_a, eval_task_b]))

        for seed in seeds:
            for weights in weight_grid:
                wlabel = _weight_label(weights)
                merge_dir = merges_dir / pair_name / f"seed_{seed}" / wlabel

                for eval_task in eval_tasks_for_pair:
                    expected_path = merged_eval_dir / f"{pair_name}_seed_{seed}_{wlabel}_{eval_task}.json"
                    if expected_path.exists():
                        try:
                            with open(expected_path) as f:
                                existing = json.load(f)
                            if not existing.get("error"):
                                skipped += 1
                                continue
                        except Exception:
                            pass

                    # Search in lm-eval subdirectories
                    encoded = str(merge_dir).replace("/", "__")
                    found = False
                    if merged_eval_dir.exists():
                        for subdir in merged_eval_dir.iterdir():
                            if not subdir.is_dir():
                                continue
                            if encoded in subdir.name or subdir.name.startswith("__"):
                                results_path = _find_results_in_dir(subdir)
                                if results_path:
                                    detected_task = _infer_eval_task_from_results(results_path)
                                    if detected_task and eval_task not in detected_task:
                                        continue

                                    if args.dry_run:
                                        print(f"  [COPY] {results_path.name} → {expected_path.name}")
                                    else:
                                        with open(results_path) as f:
                                            data = json.load(f)
                                        with open(expected_path, "w") as f:
                                            json.dump(data, f, indent=2)
                                        print(f"  [RECOVERED] {expected_path.name}")
                                    recovered += 1
                                    found = True
                                    break

                    if not found and not expected_path.exists():
                        missing += 1

    # --- Part C: Calibration evals ---
    if calibration_enabled:
        print("\n=== Recovering calibration evals ===")
        seed_pairs = list(itertools.combinations(seeds, 2))
        for task in task_names:
            eval_task = config["adapters"][task]["eval_task"]
            for seed_a, seed_b in seed_pairs:
                cal_dir_name = f"{task}_seeds_{seed_a}_{seed_b}"
                for weights in weight_grid:
                    wlabel = _weight_label(weights)
                    expected_path = merged_eval_dir / f"calibration_{cal_dir_name}_{wlabel}_{eval_task}.json"
                    if expected_path.exists():
                        try:
                            with open(expected_path) as f:
                                existing = json.load(f)
                            if not existing.get("error"):
                                skipped += 1
                                continue
                        except Exception:
                            pass

                    merge_dir = merges_dir / "calibration" / cal_dir_name / wlabel
                    encoded = str(merge_dir).replace("/", "__")
                    found = False
                    if merged_eval_dir.exists():
                        for subdir in merged_eval_dir.iterdir():
                            if not subdir.is_dir():
                                continue
                            if encoded in subdir.name:
                                results_path = _find_results_in_dir(subdir)
                                if results_path:
                                    if args.dry_run:
                                        print(f"  [COPY] {results_path.name} → {expected_path.name}")
                                    else:
                                        with open(results_path) as f:
                                            data = json.load(f)
                                        with open(expected_path, "w") as f:
                                            json.dump(data, f, indent=2)
                                        print(f"  [RECOVERED] {expected_path.name}")
                                    recovered += 1
                                    found = True
                                    break

                    if not found and not expected_path.exists():
                        missing += 1

    # --- Also do a brute-force scan ---
    print("\n=== Brute-force scan for unmapped results ===")
    all_subdirs = list(evals_dir.rglob("__*"))
    unmapped = 0
    for subdir in sorted(all_subdirs):
        if not subdir.is_dir():
            continue
        results_path = _find_results_in_dir(subdir)
        if results_path:
            detected_task = _infer_eval_task_from_results(results_path)
            # Check if this is already recovered
            relative = subdir.relative_to(evals_dir)
            print(f"  [SCAN] {relative} → task={detected_task}")
            unmapped += 1

    print("\n=== Summary ===")
    print(f"  Recovered:  {recovered}")
    print(f"  Skipped OK: {skipped}")
    print(f"  Missing:    {missing}")
    print(f"  Unmapped:   {unmapped}")


if __name__ == "__main__":
    main()
