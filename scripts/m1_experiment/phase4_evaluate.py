#!/usr/bin/env python3
"""
Phase 4: Evaluate all adapters via lm-evaluation-harness.

Evaluates:
  - 9 individual adapters (each on its own task)
  - Cross-task merged adapters (on both constituent tasks)
  - Calibration merged adapters (on their shared task)

Output: JSON results per adapter in workspace/evals/.

Usage:
    python scripts/m1_experiment/phase4_evaluate.py \\
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test:
    python scripts/m1_experiment/phase4_evaluate.py \\
        --config scripts/m1_experiment/m1_config.yaml --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config(config_path: str, smoke: bool = False) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])
        config["evaluation"]["max_eval_samples"] = smoke_cfg.get("max_eval_samples", 10)
    return config


def _weight_label(weights: list[float]) -> str:
    """Format weight pair as directory-safe label, e.g. 'linear_w0.5_0.5'."""
    return f"linear_w{weights[0]}_{weights[1]}"


def _find_lm_eval_results(search_dir: Path) -> dict | None:
    """Search for lm-eval results_*.json in a directory tree.

    lm-eval writes results into subdirectories named after the model args
    (with path separators replaced by __). This function searches recursively
    for the most recent results file.
    """
    results_files = sorted(search_dir.rglob("results_*.json"))
    if not results_files:
        return None
    # Use the most recent one
    with open(results_files[-1]) as f:
        return json.load(f)


def run_lm_eval(
    base_model: str,
    adapter_dir: str | None,
    task: str,
    output_path: Path,
    max_samples: int = 500,
    device: str = "cuda",
) -> dict | None:
    """Run lm-evaluation-harness for a single (adapter, task) combo.

    Returns parsed results dict, or None on failure.
    """
    if output_path.exists():
        print(f"      [SKIP] {output_path.name} exists")
        with open(output_path) as f:
            return json.load(f)

    # Use a dedicated temp directory for lm-eval output to avoid clutter
    # lm-eval creates subdirs based on model args, so we give it a clean dir
    lm_eval_out = output_path.parent / f".lm_eval_tmp_{output_path.stem}"
    lm_eval_out.mkdir(parents=True, exist_ok=True)

    # Check if results already exist from a previous run in the temp dir
    existing = _find_lm_eval_results(lm_eval_out)
    if existing and not existing.get("error"):
        print(f"      [RECOVERED] {output_path.name} from {lm_eval_out.name}")
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)
        return existing

    # Build lm_eval command
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={base_model}",
        "--tasks", task,
        "--batch_size", "auto",
        "--device", device,
        "--output_path", str(lm_eval_out),
        "--log_samples",
        "--confirm_run_unsafe_code",
    ]

    if adapter_dir:
        # Add PEFT adapter
        cmd[cmd.index("--model_args") + 1] += f",peft={adapter_dir}"

    if max_samples > 0:
        cmd.extend(["--limit", str(max_samples)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per eval
        )

        if result.returncode != 0:
            print(f"      [ERROR] lm_eval failed: {result.stderr[-500:]}")
            # Save error info
            error_data = {
                "error": True,
                "returncode": result.returncode,
                "stderr": result.stderr[-2000:],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(error_data, f, indent=2)
            return error_data

        # Find results in lm-eval's output tree
        results = _find_lm_eval_results(lm_eval_out)
        if results:
            # Save a normalized copy at our expected path
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            return results
        else:
            print(f"      [WARN] No results_*.json found in {lm_eval_out}")

    except subprocess.TimeoutExpired:
        print(f"      [TIMEOUT] lm_eval timed out")
        return {"error": True, "reason": "timeout"}
    except Exception as e:
        print(f"      [ERROR] {e}")
        return {"error": True, "reason": str(e)}

    return None


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 4: Evaluate adapters")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test")
    parser.add_argument(
        "--recover", action="store_true",
        help="Recover results from a previous lm-eval run (scans for results_*.json)",
    )
    args = parser.parse_args()

    # Allow lm-eval to execute generated code for pass@1 benchmarks (mbpp, humaneval).
    # Without this, code eval tasks fail with a safety disclaimer.
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    merges_dir = workspace / "merges"
    evals_dir = workspace / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    base_model = config["experiment"]["base_model"]
    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    weight_grid = config["merge"]["weight_grid"]
    max_samples = config["evaluation"]["max_eval_samples"]
    device = config["runtime"]["device"]
    calibration_enabled = config["experiment"].get("calibration_pairs", False)

    # --- Recovery mode: salvage results from previous lm-eval runs ---
    if args.recover:
        print("Recovery mode: scanning for existing lm-eval results...")
        recovered = 0
        for results_file in evals_dir.rglob("results_*.json"):
            # Already in a proper location, skip
            pass

        # Scan for lm-eval output directories (pattern: __path__segments__)
        for subdir in sorted(evals_dir.rglob("__*")):
            if not subdir.is_dir():
                continue
            results = _find_lm_eval_results(subdir)
            if results is None:
                continue
            # The directory name encodes the adapter path
            # We can't perfectly reverse-map, but we print what we find
            print(f"  Found results in: {subdir.name}")
            recovered += 1

        print(f"  Found {recovered} result directories")
        print("  Run without --recover to re-evaluate missing items")
        return

    total_start = time.monotonic()

    # --- Part A: Evaluate individual adapters on their own task ---
    print("Phase 4a: Evaluating individual adapters")
    individual_dir = evals_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for task_name, task_cfg in config["adapters"].items():
        eval_task = task_cfg["eval_task"]
        for seed in seeds:
            adapter_dir = adapters_dir / task_name / f"seed_{seed}"
            if not adapter_dir.exists():
                print(f"  [MISSING] {task_name}/seed_{seed}")
                continue

            output_path = individual_dir / f"{task_name}_seed_{seed}_{eval_task}.json"
            print(f"  Evaluating {task_name}/seed_{seed} on {eval_task}...")
            run_lm_eval(
                base_model=base_model,
                adapter_dir=str(adapter_dir),
                task=eval_task,
                output_path=output_path,
                max_samples=max_samples,
                device=device,
            )

    # --- Part B: Evaluate cross-task merged adapters on both constituent tasks ---
    print("\nPhase 4b: Evaluating cross-task merged adapters")
    merged_eval_dir = evals_dir / "merged"
    merged_eval_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(itertools.combinations(task_names, 2))

    for task_a, task_b in pairs:
        pair_name = f"{task_a}_{task_b}"
        eval_task_a = config["adapters"][task_a]["eval_task"]
        eval_task_b = config["adapters"][task_b]["eval_task"]

        # Evaluate merged adapter on both constituent tasks
        eval_tasks_for_pair = list(dict.fromkeys([eval_task_a, eval_task_b]))

        for seed in seeds:
            for weights in weight_grid:
                wlabel = _weight_label(weights)
                merge_dir = merges_dir / pair_name / f"seed_{seed}" / wlabel
                if not merge_dir.exists():
                    continue

                for eval_task in eval_tasks_for_pair:
                    output_path = (
                        merged_eval_dir
                        / f"{pair_name}_seed_{seed}_{wlabel}_{eval_task}.json"
                    )
                    print(f"  Evaluating {pair_name}/seed_{seed}/{wlabel} on {eval_task}...")
                    run_lm_eval(
                        base_model=base_model,
                        adapter_dir=str(merge_dir),
                        task=eval_task,
                        output_path=output_path,
                        max_samples=max_samples,
                        device=device,
                    )

    # --- Part C: Evaluate calibration merged adapters on their shared task ---
    if calibration_enabled:
        print("\nPhase 4c: Evaluating calibration merged adapters")
        seed_pairs = list(itertools.combinations(seeds, 2))

        for task in task_names:
            eval_task = config["adapters"][task]["eval_task"]
            for seed_a, seed_b in seed_pairs:
                cal_dir_name = f"{task}_seeds_{seed_a}_{seed_b}"
                for weights in weight_grid:
                    wlabel = _weight_label(weights)
                    merge_dir = merges_dir / "calibration" / cal_dir_name / wlabel
                    if not merge_dir.exists():
                        continue

                    output_path = (
                        merged_eval_dir
                        / f"calibration_{cal_dir_name}_{wlabel}_{eval_task}.json"
                    )
                    print(f"  Evaluating calibration/{cal_dir_name}/{wlabel} on {eval_task}...")
                    run_lm_eval(
                        base_model=base_model,
                        adapter_dir=str(merge_dir),
                        task=eval_task,
                        output_path=output_path,
                        max_samples=max_samples,
                        device=device,
                    )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 4 complete in {elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
