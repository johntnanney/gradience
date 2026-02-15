#!/usr/bin/env python3
"""
Phase 4: Evaluate all adapters via lm-evaluation-harness.

Evaluates:
  - 12 individual adapters (each on its own task)
  - 72 merged adapters (each on both constituent tasks + MMLU subset)
  - Total: ~156 evaluation runs

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


# Map M1 task eval_task names to lm-eval-harness task names
EVAL_TASK_MAP = {
    "sql_generation": "sql_generation",  # custom task or use exact match
    "mmlu": "mmlu",
    "gsm8k": "gsm8k",
    "humaneval": "humaneval",
}


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

    # Build lm_eval command
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={base_model}",
        "--tasks", task,
        "--batch_size", "auto",
        "--device", device,
        "--output_path", str(output_path.parent),
        "--log_samples",
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
            print(f"      [ERROR] lm_eval failed: {result.stderr[:200]}")
            # Save error info
            error_data = {
                "error": True,
                "returncode": result.returncode,
                "stderr": result.stderr[:1000],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(error_data, f, indent=2)
            return error_data

        # Parse results from lm_eval output directory
        results_dir = output_path.parent
        results_files = list(results_dir.glob("results_*.json"))
        if results_files:
            with open(results_files[-1]) as f:
                results = json.load(f)
            # Save a normalized copy
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            return results

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
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    merges_dir = workspace / "merges"
    evals_dir = workspace / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    base_model = config["experiment"]["base_model"]
    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    methods = config["merge"]["methods"]
    max_samples = config["evaluation"]["max_eval_samples"]
    device = config["runtime"]["device"]

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

    # --- Part B: Evaluate merged adapters on both constituent tasks + MMLU ---
    print("\nPhase 4b: Evaluating merged adapters")
    merged_eval_dir = evals_dir / "merged"
    merged_eval_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(itertools.combinations(task_names, 2))
    general_task = config["evaluation"]["general_capability"]

    for task_a, task_b in pairs:
        pair_name = f"{task_a}_{task_b}"
        eval_tasks_for_pair = [
            config["adapters"][task_a]["eval_task"],
            config["adapters"][task_b]["eval_task"],
            general_task,
        ]
        # Deduplicate (e.g., if one task's eval_task is already mmlu)
        eval_tasks_for_pair = list(dict.fromkeys(eval_tasks_for_pair))

        for seed in seeds:
            for method in methods:
                merge_dir = merges_dir / pair_name / f"seed_{seed}" / method
                if not merge_dir.exists():
                    continue

                for eval_task in eval_tasks_for_pair:
                    output_path = (
                        merged_eval_dir
                        / f"{pair_name}_seed_{seed}_{method}_{eval_task}.json"
                    )
                    print(f"  Evaluating {pair_name}/seed_{seed}/{method} on {eval_task}...")
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
