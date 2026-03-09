#!/usr/bin/env python3
"""
Check for stuck/incomplete bench runs and provide resume guidance.

Engineering hygiene utility to detect runs that may have failed due to:
- SSH timeouts
- Out of memory errors
- CUDA errors
- Process kills

Usage:
    python scripts/check_stuck_runs.py /path/to/output/dir
    python scripts/check_stuck_runs.py /workspace/evidence_runs/exp01_multiseed --auto-resume
"""

import argparse
import json
import sys
from pathlib import Path



def check_seed_status(seed_dir: Path) -> dict[str, any]:
    """Check the status of a single seed run."""
    status = {
        "seed_dir": str(seed_dir),
        "seed_name": seed_dir.name,
        "status": "unknown",
        "last_activity": None,
        "can_resume": False,
        "progress_file": None,
        "bench_json": None,
        "error_info": None,
    }

    # Check for progress file
    progress_file = seed_dir / "progress.txt"
    if progress_file.exists():
        status["progress_file"] = str(progress_file)

        try:
            with open(progress_file) as f:
                lines = f.read().strip().split("\n")

            if lines:
                last_line = lines[-1]
                status["last_activity"] = last_line

                if "COMPLETED:" in last_line:
                    status["status"] = "completed"
                elif "FAILED:" in last_line:
                    status["status"] = "failed"
                    status["error_info"] = last_line
                elif "RUNNING:" in last_line:
                    status["status"] = "stuck_or_running"
                    status["can_resume"] = True
                elif "STARTED:" in last_line:
                    status["status"] = "stuck_at_start"
                    status["can_resume"] = True

        except Exception as e:
            status["error_info"] = f"Could not read progress file: {e}"

    # Check for bench.json (completion indicator)
    bench_json = seed_dir / "bench.json"
    if bench_json.exists():
        status["bench_json"] = str(bench_json)
        if status["status"] == "unknown":
            status["status"] = "completed"

    # Check for config.yaml (confirms it's a valid seed dir)
    config_yaml = seed_dir / "config.yaml"
    if not config_yaml.exists():
        status["status"] = "invalid_seed_dir"

    return status


def check_multiseed_run(output_dir: Path) -> dict[str, any]:
    """Check the status of a multi-seed bench run."""
    results = {
        "output_dir": str(output_dir),
        "run_type": "unknown",
        "overall_status": "unknown",
        "seeds": [],
        "aggregate_complete": False,
        "can_resume": False,
        "resume_command": None,
    }

    # Check if this looks like a multi-seed run
    seed_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]

    if seed_dirs:
        results["run_type"] = "multiseed"
        results["seeds"] = []

        completed_seeds = 0
        stuck_seeds = 0
        failed_seeds = 0

        for seed_dir in sorted(seed_dirs):
            seed_status = check_seed_status(seed_dir)
            results["seeds"].append(seed_status)

            if seed_status["status"] == "completed":
                completed_seeds += 1
            elif seed_status["status"] in ["stuck_or_running", "stuck_at_start"]:
                stuck_seeds += 1
            elif seed_status["status"] == "failed":
                failed_seeds += 1

        # Determine overall status
        total_seeds = len(seed_dirs)
        if completed_seeds == total_seeds:
            results["overall_status"] = "completed"
        elif completed_seeds > 0:
            results["overall_status"] = "partially_complete"
            results["can_resume"] = True
        elif stuck_seeds > 0:
            results["overall_status"] = "stuck"
            results["can_resume"] = True
        elif failed_seeds == total_seeds:
            results["overall_status"] = "all_failed"
        else:
            results["overall_status"] = "unknown"

        # Check for aggregate files
        aggregate_json = output_dir / "bench_aggregate.json"
        aggregate_md = output_dir / "bench_aggregate.md"
        if aggregate_json.exists() and aggregate_md.exists():
            results["aggregate_complete"] = True

    else:
        # Check if it's a single-seed run
        if (output_dir / "bench.json").exists():
            results["run_type"] = "single_seed"
            seed_status = check_seed_status(output_dir)
            results["overall_status"] = seed_status["status"]
            results["can_resume"] = seed_status["can_resume"]

    return results


def print_status_report(results: dict[str, any]) -> None:
    """Print a human-readable status report."""
    print("🔍 Bench Run Status Report")
    print(f"{'=' * 50}")
    print(f"Directory: {results['output_dir']}")
    print(f"Type: {results['run_type']}")
    print(f"Status: {results['overall_status']}")
    print()

    if results["run_type"] == "multiseed":
        print("📊 Seed Summary:")
        status_counts = {}
        for seed in results["seeds"]:
            status = seed["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        for status, count in status_counts.items():
            print(f"   {status}: {count}")
        print()

        print("📋 Individual Seed Status:")
        for seed in results["seeds"]:
            status_icon = {
                "completed": "✅",
                "stuck_or_running": "⏳",
                "stuck_at_start": "🔄",
                "failed": "❌",
                "unknown": "❓",
            }.get(seed["status"], "❓")

            print(f"   {status_icon} {seed['seed_name']}: {seed['status']}")
            if seed["error_info"]:
                print(f"      Error: {seed['error_info']}")
        print()

    if results["can_resume"]:
        print("🔄 Resume Capability: YES")
        print("   This run can be resumed using --resume flag")
        if results.get("resume_command"):
            print(f"   Command: {results['resume_command']}")
    else:
        print("🔄 Resume Capability: NO")
        if results["overall_status"] == "completed":
            print("   Run already completed successfully")
        elif results["overall_status"] == "all_failed":
            print("   All seeds failed - recommend starting fresh")
    print()

    if results["aggregate_complete"]:
        print("📈 Aggregate Report: ✅ Available")
    else:
        print("📈 Aggregate Report: ❌ Not yet generated")


def main():
    parser = argparse.ArgumentParser(description="Check bench run status and detect stuck runs")
    parser.add_argument("output_dir", help="Path to bench output directory to check")
    parser.add_argument("--auto-resume", action="store_true", help="Automatically attempt to resume stuck runs")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Directory does not exist: {output_dir}")
        sys.exit(1)

    if not output_dir.is_dir():
        print(f"❌ Not a directory: {output_dir}")
        sys.exit(1)

    # Check run status
    results = check_multiseed_run(output_dir)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_status_report(results)

    # Auto-resume if requested
    if args.auto_resume and results["can_resume"]:
        print("🚀 Auto-resume not yet implemented")
        print("   Use --resume flag with your original bench command")

    # Exit code based on status
    if results["overall_status"] in ["completed"]:
        sys.exit(0)
    elif results["overall_status"] in ["stuck", "stuck_or_running", "partially_complete"]:
        sys.exit(2)  # Resumable
    else:
        sys.exit(1)  # Failed


if __name__ == "__main__":
    main()
