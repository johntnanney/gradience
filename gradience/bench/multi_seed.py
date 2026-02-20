"""Multi-seed benchmarking and aggregation for bench protocol."""
from __future__ import annotations

import json
import datetime
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from gradience.bench.model_setup import load_config
from gradience.bench.heartbeat import heartbeat_stage
from gradience.bench.reporting import (
    create_multi_seed_aggregated_report,
    create_multi_seed_markdown_report,
)



def run_multi_seed_bench_protocol(
    config_path: str | Path,
    output_dir: str | Path,
    seeds: list[int],
    variants_to_test: Optional[list[str]] = None,
    smoke: bool = False,
    ci: bool = False
) -> Dict[str, Any]:
    """
    Run bench protocol across multiple seeds and aggregate results.

    Returns aggregated report with mean ± std statistics.
    """
    # Import here to avoid circular imports since protocol.py will import from this module
    from gradience.bench.protocol import run_bench_protocol

    config = load_config(config_path)
    output_path = Path(output_dir)

    # HYGIENE: Ensure output directory exists BEFORE any logging/tee operations
    output_path.mkdir(parents=True, exist_ok=True)

    # HYGIENE: Start heartbeat for multi-seed coordination (prevent SSH timeouts)
    heartbeat_stage("multi_seed_coordination", output_dir=output_path, seed=None)

    print(f"Gradience Multi-Seed Bench Protocol v0.1")
    print("=" * 50)
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Seeds: {seeds}")
    print(f"Variants: {variants_to_test or 'all'}")
    print(f"Smoke mode: {smoke}")
    print()

    # Store individual seed results
    seed_reports = []
    seed_dirs = []

    # Run each seed
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")

        # Create seed-specific config
        seed_config = config.copy()
        seed_config["train"]["seed"] = seed

        # Remove multi-seed config from individual seeds to prevent infinite recursion
        compression = seed_config.get("compression", {}).copy()
        compression.pop("seeds", None)  # Remove seeds field to force single-seed mode

        # Filter variants if specified
        if variants_to_test:
            compression["variants_to_test"] = variants_to_test

        seed_config["compression"] = compression

        # HYGIENE: Create seed-specific directory BEFORE any operations
        seed_dir = output_path / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_dirs.append(seed_dir)

        # HYGIENE: Create progress/heartbeat file for stuck detection
        progress_file = seed_dir / "progress.txt"
        with open(progress_file, 'w') as f:
            f.write(f"STARTED: seed_{seed} at {datetime.datetime.now().isoformat()}\n")
            f.flush()

        # Write seed-specific config
        seed_config_path = seed_dir / "config.yaml"
        with open(seed_config_path, 'w') as f:
            yaml.dump(seed_config, f, indent=2)

        # Run single seed benchmark
        try:
            # Update progress before starting
            with open(progress_file, 'a') as f:
                f.write(f"RUNNING: bench protocol started at {datetime.datetime.now().isoformat()}\n")
                f.flush()

            seed_report = run_bench_protocol(
                config_path=seed_config_path,
                output_dir=seed_dir,
                smoke=smoke,
                ci=ci
            )

            # Add seed info to report
            seed_report["seed"] = seed
            seed_report["seed_index"] = i
            seed_reports.append(seed_report)

            # Mark completion in progress file
            with open(progress_file, 'a') as f:
                f.write(f"COMPLETED: seed_{seed} at {datetime.datetime.now().isoformat()}\n")
                f.flush()

            print(f"\nSeed {seed} completed successfully")

        except Exception as e:
            # Mark failure in progress file
            with open(progress_file, 'a') as f:
                f.write(f"FAILED: seed_{seed} at {datetime.datetime.now().isoformat()}: {e}\n")
                f.flush()

            print(f"\nSeed {seed} failed: {e}")
            # Continue with other seeds
            continue

    if not seed_reports:
        raise RuntimeError("All seed runs failed - cannot generate aggregated report")

    print(f"\n{'='*60}")
    print(f"AGGREGATION: {len(seed_reports)}/{len(seeds)} seeds successful")
    print(f"{'='*60}")

    # Create aggregated report
    aggregated_report = create_multi_seed_aggregated_report(
        seed_reports=seed_reports,
        config=config,
        output_dir=output_path
    )

    # Write aggregated bench.json
    agg_report_path = output_path / "bench_aggregate.json"
    with open(agg_report_path, 'w') as f:
        json.dump(aggregated_report, f, indent=2, ensure_ascii=False)

    # Create and write aggregated markdown report
    agg_markdown_content = create_multi_seed_markdown_report(
        aggregated_report=aggregated_report,
        config=config,
        output_dir=output_path
    )

    agg_markdown_path = output_path / "bench_aggregate.md"
    with open(agg_markdown_path, 'w') as f:
        f.write(agg_markdown_content)

    # Write seed summary
    seed_summary_path = output_path / "seed_summary.json"
    seed_summary = {
        "total_seeds": len(seeds),
        "successful_seeds": len(seed_reports),
        "failed_seeds": len(seeds) - len(seed_reports),
        "seed_directories": [str(d) for d in seed_dirs],
        "aggregated_report": str(agg_report_path),
        "aggregated_markdown": str(agg_markdown_path)
    }
    with open(seed_summary_path, 'w') as f:
        json.dump(seed_summary, f, indent=2, ensure_ascii=False)

    print(f"\nMulti-seed benchmark complete!")
    print(f"  Aggregated report: {agg_report_path}")
    print(f"  Aggregated markdown: {agg_markdown_path}")
    print(f"  Seed summary: {seed_summary_path}")
    print(f"  Individual seed results in: {[d.name for d in seed_dirs]}")

    return aggregated_report
