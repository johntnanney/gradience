"""
Gradience Bench CLI - LoRA Compression Benchmarking Tool

Usage:
    python -m gradience.bench.run_bench \
      --config distilbert_sst2.yaml \
      --output bench_runs/distilbert_sst2_001

Runs the complete bench protocol:
1. Train probe adapter (r=16)
2. Audit -> suggestions  
3. Generate compression configs
4. Retrain compressed variants
5. Evaluate and make verdicts
6. Generate reports (bench.json + bench.md)
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import tempfile
import yaml
from pathlib import Path
from typing import Optional

from .protocol import run_bench_protocol


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    
    parser = argparse.ArgumentParser(
        prog="python -m gradience.bench.run_bench",
        description="LoRA compression benchmarking tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python -m gradience.bench.run_bench --config configs/distilbert_sst2.yaml --output results/run_001
  
  # Smoke test (faster)
  python -m gradience.bench.run_bench --config configs/distilbert_sst2.yaml --output smoke_test --smoke
  
  # CI mode (fails if compression strategies don't pass)
  python -m gradience.bench.run_bench --config configs/distilbert_sst2.yaml --output ci_run --ci
  
  # Override device
  python -m gradience.bench.run_bench --config configs/distilbert_sst2.yaml --output gpu_run --device cuda
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/distilbert_sst2.yaml)"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        required=True,
        help="Output directory for benchmark results"
    )
    
    # Optional flags
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode (uses smoke_* limits from config for faster testing)"
    )
    
    parser.add_argument(
        "--ci",
        action="store_true", 
        help="CI mode: exit non-zero if attempted compression strategies FAIL"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        help="Override device from config (cpu, mps, or cuda)"
    )
    
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        default=True,
        help="Keep all artifacts (default: True)"
    )
    
    parser.add_argument(
        "--clean-on-pass",
        action="store_true",
        help="Clean artifacts if all strategies pass (future feature)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    
    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    if not config_path.suffix.lower() in ['.yaml', '.yml']:
        print(f"Warning: Config file should be YAML (.yaml/.yml): {config_path}")
    
    # Check output directory can be created
    output_path = Path(args.output)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error: Cannot create output directory: {output_path}")
        sys.exit(1)
    
    if args.clean_on_pass:
        print("Warning: --clean-on-pass is not yet implemented")


def check_exit_conditions(report: dict, ci_mode: bool) -> int:
    """Check exit conditions and return appropriate exit code."""
    
    if not ci_mode:
        return 0
    
    # CI mode: check if strategies passed
    # Handle both old and new report formats
    if "verdicts" in report and isinstance(report["verdicts"], dict):
        if "verdicts" in report["verdicts"]:
            # New format: report["verdicts"]["verdicts"]
            verdicts = report["verdicts"]["verdicts"]
        else:
            # Alternative format: report["verdicts"] directly contains verdicts
            verdicts = report["verdicts"]
    else:
        # Fallback: look in variants
        verdicts = {}
    
    attempted_strategies = [v for v in verdicts.values() if v.get("status") == "evaluated"]
    passed_strategies = [v for v in attempted_strategies if v.get("verdict") == "PASS"]
    
    # Debug info
    if not attempted_strategies:
        print(f"  Debug: Found verdict keys: {list(verdicts.keys())}")
        if verdicts:
            sample_verdict = list(verdicts.values())[0]
            print(f"  Debug: Sample verdict structure: {sample_verdict}")
        else:
            print(f"  Debug: Report keys: {list(report.keys())}")
    
    total_attempted = len(attempted_strategies)
    total_passed = len(passed_strategies)
    
    print(f"\nCI Mode Results:")
    print(f"  Attempted strategies: {total_attempted}")
    print(f"  Passed strategies: {total_passed}")
    
    if total_attempted == 0:
        print("  No strategies were attempted - CI FAIL")
        return 1
    
    # Fail if no strategies passed
    if total_passed == 0:
        print("  No strategies passed - CI FAIL")
        return 1
    
    # Optional: Stricter check - fail if < 2/3 passed
    # success_rate = total_passed / total_attempted
    # if success_rate < 2/3:
    #     print(f"  Success rate {success_rate:.1%} < 67% - CI FAIL")
    #     return 1
    
    print("  CI PASS")
    return 0


def main() -> int:
    """Main entry point for the CLI."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_args(args)
    
    print("Gradience Bench CLI")
    print("=" * 40)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Smoke mode: {args.smoke}")
    print(f"CI mode: {args.ci}")
    if args.device:
        print(f"Device override: {args.device}")
    print()

    try:
        # Handle device override by modifying config temporarily
        if args.device:
            # Load original config
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Override device
            if 'runtime' not in config_data:
                config_data['runtime'] = {}
            config_data['runtime']['device'] = args.device
            
            # Write temporary config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.safe_dump(config_data, f)
                temp_config_path = f.name
            
            try:
                # Run with temporary config
                report = run_bench_protocol(
                    config_path=temp_config_path,
                    output_dir=args.output,
                    smoke=args.smoke,
                    ci=args.ci
                )
            finally:
                # Clean up temporary config
                os.unlink(temp_config_path)
        else:
            # Run with original config
            report = run_bench_protocol(
                config_path=args.config,
                output_dir=args.output,
                smoke=args.smoke,
                ci=args.ci
            )
        
        # For CI mode, we need access to the internal verdicts
        # The canonical report doesn't include detailed verdict analysis
        internal_report_path = Path(args.output) / "bench_internal.json"
        if args.ci and internal_report_path.exists():
            with open(internal_report_path, 'r') as f:
                internal_report = json.load(f)
            exit_code = check_exit_conditions(internal_report, args.ci)
        else:
            exit_code = check_exit_conditions(report, args.ci)
        
        print(f"\nBenchmark complete! Results in: {args.output}")
        return exit_code
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 130
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo run bench, install dependencies:")
        print("  pip install transformers>=4.20.0 peft>=0.4.0 datasets torch pyyaml")
        return 1
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())