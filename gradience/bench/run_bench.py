"""
Bench entry point (v0.1).

Implements Step 3.1: Train probe adapter with GradienceCallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import run_bench_protocol
from .report import write_report


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m gradience.bench.run_bench")
    p.add_argument("--config", required=True, help="Path to a bench config (yaml)")
    p.add_argument("--output", required=True, help="Output directory for bench artifacts")
    p.add_argument("--smoke", action="store_true", help="Run a fast, minimal benchmark")
    p.add_argument("--ci", action="store_true", help="Fail if recommendations do not validate")
    args = p.parse_args()

    try:
        # Run the bench protocol
        report = run_bench_protocol(
            config_path=args.config,
            output_dir=args.output,
            smoke=args.smoke,
            ci=args.ci
        )
        
        # Write report
        print("\nWriting report...")
        json_path, md_path = write_report(args.output, report)
        print(f"Report written:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
        
        return 0
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo run bench, install dependencies:")
        print("  pip install transformers>=4.20.0 peft>=0.4.0 datasets torch pyyaml")
        return 1
        
    except Exception as e:
        print(f"Bench failed: {e}")
        if args.ci:
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())