#!/usr/bin/env python3
"""
Template validation script for Predibase LoRA adapters.

Runs validation samples on each task with Predibase prompt templates.
If both MNLI and QNLI achieve >= threshold accuracy, exits 0 (success).
Otherwise exits 1 (fallback to training your own adapters).

Usage::

    python validate_templates.py \\
        --base-model mistralai/Mistral-7B-v0.1 \\
        --adapter-a-dir ./adapters/adapter_a \\
        --adapter-b-dir ./adapters/adapter_b \\
        --threshold 0.70 \\
        --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Sibling imports from this script directory
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import quick_validate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Validate Predibase prompt templates against adapters")
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model HuggingFace identifier",
    )
    parser.add_argument(
        "--adapter-a-dir",
        type=str,
        required=True,
        help="Path to MNLI adapter directory",
    )
    parser.add_argument(
        "--adapter-b-dir",
        type=str,
        required=True,
        help="Path to QNLI adapter directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum accuracy to pass (default: 0.70)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of validation samples per task (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write validation results JSON",
    )
    args = parser.parse_args()

    start = time.monotonic()
    results = {}

    # -----------------------------------------------------------------------
    # Validate MNLI adapter
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Validating MNLI adapter with Predibase template...")
    print(f"  Adapter: {args.adapter_a_dir}")
    print(f"  Samples: {args.n_samples}, Threshold: {args.threshold}")
    print()

    mnli_acc, mnli_pass = quick_validate(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_a_dir,
        dataset_name="mnli",
        template_mode="predibase",
        n_samples=args.n_samples,
        device=args.device,
        threshold=args.threshold,
    )
    status = "PASS" if mnli_pass else "FAIL"
    results["mnli"] = {"accuracy": mnli_acc, "passed": mnli_pass}
    print(f"\n  MNLI: accuracy={mnli_acc:.4f} [{status}]")

    # -----------------------------------------------------------------------
    # Validate QNLI adapter
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Validating QNLI adapter with Predibase template...")
    print(f"  Adapter: {args.adapter_b_dir}")
    print()

    qnli_acc, qnli_pass = quick_validate(
        base_model_name=args.base_model,
        adapter_dir=args.adapter_b_dir,
        dataset_name="qnli",
        template_mode="predibase",
        n_samples=args.n_samples,
        device=args.device,
        threshold=args.threshold,
    )
    status = "PASS" if qnli_pass else "FAIL"
    results["qnli"] = {"accuracy": qnli_acc, "passed": qnli_pass}
    print(f"\n  QNLI: accuracy={qnli_acc:.4f} [{status}]")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.monotonic() - start
    both_pass = mnli_pass and qnli_pass
    results["both_pass"] = both_pass
    results["elapsed_seconds"] = round(elapsed, 1)
    results["threshold"] = args.threshold
    results["n_samples"] = args.n_samples

    print()
    print("=" * 60)
    print(f"  Validation completed in {elapsed:.0f}s")
    print(f"  MNLI: {mnli_acc:.4f} {'PASS' if mnli_pass else 'FAIL'}")
    print(f"  QNLI: {qnli_acc:.4f} {'PASS' if qnli_pass else 'FAIL'}")
    if both_pass:
        print("  Overall: PASS -- use Predibase templates")
    else:
        print("  Overall: FAIL -- fallback to training your own adapters")
    print("=" * 60)

    # Write optional JSON output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\n  Results written to: {output_path}")

    sys.exit(0 if both_pass else 1)


if __name__ == "__main__":
    main()
