"""Truncate command — SVD truncation of LoRA adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from gradience.exceptions import AuditError, ConfigError, DependencyError


def cmd_truncate(args: argparse.Namespace) -> None:
    """SVD truncate a PEFT LoRA adapter to a smaller rank."""

    peft_dir = Path(args.peft_dir)
    out_dir = Path(args.out_dir)
    target_rank = args.rank

    if not peft_dir.exists():
        raise ConfigError(f"Input PEFT directory not found: {peft_dir}")

    if target_rank <= 0:
        raise ConfigError(f"Target rank must be positive, got: {target_rank}")

    try:
        from gradience.vnext.svd_truncate import save_truncation_report, svd_truncate_peft_dir
    except ImportError as e:
        raise DependencyError(f"Failed to import SVD truncate module: {e}") from e

    try:
        report = svd_truncate_peft_dir(
            peft_dir=peft_dir,
            out_dir=out_dir,
            target_rank=target_rank,
            alpha_mode=args.alpha_mode,
            save_dtype=args.dtype,
        )

        if args.json:
            import json

            print(json.dumps(report.__dict__, indent=2))
        else:
            print("✅ SVD truncation completed successfully!")
            print(f"📁 Input:  {peft_dir}")
            print(f"📁 Output: {out_dir}")
            print()

            # Core metrics (as specified)
            print(f"Input rank: {report.original_rank}")
            print(f"Output rank: {report.target_rank}")
            print(f"Mean retained energy: {report.energy_retained:.1%}")

            # Calculate total LoRA parameter reduction
            total_original_lora_params = sum(int(m["original_params"]) for m in report.per_module_energy)
            total_new_lora_params = sum(int(m["new_params"]) for m in report.per_module_energy)
            lora_reduction_ratio = (
                total_original_lora_params / total_new_lora_params if total_new_lora_params > 0 else 1.0
            )

            print(
                f"LoRA parameter reduction: {total_original_lora_params:,} → {total_new_lora_params:,} ({lora_reduction_ratio:.1f}x)"
            )
            print(f"Alpha mode: {report.alpha_mode}")
            print(f"Modules processed: {report.total_modules}")

            if args.verbose:
                print("\nPer-module energy retention:")
                for module in report.per_module_energy:
                    name = module["module_name"]
                    energy = module["energy_retained"]
                    orig_params = module["original_params"]
                    new_params = module["new_params"]
                    print(f"  {name}: {energy:.1%} ({orig_params:,} → {new_params:,} params)")

        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            save_truncation_report(report, report_path)
            if not args.json:
                print(f"📄 Report saved: {report_path}")

    except (RuntimeError, ValueError, OSError) as e:
        raise AuditError(f"SVD truncation failed: {e}") from e



def _setup_truncate_command(subparsers):
    truncate_parser = subparsers.add_parser(
        "truncate", help="[ADVANCED] SVD truncate a PEFT LoRA adapter to a smaller rank"
    )
    truncate_parser.add_argument("--peft-dir", type=str, required=True, help="Path to input PEFT adapter directory")
    truncate_parser.add_argument(
        "--out-dir", type=str, required=True, help="Path to output directory for truncated adapter"
    )
    truncate_parser.add_argument(
        "--rank", type=int, required=True, help="Target rank for truncation (must be smaller than original)"
    )
    truncate_parser.add_argument(
        "--alpha-mode",
        choices=["keep_ratio", "keep_alpha"],
        default="keep_ratio",
        help="How to handle lora_alpha scaling (default: keep_ratio)",
    )
    truncate_parser.add_argument(
        "--dtype", choices=["fp16", "bf16", "fp32"], default="fp16", help="Data type for saved weights (default: fp16)"
    )
    truncate_parser.add_argument("--report", type=str, help="Path to save detailed truncation report (JSON)")
    truncate_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    truncate_parser.add_argument("--verbose", action="store_true", help="Show detailed per-module statistics")
    truncate_parser.set_defaults(func=cmd_truncate)



def setup_truncate_commands(subparsers) -> None:
    """Register truncate commands with the argument parser."""
    _setup_truncate_command(subparsers)
