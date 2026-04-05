"""Check command — config validation and restraint-first recommendations."""

from __future__ import annotations

import argparse
import json
from typing import Any

from gradience.cli_utils import (
    _apply_overrides,
    _autodetect_file_in_dir,
    _blank_vnext_dict,
    _load_config_file,
    _merge_fill_missing,
    _normalize_to_vnext_dict,
    _print_recommendations,
)
from gradience.exceptions import ConfigError, DependencyError


def cmd_check(args: argparse.Namespace) -> None:
    """Validate a LoRA/training config and emit restraint-first recommendations."""

    # Convenience alias: --task behaves like --dataset
    if getattr(args, "task", None) is not None and args.dataset is None:
        args.dataset = args.task

    # Auto-detect convenience wrapper inputs if directories are provided.
    # Explicit file paths (--peft/--training) take precedence over directories.
    if args.peft is None and getattr(args, "peft_dir", None):
        try:
            args.peft = _autodetect_file_in_dir(
                args.peft_dir,
                candidates=["adapter_config.json", "adapter_config.yaml", "adapter_config.yml"],
                label="PEFT adapter_config",
            )
        except (FileNotFoundError, OSError) as e:
            raise ConfigError(str(e)) from e

    if args.training is None and getattr(args, "training_dir", None):
        try:
            args.training = _autodetect_file_in_dir(
                args.training_dir,
                candidates=["training_args.json", "training_args.yaml", "training_args.yml"],
                label="training_args",
            )
        except (FileNotFoundError, OSError) as e:
            raise ConfigError(str(e)) from e

    # Build merged config dict from up to three sources:
    #   1) positional CONFIG (canonical or flat)
    #   2) --peft adapter_config.json (or --peft-dir)
    #   3) --training training_args.json (or --training-dir)
    # Then apply explicit CLI overrides.

    if args.config is None and args.peft is None and args.training is None:
        raise ConfigError(
            "Please provide either a CONFIG file or --peft/--training inputs.\n"
            "Examples:\n"
            "  gradience check config.yaml\n"
            "  gradience check --task gsm8k --peft adapter_config.json --training training_args.json\n"
            "  gradience check --task gsm8k --peft-dir ./peft_out --training-dir ./trainer_out"
        )

    merged = _blank_vnext_dict()
    sources: list[dict[str, str]] = []

    def _load_and_norm(path: str) -> dict[str, Any]:
        raw = _load_config_file(path)
        return _normalize_to_vnext_dict(raw)

    # (1) CONFIG
    if args.config is not None:
        try:
            d0 = _load_and_norm(args.config)
            merged = _merge_fill_missing(merged, d0)
            sources.append({"type": "config", "path": str(args.config)})
        except FileNotFoundError as e:
            raise ConfigError(f"File not found: {args.config}") from e
        except (OSError, ValueError, json.JSONDecodeError) as e:
            raise ConfigError(f"Failed to parse config file '{args.config}': {e}") from e

    # (2) PEFT adapter_config
    if args.peft is not None:
        try:
            dp = _load_and_norm(args.peft)
            merged = _merge_fill_missing(merged, dp)
            sources.append({"type": "peft", "path": str(args.peft)})
        except FileNotFoundError as e:
            raise ConfigError(f"File not found: {args.peft}") from e
        except (OSError, ValueError, json.JSONDecodeError) as e:
            raise ConfigError(f"Failed to parse PEFT config '{args.peft}': {e}") from e

    # (3) Training args
    if args.training is not None:
        try:
            dt = _load_and_norm(args.training)
            merged = _merge_fill_missing(merged, dt)
            sources.append({"type": "training", "path": str(args.training)})
        except FileNotFoundError as e:
            raise ConfigError(f"File not found: {args.training}") from e
        except (OSError, ValueError, json.JSONDecodeError) as e:
            raise ConfigError(f"Failed to parse training args '{args.training}': {e}") from e

    # Attach sources for debugging
    merged.setdefault("extras", {})
    if sources:
        merged["extras"].setdefault("sources", [])
        # Don't explode existing sources if present
        if isinstance(merged["extras"].get("sources"), list):
            merged["extras"]["sources"].extend(sources)
        else:
            merged["extras"]["sources"] = sources

    # Apply explicit CLI overrides
    d = merged
    _apply_overrides(d, args)

    try:
        from gradience.vnext import ConfigSnapshot, check_config

        config = ConfigSnapshot.from_dict(d)
        recs = check_config(config)
    except ImportError as e:
        raise DependencyError(f"Failed to build ConfigSnapshot or run policy: {e}") from e
    except (ValueError, TypeError) as e:
        raise ConfigError(f"Failed to build ConfigSnapshot or run policy: {e}") from e

    if args.json:
        payload = {
            "config": config.to_dict(),
            "recommendations": [r.to_dict() for r in recs],
        }
        print(json.dumps(payload, indent=2))
        return

    _print_recommendations(config, recs, verbose=args.verbose)


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------



def _setup_check_command(subparsers):
    check_parser = subparsers.add_parser(
        "check", help="[ADVANCED] Validate a config and emit restraint-first recommendations"
    )
    check_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to config JSON/YAML (canonical vNext or flat/PEFT-style). Optional if using --peft/--training.",
    )

    # Convenience wrapper inputs
    check_parser.add_argument("--peft", type=str, default=None, help="Path to PEFT adapter_config.json (or YAML)")
    check_parser.add_argument("--training", type=str, default=None, help="Path to training_args.json (or YAML)")

    # Convenience wrapper inputs (directories)
    check_parser.add_argument(
        "--peft-dir",
        type=str,
        default=None,
        help="Path to a PEFT output directory (auto-detects adapter_config.json). Ignored if --peft is set.",
    )
    check_parser.add_argument(
        "--training-dir",
        type=str,
        default=None,
        help="Path to a training output directory (auto-detects training_args.json). Ignored if --training is set.",
    )

    # `--task` is a convenience alias for `--dataset` (matches internal naming)
    check_parser.add_argument(
        "--task", type=str, default=None, help="Convenience alias for --dataset (e.g., gsm8k, sst2)"
    )

    # Optional overrides
    check_parser.add_argument("--model", type=str, default=None, help="Override model name")
    check_parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    check_parser.add_argument(
        "--task-profile",
        type=str,
        default=None,
        choices=["easy_classification", "hard_reasoning", "generation", "unknown"],
        help="Override task profile",
    )

    check_parser.add_argument("--optimizer", type=str, default=None, help="Override optimizer name (e.g., AdamW)")
    check_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    check_parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay")

    check_parser.add_argument("--r", type=int, default=None, help="Override LoRA rank")
    check_parser.add_argument("--alpha", type=float, default=None, help="Override LoRA alpha")
    check_parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Override target modules (space-separated and/or comma-separated)",
    )

    check_parser.add_argument("--notes", type=str, default=None, help="Attach notes to ConfigSnapshot")

    check_parser.add_argument("--verbose", action="store_true", help="Print rationale/evidence")
    check_parser.add_argument("--json", action="store_true", help="Output JSON instead of pretty text")

    check_parser.set_defaults(func=cmd_check)



def setup_check_commands(subparsers) -> None:
    """Register check commands with the argument parser."""
    _setup_check_command(subparsers)
