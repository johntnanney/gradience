"""Audit commands — structural audit and QA artifact production."""

from __future__ import annotations

import argparse
import json as jsonlib
from pathlib import Path

from gradience.cli_utils import (
    _analyze_policy_disagreements,
    _parse_rank_policies,
    _print_audit_summary,
    _print_qa_summary,
)
from gradience.exceptions import AuditError, ConfigError, DependencyError, GradienceError


def cmd_audit(args: argparse.Namespace) -> None:

    """Audit a PEFT LoRA adapter directory and print a compact efficiency summary."""

    peft_dir = getattr(args, "peft_dir", None)
    if not peft_dir:
        raise ConfigError("--peft-dir is required for audit")

    try:
        from gradience.vnext.audit import audit_lora_peft_dir
    except ImportError as e:
        raise DependencyError(f"Failed to import LoRA audit module: {e}") from e

    try:
        # Parse rank policies
        rank_policies = _parse_rank_policies(getattr(args, "rank_policies", None))

        # Extract importance configuration
        importance_config = {
            "quantile_threshold": getattr(args, "importance_quantile", 0.75),
            "uniform_mult_gate": getattr(args, "importance_uniform_mult_gate", 1.5),
            "metric": getattr(args, "importance_metric", "energy_share"),
        }

        result = audit_lora_peft_dir(
            peft_dir,
            adapter_config_path=getattr(args, "adapter_config", None),
            adapter_weights_path=getattr(args, "weights", None),
            map_location="cpu",
            include_top_singular_values=int(getattr(args, "top_singular_values", 0) or 0),
            base_model_id=getattr(args, "base_model", None),
            base_norms_cache=getattr(args, "base_norms_cache", None),
            compute_udr=not getattr(args, "no_udr", False),
            rank_policies=rank_policies,
        )
        # --- audit --append support ---
        if getattr(args, "append", None):
            import json
            import time
            from pathlib import Path

            append_path = Path(args.append)
            run_id = None
            last_step = None
            if append_path.exists():
                try:
                    with append_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = jsonlib.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(entry, dict):
                                if run_id is None and isinstance(entry.get("run_id"), str):
                                    run_id = entry.get("run_id")
                                if isinstance(entry.get("step"), int):
                                    last_step = entry.get("step")
                except (OSError, UnicodeDecodeError):
                    pass
            if run_id is None:
                run_id = f"audit_{int(time.time())}"
            # Prefer structured event helper if available
            try:
                event = result.to_metrics_event(run_id=run_id, step=last_step)
            except (AttributeError, TypeError):
                event = {
                    "schema": "gradience.vnext.telemetry/v1",
                    "ts": time.time(),
                    "run_id": run_id,
                    "event": "metrics",
                    "step": last_step,
                    "kind": "lora_audit",
                    "metrics": result.to_summary_dict(include_layers=False),
                }
            append_path.parent.mkdir(parents=True, exist_ok=True)
            with append_path.open("a", encoding="utf-8") as f:
                f.write(jsonlib.dumps(event, default=str) + "\n")
            if not getattr(args, "json", False):
                print(f"Appended lora_audit metrics to {append_path}")
    except GradienceError:
        raise
    except Exception as e:
        raise AuditError(f"Audit failed: {e}") from e

    top_wasteful = int(getattr(args, "top_wasteful", 0) or 0)
    # Include layers if --layers flag is set OR if --top-wasteful is specified
    include_layers = getattr(args, "layers", False) or top_wasteful > 0

    if args.json:
        try:
            # When --layers is set, include all layers; otherwise respect top_wasteful
            if getattr(args, "layers", False):
                payload = result.to_summary_dict(include_layers=True, topk_layers=None)
            else:
                payload = result.to_summary_dict(
                    include_layers=include_layers, topk_layers=top_wasteful if include_layers else None
                )

            # Ensure n_layers_with_udr is present when UDR is enabled via CLI args
            # Only add this if UDR computation was actually enabled AND attempted
            # Note: --no-udr explicitly disables UDR, so respect that flag strictly
            if (
                not getattr(args, "no_udr", False)
                and getattr(args, "base_model", None)  # Only when base_model provided (not just cache)
                and "n_layers_with_udr" not in payload
            ):
                # Check if any UDR-related data exists to determine if UDR was attempted
                layers = getattr(result, "layers", [])
                has_udr_data = any(
                    getattr(lyr, "udr", None) is not None
                    or getattr(lyr, "base_sigma_max", None) is not None
                    or getattr(lyr, "base_fro_norm", None) is not None
                    for lyr in layers
                )
                if has_udr_data:
                    udr_count = sum(1 for lyr in layers if getattr(lyr, "udr", None) is not None)
                    payload["n_layers_with_udr"] = udr_count

            # Add per-layer rank suggestions if requested
            suggest_per_layer = getattr(args, "suggest_per_layer", False)
            if suggest_per_layer:
                if not include_layers and not getattr(args, "layers", False):
                    raise ConfigError("--suggest-per-layer requires --layers flag")

                try:
                    from gradience.vnext.rank_suggestion import suggest_per_layer_ranks

                    rank_suggestions = suggest_per_layer_ranks(payload)
                    payload["rank_suggestions"] = rank_suggestions.to_dict()
                except (ImportError, ValueError, TypeError, RuntimeError) as e:
                    payload["rank_suggestions_error"] = str(e)

            # Add policy disagreement analysis to JSON output
            try:
                layers = getattr(result, "layers", [])
                if layers:
                    name_mapping = {
                        "energy_threshold": "energy@0.90",
                        "knee_elbow": "knee",
                        "entropy_effective": "erank",
                        "optimal_hard_threshold": "oht",
                    }
                    rationale_verbosity = getattr(args, "disagreement_rationale", "flagged_only")
                    disagreement_analysis = _analyze_policy_disagreements(
                        layers, name_mapping, importance_config, rationale_verbosity
                    )
                    payload["policy_disagreement_analysis"] = disagreement_analysis
            except (ValueError, RuntimeError) as e:
                payload["policy_disagreement_analysis_error"] = str(e)

        except (AttributeError, KeyError, TypeError):
            # Fallback if result isn't the expected dataclass
            payload = {"error": "unexpected_audit_result_type", "type": str(type(result))}
        print(jsonlib.dumps(payload, indent=2))
        return

    _print_audit_summary(result, top_wasteful=top_wasteful, importance_config=importance_config)


# ---------------------------------------------------------------------------
# audit-adapter  (single-adapter QA artifact production)
# ---------------------------------------------------------------------------


def cmd_audit_adapter(args: argparse.Namespace) -> None:
    """Audit a single adapter and produce a QA artifact."""

    peft_dir = getattr(args, "peft_dir", None)
    if not peft_dir:
        raise ConfigError("--peft-dir is required")

    try:
        from gradience.vnext.audit import audit_lora_peft_dir
        from gradience.vnext.audit.qa_artifact import build_qa_artifact
    except ImportError as e:
        raise DependencyError(f"Failed to import audit module: {e}") from e

    # Run structural audit
    try:
        result = audit_lora_peft_dir(
            peft_dir,
            adapter_config_path=getattr(args, "adapter_config", None),
            adapter_weights_path=getattr(args, "weights", None),
            map_location="cpu",
            base_model_id=getattr(args, "base_model", None),
            base_norms_cache=getattr(args, "base_norms_cache", None),
            compute_udr=not getattr(args, "no_udr", False),
        )
    except GradienceError:
        raise  # already a GradienceError subclass
    except Exception as e:
        raise AuditError(f"Audit failed: {e}") from e

    # Parse behavioral args
    adapter_score = getattr(args, "adapter_score", None)
    base_score = getattr(args, "base_score", None)

    # Build QA artifact
    artifact = build_qa_artifact(
        result,
        adapter_path=peft_dir,
        base_model=getattr(args, "base_model", None) or "",
        adapter_score=adapter_score,
        base_score=base_score,
        metric_name=getattr(args, "metric_name", None) or None,
        lower_is_better=getattr(args, "lower_is_better", True),
        eval_dataset=getattr(args, "eval_dataset", None),
        margin=float(getattr(args, "margin", 0.0) or 0.0),
    )

    # Output
    if getattr(args, "json", False):
        print(jsonlib.dumps(artifact.to_dict(), indent=2))
        return

    _print_qa_summary(artifact)

    out_path = getattr(args, "out", None)
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            jsonlib.dump(artifact.to_dict(), f, indent=2)
            f.write("\n")
        print("\nOUTPUT")
        print("\u2500" * 50)
        print(f"  Wrote QA artifact to: {p}")


# ---------------------------------------------------------------------------
# merge-audit
# ---------------------------------------------------------------------------



def _setup_audit_command(subparsers):
    audit_parser = subparsers.add_parser(
        "audit",
        help="[ADVANCED] Audit a PEFT LoRA adapter directory for spectral rank/utilization analysis",
    )
    audit_parser.add_argument(
        "--append",
        default=None,
        help="Append lora_audit metrics event to an existing vNext run JSONL",
    )
    audit_parser.add_argument(
        "--peft-dir",
        type=str,
        required=True,
        help="Path to a PEFT output directory (containing adapter_config.* and adapter weights)",
    )
    audit_parser.add_argument(
        "--adapter-config",
        type=str,
        default=None,
        help="Optional explicit path to adapter_config.json/yaml (overrides auto-detect)",
    )
    audit_parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional explicit path to adapter_model.(safetensors|bin|pt) (overrides auto-detect)",
    )
    audit_parser.add_argument(
        "--top-wasteful",
        type=int,
        default=0,
        help="Print N most wasteful layers (lowest utilization). 0 disables.",
    )
    audit_parser.add_argument(
        "--top-singular-values",
        type=int,
        default=0,
        help="Include top-k singular values per layer in JSON output (cost: small).",
    )
    audit_parser.add_argument("--json", action="store_true", help="Output JSON instead of pretty text")
    audit_parser.add_argument(
        "--layers",
        action="store_true",
        help="Include per-layer audit rows in --json output (can be large).",
    )
    audit_parser.add_argument(
        "--suggest-per-layer",
        action="store_true",
        help="Include per-layer rank suggestions in --json output (requires --layers).",
    )
    # UDR/SDI support
    audit_parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID or path for UDR computation (e.g., 'microsoft/DialoGPT-medium')",
    )
    audit_parser.add_argument(
        "--base-norms-cache",
        type=str,
        default=None,
        help="Path to save/load base model norms cache (speeds up repeated audits)",
    )
    audit_parser.add_argument(
        "--no-udr",
        action="store_true",
        help="Skip UDR computation even if base model available",
    )
    audit_parser.add_argument(
        "--rank-policies",
        type=str,
        default="energy@0.90,knee,erank",
        help="Rank selection policies to apply. Can be comma-separated (e.g., energy@0.90,knee,erank) or space-separated. "
        "Available: energy@0.90, energy@0.95, knee, erank, oht. Default: %(default)s",
    )
    # Importance threshold configuration
    audit_parser.add_argument(
        "--importance-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold for energy share filtering (default: 0.75 = top quartile). "
        "Layers must capture above this quantile of adapter's energy to be flagged as important.",
    )
    audit_parser.add_argument(
        "--importance-uniform-mult-gate",
        type=float,
        default=1.5,
        help="Uniform multiplier gate threshold (default: 1.5). "
        "Layers must have uniform_mult >= this value to be flagged as high-impact. "
        "Prevents false positives when importance distributions are flat.",
    )
    audit_parser.add_argument(
        "--importance-metric",
        choices=["energy_share", "frobenius_norm", "param_weighted"],
        default="energy_share",
        help="Metric used for energy importance calculation (default: energy_share). "
        "energy_share: Fraction of adapter's update energy (recommended). "
        "frobenius_norm: Raw ||ΔW||_F values. "
        "param_weighted: Weighted by parameter count and utilization.",
    )
    audit_parser.add_argument(
        "--disagreement-rationale",
        choices=["full", "flagged_only"],
        default="flagged_only",
        help="Detail level for JSON rationale output (default: flagged_only). "
        "flagged_only: Full rationale for flagged layers, condensed for non-flagged (reduces JSON size). "
        "full: Complete rationale for all layers (verbose, good for debugging).",
    )
    audit_parser.set_defaults(func=cmd_audit)


def _setup_audit_adapter_command(subparsers):
    p = subparsers.add_parser(
        "audit-adapter",
        help="[RECOMMENDED] Audit one adapter and produce a QA eligibility artifact",
    )
    p.add_argument(
        "--peft-dir",
        type=str,
        required=True,
        help="Path to a PEFT output directory (containing adapter_config.* and adapter weights)",
    )
    p.add_argument(
        "--adapter-config",
        type=str,
        default=None,
        help="Optional explicit path to adapter_config.json/yaml",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional explicit path to adapter_model.(safetensors|bin|pt)",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID or path (for metadata and optional UDR)",
    )
    p.add_argument(
        "--base-norms-cache",
        type=str,
        default=None,
        help="Path to save/load base model norms cache",
    )
    p.add_argument(
        "--no-udr",
        action="store_true",
        help="Skip UDR computation even if base model available",
    )
    # Behavioral evaluation args (all optional)
    p.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="Dataset used for evaluation (e.g. 'oasst2')",
    )
    p.add_argument(
        "--metric-name",
        type=str,
        default=None,
        help="Evaluation metric name (e.g. 'perplexity', 'accuracy')",
    )
    p.add_argument(
        "--adapter-score",
        type=float,
        default=None,
        help="Adapter's score on the evaluation metric",
    )
    p.add_argument(
        "--base-score",
        type=float,
        default=None,
        help="Base model's score on the evaluation metric",
    )
    p.add_argument(
        "--lower-is-better",
        action="store_true",
        default=True,
        help="Metric direction: lower values are better (default: true)",
    )
    p.add_argument(
        "--higher-is-better",
        action="store_true",
        default=False,
        help="Metric direction: higher values are better",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Tolerance margin for eligibility classification (default: 0.0)",
    )
    # Output args
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write QA artifact JSON to this path (overwrites existing file)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print QA artifact JSON to stdout instead of terminal summary",
    )

    # Handle --higher-is-better overriding --lower-is-better
    def _resolve_and_run(args):
        if getattr(args, "higher_is_better", False):
            args.lower_is_better = False
        cmd_audit_adapter(args)

    p.set_defaults(func=_resolve_and_run)



def setup_audit_commands(subparsers) -> None:
    """Register audit commands with the argument parser."""
    _setup_audit_command(subparsers)
    _setup_audit_adapter_command(subparsers)
