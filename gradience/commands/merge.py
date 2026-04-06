"""Merge commands — merge audit, plan, execute, and explain."""

from __future__ import annotations

import argparse
import json
import json as jsonlib
from pathlib import Path
from typing import Any

from gradience.cli_utils import _load_source_qa
from gradience.exceptions import ConfigError, DependencyError, GradienceError, MergeError, QASchemaError


def cmd_merge_audit(args: argparse.Namespace) -> None:
    """Run merge compatibility audit on two PEFT LoRA adapters."""

    adapter_a = getattr(args, "adapter_a", None)
    adapter_b = getattr(args, "adapter_b", None)

    if not adapter_a or not adapter_b:
        raise ConfigError("--adapter-a and --adapter-b are both required")

    # Validate paths
    for label, path_str in [("adapter-a", adapter_a), ("adapter-b", adapter_b)]:
        p = Path(path_str)
        if not p.is_dir():
            raise ConfigError(f"--{label} path does not exist or is not a directory: {p}")

    try:
        from gradience.vnext.merge import VerdictThresholds, merge_audit
    except ImportError as e:
        raise DependencyError(f"Failed to import merge audit module: {e}") from e

    # Resolve thresholds preset
    thresholds_name = getattr(args, "thresholds", "default")
    if thresholds_name == "conservative":
        thresholds = VerdictThresholds.conservative()
    elif thresholds_name == "permissive":
        thresholds = VerdictThresholds.permissive()
    else:
        thresholds = VerdictThresholds()

    # Load source QA files if provided
    source_qa_a = _load_source_qa(getattr(args, "source_a_qa", None))
    source_qa_b = _load_source_qa(getattr(args, "source_b_qa", None))

    try:
        report = merge_audit(
            adapter_a_dir=adapter_a,
            adapter_b_dir=adapter_b,
            output_dir=getattr(args, "output_dir", None),
            energy_threshold=float(getattr(args, "energy_threshold", 0.90)),
            thresholds=thresholds,
            compute_dtype=getattr(args, "compute_dtype", "float64"),
            verbose=getattr(args, "verbose", False),
            source_qa_a=source_qa_a,
            source_qa_b=source_qa_b,
            compute_core_space=bool(getattr(args, "compute_core_space", False)),
        )
    except GradienceError:
        raise
    except (FileNotFoundError, ValueError) as e:
        raise MergeError(str(e)) from e
    except Exception as e:
        raise MergeError(f"Merge audit failed: {e}") from e

    # --- Strict QA gate ---
    strict_qa = getattr(args, "strict_qa", False)
    if strict_qa:
        from gradience.vnext.merge.recommend import diagnose_pair

        diag = diagnose_pair(report)
        if not diag.eligibility.has_data:
            raise QASchemaError(
                "--strict-qa requires source QA data for both adapters.\n"
                "  Provide --source-a-qa and --source-b-qa, or remove --strict-qa."
            )
        # Block if either adapter has no QA (null eligibility) — partial data
        if diag.eligibility.status_a is None or diag.eligibility.status_b is None:
            missing = []
            if diag.eligibility.status_a is None:
                missing.append("A")
            if diag.eligibility.status_b is None:
                missing.append("B")
            raise QASchemaError(
                f"--strict-qa requires QA data for adapter(s) {', '.join(missing)}.\n"
                "  Provide --source-a-qa and --source-b-qa, or remove --strict-qa."
            )
        if diag.eligibility.any_weak:
            weak_labels = []
            if diag.eligibility.status_a and diag.eligibility.status_a.value == "flagged_weak":
                weak_labels.append("A")
            if diag.eligibility.status_b and diag.eligibility.status_b.value == "flagged_weak":
                weak_labels.append("B")
            raise QASchemaError(
                f"--strict-qa gate failed. Adapter(s) {', '.join(weak_labels)} flagged as weak.\n"
                "  Recommendations withheld. Review source adapter quality before merging."
            )
        if diag.eligibility.any_unverified:
            from gradience.vnext.merge.eligibility import EligibilityStatus

            unverified_labels = []
            if diag.eligibility.status_a == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL:
                unverified_labels.append("A")
            if diag.eligibility.status_b == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL:
                unverified_labels.append("B")
            raise QASchemaError(
                f"--strict-qa gate failed. Adapter(s) {', '.join(unverified_labels)} have no behavioral evaluation.\n"
                "  Strict mode requires behavioral evidence for eligibility. Provide evaluation scores via audit-adapter."
            )

    # --- Output ---
    if getattr(args, "json", False):
        from gradience.vnext.merge import to_json

        print(jsonlib.dumps(to_json(report), indent=2))
        return

    # Pretty-print summary
    agg = report.aggregate
    verdict = agg["overall_verdict"].upper()

    verdict_emoji = {
        "SAFE": "\u2705",
        "REDUNDANT": "\u26a0\ufe0f",
        "CONFLICTING": "\u274c",
        "IMBALANCED": "\u2696\ufe0f",
    }
    emoji = verdict_emoji.get(verdict, "")

    print(f"\n{emoji}  Merge Compatibility: {verdict}")
    print(f"   Score: {agg['compatibility_score']:.3f}")
    print(f"   Mean overlap: {agg['mean_overlap']:.3f}  |  Max: {agg['max_overlap']:.3f}")
    print(f"   Agreement: {agg['mean_agreement']:.3f}")
    print(
        f"   Layers: {agg['n_safe']} safe, "
        f"{agg['n_redundant']} redundant, "
        f"{agg['n_conflicting']} conflicting, "
        f"{agg['n_imbalanced']} imbalanced"
    )
    over_acc_advisory = agg.get("over_accumulation_advisory", "none")
    if over_acc_advisory != "none":
        high_layers = int(agg.get("high_risk_layer_count", 0))
        watch_layers = int(agg.get("watch_layer_count", 0))
        print(f"   Over-accumulation advisory: {over_acc_advisory.upper()} (high={high_layers}, watch={watch_layers})")
        over_acc_summary = str(agg.get("over_accumulation_summary", "")).strip()
        if over_acc_summary:
            print(f"   Note: {over_acc_summary}")

    if (
        getattr(args, "compute_core_space", False)
        and not getattr(args, "qa_report", False)
        and getattr(report, "core_space", None) is not None
    ):
        core = report.core_space
        if core is not None:
            print("\nCORE-SPACE DIAGNOSTIC")
            print(f"- shared basis score: {core.shared_basis_score_mean:.2f}")
            print(f"- basis distortion: {core.basis_distortion_mean:.2f}")
            print(f"- effective shared rank: {core.effective_shared_rank_median}")
            print(f"- status: {core.status}")

    # --- Strategy Recommendations ---
    user_strategy = getattr(args, "strategy", None)
    try:
        from gradience.vnext.merge.recommend import format_recommendation, recommend_merge

        merge_rec = recommend_merge(report)
        rec_output = format_recommendation(
            merge_rec,
            adapter_a_path=adapter_a,
            adapter_b_path=adapter_b,
        )
        print(rec_output)

        # Highlight user-specified strategy if different from recommendation
        if user_strategy and user_strategy != merge_rec.overall_strategy:
            print(f"\n  User-specified strategy: {user_strategy}")
            print(f"    $ gradience merge-plan --strategy {user_strategy} \\")
            print(f"        --adapter-a {adapter_a} --adapter-b {adapter_b} \\")
            print("        --output merge_plan.json")
            print()
    except Exception:
        # Fall back to old-style recommendations if recommend module fails
        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"  \u2022 {rec}")

    if report.warnings:
        print("\nWarnings:")
        for warn in report.warnings:
            print(f"  \u26a0 {warn}")

    # --- QA Report ---
    qa_for_emit = None
    if getattr(args, "qa_report", False):
        try:
            from gradience.vnext.merge.qa_report import build_qa_report, format_qa_report

            qa = build_qa_report(report)
            qa_for_emit = qa
            print(format_qa_report(qa))
            out_dir_qa = getattr(args, "output_dir", None)
            if out_dir_qa:
                qa_path = Path(out_dir_qa) / "merge_qa_report.json"
                qa.to_json(qa_path)
        except Exception as exc:
            if getattr(args, "verbose", False):
                import traceback

                traceback.print_exc()
            else:
                print(f"\n  (QA report generation failed: {exc})")

    # --- Emit structured report ---
    emit_path = getattr(args, "emit_report", None)
    if emit_path:
        if not qa_for_emit:
            from gradience.vnext.merge.qa_report import build_qa_report as _build_qa

            qa_for_emit = _build_qa(report)
        emit_p = Path(emit_path)
        emit_p.parent.mkdir(parents=True, exist_ok=True)
        with open(emit_p, "w") as f:
            jsonlib.dump(qa_for_emit.to_dict(), f, indent=2)
        print(f"\nMerge QA report written to: {emit_p}")

    out_dir = getattr(args, "output_dir", None)
    if out_dir:
        report_files = "merge_audit.{json,md}"
        if getattr(args, "qa_report", False):
            report_files = "merge_audit.{json,md}, merge_qa_report.json"
        print(f"\nReports written to: {out_dir}/{report_files}")

    print()


# ---------------------------------------------------------------------------
# merge-plan
# ---------------------------------------------------------------------------


def cmd_merge_plan(args: argparse.Namespace) -> None:
    """Generate a merge plan from two PEFT LoRA adapters."""
    adapter_a = getattr(args, "adapter_a", None)
    adapter_b = getattr(args, "adapter_b", None)
    strategy = getattr(args, "strategy", "uniform_linear")
    output_dir = getattr(args, "output_dir", None)
    output_rank = int(getattr(args, "output_rank", 8))
    output_alpha = float(getattr(args, "output_alpha", 16.0))
    verbose = getattr(args, "verbose", False)

    if not adapter_a or not adapter_b:
        raise ConfigError("--adapter-a and --adapter-b are both required")

    for label, path_str in [("adapter-a", adapter_a), ("adapter-b", adapter_b)]:
        p = Path(path_str)
        if not p.is_dir():
            raise ConfigError(f"--{label} path does not exist or is not a directory: {p}")

    if output_dir is None:
        raise ConfigError("--output-dir is required")

    try:
        from gradience.vnext.merge import PLAN_STRATEGIES, merge_audit, plan_from_audit
    except ImportError as e:
        raise DependencyError(f"Failed to import merge modules: {e}") from e

    if strategy not in PLAN_STRATEGIES:
        raise ConfigError(f"Unknown strategy '{strategy}'. Available: {sorted(PLAN_STRATEGIES.keys())}")

    # Step 1: Run merge audit
    if verbose:
        print("Running merge audit...")
    try:
        report = merge_audit(
            adapter_a_dir=adapter_a,
            adapter_b_dir=adapter_b,
            verbose=verbose,
        )
    except GradienceError:
        raise
    except (FileNotFoundError, ValueError) as e:
        raise MergeError(f"Error during audit: {e}") from e

    # Step 2: Generate plan
    if verbose:
        print(f"\nGenerating merge plan (strategy={strategy})...")

    kwargs: dict[str, Any] = {
        "output_rank": output_rank,
        "output_alpha": output_alpha,
    }
    plan = plan_from_audit(strategy, report, adapter_a, adapter_b, **kwargs)

    # Step 3: Write plan
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "merge_plan.json"
    plan.to_json(plan_path)

    print("\nMerge plan generated:")
    print(f"  Strategy: {plan.strategy_name}")
    print(f"  Layers: {len(plan.layer_configs)}")
    print(f"  Output rank: {plan.output_rank}")
    print(f"  Output alpha: {plan.output_alpha}")
    print(f"  Plan file: {plan_path}")
    print(f"\nTo execute: gradience merge --plan {plan_path} --output-dir <DIR>")


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def cmd_merge(args: argparse.Namespace) -> None:
    """Execute a merge plan to produce a PEFT-compatible adapter."""
    plan_path = getattr(args, "plan", None)
    output_dir = getattr(args, "output_dir", None)
    compute_dtype = getattr(args, "compute_dtype", "float64")
    verbose = getattr(args, "verbose", False)

    if not plan_path:
        raise ConfigError("--plan is required")

    p = Path(plan_path)
    if not p.is_file():
        raise ConfigError(f"Plan file not found: {p}")

    if output_dir is None:
        raise ConfigError("--output-dir is required")

    try:
        from gradience.vnext.merge import MergePlan, execute_merge
    except ImportError as e:
        raise DependencyError(f"Failed to import merge modules: {e}") from e

    # Load plan
    try:
        plan = MergePlan.from_json(p)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ConfigError(f"Failed to parse merge plan: {e}") from e

    if verbose:
        print(f"Plan: {plan.strategy_name}")
        print(f"  Adapter A: {plan.adapter_a_dir}")
        print(f"  Adapter B: {plan.adapter_b_dir}")
        print(f"  Layers: {len(plan.layer_configs)}")
        print(f"  Output rank: {plan.output_rank}")

    # Execute merge
    try:
        result = execute_merge(
            plan,
            output_dir,
            compute_dtype=compute_dtype,
            verbose=verbose,
        )
    except GradienceError:
        raise
    except (FileNotFoundError, ValueError) as e:
        raise MergeError(f"Error during merge: {e}") from e
    except Exception as e:
        raise MergeError(f"Merge failed: {e}") from e

    print("\nMerge complete:")
    print(f"  Output: {result.output_dir}")
    print(f"  Mean reconstruction error: {result.mean_reconstruction_error:.4f}")
    print(f"  Max reconstruction error: {result.max_reconstruction_error:.4f}")
    print(f"  Time: {result.total_time_seconds:.1f}s")
    print("\nOutput files:")
    print(f"  {result.output_dir / 'adapter_config.json'}")
    print(f"  {result.output_dir / 'adapter_model.safetensors'}")
    print(f"  {result.output_dir / 'merge_result.json'}")


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def cmd_explain(args: argparse.Namespace) -> None:
    """Explain disagreement analysis for a specific layer from audit JSON."""

    audit_json_path = getattr(args, "audit_json", None)
    layer_name = getattr(args, "layer", None)
    verbose = getattr(args, "verbose", False)

    if not audit_json_path:
        raise ConfigError("--audit-json is required")

    if not layer_name:
        raise ConfigError("--layer is required")

    # Load audit JSON
    try:
        with open(audit_json_path) as f:
            audit_data = jsonlib.load(f)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        raise ConfigError(f"Error loading audit JSON: {e}") from e

    # Extract policy disagreement analysis
    disagreement_analysis = audit_data.get("policy_disagreement_analysis")
    if not disagreement_analysis:
        raise ConfigError(
            "No 'policy_disagreement_analysis' found in audit JSON.\n"
            "Make sure the audit was run with policy disagreement analysis enabled"
        )

    # Look for the layer in both flagged and all layers
    layer_data = None
    is_flagged = False

    # Check flagged layers first
    flagged_layers = disagreement_analysis.get("flagged_layers", [])
    for layer in flagged_layers:
        if layer.get("layer_name") == layer_name:
            layer_data = layer
            is_flagged = True
            break

    # If not found, check all layers with disagreement
    if not layer_data:
        all_layers = disagreement_analysis.get("all_layers_with_disagreement", [])
        for layer in all_layers:
            if layer.get("layer_name") == layer_name:
                layer_data = layer
                is_flagged = False
                break

    if not layer_data:
        available_lines = []
        all_layers = disagreement_analysis.get("all_layers_with_disagreement", [])
        if all_layers:
            for layer in all_layers:
                status = (
                    "FLAGGED"
                    if layer.get("layer_name") in [lyr.get("layer_name") for lyr in flagged_layers]
                    else "not flagged"
                )
                available_lines.append(f"  {layer.get('layer_name')} ({status})")
        available = "\n".join(available_lines) if available_lines else "  (No layers with disagreement found)"
        raise ConfigError(
            f"Layer '{layer_name}' not found in disagreement analysis.\n"
            f"Available layers:\n{available}"
        )

    # Extract rationale
    rationale = layer_data.get("flagging_rationale", {})
    if not rationale:
        raise ConfigError(f"No flagging rationale found for layer '{layer_name}'")

    # Display explanation
    _display_layer_explanation(layer_name, rationale, is_flagged, disagreement_analysis, verbose)


def _display_layer_explanation(layer_name: str, rationale: dict, is_flagged: bool, analysis: dict, verbose: bool):
    """Display detailed explanation for a specific layer."""

    print(f"🔍 LAYER ANALYSIS: {layer_name}")
    print("=" * 80)

    # Flagging status
    status_emoji = "🔥" if is_flagged else "○"
    status_text = "HIGH-IMPACT (flagged)" if is_flagged else "not flagged"
    print(f"Status: {status_emoji} {status_text}")

    if is_flagged:
        focus_set = analysis.get("disagreement_focus_set", {})
        high_impact_layers = focus_set.get("high_impact_layers", [])
        if layer_name in high_impact_layers:
            priority_rank = high_impact_layers.index(layer_name) + 1
            print(f"Focus Priority: #{priority_rank} of {len(high_impact_layers)} high-impact layers")

    print()

    # Policy disagreement summary
    k_values = rationale.get("k_values", [])
    policies = rationale.get("policies", [])
    spread = rationale.get("spread", 0)

    if k_values and policies:
        print("📊 POLICY DISAGREEMENT:")
        print(f"  Spread: {spread} (max - min rank suggestions)")
        print("  Policy suggestions:")
        for policy, k in zip(policies, k_values):
            print(f"    • {policy}: rank {k}")
        print(f"  Range: {min(k_values)} - {max(k_values)}")

    print()

    # Threshold analysis
    print("🎯 THRESHOLD ANALYSIS:")
    _display_threshold_checks(rationale, verbose)

    print()

    # Importance metrics
    print("⚡ IMPORTANCE METRICS:")
    _display_importance_metrics(rationale, analysis, verbose)

    # Priority score
    priority_score = rationale.get("priority_score")
    if priority_score is not None:
        print(f"\n🎯 PRIORITY SCORE: {priority_score:.2f}")
        print("  (Higher = more urgent for Bench validation)")
        print(f"  Formula: spread_norm × uniform_mult = {priority_score:.2f}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    _display_recommendations(layer_name, rationale, is_flagged, analysis)


def _display_threshold_checks(rationale: dict, verbose: bool):
    """Display threshold check results."""

    checks = [
        {
            "name": "Spread Threshold",
            "value": rationale.get("spread"),
            "threshold": rationale.get("spread_threshold"),
            "meets": rationale.get("meets_spread_threshold"),
            "description": "Policy disagreement magnitude",
        },
        {
            "name": "Uniform Mult Threshold",
            "value": rationale.get("uniform_mult"),
            "threshold": rationale.get("uniform_mult_threshold"),
            "meets": rationale.get("meets_uniform_mult_threshold"),
            "description": "Energy significance vs uniform distribution",
        },
    ]

    if rationale.get("meets_quantile_threshold") is not None:
        checks.append(
            {
                "name": "Quantile Threshold",
                "value": rationale.get("importance_share"),
                "threshold": rationale.get("energy_quantile_threshold"),
                "meets": rationale.get("meets_quantile_threshold"),
                "description": "Energy share percentile ranking",
            }
        )

    for check in checks:
        if check["value"] is None or check["threshold"] is None:
            continue

        status = "✅ PASS" if check["meets"] else "❌ FAIL"
        value = check["value"]
        threshold = check["threshold"]

        if isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)

        if isinstance(threshold, float):
            threshold_str = f"{threshold:.3f}"
        else:
            threshold_str = str(threshold)

        print(f"  {check['name']}: {status}")
        print(f"    Value: {value_str}, Threshold: ≥{threshold_str}")

        if verbose:
            print(f"    Description: {check['description']}")

    # Additional context for flat distributions
    is_flat = rationale.get("is_flat_distribution", False)
    if is_flat:
        print("  📊 Distribution: FLAT (no clear importance hierarchy)")
        print("    → Quantile thresholds not applicable")


def _display_importance_metrics(rationale: dict, analysis: dict, verbose: bool):
    """Display importance and energy metrics."""

    importance_share = rationale.get("importance_share")
    uniform_mult = rationale.get("uniform_mult")

    if importance_share is not None:
        print(f"  Energy Share: {importance_share:.1%} of total adapter energy")

    if uniform_mult is not None:
        uniform_share = analysis.get("distribution", {}).get("uniform_share")
        if uniform_share:
            expected_share = uniform_share * 100
            actual_share = importance_share * 100 if importance_share else 0
            print(f"  Uniform Multiplier: {uniform_mult:.2f}×")
            print(f"    Expected share: {expected_share:.1f}% (uniform)")
            print(f"    Actual share: {actual_share:.1f}%")

    if verbose:
        distribution = analysis.get("distribution", {})
        total_energy = distribution.get("total_energy")
        max_uniform_mult = distribution.get("max_uniform_mult")
        is_flat = distribution.get("is_flat")

        print(f"  Total Adapter Energy: {total_energy:.1f}")
        print(f"  Max Uniform Mult: {max_uniform_mult:.2f}×")
        print(f"  Distribution Type: {'FLAT' if is_flat else 'HIERARCHICAL'}")


def _display_recommendations(layer_name: str, rationale: dict, is_flagged: bool, analysis: dict):
    """Display specific recommendations for this layer."""

    if is_flagged:
        print("  🔥 HIGH PRIORITY: Include in focused Bench validation")
        print("     This layer shows both high disagreement AND high importance")

        # Get suggested rank
        k_values = rationale.get("k_values", [])
        if k_values:
            suggested_rank = max(k_values)  # Use highest suggested rank as conservative choice
            print(f"     Suggested rank: {suggested_rank} (conservative choice from policy range)")
    else:
        failed_reasons = rationale.get("failed_reasons", [])
        if failed_reasons:
            print("  ○ NOT FLAGGED: Layer did not meet high-impact criteria")
            print(f"     Failure reasons: {', '.join(failed_reasons)}")
        else:
            # Full rationale available
            print("  ○ NOT FLAGGED: Layer did not pass all thresholds")

        print("     Consider uniform rank suggestion instead of per-layer optimization")

    # Focus set context
    focus_set = analysis.get("disagreement_focus_set", {})
    strategy = focus_set.get("focus_strategy")
    message = focus_set.get("message")

    if strategy and message:
        print(f"  📋 Focus Strategy: {strategy}")
        print(f"     {message}")


# ---------------------------------------------------------------------------
# CLI subcommand setup helpers
# ---------------------------------------------------------------------------



def _setup_merge_audit_command(subparsers):
    merge_audit_parser = subparsers.add_parser(
        "merge-audit",
        help="[RECOMMENDED] Audit one adapter pair and produce merge-risk output",
    )
    merge_audit_parser.add_argument(
        "--adapter-a",
        type=str,
        required=True,
        help="Path to the first PEFT adapter directory",
    )
    merge_audit_parser.add_argument(
        "--adapter-b",
        type=str,
        required=True,
        help="Path to the second PEFT adapter directory",
    )
    merge_audit_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write merge_audit.json and merge_audit.md",
    )
    merge_audit_parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.90,
        help="Energy threshold for effective rank computation (default: 0.90)",
    )
    merge_audit_parser.add_argument(
        "--thresholds",
        choices=["default", "conservative", "permissive"],
        default="default",
        help="Verdict threshold preset (default: default)",
    )
    merge_audit_parser.add_argument(
        "--compute-dtype",
        choices=["float64", "float32", "fp64", "fp32"],
        default="float64",
        help="Precision for SVD computation (default: float64)",
    )
    merge_audit_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of pretty text",
    )
    merge_audit_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-layer analysis table and progress",
    )
    merge_audit_parser.add_argument(
        "--qa-report",
        action="store_true",
        help="Print a concise QA report summarizing risk, dominant issue, and recommended action",
    )
    merge_audit_parser.add_argument(
        "--source-a-qa",
        type=str,
        default=None,
        help="Path to a JSON file with prior QA results for adapter A (AdapterQAResult format)",
    )
    merge_audit_parser.add_argument(
        "--source-b-qa",
        type=str,
        default=None,
        help="Path to a JSON file with prior QA results for adapter B (AdapterQAResult format)",
    )
    merge_audit_parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Recommended merge strategy to highlight (e.g. norm_equalized, audit_aware)",
    )
    merge_audit_parser.add_argument(
        "--emit-report",
        type=str,
        default=None,
        help="Write structured JSON report to this path (overwrites existing file)",
    )
    merge_audit_parser.add_argument(
        "--strict-qa",
        action="store_true",
        help="Refuse to produce recommendations when source QA data is missing or shows weak adapters",
    )
    merge_audit_parser.add_argument(
        "--compute-core-space",
        action="store_true",
        help="[ADVANCED/EXPERIMENTAL] Compute optional shared-basis diagnostics and include them in QA report output",
    )
    merge_audit_parser.set_defaults(func=cmd_merge_audit)


def _setup_merge_plan_command(subparsers):
    merge_plan_parser = subparsers.add_parser(
        "merge-plan",
        help="[ADVANCED] Generate a merge plan from two PEFT LoRA adapters",
    )
    merge_plan_parser.add_argument(
        "--adapter-a",
        type=str,
        required=True,
        help="Path to the first PEFT adapter directory",
    )
    merge_plan_parser.add_argument(
        "--adapter-b",
        type=str,
        required=True,
        help="Path to the second PEFT adapter directory",
    )
    merge_plan_parser.add_argument(
        "--strategy",
        type=str,
        default="uniform_linear",
        choices=["uniform_linear", "audit_aware", "norm_equalized", "overlap_ties"],
        help="Merge planning strategy (default: uniform_linear)",
    )
    merge_plan_parser.add_argument(
        "--output-rank",
        type=int,
        default=8,
        help="Target rank for the merged LoRA adapter (default: 8)",
    )
    merge_plan_parser.add_argument(
        "--output-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha for the merged adapter (default: 16.0)",
    )
    merge_plan_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write merge_plan.json",
    )
    merge_plan_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress during audit and planning",
    )
    merge_plan_parser.set_defaults(func=cmd_merge_plan)


def _setup_merge_command(subparsers):
    merge_parser = subparsers.add_parser(
        "merge",
        help="[ADVANCED] Execute a merge plan to produce a PEFT-compatible adapter",
    )
    merge_parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to merge_plan.json file",
    )
    merge_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write merged adapter",
    )
    merge_parser.add_argument(
        "--compute-dtype",
        choices=["float64", "float32", "fp64", "fp32"],
        default="float64",
        help="Precision for SVD computation (default: float64)",
    )
    merge_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-layer progress during merge",
    )
    merge_parser.set_defaults(func=cmd_merge)


def _setup_explain_command(subparsers):
    explain_parser = subparsers.add_parser(
        "explain", help="[ADVANCED] Explain disagreement analysis for a specific layer from audit JSON"
    )
    explain_parser.add_argument(
        "--audit-json", type=str, required=True, help="Path to audit JSON file containing policy_disagreement_analysis"
    )
    explain_parser.add_argument(
        "--layer", type=str, required=True, help="Layer name to explain (e.g., 'model.layers.0.self_attn.q_proj')"
    )
    explain_parser.add_argument("--verbose", action="store_true", help="Show detailed thresholds and calculations")
    explain_parser.set_defaults(func=cmd_explain)



def setup_merge_commands(subparsers) -> None:
    """Register merge commands with the argument parser."""
    _setup_merge_audit_command(subparsers)
    _setup_merge_plan_command(subparsers)
    _setup_merge_command(subparsers)
    _setup_explain_command(subparsers)
