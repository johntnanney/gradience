"""
Report generation for bench protocol.

Creates canonical JSON and human-readable Markdown reports
for both single-seed and multi-seed bench runs.

Extracted from protocol.py.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

from gradience.bench.decision_trace import DecisionTrace
from gradience.bench.types import EnvironmentInfo
from gradience.bench.escalation import EscalationTrace
from gradience.bench.metadata import (
    gather_environment_info, get_git_commit, get_git_tag,
    extract_model_dataset_info,
)
from gradience.bench._util import (
    get_primary_metric_key, create_config_hash,
)
from gradience.bench.compression import get_rank_source_from_config
from gradience.bench.constants import (
    PASS_RATE_VALID, PASS_RATE_AGGRESSIVE, DEFAULT_ACCURACY_TOLERANCE,
    CONCENTRATION_INDEX_HIGH, CONCENTRATION_INDEX_MODERATE,
    MIN_SEEDS_CERTIFIABLE, MIN_STEPS_CERTIFIABLE,
    MIN_SEEDS_SCREENING_PLUS, MIN_STEPS_SCREENING_PLUS,
    MIN_SEEDS_SUFFICIENT_POWER,
)


def write_probe_eval_json(
    probe_dir: Path,
    eval_results: Dict[str, Any],
    eval_dataset_size: int,
    config: Dict[str, Any]
) -> Path:
    """
    Step 3.2: Write probe_r{rank}/eval.json with evaluation results.

    Args:
        probe_dir: Directory where eval.json should be written
        eval_results: Results from trainer.evaluate()
        eval_dataset_size: Number of evaluation samples used
        config: Benchmark configuration

    Returns:
        Path to the written eval.json file
    """
    # Use robust metric extraction with fallback
    accuracy = _extract_accuracy_with_fallback(eval_results)

    eval_data = {
        "accuracy": accuracy,
        "eval_loss": eval_results.get("eval_loss"),
        "eval_samples": eval_dataset_size,
        "seed": config["train"]["seed"],
        "rank": config["lora"]["probe_r"],
        "eval_runtime": eval_results.get("eval_runtime"),
        "eval_samples_per_second": eval_results.get("eval_samples_per_second"),
        "eval_steps_per_second": eval_results.get("eval_steps_per_second")
    }

    # Add task-specific metrics
    if "eval_exact_match" in eval_results:
        eval_data["exact_match"] = eval_results["eval_exact_match"]
        eval_data["eval_exact_match"] = eval_results["eval_exact_match"]  # Preserve original key for probe_gate
    if "eval_correct" in eval_results:
        eval_data["correct"] = eval_results["eval_correct"]
        eval_data["eval_correct"] = eval_results["eval_correct"]  # Preserve original key for probe_gate
    if "eval_total" in eval_results:
        eval_data["total"] = eval_results["eval_total"]
        eval_data["eval_total"] = eval_results["eval_total"]  # Preserve original key for probe_gate

    eval_path = probe_dir / "eval.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    return eval_path


def _extract_accuracy_with_fallback(eval_results: Dict[str, Any], task_profile=None) -> float:
    """
    Extract accuracy metric from evaluation results with robust fallback.

    Priority:
    1. task_profile.primary_metric_key (if available)
    2. Fallback sequence: eval_accuracy, eval_exact_match, accuracy, exact_match

    Args:
        eval_results: Dictionary of evaluation metrics
        task_profile: TaskProfile instance (optional)

    Returns:
        float: Accuracy value (0.0 if not found)
    """
    # Try task profile primary metric key first
    if task_profile and hasattr(task_profile, 'primary_metric_key'):
        primary_key = task_profile.primary_metric_key
        if primary_key in eval_results:
            return eval_results[primary_key]

    # Fallback sequence
    fallback_keys = ["eval_accuracy", "eval_exact_match", "accuracy", "exact_match"]
    for key in fallback_keys:
        if key in eval_results:
            return eval_results[key]

    return 0.0


def create_canonical_bench_report(
    probe_results: Dict[str, Any],
    variant_results: Dict[str, Dict[str, Any]],
    verdict_analysis: Dict[str, Any],
    audit_data: Dict[str, Any],
    compression_configs: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
    decision_trace: Optional[DecisionTrace] = None,
    escalation_trace: Optional[EscalationTrace] = None,
) -> Dict[str, Any]:
    """
    Create the canonical bench.json report according to specification.
    """

    # Gather metadata
    timestamp = datetime.datetime.now().isoformat()
    git_commit = get_git_commit()
    git_tag = get_git_tag()
    env_info = gather_environment_info()

    # Add git information to environment
    env_info["git_commit"] = git_commit
    env_info["git_tag"] = git_tag

    # Extract model and dataset revision information
    model_dataset_metadata = extract_model_dataset_info(config)
    env_info.update(model_dataset_metadata)

    # Add validation classification to environment info
    validation_classification = verdict_analysis.get("validation_classification", {})
    env_info["validation_classification"] = validation_classification

    # Check if probe was undertrained
    probe_quality_status = verdict_analysis.get("probe_quality_status")

    if probe_quality_status in ["UNDERTRAINED", "UNDERTRAINED_SMOKE"]:
        # Create minimal bench.json for undertrained probe
        probe_data = probe_results.get("probe", {})

        # Add instrumentation sections if available (even for undertrained probes)
        instrumentation = {}

        # UDR instrumentation (if present)
        udr_instrumentation = audit_data.get("udr_instrumentation")
        if udr_instrumentation:
            instrumentation["udr"] = udr_instrumentation

        # Composition analysis (if enabled in config)
        composition_data = audit_data.get("composition")
        if composition_data:
            instrumentation["composition"] = composition_data

        # Extract seed from config for canonical placement
        _seed = config.get("train", {}).get("seed")

        minimal_report = {
            "bench_version": config.get("bench_version", "0.1"),
            "timestamp": timestamp,
            "seed": _seed,
            "git_commit": git_commit,
            "env": env_info,
            "model": config["model"]["name"],
            "task": f"{config['task']['dataset']}/{config['task']['subset']}",
            "status": probe_quality_status,
            "probe_quality_gate": {
                "metric_key": "eval_exact_match" if config.get("task", {}).get("dataset", "").lower() == "gsm8k" else "eval_accuracy",
                "metric_value": probe_data.get("accuracy"),
                "min_value": verdict_analysis.get("summary", {}).get("probe_threshold", 0.1),
                "passed": False
            },
            "probe": {
                "rank": probe_data.get("rank"),
                "params": probe_data.get("params"),
                "accuracy": probe_data.get("accuracy"),
                "threshold_required": verdict_analysis.get("summary", {}).get("probe_threshold")
            },
            "compressed": {},
            "summary": {
                "probe_quality": "FAILED",
                "recommendations_validated": "N/A",
                "best_compression": None,
                "notes": verdict_analysis.get("summary", {}).get("notes", [])
            },
            "config_metadata": {
                "primary_metric_key": get_primary_metric_key(config),
                "config_hash": create_config_hash(config),
                "seed": _seed,
                "embedded_config": config  # Complete configuration for reproducibility
            }
        }

        # Add instrumentation if available
        if instrumentation:
            minimal_report["instrumentation"] = instrumentation

        return minimal_report

    # Extract probe summary metrics from audit
    probe_summary = audit_data.get("summary", {})
    probe_baseline = verdict_analysis["probe_baseline"]

    # Build compressed section
    compressed = {}
    for variant_name, result in variant_results.items():
        if result["status"] == "completed":
            verdict_info = verdict_analysis["verdicts"][variant_name]

            if variant_name in ["per_layer", "per_layer_shuffled"]:
                # Count non-default ranks in the pattern from compression_configs
                compression_config = compression_configs.get(variant_name, {})
                rank_pattern = compression_config.get("rank_pattern", {})
                rank_pattern_nondefault = len([r for r in rank_pattern.values() if r > 0]) if rank_pattern else result.get("rank", 0)

                compressed[variant_name] = {
                    "rank_pattern_nondefault": rank_pattern_nondefault,
                    "params": result["params"],
                    "accuracy": result["accuracy"],
                    "delta_vs_probe": verdict_info["delta_vs_probe"],
                    "param_reduction": verdict_info["param_reduction"],
                    "verdict": verdict_info["verdict"]
                }

                # Include rank_check results if available
                if "rank_check" in result:
                    rank_check = result["rank_check"]
                    # Only include essential fields for the canonical report
                    compressed[variant_name]["rank_check"] = {
                        "passed": rank_check.get("passed"),
                        "unique_ranks": rank_check.get("unique_ranks"),
                        "rank_histogram": rank_check.get("rank_histogram"),
                        "total_modules": rank_check.get("total_modules")
                    }
            elif result.get("compression_method") == "svd_truncation":
                # SVD truncation variants
                compression_config = compression_configs.get(variant_name, {})

                # Build compression metadata as requested in Step 3.4
                compression_metadata = {
                    "method": "svd_truncate",
                    "policy_origin": compression_config.get("policy_type", "unknown"),
                    "rank_source": get_rank_source_from_config(compression_config),
                    "target_rank": result["rank"],
                    "source_rank": result.get("source_rank"),
                    "alpha_mode": "keep_ratio",  # Currently hardcoded, could be configurable
                    "energy_retained": result.get("energy_retained"),
                    "compression_ratio": result.get("compression_ratio"),
                    "truncation_modules": result.get("truncation_modules"),
                    "retained_energy_mean": result.get("energy_retained")  # Placeholder for future use
                }

                # Add post-tuning info if applicable
                if result.get("post_tuned", False):
                    compression_metadata["post_tune"] = result.get("post_tune_config", {
                        "enabled": True,
                        "steps": 100,  # Default fallback
                        "lr_scale": 0.1
                    })
                else:
                    compression_metadata["post_tune"] = {"enabled": False}

                compressed[variant_name] = {
                    "rank": result["rank"],
                    "params": result["params"],
                    "accuracy": result["accuracy"],
                    "delta_vs_probe": verdict_info["delta_vs_probe"],
                    "param_reduction": verdict_info["param_reduction"],
                    "verdict": verdict_info["verdict"],
                    "compression": compression_metadata
                }
            else:
                # Uniform variants (non-SVD)
                compressed[variant_name] = {
                    "rank": result["rank"],
                    "params": result["params"],
                    "accuracy": result["accuracy"],
                    "delta_vs_probe": verdict_info["delta_vs_probe"],
                    "param_reduction": verdict_info["param_reduction"],
                    "verdict": verdict_info["verdict"]
                }

            # Add stability metadata from escalation enrichment (if present)
            if "stability_status" in verdict_info:
                compressed[variant_name]["stability_status"] = verdict_info["stability_status"]
                compressed[variant_name]["failure_mode"] = verdict_info.get("failure_mode")
                if verdict_info.get("escalated_to"):
                    compressed[variant_name]["escalated_to"] = verdict_info["escalated_to"]

    # Calculate summary statistics
    completed_variants = [v for v in variant_results.values() if v["status"] == "completed"]
    passed_variants = [v for v in verdict_analysis["verdicts"].values() if v["verdict"] == "PASS"]
    recommendations_validated = f"{len(passed_variants)}/{len(completed_variants)}"

    best_compression = verdict_analysis.get("best_compression")
    best_compression_variant = best_compression["variant"] if best_compression else None

    # Construct notes
    notes = []
    if best_compression_variant == "per_layer":
        notes.append("per_layer applied successfully (verified via adapter shapes)")
    elif best_compression_variant == "per_layer_shuffled":
        notes.append("per_layer_shuffled control applied successfully")

    # Extract UDR instrumentation if available
    udr_instrumentation = {}
    if probe_summary.get("n_layers_with_udr", 0) > 0:
        udr_instrumentation = {
            "udr_median": probe_summary.get("udr_median"),
            "udr_p90": probe_summary.get("udr_p90"),
            "udr_max": probe_summary.get("udr_max"),
            "fraction_udr_gt_0_3": probe_summary.get("fraction_udr_gt_0_3"),
            "n_layers_with_udr": probe_summary.get("n_layers_with_udr")
        }

        # Add top-5 modules by UDR for debugging value
        audit_layers = audit_data.get("layers", [])
        if audit_layers:
            # Sort layers by UDR, take top 5
            layers_with_udr = [l for l in audit_layers if l.get("udr") is not None]
            layers_with_udr.sort(key=lambda x: x["udr"], reverse=True)
            top_5_modules = [
                {
                    "name": layer["name"],
                    "udr": round(layer["udr"], 4),
                    "rank": layer.get("r", "unknown")
                }
                for layer in layers_with_udr[:5]
            ]
            if top_5_modules:
                udr_instrumentation["top_modules"] = top_5_modules

    # Extract seed from config for canonical placement
    _seed = config.get("train", {}).get("seed")

    # Build the canonical report
    report = {
        "bench_version": config.get("bench_version", "0.1"),
        "timestamp": timestamp,
        "seed": _seed,
        "git_commit": git_commit,
        "env": env_info,
        "model": config["model"]["name"],
        "task": f"{config['task']['dataset']}/{config['task']['subset']}",
        "probe_quality_gate": {
            "metric_key": "eval_exact_match" if config.get("task", {}).get("dataset", "").lower() == "gsm8k" else "eval_accuracy",
            "metric_value": probe_results["probe"]["accuracy"],
            "min_value": verdict_analysis.get("summary", {}).get("probe_threshold", 0.1),
            "passed": verdict_analysis.get("probe_quality_status") not in ["UNDERTRAINED", "UNDERTRAINED_SMOKE"]
        },
        "probe": {
            "rank": probe_results["probe"]["rank"],
            "params": probe_results["probe"]["params"],
            "accuracy": probe_results["probe"]["accuracy"],
            "utilization_mean": probe_summary.get("utilization_mean"),
            "energy_rank_90_p50": probe_summary.get("energy_rank_90_p50"),
            "energy_rank_90_p90": probe_summary.get("energy_rank_90_p90"),
            "suggested_r_global_median": probe_summary.get("suggested_r_global_median"),
            "suggested_r_global_90": probe_summary.get("suggested_r_global_90")
        },
        "compressed": compressed,
        "summary": {
            "recommendations_validated": recommendations_validated,
            "best_compression": best_compression_variant,
            "notes": notes
        },
        "config_metadata": {
            "primary_metric_key": get_primary_metric_key(config),
            "config_hash": create_config_hash(config),
            "seed": _seed,
            "embedded_config": config  # Complete configuration for reproducibility
        }
    }

    # Add instrumentation sections if available
    instrumentation = {}

    # UDR instrumentation
    if udr_instrumentation:
        instrumentation["udr"] = udr_instrumentation

    # Composition analysis (if enabled in config)
    composition_data = audit_data.get("composition")
    if composition_data:
        instrumentation["composition"] = composition_data

    # Gain metrics summary
    gain_summary = audit_data.get("summary", {}).get("gain")
    if gain_summary:
        instrumentation["gain"] = gain_summary

    # Add instrumentation section if we have any data
    if instrumentation:
        report["instrumentation"] = instrumentation

    # Add decision trace for audit-driven compression decisions
    if decision_trace:
        report["decision_trace"] = decision_trace.to_dict()

    # Add escalation trace for safety fallback
    if escalation_trace and escalation_trace.triggered:
        report["escalation"] = escalation_trace.to_dict()

    # Add protocol invariants for aggregation
    probe_gate_data = report["probe_quality_gate"]
    report["protocol_invariants"] = {
        "probe_quality_gate": {
            "status": "PASSED" if probe_gate_data["passed"] else "FAILED",
            "message": f"Probe {probe_gate_data['metric_key']} {probe_gate_data['metric_value']:.4f} {'\u2265' if probe_gate_data['passed'] else '<'} {probe_gate_data['min_value']:.4f}",
            "metric_key": probe_gate_data["metric_key"],
            "metric_value": probe_gate_data["metric_value"],
            "min_value": probe_gate_data["min_value"]
        }
    }

    # Schema normalization: ensure "compressed" field is always present
    report.setdefault("compressed", {})

    return report


def create_markdown_report(
    canonical_report: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path
) -> str:
    """
    Create bench.md human-readable markdown report.
    """

    # Extract data from canonical report
    model = canonical_report["model"]
    task = canonical_report["task"]
    timestamp = canonical_report["timestamp"]
    probe_data = canonical_report["probe"]
    compressed_data = canonical_report.get("compressed", {}) or {}
    summary = canonical_report["summary"]
    instrumentation = canonical_report.get("instrumentation", {})

    # Extract validation classification
    validation_classification = canonical_report.get("env", {}).get("validation_classification", {})
    validation_level = validation_classification.get("level", "unknown")
    validation_rationale = validation_classification.get("rationale", "Not specified")

    # Build markdown content
    md_content = f"""# Gradience Bench v{canonical_report["bench_version"]}

- **Model:** {model}
- **Task:** {task}
- **Validation Level:** {validation_level.title()}
  - *{validation_rationale}*

## Probe

- **Rank:** {probe_data["rank"]}
- **LoRA params:** {probe_data["params"]:,}
- **Accuracy:** {probe_data["accuracy"]:.3f}

## Compression results

| Variant | Params | Accuracy | \u0394 vs probe | Param reduction | Verdict |
|---|---:|---:|---:|---:|---|
"""

    # Add results table rows
    for variant_name, data in compressed_data.items():
        params = f"{data['params']:,}" if data['params'] else "n/a"
        accuracy = f"{data['accuracy']:.3f}" if data['accuracy'] is not None else "n/a"
        delta = f"{data['delta_vs_probe']:+.3f}" if data['delta_vs_probe'] is not None else "n/a"
        reduction = f"{data['param_reduction']:.1%}" if data['param_reduction'] is not None else "n/a"
        verdict = data['verdict']

        # Format variant name for display
        if variant_name == "per_layer":
            variant_display = "`per_layer`"
        elif variant_name == "per_layer_shuffled":
            variant_display = "`per_layer_shuffled`"
        elif variant_name == "uniform_median":
            variant_display = "`uniform_median`"
        elif variant_name == "uniform_p90":
            variant_display = "`uniform_p90`"
        elif variant_name == "uniform_p90_control":
            variant_display = "`uniform_p90_control`"
        else:
            variant_display = f"`{variant_name}`"

        md_content += f"| {variant_display} | {params} | {accuracy} | {delta} | {reduction} | {verdict} |\n"

    # Add interpretation section
    acc_tolerance = config.get("compression", {}).get("acc_tolerance", DEFAULT_ACCURACY_TOLERANCE)

    # Create validation-level-specific interpretation
    if validation_level == "certifiable":
        interpretation_header = "## Interpretation (Certifiable)"
        validation_note = "- **Certifiable results** - Multi-seed validation with statistical rigor suitable for production decisions"
    elif validation_level == "screening_plus":
        interpretation_header = "## Interpretation (Screening+)"
        validation_note = "- **Enhanced screening** - Multi-seed but limited budget/seeds, good for development decisions"
    else:  # screening
        interpretation_header = "## Interpretation (Screening Only)"
        validation_note = "- **Screening only** - Single-seed validation, suitable for rapid development iteration"

    md_content += f"""

{interpretation_header}

{validation_note}
- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (\u00b1{acc_tolerance:.3f})
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters
"""

    # Add magnitude diagnostics if instrumentation data is available
    composition = instrumentation.get("composition", {})
    gain_summary = instrumentation.get("gain", {})

    if gain_summary or composition:
        md_content += """
## Magnitude diagnostics (LoRA \u0394W)

"""

        # Overall magnitude metrics
        if gain_summary:
            delta_fro_mean = gain_summary.get("delta_fro_mean")
            delta_op_mean = gain_summary.get("delta_op_mean")
            if delta_fro_mean is not None or delta_op_mean is not None:
                md_content += "### Update magnitude\n\n"
                if delta_fro_mean is not None:
                    md_content += f"- **Mean ||\u0394W||_F:** {delta_fro_mean:.6f}\n"
                if delta_op_mean is not None:
                    md_content += f"- **Mean ||\u0394W||_2:** {delta_op_mean:.6f}\n"
                md_content += "\n"

        # Top 5 layers by energy concentration (if composition analysis available)
        if composition and composition.get("top_k", {}).get("layers"):
            md_content += "### Top 5 layers by \u0394 energy\n\n"
            top_layers = composition["top_k"]["layers"][:5]  # Ensure max 5

            for i, layer_info in enumerate(top_layers, 1):
                layer_num = layer_info["layer"]
                share = layer_info["share"]
                energy = layer_info["energy_fro2"]
                md_content += f"{i}. **Layer {layer_num}:** {share:.1%} ({energy:.6f})\n"
            md_content += "\n"

        # Energy concentration summary (if composition analysis available)
        if composition:
            top_10pct_share = composition.get("top_10pct", {}).get("share")
            concentration_index = composition.get("concentration_index")
            if top_10pct_share is not None or concentration_index is not None:
                md_content += "### Energy concentration\n\n"
                if top_10pct_share is not None:
                    n_layers = composition.get("top_10pct", {}).get("n", 0)
                    md_content += f"- **Top-{n_layers} layers (10%):** {top_10pct_share:.1%} of energy\n"
                if concentration_index is not None:
                    md_content += f"- **Concentration index (HHI):** {concentration_index:.3f}\n"
                    # Simple interpretation
                    if concentration_index > CONCENTRATION_INDEX_HIGH:
                        md_content += "- \U0001f6a8 **Highly concentrated** adaptation\n"
                    elif concentration_index > CONCENTRATION_INDEX_MODERATE:
                        md_content += "- \u26a0\ufe0f **Moderately concentrated** adaptation\n"
                    else:
                        md_content += "- \u2705 **Well distributed** adaptation\n"
                md_content += "\n"
        elif gain_summary:
            # Show note that composition analysis was disabled
            md_content += "### Energy concentration\n\n"
            md_content += "- *Composition analysis disabled in config (audit.enable_composition_analysis: false)*\n\n"

    # Add decision trace section if available
    decision_trace_data = canonical_report.get("decision_trace")
    if decision_trace_data:
        md_content += """
## Audit-driven decisions

"""
        probe_rank = decision_trace_data.get("probe_rank", 32)
        audit_metrics = decision_trace_data.get("audit_metrics", {})
        rules_fired = decision_trace_data.get("rules_fired", [])
        rules_considered = decision_trace_data.get("rules_considered", [])

        md_content += f"- **Probe rank:** r={probe_rank}\n"

        utilization = audit_metrics.get("utilization_mean")
        stable_rank = audit_metrics.get("stable_rank_mean")
        if utilization is not None:
            md_content += f"- **Utilization mean:** {utilization:.3f}\n"
        if stable_rank is not None:
            md_content += f"- **Stable rank mean:** {stable_rank:.1f}\n"

        if rules_fired:
            md_content += "\n### Rules triggered\n\n"
            for rule in rules_fired:
                rule_id = rule.get("rule_id", "unknown")
                action = rule.get("action", "no action specified")
                evidence = rule.get("evidence", {})

                if rule_id == "tier_a_moderate_compression":
                    util_thresh = evidence.get("threshold_util", 0.55)
                    util_actual = evidence.get("utilization_mean", 0.0)
                    triggered_by_util = evidence.get("triggered_by_util", False)
                    triggered_by_suggested = evidence.get("triggered_by_suggested", False)

                    triggers = []
                    if triggered_by_util:
                        triggers.append(f"utilization {util_actual:.3f} \u2264 {util_thresh}")
                    if triggered_by_suggested:
                        triggers.append("suggested rank \u2264 0.75 \u00d7 probe rank")

                    md_content += f"- **Moderate compression:** {' OR '.join(triggers)} \u2192 {action}\n"

                elif rule_id == "tier_b_aggressive_compression":
                    util_thresh = evidence.get("threshold_util", 0.30)
                    util_actual = evidence.get("utilization_mean", 0.0)
                    stable_thresh = evidence.get("threshold_stable", 8.0)
                    stable_actual = evidence.get("stable_rank_mean", 0.0)

                    md_content += f"- **Aggressive compression:** utilization {util_actual:.3f} < {util_thresh} AND stable rank {stable_actual:.1f} \u2264 {stable_thresh:.1f} \u2192 {action}\n"

        if rules_considered and not rules_fired:
            md_content += "\n### No additional candidates added\n\n"
            for rule in rules_considered:
                rule_id = rule.get("rule_id", "unknown")
                action = rule.get("action", "")
                if "not triggered:" in action:
                    reason = action.split("not triggered: ", 1)[1]
                    if rule_id == "tier_a_moderate_compression":
                        md_content += f"- **Moderate compression not added:** {reason}\n"
                    elif rule_id == "tier_b_aggressive_compression":
                        md_content += f"- **Aggressive compression not added:** {reason}\n"

        md_content += "\n"

    # Add stability & escalation section if escalation was triggered
    escalation_data = canonical_report.get("escalation")
    if escalation_data and escalation_data.get("triggered"):
        acc_tolerance = config.get("compression", {}).get("acc_tolerance", DEFAULT_ACCURACY_TOLERANCE)

        md_content += """
## Stability & auto-escalation

"""
        for entry in escalation_data.get("escalation_trace", []):
            orig = entry["original_variant"]
            orig_rank = entry["original_rank"]
            worst_d = entry["worst_delta"]
            cat_thresh = entry["catastrophic_threshold"]

            md_content += f"- Audit candidate `{orig}` (r={orig_rank}): **unstable**\n"
            md_content += f"  - Worst delta: {worst_d:+.4f} (tolerance: {-acc_tolerance:+.4f}, catastrophic: {cat_thresh:+.4f})\n"

            esc_to = entry.get("escalated_to")
            esc_rank = entry.get("escalation_rank")
            esc_result = entry.get("escalation_result")

            if esc_to:
                result_label = f"**{esc_result}**" if esc_result else "not evaluated"
                md_content += f"- Auto-escalated to `{esc_to}` (r={esc_rank}): {result_label}\n"

                # Show the escalation candidate's delta if available
                esc_compressed = compressed_data.get(esc_to, {})
                esc_delta = esc_compressed.get("delta_vs_probe")
                if esc_delta is not None:
                    if esc_result == "PASS":
                        md_content += f"  - Delta: {esc_delta:+.4f} (within tolerance)\n"
                    else:
                        md_content += f"  - Delta: {esc_delta:+.4f}\n"
            else:
                md_content += "- No viable escalation candidate available\n"

        final_rec = escalation_data.get("final_recommendation")
        if final_rec:
            md_content += f"\n**Final recommendation:** `{final_rec}`\n"

        md_content += "\n"

    md_content += f"""
## Summary

- **Recommendations validated:** {summary["recommendations_validated"]}
- **Best compression:** {summary["best_compression"] or "None"}

*Generated on {timestamp[:19].replace('T', ' ')}*
"""

    return md_content


def create_multi_seed_aggregated_report(
    seed_reports: list[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Create aggregated report from multiple seed runs.

    Returns bench_aggregate.json format with mean ± std statistics.
    """
    import numpy as np
    from datetime import datetime

    if not seed_reports:
        raise ValueError("No seed reports provided for aggregation")

    # Extract metadata from first report
    base_report = seed_reports[0]
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Aggregate probe results
    probe_accuracies = [r["probe"]["accuracy"] for r in seed_reports]
    probe_params = [r["probe"]["params"] for r in seed_reports]

    # Calculate probe statistics
    probe_acc_mean = float(np.mean(probe_accuracies))
    probe_acc_std = float(np.std(probe_accuracies))
    probe_params_mean = float(np.mean(probe_params))  # Should be constant

    # Aggregate compressed variants
    variants_data = {}

    # Get all variant names from all reports
    all_variant_names = set()
    for report in seed_reports:
        compressed_data = report.get("compressed", {}) or {}
        all_variant_names.update(compressed_data.keys())

    for variant_name in all_variant_names:
        # Collect data for this variant across all seeds
        variant_results = []
        for report in seed_reports:
            compressed_data = report.get("compressed", {}) or {}
            if variant_name in compressed_data:
                variant_data = compressed_data[variant_name]
                if variant_data.get("accuracy") is not None:  # Only include successful runs
                    variant_results.append(variant_data)

        if not variant_results:
            continue  # Skip variants with no successful runs

        # Extract metrics
        accuracies = [v["accuracy"] for v in variant_results]
        deltas = [v["delta_vs_probe"] for v in variant_results]
        reductions = [v["param_reduction"] for v in variant_results]
        verdicts = [v["verdict"] for v in variant_results]
        params = [v["params"] for v in variant_results]

        # Calculate statistics
        acc_mean = float(np.mean(accuracies))
        acc_std = float(np.std(accuracies))
        delta_mean = float(np.mean(deltas))
        delta_std = float(np.std(deltas))
        red_mean = float(np.mean(reductions))
        red_std = float(np.std(reductions))
        params_mean = float(np.mean(params))

        # Calculate pass rate
        pass_count = sum(1 for v in verdicts if v == "PASS")
        pass_rate = pass_count / len(verdicts)

        # Overall verdict based on stringent threshold for scientific validity
        # Require at least 80% of seeds to pass for a variant to be considered valid
        # This ensures robustness and prevents cherry-picking
        overall_verdict = "PASS" if pass_rate >= PASS_RATE_VALID else "FAIL"

        # Build variant data
        variant_data = {
            "n_seeds": len(variant_results),
            "accuracy": {
                "mean": acc_mean,
                "std": acc_std,
                "values": accuracies
            },
            "delta_vs_probe": {
                "mean": delta_mean,
                "std": delta_std,
                "values": deltas
            },
            "param_reduction": {
                "mean": red_mean,
                "std": red_std,
                "values": reductions
            },
            "params": {
                "mean": params_mean,
                "std": float(np.std(params)) if len(params) > 1 else 0.0
            },
            "pass_rate": pass_rate,
            "pass_count": pass_count,
            "total_runs": len(variant_results),
            "verdict": overall_verdict,
            "individual_verdicts": verdicts
        }

        # Preserve rank information from first result (should be consistent across seeds)
        if variant_results and "rank" in variant_results[0]:
            variant_data["rank"] = variant_results[0]["rank"]

        # Preserve compression metadata if present (for SVD variants)
        if variant_results and "compression" in variant_results[0]:
            # Use compression metadata from first result (should be consistent across seeds)
            variant_data["compression"] = variant_results[0]["compression"]

            # Aggregate energy retention if present
            energy_values = [v["compression"].get("energy_retained") for v in variant_results if v.get("compression", {}).get("energy_retained") is not None]
            if energy_values:
                variant_data["compression"]["energy_retained_stats"] = {
                    "mean": float(np.mean(energy_values)),
                    "std": float(np.std(energy_values)) if len(energy_values) > 1 else 0.0,
                    "values": energy_values
                }

        variants_data[variant_name] = variant_data

    # Build detailed per-seed breakdown for self-contained reporting
    detailed_results = {
        "seeds": [],
        "candidates": {},
        "summary_statistics": {}
    }

    # Extract seed information and results
    for i, report in enumerate(seed_reports):
        seed_id = report.get("env", {}).get("seed", f"seed_{i}")
        probe_data = report["probe"]
        compressed_data = report.get("compressed", {}) or {}

        seed_detail = {
            "seed_id": seed_id,
            "probe": {
                "accuracy": probe_data["accuracy"],
                "params": probe_data["params"],
                "rank": probe_data["rank"]
            },
            "candidates": {}
        }

        # Add candidate results for this seed
        for variant_name in all_variant_names:
            if variant_name in compressed_data:
                variant_data = compressed_data[variant_name]
                seed_detail["candidates"][variant_name] = {
                    "accuracy": variant_data.get("accuracy"),
                    "params": variant_data.get("params"),
                    "rank": variant_data.get("rank", "unknown"),
                    "delta_vs_probe": variant_data.get("delta_vs_probe"),
                    "param_reduction": variant_data.get("param_reduction"),
                    "verdict": variant_data.get("verdict"),
                    "verdict_reason": variant_data.get("verdict_reason", "Accuracy within tolerance" if variant_data.get("verdict") == "PASS" else f"Accuracy drop {variant_data.get('delta_vs_probe', 0):.3f} exceeds tolerance"),
                    "policy_name": variant_data.get("policy_name", variant_name),
                    "compression_method": variant_data.get("compression", {}).get("method", "uniform") if variant_data.get("compression") else "uniform"
                }
            else:
                # Candidate wasn't run in this seed (skip or fail)
                seed_detail["candidates"][variant_name] = {
                    "accuracy": None,
                    "params": None,
                    "rank": None,
                    "delta_vs_probe": None,
                    "param_reduction": None,
                    "verdict": "SKIP",
                    "verdict_reason": "Candidate not evaluated in this seed",
                    "policy_name": variant_name,
                    "compression_method": "unknown"
                }

        detailed_results["seeds"].append(seed_detail)

    # Build candidate-centric summary statistics
    for variant_name in all_variant_names:
        if variant_name in variants_data:
            variant_stats = variants_data[variant_name]

            # Extract per-seed results for this candidate
            candidate_seeds = []
            for seed_detail in detailed_results["seeds"]:
                if variant_name in seed_detail["candidates"]:
                    candidate_data = seed_detail["candidates"][variant_name]
                    if candidate_data["accuracy"] is not None:  # Only include successful runs
                        candidate_seeds.append({
                            "seed_id": seed_detail["seed_id"],
                            "probe_accuracy": seed_detail["probe"]["accuracy"],
                            "compressed_accuracy": candidate_data["accuracy"],
                            "delta_accuracy": candidate_data["delta_vs_probe"],
                            "probe_params": seed_detail["probe"]["params"],
                            "compressed_params": candidate_data["params"],
                            "param_reduction": candidate_data["param_reduction"],
                            "verdict": candidate_data["verdict"],
                            "verdict_reason": candidate_data["verdict_reason"],
                            "rank": candidate_data["rank"]
                        })

            # Calculate worst-case and mean deltas across seeds
            if candidate_seeds:
                deltas = [s["delta_accuracy"] for s in candidate_seeds if s["delta_accuracy"] is not None]
                param_reductions = [s["param_reduction"] for s in candidate_seeds if s["param_reduction"] is not None]

                detailed_results["candidates"][variant_name] = {
                    "policy_name": variant_name,
                    "chosen_rank": candidate_seeds[0].get("rank", "unknown") if candidate_seeds else "unknown",
                    "n_seeds_evaluated": len(candidate_seeds),
                    "per_seed_results": candidate_seeds,
                    "worst_case_delta": min(deltas) if deltas else None,
                    "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                    "mean_param_reduction": sum(param_reductions) / len(param_reductions) if param_reductions else None,
                    "pass_rate": variant_stats.get("pass_rate", 0.0),
                    "overall_verdict": variant_stats.get("verdict", "UNKNOWN")
                }

    # Two-tier defensible selection system
    # Tier 1: Safe variants (100% pass rate - all seeds pass)
    safe_variants = {name: data for name, data in variants_data.items() if data["pass_rate"] == 1.0}

    # Tier 2: Aggressive variants (>=60% pass rate but <100% - majority pass, clearly labeled)
    aggressive_variants = {name: data for name, data in variants_data.items()
                         if PASS_RATE_AGGRESSIVE <= data["pass_rate"] < 1.0}

    # Select best safe variant (highest compression among 100% pass rate)
    best_safe_variant = None
    if safe_variants:
        best_safe_name = max(safe_variants.keys(), key=lambda x: safe_variants[x]["param_reduction"]["mean"])
        best_safe_data = safe_variants[best_safe_name]
        best_safe_variant = {
            "variant": best_safe_name,
            "param_reduction_mean": best_safe_data["param_reduction"]["mean"],
            "param_reduction_std": best_safe_data["param_reduction"]["std"],
            "delta_vs_probe_mean": best_safe_data["delta_vs_probe"]["mean"],
            "delta_vs_probe_std": best_safe_data["delta_vs_probe"]["std"],
            "pass_rate": best_safe_data["pass_rate"],
            "pass_count": best_safe_data["pass_count"],
            "total_runs": best_safe_data["total_runs"],
            "confidence_level": "high",
            "rationale": "All seeds pass tolerance - recommended for production"
        }

    # Select best aggressive variant (highest compression among majority-pass)
    best_aggressive_variant = None
    if aggressive_variants:
        best_aggressive_name = max(aggressive_variants.keys(), key=lambda x: aggressive_variants[x]["param_reduction"]["mean"])
        best_aggressive_data = aggressive_variants[best_aggressive_name]
        best_aggressive_variant = {
            "variant": best_aggressive_name,
            "param_reduction_mean": best_aggressive_data["param_reduction"]["mean"],
            "param_reduction_std": best_aggressive_data["param_reduction"]["std"],
            "delta_vs_probe_mean": best_aggressive_data["delta_vs_probe"]["mean"],
            "delta_vs_probe_std": best_aggressive_data["delta_vs_probe"]["std"],
            "pass_rate": best_aggressive_data["pass_rate"],
            "pass_count": best_aggressive_data["pass_count"],
            "total_runs": best_aggressive_data["total_runs"],
            "confidence_level": "moderate",
            "rationale": f"{best_aggressive_data['pass_count']}/{best_aggressive_data['total_runs']} seeds pass - higher risk, higher reward"
        }

    # Legacy best_compression for backward compatibility (prefer safe, fallback to aggressive)
    best_compression = best_safe_variant or best_aggressive_variant

    # Compute validation classification for multi-seed aggregation
    n_seeds = len(seed_reports)
    max_steps = base_report.get("env", {}).get("validation_classification", {}).get("max_steps", 0)

    if n_seeds >= MIN_SEEDS_CERTIFIABLE and max_steps >= MIN_STEPS_CERTIFIABLE:
        validation_level = "certifiable"
        validation_rationale = f"{n_seeds} seeds \u00d7 {max_steps} steps provides statistical rigor"
    elif n_seeds >= MIN_SEEDS_SCREENING_PLUS and max_steps >= MIN_STEPS_SCREENING_PLUS:
        validation_level = "screening_plus"
        validation_rationale = f"{n_seeds} seeds \u00d7 {max_steps} steps (limited budget/seeds)"
    else:
        validation_level = "screening_plus"
        validation_rationale = f"{n_seeds} seeds but only {max_steps} steps (limited budget)"

    multiseed_validation_classification = {
        "level": validation_level,
        "rationale": validation_rationale,
        "is_multiseed": True,
        "n_seeds": n_seeds,
        "max_steps": max_steps
    }

    # Copy environment but update validation_classification
    aggregated_env = base_report.get("env", {}).copy()
    aggregated_env["validation_classification"] = multiseed_validation_classification

    # Build aggregated report
    aggregated_report = {
        "bench_version": base_report["bench_version"],
        "timestamp": timestamp,
        "aggregation_type": "multi_seed",
        "n_seeds": len(seed_reports),
        "seeds": [r.get("seed", r.get("env", {}).get("seed", "unknown")) for r in seed_reports],
        "model": base_report["model"],
        "task": base_report["task"],
        "env": aggregated_env,
        "git_commit": base_report.get("git_commit"),  # Use git info from first report
        "probe": {
            "rank": base_report["probe"]["rank"],
            "accuracy": {
                "mean": probe_acc_mean,
                "std": probe_acc_std,
                "values": probe_accuracies
            },
            "params": {
                "mean": probe_params_mean,
                "std": float(np.std(probe_params)) if len(probe_params) > 1 else 0.0
            }
        },
        "compressed": variants_data,
        "detailed_results": detailed_results,  # Self-contained per-seed, per-candidate breakdown
        "summary": {
            "best_compression": best_compression,  # Legacy field for backward compatibility
            "best_safe_variant": best_safe_variant,
            "best_aggressive_variant": best_aggressive_variant,
            "total_variants": len(variants_data),
            "safe_variants": len(safe_variants),
            "aggressive_variants": len(aggressive_variants),
            "selection_strategy": {
                "safe_available": best_safe_variant is not None,
                "aggressive_available": best_aggressive_variant is not None,
                "recommendation": "use_safe" if best_safe_variant else ("use_aggressive_with_caution" if best_aggressive_variant else "no_viable_variants")
            },
            "defensible_claims": True,
            "statistical_power": "sufficient" if len(seed_reports) >= MIN_SEEDS_SUFFICIENT_POWER else "limited"
        },
        "config_metadata": base_report.get("config_metadata", {})  # Use config metadata from first report
    }

    # Aggregate escalation traces from seed reports
    escalation_traces = [
        r["escalation"] for r in seed_reports
        if r.get("escalation", {}).get("triggered")
    ]
    if escalation_traces:
        aggregated_report["escalation"] = {
            "seeds_with_escalation": len(escalation_traces),
            "total_seeds": len(seed_reports),
            "per_seed_traces": escalation_traces,
        }

    return aggregated_report


def create_multi_seed_markdown_report(
    aggregated_report: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path
) -> str:
    """
    Create bench_aggregate.md human-readable markdown report for multi-seed results.
    """
    model = aggregated_report["model"]
    task = aggregated_report["task"]
    n_seeds = aggregated_report["n_seeds"]
    timestamp = aggregated_report["timestamp"]
    probe_data = aggregated_report["probe"]
    compressed_data = aggregated_report.get("compressed", {}) or {}
    summary = aggregated_report["summary"]

    # Extract validation level from aggregated report
    validation_level = "certifiable" if n_seeds >= MIN_SEEDS_CERTIFIABLE else "screening_plus"

    # Build markdown content
    md_content = f"""# Gradience Bench v{aggregated_report["bench_version"]} (Multi-Seed)

- **Model:** {model}
- **Task:** {task}
- **Seeds:** {n_seeds} ({', '.join(str(s) for s in aggregated_report.get("seeds", []))})
- **Validation Level:** {validation_level.title()}
- **Statistical Power:** {summary["statistical_power"]}

## Probe Baseline (mean ± std)

- **Rank:** {probe_data["rank"]}
- **Accuracy:** {probe_data["accuracy"]["mean"]:.4f} ± {probe_data["accuracy"]["std"]:.4f}
- **LoRA params:** {probe_data["params"]["mean"]:,.0f}

## Compression Results (Aggregated)

| Variant | Rank Policy | Accuracy | \u0394 vs Probe | Param Reduction | Pass Rate | Verdict |
|---|---|---:|---:|---:|---:|---|
"""

    # Add results table rows
    for variant_name, data in compressed_data.items():
        acc_str = f"{data['accuracy']['mean']:.3f} ± {data['accuracy']['std']:.3f}"
        delta_str = f"{data['delta_vs_probe']['mean']:+.3f} ± {data['delta_vs_probe']['std']:.3f}"
        red_str = f"{data['param_reduction']['mean']:.1%} ± {data['param_reduction']['std']:.1%}"
        pass_rate_str = f"{data['pass_count']}/{data['total_runs']} ({data['pass_rate']:.0%})"
        verdict = data['verdict']

        # Extract policy_origin from compression metadata
        policy_origin = data.get("compression", {}).get("policy_origin", "\u2014") if isinstance(data.get("compression"), dict) else "\u2014"

        # Format variant name for display
        if variant_name == "per_layer":
            variant_display = "`per_layer`"
        elif variant_name == "per_layer_shuffled":
            variant_display = "`per_layer_shuffled`"
        elif variant_name == "uniform_median":
            variant_display = "`uniform_median`"
        elif variant_name == "uniform_p90":
            variant_display = "`uniform_p90`"
        elif variant_name == "uniform_p90_control":
            variant_display = "`uniform_p90_control`"
        else:
            variant_display = f"`{variant_name}`"

        md_content += f"| {variant_display} | {policy_origin} | {acc_str} | {delta_str} | {red_str} | {pass_rate_str} | {verdict} |\n"

    # Add per-rank validation evidence section for scientific honesty
    acc_tolerance = config.get("compression", {}).get("acc_tolerance", DEFAULT_ACCURACY_TOLERANCE)

    md_content += f"""

## Per-Rank Validation Evidence

The following statements reflect the complete evidence from all {n_seeds} seeds:

"""

    # Group variants by rank for clearer reporting
    rank_to_variants = {}
    for variant_name, data in compressed_data.items():
        # Extract rank from variant data or name
        rank = None
        if "rank" in data:
            rank = data["rank"]
        elif "r" in variant_name:
            # Try to extract from name like "uniform_r32" or "energy_p90"
            import re
            rank_match = re.search(r'r(\d+)', variant_name)
            if rank_match:
                rank = int(rank_match.group(1))

        if rank is not None:
            if rank not in rank_to_variants:
                rank_to_variants[rank] = []
            rank_to_variants[rank].append((variant_name, data))

    # Generate honest per-rank statements
    for rank in sorted(rank_to_variants.keys()):
        variants = rank_to_variants[rank]

        # Determine overall validation status for this rank
        all_pass_counts = [data["pass_count"] for _, data in variants]
        all_total_runs = [data["total_runs"] for _, data in variants]

        # Use the variant with the most comprehensive data (highest total_runs)
        best_variant_name, best_data = max(variants, key=lambda x: x[1]["total_runs"])
        pass_count = best_data["pass_count"]
        total_runs = best_data["total_runs"]
        pass_rate = best_data["pass_rate"]

        # Generate scientific validation statement
        if pass_rate == 1.0:
            validation_status = "\u2705 **validated safe**"
            detail = f"All {total_runs} seeds pass ±{acc_tolerance:.3f} tolerance"
        elif pass_count >= 2 and total_runs >= 3:
            validation_status = "\u26a0\ufe0f  **conditionally promising**"
            failed_count = total_runs - pass_count
            if failed_count == 1:
                detail = f"{pass_count}/{total_runs} seeds pass; one seed violates tolerance"
            else:
                detail = f"{pass_count}/{total_runs} seeds pass; {failed_count} seeds violate tolerance"
        elif pass_count > 0:
            validation_status = "\u274c **unreliable**"
            detail = f"Only {pass_count}/{total_runs} seeds pass tolerance"
        else:
            validation_status = "\u274c **failed validation**"
            detail = f"No seeds pass ±{acc_tolerance:.3f} tolerance"

        md_content += f"- **r={rank}** is {validation_status} ({detail})\n"

    md_content += f"""

This evidence-based approach ensures complete transparency about which compression levels can be trusted across independent random seeds.

## Two-Tier Selection System

This benchmark uses a two-tier selection system for defensible recommendations:

"""

    # Add safe variant information
    if summary["best_safe_variant"]:
        safe = summary["best_safe_variant"]
        red_mean = safe["param_reduction_mean"] * 100
        red_std = safe["param_reduction_std"] * 100
        delta_mean = safe["delta_vs_probe_mean"]
        delta_std = safe["delta_vs_probe_std"]

        md_content += f"""### \U0001f7e2 Safe Variant (Recommended)
- **Variant:** `{safe["variant"]}`
- **Parameter reduction:** {red_mean:.1f}% ± {red_std:.1f}%
- **Accuracy impact:** {delta_mean:+.4f} ± {delta_std:.4f} vs probe baseline
- **Pass rate:** {safe["pass_count"]}/{safe["total_runs"]} seeds (100%)
- **Confidence:** {safe["confidence_level"]} - {safe["rationale"]}

"""

    # Add aggressive variant information
    if summary["best_aggressive_variant"]:
        aggressive = summary["best_aggressive_variant"]
        red_mean = aggressive["param_reduction_mean"] * 100
        red_std = aggressive["param_reduction_std"] * 100
        delta_mean = aggressive["delta_vs_probe_mean"]
        delta_std = aggressive["delta_vs_probe_std"]

        md_content += f"""### \U0001f7e1 Aggressive Variant (Higher Risk)
- **Variant:** `{aggressive["variant"]}`
- **Parameter reduction:** {red_mean:.1f}% ± {red_std:.1f}%
- **Accuracy impact:** {delta_mean:+.4f} ± {delta_std:.4f} vs probe baseline
- **Pass rate:** {aggressive["pass_count"]}/{aggressive["total_runs"]} seeds ({aggressive["pass_rate"]:.0%})
- **Confidence:** {aggressive["confidence_level"]} - {aggressive["rationale"]}

"""

    # Add selection recommendation
    strategy = summary["selection_strategy"]
    md_content += f"""### Recommendation: {strategy["recommendation"].replace("_", " ").title()}
"""

    if strategy["recommendation"] == "use_safe":
        md_content += "\u2705 Safe variant available - recommended for production deployment.\n\n"
    elif strategy["recommendation"] == "use_aggressive_with_caution":
        md_content += "\u26a0\ufe0f  Only aggressive variants available - proceed with caution and additional validation.\n\n"
    else:
        md_content += "\u274c No variants meet minimum reliability thresholds.\n\n"

    # Add interpretation
    md_content += f"""## Interpretation (Statistical)

- **PASS** means ≥80% of seeds passed ±{acc_tolerance:.3f} accuracy tolerance (stringent threshold)
- **Safe variants** require 100% pass rate across all seeds
- **Aggressive variants** require >=60% pass rate but clearly flagged as higher risk
- **Statistics** are calculated as mean ± standard deviation across {n_seeds} seeds
- **Defensible claims** are supported by variance estimation across multiple random seeds
- You should still validate these results on your real workload before deployment

## Summary

- **Total variants:** {summary["total_variants"]}
- **Safe variants:** {summary["safe_variants"]} (100% pass rate)
- **Aggressive variants:** {summary["aggressive_variants"]} (>=60% pass rate)
- **Recommended approach:** {strategy["recommendation"].replace("_", " ")}
- **Statistical power:** {summary["statistical_power"]}

## Detailed Per-Seed Results

This section contains complete self-contained results for all seeds and candidates.

"""

    # Add detailed breakdown for self-contained reporting
    detailed_data = aggregated_report.get("detailed_results", {})
    if detailed_data:
        # Add per-candidate summary with worst-case delta
        md_content += """### Candidate Summary

| Candidate | Policy | Rank | Seeds | Worst \u0394 | Mean \u0394 | Param Reduction | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
"""

        candidates = detailed_data.get("candidates", {})
        for candidate_name, candidate_info in candidates.items():
            worst_delta = candidate_info.get("worst_case_delta", 0.0)
            mean_delta = candidate_info.get("mean_delta", 0.0)
            param_reduction = candidate_info.get("mean_param_reduction", 0.0)
            n_seeds = candidate_info.get("n_seeds_evaluated", 0)
            verdict = candidate_info.get("overall_verdict", "UNKNOWN")
            policy_name = candidate_info.get("policy_name", candidate_name)
            rank = candidate_info.get("chosen_rank", "?")

            md_content += f"| `{candidate_name}` | {policy_name} | r={rank} | {n_seeds} | {worst_delta:+.3f} | {mean_delta:+.3f} | {param_reduction:.1%} | {verdict} |\n"

        # Add detailed per-seed, per-candidate breakdown
        md_content += """

### Complete Per-Seed Breakdown

"""

        seeds_data = detailed_data.get("seeds", [])
        for seed_info in seeds_data:
            seed_id = seed_info.get("seed_id", "unknown")
            probe_info = seed_info.get("probe", {})

            md_content += f"""
#### Seed: {seed_id}

**Probe baseline:** {probe_info.get('accuracy', 0.0):.3f} accuracy, r={probe_info.get('rank', '?')}, {probe_info.get('params', 0):,} params

| Candidate | Accuracy | \u0394 vs Probe | Trainable Params | Param Reduction | Verdict | Reason |
|---|---:|---:|---:|---:|---|---|
"""

            candidates_data = seed_info.get("candidates", {})
            for candidate_name, candidate_data in candidates_data.items():
                if candidate_data.get("accuracy") is not None:  # Only show evaluated candidates
                    accuracy = candidate_data.get("accuracy", 0.0)
                    delta = candidate_data.get("delta_vs_probe", 0.0)
                    params = candidate_data.get("params", 0)
                    param_reduction = candidate_data.get("param_reduction", 0.0)
                    verdict = candidate_data.get("verdict", "UNKNOWN")
                    reason = candidate_data.get("verdict_reason", "No reason provided")

                    md_content += f"| `{candidate_name}` | {accuracy:.3f} | {delta:+.3f} | {params:,} | {param_reduction:.1%} | {verdict} | {reason} |\n"

    md_content += f"""

*Generated on {timestamp[:19].replace('T', ' ')}*
"""

    return md_content
