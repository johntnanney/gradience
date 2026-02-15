"""Report generation for bench protocol (canonical JSON and markdown)."""

from __future__ import annotations

import json
import os
import sys
import datetime
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

from gradience.bench.decision_trace import DecisionTrace
from gradience.bench.types import EnvironmentInfo


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


def gather_environment_info() -> EnvironmentInfo:
    """Gather comprehensive environment information for self-describing bench reports."""
    import platform
    import os

    env_info = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "hostname": platform.node(),
    }

    # Package versions
    packages = ["torch", "transformers", "peft", "datasets", "accelerate", "safetensors", "numpy"]
    for package in packages:
        try:
            module = __import__(package)
            env_info[f"{package}_version"] = module.__version__
        except ImportError:
            env_info[f"{package}_version"] = "not_installed"
        except AttributeError:
            env_info[f"{package}_version"] = "version_unavailable"

    # PyTorch and CUDA information
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["cudnn_version"] = torch.backends.cudnn.version()
            env_info["cuda_device_count"] = torch.cuda.device_count()

            # GPU information
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "device_id": i,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count
                })
            env_info["gpu_devices"] = gpu_info

            # Current device and memory
            if torch.cuda.current_device() is not None:
                current_device = torch.cuda.current_device()
                env_info["current_cuda_device"] = current_device
                env_info["cuda_memory_allocated"] = torch.cuda.memory_allocated(current_device)
                env_info["cuda_memory_reserved"] = torch.cuda.memory_reserved(current_device)
        else:
            env_info["cuda_version"] = None
            env_info["gpu_devices"] = []

    except ImportError:
        env_info["torch_version"] = "not_installed"
        env_info["cuda_available"] = False
        env_info["cuda_version"] = None
        env_info["gpu_devices"] = []

    # Environment variables that affect reproducibility
    relevant_env_vars = [
        "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
        "TORCH_HOME", "TRANSFORMERS_CACHE", "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    ]

    env_vars = {}
    for var in relevant_env_vars:
        value = os.environ.get(var)
        if value is not None:
            env_vars[var] = value
    env_info["environment_variables"] = env_vars

    return env_info


def get_git_commit() -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_git_tag() -> Optional[str]:
    """Get the current git tag or 'dirty' if there are uncommitted changes."""
    try:
        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            return "dirty"

        # Check if there are staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            return "dirty"

        # Get the exact tag for current commit
        result = subprocess.run(
            ["git", "describe", "--exact-match", "--tags"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()

        # If no exact tag, get the most recent tag with distance
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=7"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_hf_model_revision(model_id: str) -> Optional[Dict[str, str]]:
    """Get the revision hash for a HuggingFace model."""
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        return {
            "model_id": model_id,
            "revision": info.sha,
            "last_modified": info.lastModified.isoformat() if info.lastModified else None
        }
    except Exception:
        return {
            "model_id": model_id,
            "revision": "unknown",
            "last_modified": None
        }


def get_dataset_revision(dataset_id: str, split: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the revision hash and split information for a HuggingFace dataset."""
    try:
        from huggingface_hub import dataset_info
        from datasets import load_dataset_builder

        # Get dataset info from hub
        info = dataset_info(dataset_id)

        # Get split sizes
        builder = load_dataset_builder(dataset_id)
        split_info = {}
        if hasattr(builder, 'info') and hasattr(builder.info, 'splits'):
            for split_name, split_details in builder.info.splits.items():
                split_info[split_name] = split_details.num_examples

        return {
            "dataset_id": dataset_id,
            "revision": info.sha,
            "last_modified": info.lastModified.isoformat() if info.lastModified else None,
            "split_sizes": split_info,
            "requested_split": split
        }
    except Exception as e:
        return {
            "dataset_id": dataset_id,
            "revision": "unknown",
            "last_modified": None,
            "split_sizes": {},
            "requested_split": split,
            "error": str(e)
        }


def extract_model_dataset_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model and dataset information from the bench config."""
    metadata = {}

    # Extract model information
    model_id = config.get("model_id")
    if model_id:
        metadata["model_info"] = get_hf_model_revision(model_id)

    # Extract dataset information
    dataset_config = config.get("dataset", {})
    if isinstance(dataset_config, dict):
        dataset_id = dataset_config.get("name")
        split = dataset_config.get("split")
        if dataset_id:
            metadata["dataset_info"] = get_dataset_revision(dataset_id, split)

    return metadata


def get_primary_metric_key(config: Dict[str, Any]) -> str:
    """Determine the primary evaluation metric based on the task configuration."""
    task_config = config.get("task", {})
    dataset_name = task_config.get("dataset", "").lower()

    # Dataset-specific metric mappings
    if dataset_name == "gsm8k":
        return "eval_exact_match"
    elif dataset_name in ["glue", "cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]:
        return "eval_accuracy"
    else:
        # Default fallback
        return "eval_accuracy"


def create_config_hash(config: Dict[str, Any]) -> str:
    """Create a stable hash of the configuration for reference."""
    import hashlib
    import json

    # Create a stable string representation
    config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def create_canonical_bench_report(
    probe_results: Dict[str, Any],
    variant_results: Dict[str, Dict[str, Any]],
    verdict_analysis: Dict[str, Any],
    audit_data: Dict[str, Any],
    compression_configs: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
    decision_trace: Optional[DecisionTrace] = None
) -> Dict[str, Any]:
    """
    Create the canonical bench.json report according to specification.
    """

    # Import get_rank_source_from_config from protocol (it stays there)
    from gradience.bench.protocol import get_rank_source_from_config

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

        minimal_report = {
            "bench_version": config.get("bench_version", "0.1"),
            "timestamp": timestamp,
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

    # Build the canonical report
    report = {
        "bench_version": config.get("bench_version", "0.1"),
        "timestamp": timestamp,
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

    # Add protocol invariants for aggregation
    probe_gate_data = report["probe_quality_gate"]
    report["protocol_invariants"] = {
        "probe_quality_gate": {
            "status": "PASSED" if probe_gate_data["passed"] else "FAILED",
            "message": f"Probe {probe_gate_data['metric_key']} {probe_gate_data['metric_value']:.4f} {'≥' if probe_gate_data['passed'] else '<'} {probe_gate_data['min_value']:.4f}",
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
    acc_tolerance = config.get("compression", {}).get("acc_tolerance", 0.005)

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
                    if concentration_index > 0.4:
                        md_content += "- \U0001f6a8 **Highly concentrated** adaptation\n"
                    elif concentration_index > 0.25:
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

    md_content += f"""
## Summary

- **Recommendations validated:** {summary["recommendations_validated"]}
- **Best compression:** {summary["best_compression"] or "None"}

*Generated on {timestamp[:19].replace('T', ' ')}*
"""

    return md_content
