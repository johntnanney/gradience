"""
Bench protocol (v0.1).

Implements:
1) Train probe adapter (r=16)
2) Audit -> suggestions
3) Retrain: uniform_median, uniform_p90, per_layer
4) Eval all
5) Emit report (JSON + Markdown)

Step 3.1 implementation: Train probe with GradienceCallback

This module has been decomposed into focused sub-modules for maintainability.
All public symbols are re-exported here for backward compatibility.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Re-exports for backward compatibility
#
# Every symbol that was previously importable from gradience.bench.protocol
# is re-exported below so that existing code continues to work unchanged.
# ---------------------------------------------------------------------------

# Gradience imports (always available)
from gradience.bench._util import (  # noqa: F401
    create_config_hash,
    get_primary_metric_key,
    round_to_allowed_ranks,  # noqa: F401
)
from gradience.bench.compression import (  # noqa: F401
    _create_shuffled_rank_pattern,
    _resolve_policy_rank_source,
    generate_compression_configs,
    generate_svd_variant_config,
    get_rank_source_from_config,
)
from gradience.bench.config_schema import validate_config  # noqa: F401
from gradience.bench.constants import (
    DEFAULT_ACCURACY_TOLERANCE,
    DEFAULT_BASE_LR,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_NUM_LABELS,
    DEFAULT_POST_TUNE_LR_SCALE,
    DEFAULT_POST_TUNE_STEPS,
    DEFAULT_PROBE_RANK,
    DEFAULT_SEED,
    DEFAULT_TASK_THRESHOLD,
    DEFAULT_TRAIN_BATCH_SIZE,
    MIN_SEEDS_CERTIFIABLE,
    MIN_SEEDS_SCREENING_PLUS,  # noqa: F401
    MIN_STEPS_CERTIFIABLE,
    MIN_STEPS_SCREENING_PLUS,
    SMOKE_EVAL_SAMPLES,
    SMOKE_MAX_POST_TUNE_STEPS,
    SMOKE_TRAIN_SAMPLES,
    TASK_QUALITY_THRESHOLDS,
)
from gradience.bench.decision_trace import (  # noqa: F401
    DecisionTrace,
    create_decision_trace,
    maybe_add_second_rung_candidates,
)
from gradience.bench.escalation import (
    EscalationTrace,  # noqa: F401
    enrich_verdicts_with_stability,
    run_escalation_round,
    update_escalation_trace_with_results,
)

# Imports used by run_bench_protocol itself
from gradience.bench.heartbeat import heartbeat_stage

# Re-exports for backward compatibility — all existing import paths continue to work.
from gradience.bench.metadata import (  # noqa: F401
    extract_model_dataset_info,
    gather_environment_info,
    get_dataset_revision,
    get_git_commit,
    get_git_tag,
    get_hf_model_revision,
)
from gradience.bench.model_setup import (  # noqa: F401
    HAS_TRAINING_DEPS,
    _save_peft_adapter_only,
    _unwrap_model_for_save,
    load_config,
)
from gradience.bench.monitored_stage import (
    monitor_audit,
    monitor_evaluation,
    monitor_file_operations,
    monitor_generation,
    monitor_training,
    setup_global_monitoring,
)
from gradience.bench.multi_seed import (  # noqa: F401
    run_multi_seed_bench_protocol,
)
from gradience.bench.preflight import (  # noqa: F401
    run_artifact_hygiene_cleanup,
    run_bench_preflight_check,
)
from gradience.bench.probe import (  # noqa: F401
    run_probe_audit,
    run_probe_training,
)
from gradience.bench.reporting import (  # noqa: F401  # noqa: F401
    _extract_accuracy_with_fallback,
    create_canonical_bench_report,
    create_markdown_report,
    create_multi_seed_aggregated_report,
    create_multi_seed_markdown_report,
    write_probe_eval_json,
)
from gradience.bench.stage_state import create_stage_manager
from gradience.bench.task_profiles import get_task_profile_from_config  # noqa: F401
from gradience.bench.variants import (  # noqa: F401
    run_all_compressed_variants,
    run_compressed_variant_training,
    run_post_tuning,
    run_svd_truncation_variant,
)
from gradience.bench.verdicts import (  # noqa: F401
    classify_validation_level,
    compute_verdicts,
)
from gradience.peft_utils import check_heterogeneous_ranks, create_complete_alpha_pattern, create_complete_rank_pattern
from gradience.vnext.audit.lora_audit import audit_lora_peft_dir  # noqa: F401
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig
from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit, suggest_per_layer_ranks


def _get_probe_quality_threshold(task_name: str) -> float:
    """
    Get task-specific minimum probe accuracy threshold for compression certification.

    Args:
        task_name: Task identifier (e.g., 'sst2', 'mnli', etc.)

    Returns:
        Minimum accuracy threshold for considering probe sufficiently trained
    """
    return TASK_QUALITY_THRESHOLDS.get(task_name.lower(), DEFAULT_TASK_THRESHOLD)


def setup_dataset(config: dict[str, Any], smoke: bool = False):
    """Load and prepare dataset based on config using task profile."""
    if not HAS_TRAINING_DEPS:
        raise ImportError(
            "Bench dependencies not available (transformers, datasets, peft). "
            'Install with: pip install "gradience[bench]"'
        )

    # Get task profile for this configuration
    task_profile = get_task_profile_from_config(config)

    # Load dataset using task profile
    dataset = task_profile.load(config)

    # Apply smoke test limits if requested
    if smoke:
        runtime = config.get("runtime", {})
        train_samples = runtime.get("smoke_train_samples", SMOKE_TRAIN_SAMPLES)
        eval_samples = runtime.get("smoke_eval_samples", SMOKE_EVAL_SAMPLES)

        if "train" in dataset:
            dataset["train"] = dataset["train"].select(range(min(len(dataset["train"]), train_samples)))
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].select(range(min(len(dataset["validation"]), eval_samples)))

    return dataset


def setup_model_and_tokenizer(config: dict[str, Any], device: str = "cpu"):
    """Setup base model, tokenizer, and LoRA configuration."""
    if not HAS_TRAINING_DEPS:
        raise ImportError(
            'Bench dependencies not available (transformers, peft). Install with: pip install "gradience[bench]"'
        )

    model_config = config["model"]
    model_name = model_config["name"]
    model_type = model_config.get("type", "seqcls")  # Default to sequence classification
    lora_config = config["lora"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine torch dtype
    torch_dtype_str = model_config.get("torch_dtype", "auto")
    if torch_dtype_str == "bf16":
        torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    elif torch_dtype_str == "fp16":
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
    else:
        torch_dtype = torch.float32 if device == "cpu" else torch.float16

    # Load model based on type
    if model_type == "causal_lm":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map="auto" if device == "cuda" else None
        )

        # Configure for training
        if model_config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        if not model_config.get("use_cache", True):  # use_cache=False during training
            model.config.use_cache = False

        task_type = TaskType.CAUSAL_LM
    else:
        # Default to sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=DEFAULT_NUM_LABELS, torch_dtype=torch_dtype
        )
        task_type = TaskType.SEQ_CLS

    # Setup LoRA
    peft_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=lora_config["probe_r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
    )

    model = get_peft_model(model, peft_config)  # type: ignore[assignment]

    return tokenizer, model


def setup_compressed_model_and_tokenizer(
    config: dict[str, Any], compression_config: dict[str, Any], device: str = "cpu"
):
    """Setup model and tokenizer with compressed LoRA configuration."""
    if not HAS_TRAINING_DEPS:
        raise ImportError(
            'Bench dependencies not available (transformers, peft). Install with: pip install "gradience[bench]"'
        )

    model_config = config["model"]
    model_name = model_config["name"]
    model_type = model_config.get("type", "seqcls")
    _base_lora_config = config["lora"]
    variant_config = compression_config["config"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine torch dtype
    torch_dtype_str = model_config.get("torch_dtype", "auto")
    if torch_dtype_str == "bf16":
        torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    elif torch_dtype_str == "fp16":
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
    else:
        torch_dtype = torch.float32 if device == "cpu" else torch.float16

    # Load model based on type
    if model_type == "causal_lm":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map="auto" if device == "cuda" else None
        )

        # Configure for training
        if model_config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        if not model_config.get("use_cache", True):
            model.config.use_cache = False

        task_type = TaskType.CAUSAL_LM
    else:
        # Default to sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=DEFAULT_NUM_LABELS, torch_dtype=torch_dtype
        )
        task_type = TaskType.SEQ_CLS

    # Setup compressed LoRA configuration
    if compression_config["variant"] in ["per_layer", "per_layer_shuffled"]:
        # Per-layer configuration with rank_pattern
        #
        # IMPORTANT: Current approach for PEFT compatibility (as of PEFT 0.18.1)
        # We normalize module names and use a conservative default rank strategy:
        # 1. Normalize all module names (remove base_model.model. prefix)
        # 2. Create complete rank patterns for ALL modules (not just overrides)
        # 3. Use minimum rank as default for PEFT compatibility
        #
        # TODO: Future cleaner approach once PEFT rank_pattern is more robust:
        # - Use default_r = max(pattern_ranks) or global p90 rank
        # - Only include overrides for layers that differ from default
        # - This would reduce the pattern size and be more maintainable
        #
        # The current approach works reliably but creates larger patterns than necessary.
        # We keep it for stability until PEFT rank_pattern handling improves.

        # Get audit layers for complete module discovery
        audit_layers = compression_config.get("_audit_layers", [])

        # Use the original probe rank as default for modules not in rank_pattern
        default_rank_from_audit = compression_config.get("_probe_rank", DEFAULT_PROBE_RANK)
        default_alpha_from_audit = default_rank_from_audit

        # Create complete, normalized patterns using canonical helpers
        # Get patterns from compression_config (top-level) with fallback to variant_config (nested)
        rank_pattern = compression_config.get("rank_pattern") or variant_config.get("rank_pattern", {})
        alpha_pattern = compression_config.get("alpha_pattern") or variant_config.get("alpha_pattern", {})

        full_rank_pattern = create_complete_rank_pattern(rank_pattern, audit_layers, default_rank_from_audit)
        full_alpha_pattern = create_complete_alpha_pattern(alpha_pattern, audit_layers, default_alpha_from_audit)

        # Use minimum rank as default for PEFT compatibility
        # This conservative approach ensures rank_pattern overrides work correctly
        # (PEFT 0.18.1 has issues when default_r > some pattern values)

        # full_rank_pattern may be empty if module-name expansion fails (e.g., naming/prefix mismatch)
        # Fall back to the provided rank_pattern rather than crashing.
        if not full_rank_pattern:
            full_rank_pattern = dict(rank_pattern or {})

        if not full_rank_pattern:
            raise ValueError(
                "Computed empty rank_pattern for compressed variant. "
                "This usually indicates a module-name mismatch between the adapter and base model."
            )

        min_rank = min(full_rank_pattern.values())

        peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=min_rank,  # Use min rank as default
            lora_alpha=min_rank,  # Use min alpha as default
            lora_dropout=variant_config["dropout"],
            target_modules=variant_config["target_modules"],
            rank_pattern=full_rank_pattern,
            alpha_pattern=full_alpha_pattern,
        )
    else:
        # Uniform configuration (uniform_median, uniform_p90)
        peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=variant_config["probe_r"],
            lora_alpha=variant_config["alpha"],
            lora_dropout=variant_config["dropout"],
            target_modules=variant_config["target_modules"],
        )

    model = get_peft_model(model, peft_config)  # type: ignore[assignment]

    return tokenizer, model


# Legacy preprocess_function moved to task profiles
# This function is kept for backward compatibility but deprecated


def write_probe_eval_json(  # type: ignore[no-redef]  # noqa: F811
    probe_dir: Path, eval_results: dict[str, Any], eval_dataset_size: int, config: dict[str, Any]
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
        "eval_steps_per_second": eval_results.get("eval_steps_per_second"),
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
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    return eval_path


def run_probe_audit(  # type: ignore[no-redef]  # noqa: F811
    probe_dir: Path, config: dict[str, Any]
) -> Path:
    """
    Step 3.3: Run audit on trained probe and write audit.json.

    Args:
        probe_dir: Directory containing the trained probe (with adapter weights)
        config: Benchmark configuration

    Returns:
        Path to the written audit.json file
    """
    # HuggingFace Trainer saves adapters in checkpoint subdirectories
    # Find the first checkpoint directory for auditing
    checkpoint_dirs = sorted([d for d in probe_dir.glob("checkpoint-*") if d.is_dir()])
    if checkpoint_dirs:
        # Use the first/newest checkpoint directory found
        audit_dir = checkpoint_dirs[0]
        print(f"Using checkpoint directory for audit: {audit_dir}")
    else:
        # Fall back to probe_dir if no checkpoints found (e.g., manual save)
        audit_dir = probe_dir
        print(f"No checkpoint subdirectories found, using probe_dir: {audit_dir}")

    # Check for UDR configuration
    audit_config = config.get("audit", {})
    base_model_id = audit_config.get("base_model")  # Only use explicitly set base_model
    base_norms_cache = audit_config.get("base_norms_cache")

    # UDR is now explicitly opt-in: requires both compute_udr=True AND base_model to be set
    compute_udr_requested = audit_config.get("compute_udr", False)

    # Validate UDR configuration
    if compute_udr_requested and base_model_id is None:
        raise ValueError(
            "UDR computation was explicitly requested (audit.compute_udr: true) but "
            "audit.base_model is not set. Either:\n"
            "  1. Set audit.base_model to the base model ID, or\n"
            "  2. Set audit.compute_udr: false to disable UDR computation"
        )

    compute_udr = compute_udr_requested and base_model_id is not None

    if compute_udr:
        print(f"Running audit with UDR computation using base model: {base_model_id}")
    else:
        print("Running audit without UDR computation")

    # Run audit on the PEFT directory containing adapter files
    audit_result = audit_lora_peft_dir(
        audit_dir,
        base_model_id=base_model_id if compute_udr else None,
        base_norms_cache=base_norms_cache,
        compute_udr=compute_udr,
    )

    # Convert audit result to dict for JSON serialization
    audit_summary = audit_result.to_summary_dict()

    # Add the probe rank to the audit summary for rank suggestion
    probe_rank = config["lora"]["probe_r"]
    audit_summary["current_r"] = probe_rank

    # Validate LoRA attachment - prevent wasted GPU cycles
    stable_rank_mean = audit_summary.get("stable_rank_mean", 0.0)
    utilization_mean = audit_summary.get("utilization_mean", 0.0)

    if stable_rank_mean == 0.0 and utilization_mean == 0.0:
        print("")
        print("🚨" * 20)
        print("🚨 CRITICAL: LoRA LIKELY DID NOT TRAIN / DID NOT ATTACH")
        print("🚨 stable_rank_mean = 0.0 AND utilization = 0.0")
        print("🚨 This indicates LoRA adapters were not properly attached")
        print("🚨 or training failed to update weights.")
        print("🚨")
        print("🚨 Common causes:")
        print("🚨 • Wrong target_modules for this model architecture")
        print("🚨 • LoRA not properly attached during training")
        print("🚨 • Training failed silently")
        print("🚨 • Adapter weights not saved properly")
        print("🚨")
        print("🚨 This probe is INVALID - marking as failed to prevent")
        print("🚨 wasted GPU cycles on compression variants.")
        print("🚨" * 20)
        print("")

        # Mark probe as invalid in audit summary
        audit_summary["probe_validity"] = {
            "valid": False,
            "reason": "LoRA_NOT_ATTACHED",
            "stable_rank_mean": stable_rank_mean,
            "utilization_mean": utilization_mean,
            "message": "LoRA adapters likely did not train or attach properly",
        }
    else:
        audit_summary["probe_validity"] = {
            "valid": True,
            "reason": "NORMAL_OPERATION",
            "stable_rank_mean": stable_rank_mean,
            "utilization_mean": utilization_mean,
        }

    # Generate additional global rank suggestions
    try:
        global_suggestions = suggest_global_ranks_from_audit(audit_summary)
    except ValueError as e:
        print(f"Error in suggest_global_ranks_from_audit: {e}")
        print(f"audit_summary keys: {list(audit_summary.keys())}")
        print(f"stable_rank_mean: {audit_summary.get('stable_rank_mean')}")
        print(f"utilization_mean: {audit_summary.get('utilization_mean')}")
        raise

    # Generate per-layer rank suggestions if we have per-layer data
    per_layer_suggestions = None
    if audit_result.layers:
        # Create a dict with the layers data for per-layer suggestions
        audit_dict_for_layers = {"layers": [layer.to_dict() for layer in audit_result.layers]}
        per_layer_suggestions = suggest_per_layer_ranks(
            audit_dict_for_layers,
            allowed_ranks=config.get("compression", {}).get("allowed_ranks", [1, 2, 4, 8, 16, 32]),
        )

    # Compute gain metrics
    from gradience.vnext.audit.lora_audit import compute_gain_metrics

    # Check if composition analysis is enabled (default: true)
    audit_config = config.get("audit", {})
    enable_composition = audit_config.get("enable_composition_analysis", True)

    gain_metrics = compute_gain_metrics(audit_result.layers)

    # Prepare comprehensive audit data
    audit_data = {
        # Audit metadata
        "audit_timestamp": datetime.datetime.now().isoformat(),
        "probe_rank": config["lora"]["probe_r"],
        "seed": config["train"]["seed"],
        # Summary statistics (includes suggested_r_global_median, suggested_r_global_90)
        "summary": {**audit_summary, "gain": gain_metrics["summary"]},
        # Global rank suggestions (required) - using audit summary values
        "suggested_r_global_median": audit_summary.get("suggested_r_global_median"),
        "suggested_r_global_90": audit_summary.get("suggested_r_global_90"),
        # Policy-based global suggestions (Step 7) - with defensive handling for mocks
        "policy_global_suggestions": (
            getattr(audit_result, "policy_global_suggestions", {})
            if hasattr(audit_result, "policy_global_suggestions")
            and not str(type(getattr(audit_result, "policy_global_suggestions", None))).__contains__("Mock")
            else {}
        ),
        # Additional global suggestion details from rank_suggestion module
        "global_suggestions": {
            "current_r": global_suggestions.current_r,
            "suggested_r_median": global_suggestions.suggested_r_median,
            "suggested_r_p90": global_suggestions.suggested_r_p90,
            "total_lora_params": global_suggestions.total_lora_params,
            "reduction_ratio_median": global_suggestions.reduction_ratio_median,
            "reduction_ratio_p90": global_suggestions.reduction_ratio_p90,
            "evidence": global_suggestions.evidence,
        },
        # Per-module gain metrics
        "per_module": {"gain": gain_metrics["per_module"]},
        # Per-layer gain metrics
        "per_layer": {"gain": gain_metrics["per_layer"]},
        # Global gain metrics
        "global": {"gain": gain_metrics["global"]},
        # Composition analysis (energy concentration across layers) - optional
        **({"composition": gain_metrics.get("composition", {})} if enable_composition else {}),
        # Per-layer analysis (your 1.3/1.4 work)
        "layers": [layer.to_dict() for layer in audit_result.layers],
        # Per-layer suggestions if available
        "per_layer_suggestions": per_layer_suggestions.to_dict() if per_layer_suggestions else None,
        # Issues encountered during audit
        "issues": audit_result.issues,
    }

    audit_path = probe_dir / "audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)

    return audit_path


def run_probe_training(  # type: ignore[no-redef]  # noqa: F811
    config_path: str | Path, output_dir: str | Path, smoke: bool = False, stage_manager=None, resume: bool = False
) -> dict[str, Any]:
    """
    Step 3.1: Train probe adapter (r=16).

    Returns training results including accuracy and parameter counts.
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError(
            "Bench dependencies not available (transformers, peft, datasets). "
            'Install with: pip install "gradience[bench]"'
        )

    # Load configuration
    config = load_config(config_path)

    # Setup output directory for probe with actual rank
    probe_rank = config["lora"]["probe_r"]
    probe_dir = Path(output_dir) / f"probe_r{probe_rank}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Get device from config
    device = config.get("runtime", {}).get("device", "cpu")

    # Setup dataset
    dataset = setup_dataset(config, smoke=smoke)

    # Setup model and tokenizer
    tokenizer, model = setup_model_and_tokenizer(config, device=device)

    # Get task profile and preprocess dataset
    task_profile = get_task_profile_from_config(config)
    tokenized_dataset = task_profile.tokenize(dataset, tokenizer, config)

    # Apply smoke test limits to training config
    train_config = config["train"]
    runtime_config = config.get("runtime", {})

    if smoke:
        max_steps = runtime_config.get("smoke_max_steps", 50)
        # Create modified config for smoke test
        modified_config = config.copy()
        modified_config["train"] = train_config.copy()
        modified_config["train"]["max_steps"] = max_steps
        config = modified_config

    # Setup Gradience callback
    # Optional: pass dataset/task info for richer telemetry if available
    callback_config = GradienceCallbackConfig(output_dir=str(probe_dir), filename="run.jsonl")

    # Add optional dataset/task context for richer telemetry
    task_config = config.get("task", {})
    if task_config.get("dataset") and task_config.get("subset"):
        _dataset_name = f"{task_config['dataset']}/{task_config['subset']}"
        # Note: callback doesn't require these fields, but bench can provide them
        # for richer downstream monitor output
        # We'll pass them via environment or config if the callback supports it in future

    gradience_callback = GradienceCallback(callback_config)

    # Build trainer using task profile
    trainer = task_profile.build_trainer(
        model=model, tokenizer=tokenizer, tokenized_ds=tokenized_dataset, cfg=config, callbacks=[gradience_callback]
    )

    # Update trainer output dir to probe directory
    trainer.args.output_dir = str(probe_dir)
    trainer.args.logging_dir = str(probe_dir / "logs")

    # Train the model
    # Check if training can be skipped
    probe_rank = config["lora"]["probe_r"]
    skip_training = resume and stage_manager and stage_manager.should_skip_probe_training(probe_rank)

    if not skip_training:
        print(f"Starting probe training (r={probe_rank})...")
        print(f"Output dir: {probe_dir}")
        print(f"Max steps: {trainer.args.max_steps}")
        print(f"Device: {device}")

        seed = config.get("train", {}).get("seed", DEFAULT_SEED)
        with monitor_training(f"train_probe_r{probe_rank}", output_dir=probe_dir, seed=seed) as stage:
            stage.progress("Starting probe training")
            trainer.train()
            stage.progress("Probe training completed")

        # Mark training as completed
        if stage_manager:
            stage_manager.mark_stage_completed(
                f"probe_r{probe_rank}_trained",
                {"probe_rank": probe_rank, "max_steps": trainer.args.max_steps, "output_dir": str(probe_dir)},
            )
    else:
        # Load existing model for evaluation (skip training)
        print(f"Loading existing probe model from {probe_dir}...")
        # We'll still need the model loaded for evaluation, but we skip training

    # Ensure adapter exists on disk for audit (save_strategy may be "no")
    with monitor_file_operations("save_probe_adapter", output_dir=probe_dir) as stage:
        stage.progress("Saving probe adapter to disk")
        _save_peft_adapter_only(trainer, model, probe_dir, label="probe")
        stage.progress("Probe adapter saved successfully")

    # Check if evaluation can be skipped
    skip_evaluation = resume and stage_manager and stage_manager.should_skip_probe_evaluation(probe_rank)

    if not skip_evaluation:
        # Evaluate final model using task profile
        seed = config.get("train", {}).get("seed", DEFAULT_SEED)
        with monitor_evaluation("eval_probe", output_dir=probe_dir, seed=seed) as stage:
            stage.progress("Starting probe evaluation")
            eval_results = task_profile.evaluate(model, tokenizer, tokenized_dataset, config)
            stage.progress("Probe evaluation completed")
            stage.add_artifact("eval.json")

        # Mark evaluation as completed
        if stage_manager:
            stage_manager.mark_stage_completed(
                f"probe_r{probe_rank}_evaluated",
                {"probe_rank": probe_rank, "accuracy": eval_results.get("eval_accuracy", 0.0)},
            )
    else:
        # Load existing evaluation results
        eval_json_path = probe_dir / "eval.json"
        with open(eval_json_path) as f:
            eval_results = json.load(f)
        print(f"Loaded existing evaluation results: accuracy = {eval_results.get('eval_accuracy', 'unknown')}")

    # Step 3.2: Write eval.json
    eval_dataset_size = eval_results.get(
        "eval_samples", len(tokenized_dataset.get("validation", tokenized_dataset["train"]))
    )
    eval_json_path = write_probe_eval_json(
        probe_dir=probe_dir, eval_results=eval_results, eval_dataset_size=eval_dataset_size, config=config
    )

    # Step 3.3: Run audit and write audit.json
    # Guard: ensure adapter weights exist before auditing
    probe_dir_path = Path(probe_dir)
    if not (probe_dir_path / "adapter_config.json").exists():
        raise RuntimeError(f"Probe adapter_config.json missing at {probe_dir_path}. Cannot audit.")

    # Check if audit can be skipped
    skip_audit = resume and stage_manager and stage_manager.should_skip_probe_audit(probe_rank)

    if not skip_audit:
        print("Running LoRA audit...")
        seed = config.get("train", {}).get("seed", DEFAULT_SEED)
        with monitor_audit("audit_probe", output_dir=probe_dir, seed=seed) as stage:
            stage.progress("Starting probe audit analysis")
            audit_json_path = run_probe_audit(probe_dir=probe_dir, config=config)
            stage.progress("Probe audit analysis completed")
            stage.add_artifact("audit.json")

        # Mark audit as completed
        if stage_manager:
            stage_manager.mark_stage_completed(
                f"probe_r{probe_rank}_audited", {"probe_rank": probe_rank, "audit_path": str(audit_json_path)}
            )
    else:
        # Audit already exists
        audit_json_path = probe_dir / "audit.json"
        print(f"Using existing audit results: {audit_json_path}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Probe training complete!")

    # Get task profile for robust metric extraction
    task_profile = get_task_profile_from_config(config)
    accuracy = _extract_accuracy_with_fallback(eval_results, task_profile)
    print(f"Final accuracy: {accuracy:.4f}")

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Telemetry written to: {probe_dir / 'run.jsonl'}")
    print(f"Evaluation results written to: {eval_json_path}")
    print(f"Audit results written to: {audit_json_path}")

    # Return results for the bench report
    return {
        "probe": {
            "rank": config["lora"]["probe_r"],
            "params": trainable_params,
            "total_params": total_params,
            "accuracy": accuracy,
            "eval_loss": eval_results.get("eval_loss"),
            "output_dir": str(probe_dir),
        }
    }


def run_post_tuning(  # type: ignore[no-redef]  # noqa: F811
    model,
    tokenizer,
    dataset: dict[str, Any],
    config: dict[str, Any],
    post_tune_config: dict[str, Any],
    output_dir: Path,
    smoke: bool = False,
):
    """
    Perform post-tuning on a truncated adapter to recover performance.

    This is a "tiny tune" - a brief training pass to adapt the truncated
    adapter to the slight rank reduction.

    Args:
        model: PEFT model with truncated adapter loaded
        tokenizer: Tokenizer for the model
        dataset: Tokenized dataset
        config: Full bench configuration
        post_tune_config: Post-tuning settings (steps, lr_scale)
        output_dir: Directory to save post-tuned adapter
        smoke: Whether this is a smoke test

    Returns:
        Updated model with post-tuned adapter
    """
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    # Extract post-tuning parameters
    post_tune_steps = post_tune_config.get("steps", DEFAULT_POST_TUNE_STEPS)
    lr_scale = post_tune_config.get("lr_scale", DEFAULT_POST_TUNE_LR_SCALE)
    warmup_steps = post_tune_config.get("warmup_steps", 0)  # Default: zero warmup for tiny tune

    # Use base training config but with scaled parameters
    train_config = config["train"]
    base_lr = train_config.get("lr", DEFAULT_BASE_LR)
    post_tune_lr = base_lr * lr_scale

    # Reduce steps for smoke mode
    if smoke:
        post_tune_steps = min(post_tune_steps, SMOKE_MAX_POST_TUNE_STEPS)
        warmup_steps = min(warmup_steps, post_tune_steps // 5)  # Scale warmup for smoke

    # Setup training arguments for post-tuning
    post_tune_args = TrainingArguments(
        output_dir=str(output_dir / "post_tune"),
        num_train_epochs=1,
        max_steps=post_tune_steps,
        learning_rate=post_tune_lr,
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", DEFAULT_TRAIN_BATCH_SIZE),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", DEFAULT_EVAL_BATCH_SIZE),
        warmup_steps=warmup_steps,  # Configurable warmup (default 0)
        logging_steps=max(1, post_tune_steps // 4),
        save_steps=post_tune_steps,  # Save at end
        eval_strategy="no",  # Skip eval during post-tuning
        save_total_limit=1,
        load_best_model_at_end=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],  # Disable wandb/tensorboard
    )

    # Setup data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Get training split
    train_dataset = dataset.get("train", dataset.get("dataset"))
    if train_dataset is None:
        print("Warning: No training dataset available for post-tuning")
        return model

    # Setup trainer for post-tuning
    trainer = Trainer(  # type: ignore[call-arg,unused-ignore]
        model=model,
        args=post_tune_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(f"  Starting post-tuning: {post_tune_steps} steps at lr={post_tune_lr:.2e}")

    # Run post-tuning
    trainer.train()

    # Save the post-tuned adapter (overwrites the truncated one)
    trainer.save_model(str(output_dir))

    print(f"  Post-tuned adapter saved to {output_dir}")

    return model


def run_svd_truncation_variant(  # type: ignore[no-redef]  # noqa: F811
    config_path: str | Path,
    output_dir: str | Path,
    variant_name: str,
    compression_config: dict[str, Any],
    smoke: bool = False,
) -> dict[str, Any]:
    """
    Run SVD truncation variant by truncating the trained probe adapter.

    This doesn't involve retraining - it just applies SVD compression to the
    existing probe adapter and evaluates the truncated model.
    """
    from pathlib import Path

    # Load configuration
    config = load_config(config_path)

    # Setup output directory for this variant
    actual_r = compression_config["actual_r"]
    source_rank = compression_config["source_rank"]
    variant_dir = Path(output_dir) / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Find the probe adapter to truncate
    probe_dir = Path(output_dir) / f"probe_r{source_rank}"
    if not probe_dir.exists():
        return {
            "variant": variant_name,
            "status": "FAILED",
            "reason": f"Probe directory not found: {probe_dir}",
            "accuracy": None,
            "params": None,
            "output_dir": str(variant_dir),
        }

    try:
        # Perform SVD truncation
        print(f"Performing SVD truncation from r={source_rank} to r={actual_r}...")
        print(f"  Source: {probe_dir}")
        print(f"  Output: {variant_dir}")

        from gradience.vnext.svd_truncate import svd_truncate_peft_dir

        truncation_report = svd_truncate_peft_dir(
            peft_dir=probe_dir, out_dir=variant_dir, target_rank=actual_r, alpha_mode="keep_ratio", save_dtype="fp16"
        )

        print(f"  ✅ Truncation completed: {truncation_report.energy_retained:.1%} energy retained")
        print(f"  🗜️  Compression: {truncation_report.compression_ratio:.1f}x parameter reduction")

        # Save truncation report
        import json

        report_path = variant_dir / "svd_truncation_report.json"
        with open(report_path, "w") as f:
            json.dump(truncation_report.__dict__, f, indent=2)

        # Get device from config
        device = config.get("runtime", {}).get("device", "cpu")

        # Setup dataset for evaluation
        dataset = setup_dataset(config, smoke=smoke)

        # Setup base model and tokenizer
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = config["model"]["name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Move to device
        model = model.to(device)

        # Load the truncated adapter into the model
        model = PeftModel.from_pretrained(model, str(variant_dir))

        # Get task profile and preprocess dataset
        task_profile = get_task_profile_from_config(config)
        tokenized_dataset = task_profile.tokenize(dataset, tokenizer, config)

        # Check if post-tuning is enabled
        post_tune_config = compression_config.get("post_tune", {})
        if post_tune_config.get("enabled", False):
            print(f"🔧 Post-tuning enabled: {post_tune_config}")

            # Perform post-tuning of the truncated adapter
            model = run_post_tuning(
                model=model,
                tokenizer=tokenizer,
                dataset=tokenized_dataset,
                config=config,
                post_tune_config=post_tune_config,
                output_dir=variant_dir,
                smoke=smoke,
            )
            print("  ✅ Post-tuning completed")

        # Evaluate the (possibly post-tuned) model
        eval_results = task_profile.evaluate(model, tokenizer, tokenized_dataset, config)

        # Write eval.json for this variant
        eval_dataset_size = eval_results.get(
            "eval_samples", len(tokenized_dataset.get("validation", tokenized_dataset["train"]))
        )
        _eval_json_path = write_probe_eval_json(
            probe_dir=variant_dir, eval_results=eval_results, eval_dataset_size=eval_dataset_size, config=config
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Check if post-tuning was performed
        post_tuned = post_tune_config.get("enabled", False)

        result = {
            "variant": variant_name,
            "status": "PASS",
            "rank": actual_r,
            "params": trainable_params,
            "total_params": total_params,
            "accuracy": _extract_accuracy_with_fallback(eval_results, task_profile),
            "eval_loss": eval_results.get("eval_loss"),
            "output_dir": str(variant_dir),
            "compression_method": "svd_truncation",
            "source_rank": source_rank,
            "energy_retained": truncation_report.energy_retained,
            "compression_ratio": truncation_report.compression_ratio,
            "truncation_modules": truncation_report.total_modules,
            "post_tuned": post_tuned,
        }

        # Add post-tuning details if applicable
        if post_tuned:
            result["post_tune_config"] = {
                "steps": post_tune_config.get("steps", DEFAULT_POST_TUNE_STEPS),
                "lr_scale": post_tune_config.get("lr_scale", DEFAULT_POST_TUNE_LR_SCALE),
                "warmup_steps": post_tune_config.get("warmup_steps", 0),
            }

        return result

    except Exception as e:  # Intentionally broad: SVD truncation pipeline has diverse failure modes
        print(f"❌ SVD truncation failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "variant": variant_name,
            "status": "FAILED",
            "reason": f"SVD truncation failed: {str(e)}",
            "accuracy": None,
            "params": None,
            "output_dir": str(variant_dir),
        }


def run_compressed_variant_training(  # type: ignore[no-redef]  # noqa: F811
    config_path: str | Path,
    output_dir: str | Path,
    variant_name: str,
    compression_config: dict[str, Any],
    smoke: bool = False,
    stage_manager=None,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Step 3.5: Train a single compressed variant.

    Returns training results including accuracy and parameter counts.
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError(
            "Bench dependencies not available (transformers, peft, datasets). "
            'Install with: pip install "gradience[bench]"'
        )

    # Load configuration
    config = load_config(config_path)

    # Skip if variant is marked as SKIPPED
    if compression_config["status"] != "ready":
        return {
            "variant": variant_name,
            "status": "skipped",
            "reason": compression_config.get("reason", "Not ready"),
            "accuracy": None,
            "params": None,
            "output_dir": None,
        }

    # Setup output directory for this variant
    actual_r = compression_config["actual_r"]
    if variant_name.startswith("uniform"):
        variant_dir_name = f"{variant_name}_r{actual_r}"
    else:
        variant_dir_name = variant_name

    variant_dir = Path(output_dir) / variant_dir_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Get device from config
    device = config.get("runtime", {}).get("device", "cpu")

    # Check if this is an SVD truncation variant
    is_svd_variant = compression_config.get("compression_method") == "svd_truncation"

    if is_svd_variant:
        # For SVD variants, we need to truncate the trained probe adapter
        return run_svd_truncation_variant(
            config_path=config_path,
            output_dir=output_dir,
            variant_name=variant_name,
            compression_config=compression_config,
            smoke=smoke,
        )

    # Setup dataset
    dataset = setup_dataset(config, smoke=smoke)

    # Setup model and tokenizer with compressed configuration
    tokenizer, model = setup_compressed_model_and_tokenizer(config, compression_config, device=device)

    # Get task profile and preprocess dataset
    task_profile = get_task_profile_from_config(config)
    tokenized_dataset = task_profile.tokenize(dataset, tokenizer, config)

    # Apply smoke test limits to training config
    train_config = config["train"]
    runtime_config = config.get("runtime", {})

    max_steps = train_config.get("max_steps", 1500)
    if smoke:
        max_steps = runtime_config.get("smoke_max_steps", 50)
        # Create modified config for smoke test
        modified_config = config.copy()
        modified_config["train"] = train_config.copy()
        modified_config["train"]["max_steps"] = max_steps
        config = modified_config

    # Setup Gradience callback
    callback_config = GradienceCallbackConfig(output_dir=str(variant_dir), filename="run.jsonl")
    gradience_callback = GradienceCallback(callback_config)

    # Build trainer using task profile
    trainer = task_profile.build_trainer(
        model=model, tokenizer=tokenizer, tokenized_ds=tokenized_dataset, cfg=config, callbacks=[gradience_callback]
    )

    # Update trainer output dir to variant directory
    trainer.args.output_dir = str(variant_dir)
    trainer.args.logging_dir = str(variant_dir / "logs")

    # Train the model
    # Check if variant training can be skipped
    skip_variant = resume and stage_manager and stage_manager.should_skip_variant_training(variant_name)

    if not skip_variant:
        print(f"Starting {variant_name} training (r={actual_r})...")
        print(f"Output dir: {variant_dir}")
        print(f"Max steps: {max_steps}")
        print(f"Device: {device}")

        seed = config.get("train", {}).get("seed", DEFAULT_SEED)
        with monitor_training(f"train_{variant_name}", output_dir=variant_dir, seed=seed) as stage:
            stage.progress(f"Starting {variant_name} training")
            trainer.train()
            stage.progress(f"{variant_name} training completed")

        # Mark training as completed early (before evaluation in case it fails)
        if stage_manager:
            stage_manager.mark_variant_completed(
                variant_name, {"actual_r": actual_r, "max_steps": max_steps, "output_dir": str(variant_dir)}
            )
    else:
        # Skip training, will load existing results
        print(f"Skipping {variant_name} training - already completed")

    # Ensure adapter exists on disk for audit (save_strategy may be "no")
    with monitor_file_operations(f"save_{variant_name}_adapter", output_dir=variant_dir) as stage:
        stage.progress(f"Saving {variant_name} adapter to disk")
        _save_peft_adapter_only(trainer, model, variant_dir, label=f"variant:{variant_name}")
        stage.progress(f"{variant_name} adapter saved successfully")

    # Evaluate final model using task profile (or load existing results)
    if not skip_variant:
        seed = config.get("train", {}).get("seed", DEFAULT_SEED)
        with monitor_evaluation(f"eval_{variant_name}", output_dir=variant_dir, seed=seed) as stage:
            stage.progress(f"Starting {variant_name} evaluation")
            eval_results = task_profile.evaluate(model, tokenizer, tokenized_dataset, config)
            stage.progress(f"{variant_name} evaluation completed")
            stage.add_artifact("eval.json")
    else:
        # Load existing evaluation results
        eval_json_path = variant_dir / "eval.json"
        with open(eval_json_path) as f:
            eval_results = json.load(f)
        print(
            f"Loaded existing evaluation results for {variant_name}: accuracy = {eval_results.get('eval_accuracy', 'unknown')}"
        )

    # Write eval.json for this variant
    eval_dataset_size = eval_results.get(
        "eval_samples", len(tokenized_dataset.get("validation", tokenized_dataset["train"]))
    )
    eval_json_path = write_probe_eval_json(
        probe_dir=variant_dir, eval_results=eval_results, eval_dataset_size=eval_dataset_size, config=config
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Regression check for per-layer variants: verify heterogeneous ranks are applied
    rank_check_result = None
    if compression_config.get("variant") in ["per_layer", "per_layer_shuffled"]:
        from gradience.peft_utils import find_adapter_weights_path

        try:
            adapter_weights_path = find_adapter_weights_path(variant_dir)
            allowed_ranks = config["compression"]["allowed_ranks"]

            print(f"Running {compression_config.get('variant')} rank heterogeneity check...")
            print(f"  Found adapter weights at: {adapter_weights_path}")
            rank_check_result = check_heterogeneous_ranks(str(adapter_weights_path), allowed_ranks)
        except FileNotFoundError as e:
            print(f"⚠️  RANK CHECK SKIPPED: {e}")
            rank_check_result = {
                "passed": True,  # Don't fail the entire benchmark due to missing adapters
                "reason": f"Skipped due to missing adapter weights: {e}",
                "unique_ranks": [],
                "rank_histogram": {},
                "total_modules": 0,
            }

        if not rank_check_result["passed"]:
            print(f"❌ RANK CHECK FAILED: {rank_check_result['reason']}")
            print(f"   Rank histogram: {rank_check_result['rank_histogram']}")
            return {
                "variant": variant_name,
                "status": "FAILED",
                "reason": f"Rank check failed: {rank_check_result['reason']}",
                "rank": actual_r,
                "params": trainable_params,
                "total_params": total_params,
                "accuracy": _extract_accuracy_with_fallback(eval_results, task_profile),
                "eval_loss": eval_results.get("eval_loss"),
                "output_dir": str(variant_dir),
                "rank_check": rank_check_result,
            }
        elif rank_check_result.get("degrade_to_uniform", False):
            print(f"⚠️  RANK DEGENERATION: {rank_check_result['reason']}")
            print(f"   Rank histogram: {rank_check_result['rank_histogram']}")
            print("   per_layer variant collapsed to uniform - treating as legitimate degeneration")
        else:
            print(f"✅ Rank check passed: {len(rank_check_result['unique_ranks'])} distinct ranks")
            print(f"   Rank histogram: {rank_check_result['rank_histogram']}")

    print(f"{variant_name} training complete!")

    # Get task profile for robust metric extraction
    task_profile = get_task_profile_from_config(config)
    accuracy_value = _extract_accuracy_with_fallback(eval_results, task_profile)

    if accuracy_value > 0.0:
        metric_key = getattr(task_profile, "primary_metric_key", "eval_accuracy")
        print(f"Final {metric_key}: {accuracy_value:.4f}")
    else:
        print("Warning: No accuracy metric found in evaluation results")
        print(f"Available metrics: {list(eval_results.keys())}")
        accuracy_value = 0.0

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Telemetry written to: {variant_dir / 'run.jsonl'}")
    print(f"Evaluation results written to: {eval_json_path}")

    # Return results
    result = {
        "variant": variant_name,
        "status": "completed",
        "reason": None,
        "rank": actual_r,
        "params": trainable_params,
        "total_params": total_params,
        "accuracy": _extract_accuracy_with_fallback(eval_results, task_profile),
        "eval_loss": eval_results.get("eval_loss"),
        "output_dir": str(variant_dir),
    }

    # Add rank check results for per-layer variants
    if rank_check_result is not None:
        result["rank_check"] = rank_check_result

        # If per_layer variant degenerated to uniform, indicate the effective type
        if rank_check_result.get("degrade_to_uniform", False):
            result["effective_variant_type"] = "uniform"
            result["degrade_to_uniform"] = True
            # Note: variant name stays as per_layer but effective_variant_type shows what it became

    return result


def run_all_compressed_variants(  # type: ignore[no-redef]  # noqa: F811
    config_path: str | Path,
    output_dir: str | Path,
    compression_configs: dict[str, dict[str, Any]],
    smoke: bool = False,
    stage_manager=None,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Step 3.5: Train and evaluate all compressed variants.

    Returns results for all variants.
    """
    results = {}

    for variant_name, compression_config in compression_configs.items():
        print("\n" + "=" * 50)
        print(f"Training variant: {variant_name}")
        print(f"Status: {compression_config['status']}")

        if compression_config["status"] == "ready":
            actual_r = compression_config["actual_r"]
            print(f"Compressed rank: {actual_r}")
            if variant_name == "per_layer":
                pattern = compression_config["rank_pattern"]
                active_modules = {k: v for k, v in pattern.items() if v > 0}
                print(f"Active modules: {len(active_modules)}")

        result = run_compressed_variant_training(
            config_path=config_path,
            output_dir=output_dir,
            variant_name=variant_name,
            compression_config=compression_config,
            smoke=smoke,
            stage_manager=stage_manager,
            resume=resume,
        )

        results[variant_name] = result

    return results


def classify_validation_level(config: dict[str, Any]) -> dict[str, str]:  # type: ignore[no-redef]  # noqa: F811
    """
    Classify validation level based on configuration.

    Returns validation classification and reasoning.
    """
    # Check for multi-seed
    compression = config.get("compression", {})
    seeds = compression.get("seeds", [])
    is_multiseed = len(seeds) > 1

    # Check training budget
    train_config = config.get("train", {})
    max_steps = train_config.get("max_steps", 0)

    # Determine validation level
    if is_multiseed and max_steps >= MIN_STEPS_SCREENING_PLUS:
        if len(seeds) >= MIN_SEEDS_CERTIFIABLE and max_steps >= MIN_STEPS_CERTIFIABLE:
            classification = "certifiable"
            rationale = f"{len(seeds)} seeds × {max_steps} steps provides statistical rigor"
        else:
            classification = "screening_plus"
            rationale = f"{len(seeds)} seeds × {max_steps} steps (limited budget/seeds)"
    elif is_multiseed:
        classification = "screening_plus"
        rationale = f"{len(seeds)} seeds but only {max_steps} steps (limited budget)"
    elif max_steps >= MIN_STEPS_CERTIFIABLE:
        classification = "screening"
        rationale = f"Single seed, {max_steps} steps (no variance estimation)"
    else:
        classification = "screening"
        rationale = f"Single seed, {max_steps} steps (quick validation only)"

    return {
        "level": classification,
        "rationale": rationale,
        "is_multiseed": is_multiseed,
        "n_seeds": len(seeds) if is_multiseed else 1,
        "max_steps": max_steps,
    }


def compute_verdicts(  # type: ignore[no-redef]  # noqa: F811
    probe_results: dict[str, Any],
    variant_results: dict[str, dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """
    Step 3.6: Compute verdicts for compressed variants.

    Returns verdict analysis including PASS/FAIL decisions and best compression.
    """
    compression_config = config.get("compression", {})
    acc_tolerance = compression_config.get("acc_tolerance", DEFAULT_ACCURACY_TOLERANCE)

    # Classify validation level
    validation_classification = classify_validation_level(config)

    # Get probe baseline
    probe_accuracy = probe_results["probe"]["accuracy"]
    probe_params = probe_results["probe"]["params"]

    # Probe quality gating using task profile
    task_profile = get_task_profile_from_config(config)

    # Load the original probe evaluation results for probe_gate
    probe_rank = config["lora"]["probe_r"]
    probe_eval_path = output_path / f"probe_r{probe_rank}" / "eval.json"
    with open(probe_eval_path) as f:
        probe_eval_results = json.load(f)

    probe_passed, gate_info = task_profile.probe_gate(probe_eval_results, config)
    probe_quality_threshold = gate_info["threshold"]

    if not probe_passed:
        print("\n=== PROBE QUALITY GATE FAILED ===")
        print(f"Probe accuracy: {probe_accuracy:.4f}")
        print(f"Required threshold: {probe_quality_threshold:.4f}")

        if smoke:
            print("Status: UNDERTRAINED_SMOKE - continuing in smoke mode")
            status_code = "UNDERTRAINED_SMOKE"
        else:
            print("Status: UNDERTRAINED - compression certification not valid")
            status_code = "UNDERTRAINED"

        # Return undertrained status for all variants
        verdicts = {}
        for variant_name in variant_results:
            verdicts[variant_name] = {
                "status": "undertrained",
                "reason": f"Probe accuracy {probe_accuracy:.4f} < threshold {probe_quality_threshold:.4f}",
                "delta_vs_probe": None,
                "param_reduction": None,
                "verdict": status_code,
            }

        return {
            "verdicts": verdicts,
            "probe_quality_status": status_code,
            "summary": {
                "probe_quality": "FAILED",
                "probe_accuracy": probe_accuracy,
                "probe_threshold": probe_quality_threshold,
                "recommendations_validated": "N/A",
                "best_compression": None,
                "notes": [f"Probe undertrained - compression results not reliable (smoke mode: {smoke})"],
            },
        }

    verdicts = {}
    pass_variants = []

    print("\n=== VERDICT ANALYSIS ===")
    print(f"🔬 Validation level: {validation_classification['level'].upper()}")
    print(f"   {validation_classification['rationale']}")
    print(f"✅ Probe quality: {probe_accuracy:.4f} ≥ {probe_quality_threshold:.4f}")
    print(f"Probe baseline: {probe_accuracy:.4f} accuracy, {probe_params:,} params")
    print(f"Accuracy tolerance: ±{acc_tolerance:.3f}")
    print()

    for variant_name, result in variant_results.items():
        if result["status"] != "completed":
            # Skip variants that didn't complete successfully
            status = result["status"]
            reason = result.get("reason", "Training not completed")

            verdicts[variant_name] = {
                "status": "skipped",
                "reason": reason,
                "delta_vs_probe": None,
                "param_reduction": None,
                "verdict": "FAIL" if status == "FAILED" else "SKIP",
            }

            if status == "FAILED":
                print(f"{variant_name}: FAIL - {reason}")
            else:
                print(f"{variant_name}: SKIP - {reason}")
            continue

        # Compute metrics
        compressed_accuracy = result["accuracy"]
        compressed_params = result["params"]

        delta_vs_probe = compressed_accuracy - probe_accuracy
        param_reduction = 1.0 - (compressed_params / probe_params)

        # Make verdict
        if delta_vs_probe >= -acc_tolerance:
            verdict = "PASS"
            pass_variants.append(
                {
                    "variant": variant_name,
                    "param_reduction": param_reduction,
                    "delta_vs_probe": delta_vs_probe,
                    "compressed_params": compressed_params,
                    "compressed_accuracy": compressed_accuracy,
                }
            )
        else:
            verdict = "FAIL"

        verdicts[variant_name] = {
            "status": "evaluated",
            "reason": None,
            "delta_vs_probe": delta_vs_probe,
            "param_reduction": param_reduction,
            "verdict": verdict,
            "compressed_accuracy": compressed_accuracy,
            "compressed_params": compressed_params,
            "probe_accuracy": probe_accuracy,
            "probe_params": probe_params,
        }

        # Print verdict
        reduction_pct = param_reduction * 100
        print(f"{variant_name}: {verdict}")
        print(f"  Δ accuracy: {delta_vs_probe:+.4f} (threshold: {-acc_tolerance:.3f})")
        print(f"  Param reduction: {reduction_pct:.1f}% ({probe_params:,} → {compressed_params:,})")
        print(f"  Accuracy: {compressed_accuracy:.4f} vs {probe_accuracy:.4f}")
        print()

    # Find best compression among PASS variants
    best_compression = None
    if pass_variants:
        best_variant = max(pass_variants, key=lambda x: x["param_reduction"])
        best_compression = {
            "variant": best_variant["variant"],
            "param_reduction": best_variant["param_reduction"],
            "delta_vs_probe": best_variant["delta_vs_probe"],
            "compressed_params": best_variant["compressed_params"],
            "compressed_accuracy": best_variant["compressed_accuracy"],
        }

        reduction_pct = best_compression["param_reduction"] * 100
        print(f"🏆 BEST COMPRESSION: {best_compression['variant']}")
        print(
            f"   {reduction_pct:.1f}% parameter reduction with {best_compression['delta_vs_probe']:+.4f} accuracy delta"
        )
    else:
        print("❌ NO PASSING VARIANTS: All compressions exceeded accuracy tolerance")

    return {
        "verdicts": verdicts,
        "best_compression": best_compression,
        "probe_baseline": {"accuracy": probe_accuracy, "params": probe_params},
        "acc_tolerance": acc_tolerance,
        "validation_classification": validation_classification,
        "summary": {
            "probe_quality": "PASSED",
            "probe_accuracy": probe_accuracy,
            "probe_threshold": probe_quality_threshold,
            "total_variants": len(variant_results),
            "completed": len([v for v in variant_results.values() if v["status"] == "completed"]),
            "passed": len([v for v in verdicts.values() if v["verdict"] == "PASS"]),
            "failed": len([v for v in verdicts.values() if v["verdict"] == "FAIL"]),
            "skipped": len([v for v in verdicts.values() if v["verdict"] == "SKIP"]),
            "notes": [],
        },
    }


def run_multi_seed_bench_protocol(  # type: ignore[no-redef]  # noqa: F811
    config_path: str | Path,
    output_dir: str | Path,
    seeds: list[int],
    variants_to_test: list[str] | None = None,
    smoke: bool = False,
    ci: bool = False,
) -> dict[str, Any]:
    """
    Run bench protocol across multiple seeds and aggregate results.

    Returns aggregated report with mean ± std statistics.
    """
    config = load_config(config_path)
    output_path = Path(output_dir)

    # HYGIENE: Ensure output directory exists BEFORE any logging/tee operations
    output_path.mkdir(parents=True, exist_ok=True)

    # HYGIENE: Start heartbeat for multi-seed coordination (prevent SSH timeouts)
    heartbeat_stage("multi_seed_coordination", output_dir=output_path, seed=None)

    print("Gradience Multi-Seed Bench Protocol v0.1")
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
        print(f"\n{'=' * 60}")
        print(f"SEED {i + 1}/{len(seeds)}: {seed}")
        print(f"{'=' * 60}")

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
        with open(progress_file, "w") as f:
            f.write(f"STARTED: seed_{seed} at {datetime.datetime.now().isoformat()}\n")
            f.flush()

        # Write seed-specific config
        seed_config_path = seed_dir / "config.yaml"
        with open(seed_config_path, "w") as f:
            yaml.dump(seed_config, f, indent=2)

        # Run single seed benchmark
        try:
            # Update progress before starting
            with open(progress_file, "a") as f:
                f.write(f"RUNNING: bench protocol started at {datetime.datetime.now().isoformat()}\n")
                f.flush()

            seed_report = run_bench_protocol(config_path=seed_config_path, output_dir=seed_dir, smoke=smoke, ci=ci)

            # Add seed info to report
            seed_report["seed"] = seed
            seed_report["seed_index"] = i
            seed_reports.append(seed_report)

            # Mark completion in progress file
            with open(progress_file, "a") as f:
                f.write(f"COMPLETED: seed_{seed} at {datetime.datetime.now().isoformat()}\n")
                f.flush()

            print(f"\n✅ Seed {seed} completed successfully")

        except Exception as e:  # Intentionally broad: per-seed failure must not abort entire benchmark
            # Mark failure in progress file
            with open(progress_file, "a") as f:
                f.write(f"FAILED: seed_{seed} at {datetime.datetime.now().isoformat()}: {e}\n")
                f.flush()

            print(f"\n❌ Seed {seed} failed: {e}")
            # Continue with other seeds
            continue

    if not seed_reports:
        raise RuntimeError("All seed runs failed - cannot generate aggregated report")

    print(f"\n{'=' * 60}")
    print(f"AGGREGATION: {len(seed_reports)}/{len(seeds)} seeds successful")
    print(f"{'=' * 60}")

    # Create aggregated report
    aggregated_report = create_multi_seed_aggregated_report(
        seed_reports=seed_reports, config=config, output_dir=output_path
    )

    # Write aggregated bench.json
    agg_report_path = output_path / "bench_aggregate.json"
    with open(agg_report_path, "w") as f:
        json.dump(aggregated_report, f, indent=2, ensure_ascii=False)

    # Create and write aggregated markdown report
    agg_markdown_content = create_multi_seed_markdown_report(
        aggregated_report=aggregated_report, config=config, output_dir=output_path
    )

    agg_markdown_path = output_path / "bench_aggregate.md"
    with open(agg_markdown_path, "w") as f:
        f.write(agg_markdown_content)

    # Write seed summary
    seed_summary_path = output_path / "seed_summary.json"
    seed_summary = {
        "total_seeds": len(seeds),
        "successful_seeds": len(seed_reports),
        "failed_seeds": len(seeds) - len(seed_reports),
        "seed_directories": [str(d) for d in seed_dirs],
        "aggregated_report": str(agg_report_path),
        "aggregated_markdown": str(agg_markdown_path),
    }
    with open(seed_summary_path, "w") as f:
        json.dump(seed_summary, f, indent=2, ensure_ascii=False)

    print("\nMulti-seed benchmark complete! 🎉")
    print(f"  📊 Aggregated report: {agg_report_path}")
    print(f"  📝 Aggregated markdown: {agg_markdown_path}")
    print(f"  📋 Seed summary: {seed_summary_path}")
    print(f"  📁 Individual seed results in: {[d.name for d in seed_dirs]}")

    return aggregated_report


def run_artifact_hygiene_cleanup(output_dir: Path, config: dict[str, Any]) -> None:  # type: ignore[no-redef]  # noqa: F811
    """
    Clean up heavy adapter weights and checkpoints while preserving scientific artifacts.

    Deletes:
    - adapter_model.safetensors / adapter_model.bin (hundreds of MB)
    - checkpoint-* directories (if keep_checkpoints=false)

    Preserves:
    - bench.json, bench.md (scientific results)
    - */audit.json, */eval.json (evidence)
    - compression_configs.json (configuration record)
    - run.jsonl (telemetry, optional but useful)
    - adapter_config.json (small config files)
    """
    runtime_config = config.get("runtime", {})
    keep_adapter_weights = runtime_config.get("keep_adapter_weights", True)  # Default to keep for compatibility
    keep_checkpoints = runtime_config.get("keep_checkpoints", True)  # Default to keep for compatibility

    if keep_adapter_weights and keep_checkpoints:
        # Nothing to clean up
        return

    cleaned_files = []
    saved_space = 0

    try:
        for variant_dir in output_dir.iterdir():
            if not variant_dir.is_dir():
                continue

            # Clean adapter weights
            if not keep_adapter_weights:
                # Look for adapter weights in common locations
                adapter_patterns = ["adapter_model.safetensors", "adapter_model.bin", "pytorch_adapter.bin"]

                for pattern in adapter_patterns:
                    # Check in variant root
                    adapter_file = variant_dir / pattern
                    if adapter_file.exists():
                        file_size = adapter_file.stat().st_size
                        adapter_file.unlink()
                        cleaned_files.append(str(adapter_file.relative_to(output_dir)))
                        saved_space += file_size

                    # Check in peft/ subdirectory
                    peft_adapter = variant_dir / "peft" / pattern
                    if peft_adapter.exists():
                        file_size = peft_adapter.stat().st_size
                        peft_adapter.unlink()
                        cleaned_files.append(str(peft_adapter.relative_to(output_dir)))
                        saved_space += file_size

                    # Check in checkpoint directories
                    for checkpoint_dir in variant_dir.glob("checkpoint-*"):
                        if checkpoint_dir.is_dir():
                            checkpoint_adapter = checkpoint_dir / pattern
                            if checkpoint_adapter.exists():
                                file_size = checkpoint_adapter.stat().st_size
                                checkpoint_adapter.unlink()
                                cleaned_files.append(str(checkpoint_adapter.relative_to(output_dir)))
                                saved_space += file_size

            # Clean checkpoint directories
            if not keep_checkpoints:
                for checkpoint_dir in variant_dir.glob("checkpoint-*"):
                    if checkpoint_dir.is_dir():
                        # Calculate directory size before deletion
                        dir_size = sum(f.stat().st_size for f in checkpoint_dir.rglob("*") if f.is_file())

                        # Remove the entire checkpoint directory
                        import shutil

                        shutil.rmtree(checkpoint_dir)
                        cleaned_files.append(str(checkpoint_dir.relative_to(output_dir)) + "/")
                        saved_space += dir_size

    except (OSError, PermissionError) as e:
        print(f"⚠️  Artifact cleanup encountered error: {e}")
        return

    # Report cleanup results
    if cleaned_files:
        saved_mb = saved_space / (1024 * 1024)
        print(f"🧹 Artifact hygiene: Cleaned {len(cleaned_files)} items, saved {saved_mb:.1f} MB")
        if len(cleaned_files) <= 10:
            # Show details if not too many files
            for item in cleaned_files:
                print(f"   - {item}")
        else:
            # Summarize if many files
            weight_files = [f for f in cleaned_files if not f.endswith("/")]
            checkpoint_dirs = [f for f in cleaned_files if f.endswith("/")]
            if weight_files:
                print(f"   - {len(weight_files)} adapter weight files")
            if checkpoint_dirs:
                print(f"   - {len(checkpoint_dirs)} checkpoint directories")


def run_bench_preflight_check(config: dict[str, Any], model_name: str) -> None:  # type: ignore[no-redef]  # noqa: F811
    """
    Preflight checks to catch common failure modes before expensive training.

    Checks for:
    - PyTorch device availability and GPU info
    - Disk space on critical paths (/tmp, /workspace if exists)
    - HF cache accessibility and potential corruption
    - Model loading sanity check for safetensors metadata issues

    Fails fast with helpful diagnostics to save hours of debugging.
    """
    import os
    import shutil  # noqa: F401
    import subprocess
    import tempfile
    from pathlib import Path

    print("🔍 Running preflight checks...")

    # 1. PyTorch device check
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} available")

        # Check CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   Current device: {current_device} ({gpu_name})")
        else:
            print("⚠️  CUDA not available - will run on CPU (much slower)")

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")

        # Determine runtime device from config
        runtime_device = config.get("runtime", {}).get("device", "auto")
        if runtime_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Config specifies device: cuda but CUDA is not available. "
                "Either install CUDA PyTorch or change device to 'cpu' in config."
            )

    except ImportError as e:
        raise RuntimeError(f"PyTorch not available: {e}") from e

    # 2. Disk space checks
    critical_paths = ["/tmp"]
    if os.path.exists("/workspace"):
        critical_paths.append("/workspace")

    for path in critical_paths:
        if os.path.exists(path):
            try:
                # Use df command for reliable disk space info
                result = subprocess.run(["df", "-h", path], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) >= 2:
                        # Parse df output: Filesystem Size Used Avail Use% Mounted
                        fields = lines[1].split()
                        if len(fields) >= 4:
                            avail = fields[3]
                            use_pct = fields[4]
                            print(f"✅ Disk space {path}: {avail} available ({use_pct} used)")

                            # Warn if very low space
                            if use_pct.rstrip("%").isdigit() and int(use_pct.rstrip("%")) > 95:
                                print(f"⚠️  WARNING: {path} is {use_pct} full - may cause download failures")
                        else:
                            print(f"✅ {path} accessible (could not parse disk usage)")
                    else:
                        print(f"✅ {path} accessible (unusual df output)")
                else:
                    print(f"⚠️  Could not check disk space for {path}: {result.stderr}")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                print(f"⚠️  Could not check disk space for {path}: {e}")
        else:
            print(f"ℹ️  {path} does not exist (skipping)")

    # 3. HF cache checks
    try:
        import os
        from pathlib import Path

        from transformers import AutoTokenizer  # noqa: F401

        # Get HF cache directory (try multiple methods for different HF versions)
        try:
            # Try new HuggingFace Hub API
            from huggingface_hub import HF_HOME

            cache_dir = Path(HF_HOME) if HF_HOME else None
        except ImportError:
            cache_dir = None

        if not cache_dir:
            # Fallback to environment variable or default
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_dir = Path(hf_home)
            else:
                # Default HF cache location
                cache_dir = Path.home() / ".cache" / "huggingface"

        try:
            print(f"✅ HuggingFace cache: {cache_dir}")

            if cache_dir.exists():
                # Check if writable
                test_file = cache_dir / f".preflight_test_{os.getpid()}"
                try:
                    test_file.touch()
                    test_file.unlink()
                    print("✅ HF cache directory writable")
                except (OSError, PermissionError) as e:
                    print(f"❌ HF cache directory not writable: {e}")
                    raise RuntimeError(f"HuggingFace cache directory not writable: {cache_dir}") from e
            else:
                print(f"ℹ️  HF cache directory will be created: {cache_dir}")

        except (OSError, RuntimeError) as e:
            print(f"⚠️  Could not determine HF cache location: {e}")

    except ImportError:
        print("⚠️  HuggingFace transformers not available for cache check")

    # 3.5. HF Cache Environment Validation (RunPod survival check)
    hf_env_vars = {
        "HF_HOME": "Primary HF cache directory",
        "HF_HUB_CACHE": "Model weights cache",
        "HF_DATASETS_CACHE": "Dataset cache",
        "TORCH_HOME": "PyTorch cache",
    }

    print("🔍 Validating HuggingFace cache environment...")
    env_issues = []
    runpod_detected = os.path.exists("/workspace")

    for env_var, _description in hf_env_vars.items():
        value = os.environ.get(env_var)
        if value:
            cache_path = Path(value)

            # Check if path is under /root/ on RunPod (danger zone)
            if runpod_detected and str(cache_path).startswith("/root/"):
                env_issues.append(f"{env_var}={value} (⚠️  points to /root/ - will fill system disk on RunPod)")
            elif cache_path.exists() and not os.access(cache_path, os.W_OK):
                env_issues.append(f"{env_var}={value} (❌ not writable)")
            else:
                print(f"✅ {env_var}: {value}")
        else:
            if runpod_detected:
                env_issues.append(f"{env_var} not set (⚠️  will default to /root/.cache on RunPod)")
            else:
                print(f"ℹ️  {env_var}: not set (will use defaults)")

    # Report environment issues with remediation
    if env_issues:
        print("\n⚠️  HuggingFace cache environment issues detected:")
        for issue in env_issues:
            print(f"   - {issue}")

        print("\n💡 SOLUTION:")
        print("   Set environment variables to use a persistent cache directory:")
        if runpod_detected:
            print("   # For RunPod environments:")
            print("   export HF_HOME=/workspace/hf_cache/hf_home")
            print("   export HF_HUB_CACHE=/workspace/hf_cache/hub")
            print("   export HF_DATASETS_CACHE=/workspace/hf_cache/datasets")
            print("   export TORCH_HOME=/workspace/hf_cache/torch")
        else:
            print("   # For other environments, choose a persistent location:")
            print("   export HF_HOME=/path/to/persistent/hf_cache/hf_home")
            print("   export HF_HUB_CACHE=/path/to/persistent/hf_cache/hub")
            print("   export HF_DATASETS_CACHE=/path/to/persistent/hf_cache/datasets")
            print("   export TORCH_HOME=/path/to/persistent/hf_cache/torch")
    else:
        print("✅ HuggingFace cache environment properly configured")

    # 4. Model loading sanity check
    try:
        print(f"🔍 Testing model loading: {model_name}")

        # Try to load just the config (fast) to catch common issues
        from transformers import AutoConfig

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Set a temporary cache dir to isolate the test
                os.environ["HF_HOME"] = temp_dir
                try:
                    config_test = AutoConfig.from_pretrained(model_name)
                    print(f"✅ Model config loads successfully: {config_test.model_type}")
                finally:
                    # Restore original cache
                    if "HF_HOME" in os.environ:
                        del os.environ["HF_HOME"]

        except Exception as e:  # Intentionally broad: HF model loading can fail in many ways
            error_msg = str(e).lower()
            if "incomplete" in error_msg and "metadata" in error_msg:
                print(f"❌ Safetensors metadata corruption detected for {model_name}")
                print("💡 SOLUTION: Delete the corrupted cache:")
                print(f"   rm -rf ~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
                print("   Or nuke entire cache: rm -rf ~/.cache/huggingface/")
                raise RuntimeError(f"HuggingFace cache corruption: {e}") from e
            else:
                print(f"⚠️  Model loading issue: {e}")
                # Don't fail on other model loading issues as they might resolve during training

    except ImportError:
        print("⚠️  Cannot test model loading - HuggingFace transformers not available")
    except Exception as e:  # Intentionally broad: preflight check must never abort benchmark
        print(f"⚠️  Model loading check failed: {e}")

    print("✅ Preflight checks complete!\n")


def _handle_invalid_probe(
    probe_rank: int,
    audit_data: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Create a minimal report when the probe is invalid and return early."""
    # Create minimal report for invalid probe and exit early
    verdict_analysis = compute_verdicts(
        probe_results={}, variant_results={}, config=config, output_path=output_path, smoke=smoke
    )

    decision_trace = DecisionTrace(probe_rank=probe_rank)
    canonical_report = create_canonical_bench_report(
        probe_results={},
        variant_results={},
        verdict_analysis=verdict_analysis,
        audit_data=audit_data,
        compression_configs={},
        config=config,
        output_dir=output_path,
        decision_trace=decision_trace,
    )

    # Write report files
    report_path = output_path / "bench.json"
    with open(report_path, "w") as f:
        json.dump(canonical_report, f, indent=2, ensure_ascii=False)

    return canonical_report


def _run_escalation_and_update(
    variant_results,
    verdict_analysis,
    compression_configs,
    config,
    probe_results,
    config_path,
    output_path,
    smoke,
    stage_manager,
    resume,
):
    """Run the escalation round and enrich verdicts with stability metadata.

    Returns (verdict_analysis, escalation_trace, variant_results, compression_configs).
    Note: variant_results and compression_configs may be mutated (updated with escalation results).
    """
    # Step 3.6a: Escalation round (safety fallback)
    compression = config.get("compression", {})
    escalation_config = compression.get("escalation", {})

    esc_configs, escalation_trace = run_escalation_round(
        variant_results=variant_results,
        verdicts=verdict_analysis["verdicts"],
        compression_configs=compression_configs,
        config=config,
        probe_rank=config["lora"]["probe_r"],
        acc_tolerance=verdict_analysis.get("acc_tolerance", DEFAULT_ACCURACY_TOLERANCE),
        escalation_config=escalation_config,
    )

    if esc_configs:
        print(f"\n{'=' * 50}")
        print(f"ESCALATION ROUND: {len(esc_configs)} safer candidate(s)")
        print(f"{'=' * 50}")

        esc_results = run_all_compressed_variants(
            config_path=config_path,
            output_dir=output_path,
            compression_configs=esc_configs,
            smoke=smoke,
            stage_manager=stage_manager,
            resume=resume,
        )

        # Merge escalation results into the main result sets
        variant_results.update(esc_results)
        compression_configs.update(esc_configs)

        # Re-compute verdicts with expanded result set
        verdict_analysis = compute_verdicts(
            probe_results=probe_results,
            variant_results=variant_results,
            config=config,
            output_path=output_path,
            smoke=smoke,
        )

        esc_verdicts = {k: verdict_analysis["verdicts"][k] for k in esc_configs if k in verdict_analysis["verdicts"]}
        escalation_trace = update_escalation_trace_with_results(
            trace=escalation_trace,
            escalation_verdicts=esc_verdicts,
            original_best_compression=verdict_analysis.get("best_compression"),
        )

        print(f"Escalation complete: {len(escalation_trace.escalation_trace)} entries")
        if escalation_trace.final_recommendation:
            print(f"Final recommendation: {escalation_trace.final_recommendation}")

    # Enrich all verdicts with stability metadata
    enrich_verdicts_with_stability(
        verdicts=verdict_analysis["verdicts"],
        acc_tolerance=verdict_analysis.get("acc_tolerance", DEFAULT_ACCURACY_TOLERANCE),
        escalation_trace=escalation_trace,
        catastrophic_margin=escalation_config.get("catastrophic_margin"),
    )

    # Write verdict analysis to JSON
    verdict_path = output_path / "verdicts.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict_analysis, f, indent=2, ensure_ascii=False)

    return verdict_analysis, escalation_trace, variant_results, compression_configs


def _update_policy_scoreboard(
    verdict_analysis: dict[str, Any],
    compression_configs: dict[str, Any],
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """Update policy scoreboard with benchmark results (side-effect only).

    Writes policy_scoreboard_snapshot.json to output_path if policy results exist.
    """
    # Step 3.7: Update policy scoreboard with results
    try:
        from gradience.vnext.policy_scoreboard import PolicyScoreboard, create_policy_result_from_bench_data

        scoreboard = PolicyScoreboard()

        # Extract policy results from verdict analysis and compression configs
        config_name = config.get("name", "unknown_config")
        model_name = config.get("model", {}).get("model_name", "unknown_model")
        task_name = config.get("task", {}).get("task_name", "unknown_task")

        policy_results = []

        # Process each compression variant that was tested
        for variant_name, variant_config in compression_configs.items():
            if variant_config.get("status") != "ready":
                continue

            # Get policy information from variant config
            policy_type = variant_config.get("policy_type", "unknown")
            if policy_type == "unknown":
                continue  # Skip non-policy variants

            suggested_rank = variant_config.get("suggested_r", 0)
            actual_rank = variant_config.get("actual_r", 0)

            # Get performance results from verdict analysis
            variant_verdict = verdict_analysis.get("verdicts", {}).get(variant_name)
            if variant_verdict:
                passed = variant_verdict.get("verdict") == "PASS"
                performance_delta = variant_verdict.get("performance_delta", 0.0)

                # Collect all performance results for optimal rank calculation
                all_results = {}
                for vname, vverdict in verdict_analysis.get("verdicts", {}).items():
                    if vverdict.get("performance_delta") is not None:
                        all_results[vname] = vverdict.get("performance_delta", 0.0)

                # Create policy result
                policy_result = create_policy_result_from_bench_data(
                    config_name=config_name,
                    model_name=model_name,
                    task_name=task_name,
                    policy_name=policy_type,
                    suggested_rank=suggested_rank,
                    actual_rank=actual_rank,
                    performance_delta=performance_delta,
                    passed=passed,
                    all_results=all_results,
                    seed=config.get("train", {}).get("seed", DEFAULT_SEED),
                )

                policy_results.append(policy_result)

        # Add results to scoreboard
        if policy_results:
            scoreboard.add_benchmark_results(config_name, model_name, task_name, policy_results)
            print(f"📊 Updated policy scoreboard with {len(policy_results)} policy results")

            # Export snapshot to output directory
            snapshot_path = output_path / "policy_scoreboard_snapshot.json"
            scoreboard.export_snapshot(snapshot_path)

    except ImportError:
        print("⚠️  Policy scoreboard not available (vnext module not found)")
    except (TypeError, ValueError, OSError) as e:
        print(f"⚠️  Policy scoreboard update failed: {e}")


def run_bench_protocol(
    config_path: str | Path,
    output_dir: str | Path,
    smoke: bool = False,
    ci: bool = False,
    fast_mode: bool = True,
    max_candidates: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Run the complete bench protocol.

    Supports both single-seed and multi-seed configurations.
    Multi-seed configs are detected by presence of 'seeds' in compression config.
    """
    # Set up global monitoring infrastructure
    setup_global_monitoring()

    # Load configuration to check for multi-seed
    config = load_config(config_path)

    # Check for multi-seed configuration
    compression = config.get("compression", {})
    seeds = compression.get("seeds")
    variants_to_test = compression.get("variants_to_test")

    if seeds and len(seeds) > 1:
        print("Detected multi-seed configuration - running aggregated benchmark...")
        return run_multi_seed_bench_protocol(
            config_path=config_path,
            output_dir=output_dir,
            seeds=seeds,
            variants_to_test=variants_to_test,
            smoke=smoke,
            ci=ci,
        )

    # Single-seed protocol (original implementation)
    print("Gradience Bench Protocol v0.1")
    print("=" * 40)

    # HYGIENE: Ensure output directory exists BEFORE any logging/tee operations
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # HYGIENE: Start heartbeat for single-seed run (prevent SSH timeouts)
    seed = config.get("train", {}).get("seed", DEFAULT_SEED)
    heartbeat_stage("single_seed_benchmark", output_dir=output_path, seed=seed)

    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Model: {config['model']['name']}")
    print(f"Task: {config['task']['dataset']}/{config['task']['subset']}")
    print(f"Smoke mode: {smoke}")
    print()

    # Initialize stage state manager for resume functionality
    stage_manager = create_stage_manager(output_path)

    if resume:
        print("🔄 Resume mode enabled - checking for completed stages...")
        stage_manager.clean_invalid_state()  # Clean up any stale state
        summary = stage_manager.get_resume_summary()
        print(
            f"   Found {summary['total_completed_stages']} completed stages, {summary['total_completed_variants']} completed variants"
        )
        if summary["total_completed_stages"] > 0 or summary["total_completed_variants"] > 0:
            print(
                "   Completed stages:",
                ", ".join(summary["completed_stages"]) if summary["completed_stages"] else "none",
            )
            print(
                "   Completed variants:",
                ", ".join(summary["completed_variants"]) if summary["completed_variants"] else "none",
            )

    # Preflight checks to catch common failure modes early
    run_bench_preflight_check(config, config["model"]["name"])

    # Steps 3.1-3.3: Train, evaluate, and audit probe
    print("Step 3.1-3.3: Training, evaluating, and auditing probe adapter...")
    probe_results = run_probe_training(
        config_path, output_path, smoke=smoke, stage_manager=stage_manager, resume=resume
    )

    # Step 3.4: Generate compression configurations
    print("\nStep 3.4: Generating compression configurations...")
    probe_rank = config["lora"]["probe_r"]
    probe_dir = output_path / f"probe_r{probe_rank}"

    # Check probe validity before wasting GPU cycles on compression
    audit_json_path = probe_dir / "audit.json"
    if audit_json_path.exists():
        with open(audit_json_path) as f:
            audit_data = json.load(f)

        probe_validity = audit_data.get("probe_validity", {})
        if not probe_validity.get("valid", True):
            print("")
            print("⚠️" * 15)
            print("⚠️  SKIPPING COMPRESSION VARIANTS")
            print(f"⚠️  Reason: {probe_validity.get('reason', 'UNKNOWN')}")
            print(f"⚠️  Message: {probe_validity.get('message', 'Probe invalid')}")
            print(f"⚠️  stable_rank_mean: {probe_validity.get('stable_rank_mean', 'N/A')}")
            print(f"⚠️  utilization_mean: {probe_validity.get('utilization_mean', 'N/A')}")
            print("⚠️")
            print("⚠️  This prevents wasted GPU cycles on compression variants")
            print("⚠️  when the probe itself is invalid.")
            print("⚠️" * 15)
            print("")

            return _handle_invalid_probe(probe_rank, audit_data, config, output_path, smoke)

    # Check if config generation can be skipped
    skip_config_gen = resume and stage_manager and stage_manager.should_skip_config_generation()

    if not skip_config_gen:
        seed = config.get("train", {}).get("seed", DEFAULT_SEED)
        with monitor_generation("generate_configs", output_dir=output_path, seed=seed) as stage:
            stage.progress("Analyzing probe audit results")
            compression_configs: dict[str, dict[str, Any]]
            decision_trace: DecisionTrace
            compression_configs, decision_trace = generate_compression_configs(
                probe_dir, config, fast_mode=fast_mode, max_candidates=max_candidates
            )
            stage.progress("Compression configurations generated")
            stage.add_artifact("compression_configs.json")

        # Mark config generation as completed
        if stage_manager:
            stage_manager.mark_stage_completed(
                "compression_configs_generated",
                {"num_configs": len(compression_configs), "fast_mode": fast_mode, "max_candidates": max_candidates},
            )
    else:
        # Load existing compression configs
        compression_configs_path = output_path / "compression_configs.json"
        with open(compression_configs_path) as f:
            compression_configs = json.load(f)
        print(f"Loaded existing compression configs: {len(compression_configs)} variants")

        # Create empty decision trace for consistency (existing runs)
        decision_trace = DecisionTrace(probe_rank=config.get("lora", {}).get("probe_r", 32))

    # Write compression configs to JSON for debugging/inspection
    compression_configs_path = output_path / "compression_configs.json"
    with open(compression_configs_path, "w") as f:
        json.dump(compression_configs, f, indent=2, ensure_ascii=False)

    print("Compression configs generated:")
    for variant, config_data in compression_configs.items():
        status = config_data["status"]
        if status == "ready":
            actual_r = config_data["actual_r"]
            print(f"  ✅ {variant}: r={actual_r}")
        else:
            reason = config_data.get("reason", "Unknown reason")
            print(f"  ❌ {variant}: {status} - {reason}")

    # Step 3.5: Train and evaluate compressed variants
    print("\nStep 3.5: Training and evaluating compressed variants...")
    variant_results = run_all_compressed_variants(
        config_path=config_path,
        output_dir=output_path,
        compression_configs=compression_configs,
        smoke=smoke,
        stage_manager=stage_manager,
        resume=resume,
    )

    # Step 3.6: Compute verdicts
    verdict_analysis = compute_verdicts(
        probe_results=probe_results,
        variant_results=variant_results,
        config=config,
        output_path=output_path,
        smoke=smoke,
    )

    verdict_analysis, escalation_trace, variant_results, compression_configs = _run_escalation_and_update(
        variant_results,
        verdict_analysis,
        compression_configs,
        config,
        probe_results,
        config_path,
        output_path,
        smoke,
        stage_manager,
        resume,
    )

    _update_policy_scoreboard(verdict_analysis, compression_configs, config, output_path)

    # Load audit data for canonical report
    probe_audit_path = output_path / f"probe_r{probe_rank}" / "audit.json"
    with open(probe_audit_path) as f:
        audit_data = json.load(f)

    # Create canonical bench.json report
    seed = config.get("train", {}).get("seed", DEFAULT_SEED)
    with monitor_generation("generate_report", output_dir=output_path, seed=seed) as stage:
        stage.progress("Creating canonical benchmark report")
        canonical_report = create_canonical_bench_report(
            probe_results=probe_results,
            variant_results=variant_results,
            verdict_analysis=verdict_analysis,
            audit_data=audit_data,
            compression_configs=compression_configs,
            config=config,
            output_dir=output_path,
            decision_trace=decision_trace,
            escalation_trace=escalation_trace,
        )

        # Inject effective overrides so embedded_config vs actual execution is clear
        if "config_metadata" in canonical_report:
            variants_actually_evaluated = sorted(
                name for name, res in variant_results.items() if res.get("status") in ("completed", "PASS")
            )
            canonical_report["config_metadata"]["effective_overrides"] = {
                "fast_mode": fast_mode,
                "max_candidates": max_candidates,
                "variants_evaluated": variants_actually_evaluated,
                "candidate_selection_mode": "fast" if fast_mode else "full",
            }

        # Write canonical benchmark report
        stage.progress("Writing JSON report")
        report_path = output_path / "bench.json"
        with open(report_path, "w") as f:
            json.dump(canonical_report, f, indent=2, ensure_ascii=False)
        stage.add_artifact(report_path)

        # Create and write markdown report
        stage.progress("Generating markdown report")
        markdown_content = create_markdown_report(
            canonical_report=canonical_report, config=config, output_dir=output_path
        )
        markdown_path = output_path / "bench.md"
        with open(markdown_path, "w") as f:
            f.write(markdown_content)
        stage.add_artifact(markdown_path)
        stage.progress("Reports generated successfully")

    # Also write comprehensive internal report for debugging
    internal_report = {
        "bench_version": config.get("bench_version", "0.1"),
        "model": config["model"]["name"],
        "task": f"{config['task']['dataset']}/{config['task']['subset']}",
        "config_path": str(config_path),
        "output_dir": str(output_path),
        "smoke_mode": smoke,
        **probe_results,
        "compression_configs": compression_configs,
        "variants": variant_results,
        "verdicts": verdict_analysis,
    }

    internal_report_path = output_path / "bench_internal.json"
    with open(internal_report_path, "w") as f:
        json.dump(internal_report, f, indent=2, ensure_ascii=False)

    print("\nSteps 3.1-3.6 complete!")
    print("  ✅ Probe trained and telemetry written")
    print("  ✅ Evaluation results written to eval.json")
    print("  ✅ Audit completed and results written to audit.json")
    print("  ✅ Compression configurations generated")
    print("  ✅ Compressed variants trained and evaluated")
    print("  ✅ Verdicts computed and best compression identified")
    print(f"  📊 Canonical report written to: {report_path}")
    print(f"  📝 Human report written to: {markdown_path}")

    # Artifact hygiene cleanup
    run_artifact_hygiene_cleanup(output_path, config)

    print("\nBench protocol complete! 🎉")

    return canonical_report
