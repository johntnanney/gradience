"""Probe adapter training and auditing for bench protocol."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Check for optional dependencies
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False

# Gradience imports (always available)
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig
from gradience.vnext.audit.lora_audit import audit_lora_peft_dir
from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit, suggest_per_layer_ranks
from gradience.bench.task_profiles import get_task_profile_from_config
from gradience.bench.monitored_stage import (
    monitor_training, monitor_evaluation, monitor_audit,
    monitor_file_operations,
)
from gradience.bench.model_setup import (
    load_config, setup_dataset, setup_model_and_tokenizer,
    _save_peft_adapter_only, HAS_TRAINING_DEPS,
)
from gradience.bench.reporting import write_probe_eval_json, _extract_accuracy_with_fallback


def run_probe_audit(
    probe_dir: Path,
    config: Dict[str, Any]
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
        compute_udr=compute_udr
    )

    # Convert audit result to dict for JSON serialization
    audit_summary = audit_result.to_summary_dict()

    # Add the probe rank to the audit summary for rank suggestion
    probe_rank = config["lora"]["probe_r"]
    audit_summary["current_r"] = probe_rank

    # Validate LoRA attachment - prevent wasted GPU cycles
    stable_rank_mean = audit_summary.get('stable_rank_mean', 0.0)
    utilization_mean = audit_summary.get('utilization_mean', 0.0)

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
            "message": "LoRA adapters likely did not train or attach properly"
        }
    else:
        audit_summary["probe_validity"] = {
            "valid": True,
            "reason": "NORMAL_OPERATION",
            "stable_rank_mean": stable_rank_mean,
            "utilization_mean": utilization_mean
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
            allowed_ranks=config.get("compression", {}).get("allowed_ranks", [1, 2, 4, 8, 16, 32])
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
        "summary": {
            **audit_summary,
            "gain": gain_metrics["summary"]
        },

        # Global rank suggestions (required) - using audit summary values
        "suggested_r_global_median": audit_summary.get("suggested_r_global_median"),
        "suggested_r_global_90": audit_summary.get("suggested_r_global_90"),

        # Policy-based global suggestions (Step 7)
        "policy_global_suggestions": (
            getattr(audit_result, 'policy_global_suggestions', {})
            if isinstance(getattr(audit_result, 'policy_global_suggestions', None), dict)
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
            "evidence": global_suggestions.evidence
        },

        # Per-module gain metrics
        "per_module": {
            "gain": gain_metrics["per_module"]
        },

        # Per-layer gain metrics
        "per_layer": {
            "gain": gain_metrics["per_layer"]
        },

        # Global gain metrics
        "global": {
            "gain": gain_metrics["global"]
        },

        # Composition analysis (energy concentration across layers) - optional
        **({"composition": gain_metrics.get("composition", {})} if enable_composition else {}),

        # Per-layer analysis (your 1.3/1.4 work)
        "layers": [layer.to_dict() for layer in audit_result.layers],

        # Per-layer suggestions if available
        "per_layer_suggestions": per_layer_suggestions.to_dict() if per_layer_suggestions else None,

        # Issues encountered during audit
        "issues": audit_result.issues
    }

    audit_path = probe_dir / "audit.json"
    with open(audit_path, 'w') as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)

    return audit_path


def run_probe_training(
    config_path: str | Path,
    output_dir: str | Path,
    smoke: bool = False,
    stage_manager = None,
    resume: bool = False
) -> Dict[str, Any]:
    """
    Step 3.1: Train probe adapter (r=16).

    Returns training results including accuracy and parameter counts.
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError(
            "Training dependencies not available. "
            "Install: pip install transformers>=4.20.0 peft>=0.4.0 datasets torch"
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
    callback_config = GradienceCallbackConfig(
        output_dir=str(probe_dir),
        filename="run.jsonl"
    )

    # Add optional dataset/task context for richer telemetry
    task_config = config.get("task", {})
    if task_config.get("dataset") and task_config.get("subset"):
        dataset_name = f"{task_config['dataset']}/{task_config['subset']}"
        # Note: callback doesn't require these fields, but bench can provide them
        # for richer downstream monitor output
        # We'll pass them via environment or config if the callback supports it in future

    gradience_callback = GradienceCallback(callback_config)

    # Build trainer using task profile
    trainer = task_profile.build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_ds=tokenized_dataset,
        cfg=config,
        callbacks=[gradience_callback]
    )

    # Update trainer output dir to probe directory
    trainer.args.output_dir = str(probe_dir)
    trainer.args.logging_dir = str(probe_dir / "logs")

    # Train the model
    # Check if training can be skipped
    probe_rank = config['lora']['probe_r']
    skip_training = resume and stage_manager and stage_manager.should_skip_probe_training(probe_rank)

    if not skip_training:
        print(f"Starting probe training (r={probe_rank})...")
        print(f"Output dir: {probe_dir}")
        print(f"Max steps: {trainer.args.max_steps}")
        print(f"Device: {device}")

        seed = config.get("train", {}).get("seed", 42)
        with monitor_training(f"train_probe_r{probe_rank}", output_dir=probe_dir, seed=seed) as stage:
            stage.progress("Starting probe training")
            trainer.train()
            stage.progress("Probe training completed")

        # Mark training as completed
        if stage_manager:
            stage_manager.mark_stage_completed(f"probe_r{probe_rank}_trained", {
                "probe_rank": probe_rank,
                "max_steps": trainer.args.max_steps,
                "output_dir": str(probe_dir)
            })
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
        seed = config.get("train", {}).get("seed", 42)
        with monitor_evaluation("eval_probe", output_dir=probe_dir, seed=seed) as stage:
            stage.progress("Starting probe evaluation")
            eval_results = task_profile.evaluate(model, tokenizer, tokenized_dataset, config)
            stage.progress("Probe evaluation completed")
            stage.add_artifact("eval.json")

        # Mark evaluation as completed
        if stage_manager:
            stage_manager.mark_stage_completed(f"probe_r{probe_rank}_evaluated", {
                "probe_rank": probe_rank,
                "accuracy": eval_results.get("eval_accuracy", 0.0)
            })
    else:
        # Load existing evaluation results
        eval_json_path = probe_dir / "eval.json"
        with open(eval_json_path) as f:
            eval_results = json.load(f)
        print(f"Loaded existing evaluation results: accuracy = {eval_results.get('eval_accuracy', 'unknown')}")

    # Step 3.2: Write eval.json
    eval_dataset_size = eval_results.get("eval_samples", len(tokenized_dataset.get("validation", tokenized_dataset["train"])))
    eval_json_path = write_probe_eval_json(
        probe_dir=probe_dir,
        eval_results=eval_results,
        eval_dataset_size=eval_dataset_size,
        config=config
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
        seed = config.get("train", {}).get("seed", 42)
        with monitor_audit("audit_probe", output_dir=probe_dir, seed=seed) as stage:
            stage.progress("Starting probe audit analysis")
            audit_json_path = run_probe_audit(
                probe_dir=probe_dir,
                config=config
            )
            stage.progress("Probe audit analysis completed")
            stage.add_artifact("audit.json")

        # Mark audit as completed
        if stage_manager:
            stage_manager.mark_stage_completed(f"probe_r{probe_rank}_audited", {
                "probe_rank": probe_rank,
                "audit_path": str(audit_json_path)
            })
    else:
        # Audit already exists
        audit_json_path = probe_dir / "audit.json"
        print(f"Using existing audit results: {audit_json_path}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Probe training complete!")

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
            "output_dir": str(probe_dir)
        }
    }
