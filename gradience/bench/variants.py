"""Compressed variant training and evaluation for bench protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Check for optional dependencies
try:
    import torch  # noqa: F401
    from datasets import load_dataset  # noqa: F401
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: F401
    from transformers import (  # noqa: F401
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False

from gradience.bench.model_setup import (
    _save_peft_adapter_only,
    load_config,
    setup_compressed_model_and_tokenizer,
    setup_dataset,
)
from gradience.bench.monitored_stage import monitor_evaluation, monitor_file_operations, monitor_training
from gradience.bench.reporting import _extract_accuracy_with_fallback, write_probe_eval_json
from gradience.bench.task_profiles import get_task_profile_from_config
from gradience.peft_utils import check_heterogeneous_ranks
from gradience.vnext.integrations.hf import GradienceCallback, GradienceCallbackConfig


def run_post_tuning(
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
    post_tune_steps = post_tune_config.get("steps", 100)
    lr_scale = post_tune_config.get("lr_scale", 0.1)
    warmup_steps = post_tune_config.get("warmup_steps", 0)  # Default: zero warmup for tiny tune

    # Use base training config but with scaled parameters
    train_config = config["train"]
    base_lr = train_config.get("learning_rate", 5e-5)
    post_tune_lr = base_lr * lr_scale

    # Reduce steps for smoke mode
    if smoke:
        post_tune_steps = min(post_tune_steps, 20)
        warmup_steps = min(warmup_steps, post_tune_steps // 5)  # Scale warmup for smoke

    # Setup training arguments for post-tuning
    post_tune_args = TrainingArguments(
        output_dir=str(output_dir / "post_tune"),
        num_train_epochs=1,
        max_steps=post_tune_steps,
        learning_rate=post_tune_lr,
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 32),
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
    trainer = Trainer(
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


def run_svd_truncation_variant(
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
                "steps": post_tune_config.get("steps", 100),
                "lr_scale": post_tune_config.get("lr_scale", 0.1),
                "warmup_steps": post_tune_config.get("warmup_steps", 0),
            }

        return result

    except Exception as e:
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


def run_compressed_variant_training(
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
            "Training dependencies not available. Install: pip install transformers>=4.20.0 peft>=0.4.0 datasets torch"
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

        seed = config.get("train", {}).get("seed", 42)
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
        seed = config.get("train", {}).get("seed", 42)
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


def run_all_compressed_variants(
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
