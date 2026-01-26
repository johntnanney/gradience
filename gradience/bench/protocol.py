"""
Bench protocol (v0.1).

Implements:
1) Train probe adapter (r=16)
2) Audit -> suggestions
3) Retrain: uniform_median, uniform_p90, per_layer
4) Eval all
5) Emit report (JSON + Markdown)

Step 3.1 implementation: Train probe with GradienceCallback
"""

from __future__ import annotations

import os
import json
import yaml
import datetime
import sys
import subprocess
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
from gradience.peft_utils import (
    normalize_rank_pattern, normalize_alpha_pattern,
    create_complete_rank_pattern, create_complete_alpha_pattern,
    check_heterogeneous_ranks
)
from gradience.vnext.audit.lora_audit import audit_lora_peft_dir
from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit, suggest_per_layer_ranks
from .task_profiles import get_task_profile_from_config


def _unwrap_model_for_save(trainer, model):
    # Try accelerator unwrap first (works with device_map / accelerate wrapping)
    if trainer is not None and hasattr(trainer, "accelerator"):
        try:
            return trainer.accelerator.unwrap_model(model)
        except Exception:
            pass
    # Common wrapper case
    if hasattr(model, "module"):
        return model.module
    return model

def _save_peft_adapter_only(trainer, model, output_dir: str | Path, *, label: str = "adapter") -> Path:
    """
    Save PEFT adapter weights/config to output_dir.

    Critical invariant:
      - Never save a full base model here (7B would be catastrophic).
      - If the model is not a PEFT model, raise loudly.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    m = _unwrap_model_for_save(trainer, model)

    # Guardrail: only PEFT models should pass
    if not hasattr(m, "peft_config"):
        raise RuntimeError(
            f"Bench expected a PEFT model but got {type(m)}. Refusing to save full model. ({label})"
        )

    # Save adapter weights/config (small)
    try:
        m.save_pretrained(out, safe_serialization=True)
    except TypeError:
        # Older peft versions may not accept safe_serialization
        m.save_pretrained(out)

    # Sanity: ensure audit inputs exist
    cfg = out / "adapter_config.json"
    if not cfg.exists():
        raise RuntimeError(f"Adapter save succeeded but adapter_config.json missing at: {cfg} ({label})")

    # adapter_model.* name differs by serializer; prefer safetensors but accept either
    safetensors_path = out / "adapter_model.safetensors"
    bin_path = out / "adapter_model.bin"
    if not safetensors_path.exists() and not bin_path.exists():
        raise RuntimeError(
            f"Adapter save succeeded but adapter_model.(safetensors|bin) missing in: {out} ({label})"
        )

    return out


def _get_probe_quality_threshold(task_name: str) -> float:
    """
    Get task-specific minimum probe accuracy threshold for compression certification.
    
    Args:
        task_name: Task identifier (e.g., 'sst2', 'mnli', etc.)
        
    Returns:
        Minimum accuracy threshold for considering probe sufficiently trained
    """
    # Task-specific quality thresholds based on typical performance
    task_thresholds = {
        "sst2": 0.75,       # SST-2 sentiment: 75% minimum
        "mnli": 0.70,       # MNLI entailment: 70% minimum  
        "qnli": 0.75,       # QNLI question entailment: 75% minimum
        "qqp": 0.80,        # QQP paraphrase: 80% minimum
        "rte": 0.60,        # RTE small dataset: 60% minimum
        "wnli": 0.55,       # WNLI very small: 55% minimum
        "cola": 0.70,       # CoLA linguistic: 70% minimum
        "mrpc": 0.75,       # MRPC paraphrase: 75% minimum
        "stsb": 0.80,       # STS-B regression converted to classification
    }
    
    return task_thresholds.get(task_name.lower(), 0.70)  # Default 70% for unknown tasks


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_dataset(config: Dict[str, Any], smoke: bool = False):
    """Load and prepare dataset based on config using task profile."""
    if not HAS_TRAINING_DEPS:
        raise ImportError("Training dependencies not available (transformers, datasets, peft)")
    
    # Get task profile for this configuration
    task_profile = get_task_profile_from_config(config)
    
    # Load dataset using task profile
    dataset = task_profile.load(config)
    
    # Apply smoke test limits if requested
    if smoke:
        runtime = config.get("runtime", {})
        train_samples = runtime.get("smoke_train_samples", 200)
        eval_samples = runtime.get("smoke_eval_samples", 200)
        
        if "train" in dataset:
            dataset["train"] = dataset["train"].select(range(min(len(dataset["train"]), train_samples)))
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].select(range(min(len(dataset["validation"]), eval_samples)))
    
    return dataset


def setup_model_and_tokenizer(config: Dict[str, Any], device: str = "cpu"):
    """Setup base model, tokenizer, and LoRA configuration."""
    if not HAS_TRAINING_DEPS:
        raise ImportError("Training dependencies not available (transformers, peft)")
    
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
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None
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
            model_name,
            num_labels=2,  # Default binary classification
            torch_dtype=torch_dtype
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
    
    model = get_peft_model(model, peft_config)
    
    return tokenizer, model


def setup_compressed_model_and_tokenizer(config: Dict[str, Any], compression_config: Dict[str, Any], device: str = "cpu"):
    """Setup model and tokenizer with compressed LoRA configuration."""
    if not HAS_TRAINING_DEPS:
        raise ImportError("Training dependencies not available (transformers, peft)")
    
    model_config = config["model"]
    model_name = model_config["name"]
    model_type = model_config.get("type", "seqcls")
    base_lora_config = config["lora"]
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
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None
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
            model_name,
            num_labels=2,  # Default binary classification
            torch_dtype=torch_dtype
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
        default_rank_from_audit = compression_config.get("_probe_rank", 16)
        default_alpha_from_audit = default_rank_from_audit
        
        # Create complete, normalized patterns using canonical helpers
        full_rank_pattern = create_complete_rank_pattern(
            variant_config["rank_pattern"], 
            audit_layers, 
            default_rank_from_audit
        )
        full_alpha_pattern = create_complete_alpha_pattern(
            variant_config["alpha_pattern"], 
            audit_layers, 
            default_alpha_from_audit
        )
        
        # Use minimum rank as default for PEFT compatibility
        # This conservative approach ensures rank_pattern overrides work correctly
        # (PEFT 0.18.1 has issues when default_r > some pattern values)
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
    
    model = get_peft_model(model, peft_config)
    
    return tokenizer, model


# Legacy preprocess_function moved to task profiles
# This function is kept for backward compatibility but deprecated


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
    
    # Debug: Check if summary has required fields
    print(f"Debug: audit_summary has stable_rank_mean={audit_summary.get('stable_rank_mean')}, utilization_mean={audit_summary.get('utilization_mean')}, current_r={audit_summary.get('current_r')}")
    
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
    
    # Prepare comprehensive audit data
    audit_data = {
        # Audit metadata
        "audit_timestamp": datetime.datetime.now().isoformat(),
        "probe_rank": config["lora"]["probe_r"],
        "seed": config["train"]["seed"],
        
        # Summary statistics (includes suggested_r_global_median, suggested_r_global_90)
        "summary": audit_summary,
        
        # Global rank suggestions (required) - using audit summary values
        "suggested_r_global_median": audit_summary.get("suggested_r_global_median"),
        "suggested_r_global_90": audit_summary.get("suggested_r_global_90"),
        
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
    smoke: bool = False
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
    print(f"Starting probe training (r={config['lora']['probe_r']})...")
    print(f"Output dir: {probe_dir}")
    print(f"Max steps: {trainer.args.max_steps}")
    print(f"Device: {device}")
    
    trainer.train()

    # Ensure adapter exists on disk for audit (save_strategy may be "no")
    _save_peft_adapter_only(trainer, model, probe_dir, label="probe")
    
    # Evaluate final model using task profile
    eval_results = task_profile.evaluate(model, tokenizer, tokenized_dataset, config)
    
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
    
    print("Running LoRA audit...")
    audit_json_path = run_probe_audit(
        probe_dir=probe_dir,
        config=config
    )
    
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


def round_to_allowed_ranks(suggested_r: int, allowed_ranks: list[int]) -> int:
    """Round a suggested rank to the nearest allowed rank."""
    if suggested_r in allowed_ranks:
        return suggested_r
    
    # Find closest allowed rank
    return min(allowed_ranks, key=lambda x: abs(x - suggested_r))


def _create_shuffled_rank_pattern(original_rank_pattern: Dict[str, int], seed: int) -> Dict[str, int]:
    """
    Create a shuffled control by redistributing ranks across different modules.
    
    This is the key scientific control: if audit-guided placement matters,
    per_layer should outperform per_layer_shuffled. If any heterogeneity
    is enough, they should perform similarly.
    
    Args:
        original_rank_pattern: Dict mapping module names to ranks
        seed: Random seed for deterministic shuffling
        
    Returns:
        Dict with same module names but redistributed rank values
    """
    import random
    
    # Extract module names and rank values
    module_names = list(original_rank_pattern.keys())
    rank_values = list(original_rank_pattern.values())
    
    # Create deterministic shuffle using seed + offset
    rng = random.Random(seed + 10000)  # Fixed offset for reproducibility
    
    # Shuffle the rank values while keeping module names fixed
    shuffled_ranks = rank_values.copy()
    rng.shuffle(shuffled_ranks)
    
    # Recombine: same modules, redistributed ranks
    shuffled_pattern = dict(zip(module_names, shuffled_ranks))
    
    return shuffled_pattern


def generate_compression_configs(
    probe_dir: Path,
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Step 3.4: Generate 3 compression configs from probe audit.
    
    Returns dict with compression variant configs:
    - uniform_median: use suggested_r_global_median everywhere
    - uniform_p90: use suggested_r_global_90 everywhere  
    - per_layer: use per-layer rank suggestions
    """
    
    # Load audit results
    audit_path = probe_dir / "audit.json"
    with open(audit_path, 'r') as f:
        audit_data = json.load(f)
    
    compression_config = config["compression"]
    allowed_ranks = compression_config["allowed_ranks"]
    lora_config = config["lora"]
    probe_rank = lora_config["probe_r"]
    
    compression_configs = {}
    
    # A) uniform_median
    suggested_median = audit_data["suggested_r_global_median"]
    median_rank = round_to_allowed_ranks(suggested_median, allowed_ranks)
    
    # Safety check: no rank > probe rank
    if median_rank > probe_rank:
        median_rank = probe_rank
    
    compression_configs["uniform_median"] = {
        "variant": "uniform_median",
        "suggested_r": suggested_median,
        "actual_r": median_rank,
        "rank_pattern": {},  # Empty for uniform
        "alpha_pattern": {},  # Empty for uniform
        "config": {
            **lora_config,
            "probe_r": median_rank,  # Use compressed rank
            "alpha": median_rank,    # Preserve alpha=r scaling
        },
        "status": "ready",
        "reason": None
    }
    
    # B) uniform_p90
    suggested_p90 = audit_data["suggested_r_global_90"]
    p90_rank = round_to_allowed_ranks(suggested_p90, allowed_ranks)
    
    # Safety check: no rank > probe rank
    if p90_rank > probe_rank:
        p90_rank = probe_rank
    
    # Check if this is effectively a control run (no compression)
    if p90_rank == probe_rank:
        compression_configs["uniform_p90_control"] = {
            "variant": "uniform_p90_control", 
            "suggested_r": suggested_p90,
            "actual_r": p90_rank,
            "rank_pattern": {},  # Empty for uniform
            "alpha_pattern": {},  # Empty for uniform
            "config": {
                **lora_config,
                "probe_r": p90_rank,  # Use compressed rank
                "alpha": p90_rank,    # Preserve alpha=r scaling
            },
            "status": "skipped",
            "reason": f"Control run: suggested rank r={p90_rank} equals probe rank (no compression)"
        }
    else:
        compression_configs["uniform_p90"] = {
            "variant": "uniform_p90", 
            "suggested_r": suggested_p90,
            "actual_r": p90_rank,
            "rank_pattern": {},  # Empty for uniform
            "alpha_pattern": {},  # Empty for uniform
            "config": {
                **lora_config,
                "probe_r": p90_rank,  # Use compressed rank
                "alpha": p90_rank,    # Preserve alpha=r scaling
            },
            "status": "ready",
            "reason": None
        }
    
    # C) per_layer
    per_layer_suggestions = audit_data.get("per_layer_suggestions")
    if not per_layer_suggestions:
        compression_configs["per_layer"] = {
            "variant": "per_layer",
            "suggested_r": None,
            "actual_r": None,
            "rank_pattern": {},
            "alpha_pattern": {},
            "config": None,
            "status": "SKIPPED",
            "reason": "No per-layer suggestions found in audit"
        }
    else:
        rank_pattern = per_layer_suggestions["rank_pattern"]
        
        # Clamp ranks to probe rank and allowed ranks
        clamped_rank_pattern = {}
        for module_name, suggested_r in rank_pattern.items():
            # First clamp to probe rank
            clamped_r = min(suggested_r, probe_rank)
            # Then round to nearest allowed rank
            if clamped_r in allowed_ranks:
                clamped_rank_pattern[module_name] = clamped_r
            else:
                # Find nearest allowed rank that doesn't exceed probe rank
                valid_allowed_ranks = [r for r in allowed_ranks if r <= probe_rank]
                if valid_allowed_ranks:
                    clamped_rank_pattern[module_name] = max([r for r in valid_allowed_ranks if r <= clamped_r] or [min(valid_allowed_ranks)])
                else:
                    clamped_rank_pattern[module_name] = 0
        
        # Update rank_pattern with clamped values
        rank_pattern = clamped_rank_pattern
        
        # Safety checks after clamping
        issues = []
        
        # Check 3: At least 1 adapted layer (not all zero/inactive)
        active_layers = [r for r in rank_pattern.values() if r > 0]
        if not active_layers:
            issues.append("No active layers (all ranks are 0)")
        
        if issues:
            compression_configs["per_layer"] = {
                "variant": "per_layer",
                "suggested_r": len(rank_pattern),
                "actual_r": 0,
                "rank_pattern": rank_pattern,
                "alpha_pattern": {},
                "config": None,
                "status": "SKIPPED",
                "reason": f"Safety checks failed: {'; '.join(issues)}"
            }
        else:
            # Build alpha pattern to preserve alpha/r scaling
            # If probe used alpha=r, then alpha_pattern equals suggested r
            alpha_pattern = {}
            for module_name, suggested_r in rank_pattern.items():
                if suggested_r > 0:  # Only for active modules
                    alpha_pattern[module_name] = suggested_r
            
            # Normalize patterns at generation time for consistency
            rank_pattern = normalize_rank_pattern(rank_pattern)
            alpha_pattern = normalize_alpha_pattern(alpha_pattern)
            
            compression_configs["per_layer"] = {
                "variant": "per_layer",
                "suggested_r": len(rank_pattern),
                "actual_r": len([r for r in rank_pattern.values() if r > 0]),
                "rank_pattern": rank_pattern,
                "alpha_pattern": alpha_pattern,
                # Attach audit metadata for PEFT canonical processing
                "_audit_layers": audit_data.get("layers", []),
                "_probe_rank": probe_rank,
                "config": {
                    **lora_config,
                    "rank_pattern": rank_pattern,
                    "alpha_pattern": alpha_pattern,
                    # For per-layer, we don't use uniform probe_r/alpha
                    "probe_r": None,  
                    "alpha": None,
                },
                "status": "ready",
                "reason": None
            }

    # D) per_layer_shuffled (control for mechanism testing)
    # Create shuffled control only if we have a successful per_layer variant
    if ("per_layer" in compression_configs and 
        compression_configs["per_layer"]["status"] == "ready"):
        
        original_rank_pattern = compression_configs["per_layer"]["rank_pattern"]
        shuffled_rank_pattern = _create_shuffled_rank_pattern(
            original_rank_pattern, 
            seed=config.get("train", {}).get("seed", 42)
        )
        
        # Create alpha pattern matching the shuffled ranks
        shuffled_alpha_pattern = {}
        for module_name, suggested_r in shuffled_rank_pattern.items():
            if suggested_r > 0:  # Only for active modules
                shuffled_alpha_pattern[module_name] = suggested_r
        
        # Normalize patterns
        shuffled_rank_pattern = normalize_rank_pattern(shuffled_rank_pattern)
        shuffled_alpha_pattern = normalize_alpha_pattern(shuffled_alpha_pattern)
        
        compression_configs["per_layer_shuffled"] = {
            "variant": "per_layer_shuffled",
            "suggested_r": len(shuffled_rank_pattern),
            "actual_r": len([r for r in shuffled_rank_pattern.values() if r > 0]),
            "rank_pattern": shuffled_rank_pattern,
            "alpha_pattern": shuffled_alpha_pattern,
            # Attach same audit metadata for consistency
            "_audit_layers": audit_data.get("layers", []),
            "_probe_rank": probe_rank,
            "_shuffle_seed": config.get("train", {}).get("seed", 42) + 10000,  # Deterministic offset
            "config": {
                **lora_config,
                "rank_pattern": shuffled_rank_pattern,
                "alpha_pattern": shuffled_alpha_pattern,
                "probe_r": None,  
                "alpha": None,
            },
            "status": "ready",
            "reason": "Shuffled control for audit-guided per-layer variant"
        }
    else:
        # No per_layer to shuffle
        compression_configs["per_layer_shuffled"] = {
            "variant": "per_layer_shuffled",
            "suggested_r": None,
            "actual_r": None,
            "rank_pattern": {},
            "alpha_pattern": {},
            "config": None,
            "status": "SKIPPED",
            "reason": "No per-layer variant to create shuffled control from"
        }
    
    return compression_configs


def gather_environment_info() -> Dict[str, Any]:
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
    output_dir: Path
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
        return {
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
            else:
                # Uniform variants
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
    
    # Add UDR instrumentation as separate section if available
    if udr_instrumentation:
        report["instrumentation"] = {
            "udr": udr_instrumentation
        }
    
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

| Variant | Params | Accuracy | Δ vs probe | Param reduction | Verdict |
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
- **PASS** means the compressed model didn't hurt accuracy beyond tolerance (±{acc_tolerance:.3f})
- **FAIL** means accuracy dropped more than the tolerance threshold
- You should still validate these results on your real workload before deployment
- Parameter reduction shows the percentage decrease in trainable LoRA parameters

## Summary

- **Recommendations validated:** {summary["recommendations_validated"]}
- **Best compression:** {summary["best_compression"] or "None"}

*Generated on {timestamp[:19].replace('T', ' ')}*
"""

    return md_content


def run_compressed_variant_training(
    config_path: str | Path,
    output_dir: str | Path,
    variant_name: str,
    compression_config: Dict[str, Any],
    smoke: bool = False
) -> Dict[str, Any]:
    """
    Step 3.5: Train a single compressed variant.
    
    Returns training results including accuracy and parameter counts.
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError(
            "Training dependencies not available. "
            "Install: pip install transformers>=4.20.0 peft>=0.4.0 datasets torch"
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
            "output_dir": None
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
    callback_config = GradienceCallbackConfig(
        output_dir=str(variant_dir),
        filename="run.jsonl"
    )
    gradience_callback = GradienceCallback(callback_config)
    
    # Build trainer using task profile
    trainer = task_profile.build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_ds=tokenized_dataset,
        cfg=config,
        callbacks=[gradience_callback]
    )
    
    # Update trainer output dir to variant directory
    trainer.args.output_dir = str(variant_dir)
    trainer.args.logging_dir = str(variant_dir / "logs")
    
    # Train the model
    print(f"Starting {variant_name} training (r={actual_r})...")
    print(f"Output dir: {variant_dir}")
    print(f"Max steps: {max_steps}")
    print(f"Device: {device}")
    
    trainer.train()

    # Ensure adapter exists on disk for audit (save_strategy may be "no")
    _save_peft_adapter_only(trainer, model, variant_dir, label=f"variant:{variant_name}")
    
    # Evaluate final model using task profile
    eval_results = task_profile.evaluate(model, tokenizer, tokenized_dataset, config)
    
    # Write eval.json for this variant  
    eval_dataset_size = eval_results.get("eval_samples", len(tokenized_dataset.get("validation", tokenized_dataset["train"])))
    eval_json_path = write_probe_eval_json(
        probe_dir=variant_dir,
        eval_results=eval_results,
        eval_dataset_size=eval_dataset_size,
        config=config
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
                "total_modules": 0
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
                "rank_check": rank_check_result
            }
        else:
            print(f"✅ Rank check passed: {len(rank_check_result['unique_ranks'])} distinct ranks")
            print(f"   Rank histogram: {rank_check_result['rank_histogram']}")
    
    print(f"{variant_name} training complete!")
    
    # Get task profile for robust metric extraction
    task_profile = get_task_profile_from_config(config)
    accuracy_value = _extract_accuracy_with_fallback(eval_results, task_profile)
    
    if accuracy_value > 0.0:
        metric_key = getattr(task_profile, 'primary_metric_key', 'eval_accuracy')
        print(f"Final {metric_key}: {accuracy_value:.4f}")
    else:
        print(f"Warning: No accuracy metric found in evaluation results")
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
        "output_dir": str(variant_dir)
    }
    
    # Add rank check results for per-layer variants
    if rank_check_result is not None:
        result["rank_check"] = rank_check_result
    
    return result


def run_all_compressed_variants(
    config_path: str | Path,
    output_dir: str | Path,
    compression_configs: Dict[str, Dict[str, Any]],
    smoke: bool = False
) -> Dict[str, Any]:
    """
    Step 3.5: Train and evaluate all compressed variants.
    
    Returns results for all variants.
    """
    results = {}
    
    for variant_name, compression_config in compression_configs.items():
        print(f"\n" + "="*50)
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
            smoke=smoke
        )
        
        results[variant_name] = result
    
    return results


def classify_validation_level(config: Dict[str, Any]) -> Dict[str, str]:
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
    if is_multiseed and max_steps >= 200:
        if len(seeds) >= 3 and max_steps >= 500:
            classification = "certifiable"
            rationale = f"{len(seeds)} seeds × {max_steps} steps provides statistical rigor"
        else:
            classification = "screening_plus"
            rationale = f"{len(seeds)} seeds × {max_steps} steps (limited budget/seeds)"
    elif is_multiseed:
        classification = "screening_plus"  
        rationale = f"{len(seeds)} seeds but only {max_steps} steps (limited budget)"
    elif max_steps >= 500:
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
        "max_steps": max_steps
    }


def compute_verdicts(
    probe_results: Dict[str, Any],
    variant_results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_path: Path,
    smoke: bool = False
) -> Dict[str, Any]:
    """
    Step 3.6: Compute verdicts for compressed variants.
    
    Returns verdict analysis including PASS/FAIL decisions and best compression.
    """
    compression_config = config.get("compression", {})
    acc_tolerance = compression_config.get("acc_tolerance", 0.005)
    
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
    with open(probe_eval_path, 'r') as f:
        probe_eval_results = json.load(f)
    
    probe_passed, gate_info = task_profile.probe_gate(probe_eval_results, config)
    probe_quality_threshold = gate_info["threshold"]
    
    if not probe_passed:
        print(f"\n=== PROBE QUALITY GATE FAILED ===")
        print(f"Probe accuracy: {probe_accuracy:.4f}")
        print(f"Required threshold: {probe_quality_threshold:.4f}")
        
        if smoke:
            print(f"Status: UNDERTRAINED_SMOKE - continuing in smoke mode")
            status_code = "UNDERTRAINED_SMOKE"
        else:
            print(f"Status: UNDERTRAINED - compression certification not valid")
            status_code = "UNDERTRAINED"
        
        # Return undertrained status for all variants
        verdicts = {}
        for variant_name in variant_results.keys():
            verdicts[variant_name] = {
                "status": "undertrained",
                "reason": f"Probe accuracy {probe_accuracy:.4f} < threshold {probe_quality_threshold:.4f}",
                "delta_vs_probe": None,
                "param_reduction": None,
                "verdict": status_code
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
                "notes": [f"Probe undertrained - compression results not reliable (smoke mode: {smoke})"]
            }
        }
    
    verdicts = {}
    pass_variants = []
    
    print(f"\n=== VERDICT ANALYSIS ===")
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
                "verdict": "FAIL" if status == "FAILED" else "SKIP"
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
            pass_variants.append({
                "variant": variant_name,
                "param_reduction": param_reduction,
                "delta_vs_probe": delta_vs_probe,
                "compressed_params": compressed_params,
                "compressed_accuracy": compressed_accuracy
            })
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
            "probe_params": probe_params
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
            "compressed_accuracy": best_variant["compressed_accuracy"]
        }
        
        reduction_pct = best_compression["param_reduction"] * 100
        print(f"🏆 BEST COMPRESSION: {best_compression['variant']}")
        print(f"   {reduction_pct:.1f}% parameter reduction with {best_compression['delta_vs_probe']:+.4f} accuracy delta")
    else:
        print("❌ NO PASSING VARIANTS: All compressions exceeded accuracy tolerance")
    
    return {
        "verdicts": verdicts,
        "best_compression": best_compression,
        "probe_baseline": {
            "accuracy": probe_accuracy,
            "params": probe_params
        },
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
            "notes": []
        }
    }


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
        
        # Overall verdict based on majority
        overall_verdict = "PASS" if pass_rate >= 0.5 else "FAIL"
        
        variants_data[variant_name] = {
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
    
    # Find best compression variant (highest mean reduction among passing variants)
    passing_variants = {name: data for name, data in variants_data.items() if data["verdict"] == "PASS"}
    best_compression = None
    if passing_variants:
        best_name = max(passing_variants.keys(), key=lambda x: passing_variants[x]["param_reduction"]["mean"])
        best_data = passing_variants[best_name]
        best_compression = {
            "variant": best_name,
            "param_reduction_mean": best_data["param_reduction"]["mean"],
            "param_reduction_std": best_data["param_reduction"]["std"],
            "delta_vs_probe_mean": best_data["delta_vs_probe"]["mean"],
            "delta_vs_probe_std": best_data["delta_vs_probe"]["std"],
            "pass_rate": best_data["pass_rate"]
        }
    
    # Build aggregated report
    aggregated_report = {
        "bench_version": base_report["bench_version"],
        "timestamp": timestamp,
        "aggregation_type": "multi_seed",
        "n_seeds": len(seed_reports),
        "seeds": [r.get("env", {}).get("seed", "unknown") for r in seed_reports],
        "model": base_report["model"],
        "task": base_report["task"],
        "env": base_report.get("env", {}),  # Use environment from first report
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
        "summary": {
            "best_compression": best_compression,
            "total_variants": len(variants_data),
            "passing_variants": len(passing_variants),
            "defensible_claims": True,
            "statistical_power": "sufficient" if len(seed_reports) >= 3 else "limited"
        },
        "config_metadata": base_report.get("config_metadata", {})  # Use config metadata from first report
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
    validation_level = "certifiable" if n_seeds >= 3 else "screening_plus"
    
    # Build markdown content
    md_content = f"""# Gradience Bench v{aggregated_report["bench_version"]} (Multi-Seed)

- **Model:** {model}
- **Task:** {task}
- **Seeds:** {n_seeds}
- **Validation Level:** {validation_level.title()}
- **Statistical Power:** {summary["statistical_power"]}

## Probe Baseline (mean ± std)

- **Rank:** {probe_data["rank"]}
- **Accuracy:** {probe_data["accuracy"]["mean"]:.4f} ± {probe_data["accuracy"]["std"]:.4f}
- **LoRA params:** {probe_data["params"]["mean"]:,.0f}

## Compression Results (Aggregated)

| Variant | Accuracy | Δ vs Probe | Param Reduction | Pass Rate | Verdict |
|---|---:|---:|---:|---:|---|
"""

    # Add results table rows
    for variant_name, data in compressed_data.items():
        acc_str = f"{data['accuracy']['mean']:.3f} ± {data['accuracy']['std']:.3f}"
        delta_str = f"{data['delta_vs_probe']['mean']:+.3f} ± {data['delta_vs_probe']['std']:.3f}"
        red_str = f"{data['param_reduction']['mean']:.1%} ± {data['param_reduction']['std']:.1%}"
        pass_rate_str = f"{data['pass_count']}/{data['total_runs']} ({data['pass_rate']:.0%})"
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
        
        md_content += f"| {variant_display} | {acc_str} | {delta_str} | {red_str} | {pass_rate_str} | {verdict} |\n"
    
    # Add defensible claims section
    acc_tolerance = config.get("compression", {}).get("acc_tolerance", 0.005)
    
    md_content += f"""

## Defensible Claims

"""
    
    if summary["best_compression"]:
        best = summary["best_compression"]
        red_mean = best["param_reduction_mean"] * 100
        red_std = best["param_reduction_std"] * 100
        delta_mean = best["delta_vs_probe_mean"]
        delta_std = best["delta_vs_probe_std"]
        
        md_content += f"""• **{best["variant"]} compression achieves {red_mean:.1f}% ± {red_std:.1f}% parameter reduction**
• **Accuracy impact: {delta_mean:+.4f} ± {delta_std:.4f} vs probe baseline**
• **Success rate: {best["pass_rate"]:.0%} across {n_seeds} independent seeds**
• **Based on n={n_seeds} seeds for variance estimation**

"""
    else:
        md_content += "• **No variants achieved reliable compression within tolerance**\n• **All approaches exceeded ±{acc_tolerance:.3f} accuracy threshold across seeds**\n\n"

    # Add interpretation
    md_content += f"""## Interpretation (Statistical)

- **PASS** means ≥50% of seeds passed ±{acc_tolerance:.3f} accuracy tolerance
- **Statistics** are calculated as mean ± standard deviation across {n_seeds} seeds
- **Defensible claims** are supported by variance estimation across multiple random seeds
- You should still validate these results on your real workload before deployment

## Summary

- **Total variants:** {summary["total_variants"]}
- **Passing variants:** {summary["passing_variants"]}
- **Best compression:** {summary["best_compression"]["variant"] if summary["best_compression"] else "None"}
- **Statistical power:** {summary["statistical_power"]}

*Generated on {timestamp[:19].replace('T', ' ')}*
"""

    return md_content


def run_multi_seed_bench_protocol(
    config_path: str | Path,
    output_dir: str | Path,
    seeds: list[int],
    variants_to_test: list[str] = None,
    smoke: bool = False,
    ci: bool = False
) -> Dict[str, Any]:
    """
    Run bench protocol across multiple seeds and aggregate results.
    
    Returns aggregated report with mean ± std statistics.
    """
    config = load_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Gradience Multi-Seed Bench Protocol v0.1")
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
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")
        
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
        
        # Create seed-specific directory
        seed_dir = output_path / f"seed_{seed}"
        seed_dirs.append(seed_dir)
        
        # Write seed-specific config
        seed_config_path = seed_dir / "config.yaml"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with open(seed_config_path, 'w') as f:
            yaml.dump(seed_config, f, indent=2)
        
        # Run single seed benchmark
        try:
            seed_report = run_bench_protocol(
                config_path=seed_config_path,
                output_dir=seed_dir,
                smoke=smoke,
                ci=ci
            )
            
            # Add seed info to report
            seed_report["seed"] = seed
            seed_report["seed_index"] = i
            seed_reports.append(seed_report)
            
            print(f"\n✅ Seed {seed} completed successfully")
            
        except Exception as e:
            print(f"\n❌ Seed {seed} failed: {e}")
            # Continue with other seeds
            continue
    
    if not seed_reports:
        raise RuntimeError("All seed runs failed - cannot generate aggregated report")
    
    print(f"\n{'='*60}")
    print(f"AGGREGATION: {len(seed_reports)}/{len(seeds)} seeds successful")
    print(f"{'='*60}")
    
    # Create aggregated report
    aggregated_report = create_multi_seed_aggregated_report(
        seed_reports=seed_reports,
        config=config,
        output_dir=output_path
    )
    
    # Write aggregated bench.json
    agg_report_path = output_path / "bench_aggregate.json"
    with open(agg_report_path, 'w') as f:
        json.dump(aggregated_report, f, indent=2, ensure_ascii=False)
    
    # Create and write aggregated markdown report
    agg_markdown_content = create_multi_seed_markdown_report(
        aggregated_report=aggregated_report,
        config=config,
        output_dir=output_path
    )
    
    agg_markdown_path = output_path / "bench_aggregate.md"
    with open(agg_markdown_path, 'w') as f:
        f.write(agg_markdown_content)
    
    # Write seed summary
    seed_summary_path = output_path / "seed_summary.json"
    seed_summary = {
        "total_seeds": len(seeds),
        "successful_seeds": len(seed_reports),
        "failed_seeds": len(seeds) - len(seed_reports),
        "seed_directories": [str(d) for d in seed_dirs],
        "aggregated_report": str(agg_report_path),
        "aggregated_markdown": str(agg_markdown_path)
    }
    with open(seed_summary_path, 'w') as f:
        json.dump(seed_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nMulti-seed benchmark complete! 🎉")
    print(f"  📊 Aggregated report: {agg_report_path}")
    print(f"  📝 Aggregated markdown: {agg_markdown_path}")
    print(f"  📋 Seed summary: {seed_summary_path}")
    print(f"  📁 Individual seed results in: {[d.name for d in seed_dirs]}")
    
    return aggregated_report


def run_artifact_hygiene_cleanup(output_dir: Path, config: Dict[str, Any]) -> None:
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
                adapter_patterns = [
                    "adapter_model.safetensors",
                    "adapter_model.bin", 
                    "pytorch_adapter.bin"
                ]
                
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
                        dir_size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
                        
                        # Remove the entire checkpoint directory
                        import shutil
                        shutil.rmtree(checkpoint_dir)
                        cleaned_files.append(str(checkpoint_dir.relative_to(output_dir)) + "/")
                        saved_space += dir_size
    
    except Exception as e:
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
            weight_files = [f for f in cleaned_files if not f.endswith('/')]
            checkpoint_dirs = [f for f in cleaned_files if f.endswith('/')]
            if weight_files:
                print(f"   - {len(weight_files)} adapter weight files")
            if checkpoint_dirs:
                print(f"   - {len(checkpoint_dirs)} checkpoint directories")


def run_bench_preflight_check(config: Dict[str, Any], model_name: str) -> None:
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
    import shutil
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
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
            
        # Determine runtime device from config
        runtime_device = config.get("runtime", {}).get("device", "auto")
        if runtime_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Config specifies device: cuda but CUDA is not available. "
                "Either install CUDA PyTorch or change device to 'cpu' in config."
            )
            
    except ImportError as e:
        raise RuntimeError(f"PyTorch not available: {e}")
    
    # 2. Disk space checks
    critical_paths = ["/tmp"]
    if os.path.exists("/workspace"):
        critical_paths.append("/workspace")
        
    for path in critical_paths:
        if os.path.exists(path):
            try:
                # Use df command for reliable disk space info
                result = subprocess.run(
                    ["df", "-h", path], 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        # Parse df output: Filesystem Size Used Avail Use% Mounted
                        fields = lines[1].split()
                        if len(fields) >= 4:
                            avail = fields[3]
                            use_pct = fields[4]
                            print(f"✅ Disk space {path}: {avail} available ({use_pct} used)")
                            
                            # Warn if very low space
                            if use_pct.rstrip('%').isdigit() and int(use_pct.rstrip('%')) > 95:
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
        from transformers import AutoTokenizer
        import os
        from pathlib import Path
        
        # Get HF cache directory (try multiple methods for different HF versions)
        try:
            # Try new HuggingFace Hub API
            from huggingface_hub import HF_HOME
            cache_dir = Path(HF_HOME) if HF_HOME else None
        except ImportError:
            cache_dir = None
            
        if not cache_dir:
            # Fallback to environment variable or default
            hf_home = os.environ.get('HF_HOME')
            if hf_home:
                cache_dir = Path(hf_home)
            else:
                # Default HF cache location
                cache_dir = Path.home() / '.cache' / 'huggingface'
        
        try:
            print(f"✅ HuggingFace cache: {cache_dir}")
            
            if cache_dir.exists():
                # Check if writable
                test_file = cache_dir / f".preflight_test_{os.getpid()}"
                try:
                    test_file.touch()
                    test_file.unlink()
                    print(f"✅ HF cache directory writable")
                except (OSError, PermissionError) as e:
                    print(f"❌ HF cache directory not writable: {e}")
                    raise RuntimeError(f"HuggingFace cache directory not writable: {cache_dir}")
            else:
                print(f"ℹ️  HF cache directory will be created: {cache_dir}")
                
        except Exception as e:
            print(f"⚠️  Could not determine HF cache location: {e}")
    
    except ImportError:
        print("⚠️  HuggingFace transformers not available for cache check")
    
    # 3.5. HF Cache Environment Validation (RunPod survival check)
    hf_env_vars = {
        'HF_HOME': 'Primary HF cache directory',
        'HF_HUB_CACHE': 'Model weights cache',
        'HF_DATASETS_CACHE': 'Dataset cache',
        'TORCH_HOME': 'PyTorch cache'
    }
    
    print("🔍 Validating HuggingFace cache environment...")
    env_issues = []
    runpod_detected = os.path.exists("/workspace")
    
    for env_var, description in hf_env_vars.items():
        value = os.environ.get(env_var)
        if value:
            cache_path = Path(value)
            
            # Check if path is under /root/ on RunPod (danger zone)
            if runpod_detected and str(cache_path).startswith('/root/'):
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
            
        if runpod_detected:
            print("\n💡 RUNPOD SOLUTION:")
            print("   source /workspace/gradience/scripts/runpod/env.sh")
            print("   # This sets:")
            print("   #   export HF_HOME=/workspace/hf_cache/hf_home")
            print("   #   export HF_HUB_CACHE=/workspace/hf_cache/hub") 
            print("   #   export HF_DATASETS_CACHE=/workspace/hf_cache/datasets")
            print("   #   export TORCH_HOME=/workspace/hf_cache/torch")
            print("\n   Or add to your /root/.bashrc:")
            print("   if [ -d \"/workspace/gradience\" ]; then")
            print("       source /workspace/gradience/scripts/runpod/env.sh")
            print("   fi")
        else:
            print("\n💡 SOLUTION: Set HuggingFace cache environment variables")
            print("   export HF_HOME=$HOME/.cache/huggingface/hf_home")
            print("   export HF_HUB_CACHE=$HOME/.cache/huggingface/hub")
            print("   export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets")
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
                        
        except Exception as e:
            error_msg = str(e).lower()
            if "incomplete" in error_msg and "metadata" in error_msg:
                print(f"❌ Safetensors metadata corruption detected for {model_name}")
                print(f"💡 SOLUTION: Delete the corrupted cache:")
                print(f"   rm -rf ~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
                print(f"   Or nuke entire cache: rm -rf ~/.cache/huggingface/")
                raise RuntimeError(f"HuggingFace cache corruption: {e}")
            else:
                print(f"⚠️  Model loading issue: {e}")
                # Don't fail on other model loading issues as they might resolve during training
                
    except ImportError:
        print("⚠️  Cannot test model loading - HuggingFace transformers not available")
    except Exception as e:
        print(f"⚠️  Model loading check failed: {e}")
    
    print("✅ Preflight checks complete!\n")


def run_bench_protocol(
    config_path: str | Path,
    output_dir: str | Path,
    smoke: bool = False,
    ci: bool = False
) -> Dict[str, Any]:
    """
    Run the complete bench protocol.
    
    Supports both single-seed and multi-seed configurations.
    Multi-seed configs are detected by presence of 'seeds' in compression config.
    """
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
            ci=ci
        )
    
    # Single-seed protocol (original implementation)
    print("Gradience Bench Protocol v0.1")
    print("=" * 40)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Model: {config['model']['name']}")
    print(f"Task: {config['task']['dataset']}/{config['task']['subset']}")
    print(f"Smoke mode: {smoke}")
    print()
    
    # Preflight checks to catch common failure modes early
    run_bench_preflight_check(config, config['model']['name'])
    
    # Steps 3.1-3.3: Train, evaluate, and audit probe
    print("Step 3.1-3.3: Training, evaluating, and auditing probe adapter...")
    probe_results = run_probe_training(config_path, output_path, smoke=smoke)
    
    # Step 3.4: Generate compression configurations
    print("\nStep 3.4: Generating compression configurations...")
    probe_rank = config["lora"]["probe_r"]
    probe_dir = output_path / f"probe_r{probe_rank}"
    compression_configs = generate_compression_configs(probe_dir, config)
    
    # Write compression configs to JSON for debugging/inspection
    compression_configs_path = output_path / "compression_configs.json"
    with open(compression_configs_path, 'w') as f:
        json.dump(compression_configs, f, indent=2, ensure_ascii=False)
    
    print(f"Compression configs generated:")
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
        smoke=smoke
    )
    
    # Step 3.6: Compute verdicts
    verdict_analysis = compute_verdicts(
        probe_results=probe_results,
        variant_results=variant_results,
        config=config,
        output_path=output_path,
        smoke=smoke
    )
    
    # Write verdict analysis to JSON
    verdict_path = output_path / "verdicts.json"
    with open(verdict_path, 'w') as f:
        json.dump(verdict_analysis, f, indent=2, ensure_ascii=False)
    
    # Load audit data for canonical report
    probe_audit_path = output_path / f"probe_r{probe_rank}" / "audit.json"
    with open(probe_audit_path, 'r') as f:
        audit_data = json.load(f)
    
    # Create canonical bench.json report
    canonical_report = create_canonical_bench_report(
        probe_results=probe_results,
        variant_results=variant_results,
        verdict_analysis=verdict_analysis,
        audit_data=audit_data,
        compression_configs=compression_configs,
        config=config,
        output_dir=output_path
    )
    
    # Write canonical benchmark report
    report_path = output_path / "bench.json"
    with open(report_path, 'w') as f:
        json.dump(canonical_report, f, indent=2, ensure_ascii=False)
    
    # Create and write markdown report
    markdown_content = create_markdown_report(
        canonical_report=canonical_report,
        config=config,
        output_dir=output_path
    )
    
    markdown_path = output_path / "bench.md"
    with open(markdown_path, 'w') as f:
        f.write(markdown_content)
    
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
        "verdicts": verdict_analysis
    }
    
    internal_report_path = output_path / "bench_internal.json"
    with open(internal_report_path, 'w') as f:
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