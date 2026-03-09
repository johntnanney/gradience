"""Model and dataset setup for bench protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Check for optional dependencies
try:
    import torch
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

from gradience.bench.task_profiles import get_task_profile_from_config
from gradience.peft_utils import (
    create_complete_alpha_pattern,
    create_complete_rank_pattern,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate YAML configuration file."""
    from gradience.bench.config_schema import validate_config

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return validate_config(raw)


def setup_dataset(config: dict[str, Any], smoke: bool = False):
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


def _load_base_model(model_config: dict[str, Any], model_name: str, device: str):
    """Load tokenizer and base model with shared configuration logic.

    Returns:
        (tokenizer, model, task_type) tuple
    """
    if not HAS_TRAINING_DEPS:
        raise ImportError("Training dependencies not available (transformers, peft)")

    model_type = model_config.get("type", "seqcls")

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
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, torch_dtype=torch_dtype)
        task_type = TaskType.SEQ_CLS

    return tokenizer, model, task_type


def setup_model_and_tokenizer(config: dict[str, Any], device: str = "cpu"):
    """Setup base model, tokenizer, and LoRA configuration."""
    model_config = config["model"]
    model_name = model_config["name"]
    lora_config = config["lora"]

    tokenizer, model, task_type = _load_base_model(model_config, model_name, device)

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


def setup_compressed_model_and_tokenizer(
    config: dict[str, Any], compression_config: dict[str, Any], device: str = "cpu"
):
    """Setup model and tokenizer with compressed LoRA configuration."""
    model_config = config["model"]
    model_name = model_config["name"]
    base_lora_config = config["lora"]
    variant_config = compression_config["config"]

    tokenizer, model, task_type = _load_base_model(model_config, model_name, device)

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

    model = get_peft_model(model, peft_config)

    return tokenizer, model


def _unwrap_model_for_save(trainer, model):
    # Try accelerator unwrap first (works with device_map / accelerate wrapping)
    if trainer is not None and hasattr(trainer, "accelerator"):
        try:
            return trainer.accelerator.unwrap_model(model)
        except (AttributeError, RuntimeError):
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
        raise RuntimeError(f"Bench expected a PEFT model but got {type(m)}. Refusing to save full model. ({label})")

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
        raise RuntimeError(f"Adapter save succeeded but adapter_model.(safetensors|bin) missing in: {out} ({label})")

    return out
