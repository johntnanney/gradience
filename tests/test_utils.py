"""
Test utilities for creating minimal adapter fixtures.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def create_minimal_adapter_config(
    peft_dir: Path, rank: int = 16, target_modules: list | None = None, task_type: str = "SEQ_CLS"
) -> None:
    """Create minimal adapter_config.json for tests."""
    if target_modules is None:
        target_modules = ["q_lin", "v_lin"]

    config = {
        "peft_type": "LORA",
        "task_type": task_type,
        "r": rank,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": target_modules,
        "modules_to_save": None,
        "base_model_name_or_path": "distilbert-base-uncased",
        "inference_mode": False,
    }

    peft_dir.mkdir(parents=True, exist_ok=True)
    with open(peft_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)


def create_minimal_adapter_weights(
    peft_dir: Path, rank: int = 16, target_modules: list | None = None, model_dim: int = 768
) -> Path:
    """
    Create minimal adapter weights for tests.

    Returns path to the created weights file (safetensors preferred, torch fallback).
    """
    if target_modules is None:
        target_modules = ["q_lin", "v_lin"]

    state_dict = {}

    # Create LoRA weights for each target module
    for module in target_modules:
        # LoRA A matrices (rank, input_dim)
        A_key = f"base_model.model.distilbert.transformer.layer.0.attention.{module}.lora_A.default.weight"
        state_dict[A_key] = torch.randn(rank, model_dim, dtype=torch.float16) * 0.01

        # LoRA B matrices (output_dim, rank)
        B_key = f"base_model.model.distilbert.transformer.layer.0.attention.{module}.lora_B.default.weight"
        state_dict[B_key] = torch.randn(model_dim, rank, dtype=torch.float16) * 0.01

    # Try safetensors first (modern format), fallback to torch
    peft_dir.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file

        weights_path = peft_dir / "adapter_model.safetensors"
        save_file(state_dict, weights_path)
    except ImportError:
        # Fallback to torch format
        weights_path = peft_dir / "adapter_model.pt"
        torch.save(state_dict, weights_path)

    return weights_path


def create_test_adapter_directory(
    peft_dir: Path,
    rank: int = 16,
    target_modules: list | None = None,
    model_dim: int = 768,
    task_type: str = "SEQ_CLS",
) -> dict[str, Path]:
    """
    Create complete minimal adapter directory for tests.

    Returns dict with paths to created files.
    """
    create_minimal_adapter_config(peft_dir, rank, target_modules, task_type)
    weights_path = create_minimal_adapter_weights(peft_dir, rank, target_modules, model_dim)

    return {"config": peft_dir / "adapter_config.json", "weights": weights_path, "directory": peft_dir}


def verify_adapter_files_exist(peft_dir: Path) -> bool:
    """
    Verify that adapter files exist (flexible format support).

    Returns True if both config and weights are found.
    """
    # Check config
    if not (peft_dir / "adapter_config.json").exists():
        return False

    # Check for any supported weights format
    weight_files = [
        "adapter_model.safetensors",
        "adapter_model.pt",
        "adapter_model.bin",
        "adapter_model.pth",
        "pytorch_model.bin",
    ]

    return any((peft_dir / wf).exists() for wf in weight_files)
