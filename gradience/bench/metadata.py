"""
Environment and version metadata for bench reports.

Extracted from protocol.py — pure leaf functions with no gradience imports.
"""

from __future__ import annotations

import sys
import subprocess
from typing import Dict, Any, Optional

from gradience.bench.constants import GIT_SUBPROCESS_TIMEOUT


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
            timeout=GIT_SUBPROCESS_TIMEOUT
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
            timeout=GIT_SUBPROCESS_TIMEOUT
        )
        if result.returncode != 0:
            return "dirty"

        # Check if there are staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True,
            timeout=GIT_SUBPROCESS_TIMEOUT
        )
        if result.returncode != 0:
            return "dirty"

        # Get the exact tag for current commit
        result = subprocess.run(
            ["git", "describe", "--exact-match", "--tags"],
            capture_output=True,
            text=True,
            timeout=GIT_SUBPROCESS_TIMEOUT
        )
        if result.returncode == 0:
            return result.stdout.strip()

        # If no exact tag, get the most recent tag with distance
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=7"],
            capture_output=True,
            text=True,
            timeout=GIT_SUBPROCESS_TIMEOUT
        )
        if result.returncode == 0:
            return result.stdout.strip()

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_hf_model_revision(model_id: str) -> Optional[Dict[str, Optional[str]]]:
    """Get the revision hash for a HuggingFace model."""
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        return {
            "model_id": model_id,
            "revision": info.sha,
            "last_modified": info.lastModified.isoformat() if info.lastModified else None
        }
    except (ImportError, OSError, ValueError, RuntimeError):
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
    except (ImportError, OSError, ValueError, RuntimeError) as e:
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
