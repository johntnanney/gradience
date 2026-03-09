"""
Typed data structures for the bench protocol.

Provides TypedDicts for JSON-derived data (configs, reports) and
dataclasses for Python-only structures (intermediate results).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing import TypedDict

    from typing_extensions import NotRequired


# ---------------------------------------------------------------------------
# Configuration TypedDicts (from YAML config files)
# ---------------------------------------------------------------------------


class ModelConfig(TypedDict):
    """Model configuration section of a bench YAML config."""

    name_or_path: str
    type: NotRequired[str]
    torch_dtype: NotRequired[str]
    gradient_checkpointing: NotRequired[bool]
    use_cache: NotRequired[bool]


class LoRAConfig(TypedDict):
    """LoRA configuration section of a bench YAML config."""

    r: int
    alpha: NotRequired[float]
    dropout: NotRequired[float]
    target_modules: NotRequired[list[str]]
    bias: NotRequired[str]


class TrainConfig(TypedDict):
    """Training configuration section."""

    seed: NotRequired[int]
    batch_size: NotRequired[int]
    max_steps: NotRequired[int]
    learning_rate: NotRequired[float]
    weight_decay: NotRequired[float]
    epochs: NotRequired[int]
    lr_scheduler_type: NotRequired[str]
    warmup_ratio: NotRequired[float]


class TaskConfig(TypedDict):
    """Task configuration section."""

    name: NotRequired[str]
    dataset: NotRequired[str]
    type: NotRequired[str]
    task_family: NotRequired[str]


class CompressionConfig(TypedDict):
    """Compression configuration section."""

    acc_tolerance: NotRequired[float]
    seeds: NotRequired[list[int]]
    policies: NotRequired[list[str]]
    enable_fast_mode: NotRequired[bool]


class AuditConfig(TypedDict):
    """Audit configuration section."""

    compute_udr: NotRequired[bool]
    base_model: NotRequired[str]
    base_norms_cache: NotRequired[str]
    enable_composition_analysis: NotRequired[bool]


class RuntimeConfig(TypedDict):
    """Runtime configuration section."""

    device: NotRequired[str]
    smoke_train_samples: NotRequired[int]
    smoke_eval_samples: NotRequired[int]
    smoke_epochs: NotRequired[int]
    smoke_max_train_steps: NotRequired[int]
    keep_adapter_weights: NotRequired[bool]
    keep_checkpoints: NotRequired[bool]


class BenchConfig(TypedDict):
    """Top-level bench protocol configuration (from YAML)."""

    model: ModelConfig
    lora: LoRAConfig
    train: TrainConfig
    task: TaskConfig
    compression: NotRequired[CompressionConfig]
    audit: NotRequired[AuditConfig]
    runtime: NotRequired[RuntimeConfig]


# ---------------------------------------------------------------------------
# Result structures (from training/evaluation)
# ---------------------------------------------------------------------------


class EvalResult(TypedDict):
    """Evaluation metrics returned from a training run."""

    eval_loss: NotRequired[float]
    accuracy: NotRequired[float]
    exact_match: NotRequired[float]
    eval_samples: NotRequired[int]
    eval_runtime: NotRequired[float]
    eval_steps_per_second: NotRequired[float]


@dataclass
class VariantResult:
    """Result from training a single compressed variant."""

    name: str
    status: str  # "completed", "failed", "skipped"
    accuracy: float | None = None
    eval_loss: float | None = None
    param_count: int | None = None
    param_reduction: float | None = None
    delta_vs_probe: float | None = None
    training_loss: float | None = None
    error: str | None = None
    seed: int | None = None


@dataclass
class ProbeResult:
    """Result from probe adapter training."""

    accuracy: float
    eval_loss: float | None = None
    param_count: int | None = None
    training_loss: float | None = None
    eval_samples: int | None = None


# ---------------------------------------------------------------------------
# Verdict structures
# ---------------------------------------------------------------------------


class VariantVerdict(TypedDict):
    """Verdict for a single compression variant."""

    status: str  # "pass", "fail", "marginal"
    accuracy: NotRequired[float]
    delta_vs_probe: NotRequired[float]
    param_reduction: NotRequired[float]
    reason: NotRequired[str]
    tier: NotRequired[str]  # "A", "B"


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------


class GPUDeviceInfo(TypedDict):
    """Information about a single GPU device."""

    name: str
    total_memory: NotRequired[int]
    major: NotRequired[int]
    minor: NotRequired[int]


class EnvironmentInfo(TypedDict):
    """System environment information for reproducibility."""

    python_version: str
    platform: NotRequired[str]
    torch_version: NotRequired[str]
    cuda_available: NotRequired[bool]
    cuda_version: NotRequired[str]
    gpu_devices: NotRequired[list[GPUDeviceInfo]]
    transformers_version: NotRequired[str]
    peft_version: NotRequired[str]
    gradience_version: NotRequired[str]
