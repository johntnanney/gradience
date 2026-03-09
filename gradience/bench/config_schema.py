"""Bench configuration schema validation.

Lightweight validation for bench YAML configs.
Validates required keys at load time and normalizes deprecated field names.
"""

from __future__ import annotations

import warnings
from typing import Any


def _normalize_train_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize train config: lr -> learning_rate with deprecation warning."""
    result = dict(raw)
    if "lr" in result:
        if "learning_rate" not in result:
            warnings.warn(
                "Config key 'lr' is deprecated in train config. "
                "Use 'learning_rate' instead. "
                "Automatic normalization will be removed in v1.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            result["learning_rate"] = result.pop("lr")
        else:
            warnings.warn(
                "Both 'lr' and 'learning_rate' found in train config. "
                "Using 'learning_rate'. Remove 'lr' to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
            result.pop("lr")
    return result


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a raw YAML config dict.

    Checks required top-level keys exist and normalizes deprecated field names.
    Returns the normalized dict (preserves backward compat with .get() callers).

    Raises:
        ValueError: If required keys are missing.
    """
    errors = []

    if "model" not in raw:
        errors.append("Missing required top-level key: 'model'")
    elif "name" not in raw.get("model", {}):
        errors.append("Missing required key: model.name")

    if "task" not in raw:
        errors.append("Missing required top-level key: 'task'")
    elif "dataset" not in raw.get("task", {}):
        errors.append("Missing required key: task.dataset")

    if "lora" not in raw:
        errors.append("Missing required top-level key: 'lora'")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    result = dict(raw)
    if "train" in result:
        result["train"] = _normalize_train_config(result["train"])

    return result
