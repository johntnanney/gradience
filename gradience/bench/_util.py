"""
Shared utility functions for bench modules.

Small pure functions used by both reporting and compression clusters.
Extracted from protocol.py — no gradience imports.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Any

from gradience.bench.constants import CONFIG_HASH_LENGTH


def round_to_allowed_ranks(suggested_r: int, allowed_ranks: list[int]) -> int:
    """Round a suggested rank to the nearest allowed rank."""
    if suggested_r in allowed_ranks:
        return suggested_r

    # Find closest allowed rank
    return min(allowed_ranks, key=lambda x: abs(x - suggested_r))


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
    # Create a stable string representation
    config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode()).hexdigest()[:CONFIG_HASH_LENGTH]
