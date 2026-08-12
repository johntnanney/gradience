"""
Shared utility functions for bench modules.

Small pure functions used by both reporting and compression clusters.
Extracted from protocol.py — no gradience imports.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from gradience.bench.constants import CONFIG_HASH_LENGTH


def round_to_allowed_ranks(suggested_r: int | float, allowed_ranks: list[int]) -> int:
    """Round a suggested rank to the nearest allowed rank."""
    if not allowed_ranks:
        return max(1, round(suggested_r))

    if suggested_r in allowed_ranks:
        return int(suggested_r)

    # Find closest allowed rank
    return min(allowed_ranks, key=lambda x: abs(x - suggested_r))


# Hugging Face retired bare canonical dataset ids; the Hub now requires
# "namespace/name". Configs and recorded provenance still carry the legacy
# short ids, so normalize at the load boundary rather than rewriting configs
# (which would change every create_config_hash() value).
LEGACY_DATASET_IDS = {
    "glue": "nyu-mll/glue",
    "gsm8k": "openai/gsm8k",
    "sst2": "stanfordnlp/sst2",
}


def canonical_dataset_id(dataset: str) -> str:
    """Map a legacy bare dataset id to its namespaced Hub id.

    Already-namespaced ids and unknown names pass through unchanged.
    """
    if not dataset or "/" in dataset:
        return dataset
    return LEGACY_DATASET_IDS.get(dataset.lower(), dataset)


def get_primary_metric_key(config: dict[str, Any]) -> str:
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


def create_config_hash(config: dict[str, Any]) -> str:
    """Create a stable hash of the configuration for reference."""
    # Create a stable string representation
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_str.encode()).hexdigest()[:CONFIG_HASH_LENGTH]
