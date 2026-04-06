"""Reusable bench-oriented test factories shared across the main test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def make_audit_data(**overrides: Any) -> dict[str, Any]:
    """Return a realistic ``audit.json`` dict with sensible defaults."""
    data: dict[str, Any] = {
        "probe_rank": 32,
        "summary": {
            "utilization_mean": 0.2,
            "stable_rank_mean": 8.0,
            "effective_rank_mean": 20.0,
        },
        "suggested_r_global_median": 12,
        "suggested_r_global_90": 24,
        "policy_global_suggestions": {
            "energy_90": {"uniform_p90": 20, "uniform_median": 8},
            "knee": {"uniform_p90": 22, "uniform_median": 10},
            "erank": {"uniform_p90": 16, "uniform_median": 6},
            "oht": {"uniform_p90": 18},
        },
        "per_layer_suggestions": {
            "default_r": 2,
            "rank_pattern": {
                "layer.0.q": 32,
                "layer.0.k": 8,
                "layer.1.q": 16,
                "layer.1.k": 4,
            },
        },
        "layers": [],
    }
    data.update(overrides)
    return data


def make_config(**overrides: Any) -> dict[str, Any]:
    """Return a minimal bench config dict with shallow nested dict merging."""
    cfg: dict[str, Any] = {
        "compression": {
            "allowed_ranks": [1, 2, 4, 8, 16, 32],
            "fast_mode": True,
            "max_candidates": 4,
        },
        "lora": {
            "probe_r": 32,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj"],
        },
        "train": {"seed": 42},
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key] = {**cfg[key], **val}
        else:
            cfg[key] = val
    return cfg


def write_audit_file(probe_dir: Path, audit_data: dict[str, Any]) -> None:
    """Write *audit_data* as ``audit.json`` inside *probe_dir*."""
    probe_dir.mkdir(parents=True, exist_ok=True)
    with open(probe_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f)
