"""
Shared test fixtures and helpers for the Gradience test suite.

Pytest fixtures (auto-discovered, no import needed):
    tmp_dir    — clean temporary directory (auto-cleaned by pytest)
    probe_dir  — tmp directory with a default audit.json pre-written
    seed_all   — fixes torch + numpy seeds to 42 for reproducibility

Helper functions (importable by any test, including unittest.TestCase):
    make_audit_data(**overrides) — realistic audit.json dict factory
    make_config(**overrides)     — minimal bench config dict factory
    write_audit_file(path, data) — writes audit.json to disk
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a clean temporary directory (auto-cleaned by pytest)."""
    return tmp_path


@pytest.fixture
def probe_dir(tmp_path: Path) -> Path:
    """Temporary probe directory with a default audit.json pre-written.

    Uses :func:`make_audit_data` defaults so tests that only need a valid
    audit.json on disk don't have to set one up themselves.
    """
    d = tmp_path / "probe"
    d.mkdir()
    write_audit_file(d, make_audit_data())
    return d


@pytest.fixture
def seed_all():
    """Fix torch **and** numpy seeds to 42 for reproducible tests."""
    import numpy as np
    import torch

    torch.manual_seed(42)
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Helper functions — plain callables, usable from unittest.TestCase too
# ---------------------------------------------------------------------------


def make_audit_data(**overrides: Any) -> dict[str, Any]:
    """Return a realistic ``audit.json`` dict with sensible defaults.

    Callers can override any top-level key::

        data = make_audit_data(probe_rank=64)
    """
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
    """Return a minimal bench config dict.

    Nested dict overrides are *shallow-merged* into the matching section::

        cfg = make_config(compression={"fast_mode": False})
        # cfg["compression"]["allowed_ranks"] is preserved
    """
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
    """Write *audit_data* as ``audit.json`` inside *probe_dir*.

    Creates *probe_dir* (and parents) if it does not exist.
    """
    probe_dir.mkdir(parents=True, exist_ok=True)
    with open(probe_dir / "audit.json", "w") as f:
        json.dump(audit_data, f)
