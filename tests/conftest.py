"""Shared pytest fixtures and bench-helper re-exports for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.bench_fixtures import make_audit_data, make_config, write_audit_file

__all__ = [
    "make_audit_data",
    "make_config",
    "probe_dir",
    "seed_all",
    "tmp_dir",
    "write_audit_file",
]

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
