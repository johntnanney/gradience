"""Tests for dependency split: gradience (core) vs gradience[bench].

Verifies that importing core Gradience modules does NOT pull in bench-only
dependencies (transformers, peft, datasets). This ensures `pip install gradience`
remains lightweight while still providing audit, merge-audit, and monitor features.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


BENCH_ONLY_MODULES = {"transformers", "peft", "datasets"}


def _fresh_import_check(import_statement: str) -> set[str]:
    """Run an import in a subprocess and return which bench-only modules got loaded.

    We use a subprocess to get a clean sys.modules state, avoiding pollution
    from other tests that may have imported bench deps.
    """
    code = f"""
import sys
{import_statement}
bench_loaded = {{m for m in {BENCH_ONLY_MODULES!r} if m in sys.modules}}
print(",".join(sorted(bench_loaded)) if bench_loaded else "NONE")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(f"Import failed: {result.stderr.strip()}")

    output = result.stdout.strip()
    if output == "NONE":
        return set()
    return set(output.split(","))


class TestCoreDependencyBoundary:
    """Verify that core imports don't pull in bench-only deps."""

    def test_import_gradience_no_bench_deps(self):
        """import gradience should NOT load transformers/peft/datasets."""
        loaded = _fresh_import_check("import gradience")
        assert loaded == set(), (
            f"'import gradience' pulled in bench-only deps: {loaded}"
        )

    def test_import_merge_audit_no_bench_deps(self):
        """merge-audit import chain should NOT load bench deps."""
        loaded = _fresh_import_check(
            "from gradience.vnext.merge import merge_audit"
        )
        assert loaded == set(), (
            f"merge-audit import pulled in bench-only deps: {loaded}"
        )

    def test_import_lora_audit_no_bench_deps(self):
        """lora_audit import should NOT load bench deps."""
        loaded = _fresh_import_check(
            "from gradience.vnext.audit.lora_audit import LoRAAuditResult"
        )
        assert loaded == set(), (
            f"lora_audit import pulled in bench-only deps: {loaded}"
        )

    def test_import_telemetry_no_bench_deps(self):
        """Telemetry import should NOT load bench deps."""
        loaded = _fresh_import_check(
            "from gradience.vnext.telemetry import TelemetryWriter, TelemetryReader"
        )
        assert loaded == set(), (
            f"telemetry import pulled in bench-only deps: {loaded}"
        )


class TestCoreDepsPresent:
    """Verify that core dependencies are importable."""

    def test_torch_importable(self):
        """torch should be importable (core dependency)."""
        import torch  # noqa: F401

    def test_safetensors_importable(self):
        """safetensors should be importable (core dependency)."""
        import safetensors  # noqa: F401

    def test_numpy_importable(self):
        """numpy should be importable (core dependency)."""
        import numpy  # noqa: F401

    def test_yaml_importable(self):
        """pyyaml should be importable (core dependency)."""
        import yaml  # noqa: F401

    def test_rich_importable(self):
        """rich should be importable (core dependency)."""
        import rich  # noqa: F401
