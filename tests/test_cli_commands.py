"""Subprocess-based CLI tests for all primary gradience commands.

Tests cover:
- gradience --help, --version
- gradience check (valid config, missing file, no args)
- gradience audit (valid adapter, missing dir, invalid adapter)
- gradience monitor (valid JSONL, missing file, empty file, --json)
- gradience merge-audit (valid pair, single adapter, missing adapter)
- gradience truncate (valid adapter, missing args)
- gradience explain (missing args)

Each test runs the CLI as a subprocess and checks exit code + output.
Tests complete in <5s each with no GPU or model downloads required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

try:
    from safetensors.torch import save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run `python -m gradience <args>` and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "gradience", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _make_adapter_dir(base: Path, *, rank: int = 4, dim: int = 64, n_layers: int = 2) -> Path:
    """Create a minimal PEFT adapter directory with synthetic weights."""
    adapter_dir = base / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "r": rank,
        "lora_alpha": rank,
        "target_modules": ["q_proj", "v_proj"],
        "peft_type": "LORA",
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config, f)

    weights: dict[str, torch.Tensor] = {}
    for i in range(n_layers):
        for proj in ("q_proj", "v_proj"):
            prefix = f"base_model.model.layer.{i}.self_attn.{proj}"
            weights[f"{prefix}.lora_A.weight"] = torch.randn(rank, dim)
            weights[f"{prefix}.lora_B.weight"] = torch.randn(dim, rank)

    if HAS_SAFETENSORS:
        save_file(weights, adapter_dir / "adapter_model.safetensors")
    else:
        torch.save(weights, adapter_dir / "adapter_model.bin")

    return adapter_dir


def _make_check_config(base: Path) -> Path:
    """Create a minimal vNext check config JSON file."""
    cfg = {
        "model_name": "test-model",
        "dataset_name": "test-dataset",
        "task_profile": "easy_classification",
        "optimizer": {"name": "AdamW", "lr": 0.0002, "weight_decay": 0.0},
        "lora": {"r": 8, "alpha": 8, "target_modules": ["q_proj", "k_proj"]},
        "training": {"seed": 42, "max_steps": 10, "dtype": "float32"},
    }
    path = base / "check_config.json"
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _make_telemetry_jsonl(base: Path, *, n_steps: int = 3) -> Path:
    """Create a minimal valid vNext telemetry JSONL file."""
    path = base / "telemetry.jsonl"
    events = [
        {
            "schema": "gradience.vnext.telemetry/v1",
            "ts": 1.0,
            "run_id": "cli_test",
            "event": "run_start",
            "step": None,
            "config": {
                "model_name": "test-model",
                "dataset_name": "test-data",
                "task_profile": "easy_classification",
                "optimizer": {"name": "AdamW", "lr": 5e-5, "weight_decay": 0.0},
                "lora": {"r": 8, "alpha": 8, "target_modules": ["q_proj"]},
                "training": {"seed": 0, "max_steps": n_steps, "dtype": "float32"},
            },
        },
    ]
    for i in range(1, n_steps + 1):
        events.append(
            {
                "schema": "gradience.vnext.telemetry/v1",
                "ts": float(1 + i),
                "run_id": "cli_test",
                "event": "train_step",
                "step": i,
                "loss": 2.0 - 0.1 * i,
                "lr": 5e-5,
            }
        )
    events.append(
        {
            "schema": "gradience.vnext.telemetry/v1",
            "ts": float(n_steps + 2),
            "run_id": "cli_test",
            "event": "run_end",
            "step": None,
            "status": "ok",
        }
    )
    with open(path, "w") as f:
        for evt in events:
            f.write(json.dumps(evt) + "\n")
    return path


# ===========================================================================
# Tests: --help and --version
# ===========================================================================


class TestHelpAndVersion:
    """Tests for top-level --help and --version flags."""

    def test_help_exits_zero(self):
        p = run_cli("--help")
        assert p.returncode == 0
        assert "usage" in p.stdout.lower() or "gradience" in p.stdout.lower()

    def test_no_args_prints_help(self):
        """Running gradience with no args should print help and exit 0."""
        p = run_cli()
        assert p.returncode == 0
        assert "usage" in p.stdout.lower() or "gradience" in p.stdout.lower()

    def test_subcommand_help(self):
        """Each subcommand should accept --help."""
        for cmd in ("check", "audit", "monitor", "merge-audit"):
            p = run_cli(cmd, "--help")
            assert p.returncode == 0, f"{cmd} --help failed: {p.stderr}"


# ===========================================================================
# Tests: gradience check
# ===========================================================================


class TestCheckCommand:
    """Tests for the `gradience check` subcommand."""

    def test_check_valid_config(self, tmp_path: Path):
        config_path = _make_check_config(tmp_path)
        p = run_cli("check", str(config_path))
        assert p.returncode == 0

    def test_check_valid_config_json_output(self, tmp_path: Path):
        config_path = _make_check_config(tmp_path)
        p = run_cli("check", str(config_path), "--json")
        assert p.returncode == 0
        data = json.loads(p.stdout)
        assert "recommendations" in data or "config" in data

    def test_check_fixture_config(self):
        """Use the existing fixture config."""
        fixture = Path("tests/fixtures/vnext_check_config.json")
        if not fixture.exists():
            pytest.skip("Fixture vnext_check_config.json not found")
        p = run_cli("check", str(fixture))
        assert p.returncode == 0

    def test_check_missing_file(self):
        p = run_cli("check", "/nonexistent/config.yaml")
        assert p.returncode != 0

    def test_check_no_args(self):
        p = run_cli("check")
        assert p.returncode != 0

    def test_check_with_cli_overrides(self, tmp_path: Path):
        config_path = _make_check_config(tmp_path)
        p = run_cli("check", str(config_path), "--r", "16", "--alpha", "32")
        assert p.returncode == 0


# ===========================================================================
# Tests: gradience audit
# ===========================================================================


class TestAuditCommand:
    """Tests for the `gradience audit` subcommand."""

    def test_audit_valid_adapter(self, tmp_path: Path):
        adapter_dir = _make_adapter_dir(tmp_path)
        p = run_cli("audit", "--peft-dir", str(adapter_dir))
        assert p.returncode == 0

    def test_audit_with_layers(self, tmp_path: Path):
        adapter_dir = _make_adapter_dir(tmp_path)
        p = run_cli("audit", "--peft-dir", str(adapter_dir), "--layers")
        assert p.returncode == 0

    def test_audit_json_output(self, tmp_path: Path):
        adapter_dir = _make_adapter_dir(tmp_path)
        p = run_cli("audit", "--peft-dir", str(adapter_dir), "--json")
        assert p.returncode == 0
        data = json.loads(p.stdout)
        assert "n_layers" in data
        assert "total_lora_params" in data

    def test_audit_with_suggest_per_layer(self, tmp_path: Path):
        adapter_dir = _make_adapter_dir(tmp_path)
        p = run_cli("audit", "--peft-dir", str(adapter_dir), "--suggest-per-layer")
        assert p.returncode == 0

    def test_audit_missing_dir_reports_issues(self):
        """Missing dir exits 0 but reports issues in output."""
        p = run_cli("audit", "--peft-dir", "/nonexistent/adapter")
        assert p.returncode == 0
        assert "not found" in p.stdout.lower() or "issues" in p.stdout.lower()

    def test_audit_empty_dir_reports_issues(self, tmp_path: Path):
        """Empty dir exits 0 but reports issues in output."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        p = run_cli("audit", "--peft-dir", str(empty_dir))
        assert p.returncode == 0
        assert "not found" in p.stdout.lower() or "issues" in p.stdout.lower()

    def test_audit_no_args(self):
        p = run_cli("audit")
        assert p.returncode != 0


# ===========================================================================
# Tests: gradience monitor
# ===========================================================================


class TestMonitorCommand:
    """Tests for the `gradience monitor` subcommand."""

    def test_monitor_valid_jsonl(self, tmp_path: Path):
        jsonl_path = _make_telemetry_jsonl(tmp_path)
        p = run_cli("monitor", str(jsonl_path))
        assert p.returncode == 0

    def test_monitor_fixture_jsonl(self):
        fixture = Path("tests/fixtures/vnext_minimal.jsonl")
        if not fixture.exists():
            pytest.skip("Fixture vnext_minimal.jsonl not found")
        p = run_cli("monitor", str(fixture))
        assert p.returncode == 0

    def test_monitor_json_output(self, tmp_path: Path):
        jsonl_path = _make_telemetry_jsonl(tmp_path)
        p = run_cli("monitor", str(jsonl_path), "--json")
        assert p.returncode == 0
        data = json.loads(p.stdout)
        assert isinstance(data, dict)

    def test_monitor_verbose(self, tmp_path: Path):
        jsonl_path = _make_telemetry_jsonl(tmp_path)
        p = run_cli("monitor", str(jsonl_path), "--verbose")
        assert p.returncode == 0

    def test_monitor_missing_file(self):
        p = run_cli("monitor", "/nonexistent/telemetry.jsonl")
        assert p.returncode != 0

    def test_monitor_empty_file(self, tmp_path: Path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.touch()
        p = run_cli("monitor", str(empty_file))
        # Empty file may exit 0 (no events) or non-zero — either is acceptable
        # but it should not crash with a traceback
        assert "Traceback" not in p.stderr

    def test_monitor_malformed_jsonl(self, tmp_path: Path):
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text("not valid json\n{also bad\n")
        p = run_cli("monitor", str(bad_file))
        # Should handle gracefully (skip bad lines or report error)
        assert "Traceback" not in p.stderr

    def test_monitor_no_args(self):
        p = run_cli("monitor")
        assert p.returncode != 0


# ===========================================================================
# Tests: gradience merge-audit
# ===========================================================================


class TestMergeAuditCommand:
    """Tests for the `gradience merge-audit` subcommand."""

    def test_merge_audit_valid_pair(self, tmp_path: Path):
        adapter_a = _make_adapter_dir(tmp_path / "a")
        adapter_b = _make_adapter_dir(tmp_path / "b")
        p = run_cli(
            "merge-audit",
            "--adapter-a",
            str(adapter_a),
            "--adapter-b",
            str(adapter_b),
        )
        assert p.returncode == 0

    def test_merge_audit_with_output(self, tmp_path: Path):
        adapter_a = _make_adapter_dir(tmp_path / "a")
        adapter_b = _make_adapter_dir(tmp_path / "b")
        out_dir = tmp_path / "output"
        p = run_cli(
            "merge-audit",
            "--adapter-a",
            str(adapter_a),
            "--adapter-b",
            str(adapter_b),
            "--output-dir",
            str(out_dir),
        )
        assert p.returncode == 0
        assert (out_dir / "merge_audit.json").exists()
        assert (out_dir / "merge_audit.md").exists()

    def test_merge_audit_json_output(self, tmp_path: Path):
        adapter_a = _make_adapter_dir(tmp_path / "a")
        adapter_b = _make_adapter_dir(tmp_path / "b")
        p = run_cli(
            "merge-audit",
            "--adapter-a",
            str(adapter_a),
            "--adapter-b",
            str(adapter_b),
            "--json",
        )
        assert p.returncode == 0
        data = json.loads(p.stdout)
        assert "overall_verdict" in data or "verdict" in data or "per_layer" in data

    def test_merge_audit_verbose(self, tmp_path: Path):
        adapter_a = _make_adapter_dir(tmp_path / "a")
        adapter_b = _make_adapter_dir(tmp_path / "b")
        p = run_cli(
            "merge-audit",
            "--adapter-a",
            str(adapter_a),
            "--adapter-b",
            str(adapter_b),
            "--verbose",
        )
        assert p.returncode == 0
        # Verbose output should mention loading
        assert "Loading" in p.stdout or "adapter" in p.stdout.lower()

    def test_merge_audit_missing_adapter_a(self, tmp_path: Path):
        adapter_b = _make_adapter_dir(tmp_path / "b")
        p = run_cli(
            "merge-audit",
            "--adapter-a",
            "/nonexistent/adapter",
            "--adapter-b",
            str(adapter_b),
        )
        assert p.returncode != 0

    def test_merge_audit_missing_adapter_b(self, tmp_path: Path):
        adapter_a = _make_adapter_dir(tmp_path / "a")
        p = run_cli(
            "merge-audit",
            "--adapter-a",
            str(adapter_a),
            "--adapter-b",
            "/nonexistent/adapter",
        )
        assert p.returncode != 0

    def test_merge_audit_no_args(self):
        p = run_cli("merge-audit")
        assert p.returncode != 0


# ===========================================================================
# Tests: gradience truncate
# ===========================================================================


class TestTruncateCommand:
    """Tests for the `gradience truncate` subcommand."""

    def test_truncate_valid(self, tmp_path: Path):
        adapter_dir = _make_adapter_dir(tmp_path, rank=8)
        out_dir = tmp_path / "truncated"
        p = run_cli(
            "truncate",
            "--peft-dir",
            str(adapter_dir),
            "--out-dir",
            str(out_dir),
            "--rank",
            "4",
        )
        assert p.returncode == 0
        assert (out_dir / "adapter_config.json").exists()

    def test_truncate_with_report(self, tmp_path: Path):
        adapter_dir = _make_adapter_dir(tmp_path, rank=8)
        out_dir = tmp_path / "truncated"
        p = run_cli(
            "truncate",
            "--peft-dir",
            str(adapter_dir),
            "--out-dir",
            str(out_dir),
            "--rank",
            "4",
        )
        assert p.returncode == 0
        assert "truncation" in p.stdout.lower() or "rank" in p.stdout.lower()

    def test_truncate_missing_peft_dir(self, tmp_path: Path):
        out_dir = tmp_path / "truncated"
        p = run_cli(
            "truncate",
            "--peft-dir",
            "/nonexistent/adapter",
            "--out-dir",
            str(out_dir),
            "--rank",
            "4",
        )
        assert p.returncode != 0

    def test_truncate_no_args(self):
        p = run_cli("truncate")
        assert p.returncode != 0


# ---------------------------------------------------------------------------
# Compress → Merge CLI pipeline smoke test
# ---------------------------------------------------------------------------


class TestCompressMergeCLIPipeline:
    """End-to-end CLI smoke test: truncate → merge-audit → merge-plan → merge."""

    def test_full_cli_pipeline(self, tmp_path: Path):
        # Create two synthetic adapters (rank=8, separate dirs so names don't clash)
        torch.manual_seed(42)
        dir_a = _make_adapter_dir(tmp_path / "a", rank=8)
        torch.manual_seed(99)
        dir_b = _make_adapter_dir(tmp_path / "b", rank=8)

        # Step 1: Truncate both adapters from rank 8 → rank 4
        compressed_a = tmp_path / "compressed_a"
        p = run_cli("truncate", "--peft-dir", str(dir_a), "--out-dir", str(compressed_a), "--rank", "4")
        assert p.returncode == 0, f"truncate A failed: {p.stderr}"
        assert (compressed_a / "adapter_config.json").exists()

        compressed_b = tmp_path / "compressed_b"
        p = run_cli("truncate", "--peft-dir", str(dir_b), "--out-dir", str(compressed_b), "--rank", "4")
        assert p.returncode == 0, f"truncate B failed: {p.stderr}"
        assert (compressed_b / "adapter_config.json").exists()

        # Step 2: Merge audit on compressed pair
        audit_out = tmp_path / "audit_output"
        p = run_cli(
            "merge-audit",
            "--adapter-a", str(compressed_a),
            "--adapter-b", str(compressed_b),
            "--output-dir", str(audit_out),
        )
        assert p.returncode == 0, f"merge-audit failed: {p.stderr}"
        assert (audit_out / "merge_audit.json").exists()
        assert (audit_out / "merge_audit.md").exists()

        # Step 3: Generate merge plan
        plan_out = tmp_path / "plan_output"
        p = run_cli(
            "merge-plan",
            "--adapter-a", str(compressed_a),
            "--adapter-b", str(compressed_b),
            "--strategy", "audit_aware",
            "--output-dir", str(plan_out),
            "--output-rank", "4",
        )
        assert p.returncode == 0, f"merge-plan failed: {p.stderr}"
        plan_path = plan_out / "merge_plan.json"
        assert plan_path.exists()

        # Step 4: Execute the merge
        merged_out = tmp_path / "merged_adapter"
        p = run_cli(
            "merge",
            "--plan", str(plan_path),
            "--output-dir", str(merged_out),
        )
        assert p.returncode == 0, f"merge failed: {p.stderr}"
        assert (merged_out / "adapter_config.json").exists()
        has_weights = (
            (merged_out / "adapter_model.safetensors").exists()
            or (merged_out / "adapter_model.bin").exists()
        )
        assert has_weights, "merged adapter missing weight file"
        assert (merged_out / "merge_result.json").exists()

        # Validate merged adapter config
        with open(merged_out / "adapter_config.json") as f:
            config = json.load(f)
        assert config["peft_type"] == "LORA"
        assert isinstance(config["r"], int) and config["r"] > 0

        # Provenance: merge metadata embedded in adapter config
        provenance = config.get("gradience_merge")
        assert provenance is not None, "Missing gradience_merge provenance"
        assert provenance["strategy"] == "audit_aware"
        assert "adapter_a" in provenance
        assert "adapter_b" in provenance

        # Provenance: merge_result.json has per-layer reconstruction data
        with open(merged_out / "merge_result.json") as f:
            result = json.load(f)
        assert "per_layer" in result
        assert len(result["per_layer"]) > 0

        # Provenance: merge_plan.json records strategy
        with open(plan_path) as f:
            plan_data = json.load(f)
        assert plan_data["strategy_name"] == "audit_aware"

        # Provenance: merge_audit.json has schema and verdicts
        with open(audit_out / "merge_audit.json") as f:
            audit_data = json.load(f)
        assert "per_layer" in audit_data
        assert len(audit_data["per_layer"]) > 0
