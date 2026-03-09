"""Error-path tests for Gradience public API modules.

Tests verify that the correct exception types are raised with useful messages
for invalid inputs, missing data, and constraint violations. Each test targets
a specific `raise` statement in the source code.

Organized by module:
- lora_audit: DependencyError, AuditError for shape/format issues
- merge: MergeError for config/dtype issues
- rank_suggestion: TypeError, ValueError for bad audit dicts
- svd_truncate: AuditError, ConfigError for truncation constraints
- telemetry: ConfigError, TelemetryError for writer violations
- telemetry_reader: TelemetrySchemaError, TelemetryFormatError in strict mode
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
import torch

from gradience.exceptions import (
    AuditError,
    ConfigError,
    DependencyError,
    MergeError,
    TelemetryError,
)

# ===========================================================================
# lora_audit error paths
# ===========================================================================


class TestLoraAuditErrors:
    """Error paths in gradience.vnext.audit.lora_audit."""

    def test_orient_lora_factors_non_2d(self):
        """LoRA factors must be 2D tensors."""
        from gradience.vnext.audit.lora_audit import _orient_lora_factors

        A_3d = torch.randn(2, 3, 4)
        B_2d = torch.randn(4, 2)
        with pytest.raises(AuditError, match="must be 2D"):
            _orient_lora_factors(A_3d, B_2d)

    def test_orient_lora_factors_incompatible_shapes(self):
        """LoRA factors with no valid alignment should raise."""
        from gradience.vnext.audit.lora_audit import _orient_lora_factors

        A = torch.randn(3, 5)
        B = torch.randn(7, 11)
        with pytest.raises(AuditError, match="Cannot align LoRA shapes"):
            _orient_lora_factors(A, B)

    def test_orient_lora_factors_valid_standard(self):
        """Standard orientation (r, in) x (out, r) should work."""
        from gradience.vnext.audit.lora_audit import _orient_lora_factors

        A = torch.randn(4, 64)
        B = torch.randn(64, 4)
        A_out, B_out, r = _orient_lora_factors(A, B)
        assert r == 4

    def test_orient_lora_factors_valid_transposed(self):
        """Transposed orientation (in, r) x (r, out) should be auto-fixed."""
        from gradience.vnext.audit.lora_audit import _orient_lora_factors

        # Use asymmetric dims so standard check (A[0]==B[1]) fails
        A = torch.randn(64, 4)  # (in, r) — transposed
        B = torch.randn(4, 32)  # (r, out) — transposed
        A_out, B_out, r = _orient_lora_factors(A, B)
        assert r == 4

    def test_load_adapter_state_dict_missing_file(self):
        """Missing weights file should raise FileNotFoundError."""
        from gradience.vnext.audit.lora_audit import load_adapter_state_dict

        with pytest.raises(FileNotFoundError):
            load_adapter_state_dict("/nonexistent/weights.safetensors")

    def test_load_adapter_state_dict_unsupported_type(self, tmp_path: Path):
        """Non-dict payload from torch.load should raise AuditError."""
        from gradience.vnext.audit.lora_audit import load_adapter_state_dict

        # Save a list (not a dict) as weights
        weights_path = tmp_path / "bad_weights.bin"
        torch.save([1, 2, 3], str(weights_path))
        with pytest.raises(AuditError, match="Unsupported.*payload type"):
            load_adapter_state_dict(str(weights_path))

    def test_load_yaml_config_without_pyyaml(self, tmp_path: Path):
        """Loading YAML config when PyYAML is not installed raises DependencyError."""
        from gradience.vnext.audit import lora_audit

        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("r: 4\nlora_alpha: 4\n")

        with patch.object(lora_audit, "yaml", None), pytest.raises(DependencyError, match="PyYAML"):
            lora_audit._load_json_or_yaml(yaml_path)

    def test_load_json_config_valid(self, tmp_path: Path):
        """Loading valid JSON config should succeed."""
        from gradience.vnext.audit.lora_audit import _load_json_or_yaml

        json_path = tmp_path / "config.json"
        json_path.write_text('{"r": 4, "lora_alpha": 4}')
        result = _load_json_or_yaml(json_path)
        assert result["r"] == 4


# ===========================================================================
# merge error paths
# ===========================================================================


class TestMergeErrors:
    """Error paths in gradience.vnext.merge modules."""

    def test_merge_audit_invalid_dtype(self, tmp_path: Path):
        """Unsupported compute_dtype should raise MergeError."""
        from gradience.vnext.merge import merge_audit

        # Create two minimal adapter dirs
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            json.dump(
                {"r": 4, "lora_alpha": 4, "target_modules": ["q_proj"], "peft_type": "LORA"},
                open(d / "adapter_config.json", "w"),
            )
            weights = {
                "base_model.model.layer.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 64),
                "base_model.model.layer.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 4),
            }
            try:
                from safetensors.torch import save_file

                save_file(weights, d / "adapter_model.safetensors")
            except ImportError:
                torch.save(weights, d / "adapter_model.bin")

        with pytest.raises(MergeError, match="Unsupported compute_dtype"):
            merge_audit(tmp_path / "a", tmp_path / "b", compute_dtype="bfloat16")

    def test_load_adapter_missing_rank(self, tmp_path: Path):
        """Adapter config missing 'r' should raise MergeError."""
        from gradience.vnext.merge.io import load_adapter

        d = tmp_path / "adapter"
        d.mkdir()
        # Config with no 'r'
        json.dump(
            {"lora_alpha": 4, "target_modules": ["q_proj"], "peft_type": "LORA"},
            open(d / "adapter_config.json", "w"),
        )
        weights = {
            "base_model.model.layer.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 64),
            "base_model.model.layer.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 4),
        }
        try:
            from safetensors.torch import save_file

            save_file(weights, d / "adapter_model.safetensors")
        except ImportError:
            torch.save(weights, d / "adapter_model.bin")

        with pytest.raises(MergeError, match="missing 'r'"):
            load_adapter(d)

    def test_load_adapter_missing_alpha(self, tmp_path: Path):
        """Adapter config missing 'lora_alpha' should raise MergeError."""
        from gradience.vnext.merge.io import load_adapter

        d = tmp_path / "adapter"
        d.mkdir()
        # Config with no 'lora_alpha'
        json.dump(
            {"r": 4, "target_modules": ["q_proj"], "peft_type": "LORA"},
            open(d / "adapter_config.json", "w"),
        )
        weights = {
            "base_model.model.layer.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 64),
            "base_model.model.layer.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 4),
        }
        try:
            from safetensors.torch import save_file

            save_file(weights, d / "adapter_model.safetensors")
        except ImportError:
            torch.save(weights, d / "adapter_model.bin")

        with pytest.raises(MergeError, match="missing 'lora_alpha'"):
            load_adapter(d)

    def test_execute_merge_empty_plan(self, tmp_path: Path):
        """MergePlan with no layer_configs should raise MergeError."""
        from gradience.vnext.merge.executor import execute_merge
        from gradience.vnext.merge.plan import MergePlan

        plan = MergePlan(
            plan_id="test",
            strategy_name="uniform_linear",
            adapter_a_dir=str(tmp_path / "a"),
            adapter_b_dir=str(tmp_path / "b"),
            output_rank=4,
            output_alpha=4.0,
            layer_configs=[],
            metadata={},
        )
        with pytest.raises(MergeError, match="no layer_configs"):
            execute_merge(plan, tmp_path / "out")


# ===========================================================================
# rank_suggestion error paths
# ===========================================================================


class TestRankSuggestionErrors:
    """Error paths in gradience.vnext.rank_suggestion."""

    def test_suggest_global_ranks_not_dict(self):
        """Non-dict audit input should raise TypeError."""
        from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit

        with pytest.raises(TypeError, match="audit must be a dict"):
            suggest_global_ranks_from_audit("not a dict")  # type: ignore[arg-type]

    def test_suggest_global_ranks_no_current_r(self):
        """Audit dict without rank info should raise ValueError."""
        from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit

        audit: dict[str, Any] = {
            "total_lora_params": 1000,
            # No r, current_r, stable_rank_mean, or utilization_mean
        }
        with pytest.raises(ValueError, match="Could not infer current_r"):
            suggest_global_ranks_from_audit(audit)

    def test_suggest_per_layer_ranks_not_dict(self):
        """Non-dict audit input should raise TypeError."""
        from gradience.vnext.rank_suggestion import suggest_per_layer_ranks

        with pytest.raises(TypeError, match="audit must be a dict"):
            suggest_per_layer_ranks(42)  # type: ignore[arg-type]

    def test_suggest_per_layer_ranks_no_layers(self):
        """Audit dict without layer data should raise AuditError."""
        from gradience.vnext.rank_suggestion import suggest_per_layer_ranks

        audit: dict[str, Any] = {"n_layers": 0}
        with pytest.raises(AuditError, match="layer data"):
            suggest_per_layer_ranks(audit)

    def test_suggest_global_ranks_valid(self):
        """Valid audit dict should produce a suggestion without error."""
        from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit

        audit = {
            "total_lora_params": 10000,
            "r": 16,
            "energy_rank_90_p50": 6.0,
            "energy_rank_90_p90": 12.0,
        }
        result = suggest_global_ranks_from_audit(audit)
        assert result.current_r == 16
        assert result.suggested_r_median > 0
        assert result.suggested_r_p90 > 0


# ===========================================================================
# svd_truncate error paths
# ===========================================================================


class TestSvdTruncateErrors:
    """Error paths in gradience.vnext.svd_truncate."""

    def _make_adapter(self, tmp_path: Path, rank: int = 8) -> Path:
        """Create a minimal PEFT adapter directory."""
        d = tmp_path / "adapter"
        d.mkdir(parents=True, exist_ok=True)
        json.dump(
            {"r": rank, "lora_alpha": rank, "target_modules": ["q_proj"], "peft_type": "LORA"},
            open(d / "adapter_config.json", "w"),
        )
        weights = {
            "base_model.model.layer.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, 64),
            "base_model.model.layer.0.self_attn.q_proj.lora_B.weight": torch.randn(64, rank),
        }
        try:
            from safetensors.torch import save_file

            save_file(weights, d / "adapter_model.safetensors")
        except ImportError:
            torch.save(weights, d / "adapter_model.bin")
        return d

    def test_truncate_missing_peft_files(self, tmp_path: Path):
        """Missing PEFT files should raise FileNotFoundError."""
        from gradience.vnext.svd_truncate import svd_truncate_peft_dir

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Could not find PEFT"):
            svd_truncate_peft_dir(empty_dir, tmp_path / "out", target_rank=4)

    def test_truncate_rank_too_large(self, tmp_path: Path):
        """Target rank >= original rank should raise ConfigError."""
        from gradience.vnext.svd_truncate import svd_truncate_peft_dir

        adapter = self._make_adapter(tmp_path, rank=8)
        with pytest.raises(ConfigError, match="must be < original rank"):
            svd_truncate_peft_dir(adapter, tmp_path / "out", target_rank=8)

    def test_truncate_rank_exceeds_original(self, tmp_path: Path):
        """Target rank > original rank should also raise ConfigError."""
        from gradience.vnext.svd_truncate import svd_truncate_peft_dir

        adapter = self._make_adapter(tmp_path, rank=4)
        with pytest.raises(ConfigError, match="must be < original rank"):
            svd_truncate_peft_dir(adapter, tmp_path / "out", target_rank=10)

    def test_truncate_no_lora_pairs(self, tmp_path: Path):
        """Adapter with no LoRA pairs should raise AuditError."""
        from gradience.vnext.svd_truncate import svd_truncate_peft_dir

        d = tmp_path / "adapter"
        d.mkdir()
        json.dump(
            {"r": 8, "lora_alpha": 8, "target_modules": ["q_proj"], "peft_type": "LORA"},
            open(d / "adapter_config.json", "w"),
        )
        # Weights with no LoRA keys
        weights = {"some_other_key": torch.randn(4, 4)}
        try:
            from safetensors.torch import save_file

            save_file(weights, d / "adapter_model.safetensors")
        except ImportError:
            torch.save(weights, d / "adapter_model.bin")

        with pytest.raises(AuditError, match="No complete LoRA"):
            svd_truncate_peft_dir(d, tmp_path / "out", target_rank=4)

    def test_corresponding_lora_b_key_bad_pattern(self):
        """Unexpected LoRA A key pattern should raise AuditError."""
        from gradience.vnext.svd_truncate import _corresponding_lora_B_key

        with pytest.raises(AuditError, match="Unexpected LoRA A key"):
            _corresponding_lora_B_key("some.random.key.not_lora")

    def test_truncate_valid(self, tmp_path: Path):
        """Valid truncation should succeed without error."""
        from gradience.vnext.svd_truncate import svd_truncate_peft_dir

        adapter = self._make_adapter(tmp_path, rank=8)
        out = tmp_path / "truncated"
        report = svd_truncate_peft_dir(adapter, out, target_rank=4)
        assert report.target_rank == 4
        assert report.original_rank == 8


# ===========================================================================
# telemetry writer error paths
# ===========================================================================


class TestTelemetryWriterErrors:
    """Error paths in gradience.vnext.telemetry.TelemetryWriter."""

    def test_invalid_on_text_violation(self, tmp_path: Path):
        """Invalid on_text_violation value should raise ConfigError."""
        from gradience.vnext.telemetry import TelemetryWriter

        with pytest.raises(ConfigError, match="on_text_violation"):
            TelemetryWriter(tmp_path / "out.jsonl", on_text_violation="invalid")

    def test_text_too_long_raise_mode(self, tmp_path: Path):
        """Long text in raise mode should raise TelemetryError."""
        from gradience.vnext.telemetry import TelemetryWriter

        writer = TelemetryWriter(
            tmp_path / "out.jsonl",
            on_text_violation="raise",
            max_str_len=10,
        )
        try:
            with pytest.raises(TelemetryError, match="text too long"):
                writer.metrics(step=1, kind="test", metrics={"text": "x" * 100})
        finally:
            writer.close()

    def test_text_too_long_redact_mode(self, tmp_path: Path):
        """Long text in redact mode should silently redact."""
        from gradience.vnext.telemetry import TelemetryWriter

        writer = TelemetryWriter(
            tmp_path / "out.jsonl",
            on_text_violation="redact",
            max_str_len=10,
        )
        try:
            # Should not raise
            writer.metrics(step=1, kind="test", metrics={"text": "x" * 100})
        finally:
            writer.close()

        content = (tmp_path / "out.jsonl").read_text()
        assert "[REDACTED" in content

    def test_valid_on_text_violation_values(self, tmp_path: Path):
        """Both valid on_text_violation values should work."""
        from gradience.vnext.telemetry import TelemetryWriter

        for mode in ("redact", "raise"):
            w = TelemetryWriter(tmp_path / f"out_{mode}.jsonl", on_text_violation=mode)
            w.close()


# ===========================================================================
# telemetry reader error paths
# ===========================================================================


class TestTelemetryReaderErrors:
    """Error paths in gradience.vnext.telemetry_reader.TelemetryReader."""

    def test_strict_schema_mismatch(self, tmp_path: Path):
        """Schema mismatch in strict mode should raise TelemetrySchemaError."""
        from gradience.exceptions import TelemetrySchemaError
        from gradience.vnext.telemetry_reader import TelemetryReader

        jsonl = tmp_path / "bad_schema.jsonl"
        jsonl.write_text(json.dumps({"schema": "wrong/v999", "ts": 1.0, "run_id": "x", "event": "run_start"}) + "\n")

        reader = TelemetryReader(jsonl, strict_schema=True)
        with pytest.raises(TelemetrySchemaError, match="schema mismatch"):
            list(reader.iter_events())

    def test_strict_missing_required_key(self, tmp_path: Path):
        """Missing required envelope key in strict mode should raise TelemetryFormatError."""
        from gradience.exceptions import TelemetryFormatError
        from gradience.vnext.telemetry_reader import TelemetryReader

        jsonl = tmp_path / "bad_format.jsonl"
        # Valid schema but missing 'run_id'
        jsonl.write_text(json.dumps({"schema": "gradience.vnext.telemetry/v1", "ts": 1.0, "event": "run_start"}) + "\n")

        reader = TelemetryReader(jsonl, strict_schema=True)
        with pytest.raises(TelemetryFormatError, match="missing required key"):
            list(reader.iter_events())

    def test_non_strict_schema_mismatch_records_issue(self, tmp_path: Path):
        """Schema mismatch in non-strict mode should record issue, not raise."""
        from gradience.vnext.telemetry_reader import TelemetryReader

        jsonl = tmp_path / "bad_schema.jsonl"
        jsonl.write_text(json.dumps({"schema": "wrong/v999", "ts": 1.0, "run_id": "x", "event": "run_start"}) + "\n")

        reader = TelemetryReader(jsonl, strict_schema=False)
        events = list(reader.iter_events())
        assert len(events) == 0  # skipped
        assert len(reader.issues) == 1
        assert "schema mismatch" in reader.issues[0].message

    def test_non_strict_missing_key_records_issue(self, tmp_path: Path):
        """Missing key in non-strict mode should record issue, not raise."""
        from gradience.vnext.telemetry_reader import TelemetryReader

        jsonl = tmp_path / "bad_format.jsonl"
        jsonl.write_text(json.dumps({"schema": "gradience.vnext.telemetry/v1", "ts": 1.0, "event": "run_start"}) + "\n")

        reader = TelemetryReader(jsonl, strict_schema=False)
        events = list(reader.iter_events())
        assert len(events) == 0  # skipped
        assert len(reader.issues) == 1
        assert "missing required key" in reader.issues[0].message

    def test_malformed_json_records_issue(self, tmp_path: Path):
        """Malformed JSON should record issue in non-strict mode."""
        from gradience.vnext.telemetry_reader import TelemetryReader

        jsonl = tmp_path / "malformed.jsonl"
        jsonl.write_text("not valid json\n")

        reader = TelemetryReader(jsonl, strict_schema=False)
        events = list(reader.iter_events())
        assert len(events) == 0
        assert len(reader.issues) >= 1


# ===========================================================================
# config_schema error paths (verify exception types match hierarchy)
# ===========================================================================


class TestConfigSchemaExceptionTypes:
    """Verify config validation raises ValueError (backward compatible)."""

    def test_missing_model_raises_valueerror(self):
        from gradience.bench.config_schema import validate_config

        with pytest.raises(ValueError, match="model"):
            validate_config({"task": {"dataset": "sst2"}, "lora": {}})

    def test_missing_task_raises_valueerror(self):
        from gradience.bench.config_schema import validate_config

        with pytest.raises(ValueError, match="task"):
            validate_config({"model": {"name": "bert"}, "lora": {}})

    def test_empty_config_raises_valueerror(self):
        from gradience.bench.config_schema import validate_config

        with pytest.raises(ValueError):
            validate_config({})
