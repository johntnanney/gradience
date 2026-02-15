"""Tests for merge executor — end-to-end merge execution.

Uses the conftest.py orthogonal_pair fixture to create real adapter directories,
then runs a merge and validates the output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pytest
import torch

from gradience.vnext.merge import merge_audit
from gradience.vnext.merge.executor import (
    MergeResult,
    LayerMergeResult,
    execute_merge,
)
from gradience.vnext.merge.io import load_adapter
from gradience.vnext.merge.plan import plan_from_audit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_merge(
    adapter_pair: Tuple[Path, Path],
    strategy_name: str,
    output_dir: Path,
    output_rank: int = 4,
) -> MergeResult:
    """Helper: audit → plan → merge in one call."""
    dir_a, dir_b = adapter_pair
    report = merge_audit(dir_a, dir_b)
    plan = plan_from_audit(
        strategy_name, report,
        str(dir_a), str(dir_b),
        output_rank=output_rank,
        output_alpha=8.0,
    )
    return execute_merge(plan, output_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExecuteMerge:
    def test_produces_peft_directory(self, orthogonal_pair, tmp_path):
        """Merge output should contain adapter_config.json + adapter_model.safetensors."""
        output_dir = tmp_path / "merged"
        result = _run_merge(orthogonal_pair, "uniform_linear", output_dir)

        assert (output_dir / "adapter_config.json").exists()
        assert (
            (output_dir / "adapter_model.safetensors").exists()
            or (output_dir / "adapter_model.bin").exists()
        )
        assert (output_dir / "merge_plan.json").exists()
        assert (output_dir / "merge_result.json").exists()

    def test_adapter_config_valid(self, orthogonal_pair, tmp_path):
        """adapter_config.json should be valid PEFT config."""
        output_dir = tmp_path / "merged"
        _run_merge(orthogonal_pair, "uniform_linear", output_dir, output_rank=4)

        with open(output_dir / "adapter_config.json") as f:
            config = json.load(f)

        assert config["peft_type"] == "LORA"
        assert config["r"] == 4
        assert config["lora_alpha"] == 8.0
        assert isinstance(config["target_modules"], list)
        assert len(config["target_modules"]) > 0

    def test_result_has_metrics(self, orthogonal_pair, tmp_path):
        """MergeResult should have reconstruction errors and timing."""
        output_dir = tmp_path / "merged"
        result = _run_merge(orthogonal_pair, "uniform_linear", output_dir)

        assert len(result.layer_results) == 2  # q_proj + v_proj
        assert result.total_time_seconds > 0
        assert result.mean_reconstruction_error >= 0
        assert result.max_reconstruction_error >= 0

        for lr in result.layer_results:
            assert isinstance(lr, LayerMergeResult)
            assert lr.reconstruction_error >= 0
            assert lr.output_rank == 4
            assert lr.merged_frobenius > 0

    def test_result_json_written(self, orthogonal_pair, tmp_path):
        """merge_result.json should be valid and match MergeResult data."""
        output_dir = tmp_path / "merged"
        result = _run_merge(orthogonal_pair, "uniform_linear", output_dir)

        with open(output_dir / "merge_result.json") as f:
            data = json.load(f)

        assert data["schema_version"] == "gradience.merge_result/v1"
        assert data["plan_id"] == result.plan.plan_id
        assert data["n_layers"] == len(result.layer_results)
        assert data["total_time_seconds"] >= 0

    def test_merged_weights_loadable(self, orthogonal_pair, tmp_path):
        """Merged adapter should be loadable via load_adapter."""
        output_dir = tmp_path / "merged"
        _run_merge(orthogonal_pair, "uniform_linear", output_dir, output_rank=4)

        # Should load without errors
        info = load_adapter(output_dir)
        assert info.rank == 4
        assert info.alpha == 8.0
        assert len(info.lora_pairs) == 2

    def test_merged_tensor_shapes(self, orthogonal_pair, tmp_path):
        """Output tensors should have correct shapes for target_rank."""
        output_dir = tmp_path / "merged"
        target_rank = 4
        _run_merge(
            orthogonal_pair, "uniform_linear", output_dir,
            output_rank=target_rank,
        )

        info = load_adapter(output_dir)
        for prefix, a_key, b_key in info.lora_pairs:
            A = info.state_dict[a_key]
            B = info.state_dict[b_key]
            assert A.shape[0] == target_rank, f"A should have rank rows: {A.shape}"
            assert B.shape[1] == target_rank, f"B should have rank cols: {B.shape}"

    def test_all_strategies_work(self, orthogonal_pair, tmp_path):
        """All 3 merge strategies should produce valid output."""
        for strategy in ["uniform_linear", "audit_aware", "overlap_ties"]:
            output_dir = tmp_path / f"merged_{strategy}"
            result = _run_merge(orthogonal_pair, strategy, output_dir)

            assert (output_dir / "adapter_config.json").exists()
            assert len(result.layer_results) == 2

    def test_result_to_dict(self, orthogonal_pair, tmp_path):
        """MergeResult.to_dict() should produce valid JSON-serializable dict."""
        output_dir = tmp_path / "merged"
        result = _run_merge(orthogonal_pair, "uniform_linear", output_dir)

        d = result.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert d["strategy_name"] == "uniform_linear"
        assert "per_layer" in d


class TestMergeWithDifferentFixtures:
    """Test merging with various adapter pair types."""

    def test_redundant_pair(self, redundant_pair, tmp_path):
        """Merging redundant adapters should work with TIES."""
        output_dir = tmp_path / "merged"
        result = _run_merge(redundant_pair, "audit_aware", output_dir)
        assert len(result.layer_results) == 2
        # Redundant pair: audit_aware should choose TIES for some layers
        assert any(lr.strategy_used == "ties" for lr in result.layer_results)

    def test_conflicting_pair(self, conflicting_pair, tmp_path):
        """Merging conflicting adapters should complete without errors."""
        output_dir = tmp_path / "merged"
        result = _run_merge(conflicting_pair, "overlap_ties", output_dir)
        assert len(result.layer_results) == 2
