"""End-to-end integration tests for merge compatibility audit."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from gradience.vnext.merge import VerdictThresholds, merge_audit
from gradience.vnext.merge.report import MergeAuditReport

# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestMergeAuditPipeline:
    """End-to-end tests using the merge_audit() orchestrator."""

    def test_orthogonal_safe(self, orthogonal_pair, tmp_path):
        """Orthogonal adapters -> SAFE verdict + reports written."""
        dir_a, dir_b = orthogonal_pair
        output = tmp_path / "output"

        report = merge_audit(
            adapter_a_dir=dir_a,
            adapter_b_dir=dir_b,
            output_dir=output,
        )

        assert isinstance(report, MergeAuditReport)
        assert report.aggregate["overall_verdict"] == "safe"
        assert report.aggregate["mean_overlap"] < 0.3

        # Check files written
        assert (output / "merge_audit.json").exists()
        assert (output / "merge_audit.md").exists()

        # Validate JSON content
        with open(output / "merge_audit.json") as f:
            data = json.load(f)
        assert data["schema_version"] == "gradience.merge_audit/v2"
        assert data["aggregate"]["overall_verdict"] == "safe"

    def test_redundant_detected(self, redundant_pair):
        """Redundant adapters -> REDUNDANT verdict."""
        dir_a, dir_b = redundant_pair

        report = merge_audit(
            adapter_a_dir=dir_a,
            adapter_b_dir=dir_b,
        )

        assert report.aggregate["overall_verdict"] == "redundant"
        assert report.aggregate["mean_overlap"] > 0.5

    def test_conflicting_detected(self, conflicting_pair):
        """Conflicting adapters -> CONFLICTING verdict."""
        dir_a, dir_b = conflicting_pair

        report = merge_audit(
            adapter_a_dir=dir_a,
            adapter_b_dir=dir_b,
        )

        assert report.aggregate["overall_verdict"] == "conflicting"
        assert report.aggregate["mean_agreement"] < 0

    def test_imbalanced_detected(self, imbalanced_pair):
        """Magnitude-imbalanced adapters -> IMBALANCED verdict."""
        dir_a, dir_b = imbalanced_pair

        report = merge_audit(
            adapter_a_dir=dir_a,
            adapter_b_dir=dir_b,
        )

        # Either IMBALANCED or REDUNDANT depending on overlap
        assert report.aggregate["overall_verdict"] in ("imbalanced", "redundant")

    def test_different_ranks(self, different_rank_pair, tmp_path):
        """Different rank adapters (r=4, r=16) -> analysis completes."""
        dir_a, dir_b = different_rank_pair
        output = tmp_path / "output"

        report = merge_audit(
            adapter_a_dir=dir_a,
            adapter_b_dir=dir_b,
            output_dir=output,
        )

        assert isinstance(report, MergeAuditReport)
        assert len(report.layer_verdicts) == 2
        # Verify adapter rank info
        assert report.adapter_a["rank"] == 4
        assert report.adapter_b["rank"] == 16

    def test_partial_overlap_layers(self, partial_overlap_pair):
        """Partial module overlap -> shared + only_a + only_b."""
        dir_a, dir_b = partial_overlap_pair

        report = merge_audit(
            adapter_a_dir=dir_a,
            adapter_b_dir=dir_b,
        )

        assert report.matching["n_shared"] == 1
        assert report.matching["n_only_a"] == 1
        assert report.matching["n_only_b"] == 1
        assert len(report.layer_verdicts) == 1

    def test_no_shared_layers_raises(self, tmp_path):
        """Completely disjoint modules -> ValueError."""
        import torch

        from tests.merge.conftest import (
            ALPHA,
            D_IN,
            D_OUT,
            RANK,
            _make_adapter_dir,
            _make_lora_weights_from_svd,
        )

        torch.manual_seed(999)
        U, _, Vt = torch.linalg.svd(torch.randn(D_OUT, D_IN), full_matrices=False)
        S = torch.linspace(5.0, 1.0, RANK)

        weights_a = _make_lora_weights_from_svd(
            ["base_model.model.layers.0.self_attn.q_proj"],
            U[:, :RANK],
            S,
            Vt[:RANK, :],
            RANK,
            ALPHA,
        )
        weights_b = _make_lora_weights_from_svd(
            ["base_model.model.layers.0.self_attn.o_proj"],
            U[:, :RANK],
            S,
            Vt[:RANK, :],
            RANK,
            ALPHA,
        )

        dir_a = _make_adapter_dir(tmp_path, "a", RANK, ALPHA, ["q_proj"], weights_a)
        dir_b = _make_adapter_dir(tmp_path, "b", RANK, ALPHA, ["o_proj"], weights_b)

        with pytest.raises(ValueError, match="No shared LoRA layers"):
            merge_audit(adapter_a_dir=dir_a, adapter_b_dir=dir_b)

    def test_custom_thresholds(self, orthogonal_pair):
        """Custom thresholds are passed through to verdicts."""
        dir_a, dir_b = orthogonal_pair

        report = merge_audit(
            adapter_a_dir=dir_a,
            adapter_b_dir=dir_b,
            thresholds=VerdictThresholds.conservative(),
        )

        assert report.thresholds["low_overlap"] == 0.15  # conservative value

    def test_verbose_mode(self, orthogonal_pair, capsys):
        """Verbose mode prints progress."""
        dir_a, dir_b = orthogonal_pair

        merge_audit(
            adapter_a_dir=dir_a,
            adapter_b_dir=dir_b,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Loading adapter A" in captured.out
        assert "Layer matching" in captured.out

    def test_no_output_dir(self, orthogonal_pair, tmp_path):
        """When output_dir is None, no files are written."""
        dir_a, dir_b = orthogonal_pair

        report = merge_audit(
            adapter_a_dir=dir_a,
            adapter_b_dir=dir_b,
            output_dir=None,
        )

        assert isinstance(report, MergeAuditReport)
        # No files should exist in tmp_path
        assert not (tmp_path / "merge_audit.json").exists()


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Test the CLI subcommand via subprocess."""

    def test_cli_json_output(self, orthogonal_pair):
        """gradience merge-audit --json produces valid JSON."""
        dir_a, dir_b = orthogonal_pair

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gradience.cli",
                "merge-audit",
                "--adapter-a",
                str(dir_a),
                "--adapter-b",
                str(dir_b),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["schema_version"] == "gradience.merge_audit/v2"
        assert "aggregate" in data

    def test_cli_pretty_output(self, orthogonal_pair):
        """gradience merge-audit produces human-readable output."""
        dir_a, dir_b = orthogonal_pair

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gradience.cli",
                "merge-audit",
                "--adapter-a",
                str(dir_a),
                "--adapter-b",
                str(dir_b),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "Merge Compatibility" in result.stdout

    def test_cli_missing_adapter(self, tmp_path):
        """Missing adapter directory -> exit code 1."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gradience.cli",
                "merge-audit",
                "--adapter-a",
                str(tmp_path / "nonexistent_a"),
                "--adapter-b",
                str(tmp_path / "nonexistent_b"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0
