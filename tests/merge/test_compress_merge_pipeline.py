"""
Integration test: compress-then-merge pipeline on CPU with synthetic adapters.

Exercises the full pipeline:
  1. SVD truncation (rank 8 -> 4) of both adapters
  2. Merge audit on the compressed pair
  3. Plan generation (audit_aware strategy)
  4. Merge execution producing a PEFT-compatible adapter
"""

from __future__ import annotations

import json
from pathlib import Path

from gradience.vnext.merge import merge_audit
from gradience.vnext.merge.executor import execute_merge
from gradience.vnext.merge.plan import plan_from_audit
from gradience.vnext.svd_truncate import svd_truncate_peft_dir

VALID_VERDICTS = {"safe", "redundant", "conflicting", "imbalanced"}


class TestCompressMergePipeline:
    """End-to-end compress -> audit -> plan -> merge pipeline."""

    def test_full_pipeline(self, orthogonal_pair: tuple[Path, Path], tmp_path: Path) -> None:
        dir_a, dir_b = orthogonal_pair

        # --- Step 1: Compress both adapters from rank 8 to rank 4 ---
        compressed_a = tmp_path / "compressed_a"
        compressed_b = tmp_path / "compressed_b"

        report_a = svd_truncate_peft_dir(dir_a, compressed_a, target_rank=4)
        report_b = svd_truncate_peft_dir(dir_b, compressed_b, target_rank=4)

        # Truncation assertions
        assert report_a.energy_retained > 0.5, f"Adapter A energy too low: {report_a.energy_retained}"
        assert report_b.energy_retained > 0.5, f"Adapter B energy too low: {report_b.energy_retained}"
        assert (compressed_a / "adapter_config.json").exists()
        assert (compressed_b / "adapter_config.json").exists()

        # --- Step 2: Merge audit on compressed pair ---
        audit_report = merge_audit(str(compressed_a), str(compressed_b))

        assert len(audit_report.layer_verdicts) > 0, "Expected at least one layer verdict"
        assert audit_report.aggregate.overall_verdict in VALID_VERDICTS, (
            f"Unexpected verdict: {audit_report.aggregate.overall_verdict}"
        )

        # --- Step 3: Plan from audit ---
        plan = plan_from_audit(
            "audit_aware",
            audit_report,
            str(compressed_a),
            str(compressed_b),
        )

        assert plan is not None

        # --- Step 4: Execute merge ---
        merged_dir = tmp_path / "merged_adapter"
        _result = execute_merge(plan, merged_dir)

        # Merged output assertions
        assert (merged_dir / "adapter_config.json").exists(), "Missing adapter_config.json in merged output"

        has_weights = (merged_dir / "adapter_model.safetensors").exists() or (
            merged_dir / "adapter_model.bin"
        ).exists()
        assert has_weights, "Missing weights file (.safetensors or .bin) in merged output"

        assert (merged_dir / "merge_result.json").exists(), "Missing merge_result.json"
        assert (merged_dir / "merge_plan.json").exists(), "Missing merge_plan.json"

        # Validate merged adapter config
        with open(merged_dir / "adapter_config.json") as f:
            merged_config = json.load(f)

        assert merged_config["peft_type"] == "LORA", f"Expected peft_type=LORA, got {merged_config['peft_type']}"
        assert isinstance(merged_config["r"], int) and merged_config["r"] > 0, (
            f"Expected valid r field, got {merged_config.get('r')}"
        )
