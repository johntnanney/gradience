"""Study 17 dry run: two-condition merge experiment on synthetic adapters.

Validates the bookkeeping layer that real GPU studies depend on:
- Distinct output directories per condition
- Distinct labels in the summary JSON
- File completeness in each condition directory
- Provenance divergence between conditions
- Summary JSON is ingestible with expected structure
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gradience.vnext.merge import merge_audit
from gradience.vnext.merge.executor import execute_merge
from gradience.vnext.merge.plan import plan_from_audit
from gradience.vnext.svd_truncate import svd_truncate_peft_dir

CONDITION_FILES = {"adapter_config.json", "merge_plan.json", "merge_result.json"}


@dataclass
class ConditionResult:
    """Metrics from one merge condition."""

    strategy: str
    mean_reconstruction_error: float
    max_reconstruction_error: float
    plan_id: str
    output_rank: int


@dataclass
class PairResult:
    """Aggregated results for one adapter pair across all conditions."""

    pair_id: str
    overall_verdict: str
    compatibility_score: float
    conditions: dict[str, dict] = field(default_factory=dict)


def _run_condition(
    name: str,
    strategy: str,
    report,
    dir_a: Path,
    dir_b: Path,
    cond_dir: Path,
    output_rank: int,
) -> ConditionResult:
    """Run one merge condition and return metrics."""
    plan = plan_from_audit(strategy, report, str(dir_a), str(dir_b), output_rank=output_rank)
    result = execute_merge(plan, cond_dir)
    return ConditionResult(
        strategy=strategy,
        mean_reconstruction_error=result.mean_reconstruction_error,
        max_reconstruction_error=result.max_reconstruction_error,
        plan_id=plan.plan_id,
        output_rank=output_rank,
    )


class TestStudyDryRun:
    """Validates the bookkeeping layer for a two-condition merge study."""

    def test_two_condition_study(self, orthogonal_pair: tuple[Path, Path], tmp_path: Path):
        dir_a, dir_b = orthogonal_pair
        study_dir = tmp_path / "study17_dry_run"
        pair_dir = study_dir / "pair_01_synthetic"

        # --- Audit the original pair ---
        audit_dir = pair_dir / "audit"
        report = merge_audit(str(dir_a), str(dir_b), output_dir=str(audit_dir))

        # --- Condition 1: full-rank norm-equalized ---
        cond1_dir = pair_dir / "full_rank_norm_eq"
        cond1 = _run_condition("full_rank_norm_eq", "norm_equalized", report, dir_a, dir_b, cond1_dir, output_rank=8)

        # --- Condition 2: compress to rank 4, then norm-equalized ---
        compressed_a = tmp_path / "compressed_a"
        compressed_b = tmp_path / "compressed_b"
        svd_truncate_peft_dir(dir_a, compressed_a, target_rank=4)
        svd_truncate_peft_dir(dir_b, compressed_b, target_rank=4)

        # Re-audit compressed pair
        compressed_report = merge_audit(str(compressed_a), str(compressed_b))

        cond2_dir = pair_dir / "compressed_norm_eq"
        cond2 = _run_condition(
            "compressed_norm_eq", "norm_equalized", compressed_report, compressed_a, compressed_b, cond2_dir,
            output_rank=4,
        )

        # --- Build summary JSON (mirrors study16 structure) ---
        pair_result = PairResult(
            pair_id="pair_01_synthetic",
            overall_verdict=report.aggregate.overall_verdict,
            compatibility_score=report.aggregate.compatibility_score,
            conditions={
                "full_rank_norm_eq": asdict(cond1),
                "compressed_norm_eq": asdict(cond2),
            },
        )
        summary = {
            "study_id": "study17_dry_run",
            "pair_results": [asdict(pair_result)],
        }
        summary_path = study_dir / "study17_dry_run.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # ===================================================================
        # ASSERTIONS: Validate the bookkeeping layer
        # ===================================================================

        # 1. Distinct output directories exist
        assert cond1_dir.exists() and cond2_dir.exists()
        assert cond1_dir != cond2_dir

        # 2. Audit directory populated
        assert (audit_dir / "merge_audit.json").exists()
        assert (audit_dir / "merge_audit.md").exists()
        with open(audit_dir / "merge_audit.json") as f:
            audit_json = json.load(f)
        assert "per_layer" in audit_json
        assert len(audit_json["per_layer"]) > 0

        # 3. File completeness per condition
        for cond_dir in [cond1_dir, cond2_dir]:
            for fname in CONDITION_FILES:
                assert (cond_dir / fname).exists(), f"Missing {fname} in {cond_dir.name}"
            has_weights = (
                (cond_dir / "adapter_model.safetensors").exists()
                or (cond_dir / "adapter_model.bin").exists()
            )
            assert has_weights, f"Missing weight file in {cond_dir.name}"

        # 4. Provenance divergence — different plan IDs
        assert cond1.plan_id != cond2.plan_id, "Conditions should have distinct plan IDs"

        # 5. Provenance divergence — different ranks in adapter configs
        with open(cond1_dir / "adapter_config.json") as f:
            config1 = json.load(f)
        with open(cond2_dir / "adapter_config.json") as f:
            config2 = json.load(f)
        assert config1["r"] == 8
        assert config2["r"] == 4

        # 6. Both conditions use norm_equalized strategy
        assert config1["gradience_merge"]["strategy"] == "norm_equalized"
        assert config2["gradience_merge"]["strategy"] == "norm_equalized"

        # 7. Summary JSON is ingestible
        with open(summary_path) as f:
            loaded = json.load(f)
        assert loaded["study_id"] == "study17_dry_run"
        assert len(loaded["pair_results"]) == 1
        pr = loaded["pair_results"][0]
        assert "full_rank_norm_eq" in pr["conditions"]
        assert "compressed_norm_eq" in pr["conditions"]
        assert isinstance(pr["conditions"]["full_rank_norm_eq"]["mean_reconstruction_error"], float)
        assert isinstance(pr["conditions"]["compressed_norm_eq"]["mean_reconstruction_error"], float)

        # 8. Distinct labels — condition names differ
        assert set(pr["conditions"].keys()) == {"full_rank_norm_eq", "compressed_norm_eq"}
