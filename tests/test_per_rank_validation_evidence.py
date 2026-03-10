"""
Unit tests for per-rank validation evidence in markdown reports.

Tests Priority 3: explicit per-rank validation statements that match the evidence pack.
"""

import json
import tempfile
import unittest
from pathlib import Path


class TestPerRankValidationEvidence(unittest.TestCase):
    """Test per-rank validation evidence in markdown reports."""

    def setUp(self):
        """Set up test fixtures for validation evidence."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create realistic seed reports with mixed pass/fail patterns
        self.seed_reports = []

        # Scenario: r=48 passes all 3 seeds, r=32 passes 2/3 seeds
        pass_patterns = {
            "energy_p90": ["PASS", "PASS", "PASS"],  # r=48: 3/3 pass
            "uniform_r32": ["PASS", "PASS", "FAIL"],  # r=32: 2/3 pass
        }

        for i, seed in enumerate([42, 123, 456]):
            report = {
                "bench_version": "0.1",
                "timestamp": f"2026-02-06T02:3{3 + i}:57.998014",
                "model": "test-model",
                "task": "test-task",
                "env": {"seed": seed, "validation_classification": {"max_steps": 1000}},
                "probe": {"rank": 64, "params": 167772160, "accuracy": 0.336},
                "compressed": {
                    "energy_p90": {
                        "rank": 48,
                        "params": 125829120,
                        "accuracy": 0.368,
                        "delta_vs_probe": 0.032,
                        "param_reduction": 0.25,
                        "verdict": pass_patterns["energy_p90"][i],
                    },
                    "uniform_r32": {
                        "rank": 32,
                        "params": 83886080,
                        "accuracy": 0.348,
                        "delta_vs_probe": 0.012,
                        "param_reduction": 0.5,
                        "verdict": pass_patterns["uniform_r32"][i],
                    },
                },
            }
            self.seed_reports.append(report)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_per_rank_validation_evidence_statements(self):
        """Test that markdown contains explicit per-rank validation statements."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report, create_multi_seed_markdown_report

        config = {"bench_version": "0.1", "compression": {"acc_tolerance": 0.025}}

        # Create aggregated report
        agg_report = create_multi_seed_aggregated_report(self.seed_reports, config, self.temp_dir)

        # Generate markdown report
        md_content = create_multi_seed_markdown_report(agg_report, config, self.temp_dir)

        # Verify the new "Per-Rank Validation Evidence" section exists
        assert "## Per-Rank Validation Evidence" in md_content, "Missing per-rank validation evidence section"

        # Verify honest statements about r=48 (3/3 pass - validated safe)
        assert "r=48** is ✅ **validated safe**" in md_content, "Missing validated safe statement for r=48"
        assert "All 3 seeds pass ±0.025 tolerance" in md_content, "Missing detailed evidence for r=48"

        # Verify honest statements about r=32 (2/3 pass - conditionally promising)
        assert "r=32** is ⚠️  **conditionally promising**" in md_content, (
            "Missing conditionally promising statement for r=32"
        )
        assert "2/3 seeds pass; one seed violates tolerance" in md_content, "Missing detailed evidence for r=32"

        print("✅ Generated markdown with honest per-rank validation:")
        print("   - r=48 is validated safe (3/3)")
        print("   - r=32 is conditionally promising (2/3; one seed violates tolerance)")

    def test_failed_validation_statement(self):
        """Test statement for ranks that fail validation."""
        # Create scenario where r=24 fails completely (0/3 pass)
        failing_reports = []
        for i, _seed in enumerate([42, 123, 456]):
            report = json.loads(json.dumps(self.seed_reports[i]))  # Deep copy

            # Add a failing variant
            report["compressed"]["uniform_r24"] = {
                "rank": 24,
                "params": 50000000,
                "accuracy": 0.200,  # Much worse accuracy
                "delta_vs_probe": -0.136,
                "param_reduction": 0.7,
                "verdict": "FAIL",
            }
            failing_reports.append(report)

        from gradience.bench.protocol import create_multi_seed_aggregated_report, create_multi_seed_markdown_report

        config = {"bench_version": "0.1", "compression": {"acc_tolerance": 0.025}}

        # Create aggregated report and markdown
        agg_report = create_multi_seed_aggregated_report(failing_reports, config, self.temp_dir)
        md_content = create_multi_seed_markdown_report(agg_report, config, self.temp_dir)

        # Should show honest failure statement
        assert "r=24** is ❌ **failed validation**" in md_content, "Missing failed validation statement"
        assert "No seeds pass ±0.025 tolerance" in md_content, "Missing failure detail"

    def test_unreliable_rank_statement(self):
        """Test statement for ranks with poor but non-zero pass rates."""
        # Create scenario where r=16 has 1/3 pass rate
        unreliable_reports = []
        for i, _seed in enumerate([42, 123, 456]):
            report = json.loads(json.dumps(self.seed_reports[i]))  # Deep copy

            # Add an unreliable variant
            report["compressed"]["uniform_r16"] = {
                "rank": 16,
                "params": 42000000,
                "accuracy": 0.320,
                "delta_vs_probe": -0.016,
                "param_reduction": 0.75,
                "verdict": "PASS" if i == 0 else "FAIL",  # Only first seed passes
            }
            unreliable_reports.append(report)

        from gradience.bench.protocol import create_multi_seed_aggregated_report, create_multi_seed_markdown_report

        config = {"bench_version": "0.1", "compression": {"acc_tolerance": 0.025}}

        # Create aggregated report and markdown
        agg_report = create_multi_seed_aggregated_report(unreliable_reports, config, self.temp_dir)
        md_content = create_multi_seed_markdown_report(agg_report, config, self.temp_dir)

        # Should show unreliable statement
        assert "r=16** is ❌ **unreliable**" in md_content, "Missing unreliable statement"
        assert "Only 1/3 seeds pass tolerance" in md_content, "Missing unreliability detail"

    def test_evidence_based_transparency_message(self):
        """Test that the transparency message is included."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report, create_multi_seed_markdown_report

        config = {"bench_version": "0.1", "compression": {"acc_tolerance": 0.025}}

        agg_report = create_multi_seed_aggregated_report(self.seed_reports, config, self.temp_dir)
        md_content = create_multi_seed_markdown_report(agg_report, config, self.temp_dir)

        # Should include transparency message
        assert "evidence-based approach ensures complete transparency" in md_content, "Missing transparency message"
        assert "which compression levels can be trusted across independent random seeds" in md_content, (
            "Missing trust statement"
        )

    def test_scientific_honesty_integration(self):
        """Test that per-rank evidence integrates well with two-tier selection."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report, create_multi_seed_markdown_report

        config = {"bench_version": "0.1", "compression": {"acc_tolerance": 0.025}}

        agg_report = create_multi_seed_aggregated_report(self.seed_reports, config, self.temp_dir)
        md_content = create_multi_seed_markdown_report(agg_report, config, self.temp_dir)

        # Should have both sections in logical order
        evidence_pos = md_content.find("## Per-Rank Validation Evidence")
        tier_pos = md_content.find("## Two-Tier Selection System")

        assert evidence_pos > 0, "Missing per-rank evidence section"
        assert tier_pos > 0, "Missing two-tier selection section"
        assert evidence_pos < tier_pos, "Per-rank evidence should come before two-tier selection"

        # The evidence should inform the two-tier recommendations
        assert "validated safe" in md_content, "Should show validated safe variants"
        assert "conditionally promising" in md_content, "Should show conditionally promising variants"


if __name__ == "__main__":
    unittest.main()
