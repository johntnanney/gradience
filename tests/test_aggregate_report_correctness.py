"""
Unit tests for aggregate report correctness.

Tests the 4 critical fixes for validation evidence:
1. Seed identity propagation
2. Validation classification for multi-seed
3. Verdict logic for failed seeds
4. Chosen rank serialization
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


class TestAggregateReportCorrectness(unittest.TestCase):
    """Test aggregate report generation with critical bug fixes."""

    def setUp(self):
        """Set up test fixtures for 3-seed aggregation."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create 3 realistic bench.json fixtures based on actual v0.9.0 data
        self.seed_reports = []

        for i, seed in enumerate([42, 123, 456]):
            report = {
                "bench_version": "0.1",
                "timestamp": f"2026-02-06T02:3{3 + i}:57.998014",
                "git_commit": "66f03ce7d416f90a63c3ff28a26b9e2c09226db4",
                "model": "mistralai/Mistral-7B-v0.1",
                "task": "gsm8k/main",
                "env": {
                    "python_version": "3.11.10",
                    "torch_version": "2.10.0+cu128",
                    "cuda_available": True,
                    "seed": seed,
                    "validation_classification": {
                        "level": "screening",
                        "rationale": "Single seed, 1000 steps (no variance estimation)",
                        "is_multiseed": False,
                        "n_seeds": 1,
                        "max_steps": 1000,
                    },
                },
                "probe_quality_gate": {
                    "metric_key": "eval_exact_match",
                    "metric_value": 0.336,
                    "min_value": 0.15,
                    "passed": True,
                },
                "probe": {
                    "rank": 64,
                    "params": 167772160,
                    "accuracy": 0.336 + i * 0.001,  # Small variance: 0.336, 0.337, 0.338
                },
                "compressed": {
                    "energy_p90": {
                        "rank": 48,
                        "params": 125829120,
                        "accuracy": 0.368 + i * 0.001,  # 0.368, 0.369, 0.370
                        "delta_vs_probe": 0.032 + i * 0.001,
                        "param_reduction": 0.25,
                        "verdict": "PASS",
                    },
                    "uniform_r32": {
                        "rank": 32,
                        "params": 83886080,
                        "accuracy": 0.348 + i * 0.001,  # 0.348, 0.349, 0.350
                        "delta_vs_probe": 0.012 + i * 0.001,
                        "param_reduction": 0.5,
                        "verdict": "PASS",
                    },
                },
                "summary": {"recommendations_validated": "2/2", "best_compression": "uniform_r32"},
                "config_metadata": {
                    "primary_metric_key": "eval_exact_match",
                    "embedded_config": {"train": {"max_steps": 1000, "seed": seed}},
                },
            }
            self.seed_reports.append(report)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_seed_identity_propagation_fix(self):
        """Test that aggregate seeds are correctly extracted (Bug #1 fix)."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(self.seed_reports, config, self.temp_dir)

        # BEFORE BUG FIX: would be ["unknown", "unknown", "unknown"]
        # AFTER BUG FIX: should be actual seed values
        assert "seeds" in result
        assert result["seeds"] == [42, 123, 456], f"Expected [42, 123, 456], got {result['seeds']}"
        assert result["n_seeds"] == 3

    def test_validation_classification_multiseed_fix(self):
        """Test that validation classification correctly reflects multi-seed (Bug #2 fix)."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(self.seed_reports, config, self.temp_dir)

        # Check that env.validation_classification is corrected for multi-seed
        validation_classification = result["env"]["validation_classification"]

        # BEFORE BUG FIX: would be {"is_multiseed": false, "n_seeds": 1, "rationale": "Single seed..."}
        # AFTER BUG FIX: should reflect multi-seed reality
        assert validation_classification["is_multiseed"] == True, "Should be marked as multi-seed"
        assert validation_classification["n_seeds"] == 3, (
            f"Should show 3 seeds, got {validation_classification['n_seeds']}"
        )
        assert "certifiable" in validation_classification["level"], (
            f"Expected certifiable level with 3 seeds x 1000 steps, got {validation_classification['level']}"
        )
        assert "statistical rigor" in validation_classification["rationale"], (
            "Rationale should mention statistical rigor"
        )

    def test_verdict_logic_stringent_threshold_fix(self):
        """Test that verdict logic uses 80% threshold for scientific validity (Bug #3 fix)."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        # Test case 1: All seeds pass (100% pass rate) -> should PASS
        all_pass_reports = []
        for report in self.seed_reports:
            modified_report = json.loads(json.dumps(report))  # Deep copy
            modified_report["compressed"]["energy_p90"]["verdict"] = "PASS"
            modified_report["compressed"]["uniform_r32"]["verdict"] = "PASS"
            all_pass_reports.append(modified_report)

        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(all_pass_reports, config, self.temp_dir)

        # All variants should pass with 100% pass rate
        assert result["compressed"]["energy_p90"]["verdict"] == "PASS"
        assert result["compressed"]["uniform_r32"]["verdict"] == "PASS"
        assert result["compressed"]["energy_p90"]["pass_rate"] == 1.0
        assert result["compressed"]["uniform_r32"]["pass_rate"] == 1.0

        # Test case 2: 2/3 seeds pass (66.7% pass rate) -> should FAIL with 80% threshold
        mixed_reports = []
        for i, report in enumerate(self.seed_reports):
            modified_report = json.loads(json.dumps(report))  # Deep copy
            # First 2 seeds pass, third fails
            verdict = "PASS" if i < 2 else "FAIL"
            modified_report["compressed"]["energy_p90"]["verdict"] = verdict
            modified_report["compressed"]["uniform_r32"]["verdict"] = verdict
            mixed_reports.append(modified_report)

        result_mixed = create_multi_seed_aggregated_report(mixed_reports, config, self.temp_dir)

        # BEFORE BUG FIX: 66.7% would pass with 50% threshold
        # AFTER BUG FIX: 66.7% should fail with 80% threshold
        assert result_mixed["compressed"]["energy_p90"]["verdict"] == "FAIL", (
            "Should fail with 66.7% pass rate (below 80% threshold)"
        )
        assert result_mixed["compressed"]["uniform_r32"]["verdict"] == "FAIL", (
            "Should fail with 66.7% pass rate (below 80% threshold)"
        )
        assert abs(result_mixed["compressed"]["energy_p90"]["pass_rate"] - 0.667) < 0.01, "Pass rate should be ~66.7%"

    def test_chosen_rank_serialization_fix(self):
        """Test that chosen_rank uses actual rank values (Bug #4 fix)."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(self.seed_reports, config, self.temp_dir)

        # Check detailed_results.candidates for correct rank values
        detailed_results = result["detailed_results"]
        candidates = detailed_results["candidates"]

        # BEFORE BUG FIX: chosen_rank would be param_count // 1000000 = nonsense like 125, 83
        # AFTER BUG FIX: chosen_rank should be actual rank values
        if "energy_p90" in candidates:
            energy_rank = candidates["energy_p90"]["chosen_rank"]
            assert energy_rank == 48, f"energy_p90 should have rank 48, got {energy_rank}"

        if "uniform_r32" in candidates:
            uniform_rank = candidates["uniform_r32"]["chosen_rank"]
            assert uniform_rank == 32, f"uniform_r32 should have rank 32, got {uniform_rank}"

    def test_worst_case_delta_calculation(self):
        """Test that worst_case_delta is calculated correctly across seeds."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        # Modify reports to have known delta values for testing
        test_reports = []
        expected_deltas = [0.012, 0.013, 0.014]  # Known worst case should be 0.012

        for i, report in enumerate(self.seed_reports):
            modified_report = json.loads(json.dumps(report))  # Deep copy
            modified_report["compressed"]["uniform_r32"]["delta_vs_probe"] = expected_deltas[i]
            test_reports.append(modified_report)

        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(test_reports, config, self.temp_dir)

        # Check worst_case_delta in detailed results
        candidates = result["detailed_results"]["candidates"]
        if "uniform_r32" in candidates:
            worst_delta = candidates["uniform_r32"]["worst_case_delta"]
            expected_worst = min(expected_deltas)
            assert worst_delta == expected_worst, f"Expected worst_case_delta={expected_worst}, got {worst_delta}"

    def test_overall_verdict_policy_compliance(self):
        """Test that overall verdicts match the intended 80% threshold policy."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        config = {"bench_version": "0.1"}

        # Test various pass rate scenarios
        test_cases = [
            ([True, True, True], 1.0, "PASS"),  # 100% -> PASS
            ([True, True, False], 0.667, "FAIL"),  # 66.7% -> FAIL (below 80%)
            ([True, False, False], 0.333, "FAIL"),  # 33.3% -> FAIL
            ([False, False, False], 0.0, "FAIL"),  # 0% -> FAIL
        ]

        for passes, expected_rate, expected_verdict in test_cases:
            test_reports = []
            for i, should_pass in enumerate(passes):
                modified_report = json.loads(json.dumps(self.seed_reports[i]))
                verdict = "PASS" if should_pass else "FAIL"
                modified_report["compressed"]["energy_p90"]["verdict"] = verdict
                test_reports.append(modified_report)

            result = create_multi_seed_aggregated_report(test_reports, config, self.temp_dir)

            actual_rate = result["compressed"]["energy_p90"]["pass_rate"]
            actual_verdict = result["compressed"]["energy_p90"]["verdict"]

            assert abs(actual_rate - expected_rate) < 0.01, f"Expected pass rate {expected_rate}, got {actual_rate}"
            assert actual_verdict == expected_verdict, (
                f"Expected verdict {expected_verdict}, got {actual_verdict} for {expected_rate * 100}% pass rate"
            )

    def test_two_tier_selection_system(self):
        """Test the two-tier selection system (safe vs aggressive variants)."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        # Create test case with mixed pass rates
        mixed_reports = []
        for i, seed in enumerate([42, 123, 456]):
            report = json.loads(json.dumps(self.seed_reports[i]))  # Deep copy

            # energy_p90: passes all 3 seeds (100% rate) -> safe
            report["compressed"]["energy_p90"]["verdict"] = "PASS"

            # uniform_r32: passes 2/3 seeds (67% rate) -> aggressive
            report["compressed"]["uniform_r32"]["verdict"] = "PASS" if i < 2 else "FAIL"

            mixed_reports.append(report)

        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(mixed_reports, config, self.temp_dir)

        summary = result["summary"]

        # Should have both tiers available
        assert summary["best_safe_variant"] is not None, "Should have a safe variant"
        assert summary["best_aggressive_variant"] is not None, "Should have an aggressive variant"

        # Safe variant should be energy_p90 (100% pass rate)
        safe_variant = summary["best_safe_variant"]
        assert safe_variant["variant"] == "energy_p90"
        assert safe_variant["pass_rate"] == 1.0
        assert safe_variant["confidence_level"] == "high"
        assert "recommended for production" in safe_variant["rationale"]

        # Aggressive variant should be uniform_r32 (67% pass rate)
        aggressive_variant = summary["best_aggressive_variant"]
        assert aggressive_variant["variant"] == "uniform_r32"
        assert abs(aggressive_variant["pass_rate"] - 0.667) < 0.01
        assert aggressive_variant["confidence_level"] == "moderate"
        assert "2/3 seeds pass" in aggressive_variant["rationale"]
        assert "higher risk, higher reward" in aggressive_variant["rationale"]

        # Selection strategy should recommend safe
        strategy = summary["selection_strategy"]
        assert strategy["safe_available"] == True
        assert strategy["aggressive_available"] == True
        assert strategy["recommendation"] == "use_safe"

        # Legacy best_compression should prefer safe variant
        assert summary["best_compression"]["variant"] == "energy_p90"

    def test_aggressive_only_scenario(self):
        """Test scenario where only aggressive variants are available."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        # Create test case where all variants have <100% pass rate
        aggressive_only_reports = []
        for i, seed in enumerate([42, 123, 456]):
            report = json.loads(json.dumps(self.seed_reports[i]))  # Deep copy

            # energy_p90: passes 2/3 seeds (67% rate)
            report["compressed"]["energy_p90"]["verdict"] = "PASS" if i < 2 else "FAIL"

            # uniform_r32: passes 2/3 seeds (67% rate) but lower compression
            report["compressed"]["uniform_r32"]["verdict"] = "PASS" if i != 1 else "FAIL"  # Different pattern

            aggressive_only_reports.append(report)

        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(aggressive_only_reports, config, self.temp_dir)

        summary = result["summary"]

        # Should have no safe variants, only aggressive
        assert summary["best_safe_variant"] is None, "Should have no safe variants"
        assert summary["best_aggressive_variant"] is not None, "Should have aggressive variants"

        # Should recommend aggressive with caution
        strategy = summary["selection_strategy"]
        assert strategy["safe_available"] == False
        assert strategy["aggressive_available"] == True
        assert strategy["recommendation"] == "use_aggressive_with_caution"

        # Legacy best_compression should fallback to aggressive
        assert summary["best_compression"] is not None
        assert summary["best_compression"]["confidence_level"] == "moderate"

    def test_no_viable_variants_scenario(self):
        """Test scenario where no variants meet minimum thresholds."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        # Create test case where all variants fail majority of seeds
        failing_reports = []
        for i, seed in enumerate([42, 123, 456]):
            report = json.loads(json.dumps(self.seed_reports[i]))  # Deep copy

            # All variants pass only 1/3 seeds (33% rate) - below 60% threshold
            report["compressed"]["energy_p90"]["verdict"] = "PASS" if i == 0 else "FAIL"
            report["compressed"]["uniform_r32"]["verdict"] = "PASS" if i == 0 else "FAIL"

            failing_reports.append(report)

        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(failing_reports, config, self.temp_dir)

        summary = result["summary"]

        # Should have no viable variants
        assert summary["best_safe_variant"] is None, "Should have no safe variants"
        assert summary["best_aggressive_variant"] is None, "Should have no aggressive variants"

        # Strategy should indicate no viable options
        strategy = summary["selection_strategy"]
        assert strategy["safe_available"] == False
        assert strategy["aggressive_available"] == False
        assert strategy["recommendation"] == "no_viable_variants"

        # Legacy best_compression should be None
        assert summary["best_compression"] is None

    def test_integration_all_fixes_together(self):
        """Integration test verifying all 4 fixes work together."""
        from gradience.bench.protocol import create_multi_seed_aggregated_report

        config = {"bench_version": "0.1"}
        result = create_multi_seed_aggregated_report(self.seed_reports, config, self.temp_dir)

        # Fix 1: Seed identity
        assert result["seeds"] == [42, 123, 456]

        # Fix 2: Multi-seed validation classification
        assert result["env"]["validation_classification"]["is_multiseed"] == True
        assert result["env"]["validation_classification"]["n_seeds"] == 3

        # Fix 3: Verdicts use 80% threshold
        for variant_name in ["energy_p90", "uniform_r32"]:
            if variant_name in result["compressed"]:
                # All our test seeds pass, so should get PASS with 100% rate
                assert result["compressed"][variant_name]["pass_rate"] == 1.0
                assert result["compressed"][variant_name]["verdict"] == "PASS"

        # Fix 4: Chosen ranks are correct
        candidates = result["detailed_results"]["candidates"]
        if "energy_p90" in candidates:
            assert candidates["energy_p90"]["chosen_rank"] == 48
        if "uniform_r32" in candidates:
            assert candidates["uniform_r32"]["chosen_rank"] == 32

        # NEW: Two-tier selection system
        summary = result["summary"]
        assert "best_safe_variant" in summary
        assert "best_aggressive_variant" in summary
        assert "selection_strategy" in summary

        # All variants pass in our test data, so should be safe variants
        assert summary["best_safe_variant"] is not None
        assert summary["selection_strategy"]["recommendation"] == "use_safe"

        # Overall structure integrity
        assert result["aggregation_type"] == "multi_seed"
        assert result["n_seeds"] == 3
        assert "compressed" in result
        assert "detailed_results" in result


if __name__ == "__main__":
    unittest.main()
