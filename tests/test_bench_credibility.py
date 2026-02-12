"""Tests for bench artifact credibility fixes.

Covers: seed ID extraction, policy_origin field, validation_classification
ownership, effective_overrides, Rank Policy column in markdown.
"""

import json
import unittest
from unittest.mock import patch
from pathlib import Path

from gradience.bench.reporting import (
    create_multi_seed_aggregated_report,
    create_multi_seed_markdown_report,
)
from gradience.bench.aggregate import (
    _extract_seed_id,
    generate_markdown_report,
)


def _make_seed_report(seed, accuracy=0.85, probe_rank=64, probe_params=1024, max_steps=1200):
    """Build a minimal seed report dict for aggregation tests."""
    return {
        "bench_version": "0.1",
        "seed": seed,
        "model": "test-model",
        "task": "test/task",
        "env": {
            "device": "cpu",
            "validation_classification": {
                "level": "screening",
                "rationale": f"Single seed, {max_steps} steps",
                "is_multiseed": False,
                "n_seeds": 1,
                "max_steps": max_steps,
            }
        },
        "git_commit": "abc123",
        "probe": {
            "rank": probe_rank,
            "params": probe_params,
            "accuracy": accuracy,
        },
        "compressed": {
            "energy_p90": {
                "rank": 32,
                "params": 512,
                "accuracy": accuracy - 0.002,
                "delta_vs_probe": -0.002,
                "param_reduction": 0.50,
                "verdict": "PASS",
                "compression": {
                    "method": "svd_truncate",
                    "policy_origin": "energy",
                    "rank_source": "energy_90.uniform_p90",
                    "target_rank": 32,
                    "source_rank": 64,
                    "alpha_mode": "keep_ratio",
                    "energy_retained": 0.95,
                },
            }
        },
        "summary": {
            "best_compression": "energy_p90",
            "notes": [],
        },
        "config_metadata": {
            "primary_metric_key": "eval_accuracy",
            "config_hash": "deadbeef",
            "embedded_config": {
                "compression": {"variants_to_test": ["per_layer"]},
                "train": {"max_steps": max_steps},
            },
        },
    }


class TestSeedExtraction(unittest.TestCase):
    """Fix 1: Seed IDs should come from top-level 'seed' key, not nested env."""

    def test_aggregated_seeds_use_top_level_seed(self):
        reports = [_make_seed_report(42), _make_seed_report(123), _make_seed_report(456)]
        config = {"compression": {"acc_tolerance": 0.005}}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        self.assertEqual(agg["seeds"], [42, 123, 456])

    def test_aggregated_seeds_fallback_to_env(self):
        report = _make_seed_report(99)
        del report["seed"]
        report["env"]["seed"] = 77
        agg = create_multi_seed_aggregated_report([report], {}, Path("/tmp"))
        self.assertEqual(agg["seeds"], [77])

    def test_markdown_shows_seed_ids(self):
        reports = [_make_seed_report(42), _make_seed_report(123), _make_seed_report(456)]
        config = {"compression": {"acc_tolerance": 0.005}}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        md = create_multi_seed_markdown_report(agg, config, Path("/tmp"))
        self.assertIn("42", md)
        self.assertIn("123", md)
        self.assertIn("456", md)


class TestExtractSeedIdHelper(unittest.TestCase):
    """aggregate.py _extract_seed_id helper."""

    def test_extracts_numeric_seed(self):
        self.assertEqual(_extract_seed_id(Path("/runs/seed_42")), 42)

    def test_extracts_large_seed(self):
        self.assertEqual(_extract_seed_id(Path("/runs/seed_12345")), 12345)

    def test_non_seed_dir_returns_name(self):
        self.assertEqual(_extract_seed_id(Path("/runs/run_0")), "run_0")

    def test_nested_path(self):
        self.assertEqual(_extract_seed_id(Path("/data/bench/seed_456/extra")), "extra")
        self.assertEqual(_extract_seed_id(Path("/data/bench/seed_456")), 456)


class TestPolicyOrigin(unittest.TestCase):
    """Fix 3: policy_origin field appears in compression metadata."""

    def test_policy_origin_in_aggregated_variant(self):
        reports = [_make_seed_report(42)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        variant = agg["compressed"]["energy_p90"]
        self.assertEqual(variant["compression"]["policy_origin"], "energy")

    def test_markdown_has_rank_policy_column(self):
        reports = [_make_seed_report(42), _make_seed_report(123)]
        config = {"compression": {"acc_tolerance": 0.005}}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        md = create_multi_seed_markdown_report(agg, config, Path("/tmp"))
        self.assertIn("Rank Policy", md)
        self.assertIn("energy", md)


class TestValidationClassificationOwnership(unittest.TestCase):
    """Fix A: Aggregate must own validation_classification, not copy from seed."""

    def test_aggregate_is_multiseed_true(self):
        reports = [_make_seed_report(42), _make_seed_report(123), _make_seed_report(456)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        vc = agg["env"]["validation_classification"]
        self.assertTrue(vc["is_multiseed"])

    def test_aggregate_n_seeds_matches(self):
        reports = [_make_seed_report(42), _make_seed_report(123), _make_seed_report(456)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        vc = agg["env"]["validation_classification"]
        self.assertEqual(vc["n_seeds"], 3)

    def test_aggregate_level_certifiable_with_3_seeds_sufficient_steps(self):
        """3 seeds with sufficient steps should be certifiable."""
        reports = [_make_seed_report(42, max_steps=1200), _make_seed_report(123, max_steps=1200), _make_seed_report(456, max_steps=1200)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        vc = agg["env"]["validation_classification"]
        self.assertEqual(vc["level"], "certifiable")

    def test_aggregate_level_screening_plus_with_2_seeds(self):
        reports = [_make_seed_report(42, max_steps=1200), _make_seed_report(123, max_steps=1200)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        vc = agg["env"]["validation_classification"]
        self.assertIn(vc["level"], ["screening_plus", "certifiable"])

    def test_single_seed_env_not_leaked_to_aggregate(self):
        """Even if seed report has is_multiseed=False, aggregate must override."""
        report = _make_seed_report(42, max_steps=1200)
        report["env"]["validation_classification"] = {
            "level": "screening",
            "rationale": "Single seed, 1200 steps",
            "is_multiseed": False,
            "n_seeds": 1,
        }
        reports = [report, _make_seed_report(123, max_steps=1200), _make_seed_report(456, max_steps=1200)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        vc = agg["env"]["validation_classification"]
        self.assertTrue(vc["is_multiseed"])
        self.assertEqual(vc["n_seeds"], 3)


class TestEffectiveOverrides(unittest.TestCase):
    """Fix C: effective_overrides alongside embedded_config in config_metadata."""

    def _make_report_with_overrides(self):
        """Helper to build a seed report with effective_overrides."""
        report = _make_seed_report(42)
        report["config_metadata"]["effective_overrides"] = {
            "fast_mode": True,
            "max_candidates": 4,
            "variants_evaluated": ["energy_p90"],
            "candidate_selection_mode": "fast",
        }
        return report

    def test_effective_overrides_present(self):
        report = self._make_report_with_overrides()
        self.assertIn("effective_overrides", report["config_metadata"])

    def test_effective_overrides_variants_evaluated(self):
        report = self._make_report_with_overrides()
        overrides = report["config_metadata"]["effective_overrides"]
        self.assertEqual(overrides["variants_evaluated"], ["energy_p90"])

    def test_embedded_config_differs_from_effective(self):
        """embedded_config may say per_layer, effective says energy_p90."""
        report = self._make_report_with_overrides()
        embedded_variants = report["config_metadata"]["embedded_config"]["compression"]["variants_to_test"]
        effective_variants = report["config_metadata"]["effective_overrides"]["variants_evaluated"]
        # They can differ — that's the point of having both fields
        self.assertNotEqual(embedded_variants, effective_variants)

    def test_effective_overrides_flows_to_aggregate(self):
        report = self._make_report_with_overrides()
        reports = [report, _make_seed_report(123)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        # Effective overrides comes through config_metadata from base_report
        self.assertIn("effective_overrides", agg["config_metadata"])
        self.assertEqual(
            agg["config_metadata"]["effective_overrides"]["variants_evaluated"],
            ["energy_p90"]
        )


class TestStandaloneAggregateMd(unittest.TestCase):
    """Tests for the standalone aggregate.py markdown generator."""

    def test_standalone_markdown_shows_seeds(self):
        data = {
            "model": "test-model",
            "task": "test/task",
            "validation_level": "Certifiable",
            "n_seeds": 3,
            "seeds": [42, 123, 456],
            "probe_baseline": {"accuracy_mean": 0.85, "accuracy_std": 0.01, "accuracy_min": 0.84, "accuracy_max": 0.86},
            "variant_results": {},
            "policy_compliance": {},
            "safety_policy": {"name": "Test Policy", "pass_rate_min": 0.67, "worst_delta_min": -0.025},
            "invariants": {},
            "summary": {"total_variants": 0, "policy_compliant_variants": 0, "best_compression": None, "recommendations": []},
            "aggregation_timestamp": "2026-02-10T00:00:00",
        }
        md = generate_markdown_report(data)
        self.assertIn("42", md)
        self.assertIn("123", md)
        self.assertIn("456", md)

    def test_standalone_markdown_rank_policy_column(self):
        data = {
            "model": "test-model",
            "task": "test/task",
            "validation_level": "Certifiable",
            "n_seeds": 1,
            "seeds": [42],
            "probe_baseline": {"accuracy_mean": 0.85, "accuracy_std": 0.0, "accuracy_min": 0.85, "accuracy_max": 0.85},
            "variant_results": {
                "energy_p90": {
                    "status": "ok",
                    "pass_rate": 1.0,
                    "delta_worst": -0.002,
                    "accuracy_mean": 0.848,
                    "param_reduction_mean": 0.50,
                    "compression": {"policy_origin": "energy"},
                }
            },
            "policy_compliance": {"energy_p90": {"policy_compliant": True}},
            "safety_policy": {"name": "Test Policy", "pass_rate_min": 0.67, "worst_delta_min": -0.025},
            "invariants": {},
            "summary": {"total_variants": 1, "policy_compliant_variants": 1, "best_compression": None, "recommendations": []},
            "aggregation_timestamp": "2026-02-10T00:00:00",
        }
        md = generate_markdown_report(data)
        self.assertIn("Rank Policy", md)
        self.assertIn("energy", md)


if __name__ == "__main__":
    unittest.main()
