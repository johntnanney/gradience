"""Tests for bench artifact credibility fixes.

Covers: seed ID extraction, policy_origin field, candidate selection trace,
audit context markdown sections.
"""

import json
import unittest
from unittest.mock import patch
from pathlib import Path

from gradience.bench.protocol import (
    create_multi_seed_aggregated_report,
    create_multi_seed_markdown_report,
)
from gradience.bench.aggregate import (
    _extract_seed_id,
    generate_markdown_report,
)


def _make_seed_report(seed, accuracy=0.85, probe_rank=64, probe_params=1024):
    """Build a minimal seed report dict for aggregation tests."""
    return {
        "bench_version": "0.1",
        "seed": seed,
        "model": "test-model",
        "task": "test/task",
        "env": {"device": "cpu"},
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
            "candidate_selection": {
                "mode": "fast",
                "max_candidates": 4,
                "total_policies_evaluated": 3,
                "after_dedup": 2,
                "final_count": 2,
                "dedup_events": [
                    {
                        "rank": 32,
                        "policies": ["energy", "knee"],
                        "kept": "energy_p90",
                        "dropped": ["knee_p90"],
                    }
                ],
                "candidates": [
                    {"name": "energy_p90", "policy_type": "energy", "suggested_r": 32, "actual_r": 32, "conservatism_score": 3.0},
                    {"name": "knee_p90", "policy_type": "knee", "suggested_r": 33, "actual_r": 32, "conservatism_score": 2.0},
                    {"name": "erank_p90", "policy_type": "erank", "suggested_r": 24, "actual_r": 24, "conservatism_score": 1.5},
                ],
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


class TestCandidateSelectionTrace(unittest.TestCase):
    """Fix 2: Selection trace flows through to aggregate and markdown."""

    def test_selection_trace_in_aggregate_config_metadata(self):
        reports = [_make_seed_report(42)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        sel = agg["config_metadata"]["candidate_selection"]
        self.assertEqual(sel["mode"], "fast")
        self.assertEqual(sel["total_policies_evaluated"], 3)
        self.assertEqual(sel["after_dedup"], 2)

    def test_markdown_contains_candidate_selection_section(self):
        reports = [_make_seed_report(42), _make_seed_report(123)]
        config = {"compression": {"acc_tolerance": 0.005}}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        md = create_multi_seed_markdown_report(agg, config, Path("/tmp"))
        self.assertIn("## Candidate Selection", md)
        self.assertIn("Policies evaluated", md)
        self.assertIn("de-duplication", md.lower())

    def test_dedup_events_in_markdown(self):
        reports = [_make_seed_report(42)]
        config = {"compression": {"acc_tolerance": 0.005}}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        md = create_multi_seed_markdown_report(agg, config, Path("/tmp"))
        self.assertIn("energy_p90", md)
        self.assertIn("r=32", md)


class TestAuditContextTrace(unittest.TestCase):
    """Fix 4: Audit summary appears in aggregate JSON and markdown."""

    def test_audit_summary_in_aggregate(self):
        reports = [_make_seed_report(42)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        self.assertIn("audit_summary", agg)
        self.assertEqual(agg["audit_summary"]["probe_rank"], 64)
        self.assertIn("energy_p90", agg["audit_summary"]["policy_suggestions"])

    def test_audit_summary_dedup_annotation(self):
        reports = [_make_seed_report(42)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        knee = agg["audit_summary"]["policy_suggestions"].get("knee_p90", {})
        self.assertIn("dedup", knee)
        self.assertIn("energy_p90", knee["dedup"])

    def test_markdown_contains_audit_context(self):
        reports = [_make_seed_report(42), _make_seed_report(123)]
        config = {"compression": {"acc_tolerance": 0.005}}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        md = create_multi_seed_markdown_report(agg, config, Path("/tmp"))
        self.assertIn("## Audit Context", md)
        self.assertIn("r=64", md)

    def test_selection_reasoning_in_audit_summary(self):
        reports = [_make_seed_report(42)]
        config = {}
        agg = create_multi_seed_aggregated_report(reports, config, Path("/tmp"))
        reasoning = agg["audit_summary"]["selection_reasoning"]
        self.assertIn("Fast", reasoning)
        self.assertIn("3", reasoning)


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
