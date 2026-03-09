"""
Integration tests for the compression pipeline.

Tests the full compression config generation pipeline:
- _resolve_policy_rank_source(): dotted-path policy resolution
- _create_shuffled_rank_pattern(): deterministic shuffle control
- generate_svd_variant_config(): SVD truncation config builder
- get_rank_source_from_config(): rank source extraction/inference
- generate_compression_configs(): end-to-end pipeline integration
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from conftest import make_audit_data, make_config, write_audit_file

from gradience.bench.compression import (
    _create_shuffled_rank_pattern,
    _resolve_policy_rank_source,
    generate_compression_configs,
    generate_svd_variant_config,
    get_rank_source_from_config,
)
from gradience.bench.constants import (
    CONSERVATISM_SCORES,
    DEFAULT_POST_TUNE_LR_SCALE,
    DEFAULT_POST_TUNE_STEPS,
)
from gradience.bench.decision_trace import DecisionTrace

# ===================================================================
# Class 1: _resolve_policy_rank_source()
# ===================================================================


class TestResolvePolicyRankSource(unittest.TestCase):
    """Tests for _resolve_policy_rank_source() — dotted-path policy resolution."""

    def setUp(self):
        self.audit_data = make_audit_data()

    def test_valid_dotted_path_resolves(self):
        result = _resolve_policy_rank_source(self.audit_data, "audit.rank_suggestions.knee.uniform_p90")
        assert result == 22.0

    def test_all_policies_resolve(self):
        expected = {
            "audit.rank_suggestions.energy_90.uniform_p90": 20.0,
            "audit.rank_suggestions.energy_90.uniform_median": 8.0,
            "audit.rank_suggestions.knee.uniform_p90": 22.0,
            "audit.rank_suggestions.knee.uniform_median": 10.0,
            "audit.rank_suggestions.erank.uniform_p90": 16.0,
            "audit.rank_suggestions.erank.uniform_median": 6.0,
            "audit.rank_suggestions.oht.uniform_p90": 18.0,
        }
        for path, expected_val in expected.items():
            result = _resolve_policy_rank_source(self.audit_data, path)
            assert result == expected_val, f"{path}: expected {expected_val}, got {result}"

    def test_malformed_path_returns_none(self):
        malformed = [
            "audit.rank_suggestions.knee",  # 3 parts
            "audit.rank_suggestions",  # 2 parts
            "wrong.rank_suggestions.knee.uniform_p90",  # wrong prefix
            "audit.wrong_field.knee.uniform_p90",  # wrong second part
            "",  # empty
            "single",  # 1 part
            "a.b.c.d.e",  # 5 parts
        ]
        for path in malformed:
            result = _resolve_policy_rank_source(self.audit_data, path)
            assert result is None, f"Expected None for malformed path '{path}', got {result}"

    def test_missing_policy_returns_none(self):
        result = _resolve_policy_rank_source(self.audit_data, "audit.rank_suggestions.nonexistent.uniform_p90")
        assert result is None

    def test_missing_statistic_returns_none(self):
        # Statistic that doesn't exist under a valid policy
        result = _resolve_policy_rank_source(self.audit_data, "audit.rank_suggestions.knee.nonexistent_stat")
        assert result is None

        # Empty policy_global_suggestions
        data = make_audit_data(policy_global_suggestions={})
        result = _resolve_policy_rank_source(data, "audit.rank_suggestions.knee.uniform_p90")
        assert result is None

        # Missing policy_global_suggestions key entirely
        data = make_audit_data()
        del data["policy_global_suggestions"]
        result = _resolve_policy_rank_source(data, "audit.rank_suggestions.knee.uniform_p90")
        assert result is None


# ===================================================================
# Class 2: _create_shuffled_rank_pattern()
# ===================================================================


class TestCreateShuffledRankPattern(unittest.TestCase):
    """Tests for _create_shuffled_rank_pattern() — deterministic shuffle control."""

    def _make_pattern(self, n=8):
        """Return a rank pattern with *n* modules and distinct values."""
        return {f"layer.{i}.proj": (i + 1) * 2 for i in range(n)}

    def test_preserves_values(self):
        original = self._make_pattern()
        result = _create_shuffled_rank_pattern(original, seed=42)
        assert sorted(result.values()) == sorted(original.values())

    def test_preserves_keys(self):
        original = self._make_pattern()
        result = _create_shuffled_rank_pattern(original, seed=42)
        assert set(result.keys()) == set(original.keys())

    def test_deterministic_with_same_seed(self):
        original = self._make_pattern()
        r1 = _create_shuffled_rank_pattern(original, seed=99)
        r2 = _create_shuffled_rank_pattern(original, seed=99)
        assert r1 == r2

    def test_different_seeds_differ(self):
        original = self._make_pattern(n=12)  # more modules to minimise coincidence
        r1 = _create_shuffled_rank_pattern(original, seed=1)
        r2 = _create_shuffled_rank_pattern(original, seed=2)
        # With 12! = 479M permutations, collision probability is negligible
        assert r1 != r2, "Different seeds should produce different shuffles"


# ===================================================================
# Class 3: generate_svd_variant_config()
# ===================================================================


class TestGenerateSvdVariantConfig(unittest.TestCase):
    """Tests for generate_svd_variant_config() — SVD truncation config builder."""

    def setUp(self):
        self.audit_data = make_audit_data()
        self.allowed_ranks = [1, 2, 4, 8, 16, 32]
        self.probe_rank = 32
        self.lora_config = {
            "probe_r": 32,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj"],
        }

    def test_direct_int_rank_source(self):
        vdef = {"name": "svd_r8", "rank_source": 8}
        result = generate_svd_variant_config(
            vdef,
            self.audit_data,
            self.probe_rank,
            self.lora_config,
            self.allowed_ranks,
        )
        assert result["status"] == "ready"
        assert result["actual_r"] == 8
        assert result["suggested_r"] == 8
        assert result["config"]["probe_r"] == 8
        assert result["config"]["alpha"] == 8
        assert result["compression_method"] == "svd_truncation"
        assert result["source_rank"] == 32

    def test_audit_global_median_source(self):
        vdef = {"name": "svd_median", "rank_source": "audit_global_median"}
        result = generate_svd_variant_config(
            vdef,
            self.audit_data,
            self.probe_rank,
            self.lora_config,
            self.allowed_ranks,
        )
        # suggested_r_global_median = 12 → rounds to 8 in [1,2,4,8,16,32]
        # (equidistant from 8 and 16; min() picks 8 first)
        assert result["status"] == "ready"
        assert result["suggested_r"] == 12
        assert result["actual_r"] == 8

    def test_policy_rank_source(self):
        vdef = {"name": "svd_knee", "rank_source": "audit.rank_suggestions.knee.uniform_p90"}
        result = generate_svd_variant_config(
            vdef,
            self.audit_data,
            self.probe_rank,
            self.lora_config,
            self.allowed_ranks,
        )
        # knee.uniform_p90 = 22 → rounds to 16 in [1,2,4,8,16,32]
        assert result["status"] == "ready"
        assert result["suggested_r"] == 22.0
        assert result["actual_r"] == 16

    def test_rank_ge_probe_skipped(self):
        vdef = {"name": "svd_noop", "rank_source": 32}
        result = generate_svd_variant_config(
            vdef,
            self.audit_data,
            self.probe_rank,
            self.lora_config,
            self.allowed_ranks,
        )
        assert result["status"] == "skipped"
        assert result["config"] is None
        assert "no compression" in result["reason"]

    def test_post_tune_config(self):
        # Explicit steps/lr_scale
        vdef = {
            "name": "svd_tune",
            "rank_source": 8,
            "post_tune": {"enabled": True, "steps": 200, "lr_scale": 0.05},
        }
        result = generate_svd_variant_config(
            vdef,
            self.audit_data,
            self.probe_rank,
            self.lora_config,
            self.allowed_ranks,
        )
        assert result["status"] == "ready"
        assert result["post_tune"]["enabled"] is True
        assert result["post_tune"]["steps"] == 200
        assert result["post_tune"]["lr_scale"] == 0.05

        # Defaults from constants when omitted
        vdef2 = {
            "name": "svd_tune_defaults",
            "rank_source": 8,
            "post_tune": {"enabled": True},
        }
        result2 = generate_svd_variant_config(
            vdef2,
            self.audit_data,
            self.probe_rank,
            self.lora_config,
            self.allowed_ranks,
        )
        assert result2["post_tune"]["steps"] == DEFAULT_POST_TUNE_STEPS
        assert result2["post_tune"]["lr_scale"] == DEFAULT_POST_TUNE_LR_SCALE


# ===================================================================
# Class 4: get_rank_source_from_config()
# ===================================================================


class TestGetRankSourceFromConfig(unittest.TestCase):
    """Tests for get_rank_source_from_config() — rank source extraction."""

    def test_explicit_rank_source(self):
        assert get_rank_source_from_config({"rank_source": "audit_global_median"}) == "audit_global_median"
        assert get_rank_source_from_config({"rank_source": 8}) == "8"

    def test_legacy_inference(self):
        # "median" keyword in reason
        result = get_rank_source_from_config({"reason": "SVD truncation using audit_global_median", "actual_r": 12})
        assert result == "audit_global_median"

        # "p90" keyword in reason
        result = get_rank_source_from_config({"reason": "SVD truncation using p90 estimate", "actual_r": 24})
        assert result == "audit_global_p90"

    def test_fallback_to_actual_r(self):
        result = get_rank_source_from_config({"actual_r": 8, "reason": "custom variant"})
        assert result == "8"

        # Empty dict
        result = get_rank_source_from_config({})
        assert result == "unknown"


# ===================================================================
# Class 5: generate_compression_configs() — full pipeline integration
# ===================================================================


class TestGenerateCompressionConfigsPipeline(unittest.TestCase):
    """Integration tests for the full compression config generation pipeline."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.probe_dir = self.temp_dir / "probe"
        self.audit_data = make_audit_data()
        write_audit_file(self.probe_dir, self.audit_data)
        self.config = make_config()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_fast_mode_returns_policy_candidates(self):
        configs, trace = generate_compression_configs(
            self.probe_dir,
            self.config,
            fast_mode=True,
            max_candidates=4,
        )
        # Returns the right types
        assert isinstance(configs, dict)
        assert isinstance(trace, DecisionTrace)

        # All configs should be ready with valid policy types
        valid_policy_types = {
            "energy",
            "knee",
            "erank",
            "oht",
            "legacy",
            "per_layer",
            "second_rung_tier_a",
            "second_rung_tier_b",
        }
        for name, cfg in configs.items():
            assert cfg["status"] == "ready", f"{name} status is {cfg['status']}"
            assert cfg.get("policy_type") in valid_policy_types, (
                f"{name} has unexpected policy_type: {cfg.get('policy_type')}"
            )

        # Respect max_candidates
        assert len(configs) <= 4 + 2  # +2 for possible second-rung candidates

    def test_full_mode_includes_additional_policies(self):
        cfg = make_config(
            compression={
                "allowed_ranks": [1, 2, 4, 8, 16, 32],
                "fast_mode": False,
                "max_candidates": 10,
            }
        )
        configs, _ = generate_compression_configs(
            self.probe_dir,
            cfg,
            fast_mode=False,
            max_candidates=10,
        )
        policy_types = {c.get("policy_type") for c in configs.values()}
        # Full mode should include legacy or oht candidates (if audit data has them)
        assert policy_types & {"legacy", "oht"}, (
            f"Full mode should include legacy/oht, got policy_types: {policy_types}"
        )

    def test_decision_trace_populated(self):
        # Low utilization triggers second-rung rules
        audit = make_audit_data()
        audit["summary"]["utilization_mean"] = 0.2
        audit["summary"]["stable_rank_mean"] = 6.0
        write_audit_file(self.probe_dir, audit)

        _, trace = generate_compression_configs(
            self.probe_dir,
            self.config,
            fast_mode=True,
        )
        assert trace.probe_rank == 32
        assert "utilization_mean" in trace.audit_metrics
        assert "stable_rank_mean" in trace.audit_metrics
        # At least one rule should have been considered or fired
        total_rules = len(trace.rules_fired) + len(trace.rules_considered)
        assert total_rules > 0, "Decision trace should contain rules"

    def test_per_layer_included_when_different(self):
        # per_layer avg rank = (32+8+16+4)/4 = 15 → rounds to 16
        # We need the uniform candidates NOT to include r=16 for per_layer to be added.
        # With default fixture: energy_p90=20→16, knee_p90=22→16, erank_p90=16
        # All round to 16, so per_layer at r=16 won't be "different".
        # Adjust audit so uniform candidates use r=8 instead.
        audit = make_audit_data()
        audit["policy_global_suggestions"] = {
            "energy_90": {"uniform_p90": 8},
            "knee": {"uniform_p90": 6},
            "erank": {"uniform_p90": 4},
        }
        # per_layer average = (32+8+16+4)/4 = 15 → rounds to 16
        # Uniform candidates will use r=8, r=4 (6→4 or 8) — distinct from r=16
        write_audit_file(self.probe_dir, audit)

        cfg = make_config(
            compression={
                "allowed_ranks": [1, 2, 4, 8, 16, 32],
                "fast_mode": True,
                "max_candidates": 8,  # room for per_layer
            }
        )
        configs, _ = generate_compression_configs(
            self.probe_dir,
            cfg,
            fast_mode=True,
            max_candidates=8,
        )
        # per_layer should be present since its avg rank (16) differs from uniform ranks
        assert "per_layer" in configs, f"per_layer should be included, got: {list(configs.keys())}"
        assert configs["per_layer"]["rank_pattern"], "per_layer should have non-empty rank_pattern"

    def test_per_layer_shuffled_not_in_final_output(self):
        # per_layer_shuffled does not have policy_type in the allowed set,
        # so the final filter drops it. This documents current behavior.
        audit = make_audit_data()
        audit["policy_global_suggestions"] = {
            "energy_90": {"uniform_p90": 8},
            "knee": {"uniform_p90": 6},
            "erank": {"uniform_p90": 4},
        }
        write_audit_file(self.probe_dir, audit)

        cfg = make_config(
            compression={
                "allowed_ranks": [1, 2, 4, 8, 16, 32],
                "fast_mode": True,
                "max_candidates": 8,
            }
        )
        configs, _ = generate_compression_configs(
            self.probe_dir,
            cfg,
            fast_mode=True,
            max_candidates=8,
        )
        # per_layer_shuffled is filtered out by the policy_type gate
        assert "per_layer_shuffled" not in configs, "per_layer_shuffled should be filtered out by policy_type gate"

    def test_max_candidates_caps_output(self):
        cfg = make_config(
            compression={
                "allowed_ranks": [1, 2, 4, 8, 16, 32],
                "fast_mode": False,
                "max_candidates": 2,
            }
        )
        configs, _ = generate_compression_configs(
            self.probe_dir,
            cfg,
            fast_mode=False,
            max_candidates=2,
        )
        # Should not exceed 2 + possible second-rung candidates
        # The final filter re-caps, so total should be bounded
        assert len(configs) <= 4, f"Expected at most 4 configs with cap=2, got {len(configs)}"

    def test_missing_policy_suggestions_graceful(self):
        audit = make_audit_data(policy_global_suggestions={})
        write_audit_file(self.probe_dir, audit)

        # Should not raise
        configs, trace = generate_compression_configs(
            self.probe_dir,
            self.config,
            fast_mode=True,
        )
        assert isinstance(configs, dict)
        assert isinstance(trace, DecisionTrace)

    def test_constants_integration(self):
        configs, _ = generate_compression_configs(
            self.probe_dir,
            self.config,
            fast_mode=True,
        )
        for name, cfg in configs.items():
            score = cfg.get("conservatism_score")
            policy = cfg.get("policy_type")
            if score is not None and policy in CONSERVATISM_SCORES:
                assert score == CONSERVATISM_SCORES[policy], (
                    f"{name}: conservatism_score {score} != "
                    f"CONSERVATISM_SCORES[{policy}] ({CONSERVATISM_SCORES[policy]})"
                )


if __name__ == "__main__":
    unittest.main()
