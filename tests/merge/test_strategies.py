"""Tests for merge strategy implementations.

Validates LinearMerge and TIESMerge against known mathematical properties.
"""

from __future__ import annotations

import pytest
import torch

from gradience.vnext.merge.strategies import (
    LayerMergeConfig,
    LinearMerge,
    TIESMerge,
    DARELinearMerge,
    DARETIESMerge,
    get_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_dW_pair():
    """Two small ΔW matrices with known values for deterministic testing."""
    torch.manual_seed(42)
    dW_a = torch.randn(8, 6)
    dW_b = torch.randn(8, 6)
    return dW_a, dW_b


@pytest.fixture
def make_config():
    """Factory for LayerMergeConfig."""
    def _make(
        strategy: str = "linear",
        coefficients=(0.5, 0.5),
        trim_fraction: float = 0.0,
    ) -> LayerMergeConfig:
        return LayerMergeConfig(
            module_prefix="test.layer.q_proj",
            strategy=strategy,
            coefficients=coefficients,
            trim_fraction=trim_fraction,
        )
    return _make


# ---------------------------------------------------------------------------
# LinearMerge
# ---------------------------------------------------------------------------


class TestLinearMerge:
    def test_equal_coefficients(self, simple_dW_pair, make_config):
        """dW_merged = 0.5 * dW_a + 0.5 * dW_b."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="linear", coefficients=(0.5, 0.5))
        strategy = LinearMerge()

        result = strategy.merge(dW_a, dW_b, config)

        expected = 0.5 * dW_a + 0.5 * dW_b
        torch.testing.assert_close(result, expected)

    def test_asymmetric_coefficients(self, simple_dW_pair, make_config):
        """dW_merged = 0.7 * dW_a + 0.3 * dW_b."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="linear", coefficients=(0.7, 0.3))
        strategy = LinearMerge()

        result = strategy.merge(dW_a, dW_b, config)

        expected = 0.7 * dW_a + 0.3 * dW_b
        torch.testing.assert_close(result, expected)

    def test_zero_coefficient_a(self, simple_dW_pair, make_config):
        """When coeff_a=0, result is just coeff_b * dW_b."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="linear", coefficients=(0.0, 1.0))
        strategy = LinearMerge()

        result = strategy.merge(dW_a, dW_b, config)

        torch.testing.assert_close(result, dW_b)

    def test_output_shape_preserved(self, simple_dW_pair, make_config):
        """Output has the same shape as inputs."""
        dW_a, dW_b = simple_dW_pair
        config = make_config()
        result = LinearMerge().merge(dW_a, dW_b, config)
        assert result.shape == dW_a.shape


# ---------------------------------------------------------------------------
# TIESMerge
# ---------------------------------------------------------------------------


class TestTIESMerge:
    def test_ties_no_trim(self, simple_dW_pair, make_config):
        """TIES with trim_fraction=0.0 does elect-sign + disjoint-mean only."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="ties", coefficients=(0.5, 0.5), trim_fraction=0.0)
        strategy = TIESMerge()

        result = strategy.merge(dW_a, dW_b, config)

        # Result should have the same shape
        assert result.shape == dW_a.shape
        # Result should not be all zeros (inputs have non-zero values)
        assert result.abs().sum() > 0

    def test_ties_with_trim(self, simple_dW_pair, make_config):
        """TIES with trim_fraction=0.3 should produce sparser intermediate results."""
        dW_a, dW_b = simple_dW_pair
        config_no_trim = make_config(strategy="ties", trim_fraction=0.0)
        config_trim = make_config(strategy="ties", trim_fraction=0.3)
        strategy = TIESMerge()

        result_no_trim = strategy.merge(dW_a, dW_b, config_no_trim)
        result_trim = strategy.merge(dW_a, dW_b, config_trim)

        # Both should have same shape
        assert result_no_trim.shape == result_trim.shape
        # Trimmed result may differ from untrimmed
        # (exact equality would be surprising — different trimming produces
        # different elected signs and different disjoint means)
        # We just verify both produce valid outputs
        assert result_trim.shape == dW_a.shape

    def test_ties_full_trim_gives_zeros(self, simple_dW_pair, make_config):
        """TIES with trim_fraction=1.0 should produce all zeros."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="ties", trim_fraction=1.0)
        strategy = TIESMerge()

        result = strategy.merge(dW_a, dW_b, config)

        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_ties_identical_inputs_aligned(self, make_config):
        """When inputs are identical and positive, TIES should return them."""
        dW = torch.ones(4, 4) * 3.0
        config = make_config(strategy="ties", coefficients=(0.5, 0.5))
        strategy = TIESMerge()

        result = strategy.merge(dW, dW, config)

        # Both task vectors are 0.5 * dW = 1.5, identical positive values.
        # Sign election: +1 everywhere.
        # Disjoint mean: both agree, mean of [1.5, 1.5] = 1.5.
        expected = 0.5 * dW
        torch.testing.assert_close(result, expected)

    def test_ties_opposing_inputs_cancel(self, make_config):
        """When inputs are exact negatives, TIES should partially cancel."""
        dW = torch.ones(4, 4) * 2.0
        config = make_config(strategy="ties", coefficients=(0.5, 0.5))
        strategy = TIESMerge()

        # dW_a = +1.0 (after 0.5 scaling), dW_b = -1.0 (after 0.5 scaling)
        result = strategy.merge(dW, -dW, config)

        # Task vectors: tv_a = +1.0, tv_b = -1.0
        # Sign election: sum of signs = +1 + (-1) = 0, tie -> positive (convention)
        # Disjoint mean: only tv_a agrees (+1 sign), count=1, result = +1.0
        expected = torch.ones(4, 4)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestGetStrategy:
    def test_linear(self):
        s = get_strategy("linear")
        assert isinstance(s, LinearMerge)

    def test_ties(self):
        s = get_strategy("ties")
        assert isinstance(s, TIESMerge)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown merge strategy"):
            get_strategy("nonexistent")


# ---------------------------------------------------------------------------
# LayerMergeConfig serialization
# ---------------------------------------------------------------------------


class TestLayerMergeConfig:
    def test_roundtrip(self):
        config = LayerMergeConfig(
            module_prefix="model.layers.0.q_proj",
            strategy="ties",
            coefficients=(0.6, 0.4),
            target_rank=8,
            trim_fraction=0.2,
        )
        d = config.to_dict()
        restored = LayerMergeConfig.from_dict(d)
        assert restored == config

    def test_defaults(self):
        config = LayerMergeConfig(
            module_prefix="test",
            strategy="linear",
        )
        assert config.coefficients == (0.5, 0.5)
        assert config.target_rank is None
        assert config.trim_fraction == 0.0


# ---------------------------------------------------------------------------
# DARELinearMerge
# ---------------------------------------------------------------------------


class TestDARELinearMerge:
    def test_output_shape_preserved(self, simple_dW_pair, make_config):
        """Output has the same shape as inputs."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_linear", trim_fraction=0.3)
        strategy = get_strategy("dare_linear")
        torch.manual_seed(0)
        result = strategy.merge(dW_a, dW_b, config)
        assert result.shape == dW_a.shape

    def test_no_dropout_equals_linear(self, simple_dW_pair, make_config):
        """With trim_fraction=0.0 (no dropout), DARE-Linear = Linear."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_linear", trim_fraction=0.0)
        strategy = get_strategy("dare_linear")
        result = strategy.merge(dW_a, dW_b, config)
        expected = 0.5 * dW_a + 0.5 * dW_b
        torch.testing.assert_close(result, expected)

    def test_full_dropout_gives_zeros(self, simple_dW_pair, make_config):
        """With trim_fraction=1.0, all params dropped -> zeros."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_linear", trim_fraction=1.0)
        strategy = get_strategy("dare_linear")
        result = strategy.merge(dW_a, dW_b, config)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_rescaling_preserves_expected_value(self, make_config):
        """After drop+rescale, expected value matches original."""
        torch.manual_seed(42)
        # Large tensor for statistical convergence
        dW_a = torch.ones(1000, 1000)
        dW_b = torch.ones(1000, 1000)
        config = make_config(strategy="dare_linear", trim_fraction=0.3, coefficients=(0.5, 0.5))
        strategy = get_strategy("dare_linear")
        # Average over many trials
        results = []
        for seed in range(50):
            torch.manual_seed(seed)
            r = strategy.merge(dW_a, dW_b, config)
            results.append(r.mean().item())
        avg = sum(results) / len(results)
        # Expected: 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert abs(avg - 1.0) < 0.05, f"Expected ~1.0, got {avg}"

    def test_deterministic_with_same_seed(self, simple_dW_pair, make_config):
        """Same manual seed -> same result."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_linear", trim_fraction=0.5)
        strategy = get_strategy("dare_linear")
        torch.manual_seed(99)
        r1 = strategy.merge(dW_a, dW_b, config)
        torch.manual_seed(99)
        r2 = strategy.merge(dW_a, dW_b, config)
        torch.testing.assert_close(r1, r2)


# ---------------------------------------------------------------------------
# DARETIESMerge
# ---------------------------------------------------------------------------


class TestDARETIESMerge:
    def test_output_shape_preserved(self, simple_dW_pair, make_config):
        """Output has the same shape as inputs."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_ties", trim_fraction=0.3)
        strategy = get_strategy("dare_ties")
        torch.manual_seed(0)
        result = strategy.merge(dW_a, dW_b, config)
        assert result.shape == dW_a.shape

    def test_no_dropout_equals_ties(self, simple_dW_pair, make_config):
        """With trim_fraction=0.0, DARE-TIES = TIES (no DARE dropout applied)."""
        dW_a, dW_b = simple_dW_pair
        config_dare = make_config(strategy="dare_ties", trim_fraction=0.0)
        config_ties = make_config(strategy="ties", trim_fraction=0.0)
        dare_strategy = get_strategy("dare_ties")
        ties_strategy = get_strategy("ties")
        result_dare = dare_strategy.merge(dW_a, dW_b, config_dare)
        result_ties = ties_strategy.merge(dW_a, dW_b, config_ties)
        torch.testing.assert_close(result_dare, result_ties)

    def test_full_dropout_gives_zeros(self, simple_dW_pair, make_config):
        """With trim_fraction=1.0, all params dropped -> zeros."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_ties", trim_fraction=1.0)
        strategy = get_strategy("dare_ties")
        result = strategy.merge(dW_a, dW_b, config)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_deterministic_with_same_seed(self, simple_dW_pair, make_config):
        """Same manual seed -> same result."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_ties", trim_fraction=0.5)
        strategy = get_strategy("dare_ties")
        torch.manual_seed(99)
        r1 = strategy.merge(dW_a, dW_b, config)
        torch.manual_seed(99)
        r2 = strategy.merge(dW_a, dW_b, config)
        torch.testing.assert_close(r1, r2)


# ---------------------------------------------------------------------------
# Updated factory tests
# ---------------------------------------------------------------------------


class TestGetStrategyExtended:
    def test_dare_linear(self):
        s = get_strategy("dare_linear")
        assert isinstance(s, DARELinearMerge)

    def test_dare_ties(self):
        s = get_strategy("dare_ties")
        assert isinstance(s, DARETIESMerge)
