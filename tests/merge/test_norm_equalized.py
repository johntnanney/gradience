"""Tests for norm-equalized merge strategy.

Validates that Frobenius-norm equalization works correctly before
performing the weighted linear combination.
"""

from __future__ import annotations

import math

import pytest
import torch

from gradience.vnext.merge.norm_equalized import norm_equalized_merge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fro_norm(t: torch.Tensor) -> float:
    return torch.linalg.norm(t, "fro").item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormEqualizedMerge:
    """Core tests for norm_equalized_merge."""

    def test_equal_norm_inputs_no_scaling(self):
        """When both inputs already have the same Frobenius norm, scale factors
        should be 1.0 (no-op scaling)."""
        torch.manual_seed(0)
        dW = torch.randn(8, 6)
        # Make B the same norm as A but different direction.
        norm_A = _fro_norm(dW)
        dW_B = torch.randn(8, 6)
        dW_B = dW_B * (norm_A / _fro_norm(dW_B))

        result = norm_equalized_merge(dW, dW_B)

        assert result["scale_factor_A"] == pytest.approx(1.0, abs=1e-6)
        assert result["scale_factor_B"] == pytest.approx(1.0, abs=1e-6)
        assert result["target_norm"] == pytest.approx(norm_A, rel=1e-6)
        # merged should be the simple average
        expected = 0.5 * dW + 0.5 * dW_B
        assert torch.allclose(result["merged"], expected, atol=1e-5)

    def test_imbalanced_10x_inputs_equalized(self):
        """When one adapter is 10x the norm of the other, both should be
        scaled to the geometric mean before merging."""
        torch.manual_seed(1)
        dW_A = torch.randn(8, 6)
        dW_B = torch.randn(8, 6)

        # Scale B to be 10x the norm of A.
        norm_A = _fro_norm(dW_A)
        dW_B = dW_B * (10.0 * norm_A / _fro_norm(dW_B))
        norm_B = _fro_norm(dW_B)

        result = norm_equalized_merge(dW_A, dW_B)

        # Target norm should be geometric mean.
        expected_target = math.sqrt(norm_A * norm_B)
        assert result["target_norm"] == pytest.approx(expected_target, rel=1e-6)

        # After scaling, both contributions should have the target norm.
        scaled_A = result["scale_factor_A"] * dW_A
        scaled_B = result["scale_factor_B"] * dW_B
        assert _fro_norm(scaled_A) == pytest.approx(expected_target, rel=1e-5)
        assert _fro_norm(scaled_B) == pytest.approx(expected_target, rel=1e-5)

        # Scale factors should reflect the 10x imbalance.
        # A gets scaled up, B gets scaled down.
        assert result["scale_factor_A"] > 1.0
        assert result["scale_factor_B"] < 1.0

    def test_weight_A_zero_gives_scaled_B(self):
        """weight_A=0 means the result is entirely scaled dW_B."""
        torch.manual_seed(2)
        dW_A = torch.randn(8, 6)
        dW_B = torch.randn(8, 6) * 3.0  # different norm

        result = norm_equalized_merge(dW_A, dW_B, weight_A=0.0)

        expected = result["scale_factor_B"] * dW_B
        assert torch.allclose(result["merged"], expected, atol=1e-6)

    def test_weight_A_one_gives_scaled_A(self):
        """weight_A=1 means the result is entirely scaled dW_A."""
        torch.manual_seed(3)
        dW_A = torch.randn(8, 6) * 5.0
        dW_B = torch.randn(8, 6)

        result = norm_equalized_merge(dW_A, dW_B, weight_A=1.0)

        expected = result["scale_factor_A"] * dW_A
        assert torch.allclose(result["merged"], expected, atol=1e-6)

    def test_zero_norm_A(self):
        """A zero-norm dW_A should not cause division by zero; scaling is
        skipped and B passes through unchanged."""
        dW_A = torch.zeros(4, 4)
        dW_B = torch.randn(4, 4)

        result = norm_equalized_merge(dW_A, dW_B)

        # A is zero, so scale factors should be 1.0 (no scaling applied).
        assert result["scale_factor_A"] == 1.0
        assert result["scale_factor_B"] == 1.0
        assert result["norm_A_original"] < 1e-12
        # Merged is weight_B * dW_B since A contributes nothing.
        expected = 0.5 * dW_B
        assert torch.allclose(result["merged"], expected, atol=1e-7)

    def test_zero_norm_B(self):
        """A zero-norm dW_B should not cause division by zero; scaling is
        skipped and A passes through unchanged."""
        dW_A = torch.randn(4, 4)
        dW_B = torch.zeros(4, 4)

        result = norm_equalized_merge(dW_A, dW_B)

        assert result["scale_factor_A"] == 1.0
        assert result["scale_factor_B"] == 1.0
        assert result["norm_B_original"] < 1e-12
        # Merged is weight_A * dW_A since B contributes nothing.
        expected = 0.5 * dW_A
        assert torch.allclose(result["merged"], expected, atol=1e-7)

    def test_both_zero_norm(self):
        """Both zero-norm tensors should produce a zero merged result."""
        dW_A = torch.zeros(4, 4)
        dW_B = torch.zeros(4, 4)

        result = norm_equalized_merge(dW_A, dW_B)

        assert result["target_norm"] == 0.0
        assert result["scale_factor_A"] == 1.0
        assert result["scale_factor_B"] == 1.0
        assert torch.allclose(result["merged"], torch.zeros(4, 4))

    def test_return_keys(self):
        """All expected keys are present in the result dict."""
        result = norm_equalized_merge(torch.randn(4, 4), torch.randn(4, 4))

        expected_keys = {
            "merged",
            "norm_A_original",
            "norm_B_original",
            "target_norm",
            "scale_factor_A",
            "scale_factor_B",
        }
        assert set(result.keys()) == expected_keys

    def test_merged_shape_matches_input(self):
        """Output shape should match input shape."""
        dW_A = torch.randn(16, 8)
        dW_B = torch.randn(16, 8)

        result = norm_equalized_merge(dW_A, dW_B)

        assert result["merged"].shape == (16, 8)
