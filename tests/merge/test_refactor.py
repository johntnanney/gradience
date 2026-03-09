"""Tests for SVD refactoring of dense ΔW back into LoRA (B, A) format.

Validates that refactor_to_lora produces correct factors and meaningful
reconstruction error measurements.
"""

from __future__ import annotations

import pytest
import torch

from gradience.vnext.merge.refactor import refactor_to_lora

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def known_dW():
    """A dense ΔW with known singular value structure."""
    torch.manual_seed(99)
    d_out, d_in = 32, 24
    # Build from known SVD to have controlled rank structure
    U, _, Vt = torch.linalg.svd(torch.randn(d_out, d_in), full_matrices=False)
    # Exponentially decaying singular values
    S = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.6, 0.3, 0.15, 0.07])
    # Pad with zeros to full min(d_out, d_in) = 24
    S_full = torch.zeros(min(d_out, d_in))
    S_full[: len(S)] = S
    dW = U @ torch.diag(S_full) @ Vt
    return dW, S


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRefactorToLora:
    def test_full_rank_roundtrip(self, known_dW):
        """At full rank, reconstruction error should be near zero."""
        dW, S = known_dW
        # 8 non-zero singular values
        B, A, error = refactor_to_lora(dW, target_rank=8, alpha=16.0)

        assert B.shape == (32, 8)
        assert A.shape == (8, 24)
        assert error < 1e-6, f"Full-rank error too high: {error}"

        # Verify scaling * B @ A ≈ dW
        scaling = 16.0 / 8
        reconstructed = scaling * B @ A
        torch.testing.assert_close(reconstructed, dW, atol=1e-5, rtol=1e-5)

    def test_truncation_has_positive_error(self, known_dW):
        """At lower rank, error should be positive but bounded."""
        dW, S = known_dW
        B, A, error = refactor_to_lora(dW, target_rank=4, alpha=16.0)

        assert B.shape == (32, 4)
        assert A.shape == (4, 24)
        assert error > 0.01, f"Truncation should produce nonzero error: {error}"
        assert error < 1.0, f"Error should be < 1.0: {error}"

    def test_scaling_consistency(self, known_dW):
        """scaling * B @ A should approximate dW within the reported error."""
        dW, _ = known_dW
        target_rank = 6
        alpha = 16.0

        B, A, error = refactor_to_lora(dW, target_rank=target_rank, alpha=alpha)

        scaling = alpha / target_rank
        reconstructed = scaling * B @ A
        actual_error = torch.linalg.norm(dW - reconstructed) / torch.linalg.norm(dW)

        # Reported error should match computed error
        assert abs(error - actual_error.item()) < 1e-6

    def test_higher_rank_lower_error(self, known_dW):
        """More rank should give lower reconstruction error."""
        dW, _ = known_dW

        _, _, error_r4 = refactor_to_lora(dW, target_rank=4, alpha=16.0)
        _, _, error_r8 = refactor_to_lora(dW, target_rank=8, alpha=16.0)

        assert error_r8 < error_r4

    def test_dtype_float32(self, known_dW):
        """Float32 compute path should work correctly."""
        dW, _ = known_dW
        B, A, error = refactor_to_lora(dW, target_rank=4, alpha=16.0, compute_dtype="float32")

        assert B.shape == (32, 4)
        assert A.shape == (4, 24)
        assert error > 0  # truncated, so positive error
        assert error < 1.0

    def test_output_dtype_matches_input(self, known_dW):
        """Output tensors should have the same dtype as input."""
        dW, _ = known_dW

        # Float32 input
        dW_f32 = dW.float()
        B, A, _ = refactor_to_lora(dW_f32, target_rank=4, alpha=16.0)
        assert B.dtype == torch.float32
        assert A.dtype == torch.float32

        # Float64 input
        dW_f64 = dW.double()
        B, A, _ = refactor_to_lora(dW_f64, target_rank=4, alpha=16.0)
        assert B.dtype == torch.float64
        assert A.dtype == torch.float64

    def test_invalid_rank_too_low(self):
        """target_rank < 1 should raise ValueError."""
        dW = torch.randn(8, 6)
        with pytest.raises(ValueError, match="target_rank must be >= 1"):
            refactor_to_lora(dW, target_rank=0, alpha=16.0)

    def test_invalid_rank_too_high(self):
        """target_rank exceeding matrix dimensions should raise ValueError."""
        dW = torch.randn(8, 6)
        with pytest.raises(ValueError, match="exceeds max possible rank"):
            refactor_to_lora(dW, target_rank=7, alpha=16.0)

    def test_non_2d_raises(self):
        """Non-2D input should raise ValueError."""
        dW = torch.randn(8, 6, 4)
        with pytest.raises(ValueError, match="must be 2-D"):
            refactor_to_lora(dW, target_rank=4, alpha=16.0)

    def test_zero_matrix(self):
        """All-zero dW should give zero error and zero factors."""
        dW = torch.zeros(8, 6)
        B, A, error = refactor_to_lora(dW, target_rank=4, alpha=16.0)

        assert B.shape == (8, 4)
        assert A.shape == (4, 6)
        assert error == 0.0
        torch.testing.assert_close(B, torch.zeros_like(B))
        torch.testing.assert_close(A, torch.zeros_like(A))
