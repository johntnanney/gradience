"""Tests for gradience.vnext.merge.null_controls — null control baselines."""

from __future__ import annotations

import pytest
import torch

from gradience.vnext.merge.null_controls import (
    layer_shuffle_control,
    randomized_subspace_control,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orthonormal_columns(d: int, k: int, seed: int = 0) -> torch.Tensor:
    """Return a (d, k) orthonormal matrix via QR."""
    torch.manual_seed(seed)
    M = torch.randn(d, k, dtype=torch.float64)
    Q, _ = torch.linalg.qr(M)
    return Q


# ---------------------------------------------------------------------------
# randomized_subspace_control
# ---------------------------------------------------------------------------


class TestRandomizedSubspaceControl:
    """Tests for the randomized subspace null control."""

    def test_return_structure(self):
        """Result dict contains all expected keys with correct types."""
        U_A = _orthonormal_columns(64, 4, seed=1)
        U_B = _orthonormal_columns(64, 4, seed=2)

        result = randomized_subspace_control(U_A, U_B, n_random=5, seed=42)

        assert set(result.keys()) == {
            "null_mean_overlaps",
            "null_max_overlaps",
            "null_mean_of_means",
            "null_std_of_means",
            "n_trials",
        }
        assert isinstance(result["null_mean_overlaps"], list)
        assert isinstance(result["null_max_overlaps"], list)
        assert len(result["null_mean_overlaps"]) == 5
        assert len(result["null_max_overlaps"]) == 5
        assert result["n_trials"] == 5

    def test_null_overlaps_below_aligned(self):
        """Random subspaces have lower overlap than a truly aligned pair.

        Build U_A and U_B that share a subspace, confirm the real
        overlap exceeds the null distribution.
        """
        torch.manual_seed(10)
        # Shared orthonormal basis -- perfect alignment
        U_shared = _orthonormal_columns(128, 8, seed=10)
        # Add tiny noise so they aren't bitwise identical (still nearly aligned)
        noise = torch.randn_like(U_shared) * 1e-6
        U_A = U_shared + noise
        # Re-orthonormalize
        U_A, _ = torch.linalg.qr(U_A)
        U_B = U_shared.clone()

        # Real overlap
        cross = U_A.T @ U_B
        real_cosines = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
        real_mean = real_cosines.mean().item()

        # Null distribution
        result = randomized_subspace_control(U_A, U_B, n_random=20, seed=99)

        # The real aligned overlap should be well above the null mean
        assert real_mean > result["null_mean_of_means"] + 2 * result["null_std_of_means"], (
            f"Real overlap {real_mean:.4f} not significantly above null "
            f"{result['null_mean_of_means']:.4f} +/- {result['null_std_of_means']:.4f}"
        )

    def test_seed_reproducibility(self):
        """Same seed produces identical results."""
        U_A = _orthonormal_columns(64, 4, seed=1)
        U_B = _orthonormal_columns(64, 4, seed=2)

        r1 = randomized_subspace_control(U_A, U_B, n_random=5, seed=42)
        r2 = randomized_subspace_control(U_A, U_B, n_random=5, seed=42)

        assert r1["null_mean_overlaps"] == r2["null_mean_overlaps"]
        assert r1["null_max_overlaps"] == r2["null_max_overlaps"]

    def test_n_random_one(self):
        """Edge case: single trial still works."""
        U_A = _orthonormal_columns(32, 4, seed=3)
        U_B = _orthonormal_columns(32, 4, seed=4)

        result = randomized_subspace_control(U_A, U_B, n_random=1, seed=0)

        assert result["n_trials"] == 1
        assert len(result["null_mean_overlaps"]) == 1
        # With n=1, std should be 0.0 (fallback)
        assert result["null_std_of_means"] == 0.0

    def test_overlaps_in_unit_interval(self):
        """All overlap values are in [0, 1]."""
        U_A = _orthonormal_columns(64, 8, seed=5)
        U_B = _orthonormal_columns(64, 8, seed=6)

        result = randomized_subspace_control(U_A, U_B, n_random=10, seed=42)

        for v in result["null_mean_overlaps"]:
            assert 0.0 <= v <= 1.0 + 1e-9
        for v in result["null_max_overlaps"]:
            assert 0.0 <= v <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# layer_shuffle_control
# ---------------------------------------------------------------------------


class TestLayerShuffleControl:
    """Tests for the layer-shuffle null control."""

    def test_return_structure(self):
        """Result dict contains all expected keys with correct types."""
        bases_A = [_orthonormal_columns(64, 4, seed=i) for i in range(5)]
        bases_B = [_orthonormal_columns(64, 4, seed=i + 100) for i in range(5)]

        result = layer_shuffle_control(bases_A, bases_B, n_shuffles=3, seed=42)

        assert set(result.keys()) == {
            "shuffle_mean_overlaps",
            "shuffle_mean_of_means",
            "shuffle_std_of_means",
            "n_trials",
            "n_layers",
        }
        assert isinstance(result["shuffle_mean_overlaps"], list)
        assert len(result["shuffle_mean_overlaps"]) == 3
        assert result["n_trials"] == 3
        assert result["n_layers"] == 5

    def test_single_layer(self):
        """Edge case: single layer still returns a result."""
        bases_A = [_orthonormal_columns(64, 4, seed=0)]
        bases_B = [_orthonormal_columns(64, 4, seed=1)]

        result = layer_shuffle_control(bases_A, bases_B, n_shuffles=5, seed=42)

        assert result["n_layers"] == 1
        # Single layer means derangement is impossible; identity perm used
        assert result["n_trials"] == 5
        assert len(result["shuffle_mean_overlaps"]) == 5

    def test_seed_reproducibility(self):
        """Same seed produces identical results."""
        bases_A = [_orthonormal_columns(64, 4, seed=i) for i in range(4)]
        bases_B = [_orthonormal_columns(64, 4, seed=i + 50) for i in range(4)]

        r1 = layer_shuffle_control(bases_A, bases_B, n_shuffles=5, seed=42)
        r2 = layer_shuffle_control(bases_A, bases_B, n_shuffles=5, seed=42)

        assert r1["shuffle_mean_overlaps"] == r2["shuffle_mean_overlaps"]

    def test_mismatched_d_out_skipped(self):
        """Layers with different d_out are skipped during pairing."""
        # Layer 0: d_out=64, Layer 1: d_out=128
        bases_A = [
            _orthonormal_columns(64, 4, seed=0),
            _orthonormal_columns(128, 4, seed=1),
        ]
        bases_B = [
            _orthonormal_columns(64, 4, seed=10),
            _orthonormal_columns(128, 4, seed=11),
        ]

        # With 2 layers, a derangement swaps them: (0->1, 1->0).
        # Layer 0 of A (d_out=64) paired with layer 1 of B (d_out=128) -> skip
        # Layer 1 of A (d_out=128) paired with layer 0 of B (d_out=64) -> skip
        # All pairs skipped, so that trial produces no overlap.
        # Result should handle this gracefully.
        result = layer_shuffle_control(bases_A, bases_B, n_shuffles=5, seed=42)

        # The function should still return without error
        assert result["n_layers"] == 2

    def test_empty_layers(self):
        """Empty layer lists return zero-valued result."""
        result = layer_shuffle_control([], [], n_shuffles=5, seed=42)

        assert result["n_layers"] == 0
        assert result["n_trials"] == 0
        assert result["shuffle_mean_overlaps"] == []
        assert result["shuffle_mean_of_means"] == 0.0
        assert result["shuffle_std_of_means"] == 0.0

    def test_overlaps_in_unit_interval(self):
        """All shuffle overlap values are in [0, 1]."""
        bases_A = [_orthonormal_columns(64, 4, seed=i) for i in range(6)]
        bases_B = [_orthonormal_columns(64, 4, seed=i + 200) for i in range(6)]

        result = layer_shuffle_control(bases_A, bases_B, n_shuffles=10, seed=42)

        for v in result["shuffle_mean_overlaps"]:
            assert 0.0 <= v <= 1.0 + 1e-9
