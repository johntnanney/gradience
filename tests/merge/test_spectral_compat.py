"""Tests for gradience.vnext.merge.spectral_compat — SVD math and subspace metrics."""

from __future__ import annotations

import pytest
import torch

from gradience.vnext.merge.spectral_compat import (
    SubspaceMetrics,
    _energy_rank,
    _stable_rank,
    compute_layer_svd,
    compute_subspace_metrics,
)


# ---------------------------------------------------------------------------
# compute_layer_svd
# ---------------------------------------------------------------------------


class TestComputeLayerSVD:
    """Tests for QR-based SVD decomposition."""

    def test_shapes(self):
        """U, S, Vt have correct shapes."""
        torch.manual_seed(1)
        A = torch.randn(8, 48)   # (r, d_in)
        B = torch.randn(64, 8)   # (d_out, r)

        U, S, Vt = compute_layer_svd(A, B)

        assert U.shape == (64, 8)   # (d_out, r)
        assert S.shape == (8,)
        assert Vt.shape == (8, 48)  # (r, d_in)

    def test_reconstruction(self):
        """U @ diag(S) @ Vt ≈ B @ A (within floating-point tolerance)."""
        torch.manual_seed(2)
        A = torch.randn(8, 48)
        B = torch.randn(64, 8)

        U, S, Vt = compute_layer_svd(A, B)
        reconstructed = U @ torch.diag(S) @ Vt
        expected = B.double() @ A.double()

        assert torch.allclose(reconstructed, expected, atol=1e-10)

    def test_scaling(self):
        """Scaling factor is correctly applied."""
        torch.manual_seed(3)
        A = torch.randn(4, 32)
        B = torch.randn(16, 4)
        scaling = 2.5

        U, S, Vt = compute_layer_svd(A, B, scaling=scaling)
        reconstructed = U @ torch.diag(S) @ Vt
        expected = scaling * B.double() @ A.double()

        assert torch.allclose(reconstructed, expected, atol=1e-10)

    def test_singular_values_descending(self):
        """Singular values are in descending order."""
        torch.manual_seed(4)
        A = torch.randn(8, 48)
        B = torch.randn(64, 8)

        _, S, _ = compute_layer_svd(A, B)

        for i in range(len(S) - 1):
            assert S[i] >= S[i + 1], f"S[{i}]={S[i]} < S[{i + 1}]={S[i + 1]}"

    def test_orthonormal_U(self):
        """U columns are orthonormal."""
        torch.manual_seed(5)
        A = torch.randn(8, 48)
        B = torch.randn(64, 8)

        U, _, _ = compute_layer_svd(A, B)
        gram = U.T @ U

        assert torch.allclose(gram, torch.eye(8, dtype=torch.float64), atol=1e-10)

    def test_orthonormal_Vt(self):
        """Vt rows are orthonormal."""
        torch.manual_seed(6)
        A = torch.randn(8, 48)
        B = torch.randn(64, 8)

        _, _, Vt = compute_layer_svd(A, B)
        gram = Vt @ Vt.T

        assert torch.allclose(gram, torch.eye(8, dtype=torch.float64), atol=1e-10)


# ---------------------------------------------------------------------------
# _energy_rank
# ---------------------------------------------------------------------------


class TestEnergyRank:
    """Tests for effective rank via energy threshold."""

    def test_single_dominant(self):
        """One large value dominates: energy rank = 1."""
        s = torch.tensor([10.0, 0.01, 0.001, 0.0001])
        assert _energy_rank(s, threshold=0.90) == 1

    def test_uniform(self):
        """Uniform spectrum: need many dimensions."""
        s = torch.ones(10)
        assert _energy_rank(s, threshold=0.90) == 9  # need 9/10 = 90%

    def test_empty(self):
        """Empty tensor returns 1."""
        s = torch.tensor([])
        assert _energy_rank(s) == 1

    def test_near_zero(self):
        """All near-zero returns 1."""
        s = torch.tensor([1e-15, 1e-16])
        assert _energy_rank(s) == 1

    def test_threshold_boundary(self):
        """Check exact boundary behavior."""
        s = torch.tensor([1.0, 1.0])  # each contributes 50%
        assert _energy_rank(s, threshold=0.50) == 1  # first captures 50%
        assert _energy_rank(s, threshold=0.90) == 2  # need both for 90%


# ---------------------------------------------------------------------------
# _stable_rank
# ---------------------------------------------------------------------------


class TestStableRank:
    """Tests for stable rank computation."""

    def test_rank_one(self):
        """Single nonzero value: stable rank = 1."""
        s = torch.tensor([5.0, 0.0, 0.0])
        assert _stable_rank(s) == pytest.approx(1.0)

    def test_uniform(self):
        """Uniform spectrum: stable rank = n."""
        s = torch.ones(5)
        assert _stable_rank(s) == pytest.approx(5.0)

    def test_empty(self):
        """Empty or near-zero returns 1.0."""
        assert _stable_rank(torch.tensor([])) == pytest.approx(1.0)
        assert _stable_rank(torch.tensor([0.0])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_subspace_metrics
# ---------------------------------------------------------------------------


class TestComputeSubspaceMetrics:
    """Tests for the full subspace comparison pipeline."""

    def _make_pair_from_svd(self, U_a, S_a, Vt_a, U_b, S_b, Vt_b, rank=8, alpha=16.0):
        """Helper: build A/B factors from SVD and compute metrics."""
        scaling = alpha / rank

        # Factor ΔW = U @ diag(S) @ Vt into B @ A
        sqrt_S_a = torch.sqrt(S_a).unsqueeze(1)
        A_a = (sqrt_S_a * Vt_a) / (scaling ** 0.5)
        B_a = (U_a * sqrt_S_a.T) / (scaling ** 0.5)

        sqrt_S_b = torch.sqrt(S_b).unsqueeze(1)
        A_b = (sqrt_S_b * Vt_b) / (scaling ** 0.5)
        B_b = (U_b * sqrt_S_b.T) / (scaling ** 0.5)

        return compute_subspace_metrics(
            A_a.float(), B_a.float(), alpha, rank,
            A_b.float(), B_b.float(), alpha, rank,
        )

    def test_identical_matrices_high_overlap(self):
        """Same subspace + same direction -> overlap ≈ 1.0, agreement ≈ 1.0."""
        torch.manual_seed(10)
        U, _, Vt = torch.linalg.svd(torch.randn(64, 48, dtype=torch.float64), full_matrices=False)
        U_r = U[:, :8]
        Vt_r = Vt[:8, :]
        S = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)

        metrics = self._make_pair_from_svd(U_r, S, Vt_r, U_r, S, Vt_r)

        assert metrics.mean_overlap > 0.9
        assert metrics.directional_agreement > 0.5
        assert metrics.magnitude_ratio == pytest.approx(1.0, abs=0.1)

    def test_orthogonal_matrices_low_overlap(self):
        """Orthogonal subspaces -> overlap ≈ 0.0."""
        torch.manual_seed(11)
        U, _, Vt = torch.linalg.svd(torch.randn(64, 48, dtype=torch.float64), full_matrices=False)

        U_a = U[:, :8]
        Vt_a = Vt[:8, :]
        U_b = U[:, 8:16]
        Vt_b = Vt[8:16, :]
        S = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)

        metrics = self._make_pair_from_svd(U_a, S, Vt_a, U_b, S, Vt_b)

        assert metrics.mean_overlap < 0.2
        assert metrics.max_overlap < 0.3

    def test_opposing_directions_negative_agreement(self):
        """Same subspace but negated -> overlap ≈ 1.0, agreement < 0."""
        torch.manual_seed(12)
        U, _, Vt = torch.linalg.svd(torch.randn(64, 48, dtype=torch.float64), full_matrices=False)
        U_r = U[:, :8]
        Vt_r = Vt[:8, :]
        S = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)

        metrics = self._make_pair_from_svd(U_r, S, Vt_r, -U_r, S, Vt_r)

        assert metrics.mean_overlap > 0.9
        assert metrics.directional_agreement < 0

    def test_magnitude_ratio_always_ge_one(self):
        """Magnitude ratio should always be >= 1.0."""
        torch.manual_seed(13)
        U, _, Vt = torch.linalg.svd(torch.randn(64, 48, dtype=torch.float64), full_matrices=False)
        U_r = U[:, :8]
        Vt_r = Vt[:8, :]

        S_large = torch.linspace(10.0, 2.0, 8, dtype=torch.float64)
        S_small = torch.linspace(1.0, 0.2, 8, dtype=torch.float64)

        metrics = self._make_pair_from_svd(U_r, S_small, Vt_r, U_r, S_large, Vt_r)

        assert metrics.magnitude_ratio >= 1.0
        assert metrics.frobenius_ratio >= 1.0

    def test_magnitude_imbalance_detected(self):
        """Large magnitude difference -> high magnitude_ratio."""
        torch.manual_seed(14)
        U, _, Vt = torch.linalg.svd(torch.randn(64, 48, dtype=torch.float64), full_matrices=False)
        U_r = U[:, :8]
        Vt_r = Vt[:8, :]

        S_a = torch.linspace(50.0, 10.0, 8, dtype=torch.float64)
        S_b = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)

        metrics = self._make_pair_from_svd(U_r, S_a, Vt_r, U_r, S_b, Vt_r)

        assert metrics.magnitude_ratio > 5.0

    def test_different_ranks(self):
        """Different nominal ranks: metrics still computed."""
        torch.manual_seed(15)
        U, _, Vt = torch.linalg.svd(torch.randn(64, 48, dtype=torch.float64), full_matrices=False)

        r_a, r_b = 4, 16
        alpha = 16.0
        scaling_a = alpha / r_a
        scaling_b = alpha / r_b

        S_a = torch.linspace(5.0, 1.0, r_a, dtype=torch.float64)
        S_b = torch.linspace(3.0, 0.5, r_b, dtype=torch.float64)

        sqrt_Sa = torch.sqrt(S_a).unsqueeze(1)
        A_a = (sqrt_Sa * Vt[:r_a, :]) / (scaling_a ** 0.5)
        B_a = (U[:, :r_a] * sqrt_Sa.T) / (scaling_a ** 0.5)

        sqrt_Sb = torch.sqrt(S_b).unsqueeze(1)
        A_b = (sqrt_Sb * Vt[:r_b, :]) / (scaling_b ** 0.5)
        B_b = (U[:, :r_b] * sqrt_Sb.T) / (scaling_b ** 0.5)

        metrics = compute_subspace_metrics(
            A_a.float(), B_a.float(), alpha, r_a,
            A_b.float(), B_b.float(), alpha, r_b,
        )

        assert isinstance(metrics, SubspaceMetrics)
        assert metrics.nominal_rank_a == 4
        assert metrics.nominal_rank_b == 16
        assert 0 <= metrics.mean_overlap <= 1.0

    def test_to_dict_serializable(self):
        """to_dict() produces JSON-serializable output."""
        import json

        torch.manual_seed(16)
        A_a = torch.randn(4, 32)
        B_a = torch.randn(16, 4)
        A_b = torch.randn(4, 32)
        B_b = torch.randn(16, 4)

        metrics = compute_subspace_metrics(
            A_a, B_a, 16.0, 4, A_b, B_b, 16.0, 4,
        )

        d = metrics.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert "principal_angle_cosines" in json_str

    def test_principal_angles_clamped(self):
        """Principal angle cosines are in [0, 1]."""
        torch.manual_seed(17)
        A_a = torch.randn(8, 48)
        B_a = torch.randn(64, 8)
        A_b = torch.randn(8, 48)
        B_b = torch.randn(64, 8)

        metrics = compute_subspace_metrics(
            A_a, B_a, 16.0, 8, A_b, B_b, 16.0, 8,
        )

        for cos_a in metrics.principal_angle_cosines:
            assert 0.0 <= cos_a <= 1.0 + 1e-6

    def test_near_zero_matrix_handled(self):
        """Near-zero matrices don't crash."""
        eps = 1e-15
        A_a = torch.full((4, 32), eps)
        B_a = torch.full((16, 4), eps)
        A_b = torch.randn(4, 32)
        B_b = torch.randn(16, 4)

        metrics = compute_subspace_metrics(
            A_a, B_a, 16.0, 4, A_b, B_b, 16.0, 4,
        )

        assert isinstance(metrics, SubspaceMetrics)
        # Near-zero adapter should have finite metrics
        assert not any(
            x != x  # NaN check
            for x in [metrics.mean_overlap, metrics.directional_agreement]
        )


# ---------------------------------------------------------------------------
# Symmetric scale fields (Section 3)
# ---------------------------------------------------------------------------


class TestSymmetricScaleFields:
    """Tests for the symmetric scale fields on SubspaceMetrics."""

    def _make_metrics(self, S_a, S_b, seed=20, rank=8, alpha=16.0):
        """Helper: build metrics from specified singular value spectra."""
        torch.manual_seed(seed)
        U, _, Vt = torch.linalg.svd(
            torch.randn(64, 48, dtype=torch.float64), full_matrices=False
        )
        U_r = U[:, :rank]
        Vt_r = Vt[:rank, :]

        scaling = alpha / rank
        sqrt_Sa = torch.sqrt(S_a).unsqueeze(1)
        A_a = (sqrt_Sa * Vt_r) / (scaling ** 0.5)
        B_a = (U_r * sqrt_Sa.T) / (scaling ** 0.5)

        sqrt_Sb = torch.sqrt(S_b).unsqueeze(1)
        A_b = (sqrt_Sb * Vt_r) / (scaling ** 0.5)
        B_b = (U_r * sqrt_Sb.T) / (scaling ** 0.5)

        return compute_subspace_metrics(
            A_a.float(), B_a.float(), alpha, rank,
            A_b.float(), B_b.float(), alpha, rank,
        )

    def test_scale_bounded_ratio_in_range(self):
        """scale_bounded_ratio must be in [0, 1]."""
        S_a = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)
        S_b = torch.linspace(3.0, 0.5, 8, dtype=torch.float64)
        metrics = self._make_metrics(S_a, S_b)
        assert 0.0 <= metrics.scale_bounded_ratio <= 1.0

    def test_frob_bounded_ratio_in_range(self):
        """frob_bounded_ratio must be in [0, 1]."""
        S_a = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)
        S_b = torch.linspace(3.0, 0.5, 8, dtype=torch.float64)
        metrics = self._make_metrics(S_a, S_b)
        assert 0.0 <= metrics.frob_bounded_ratio <= 1.0

    def test_identical_adapters_bounded_ratios_near_one(self):
        """Identical adapters should have bounded ratios close to 1.0."""
        S = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)
        metrics = self._make_metrics(S, S)
        assert metrics.scale_bounded_ratio == pytest.approx(1.0, abs=0.05)
        assert metrics.frob_bounded_ratio == pytest.approx(1.0, abs=0.05)

    def test_identical_adapters_log_ratios_near_zero(self):
        """Identical adapters should have log ratios close to 0.0."""
        S = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)
        metrics = self._make_metrics(S, S)
        assert metrics.scale_log_ratio == pytest.approx(0.0, abs=0.05)
        assert metrics.frob_log_ratio == pytest.approx(0.0, abs=0.05)

    def test_very_different_magnitudes_bounded_ratios_small(self):
        """Large magnitude difference -> bounded ratios well below 1."""
        S_a = torch.linspace(50.0, 10.0, 8, dtype=torch.float64)
        S_b = torch.linspace(0.5, 0.1, 8, dtype=torch.float64)
        metrics = self._make_metrics(S_a, S_b)
        assert metrics.scale_bounded_ratio < 0.05
        assert metrics.frob_bounded_ratio < 0.05

    def test_json_serialization_includes_new_fields(self):
        """to_dict() and JSON serialization include the symmetric scale fields."""
        import json

        S = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)
        metrics = self._make_metrics(S, S)
        d = metrics.to_dict()
        json_str = json.dumps(d)

        assert "scale_bounded_ratio" in json_str
        assert "scale_log_ratio" in json_str
        assert "frob_bounded_ratio" in json_str
        assert "frob_log_ratio" in json_str

    def test_backward_compat_old_fields_still_present(self):
        """Old asymmetric fields are still present alongside new symmetric ones."""
        S = torch.linspace(5.0, 1.0, 8, dtype=torch.float64)
        metrics = self._make_metrics(S, S)

        # Old fields still exist
        assert hasattr(metrics, "magnitude_ratio")
        assert hasattr(metrics, "frobenius_ratio")
        assert metrics.magnitude_ratio >= 1.0
        assert metrics.frobenius_ratio >= 1.0

        # New fields also exist
        assert hasattr(metrics, "scale_bounded_ratio")
        assert hasattr(metrics, "scale_log_ratio")
        assert hasattr(metrics, "frob_bounded_ratio")
        assert hasattr(metrics, "frob_log_ratio")
