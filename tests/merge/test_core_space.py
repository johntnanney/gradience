"""Tests for optional core-space diagnostics."""

from __future__ import annotations

import torch

from gradience.vnext.merge.core_space import (
    aggregate_core_space_diagnostics,
    compute_core_space_layer_diagnostic,
)
from gradience.vnext.merge.core_space_types import CoreSpaceLayerDiagnostic


def _rand_factors(*, r: int, d_in: int = 32, d_out: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(r, d_in), torch.randn(d_out, r)


def _pair_diag_for_archetype(kind: str, *, n_layers: int = 8) -> object:
    torch.manual_seed(0)
    layer_diags = []
    for i in range(n_layers):
        A_a, B_a = _rand_factors(r=8, d_in=64, d_out=64)
        if kind == "self":
            A_b = A_a.clone()
            B_b = B_a.clone()
        elif kind == "near":
            A_b = A_a + 0.05 * torch.randn_like(A_a)
            B_b = B_a + 0.05 * torch.randn_like(B_a)
        elif kind == "imbalanced_related":
            A_b = (A_a + 0.01 * torch.randn_like(A_a)) * 0.08
            B_b = (B_a + 0.01 * torch.randn_like(B_a)) * 0.08
        elif kind == "random_other":
            A_b, B_b = _rand_factors(r=8, d_in=64, d_out=64)
        else:
            raise ValueError(f"Unknown archetype '{kind}'")

        layer_diags.append(
            compute_core_space_layer_diagnostic(
                layer_name=f"layer_{i}",
                A_a=A_a,
                B_a=B_a,
                alpha_a=16.0,
                r_a=8,
                A_b=A_b,
                B_b=B_b,
                alpha_b=16.0,
                r_b=8,
                compute_dtype=torch.float64,
            )
        )

    return aggregate_core_space_diagnostics(layer_diags)


class TestCoreSpaceLayerDiagnostic:
    def test_metric_ranges_for_identical_updates(self) -> None:
        torch.manual_seed(0)
        A, B = _rand_factors(r=4)
        diag = compute_core_space_layer_diagnostic(
            layer_name="model.layers.0.self_attn.q_proj",
            A_a=A,
            B_a=B,
            alpha_a=8.0,
            r_a=4,
            A_b=A.clone(),
            B_b=B.clone(),
            alpha_b=8.0,
            r_b=4,
            energy_threshold=0.9,
            compute_dtype=torch.float64,
        )

        assert 0.0 <= diag.shared_basis_score <= 1.0
        assert 0.0 <= diag.basis_distortion <= 1.0
        assert diag.effective_shared_rank >= 1
        assert diag.status in {"compatible", "marginal", "incompatible", "not_applicable"}
        # Identical updates should have very low distortion.
        assert diag.basis_distortion <= 0.05

    def test_zero_updates_are_not_applicable(self) -> None:
        A = torch.zeros(4, 16)
        B = torch.zeros(32, 4)
        diag = compute_core_space_layer_diagnostic(
            layer_name="model.layers.0.self_attn.q_proj",
            A_a=A,
            B_a=B,
            alpha_a=8.0,
            r_a=4,
            A_b=A.clone(),
            B_b=B.clone(),
            alpha_b=8.0,
            r_b=4,
        )
        assert diag.status == "not_applicable"
        assert diag.shared_basis_score == 0.0
        assert diag.basis_distortion == 0.0
        assert diag.effective_shared_rank == 0

    def test_rank_mismatch_supported(self) -> None:
        torch.manual_seed(1)
        A_a, B_a = _rand_factors(r=2)
        A_b, B_b = _rand_factors(r=6)

        diag = compute_core_space_layer_diagnostic(
            layer_name="model.layers.0.self_attn.v_proj",
            A_a=A_a,
            B_a=B_a,
            alpha_a=4.0,
            r_a=2,
            A_b=A_b,
            B_b=B_b,
            alpha_b=24.0,
            r_b=6,
        )
        assert diag.effective_shared_rank >= 1
        assert diag.status in {"compatible", "marginal", "incompatible", "not_applicable"}


class TestCoreSpacePairAggregation:
    def test_aggregate_counts_and_status(self) -> None:
        layers = [
            CoreSpaceLayerDiagnostic(
                layer_name="layer1",
                shared_basis_score=0.82,
                basis_distortion=0.10,
                effective_shared_rank=4,
                status="compatible",
            ),
            CoreSpaceLayerDiagnostic(
                layer_name="layer2",
                shared_basis_score=0.48,
                basis_distortion=0.42,
                effective_shared_rank=7,
                status="incompatible",
            ),
            CoreSpaceLayerDiagnostic(
                layer_name="layer3",
                shared_basis_score=0.0,
                basis_distortion=0.0,
                effective_shared_rank=0,
                status="not_applicable",
            ),
        ]

        pair = aggregate_core_space_diagnostics(layers)
        assert pair.n_layers_scored == 2
        assert pair.n_layers_incompatible == 1
        assert pair.effective_shared_rank_median in {4, 5, 6, 7}
        assert pair.status in {"compatible", "marginal", "incompatible", "not_applicable"}

    def test_status_spread_regression_across_archetypes(self) -> None:
        """Regression guard: statuses should not collapse to one bucket."""
        self_pair = _pair_diag_for_archetype("self")
        near_pair = _pair_diag_for_archetype("near")
        imbalanced_related = _pair_diag_for_archetype("imbalanced_related")
        random_pair = _pair_diag_for_archetype("random_other")

        # Expected directional behavior:
        # - self/near are usually compatible
        # - imbalanced-related can remain compatible or become marginal
        # - random other should be downgraded from compatible
        assert self_pair.status == "compatible"
        assert near_pair.status in {"compatible", "marginal"}
        assert imbalanced_related.status in {"compatible", "marginal"}
        assert random_pair.status in {"marginal", "incompatible"}

        # Non-trivial spread is the key invariant for this regression.
        statuses = {
            self_pair.status,
            near_pair.status,
            imbalanced_related.status,
            random_pair.status,
        }
        assert len(statuses) >= 2
