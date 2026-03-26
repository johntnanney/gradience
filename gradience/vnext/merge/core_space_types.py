"""Typed containers for optional core-space diagnostics.

These objects are internal, additive diagnostics used by the merge-audit
opt-in path. They are intentionally not part of the frozen public v1 artifact
contracts unless explicitly mapped into optional fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

CORE_SPACE_STATUS_VALUES = frozenset(
    {
        "compatible",
        "marginal",
        "incompatible",
        "not_applicable",
    }
)


def _validate_status(status: str, field: str) -> str:
    if status not in CORE_SPACE_STATUS_VALUES:
        raise ValueError(f"{field} must be one of {sorted(CORE_SPACE_STATUS_VALUES)}, got '{status}'")
    return status


@dataclass(frozen=True)
class CoreSpaceLayerDiagnostic:
    """Per-layer shared-basis diagnostic."""

    layer_name: str
    shared_basis_score: float
    basis_distortion: float
    effective_shared_rank: int
    status: str

    def __post_init__(self) -> None:
        _validate_status(self.status, "status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "shared_basis_score": float(self.shared_basis_score),
            "basis_distortion": float(self.basis_distortion),
            "effective_shared_rank": int(self.effective_shared_rank),
            "status": self.status,
        }


@dataclass(frozen=True)
class CoreSpacePairDiagnostic:
    """Pair-level aggregate of shared-basis diagnostics."""

    shared_basis_score_mean: float
    basis_distortion_mean: float
    effective_shared_rank_median: int
    status: str
    n_layers_scored: int
    n_layers_incompatible: int
    layer_diagnostics: tuple[CoreSpaceLayerDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _validate_status(self.status, "status")

    def to_dict(self, *, include_layers: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "shared_basis_score_mean": float(self.shared_basis_score_mean),
            "basis_distortion_mean": float(self.basis_distortion_mean),
            "effective_shared_rank_median": int(self.effective_shared_rank_median),
            "status": self.status,
            "n_layers_scored": int(self.n_layers_scored),
            "n_layers_incompatible": int(self.n_layers_incompatible),
        }
        if include_layers:
            d["layers"] = [ld.to_dict() for ld in self.layer_diagnostics]
        return d

    @classmethod
    def from_layers(
        cls,
        layer_diagnostics: list[CoreSpaceLayerDiagnostic],
        *,
        status: str,
    ) -> CoreSpacePairDiagnostic:
        """Build pair-level aggregate fields from per-layer diagnostics."""
        scored = [ld for ld in layer_diagnostics if ld.status != "not_applicable"]
        if not scored:
            return cls(
                shared_basis_score_mean=0.0,
                basis_distortion_mean=0.0,
                effective_shared_rank_median=0,
                status="not_applicable",
                n_layers_scored=0,
                n_layers_incompatible=0,
                layer_diagnostics=tuple(layer_diagnostics),
            )

        mean_score = sum(ld.shared_basis_score for ld in scored) / len(scored)
        mean_distortion = sum(ld.basis_distortion for ld in scored) / len(scored)
        median_rank = int(round(median(ld.effective_shared_rank for ld in scored)))
        n_incompatible = sum(1 for ld in scored if ld.status == "incompatible")

        return cls(
            shared_basis_score_mean=mean_score,
            basis_distortion_mean=mean_distortion,
            effective_shared_rank_median=median_rank,
            status=_validate_status(status, "status"),
            n_layers_scored=len(scored),
            n_layers_incompatible=n_incompatible,
            layer_diagnostics=tuple(layer_diagnostics),
        )
