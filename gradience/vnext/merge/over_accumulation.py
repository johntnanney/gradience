"""Over-accumulation diagnostic helpers for merge audit.

This module implements a diagnostic-only additive signal that estimates when
high-overlap layer pairs may over-amplify shared directions under naive linear
combination.

The intent is cautionary:
- This is a risk estimate, not proof of harmful inflation.
- It does not replace the existing verdict taxonomy.
- It should stay lightweight and reuse existing spectral metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gradience.vnext.merge.spectral_compat import SubspaceMetrics
from gradience.vnext.merge.spectral_theory import compute_rankr_interaction_geometry


def _clamp01(value: float) -> float:
    """Clamp a scalar into [0, 1]."""
    return min(max(float(value), 0.0), 1.0)


def _normalize_coefficients(assumed_coefficients: tuple[float, float] | None) -> tuple[float, float]:
    """Normalize merge coefficients for exposure estimation.

    If no coefficients are provided, defaults to symmetric naive merge.
    """
    if assumed_coefficients is None:
        return (0.5, 0.5)

    coeff_a = max(float(assumed_coefficients[0]), 0.0)
    coeff_b = max(float(assumed_coefficients[1]), 0.0)
    total = coeff_a + coeff_b
    if total <= 1e-12:
        return (0.5, 0.5)
    return (coeff_a / total, coeff_b / total)


def _alignment_component(metrics: SubspaceMetrics) -> float:
    """Estimate strength of aligned shared subspace."""
    overlap_strength = _clamp01(0.6 * metrics.mean_overlap + 0.4 * metrics.max_overlap)
    agreement_aligned = _clamp01(max(metrics.directional_agreement, 0.0))
    # Require both overlap and positive directional alignment.
    return _clamp01(overlap_strength * (0.4 + 0.6 * agreement_aligned))


def _adapter_concentration(stable_rank: float, effective_rank: int, nominal_rank: int) -> float:
    """Estimate spectral concentration for one adapter in one layer."""
    eff = max(float(effective_rank), 1.0)
    stable = min(max(float(stable_rank), 1.0), eff)

    # Lower stable-rank relative to effective-rank means stronger top-mode concentration.
    stable_concentration = 1.0 - (stable / eff)

    # Lower effective utilization of nominal rank also suggests concentrated mass.
    if nominal_rank > 0:
        utilization = _clamp01(float(effective_rank) / float(nominal_rank))
    else:
        utilization = 1.0
    utilization_concentration = 1.0 - utilization

    return _clamp01(0.7 * stable_concentration + 0.3 * utilization_concentration)


def _concentration_component(metrics: SubspaceMetrics) -> float:
    """Estimate shared spectral concentration across the pair."""
    conc_a = _adapter_concentration(metrics.stable_rank_a, metrics.effective_rank_a, metrics.nominal_rank_a)
    conc_b = _adapter_concentration(metrics.stable_rank_b, metrics.effective_rank_b, metrics.nominal_rank_b)
    return _clamp01(0.5 * (conc_a + conc_b))


def _coefficient_exposure_component(coeff_a: float, coeff_b: float) -> float:
    """Estimate how strongly coefficients reinforce shared aligned directions.

    Uses 4ab in [0, 1]:
    - 1.0 when both adapters are equally weighted (max reinforcement opportunity)
    - 0.0 when one side is effectively muted
    """
    return _clamp01(4.0 * coeff_a * coeff_b)


@dataclass(frozen=True)
class OverAccumulationFactors:
    """Sub-components of the over-accumulation score."""

    alignment: float
    concentration: float
    coefficient_exposure: float

    def to_dict(self) -> dict[str, float]:
        return {
            "alignment": round(self.alignment, 4),
            "concentration": round(self.concentration, 4),
            "coefficient_exposure": round(self.coefficient_exposure, 4),
        }


@dataclass(frozen=True)
class OverAccumulationEstimate:
    """Per-layer over-accumulation estimate."""

    score: float
    band: str  # "low" | "watch" | "high"
    factors: OverAccumulationFactors

    def to_dict(self) -> dict[str, Any]:
        return {
            "over_accumulation_score": round(self.score, 4),
            "over_accumulation_band": self.band,
            "over_accumulation_factors": self.factors.to_dict(),
        }


@dataclass(frozen=True)
class OverAccumulationFactorsV2:
    """
    OA-v2 factor decomposition.

    OA-v2 is interaction-first:
    - interaction_primary from full rank-r spectral interaction
    - concentration as secondary modulation only
    """

    interaction_primary_raw: float
    spectral_overlap_weighted: float
    interaction_top1: float
    interaction_top3_mean: float
    directional_gate: float
    coefficient_exposure: float
    interaction_primary: float
    concentration_secondary: float
    input_confidence: float
    degraded_input: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_primary_raw": round(self.interaction_primary_raw, 4),
            "spectral_overlap_weighted": round(self.spectral_overlap_weighted, 4),
            "interaction_top1": round(self.interaction_top1, 4),
            "interaction_top3_mean": round(self.interaction_top3_mean, 4),
            "directional_gate": round(self.directional_gate, 4),
            "coefficient_exposure": round(self.coefficient_exposure, 4),
            "interaction_primary": round(self.interaction_primary, 4),
            "concentration_secondary": round(self.concentration_secondary, 4),
            "input_confidence": round(self.input_confidence, 4),
            "degraded_input": bool(self.degraded_input),
        }


@dataclass(frozen=True)
class OverAccumulationEstimateV2:
    """Preview-only OA-v2 layer estimate (parallel to OA-v1, not authoritative)."""

    score: float
    band: str  # "low" | "watch" | "high"
    factors: OverAccumulationFactorsV2

    def to_dict(self) -> dict[str, Any]:
        return {
            "over_accumulation_v2_score": round(self.score, 4),
            "over_accumulation_v2_band": self.band,
            "over_accumulation_v2_factors": self.factors.to_dict(),
        }


@dataclass(frozen=True)
class OverAccumulationPairSummary:
    """Pair-level advisory synthesized from per-layer estimates."""

    advisory: str  # "none" | "watch" | "elevated"
    summary: str
    high_risk_layer_count: int
    watch_layer_count: int
    max_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "over_accumulation_advisory": self.advisory,
            "over_accumulation_summary": self.summary,
            "high_risk_layer_count": int(self.high_risk_layer_count),
            "watch_layer_count": int(self.watch_layer_count),
            "max_over_accumulation_score": round(self.max_score, 4),
        }


def estimate_over_accumulation(
    metrics: SubspaceMetrics,
    *,
    assumed_coefficients: tuple[float, float] | None = None,
) -> OverAccumulationEstimate:
    """Estimate layer-level risk of shared-direction over-accumulation."""
    coeff_a, coeff_b = _normalize_coefficients(assumed_coefficients)

    alignment = _alignment_component(metrics)
    concentration = _concentration_component(metrics)
    coeff_exposure = _coefficient_exposure_component(coeff_a, coeff_b)

    # Alignment gates the score; concentration dominates exposure in phase 1.
    score = _clamp01(alignment * (0.7 * concentration + 0.3 * coeff_exposure))

    if alignment >= 0.65 and score >= 0.60:
        band = "high"
    elif alignment >= 0.45 and score >= 0.35:
        band = "watch"
    else:
        band = "low"

    return OverAccumulationEstimate(
        score=score,
        band=band,
        factors=OverAccumulationFactors(
            alignment=alignment,
            concentration=concentration,
            coefficient_exposure=coeff_exposure,
        ),
    )


def estimate_over_accumulation_v2(
    metrics: SubspaceMetrics,
    *,
    assumed_coefficients: tuple[float, float] | None = None,
) -> OverAccumulationEstimateV2:
    """
    Experimental OA-v2 score (parallel line; does not replace OA-v1).

    Contract:
      directional_gate = clamp(max(direction_agreement,0),0,1)
      coeff_exposure = 4ab
      interaction_primary = directional_gate * coeff_exposure * spectral_overlap_weighted
      score_v2 = clamp(interaction_primary * (0.85 + 0.15 * concentration_secondary), 0, 1)
    """
    coeff_a, coeff_b = _normalize_coefficients(assumed_coefficients)
    geometry = compute_rankr_interaction_geometry(metrics)
    directional_gate = _clamp01(max(float(metrics.directional_agreement), 0.0))
    coeff_exposure = _coefficient_exposure_component(coeff_a, coeff_b)
    concentration_secondary = _concentration_component(metrics)

    interaction_primary = directional_gate * coeff_exposure * geometry.spectral_overlap_weighted
    score_v2 = _clamp01(interaction_primary * (0.85 + 0.15 * concentration_secondary))

    if interaction_primary >= 0.60 and score_v2 >= 0.60:
        band = "high"
    elif interaction_primary >= 0.35 and score_v2 >= 0.35:
        band = "watch"
    else:
        band = "low"

    # Degraded input marks older artifacts lacking v2 geometry fields.
    input_confidence = 0.6 if geometry.degraded_input else 1.0

    return OverAccumulationEstimateV2(
        score=score_v2,
        band=band,
        factors=OverAccumulationFactorsV2(
            interaction_primary_raw=geometry.interaction_primary_raw,
            spectral_overlap_weighted=geometry.spectral_overlap_weighted,
            interaction_top1=geometry.interaction_top1,
            interaction_top3_mean=geometry.interaction_top3_mean,
            directional_gate=directional_gate,
            coefficient_exposure=coeff_exposure,
            interaction_primary=_clamp01(interaction_primary),
            concentration_secondary=concentration_secondary,
            input_confidence=input_confidence,
            degraded_input=geometry.degraded_input,
        ),
    )


def summarize_over_accumulation(
    *,
    layer_scores: list[float],
    layer_bands: list[str],
    non_overlap_issue_count: int = 0,
) -> OverAccumulationPairSummary:
    """Aggregate layer-level over-accumulation estimates to a pair advisory."""
    if not layer_scores or not layer_bands:
        return OverAccumulationPairSummary(
            advisory="none",
            summary="",
            high_risk_layer_count=0,
            watch_layer_count=0,
            max_score=0.0,
        )

    scores = [_clamp01(s) for s in layer_scores]
    n_layers = max(min(len(scores), len(layer_bands)), 1)
    high_count = sum(1 for b in layer_bands[:n_layers] if b == "high")
    watch_count = sum(1 for b in layer_bands[:n_layers] if b == "watch")
    max_score = max(scores[:n_layers]) if scores else 0.0

    high_fraction = high_count / n_layers
    watch_fraction = (high_count + watch_count) / n_layers

    if high_count > 0 and (high_fraction >= 0.15 or max_score >= 0.75):
        advisory = "elevated"
    elif high_count > 0 or watch_fraction >= 0.25:
        advisory = "watch"
    else:
        advisory = "none"

    if advisory == "none":
        summary = ""
    else:
        summary = (
            "High-overlap layer(s) may be susceptible to shared-direction inflation under naive linear merge "
            f"(high={high_count}, watch={watch_count})."
        )
        if non_overlap_issue_count > 0:
            summary += " Existing conflict/norm-imbalance findings remain the primary concern in this pair."

    return OverAccumulationPairSummary(
        advisory=advisory,
        summary=summary,
        high_risk_layer_count=high_count,
        watch_layer_count=watch_count,
        max_score=max_score,
    )
