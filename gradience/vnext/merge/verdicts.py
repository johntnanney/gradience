"""
Decision logic for merge compatibility.

Translates raw SubspaceMetrics into actionable per-layer assessments and
an aggregate compatibility verdict.  The decision tree follows a six-branch
structure with Frobenius imbalance checked first:

    0. Frobenius imbalance + low/moderate overlap -> IMBALANCED (rebalanced)
    1. Low overlap  -> SAFE (orthogonal subspaces)
    2. High overlap + aligned  -> REDUNDANT (de-dup needed)
    3. High overlap + opposing -> CONFLICTING (danger zone)
    4. Frobenius imbalance + high overlap (remainder) -> IMBALANCED
    5. Everything else         -> SAFE with moderate confidence

Thresholds ship as defaults but can be overridden via CLI flags or a
VerdictThresholds instance.

Additive extension:
- Per-layer over-accumulation score/band/factors
- Pair-level over-accumulation watch/elevated advisory (computed downstream)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from gradience.vnext.merge.over_accumulation import (
    estimate_over_accumulation,
    summarize_over_accumulation,
)
from gradience.vnext.merge.spectral_compat import SubspaceMetrics

# ---------------------------------------------------------------------------
# Enums and thresholds
# ---------------------------------------------------------------------------


class CompatibilityVerdict(Enum):
    SAFE = "safe"
    REDUNDANT = "redundant"
    CONFLICTING = "conflicting"
    IMBALANCED = "imbalanced"


@dataclass(frozen=True)
class VerdictThresholds:
    """Tunable thresholds for verdict decisions.

    Attributes
    ----------
    low_overlap : below this -> orthogonal subspaces
    high_overlap : above this -> significant shared subspace
    aligned : agreement above this -> same direction (redundant)
    conflicting : agreement below this -> opposing (conflicting)
    imbalanced : magnitude ratio above this -> imbalanced
    imbalanced_frob : Frobenius norm ratio above this -> imbalanced
    """

    low_overlap: float = 0.2
    high_overlap: float = 0.5
    aligned: float = 0.5
    conflicting: float = -0.3
    imbalanced: float = 5.0
    imbalanced_frob: float = 5.0

    @classmethod
    def conservative(cls) -> VerdictThresholds:
        """Flag more potential issues. Good for early adoption."""
        return cls(
            low_overlap=0.15,
            high_overlap=0.35,
            aligned=0.6,
            conflicting=-0.2,
            imbalanced=3.0,
            imbalanced_frob=3.0,
        )

    @classmethod
    def permissive(cls) -> VerdictThresholds:
        """Flag only obvious problems. For experienced users."""
        return cls(
            low_overlap=0.3,
            high_overlap=0.7,
            aligned=0.4,
            conflicting=-0.5,
            imbalanced=10.0,
            imbalanced_frob=10.0,
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-layer verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerVerdict:
    """Assessment for one matched layer pair."""

    layer_name: str
    module_type: str
    metrics: SubspaceMetrics
    verdict: CompatibilityVerdict
    confidence: float  # [0, 1] — how clearly metrics match verdict
    recommendation: str  # human-readable
    conflict_dimensions: int  # number of conflicting singular directions
    safe_merge_rank: int  # suggested merged adapter rank
    suggested_strategy: str  # "linear" | "ties" | "dare_ties" | "dare_linear" | "exclude"
    suggested_coefficients: tuple[float, float] | None  # (coeff_a, coeff_b)
    over_accumulation_score: float = 0.0
    over_accumulation_band: str = "low"
    over_accumulation_factors: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "layer_name": self.layer_name,
            "module_type": self.module_type,
            "metrics": self.metrics.to_dict(),
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "conflict_dimensions": self.conflict_dimensions,
            "safe_merge_rank": self.safe_merge_rank,
            "suggested_strategy": self.suggested_strategy,
            "suggested_coefficients": (list(self.suggested_coefficients) if self.suggested_coefficients else None),
            "over_accumulation_score": self.over_accumulation_score,
            "over_accumulation_band": self.over_accumulation_band,
            "over_accumulation_factors": self.over_accumulation_factors
            or {
                "alignment": 0.0,
                "concentration": 0.0,
                "coefficient_exposure": 0.0,
            },
        }
        return d


# ---------------------------------------------------------------------------
# Layer assessment
# ---------------------------------------------------------------------------


def _overlap_confidence(value: float, threshold: float) -> float:
    """Distance from threshold, normalized to [0, 1]."""
    if threshold < 1e-10:
        return 1.0
    return min(abs(value - threshold) / threshold, 1.0)


def assess_layer(
    layer_name: str,
    module_type: str,
    metrics: SubspaceMetrics,
    thresholds: VerdictThresholds | None = None,
) -> LayerVerdict:
    """Six-branch decision tree for layer-level verdict.

    Branch ordering prioritises magnitude imbalance (Frobenius-based) before
    orthogonal-safe, so that adapters with large energy mismatches get
    rebalanced coefficients even when their subspaces are nearly orthogonal.

    When overlap is high, the overlap-based branches (REDUNDANT, CONFLICTING)
    take priority because subspace interaction is the dominant concern.
    """
    if thresholds is None:
        thresholds = VerdictThresholds()

    def _over_accum_fields(assumed_coefficients: tuple[float, float] | None) -> dict[str, Any]:
        est = estimate_over_accumulation(metrics, assumed_coefficients=assumed_coefficients)
        return {
            "over_accumulation_score": est.score,
            "over_accumulation_band": est.band,
            "over_accumulation_factors": est.factors.to_dict(),
        }

    # --- Branch 0 (NEW): Frobenius imbalanced + low-to-moderate overlap ---
    if metrics.frobenius_ratio > thresholds.imbalanced_frob and metrics.mean_overlap < thresholds.high_overlap:
        ratio = metrics.frobenius_ratio
        # coeff_dominant is the smaller value (down-weights the stronger adapter)
        # coeff_subordinate is the larger value (up-weights the weaker adapter)
        coeff_dominant = 1.0 / (1.0 + ratio)
        coeff_subordinate = ratio / (1.0 + ratio)

        if metrics.frobenius_norm_a >= metrics.frobenius_norm_b:
            coefficients = (coeff_dominant, coeff_subordinate)
        else:
            coefficients = (coeff_subordinate, coeff_dominant)

        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.IMBALANCED,
            confidence=_overlap_confidence(metrics.frobenius_ratio, thresholds.imbalanced_frob),
            recommendation=(
                f"Frobenius imbalance ({ratio:.1f}x, "
                f"norms: {metrics.frobenius_norm_a:.1f} vs "
                f"{metrics.frobenius_norm_b:.1f}). "
                f"The weaker adapter's contribution will be drowned out "
                f"with equal coefficients. Suggested rebalancing: "
                f"A={coefficients[0]:.3f}, B={coefficients[1]:.3f}."
            ),
            conflict_dimensions=0,
            safe_merge_rank=max(metrics.effective_rank_a, metrics.effective_rank_b),
            suggested_strategy="linear",
            suggested_coefficients=coefficients,
            **_over_accum_fields(coefficients),
        )

    # --- Branch 1: Orthogonal subspaces ---
    if metrics.mean_overlap < thresholds.low_overlap:
        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.SAFE,
            confidence=_overlap_confidence(metrics.mean_overlap, thresholds.low_overlap),
            recommendation=(
                f"Orthogonal subspaces (overlap={metrics.mean_overlap:.3f}). "
                f"Safe to merge with any method. Combined effective rank: "
                f"{metrics.effective_rank_a + metrics.effective_rank_b}."
            ),
            conflict_dimensions=0,
            safe_merge_rank=metrics.effective_rank_a + metrics.effective_rank_b,
            suggested_strategy="linear",
            suggested_coefficients=(0.5, 0.5),
            **_over_accum_fields((0.5, 0.5)),
        )

    # --- Branch 2: Redundant (high overlap, aligned) ---
    if metrics.mean_overlap > thresholds.high_overlap and metrics.directional_agreement > thresholds.aligned:
        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.REDUNDANT,
            confidence=min(
                _overlap_confidence(metrics.mean_overlap, thresholds.high_overlap),
                abs(metrics.directional_agreement - thresholds.aligned),
            ),
            recommendation=(
                f"High redundancy (overlap={metrics.mean_overlap:.3f}, "
                f"agreement={metrics.directional_agreement:.3f}). "
                f"Adapters learn similar features. TIES recommended to "
                f"deduplicate shared directions. Merged rank ~ "
                f"{max(metrics.effective_rank_a, metrics.effective_rank_b)}."
            ),
            conflict_dimensions=0,
            safe_merge_rank=max(metrics.effective_rank_a, metrics.effective_rank_b),
            suggested_strategy="ties",
            suggested_coefficients=(0.5, 0.5),
            **_over_accum_fields((0.5, 0.5)),
        )

    # --- Branch 3: Conflicting (high overlap, opposing) ---
    if metrics.mean_overlap > thresholds.high_overlap and metrics.directional_agreement < thresholds.conflicting:
        n_conflict = sum(1 for cos_a in metrics.principal_angle_cosines if cos_a > thresholds.high_overlap)

        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.CONFLICTING,
            confidence=min(
                _overlap_confidence(metrics.mean_overlap, thresholds.high_overlap),
                abs(metrics.directional_agreement - thresholds.conflicting),
            ),
            recommendation=(
                f"CONFLICT: {n_conflict} shared direction(s) have opposing effects "
                f"(overlap={metrics.mean_overlap:.3f}, "
                f"agreement={metrics.directional_agreement:.3f}). "
                f"Direct merging will cause cancellation. Options: "
                f"(1) Use DARE with high drop rate, "
                f"(2) Reduce merge coefficient for weaker adapter, "
                f"(3) Exclude this layer from merge."
            ),
            conflict_dimensions=n_conflict,
            safe_merge_rank=metrics.effective_rank_a,
            suggested_strategy="dare_ties",
            suggested_coefficients=None,
            **_over_accum_fields((0.5, 0.5)),
        )

    # --- Branch 4: Frobenius imbalanced (high-overlap remainder) ---
    if metrics.frobenius_ratio > thresholds.imbalanced_frob:
        ratio = metrics.frobenius_ratio
        # coeff_dominant is the smaller value (down-weights the stronger adapter)
        # coeff_subordinate is the larger value (up-weights the weaker adapter)
        coeff_dominant = 1.0 / (1.0 + ratio)
        coeff_subordinate = ratio / (1.0 + ratio)

        if metrics.frobenius_norm_a >= metrics.frobenius_norm_b:
            coefficients = (coeff_dominant, coeff_subordinate)
        else:
            coefficients = (coeff_subordinate, coeff_dominant)

        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.IMBALANCED,
            confidence=_overlap_confidence(metrics.frobenius_ratio, thresholds.imbalanced_frob),
            recommendation=(
                f"Frobenius imbalance ({ratio:.1f}x, "
                f"norms: {metrics.frobenius_norm_a:.1f} vs "
                f"{metrics.frobenius_norm_b:.1f}). "
                f"The weaker adapter's contribution will be drowned out "
                f"with equal coefficients. Suggested rebalancing: "
                f"A={coefficients[0]:.3f}, B={coefficients[1]:.3f}."
            ),
            conflict_dimensions=0,
            safe_merge_rank=max(metrics.effective_rank_a, metrics.effective_rank_b),
            suggested_strategy="linear",
            suggested_coefficients=coefficients,
            **_over_accum_fields(coefficients),
        )

    # --- Branch 5: Moderate / ambiguous -> default SAFE ---
    return LayerVerdict(
        layer_name=layer_name,
        module_type=module_type,
        metrics=metrics,
        verdict=CompatibilityVerdict.SAFE,
        confidence=0.5,
        recommendation=(
            f"Moderate subspace interaction (overlap={metrics.mean_overlap:.3f}, "
            f"agreement={metrics.directional_agreement:.3f}). "
            f"Standard merge methods should work. TIES is a safe default."
        ),
        conflict_dimensions=0,
        safe_merge_rank=(metrics.effective_rank_a + metrics.effective_rank_b) // 2,
        suggested_strategy="ties",
        suggested_coefficients=(0.5, 0.5),
        **_over_accum_fields((0.5, 0.5)),
    )


# ---------------------------------------------------------------------------
# Aggregate assessment
# ---------------------------------------------------------------------------


def assess_overall(
    layer_verdicts: list[LayerVerdict],
) -> tuple[CompatibilityVerdict, float, list[str]]:
    """Compute aggregate compatibility verdict, score, and recommendations.

    Parameters
    ----------
    layer_verdicts : per-layer assessments

    Returns
    -------
    overall_verdict : worst-case verdict across layers
    compatibility_score : energy-weighted mean overlap [0, 1]
        (0 = fully orthogonal, 1 = fully overlapping)
    recommendations : list of human-readable recommendation strings
    """
    if not layer_verdicts:
        return CompatibilityVerdict.SAFE, 0.0, ["No shared layers to analyze."]

    # Priority ordering: CONFLICTING > IMBALANCED > REDUNDANT > SAFE
    verdict_priority = {
        CompatibilityVerdict.SAFE: 0,
        CompatibilityVerdict.REDUNDANT: 1,
        CompatibilityVerdict.IMBALANCED: 2,
        CompatibilityVerdict.CONFLICTING: 3,
    }

    overall = max(
        (lv.verdict for lv in layer_verdicts),
        key=lambda v: verdict_priority[v],
    )

    # Energy-weighted mean overlap
    total_energy = 0.0
    weighted_overlap = 0.0
    for lv in layer_verdicts:
        # Weight by the larger Frobenius norm: layers with bigger updates
        # contribute more to the aggregate score.
        energy = max(lv.metrics.frobenius_norm_a, lv.metrics.frobenius_norm_b)
        total_energy += energy
        weighted_overlap += energy * lv.metrics.mean_overlap

    score = weighted_overlap / total_energy if total_energy > 0 else 0.0

    # Build recommendations
    recommendations: list[str] = []

    conflicting = [lv for lv in layer_verdicts if lv.verdict == CompatibilityVerdict.CONFLICTING]
    redundant = [lv for lv in layer_verdicts if lv.verdict == CompatibilityVerdict.REDUNDANT]
    imbalanced = [lv for lv in layer_verdicts if lv.verdict == CompatibilityVerdict.IMBALANCED]

    if conflicting:
        names = ", ".join(lv.layer_name.split(".")[-1] for lv in conflicting[:3])
        recommendations.append(
            f"{len(conflicting)} layer(s) show subspace conflicts ({names}). "
            f"Consider DARE with high drop rate or excluding these layers."
        )

    if redundant:
        recommendations.append(
            f"{len(redundant)} layer(s) show high redundancy. TIES merge recommended to deduplicate shared features."
        )

    if imbalanced:
        recommendations.append(
            f"{len(imbalanced)} layer(s) show magnitude imbalance. "
            f"Use rebalanced coefficients or normalize adapter scales."
        )

    if not conflicting and not redundant and not imbalanced:
        recommendations.append(
            "Adapters are spectrally compatible. Linear merge (equal coefficients) should preserve both signals."
        )

    over_accum_summary = summarize_over_accumulation(
        layer_scores=[lv.over_accumulation_score for lv in layer_verdicts],
        layer_bands=[lv.over_accumulation_band for lv in layer_verdicts],
        non_overlap_issue_count=len(conflicting) + len(imbalanced),
    )
    if over_accum_summary.advisory != "none" and over_accum_summary.summary:
        recommendations.append(over_accum_summary.summary)

    return overall, score, recommendations
