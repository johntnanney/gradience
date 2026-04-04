"""
Merge strategy recommendation engine.

Two-stage architecture:

**Stage A — Diagnosis** (pure spectral analysis, no policy decisions):
  ``diagnose_layer`` and ``diagnose_pair`` extract structured facts from
  a MergeAuditReport: magnitude imbalance, overlap/conflict/redundancy
  classification, compression needs, and structural risk summary.

**Stage B — Policy** (decision logic, uses diagnosis + eligibility context):
  ``_apply_layer_policy`` and ``_apply_policy`` translate a diagnosis into
  concrete strategy choices with tuned parameters.  Eligibility context
  modulates the policy: weak sources get conservative treatment, strong
  sources get full rebalancing, unknown sources get warnings.

The public API (``recommend_merge``, ``_recommend_layer``) composes both
stages.  The separation pays off immediately: diagnosis is reusable and
testable independently of policy, and policy is the clear hook for
future eligibility-aware refinements.

Usage::

    from gradience.vnext.merge.recommend import recommend_merge, diagnose_pair

    report = merge_audit("./adapter_a", "./adapter_b")

    # Stage A only — useful for inspection / custom policy
    diagnosis = diagnose_pair(report)

    # Both stages — full recommendation
    rec = recommend_merge(report)
    print(rec.format_cli())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from gradience.vnext.merge.eligibility import EligibilityStatus
from gradience.vnext.merge.over_accumulation import summarize_over_accumulation
from gradience.vnext.merge.report import _shorten_layer_name

# ===================================================================
# Data structures
# ===================================================================


# ---------------------------------------------------------------------------
# Stage A outputs — diagnosis (pure facts, no policy decisions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerDiagnosis:
    """Spectral diagnosis for one layer pair.

    Contains only facts derived from spectral metrics — no strategy
    choices or coefficient decisions.  Produced by ``diagnose_layer``.
    """

    layer_name: str
    verdict: str  # "safe" | "redundant" | "conflicting" | "imbalanced"
    confidence: float
    risk_level: str  # "low" | "medium" | "high"

    # Key spectral metrics (extracted for convenience)
    mean_overlap: float
    directional_agreement: float
    magnitude_ratio: float
    n_conflict: int
    n_principal: int

    # Compression diagnosis
    compress_first: bool
    compress_target_rank_a: int | None
    compress_target_rank_b: int | None

    # Upstream suggestion from the verdict engine (may be None)
    suggested_coefficients: tuple[float, float] | None
    over_accumulation_score: float
    over_accumulation_band: str  # "low" | "watch" | "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "risk_level": self.risk_level,
            "mean_overlap": round(self.mean_overlap, 4),
            "directional_agreement": round(self.directional_agreement, 4),
            "magnitude_ratio": round(self.magnitude_ratio, 4),
            "n_conflict": self.n_conflict,
            "n_principal": self.n_principal,
            "compress_first": self.compress_first,
            "compress_target_rank_a": self.compress_target_rank_a,
            "compress_target_rank_b": self.compress_target_rank_b,
            "suggested_coefficients": list(self.suggested_coefficients) if self.suggested_coefficients else None,
            "over_accumulation_score": round(self.over_accumulation_score, 4),
            "over_accumulation_band": self.over_accumulation_band,
        }


@dataclass(frozen=True)
class EligibilityContext:
    """Parsed eligibility state for the adapter pair.

    Produced by ``_parse_eligibility`` from the report's ``source_qa``.
    """

    status_a: EligibilityStatus | None
    status_b: EligibilityStatus | None

    @property
    def has_data(self) -> bool:
        """True if any source QA data was provided."""
        return self.status_a is not None or self.status_b is not None

    @property
    def any_weak(self) -> bool:
        return self.status_a == EligibilityStatus.FLAGGED_WEAK or self.status_b == EligibilityStatus.FLAGGED_WEAK

    @property
    def both_weak(self) -> bool:
        return self.status_a == EligibilityStatus.FLAGGED_WEAK and self.status_b == EligibilityStatus.FLAGGED_WEAK

    @property
    def both_eligible(self) -> bool:
        return self.status_a == EligibilityStatus.ELIGIBLE and self.status_b == EligibilityStatus.ELIGIBLE

    @property
    def any_unverified(self) -> bool:
        """True if either adapter has no behavioral evaluation."""
        return (
            self.status_a == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL
            or self.status_b == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL
        )


@dataclass(frozen=True)
class PairDiagnosis:
    """Aggregate diagnosis for an adapter pair.

    Combines per-layer diagnoses with eligibility context.  Produced
    by ``diagnose_pair``.  This is a complete Stage A output — everything
    downstream policy needs to make decisions.
    """

    layer_diagnoses: tuple[LayerDiagnosis, ...]
    overall_risk: str  # "low" | "medium" | "high"
    compression_needed: bool
    n_layers_needing_compression: int
    eligibility: EligibilityContext
    over_accumulation_advisory: str = "none"  # "none" | "watch" | "elevated"
    over_accumulation_summary: str = ""
    over_accumulation_high_risk_layers: int = 0
    over_accumulation_watch_layers: int = 0
    over_accumulation_max_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_diagnoses": [ld.to_dict() for ld in self.layer_diagnoses],
            "overall_risk": self.overall_risk,
            "compression_needed": self.compression_needed,
            "n_layers_needing_compression": self.n_layers_needing_compression,
            "over_accumulation_advisory": self.over_accumulation_advisory,
            "over_accumulation_summary": self.over_accumulation_summary,
            "over_accumulation_high_risk_layers": self.over_accumulation_high_risk_layers,
            "over_accumulation_watch_layers": self.over_accumulation_watch_layers,
            "over_accumulation_max_score": round(self.over_accumulation_max_score, 4),
            "eligibility": {
                "status_a": self.eligibility.status_a.value if self.eligibility.status_a else None,
                "status_b": self.eligibility.status_b.value if self.eligibility.status_b else None,
                "has_data": self.eligibility.has_data,
                "any_weak": self.eligibility.any_weak,
                "both_weak": self.eligibility.both_weak,
                "both_eligible": self.eligibility.both_eligible,
            },
        }


# ---------------------------------------------------------------------------
# Stage B outputs — recommendations (policy decisions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerRecommendation:
    """Actionable recommendation for one layer pair."""

    layer_name: str
    verdict: str  # "safe" | "redundant" | "conflicting" | "imbalanced"
    strategy: str  # "linear" | "ties" | "dare_linear" | "dare_ties"
    coefficients: tuple[float, float]  # (coeff_a, coeff_b)
    trim_fraction: float  # TIES trim or DARE drop rate
    risk_level: str  # "low" | "medium" | "high"
    compress_first: bool  # True if pre-compression recommended
    compress_target_rank_a: int | None  # suggested rank for adapter A
    compress_target_rank_b: int | None  # suggested rank for adapter B
    reasoning: str  # one-line explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "verdict": self.verdict,
            "strategy": self.strategy,
            "coefficients": list(self.coefficients),
            "trim_fraction": round(self.trim_fraction, 4),
            "risk_level": self.risk_level,
            "compress_first": self.compress_first,
            "compress_target_rank_a": self.compress_target_rank_a,
            "compress_target_rank_b": self.compress_target_rank_b,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class MergeRecommendation:
    """Complete merge recommendation from an audit report."""

    overall_strategy: str  # dominant strategy across layers
    overall_risk: str  # "low" | "medium" | "high"
    layer_recommendations: tuple[LayerRecommendation, ...]
    compression_needed: bool
    n_layers_needing_compression: int
    fallback_strategies: tuple[str, ...]  # alternative approaches
    warnings: tuple[str, ...] = ()  # hard warnings from eligibility screening

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_strategy": self.overall_strategy,
            "overall_risk": self.overall_risk,
            "layer_recommendations": [lr.to_dict() for lr in self.layer_recommendations],
            "compression_needed": self.compression_needed,
            "n_layers_needing_compression": self.n_layers_needing_compression,
            "fallback_strategies": list(self.fallback_strategies),
            "warnings": list(self.warnings),
        }

    def format_cli(
        self,
        adapter_a_path: str = "./adapter-a",
        adapter_b_path: str = "./adapter-b",
    ) -> str:
        """Format recommendation as CLI-ready output."""
        return format_recommendation(self, adapter_a_path, adapter_b_path)


# ===================================================================
# Parameter computation — deterministic functions of spectral metrics
# ===================================================================


def rebalance_coefficients(magnitude_ratio: float) -> tuple[float, float]:
    """Compute merge coefficients that compensate for magnitude imbalance.

    Given the ratio ``max(σ₁_a, σ₁_b) / min(σ₁_a, σ₁_b)``, returns
    ``(coeff_dominant, coeff_subordinate)`` where the dominant (stronger)
    adapter is down-weighted and the subordinate (weaker) adapter is
    up-weighted proportionally to the imbalance.

    This is the per-layer rebalancing used by the ``imbalanced`` verdict path
    and is also available for manual coefficient tuning.

    Parameters
    ----------
    magnitude_ratio : ratio ≥ 1.0 of the larger to smaller leading singular value

    Returns
    -------
    (coeff_dominant, coeff_subordinate) — sum to 1.0; coeff_dominant < coeff_subordinate
    """
    ratio = max(magnitude_ratio, 1.0)
    coeff_dominant = round(1.0 / (1.0 + ratio), 4)
    coeff_subordinate = 1.0 - coeff_dominant
    return (coeff_dominant, coeff_subordinate)


def norm_equalized_coefficients(
    frobenius_norm_a: float,
    frobenius_norm_b: float,
    weight_a: float = 0.5,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Compute effective per-adapter scale factors for norm-equalized merging.

    Both adapters are rescaled to their geometric-mean Frobenius norm.
    The returned coefficients incorporate both the norm-equalization scaling
    and the user-specified weight, so a simple ``coeff_a * dW_a + coeff_b * dW_b``
    reproduces the norm-equalized merge.

    Parameters
    ----------
    frobenius_norm_a : ‖ΔW_a‖_F
    frobenius_norm_b : ‖ΔW_b‖_F
    weight_a : mixing weight for adapter A (weight_b = 1 - weight_a)
    eps : numerical stability

    Returns
    -------
    (effective_coeff_a, effective_coeff_b)
    """
    na = max(frobenius_norm_a, eps)
    nb = max(frobenius_norm_b, eps)
    target = (na * nb) ** 0.5
    scale_a = target / na
    scale_b = target / nb
    weight_b = 1.0 - weight_a
    return (round(weight_a * scale_a, 4), round(weight_b * scale_b, 4))


def _compute_trim_fraction(mean_overlap: float, high_overlap_threshold: float = 0.5) -> float:
    """Compute TIES trim fraction from overlap.

    Scales linearly from 0.1 (at threshold) to 0.5 (at overlap=1.0).
    Higher overlap means more redundancy, so trim more aggressively.
    """
    if mean_overlap <= high_overlap_threshold:
        return 0.0
    t = (mean_overlap - high_overlap_threshold) / (1.0 - high_overlap_threshold)
    return round(0.1 + 0.4 * t, 3)


def _compute_dare_drop_rate(
    n_conflict: int,
    n_principal: int,
) -> float:
    """Compute DARE drop rate from conflict count.

    Scales from 0.15 (1 conflict) to 0.5 (all dimensions conflicting).
    More conflict → more aggressive random dropout to break interference.
    """
    if n_conflict <= 0 or n_principal <= 0:
        return 0.15
    t = min(n_conflict / max(n_principal, 1), 1.0)
    return round(0.15 + 0.35 * t, 3)


def _should_compress(
    mean_overlap: float,
    effective_rank: int,
    nominal_rank: int,
    overlap_threshold: float = 0.5,
    utilization_threshold: float = 0.4,
) -> bool:
    """Check if pre-compression is recommended for an adapter.

    Compression helps when:
    1. High overlap with the other adapter (redundant dimensions)
    2. The adapter itself is over-provisioned (low utilization)
    """
    if nominal_rank <= 0:
        return False
    utilization = effective_rank / nominal_rank
    return mean_overlap > overlap_threshold and utilization < utilization_threshold


def _compression_target(effective_rank: int, min_rank: int = 4) -> int:
    """Suggest target rank for compression.

    Uses effective_rank * 1.1 (10% safety margin), minimum 4.
    """
    return max(math.ceil(effective_rank * 1.1), min_rank)


def _risk_level(verdict: str, confidence: float) -> str:
    """Map verdict + confidence to a risk level."""
    if verdict == "conflicting":
        return "high"
    if verdict == "imbalanced":
        return "medium" if confidence > 0.5 else "high"
    if verdict == "redundant":
        return "medium"
    return "low"


# ===================================================================
# Stage A — Diagnosis (pure spectral analysis, no policy decisions)
# ===================================================================


def diagnose_layer(lv_dict: dict[str, Any]) -> LayerDiagnosis:
    """Diagnose one layer pair from its verdict dict.

    Extracts and computes structural facts: risk level, compression
    needs, conflict counts.  Does NOT choose a merge strategy — that
    is Stage B (policy).

    Parameters
    ----------
    lv_dict : per-layer verdict dict from MergeAuditReport.layer_verdicts

    Returns
    -------
    LayerDiagnosis with all facts needed by the policy stage.
    """
    verdict = lv_dict["verdict"]
    metrics = lv_dict.get("metrics", {})
    layer_name = lv_dict["layer_name"]

    mean_overlap = metrics.get("mean_overlap", 0.0)
    directional_agreement = metrics.get("directional_agreement", 0.0)
    magnitude_ratio = metrics.get("magnitude_ratio", 1.0)
    effective_rank_a = metrics.get("effective_rank_a", 0)
    effective_rank_b = metrics.get("effective_rank_b", 0)
    nominal_rank_a = metrics.get("nominal_rank_a", 0)
    nominal_rank_b = metrics.get("nominal_rank_b", 0)
    n_conflict = lv_dict.get("conflict_dimensions", 0)
    confidence = lv_dict.get("confidence", 0.5)
    suggested_coeffs = lv_dict.get("suggested_coefficients")
    over_accumulation_score = float(lv_dict.get("over_accumulation_score", 0.0))
    over_accumulation_band = str(lv_dict.get("over_accumulation_band", "low"))

    # Number of principal angles available
    pac = metrics.get("principal_angle_cosines", ())
    n_principal = len(pac) if pac else max(effective_rank_a, effective_rank_b, 1)

    # Compression diagnosis
    compress_a = _should_compress(mean_overlap, effective_rank_a, nominal_rank_a)
    compress_b = _should_compress(mean_overlap, effective_rank_b, nominal_rank_b)
    target_a = _compression_target(effective_rank_a) if compress_a else None
    target_b = _compression_target(effective_rank_b) if compress_b else None

    risk = _risk_level(verdict, confidence)

    return LayerDiagnosis(
        layer_name=layer_name,
        verdict=verdict,
        confidence=confidence,
        risk_level=risk,
        mean_overlap=mean_overlap,
        directional_agreement=directional_agreement,
        magnitude_ratio=magnitude_ratio,
        n_conflict=n_conflict,
        n_principal=n_principal,
        compress_first=compress_a or compress_b,
        compress_target_rank_a=target_a,
        compress_target_rank_b=target_b,
        suggested_coefficients=tuple(suggested_coeffs) if suggested_coeffs else None,
        over_accumulation_score=over_accumulation_score,
        over_accumulation_band=over_accumulation_band,
    )


def _parse_eligibility(report: Any) -> EligibilityContext:
    """Parse eligibility status from a report's source_qa field.

    Parameters
    ----------
    report : MergeAuditReport (or any object with an optional ``source_qa`` attr)

    Returns
    -------
    EligibilityContext with parsed status for both adapters.
    """
    source_qa = getattr(report, "source_qa", None)

    if source_qa is None or not source_qa:
        return EligibilityContext(status_a=None, status_b=None)

    def _status(key: str) -> EligibilityStatus | None:
        entry = source_qa.get(key)
        if entry is None:
            return None
        raw = entry.get("status", EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL.value)
        try:
            return EligibilityStatus(raw)
        except ValueError:
            return EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL

    return EligibilityContext(
        status_a=_status("adapter_a"),
        status_b=_status("adapter_b"),
    )


def diagnose_pair(report: Any) -> PairDiagnosis:
    """Run Stage A diagnosis on an audit report.

    Produces a ``PairDiagnosis`` containing per-layer diagnoses,
    aggregate risk, compression summary, and eligibility context.
    No strategy or coefficient decisions are made here.

    Parameters
    ----------
    report : MergeAuditReport
        Output from ``merge_audit()``.

    Returns
    -------
    PairDiagnosis — complete Stage A output.
    """
    layer_diags = tuple(diagnose_layer(lv) for lv in report.layer_verdicts)

    # Aggregate risk: worst case across layers
    risk_order = {"low": 0, "medium": 1, "high": 2}
    overall_risk = max(
        (ld.risk_level for ld in layer_diags),
        key=lambda r: risk_order.get(r, 0),
        default="low",
    )

    n_compress = sum(1 for ld in layer_diags if ld.compress_first)
    over_accum = summarize_over_accumulation(
        layer_scores=[ld.over_accumulation_score for ld in layer_diags],
        layer_bands=[ld.over_accumulation_band for ld in layer_diags],
        non_overlap_issue_count=sum(1 for ld in layer_diags if ld.verdict in ("conflicting", "imbalanced")),
    )

    eligibility = _parse_eligibility(report)

    return PairDiagnosis(
        layer_diagnoses=layer_diags,
        overall_risk=overall_risk,
        compression_needed=n_compress > 0,
        n_layers_needing_compression=n_compress,
        eligibility=eligibility,
        over_accumulation_advisory=over_accum.advisory,
        over_accumulation_summary=over_accum.summary,
        over_accumulation_high_risk_layers=over_accum.high_risk_layer_count,
        over_accumulation_watch_layers=over_accum.watch_layer_count,
        over_accumulation_max_score=over_accum.max_score,
    )


# ===================================================================
# Stage B — Policy (decision logic, uses diagnosis + eligibility)
# ===================================================================


def _apply_layer_policy(
    diag: LayerDiagnosis,
    eligibility: EligibilityContext,
) -> LayerRecommendation:
    """Apply recommendation policy to a single layer diagnosis.

    Maps a ``LayerDiagnosis`` to a concrete ``LayerRecommendation``
    (strategy, coefficients, trim fraction) based on the verdict and
    eligibility context.

    Parameters
    ----------
    diag : LayerDiagnosis from Stage A
    eligibility : EligibilityContext for the adapter pair

    Returns
    -------
    LayerRecommendation with strategy and parameter choices.
    """
    verdict = diag.verdict

    # --- Strategy and parameter selection per verdict ---

    if verdict == "safe":
        return LayerRecommendation(
            layer_name=diag.layer_name,
            verdict=verdict,
            strategy="linear",
            coefficients=(0.5, 0.5),
            trim_fraction=0.0,
            risk_level=diag.risk_level,
            compress_first=diag.compress_first,
            compress_target_rank_a=diag.compress_target_rank_a,
            compress_target_rank_b=diag.compress_target_rank_b,
            reasoning=(f"Orthogonal subspaces (overlap={diag.mean_overlap:.3f}). Linear merge preserves both signals."),
        )

    if verdict == "redundant":
        trim = _compute_trim_fraction(diag.mean_overlap)
        return LayerRecommendation(
            layer_name=diag.layer_name,
            verdict=verdict,
            strategy="ties",
            coefficients=(0.5, 0.5),
            trim_fraction=trim,
            risk_level=diag.risk_level,
            compress_first=diag.compress_first,
            compress_target_rank_a=diag.compress_target_rank_a,
            compress_target_rank_b=diag.compress_target_rank_b,
            reasoning=(
                f"Redundant subspaces (overlap={diag.mean_overlap:.3f}, "
                f"agreement={diag.directional_agreement:.3f}). "
                f"TIES with trim={trim:.2f} deduplicates shared directions."
            ),
        )

    if verdict == "conflicting":
        drop_rate = _compute_dare_drop_rate(diag.n_conflict, diag.n_principal)
        return LayerRecommendation(
            layer_name=diag.layer_name,
            verdict=verdict,
            strategy="dare_ties",
            coefficients=(0.5, 0.5),
            trim_fraction=drop_rate,
            risk_level=diag.risk_level,
            compress_first=diag.compress_first,
            compress_target_rank_a=diag.compress_target_rank_a,
            compress_target_rank_b=diag.compress_target_rank_b,
            reasoning=(
                f"{diag.n_conflict} conflicting dimension(s) "
                f"(overlap={diag.mean_overlap:.3f}, agreement={diag.directional_agreement:.3f}). "
                f"DARE-TIES with drop={drop_rate:.2f} breaks interference."
            ),
        )

    if verdict == "imbalanced":
        coeffs = diag.suggested_coefficients or rebalance_coefficients(diag.magnitude_ratio)
        return LayerRecommendation(
            layer_name=diag.layer_name,
            verdict=verdict,
            strategy="linear",
            coefficients=coeffs,
            trim_fraction=0.0,
            risk_level=diag.risk_level,
            compress_first=diag.compress_first,
            compress_target_rank_a=diag.compress_target_rank_a,
            compress_target_rank_b=diag.compress_target_rank_b,
            reasoning=(
                f"Magnitude imbalance ({diag.magnitude_ratio:.1f}x). "
                f"Rebalanced coefficients: A={coeffs[0]:.2f}, B={coeffs[1]:.2f}."
            ),
        )

    # Fallback: moderate/ambiguous → TIES with light trim
    trim = _compute_trim_fraction(diag.mean_overlap) if diag.mean_overlap > 0.3 else 0.0
    strategy = "ties" if trim > 0 else "linear"
    return LayerRecommendation(
        layer_name=diag.layer_name,
        verdict=verdict,
        strategy=strategy,
        coefficients=(0.5, 0.5),
        trim_fraction=trim,
        risk_level="low",
        compress_first=diag.compress_first,
        compress_target_rank_a=diag.compress_target_rank_a,
        compress_target_rank_b=diag.compress_target_rank_b,
        reasoning=(
            f"Moderate interaction (overlap={diag.mean_overlap:.3f}). "
            f"{'TIES' if strategy == 'ties' else 'Linear'} merge should work."
        ),
    )


def _eligibility_warnings(eligibility: EligibilityContext) -> list[str]:
    """Generate hard warnings from eligibility context.

    Returns concise, non-cheerful warnings so the recommendation is honest
    about deployment risk.
    """
    if not eligibility.has_data:
        return [
            "No source-eligibility data provided; recommendation optimizes structural balance only.",
        ]

    warnings: list[str] = []

    # Both flagged weak — most severe
    if eligibility.both_weak:
        warnings.append(
            "Both source adapters underperform base or lack eligibility evidence; "
            "merge recommendation is structurally valid but deployment value is uncertain."
        )
        return warnings

    # Exactly one flagged weak
    if eligibility.any_weak:
        warnings.append("Structural rebalance may preserve a behaviorally weak adapter.")

    return warnings


def _apply_policy(pair_diag: PairDiagnosis) -> MergeRecommendation:
    """Apply recommendation policy to a pair diagnosis.

    Takes the complete Stage A output and produces a ``MergeRecommendation``
    with per-layer strategies, fallbacks, and eligibility warnings.

    Parameters
    ----------
    pair_diag : PairDiagnosis from ``diagnose_pair``

    Returns
    -------
    MergeRecommendation — complete Stage B output.
    """
    layer_recs = tuple(_apply_layer_policy(ld, pair_diag.eligibility) for ld in pair_diag.layer_diagnoses)

    # Hard warnings from eligibility screening
    hard_warnings = _eligibility_warnings(pair_diag.eligibility)

    return MergeRecommendation(
        # Always "audit_aware": the per-layer recommendations carry the real
        # strategy signal; the top-level field is a stable sentinel that
        # downstream consumers can key on without inspecting every layer.
        overall_strategy="audit_aware",
        overall_risk=pair_diag.overall_risk,
        layer_recommendations=layer_recs,
        compression_needed=pair_diag.compression_needed,
        n_layers_needing_compression=pair_diag.n_layers_needing_compression,
        fallback_strategies=("norm_equalized", "uniform_linear"),
        warnings=tuple(hard_warnings),
    )


# ===================================================================
# Public API — composes Stage A + Stage B
# ===================================================================


def _recommend_layer(lv_dict: dict[str, Any]) -> LayerRecommendation:
    """Generate recommendation for a single layer from its verdict dict.

    Convenience function that composes ``diagnose_layer`` (Stage A) with
    ``_apply_layer_policy`` (Stage B) using a no-data eligibility context.
    """
    diag = diagnose_layer(lv_dict)
    return _apply_layer_policy(diag, EligibilityContext(status_a=None, status_b=None))


def recommend_merge(
    report: Any,  # MergeAuditReport — Any to avoid circular import
) -> MergeRecommendation:
    """Generate complete merge recommendation from an audit report.

    Composes Stage A (``diagnose_pair``) with Stage B (``_apply_policy``).

    Parameters
    ----------
    report : MergeAuditReport
        Output from ``merge_audit()``.

    Returns
    -------
    MergeRecommendation with per-layer strategies, parameters, and
    compression guidance.
    """
    pair_diag = diagnose_pair(report)
    return _apply_policy(pair_diag)


# ---------------------------------------------------------------------------
# CLI formatting
# ---------------------------------------------------------------------------


### _shorten_layer_name — canonical definition lives in report.py


def _format_params(lr: LayerRecommendation) -> str:
    """Format strategy parameters for display."""
    if lr.strategy in ("linear", "norm_equalized"):
        return f"a={lr.coefficients[0]:.2f}, b={lr.coefficients[1]:.2f}"
    if lr.strategy == "ties":
        return f"trim={lr.trim_fraction:.2f}"
    if lr.strategy in ("dare_linear", "dare_ties"):
        return f"drop={lr.trim_fraction:.2f}"
    return ""


def format_recommendation(
    rec: MergeRecommendation,
    adapter_a_path: str = "./adapter-a",
    adapter_b_path: str = "./adapter-b",
) -> str:
    """Format recommendation as rich CLI output."""
    lines = []

    # Header
    risk_indicator = {"low": "OK", "medium": "CAUTION", "high": "WARNING"}
    risk_str = risk_indicator.get(rec.overall_risk, rec.overall_risk.upper())

    lines.append("")
    lines.append("  MERGE STRATEGY RECOMMENDATION")
    lines.append("  " + "=" * 47)
    lines.append("")
    lines.append("  Recommended approach: audit_aware (per-layer)")
    lines.append(f"  Overall risk: {risk_str}")
    lines.append("")

    # Per-layer table
    lines.append("  Per-layer breakdown:")
    header = f"  {'Layer':<30s} {'Verdict':<13s} {'Strategy':<12s} {'Parameters':<22s} {'Risk':<6s}"
    lines.append(header)
    lines.append("  " + "-" * 83)

    for lr in rec.layer_recommendations:
        short = _shorten_layer_name(lr.layer_name)
        params = _format_params(lr)
        lines.append(f"  {short:<30s} {lr.verdict:<13s} {lr.strategy:<12s} {params:<22s} {lr.risk_level:<6s}")

    lines.append("")

    # Compression recommendation
    if rec.compression_needed:
        compress_a = [lr for lr in rec.layer_recommendations if lr.compress_target_rank_a is not None]
        compress_b = [lr for lr in rec.layer_recommendations if lr.compress_target_rank_b is not None]

        lines.append("  Pre-compression recommended:")
        if compress_a:
            # Use the median target rank across layers
            targets_a = [lr.compress_target_rank_a for lr in compress_a if lr.compress_target_rank_a is not None]
            median_a = sorted(targets_a)[len(targets_a) // 2]
            lines.append(f"    Adapter A: {len(compress_a)} layer(s) over-provisioned with high overlap")
            lines.append(f"    $ gradience truncate --peft-dir {adapter_a_path} --rank {median_a}")
        if compress_b:
            targets_b = [lr.compress_target_rank_b for lr in compress_b if lr.compress_target_rank_b is not None]
            median_b = sorted(targets_b)[len(targets_b) // 2]
            lines.append(f"    Adapter B: {len(compress_b)} layer(s) over-provisioned with high overlap")
            lines.append(f"    $ gradience truncate --peft-dir {adapter_b_path} --rank {median_b}")
        lines.append(
            f"    Then re-run: gradience merge-audit --adapter-a {adapter_a_path} --adapter-b {adapter_b_path}"
        )
        lines.append("")

    # Ready-to-run command
    lines.append("  Ready-to-run merge:")
    lines.append("    $ gradience merge-plan --strategy audit_aware \\")
    lines.append(f"        --adapter-a {adapter_a_path} --adapter-b {adapter_b_path} \\")
    lines.append("        --output merge_plan.json")
    lines.append("")

    # Norm-equalized baseline
    lines.append("  Norm-equalized baseline (often competitive, simpler):")
    lines.append("    $ gradience merge-plan --strategy norm_equalized \\")
    lines.append(f"        --adapter-a {adapter_a_path} --adapter-b {adapter_b_path} \\")
    lines.append("        --output merge_plan_norm_eq.json")
    lines.append("")

    # Hard warnings from eligibility screening
    if rec.warnings:
        lines.append("  Warnings:")
        for warn in rec.warnings:
            lines.append(f"    * {warn}")
        lines.append("")

    # Fallback strategies
    if rec.fallback_strategies:
        alts = ", ".join(rec.fallback_strategies)
        lines.append(f"  Alternative strategies: {alts}")
        lines.append("")

    return "\n".join(lines)
