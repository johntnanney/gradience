"""
Merge strategy recommendation engine.

Translates a MergeAuditReport into specific, actionable recommendations:
concrete strategy choices with tuned parameters per layer, compression
suggestions when adapters are over-provisioned, and ready-to-paste CLI
commands.

The recommendation logic is deterministic: all parameters are computed
from spectral metrics, no GPU or task data required.

Usage::

    from gradience.vnext.merge.recommend import recommend_merge

    report = merge_audit("./adapter_a", "./adapter_b")
    rec = recommend_merge(report)
    print(rec.format_cli())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from gradience.vnext.merge.eligibility import EligibilityStatus


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerRecommendation:
    """Actionable recommendation for one layer pair."""

    layer_name: str
    verdict: str                            # "safe" | "redundant" | "conflicting" | "imbalanced"
    strategy: str                           # "linear" | "ties" | "dare_linear" | "dare_ties"
    coefficients: Tuple[float, float]       # (coeff_a, coeff_b)
    trim_fraction: float                    # TIES trim or DARE drop rate
    risk_level: str                         # "low" | "medium" | "high"
    compress_first: bool                    # True if pre-compression recommended
    compress_target_rank_a: Optional[int]   # suggested rank for adapter A
    compress_target_rank_b: Optional[int]   # suggested rank for adapter B
    reasoning: str                          # one-line explanation

    def to_dict(self) -> Dict[str, Any]:
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

    overall_strategy: str                   # dominant strategy across layers
    overall_risk: str                       # "low" | "medium" | "high"
    layer_recommendations: Tuple[LayerRecommendation, ...]
    compression_needed: bool
    n_layers_needing_compression: int
    fallback_strategies: Tuple[str, ...]    # alternative approaches
    warnings: Tuple[str, ...] = ()          # hard warnings from eligibility screening

    def to_dict(self) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Parameter computation — deterministic functions of spectral metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core recommendation logic
# ---------------------------------------------------------------------------


def _recommend_layer(lv_dict: Dict[str, Any]) -> LayerRecommendation:
    """Generate recommendation for a single layer from its verdict dict."""
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

    # Number of principal angles available
    pac = metrics.get("principal_angle_cosines", ())
    n_principal = len(pac) if pac else max(effective_rank_a, effective_rank_b, 1)

    # Compression check (for both adapters independently)
    compress_a = _should_compress(mean_overlap, effective_rank_a, nominal_rank_a)
    compress_b = _should_compress(mean_overlap, effective_rank_b, nominal_rank_b)
    compress_first = compress_a or compress_b
    target_a = _compression_target(effective_rank_a) if compress_a else None
    target_b = _compression_target(effective_rank_b) if compress_b else None

    risk = _risk_level(verdict, confidence)

    # --- Strategy and parameter selection per verdict ---

    if verdict == "safe":
        return LayerRecommendation(
            layer_name=layer_name,
            verdict=verdict,
            strategy="linear",
            coefficients=(0.5, 0.5),
            trim_fraction=0.0,
            risk_level=risk,
            compress_first=compress_first,
            compress_target_rank_a=target_a,
            compress_target_rank_b=target_b,
            reasoning=(
                f"Orthogonal subspaces (overlap={mean_overlap:.3f}). "
                f"Linear merge preserves both signals."
            ),
        )

    if verdict == "redundant":
        trim = _compute_trim_fraction(mean_overlap)
        return LayerRecommendation(
            layer_name=layer_name,
            verdict=verdict,
            strategy="ties",
            coefficients=(0.5, 0.5),
            trim_fraction=trim,
            risk_level=risk,
            compress_first=compress_first,
            compress_target_rank_a=target_a,
            compress_target_rank_b=target_b,
            reasoning=(
                f"Redundant subspaces (overlap={mean_overlap:.3f}, "
                f"agreement={directional_agreement:.3f}). "
                f"TIES with trim={trim:.2f} deduplicates shared directions."
            ),
        )

    if verdict == "conflicting":
        drop_rate = _compute_dare_drop_rate(n_conflict, n_principal)
        return LayerRecommendation(
            layer_name=layer_name,
            verdict=verdict,
            strategy="dare_ties",
            coefficients=(0.5, 0.5),
            trim_fraction=drop_rate,
            risk_level=risk,
            compress_first=compress_first,
            compress_target_rank_a=target_a,
            compress_target_rank_b=target_b,
            reasoning=(
                f"{n_conflict} conflicting dimension(s) "
                f"(overlap={mean_overlap:.3f}, agreement={directional_agreement:.3f}). "
                f"DARE-TIES with drop={drop_rate:.2f} breaks interference."
            ),
        )

    if verdict == "imbalanced":
        coeffs = tuple(suggested_coeffs) if suggested_coeffs else (0.5, 0.5)
        return LayerRecommendation(
            layer_name=layer_name,
            verdict=verdict,
            strategy="linear",
            coefficients=coeffs,
            trim_fraction=0.0,
            risk_level=risk,
            compress_first=compress_first,
            compress_target_rank_a=target_a,
            compress_target_rank_b=target_b,
            reasoning=(
                f"Magnitude imbalance ({magnitude_ratio:.1f}x). "
                f"Rebalanced coefficients: A={coeffs[0]:.2f}, B={coeffs[1]:.2f}."
            ),
        )

    # Fallback: moderate/ambiguous → TIES with light trim
    trim = _compute_trim_fraction(mean_overlap) if mean_overlap > 0.3 else 0.0
    strategy = "ties" if trim > 0 else "linear"
    return LayerRecommendation(
        layer_name=layer_name,
        verdict=verdict,
        strategy=strategy,
        coefficients=(0.5, 0.5),
        trim_fraction=trim,
        risk_level="low",
        compress_first=compress_first,
        compress_target_rank_a=target_a,
        compress_target_rank_b=target_b,
        reasoning=(
            f"Moderate interaction (overlap={mean_overlap:.3f}). "
            f"{'TIES' if strategy == 'ties' else 'Linear'} merge should work."
        ),
    )


def _eligibility_warnings(report: Any) -> list[str]:
    """Generate hard warnings based on source adapter eligibility data.

    Inspects ``report.source_qa`` (dict with optional ``adapter_a`` /
    ``adapter_b`` keys, each containing an ``AdapterQAResult.to_dict()``).
    Returns concise, non-cheerful warnings so the recommendation is honest
    about deployment risk.
    """
    source_qa = getattr(report, "source_qa", None)

    # Case 1: no source QA provided at all
    if source_qa is None:
        return [
            "No source-eligibility data provided; recommendation optimizes structural balance only.",
        ]

    def _status(key: str) -> EligibilityStatus | None:
        entry = source_qa.get(key)
        if entry is None:
            return None
        raw = entry.get("status", EligibilityStatus.UNKNOWN.value)
        try:
            return EligibilityStatus(raw)
        except ValueError:
            return EligibilityStatus.UNKNOWN

    status_a = _status("adapter_a")
    status_b = _status("adapter_b")

    # If both entries are missing despite source_qa being non-None (empty dict),
    # treat as no data provided.
    if status_a is None and status_b is None:
        return [
            "No source-eligibility data provided; recommendation optimizes structural balance only.",
        ]

    warnings: list[str] = []

    # Both flagged weak — most severe
    if status_a == EligibilityStatus.FLAGGED_WEAK and status_b == EligibilityStatus.FLAGGED_WEAK:
        warnings.append(
            "Both source adapters underperform base or lack eligibility evidence; "
            "merge recommendation is structurally valid but deployment value is uncertain."
        )
        return warnings

    # Exactly one flagged weak
    if status_a == EligibilityStatus.FLAGGED_WEAK or status_b == EligibilityStatus.FLAGGED_WEAK:
        warnings.append(
            "Structural rebalance may preserve a behaviorally weak adapter."
        )

    return warnings


def recommend_merge(
    report: Any,  # MergeAuditReport — Any to avoid circular import
) -> MergeRecommendation:
    """Generate complete merge recommendation from an audit report.

    Parameters
    ----------
    report : MergeAuditReport
        Output from ``merge_audit()``.

    Returns
    -------
    MergeRecommendation with per-layer strategies, parameters, and
    compression guidance.
    """
    layer_recs = []
    for lv_dict in report.layer_verdicts:
        layer_recs.append(_recommend_layer(lv_dict))

    # --- Overall strategy: majority vote across layers ---
    strategy_counts: Dict[str, int] = {}
    for lr in layer_recs:
        strategy_counts[lr.strategy] = strategy_counts.get(lr.strategy, 0) + 1
    overall_strategy = max(strategy_counts, key=strategy_counts.get) if strategy_counts else "linear"

    # --- Overall risk: worst case ---
    risk_order = {"low": 0, "medium": 1, "high": 2}
    overall_risk = max(
        (lr.risk_level for lr in layer_recs),
        key=lambda r: risk_order.get(r, 0),
        default="low",
    )

    # --- Compression ---
    n_compress = sum(1 for lr in layer_recs if lr.compress_first)

    # --- Fallback strategies ---
    fallbacks = []
    if overall_strategy != "linear":
        fallbacks.append("uniform_linear")
    if overall_strategy != "dare_ties":
        fallbacks.append("dare_ties")
    if overall_strategy != "ties":
        fallbacks.append("overlap_ties")

    # --- Hard warnings from eligibility screening ---
    hard_warnings = _eligibility_warnings(report)

    return MergeRecommendation(
        overall_strategy="audit_aware",  # always audit_aware since it's per-layer
        overall_risk=overall_risk,
        layer_recommendations=tuple(layer_recs),
        compression_needed=n_compress > 0,
        n_layers_needing_compression=n_compress,
        fallback_strategies=tuple(fallbacks[:2]),
        warnings=tuple(hard_warnings),
    )


# ---------------------------------------------------------------------------
# CLI formatting
# ---------------------------------------------------------------------------


def _shorten_layer_name(name: str) -> str:
    """Shorten a fully-qualified layer name for display.

    "base_model.model.model.layers.10.self_attn.k_proj" -> "L10.k_proj"
    """
    parts = name.split(".")
    block_idx = None
    suffix_start = len(parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            block_idx = parts[i]
            suffix_start = i + 1
            break
    suffix = ".".join(parts[suffix_start:]) if suffix_start < len(parts) else parts[-1]
    if block_idx is not None:
        return f"L{block_idx}.{suffix}"
    return suffix


def _format_params(lr: LayerRecommendation) -> str:
    """Format strategy parameters for display."""
    if lr.strategy == "linear":
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
    lines.append(f"  Recommended approach: audit_aware (per-layer)")
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
        lines.append(
            f"  {short:<30s} {lr.verdict:<13s} {lr.strategy:<12s} {params:<22s} {lr.risk_level:<6s}"
        )

    lines.append("")

    # Compression recommendation
    if rec.compression_needed:
        compress_a = [lr for lr in rec.layer_recommendations if lr.compress_target_rank_a is not None]
        compress_b = [lr for lr in rec.layer_recommendations if lr.compress_target_rank_b is not None]

        lines.append("  Pre-compression recommended:")
        if compress_a:
            # Use the median target rank across layers
            targets_a = [lr.compress_target_rank_a for lr in compress_a]
            median_a = sorted(targets_a)[len(targets_a) // 2]
            lines.append(
                f"    Adapter A: {len(compress_a)} layer(s) over-provisioned with high overlap"
            )
            lines.append(
                f"    $ gradience compress --peft-dir {adapter_a_path} --target-rank {median_a}"
            )
        if compress_b:
            targets_b = [lr.compress_target_rank_b for lr in compress_b]
            median_b = sorted(targets_b)[len(targets_b) // 2]
            lines.append(
                f"    Adapter B: {len(compress_b)} layer(s) over-provisioned with high overlap"
            )
            lines.append(
                f"    $ gradience compress --peft-dir {adapter_b_path} --target-rank {median_b}"
            )
        lines.append(
            f"    Then re-run: gradience merge-audit --adapter-a {adapter_a_path} --adapter-b {adapter_b_path}"
        )
        lines.append("")

    # Ready-to-run command
    lines.append("  Ready-to-run merge:")
    lines.append(
        f"    $ gradience merge-plan --strategy audit_aware \\"
    )
    lines.append(
        f"        --adapter-a {adapter_a_path} --adapter-b {adapter_b_path} \\"
    )
    lines.append(
        f"        --output merge_plan.json"
    )
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
