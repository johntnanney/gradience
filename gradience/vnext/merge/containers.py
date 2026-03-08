"""
Typed containers for merge audit data.

Replaces the ``Dict[str, Any]`` fields that previously lived inside
``MergeAuditReport`` with frozen dataclasses that provide attribute access,
IDE auto-completion, and type-checked field validation.

Container types
---------------
- ``AdapterMetadata`` — structural summary of one PEFT adapter
- ``MatchingSummary`` — layer-matching statistics
- ``AggregateResult`` — aggregate spectral metrics across all shared layers
- ``WarningCode`` — categorical enum for merge warnings
- ``MergeWarning`` — a single typed warning (code + message)
- ``PairAuditResult`` — clean public-facing result of a full merge audit
- ``RecommendationResult`` — clean public-facing merge recommendation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# ===================================================================
# Adapter metadata
# ===================================================================


@dataclass(frozen=True)
class AdapterMetadata:
    """Structural summary of one PEFT LoRA adapter."""

    path: str
    rank: int
    alpha: float
    base_model: str = "unknown"
    n_layers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rank": self.rank,
            "alpha": self.alpha,
            "base_model": self.base_model,
            "n_layers": self.n_layers,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AdapterMetadata:
        return cls(
            path=d.get("path", "unknown"),
            rank=d.get("rank", 0),
            alpha=d.get("alpha", 0.0),
            base_model=d.get("base_model", "unknown"),
            n_layers=d.get("n_layers", 0),
        )

    # Dict-compat: allow d["key"] and d.get("key") for transition period
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def __iter__(self):
        return iter(self.to_dict())


# ===================================================================
# Layer matching
# ===================================================================


@dataclass(frozen=True)
class MatchingSummary:
    """Layer-matching statistics between two adapters."""

    n_shared: int
    n_only_a: int = 0
    n_only_b: int = 0
    shared_layers: tuple[str, ...] = ()
    only_a_layers: tuple[str, ...] = ()
    only_b_layers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_shared": self.n_shared,
            "n_only_a": self.n_only_a,
            "n_only_b": self.n_only_b,
            "shared_layers": list(self.shared_layers),
            "only_a_layers": list(self.only_a_layers),
            "only_b_layers": list(self.only_b_layers),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MatchingSummary:
        return cls(
            n_shared=d.get("n_shared", 0),
            n_only_a=d.get("n_only_a", 0),
            n_only_b=d.get("n_only_b", 0),
            shared_layers=tuple(d.get("shared_layers", ())),
            only_a_layers=tuple(d.get("only_a_layers", ())),
            only_b_layers=tuple(d.get("only_b_layers", ())),
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def __iter__(self):
        return iter(self.to_dict())


# ===================================================================
# Aggregate spectral result
# ===================================================================


@dataclass(frozen=True)
class AggregateResult:
    """Aggregate spectral metrics across all shared layers."""

    overall_verdict: str              # "safe" | "redundant" | "conflicting" | "imbalanced"
    compatibility_score: float

    # Overlap statistics
    mean_overlap: float = 0.0
    median_overlap: float = 0.0
    max_overlap: float = 0.0

    # Agreement
    mean_agreement: float = 0.0

    # Verdict distribution
    n_safe: int = 0
    n_redundant: int = 0
    n_conflicting: int = 0
    n_imbalanced: int = 0

    # Symmetric scale metrics
    mean_scale_bounded_ratio: float = 0.0
    mean_scale_log_ratio: float = 0.0
    mean_frob_bounded_ratio: float = 0.0
    mean_frob_log_ratio: float = 0.0

    # Legacy magnitude metric
    mean_magnitude_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_verdict": self.overall_verdict,
            "compatibility_score": self.compatibility_score,
            "mean_overlap": self.mean_overlap,
            "median_overlap": self.median_overlap,
            "max_overlap": self.max_overlap,
            "mean_agreement": self.mean_agreement,
            "n_safe": self.n_safe,
            "n_redundant": self.n_redundant,
            "n_conflicting": self.n_conflicting,
            "n_imbalanced": self.n_imbalanced,
            "mean_scale_bounded_ratio": self.mean_scale_bounded_ratio,
            "mean_scale_log_ratio": self.mean_scale_log_ratio,
            "mean_frob_bounded_ratio": self.mean_frob_bounded_ratio,
            "mean_frob_log_ratio": self.mean_frob_log_ratio,
            "mean_magnitude_ratio": self.mean_magnitude_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AggregateResult:
        return cls(
            overall_verdict=d.get("overall_verdict", "unknown"),
            compatibility_score=d.get("compatibility_score", 0.0),
            mean_overlap=d.get("mean_overlap", 0.0),
            median_overlap=d.get("median_overlap", 0.0),
            max_overlap=d.get("max_overlap", 0.0),
            mean_agreement=d.get("mean_agreement", 0.0),
            n_safe=d.get("n_safe", 0),
            n_redundant=d.get("n_redundant", 0),
            n_conflicting=d.get("n_conflicting", 0),
            n_imbalanced=d.get("n_imbalanced", 0),
            mean_scale_bounded_ratio=d.get("mean_scale_bounded_ratio", 0.0),
            mean_scale_log_ratio=d.get("mean_scale_log_ratio", 0.0),
            mean_frob_bounded_ratio=d.get("mean_frob_bounded_ratio", 0.0),
            mean_frob_log_ratio=d.get("mean_frob_log_ratio", 0.0),
            mean_magnitude_ratio=d.get("mean_magnitude_ratio", 0.0),
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def __iter__(self):
        return iter(self.to_dict())


# ===================================================================
# Warnings
# ===================================================================


class WarningCode(str, Enum):
    """Categorical warning codes for merge audit."""

    NO_ELIGIBILITY_DATA = "no_eligibility_data"
    ADAPTER_WEAK = "adapter_weak"
    BOTH_ADAPTERS_WEAK = "both_adapters_weak"
    ADAPTER_UNCERTAIN = "adapter_uncertain"
    ADAPTER_UNKNOWN = "adapter_unknown"
    LAYERS_ONLY_IN_A = "layers_only_in_a"
    LAYERS_ONLY_IN_B = "layers_only_in_b"
    HIGH_STRUCTURAL_RISK = "high_structural_risk"
    COMPRESSION_NEEDED = "compression_needed"


@dataclass(frozen=True)
class MergeWarning:
    """A single typed merge warning.

    Carries a machine-readable ``code`` alongside a human-readable ``message``,
    so downstream code can branch on code rather than parsing strings.
    """

    code: WarningCode
    message: str
    severity: str = "warning"         # "info" | "warning" | "error"
    adapter: str | None = None        # "a" | "b" | None (pair-level)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity,
        }
        if self.adapter is not None:
            d["adapter"] = self.adapter
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MergeWarning:
        return cls(
            code=WarningCode(d["code"]),
            message=d["message"],
            severity=d.get("severity", "warning"),
            adapter=d.get("adapter"),
        )


# ===================================================================
# Composed public-facing result types
# ===================================================================


@dataclass(frozen=True)
class PairAuditResult:
    """Clean public-facing result of a full merge compatibility audit.

    Composes typed containers instead of dict fields, suitable for
    programmatic consumption and IDE auto-completion.
    """

    adapter_a: AdapterMetadata
    adapter_b: AdapterMetadata
    matching: MatchingSummary
    aggregate: AggregateResult
    layer_verdicts: tuple[dict[str, Any], ...]   # LayerVerdict.to_dict() form for now
    recommendations: tuple[str, ...]
    typed_warnings: tuple[MergeWarning, ...] = ()
    source_qa: dict[str, Any] | None = None      # preserved for eligibility parsing

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "adapter_a": self.adapter_a.to_dict(),
            "adapter_b": self.adapter_b.to_dict(),
            "matching": self.matching.to_dict(),
            "aggregate": self.aggregate.to_dict(),
            "layer_verdicts": list(self.layer_verdicts),
            "recommendations": list(self.recommendations),
        }
        if self.typed_warnings:
            d["warnings"] = [w.to_dict() for w in self.typed_warnings]
        if self.source_qa is not None:
            d["source_qa"] = self.source_qa
        return d

    @classmethod
    def from_audit_report(cls, report: Any) -> PairAuditResult:
        """Construct from a MergeAuditReport (typed or dict-based)."""
        adapter_a = (
            report.adapter_a if isinstance(report.adapter_a, AdapterMetadata)
            else AdapterMetadata.from_dict(report.adapter_a)
        )
        adapter_b = (
            report.adapter_b if isinstance(report.adapter_b, AdapterMetadata)
            else AdapterMetadata.from_dict(report.adapter_b)
        )
        matching = (
            report.matching if isinstance(report.matching, MatchingSummary)
            else MatchingSummary.from_dict(report.matching)
        )
        aggregate = (
            report.aggregate if isinstance(report.aggregate, AggregateResult)
            else AggregateResult.from_dict(report.aggregate)
        )

        # Convert string warnings to typed warnings where possible
        typed = []
        for w in getattr(report, "warnings", []):
            if isinstance(w, MergeWarning):
                typed.append(w)
            elif isinstance(w, str):
                typed.append(MergeWarning(
                    code=_classify_warning_string(w),
                    message=w,
                ))

        return cls(
            adapter_a=adapter_a,
            adapter_b=adapter_b,
            matching=matching,
            aggregate=aggregate,
            layer_verdicts=tuple(report.layer_verdicts),
            recommendations=tuple(report.recommendations),
            typed_warnings=tuple(typed),
            source_qa=getattr(report, "source_qa", None),
        )


@dataclass(frozen=True)
class RecommendationResult:
    """Clean public-facing merge recommendation.

    Distills the full recommendation into fields a practitioner
    needs to decide whether and how to merge.
    """

    strategy: str                           # recommended merge strategy
    risk: str                               # "low" | "medium" | "high"
    action: str                             # one-sentence recommended action
    confidence: str                         # confidence note
    dominant_issue: str                     # primary concern
    caveats: tuple[str, ...]                # things the user should know
    typed_warnings: tuple[MergeWarning, ...]  # machine-readable warnings
    fallback_strategies: tuple[str, ...]    # alternative approaches

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "risk": self.risk,
            "action": self.action,
            "confidence": self.confidence,
            "dominant_issue": self.dominant_issue,
            "caveats": list(self.caveats),
            "warnings": [w.to_dict() for w in self.typed_warnings],
            "fallback_strategies": list(self.fallback_strategies),
        }


# ===================================================================
# Internal helpers
# ===================================================================


def _classify_warning_string(msg: str) -> WarningCode:
    """Best-effort classification of a free-form warning string."""
    lower = msg.lower()
    if "both" in lower and ("weak" in lower or "underperform" in lower):
        return WarningCode.BOTH_ADAPTERS_WEAK
    if "weak" in lower or "underperform" in lower:
        return WarningCode.ADAPTER_WEAK
    if "uncertain" in lower:
        return WarningCode.ADAPTER_UNCERTAIN
    if "no source-eligibility" in lower or "no behavioral" in lower:
        return WarningCode.NO_ELIGIBILITY_DATA
    if "only in adapter a" in lower:
        return WarningCode.LAYERS_ONLY_IN_A
    if "only in adapter b" in lower:
        return WarningCode.LAYERS_ONLY_IN_B
    if "compress" in lower:
        return WarningCode.COMPRESSION_NEEDED
    if "risk" in lower:
        return WarningCode.HIGH_STRUCTURAL_RISK
    # Fallback — can't classify, but still a warning about an adapter
    if "unknown" in lower or "no data" in lower:
        return WarningCode.ADAPTER_UNKNOWN
    return WarningCode.NO_ELIGIBILITY_DATA  # safe default
