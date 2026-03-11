"""
Merge QA Report — clean, human-readable summary of a merge audit.

Produces a ``MergeQAReport`` that distills the full spectral audit and
diagnosis into the fields a practitioner actually needs to make a decision:

- Adapter structural summaries (rank, alpha, layer count, base model)
- Eligibility status for both adapters
- Pair risk level
- Dominant issue (norm imbalance / subspace conflict / redundancy / none)
- Recommended action
- Confidence note
- Caveats

Usage::

    from gradience.vnext.merge.qa_report import build_qa_report, format_qa_report

    report = merge_audit("./adapter_a", "./adapter_b")
    qa = build_qa_report(report)
    print(format_qa_report(qa))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gradience.vnext.merge.recommend import (
    MergeRecommendation,
    PairDiagnosis,
    diagnose_pair,
    recommend_merge,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMINANT_ISSUE_LABELS = frozenset(
    {
        "norm_imbalance",
        "subspace_conflict",
        "high_redundancy",
        "partial_redundancy",
        "none",
        "unknown",
    }
)

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterSummary:
    """Structural summary of one adapter."""

    path: str
    rank: int
    alpha: float
    n_layers: int
    base_model: str
    eligibility_status: str | None  # EligibilityStatus value or None when no QA provided

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rank": self.rank,
            "alpha": float(self.alpha),
            "n_layers": self.n_layers,
            "base_model": self.base_model,
            "eligibility_status": self.eligibility_status,
        }


@dataclass(frozen=True)
class MergeQAReport:
    """Clean, practitioner-facing merge quality assessment.

    Every field answers one question a user would actually ask before
    deciding whether to merge two adapters.
    """

    adapter_a: AdapterSummary
    adapter_b: AdapterSummary
    pair_risk: str  # "low" | "medium" | "high"
    dominant_issue: str  # machine-readable label from DOMINANT_ISSUE_LABELS
    dominant_issue_detail: str  # human-readable explanation
    recommended_action: str  # one-sentence action
    recommended_strategy: str  # strategy name for the merge plan
    confidence: str  # "high" | "medium" | "low"
    confidence_note: str  # how much to trust the recommendation
    caveats: tuple[str, ...]  # things the user should know
    verdict_distribution: dict[str, int]  # {"safe": N, "redundant": N, ...}
    compatibility_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "gradience.merge_qa_report/v1",
            "adapter_a": self.adapter_a.to_dict(),
            "adapter_b": self.adapter_b.to_dict(),
            "pair_risk": self.pair_risk,
            "dominant_issue": self.dominant_issue,
            "dominant_issue_detail": self.dominant_issue_detail,
            "recommended_action": self.recommended_action,
            "recommended_strategy": self.recommended_strategy,
            "confidence": self.confidence,
            "confidence_note": self.confidence_note,
            "caveats": list(self.caveats),
            "verdict_distribution": self.verdict_distribution,
            "compatibility_score": round(self.compatibility_score, 4),
        }

    def to_json(self, path: Path | str) -> None:
        """Write QA report to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MergeQAReport:
        """Reconstruct from a dict (e.g. loaded from JSON)."""
        return cls(
            adapter_a=AdapterSummary(**d["adapter_a"]),
            adapter_b=AdapterSummary(**d["adapter_b"]),
            pair_risk=d["pair_risk"],
            dominant_issue=d["dominant_issue"],
            dominant_issue_detail=d.get("dominant_issue_detail", ""),
            recommended_action=d["recommended_action"],
            recommended_strategy=d["recommended_strategy"],
            confidence=d.get("confidence", "medium"),
            confidence_note=d["confidence_note"],
            caveats=tuple(d["caveats"]),
            verdict_distribution=d["verdict_distribution"],
            compatibility_score=d["compatibility_score"],
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _eligibility_label(diag: PairDiagnosis, which: str) -> str | None:
    """Get eligibility status for adapter A or B, or None if no QA provided."""
    status = diag.eligibility.status_a if which == "a" else diag.eligibility.status_b
    if status is None:
        return None
    return status.value


def _dominant_issue(
    diag: PairDiagnosis,
    agg: Any,
) -> tuple[str, str]:
    """Identify the single biggest concern for this pair.

    Returns (machine_label, human_detail).
    """
    n_imbalanced = getattr(agg, "n_imbalanced", 0)
    n_conflicting = getattr(agg, "n_conflicting", 0)
    n_redundant = getattr(agg, "n_redundant", 0)
    n_safe = getattr(agg, "n_safe", 0)
    total = n_safe + n_redundant + n_conflicting + n_imbalanced

    if total == 0:
        return "unknown", "no layer data available"

    # Check magnitude imbalance first — it's the most actionable
    mean_mag = getattr(agg, "mean_magnitude_ratio", 1.0)
    if n_imbalanced > 0 and mean_mag > 3.0:
        return "norm_imbalance", f"{mean_mag:.1f}x mean magnitude ratio across {n_imbalanced} layer(s)"

    if n_conflicting > 0 and n_conflicting >= n_imbalanced:
        return "subspace_conflict", f"{n_conflicting} conflicting layer(s)"

    if n_imbalanced > 0:
        return "norm_imbalance", f"{n_imbalanced} imbalanced layer(s)"

    if n_redundant > 0 and n_redundant > n_safe:
        return "high_redundancy", f"{n_redundant} redundant layer(s)"

    if n_redundant > 0:
        return "partial_redundancy", f"{n_redundant} redundant layer(s)"

    return "none", "adapters are spectrally compatible"


def _recommended_action(
    diag: PairDiagnosis,
    rec: MergeRecommendation,
    agg: Any,
) -> str:
    """One-sentence recommended action."""
    if diag.eligibility.both_weak:
        return (
            "Reconsider merging: both adapters underperform the base model. "
            "Merge will produce a structurally valid but likely unhelpful result."
        )

    if diag.overall_risk == "high":
        n_conf = getattr(agg, "n_conflicting", 0)
        if n_conf > 0:
            return (
                f"Merge with caution using audit-aware strategy (DARE-TIES on "
                f"{n_conf} conflicting layer(s)). Validate merged adapter on downstream task."
            )
        return "Merge with caution using audit-aware strategy. Validate on downstream task."

    if rec.compression_needed:
        return (
            f"Pre-compress {rec.n_layers_needing_compression} over-provisioned layer(s), "
            f"then merge with audit-aware strategy."
        )

    if diag.overall_risk == "low":
        return "Merge is safe. Use audit-aware strategy or norm-equalized baseline."

    # medium risk
    return "Merge with audit-aware strategy. Consider norm-equalized as simpler alternative."


def _confidence_note(diag: PairDiagnosis, score: float) -> str:
    """How much to trust the recommendation."""
    parts: list[str] = []

    if score >= 0.8:
        parts.append("High spectral compatibility")
    elif score >= 0.5:
        parts.append("Moderate spectral compatibility")
    else:
        parts.append("Low spectral compatibility")

    parts.append(f"(score={score:.3f})")

    if not diag.eligibility.has_data:
        parts.append(
            "— no behavioral evaluation data available, so recommendation is based on structural analysis only"
        )
    elif diag.eligibility.any_weak:
        parts.append("— at least one adapter lacks behavioral evidence of quality")
    elif diag.eligibility.both_eligible:
        parts.append("— both adapters have verified behavioral quality")

    return ". ".join(p if p.startswith("—") or p.startswith("(") else p for p in parts[:1]) + " " + " ".join(parts[1:])


def _caveats(diag: PairDiagnosis, rec: MergeRecommendation) -> tuple[str, ...]:
    """Build caveats list from diagnosis and recommendation."""
    caveats: list[str] = []

    # Eligibility caveats
    if not diag.eligibility.has_data:
        caveats.append(
            "No source-eligibility data was provided. The recommendation optimizes "
            "structural balance only and cannot predict downstream task performance."
        )
    if diag.eligibility.any_weak and not diag.eligibility.both_weak:
        weak_label = "A" if (diag.eligibility.status_a and diag.eligibility.status_a.value == "flagged_weak") else "B"
        caveats.append(f"Adapter {weak_label} underperforms the base model. Rebalancing may preserve a weak signal.")
    if diag.eligibility.both_weak:
        caveats.append(
            "Both adapters underperform the base model. Merging two weak adapters rarely produces a strong one."
        )

    # Structural caveats
    if diag.compression_needed:
        caveats.append(
            f"{diag.n_layers_needing_compression} layer(s) are over-provisioned. "
            f"Pre-compression before merging may improve results."
        )

    if diag.overall_risk == "high":
        caveats.append(
            "High structural risk. Always validate the merged adapter on your target task before deployment."
        )

    # From recommendation warnings
    for w in rec.warnings:
        if w not in "\n".join(caveats):
            caveats.append(w)

    return tuple(caveats)


def _derive_strategy(diag: PairDiagnosis, rec: MergeRecommendation) -> str:
    """Derive the primary recommended strategy from diagnosis.

    Policy:
      low risk + no compression  -> "linear"
      medium risk + no compression -> "norm_equalized"
      otherwise (high risk or compression needed) -> "audit_aware"
    """
    if diag.compression_needed:
        return "audit_aware"
    if diag.overall_risk == "low":
        return "linear"
    if diag.overall_risk == "medium":
        return "norm_equalized"
    return "audit_aware"


def _derive_confidence(diag: PairDiagnosis, score: float) -> str:
    """Derive categorical confidence level."""
    if not diag.eligibility.has_data:
        return "low"
    if diag.overall_risk == "high":
        return "low"
    if diag.eligibility.both_eligible and score >= 0.8 and diag.overall_risk == "low":
        return "high"
    return "medium"


def build_qa_report(report: Any) -> MergeQAReport:
    """Build a QA report from a MergeAuditReport.

    Runs Stage A diagnosis and Stage B recommendation internally,
    then distills results into a clean practitioner-facing summary.

    Parameters
    ----------
    report : MergeAuditReport
        Output from ``merge_audit()``.

    Returns
    -------
    MergeQAReport
    """
    diag = diagnose_pair(report)
    rec = recommend_merge(report)
    agg = report.aggregate

    adapter_a_info = report.adapter_a
    adapter_b_info = report.adapter_b

    adapter_a = AdapterSummary(
        path=getattr(adapter_a_info, "path", "unknown"),
        rank=getattr(adapter_a_info, "rank", 0),
        alpha=getattr(adapter_a_info, "alpha", 0.0),
        n_layers=getattr(adapter_a_info, "n_layers", 0),
        base_model=getattr(adapter_a_info, "base_model", "unknown"),
        eligibility_status=_eligibility_label(diag, "a"),
    )

    adapter_b = AdapterSummary(
        path=getattr(adapter_b_info, "path", "unknown"),
        rank=getattr(adapter_b_info, "rank", 0),
        alpha=getattr(adapter_b_info, "alpha", 0.0),
        n_layers=getattr(adapter_b_info, "n_layers", 0),
        base_model=getattr(adapter_b_info, "base_model", "unknown"),
        eligibility_status=_eligibility_label(diag, "b"),
    )

    score = getattr(agg, "compatibility_score", agg.get("compatibility_score", 0.0) if hasattr(agg, "get") else 0.0)

    verdict_dist = {
        "safe": getattr(agg, "n_safe", 0),
        "redundant": getattr(agg, "n_redundant", 0),
        "conflicting": getattr(agg, "n_conflicting", 0),
        "imbalanced": getattr(agg, "n_imbalanced", 0),
    }

    issue_label, issue_detail = _dominant_issue(diag, agg)
    confidence = _derive_confidence(diag, score)

    return MergeQAReport(
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        pair_risk=diag.overall_risk,
        dominant_issue=issue_label,
        dominant_issue_detail=issue_detail,
        recommended_action=_recommended_action(diag, rec, agg),
        recommended_strategy=_derive_strategy(diag, rec),
        confidence=confidence,
        confidence_note=_confidence_note(diag, score),
        caveats=_caveats(diag, rec),
        verdict_distribution=verdict_dist,
        compatibility_score=score,
    )


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


def format_qa_report(qa: MergeQAReport) -> str:
    """Format a MergeQAReport as clean, human-readable text.

    Output is organized into four explicit sections:

    1. **Structural Result** — spectral compatibility verdict, score,
       layer distribution, dominant structural issue.
    2. **Behavioral Status** — source adapter eligibility from QA data,
       confidence note based on available evidence.
    3. **Eligibility Warning** — warnings and caveats about data gaps
       or weak adapters that affect recommendation reliability.
    4. **Recommended Action** — concrete action and strategy, informed
       by both structural and behavioral analysis.
    """
    lines: list[str] = []

    lines.append("")
    lines.append("  MERGE QA REPORT")
    lines.append("  " + "=" * 60)

    # ---------------------------------------------------------------
    # Section 1: Structural Result
    # ---------------------------------------------------------------
    lines.append("")
    lines.append("  1. STRUCTURAL RESULT")
    lines.append("  " + "-" * 40)

    risk_indicator = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
    lines.append(f"  Pair risk:       {risk_indicator.get(qa.pair_risk, qa.pair_risk.upper())}")
    lines.append(f"  Compat. score:   {qa.compatibility_score:.3f}")
    lines.append(f"  Dominant issue:  {qa.dominant_issue.upper().replace('_', ' ')}")
    if qa.dominant_issue_detail:
        lines.append(f"                   {qa.dominant_issue_detail}")

    # Verdict distribution
    dist = qa.verdict_distribution
    total = sum(dist.values())
    if total > 0:
        dist_parts = []
        for k in ("safe", "redundant", "conflicting", "imbalanced"):
            n = dist.get(k, 0)
            if n > 0:
                dist_parts.append(f"{n} {k}")
        lines.append(f"  Layer verdicts:  {', '.join(dist_parts)} ({total} total)")

    # Adapter structural summaries
    lines.append("")
    lines.append(
        f"  Adapter A:  rank={qa.adapter_a.rank}, alpha={qa.adapter_a.alpha}, "
        f"{qa.adapter_a.n_layers} layers ({qa.adapter_a.base_model})"
    )
    lines.append(f"              {qa.adapter_a.path}")
    lines.append(
        f"  Adapter B:  rank={qa.adapter_b.rank}, alpha={qa.adapter_b.alpha}, "
        f"{qa.adapter_b.n_layers} layers ({qa.adapter_b.base_model})"
    )
    lines.append(f"              {qa.adapter_b.path}")

    # ---------------------------------------------------------------
    # Section 2: Behavioral Status
    # ---------------------------------------------------------------
    lines.append("")
    lines.append("  2. BEHAVIORAL STATUS")
    lines.append("  " + "-" * 40)

    a_status = qa.adapter_a.eligibility_status or "not provided"
    b_status = qa.adapter_b.eligibility_status or "not provided"
    lines.append(f"  Adapter A eligibility: {a_status}")
    lines.append(f"  Adapter B eligibility: {b_status}")

    lines.append("")
    lines.append(f"  Confidence:      {qa.confidence}")
    lines.append(f"                   {qa.confidence_note}")

    # ---------------------------------------------------------------
    # Section 3: Eligibility Warning
    # ---------------------------------------------------------------
    lines.append("")
    lines.append("  3. ELIGIBILITY WARNING")
    lines.append("  " + "-" * 40)

    if qa.caveats:
        for i, caveat in enumerate(qa.caveats, 1):
            lines.append(f"  {i}. {caveat}")
    else:
        lines.append("  No eligibility concerns detected.")

    # ---------------------------------------------------------------
    # Section 4: Recommended Action
    # ---------------------------------------------------------------
    lines.append("")
    lines.append("  4. RECOMMENDED ACTION")
    lines.append("  " + "-" * 40)
    lines.append(f"  {qa.recommended_action}")
    lines.append(f"  Strategy: {qa.recommended_strategy}")

    lines.append("")

    return "\n".join(lines)
