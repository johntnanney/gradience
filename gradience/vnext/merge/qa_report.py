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
    eligibility: str   # "eligible" | "flagged_weak" | "uncertain" | "unknown" | "not provided"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rank": self.rank,
            "alpha": float(self.alpha),
            "n_layers": self.n_layers,
            "base_model": self.base_model,
            "eligibility": self.eligibility,
        }


@dataclass(frozen=True)
class MergeQAReport:
    """Clean, practitioner-facing merge quality assessment.

    Every field answers one question a user would actually ask before
    deciding whether to merge two adapters.
    """

    adapter_a: AdapterSummary
    adapter_b: AdapterSummary
    pair_risk: str                  # "low" | "medium" | "high"
    dominant_issue: str             # human-readable label
    recommended_action: str         # one-sentence action
    recommended_strategy: str       # strategy name for the merge plan
    confidence_note: str            # how much to trust the recommendation
    caveats: tuple[str, ...]        # things the user should know
    verdict_distribution: dict[str, int]  # {"safe": N, "redundant": N, ...}
    compatibility_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "gradience.merge_qa_report/v1",
            "adapter_a": self.adapter_a.to_dict(),
            "adapter_b": self.adapter_b.to_dict(),
            "pair_risk": self.pair_risk,
            "dominant_issue": self.dominant_issue,
            "recommended_action": self.recommended_action,
            "recommended_strategy": self.recommended_strategy,
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
            recommended_action=d["recommended_action"],
            recommended_strategy=d["recommended_strategy"],
            confidence_note=d["confidence_note"],
            caveats=tuple(d["caveats"]),
            verdict_distribution=d["verdict_distribution"],
            compatibility_score=d["compatibility_score"],
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _eligibility_label(diag: PairDiagnosis, which: str) -> str:
    """Get a human-readable eligibility label for adapter A or B."""
    status = diag.eligibility.status_a if which == "a" else diag.eligibility.status_b
    if status is None:
        return "not provided"
    return status.value


def _dominant_issue(
    diag: PairDiagnosis,
    agg: dict[str, Any],
) -> str:
    """Identify the single biggest concern for this pair."""
    n_imbalanced = agg.get("n_imbalanced", 0)
    n_conflicting = agg.get("n_conflicting", 0)
    n_redundant = agg.get("n_redundant", 0)
    n_safe = agg.get("n_safe", 0)
    total = n_safe + n_redundant + n_conflicting + n_imbalanced

    if total == 0:
        return "unknown (no layer data)"

    # Check magnitude imbalance first — it's the most actionable
    mean_mag = agg.get("mean_magnitude_ratio", 1.0)
    if n_imbalanced > 0 and mean_mag > 3.0:
        return f"norm imbalance ({mean_mag:.1f}x mean magnitude ratio across {n_imbalanced} layer(s))"

    if n_conflicting > 0 and n_conflicting >= n_imbalanced:
        return f"subspace conflict ({n_conflicting} conflicting layer(s))"

    if n_imbalanced > 0:
        return f"norm imbalance ({n_imbalanced} imbalanced layer(s))"

    if n_redundant > 0 and n_redundant > n_safe:
        return f"high redundancy ({n_redundant} redundant layer(s))"

    if n_redundant > 0:
        return f"partial redundancy ({n_redundant} redundant layer(s))"

    return "none — adapters are spectrally compatible"


def _recommended_action(
    diag: PairDiagnosis,
    rec: MergeRecommendation,
    agg: dict[str, Any],
) -> str:
    """One-sentence recommended action."""
    if diag.eligibility.both_weak:
        return (
            "Reconsider merging: both adapters underperform the base model. "
            "Merge will produce a structurally valid but likely unhelpful result."
        )

    if diag.overall_risk == "high":
        n_conf = agg.get("n_conflicting", 0)
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
            "— no behavioral evaluation data available, so recommendation "
            "is based on structural analysis only"
        )
    elif diag.eligibility.any_weak:
        parts.append(
            "— at least one adapter lacks behavioral evidence of quality"
        )
    elif diag.eligibility.both_eligible:
        parts.append(
            "— both adapters have verified behavioral quality"
        )

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
        weak_label = "A" if (
            diag.eligibility.status_a
            and diag.eligibility.status_a.value == "flagged_weak"
        ) else "B"
        caveats.append(
            f"Adapter {weak_label} underperforms the base model. "
            f"Rebalancing may preserve a weak signal."
        )
    if diag.eligibility.both_weak:
        caveats.append(
            "Both adapters underperform the base model. "
            "Merging two weak adapters rarely produces a strong one."
        )

    # Structural caveats
    if diag.compression_needed:
        caveats.append(
            f"{diag.n_layers_needing_compression} layer(s) are over-provisioned. "
            f"Pre-compression before merging may improve results."
        )

    if diag.overall_risk == "high":
        caveats.append(
            "High structural risk. Always validate the merged adapter on "
            "your target task before deployment."
        )

    # From recommendation warnings
    for w in rec.warnings:
        if w not in "\n".join(caveats):
            caveats.append(w)

    return tuple(caveats)


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
        path=adapter_a_info.get("path", "unknown"),
        rank=adapter_a_info.get("rank", 0),
        alpha=adapter_a_info.get("alpha", 0.0),
        n_layers=adapter_a_info.get("n_layers", 0),
        base_model=adapter_a_info.get("base_model", "unknown"),
        eligibility=_eligibility_label(diag, "a"),
    )

    adapter_b = AdapterSummary(
        path=adapter_b_info.get("path", "unknown"),
        rank=adapter_b_info.get("rank", 0),
        alpha=adapter_b_info.get("alpha", 0.0),
        n_layers=adapter_b_info.get("n_layers", 0),
        base_model=adapter_b_info.get("base_model", "unknown"),
        eligibility=_eligibility_label(diag, "b"),
    )

    score = agg.get("compatibility_score", 0.0)

    verdict_dist = {
        "safe": agg.get("n_safe", 0),
        "redundant": agg.get("n_redundant", 0),
        "conflicting": agg.get("n_conflicting", 0),
        "imbalanced": agg.get("n_imbalanced", 0),
    }

    return MergeQAReport(
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        pair_risk=diag.overall_risk,
        dominant_issue=_dominant_issue(diag, agg),
        recommended_action=_recommended_action(diag, rec, agg),
        recommended_strategy=rec.overall_strategy,
        confidence_note=_confidence_note(diag, score),
        caveats=_caveats(diag, rec),
        verdict_distribution=verdict_dist,
        compatibility_score=score,
    )


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


def format_qa_report(qa: MergeQAReport) -> str:
    """Format a MergeQAReport as clean, human-readable text."""
    lines: list[str] = []

    lines.append("")
    lines.append("  MERGE QA REPORT")
    lines.append("  " + "=" * 60)

    # Adapter summaries
    lines.append("")
    lines.append("  Adapter A")
    lines.append(f"    Path:        {qa.adapter_a.path}")
    lines.append(f"    Rank:        {qa.adapter_a.rank}")
    lines.append(f"    Alpha:       {qa.adapter_a.alpha}")
    lines.append(f"    Layers:      {qa.adapter_a.n_layers}")
    lines.append(f"    Base model:  {qa.adapter_a.base_model}")
    lines.append(f"    Eligibility: {qa.adapter_a.eligibility}")

    lines.append("")
    lines.append("  Adapter B")
    lines.append(f"    Path:        {qa.adapter_b.path}")
    lines.append(f"    Rank:        {qa.adapter_b.rank}")
    lines.append(f"    Alpha:       {qa.adapter_b.alpha}")
    lines.append(f"    Layers:      {qa.adapter_b.n_layers}")
    lines.append(f"    Base model:  {qa.adapter_b.base_model}")
    lines.append(f"    Eligibility: {qa.adapter_b.eligibility}")

    # Risk and issue
    lines.append("")
    risk_indicator = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
    lines.append(f"  Pair risk:       {risk_indicator.get(qa.pair_risk, qa.pair_risk.upper())}")
    lines.append(f"  Dominant issue:  {qa.dominant_issue}")

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

    lines.append(f"  Compat. score:   {qa.compatibility_score:.3f}")

    # Recommended action
    lines.append("")
    lines.append("  Recommended action")
    lines.append(f"    {qa.recommended_action}")

    # Confidence
    lines.append("")
    lines.append("  Confidence")
    lines.append(f"    {qa.confidence_note}")

    # Caveats
    if qa.caveats:
        lines.append("")
        lines.append("  Caveats")
        for i, caveat in enumerate(qa.caveats, 1):
            lines.append(f"    {i}. {caveat}")

    lines.append("")

    return "\n".join(lines)
