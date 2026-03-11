"""
Source adapter eligibility screening.

Study 16 showed that merge recommendation assumes both source adapters are
worth preserving.  When one or both sources are weaker than the base model,
structurally fair merging can still produce behaviorally disappointing results.

This module provides lightweight plumbing for source-quality gating:

    1. ``EligibilityStatus`` — a four-state enum capturing what we know
       about an adapter's standalone quality.
    2. ``AdapterQAResult`` — a small data structure carrying the status
       plus whatever evidence produced it.
    3. ``screen_adapters`` — given two optional QA results, returns
       warnings that should be surfaced in the merge audit report.

The heuristics are intentionally simple today.  The architecture matters
more than the decision logic: this is the hook where better screening
will live later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class EligibilityStatus(str, Enum):
    """Source adapter eligibility for merge preservation.

    Values
    ------
    ELIGIBLE
        Adapter has demonstrated standalone quality above baseline.
    UNCERTAIN
        Some evidence exists but is inconclusive.
    FLAGGED_WEAK
        Adapter appears weaker than the base model on its target task.
        Merging may not be worth preserving this side.
    UNKNOWN_NO_BEHAVIORAL_EVAL
        No behavioral evaluation is available.  This is the default
        when users run a purely structural audit.
    """

    ELIGIBLE = "eligible"
    UNCERTAIN = "uncertain"
    FLAGGED_WEAK = "flagged_weak"
    UNKNOWN_NO_BEHAVIORAL_EVAL = "unknown_no_behavioral_eval"


# ---------------------------------------------------------------------------
# QA result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterQAResult:
    """Lightweight quality-assessment summary for one source adapter.

    Parameters
    ----------
    adapter_path : str
        Path or identifier for the adapter.
    status : EligibilityStatus
        Screening verdict.
    adapter_metric : float or None
        Primary quality metric for the adapter on its target task
        (e.g., perplexity, accuracy).
    base_metric : float or None
        Same metric evaluated on the bare base model.
    metric_name : str
        Name of the metric (e.g. ``"perplexity"``, ``"accuracy"``).
    lower_is_better : bool
        If True, adapter_metric < base_metric means the adapter is better
        (e.g. perplexity).  If False, higher is better (e.g. accuracy).
    eval_dataset : str or None
        Dataset or benchmark used for evaluation.
    notes : str or None
        Free-text explanation of the screening decision.
    evidence : dict
        Arbitrary extra evidence (model card metadata, audit summary, etc.).
    """

    adapter_path: str
    status: EligibilityStatus
    adapter_metric: float | None = None
    base_metric: float | None = None
    metric_name: str = ""
    lower_is_better: bool = True
    eval_dataset: str | None = None
    notes: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_path": self.adapter_path,
            "status": self.status.value,
            "adapter_metric": self.adapter_metric,
            "base_metric": self.base_metric,
            "metric_name": self.metric_name,
            "lower_is_better": self.lower_is_better,
            "eval_dataset": self.eval_dataset,
            "notes": self.notes,
            "evidence": dict(self.evidence),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> AdapterQAResult:
        status_raw = d.get("status", EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL.value)
        try:
            status = EligibilityStatus(status_raw)
        except ValueError:
            status = EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL

        return AdapterQAResult(
            adapter_path=str(d.get("adapter_path", "")),
            status=status,
            adapter_metric=d.get("adapter_metric"),
            base_metric=d.get("base_metric"),
            metric_name=str(d.get("metric_name", "")),
            lower_is_better=bool(d.get("lower_is_better", True)),
            eval_dataset=d.get("eval_dataset"),
            notes=d.get("notes"),
            evidence=dict(d.get("evidence") or {}),
        )


# ---------------------------------------------------------------------------
# Simple heuristic: classify from metric comparison
# ---------------------------------------------------------------------------


def classify_eligibility(
    adapter_path: str,
    adapter_metric: float | None,
    base_metric: float | None,
    *,
    metric_name: str = "",
    lower_is_better: bool = True,
    eval_dataset: str | None = None,
    margin: float = 0.0,
) -> AdapterQAResult:
    """Classify adapter eligibility from a single metric comparison.

    Parameters
    ----------
    adapter_path : str
        Identifier for the adapter.
    adapter_metric : float or None
        Adapter's score on its target task.
    base_metric : float or None
        Base model's score on the same task.
    metric_name : str
        Human-readable metric name.
    lower_is_better : bool
        True for metrics like perplexity where lower = better.
    eval_dataset : str or None
        Dataset used for evaluation.
    margin : float
        Tolerance margin.  Adapter must beat base by at least this much
        to be classified as ELIGIBLE.  Default 0.0.

    Returns
    -------
    AdapterQAResult
    """
    if adapter_metric is None or base_metric is None:
        return AdapterQAResult(
            adapter_path=adapter_path,
            status=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL,
            adapter_metric=adapter_metric,
            base_metric=base_metric,
            metric_name=metric_name,
            lower_is_better=lower_is_better,
            eval_dataset=eval_dataset,
            notes="No behavioral evaluation available.",
        )

    if lower_is_better:
        delta = base_metric - adapter_metric  # positive = adapter is better
    else:
        delta = adapter_metric - base_metric  # positive = adapter is better

    if delta > margin:
        status = EligibilityStatus.ELIGIBLE
        notes = (
            f"Adapter beats base by {abs(delta):.4f} "
            f"({metric_name}, {'lower' if lower_is_better else 'higher'} is better)."
        )
    elif delta >= -margin:
        status = EligibilityStatus.UNCERTAIN
        notes = f"Adapter is within margin of base (delta={delta:.4f}, margin=\u00b1{margin:.4f})."
    else:
        status = EligibilityStatus.FLAGGED_WEAK
        notes = (
            f"Adapter is worse than base by {abs(delta):.4f} ({metric_name}).  Merging may not preserve useful signal."
        )

    return AdapterQAResult(
        adapter_path=adapter_path,
        status=status,
        adapter_metric=adapter_metric,
        base_metric=base_metric,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        eval_dataset=eval_dataset,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Screening for merge audit
# ---------------------------------------------------------------------------


def screen_adapters(
    qa_a: AdapterQAResult | None = None,
    qa_b: AdapterQAResult | None = None,
) -> list[str]:
    """Generate merge-audit warnings from source QA results.

    Returns a list of human-readable warning strings.  Empty list means
    no eligibility concerns were detected (or no QA data was provided).
    """
    warnings: list[str] = []

    for label, qa in [("A", qa_a), ("B", qa_b)]:
        if qa is None:
            continue
        if qa.status == EligibilityStatus.FLAGGED_WEAK:
            msg = f"Source adapter {label} ({qa.adapter_path}) is flagged as weaker than the base model"
            if qa.metric_name:
                msg += f" on {qa.metric_name}"
            if qa.eval_dataset:
                msg += f" ({qa.eval_dataset})"
            msg += (
                ".  Structural merge balancing may not translate to "
                "behavioral improvement.  Consider whether this adapter "
                "is worth preserving."
            )
            warnings.append(msg)
        elif qa.status == EligibilityStatus.UNCERTAIN:
            msg = f"Source adapter {label} ({qa.adapter_path}) shows uncertain standalone quality"
            if qa.metric_name:
                msg += f" on {qa.metric_name}"
            msg += ".  Merge results may be unpredictable."
            warnings.append(msg)
        elif qa.status == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL:
            msg = (
                f"No behavioral evaluation available for adapter {label} "
                f"({qa.adapter_path}).  Merge recommendation is based on "
                f"structural analysis only."
            )
            warnings.append(msg)

    # Both flagged weak is an especially important case
    if (
        qa_a is not None
        and qa_b is not None
        and qa_a.status == EligibilityStatus.FLAGGED_WEAK
        and qa_b.status == EligibilityStatus.FLAGGED_WEAK
    ):
        warnings.append(
            "Both source adapters are flagged as weaker than the base model.  "
            "The merge problem may be ill-posed: consider whether either "
            "adapter should be used at all."
        )

    return warnings
