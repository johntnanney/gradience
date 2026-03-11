"""
Single-adapter QA artifact schema (v1).

Produces a structured JSON artifact from a LoRA adapter audit that captures:

    1. What adapter is this?  (adapter metadata)
    2. What did the structural audit observe?  (structural_summary)
    3. What is the current eligibility judgment?  (eligibility)
    4. Why was that judgment made?  (eligibility.reasons)

The artifact is the output of ``gradience audit-adapter`` and can be fed into
``gradience merge-audit`` via ``--source-a-qa`` / ``--source-b-qa``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from gradience.vnext.merge.eligibility import AdapterQAResult, EligibilityStatus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "gradience.adapter_qa/v1"

# Structural flag thresholds (initial policy — not gospel)
_UTILIZATION_LOW_THRESHOLD = 0.25
_RANK_WASTE_HIGH_THRESHOLD = 0.75  # 1 - utilization
_CONCENTRATED_SPECTRUM_ENERGY_RANK = 2.0
_CONCENTRATED_SPECTRUM_MIN_RANK = 8
_UNDERUTILIZED_STABLE_RANK_RATIO = 0.2


# ---------------------------------------------------------------------------
# Confidence enum (kept as strings, not Enum, to stay lightweight)
# ---------------------------------------------------------------------------

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"


# ---------------------------------------------------------------------------
# Structural flag derivation
# ---------------------------------------------------------------------------


def derive_structural_flags(
    utilization_mean: float,
    rank_waste_ratio: float,
    energy_rank_90_p50: float | None,
    stable_rank_mean: float,
    rank_nominal: int,
) -> list[str]:
    """Derive structural warning flags from audit metrics.

    Returns a list of short flag strings.  Empty list means no anomalies
    above threshold.
    """
    flags: list[str] = []

    if utilization_mean < _UTILIZATION_LOW_THRESHOLD:
        flags.append("low_utilization")

    if rank_waste_ratio > _RANK_WASTE_HIGH_THRESHOLD:
        flags.append("high_rank_waste")

    if (
        energy_rank_90_p50 is not None
        and energy_rank_90_p50 <= _CONCENTRATED_SPECTRUM_ENERGY_RANK
        and rank_nominal >= _CONCENTRATED_SPECTRUM_MIN_RANK
    ):
        flags.append("concentrated_spectrum")

    if rank_nominal > 0 and stable_rank_mean < rank_nominal * _UNDERUTILIZED_STABLE_RANK_RATIO:
        flags.append("underutilized_capacity")

    return flags


# ---------------------------------------------------------------------------
# Confidence derivation
# ---------------------------------------------------------------------------

_FLAG_DESCRIPTIONS: dict[str, str] = {
    "low_utilization": "low utilization across layers",
    "high_rank_waste": "high rank waste",
    "concentrated_spectrum": "spectrum concentrated in very few singular values",
    "underutilized_capacity": "stable rank far below nominal rank",
}


def derive_confidence(
    eval_available: bool,
    status: EligibilityStatus,
    margin: float = 0.0,
    delta: float | None = None,
) -> str:
    """Derive confidence level for the eligibility judgment.

    Key constraint: never assign ``high`` without behavioral evidence.
    """
    if not eval_available:
        return CONFIDENCE_LOW

    # Behavioral eval available — confidence depends on clarity of result
    if status == EligibilityStatus.UNCERTAIN:
        return CONFIDENCE_MEDIUM

    # Clear result (ELIGIBLE or FLAGGED_WEAK) with behavioral data
    if delta is not None and abs(delta) > margin * 2:
        return CONFIDENCE_HIGH

    return CONFIDENCE_MEDIUM


# ---------------------------------------------------------------------------
# Reason list generation
# ---------------------------------------------------------------------------


def build_reasons(
    eval_available: bool,
    status: EligibilityStatus,
    metric_name: str,
    eval_dataset: str | None,
    structural_flags: list[str],
) -> list[str]:
    """Build a human-readable list of reasons for the eligibility judgment."""
    reasons: list[str] = []

    # Behavioral reason
    if eval_available:
        metric_part = metric_name if metric_name else "evaluation metric"
        dataset_part = f" ({eval_dataset})" if eval_dataset else ""
        if status == EligibilityStatus.ELIGIBLE:
            reasons.append(f"adapter outperforms base on {metric_part}{dataset_part}")
        elif status == EligibilityStatus.FLAGGED_WEAK:
            reasons.append(f"adapter underperforms base on {metric_part}{dataset_part}")
        elif status == EligibilityStatus.UNCERTAIN:
            reasons.append(f"adapter performance within margin of base on {metric_part}{dataset_part}")
    else:
        reasons.append("no behavioral evaluation available")

    # Structural reasons
    for flag in structural_flags:
        desc = _FLAG_DESCRIPTIONS.get(flag, flag)
        reasons.append(desc)

    return reasons


# ---------------------------------------------------------------------------
# QA artifact dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterQAArtifact:
    """Structured QA artifact for a single adapter.

    This is the user-facing output of ``audit-adapter``.  It is richer than
    :class:`AdapterQAResult` (which is the merge-facing type) because it
    includes the full structural summary and confidence level.
    """

    # Adapter metadata
    adapter_name: str
    adapter_path: str
    base_model: str
    rank_nominal: int
    n_layers: int

    # Structural summary
    utilization_mean: float
    utilization_median: float
    stable_rank_mean: float
    energy_rank_90_p50: float | None
    rank_waste_ratio: float
    structural_flags: list[str]

    # Behavioral summary
    eval_available: bool
    eval_dataset: str | None = None
    metric_name: str = ""
    adapter_score: float | None = None
    base_score: float | None = None
    lower_is_better: bool = True
    beats_base: bool | None = None

    # Eligibility judgment
    status: EligibilityStatus = EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL
    confidence: str = CONFIDENCE_LOW
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict matching the v1 schema."""
        return {
            "schema": SCHEMA_VERSION,
            "adapter": {
                "name": self.adapter_name,
                "path": self.adapter_path,
                "base_model": self.base_model,
                "rank_nominal": self.rank_nominal,
                "n_layers": self.n_layers,
            },
            "structural_summary": {
                "utilization_mean": self.utilization_mean,
                "utilization_median": self.utilization_median,
                "stable_rank_mean": self.stable_rank_mean,
                "effective_rank_90_median": self.energy_rank_90_p50,
                "rank_waste_ratio": self.rank_waste_ratio,
                "flags": list(self.structural_flags),
            },
            "behavioral_summary": {
                "eval_available": self.eval_available,
                "eval_dataset": self.eval_dataset,
                "metric_name": self.metric_name,
                "adapter_score": self.adapter_score,
                "base_score": self.base_score,
                "lower_is_better": self.lower_is_better,
                "beats_base": self.beats_base,
            },
            "eligibility": {
                "status": self.status.value,
                "confidence": self.confidence,
                "reasons": list(self.reasons),
            },
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> AdapterQAArtifact:
        """Deserialize from a v1 schema dict."""
        adapter = d.get("adapter", {})
        structural = d.get("structural_summary", {})
        behavioral = d.get("behavioral_summary", {})
        eligibility = d.get("eligibility", {})

        try:
            status = EligibilityStatus(eligibility.get("status", "unknown_no_behavioral_eval"))
        except ValueError:
            status = EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL

        return AdapterQAArtifact(
            adapter_name=str(adapter.get("name", "")),
            adapter_path=str(adapter.get("path", "")),
            base_model=str(adapter.get("base_model", "")),
            rank_nominal=int(adapter.get("rank_nominal", 0)),
            n_layers=int(adapter.get("n_layers", 0)),
            utilization_mean=float(structural.get("utilization_mean", 0.0)),
            utilization_median=float(structural.get("utilization_median", 0.0)),
            stable_rank_mean=float(structural.get("stable_rank_mean", 0.0)),
            energy_rank_90_p50=structural.get("effective_rank_90_median", structural.get("energy_rank_90_p50")),
            rank_waste_ratio=float(structural.get("rank_waste_ratio", 0.0)),
            structural_flags=list(structural.get("flags", [])),
            eval_available=bool(behavioral.get("eval_available", False)),
            eval_dataset=behavioral.get("eval_dataset"),
            metric_name=str(behavioral.get("metric_name", "")),
            adapter_score=behavioral.get("adapter_score"),
            base_score=behavioral.get("base_score"),
            lower_is_better=bool(behavioral.get("lower_is_better", True)),
            beats_base=behavioral.get("beats_base"),
            status=status,
            confidence=str(eligibility.get("confidence", CONFIDENCE_LOW)),
            reasons=list(eligibility.get("reasons", [])),
        )

    def to_qa_result(self) -> AdapterQAResult:
        """Bridge to the merge-facing :class:`AdapterQAResult` type.

        Extracts the fields that ``merge-audit`` needs for eligibility
        screening.
        """
        return AdapterQAResult(
            adapter_path=self.adapter_path,
            status=self.status,
            adapter_metric=self.adapter_score,
            base_metric=self.base_score,
            metric_name=self.metric_name,
            lower_is_better=self.lower_is_better,
            eval_dataset=self.eval_dataset,
            notes="; ".join(self.reasons) if self.reasons else None,
            evidence={
                "confidence": self.confidence,
                "structural_flags": list(self.structural_flags),
                "rank_waste_ratio": self.rank_waste_ratio,
                "utilization_mean": self.utilization_mean,
            },
        )


# ---------------------------------------------------------------------------
# Builder — the main entry point
# ---------------------------------------------------------------------------


def build_qa_artifact(
    audit_result: Any,
    *,
    adapter_path: str = "",
    base_model: str = "",
    adapter_score: float | None = None,
    base_score: float | None = None,
    metric_name: str = "",
    lower_is_better: bool = True,
    eval_dataset: str | None = None,
    margin: float = 0.0,
) -> AdapterQAArtifact:
    """Build a QA artifact from a :class:`LoRAAuditResult` and optional behavioral data.

    Parameters
    ----------
    audit_result
        A ``LoRAAuditResult`` from ``audit_lora_peft_dir``.
    adapter_path
        Path or identifier for the adapter.  Falls back to ``audit_result.peft_dir``.
    base_model
        Base model identifier (e.g. ``meta-llama/Llama-2-7b-hf``).
    adapter_score, base_score
        Behavioral evaluation scores.  Both must be provided for behavioral
        eligibility classification.
    metric_name
        Name of the evaluation metric.
    lower_is_better
        True for metrics like perplexity where lower = better.
    eval_dataset
        Dataset used for evaluation.
    margin
        Tolerance margin for eligibility classification.
    """
    from pathlib import Path

    # --- Adapter metadata ---
    path_str = adapter_path or str(audit_result.peft_dir or "")
    name = Path(path_str).name if path_str else ""

    rank_nominal = 0
    if audit_result.layers:
        rank_nominal = audit_result.layers[0].r

    # --- Structural summary ---
    utilization_mean = audit_result.utilization_mean
    rank_waste_ratio = 1.0 - utilization_mean

    # Compute median utilization from layers
    if audit_result.layers:
        utilization_median = statistics.median(layer.utilization for layer in audit_result.layers)
    else:
        utilization_median = utilization_mean

    structural_flags = derive_structural_flags(
        utilization_mean=utilization_mean,
        rank_waste_ratio=rank_waste_ratio,
        energy_rank_90_p50=audit_result.energy_rank_90_p50,
        stable_rank_mean=audit_result.stable_rank_mean,
        rank_nominal=rank_nominal,
    )

    # --- Behavioral summary ---
    eval_available = adapter_score is not None and base_score is not None

    beats_base: bool | None = None
    delta: float | None = None
    if eval_available:
        assert adapter_score is not None and base_score is not None
        if lower_is_better:
            delta = base_score - adapter_score
        else:
            delta = adapter_score - base_score
        beats_base = delta > 0

    # --- Eligibility ---
    from gradience.vnext.merge.eligibility import classify_eligibility

    qa_result = classify_eligibility(
        adapter_path=path_str,
        adapter_metric=adapter_score,
        base_metric=base_score,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        eval_dataset=eval_dataset,
        margin=margin,
    )
    status = qa_result.status

    confidence = derive_confidence(
        eval_available=eval_available,
        status=status,
        margin=margin,
        delta=delta,
    )

    reasons = build_reasons(
        eval_available=eval_available,
        status=status,
        metric_name=metric_name,
        eval_dataset=eval_dataset,
        structural_flags=structural_flags,
    )

    return AdapterQAArtifact(
        adapter_name=name,
        adapter_path=path_str,
        base_model=base_model,
        rank_nominal=rank_nominal,
        n_layers=audit_result.n_layers,
        utilization_mean=utilization_mean,
        utilization_median=utilization_median,
        stable_rank_mean=audit_result.stable_rank_mean,
        energy_rank_90_p50=audit_result.energy_rank_90_p50,
        rank_waste_ratio=rank_waste_ratio,
        structural_flags=structural_flags,
        eval_available=eval_available,
        eval_dataset=eval_dataset,
        metric_name=metric_name,
        adapter_score=adapter_score,
        base_score=base_score,
        lower_is_better=lower_is_better,
        beats_base=beats_base,
        status=status,
        confidence=confidence,
        reasons=reasons,
    )
