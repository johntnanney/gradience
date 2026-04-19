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

from gradience.exceptions import QASchemaError
from gradience.vnext.merge.eligibility import ConfidenceLevel, EligibilityStatus
from gradience.vnext.merge.recommend import (
    MergeRecommendation,
    PairDiagnosis,
    diagnose_pair,
    recommend_merge,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_ID = "gradience.merge_qa_report/v1.1"
_SCHEMA_V1 = "gradience.merge_qa_report/v1"
_ACCEPTED_SCHEMAS = frozenset({SCHEMA_ID, _SCHEMA_V1})

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

PAIR_RISK_VALUES = frozenset({"low", "medium", "high"})
CORE_SPACE_STATUS_VALUES = frozenset(
    {
        "compatible",
        "marginal",
        "incompatible",
        "not_applicable",
    }
)
OVER_ACCUMULATION_ADVISORY_VALUES = frozenset({"none", "watch", "elevated"})

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
class CoreSpaceSummary:
    """Optional pair-level core-space diagnostic block."""

    shared_basis_score: float
    basis_distortion: float
    effective_shared_rank: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_basis_score": float(self.shared_basis_score),
            "basis_distortion": float(self.basis_distortion),
            "effective_shared_rank": int(self.effective_shared_rank),
            "status": self.status,
        }


def _validate_adapter_summary(raw: dict[str, Any], section_name: str) -> AdapterSummary:
    """Validate and construct an AdapterSummary from a raw dict.

    Required fields: ``path`` (str), ``rank`` (numeric -> int).
    Optional: ``alpha`` (numeric -> float, default 0.0),
    ``n_layers`` (int, default 0), ``base_model`` (str, default ""),
    ``eligibility_status`` (valid EligibilityStatus value or None).

    Raises ``QASchemaError`` on contract violations.
    Extra keys are silently ignored.
    """
    # path (required, str)
    if "path" not in raw:
        raise QASchemaError(f"Missing required field: {section_name}.path")
    path = str(raw["path"])

    # rank (required, numeric -> int)
    if "rank" not in raw:
        raise QASchemaError(f"Missing required field: {section_name}.rank")
    raw_rank = raw["rank"]
    if not isinstance(raw_rank, (int, float)):
        raise QASchemaError(f"Field '{section_name}.rank' must be numeric, got {type(raw_rank).__name__}")
    rank = int(raw_rank)

    # alpha (optional, numeric -> float)
    raw_alpha = raw.get("alpha", 0.0)
    if not isinstance(raw_alpha, (int, float)):
        raise QASchemaError(f"Field '{section_name}.alpha' must be numeric, got {type(raw_alpha).__name__}")
    alpha = float(raw_alpha)

    # n_layers (optional, int)
    n_layers = int(raw.get("n_layers", 0))

    # base_model (optional, str)
    base_model = str(raw.get("base_model", ""))

    # eligibility_status (optional; if present and non-null, must be valid EligibilityStatus)
    raw_eligibility = raw.get("eligibility_status")
    if raw_eligibility is None:
        eligibility_status = None
    else:
        try:
            EligibilityStatus(raw_eligibility)
        except ValueError:
            raise QASchemaError(
                f"Unknown eligibility_status in {section_name}: '{raw_eligibility}'. "
                f"Valid values: {[e.value for e in EligibilityStatus]}"
            ) from None
        eligibility_status = raw_eligibility

    return AdapterSummary(
        path=path,
        rank=rank,
        alpha=alpha,
        n_layers=n_layers,
        base_model=base_model,
        eligibility_status=eligibility_status,
    )


def _validate_core_space_summary(raw: dict[str, Any]) -> CoreSpaceSummary:
    """Validate optional core_space block when present."""
    required = ("shared_basis_score", "basis_distortion", "effective_shared_rank", "status")
    for field_name in required:
        if field_name not in raw:
            raise QASchemaError(f"Missing required field: core_space.{field_name}")

    score = raw["shared_basis_score"]
    if not isinstance(score, (int, float)):
        raise QASchemaError(f"Field 'core_space.shared_basis_score' must be numeric, got {type(score).__name__}")

    distortion = raw["basis_distortion"]
    if not isinstance(distortion, (int, float)):
        raise QASchemaError(f"Field 'core_space.basis_distortion' must be numeric, got {type(distortion).__name__}")

    rank = raw["effective_shared_rank"]
    if not isinstance(rank, int):
        raise QASchemaError(f"Field 'core_space.effective_shared_rank' must be int, got {type(rank).__name__}")

    status = str(raw["status"])
    if status not in CORE_SPACE_STATUS_VALUES:
        raise QASchemaError(
            f"Invalid core_space.status '{status}'. Must be one of: {sorted(CORE_SPACE_STATUS_VALUES)}"
        )

    return CoreSpaceSummary(
        shared_basis_score=float(score),
        basis_distortion=float(distortion),
        effective_shared_rank=int(rank),
        status=status,
    )


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
    confidence: ConfidenceLevel
    confidence_note: str  # how much to trust the recommendation
    caveats: tuple[str, ...]  # things the user should know
    verdict_distribution: dict[str, int]  # {"safe": N, "redundant": N, ...}
    compatibility_score: float
    core_space: CoreSpaceSummary | None = None
    task_relationship_advisory: str | None = None  # additive advisory when adapters trained on different tasks
    task_relationship: str | None = None  # machine-readable: "same_task", "same_family", "cross_task", or None
    over_accumulation_advisory: str | None = None  # "watch" | "elevated" (None when no watch)
    over_accumulation_summary: str | None = None
    over_accumulation_high_risk_layers: int = 0
    over_accumulation_watch_layers: int = 0
    scale_validation: dict[str, str] | None = None  # provenance block for decoder-scale models

    def to_dict(self) -> dict[str, Any]:
        d = {
            "schema": SCHEMA_ID,
            "adapter_a": self.adapter_a.to_dict(),
            "adapter_b": self.adapter_b.to_dict(),
            "pair_risk": self.pair_risk,
            "dominant_issue": self.dominant_issue,
            "dominant_issue_detail": self.dominant_issue_detail,
            "recommended_action": self.recommended_action,
            "recommended_strategy": self.recommended_strategy,
            "confidence": self.confidence.value,
            "confidence_note": self.confidence_note,
            "caveats": list(self.caveats),
            "verdict_distribution": self.verdict_distribution,
            "compatibility_score": round(self.compatibility_score, 4),
        }
        if self.core_space is not None:
            d["core_space"] = self.core_space.to_dict()
        if self.task_relationship_advisory is not None:
            d["task_relationship_advisory"] = self.task_relationship_advisory
        if self.task_relationship is not None:
            d["task_relationship"] = self.task_relationship
        if self.over_accumulation_advisory is not None:
            d["over_accumulation_advisory"] = self.over_accumulation_advisory
            d["over_accumulation_summary"] = self.over_accumulation_summary or ""
            d["over_accumulation_high_risk_layers"] = int(self.over_accumulation_high_risk_layers)
            d["over_accumulation_watch_layers"] = int(self.over_accumulation_watch_layers)
        if self.scale_validation is not None:
            d["scale_validation"] = self.scale_validation
        return d

    def to_json(self, path: Path | str) -> None:
        """Write QA report to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MergeQAReport:
        """Deserialize from a v1 schema dict.

        This is the single canonical gatekeeper for the merge_qa_report/v1
        schema.  Validates schema identity, required sections, type
        enforcement, and controlled vocabularies.  Raises
        ``QASchemaError`` for contract violations.  Extra keys are
        silently ignored for forward compatibility.
        """
        # --- Schema identity ---
        if "schema" not in d:
            raise QASchemaError("Missing required field: schema")
        if d["schema"] not in _ACCEPTED_SCHEMAS:
            raise QASchemaError(f"Expected schema in {sorted(_ACCEPTED_SCHEMAS)}, got '{d['schema']}'")

        # --- Required adapter sections ---
        for section_name in ("adapter_a", "adapter_b"):
            if section_name not in d:
                raise QASchemaError(f"Missing required section: {section_name}")
            if not isinstance(d[section_name], dict):
                raise QASchemaError(f"Section '{section_name}' must be a dict")

        adapter_a = _validate_adapter_summary(d["adapter_a"], "adapter_a")
        adapter_b = _validate_adapter_summary(d["adapter_b"], "adapter_b")

        # --- pair_risk ---
        if "pair_risk" not in d:
            raise QASchemaError("Missing required field: pair_risk")
        pair_risk = d["pair_risk"]
        if pair_risk not in PAIR_RISK_VALUES:
            raise QASchemaError(f"Invalid pair_risk '{pair_risk}'. Must be one of: {sorted(PAIR_RISK_VALUES)}")

        # --- dominant_issue ---
        if "dominant_issue" not in d:
            raise QASchemaError("Missing required field: dominant_issue")
        dominant_issue = d["dominant_issue"]
        if dominant_issue not in DOMINANT_ISSUE_LABELS:
            raise QASchemaError(
                f"Unknown dominant_issue '{dominant_issue}'. Must be one of: {sorted(DOMINANT_ISSUE_LABELS)}"
            )

        # --- recommended_strategy (required, but lenient on values for forward compat) ---
        if "recommended_strategy" not in d:
            raise QASchemaError("Missing required field: recommended_strategy")
        recommended_strategy = str(d["recommended_strategy"])

        # --- confidence ---
        if "confidence" not in d:
            raise QASchemaError("Missing required field: confidence")
        try:
            confidence = ConfidenceLevel(d["confidence"])
        except ValueError:
            raise QASchemaError(
                f"Invalid confidence '{d['confidence']}'. Must be one of: {[c.value for c in ConfidenceLevel]}"
            ) from None

        # --- compatibility_score (numeric -> float) ---
        if "compatibility_score" not in d:
            raise QASchemaError("Missing required field: compatibility_score")
        raw_score = d["compatibility_score"]
        if not isinstance(raw_score, (int, float)):
            raise QASchemaError(f"Field 'compatibility_score' must be numeric, got {type(raw_score).__name__}")
        compatibility_score = float(raw_score)

        # --- optional core_space block ---
        raw_core_space = d.get("core_space")
        if raw_core_space is None:
            core_space = None
        else:
            if not isinstance(raw_core_space, dict):
                raise QASchemaError("Field 'core_space' must be a dict when present")
            core_space = _validate_core_space_summary(raw_core_space)

        # --- Optional string fields (backfill to "" if absent) ---
        dominant_issue_detail = str(d.get("dominant_issue_detail", ""))
        confidence_note = str(d.get("confidence_note", ""))
        recommended_action = str(d.get("recommended_action", ""))

        # --- caveats (list[str] if present, backfill to () if absent) ---
        raw_caveats = d.get("caveats")
        if raw_caveats is None:
            caveats: tuple[str, ...] = ()
        else:
            if not isinstance(raw_caveats, list) or not all(isinstance(x, str) for x in raw_caveats):
                raise QASchemaError("Field 'caveats' must be a list of strings")
            caveats = tuple(raw_caveats)

        # --- verdict_distribution (dict with int values if present, backfill to {} if absent) ---
        raw_vd = d.get("verdict_distribution")
        if raw_vd is None:
            verdict_distribution: dict[str, int] = {}
        else:
            if not isinstance(raw_vd, dict):
                raise QASchemaError("Field 'verdict_distribution' must be a dict")
            for k, v in raw_vd.items():
                if not isinstance(v, int):
                    raise QASchemaError(f"verdict_distribution['{k}'] must be int, got {type(v).__name__}")
            verdict_distribution = raw_vd

        # --- Optional task_relationship_advisory (str or None) ---
        raw_advisory = d.get("task_relationship_advisory")
        task_relationship_advisory = str(raw_advisory) if raw_advisory is not None else None

        # --- Optional task_relationship (str or None) ---
        _VALID_TASK_RELATIONSHIPS = {"same_task", "same_family", "cross_task"}
        raw_rel = d.get("task_relationship")
        task_relationship: str | None = None
        if raw_rel is not None:
            raw_rel_str = str(raw_rel)
            if raw_rel_str in _VALID_TASK_RELATIONSHIPS:
                task_relationship = raw_rel_str
            # silently ignore unknown values for forward compat

        # --- Optional over-accumulation advisory ---
        raw_oa_advisory = d.get("over_accumulation_advisory")
        over_accumulation_advisory: str | None = None
        over_accumulation_summary: str | None = None
        over_accumulation_high_risk_layers = 0
        over_accumulation_watch_layers = 0
        if raw_oa_advisory is not None:
            oa_value = str(raw_oa_advisory)
            if oa_value not in OVER_ACCUMULATION_ADVISORY_VALUES:
                raise QASchemaError(
                    "Invalid over_accumulation_advisory "
                    f"'{oa_value}'. Must be one of: {sorted(OVER_ACCUMULATION_ADVISORY_VALUES)}"
                )
            if oa_value != "none":
                over_accumulation_advisory = oa_value
                over_accumulation_summary = str(d.get("over_accumulation_summary", ""))
                over_accumulation_high_risk_layers = int(d.get("over_accumulation_high_risk_layers", 0))
                over_accumulation_watch_layers = int(d.get("over_accumulation_watch_layers", 0))

        # --- Optional scale_validation block (v1.1) ---
        raw_sv = d.get("scale_validation")
        scale_validation: dict[str, str] | None = None
        if raw_sv is not None and isinstance(raw_sv, dict):
            scale_validation = {str(k): str(v) for k, v in raw_sv.items()}

        return cls(
            adapter_a=adapter_a,
            adapter_b=adapter_b,
            pair_risk=pair_risk,
            dominant_issue=dominant_issue,
            dominant_issue_detail=dominant_issue_detail,
            recommended_action=recommended_action,
            recommended_strategy=recommended_strategy,
            confidence=confidence,
            confidence_note=confidence_note,
            caveats=caveats,
            verdict_distribution=verdict_distribution,
            compatibility_score=compatibility_score,
            core_space=core_space,
            task_relationship_advisory=task_relationship_advisory,
            task_relationship=task_relationship,
            over_accumulation_advisory=over_accumulation_advisory,
            over_accumulation_summary=over_accumulation_summary,
            over_accumulation_high_risk_layers=over_accumulation_high_risk_layers,
            over_accumulation_watch_layers=over_accumulation_watch_layers,
            scale_validation=scale_validation,
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
    is_decoder_scale: bool = False,
) -> str:
    """One-sentence recommended action."""
    if diag.eligibility.both_weak:
        return (
            "Reconsider merging: both adapters underperform the base model. "
            "Merge will produce a structurally valid but likely unhelpful result."
        )

    if diag.overall_risk == "high":
        n_conf = getattr(agg, "n_conflicting", 0)
        if is_decoder_scale:
            return (
                "Structural analysis suggests caution (risk prediction unvalidated "
                "at this scale). Validate on downstream task."
            )
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
        if is_decoder_scale:
            return "Merge appears structurally compatible (not validated at this scale)."
        return "Merge is safe. Use audit-aware strategy or norm-equalized baseline."

    # medium risk
    if is_decoder_scale:
        return (
            "Structural analysis suggests moderate compatibility "
            "(risk prediction unvalidated at this scale). Validate on downstream task."
        )
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
        parts.append("— both adapters have user-reported behavioral quality")

    return ". ".join(p if p.startswith("—") or p.startswith("(") else p for p in parts[:1]) + " " + " ".join(parts[1:])


def _caveats(diag: PairDiagnosis, rec: MergeRecommendation, is_decoder_scale: bool = False) -> tuple[str, ...]:
    """Build caveats list from diagnosis and recommendation."""
    caveats: list[str] = []

    # Decoder-scale provenance caveat (most important, goes first)
    if is_decoder_scale:
        caveats.append(
            "DECODER-SCALE PROVENANCE: Verdict thresholds, risk levels, and strategy "
            "recommendations were calibrated on encoder-scale experiments and have not "
            "been validated against merge outcomes at this model scale. The spectral "
            "metrics and same-task/cross-task classification remain valid. "
            "Treat risk predictions as indicative, not calibrated."
        )

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

    if diag.over_accumulation_advisory != "none" and diag.over_accumulation_summary:
        caveats.append(diag.over_accumulation_summary)

    # From recommendation warnings (deduplicate by exact match)
    existing = set(caveats)
    for w in rec.warnings:
        if w not in existing:
            caveats.append(w)
            existing.add(w)

    return tuple(caveats)


def _derive_strategy(diag: PairDiagnosis, rec: MergeRecommendation) -> str:
    """Derive the primary recommended strategy from diagnosis.

    Policy:
      low risk + no compression                      -> "linear"
      medium risk + imbalance dominant + no compress  -> "linear"  (rebalanced coefficients)
      medium risk + no imbalance + no compression     -> "norm_equalized"
      otherwise (high risk or compression needed)     -> "audit_aware"

    Note: norm equalization is contraindicated for imbalanced pairs because
    equalizing norms amplifies the weaker adapter's spectrum into the overlap
    zone, increasing cross-term spectral inflation.  The per-layer rebalanced
    coefficients (from the imbalanced verdict) are more appropriate.
    """
    if diag.compression_needed:
        return "audit_aware"
    if diag.overall_risk == "low":
        return "linear"
    if diag.overall_risk == "medium":
        # Check whether the dominant concern is norm imbalance.
        # Norm equalization can *increase* spectral inflation for imbalanced
        # pairs by amplifying the weaker adapter into the overlap zone.
        # Use linear with rebalanced coefficients instead.
        has_imbalanced = any(ld.verdict == "imbalanced" for ld in diag.layer_diagnoses)
        if has_imbalanced:
            return "linear"
        return "norm_equalized"
    return "audit_aware"


def _derive_confidence(diag: PairDiagnosis, score: float) -> ConfidenceLevel:
    """Derive categorical confidence level."""
    if not diag.eligibility.has_data:
        return ConfidenceLevel.LOW
    if diag.overall_risk == "high":
        return ConfidenceLevel.LOW
    if diag.eligibility.both_eligible and score >= 0.8 and diag.overall_risk == "low":
        return ConfidenceLevel.HIGH
    return ConfidenceLevel.MEDIUM


def _task_relationship(report: Any) -> tuple[str | None, str | None]:
    """Classify the task relationship and build an advisory if needed.

    Returns (task_relationship, advisory_text):
    - ("same_task", None) — exact same eval_dataset
    - ("same_family", advisory_text) — different datasets, same validated family
    - ("cross_task", advisory_text) — different datasets, different/unknown families
    - (None, None) — insufficient metadata to classify
    """
    from gradience.vnext.merge.task_families import family_label

    source_qa = getattr(report, "source_qa", None)
    if source_qa is None:
        return None, None

    qa_a = source_qa.get("adapter_a") if isinstance(source_qa, dict) else None
    qa_b = source_qa.get("adapter_b") if isinstance(source_qa, dict) else None
    if qa_a is None or qa_b is None:
        return None, None

    ds_a = qa_a.get("eval_dataset")
    ds_b = qa_b.get("eval_dataset")
    if ds_a is None or ds_b is None:
        return None, None
    if ds_a == ds_b:
        return "same_task", None

    # Different datasets — check family registry
    fam = family_label(ds_a, ds_b)
    if fam is not None:
        advisory = (
            f"Same-family merge: adapters were evaluated on different datasets "
            f"({ds_a} vs {ds_b}) but both belong to the {fam} family. "
            f"Similar tasks tend to merge well."
        )
        return "same_family", advisory

    advisory = (
        f"Cross-task merge: adapters were evaluated on different tasks "
        f"({ds_a} vs {ds_b}). Linear merges across task boundaries "
        f"may degrade the weaker task's performance."
    )
    return "cross_task", advisory


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
    from gradience.vnext.merge.scale_detection import DetectedScale, detect_scale, needs_risk_provenance_warning

    diag = diagnose_pair(report)
    rec = recommend_merge(report)
    agg = report.aggregate

    adapter_a_info = report.adapter_a
    adapter_b_info = report.adapter_b

    # Scale detection for provenance warnings
    base_model_a = getattr(adapter_a_info, "base_model", "") or ""
    base_model_b = getattr(adapter_b_info, "base_model", "") or ""
    # Use adapter A's base model as the primary signal (they should match)
    primary_base = base_model_a or base_model_b
    is_decoder_scale = needs_risk_provenance_warning(primary_base) if primary_base else False
    detected_scale = detect_scale(primary_base) if primary_base else DetectedScale.UNKNOWN

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

    core_space_summary = None
    raw_core_space = getattr(report, "core_space", None)
    if raw_core_space is not None:
        # Accept either the typed internal aggregate object or a plain dict.
        if hasattr(raw_core_space, "shared_basis_score_mean"):
            core_space_summary = CoreSpaceSummary(
                shared_basis_score=float(raw_core_space.shared_basis_score_mean),
                basis_distortion=float(raw_core_space.basis_distortion_mean),
                effective_shared_rank=int(raw_core_space.effective_shared_rank_median),
                status=str(raw_core_space.status),
            )
        elif isinstance(raw_core_space, dict):
            score_value = raw_core_space.get("shared_basis_score")
            if score_value is None:
                score_value = raw_core_space.get("shared_basis_score_mean", 0.0)
            distortion_value = raw_core_space.get("basis_distortion")
            if distortion_value is None:
                distortion_value = raw_core_space.get("basis_distortion_mean", 0.0)
            rank_value = raw_core_space.get("effective_shared_rank")
            if rank_value is None:
                rank_value = raw_core_space.get("effective_shared_rank_median", 0)
            core_space_summary = CoreSpaceSummary(
                shared_basis_score=float(score_value),
                basis_distortion=float(distortion_value),
                effective_shared_rank=int(rank_value),
                status=str(raw_core_space.get("status", "not_applicable")),
            )

    _task_rel = _task_relationship(report)

    # Build scale_validation block for decoder-scale models
    scale_validation_block: dict[str, str] | None = None
    if is_decoder_scale:
        scale_validation_block = {
            "detected_scale": detected_scale.value,
            "risk_prediction_status": "unvalidated",
            "source_geometry_status": "validated",
            "note": (
                "Per-pair risk prediction not validated at decoder scale (N133). "
                "Source-adapter geometry validated."
            ),
        }

    return MergeQAReport(
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        pair_risk=diag.overall_risk,
        dominant_issue=issue_label,
        dominant_issue_detail=issue_detail,
        recommended_action=_recommended_action(diag, rec, agg, is_decoder_scale=is_decoder_scale),
        recommended_strategy=_derive_strategy(diag, rec),
        confidence=confidence,
        confidence_note=_confidence_note(diag, score),
        caveats=_caveats(diag, rec, is_decoder_scale=is_decoder_scale),
        verdict_distribution=verdict_dist,
        compatibility_score=score,
        core_space=core_space_summary,
        task_relationship_advisory=_task_rel[1],
        task_relationship=_task_rel[0],
        over_accumulation_advisory=(
            diag.over_accumulation_advisory if diag.over_accumulation_advisory != "none" else None
        ),
        over_accumulation_summary=(
            diag.over_accumulation_summary if diag.over_accumulation_advisory != "none" else None
        ),
        over_accumulation_high_risk_layers=diag.over_accumulation_high_risk_layers,
        over_accumulation_watch_layers=diag.over_accumulation_watch_layers,
        scale_validation=scale_validation_block,
    )


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


def _report_headline(qa: MergeQAReport) -> str:
    """Generate a one-line report headline for quick scanning."""
    has_advisory = qa.task_relationship_advisory is not None
    has_weak = any(
        s in (qa.adapter_a.eligibility_status, qa.adapter_b.eligibility_status)
        for s in ("flagged_weak", "unknown_no_behavioral_eval")
    )

    if has_weak:
        return "Weak-source pair — low-priority candidate"
    if has_advisory:
        if qa.task_relationship == "same_family":
            return "Same-family pair — plausible candidate"
        if qa.pair_risk == "high":
            return "Cross-task pair — high structural risk — caution region"
        return "Cross-task pair — caution region"
    if qa.pair_risk == "low":
        return "Same-task pair — safe region"
    return "Same-task pair — reasonable candidate"


def _report_interpretation(qa: MergeQAReport) -> str:
    """Generate a short interpretation line for the end of the report."""
    has_advisory = qa.task_relationship_advisory is not None
    has_weak = any(
        s in (qa.adapter_a.eligibility_status, qa.adapter_b.eligibility_status)
        for s in ("flagged_weak", "unknown_no_behavioral_eval")
    )

    if has_weak:
        return "Weak-source influence likely dominates; low-value candidate."
    if has_advisory:
        if qa.task_relationship == "same_family":
            return "Same-family pair (different datasets); structurally similar tasks tend to merge well."
        return "Cross-task caution pair; do not prioritize for casual merge exploration."
    return "Same-task pair; reasonable candidate to keep in the evaluation subset."


def format_qa_report(qa: MergeQAReport) -> str:
    """Format a MergeQAReport as clean, human-readable text.

    Output is organized for quick scanning and decision support:

    - **Headline** — one-line summary of pair status
    - **Task-boundary warning** (if cross-task) — visually primary
    - **Structural result** — risk, compatibility, layer verdicts
    - **Behavioral status** — source eligibility and confidence
    - **Eligibility warnings** — caveats about data gaps
    - **Recommended action** — strategy and next step
    - **Interpretation** — plain-language summary
    - **Advanced diagnostic detail** (if core-space present)
    """
    lines: list[str] = []

    # ---------------------------------------------------------------
    # Headline
    # ---------------------------------------------------------------
    lines.append("")
    lines.append("  MERGE QA REPORT")
    lines.append("  " + "=" * 60)
    lines.append(f"  {_report_headline(qa)}")

    # ---------------------------------------------------------------
    # Task-boundary section (visually primary, before structural)
    # ---------------------------------------------------------------
    if qa.task_relationship_advisory is not None:
        lines.append("")
        if qa.task_relationship == "same_family":
            lines.append("  TASK-FAMILY NOTE")
            lines.append("  " + "-" * 40)
            lines.append(f"  {qa.task_relationship_advisory}")
            lines.append("  Note: Different datasets but same validated task family.")
            lines.append("        Treat this pair like a same-task candidate.")
        else:
            lines.append("  TASK-BOUNDARY WARNING")
            lines.append("  " + "-" * 40)
            lines.append(f"  {qa.task_relationship_advisory}")
            lines.append("  Action: Do not prioritize this pair unless you have a specific")
            lines.append("          reason to merge across task boundaries.")

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
    if qa.over_accumulation_advisory is not None:
        lines.append(
            f"  Over-accumulation: {qa.over_accumulation_advisory.upper()} "
            f"(high={qa.over_accumulation_high_risk_layers}, watch={qa.over_accumulation_watch_layers})"
        )
        if qa.over_accumulation_summary:
            lines.append(f"                     {qa.over_accumulation_summary}")

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
    lines.append(f"  Confidence:      {qa.confidence.value}")
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

    # ---------------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------------
    lines.append("")
    lines.append("  INTERPRETATION")
    lines.append("  " + "-" * 40)
    lines.append(f"  {_report_interpretation(qa)}")

    # ---------------------------------------------------------------
    # Advanced diagnostic detail (subordinate, at end)
    # ---------------------------------------------------------------
    if qa.core_space is not None:
        lines.append("")
        lines.append("  ADVANCED DIAGNOSTIC DETAIL")
        lines.append("  " + "-" * 40)
        lines.append(f"  Core-space shared basis: {qa.core_space.shared_basis_score:.3f}")
        lines.append(f"  Basis distortion:        {qa.core_space.basis_distortion:.3f}")
        lines.append(f"  Effective shared rank:    {qa.core_space.effective_shared_rank}")
        lines.append(f"  Status:                  {qa.core_space.status}")

    lines.append("")

    return "\n".join(lines)
