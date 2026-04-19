"""
Empirically calibrated threshold profiles for merge triage.

Derives architecture-specific thresholds from summary statistics in
FINDINGS.md rather than hardcoded guesses.  Each reference point cites
its source experiment so the provenance chain is auditable.

Available profiles
------------------
STANDARD_ATTENTION : DistilBERT, RoBERTa, Mistral, LLaMA, etc.
    Calibrated from N127 (Mistral-7B merge study), N127 Ext 1-2
    (DistilBERT spectral partitioning), N130-N131 (erank/C_k).

DISENTANGLED_ATTENTION : DeBERTa-v3 family
    Calibrated from N07 (DeBERTa adjudication), N132 (DeBERTa erank
    replication).  Tighter erank and alignment ranges reflecting the
    compressed spectral geometry of disentangled attention.

Limitations
-----------
These thresholds are derived from summary statistics (means, SDs,
correlations), not from a proper ROC curve or cross-validated decision
boundary.  A full calibration requires the per-pair CSVs from
``sidecar/`` — specifically the per-layer metrics and binary merge
outcome labels.  This module documents the gap explicitly and should
be updated when raw data becomes available.

References
----------
- FINDINGS.md §7  (Mistral-7B merge study, Post 3)
- FINDINGS.md §12 (N127 Ext 2, task-dependent partitioning)
- FINDINGS.md §13 (N127 Ext 1, W₀ energy concentration)
- FINDINGS.md §24 (N07, per-module curvature)
- FINDINGS.md §25 (N130, spectral susceptibility / erank)
- FINDINGS.md §26 (N131, composite predictor)
- FINDINGS.md §27 (N132, DeBERTa erank replication)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ScaleValidation(str, Enum):
    """Empirical validation status for a claim or threshold."""

    VALIDATED = "validated"  # Confirmed by experiment with merge outcomes
    PARTIAL = "partial"  # Some aspects confirmed, others not
    UNVALIDATED = "unvalidated"  # No merge-outcome validation at this scale
    REFUTED = "refuted"  # Experiment produced a negative result

# ---------------------------------------------------------------------------
# Reference points — every number traces to a FINDINGS.md section
# ---------------------------------------------------------------------------

# fmt: off
REFERENCE_POINTS: dict[str, dict[str, Any]] = {
    # --- Alignment / overlap ---
    "mistral_same_task_overlap": {
        "value": 0.473,
        "source": "FINDINGS.md §7 (Post 3, Mistral-7B, 9 same-task pairs)",
        "note": "Mean subspace overlap for same-task pairs (different seeds)",
    },
    "mistral_cross_task_overlap": {
        "value": 0.200,
        "source": "FINDINGS.md §7 (Post 3, Mistral-7B, 27 cross-task pairs)",
        "note": "Mean subspace overlap for cross-task pairs",
    },
    "mistral_same_cross_separation": {
        "value": 2.4,
        "source": "FINDINGS.md §7 (t=12.985, p<0.0001)",
        "note": "Ratio of same-task to cross-task mean overlap",
    },
    "distilbert_high_sv_same_task": {
        "value": 0.634,
        "source": "FINDINGS.md §12 (N127 Ext 2, SST-2×SST-2, high-SV band)",
        "note": "Same-task alignment in the high-SV partition",
    },
    "distilbert_high_sv_cross_task": {
        "value": 0.133,
        "source": "FINDINGS.md §12 (N127 Ext 2, SST-2×QNLI, high-SV band)",
        "note": "Cross-task alignment in the high-SV partition",
    },
    "deberta_same_task_compat": {
        "value": 0.449,
        "source": "FINDINGS.md §23 (N07, DeBERTa-v3-base, 4 same-task pairs)",
        "note": "Same-task compatibility score on DeBERTa",
    },
    "deberta_cross_task_compat": {
        "value": 0.160,
        "source": "FINDINGS.md §23 (N07, DeBERTa-v3-base, cross-task)",
        "note": "Cross-task compatibility score on DeBERTa",
    },
    "deberta_same_cross_ratio": {
        "value": 2.8,
        "source": "FINDINGS.md §23 (t=15.87, p<10⁻¹⁵)",
        "note": "Ratio 0.449/0.160 ≈ 2.8×",
    },

    # --- C_k (energy concentration) ---
    "ck_alignment_rho_qnli": {
        "value": 0.56,
        "source": "FINDINGS.md §13 / §25 (N127 Ext 1 + N130, QNLI, DistilBERT)",
        "note": "Spearman ρ between C_k and per-layer alignment, QNLI",
    },
    "ck_alignment_rho_sst2": {
        "value": -0.076,
        "source": "FINDINGS.md §26 (N131, SST-2, DistilBERT)",
        "note": "C_k not predictive for low-erank tasks",
    },
    "ck_alignment_rho_deberta_max": {
        "value": 0.216,
        "source": "FINDINGS.md §27 (N132, MRPC, DeBERTa, p=0.14)",
        "note": "Highest ρ on DeBERTa — still not significant",
    },

    # --- Effective rank (erank) ---
    "erank_distilbert_sst2": {
        "value": 5.47,
        "source": "FINDINGS.md §25 (N130, DistilBERT, SST-2)",
        "note": "Mean entropy effective rank, low-erank task",
    },
    "erank_distilbert_qnli": {
        "value": 13.32,
        "source": "FINDINGS.md §25 (N130, DistilBERT, QNLI)",
        "note": "Mean entropy effective rank, high-erank task",
    },
    "erank_distilbert_ratio": {
        "value": 2.43,
        "source": "FINDINGS.md §25 (13.32/5.47, d=2.05)",
        "note": "QNLI/SST-2 erank ratio — canonical 'very different tasks'",
    },
    "erank_deberta_sst2": {
        "value": 2.03,
        "source": "FINDINGS.md §27 (N132, DeBERTa-v3-base)",
        "note": "DeBERTa erank much lower than DistilBERT",
    },
    "erank_deberta_rte": {
        "value": 3.95,
        "source": "FINDINGS.md §27 (N132, DeBERTa-v3-base, highest erank)",
        "note": "DeBERTa max erank, compressed range (2.0–4.0)",
    },
    "erank_deberta_ratio": {
        "value": 1.23,
        "source": "FINDINGS.md §27 (QNLI/SST-2 = 2.49/2.03)",
        "note": "Compressed erank ratio on disentangled attention",
    },
}
# fmt: on


# ---------------------------------------------------------------------------
# ThresholdProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdProfile:
    """Architecture-specific calibrated thresholds for merge triage.

    Each field documents its derivation from the reference points above.

    .. note::

       These profiles are validated for *source-adapter characterization*
       (spectral metrics, same-task vs. cross-task separation) but not
       necessarily for *per-pair risk prediction* at decoder scale.  The
       ``risk_prediction_validation`` field records the empirical status.

    Attributes
    ----------
    name : human-readable profile name
    alignment_safe : overlap below this → orthogonal, safe to merge
    alignment_risk : overlap above this → significant shared subspace
    erank_ratio_safe : erank ratio below this → similar task complexity
    erank_ratio_risk : erank ratio above this → very different tasks
    ck_predictive : whether C_k is a reliable per-layer alignment predictor
    imbalanced_frob : Frobenius norm ratio above this → imbalanced
    risk_prediction_validation : empirical validation status for risk prediction
    risk_prediction_note : one-sentence provenance
    """

    name: str
    alignment_safe: float
    alignment_risk: float
    erank_ratio_safe: float
    erank_ratio_risk: float
    ck_predictive: bool
    imbalanced_frob: float
    risk_prediction_validation: ScaleValidation = ScaleValidation.UNVALIDATED
    risk_prediction_note: str = ""

    def to_verdict_thresholds(self) -> dict[str, float]:
        """Convert to VerdictThresholds-compatible kwargs.

        Maps calibrated alignment thresholds to the verdict decision tree's
        low_overlap / high_overlap parameters.  The alignment_safe threshold
        becomes the orthogonal-safe boundary (low_overlap), and alignment_risk
        becomes the significant-interaction boundary (high_overlap).
        """
        return {
            "low_overlap": self.alignment_safe,
            "high_overlap": self.alignment_risk,
            "imbalanced_frob": self.imbalanced_frob,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alignment_safe": self.alignment_safe,
            "alignment_risk": self.alignment_risk,
            "erank_ratio_safe": self.erank_ratio_safe,
            "erank_ratio_risk": self.erank_ratio_risk,
            "ck_predictive": self.ck_predictive,
            "imbalanced_frob": self.imbalanced_frob,
            "risk_prediction_validation": self.risk_prediction_validation.value,
            "risk_prediction_note": self.risk_prediction_note,
        }


# ---------------------------------------------------------------------------
# Calibrated profiles
# ---------------------------------------------------------------------------

# Standard attention (DistilBERT, RoBERTa, Mistral, LLaMA, GPT-NeoX, etc.)
#
# Derivation:
#   alignment_safe = 0.15
#     Cross-task full overlap on DistilBERT = 0.089 (§12).  Midpoint between
#     cross-task (0.089) and the low end of the moderate band.  We want truly
#     orthogonal pairs to be below this — 0.15 gives margin above noise.
#
#   alignment_risk = 0.40
#     Same-task overlap on Mistral = 0.473 (§7).  Cross-task = 0.200 (§7).
#     Midpoint = 0.337.  We set the risk threshold at 0.40 — above the midpoint
#     but below same-task mean — so that same-task pairs are flagged for
#     deduplication review, while cross-task pairs mostly fall below.
#
#   erank_ratio_safe = 1.5
#     Within-seed erank variation is small (SD << mean).  A ratio of 1.5×
#     accommodates normal same-task variation without triggering alerts.
#
#   erank_ratio_risk = 2.5
#     The canonical "very different tasks" ratio is QNLI/SST-2 = 2.43× (§25).
#     Setting the threshold at 2.5 means this pair is near the boundary —
#     appropriately, since it *is* the boundary between "same-category" and
#     "very different" tasks.
#
#   ck_predictive = False
#     N133 Phase 2 showed the apparent C_k signal at decoder scale is a
#     Simpson's paradox artifact driven by module-type differences (Q/K/V/O).
#     Stratified analysis shows no within-module C_k → alignment relationship.
#     C_k remains predictive on DistilBERT for QNLI (ρ=0.56, §13/§25) but
#     this is an encoder-scale, within-task finding that does not generalize.
#     Updated April 2026 per N133 Phase 2 verdict.
#
#   imbalanced_frob = 5.0
#     Retained from original defaults.  No experimental data directly
#     calibrates this threshold yet.
#
STANDARD_ATTENTION = ThresholdProfile(
    name="standard_attention",
    alignment_safe=0.15,
    alignment_risk=0.40,
    erank_ratio_safe=1.5,
    erank_ratio_risk=2.5,
    ck_predictive=False,
    imbalanced_frob=5.0,
    risk_prediction_validation=ScaleValidation.UNVALIDATED,
    risk_prediction_note=(
        "Thresholds derived from encoder-scale summary statistics (DistilBERT, "
        "Mistral-7B same/cross-task separation). Per-pair risk prediction not "
        "validated against merge outcomes at decoder scale (N133)."
    ),
)


# Disentangled attention (DeBERTa-v3 family)
#
# Derivation:
#   alignment_safe = 0.12
#     DeBERTa cross-task compat = 0.160 (§23).  Lower than standard-attention
#     cross-task (0.200).  Orthogonal threshold set below cross-task mean.
#
#   alignment_risk = 0.35
#     DeBERTa same-task compat = 0.449 (§23).  Cross-task = 0.160.
#     Midpoint = 0.305.  Risk threshold at 0.35 — below same-task mean
#     to catch redundancy, above midpoint to avoid false positives.
#
#   erank_ratio_safe = 1.3
#     DeBERTa erank range is compressed (2.0–4.0 vs 5.5–13.3 on DistilBERT).
#     The same-task ratio is QNLI/SST-2 = 1.23× (§27).  A 1.3 threshold
#     accommodates this without being too permissive.
#
#   erank_ratio_risk = 1.8
#     The max ratio is RTE/SST-2 = 3.95/2.03 = 1.95× (§27).  Setting risk
#     at 1.8 means only the most extreme DeBERTa task pairs trigger the alert.
#
#   ck_predictive = False
#     C_k → alignment ρ = 0.11–0.22, all p > 0.14 on DeBERTa (§27).
#     The relationship does NOT replicate on disentangled attention.
#
#   imbalanced_frob = 5.0
#     Same as standard — no architecture-specific data available.
#
DISENTANGLED_ATTENTION = ThresholdProfile(
    name="disentangled_attention",
    alignment_safe=0.12,
    alignment_risk=0.35,
    erank_ratio_safe=1.3,
    erank_ratio_risk=1.8,
    ck_predictive=False,
    imbalanced_frob=5.0,
    risk_prediction_validation=ScaleValidation.PARTIAL,
    risk_prediction_note=(
        "Thresholds derived from DeBERTa-v3 summary statistics. "
        "Same/cross-task separation validated; per-pair risk prediction "
        "not validated against merge outcomes."
    ),
)


# Registry for lookup by name
PROFILES: dict[str, ThresholdProfile] = {
    "standard_attention": STANDARD_ATTENTION,
    "disentangled_attention": DISENTANGLED_ATTENTION,
}


def get_profile(name: str) -> ThresholdProfile:
    """Look up a threshold profile by name.

    Raises
    ------
    KeyError
        If *name* is not a registered profile.
    """
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown threshold profile {name!r}. Available: {available}")
    return PROFILES[name]


# ---------------------------------------------------------------------------
# Calibration gap documentation
# ---------------------------------------------------------------------------

CALIBRATION_GAPS = """
Known calibration gaps (to be addressed when raw data is available):

1. ROC / decision boundary optimization.  The current thresholds are derived
   from summary statistics (means, SDs, ratios).  A proper calibration
   requires per-pair (metrics, outcome) tuples from sidecar/ to compute
   sensitivity, specificity, and optimal operating points.

2. Frobenius imbalance threshold.  No experiment has directly measured the
   relationship between Frobenius norm ratio and merge degradation.  The
   5.0× default is a conservative engineering choice, not an empirical
   finding.

3. Directional agreement thresholds (aligned=0.5, conflicting=-0.3).
   These control the REDUNDANT/CONFLICTING distinction within the
   high-overlap region.  No experiment has calibrated them against
   merge outcomes.

4. Cross-architecture generalization.  Standard-attention profile is
   calibrated from DistilBERT + Mistral-7B data.  Whether the same
   thresholds hold for GPT-2, LLaMA-3, Phi-3, etc. is untested.

5. Scale dependence.  Whether 7B and 70B+ models need different
   thresholds is unknown.

6. Decoder-scale per-pair risk prediction.  N133 (Mistral-7B, 6 tasks,
   12 cross-task pairs) showed that per-pair alignment metrics do not
   predict merge degradation beyond task-family membership.  The three
   diagnosed confounds (metric saturation, task-family aliasing,
   insufficient seed replication) are addressed in the N134 experimental
   design.  Until N134 produces results, risk predictions at decoder
   scale should be treated as unvalidated.
"""
