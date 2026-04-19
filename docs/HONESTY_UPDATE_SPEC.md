# Honesty Update: Merge-Risk Module

**Status:** Spec (pre-implementation)
**Author:** John Nanney
**Date:** April 2026
**Scope:** `gradience.vnext.merge.*`, `gradience.commands.merge`, `gradience.api`

## 1. Motivation

Experiment N133 established that per-pair merge-risk prediction fails at decoder scale for diagnosed reasons (metric saturation at ~2e-3 resolution, task-family aliasing explaining R^2 = 0.966, and failure of all 16 alternative alignment aggregations to exceed delta-R^2 = 0.01 after family residualization). The merge triage pipeline — verdicts, recommendations, risk levels, and the QA report — was calibrated on encoder-scale experiments (DistilBERT, DeBERTa-v3) and has not been validated against decoder-scale merge outcomes.

A user who installs Gradience today and runs `gradience merge-audit` on two Mistral-7B or LLaMA adapters will receive verdicts (SAFE, CONFLICTING, REDUNDANT, IMBALANCED), risk levels (low/medium/high), compatibility scores, and strategy recommendations — all presented with the same confidence as the encoder-scale case, despite the fact that none of these outputs have been validated against actual merge degradation at decoder scale.

This spec enumerates every user-facing claim that exceeds the evidence, specifies the changes required, and defines the acceptance criteria for a coding agent to implement the update.

## 2. What the Evidence Supports

These capabilities are validated across architectures and scales. They should be preserved without qualification.

**S1. Source-adapter spectral characterization.** SVD decomposition, energy concentration, entropy effective rank, stable rank, and utilization are valid at all scales tested (DistilBERT, DeBERTa-v3, Mistral-7B, and 86 public adapters across 22 architectures). No changes needed to `gradience.spectral.metrics` or `gradience.vnext.audit`.

**S2. Same-task vs. cross-task separation.** Adapters trained on the same task show substantially higher subspace overlap than cross-task adapters. Confirmed at encoder scale (7.8x on DistilBERT, 2.8x on DeBERTa-v3) and decoder scale (B-P1: 0.125 vs 0.041, zero misclassifications on N133). The binary classification "are these adapters from the same task?" is robust.

**S3. Task complexity predicts alignment.** Low-erank tasks produce more tightly aligned adapters. Confirmed across DistilBERT (d = 2.05), DeBERTa (ordering preserved), and Mistral-7B (B-P2: 3.06x, t = 35.8).

**S4. Spectral compression estimates.** Rank-utilization analysis and compression-potential estimates are valid across scales. Median utilization 0.154 across 86 adapters.

**S5. Per-module SNR ordering (decoder scale).** O > V > Q > K module ordering for same-task/cross-task separation is confirmed at decoder scale (N133 Phase 2, F = 308, non-significant interaction p = 0.26).

## 3. What the Evidence Does Not Support

These claims are currently made or implied by the software but lack validation at decoder scale.

**U1. Per-pair risk ranking.** The claim that a higher compatibility score, lower overlap, or fewer conflicting layers predicts lower merge degradation has not been validated at decoder scale. N133 B-P5 showed that `mean_alignment` has structural resolution of ~2e-3 with 63% of cross-task pairs in tie clusters, and that task-family identity alone explains R^2 = 0.966 of merge degradation.

**U2. Verdict thresholds.** The overlap thresholds that drive SAFE/REDUNDANT/CONFLICTING/IMBALANCED verdicts (low_overlap=0.2, high_overlap=0.5, aligned=0.5, conflicting=-0.3) are calibrated from encoder-scale summary statistics. The `calibration.py` module already documents these gaps (see `CALIBRATION_GAPS`), but this documentation is not surfaced to users.

**U3. Strategy recommendations.** The recommendation engine's mapping from risk level to merge strategy (linear for low risk, norm-equalized for medium, audit-aware for high) has not been validated against decoder-scale outcomes. Study 17 showed compression does not improve merge outcomes; the strategy recommendations may inherit similar overconfidence.

**U4. Compatibility score.** The scalar compatibility score presented in CLI output and QA reports has no demonstrated correlation with merge quality at decoder scale.

**U5. Over-accumulation advisory.** The sigma-1 inflation risk model is analytical (derived from spectral theory) but has not been validated against observed decoder-scale merge degradation.

**U6. C_k as per-layer predictor on standard attention at decoder scale.** The `STANDARD_ATTENTION` threshold profile in `calibration.py` sets `ck_predictive=True` and lists Mistral in its description. N133 Phase 2 showed this is a Simpson's paradox artifact at decoder scale — the pooled C_k signal disappears when stratified by module type.

## 4. Required Changes

### 4.1 New: `ScaleValidation` enum and provenance metadata

**File:** `gradience/vnext/merge/calibration.py`

Add an enum and attach it to every threshold profile and every user-facing output:

```python
class ScaleValidation(str, Enum):
    """Empirical validation status for a claim or threshold."""
    VALIDATED = "validated"           # Confirmed by experiment with merge outcomes
    PARTIAL = "partial"              # Some aspects confirmed, others not
    UNVALIDATED = "unvalidated"      # No merge-outcome validation at this scale
    REFUTED = "refuted"              # Experiment produced a negative result

class ScaleContext(str, Enum):
    """Architecture scale category."""
    ENCODER = "encoder"              # DistilBERT, BERT, DeBERTa, RoBERTa
    DECODER_7B = "decoder_7b"        # Mistral-7B, LLaMA-2-7B, etc.
    DECODER_LARGE = "decoder_large"  # 13B+, untested
```

Add a `validation_status` field to `ThresholdProfile`:

```python
@dataclass(frozen=True)
class ThresholdProfile:
    # ... existing fields ...
    risk_prediction_validation: ScaleValidation
    risk_prediction_note: str  # one-sentence provenance
```

Set the values:

- `STANDARD_ATTENTION`: `risk_prediction_validation=ScaleValidation.UNVALIDATED`, note: `"Thresholds derived from encoder-scale summary statistics (DistilBERT, Mistral-7B same/cross-task separation). Per-pair risk prediction not validated against merge outcomes at decoder scale (N133)."`
- `DISENTANGLED_ATTENTION`: `risk_prediction_validation=ScaleValidation.PARTIAL`, note: `"Thresholds derived from DeBERTa-v3 summary statistics. Same/cross-task separation validated; per-pair risk prediction not validated against merge outcomes."`

### 4.2 Fix: `STANDARD_ATTENTION.ck_predictive`

**File:** `gradience/vnext/merge/calibration.py`

Change `ck_predictive=True` to `ck_predictive=False` for the `STANDARD_ATTENTION` profile.

Add a comment:

```python
# ck_predictive = False
#   N133 Phase 2 showed the apparent C_k signal at decoder scale is a
#   Simpson's paradox artifact driven by module-type differences (Q/K/V/O).
#   Stratified analysis shows no within-module C_k → alignment relationship.
#   C_k remains predictive on DistilBERT for QNLI (ρ=0.56, §13/§25) but
#   this is an encoder-scale, within-task finding that does not generalize.
#   Updated April 2026 per N133 Phase 2 verdict.
```

Update the `STANDARD_ATTENTION` profile `name` to include Mistral only for source-adapter geometry, not risk prediction. Alternatively, add a `decoder_scale_note` field.

### 4.3 Add: Scale-aware warnings in CLI output

**File:** `gradience/commands/merge.py` (function `cmd_merge_audit`)

After the existing pretty-print summary (around line 132–149), add scale-detection logic:

```python
# Detect decoder-scale models from adapter metadata
def _is_decoder_scale(report) -> bool:
    """Heuristic: check if base model name suggests decoder-scale."""
    base = getattr(report.adapter_a, 'base_model', '') or ''
    base_lower = base.lower()
    decoder_keywords = ['mistral', 'llama', 'gpt', 'phi', 'gemma', 'qwen', 'falcon', 'yi']
    return any(kw in base_lower for kw in decoder_keywords)
```

When decoder-scale is detected, emit a provenance notice after the verdict line:

```
  Note: Per-pair risk prediction (verdicts, compatibility score, strategy
  recommendations) has not been validated against merge outcomes at decoder
  scale. Source-adapter characterization (spectral metrics, same-task vs.
  cross-task separation) remains valid. See FINDINGS.md §28 for details.
```

This notice should appear in both text and JSON output modes. In JSON, add a top-level field:

```json
{
  "scale_validation": {
    "detected_scale": "decoder_7b",
    "risk_prediction_status": "unvalidated",
    "source_geometry_status": "validated",
    "note": "Per-pair risk prediction not validated at decoder scale (N133). Source-adapter geometry validated."
  }
}
```

### 4.4 Update: QA report caveats

**File:** `gradience/vnext/merge/qa_report.py` (function `_caveats`)

Add a decoder-scale caveat that is injected when the base model is detected as decoder-scale:

```python
# At the TOP of the caveats list (this is the most important caveat)
if _is_decoder_scale(report):
    caveats.insert(0,
        "DECODER-SCALE PROVENANCE: Verdict thresholds, risk levels, and strategy "
        "recommendations were calibrated on encoder-scale experiments and have not "
        "been validated against merge outcomes at this model scale. The spectral "
        "metrics and same-task/cross-task classification remain valid. "
        "Treat risk predictions as indicative, not calibrated."
    )
```

### 4.5 Update: Recommendation language

**File:** `gradience/vnext/merge/qa_report.py` (function `_recommended_action`)

Soften the language for decoder-scale models. Currently:

- `"Merge is safe."` → At decoder scale: `"Merge appears structurally compatible (not validated at this scale)."`
- `"Merge with caution using audit-aware strategy."` → At decoder scale: `"Structural analysis suggests caution (risk prediction unvalidated at this scale). Validate on downstream task."`

The encoder-scale language can remain unchanged. The conditional should use the same `_is_decoder_scale` heuristic.

### 4.6 Update: `format_recommendation` in `recommend.py`

**File:** `gradience/vnext/merge/recommend.py`

The `format_recommendation` function produces the strategy recommendation block in CLI output. Add a `scale_context` parameter that, when set to decoder, appends a one-line provenance note:

```
  [Provenance: risk prediction unvalidated at decoder scale — N133]
```

### 4.7 Update: `CALIBRATION_GAPS` and docstring

**File:** `gradience/vnext/merge/calibration.py`

Add a sixth calibration gap to the `CALIBRATION_GAPS` string:

```
6. Decoder-scale per-pair risk prediction.  N133 (Mistral-7B, 6 tasks,
   12 cross-task pairs) showed that per-pair alignment metrics do not
   predict merge degradation beyond task-family membership.  The three
   diagnosed confounds (metric saturation, task-family aliasing,
   insufficient seed replication) are addressed in the N134 experimental
   design.  Until N134 produces results, risk predictions at decoder
   scale should be treated as unvalidated.
```

Update the module docstring to note that the profiles are validated for *source-adapter characterization* but not for *per-pair risk prediction* at decoder scale.

### 4.8 Update: Public API docstring

**File:** `gradience/api.py` (function `merge_risk_report`)

Add a note to the docstring:

```python
"""Run merge-audit and return MergeQAReport.

...existing docstring...

.. note::

   Risk predictions (verdicts, compatibility scores, strategy
   recommendations) are calibrated on encoder-scale experiments.
   At decoder scale (7B+ parameter models), source-adapter spectral
   characterization is validated, but per-pair risk ranking is not.
   See FINDINGS.md §28 and the N134 experimental design for details.
"""
```

### 4.9 Update: JSON schema version

**File:** `gradience/vnext/merge/qa_report.py`

Bump the schema ID from `"gradience.merge_qa_report/v1"` to `"gradience.merge_qa_report/v1.1"`. The v1.1 schema adds:

- `scale_validation` block (optional, present when decoder-scale detected)
- Decoder-scale caveat in `caveats` list

Ensure `from_dict` accepts both `v1` and `v1.1` for backward compatibility.

### 4.10 New: Scale detection utility

**File:** `gradience/vnext/merge/scale_detection.py` (new file)

Centralize the decoder-scale heuristic so it is not duplicated across modules:

```python
"""
Heuristic scale detection from base-model identifiers.

This module classifies adapters as encoder-scale or decoder-scale based
on the base_model string in adapter metadata. The classification drives
provenance warnings in the merge pipeline.

The heuristic is intentionally conservative: unknown models are classified
as UNKNOWN, which triggers the decoder-scale warning (erring on the side
of honesty).
"""

from enum import Enum

class DetectedScale(str, Enum):
    ENCODER = "encoder"
    DECODER_7B = "decoder_7b"
    DECODER_LARGE = "decoder_large"
    UNKNOWN = "unknown"

# Known encoder-scale model families
_ENCODER_PATTERNS = [
    'distilbert', 'bert-base', 'bert-large', 'roberta', 'deberta',
    'albert', 'electra', 'xlm-roberta',
]

# Known decoder-scale model families (7B class)
_DECODER_7B_PATTERNS = [
    'mistral-7b', 'llama-2-7b', 'llama-3-8b', 'llama-3.1-8b',
    'phi-2', 'phi-3', 'gemma-7b', 'gemma-2b', 'qwen-7b',
    'falcon-7b', 'yi-6b',
]

# Known large decoder models
_DECODER_LARGE_PATTERNS = [
    'llama-2-13b', 'llama-2-70b', 'llama-3-70b', 'mistral-8x7b',
    'mixtral', 'falcon-40b', 'falcon-180b', 'qwen-14b', 'qwen-72b',
    'yi-34b',
]

def detect_scale(base_model: str) -> DetectedScale:
    """Classify base model into scale category.

    Parameters
    ----------
    base_model : str
        The base_model identifier from adapter config (e.g.,
        "mistralai/Mistral-7B-v0.3").

    Returns
    -------
    DetectedScale
        The detected scale. UNKNOWN triggers the same warnings as
        decoder-scale (conservative default).
    """
    lower = base_model.lower()
    for pat in _ENCODER_PATTERNS:
        if pat in lower:
            return DetectedScale.ENCODER
    for pat in _DECODER_LARGE_PATTERNS:
        if pat in lower:
            return DetectedScale.DECODER_LARGE
    for pat in _DECODER_7B_PATTERNS:
        if pat in lower:
            return DetectedScale.DECODER_7B
    return DetectedScale.UNKNOWN

def needs_risk_provenance_warning(base_model: str) -> bool:
    """Return True if risk predictions should carry a provenance warning.

    Currently returns True for everything except known encoder-scale models,
    because risk prediction has only been validated at encoder scale.
    """
    return detect_scale(base_model) != DetectedScale.ENCODER
```

### 4.11 Update: Inventory and neighborhood modules

**Files:** `gradience/vnext/inventory/summary.py`, `gradience/vnext/inventory/neighborhoods.py`, `gradience/vnext/inventory/html_report.py`

These modules aggregate `pair_risk`, `compatibility_score`, and `recommended_strategy` from individual QA reports into portfolio-level views. They inherit the same provenance gap: at decoder scale, the risk counts, compatibility rankings, and neighborhood severity labels are built from unvalidated per-pair predictions.

Changes required:
- `summary.py`: When formatting the inventory summary (text, JSON, and HTML), include a provenance note if *any* adapter pair in the inventory has a decoder-scale base model. The note should appear once, not per-pair. Add a `scale_validation_note` field to `InventorySummary`.
- `neighborhoods.py`: The `_edge_severity` function maps `pair_risk` to neighborhood edge labels. Add a `provenance` field to `MergeNeighborhoodEdge` that carries `"validated"` or `"unvalidated"` based on scale detection.
- `html_report.py`: The HTML report renders pair risk with color-coded badges. For decoder-scale pairs, append a subtle footnote marker (e.g., dagger) that links to a provenance note at the bottom of the report.

### 4.12 Update: Merge-aware monitor

**File:** `gradience/vnext/integrations/merge_aware_monitor.py`

This module tracks `compatibility_score` over training and reports trends. Add a note to the `MergeAwareSnapshot.to_dict()` output when the base model is decoder-scale, indicating that the compatibility score trend is structurally informative but not validated as a risk predictor.

## 5. Files NOT Changed

The following modules are **not** modified by this update:

- `gradience/spectral/metrics.py` — core spectral math, validated at all scales.
- `gradience/vnext/audit/lora_audit.py` — single-adapter audit, not affected.
- `gradience/vnext/audit/rank_policies.py` — rank suggestions, not merge-risk claims.
- `gradience/vnext/merge/strategies.py` — merge execution mechanics, not predictions.
- `gradience/vnext/merge/executor.py` — merge I/O, no claims.
- `gradience/vnext/merge/null_controls.py` — validation baselines, no user-facing claims.
- `gradience/vnext/merge/spectral_compat.py` — principal angle computation, math is correct.
- `gradience/vnext/merge/scale.py` — symmetric scale metrics, no risk claims.

## 6. Test Changes

### 6.1 New tests

**File:** `tests/merge/test_scale_detection.py`

- `test_encoder_models_detected` — DistilBERT, DeBERTa, RoBERTa → ENCODER
- `test_decoder_7b_detected` — Mistral-7B, LLaMA-2-7B → DECODER_7B
- `test_decoder_large_detected` — LLaMA-2-70B, Mixtral → DECODER_LARGE
- `test_unknown_defaults_to_unknown` — arbitrary string → UNKNOWN
- `test_needs_warning_encoder_false` — encoder models → no warning
- `test_needs_warning_decoder_true` — decoder models → warning
- `test_needs_warning_unknown_true` — unknown models → warning (conservative)

**File:** `tests/merge/test_qa_report_honesty.py`

- `test_decoder_scale_caveat_present` — build QA report with Mistral base model, assert the decoder-scale provenance caveat appears in caveats list
- `test_encoder_scale_no_caveat` — build QA report with DistilBERT base model, assert no decoder-scale caveat
- `test_json_output_includes_scale_validation` — merge-audit JSON output includes `scale_validation` block for decoder-scale adapters
- `test_schema_v1_1_round_trip` — serialize and deserialize v1.1 schema
- `test_schema_v1_backward_compat` — v1 schema still deserializes correctly

### 6.2 Updated tests

Any existing test that asserts exact CLI output text or exact JSON schema must be updated to account for:
- The new provenance notice in text output
- The new `scale_validation` block in JSON output
- The schema version bump to v1.1
- Changed `ck_predictive` value in `STANDARD_ATTENTION`

Search for tests that reference `ck_predictive` or `STANDARD_ATTENTION` and update expected values.

## 7. Documentation Changes

### 7.1 FINDINGS.md

No changes needed — the empirical record is already correct.

### 7.2 docs/USER_MANUAL.md

Add a section titled "Scale-Dependent Validation" that explains:
- What Gradience can tell you at encoder scale vs. decoder scale
- Why verdicts and risk predictions carry provenance warnings at decoder scale
- What the N134 experiment is designed to resolve

### 7.3 docs/CLI_CHEATSHEET.md

Add a note under the `merge-audit` command that decoder-scale risk predictions are unvalidated.

## 8. Acceptance Criteria

The update is complete when:

1. `ck_predictive=False` in `STANDARD_ATTENTION` with N133 citation.
2. `ThresholdProfile` carries `risk_prediction_validation` field.
3. `scale_detection.py` exists with `detect_scale()` and `needs_risk_provenance_warning()`.
4. CLI text output includes provenance notice for decoder-scale models.
5. CLI JSON output includes `scale_validation` block for decoder-scale models.
6. QA report `caveats` includes decoder-scale provenance caveat when applicable.
7. Recommendation language is softened for decoder-scale models.
8. `CALIBRATION_GAPS` includes the N133 decoder-scale gap.
9. `api.py::merge_risk_report` docstring includes the provenance note.
10. Schema version bumped to v1.1 with backward compatibility.
11. All new tests pass.
12. All existing tests pass (updated where necessary).
13. No changes to `gradience/spectral/metrics.py` or single-adapter audit modules.

## 9. What This Update Does NOT Do

This update does not:
- Remove or disable the merge triage pipeline. The verdicts and recommendations are still computed — they are annotated with their provenance, not deleted.
- Change any threshold values (except `ck_predictive`). The N134 experiment may produce new thresholds; this update only adds honesty about the current ones.
- Add new metrics or predictors. That is the job of N134.
- Modify the spectral analysis core. The math is correct; the issue is the operationalization layer.

## 10. Relation to Other Work

- **N134 experimental design:** If N134 produces a positive H1 result, the `STANDARD_ATTENTION` profile's `risk_prediction_validation` can be updated to `VALIDATED` (or `PARTIAL`) and the provenance warnings can be relaxed for the specific metric and threshold that passed validation. The honesty infrastructure built here makes that update trivial.
- **Paper:** This update forces the articulation of what the software can and cannot claim — language that feeds directly into the paper's methods and limitations sections.
- **DeBERTa/C_k investigation:** The `ck_predictive=False` change for `STANDARD_ATTENTION` aligns the software with the current empirical state. If the architecture-specificity investigation (§4 of the paths-forward analysis) reveals a mechanism, `ck_predictive` can be made conditional on module type rather than a blanket boolean.

---

*Last updated: April 2026. This spec reflects the state of the evidence after N133 Phase 4 and the B-P5 confound cascade.*
