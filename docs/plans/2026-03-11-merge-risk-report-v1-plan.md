# Merge Risk Report v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze the existing `MergeQAReport` as a stable, decision-bearing pair-level product object with strict validation, public API promotion, and canonical examples.

**Architecture:** Stabilize the existing `MergeQAReport` in `gradience/vnext/merge/qa_report.py` by renaming fields for clarity (`eligibility` -> `eligibility_status`, split `dominant_issue`), adding a categorical `confidence` field, making `recommended_strategy` operational, adding strict `from_dict()` validation, promoting to public API, and adding examples/docs.

**Tech Stack:** Python dataclasses, pytest, ruff, mypy

---

## Context

The design document is at `docs/plans/2026-03-11-merge-risk-report-v1-design.md`. Reference it for schema shape, validation rules, and policy decisions.

Key files:
- `gradience/vnext/merge/qa_report.py` — `MergeQAReport`, `AdapterSummary`, `build_qa_report`, `format_qa_report`
- `gradience/vnext/merge/recommend.py` — `PairDiagnosis`, `MergeRecommendation`, `diagnose_pair`, `recommend_merge`
- `gradience/vnext/merge/eligibility.py` — `EligibilityStatus`, `screen_adapters`
- `gradience/cli.py` — `cmd_merge_audit`, `--emit-report`, `--qa-report`, `--strict-qa`
- `gradience/exceptions.py` — `QASchemaError`
- `gradience/api.py` — stable Python API wrappers
- `gradience/__init__.py` — public exports
- `tests/merge/test_qa_report.py` — existing tests

Phase A precedent: `gradience/vnext/audit/qa_artifact.py` has the validation pattern to follow (`from_dict()` with `_require_field`, `_require_list_of_str`, `_to_float`, `QASchemaError`).

---

### Task 1: Rename `eligibility` to `eligibility_status` in AdapterSummary

**Files:**
- Modify: `gradience/vnext/merge/qa_report.py`
- Modify: `tests/merge/test_qa_report.py`

**Step 1: Update `AdapterSummary` dataclass field name**

In `gradience/vnext/merge/qa_report.py`, rename the field in the dataclass:

```python
@dataclass(frozen=True)
class AdapterSummary:
    """Structural summary of one adapter."""

    path: str
    rank: int
    alpha: float
    n_layers: int
    base_model: str
    eligibility_status: str | None  # EligibilityStatus value or None when no QA provided
```

**Step 2: Update `AdapterSummary.to_dict()` key**

Change the key from `"eligibility"` to `"eligibility_status"`:

```python
    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rank": self.rank,
            "alpha": float(self.alpha),
            "n_layers": self.n_layers,
            "base_model": self.base_model,
            "eligibility_status": self.eligibility_status,
        }
```

**Step 3: Update `_eligibility_label()` to return `None` instead of `"not provided"`**

```python
def _eligibility_label(diag: PairDiagnosis, which: str) -> str | None:
    """Get eligibility status for adapter A or B, or None if no QA provided."""
    status = diag.eligibility.status_a if which == "a" else diag.eligibility.status_b
    if status is None:
        return None
    return status.value
```

**Step 4: Update all references in `build_qa_report()` and `format_qa_report()`**

In `build_qa_report()`, the `AdapterSummary` constructor calls use `eligibility=...` — change to `eligibility_status=...`.

In `format_qa_report()`, update references from `qa.adapter_a.eligibility` to `qa.adapter_a.eligibility_status`. For terminal display, show `None` as `"not provided"`:

```python
    a_status = qa.adapter_a.eligibility_status or "not provided"
    b_status = qa.adapter_b.eligibility_status or "not provided"
    lines.append(f"  Adapter A eligibility: {a_status}")
    lines.append(f"  Adapter B eligibility: {b_status}")
```

**Step 5: Update existing tests**

In `tests/merge/test_qa_report.py`, update all references from `.eligibility` to `.eligibility_status` and from `d["eligibility"]` to `d["eligibility_status"]`. The `test_eligibility_not_provided` test should now assert `None` instead of `"not provided"`:

```python
    def test_eligibility_not_provided(self):
        report = _FakeReport([_make_lv_dict("safe")])
        qa = build_qa_report(report)
        assert qa.adapter_a.eligibility_status is None
        assert qa.adapter_b.eligibility_status is None
```

**Step 6: Run tests**

Run: `python3 -m pytest tests/merge/test_qa_report.py -v`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add gradience/vnext/merge/qa_report.py tests/merge/test_qa_report.py
git commit -m "Rename eligibility to eligibility_status in MergeQAReport

Use None instead of 'not provided' for missing QA data.
Keeps canonical EligibilityStatus vocabulary or null."
```

---

### Task 2: Split `dominant_issue` into structured label + detail

**Files:**
- Modify: `gradience/vnext/merge/qa_report.py`
- Modify: `tests/merge/test_qa_report.py`

**Step 1: Add `DOMINANT_ISSUE_LABELS` constant and update `MergeQAReport`**

Add a frozen set of valid labels at the top of `qa_report.py`:

```python
DOMINANT_ISSUE_LABELS = frozenset({
    "norm_imbalance",
    "subspace_conflict",
    "high_redundancy",
    "partial_redundancy",
    "none",
    "unknown",
})
```

Add `dominant_issue_detail` field to `MergeQAReport`:

```python
@dataclass(frozen=True)
class MergeQAReport:
    adapter_a: AdapterSummary
    adapter_b: AdapterSummary
    pair_risk: str
    dominant_issue: str        # machine-readable label from DOMINANT_ISSUE_LABELS
    dominant_issue_detail: str  # human-readable explanation
    recommended_action: str
    recommended_strategy: str
    confidence: str            # "high" | "medium" | "low" (new)
    confidence_note: str
    caveats: tuple[str, ...]
    verdict_distribution: dict[str, int]
    compatibility_score: float
```

**Step 2: Update `to_dict()` to emit both fields**

```python
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
```

**Step 3: Refactor `_dominant_issue()` to return `(label, detail)` tuple**

```python
def _dominant_issue(diag: PairDiagnosis, agg: Any) -> tuple[str, str]:
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
```

**Step 4: Add `_derive_confidence()` function**

```python
def _derive_confidence(diag: PairDiagnosis, score: float) -> str:
    """Derive categorical confidence level."""
    if not diag.eligibility.has_data:
        return "low"
    if diag.overall_risk == "high":
        return "low"
    if diag.eligibility.both_eligible and score >= 0.8 and diag.overall_risk == "low":
        return "high"
    return "medium"
```

**Step 5: Update `build_qa_report()` to use new signatures**

```python
    issue_label, issue_detail = _dominant_issue(diag, agg)
    confidence = _derive_confidence(diag, score)

    return MergeQAReport(
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        pair_risk=diag.overall_risk,
        dominant_issue=issue_label,
        dominant_issue_detail=issue_detail,
        recommended_action=_recommended_action(diag, rec, agg),
        recommended_strategy=...,  # handled in Task 3
        confidence=confidence,
        confidence_note=_confidence_note(diag, score),
        caveats=_caveats(diag, rec),
        verdict_distribution=verdict_dist,
        compatibility_score=score,
    )
```

**Step 6: Update `format_qa_report()` for the split**

Replace the old single dominant_issue line:
```python
    lines.append(f"  Dominant issue:  {qa.dominant_issue}")
```

With:
```python
    lines.append(f"  Dominant issue:  {qa.dominant_issue.upper().replace('_', ' ')}")
    if qa.dominant_issue_detail:
        lines.append(f"                   {qa.dominant_issue_detail}")
```

Add categorical confidence display:
```python
    lines.append(f"  Confidence:      {qa.confidence}")
    lines.append(f"                   {qa.confidence_note}")
```

**Step 7: Update existing tests**

Tests currently assert things like `"imbalance" in qa.dominant_issue.lower()`. Update to check the structured label:

```python
    def test_all_safe_layers(self):
        ...
        assert qa.dominant_issue == "none"
        assert qa.dominant_issue_detail  # non-empty

    def test_conflicting_layers_high_risk(self):
        ...
        assert qa.dominant_issue == "subspace_conflict"

    def test_imbalanced_layers_norm_issue(self):
        ...
        assert qa.dominant_issue == "norm_imbalance"

    def test_redundant_layers(self):
        ...
        assert qa.dominant_issue in ("high_redundancy", "partial_redundancy")
```

Also update the `test_to_dict_has_schema` test to check for `"dominant_issue_detail"` and `"confidence"`.

**Step 8: Run tests**

Run: `python3 -m pytest tests/merge/test_qa_report.py -v`
Expected: All tests pass.

**Step 9: Commit**

```bash
git add gradience/vnext/merge/qa_report.py tests/merge/test_qa_report.py
git commit -m "Split dominant_issue into structured label + detail

Add DOMINANT_ISSUE_LABELS frozen set. Add categorical confidence
field derived from eligibility and structural risk."
```

---

### Task 3: Make `recommended_strategy` operational

**Files:**
- Modify: `gradience/vnext/merge/qa_report.py`
- Modify: `tests/merge/test_qa_report.py`

**Step 1: Add `_derive_strategy()` function**

```python
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
```

**Step 2: Update `build_qa_report()` to call it**

Replace `recommended_strategy=rec.overall_strategy` with:

```python
    recommended_strategy=_derive_strategy(diag, rec),
```

**Step 3: Add tests for strategy derivation**

```python
class TestStrategyDerivation:
    def test_low_risk_no_compression_is_linear(self):
        """Low-risk, no compression → linear."""
        report = _FakeReport([
            _make_lv_dict("safe", layer_name="model.layers.0.self_attn.q_proj"),
            _make_lv_dict("safe", layer_name="model.layers.0.self_attn.v_proj"),
        ])
        qa = build_qa_report(report)
        assert qa.recommended_strategy == "linear"

    def test_high_risk_is_audit_aware(self):
        """High-risk → audit_aware."""
        report = _FakeReport([
            _make_lv_dict(
                "conflicting",
                mean_overlap=0.6,
                directional_agreement=-0.5,
                conflict_dimensions=3,
                layer_name="model.layers.0.self_attn.q_proj",
            ),
        ])
        qa = build_qa_report(report)
        assert qa.recommended_strategy == "audit_aware"
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/merge/test_qa_report.py -v`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add gradience/vnext/merge/qa_report.py tests/merge/test_qa_report.py
git commit -m "Make recommended_strategy operational in MergeQAReport

Derive from pair_risk + compression_needed instead of hardcoding
audit_aware. low->linear, medium->norm_equalized, high->audit_aware."
```

---

### Task 4: Update `from_dict()` with strict validation

**Files:**
- Modify: `gradience/vnext/merge/qa_report.py`
- Modify: `tests/merge/test_qa_report.py`

**Step 1: Rewrite `MergeQAReport.from_dict()` with validation**

Follow the Phase A pattern from `gradience/vnext/audit/qa_artifact.py`. Import `QASchemaError` and use validation helpers:

```python
from gradience.exceptions import QASchemaError
from gradience.vnext.merge.eligibility import EligibilityStatus

SCHEMA_ID = "gradience.merge_qa_report/v1"

PAIR_RISK_VALUES = frozenset({"low", "medium", "high"})
CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})


def _require_field(section: dict, field_name: str, section_name: str) -> Any:
    if field_name not in section:
        raise QASchemaError(f"Missing required field: {section_name}.{field_name}")
    return section[field_name]


def _validate_adapter_summary(d: dict, label: str) -> AdapterSummary:
    """Validate and construct an AdapterSummary from a dict."""
    path = str(_require_field(d, "path", label))
    rank_raw = _require_field(d, "rank", label)
    if not isinstance(rank_raw, (int, float)):
        raise QASchemaError(f"Field '{label}.rank' must be numeric")
    rank = int(rank_raw)

    alpha_raw = d.get("alpha", 0.0)
    if not isinstance(alpha_raw, (int, float)):
        raise QASchemaError(f"Field '{label}.alpha' must be numeric")
    alpha = float(alpha_raw)

    n_layers = int(d.get("n_layers", 0))
    base_model = str(d.get("base_model", ""))

    eligibility_raw = d.get("eligibility_status")
    if eligibility_raw is not None:
        try:
            EligibilityStatus(eligibility_raw)
        except ValueError:
            raise QASchemaError(
                f"Unknown eligibility status in {label}: '{eligibility_raw}'. "
                f"Valid values: {[e.value for e in EligibilityStatus]}"
            ) from None
    eligibility_status = eligibility_raw  # str | None

    return AdapterSummary(
        path=path,
        rank=rank,
        alpha=alpha,
        n_layers=n_layers,
        base_model=base_model,
        eligibility_status=eligibility_status,
    )
```

Then the `from_dict` classmethod:

```python
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MergeQAReport:
        """Deserialize from a v1 schema dict.

        Single canonical gatekeeper for the v1 schema. Validates required
        sections, required fields with type enforcement, and controlled
        vocabularies. Raises ``QASchemaError`` for contract violations.
        Extra keys are silently ignored for forward compatibility.
        """
        # Schema identity
        if "schema" not in d:
            raise QASchemaError("Missing required field: schema")
        if d["schema"] != SCHEMA_ID:
            raise QASchemaError(f"Expected schema '{SCHEMA_ID}', got '{d['schema']}'")

        # Required sections
        for section_name in ("adapter_a", "adapter_b"):
            if section_name not in d:
                raise QASchemaError(f"Missing required section: {section_name}")
            if not isinstance(d[section_name], dict):
                raise QASchemaError(f"Section '{section_name}' must be a dict")

        adapter_a = _validate_adapter_summary(d["adapter_a"], "adapter_a")
        adapter_b = _validate_adapter_summary(d["adapter_b"], "adapter_b")

        # pair_risk
        pair_risk = str(_require_field(d, "pair_risk", "root"))
        if pair_risk not in PAIR_RISK_VALUES:
            raise QASchemaError(
                f"Invalid pair_risk '{pair_risk}'. Must be one of: {sorted(PAIR_RISK_VALUES)}"
            )

        # dominant_issue
        dominant_issue = str(_require_field(d, "dominant_issue", "root"))
        if dominant_issue not in DOMINANT_ISSUE_LABELS:
            raise QASchemaError(
                f"Unknown dominant_issue '{dominant_issue}'. "
                f"Valid values: {sorted(DOMINANT_ISSUE_LABELS)}"
            )

        dominant_issue_detail = str(d.get("dominant_issue_detail", ""))

        # recommended fields
        recommended_action = str(d.get("recommended_action", ""))
        recommended_strategy = str(_require_field(d, "recommended_strategy", "root"))

        # confidence
        confidence = str(_require_field(d, "confidence", "root"))
        if confidence not in CONFIDENCE_VALUES:
            raise QASchemaError(
                f"Invalid confidence '{confidence}'. Must be one of: {sorted(CONFIDENCE_VALUES)}"
            )
        confidence_note = str(d.get("confidence_note", ""))

        # caveats
        raw_caveats = d.get("caveats", [])
        if not isinstance(raw_caveats, list) or not all(isinstance(c, str) for c in raw_caveats):
            raise QASchemaError("Field 'caveats' must be a list of strings")
        caveats = tuple(raw_caveats)

        # verdict_distribution
        raw_vd = d.get("verdict_distribution", {})
        if not isinstance(raw_vd, dict):
            raise QASchemaError("Field 'verdict_distribution' must be a dict")
        for k, v in raw_vd.items():
            if not isinstance(v, int):
                raise QASchemaError(f"verdict_distribution['{k}'] must be an integer")

        # compatibility_score
        score_raw = _require_field(d, "compatibility_score", "root")
        if not isinstance(score_raw, (int, float)):
            raise QASchemaError("Field 'compatibility_score' must be numeric")
        compatibility_score = float(score_raw)

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
            verdict_distribution=raw_vd,
            compatibility_score=compatibility_score,
        )
```

**Step 2: Add validation tests**

```python
class TestFromDictValidation:
    """Strict from_dict validation (parallel to Phase A test_qa_artifact.py)."""

    def _valid_dict(self) -> dict:
        """Minimal valid v1 dict."""
        return {
            "schema": "gradience.merge_qa_report/v1",
            "adapter_a": {
                "path": "/tmp/a", "rank": 8, "alpha": 16.0,
                "n_layers": 32, "base_model": "llama",
                "eligibility_status": "eligible",
            },
            "adapter_b": {
                "path": "/tmp/b", "rank": 8, "alpha": 16.0,
                "n_layers": 32, "base_model": "llama",
                "eligibility_status": None,
            },
            "pair_risk": "low",
            "dominant_issue": "none",
            "dominant_issue_detail": "adapters are spectrally compatible",
            "recommended_action": "Merge is safe.",
            "recommended_strategy": "linear",
            "confidence": "high",
            "confidence_note": "High spectral compatibility.",
            "caveats": [],
            "verdict_distribution": {"safe": 32, "redundant": 0, "conflicting": 0, "imbalanced": 0},
            "compatibility_score": 0.95,
        }

    def test_valid_roundtrip(self):
        d = self._valid_dict()
        report = MergeQAReport.from_dict(d)
        assert report.pair_risk == "low"
        assert report.adapter_a.eligibility_status == "eligible"
        assert report.adapter_b.eligibility_status is None

    def test_missing_schema_raises(self):
        d = self._valid_dict()
        del d["schema"]
        with pytest.raises(QASchemaError, match="schema"):
            MergeQAReport.from_dict(d)

    def test_wrong_schema_raises(self):
        d = self._valid_dict()
        d["schema"] = "wrong/v1"
        with pytest.raises(QASchemaError, match="Expected schema"):
            MergeQAReport.from_dict(d)

    def test_missing_adapter_section_raises(self):
        d = self._valid_dict()
        del d["adapter_a"]
        with pytest.raises(QASchemaError, match="adapter_a"):
            MergeQAReport.from_dict(d)

    def test_bad_pair_risk_raises(self):
        d = self._valid_dict()
        d["pair_risk"] = "extreme"
        with pytest.raises(QASchemaError, match="pair_risk"):
            MergeQAReport.from_dict(d)

    def test_unknown_dominant_issue_raises(self):
        d = self._valid_dict()
        d["dominant_issue"] = "cosmic_radiation"
        with pytest.raises(QASchemaError, match="dominant_issue"):
            MergeQAReport.from_dict(d)

    def test_unknown_eligibility_status_raises(self):
        d = self._valid_dict()
        d["adapter_a"]["eligibility_status"] = "super_good"
        with pytest.raises(QASchemaError, match="eligibility status"):
            MergeQAReport.from_dict(d)

    def test_bad_confidence_raises(self):
        d = self._valid_dict()
        d["confidence"] = "very_high"
        with pytest.raises(QASchemaError, match="confidence"):
            MergeQAReport.from_dict(d)

    def test_caveats_must_be_list_of_str(self):
        d = self._valid_dict()
        d["caveats"] = [1, 2, 3]
        with pytest.raises(QASchemaError, match="caveats"):
            MergeQAReport.from_dict(d)

    def test_verdict_distribution_values_must_be_int(self):
        d = self._valid_dict()
        d["verdict_distribution"] = {"safe": "ten"}
        with pytest.raises(QASchemaError, match="verdict_distribution"):
            MergeQAReport.from_dict(d)

    def test_extra_keys_ignored(self):
        d = self._valid_dict()
        d["future_field"] = "should not fail"
        report = MergeQAReport.from_dict(d)
        assert report.pair_risk == "low"

    def test_null_eligibility_accepted(self):
        d = self._valid_dict()
        d["adapter_a"]["eligibility_status"] = None
        report = MergeQAReport.from_dict(d)
        assert report.adapter_a.eligibility_status is None

    def test_missing_optional_fields_backfilled(self):
        d = self._valid_dict()
        del d["dominant_issue_detail"]
        del d["confidence_note"]
        del d["caveats"]
        del d["verdict_distribution"]
        del d["recommended_action"]
        report = MergeQAReport.from_dict(d)
        assert report.dominant_issue_detail == ""
        assert report.confidence_note == ""
        assert report.caveats == ()
        assert report.verdict_distribution == {}
        assert report.recommended_action == ""

    def test_numeric_rank_normalized_to_int(self):
        d = self._valid_dict()
        d["adapter_a"]["rank"] = 16.0
        report = MergeQAReport.from_dict(d)
        assert report.adapter_a.rank == 16
        assert isinstance(report.adapter_a.rank, int)

    def test_unknown_strategy_accepted(self):
        """recommended_strategy is lenient — accepts unknown values."""
        d = self._valid_dict()
        d["recommended_strategy"] = "dare_ties"
        report = MergeQAReport.from_dict(d)
        assert report.recommended_strategy == "dare_ties"
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/merge/test_qa_report.py -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add gradience/vnext/merge/qa_report.py tests/merge/test_qa_report.py
git commit -m "Add strict from_dict validation to MergeQAReport

Single canonical gatekeeper for merge_qa_report/v1 schema.
Validates schema identity, required sections, type enforcement,
controlled vocabularies. Raises QASchemaError on violations."
```

---

### Task 5: Update CLI `--emit-report` and `--strict-qa` for null eligibility

**Files:**
- Modify: `gradience/cli.py`
- Modify: `tests/test_cli_commands.py` (if merge-audit CLI tests exist there)

**Step 1: Update `--emit-report` to write `MergeQAReport` v1 JSON**

In `cmd_merge_audit()`, replace the current `--emit-report` block (around line 2382-2390):

```python
    # --- Emit structured report ---
    emit_path = getattr(args, "emit_report", None)
    if emit_path:
        from gradience.vnext.merge.qa_report import build_qa_report as _build_qa

        emit_p = Path(emit_path)
        emit_p.parent.mkdir(parents=True, exist_ok=True)
        qa_for_emit = _build_qa(report)
        with open(emit_p, "w") as f:
            jsonlib.dump(qa_for_emit.to_dict(), f, indent=2)
        print(f"\nMerge QA report written to: {emit_p}")
```

Note: the `build_qa_report` import may already exist if `--qa-report` was used above. Use a local alias `_build_qa` to avoid collisions, or reuse the already-built `qa` object if `--qa-report` was also set. Check for that optimization:

```python
    # --- Emit structured report ---
    emit_path = getattr(args, "emit_report", None)
    if emit_path:
        if not qa_for_emit:
            from gradience.vnext.merge.qa_report import build_qa_report as _build_qa
            qa_for_emit = _build_qa(report)
        emit_p = Path(emit_path)
        emit_p.parent.mkdir(parents=True, exist_ok=True)
        with open(emit_p, "w") as f:
            jsonlib.dump(qa_for_emit.to_dict(), f, indent=2)
        print(f"\nMerge QA report written to: {emit_p}")
```

Initialize `qa_for_emit = None` before the QA report section, and set `qa_for_emit = qa` if `--qa-report` was used.

**Step 2: Update `--strict-qa` to handle null eligibility (no QA provided)**

The current strict-qa gate (around line 2266-2298) already blocks when `not diag.eligibility.has_data`. This covers the case where no `--source-a-qa` / `--source-b-qa` is provided (both null). No change needed for that case.

However, the current gate doesn't handle the case where one adapter has QA and the other doesn't. Check if `EligibilityContext` handles partial QA — if `status_a` is set but `status_b` is `None`, the `has_data` property returns `True` (since one has data), but the null adapter isn't blocked.

If needed, add a check for individual null statuses in strict mode:

```python
    if strict_qa:
        diag = diagnose_pair(report)
        # Block if no QA data at all
        if not diag.eligibility.has_data:
            print("\nError: --strict-qa requires source QA data for both adapters.")
            sys.exit(1)
        # Block if either adapter has no QA (null eligibility)
        if diag.eligibility.status_a is None or diag.eligibility.status_b is None:
            missing = []
            if diag.eligibility.status_a is None:
                missing.append("A")
            if diag.eligibility.status_b is None:
                missing.append("B")
            print(f"\nError: --strict-qa requires QA data for adapter(s) {', '.join(missing)}.")
            print("  Provide --source-a-qa and --source-b-qa, or remove --strict-qa.")
            sys.exit(1)
        # ... existing weak/unverified checks ...
```

**Step 3: Run existing CLI tests**

Run: `python3 -m pytest tests/test_cli_commands.py -v -k merge`
Expected: Existing tests pass (may need minor updates if they check emit-report output format).

**Step 4: Commit**

```bash
git add gradience/cli.py
git commit -m "Update --emit-report to write MergeQAReport v1 JSON

Repurpose --emit-report from raw merge_audit/v2 to the frozen
merge_qa_report/v1 schema. Update --strict-qa to block when
either adapter has null eligibility (no QA artifact provided)."
```

---

### Task 6: Public API promotion

**Files:**
- Modify: `gradience/__init__.py`
- Modify: `gradience/api.py`
- Modify: `tests/test_qa_artifact.py` (add public API test)

**Step 1: Export `MergeQAReport` from `gradience/__init__.py`**

Add the import and `__all__` entry:

```python
from gradience.vnext.merge.qa_report import MergeQAReport

__all__ = [
    # ... existing entries ...
    # Merge QA report
    "MergeQAReport",
]
```

**Step 2: Add `merge_risk_report()` to `gradience/api.py`**

Add a new dataclass for the result and the function:

```python
@dataclass(frozen=True)
class MergeRiskReportArtifacts:
    """Paths produced by `merge_risk_report`."""

    report_json: Path


def merge_risk_report(
    *,
    adapter_a: str | Path,
    adapter_b: str | Path,
    source_a_qa: str | Path | None = None,
    source_b_qa: str | Path | None = None,
    thresholds: str = "default",
    python: str | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> Any:
    """Run merge-audit and return the pair-level MergeQAReport.

    This is the stable Python wrapper for the ``merge-audit --qa-report
    --emit-report`` workflow. It delegates report generation to the CLI,
    then loads the resulting JSON as a ``MergeQAReport``.

    Parameters
    ----------
    adapter_a, adapter_b
        Paths to the two PEFT adapter directories.
    source_a_qa, source_b_qa
        Optional paths to adapter QA artifact JSON files.
    thresholds
        Threshold preset: ``"default"``, ``"conservative"``, or ``"permissive"``.

    Returns
    -------
    MergeQAReport
        The canonical pair-level merge risk report.
    """
    import tempfile

    adapter_a_p = Path(adapter_a)
    adapter_b_p = Path(adapter_b)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        emit_path = Path(tmp.name)

    try:
        argv = [
            _pyexe(python),
            "-m", "gradience",
            "merge-audit",
            "--adapter-a", str(adapter_a_p),
            "--adapter-b", str(adapter_b_p),
            "--qa-report",
            "--emit-report", str(emit_path),
            "--thresholds", thresholds,
        ]
        if source_a_qa:
            argv.extend(["--source-a-qa", str(Path(source_a_qa))])
        if source_b_qa:
            argv.extend(["--source-b-qa", str(Path(source_b_qa))])

        _run(
            argv,
            env=env,
            check=check,
            log_path=Path(log_path) if log_path else None,
        )

        report_data = _read_json(emit_path)
        from gradience.vnext.merge.qa_report import MergeQAReport as _MergeQAReport
        return _MergeQAReport.from_dict(report_data)
    finally:
        emit_path.unlink(missing_ok=True)
```

**Step 3: Add public API test**

In `tests/test_qa_artifact.py` (or a new `tests/test_public_api.py`), add:

```python
def test_merge_qa_report_importable_from_gradience():
    from gradience import MergeQAReport
    assert hasattr(MergeQAReport, "from_dict")
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_qa_artifact.py -v -k public`
Expected: Pass.

**Step 5: Commit**

```bash
git add gradience/__init__.py gradience/api.py tests/test_qa_artifact.py
git commit -m "Promote MergeQAReport to public API

Export from gradience.__init__. Add merge_risk_report() to api.py
as stable Python wrapper (CLI-delegating, not alternate logic path)."
```

---

### Task 7: Create canonical example files

**Files:**
- Create: `examples/reports/safe_merge_report.json`
- Create: `examples/reports/high_risk_warn_report.json`
- Create: `examples/reports/strict_blocked_report.json`

**Step 1: Create `examples/reports/` directory**

```bash
mkdir -p examples/reports
```

**Step 2: Create `safe_merge_report.json`**

Both eligible, low risk, no issues.

```json
{
  "schema": "gradience.merge_qa_report/v1",
  "adapter_a": {
    "path": "./adapters/good-adapter-a",
    "rank": 16,
    "alpha": 16.0,
    "n_layers": 32,
    "base_model": "meta-llama/Llama-2-7b-hf",
    "eligibility_status": "eligible"
  },
  "adapter_b": {
    "path": "./adapters/good-adapter-b",
    "rank": 16,
    "alpha": 16.0,
    "n_layers": 32,
    "base_model": "meta-llama/Llama-2-7b-hf",
    "eligibility_status": "eligible"
  },
  "pair_risk": "low",
  "dominant_issue": "none",
  "dominant_issue_detail": "adapters are spectrally compatible",
  "recommended_action": "Merge is safe. Use audit-aware strategy or norm-equalized baseline.",
  "recommended_strategy": "linear",
  "confidence": "high",
  "confidence_note": "High spectral compatibility (score=0.920) — both adapters have verified behavioral quality",
  "caveats": [],
  "verdict_distribution": {"safe": 30, "redundant": 2, "conflicting": 0, "imbalanced": 0},
  "compatibility_score": 0.920
}
```

**Step 3: Create `high_risk_warn_report.json`**

One flagged_weak, high risk, norm imbalance.

```json
{
  "schema": "gradience.merge_qa_report/v1",
  "adapter_a": {
    "path": "./adapters/catsubcat-r16",
    "rank": 16,
    "alpha": 16.0,
    "n_layers": 32,
    "base_model": "meta-llama/Llama-2-7b-hf",
    "eligibility_status": "flagged_weak"
  },
  "adapter_b": {
    "path": "./adapters/btgenbot-r8",
    "rank": 8,
    "alpha": 8.0,
    "n_layers": 32,
    "base_model": "meta-llama/Llama-2-7b-hf",
    "eligibility_status": "eligible"
  },
  "pair_risk": "high",
  "dominant_issue": "norm_imbalance",
  "dominant_issue_detail": "11.3x mean magnitude ratio across 15 layer(s)",
  "recommended_action": "Merge with caution using audit-aware strategy. Validate merged adapter on downstream task.",
  "recommended_strategy": "audit_aware",
  "confidence": "low",
  "confidence_note": "Low spectral compatibility (score=0.340) — at least one adapter lacks behavioral evidence of quality",
  "caveats": [
    "Adapter A underperforms the base model. Rebalancing may preserve a weak signal.",
    "High structural risk. Always validate the merged adapter on your target task before deployment."
  ],
  "verdict_distribution": {"safe": 10, "redundant": 2, "conflicting": 5, "imbalanced": 15},
  "compatibility_score": 0.340
}
```

**Step 4: Create `strict_blocked_report.json`**

One null eligibility (no QA provided), high risk.

```json
{
  "schema": "gradience.merge_qa_report/v1",
  "adapter_a": {
    "path": "./adapters/catsubcat-r16",
    "rank": 16,
    "alpha": 16.0,
    "n_layers": 32,
    "base_model": "meta-llama/Llama-2-7b-hf",
    "eligibility_status": "eligible"
  },
  "adapter_b": {
    "path": "./adapters/unknown-adapter",
    "rank": 8,
    "alpha": 8.0,
    "n_layers": 32,
    "base_model": "meta-llama/Llama-2-7b-hf",
    "eligibility_status": null
  },
  "pair_risk": "high",
  "dominant_issue": "norm_imbalance",
  "dominant_issue_detail": "8.5x mean magnitude ratio across 12 layer(s)",
  "recommended_action": "Merge with caution using audit-aware strategy. Validate on downstream task.",
  "recommended_strategy": "audit_aware",
  "confidence": "low",
  "confidence_note": "Low spectral compatibility (score=0.410) — no behavioral evaluation data available",
  "caveats": [
    "No QA artifact provided for adapter B. Recommendation is structural only.",
    "High structural risk. Always validate the merged adapter on your target task before deployment."
  ],
  "verdict_distribution": {"safe": 8, "redundant": 4, "conflicting": 8, "imbalanced": 12},
  "compatibility_score": 0.410
}
```

**Step 5: Add smoke test for example files**

In `tests/merge/test_qa_report.py`:

```python
import glob

class TestExampleFiles:
    @pytest.mark.parametrize("path", sorted(glob.glob("examples/reports/*.json")))
    def test_example_loads_via_from_dict(self, path):
        with open(path) as f:
            d = json.load(f)
        report = MergeQAReport.from_dict(d)
        assert report.pair_risk in ("low", "medium", "high")
        assert report.dominant_issue in DOMINANT_ISSUE_LABELS
```

Import `DOMINANT_ISSUE_LABELS` at the top of the test file.

**Step 6: Run tests**

Run: `python3 -m pytest tests/merge/test_qa_report.py -v`
Expected: All tests pass including example file smoke tests.

**Step 7: Commit**

```bash
git add examples/reports/ tests/merge/test_qa_report.py
git commit -m "Add canonical merge QA report examples

Three files: safe_merge_report.json, high_risk_warn_report.json,
strict_blocked_report.json. Add smoke test for example files."
```

---

### Task 8: Write definition doc

**Files:**
- Create: `docs/merge-risk-report.md`

Write the definition doc following the same structure as `docs/adapter-qa-artifact.md`. Headings:

1. What it is (one sentence)
2. How to produce it (CLI `--emit-report` and Python `merge_risk_report()`)
3. How to read it (section walkthrough: adapter summaries, pair_risk, dominant_issue, strategy, confidence, caveats)
4. How to consume it (scripting, `--strict-qa`, exit codes)
5. Schema contract (field table with types, required/optional)
6. Decision semantics (strategy derivation table, confidence rules, strict-qa policy table, `compatibility_score` definition: 0-1 range, higher=more compatible)
7. Versioning policy (additive only)

Keep it short. Definition document, not tutorial.

**Step 1: Write the doc**

**Step 2: Commit**

```bash
git add docs/merge-risk-report.md
git commit -m "Add merge risk report definition document"
```

---

### Task 9: Final validation and CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Run full test suite**

```bash
python3 -m pytest tests/ -x -q
```

Expected: All tests pass (1042+).

**Step 2: Run lint and format**

```bash
ruff check gradience/ tests/merge/test_qa_report.py
ruff format --check gradience/ tests/merge/test_qa_report.py
```

Fix any issues.

**Step 3: Run mypy**

```bash
mypy gradience/
```

Expected: No errors.

**Step 4: Verify example files smoke test**

```bash
python3 -c "
import json, glob
from gradience.vnext.merge.qa_report import MergeQAReport
for f in sorted(glob.glob('examples/reports/*.json')):
    with open(f) as fp:
        d = json.load(fp)
    r = MergeQAReport.from_dict(d)
    print(f'{f}: {r.pair_risk} / {r.dominant_issue} / {r.recommended_strategy} ({r.confidence})')
"
```

**Step 5: Update CLAUDE.md**

Add a `### Merge QA Report` section after the Adapter QA Artifact section:

```markdown
### Merge QA Report (`vnext/merge/qa_report.py`)

- Schema: `gradience.merge_qa_report/v1` — frozen, additive-only versioning
- `MergeQAReport` is stable public API (exported from `gradience.__init__`)
- `gradience.api.merge_risk_report()` is the stable Python entry point (CLI-delegating wrapper)
- `from_dict()` is the single validation gatekeeper — raises `QASchemaError` on contract violations
- `eligibility_status` per adapter: canonical `EligibilityStatus` value or `null` (no QA provided)
- `dominant_issue`: machine-readable label from frozen set (`norm_imbalance`, `subspace_conflict`, `high_redundancy`, `partial_redundancy`, `none`, `unknown`)
- `recommended_strategy`: operational — `"linear"` (low risk), `"norm_equalized"` (medium), `"audit_aware"` (high/compression)
- `--emit-report` writes v1 JSON; `--qa-report` prints 4-section terminal format
- `--strict-qa` blocks `flagged_weak`, `unknown_no_behavioral_eval`, and `null` eligibility
- Canonical examples in `examples/reports/` — one per scenario (safe, high-risk warning, strict-blocked)
- Definition doc: `docs/merge-risk-report.md`
```

**Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md with merge QA report conventions"
```
