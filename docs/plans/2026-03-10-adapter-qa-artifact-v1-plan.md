# Adapter QA Artifact v1 Freeze — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze the single-adapter QA artifact as a stable, validated, decision-bearing product object with public API.

**Architecture:** Tighten the existing `AdapterQAArtifact` with strict validation, rename two fields and one enum value for clarity, add `QASchemaError`, promote to public API via `audit_adapter()` in `gradience/api.py`, update downstream `--strict-qa` policy, and update examples/docs.

**Tech Stack:** Python dataclasses, pytest, ruff, mypy

**Design doc:** `docs/plans/2026-03-10-adapter-qa-artifact-v1-design.md`

---

### Task 1: Add `QASchemaError` to exception hierarchy

**Files:**
- Modify: `gradience/exceptions.py`
- Test: `tests/test_qa_artifact.py` (validation tests in Task 3 will exercise it)

**Step 1: Add the exception class**

In `gradience/exceptions.py`, add after `MergeError`:

```python
class QASchemaError(GradienceError, ValueError):
    """Raised when a QA artifact fails schema validation (missing fields, wrong types, unknown schema)."""
```

**Step 2: Run lint to verify**

Run: `ruff check gradience/exceptions.py`
Expected: PASS

**Step 3: Commit**

```bash
git add gradience/exceptions.py
git commit -m "Add QASchemaError to exception hierarchy"
```

---

### Task 2: Rename enum value `UNKNOWN` -> `UNKNOWN_NO_BEHAVIORAL_EVAL`

**Files:**
- Modify: `gradience/vnext/merge/eligibility.py`
- Modify: `gradience/vnext/audit/qa_artifact.py`
- Modify: `gradience/cli.py` (strict-qa gate references `"flagged_weak"` by value string — check for `"unknown"` references)
- Test: `tests/test_qa_artifact.py`
- Test: `tests/merge/test_eligibility.py`

**Step 1: Write a failing test confirming the new enum value exists**

In `tests/test_qa_artifact.py`, add to `TestSchemaOutput`:

```python
def test_unknown_status_is_explicit(self):
    """The unknown status must use the explicit name, not bare 'unknown'."""
    art = _make_artifact(status=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL)
    d = art.to_dict()
    assert d["eligibility"]["status"] == "unknown_no_behavioral_eval"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_qa_artifact.py::TestSchemaOutput::test_unknown_status_is_explicit -v`
Expected: FAIL with `AttributeError: UNKNOWN_NO_BEHAVIORAL_EVAL`

**Step 3: Rename the enum value**

In `gradience/vnext/merge/eligibility.py`, change:

```python
# Old:
UNKNOWN = "unknown"

# New:
UNKNOWN_NO_BEHAVIORAL_EVAL = "unknown_no_behavioral_eval"
```

Update the docstring for the enum value to match.

**Step 4: Update all references to `EligibilityStatus.UNKNOWN`**

Search the entire codebase for `EligibilityStatus.UNKNOWN` and `EligibilityStatus.UNKNOWN`:

```bash
rg "EligibilityStatus\.UNKNOWN" gradience/ tests/
```

Update every reference to `EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL`. Key files:
- `gradience/vnext/merge/eligibility.py`: `classify_eligibility()` returns `UNKNOWN` when no metrics — change to `UNKNOWN_NO_BEHAVIORAL_EVAL`
- `gradience/vnext/audit/qa_artifact.py`: default status in `AdapterQAArtifact`, `from_dict()` fallback, `CONFIDENCE_LOW` derivation
- `gradience/vnext/merge/recommend.py`: `_parse_eligibility()` fallback
- `gradience/cli.py`: strict-qa gate logic, `_load_source_qa` fallback
- `tests/test_qa_artifact.py`: all `EligibilityStatus.UNKNOWN` references
- `tests/merge/test_eligibility.py`: all `EligibilityStatus.UNKNOWN` references

Also update `from_dict()` in `eligibility.py` — the fallback value changes from `EligibilityStatus.UNKNOWN.value` to `EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL.value`.

**Step 5: Run tests to verify**

Run: `pytest tests/test_qa_artifact.py tests/merge/test_eligibility.py -v`
Expected: ALL PASS

**Step 6: Run full test suite to catch any missed references**

Run: `pytest tests/ -x -q`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add -A
git commit -m "Rename EligibilityStatus.UNKNOWN to UNKNOWN_NO_BEHAVIORAL_EVAL

Precise about why eligibility is unknown. A /v1 producer that adds
new status values should not still call itself /v1."
```

---

### Task 3: Rename `energy_rank_90_p50` -> `effective_rank_90_median` in JSON output

This rename only affects the **JSON serialization** (`to_dict()` / `from_dict()`). The Python dataclass field name stays `energy_rank_90_p50` internally for now to minimize churn across the codebase. The JSON key is what the schema freezes.

**Files:**
- Modify: `gradience/vnext/audit/qa_artifact.py` (`to_dict()`, `from_dict()`)
- Modify: `gradience/cli.py` (`_print_qa_summary` display label)
- Modify: `examples/qa/*.json` (all four example files)
- Test: `tests/test_qa_artifact.py`

**Step 1: Write a failing test for the new JSON key name**

In `tests/test_qa_artifact.py`, add to `TestSchemaOutput`:

```python
def test_effective_rank_90_median_key(self):
    """JSON output uses 'effective_rank_90_median', not 'energy_rank_90_p50'."""
    d = _make_artifact().to_dict()
    assert "effective_rank_90_median" in d["structural_summary"]
    assert "energy_rank_90_p50" not in d["structural_summary"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_qa_artifact.py::TestSchemaOutput::test_effective_rank_90_median_key -v`
Expected: FAIL

**Step 3: Update `to_dict()` in `qa_artifact.py`**

In the `structural_summary` dict returned by `to_dict()`, change:

```python
# Old:
"energy_rank_90_p50": self.energy_rank_90_p50,

# New:
"effective_rank_90_median": self.energy_rank_90_p50,
```

**Step 4: Update `from_dict()` in `qa_artifact.py`**

Change the field read:

```python
# Old:
energy_rank_90_p50=structural.get("energy_rank_90_p50"),

# New — accept both for backward compat on load:
energy_rank_90_p50=structural.get("effective_rank_90_median", structural.get("energy_rank_90_p50")),
```

**Step 5: Update `_print_qa_summary` in `cli.py`**

Change the display label from `"Energy rank 90 p50"` to `"Effective rank 90 (median)"`.

**Step 6: Update all four example JSON files**

In each file under `examples/qa/`, rename the key from `"energy_rank_90_p50"` to `"effective_rank_90_median"`.

**Step 7: Run tests**

Run: `pytest tests/test_qa_artifact.py -v`
Expected: ALL PASS

**Step 8: Commit**

```bash
git add gradience/vnext/audit/qa_artifact.py gradience/cli.py examples/qa/ tests/test_qa_artifact.py
git commit -m "Rename energy_rank_90_p50 to effective_rank_90_median in QA JSON

More readable in the frozen schema. Internal Python field name
unchanged to minimize churn. from_dict() accepts both keys for
backward compatibility on load."
```

---

### Task 4: Make `metric_name` and `lower_is_better` nullable, add `notes` field

**Files:**
- Modify: `gradience/vnext/audit/qa_artifact.py`
- Modify: `gradience/cli.py` (`cmd_audit_adapter` passes `metric_name`)
- Test: `tests/test_qa_artifact.py`

**Step 1: Write failing tests**

In `tests/test_qa_artifact.py`, add to `TestSchemaOutput`:

```python
def test_metric_name_null_when_no_eval(self):
    """metric_name should be null (not empty string) when no eval."""
    art = _make_artifact(eval_available=False, metric_name=None)
    d = art.to_dict()
    assert d["behavioral_summary"]["metric_name"] is None

def test_lower_is_better_null_when_no_eval(self):
    """lower_is_better should be null when no eval."""
    art = _make_artifact(eval_available=False, lower_is_better=None)
    d = art.to_dict()
    assert d["behavioral_summary"]["lower_is_better"] is None

def test_notes_field_present(self):
    """notes must always be present in output."""
    d = _make_artifact().to_dict()
    assert "notes" in d
    assert isinstance(d["notes"], list)

def test_notes_roundtrip(self):
    art = _make_artifact()
    # Manually set notes via dict manipulation for roundtrip
    d = art.to_dict()
    d["notes"] = ["structural audit only"]
    restored = AdapterQAArtifact.from_dict(d)
    assert restored.notes == ["structural audit only"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qa_artifact.py::TestSchemaOutput::test_metric_name_null_when_no_eval tests/test_qa_artifact.py::TestSchemaOutput::test_lower_is_better_null_when_no_eval tests/test_qa_artifact.py::TestSchemaOutput::test_notes_field_present -v`
Expected: FAIL

**Step 3: Update `AdapterQAArtifact` dataclass**

In `gradience/vnext/audit/qa_artifact.py`:

1. Change `metric_name` field type from `str` to `str | None`, default `None`
2. Change `lower_is_better` field type from `bool` to `bool | None`, default `None`
3. Add `notes: list[str] = field(default_factory=list)` after the `reasons` field

**Step 4: Update `to_dict()`**

Add `notes` to the output:

```python
def to_dict(self) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "adapter": { ... },
        "structural_summary": { ... },
        "behavioral_summary": { ... },
        "eligibility": { ... },
        "notes": list(self.notes),
    }
```

**Step 5: Update `from_dict()`**

Add `notes` parsing:

```python
notes=list(d.get("notes", [])),
```

Update `metric_name` parsing to preserve `None`:

```python
# Old:
metric_name=str(behavioral.get("metric_name", "")),

# New:
metric_name=behavioral.get("metric_name"),
```

Update `lower_is_better` parsing to preserve `None`:

```python
# Old:
lower_is_better=bool(behavioral.get("lower_is_better", True)),

# New:
lower_is_better=behavioral.get("lower_is_better"),
```

**Step 6: Update `build_qa_artifact()`**

When `eval_available` is False, pass `metric_name=None` and `lower_is_better=None` to the constructor. When eval is available, pass the caller-provided values.

Add `notes` parameter to `build_qa_artifact()`:

```python
def build_qa_artifact(
    audit_result: Any,
    *,
    ...,
    notes: list[str] | None = None,
) -> AdapterQAArtifact:
```

Pass `notes=notes or []` to the constructor.

**Step 7: Update `_make_artifact` test helper defaults**

Change defaults to match new types:

```python
metric_name=None,  # was ""
lower_is_better=None,  # was True
```

**Step 8: Update `to_qa_result()` bridge**

The `AdapterQAResult` still expects `metric_name: str` and `lower_is_better: bool`. In `to_qa_result()`, coerce:

```python
metric_name=self.metric_name or "",
lower_is_better=self.lower_is_better if self.lower_is_better is not None else True,
```

**Step 9: Update CLI `cmd_audit_adapter`**

When no metric_name is provided by CLI, pass `None` instead of `""`:

```python
metric_name=getattr(args, "metric_name", None) or None,  # was: or ""
```

**Step 10: Run tests**

Run: `pytest tests/test_qa_artifact.py -v`
Expected: ALL PASS

**Step 11: Commit**

```bash
git add gradience/vnext/audit/qa_artifact.py gradience/cli.py tests/test_qa_artifact.py
git commit -m "Make metric_name/lower_is_better nullable, add notes field

Absent values use null not empty string. lower_is_better is null
when no eval exists. notes field added as top-level list[str] for
caveats that aren't reasons or flags."
```

---

### Task 5: Add strict validation to `from_dict()`

**Files:**
- Modify: `gradience/vnext/audit/qa_artifact.py`
- Test: `tests/test_qa_artifact.py`

**Step 1: Write failing tests for validation**

In `tests/test_qa_artifact.py`, add a new test class:

```python
from gradience.exceptions import QASchemaError

class TestFromDictValidation:
    def test_rejects_missing_schema(self):
        with pytest.raises(QASchemaError, match="Missing required field: schema"):
            AdapterQAArtifact.from_dict({"adapter": {}})

    def test_rejects_wrong_schema(self):
        d = _make_artifact().to_dict()
        d["schema"] = "gradience.adapter_qa/v2"
        with pytest.raises(QASchemaError, match="Expected schema"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_missing_section(self):
        d = _make_artifact().to_dict()
        del d["eligibility"]
        with pytest.raises(QASchemaError, match="Missing required section: eligibility"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_section_not_dict(self):
        d = _make_artifact().to_dict()
        d["adapter"] = "not a dict"
        with pytest.raises(QASchemaError, match="must be a dict"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_missing_required_field(self):
        d = _make_artifact().to_dict()
        del d["adapter"]["rank_nominal"]
        with pytest.raises(QASchemaError, match="rank_nominal"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_unknown_status(self):
        d = _make_artifact().to_dict()
        d["eligibility"]["status"] = "experimental_new_status"
        with pytest.raises(QASchemaError, match="Unknown eligibility status"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_bad_confidence(self):
        d = _make_artifact().to_dict()
        d["eligibility"]["confidence"] = "super_high"
        with pytest.raises(QASchemaError, match="confidence"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_flags_not_list_of_str(self):
        d = _make_artifact().to_dict()
        d["structural_summary"]["flags"] = [1, 2, 3]
        with pytest.raises(QASchemaError, match="flags"):
            AdapterQAArtifact.from_dict(d)

    def test_rejects_notes_not_list_of_str(self):
        d = _make_artifact().to_dict()
        d["notes"] = "not a list"
        with pytest.raises(QASchemaError, match="notes"):
            AdapterQAArtifact.from_dict(d)

    def test_accepts_numeric_as_float(self):
        """Integer values for float fields should be accepted and normalized."""
        d = _make_artifact().to_dict()
        d["structural_summary"]["utilization_mean"] = 0  # int, not float
        d["structural_summary"]["rank_waste_ratio"] = 1  # int, not float
        art = AdapterQAArtifact.from_dict(d)
        assert isinstance(art.utilization_mean, float)
        assert isinstance(art.rank_waste_ratio, float)

    def test_missing_notes_backfilled(self):
        d = _make_artifact().to_dict()
        del d["notes"]
        art = AdapterQAArtifact.from_dict(d)
        assert art.notes == []

    def test_missing_reasons_backfilled(self):
        d = _make_artifact().to_dict()
        del d["eligibility"]["reasons"]
        art = AdapterQAArtifact.from_dict(d)
        assert art.reasons == []

    def test_extra_keys_ignored(self):
        d = _make_artifact().to_dict()
        d["provenance"] = {"generated_by": "test"}
        d["adapter"]["extra_field"] = "ignored"
        art = AdapterQAArtifact.from_dict(d)
        assert art.adapter_name == "test-adapter"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qa_artifact.py::TestFromDictValidation -v`
Expected: FAIL (no validation logic yet)

**Step 3: Implement validation in `from_dict()`**

Replace the current `from_dict()` with a validated version. Import `QASchemaError` from `gradience.exceptions`.

Key validation helper (private, inside `qa_artifact.py`):

```python
def _require_field(section: dict, field: str, section_name: str) -> Any:
    if field not in section:
        raise QASchemaError(f"Missing required field: {section_name}.{field}")
    return section[field]

def _require_list_of_str(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise QASchemaError(f"Field '{field_name}' must be a list of strings")
    return value

def _to_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise QASchemaError(f"Field '{field_name}' must be numeric, got {type(value).__name__}")
    return float(value)
```

Validation order in `from_dict()`:
1. Check `schema` key present and equals `"gradience.adapter_qa/v1"`
2. Check four required sections are present and are dicts
3. Check required fields within each section with type enforcement
4. Check `eligibility.status` is a known enum value (raise `QASchemaError` if not)
5. Check `eligibility.confidence` if present is one of `"high"`, `"medium"`, `"low"`
6. Check `flags`, `reasons`, `notes` are `list[str]` if present
7. Backfill `notes` and `reasons` to `[]` if missing
8. Ignore extra keys

**Step 4: Remove the old `test_from_dict_unknown_status_on_bad_value` test**

This test expects silent fallback to UNKNOWN on bad status. Now it should expect `QASchemaError`. The new `test_rejects_unknown_status` in `TestFromDictValidation` covers this case. Remove or replace the old test.

Also update `test_from_dict_empty` — an empty dict should now raise `QASchemaError`, not silently construct a default artifact.

**Step 5: Run tests**

Run: `pytest tests/test_qa_artifact.py -v`
Expected: ALL PASS

**Step 6: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add gradience/vnext/audit/qa_artifact.py tests/test_qa_artifact.py
git commit -m "Add strict validation to AdapterQAArtifact.from_dict()

from_dict() is now the single canonical gatekeeper for v1 schema.
Validates required sections, required fields with type enforcement,
known status values, and known confidence levels. Raises
QASchemaError for contract violations. Extra keys silently ignored
for forward compatibility."
```

---

### Task 6: Update `_load_source_qa` loader boundary

**Files:**
- Modify: `gradience/cli.py`
- Test: `tests/test_qa_artifact.py` (existing `TestLoadSourceQA`)

**Step 1: Write failing test for wrong-schema rejection**

In `tests/test_qa_artifact.py`, add to `TestLoadSourceQA`:

```python
def test_load_rejects_wrong_schema(self, tmp_path):
    """Wrong schema key should raise QASchemaError, not fall back to legacy."""
    d = {"schema": "gradience.adapter_qa/v99", "adapter": {}}
    p = tmp_path / "bad_schema.json"
    p.write_text(json.dumps(d))

    from gradience.cli import _load_source_qa
    # Should exit(1) due to QASchemaError, not silently fall through
    with pytest.raises(SystemExit):
        _load_source_qa(str(p))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_qa_artifact.py::TestLoadSourceQA::test_load_rejects_wrong_schema -v`
Expected: FAIL (currently catches all exceptions generically)

**Step 3: Update `_load_source_qa` in `cli.py`**

Update the three-way routing:

```python
def _load_source_qa(path_str: str | None) -> Any:
    if path_str is None:
        return None
    import json as jsonlib

    p = Path(path_str)
    if not p.is_file():
        print(f"Error: --source-*-qa path does not exist: {p}")
        sys.exit(1)
    try:
        with open(p) as f:
            data = jsonlib.load(f)
    except Exception as e:
        print(f"Error: Failed to parse QA file {p}: {e}")
        sys.exit(1)

    # Three-way routing
    schema_key = data.get("schema") if isinstance(data, dict) else None

    if schema_key is not None:
        # Schema present — must go through strict v1 loader
        try:
            from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
            return AdapterQAArtifact.from_dict(data).to_qa_result()
        except Exception as e:
            print(f"Error: Invalid QA artifact {p}: {e}")
            sys.exit(1)

    # Schema absent — legacy flat format
    try:
        from gradience.vnext.merge.eligibility import AdapterQAResult
        return AdapterQAResult.from_dict(data)
    except Exception as e:
        print(f"Error: Failed to load legacy QA file {p}: {e}")
        sys.exit(1)
```

**Step 4: Run tests**

Run: `pytest tests/test_qa_artifact.py::TestLoadSourceQA -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add gradience/cli.py tests/test_qa_artifact.py
git commit -m "Update _load_source_qa with three-way schema routing

Schema present + correct: strict v1 loader. Schema absent: legacy
parser. Schema present + wrong: hard fail, no fallback to legacy."
```

---

### Task 7: Implement symmetric margin in `classify_eligibility()`

**Files:**
- Modify: `gradience/vnext/merge/eligibility.py`
- Test: `tests/merge/test_eligibility.py`

**Step 1: Write failing test for symmetric margin**

In `tests/merge/test_eligibility.py`, add:

```python
def test_uncertain_symmetric_negative_delta_within_margin():
    """Small negative delta within margin should be uncertain, not flagged_weak."""
    result = classify_eligibility(
        adapter_path="./test",
        adapter_metric=4.70,  # slightly worse
        base_metric=4.66,     # base is better (lower is better)
        metric_name="perplexity",
        lower_is_better=True,
        margin=0.1,
    )
    # delta = 4.66 - 4.70 = -0.04, within [-0.1, 0.1] → uncertain
    assert result.status == EligibilityStatus.UNCERTAIN
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/merge/test_eligibility.py::test_uncertain_symmetric_negative_delta_within_margin -v`
Expected: FAIL (currently returns `flagged_weak` for any negative delta)

**Step 3: Fix `classify_eligibility()`**

In `gradience/vnext/merge/eligibility.py`, change:

```python
# Old:
if delta > margin:
    status = EligibilityStatus.ELIGIBLE
    ...
elif delta >= 0:
    status = EligibilityStatus.UNCERTAIN
    ...
else:
    status = EligibilityStatus.FLAGGED_WEAK
    ...

# New:
if delta > margin:
    status = EligibilityStatus.ELIGIBLE
    ...
elif delta >= -margin:
    status = EligibilityStatus.UNCERTAIN
    ...
else:
    status = EligibilityStatus.FLAGGED_WEAK
    ...
```

Update the notes string for the uncertain case to mention the symmetric band:

```python
notes = f"Adapter is within margin of base (delta={delta:.4f}, margin=\u00b1{margin:.4f})."
```

**Step 4: Run tests**

Run: `pytest tests/merge/test_eligibility.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add gradience/vnext/merge/eligibility.py tests/merge/test_eligibility.py
git commit -m "Implement symmetric margin for eligibility classification

Margin now applies symmetrically: -margin <= delta <= margin is
uncertain. Previously any negative delta was immediately flagged_weak.
With default margin=0.0, behavior is identical to before."
```

---

### Task 8: Update `--strict-qa` to block `unknown_no_behavioral_eval`

**Files:**
- Modify: `gradience/cli.py`
- Modify: `gradience/vnext/merge/recommend.py` (add `any_unverified` property)
- Test: `tests/test_qa_artifact.py` or a new test in `tests/merge/`

**Step 1: Write failing test**

Add a test that verifies strict-qa blocks unknown status. This can be tested via the `EligibilityContext` helper since the CLI calls `diagnose_pair`:

In `tests/merge/test_eligibility.py`:

```python
def test_eligibility_context_any_unverified():
    """any_unverified should be True when either adapter has unknown status."""
    from gradience.vnext.merge.recommend import EligibilityContext
    ctx = EligibilityContext(
        status_a=EligibilityStatus.ELIGIBLE,
        status_b=EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL,
    )
    assert ctx.any_unverified is True

def test_eligibility_context_not_unverified_when_both_known():
    from gradience.vnext.merge.recommend import EligibilityContext
    ctx = EligibilityContext(
        status_a=EligibilityStatus.ELIGIBLE,
        status_b=EligibilityStatus.UNCERTAIN,
    )
    assert ctx.any_unverified is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/merge/test_eligibility.py::test_eligibility_context_any_unverified -v`
Expected: FAIL (no `any_unverified` property)

**Step 3: Add `any_unverified` property to `EligibilityContext`**

In `gradience/vnext/merge/recommend.py`:

```python
@property
def any_unverified(self) -> bool:
    return (
        self.status_a == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL
        or self.status_b == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL
    )
```

**Step 4: Update strict-qa gate in CLI**

In `gradience/cli.py`, in the `--strict-qa` block (around line 2250), after the `any_weak` check, add:

```python
if diag.eligibility.any_unverified:
    unverified_labels = []
    if diag.eligibility.status_a == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL:
        unverified_labels.append("A")
    if diag.eligibility.status_b == EligibilityStatus.UNKNOWN_NO_BEHAVIORAL_EVAL:
        unverified_labels.append("B")
    print(f"\nError: --strict-qa gate failed. Adapter(s) {', '.join(unverified_labels)} have no behavioral evaluation.")
    print("  Strict mode requires behavioral evidence for eligibility. Provide evaluation scores via audit-adapter.")
    sys.exit(1)
```

This check should come after the `has_data` check and before or after the `any_weak` check.

**Step 5: Run tests**

Run: `pytest tests/merge/test_eligibility.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add gradience/cli.py gradience/vnext/merge/recommend.py tests/merge/test_eligibility.py
git commit -m "Block unknown_no_behavioral_eval under --strict-qa

Strict mode means behavioral evidence is required before merge
recommendation proceeds. Both flagged_weak and unknown are now
blocked. Uncertain is allowed with warning."
```

---

### Task 9: Promote to public API

**Files:**
- Modify: `gradience/__init__.py`
- Modify: `gradience/api.py`
- Test: `tests/test_qa_artifact.py`

**Step 1: Write failing test for public imports**

In `tests/test_qa_artifact.py`, add:

```python
class TestPublicAPI:
    def test_import_from_init(self):
        from gradience import AdapterQAArtifact, EligibilityStatus
        assert AdapterQAArtifact is not None
        assert EligibilityStatus is not None

    def test_audit_adapter_importable(self):
        from gradience.api import audit_adapter
        assert callable(audit_adapter)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qa_artifact.py::TestPublicAPI -v`
Expected: FAIL (not exported yet)

**Step 3: Add exports to `gradience/__init__.py`**

Add after the existing exception import:

```python
from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.merge.eligibility import EligibilityStatus
```

Add to `__all__`:

```python
"AdapterQAArtifact",
"EligibilityStatus",
```

**Step 4: Add `audit_adapter()` to `gradience/api.py`**

```python
def audit_adapter(
    *,
    peft_dir: str | Path,
    base_model: str | None = None,
    adapter_score: float | None = None,
    base_score: float | None = None,
    metric_name: str | None = None,
    lower_is_better: bool = True,
    eval_dataset: str | None = None,
    margin: float = 0.0,
    notes: list[str] | None = None,
    adapter_config_path: str | Path | None = None,
    adapter_weights_path: str | Path | None = None,
    base_norms_cache: str | Path | None = None,
    compute_udr: bool = True,
) -> Any:
    """Produce an AdapterQAArtifact from a PEFT adapter directory.

    This is the preferred stable Python entry point for producing adapter
    QA artifacts.  It runs the structural audit internally and builds the
    artifact through the same policy path as the CLI.

    Parameters
    ----------
    peft_dir
        Path to the PEFT adapter directory.
    base_model
        Base model identifier (e.g. ``meta-llama/Llama-2-7b-hf``).
    adapter_score, base_score
        Behavioral evaluation scores.  Both must be provided for
        behavioral eligibility classification.
    metric_name
        Name of the evaluation metric.
    lower_is_better
        True for metrics like perplexity where lower = better.
    eval_dataset
        Dataset used for evaluation.
    margin
        Tolerance margin for eligibility classification (symmetric).
    notes
        Optional list of caveats or annotations.
    adapter_config_path, adapter_weights_path
        Override default adapter config/weights file detection.
    base_norms_cache
        Path to cached base model norms for UDR computation.
    compute_udr
        Whether to compute Update Dominance Ratio metrics.

    Returns
    -------
    AdapterQAArtifact
        The canonical QA artifact for this adapter.
    """
    from gradience.vnext.audit import audit_lora_peft_dir
    from gradience.vnext.audit.qa_artifact import AdapterQAArtifact, build_qa_artifact

    result = audit_lora_peft_dir(
        str(peft_dir),
        adapter_config_path=str(adapter_config_path) if adapter_config_path else None,
        adapter_weights_path=str(adapter_weights_path) if adapter_weights_path else None,
        map_location="cpu",
        base_model_id=base_model,
        base_norms_cache=str(base_norms_cache) if base_norms_cache else None,
        compute_udr=compute_udr,
    )

    artifact = build_qa_artifact(
        result,
        adapter_path=str(peft_dir),
        base_model=base_model or "",
        adapter_score=adapter_score,
        base_score=base_score,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        eval_dataset=eval_dataset,
        margin=margin,
        notes=notes,
    )

    return artifact
```

Note: return type is `Any` to avoid importing the dataclass at module level (follows existing api.py pattern of lazy imports). The docstring documents the actual return type.

**Step 5: Run tests**

Run: `pytest tests/test_qa_artifact.py::TestPublicAPI -v`
Expected: ALL PASS

**Step 6: Run lint + type check**

Run: `ruff check gradience/__init__.py gradience/api.py && mypy gradience/api.py`
Expected: PASS

**Step 7: Commit**

```bash
git add gradience/__init__.py gradience/api.py tests/test_qa_artifact.py
git commit -m "Promote AdapterQAArtifact and EligibilityStatus to public API

audit_adapter() in gradience.api is the stable Python entry point.
AdapterQAArtifact and EligibilityStatus exported from gradience.__init__.
Python API and CLI use the same builder policy path."
```

---

### Task 10: Update example files to frozen v1 format

**Files:**
- Modify: `examples/qa/eligible_adapter_qa.json`
- Modify: `examples/qa/catsubcat_r16_qa.json`
- Modify: `examples/qa/btgenbot_r8_qa.json`
- Modify: `examples/qa/structural_only_qa.json`
- Create: `examples/qa/uncertain_adapter_qa.json`
- Test: `tests/test_qa_artifact.py` (`TestExampleFiles`)

**Step 1: Update all existing example files**

Apply all schema changes to each file:
- Rename `energy_rank_90_p50` key to `effective_rank_90_median`
- Change `"unknown"` status to `"unknown_no_behavioral_eval"`
- Change `metric_name: ""` to `metric_name: null` where no eval
- Change `lower_is_better: true` to `lower_is_better: null` where no eval
- Add `"notes": []` to all files that don't have it

**Step 2: Create `uncertain_adapter_qa.json`**

Use the example from the design doc Section 3a.

**Step 3: Update `TestExampleFiles` parametrization**

Add the uncertain example to the test parametrization:

```python
@pytest.mark.parametrize(
    "filename,expected_status",
    [
        ("catsubcat_r16_qa.json", "flagged_weak"),
        ("btgenbot_r8_qa.json", "flagged_weak"),
        ("eligible_adapter_qa.json", "eligible"),
        ("structural_only_qa.json", "unknown_no_behavioral_eval"),
        ("uncertain_adapter_qa.json", "uncertain"),
    ],
)
```

**Step 4: Run tests**

Run: `pytest tests/test_qa_artifact.py::TestExampleFiles -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add examples/qa/ tests/test_qa_artifact.py
git commit -m "Update example QA files to frozen v1 format

All examples use the frozen field names and status values. Added
uncertain_adapter_qa.json to cover all four eligibility statuses."
```

---

### Task 11: Write definition doc

**Files:**
- Create: `docs/adapter-qa-artifact.md`

**Step 1: Write the definition document**

Follow the headings from the design doc Section 3b:

1. What it is (one sentence definition)
2. How to produce it (CLI `gradience audit-adapter` + Python `gradience.api.audit_adapter()`)
3. How to read it (section walkthrough)
4. How to consume it (`merge-audit --source-a-qa`)
5. Schema contract (field table with types)
6. Decision semantics (status table, confidence rules, margin)
7. Versioning policy (additive only)

Keep it short. Definition document, not tutorial.

**Step 2: Commit**

```bash
git add docs/adapter-qa-artifact.md
git commit -m "Add adapter QA artifact definition document

Defines what the QA artifact is, how to produce and consume it,
the frozen schema contract, and decision semantics."
```

---

### Task 12: Final validation

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Run lint + format check**

Run: `ruff check . && ruff format --check .`
Expected: PASS

**Step 3: Run mypy**

Run: `mypy gradience/`
Expected: PASS (or no new errors)

**Step 4: Verify example files load through from_dict**

Quick smoke test — verify each example file passes strict validation:

```bash
python3 -c "
import json
from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
for f in ['eligible_adapter_qa', 'uncertain_adapter_qa', 'catsubcat_r16_qa', 'btgenbot_r8_qa', 'structural_only_qa']:
    with open(f'examples/qa/{f}.json') as fh:
        AdapterQAArtifact.from_dict(json.load(fh))
    print(f'  {f}: OK')
print('All example files pass strict validation.')
"
```

**Step 5: Commit any fixups**

If anything needed fixing, commit with descriptive message.

**Step 6: Final commit — update CLAUDE.md**

Add QA artifact to the Architecture & Conventions section of `CLAUDE.md`:

```markdown
### Adapter QA Artifact (`vnext/audit/qa_artifact.py`)

- Schema: `gradience.adapter_qa/v1` (frozen, additive-only versioning)
- Canonical producer: `gradience audit-adapter` CLI / `gradience.api.audit_adapter()` Python API
- Canonical consumer: `gradience merge-audit --source-a-qa` / `--source-b-qa`
- Public exports: `AdapterQAArtifact`, `EligibilityStatus` from `gradience.__init__`
- Status values: `eligible`, `uncertain`, `flagged_weak`, `unknown_no_behavioral_eval`
- `--strict-qa` blocks both `flagged_weak` and `unknown_no_behavioral_eval`
- Definition doc: `docs/adapter-qa-artifact.md`
```

```bash
git add CLAUDE.md
git commit -m "Add QA artifact conventions to CLAUDE.md"
```
