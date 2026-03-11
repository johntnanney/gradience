# Inventory Summary v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a stable, descriptive inventory-level summary object that aggregates adapter QA artifacts and pairwise merge-risk reports into counts and distributions.

**Architecture:** New `gradience/vnext/inventory/summary.py` module containing `InventorySummary` frozen dataclass with `build_inventory_summary()` builder and `format_inventory_summary()` formatter. Direct Python aggregation (no subprocess). `summarize_inventory()` in `gradience/api.py` as stable entry point. CLI command `summarize-inventory` in `gradience/cli.py`.

**Tech Stack:** Python dataclasses, pytest, ruff, mypy

---

## Context

The design document is at `docs/plans/2026-03-11-inventory-summary-v1-design.md`. Reference it for schema shape, validation rules, and policy decisions.

Key files:
- `gradience/vnext/audit/qa_artifact.py` — `AdapterQAArtifact` (status at `.status.value`, flags at `.structural_flags`)
- `gradience/vnext/merge/qa_report.py` — `MergeQAReport` (pair_risk, dominant_issue, recommended_strategy, adapter_a/b.eligibility_status)
- `gradience/exceptions.py` — `QASchemaError`
- `gradience/api.py` — stable Python API wrappers
- `gradience/__init__.py` — public exports
- `gradience/cli.py` — CLI commands (`main()` at line 3364, `_setup_*_command` pattern)

Phase A/B precedents:
- `AdapterQAArtifact.from_dict()` in `qa_artifact.py` — validation pattern
- `MergeQAReport.from_dict()` in `qa_report.py` — validation pattern
- `format_qa_report()` in `qa_report.py` — terminal format pattern

---

### Task 1: Create `InventorySummary` dataclass and `from_dict()` validation

**Files:**
- Create: `gradience/vnext/inventory/__init__.py`
- Create: `gradience/vnext/inventory/summary.py`
- Create: `tests/test_inventory_summary.py`

**Step 1: Create package init**

Create empty `gradience/vnext/inventory/__init__.py`:

```python
```

**Step 2: Write the failing tests**

Create `tests/test_inventory_summary.py`:

```python
"""Tests for gradience.vnext.inventory.summary — inventory summary builder and validator."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gradience.exceptions import QASchemaError
from gradience.vnext.inventory.summary import (
    SCHEMA_ID,
    InventorySummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_dict() -> dict[str, Any]:
    """Minimal valid v1 dict."""
    return {
        "schema": "gradience.inventory_summary/v1",
        "sources": {
            "qa_artifact_count": 3,
            "merge_report_count": 2,
        },
        "adapter_status_counts": {
            "eligible": 2,
            "flagged_weak": 1,
        },
        "adapter_flag_counts": {
            "low_utilization": 1,
        },
        "pair_risk_counts": {
            "low": 1,
            "high": 1,
        },
        "recommended_strategy_counts": {
            "linear": 1,
            "audit_aware": 1,
        },
        "dominant_issue_counts": {
            "none": 1,
            "norm_imbalance": 1,
        },
        "strict_qa_block_candidates": 1,
        "notes": [],
    }


# ---------------------------------------------------------------------------
# Tests — from_dict validation
# ---------------------------------------------------------------------------


class TestFromDictValidation:
    """Strict from_dict validation (parallel to Phase A/B)."""

    def test_valid_roundtrip(self):
        d = _valid_dict()
        summary = InventorySummary.from_dict(d)
        assert summary.sources["qa_artifact_count"] == 3
        assert summary.sources["merge_report_count"] == 2
        assert summary.adapter_status_counts["eligible"] == 2
        assert summary.strict_qa_block_candidates == 1

    def test_to_dict_roundtrip(self):
        d = _valid_dict()
        summary = InventorySummary.from_dict(d)
        d2 = summary.to_dict()
        summary2 = InventorySummary.from_dict(d2)
        assert summary == summary2

    def test_missing_schema_raises(self):
        d = _valid_dict()
        del d["schema"]
        with pytest.raises(QASchemaError, match="schema"):
            InventorySummary.from_dict(d)

    def test_wrong_schema_raises(self):
        d = _valid_dict()
        d["schema"] = "wrong/v1"
        with pytest.raises(QASchemaError, match="Expected schema"):
            InventorySummary.from_dict(d)

    def test_missing_sources_raises(self):
        d = _valid_dict()
        del d["sources"]
        with pytest.raises(QASchemaError, match="sources"):
            InventorySummary.from_dict(d)

    def test_missing_adapter_status_counts_raises(self):
        d = _valid_dict()
        del d["adapter_status_counts"]
        with pytest.raises(QASchemaError, match="adapter_status_counts"):
            InventorySummary.from_dict(d)

    def test_missing_pair_risk_counts_raises(self):
        d = _valid_dict()
        del d["pair_risk_counts"]
        with pytest.raises(QASchemaError, match="pair_risk_counts"):
            InventorySummary.from_dict(d)

    def test_missing_strict_qa_block_candidates_raises(self):
        d = _valid_dict()
        del d["strict_qa_block_candidates"]
        with pytest.raises(QASchemaError, match="strict_qa_block_candidates"):
            InventorySummary.from_dict(d)

    def test_non_int_count_value_raises(self):
        d = _valid_dict()
        d["adapter_status_counts"]["eligible"] = "two"
        with pytest.raises(QASchemaError, match="adapter_status_counts"):
            InventorySummary.from_dict(d)

    def test_non_int_source_count_raises(self):
        d = _valid_dict()
        d["sources"]["qa_artifact_count"] = 3.5
        with pytest.raises(QASchemaError, match="sources"):
            InventorySummary.from_dict(d)

    def test_non_int_strict_qa_raises(self):
        d = _valid_dict()
        d["strict_qa_block_candidates"] = "many"
        with pytest.raises(QASchemaError, match="strict_qa_block_candidates"):
            InventorySummary.from_dict(d)

    def test_caveats_must_be_list_of_str(self):
        d = _valid_dict()
        d["notes"] = [1, 2, 3]
        with pytest.raises(QASchemaError, match="notes"):
            InventorySummary.from_dict(d)

    def test_notes_backfilled(self):
        d = _valid_dict()
        del d["notes"]
        summary = InventorySummary.from_dict(d)
        assert summary.notes == ()

    def test_extra_keys_ignored(self):
        d = _valid_dict()
        d["future_field"] = "should not fail"
        summary = InventorySummary.from_dict(d)
        assert summary.sources["qa_artifact_count"] == 3

    def test_empty_count_dicts_accepted(self):
        d = _valid_dict()
        d["adapter_status_counts"] = {}
        d["adapter_flag_counts"] = {}
        d["pair_risk_counts"] = {}
        d["recommended_strategy_counts"] = {}
        d["dominant_issue_counts"] = {}
        summary = InventorySummary.from_dict(d)
        assert summary.adapter_status_counts == {}
```

**Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_inventory_summary.py -v`
Expected: FAIL with `ModuleNotFoundError` (module doesn't exist yet)

**Step 4: Write `InventorySummary` dataclass and `from_dict()`**

Create `gradience/vnext/inventory/summary.py`:

```python
"""
Inventory Summary — batch aggregation of adapter QA artifacts and merge risk reports.

Produces an ``InventorySummary`` that counts adapter statuses, structural flags,
pair risks, recommended strategies, and dominant issues across an inventory of
already-produced artifacts. This is a descriptive object, not a decision-bearing one.

Schema identifier: ``gradience.inventory_summary/v1``

Usage::

    from gradience.vnext.inventory.summary import build_inventory_summary

    summary = build_inventory_summary(qa_artifacts, merge_reports)
    print(format_inventory_summary(summary))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gradience.exceptions import QASchemaError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_ID = "gradience.inventory_summary/v1"

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InventorySummary:
    """Descriptive summary of an adapter inventory.

    Aggregates counts from adapter QA artifacts and merge risk reports.
    Does not invent new judgments — only summarizes existing ones.
    """

    sources: dict[str, int]
    adapter_status_counts: dict[str, int]
    adapter_flag_counts: dict[str, int]
    pair_risk_counts: dict[str, int]
    recommended_strategy_counts: dict[str, int]
    dominant_issue_counts: dict[str, int]
    strict_qa_block_candidates: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_ID,
            "sources": dict(self.sources),
            "adapter_status_counts": dict(self.adapter_status_counts),
            "adapter_flag_counts": dict(self.adapter_flag_counts),
            "pair_risk_counts": dict(self.pair_risk_counts),
            "recommended_strategy_counts": dict(self.recommended_strategy_counts),
            "dominant_issue_counts": dict(self.dominant_issue_counts),
            "strict_qa_block_candidates": self.strict_qa_block_candidates,
            "notes": list(self.notes),
        }

    def to_json(self, path: Path | str) -> None:
        """Write inventory summary to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InventorySummary:
        """Deserialize from a v1 schema dict.

        Single canonical gatekeeper for the inventory_summary/v1 schema.
        Validates schema identity, required sections, and type enforcement.
        Raises ``QASchemaError`` for contract violations.
        Extra keys are silently ignored for forward compatibility.
        """
        # --- Schema identity ---
        if "schema" not in d:
            raise QASchemaError("Missing required field: schema")
        if d["schema"] != SCHEMA_ID:
            raise QASchemaError(f"Expected schema '{SCHEMA_ID}', got '{d['schema']}'")

        # --- Required count-map sections ---
        count_map_sections = [
            "sources",
            "adapter_status_counts",
            "adapter_flag_counts",
            "pair_risk_counts",
            "recommended_strategy_counts",
            "dominant_issue_counts",
        ]
        validated_maps: dict[str, dict[str, int]] = {}
        for section_name in count_map_sections:
            if section_name not in d:
                raise QASchemaError(f"Missing required section: {section_name}")
            raw = d[section_name]
            if not isinstance(raw, dict):
                raise QASchemaError(f"Section '{section_name}' must be a dict")
            for k, v in raw.items():
                if not isinstance(v, int):
                    raise QASchemaError(
                        f"Field '{section_name}[\"{k}\"]' must be int, got {type(v).__name__}"
                    )
            validated_maps[section_name] = raw

        # --- strict_qa_block_candidates (required, int) ---
        if "strict_qa_block_candidates" not in d:
            raise QASchemaError("Missing required field: strict_qa_block_candidates")
        raw_block = d["strict_qa_block_candidates"]
        if not isinstance(raw_block, int):
            raise QASchemaError(
                f"Field 'strict_qa_block_candidates' must be int, got {type(raw_block).__name__}"
            )

        # --- notes (optional, list[str]) ---
        raw_notes = d.get("notes")
        if raw_notes is None:
            notes: tuple[str, ...] = ()
        else:
            if not isinstance(raw_notes, list) or not all(isinstance(x, str) for x in raw_notes):
                raise QASchemaError("Field 'notes' must be a list of strings")
            notes = tuple(raw_notes)

        return cls(
            sources=validated_maps["sources"],
            adapter_status_counts=validated_maps["adapter_status_counts"],
            adapter_flag_counts=validated_maps["adapter_flag_counts"],
            pair_risk_counts=validated_maps["pair_risk_counts"],
            recommended_strategy_counts=validated_maps["recommended_strategy_counts"],
            dominant_issue_counts=validated_maps["dominant_issue_counts"],
            strict_qa_block_candidates=raw_block,
            notes=notes,
        )
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_inventory_summary.py -v`
Expected: All 15 tests pass.

**Step 6: Commit**

```bash
git add gradience/vnext/inventory/__init__.py gradience/vnext/inventory/summary.py tests/test_inventory_summary.py
git commit -m "Add InventorySummary dataclass with strict from_dict validation

Frozen schema gradience.inventory_summary/v1. Validates required
sections, int count values, and notes list. Reuses QASchemaError."
```

---

### Task 2: Add `build_inventory_summary()` aggregation function

**Files:**
- Modify: `gradience/vnext/inventory/summary.py`
- Modify: `tests/test_inventory_summary.py`

**Step 1: Write failing tests for aggregation**

Add to `tests/test_inventory_summary.py`:

```python
from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.summary import build_inventory_summary
from gradience.vnext.merge.eligibility import EligibilityStatus
from gradience.vnext.merge.qa_report import MergeQAReport


def _make_qa_artifact(
    status: str = "eligible",
    flags: list[str] | None = None,
) -> AdapterQAArtifact:
    """Create a minimal AdapterQAArtifact for testing."""
    return AdapterQAArtifact(
        adapter_name="test",
        adapter_path="/tmp/test",
        base_model="llama",
        rank_nominal=8,
        n_layers=32,
        utilization_mean=0.5,
        utilization_median=0.5,
        stable_rank_mean=4.0,
        energy_rank_90_p50=4.0,
        rank_waste_ratio=0.5,
        structural_flags=flags or [],
        eval_available=True,
        status=EligibilityStatus(status),
    )


def _make_merge_report_dict(
    pair_risk: str = "low",
    dominant_issue: str = "none",
    strategy: str = "linear",
    eligibility_a: str | None = "eligible",
    eligibility_b: str | None = "eligible",
) -> dict:
    """Create a minimal valid MergeQAReport dict for from_dict loading."""
    return {
        "schema": "gradience.merge_qa_report/v1",
        "adapter_a": {
            "path": "/tmp/a",
            "rank": 8,
            "eligibility_status": eligibility_a,
        },
        "adapter_b": {
            "path": "/tmp/b",
            "rank": 8,
            "eligibility_status": eligibility_b,
        },
        "pair_risk": pair_risk,
        "dominant_issue": dominant_issue,
        "recommended_strategy": strategy,
        "confidence": "high",
        "compatibility_score": 0.9,
    }


# ---------------------------------------------------------------------------
# Tests — aggregation
# ---------------------------------------------------------------------------


class TestBuildInventorySummary:
    def test_empty_inputs(self):
        summary = build_inventory_summary([], [])
        assert summary.sources == {"qa_artifact_count": 0, "merge_report_count": 0}
        assert summary.adapter_status_counts == {}
        assert summary.adapter_flag_counts == {}
        assert summary.pair_risk_counts == {}
        assert summary.strict_qa_block_candidates == 0

    def test_adapter_status_counts(self):
        artifacts = [
            _make_qa_artifact("eligible"),
            _make_qa_artifact("eligible"),
            _make_qa_artifact("flagged_weak"),
        ]
        summary = build_inventory_summary(artifacts, [])
        assert summary.adapter_status_counts == {"eligible": 2, "flagged_weak": 1}

    def test_adapter_flag_counts(self):
        artifacts = [
            _make_qa_artifact(flags=["low_utilization", "high_rank_waste"]),
            _make_qa_artifact(flags=["low_utilization"]),
            _make_qa_artifact(flags=[]),
        ]
        summary = build_inventory_summary(artifacts, [])
        assert summary.adapter_flag_counts == {"low_utilization": 2, "high_rank_waste": 1}

    def test_pair_risk_counts(self):
        reports = [
            MergeQAReport.from_dict(_make_merge_report_dict(pair_risk="low")),
            MergeQAReport.from_dict(_make_merge_report_dict(pair_risk="high")),
            MergeQAReport.from_dict(_make_merge_report_dict(pair_risk="high")),
        ]
        summary = build_inventory_summary([], reports)
        assert summary.pair_risk_counts == {"low": 1, "high": 2}

    def test_strategy_counts(self):
        reports = [
            MergeQAReport.from_dict(_make_merge_report_dict(strategy="linear")),
            MergeQAReport.from_dict(_make_merge_report_dict(strategy="audit_aware")),
        ]
        summary = build_inventory_summary([], reports)
        assert summary.recommended_strategy_counts == {"linear": 1, "audit_aware": 1}

    def test_dominant_issue_counts(self):
        reports = [
            MergeQAReport.from_dict(_make_merge_report_dict(dominant_issue="none")),
            MergeQAReport.from_dict(_make_merge_report_dict(dominant_issue="norm_imbalance")),
        ]
        summary = build_inventory_summary([], reports)
        assert summary.dominant_issue_counts == {"none": 1, "norm_imbalance": 1}

    def test_strict_qa_block_candidates_flagged_weak(self):
        reports = [
            MergeQAReport.from_dict(_make_merge_report_dict(eligibility_a="flagged_weak")),
            MergeQAReport.from_dict(_make_merge_report_dict()),  # both eligible
        ]
        summary = build_inventory_summary([], reports)
        assert summary.strict_qa_block_candidates == 1

    def test_strict_qa_block_candidates_null(self):
        reports = [
            MergeQAReport.from_dict(_make_merge_report_dict(eligibility_b=None)),
        ]
        summary = build_inventory_summary([], reports)
        assert summary.strict_qa_block_candidates == 1

    def test_strict_qa_block_candidates_unknown(self):
        reports = [
            MergeQAReport.from_dict(
                _make_merge_report_dict(eligibility_a="unknown_no_behavioral_eval")
            ),
        ]
        summary = build_inventory_summary([], reports)
        assert summary.strict_qa_block_candidates == 1

    def test_strict_qa_block_not_double_counted(self):
        """A report with both adapters flagged counts as 1, not 2."""
        reports = [
            MergeQAReport.from_dict(
                _make_merge_report_dict(eligibility_a="flagged_weak", eligibility_b=None)
            ),
        ]
        summary = build_inventory_summary([], reports)
        assert summary.strict_qa_block_candidates == 1

    def test_to_dict_produces_valid_schema(self):
        artifacts = [_make_qa_artifact("eligible")]
        reports = [MergeQAReport.from_dict(_make_merge_report_dict())]
        summary = build_inventory_summary(artifacts, reports)
        d = summary.to_dict()
        assert d["schema"] == "gradience.inventory_summary/v1"
        # Round-trip
        summary2 = InventorySummary.from_dict(d)
        assert summary2 == summary
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_inventory_summary.py::TestBuildInventorySummary -v`
Expected: FAIL with `ImportError` (build_inventory_summary not defined)

**Step 3: Write `build_inventory_summary()`**

Add to `gradience/vnext/inventory/summary.py`, after the `InventorySummary` class:

```python
# ---------------------------------------------------------------------------
# Strict-QA block statuses
# ---------------------------------------------------------------------------

_STRICT_QA_BLOCK_STATUSES = frozenset({"flagged_weak", "unknown_no_behavioral_eval"})


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_inventory_summary(
    qa_artifacts: list[Any],
    merge_reports: list[Any],
) -> InventorySummary:
    """Build an inventory summary from already-parsed artifacts.

    Pure counting — no recomputation of eligibility or risk.

    Parameters
    ----------
    qa_artifacts
        List of ``AdapterQAArtifact`` objects.
    merge_reports
        List of ``MergeQAReport`` objects.

    Returns
    -------
    InventorySummary
    """
    from collections import Counter

    # --- Adapter QA counts ---
    status_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()

    for artifact in qa_artifacts:
        status_counter[artifact.status.value] += 1
        for flag in artifact.structural_flags:
            flag_counter[flag] += 1

    # --- Merge report counts ---
    risk_counter: Counter[str] = Counter()
    strategy_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    block_candidates = 0

    for report in merge_reports:
        risk_counter[report.pair_risk] += 1
        strategy_counter[report.recommended_strategy] += 1
        issue_counter[report.dominant_issue] += 1

        # Count strict-QA block candidates
        a_status = report.adapter_a.eligibility_status
        b_status = report.adapter_b.eligibility_status
        would_block = (
            a_status is None
            or b_status is None
            or a_status in _STRICT_QA_BLOCK_STATUSES
            or b_status in _STRICT_QA_BLOCK_STATUSES
        )
        if would_block:
            block_candidates += 1

    return InventorySummary(
        sources={
            "qa_artifact_count": len(qa_artifacts),
            "merge_report_count": len(merge_reports),
        },
        adapter_status_counts=dict(status_counter),
        adapter_flag_counts=dict(flag_counter),
        pair_risk_counts=dict(risk_counter),
        recommended_strategy_counts=dict(strategy_counter),
        dominant_issue_counts=dict(issue_counter),
        strict_qa_block_candidates=block_candidates,
        notes=(),
    )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_inventory_summary.py -v`
Expected: All tests pass (validation + aggregation).

**Step 5: Commit**

```bash
git add gradience/vnext/inventory/summary.py tests/test_inventory_summary.py
git commit -m "Add build_inventory_summary aggregation function

Pure counting over AdapterQAArtifact and MergeQAReport lists.
Counts statuses, flags, risks, strategies, issues, and strict-QA
block candidates."
```

---

### Task 3: Add `format_inventory_summary()` terminal formatter

**Files:**
- Modify: `gradience/vnext/inventory/summary.py`
- Modify: `tests/test_inventory_summary.py`

**Step 1: Write failing tests**

Add to `tests/test_inventory_summary.py`:

```python
from gradience.vnext.inventory.summary import format_inventory_summary


class TestFormatInventorySummary:
    def test_contains_header(self):
        summary = InventorySummary.from_dict(_valid_dict())
        output = format_inventory_summary(summary)
        assert "INVENTORY SUMMARY" in output

    def test_contains_sources(self):
        summary = InventorySummary.from_dict(_valid_dict())
        output = format_inventory_summary(summary)
        assert "QA artifacts:" in output
        assert "Merge reports:" in output

    def test_contains_adapter_status(self):
        summary = InventorySummary.from_dict(_valid_dict())
        output = format_inventory_summary(summary)
        assert "eligible" in output
        assert "flagged_weak" in output

    def test_contains_strict_qa_count(self):
        summary = InventorySummary.from_dict(_valid_dict())
        output = format_inventory_summary(summary)
        assert "STRICT-QA BLOCK CANDIDATES" in output

    def test_empty_section_omitted(self):
        d = _valid_dict()
        d["adapter_flag_counts"] = {}
        summary = InventorySummary.from_dict(d)
        output = format_inventory_summary(summary)
        assert "STRUCTURAL FLAGS" not in output
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_inventory_summary.py::TestFormatInventorySummary -v`
Expected: FAIL with `ImportError`

**Step 3: Write `format_inventory_summary()`**

Add to `gradience/vnext/inventory/summary.py`:

```python
# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


def format_inventory_summary(summary: InventorySummary) -> str:
    """Format an InventorySummary as clean, human-readable text."""
    lines: list[str] = []

    lines.append("")
    lines.append("  INVENTORY SUMMARY")
    lines.append("  " + "=" * 60)

    # Sources
    lines.append("")
    lines.append("  SOURCES")
    lines.append("  " + "-" * 40)
    lines.append(f"  QA artifacts:    {summary.sources.get('qa_artifact_count', 0)}")
    lines.append(f"  Merge reports:   {summary.sources.get('merge_report_count', 0)}")

    # Adapter status
    if summary.adapter_status_counts:
        lines.append("")
        lines.append("  ADAPTER STATUS")
        lines.append("  " + "-" * 40)
        for status, count in sorted(summary.adapter_status_counts.items()):
            lines.append(f"  {status}:  {count}")

    # Structural flags
    if summary.adapter_flag_counts:
        lines.append("")
        lines.append("  STRUCTURAL FLAGS")
        lines.append("  " + "-" * 40)
        for flag, count in sorted(summary.adapter_flag_counts.items()):
            lines.append(f"  {flag}:  {count}")

    # Pair risk
    if summary.pair_risk_counts:
        lines.append("")
        lines.append("  PAIR RISK")
        lines.append("  " + "-" * 40)
        for risk, count in sorted(summary.pair_risk_counts.items()):
            lines.append(f"  {risk}:  {count}")

    # Recommended strategies
    if summary.recommended_strategy_counts:
        lines.append("")
        lines.append("  RECOMMENDED STRATEGIES")
        lines.append("  " + "-" * 40)
        for strategy, count in sorted(summary.recommended_strategy_counts.items()):
            lines.append(f"  {strategy}:  {count}")

    # Dominant issues
    if summary.dominant_issue_counts:
        lines.append("")
        lines.append("  DOMINANT ISSUES")
        lines.append("  " + "-" * 40)
        for issue, count in sorted(summary.dominant_issue_counts.items()):
            lines.append(f"  {issue}:  {count}")

    # Strict-QA block candidates
    lines.append("")
    lines.append(f"  STRICT-QA BLOCK CANDIDATES: {summary.strict_qa_block_candidates}")
    lines.append("")

    return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_inventory_summary.py -v`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add gradience/vnext/inventory/summary.py tests/test_inventory_summary.py
git commit -m "Add format_inventory_summary terminal formatter

Human-readable output with sections for sources, adapter status,
structural flags, pair risk, strategies, issues, and block count.
Empty sections are omitted."
```

---

### Task 4: Add `summarize_inventory()` to public API

**Files:**
- Modify: `gradience/api.py`
- Modify: `gradience/__init__.py`
- Modify: `tests/test_qa_artifact.py` (add import test)

**Step 1: Write failing test**

Add to `tests/test_qa_artifact.py`:

```python
def test_inventory_summary_importable_from_gradience():
    from gradience import InventorySummary
    assert hasattr(InventorySummary, "from_dict")
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_qa_artifact.py::test_inventory_summary_importable_from_gradience -v`
Expected: FAIL with `ImportError`

**Step 3: Add export to `gradience/__init__.py`**

Add after the `MergeQAReport` import (line 73):

```python
from gradience.vnext.inventory.summary import InventorySummary
```

Add to `__all__` (after `"MergeQAReport"` entry):

```python
    # Inventory summary
    "InventorySummary",
```

**Step 4: Add `summarize_inventory()` to `gradience/api.py`**

Add after the `merge_risk_report()` function (after line 486):

```python
# -----------------------------
# Public API: Inventory Summary
# -----------------------------


def summarize_inventory(
    *,
    qa_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    qa_paths: list[str | Path] | None = None,
    report_paths: list[str | Path] | None = None,
    strict_input: bool = False,
) -> Any:
    """Summarize an inventory of adapter QA artifacts and merge risk reports.

    This is the stable Python entry point for inventory summarization.
    Scans directories or loads explicit paths, parses valid v1 artifacts,
    and aggregates counts into an ``InventorySummary``.

    Parameters
    ----------
    qa_dir
        Directory to scan for adapter QA artifact JSON files.
    report_dir
        Directory to scan for merge risk report JSON files.
    qa_paths
        Explicit list of QA artifact file paths.
    report_paths
        Explicit list of merge report file paths.
    strict_input
        If True, raise on first malformed file instead of skipping.

    Returns
    -------
    InventorySummary
        The inventory-level summary.

    Raises
    ------
    ValueError
        If no input sources are provided.
    QASchemaError
        If ``strict_input`` is True and a file fails validation.
    """
    import sys

    from gradience.exceptions import QASchemaError as _QASchemaError
    from gradience.vnext.audit.qa_artifact import AdapterQAArtifact as _Artifact
    from gradience.vnext.inventory.summary import build_inventory_summary as _build
    from gradience.vnext.merge.qa_report import MergeQAReport as _MergeReport

    if not any([qa_dir, report_dir, qa_paths, report_paths]):
        raise ValueError("At least one input source must be provided (qa_dir, report_dir, qa_paths, or report_paths)")

    def _collect_json_paths(directory: str | Path | None, explicit: list[str | Path] | None) -> list[Path]:
        paths: list[Path] = []
        if directory:
            paths.extend(sorted(Path(directory).glob("*.json")))
        if explicit:
            paths.extend(Path(p) for p in explicit)
        return paths

    def _load_artifacts(paths: list[Path]) -> list[Any]:
        artifacts = []
        for p in paths:
            try:
                data = _read_json(p)
                schema = data.get("schema", "")
                if schema.startswith("gradience.adapter_qa/"):
                    artifacts.append(_Artifact.from_dict(data))
            except (json.JSONDecodeError, _QASchemaError, KeyError) as exc:
                if strict_input:
                    raise
                print(f"Warning: skipping {p}: {exc}", file=sys.stderr)
        return artifacts

    def _load_reports(paths: list[Path]) -> list[Any]:
        reports = []
        for p in paths:
            try:
                data = _read_json(p)
                schema = data.get("schema", "")
                if schema.startswith("gradience.merge_qa_report/"):
                    reports.append(_MergeReport.from_dict(data))
            except (json.JSONDecodeError, _QASchemaError, KeyError) as exc:
                if strict_input:
                    raise
                print(f"Warning: skipping {p}: {exc}", file=sys.stderr)
        return reports

    qa_file_paths = _collect_json_paths(qa_dir, qa_paths)
    report_file_paths = _collect_json_paths(report_dir, report_paths)

    artifacts = _load_artifacts(qa_file_paths)
    reports = _load_reports(report_file_paths)

    return _build(artifacts, reports)
```

**Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_qa_artifact.py::test_inventory_summary_importable_from_gradience -v`
Expected: PASS

**Step 6: Commit**

```bash
git add gradience/__init__.py gradience/api.py tests/test_qa_artifact.py
git commit -m "Promote InventorySummary to public API

Export from gradience.__init__. Add summarize_inventory() to api.py
as stable Python entry point with directory scanning and malformed
file skip-with-warning behavior."
```

---

### Task 5: Add CLI `summarize-inventory` command

**Files:**
- Modify: `gradience/cli.py`

**Step 1: Add `_setup_summarize_inventory_command()` and `cmd_summarize_inventory()`**

Add before the `main()` function (before line 3364):

```python
def _setup_summarize_inventory_command(subparsers):
    p = subparsers.add_parser(
        "summarize-inventory",
        help="Summarize an inventory of adapter QA artifacts and merge risk reports",
    )
    p.add_argument(
        "--qa-dir",
        type=str,
        default=None,
        help="Directory to scan for adapter QA artifact JSON files",
    )
    p.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to scan for merge risk report JSON files",
    )
    p.add_argument(
        "--emit-report",
        type=str,
        default=None,
        help="Write inventory summary v1 JSON to this path",
    )
    p.add_argument(
        "--strict-input",
        action="store_true",
        help="Fail on first malformed input file instead of skipping",
    )
    p.set_defaults(func=cmd_summarize_inventory)


def cmd_summarize_inventory(args) -> None:
    """Summarize an inventory of adapter QA artifacts and merge risk reports."""
    from gradience.api import summarize_inventory

    qa_dir = getattr(args, "qa_dir", None)
    report_dir = getattr(args, "report_dir", None)
    strict_input = getattr(args, "strict_input", False)
    emit_path = getattr(args, "emit_report", None)

    if not qa_dir and not report_dir:
        print("Error: at least one of --qa-dir or --report-dir must be provided.")
        sys.exit(1)

    try:
        summary = summarize_inventory(
            qa_dir=qa_dir,
            report_dir=report_dir,
            strict_input=strict_input,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Print terminal summary
    from gradience.vnext.inventory.summary import format_inventory_summary

    print(format_inventory_summary(summary))

    # Optionally emit JSON
    if emit_path:
        summary.to_json(emit_path)
        print(f"\nInventory summary written to: {emit_path}")
```

**Step 2: Register the command in `main()`**

Add `_setup_summarize_inventory_command(subparsers)` in the `main()` function after the `_setup_monitor_command(subparsers)` call (line 3381):

```python
    _setup_summarize_inventory_command(subparsers)
```

**Step 3: Run a quick smoke test**

Run: `python3 -m gradience summarize-inventory --qa-dir examples/qa --report-dir examples/reports`
Expected: Terminal summary output with counts.

**Step 4: Commit**

```bash
git add gradience/cli.py
git commit -m "Add summarize-inventory CLI command

Scans directories for QA artifacts and merge reports, aggregates
counts, prints terminal summary. --emit-report writes v1 JSON.
--strict-input fails on malformed files."
```

---

### Task 6: Create canonical example and example file smoke test

**Files:**
- Create: `examples/inventory/inventory_summary.json`
- Modify: `tests/test_inventory_summary.py`

**Step 1: Create `examples/inventory/` directory and example file**

```bash
mkdir -p examples/inventory
```

Create `examples/inventory/inventory_summary.json`:

```json
{
  "schema": "gradience.inventory_summary/v1",
  "sources": {
    "qa_artifact_count": 5,
    "merge_report_count": 3
  },
  "adapter_status_counts": {
    "eligible": 2,
    "uncertain": 1,
    "flagged_weak": 1,
    "unknown_no_behavioral_eval": 1
  },
  "adapter_flag_counts": {
    "low_utilization": 3,
    "high_rank_waste": 2,
    "concentrated_spectrum": 1,
    "underutilized_capacity": 1
  },
  "pair_risk_counts": {
    "low": 1,
    "medium": 1,
    "high": 1
  },
  "recommended_strategy_counts": {
    "linear": 1,
    "norm_equalized": 1,
    "audit_aware": 1
  },
  "dominant_issue_counts": {
    "none": 1,
    "norm_imbalance": 1,
    "subspace_conflict": 1
  },
  "strict_qa_block_candidates": 2,
  "notes": []
}
```

**Step 2: Add example file smoke test**

Add to `tests/test_inventory_summary.py`:

```python
import glob


class TestExampleFiles:
    @pytest.mark.parametrize("path", sorted(glob.glob("examples/inventory/*.json")))
    def test_example_loads_via_from_dict(self, path):
        with open(path) as f:
            d = json.load(f)
        summary = InventorySummary.from_dict(d)
        assert summary.sources["qa_artifact_count"] >= 0
```

**Step 3: Add integration test using real example files**

```python
class TestIntegrationWithExampleFiles:
    def test_summarize_from_existing_examples(self):
        """Build summary from the existing QA and report example files."""
        from gradience.api import summarize_inventory

        summary = summarize_inventory(
            qa_dir="examples/qa",
            report_dir="examples/reports",
        )
        # Should have loaded some artifacts
        assert summary.sources["qa_artifact_count"] >= 1
        assert summary.sources["merge_report_count"] >= 1
        # Round-trip through from_dict
        d = summary.to_dict()
        summary2 = InventorySummary.from_dict(d)
        assert summary2 == summary
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_inventory_summary.py -v`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add examples/inventory/inventory_summary.json tests/test_inventory_summary.py
git commit -m "Add canonical inventory summary example and smoke tests

One example file in examples/inventory/. Integration test loads
existing QA and report examples through summarize_inventory()."
```

---

### Task 7: Write definition doc

**Files:**
- Create: `docs/inventory-summary.md`

**Step 1: Write the doc**

Create `docs/inventory-summary.md` following the same structure as `docs/adapter-qa-artifact.md` and `docs/merge-risk-report.md`. Headings:

1. What it is
2. How to produce it (CLI `summarize-inventory` and Python `summarize_inventory()`)
3. How to read it (section walkthrough)
4. How to consume it (scripting, `strict_input`, skip-with-warning default)
5. Schema contract (field table)
6. Malformed input behavior (skip vs strict)
7. Versioning policy

Keep it definition-style.

**Step 2: Commit**

```bash
git add docs/inventory-summary.md
git commit -m "Add inventory summary definition document"
```

---

### Task 8: Final validation and CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass.

**Step 2: Run lint and format**

Run: `ruff check gradience/ tests/test_inventory_summary.py && ruff format --check gradience/ tests/test_inventory_summary.py`
Fix any issues with `ruff check --fix` and `ruff format`.

**Step 3: Run mypy**

Run: `mypy gradience/`
Expected: No errors.

**Step 4: Verify example file smoke test**

Run:
```bash
python3 -c "
import json, glob
from gradience.vnext.inventory.summary import InventorySummary
for f in sorted(glob.glob('examples/inventory/*.json')):
    with open(f) as fp:
        d = json.load(fp)
    s = InventorySummary.from_dict(d)
    print(f'{f}: {s.sources} / block_candidates={s.strict_qa_block_candidates}')
"
```

**Step 5: Verify CLI**

Run: `python3 -m gradience summarize-inventory --qa-dir examples/qa --report-dir examples/reports`
Expected: Terminal summary output.

**Step 6: Update CLAUDE.md**

Add after the Merge QA Report section:

```markdown
### Inventory Summary (`vnext/inventory/summary.py`)

- Schema: `gradience.inventory_summary/v1` — frozen, additive-only versioning
- `InventorySummary` is stable public API (exported from `gradience.__init__`)
- `gradience.api.summarize_inventory()` is the stable Python entry point (direct aggregation, not subprocess)
- `from_dict()` is the single validation gatekeeper — raises `QASchemaError` on contract violations
- Descriptive object, not decision-bearing — only aggregates existing judgments
- Count maps: `adapter_status_counts`, `adapter_flag_counts`, `pair_risk_counts`, `recommended_strategy_counts`, `dominant_issue_counts`
- `strict_qa_block_candidates`: pair reports that would be blocked under `--strict-qa`
- CLI: `gradience summarize-inventory --qa-dir ... --report-dir ... [--emit-report ...] [--strict-input]`
- Malformed input: skip with warning by default, `--strict-input` fails hard
- Canonical example in `examples/inventory/`
- Definition doc: `docs/inventory-summary.md`
```

**Step 7: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md with inventory summary conventions"
```
