# Pipeline Integration Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an integration test that wires real adapter fixtures through the full merge pipeline: audit → diagnose → recommend → format.

**Architecture:** Single test file with one class, four methods (one per fixture type). Each method runs all four pipeline stages and asserts structural properties at each boundary. Uses existing conftest fixtures — no new fixture creation.

**Tech Stack:** pytest, existing `tests/merge/conftest.py` fixtures

---

### Task 1: Write the test file with all four pipeline tests

**Files:**
- Create: `tests/merge/test_pipeline_integration.py`

**Step 1: Write the complete test file**

```python
"""Integration test: audit → diagnose → recommend → format pipeline.

Verifies that real adapter fixtures flow through all four stages
with structurally valid outputs at each boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gradience.vnext.merge import (
    diagnose_pair,
    format_recommendation,
    merge_audit,
    recommend_merge,
)


VALID_VERDICTS = {"safe", "redundant", "conflicting", "imbalanced"}
VALID_RISKS = {"low", "medium", "high"}
VALID_STRATEGIES = {"linear", "ties", "dare_ties", "dare_linear"}
LAYER_VERDICT_REQUIRED_KEYS = {"layer_name", "verdict", "confidence", "metrics"}


def _assert_report(report, *, expected_n_layers: int = 2):
    """Structural assertions on MergeAuditReport."""
    assert len(report.layer_verdicts) == expected_n_layers
    for lv in report.layer_verdicts:
        assert LAYER_VERDICT_REQUIRED_KEYS <= set(lv.keys())
        assert lv["verdict"] in VALID_VERDICTS
        assert isinstance(lv["confidence"], float)
    assert report.aggregate.overall_verdict in VALID_VERDICTS
    score = report.aggregate.compatibility_score
    assert isinstance(score, float) and 0.0 <= score <= 1.0


def _assert_diagnosis(diag, report):
    """Structural assertions on PairDiagnosis."""
    assert len(diag.layer_diagnoses) == len(report.layer_verdicts)
    for ld in diag.layer_diagnoses:
        assert ld.verdict in VALID_VERDICTS
        assert ld.risk_level in VALID_RISKS
        assert isinstance(ld.compress_first, bool)
    assert diag.overall_risk in VALID_RISKS
    assert isinstance(diag.compression_needed, bool)


def _assert_recommendation(rec, report, *, expected_strategy: str):
    """Structural assertions on MergeRecommendation."""
    assert len(rec.layer_recommendations) == len(report.layer_verdicts)
    for lr in rec.layer_recommendations:
        assert lr.strategy in VALID_STRATEGIES
        assert lr.strategy == expected_strategy
        assert isinstance(lr.coefficients, tuple) and len(lr.coefficients) == 2
        assert lr.risk_level in VALID_RISKS
        assert isinstance(lr.reasoning, str) and len(lr.reasoning) > 0
    assert rec.overall_strategy == "audit_aware"
    assert rec.overall_risk in VALID_RISKS
    assert isinstance(rec.warnings, tuple)


def _assert_formatted(text, rec):
    """Structural assertions on formatted output string."""
    assert isinstance(text, str) and len(text) > 0
    assert "MERGE STRATEGY RECOMMENDATION" in text
    # At least one shortened layer name appears
    assert "L0." in text


class TestMergeRecommendPipeline:
    """End-to-end pipeline: merge_audit → diagnose → recommend → format."""

    def test_orthogonal_safe_pipeline(self, orthogonal_pair: tuple[Path, Path]):
        dir_a, dir_b = orthogonal_pair
        report = merge_audit(str(dir_a), str(dir_b))
        _assert_report(report)
        assert report.aggregate.overall_verdict == "safe"

        diag = diagnose_pair(report)
        _assert_diagnosis(diag, report)
        assert all(ld.risk_level == "low" for ld in diag.layer_diagnoses)

        rec = recommend_merge(report)
        _assert_recommendation(rec, report, expected_strategy="linear")

        text = format_recommendation(rec)
        _assert_formatted(text, rec)

    def test_redundant_pipeline(self, redundant_pair: tuple[Path, Path]):
        dir_a, dir_b = redundant_pair
        report = merge_audit(str(dir_a), str(dir_b))
        _assert_report(report)
        assert report.aggregate.overall_verdict == "redundant"

        diag = diagnose_pair(report)
        _assert_diagnosis(diag, report)
        assert all(ld.risk_level == "medium" for ld in diag.layer_diagnoses)

        rec = recommend_merge(report)
        _assert_recommendation(rec, report, expected_strategy="ties")

        text = format_recommendation(rec)
        _assert_formatted(text, rec)

    def test_conflicting_pipeline(self, conflicting_pair: tuple[Path, Path]):
        dir_a, dir_b = conflicting_pair
        report = merge_audit(str(dir_a), str(dir_b))
        _assert_report(report)
        assert report.aggregate.overall_verdict == "conflicting"

        diag = diagnose_pair(report)
        _assert_diagnosis(diag, report)
        assert all(ld.risk_level == "high" for ld in diag.layer_diagnoses)

        rec = recommend_merge(report)
        _assert_recommendation(rec, report, expected_strategy="dare_ties")

        text = format_recommendation(rec)
        _assert_formatted(text, rec)

    def test_imbalanced_pipeline(self, imbalanced_pair: tuple[Path, Path]):
        dir_a, dir_b = imbalanced_pair
        report = merge_audit(str(dir_a), str(dir_b))
        _assert_report(report)
        assert report.aggregate.overall_verdict == "imbalanced"

        diag = diagnose_pair(report)
        _assert_diagnosis(diag, report)

        rec = recommend_merge(report)
        _assert_recommendation(rec, report, expected_strategy="linear")
        # Imbalanced should have rebalanced (non-equal) coefficients
        for lr in rec.layer_recommendations:
            a, b = lr.coefficients
            assert abs(a - b) > 0.01, "imbalanced should rebalance coefficients"

        text = format_recommendation(rec)
        _assert_formatted(text, rec)
```

**Step 2: Run tests to verify they pass**

Run: `python3 -m pytest tests/merge/test_pipeline_integration.py -v`
Expected: all 4 tests PASS

**Step 3: Run lint on new file**

Run: `ruff check tests/merge/test_pipeline_integration.py`
Expected: no errors

**Step 4: Run full test suite to check for regressions**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: 1015 passed (1011 existing + 4 new)

**Step 5: Commit**

```bash
git add tests/merge/test_pipeline_integration.py
git commit -m "Add end-to-end pipeline integration test for merge recommend workflow"
```
