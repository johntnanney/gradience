# Pre-GPU CPU-Side Stabilization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the three-tier artifact spine (AdapterQAArtifact → MergeQAReport → InventorySummary) boringly reliable before GPU experiments begin.

**Architecture:** Four priority bands executed sequentially: (1) canonical workflow with real adapter fixture, (2) cross-artifact policy regression tests, (3) CLI scripting reliability, (4) inventory formatting fixes. All work is pure-Python, no GPU, no new features.

**Tech Stack:** Python 3.10+, pytest, bash, ruff, mypy

---

### Task 1: Terminal Formatter Label Fix

The `format_inventory_summary()` function in `gradience/vnext/inventory/summary.py` has a cosmetic bug: source labels render as "Qa artifact" and "Merge report" instead of "QA artifacts" and "Merge reports".

**Files:**
- Modify: `gradience/vnext/inventory/summary.py:243-247`
- Modify: `tests/test_inventory_summary.py` (add label tests)

**Step 1: Write failing tests**

Add these tests to the `TestFormatInventorySummary` class in `tests/test_inventory_summary.py`:

```python
def test_sources_label_qa_artifacts(self) -> None:
    """Source label for QA artifact count should read 'QA artifacts:'."""
    d = _valid_dict()
    summary = InventorySummary.from_dict(d)
    text = format_inventory_summary(summary)
    assert "QA artifacts:" in text

def test_sources_label_merge_reports(self) -> None:
    """Source label for merge report count should read 'Merge reports:'."""
    d = _valid_dict()
    summary = InventorySummary.from_dict(d)
    text = format_inventory_summary(summary)
    assert "Merge reports:" in text

def test_sources_label_not_mangled(self) -> None:
    """Source labels should NOT contain the old mangled forms."""
    d = _valid_dict()
    summary = InventorySummary.from_dict(d)
    text = format_inventory_summary(summary)
    assert "Qa artifact" not in text
    assert "Merge report:" not in text or "Merge reports:" in text
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_inventory_summary.py::TestFormatInventorySummary::test_sources_label_qa_artifacts tests/test_inventory_summary.py::TestFormatInventorySummary::test_sources_label_merge_reports tests/test_inventory_summary.py::TestFormatInventorySummary::test_sources_label_not_mangled -v`
Expected: FAIL (labels are currently mangled)

**Step 3: Fix the formatter**

In `gradience/vnext/inventory/summary.py`, replace lines 243-247 (the SOURCES formatting loop) with:

```python
    _SOURCE_LABELS = {
        "qa_artifact_count": "QA artifacts:",
        "merge_report_count": "Merge reports:",
    }
    for key in sorted(summary.sources):
        label = _SOURCE_LABELS.get(key, key + ":")
        lines.append(f"  {label:<20s}{summary.sources[key]}")
```

Move the `_SOURCE_LABELS` dict to module level (near `_SECTION_DEFS` around line 217) as a private constant:

```python
_SOURCE_LABELS: dict[str, str] = {
    "qa_artifact_count": "QA artifacts:",
    "merge_report_count": "Merge reports:",
}
```

And the loop becomes:

```python
    for key in sorted(summary.sources):
        label = _SOURCE_LABELS.get(key, key + ":")
        lines.append(f"  {label:<20s}{summary.sources[key]}")
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_inventory_summary.py -v`
Expected: ALL PASS

**Step 5: Run lint**

Run: `ruff check gradience/vnext/inventory/summary.py tests/test_inventory_summary.py && ruff format --check gradience/vnext/inventory/summary.py tests/test_inventory_summary.py`
Expected: Clean

**Step 6: Commit**

```bash
git add gradience/vnext/inventory/summary.py tests/test_inventory_summary.py
git commit -m "fix: correct inventory summary source labels (QA artifacts, Merge reports)"
```

---

### Task 2: Strict Reload Invariant Tests

Verify `from_dict(obj.to_dict())` round-trips produce identical objects for all three artifact types.

**Files:**
- Modify: `tests/test_inventory_summary.py`
- Modify: `tests/test_qa_artifact.py`
- Modify: `tests/merge/test_qa_report.py`

**Context:** Existing round-trip tests check individual fields. These tests verify strict equality of the full object after serialization/deserialization.

**Step 1: Write the round-trip tests**

In `tests/test_inventory_summary.py`, add a new class:

```python
class TestStrictReloadInvariant:
    """from_dict(to_dict(obj)) must produce an identical object."""

    def test_roundtrip_identity(self) -> None:
        d = _valid_dict()
        original = InventorySummary.from_dict(d)
        reloaded = InventorySummary.from_dict(original.to_dict())
        assert reloaded == original

    def test_roundtrip_with_empty_counts(self) -> None:
        d = _valid_dict()
        d["adapter_flag_counts"] = {}
        d["pair_risk_counts"] = {}
        original = InventorySummary.from_dict(d)
        reloaded = InventorySummary.from_dict(original.to_dict())
        assert reloaded == original

    def test_roundtrip_with_notes(self) -> None:
        d = _valid_dict()
        d["notes"] = ["note one", "note two"]
        original = InventorySummary.from_dict(d)
        reloaded = InventorySummary.from_dict(original.to_dict())
        assert reloaded == original
```

In `tests/test_qa_artifact.py`, add to the appropriate class or as standalone:

```python
class TestStrictReloadInvariant:
    """from_dict(to_dict(obj)) must produce an identical object."""

    def test_roundtrip_identity_eligible(self) -> None:
        """Load an eligible example, round-trip, verify equality."""
        import json
        from pathlib import Path
        p = Path("examples/qa/eligible_adapter_qa.json")
        with open(p) as f:
            d = json.load(f)
        original = AdapterQAArtifact.from_dict(d)
        reloaded = AdapterQAArtifact.from_dict(original.to_dict())
        assert reloaded == original

    def test_roundtrip_identity_structural_only(self) -> None:
        """Load a structural-only example, round-trip, verify equality."""
        import json
        from pathlib import Path
        p = Path("examples/qa/structural_only_qa.json")
        with open(p) as f:
            d = json.load(f)
        original = AdapterQAArtifact.from_dict(d)
        reloaded = AdapterQAArtifact.from_dict(original.to_dict())
        assert reloaded == original
```

In `tests/merge/test_qa_report.py`, add:

```python
class TestStrictReloadInvariant:
    """from_dict(to_dict(obj)) must produce an identical object."""

    def test_roundtrip_identity_safe(self) -> None:
        import json
        from pathlib import Path
        p = Path("examples/reports/safe_merge_report.json")
        with open(p) as f:
            d = json.load(f)
        original = MergeQAReport.from_dict(d)
        reloaded = MergeQAReport.from_dict(original.to_dict())
        assert reloaded == original

    def test_roundtrip_identity_high_risk(self) -> None:
        import json
        from pathlib import Path
        p = Path("examples/reports/high_risk_warn_report.json")
        with open(p) as f:
            d = json.load(f)
        original = MergeQAReport.from_dict(d)
        reloaded = MergeQAReport.from_dict(original.to_dict())
        assert reloaded == original
```

**Step 2: Run all three test files**

Run: `python3 -m pytest tests/test_inventory_summary.py::TestStrictReloadInvariant tests/test_qa_artifact.py::TestStrictReloadInvariant tests/merge/test_qa_report.py::TestStrictReloadInvariant -v`
Expected: ALL PASS (these are invariants that should already hold)

If any fail, that's a real bug — investigate and fix the serialization before proceeding.

**Step 3: Run lint**

Run: `ruff check tests/test_inventory_summary.py tests/test_qa_artifact.py tests/merge/test_qa_report.py && ruff format --check tests/test_inventory_summary.py tests/test_qa_artifact.py tests/merge/test_qa_report.py`

**Step 4: Commit**

```bash
git add tests/test_inventory_summary.py tests/test_qa_artifact.py tests/merge/test_qa_report.py
git commit -m "test: add strict reload invariant tests for all three artifact types"
```

---

### Task 3: Cross-Artifact Policy Regression Tests

Test the eligibility status flow across the full artifact spine and strategy/action alignment.

**Files:**
- Create: `tests/test_cross_artifact_policy.py`

**Context:**
- `AdapterQAArtifact` has `eligibility.status` (an `EligibilityStatus` enum value).
- `MergeQAReport` has `adapter_a.eligibility_status` and `adapter_b.eligibility_status` (string or `None`).
- `InventorySummary.build_inventory_summary()` counts block candidates where either adapter has status in `{"flagged_weak", "unknown_no_behavioral_eval"}` or `None`.
- Strategy alignment: `pair_risk="low"` → `recommended_strategy="linear"`, `"medium"` → `"norm_equalized"`, `"high"` → `"audit_aware"`.
- Key imports:
  - `from gradience.vnext.audit.qa_artifact import AdapterQAArtifact`
  - `from gradience.vnext.merge.qa_report import MergeQAReport`
  - `from gradience.vnext.inventory.summary import build_inventory_summary, InventorySummary`
  - `from gradience.vnext.merge.eligibility import EligibilityStatus`

**Step 1: Write the test file**

Create `tests/test_cross_artifact_policy.py`:

```python
"""Cross-artifact policy regression tests.

Verifies that eligibility status flows correctly from AdapterQAArtifact
through MergeQAReport to InventorySummary, and that strategy/risk
alignment is consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
from gradience.vnext.inventory.summary import build_inventory_summary
from gradience.vnext.merge.eligibility import EligibilityStatus
from gradience.vnext.merge.qa_report import MergeQAReport


# ---------------------------------------------------------------------------
# Helpers — load canonical examples
# ---------------------------------------------------------------------------

def _load_qa(name: str) -> AdapterQAArtifact:
    """Load a QA artifact from examples/qa/{name}.json."""
    p = Path(f"examples/qa/{name}.json")
    with open(p) as f:
        return AdapterQAArtifact.from_dict(json.load(f))


def _load_report(name: str) -> MergeQAReport:
    """Load a merge report from examples/reports/{name}.json."""
    p = Path(f"examples/reports/{name}.json")
    with open(p) as f:
        return MergeQAReport.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Eligibility flow tests
# ---------------------------------------------------------------------------


class TestEligibilityFlow:
    """Verify eligibility_status propagates from QA → merge → inventory."""

    def test_eligible_adapters_zero_block_candidates(self) -> None:
        """Two eligible adapters produce zero strict-QA block candidates."""
        report = _load_report("safe_merge_report")
        # Both adapters should be eligible in the safe report
        assert report.adapter_a.eligibility_status == "eligible"
        assert report.adapter_b.eligibility_status == "eligible"

        summary = build_inventory_summary([], [report])
        assert summary.strict_qa_block_candidates == 0

    def test_flagged_weak_adapter_counted_as_block_candidate(self) -> None:
        """A report with a flagged_weak adapter is a block candidate."""
        report = _load_report("strict_blocked_report")
        # At least one adapter should be flagged or unknown
        a_elig = report.adapter_a.eligibility_status
        b_elig = report.adapter_b.eligibility_status
        blocked_statuses = {"flagged_weak", "unknown_no_behavioral_eval", None}
        assert a_elig in blocked_statuses or b_elig in blocked_statuses

        summary = build_inventory_summary([], [report])
        assert summary.strict_qa_block_candidates >= 1

    def test_null_eligibility_counted_as_block_candidate(self) -> None:
        """A report with null eligibility_status is a block candidate."""
        # Build a synthetic report with null eligibility
        report = _load_report("safe_merge_report")
        d = report.to_dict()
        d["adapter_a"]["eligibility_status"] = None
        patched = MergeQAReport.from_dict(d)

        summary = build_inventory_summary([], [patched])
        assert summary.strict_qa_block_candidates == 1

    def test_adapter_status_counts_match_qa_artifacts(self) -> None:
        """Inventory adapter_status_counts matches the statuses from QA artifacts."""
        eligible = _load_qa("eligible_adapter_qa")
        structural = _load_qa("structural_only_qa")

        summary = build_inventory_summary([eligible, structural], [])
        # eligible_adapter_qa has status "eligible"
        assert summary.adapter_status_counts.get("eligible", 0) >= 1
        # structural_only_qa has status "unknown_no_behavioral_eval"
        assert summary.adapter_status_counts.get("unknown_no_behavioral_eval", 0) >= 1
        # Total should match artifact count
        total = sum(summary.adapter_status_counts.values())
        assert total == 2


# ---------------------------------------------------------------------------
# Strategy/risk alignment tests
# ---------------------------------------------------------------------------


class TestStrategyAlignment:
    """Verify pair_risk → recommended_strategy alignment in canonical examples."""

    def test_low_risk_maps_to_linear(self) -> None:
        """Low-risk merge reports should recommend 'linear' strategy."""
        report = _load_report("safe_merge_report")
        assert report.pair_risk == "low"
        assert report.recommended_strategy == "linear"

    def test_high_risk_maps_to_audit_aware(self) -> None:
        """High-risk merge reports should recommend 'audit_aware' strategy."""
        report = _load_report("high_risk_warn_report")
        assert report.pair_risk == "high"
        assert report.recommended_strategy == "audit_aware"

    def test_strategy_counts_match_risk_counts_in_inventory(self) -> None:
        """Each merge report contributes exactly one risk and one strategy count."""
        safe = _load_report("safe_merge_report")
        risky = _load_report("high_risk_warn_report")

        summary = build_inventory_summary([], [safe, risky])

        total_risk = sum(summary.pair_risk_counts.values())
        total_strategy = sum(summary.recommended_strategy_counts.values())
        assert total_risk == 2
        assert total_strategy == 2


# ---------------------------------------------------------------------------
# End-to-end spine test
# ---------------------------------------------------------------------------


class TestFullSpine:
    """Load all canonical examples, build inventory, verify consistency."""

    def test_all_examples_produce_valid_inventory(self) -> None:
        """Loading all example QA + reports produces a valid InventorySummary."""
        qa_dir = Path("examples/qa")
        report_dir = Path("examples/reports")

        qa_artifacts = []
        for p in sorted(qa_dir.glob("*.json")):
            with open(p) as f:
                qa_artifacts.append(AdapterQAArtifact.from_dict(json.load(f)))

        merge_reports = []
        for p in sorted(report_dir.glob("*.json")):
            with open(p) as f:
                merge_reports.append(MergeQAReport.from_dict(json.load(f)))

        summary = build_inventory_summary(qa_artifacts, merge_reports)

        # Source counts match inputs
        assert summary.sources["qa_artifact_count"] == len(qa_artifacts)
        assert summary.sources["merge_report_count"] == len(merge_reports)

        # Adapter status counts sum to artifact count
        assert sum(summary.adapter_status_counts.values()) == len(qa_artifacts)

        # Risk counts sum to report count
        assert sum(summary.pair_risk_counts.values()) == len(merge_reports)

        # Strategy counts sum to report count
        assert sum(summary.recommended_strategy_counts.values()) == len(merge_reports)

        # Block candidates <= report count
        assert summary.strict_qa_block_candidates <= len(merge_reports)
```

**Step 2: Run the tests**

Run: `python3 -m pytest tests/test_cross_artifact_policy.py -v`
Expected: ALL PASS

**Step 3: Run lint**

Run: `ruff check tests/test_cross_artifact_policy.py && ruff format --check tests/test_cross_artifact_policy.py`

**Step 4: Commit**

```bash
git add tests/test_cross_artifact_policy.py
git commit -m "test: add cross-artifact policy regression tests for eligibility flow and strategy alignment"
```

---

### Task 4: CLI Exit Code Tests

Test that all three artifact-producing CLI commands return correct exit codes.

**Files:**
- Create: `tests/test_cli_exit_codes.py`

**Context:**
- CLI entrypoint: `gradience.cli:main`
- `audit-adapter` takes `--adapter-dir` and `--out` (writes QA artifact JSON)
- `merge-audit` takes `--adapter-a`, `--adapter-b`, `--emit-report` (writes merge report JSON)
- `summarize-inventory` takes `--qa-dir`, `--report-dir`, `--emit-report`, `--strict-input`
- Real adapter fixture: `examples/adapters/tiny_lora/`
- Pre-built JSON artifacts: `examples/qa/`, `examples/reports/`
- Run CLI via `subprocess.run([sys.executable, "-m", "gradience", ...])`

**Step 1: Write the test file**

Create `tests/test_cli_exit_codes.py`:

```python
"""CLI exit code tests for artifact-producing commands.

Verifies that audit-adapter, merge-audit, and summarize-inventory
return correct exit codes on success and failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _run_gradience(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run 'python -m gradience <args>' and capture output."""
    return subprocess.run(
        [sys.executable, "-m", "gradience", *args],
        capture_output=True,
        text=True,
        check=check,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# summarize-inventory
# ---------------------------------------------------------------------------


class TestSummarizeInventoryExitCodes:
    """Exit code tests for 'gradience summarize-inventory'."""

    def test_success_with_valid_dirs(self) -> None:
        """Exit 0 when given valid QA and report directories."""
        result = _run_gradience(
            "summarize-inventory",
            "--qa-dir", "examples/qa",
            "--report-dir", "examples/reports",
        )
        assert result.returncode == 0

    def test_success_emits_valid_json(self) -> None:
        """--emit-report writes valid JSON and exits 0."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        try:
            result = _run_gradience(
                "summarize-inventory",
                "--qa-dir", "examples/qa",
                "--report-dir", "examples/reports",
                "--emit-report", out_path,
            )
            assert result.returncode == 0
            with open(out_path) as f:
                data = json.load(f)
            assert data["schema"] == "gradience.inventory_summary/v1"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_strict_input_fails_on_malformed(self) -> None:
        """--strict-input causes non-zero exit on malformed JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a malformed QA file
            bad_file = Path(tmpdir) / "bad.json"
            bad_file.write_text('{"schema": "gradience.adapter_qa/v1"}')  # missing required fields
            result = _run_gradience(
                "summarize-inventory",
                "--qa-dir", tmpdir,
                "--strict-input",
            )
            assert result.returncode != 0

    def test_no_args_fails(self) -> None:
        """No arguments at all → non-zero exit."""
        result = _run_gradience("summarize-inventory")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# audit-adapter
# ---------------------------------------------------------------------------


class TestAuditAdapterExitCodes:
    """Exit code tests for 'gradience audit-adapter'."""

    def test_success_with_real_adapter(self) -> None:
        """Exit 0 when auditing a real PEFT adapter directory."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        try:
            result = _run_gradience(
                "audit-adapter",
                "--adapter-dir", "examples/adapters/tiny_lora",
                "--out", out_path,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            with open(out_path) as f:
                data = json.load(f)
            assert data["schema"] == "gradience.adapter_qa/v1"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_nonexistent_adapter_fails(self) -> None:
        """Non-existent adapter directory → non-zero exit."""
        result = _run_gradience(
            "audit-adapter",
            "--adapter-dir", "/nonexistent/path/to/adapter",
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# merge-audit
# ---------------------------------------------------------------------------


class TestMergeAuditExitCodes:
    """Exit code tests for 'gradience merge-audit'."""

    def test_success_with_real_adapters(self) -> None:
        """Exit 0 when merging two copies of the same adapter."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        try:
            result = _run_gradience(
                "merge-audit",
                "--adapter-a", "examples/adapters/tiny_lora",
                "--adapter-b", "examples/adapters/tiny_lora",
                "--emit-report", out_path,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            with open(out_path) as f:
                data = json.load(f)
            assert data["schema"] == "gradience.merge_qa_report/v1"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_nonexistent_adapter_fails(self) -> None:
        """Non-existent adapter directory → non-zero exit."""
        result = _run_gradience(
            "merge-audit",
            "--adapter-a", "/nonexistent/path",
            "--adapter-b", "examples/adapters/tiny_lora",
        )
        assert result.returncode != 0
```

**Step 2: Run the tests**

Run: `python3 -m pytest tests/test_cli_exit_codes.py -v --timeout=120`
Expected: ALL PASS

Note: `audit-adapter` and `merge-audit` tests load PEFT weights and may take 10-20 seconds each. If any test times out, increase the `timeout` parameter in `_run_gradience`.

**Step 3: Run lint**

Run: `ruff check tests/test_cli_exit_codes.py && ruff format --check tests/test_cli_exit_codes.py`

**Step 4: Commit**

```bash
git add tests/test_cli_exit_codes.py
git commit -m "test: add CLI exit code tests for audit-adapter, merge-audit, summarize-inventory"
```

---

### Task 5: CLI Overwrite Behavior Documentation

Document that `--emit-report`, `--emit-artifact`, and `--out` silently overwrite existing files.

**Files:**
- Modify: `gradience/cli.py` (help text for `--out`, `--emit-report`)

**Step 1: Update help text**

Find the `--out` argument for `audit-adapter` (around line 3110-3113 in `gradience/cli.py`). Change:

```python
help="Write QA artifact JSON to this path",
```

to:

```python
help="Write QA artifact JSON to this path (overwrites existing file)",
```

Find the `--emit-report` argument for `merge-audit` (around line 3205-3208). Change:

```python
help="Write structured JSON report to this path (e.g. report.json)",
```

to:

```python
help="Write structured JSON report to this path (overwrites existing file)",
```

Find the `--emit-report` argument for `summarize-inventory` (around line 3366). Change:

```python
help="Write inventory summary v1 JSON to this path",
```

to:

```python
help="Write inventory summary v1 JSON to this path (overwrites existing file)",
```

**Step 2: Verify help text renders correctly**

Run: `python3 -m gradience audit-adapter --help | grep overwrites`
Run: `python3 -m gradience merge-audit --help | grep overwrites`
Run: `python3 -m gradience summarize-inventory --help | grep overwrites`
Expected: Each command's help shows "(overwrites existing file)"

**Step 3: Run lint**

Run: `ruff check gradience/cli.py && ruff format --check gradience/cli.py`

**Step 4: Commit**

```bash
git add gradience/cli.py
git commit -m "docs: document overwrite behavior in CLI --emit-report/--out help text"
```

---

### Task 6: Realistic Inventory Example

Create a realistic inventory summary example that looks like a real 10-adapter inventory.

**Files:**
- Create: `examples/inventory/realistic_inventory_summary.json`
- Modify: `tests/test_inventory_summary.py` (add to example file smoke test)

**Step 1: Create the example file**

Create `examples/inventory/realistic_inventory_summary.json`:

```json
{
  "schema": "gradience.inventory_summary/v1",
  "sources": {
    "qa_artifact_count": 10,
    "merge_report_count": 8
  },
  "adapter_status_counts": {
    "eligible": 6,
    "uncertain": 2,
    "flagged_weak": 1,
    "unknown_no_behavioral_eval": 1
  },
  "adapter_flag_counts": {
    "low_utilization": 4,
    "high_rank_waste": 3,
    "concentrated_spectrum": 2,
    "underutilized_capacity": 1
  },
  "pair_risk_counts": {
    "low": 4,
    "medium": 3,
    "high": 1
  },
  "recommended_strategy_counts": {
    "linear": 4,
    "norm_equalized": 3,
    "audit_aware": 1
  },
  "dominant_issue_counts": {
    "none": 4,
    "norm_imbalance": 2,
    "subspace_conflict": 1,
    "partial_redundancy": 1
  },
  "strict_qa_block_candidates": 3,
  "notes": [
    "2 adapters pending behavioral evaluation",
    "1 adapter flagged for retraining at lower rank"
  ]
}
```

**Step 2: Verify it loads**

Run: `python3 -c "import json; from gradience.vnext.inventory.summary import InventorySummary; d = json.load(open('examples/inventory/realistic_inventory_summary.json')); s = InventorySummary.from_dict(d); print('OK:', s.sources)"`
Expected: `OK: {'qa_artifact_count': 10, 'merge_report_count': 8}`

**Step 3: Check that the parametrized smoke test in `tests/test_inventory_summary.py` picks it up**

The existing `TestExampleFiles` class uses `glob.glob("examples/inventory/*.json")` to find example files. Verify by running:

Run: `python3 -m pytest tests/test_inventory_summary.py::TestExampleFiles -v`
Expected: PASS for both `inventory_summary.json` and `realistic_inventory_summary.json`

**Step 4: Run lint**

Run: `ruff check tests/test_inventory_summary.py && ruff format --check tests/test_inventory_summary.py`

**Step 5: Commit**

```bash
git add examples/inventory/realistic_inventory_summary.json
git commit -m "docs: add realistic 10-adapter inventory summary example"
```

---

### Task 7: Preflight Policy Document

Document the cross-artifact contracts and consistency rules.

**Files:**
- Create: `docs/preflight-policy.md`

**Step 1: Write the document**

Create `docs/preflight-policy.md`:

```markdown
# Preflight Policy: Cross-Artifact Contracts

This document defines the consistency contracts between Gradience's three artifact types. These contracts are tested by `tests/test_cross_artifact_policy.py`.

## Artifact Spine

```
AdapterQAArtifact → MergeQAReport → InventorySummary
   (per-adapter)      (per-pair)      (inventory-level)
```

## Eligibility Status Flow

The `EligibilityStatus` enum has four values:

| Status | Meaning |
|--------|---------|
| `eligible` | Adapter outperforms base model on target task |
| `uncertain` | Evidence exists but is inconclusive |
| `flagged_weak` | Adapter appears weaker than base model |
| `unknown_no_behavioral_eval` | No behavioral evaluation provided |

### How status propagates

1. **QA artifact** records the status in `eligibility.status`.
2. **Merge report** copies each adapter's status into `adapter_a.eligibility_status` / `adapter_b.eligibility_status`. If no QA artifact was provided, the value is `null`.
3. **Inventory summary** counts statuses in `adapter_status_counts` (from QA artifacts) and identifies `strict_qa_block_candidates` (from merge reports).

## Strict-QA Blocking

The `--strict-qa` flag (on `merge-audit`) and the `strict_qa_block_candidates` count (in inventory summaries) use the same blocking rule:

A pair is blocked if **either** adapter has:
- `eligibility_status == "flagged_weak"`
- `eligibility_status == "unknown_no_behavioral_eval"`
- `eligibility_status` is `null` (no QA provided)

## Strategy/Risk Alignment

Merge reports map `pair_risk` to `recommended_strategy`:

| Risk Level | Strategy | Meaning |
|------------|----------|---------|
| `low` | `linear` | Safe to merge with simple linear combination |
| `medium` | `norm_equalized` | Merge with norm equalization to handle imbalance |
| `high` | `audit_aware` | Requires careful audit-guided merge or manual review |

The `recommended_action` field is explanatory prose and does not override `recommended_strategy`.

## Inventory Aggregation Rules

- `adapter_status_counts` sums to the number of QA artifacts
- `pair_risk_counts` sums to the number of merge reports
- `recommended_strategy_counts` sums to the number of merge reports
- `strict_qa_block_candidates` is at most the number of merge reports
- Count maps only include keys with non-zero values
```

**Step 2: Commit**

```bash
git add docs/preflight-policy.md
git commit -m "docs: add preflight policy document for cross-artifact contracts"
```

---

### Task 8: Getting Started Preflight Guide

Write the user-facing guide that walks through the full artifact spine.

**Files:**
- Create: `docs/getting-started-preflight.md`

**Step 1: Write the document**

Create `docs/getting-started-preflight.md`:

```markdown
# Getting Started: Preflight Check

This guide walks through the complete Gradience artifact pipeline using the bundled example adapter. By the end, you'll have produced all three artifact types and verified they're consistent.

## Prerequisites

```bash
pip install gradience[hf]
```

## Step 1: Audit a Single Adapter

Produce an `AdapterQAArtifact` from the bundled example adapter:

```bash
gradience audit-adapter \
  --adapter-dir examples/adapters/tiny_lora \
  --out /tmp/gradience_preflight/qa_artifact.json
```

Expected: exit code 0, JSON file written with `"schema": "gradience.adapter_qa/v1"`.

Since this adapter has no behavioral evaluation, the artifact will have `eligibility.status = "unknown_no_behavioral_eval"`.

## Step 2: Run a Merge Audit

Produce a `MergeQAReport` comparing two adapters. For this demo, we compare the adapter against itself:

```bash
gradience merge-audit \
  --adapter-a examples/adapters/tiny_lora \
  --adapter-b examples/adapters/tiny_lora \
  --emit-report /tmp/gradience_preflight/merge_report.json
```

Expected: exit code 0, JSON file written with `"schema": "gradience.merge_qa_report/v1"`.

## Step 3: Summarize the Inventory

Produce an `InventorySummary` from the artifacts:

```bash
gradience summarize-inventory \
  --qa-dir /tmp/gradience_preflight \
  --report-dir /tmp/gradience_preflight \
  --emit-report /tmp/gradience_preflight/inventory_summary.json
```

Expected: exit code 0, terminal summary printed, JSON file written with `"schema": "gradience.inventory_summary/v1"`.

## Step 4: Verify

Check that all three files were created and contain valid JSON:

```bash
for f in /tmp/gradience_preflight/*.json; do
  echo "=== $(basename "$f") ==="
  python3 -c "import json; d=json.load(open('$f')); print(d.get('schema', 'NO SCHEMA'))"
done
```

Expected output:
```
=== qa_artifact.json ===
gradience.adapter_qa/v1
=== merge_report.json ===
gradience.merge_qa_report/v1
=== inventory_summary.json ===
gradience.inventory_summary/v1
```

## Strict-QA Blocking Example

The `--strict-qa` flag on `merge-audit` blocks merges when either adapter lacks behavioral evaluation. Since our example adapter has no eval:

```bash
gradience merge-audit \
  --adapter-a examples/adapters/tiny_lora \
  --adapter-b examples/adapters/tiny_lora \
  --strict-qa \
  --emit-report /tmp/gradience_preflight/strict_report.json
```

Expected: non-zero exit code, error message about adapter eligibility.

## Cleanup

```bash
rm -rf /tmp/gradience_preflight
```

## Python API

The same workflow is available programmatically:

```python
from gradience.api import audit_adapter, merge_risk_report, summarize_inventory

# Step 1: Audit
qa = audit_adapter(peft_dir="examples/adapters/tiny_lora")

# Step 2: Merge audit (delegates to CLI subprocess)
report = merge_risk_report(
    adapter_a="examples/adapters/tiny_lora",
    adapter_b="examples/adapters/tiny_lora",
)

# Step 3: Summarize (direct aggregation from JSON files)
summary = summarize_inventory(qa_dir="examples/qa", report_dir="examples/reports")
```

## Next Steps

- See `docs/adapter-qa-artifact.md` for the adapter QA schema contract
- See `docs/merge-risk-report.md` for the merge report schema contract
- See `docs/inventory-summary.md` for the inventory summary schema contract
- See `docs/preflight-policy.md` for cross-artifact consistency rules
```

**Step 2: Commit**

```bash
git add docs/getting-started-preflight.md
git commit -m "docs: add getting-started preflight guide for full artifact pipeline"
```

---

### Task 9: Workflow Smoke Script

Create the automated smoke script that exercises the full pipeline.

**Files:**
- Create: `scripts/preflight_smoke.sh`

**Step 1: Write the script**

Create `scripts/preflight_smoke.sh`:

```bash
#!/usr/bin/env bash
# preflight_smoke.sh — End-to-end smoke test for the Gradience artifact spine.
#
# Exercises: audit-adapter → merge-audit → summarize-inventory
# Uses the bundled examples/adapters/tiny_lora fixture.
#
# Exit 0 = all green, non-zero = something broke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

echo "=== Gradience Preflight Smoke Test ==="
echo "Working directory: $WORK_DIR"
echo ""

ADAPTER_DIR="$REPO_ROOT/examples/adapters/tiny_lora"
QA_DIR="$WORK_DIR/qa"
REPORT_DIR="$WORK_DIR/reports"
INVENTORY_DIR="$WORK_DIR/inventory"
mkdir -p "$QA_DIR" "$REPORT_DIR" "$INVENTORY_DIR"

# --- Step 1: audit-adapter ---
echo "Step 1: audit-adapter"
python3 -m gradience audit-adapter \
    --adapter-dir "$ADAPTER_DIR" \
    --out "$QA_DIR/tiny_lora_qa.json"
echo "  -> OK"
echo ""

# --- Step 2: merge-audit ---
echo "Step 2: merge-audit"
python3 -m gradience merge-audit \
    --adapter-a "$ADAPTER_DIR" \
    --adapter-b "$ADAPTER_DIR" \
    --emit-report "$REPORT_DIR/self_merge_report.json"
echo "  -> OK"
echo ""

# --- Step 3: summarize-inventory ---
echo "Step 3: summarize-inventory"
python3 -m gradience summarize-inventory \
    --qa-dir "$QA_DIR" \
    --report-dir "$REPORT_DIR" \
    --emit-report "$INVENTORY_DIR/inventory_summary.json"
echo "  -> OK"
echo ""

# --- Step 4: Validate outputs ---
echo "Step 4: Validate JSON schemas"

validate_schema() {
    local file="$1"
    local expected_schema="$2"
    local actual
    actual=$(python3 -c "import json; print(json.load(open('$file'))['schema'])")
    if [ "$actual" != "$expected_schema" ]; then
        echo "  FAIL: $file — expected schema '$expected_schema', got '$actual'"
        exit 1
    fi
    echo "  $file -> $expected_schema OK"
}

validate_schema "$QA_DIR/tiny_lora_qa.json" "gradience.adapter_qa/v1"
validate_schema "$REPORT_DIR/self_merge_report.json" "gradience.merge_qa_report/v1"
validate_schema "$INVENTORY_DIR/inventory_summary.json" "gradience.inventory_summary/v1"
echo ""

echo "=== All preflight checks passed ==="
```

**Step 2: Make it executable**

Run: `chmod +x scripts/preflight_smoke.sh`

**Step 3: Test it**

Run: `bash scripts/preflight_smoke.sh`
Expected: All steps pass, ends with "All preflight checks passed"

**Step 4: Commit**

```bash
git add scripts/preflight_smoke.sh
git commit -m "feat: add preflight smoke script for end-to-end artifact spine validation"
```

---

### Task 10: Full Test Suite Verification

Run the complete test suite and lint checks to confirm nothing is broken.

**Step 1: Run lint**

Run: `ruff check . && ruff format --check .`
Expected: Clean

**Step 2: Run mypy**

Run: `mypy gradience/`
Expected: Clean

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -x -q --timeout=120`
Expected: All tests pass (1011+ existing tests plus new ones)

**Step 4: Verify test count increased**

The new tests should add approximately:
- 3 formatter label tests (Task 1)
- 6 strict reload tests (Task 2)
- 9 cross-artifact policy tests (Task 3)
- 7 CLI exit code tests (Task 4)

Total: ~25 new tests. Final count should be 1035+.

If any tests fail, investigate and fix before committing.
