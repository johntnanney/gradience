# CLI Refactoring Plan

**Date:** April 5, 2026
**Status:** Planned
**Scope:** `gradience/cli.py` (3,874 lines → modular package)
**Estimated effort:** 4–6 sessions, each self-contained and independently shippable

---

## The Problem

`cli.py` is the largest file in the project at 3,874 lines. It contains 59 functions that handle four distinct responsibilities: argument parsing, input validation, business logic orchestration, and terminal output formatting. These responsibilities are tangled together inside 16 command handlers, each of which catches its own exceptions and calls `sys.exit(1)` directly — 80 times total across the file.

This creates three concrete problems.

**Testability.** Command handlers cannot be tested from Python because they call `sys.exit()` instead of returning or raising. All 54+ CLI tests must spawn subprocesses, which is slow and limits what can be asserted. No command handler has a direct unit test.

**Reusability.** A user who wants to call `cmd_merge_audit` from a script gets a system exit instead of a return value or exception. The `api.py` module exists partly to work around this, but it delegates to subprocess calls rather than directly invoking command logic.

**Maintainability.** At 3,874 lines, the file is difficult to navigate. Adding a new command requires understanding the full file's conventions. Output formatting (867 lines, 22% of the file) is interleaved with logic rather than separated.

## Design Principles

The refactoring follows three rules:

1. **Every phase is independently shippable.** No phase depends on a later phase being completed. After each phase, all existing tests pass, the CLI behaves identically, and the codebase is in a better state than before.

2. **Existing integration tests are the safety net.** The 54+ subprocess-based tests verify end-to-end behavior. We don't need to write new tests before refactoring — the existing tests catch regressions. New unit tests are added *after* each phase makes them possible.

3. **No behavior changes.** The user-visible CLI (arguments, output, exit codes) is unchanged throughout. This is a structural refactoring, not a feature change.

---

## Phase 1: Exception-Based Error Handling in main()

**Effort:** 1 session (~2 hours)
**Risk:** Low
**Prerequisite:** None

This is the critical enabling change. Everything else builds on it.

### What changes

Wrap `args.func(args)` in a try/except that catches `GradienceError` and `SystemExit`, prints the error message, and exits with code 1. This makes `main()` the single exit point:

```python
def main() -> None:
    parser = ...
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except GradienceError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
```

Then, one command handler at a time, replace `sys.exit(1)` calls with `raise`. Start with the simplest handler (`cmd_verify`, 12 lines) and work outward. Each handler can be migrated independently — partially migrated handlers still work because the old `sys.exit(1)` calls still function; they're just bypassing the new catch.

### Migration pattern for each handler

Before:
```python
def cmd_check(args):
    try:
        config = _load_config_file(args.config)
    except (OSError, ValueError) as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
```

After:
```python
def cmd_check(args):
    try:
        config = _load_config_file(args.config)
    except (OSError, ValueError) as e:
        raise ConfigError(f"Error loading config: {e}") from e
```

### Priority order for handler migration

Migrate handlers in this order (simplest first, most error-prone last):

1. `cmd_verify` (12 lines, 0 exits) — trivial, proves the pattern works
2. `cmd_report` (90 lines, 3 exits) — simple file loading
3. `cmd_truncate` (84 lines, 4 exits) — self-contained SVD operation
4. `cmd_audit_adapter` (72 lines, 3 exits) — clean QA production
5. `cmd_monitor` (139 lines, 5 exits) — telemetry parsing
6. `cmd_audit` (170 lines, 4 exits) — structural audit
7. `cmd_check` (130 lines, 10 exits) — most validation logic
8. `cmd_merge_plan`, `cmd_merge`, `cmd_explain` (77+75+79 lines, 19 exits combined) — merge family
9. `cmd_merge_audit` (237 lines, 9 exits) — longest handler
10. `cmd_summarize_inventory`, `cmd_suggest_neighborhoods`, `cmd_batch_summary`, `cmd_portfolio`, `cmd_preflight_report` — inventory family

### What this enables

Once handlers raise instead of exiting, they become directly callable from Python:

```python
from gradience.cli import cmd_audit
try:
    cmd_audit(args)
except AuditError as e:
    # handle programmatically
```

And unit-testable without subprocesses:

```python
def test_check_missing_config():
    args = make_args(config="/nonexistent")
    with pytest.raises(ConfigError, match="not found"):
        cmd_check(args)
```

### Verification

All 54+ existing subprocess tests pass unchanged (they test exit codes and stdout, which are preserved). After each handler migration, run `make test-quick`.

---

## Phase 2: Extract Output Formatting

**Effort:** 1 session (~2 hours)
**Risk:** Low
**Prerequisite:** None (can be done before, after, or in parallel with Phase 1)

### What changes

Create `gradience/cli_format.py` containing all output formatting functions. These are the 9 `_print_*` and `_display_*` functions (867 lines total) that take structured data and produce terminal output.

The key insight: several of these functions are "mixed" — they compute summaries *and* print them. The extraction separates these into two steps:

```
# Before (in cli.py):
def _print_monitor_result(config, result, args):
    # 221 lines of compute + print interleaved

# After:
# In cli_format.py:
def format_monitor_result(config, result, verbose=False) -> str:
    # Returns formatted string

# In cli.py:
def _print_monitor_result(config, result, args):
    print(format_monitor_result(config, result, verbose=args.verbose))
```

### Functions to extract

| Function | Lines | Type | Notes |
|----------|-------|------|-------|
| `_print_monitor_result` | 221 | Mixed | Largest; computes alert summaries then prints |
| `_print_policy_disagreement_summary` | 162 | Mixed | Uses policy_analysis module for compute |
| `_print_audit_summary` | 165 | Mixed | Computes layer stats then formats |
| `_print_recommendations` | ~80 | Pure output | Takes recommendation data, formats table |
| `_print_qa_summary` | ~60 | Pure output | QA artifact formatting |
| `_display_audit_adapter_summary` | ~50 | Pure output | Adapter audit display |
| `_fmt` | 8 | Pure utility | Number formatting |
| `_fmt_params` | 12 | Pure utility | Parameter count formatting |
| `_severity_rank` | 10 | Pure utility | Severity ordering |

### What this enables

Output formatting becomes testable without subprocess:

```python
def test_monitor_output_includes_alerts():
    result = make_monitor_result(alerts=["loss_plateau"])
    output = format_monitor_result(config, result)
    assert "Loss plateau" in output
```

And alternative output formats (JSON, HTML, structured logging) can reuse the same compute path with a different formatter.

### Verification

All existing tests pass unchanged — the functions are called from the same places, just defined in a different file.

---

## Phase 3: Extract Command Handlers into a Package

**Effort:** 2 sessions (~3-4 hours)
**Risk:** Medium (more files to coordinate)
**Prerequisite:** Phase 1 (handlers should raise, not exit)

### What changes

Create `gradience/commands/` package:

```
gradience/commands/
├── __init__.py          # re-exports all cmd_* functions
├── audit.py             # cmd_audit, cmd_audit_adapter
├── check.py             # cmd_check
├── merge.py             # cmd_merge_audit, cmd_merge_plan, cmd_merge, cmd_explain
├── monitor.py           # cmd_monitor, cmd_report
├── inventory.py         # cmd_summarize_inventory, cmd_suggest_neighborhoods,
│                        # cmd_batch_summary, cmd_portfolio, cmd_preflight_report
├── truncate.py          # cmd_truncate
└── verify.py            # cmd_verify
```

Each module contains:
- The `cmd_*` function(s) for that command group
- The `_setup_*_command` function(s) that wire argparse
- Any helper functions used *only* by that command group

### Shared helpers stay in cli.py (temporarily) or move to cli_utils.py

These functions are used across multiple commands and should be extracted to `gradience/cli_utils.py`:

- Config pipeline: `_load_config_file`, `_autodetect_file_in_dir`, `_normalize_to_vnext_dict`, `_blank_vnext_dict`, `_merge_fill_missing`, `_apply_overrides` (6 functions, ~320 lines)
- QA loading: `_load_source_qa` (~80 lines)
- Analysis: `_analyze_policy_disagreements` (202 lines) — already partially extracted to `policy_analysis.py`; the CLI version adds JSON-specific rationale formatting

### What cli.py becomes

After Phase 3, `cli.py` is reduced to ~100 lines: imports, argparse setup, the `main()` dispatcher, and nothing else.

```python
"""Gradience CLI — argument parsing and dispatch."""

import argparse
import sys

from gradience.exceptions import GradienceError
from gradience.commands import (
    setup_audit_commands, setup_check_commands, setup_merge_commands,
    setup_monitor_commands, setup_inventory_commands, setup_truncate_commands,
    setup_verify_commands,
)


def main() -> None:
    parser = argparse.ArgumentParser(...)
    subparsers = parser.add_subparsers(dest="command")

    setup_audit_commands(subparsers)
    setup_check_commands(subparsers)
    setup_merge_commands(subparsers)
    setup_monitor_commands(subparsers)
    setup_inventory_commands(subparsers)
    setup_truncate_commands(subparsers)
    setup_verify_commands(subparsers)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except GradienceError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
```

### Migration strategy

Move one command group at a time. After each move: run `make test-quick`. The `__init__.py` re-exports ensure nothing breaks for any external code that might import from `gradience.cli`.

Recommended order:
1. `verify.py` (smallest, proves the pattern)
2. `truncate.py` (self-contained)
3. `check.py` (heavy validation, good test of shared helpers)
4. `monitor.py` + `report` (telemetry family)
5. `audit.py` (adapter auditing)
6. `merge.py` (largest group, most cross-dependencies)
7. `inventory.py` (largest group by command count)

### Verification

All existing subprocess tests pass unchanged. Import-based tests (like `test_qa_artifact.py` importing `_load_source_qa`) need their import paths updated — or `cli.py` can re-export for backward compatibility.

---

## Phase 4: Add Unit Tests for Command Handlers

**Effort:** 1–2 sessions (~2-3 hours)
**Risk:** Low
**Prerequisite:** Phase 1 (handlers must raise, not exit)

### What changes

Add direct unit tests for each command handler. These complement the existing integration tests — integration tests verify end-to-end behavior, unit tests verify specific error paths and edge cases.

### Test structure

```
tests/
├── commands/
│   ├── test_audit.py
│   ├── test_check.py
│   ├── test_merge.py
│   ├── test_monitor.py
│   ├── test_inventory.py
│   └── test_truncate.py
```

### Test patterns

Create a lightweight `make_args` helper that builds argparse.Namespace objects without running the parser:

```python
def make_args(**kwargs):
    """Build a minimal args Namespace for testing command handlers directly."""
    defaults = {"verbose": False, "json": False, "output": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)
```

Then test handlers directly:

```python
def test_check_valid_config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(make_config()))
    args = make_args(config=str(config_file), json=True)
    # Returns normally (no exception = success)
    cmd_check(args)

def test_check_missing_file():
    args = make_args(config="/nonexistent/config.json")
    with pytest.raises(ConfigError, match="not found"):
        cmd_check(args)

def test_merge_audit_incompatible_ranks(tmp_path, orthogonal_pair):
    a, b = orthogonal_pair
    args = make_args(adapter_a=str(a), adapter_b=str(b), ...)
    result = cmd_merge_audit(args)  # or capture structured output
    assert result.pair_risk == "safe"
```

### Priority

Focus on error paths that are currently hard to test via subprocess: specific exception types, edge cases in config validation, boundary conditions in QA gating logic. The integration tests already cover the happy paths well.

---

## Phase Summary

| Phase | Effort | Risk | Enables | Independently Shippable |
|-------|--------|------|---------|------------------------|
| 1. Exception handling | 2 hrs | Low | Unit testing, programmatic use | Yes |
| 2. Extract formatting | 2 hrs | Low | Output testing, alternative formats | Yes |
| 3. Extract commands | 3-4 hrs | Medium | Navigability, per-command ownership | Yes |
| 4. Add unit tests | 2-3 hrs | Low | Coverage of error paths | Yes |

Phases 1 and 2 can be done in either order or in parallel. Phase 3 builds on Phase 1. Phase 4 builds on Phase 1 but can start as soon as the first few handlers are migrated.

The minimum viable improvement is Phase 1 alone — it solves the testability problem and takes one session.

---

## What This Does Not Change

- The user-facing CLI (arguments, subcommands, output, exit codes)
- The `api.py` stable API surface
- The vnext module architecture
- Any algorithmic behavior
- The pre-commit, CI, or release workflows

## Risks and Mitigations

**Risk: Import path breakage.** Any external code importing from `gradience.cli` (e.g., `from gradience.cli import _load_source_qa` in tests) will break when functions move. **Mitigation:** Add backward-compatible re-exports in `cli.py`, with deprecation warnings. One test file (`test_qa_artifact.py`) is known to import `_load_source_qa` directly.

**Risk: Subtle behavior changes from exception propagation.** A `sys.exit(1)` terminates immediately; a raised exception unwinds the stack, which could trigger `finally` blocks or context manager `__exit__` methods. **Mitigation:** Review each handler for context managers before migrating. Most handlers don't use them.

**Risk: Merge conflicts if other work touches cli.py concurrently.** **Mitigation:** Do Phase 1 first (small, surgical changes within the existing file). Only Phase 3 creates new files and large diffs.
