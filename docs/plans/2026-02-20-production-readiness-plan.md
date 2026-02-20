# Production Readiness Plan

**Date:** 2026-02-20
**Goal:** Bring Gradience to publication-quality before PyPI release and public dissemination.

## Current State

Gradience 0.10.0 is structurally sound: modern pyproject.toml packaging, 6 CI workflows, 697 tests, typed (py.typed marker), linted (ruff), pre-commit hooks, Apache 2.0 license, comprehensive documentation (32 docs), and a Docker build. The project exceeds most open-source standards.

However, a full audit reveals **293 mypy errors across 36 files**, no exception hierarchy, zero `logging` usage, and significant CLI test gaps. This plan addresses all issues in priority order.

---

## Phase 1: Type Safety (293 mypy errors)

**Why first:** Type errors are the most visible quality signal. A typed package (`py.typed`) that fails `mypy` undermines credibility.

### Error breakdown by category

| Error Type | Count | Fix Strategy |
|------------|-------|--------------|
| `has-type` | 60 | Add type annotations to variables |
| `assignment` | 42 | Fix type mismatches in assignments |
| `arg-type` | 33 | Correct function call argument types |
| `name-defined` | 32 | Guard conditional imports with `TYPE_CHECKING` |
| `no-any-return` | 28 | Add return type annotations |
| `var-annotated` | 18 | Annotate module-level variables |
| `no-redef` | 15 | Fix re-export patterns in protocol.py |
| `operator` | 13 | Add None guards before comparisons |
| `attr-defined` | 11 | Fix attribute access on wrong types |
| `return-value` | 10 | Fix return type mismatches |
| `dict-item` | 9 | Fix dict value type annotations |
| `index` | 6 | Fix indexing on wrong types |
| `union-attr` | 5 | Narrow union types before attribute access |
| `valid-type` | 4 | Fix invalid type syntax |
| `misc` | 4 | Miscellaneous fixes |
| Other | 3 | import-untyped, call-arg, abstract |

### Top files by error count

| File | Errors | Root Cause |
|------|--------|------------|
| `vnext/integrations/hf.py` | 63 | HF Trainer types unresolved |
| `bench/protocol.py` | 46 | Conditional imports (`BENCH_AVAILABLE`) |
| `vnext/audit/lora_audit.py` | 44 | Missing annotations on audit functions |
| `bench/metadata.py` | 17 | Dynamic attribute access |
| `cli.py` | 13 | Variable shadowing, None comparisons |
| `bench/invariants.py` | 13 | Dict type mismatches |
| `finetune/lora.py` | 10 | Missing return types |
| `finetune/alerts.py` | 9 | Dataclass field initialization |

### Approach

1. **Conditional imports** (protocol.py, hf.py): Use `TYPE_CHECKING` blocks for type-only imports. Guard runtime uses with `if BENCH_AVAILABLE:` patterns.
2. **Re-export collisions** (protocol.py `no-redef`): The re-export pattern defines local fallback functions when bench deps aren't available, then re-imports. Fix with explicit `if/else` blocks or `# type: ignore[no-redef]` where the pattern is intentional.
3. **Implicit Optional** (1 instance): Change `x: list[str] = None` to `x: list[str] | None = None`.
4. **Missing annotations**: Add return types and variable annotations file by file.
5. **Operator errors** (cli.py): Add `None` guards before numeric comparisons.

### Acceptance criteria

- `mypy gradience/` exits with 0 errors
- No `# type: ignore` without a specific error code (e.g., `# type: ignore[import-untyped]`)

---

## Phase 2: Exception Hierarchy

**Why:** Currently only 2 custom exceptions exist (`TelemetrySchemaError`, `TelemetryFormatError`, both in `telemetry_reader.py`). Users catching Gradience errors must catch `ValueError`, `RuntimeError`, etc., with no way to distinguish Gradience errors from stdlib errors.

### Design

```python
# gradience/exceptions.py

class GradienceError(Exception):
    """Base exception for all Gradience errors."""

class ConfigError(GradienceError):
    """Invalid configuration (YAML parsing, missing fields, constraint violations)."""

class AuditError(GradienceError):
    """Spectral audit failures (missing weights, incompatible shapes, SVD failures)."""

class MergeError(GradienceError):
    """Merge audit failures (incompatible adapters, shape mismatches)."""

class TelemetryError(GradienceError):
    """Telemetry read/write errors."""

class TelemetrySchemaError(TelemetryError):
    """Schema validation failures in telemetry events."""

class TelemetryFormatError(TelemetryError):
    """Format/parsing errors in telemetry files."""

class BenchError(GradienceError):
    """Benchmark protocol failures."""

class DependencyError(GradienceError):
    """Missing optional dependency."""
```

### Migration

1. Create `gradience/exceptions.py` with the hierarchy above.
2. Move `TelemetrySchemaError` and `TelemetryFormatError` from `telemetry_reader.py` to `exceptions.py`, re-export from the original location for backward compatibility.
3. Replace bare `ValueError`/`RuntimeError` raises in public-facing code with specific exception types.
4. Export base `GradienceError` from `gradience/__init__.py`.
5. Add `GradienceError` to `__all__`.
6. Update `PUBLIC_API.md` to document the exception hierarchy.

### Scope limit

Only replace exceptions in **public-facing code paths** (CLI commands, API functions, public module functions). Internal bench protocol code can keep generic exceptions since it's not part of the stable API.

---

## Phase 3: Stricter MyPy Configuration

**Why:** Current config is permissive (`ignore_missing_imports = true`, no `no_implicit_optional`). For a published typed package, stricter settings catch real bugs.

### Changes to `pyproject.toml`

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
check_untyped_defs = true
no_implicit_optional = true          # NEW: explicit Optional required
warn_redundant_casts = true          # NEW
warn_unused_ignores = true           # NEW
exclude = ["tests/", "scripts/", "dev/", "docs/"]

# Per-module overrides for optional dependency modules
[[tool.mypy.overrides]]
module = [
    "transformers.*",
    "peft.*",
    "datasets.*",
    "accelerate.*",
    "evaluate.*",
    "pandas.*",
    "sklearn.*",
    "safetensors.*",
]
ignore_missing_imports = true
```

### Approach

Move `ignore_missing_imports` from global to per-module overrides so that missing imports in Gradience's own code are caught, while optional third-party deps are still allowed to be untyped.

---

## Phase 4: Metadata and Documentation Fixes

Small but visible issues that affect first impressions.

### 4a. CITATION.cff version mismatch

Current: `version: "0.9.10"`. Should be `"0.10.0"`.

### 4b. README_PYPI.md Python version

Add `Requires Python 3.10+` to the Quick Start section, before the `pip install` commands.

### 4c. Verify version consistency

Run `make verify-version` and fix any discrepancies.

---

## Phase 5: Logging Infrastructure

**Why:** The package currently has zero `logging` usage. All output goes through `rich.console.Console.print()` or bare `print()`. This is fine for CLI output, but library consumers (who call `gradience.api.*` or import modules directly) have no way to control verbosity or capture diagnostic output.

### Design

- Add `logging.getLogger(__name__)` to library modules (not CLI modules).
- CLI modules (`cli.py`, `run_bench.py`) keep using Rich for formatted output.
- Library modules (`api.py`, `vnext/audit/`, `vnext/merge/`, `vnext/telemetry.py`) use `logger.info()` for progress, `logger.warning()` for recoverable issues, `logger.debug()` for diagnostics.
- Do NOT configure logging (no `basicConfig`). Let consumers configure their own handlers.

### Scope

Only add logging to modules in the **public API surface**: `api.py`, `vnext/audit/lora_audit.py`, `vnext/merge/`, `vnext/rank_suggestion.py`, `vnext/telemetry.py`. Internal bench code is out of scope.

---

## Phase 6: CLI Test Coverage

**Why:** `cli.py` is ~3,500 lines with only 1 smoke test (`gradience --help`). This is the primary user-facing interface.

### Approach

Create `tests/test_cli_commands.py` with subprocess-based tests for each major command:

1. **`gradience check`** — valid config, invalid config, missing file
2. **`gradience audit`** — with a fixture adapter dir, missing dir, invalid adapter
3. **`gradience merge-audit`** — with fixture adapter pair, single adapter (error), incompatible shapes
4. **`gradience monitor`** — with a fixture JSONL file, empty file, missing file
5. **`gradience --version`** — verify version output matches `__version__`

Each command gets 3-5 tests: happy path, invalid input, missing input, edge cases.

### Fixture strategy

Use existing test fixtures (`tests/fixtures/`) and `tmp_path` to create minimal adapter directories. Do not require GPU or model downloads. Tests should complete in <5s each.

### Target

~25-30 new CLI tests covering all 4 primary commands + version + help.

---

## Phase 7: Error Path Tests

**Why:** Only 4.6% of tests (32/697) test error scenarios. Production code should verify that errors are raised with useful messages.

### Approach

Add error-path tests to existing test files rather than creating new ones:

1. **Config validation errors** — malformed YAML, missing required fields, invalid values
2. **Audit errors** — non-existent adapter directory, empty directory, corrupted safetensors
3. **Telemetry errors** — malformed JSONL, truncated events, schema violations
4. **Merge errors** — shape mismatches, single adapter, incompatible base models
5. **API errors** — invalid arguments to public API functions

### Target

~40-50 new error-path tests distributed across existing test modules. After Phase 2 (exception hierarchy), these tests can verify specific exception types.

---

## Execution Order and Dependencies

```
Phase 1 (mypy) ─────────────────────┐
                                     ├── Phase 3 (stricter mypy)
Phase 2 (exceptions) ───────────────┤
                                     ├── Phase 7 (error path tests)
Phase 4 (metadata) ──── independent  │
                                     │
Phase 5 (logging) ──── independent   │
                                     │
Phase 6 (CLI tests) ── independent ──┘
```

- Phase 1 must precede Phase 3 (fix errors before tightening rules).
- Phase 2 should precede Phase 7 (define exceptions before testing them).
- Phases 4, 5, 6 are independent and can run in any order or in parallel.

---

## Out of Scope

These items were identified but are **not essential** for initial publication:

- **Research module tests** (`research/fisher.py`, `hessian.py`, `phase_transitions.py`) — experimental, not part of public API
- **Performance benchmarks** — useful but not blocking
- **Cross-platform CI** (Windows/macOS) — already tested in `pip-install-ready.yml`
- **Dependabot configuration** — nice to have
- **pytest-benchmark** — not needed for initial release
- **Property-based testing** (hypothesis) — overkill for initial release
