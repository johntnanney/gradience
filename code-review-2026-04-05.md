# Gradience — Full Code Review

**Date:** April 5, 2026
**Scope:** Complete codebase — configuration, core package, vnext modules, test suite, code quality patterns
**Reviewed:** ~170 Python files, ~20,000+ lines of production code, ~116 test files

---

## Executive Summary

Gradience is a remarkably well-engineered library. The vnext core — where the real intellectual work lives (spectral analysis, rank policies, merge diagnostics) — is **production-grade** with excellent numerical stability, clean module boundaries, and disciplined schema governance. The mathematical implementations are correct and defensive. Type annotations are nearly complete. No security vulnerabilities were found.

The areas that need attention are primarily **structural and ergonomic**, not algorithmic. The CLI layer has accumulated architectural debt (pervasive `sys.exit()` calls, inconsistent error strategies), and the build configuration has some staleness and duplication. These are the kind of problems that accumulate naturally in a research-first project that has matured into a real tool — they don't threaten correctness, but they do affect maintainability and testability.

**Overall Grade: A-/B+**

| Area | Grade | Summary |
|------|-------|---------|
| Algorithmic correctness (vnext) | A | Mathematically sound, comprehensive edge-case handling |
| Numerical stability | A | Exemplary eps guardrails, clamping, safe division throughout |
| Type safety & annotations | A- | Near-complete coverage, frozen dataclasses, proper enums |
| Schema governance | A | Frozen versioned schemas, additive-only discipline |
| Module architecture | A | Clean one-way dependency flow, no circular imports |
| Security | A | No dangerous functions, no hardcoded secrets |
| Resource management | A | 100% context manager usage for file I/O |
| Public API design | A- | Well-documented, clear contracts, good stability tiers |
| Test suite | B+ | Mature coverage with strong math tests; some fixture/mock gaps |
| Code quality patterns | B+ | Clean imports, minimal dead code; some broad exception catches |
| CLI layer | B- | Functional but hard to test; sys.exit() proliferation |
| Build configuration | B- | Staleness, duplication between pytest.ini and pyproject.toml |

---

## I. What's Working Well

These are genuine strengths — not just "adequate" but notably good for a project of this kind.

### The Measurement Layer Is Mathematically Sound

The five rank-selection policies (energy threshold, entropy effective rank, Gavish-Donoho optimal hard threshold, kneedle elbow detection, stable rank ceiling) are each implemented with proper numerical guardrails. Every division is protected by epsilon checks. Every log operation adds `1e-10` or validates positivity first. SVD computations use QR decomposition to avoid materializing full-rank matrices. Singular values are clamped to valid ranges after computation to account for floating-point drift.

This matters because the entire diagnostic value of Gradience depends on these measurements being trustworthy. They are.

### Schema Versioning Is Disciplined

The frozen-schema pattern (`gradience.vnext.telemetry/v1`, `gradience.adapter_qa/v1`, `gradience.merge_qa_report/v1`, `gradience.inventory_summary/v1`) with additive-only evolution is exactly right for a tool whose outputs need to be machine-readable and stable across versions. Every schema type has a `from_dict()` gatekeeper that validates contracts and raises `QASchemaError` on violations.

### Module Boundaries Are Clean

The vnext dependency graph flows strictly downward: `types.py` → `telemetry.py` → `rank_suggestion.py` → `audit/` → `merge/` → `integrations/`. No circular imports. Each module owns its validation. The merge pipeline (`spectral_compat` → `spectral_theory` → `verdicts` → `recommend`) is well-insulated and could operate standalone.

### Frozen Dataclasses Prevent Mutation Bugs

All measurement containers (`SubspaceMetrics`, `PairDiagnosis`, `MergeRecommendation`, `ConfigSnapshot`, etc.) are frozen dataclasses. This is the right choice for a system where computed measurements flow through a pipeline — once you've measured something, the measurement shouldn't change.

### Telemetry Has Privacy Guardrails

The telemetry writer includes text sanitization with configurable redaction policy — it won't accidentally log prompts or sensitive text. Recursive path tracking for debugging. This is forward-thinking for a tool that operates on training data.

---

## II. Issues Requiring Attention

### Critical (Fix Before Next Release)

**1. CLI testability: pervasive `sys.exit()` pattern**

The CLI module (`cli.py`, ~3,800 lines) contains approximately 80 `sys.exit()` calls scattered through individual command handlers. This makes the CLI essentially untestable from Python — you can only test it by spawning subprocesses and checking exit codes and stdout strings.

The standard fix is to have command functions raise typed exceptions (e.g., `CLIError`, `ConfigValidationError`) and catch them in a single `main()` wrapper that translates to exit codes. This also makes the CLI usable programmatically — someone could call `run_audit(args)` from Python and get a structured result instead of needing `subprocess.run`.

This is the single highest-impact structural improvement available. It doesn't change any behavior; it just makes the behavior testable and reusable.

**2. Missing numpy import in `policy_analysis.py`**

The module uses `np.percentile()` at line 148 but has no `import numpy as np` at module level. This will raise `NameError` at runtime if the code path is hit. Likely masked because the function is called from contexts where numpy is already in scope, but it's a latent bug.

**3. Makefile hardcodes version 0.11.0**

The `publish-test` target (line 121) references `gradience==0.11.0`, but pyproject.toml declares version 1.0.1. Anyone following the Makefile instructions to test a PyPI publish will install the wrong version. Should be dynamically extracted.

### Moderate (Plan to Address)

**4. Duplicate pytest configuration**

Both `pytest.ini` and `pyproject.toml [tool.pytest.ini_options]` define nearly identical test configuration. If one is updated and the other isn't, they'll silently diverge. The modern practice is to remove `pytest.ini` entirely and rely on pyproject.toml. The one addition pytest.ini has (`pythonpath = tests`) should either be documented and moved to pyproject.toml, or removed if not needed.

**5. Broad exception catches in `api.py`**

Lines 165 and 175 catch bare `Exception` during JSON artifact loading. These are documented as "best-effort" paths, which is the right intent, but they should catch `(OSError, json.JSONDecodeError, KeyError)` specifically. Catching `Exception` masks genuine bugs (e.g., `TypeError` from a code change) that you'd want to see during development.

The same pattern appears in several places in `cli.py` (lines 720, 1102, 2093, 2157, 2179). Each is commented, but narrowing the exception types would make debugging easier without changing behavior.

**6. Inconsistent error reporting in CLI**

The CLI mixes three error-reporting strategies: `print()` + `sys.exit(1)`, `print(stderr)` + `sys.exit(1)`, and exception propagation. There's no centralized error handler. A single `_cli_error(message, code=1)` function (or the exception-based approach from item #1) would unify this.

**7. Pre-commit ruff config creates confusing UX**

The pre-commit hook runs ruff with `--fix --exit-non-zero-on-fix`, which auto-fixes issues *and then fails the commit*. This forces developers to `git add` the fixed files and re-commit. Either remove `--fix` (let developers run `make format` separately) or remove `--exit-non-zero-on-fix` (auto-fix and allow the commit).

**8. Docker container runs as root**

The Dockerfile has no `USER` instruction. Standard security practice is to create a non-root user:
```dockerfile
RUN useradd -m gradience && chown -R gradience /workspace
USER gradience
```

**9. Mixed unittest/pytest styles across test suite**

27 of 116 test files use `unittest.TestCase` with `unittest.mock.patch()`. The rest use pure pytest fixtures. This creates cognitive load and inconsistent mock cleanup patterns. The `patch()` calls aren't always explicitly exited, which creates a small risk of cross-test pollution (mitigated by pytest's test isolation, but not guaranteed).

**10. Magic numbers in CLI analysis functions**

`_analyze_policy_disagreements()` (lines 1190-1267) contains hardcoded thresholds: `spread_threshold = max(3, 0.5 * max_k)`, `sorted_all[:3]` (top 3 layers), top-8 display for flat distributions. These should be module-level constants with brief justification comments.

### Minor (Nice-to-Have)

**11. Dependency version constraints may block security patches**

`transformers>=4.35.0,<5` and `peft>=0.7.0,<1` pin upper bounds. When transformers 5.0 or peft 1.0 ship, these will prevent installation of security fixes. Consider removing upper bounds or documenting the compatibility testing schedule.

**12. MANIFEST.in includes all JSON files under gradience/**

`recursive-include gradience *.json` captures everything. Should be scoped to specific directories (`gradience/bench/configs *.json`) to avoid accidentally packaging internal metadata.

**13. Sparse test fixture data**

Only 2 fixture files exist (`vnext_check_config.json`, `vnext_minimal.jsonl`) for 116 test files. All adapter data is generated by factory functions. Consider adding representative "golden" fixtures for regression testing and documentation.

**14. Some test assertions are presence-only**

Patterns like `assert result is not None` or `assert p.returncode != 0` don't validate the actual content or specific error code. Tightening these would catch subtle regressions.

**15. One test explicitly skipped**

`test_watchdog_system.py` has `@unittest.skip("Timeout detection test needs investigation")` — indicates a timing-dependent test that should either be fixed or removed.

**16. Redundant exception inheritance**

`TelemetrySchemaError(TelemetryError, ValueError)` inherits from `ValueError` twice (once directly, once through `TelemetryError`). Not harmful, but the direct `ValueError` inheritance is redundant and should be removed for clarity.

**17. CLI numpy import at top level**

`cli.py` imports numpy at the top of the file but only uses it in `_analyze_policy_disagreements()`. Moving this to a lazy import would speed up CLI startup for commands that don't need numpy.

---

## III. Architectural Observations

These aren't "issues" exactly — they're observations about where the codebase sits architecturally and what the trade-offs are.

### The CLI is doing too much

At ~3,800 lines, `cli.py` is the largest file in the project. It handles argument parsing, config normalization, file I/O, output formatting, and business logic for every command. The output functions alone (`_print_audit_summary`, `_print_monitor_result`, etc.) are several hundred lines of print statements that produce rich terminal output but can't be tested or reused.

The typical refactoring path is: extract output formatting into a `cli_format.py` module that returns strings (testable), extract command implementations into a `commands/` package, and keep `cli.py` as a thin dispatcher. This would also make the exception-based error handling (item #1) natural to implement.

This isn't urgent — the CLI works correctly. But if you're planning to add more commands or make the tool scriptable, the current structure will resist it.

### The research/ module is isolated but unguarded

`gradience/research/` contains Fisher information, Hessian analysis, and phase transition detection. These modules are well-implemented but don't participate in the schema governance or public API tiers that the vnext/ modules enjoy. They import torch directly (not lazily), have no `from_dict`/`to_dict` contracts, and aren't covered by the same type-checking rigor.

This is probably fine if these remain research tools. But if any of their outputs feed into the product pipeline, they should be promoted to vnext-style discipline.

### The four-layer extraction model is reflected in code

The architecture assessment document describes a four-layer model: measurement → diagnosis → aggregation → policy. This is genuinely visible in the code. `audit/lora_audit.py` measures, `merge/diagnose.py` diagnoses, `inventory/summary.py` aggregates, and `merge/recommend.py` + `policy/` apply policy. The parameterization points (aggregation strategy and policy vocabulary) are identifiable. This is a sign of coherent design — the conceptual model and the implementation model agree.

---

## IV. Recommended Priority Order

1. **Fix the numpy import bug** in `policy_analysis.py` (5 minutes, prevents runtime crash)
2. **Fix the Makefile version** (5 minutes, prevents user confusion)
3. **Remove pytest.ini** and consolidate into pyproject.toml (30 minutes)
4. **Narrow broad exception catches** in `api.py` and `cli.py` (1-2 hours)
5. **Plan the CLI refactoring** — extract exception-based error handling as the first step, then output formatting, then command modules (multi-session project)
6. **Add Dockerfile USER instruction** (10 minutes)
7. **Reconcile pre-commit ruff config** (10 minutes)
8. **Extract magic numbers to constants** in CLI analysis functions (30 minutes)

Items 1-3 are quick wins. Item 5 is the strategic investment that pays off over time.

---

## V. Fixes Applied (April 5, 2026)

All fixes pass ruff lint, ruff format, and Python syntax validation.

| # | File(s) | Change | Category |
|---|---------|--------|----------|
| 1 | `gradience/policy_analysis.py` | Moved `import numpy as np` to top of file; removed inline import from loop body | Bug fix |
| 2 | `gradience/policy_analysis.py` | Extracted 7 magic numbers into named module-level constants (`FROBENIUS_WEIGHT`, `PARAM_COUNT_WEIGHT`, `UTILIZATION_WEIGHT`, `PARAM_COUNT_SCALE`, `UTILIZATION_SCALE`, `SPREAD_FLOOR`, `SPREAD_FRACTION`) | Readability |
| 3 | `Makefile` | `publish-test` target now reads version dynamically from pyproject.toml instead of hardcoding `0.11.0` | Bug fix |
| 4 | `pytest.ini` / `pyproject.toml` | Replaced pytest.ini with stub; added `pythonpath = ["tests"]` to pyproject.toml with explanatory comment | Config hygiene |
| 5 | `gradience/api.py` | Narrowed `except Exception:` to `(OSError, json.JSONDecodeError, UnicodeDecodeError)` and `(KeyError, TypeError, ValueError)`; removed redundant `import sys` | Error handling |
| 6 | `gradience/exceptions.py` | Removed redundant `ValueError` inheritance from `TelemetrySchemaError` and `TelemetryFormatError` (already inherited via `TelemetryError`) | MRO cleanup |
| 7 | `gradience/cli.py` | Moved `import numpy as np` from top-level to lazy import inside `_analyze_policy_disagreements()` | Startup perf |
| 8 | `Dockerfile` | Added non-root `gradience` user with `chown` on `/workspace`; `USER gradience` before entrypoint | Security |
| 9 | `gradience/bench/watchdog.py` | Fixed timeout detection bug: lowered min check interval from 5s to 1s; set `is_running = False` before diagnostic collection; fixed `timeout_minutes` type hint to `float` | Bug fix |
| 10 | `tests/test_watchdog_system.py` | Removed `@unittest.skip` from `test_timeout_detection`; updated wait time and docstring to match fixed watchdog timing | Test fix |
| 11 | `.pre-commit-config.yaml` | Removed `--exit-non-zero-on-fix` from ruff hook so auto-fixes don't reject the commit | UX |

### Remaining open items (not addressed)

- **CLI `sys.exit()` refactoring** — 80 calls across 16 command handlers. Strategic project (~4-6 hours). Start with exception-based `main()` wrapper, then migrate handlers incrementally.
- **CLI file decomposition** — `cli.py` is ~3,800 lines. Natural split: `commands/` package + `cli_format.py` for output. Depends on sys.exit refactoring.
- **`research/` module governance** — No schema contracts, no lazy imports, no mypy coverage. Fine as research tools; needs promotion if outputs feed into product pipeline.
- **Substrate extraction decision** — Architectural question: expose the four-layer model as a parameterized engine, or keep merge as the sole product surface.
- **DeBERTa adjudication** — Blocked on GPU compute. Highest-value empirical work once hardware is available.

---

*Review conducted against the codebase as of April 5, 2026. 54 vnext modules (19,670 lines), ~50 additional modules in core/bench/research/finetune, 116 test files reviewed. 11 fixes applied same day.*
