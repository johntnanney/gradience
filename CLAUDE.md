# CLAUDE.md — Gradience

## Project Overview

Gradience is a Python library for **spectral analysis of LoRA (Low-Rank Adaptation) fine-tuning dynamics**. It measures rank evolution, detects phase transitions, and studies training geometry via SVD-based analysis. Published on PyPI as `gradience` (v0.11.0).

## Quick Reference

```bash
# Install for development
make setup                    # Creates .venv, installs [hf,dev] extras
source .venv/bin/activate

# Run checks (lint + format-check + tests)
make check

# Individual commands
make test-smoke               # CPU-only smoke tests (~6s, no GPU)
make test-quick                # Fast tests without coverage (~30s)
make test                      # Full test suite with coverage
make lint                      # ruff check + mypy
make format                    # Auto-format with ruff
```

## Repository Structure

```
gradience/                     # Main package
├── __init__.py                # Public API re-exports, version
├── api.py                     # Stable Python API wrappers
├── cli.py                     # CLI entrypoint (gradience command)
├── exceptions.py              # Exception hierarchy (GradienceError base)
├── peft_utils.py              # PEFT/LoRA weight utilities
├── policy_analysis.py         # Rank policy analysis
├── analysis/                  # Time-series & lead-lag analysis
├── bench/                     # Benchmarking suite (compression validation)
│   ├── configs/               # YAML benchmark configurations
│   ├── policies/              # YAML rank policies
│   ├── run_bench.py           # Bench entrypoint (gradience-bench command)
│   └── ...                    # Compression, reporting, decision trace, etc.
├── finetune/                  # LoRA fine-tuning utilities & alerts
├── integrations/              # Framework integrations (HuggingFace callback)
├── research/                  # Research tools (Fisher, Hessian, phase transitions)
└── vnext/                     # Current-generation core modules
    ├── audit/                 # Spectral audit implementation
    ├── merge/                 # Merge compatibility analysis
    ├── policy/                # Rank suggestion policies
    ├── integrations/          # HF callback (GradienceCallback)
    ├── telemetry.py           # JSONL telemetry writer/reader
    ├── rank_suggestion.py     # Global & per-layer rank suggestions
    ├── svd_truncate.py        # SVD truncation utilities
    └── types.py               # Shared type definitions

tests/                         # Test suite (~60 test files)
├── conftest.py                # Shared fixtures
├── fixtures/                  # Test data fixtures
├── helpers/                   # Test helper utilities
├── merge/                     # Merge-specific tests
└── test_bench/                # Bench-specific tests

scripts/                       # Development & CI scripts
experiments/                   # Research experiment scripts
examples/                      # Usage examples
docs/                          # Documentation
```

## Build & Tooling

- **Python**: 3.10+ required (3.10, 3.11, 3.12 tested in CI)
- **Build system**: setuptools (pyproject.toml)
- **Package manager**: pip (no poetry/pdm)
- **Linter**: ruff (line-length=120, rules: E, F, W, I, UP, B, SIM)
- **Type checker**: mypy (check_untyped_defs=true, excludes tests/scripts/dev/docs)
- **Formatter**: ruff format
- **Test framework**: pytest with pytest-cov, pytest-timeout (30s default)
- **Pre-commit hooks**: trailing-whitespace, end-of-file-fixer, ruff, ruff-format, mypy

## Key Commands

| Command | Description |
|---------|-------------|
| `pytest tests/ -v` | Run tests verbosely |
| `pytest tests/ -q` | Run tests quietly |
| `ruff check .` | Lint check |
| `ruff check . --fix` | Lint with auto-fix |
| `ruff format .` | Format code |
| `ruff format --check .` | Check formatting |
| `mypy gradience/` | Type check |
| `make check` | All quality checks (lint + format-check + test-quick) |
| `make pre-release` | Full pre-release validation |

## Architecture & Conventions

### Public API Tiers

1. **Stable (public API)** — `gradience.api`, CLI commands, `gradience.vnext.telemetry/v1` schema, exports from `gradience.__init__`
2. **Internal** — Everything else; may change without notice

### CLI Commands

Entry point: `gradience.cli:main` (registered as `gradience` console script)

- `gradience check` — Config validation
- `gradience monitor` — Training telemetry analysis
- `gradience audit` — Spectral measurement of LoRA adapters
- `gradience merge-audit` — Geometric compatibility between adapter pairs
- `gradience verify` — Installation verification

Bench entry point: `gradience.bench.run_bench:main` (registered as `gradience-bench`)

### Exception Hierarchy

All exceptions inherit from `GradienceError`. Specific types: `ConfigError`, `AuditError`, `MergeError`, `TelemetryError`, `TelemetrySchemaError`, `TelemetryFormatError`, `DependencyError`, `QASchemaError`.

### Optional Dependencies

- `gradience[hf]` — HuggingFace/PEFT integration (transformers, peft, sentencepiece)
- `gradience[bench]` — Full benchmarking suite (adds datasets, evaluate, scikit-learn, pandas)
- `gradience[dev]` — Development tools (pytest, ruff, mypy, pre-commit)
- `gradience[all]` — Everything

### Code Style

- Line length: 120 characters
- Imports sorted by ruff (isort rules via `I` selector)
- Type hints used throughout; `from __future__ import annotations` in most files
- `typing_extensions` used for Python 3.10 compatibility
- Tests use `pytest` conventions with `test_` prefix; markers: `slow`
- Test timeout: 30 seconds per test
- When combining `# type: ignore` and `# noqa:` on one line, mypy's directive must come first: `# type: ignore[no-redef]  # noqa: F811`

### Key Patterns

- Lazy imports for optional dependencies (transformers, peft, etc.)
- `vnext/` namespace contains the current-generation implementation
- Deprecated Guard API raises `DeprecationWarning` + `ImportError` with migration guidance
- Telemetry uses structured JSONL format (`gradience.vnext.telemetry/v1`)
- Bench configs are YAML files validated by `config_schema.py`

### Merge Pipeline (`vnext/merge/`)

- Pipeline: `merge_audit()` → `diagnose_pair()` → `recommend_merge()` → `format_recommendation()`
- Strategy strings are lowercase: `"linear"`, `"ties"`, `"dare_ties"`, `"dare_linear"` — not class names
- `MergeRecommendation.overall_strategy` is always `"audit_aware"`
- `PairDiagnosis.layer_diagnoses` (tuple, not `.layers`)
- `report.aggregate.overall_verdict` / `.compatibility_score` (not direct on report)
- `layer_verdicts` is `list[dict]`, not list of dataclasses
- Canonical helpers: `_energy_rank()` in `audit/lora_audit.py`, `_shorten_layer_name()` in `merge/report.py`
- Verdict branch order: Branch 0 (IMBALANCED, low overlap) → Branch 1 (SAFE, orthogonal) → Branch 2 (REDUNDANT) → Branch 3 (CONFLICTING) → Branch 4 (IMBALANCED, high overlap) → Branch 5 (SAFE, default)
- Merge test fixtures in `tests/merge/conftest.py`: `orthogonal_pair`, `redundant_pair`, `conflicting_pair`, `imbalanced_pair` — each returns `tuple[Path, Path]`

### Adapter QA Artifact (`vnext/audit/qa_artifact.py`)

- Schema: `gradience.adapter_qa/v1` — frozen, additive-only versioning
- `AdapterQAArtifact` and `EligibilityStatus` are stable public API (exported from `gradience.__init__`)
- `gradience.api.audit_adapter()` is the stable Python entry point; same builder path as CLI
- Four eligibility statuses: `eligible`, `uncertain`, `flagged_weak`, `unknown_no_behavioral_eval`
- `behavioral_summary` = evidence, `eligibility` = policy judgment (not raw measurement)
- Nullable fields: `metric_name` and `lower_is_better` are `None` (not `""` / `True`) when no eval
- `from_dict()` is the single validation gatekeeper — raises `QASchemaError` on contract violations
- Three-way loader in CLI (`_load_source_qa`): schema present+correct → strict v1, absent → legacy, wrong → hard fail
- `--strict-qa` blocks both `flagged_weak` and `unknown_no_behavioral_eval`
- JSON key: `effective_rank_90_median` in output (internal field: `energy_rank_90_p50`)
- Canonical examples in `examples/qa/` — one per status
- Definition doc: `docs/adapter-qa-artifact.md`

### Merge QA Report (`vnext/merge/qa_report.py`)

- Schema: `gradience.merge_qa_report/v1` — frozen, additive-only versioning
- `MergeQAReport` is stable public API (exported from `gradience.__init__`)
- `gradience.api.merge_risk_report()` is the stable Python entry point (CLI-delegating wrapper)
- `from_dict()` is the single validation gatekeeper — raises `QASchemaError` on contract violations
- `eligibility_status` per adapter: canonical `EligibilityStatus` value or `null` (no QA provided)
- `dominant_issue`: machine-readable label from frozen set (`norm_imbalance`, `subspace_conflict`, `high_redundancy`, `partial_redundancy`, `none`, `unknown`)
- `dominant_issue_detail`: human-readable explanation companion to `dominant_issue`
- `recommended_strategy`: operational — `"linear"` (low risk), `"norm_equalized"` (medium), `"audit_aware"` (high/compression)
- `recommended_action`: explanatory prose, does not override `recommended_strategy`
- `confidence`: categorical (`"high"`, `"medium"`, `"low"`); `confidence_note` is prose companion
- `--emit-report` writes v1 JSON; `--qa-report` prints 4-section terminal format
- `--strict-qa` blocks `flagged_weak`, `unknown_no_behavioral_eval`, and `null` eligibility
- Canonical examples in `examples/reports/` — one per scenario (safe, high-risk warning, strict-blocked)
- Definition doc: `docs/merge-risk-report.md`

## CI/CD

GitHub Actions workflows:
- **ci.yml** — Main CI: lint (ruff), type check (mypy), tests with coverage, CPU bench smoke test. Runs on Python 3.10/3.11/3.12. Also validates YAML configs.
- **tests.yml** — Additional test configuration
- **security.yml** — Security scanning
- **nightly.yml** — Nightly builds
- **pip-install-ready.yml** — Package installability check

Pre-commit hooks enforce: trailing whitespace, EOF fixers, YAML/TOML/JSON validity, no debug statements, no direct commits to master/main, ruff lint+format, mypy.

## Testing Guidelines

- All tests live in `tests/` with `test_` prefix
- Use existing fixtures from `tests/conftest.py`, `tests/merge/conftest.py`, and `tests/fixtures/`
- Default test timeout is 30 seconds; mark slow tests with `@pytest.mark.slow`
- `DeprecationWarning` is treated as error except from gradience itself
- Run `make test-smoke` for quick validation; `make test` for full coverage
- CI runs `pytest tests/ -v --tb=short --cov=gradience`
