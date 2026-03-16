# Development Setup

## Prerequisites

- Python 3.10 or later
- git

## Setup

```bash
# Clone the repository
git clone https://github.com/johntnanney/gradience.git
cd gradience

# Create virtual environment and install all development dependencies
make setup
source .venv/bin/activate

# Verify everything works
gradience verify
make check
```

The `make setup` command:

1. Creates a `.venv` virtual environment
2. Installs Gradience in editable mode with `[hf,dev]` extras
3. Sets up all development tools (ruff, mypy, pytest, pre-commit)

## Common commands

| Command | Description | Time |
|---------|-------------|------|
| `make check` | Lint + format check + tests | ~30s |
| `make test-smoke` | CPU-only smoke tests | ~6s |
| `make test-quick` | Tests without coverage | ~30s |
| `make test` | Full tests with coverage | ~60s |
| `make lint` | ruff check + mypy | ~10s |
| `make format` | Auto-format code | ~2s |
| `make format-check` | Check formatting (no changes) | ~2s |

## Pre-commit hooks

Install pre-commit hooks for automatic checking on every commit:

```bash
pre-commit install
```

Hooks include: trailing whitespace, EOF fixer, YAML/TOML validation, debug statement detection, ruff lint, ruff format, mypy.

## Running specific tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_rank_suggestion.py -v

# Run tests matching a pattern
pytest tests/ -k "test_audit" -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Run with coverage
pytest tests/ --cov=gradience --cov-report=term-missing
```

## Project structure

See [Module Map](../architecture/modules.md) for a complete guide to the codebase.
