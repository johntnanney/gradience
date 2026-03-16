# Contributing

Thank you for your interest in contributing to Gradience.

## Sections

- [Development Setup](development.md) — Set up your local development environment
- [CI & Testing](ci.md) — How CI works and how to run tests locally

## Quick start for contributors

```bash
git clone https://github.com/johntnanney/gradience.git
cd gradience
make setup
source .venv/bin/activate
make check    # lint + format-check + tests
```

## Code style

- **Line length**: 120 characters
- **Formatter**: ruff format
- **Linter**: ruff check (rules: E, F, W, I, UP, B, SIM)
- **Type checker**: mypy (check_untyped_defs=true)
- **Test framework**: pytest with 30-second timeout per test

## Pull request checklist

1. `make check` passes (lint + format + tests)
2. New tests for new functionality
3. No regressions in existing tests
4. Type hints for public functions
5. Docstrings for public API additions
