# CI & Testing

## GitHub Actions workflows

### Main CI (`ci.yml`)

Runs on every push and pull request:

1. **Lint** — ruff check with GitHub output format
2. **Type check** — mypy on `gradience/`
3. **Tests** — pytest with coverage on Python 3.10, 3.11, 3.12 matrix
4. **Coverage** — Uploaded to Codecov (Python 3.11 only)
5. **Bench smoke test** — Validates bench artifact structure
6. **Config validation** — Checks all YAML configs against schema

Timeout: 15 minutes.

### Tests (`tests.yml`)

Additional validation:

- Syntax checking on key modules (`py_compile`)
- CLI sanity checks (`gradience --help`, `gradience verify`)
- Functional checks (telemetry reader, rank suggestion)
- HF integration tests (with dedicated environment variable)
- Package validation (build wheel, install in fresh venv, test imports)

### Security (`security.yml`)

- Gitleaks for secret detection (full history)
- Runs weekly (Monday 6 AM UTC) and on push

### Other workflows

- **Nightly** (`nightly.yml`) — Extended test runs
- **pip-install-ready** (`pip-install-ready.yml`) — Package installability verification

## Testing guidelines

### Test organization

- All tests in `tests/` with `test_` prefix
- Shared fixtures in `tests/conftest.py`
- Test data in `tests/fixtures/`

### Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Long-running tests (excluded in quick mode) |

### Timeouts

Default timeout is 30 seconds per test. Tests exceeding this are killed.

### Deprecation warnings

`DeprecationWarning` is treated as an error, except warnings from `gradience` itself (which are expected during migration periods).

### Running CI locally

```bash
# Equivalent to the full CI check
make check

# Equivalent to pre-release validation
make pre-release
```
