# Packaging Infrastructure Design

**Date:** 2026-02-15
**Status:** Approved
**Goal:** Make `pip install gradience` work from PyPI with correct metadata, and `make publish` a single command.

## Decisions

- **Version:** 0.10.0 (minor bump signals merge-audit + M1 modules; stays pre-1.0)
- **PyPI voice:** Research-first (spectral analysis of training dynamics)
- **GitHub URL:** `johntnanney/gradience` (personal repo, can transfer later)
- **Approach:** Full polish (B) — pyproject.toml fixup, README_PYPI.md, Makefile publish targets

## Changes

### 1. Version bump: 0.9.10 → 0.10.0

Files:
- `pyproject.toml` → `version = "0.10.0"`
- `gradience/__init__.py` fallback → `__version__ = "0.10.0"`
- `README.md` citation block → `note = {Version 0.10.0}`

Validate with `scripts/verify_version.py`.

### 2. pyproject.toml

- **URLs**: `gradience-ai/gradience` → `johntnanney/gradience` (4 occurrences)
- **Classifiers**: Add `Intended Audience :: Developers`, `Typing :: Typed`
- **Keywords**: Add `merge-audit`, `adapter-merge`, `compression`, `model-compression`
- **Core deps**: Add `scipy>=1.7.0` (used by null_controls.py t-test)
- **readme**: Change to `README_PYPI.md`
- **No changes to**: build-system, entry points, optional deps, tool config, package-data

### 3. README_PYPI.md (new)

~60-70 line condensed research-framed README for PyPI listing. Covers:
- What it is (one line)
- Quick start (audit, merge-audit, bench)
- What you get (spectral measurements, merge analysis, bench protocol)
- Install extras table
- Links, citation

### 4. Makefile targets

- `make build` — `python3 -m build`
- `make publish` — `clean → test-quick → build → twine check → twine upload`
- `make publish-test` — same but `--repository testpypi`

### 5. Cleanup

- Remove stale dist/ artifacts (0.9.1, 0.9.4)
- `make clean` already handles build artifacts

## Scope boundary

Not included: GitHub Actions publish workflow, CI changes, test changes, runtime code changes.

## Implementation sequence

1. Create README_PYPI.md
2. Update pyproject.toml (version, readme, URLs, classifiers, keywords, scipy dep)
3. Update __init__.py fallback version
4. Update README.md citation block
5. Add Makefile targets (build, publish, publish-test)
6. Clean dist/
7. Test: `make build && twine check dist/*`
8. Run `scripts/verify_version.py`
9. Run `make test-quick` to confirm nothing broke
10. Commit
