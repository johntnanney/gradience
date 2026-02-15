# Packaging Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship `pip install gradience` 0.10.0 to PyPI with correct metadata and a one-command `make publish`.

**Architecture:** Update pyproject.toml metadata (version, URLs, classifiers, keywords, deps), create a condensed README_PYPI.md for the PyPI listing, add Makefile publish targets, and validate the entire build pipeline end-to-end.

**Tech Stack:** setuptools, build, twine, Make

**Design doc:** `docs/plans/2026-02-15-packaging-infrastructure-design.md`

---

### Task 1: Create README_PYPI.md

**Files:**
- Create: `README_PYPI.md`

**Step 1: Write the file**

```markdown
# Gradience

**Spectral analysis of low-rank adaptation dynamics.**

Gradience is a research instrument for studying the geometry of LoRA fine-tuning.
It measures rank structure, energy concentration, and subspace alignment across
adapter layers — and provides reproducible, multi-seed experimental infrastructure
for validating spectral hypotheses.

## Quick Start

```bash
pip install gradience

# Audit a LoRA adapter's spectral structure
gradience audit --peft-dir ./your-adapter --suggest-ranks

# Measure merge compatibility between two adapters
gradience merge-audit --adapter-a ./adapter_a --adapter-b ./adapter_b

# Run a full compression validation benchmark
gradience bench --config bench_config.yaml
```

## What You Get

- **Spectral measurements** — Per-layer SVD analysis: stable rank, energy concentration, utilization ratios, rank waste quantification
- **Merge compatibility analysis** — Principal angles, directional agreement, and magnitude balance between adapter pairs, with per-layer geometric verdicts
- **Training telemetry** — Structured JSONL recording of spectral evolution across training steps
- **Reproducible benchmarking** — Multi-seed compression validation with statistical aggregation and tolerance-based safety policies
- **Publication-ready artifacts** — JSON data, Markdown reports, and aggregate statistics for tables and figures

## Install

```bash
pip install gradience                # Core (torch + safetensors + scipy)
pip install "gradience[hf]"          # + HuggingFace Trainer integration
pip install "gradience[bench]"       # + Full benchmark protocol with eval
pip install "gradience[all]"         # Everything
```

## Links

- **GitHub:** [github.com/johntnanney/gradience](https://github.com/johntnanney/gradience)
- **License:** Apache 2.0

## Citation

```bibtex
@software{gradience2026,
  title  = {Gradience: Spectral Analysis of Low-Rank Adaptation Dynamics},
  author = {Nanney, John T.},
  year   = {2026},
  url    = {https://github.com/johntnanney/gradience}
}
```
```

**Step 2: Verify the file renders**

Run: `python3 -c "from pathlib import Path; content = Path('README_PYPI.md').read_text(); print(f'Lines: {len(content.splitlines())}'); assert 40 < len(content.splitlines()) < 80, 'Unexpected line count'"`
Expected: `Lines: ~55` (no error)

**Step 3: Commit**

```bash
git add README_PYPI.md
git commit -m "docs: add condensed README_PYPI.md for PyPI listing"
```

---

### Task 2: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Make all edits**

Changes (apply with Edit tool):

1. `version = "0.9.10"` → `version = "0.10.0"`
2. `readme = "README.md"` → `readme = "README_PYPI.md"`
3. All 4 URLs: `gradience-ai/gradience` → `johntnanney/gradience`
4. Add to classifiers (after existing ones):
   - `"Intended Audience :: Developers",`
   - `"Typing :: Typed",`
5. Add to keywords: `"merge-audit", "adapter-merge", "compression", "model-compression"`
6. Add to dependencies: `"scipy>=1.7.0",` (after `"rich>=10.0.0"`)

**Step 2: Verify edits are consistent**

Run: `python3 -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb')); print(d['project']['version']); print(d['project']['readme']); print(d['project']['urls']); assert d['project']['version'] == '0.10.0'; assert 'scipy' in str(d['project']['dependencies']); assert 'johntnanney' in d['project']['urls']['Homepage']"`
Expected: Prints version 0.10.0, README_PYPI.md, URLs with johntnanney, no assertion error.

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump to 0.10.0, fix URLs, add classifiers and scipy dep"
```

---

### Task 3: Update version references

**Files:**
- Modify: `gradience/__init__.py` (line 60)
- Modify: `README.md` (lines 12-13)

**Step 1: Update __init__.py fallback**

Change `__version__ = "0.9.10"` → `__version__ = "0.10.0"` on line 60.

**Step 2: Update README.md citation block**

Change:
- `url = {https://github.com/gradience-ai/gradience},` → `url = {https://github.com/johntnanney/gradience},`
- `note = {Version 0.9.10}` → `note = {Version 0.10.0}`

**Step 3: Run version verification**

Run: `python3 scripts/verify_version.py`
Expected: pyproject.toml shows 0.10.0, module __version__ shows 0.10.0. May warn about installed version mismatch (that's fine — we haven't reinstalled yet).

**Step 4: Reinstall and re-verify**

Run: `pip3 install -e . && python3 scripts/verify_version.py`
Expected: All checks pass (pyproject.toml = installed = module = 0.10.0).

**Step 5: Commit**

```bash
git add gradience/__init__.py README.md
git commit -m "chore: update version references to 0.10.0"
```

---

### Task 4: Add Makefile publish targets

**Files:**
- Modify: `Makefile`

**Step 1: Add three new targets**

Add after the existing `pre-release` target (line 103), before the version management section:

```makefile
build: clean ## Build sdist and wheel
	@echo "📦 Building package..."
	@python3 -m build
	@echo "✅ Build complete. Artifacts in dist/"

publish: pre-release build ## Build and publish to PyPI
	@echo "🚀 Publishing to PyPI..."
	@python3 -m twine check dist/*
	@python3 -m twine upload dist/*
	@echo "✅ Published to PyPI!"

publish-test: pre-release build ## Build and publish to TestPyPI
	@echo "🧪 Publishing to TestPyPI..."
	@python3 -m twine check dist/*
	@python3 -m twine upload --repository testpypi dist/*
	@echo "✅ Published to TestPyPI!"
	@echo "Install with: pip install -i https://test.pypi.org/simple/ gradience==0.10.0"
```

**Step 2: Update .PHONY line**

Add `build publish publish-test` to the `.PHONY` declaration on line 2.

**Step 3: Verify targets appear**

Run: `make help | grep -E '(build|publish)'`
Expected: Shows build, publish, publish-test with descriptions.

**Step 4: Commit**

```bash
git add Makefile
git commit -m "chore: add make build/publish/publish-test targets"
```

---

### Task 5: Clean dist/ and validate build pipeline

**Files:**
- No new files — cleanup and validation only

**Step 1: Remove stale dist/ artifacts**

Run: `rm -f dist/gradience-0.9.1* dist/gradience-0.9.4* dist/gradience-0.9.10*`
Expected: Old wheels and tarballs removed.

**Step 2: Build fresh**

Run: `make build`
Expected: `dist/gradience-0.10.0-py3-none-any.whl` and `dist/gradience-0.10.0.tar.gz` created.

**Step 3: Twine check**

Run: `python3 -m twine check dist/*`
Expected: Both PASSED.

**Step 4: Inspect wheel metadata**

Run: `python3 -c "import zipfile; z=zipfile.ZipFile('dist/gradience-0.10.0-py3-none-any.whl'); m=z.read('gradience-0.10.0.dist-info/METADATA').decode(); print(m[:1500])"`
Expected: Version 0.10.0, URLs point to johntnanney/gradience, scipy in Requires-Dist, Typing :: Typed in classifiers.

**Step 5: Inspect entry points**

Run: `python3 -c "import zipfile; z=zipfile.ZipFile('dist/gradience-0.10.0-py3-none-any.whl'); print(z.read('gradience-0.10.0.dist-info/entry_points.txt').decode())"`
Expected: `gradience = gradience.cli:main` and `gradience-bench = gradience.bench.run_bench:main`.

**Step 6: Run tests to confirm nothing broke**

Run: `python3 -m pytest tests/ -q`
Expected: All tests pass (690+, 0 failures).

**Step 7: Final commit**

```bash
git add -A
git commit -m "feat: packaging infrastructure for PyPI 0.10.0 release

- Add README_PYPI.md (condensed PyPI listing)
- Bump version to 0.10.0
- Fix URLs to johntnanney/gradience
- Add classifiers (Developers, Typed) and keywords (merge-audit, compression)
- Add scipy>=1.7.0 to core deps
- Add make build/publish/publish-test targets
- Clean stale dist/ artifacts"
```

Note: This final commit is a squash-style summary. If earlier tasks were committed individually, this step can be skipped — the work is already committed. Only use this if implementing all tasks in a single pass.

---

### Task 6: Publish to TestPyPI (manual verification)

**This is a human-performed step, not automated.**

**Step 1: Publish to TestPyPI**

Run: `make publish-test`
Expected: Upload succeeds. Prints install command.

**Step 2: Verify install from TestPyPI**

Run in a fresh venv:
```bash
python3 -m venv /tmp/test-gradience && source /tmp/test-gradience/bin/activate
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ gradience==0.10.0
python -c "import gradience; print(gradience.__version__)"
gradience --help
deactivate && rm -rf /tmp/test-gradience
```
Expected: Version 0.10.0, help text prints.

**Step 3: If TestPyPI is good, publish for real**

Run: `make publish`
Expected: Package live on PyPI.
