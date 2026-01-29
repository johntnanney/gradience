# Development Scripts

⚠️ **These scripts are for local experimentation and are NOT CI tests.**

## Purpose

This directory contains development scripts that were moved from the repository root to maintain clean CI test hygiene. These scripts may:

- Require model downloads (HuggingFace transformers, etc.)
- Take longer to run than typical unit tests
- Test integration scenarios or end-to-end workflows
- Serve as step-by-step demonstrations of features

## Usage Guidelines

### ✅ **Do**
- Use these scripts for local development and experimentation
- Run them manually when testing features or debugging issues
- Modify them as needed for your development workflow
- Use repo-root detection for path resolution:
  ```python
  import os
  from pathlib import Path
  
  # Find repo root dynamically
  repo_root = Path(__file__).parent.parent
  gradience_path = repo_root / "gradience"
  ```

### ❌ **Don't** 
- Include these in CI/CD pipelines
- Hardcode absolute paths like `/Users/john/...`
- Assume they will run in all environments
- Expect them to be as stable as tests in `tests/`

## Path Guidelines

Scripts must NOT hardcode absolute paths. Instead, use dynamic path resolution:

```python
# ❌ Bad - hardcoded path
sys.path.insert(0, '/Users/john/code/gradience')

# ✅ Good - dynamic resolution  
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
```

## CI Test Separation

**Real CI tests belong in `tests/`** - these dev scripts are explicitly excluded from pytest discovery via `pytest.ini` containment.

For production-quality tests, create them in the `tests/` directory following CI testing standards.