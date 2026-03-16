# Installation

## Requirements

- **Python**: 3.10 or later (3.10, 3.11, 3.12 tested in CI)
- **PyTorch**: 2.0+ (installed automatically with the core package)

## Install from PyPI

### Core package

The core package includes spectral analysis, merge audit, and LoRA weight math:

```bash
pip install gradience
```

Core dependencies: `torch`, `safetensors`, `scipy`, `numpy`, `pyyaml`, `rich`, `packaging`

### With HuggingFace integration

Adds `transformers`, `peft`, and `sentencepiece` for training callbacks:

```bash
pip install "gradience[hf]"
```

### Full benchmarking suite

Adds `datasets`, `evaluate`, `scikit-learn`, `pandas`, and `accelerate` for running the complete bench protocol:

```bash
pip install "gradience[bench]"
```

### All extras

```bash
pip install "gradience[all]"
```

## Install from source

```bash
git clone https://github.com/johntnanney/gradience.git
cd gradience
pip install -e ".[dev]"
```

Or use the Makefile:

```bash
make setup        # Creates .venv, installs [hf,dev] extras
source .venv/bin/activate
```

## Verify installation

```bash
gradience verify
```

This checks that all core dependencies are importable and the CLI is functional.

## Optional: configure caching

Gradience and its dependencies (HuggingFace, torch) can download large files. To control where caches are stored:

```bash
make setup-cache
```

See [Storage & Caching](../guide/storage.md) for details.

## Extras summary

| Extra | What it adds | Use case |
|-------|-------------|----------|
| `[hf]` | transformers, peft, sentencepiece | HF Trainer callback |
| `[bench]` | datasets, evaluate, scikit-learn, pandas | Full benchmarking protocol |
| `[dev]` | pytest, ruff, mypy, pre-commit | Contributing to Gradience |
| `[fast]` | hf_transfer | Faster HuggingFace downloads |
| `[all]` | Everything above | Full functionality |
