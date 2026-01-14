# Bench v0.1 Certifiable Standard

The canonical benchmark specification for public validation claims and blog post references.

## Specification

**Fixed Parameters:**
- **Model/Task**: DistilBERT + SST-2 (GLUE)
- **Probe rank**: r=32
- **Training budget**: 500 steps, 2000 train samples, 500 eval samples
- **Seeds**: 3 canonical seeds (42, 123, 456) for statistical robustness
- **Pass criterion**: ≥67% seeds PASS AND worst-seed Δ ≥ -2.5%

**Compression Variants Tested:**
- **`per_layer`** - Adaptive per-layer rank optimization (primary recommendation)
- **`uniform_r20`** - Safe uniform baseline (empirically validated as safe)
- **`uniform_r24`** - Conservative uniform baseline (optional, ultra-safe)

## Usage

### Individual seed runs (recommended for parallel execution):

```bash
# Seed 42 (canonical seed 1/3)
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/distilbert_sst2_certifiable_seed42.yaml \
  --output bench_runs/cert_v0.1_seed42 --ci

# Seed 123 (canonical seed 2/3)  
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/distilbert_sst2_certifiable_seed123.yaml \
  --output bench_runs/cert_v0.1_seed123 --ci

# Seed 456 (canonical seed 3/3)
python -m gradience.bench.run_bench \
  --config gradience/bench/configs/distilbert_sst2_certifiable_seed456.yaml \
  --output bench_runs/cert_v0.1_seed456 --ci
```

### Batch execution:

```bash
# Run the convenience script for all 3 seeds
./scripts/run_certifiable_v0.1.sh
```

### Analysis:

```bash
# Aggregate results across all 3 seeds
python scripts/analyze_safe_baselines.py \
  --certifiable-v01 bench_runs/cert_v0.1_seed*
```

## Validation Results

This benchmark validates the following safety policy decisions:

- ✅ **per_layer adaptive compression** - Primary recommendation for efficiency
- ✅ **uniform r=20** - Safe uniform baseline (25% compression vs r=32 probe)
- ✅ **uniform r=24** - Conservative uniform baseline (16.6% compression)
- ❌ **uniform r=16** - **Unsafe** (0% pass rate, -8% worst-case accuracy drop)

## Purpose

The Bench v0.1 Certifiable standard serves as:

1. **Public reference** - Citable benchmark for blog posts and papers
2. **Regression protection** - Baseline for detecting policy drift in future releases
3. **Statistical confidence** - Multi-seed results provide defensible statistical evidence
4. **Reproducibility** - Fixed specification enables independent validation

## Quality Level

**Validation Level**: **Certifiable**
- ✅ Multi-seed statistical robustness (3 seeds)
- ✅ Full training budget (500 steps)
- ✅ Realistic dataset size (2000 train samples)
- ✅ Policy-compliant pass criteria (≥67% pass rate, ≥-2.5% worst delta)
- ✅ Regression test protection

## Machine-Consumable Policy

The safety decisions from this benchmark are exported in machine-consumable format:
- **Policy file**: `gradience/bench/policies/safe_uniform.yaml`
- **Regression tests**: `tests/test_bench/test_policy_regression.py`

## Integration

This benchmark is integrated into the release engineering process:
- **Pre-release validation**: `./scripts/pre_release_bench_check.sh`
- **Release checklist**: `RELEASE_CHECKLIST.md`

The certifiable benchmark ensures that safety decisions are never accidentally regressed in future releases.