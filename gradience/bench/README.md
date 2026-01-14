# Gradience Bench (v0.1)

Bench is a minimal validation harness for Gradience recommendations.

**Goal:** turn "suggested rank" into a testable claim by running:

1) Train probe adapter (high-rank)
2) Audit -> get compression suggestions
3) Retrain with compressed ranks
4) Eval and compare
5) Emit a report (JSON + Markdown)

## Setup

**Prerequisites**: Ensure you have installed Gradience from the repo root directory to avoid import issues:

```bash
# From the repo root (directory containing pyproject.toml)
git clone https://github.com/johntnanney/gradience.git
cd gradience

# Option A: Use the helper script (recommended)
./scripts/setup_venv.sh

# Option B: Manual installation
pip install -e .
```

> **⚠️ Important:** Always install from the repo root (not from `gradience/gradience/`). The helper script automatically validates this for you.

**Quick test:**
```bash
python -m gradience.bench.run_bench --help
```

## Smoke Mode

Bench supports a **smoke mode** (`--smoke`) for fast pipeline validation:

```bash
python -m gradience.bench.run_bench --config configs/distilbert_sst2.yaml --output smoke_test --smoke
```

**Purpose:** Smoke mode is intended to validate the pipeline end-to-end; probe quality gates are reported as `UNDERTRAINED_SMOKE` and do not indicate a failure of Bench itself. This mode uses reduced training steps for faster testing and is not suitable for certification.

**Key Behaviors:**
- Uses `smoke_max_steps`, `smoke_train_samples`, `smoke_eval_samples` from config
- Probe failures result in `UNDERTRAINED_SMOKE` status (exit code 0)
- Results are automatically excluded from aggregation by default
- Use `--include-smoke` flag in aggregation scripts to include smoke runs explicitly

## Safe Uniform Baseline Policy

The **Safe Uniform Baseline** is Gradience's default compression recommendation policy:
- **Pass Rate**: ≥67% of seeds must pass accuracy tolerance (Δ ≥ -2.5%)
- **Worst-Case**: The worst-performing seed must have Δ ≥ -2.5%

### Current Validated Baselines (DistilBERT/SST-2)

**Certifiable v0.1 Results** (3 seeds × 500 steps):
- **uniform_median**: 61% compression, 100% pass rate, worst Δ = -1.0% ✅ POLICY COMPLIANT

**⚠️ TASK/MODEL DEPENDENT**: These baselines are empirically validated for DistilBERT on SST-2. Always validate on your specific task/model combination before production deployment.

### Control Variants

The `uniform_p90_control` variant is automatically skipped when the suggested 90th percentile rank equals the probe rank (no compression possible). Output clearly labels this as:
```
"Control run: suggested rank r=32 equals probe rank (no compression)"
```

**Full policy documentation**: [VALIDATION_POLICY.md](/VALIDATION_POLICY.md)  
**Machine-consumable policy**: `gradience/bench/policies/safe_uniform.yaml`

## Outsider Drill - First-Class Acceptance Test

For release validation, use the **Outsider Drill** to simulate a fresh external user experience:

```bash
# Full acceptance test (smoke + certification)
./scripts/outsider_bench_drill.sh v0.3.5

# Custom output directory  
./scripts/outsider_bench_drill.sh v0.3.5 /tmp/my_test_output

# Fast smoke-only validation (for CI/development)
./scripts/outsider_bench_drill.sh v0.3.5 /tmp/smoke_test true
```

**What it validates:**
- **Fresh Environment**: Clean clone, fresh venv, correct installation from source
- **Smoke Mode**: Exit code 0 even with undertrained probe (`UNDERTRAINED_SMOKE` status)  
- **Certification Mode**: Multi-seed validation + aggregate report generation
- **Aggregation Safety**: Smoke runs excluded from cert aggregation by default
- **Artifact Packaging**: Timestamped tarball with all results

**Expected Final Line:**
```
✅ ACCEPTANCE TEST: PASSED
   Gradience v0.3.5 is ready for external users
```

Use this before any release to ensure Gradience provides a smooth experience for external users without tribal knowledge.

## Status

This directory is the **v0.1 scaffold** (layout + config + reporting utilities).
The actual train/audit/retrain protocol wiring lands in later commits.

## Layout

```
gradience/bench/
├── run_bench.py
├── configs/
│   └── distilbert_sst2*.yaml
├── policies/
│   ├── safe_uniform.yaml      # Machine-consumable policy export
│   └── README.md
├── protocol.py
├── report.py
└── README.md
```
## Reference Results

📁 **Frozen reference results available at:** `gradience/bench/results/distilbert_sst2_v0.1/`

This directory contains canonical aggregate results that can be cited:
- `bench_aggregate.json` - Machine-readable 3-seed aggregate
- `bench_aggregate.md` - Human-readable report
- `env.txt` - Complete environment metadata
- `CITATION.txt` - Ready-to-use citation

**Key Results:**
- Model: DistilBERT (distilbert-base-uncased)
- Task: SST-2 sentiment classification  
- Compression: uniform_median (61% parameter reduction)
- Policy compliance: COMPLIANT (100% pass rate, 3/3 seeds)
- Worst-case accuracy delta: -1.0%

To reproduce: `./scripts/freeze_reference_results.sh v0.1 cpu`

## Safe uniform baseline (current)

Current policy-validated safe uniform baseline: **r=20**

⚠️ **TASK/MODEL DEPENDENT** — this baseline is calibrated for a specific benchmark setup.
Re-run Bench to calibrate for new model/task combinations.

