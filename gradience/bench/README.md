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
## Safe uniform baseline (current)

Current policy-validated safe uniform baseline: **r=20**

⚠️ **TASK/MODEL DEPENDENT** — this baseline is calibrated for a specific benchmark setup.
Re-run Bench to calibrate for new model/task combinations.

