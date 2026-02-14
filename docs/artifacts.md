# Artifacts & Evidence

**Understanding Gradience's evidence-based compression artifacts and how to use them for reproducible research.**

## Overview

Gradience is designed as an **evidence-based process**, not just a tool. Every benchmark run generates a complete evidence package that can be:

- **Reproduced** by others using the same config and seed
- **Cited** in papers with full methodological transparency  
- **Attached** to PRs for compression validation
- **Aggregated** across multiple seeds for statistical confidence

This document explains every artifact Gradience produces and how to use them professionally.

## Artifact Hierarchy

Every benchmark run creates this structure:

```
benchmark_output/
├── audit.json                    # Probe adapter analysis
├── compression_configs.json      # Generated compression candidates  
├── bench.json                    # Public benchmark results
├── bench_internal.json           # Internal metrics and debug info
├── probe_r16/                    # Probe adapter artifacts
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── audit.json               # Detailed per-layer analysis
└── compressed_variants/          # Compression results
    ├── energy_p90/
    ├── knee_p90/
    └── per_layer/
```

**Multi-seed aggregation:**
```
multi_seed_experiment/
├── seed_42/
│   ├── bench.json
│   └── audit.json
├── seed_123/
│   ├── bench.json  
│   └── audit.json
├── seed_456/
│   ├── bench.json
│   └── audit.json
├── bench_aggregate.json          # Statistical summary
├── seed_summary.json             # Cross-seed comparisons
└── evidence_package.md           # Human-readable report
```

## Core Artifacts

### audit.json - Spectral Analysis Evidence

**The foundation of all compression decisions.** Contains spectral analysis of the probe adapter revealing rank utilization, energy distribution, and compression opportunities.

#### Structure Overview
```json
{
  "audit_timestamp": "2026-01-15T10:30:00.000000",
  "probe_rank": 16,
  "seed": 42,
  "summary": {
    "total_lora_params": 294912,
    "n_layers": 12,
    "stable_rank_mean": 2.27,           # Average stable rank across layers
    "stable_rank_median": 1.98,
    "effective_rank_mean": 12.12,       # Entropy-based effective rank
    "utilization_mean": 0.142,          # How much of the rank is used
    "energy_rank_90_p50": 8.0,          # Median rank to capture 90% energy
    "energy_rank_90_p90": 10.9          # 90th percentile rank for 90% energy
  },
  "policy_global_suggestions": {
    "energy_90": {
      "uniform_median": 8.0,            # Uniform rank based on median
      "uniform_p90": 10.9,              # Uniform rank based on p90
      "uniform_max": 11.0               # Conservative uniform rank
    }
  },
  "gain": {
    "delta_fro_mean": 0.0086,           # Average Frobenius norm of updates
    "delta_op_mean": 0.0059,            # Average spectral norm of updates
    "energy_concentration_top10pct": 0.182  # How much energy is in top 10% layers
  }
}
```

#### Key Metrics Explained

**Stable Rank**: `||A||_F^2 / ||A||_2^2` - How "low-rank" the learned update is
- **Value < 5**: Very low-rank, aggressive compression possible
- **Value 5-10**: Moderate rank structure, careful compression
- **Value > 10**: Full-rank behavior, conservative compression

**Effective Rank**: Entropy-based measure of rank diversity
- **Lower values**: Concentrated singular values, good for compression
- **Higher values**: Uniform singular values, harder to compress

**Energy Rank 90**: Minimum rank to capture 90% of update energy
- **Direct compression target**: Use this rank for energy-based policies
- **90th percentile**: Conservative estimate across layers

**Utilization**: `stable_rank / probe_rank` - How efficiently the probe rank is used
- **< 0.3**: Significant over-parameterization, compress aggressively
- **0.3-0.7**: Moderate waste, standard compression  
- **> 0.7**: Efficient usage, conservative compression

#### Per-Layer Analysis

When using `--layers` flag, audit.json includes detailed per-module breakdowns:

```json
{
  "per_module_gain": [
    {
      "module": "base_model.model.distilbert.transformer.layer.0.attention.q_lin",
      "layer": 0,
      "r": 16,
      "stable_rank": 2.34,
      "effective_rank": 11.8,
      "utilization": 0.146,
      "delta_sigma_max": 0.0062,         # Largest singular value  
      "delta_frob_norm": 0.0089,         # Frobenius norm
      "energy_rank_90": 9,               # Rank for 90% energy
      "suggested_r": 8,                  # Recommended compressed rank
      "compression_ratio": 0.5           # Parameter reduction
    }
  ]
}
```

#### Decision Traces

Audit includes **algorithmic decision traces** for transparency:

```json
{
  "decision_trace": {
    "energy_policy": {
      "threshold": 0.90,                # 90% energy target
      "per_layer_ranks": [9, 8, 10, 7], # Per-layer energy ranks
      "uniform_strategy": "p90",         # Use 90th percentile
      "final_rank": 10,                 # Selected uniform rank
      "reasoning": "Conservative p90 selection for safety"
    }
  }
}
```

### compression_configs.json - Generated Candidates

**The compression strategies generated from audit analysis.** Each candidate represents a different compression approach with specific rank patterns and expected performance.

#### Structure
```json
{
  "energy_p90": {
    "variant": "energy_p90",           # Policy identifier
    "suggested_r": 10.9,               # Original suggestion (may be non-integer)
    "actual_r": 12,                    # Clamped to allowed ranks
    "rank_pattern": {},                # Per-layer ranks (empty = uniform)
    "alpha_pattern": {},               # Per-layer alphas (empty = use r)
    "config": {
      "alpha": 12,                     # LoRA scaling factor
      "dropout": 0.05,
      "probe_r": 12,                   # Uniform rank
      "target_modules": ["q_lin", "v_lin"]
    },
    "status": "ready",                 # ready | skipped | failed
    "policy_type": "energy",           # energy | knee | erank | uniform
    "conservatism_score": 3.0          # Higher = more conservative
  },
  "per_layer": {
    "variant": "per_layer",
    "suggested_r": 6.67,               # Average across layers
    "actual_r": 8,                     # Average actual rank
    "rank_pattern": {                  # Layer-specific ranks
      "base_model.model.distilbert.transformer.layer.2.attention.v_lin": 8,
      "base_model.model.distilbert.transformer.layer.5.attention.q_lin": 4
    },
    "alpha_pattern": {                 # Layer-specific alphas
      "base_model.model.distilbert.transformer.layer.2.attention.v_lin": 8,
      "base_model.model.distilbert.transformer.layer.5.attention.q_lin": 4  
    },
    "status": "ready",
    "policy_type": "per_layer",
    "conservatism_score": 2.5          # More aggressive than uniform
  }
}
```

#### Policy Types

**energy_p90**: Energy-based rank selection at 90th percentile
- **Basis**: Captures 90% of adapter energy with minimal rank
- **Use case**: Balanced compression with quality preservation

**knee_p90**: Elbow detection in singular value spectrum
- **Basis**: Finds "knee" where singular values drop sharply
- **Use case**: Natural compression point based on spectral structure

**erank_p90**: Effective rank at 90th percentile
- **Basis**: Entropy-based rank estimation
- **Use case**: Information-theoretic compression

**per_layer**: Layer-specific rank allocation
- **Basis**: Different ranks per layer based on individual analysis
- **Use case**: Maximum parameter efficiency

#### Compression Metrics
```json
{
  "compression_metrics": {
    "parameter_reduction": 0.61,       # 61% fewer parameters
    "rank_reduction": 0.50,            # 50% rank reduction (16→8)
    "expected_speedup": 1.38,          # Estimated inference speedup
    "memory_reduction": 0.61           # Memory usage reduction
  }
}
```

### bench.json - Public Benchmark Results

**The canonical result file for sharing and citation.** Contains all information needed to reproduce and validate the benchmark.

#### Structure
```json
{
  "bench_version": "0.1",             # Schema version
  "timestamp": "2026-01-15T10:30:00.000000",
  "git_commit": "abc123...",          # Code version used
  "model": "distilbert-base-uncased",
  "task": "sst2/default",
  "status": "COMPLETED",              # COMPLETED | UNDERTRAINED | FAILED
  
  "env": {
    "python_version": "3.11.0",
    "torch_version": "2.1.0",
    "transformers_version": "4.35.0",
    "platform": "Linux-5.15.0-x86_64",
    "cuda_available": true,
    "cuda_version": "12.1",
    "gpu_devices": ["NVIDIA RTX 4090"]
  },
  
  "probe_quality_gate": {
    "metric_key": "eval_accuracy",
    "metric_value": 0.847,             # Achieved accuracy
    "min_value": 0.75,                 # Required threshold
    "passed": true                     # Quality gate passed
  },
  
  "probe": {
    "rank": 16,
    "params": 294912,
    "accuracy": 0.847,
    "training_time_seconds": 180.5
  },
  
  "compressed": {
    "energy_p90": {
      "rank": 12,
      "params": 221184,
      "accuracy": 0.843,               # -0.4% delta
      "compression_ratio": 0.75,       # 25% parameter reduction
      "training_time_seconds": 145.2,
      "verdict": "PASSED",             # PASSED | FAILED
      "delta_accuracy": -0.004         # Accuracy loss
    }
  },
  
  "summary": {
    "probe_quality": "PASSED",
    "recommendations_validated": "2/3", # 2 out of 3 variants passed
    "best_compression": {
      "variant": "energy_p90",
      "compression_ratio": 0.75,
      "accuracy_delta": -0.004
    },
    "verdict": "✅ Quality-preserving compression validated",
    "notes": [
      "energy_p90: 25% compression with -0.4% accuracy loss",
      "per_layer: 39% compression with -1.2% accuracy loss (PASSED)",
      "knee_p90: 45% compression with -3.1% accuracy loss (FAILED)"
    ]
  }
}
```

### bench_internal.json - Debug Information

**Extended metrics and debug information.** Used for development and detailed analysis.

#### Additional Content
```json
{
  "training_curves": {
    "probe": {
      "train_loss": [2.3, 1.8, 1.2, 0.9, 0.7],
      "eval_accuracy": [0.52, 0.67, 0.79, 0.84, 0.85],
      "steps": [100, 200, 300, 400, 500]
    }
  },
  
  "timing_breakdown": {
    "model_loading": 15.2,
    "data_preprocessing": 8.7,
    "probe_training": 180.5,
    "audit_analysis": 12.3,
    "compression_training": 435.8,
    "evaluation": 45.2
  },
  
  "memory_usage": {
    "peak_gpu_memory_gb": 8.4,
    "peak_cpu_memory_gb": 12.1,
    "model_size_gb": 0.25
  },
  
  "validation_details": {
    "probe_validation_accuracy": 0.851,
    "probe_test_accuracy": 0.847,
    "compression_validation_accuracy": 0.846,
    "compression_test_accuracy": 0.843,
    "overfitting_gap": 0.003         # Low gap = good generalization
  }
}
```

## Multi-Seed Aggregation

### bench_aggregate.json - Statistical Summary

**Cross-seed statistical analysis for research-grade validation.**

#### Structure
```json
{
  "bench_version": "0.1",
  "aggregate_timestamp": "2026-01-15T12:00:00.000000", 
  "model": "distilbert-base-uncased",
  "task": "glue/sst2",
  "policy": "Safe Uniform Baseline v0.1",
  "aggregation_type": "multi_seed",
  
  "runs": [
    {
      "path": "seed_42/bench.json",
      "seed": 42,
      "accuracy_delta": -0.004,        # -0.4% loss
      "compression_ratio": 0.75,       # 25% compression
      "variant": "energy_p90"
    },
    {
      "path": "seed_123/bench.json", 
      "seed": 123,
      "accuracy_delta": -0.006,        # -0.6% loss
      "compression_ratio": 0.75,       # Same compression
      "variant": "energy_p90"
    },
    {
      "path": "seed_456/bench.json",
      "seed": 456, 
      "accuracy_delta": -0.003,        # -0.3% loss
      "compression_ratio": 0.75,
      "variant": "energy_p90"
    }
  ],
  
  "policy_compliance": {
    "status": "COMPLIANT",             # All seeds passed
    "pass_rate": "100% (3/3)",
    "worst_case": -0.006,              # Worst accuracy loss
    "threshold": -0.025,               # Policy threshold (-2.5%)
    "details": "All seeds passed accuracy tolerance"
  },
  
  "statistical_summary": {
    "mean_accuracy_delta": -0.0043,   # Mean across seeds
    "std_accuracy_delta": 0.0015,     # Standard deviation  
    "confidence_interval_95": [-0.0058, -0.0028], # 95% CI
    "effect_size_cohens_d": -0.87,    # Effect size
    "compression_consistency": 1.0     # All seeds same compression
  },
  
  "best_compression": {
    "variant": "energy_p90",
    "compression_ratio": 0.75,
    "mean_accuracy_delta": -0.0043,
    "reproducibility_score": 0.95     # High reproducibility
  }
}
```

### seed_summary.json - Cross-Seed Analysis

**Detailed comparison of results across seeds.**

```json
{
  "seed_comparison": {
    "probe_accuracy": {
      "seed_42": 0.847,
      "seed_123": 0.851, 
      "seed_456": 0.845,
      "mean": 0.848,
      "std": 0.003,
      "coefficient_of_variation": 0.004  # Low = consistent
    },
    
    "compression_accuracy": {
      "energy_p90": {
        "seed_42": 0.843,
        "seed_123": 0.845,
        "seed_456": 0.842,
        "mean": 0.843,
        "std": 0.0015
      }
    },
    
    "rank_suggestions": {
      "energy_p90_uniform": {
        "seed_42": 12,
        "seed_123": 11, 
        "seed_456": 12,
        "mode": 12,                     # Most common suggestion
        "agreement_rate": 0.67          # 2/3 seeds agree
      }
    }
  },
  
  "reproducibility_metrics": {
    "probe_variance": 0.000009,        # Very low variance
    "compression_variance": 0.000003,  
    "rank_suggestion_stability": 0.67, 
    "overall_reproducibility": 0.89    # High reproducibility
  }
}
```

## Evidence Package Generation

### Automatic Evidence Reports

Gradience automatically generates human-readable evidence packages:

```markdown
# Evidence Package: DistilBERT SST-2 Compression

## Summary
- **Model**: distilbert-base-uncased
- **Task**: GLUE SST-2 sentiment classification
- **Seeds**: 42, 123, 456 (3 seeds)
- **Status**: ✅ VALIDATED

## Results
- **Best compression**: energy_p90 (25% parameter reduction)
- **Quality impact**: -0.43% accuracy (±0.15%, 95% CI)
- **Reproducibility**: 95% (high consistency across seeds)

## Evidence Files
- `bench_aggregate.json`: Statistical summary
- `seed_*/bench.json`: Per-seed results  
- `seed_*/audit.json`: Spectral analysis
- `compression_configs.json`: Generated candidates

## Reproduction
```bash
gradience-bench --config evidence/distilbert_sst2.yaml \
                --output reproduction_attempt \
                --ci
```

## Citation
If using these results, cite: [DOI or GitHub release]
```

## Professional Use Cases

### 1. Research Paper Citation

**In methods section:**
```
LoRA compression was performed using Gradience v0.9.0 [cite]. We used 
the energy_p90 policy which selects uniform ranks based on the 90th 
percentile of per-layer energy requirements. Compression candidates 
were validated on 3 random seeds (42, 123, 456) with a quality 
threshold of -2.5% accuracy loss.
```

**In results:**
```
The energy_p90 policy achieved 25% parameter reduction with 
-0.43% accuracy loss (±0.15%, 95% CI, n=3 seeds). Statistical 
analysis showed high reproducibility (95% consistency score).
See supplementary materials for complete evidence package.
```

### 2. Pull Request Validation

**PR Description Template:**
```markdown
## LoRA Compression Validation

This PR implements rank-16 LoRA training. Compression validation:

**Results Summary:**
- ✅ energy_p90: 25% compression, -0.4% accuracy 
- ✅ per_layer: 39% compression, -1.2% accuracy
- ❌ knee_p90: 45% compression, -3.1% accuracy (exceeds -2.5% threshold)

**Recommendation:** Deploy energy_p90 variant for 25% memory savings.

**Evidence Package:** [Link to bench.json and audit.json files]

**Reproduction:**
```bash
gradience-bench --config pr_validation.yaml --output validation_results --ci
```
```

### 3. Model Release Documentation

**Model card addition:**
```yaml
compression_evidence:
  validated_by: "Gradience v0.9.0"
  evidence_package: "https://github.com/org/model/releases/tag/v1.0.0/evidence.zip"
  validation_summary:
    compression_ratio: 0.75
    quality_impact: "-0.43% accuracy (±0.15%, 95% CI)"
    seeds_tested: [42, 123, 456] 
    reproducibility_score: 0.95
  reproduction_command: |
    gradience-bench --config model_validation.yaml --output reproduction --ci
```

### 4. Continuous Integration

**CI pipeline integration:**
```yaml
- name: Validate LoRA Compression
  run: |
    gradience-bench --config ci_validation.yaml \
                    --output ${{ github.sha }}_validation \
                    --ci
    
    # Upload evidence to artifacts
    tar czf evidence_${{ github.sha }}.tar.gz \
        ${{ github.sha }}_validation/bench.json \
        ${{ github.sha }}_validation/audit.json
        
- name: Archive Evidence
  uses: actions/upload-artifact@v3
  with:
    name: compression-evidence-${{ github.sha }}
    path: evidence_${{ github.sha }}.tar.gz
```

## Artifact Validation

### Integrity Checks

**Validate evidence package completeness:**
```python
import json
from pathlib import Path

def validate_evidence_package(benchmark_dir: Path) -> bool:
    """Validate that evidence package is complete."""
    required_files = [
        "bench.json",
        "audit.json", 
        "compression_configs.json"
    ]
    
    for file in required_files:
        if not (benchmark_dir / file).exists():
            print(f"❌ Missing {file}")
            return False
    
    # Validate schema versions match
    with open(benchmark_dir / "bench.json") as f:
        bench_data = json.load(f)
    
    if bench_data.get("bench_version") != "0.1":
        print("❌ Schema version mismatch")
        return False
        
    print("✅ Evidence package validation passed")
    return True
```

**Cross-check audit and bench consistency:**
```python
def validate_audit_bench_consistency(benchmark_dir: Path) -> bool:
    """Ensure audit and bench results are consistent."""
    with open(benchmark_dir / "audit.json") as f:
        audit = json.load(f)
    with open(benchmark_dir / "bench.json") as f:
        bench = json.load(f)
    
    # Check probe rank consistency
    audit_rank = audit["probe_rank"] 
    bench_rank = bench["probe"]["rank"]
    
    if audit_rank != bench_rank:
        print(f"❌ Probe rank mismatch: audit={audit_rank}, bench={bench_rank}")
        return False
        
    print("✅ Audit-bench consistency validated")
    return True
```

## Schema Evolution

### Version Compatibility

Gradience maintains **semantic versioning** for evidence schemas:

- **Major version change** (0.x → 1.x): Breaking schema changes
- **Minor version change** (0.1 → 0.2): Backward-compatible additions  
- **Patch version change** (0.1.0 → 0.1.1): Bug fixes, no schema change

### Migration Support

Future versions will include migration utilities:

```bash
# Migrate evidence from v0.1 to v0.2
gradience migrate-evidence --from v0.1 --to v0.2 --dir old_evidence/

# Validate evidence compatibility
gradience validate-evidence --min-version v0.1 --dir evidence_package/
```

## Troubleshooting

### Missing Artifacts

**Partial evidence packages:**
```bash
# Check what artifacts exist
find benchmark_output/ -name "*.json" -type f

# Regenerate missing audit
gradience audit --peft-dir benchmark_output/probe_r16 --json > audit_recovered.json

# Validate benchmark completion
grep '"status"' benchmark_output/bench.json
```

### Schema Validation Errors

**Check schema compatibility:**
```python
import json

def check_schema_version(artifact_path):
    with open(artifact_path) as f:
        data = json.load(f)
    
    version = data.get("bench_version") or data.get("audit_version", "unknown")
    print(f"Schema version: {version}")
    
    if version != "0.1":
        print("⚠️ Schema version mismatch - may need migration")
```

### Reproducibility Issues

**Debug seed differences:**
```bash
# Compare audit results across seeds
diff -u seed_42/audit.json seed_123/audit.json

# Check for environment differences  
diff -u seed_42/bench.json seed_123/bench.json | grep '"env"' -A 20
```

The artifact system transforms Gradience from a compression tool into a **reproducible research process** with full evidence traceability.

## Merge Compatibility Artifacts

### merge_audit.json - Adapter Merge Compatibility Report

**Spectral compatibility analysis between two PEFT LoRA adapters.** Produced by `gradience merge-audit` to assess whether two adapters can be safely merged, and which merge strategy to use.

#### Structure Overview
```json
{
  "schema": "gradience.merge_audit/v1",
  "timestamp": "2026-02-13T10:30:00Z",
  "adapter_a": {
    "path": "./adapter_a",
    "base_model": "mistralai/Mistral-7B-v0.1",
    "rank": 16,
    "alpha": 16,
    "n_layers": 32
  },
  "adapter_b": {
    "path": "./adapter_b",
    "base_model": "mistralai/Mistral-7B-v0.1",
    "rank": 8,
    "alpha": 8,
    "n_layers": 32
  },
  "matching": {
    "shared": 32,
    "only_a": 0,
    "only_b": 0
  },
  "aggregate": {
    "overall_verdict": "safe",
    "compatibility_score": 0.142,
    "mean_overlap": 0.142,
    "max_overlap": 0.387,
    "mean_agreement": 0.056,
    "n_safe": 30,
    "n_redundant": 1,
    "n_conflicting": 0,
    "n_imbalanced": 1
  },
  "per_layer": [
    {
      "layer_name": "model.layers.0.self_attn.q_proj",
      "module_type": "attention",
      "verdict": "safe",
      "confidence": 0.85,
      "recommendation": "Orthogonal subspaces...",
      "suggested_strategy": "linear",
      "suggested_coefficients": [0.5, 0.5],
      "metrics": {
        "mean_overlap": 0.087,
        "max_overlap": 0.213,
        "directional_agreement": 0.034,
        "magnitude_ratio": 1.23,
        "effective_rank_a": 4,
        "effective_rank_b": 3
      }
    }
  ],
  "recommendations": [
    "Adapters are spectrally compatible. Linear merge (equal coefficients) should preserve both signals."
  ],
  "thresholds": {
    "low_overlap": 0.2,
    "high_overlap": 0.5,
    "aligned": 0.5,
    "conflicting": -0.3,
    "imbalanced": 5.0
  }
}
```

#### Key Metrics Explained

**Mean Overlap**: Average cosine of principal angles between adapter subspaces [0, 1]
- **< 0.2**: Orthogonal subspaces, safe to merge with any method
- **0.2 - 0.5**: Moderate interaction, standard merge methods work
- **> 0.5**: Significant shared subspace, check directional agreement

**Directional Agreement**: Projection cosine similarity [-1, 1]
- **> 0.5**: Aligned directions (same effect) — REDUNDANT, use TIES to deduplicate
- **-0.3 to 0.5**: Neutral — SAFE
- **< -0.3**: Opposing directions — CONFLICTING, merging causes cancellation

**Magnitude Ratio**: Frobenius norm ratio of larger to smaller adapter, >= 1
- **< 5.0**: Balanced — equal merge coefficients work
- **> 5.0**: IMBALANCED — weaker adapter drowned out, use rebalanced coefficients

**Compatibility Score**: Energy-weighted mean overlap across all layers [0, 1]
- **0.0**: Fully orthogonal adapters (ideal for merging)
- **1.0**: Fully overlapping adapters (redundant)

#### Verdict Decision Tree

1. Low overlap → **SAFE** (orthogonal subspaces)
2. High overlap + aligned → **REDUNDANT** (de-dup needed via TIES)
3. High overlap + opposing → **CONFLICTING** (danger zone — use DARE or exclude)
4. Extreme magnitude ratio → **IMBALANCED** (coefficient tuning needed)
5. Moderate / ambiguous → **SAFE** with moderate confidence

### merge_audit.md - Human-Readable Merge Report

**Markdown report** with formatted tables, recommendations, and per-layer analysis. Suitable for PR descriptions, documentation, and team review.

Contains:
- Adapter metadata comparison (base model, rank, alpha, target modules)
- Aggregate compatibility summary
- Per-layer verdict table with overlap, agreement, and strategy columns
- Actionable recommendations
- Warnings for mismatched base models or target modules