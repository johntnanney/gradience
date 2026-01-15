# DistilBERT/SST-2 Reference Results

This directory contains frozen reference results for DistilBERT fine-tuning on SST-2
using Gradience compression recommendations.

## Files

- `bench_aggregate.json` - Machine-readable aggregate results
- `bench_aggregate.md` - Human-readable aggregate report  
- `bench_seed{42,123,456}.json` - Individual seed results (when available)
- `env.txt` - Complete environment information
- `README.md` - This file

## Key Results

The aggregate results demonstrate:
- 3-seed certification with policy compliance
- Compression ratio and accuracy preservation
- Protocol invariant verification

## Usage

These results serve as a versioned reference that can be cited:

```
Gradience DistilBERT/SST-2 Benchmark v0.1
Policy-compliant compression: 61% (uniform_median)
Pass rate: 100% (3/3 seeds)
Worst-case accuracy delta: -1.0%
```

## Reproducibility

To reproduce these results:
1. Check out the same git commit (see env.txt)
2. Install the same dependency versions (see env.txt)
3. Run: `./scripts/freeze_reference_results.sh v0.1 cpu`

## Caveats

These results are specific to:
- Model: DistilBERT (distilbert-base-uncased)
- Task: SST-2 sentiment classification
- Hardware: See env.txt for device details

Results may vary on different hardware or with different dependency versions.