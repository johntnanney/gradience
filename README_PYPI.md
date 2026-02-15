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
