# Gradience

**Spectral metrics as empirical probes into the geometry of LoRA training.**

Gradience is a Python library for studying the geometry of LoRA (Low-Rank Adaptation) fine-tuning. It measures rank evolution, detects phase transitions, and studies how spectral structure relates to generalization — with reproducible, multi-seed experimental infrastructure.

## What Gradience does

- **Spectral measurements** — Per-layer SVD analysis yielding stable rank, energy concentration, utilization ratios, and rank waste quantification
- **Training telemetry** — Structured JSONL recording of spectral evolution across training steps
- **Merge compatibility analysis** — Principal angle and directional agreement measurements between adapter pairs
- **Reproducible benchmarking** — Multi-seed experimental infrastructure with statistical aggregation and tolerance-based validation

## Who it's for

- **ML researchers** studying training dynamics via spectral measurements
- **Researchers comparing adaptation strategies** with reproducible, statistically rigorous evidence
- **Practitioners** translating spectral insights into validated compression configurations

## Quick install

```bash
pip install gradience                    # Core (torch, safetensors, scipy)
pip install "gradience[hf]"              # + HuggingFace integration
pip install "gradience[bench]"           # + Full benchmarking suite
```

## 60-second example

```bash
# Verify installation
gradience verify

# Audit a LoRA adapter
gradience audit --peft-dir ./my-adapter --layers

# Check compatibility of two adapters for merging
gradience merge-audit --adapter-a ./adapter-1 --adapter-b ./adapter-2
```

## Next steps

- [Getting Started](getting-started/index.md) — Installation, quick start, and first audit
- [User Guide](guide/index.md) — CLI reference, configuration, and output formats
- [API Reference](api/index.md) — Python API, types, and exceptions
- [Architecture](architecture/index.md) — Design principles and module organization

## Citation

```bibtex
@software{gradience2026,
  title = {Gradience: Spectral Analysis of Low-Rank Adaptation Dynamics},
  author = {Nanney, John T.},
  year = {2026},
  url = {https://github.com/johntnanney/gradience},
  note = {Version 0.11.0}
}
```
