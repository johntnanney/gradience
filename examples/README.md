# Gradience Examples

This directory contains example artifacts and fixtures to help you understand Gradience's data formats and get started quickly.

## 📁 Directory Structure

```
examples/
├── adapters/                  # Sample LoRA adapters
│   └── tiny_lora/            # Minimal LoRA for testing
├── bench_artifacts/           # Bench output examples
│   ├── bench.json            # Single-seed bench result
│   ├── bench_aggregate.json  # Multi-seed aggregated result
│   └── bench.md              # Human-readable report
├── telemetry/                 # JSONL telemetry examples
│   ├── training_log.jsonl    # Training telemetry stream
│   └── callback_output.jsonl # HuggingFace callback output
└── configs/                   # Example configuration files
    ├── smoke_gsm8k.yaml      # Quick GSM8K bench config
    └── glue_sst2.yaml        # GLUE SST-2 bench config
```

## 🚀 Quick Start

### Audit an Adapter
```bash
# Audit the tiny example adapter
gradience audit --peft-dir examples/adapters/tiny_lora/

# With rank suggestions
gradience audit --peft-dir examples/adapters/tiny_lora/ --layers --suggest-per-layer
```

### Run a Smoke Bench
```bash
# Quick GSM8K bench (~5 minutes)
gradience-bench \
    --config examples/configs/smoke_gsm8k.yaml \
    --output bench_output/
```

### Inspect Artifacts
```bash
# View bench result structure
cat examples/bench_artifacts/bench.json | jq .

# View telemetry format
head -5 examples/telemetry/training_log.jsonl
```

## 📋 Artifact Schemas

### bench.json Schema
See `examples/bench_artifacts/bench.json` for the complete structure including:
- Environment metadata (git, python, torch versions)
- Model and dataset revision tracking  
- Probe and compression results
- Complete config embedding
- Primary metric identification

### Telemetry JSONL Schema  
See `examples/telemetry/training_log.jsonl` for event streaming format:
```json
{"timestamp": "2024-01-01T00:00:00Z", "event": "training_start", "data": {...}}
{"timestamp": "2024-01-01T00:00:01Z", "event": "step_metrics", "data": {...}}
```

## 🎯 Use Cases

### Testing Your Changes
- Use `examples/adapters/tiny_lora/` for quick audit testing
- Use `examples/configs/smoke_gsm8k.yaml` for bench pipeline testing
- Compare your bench output against `examples/bench_artifacts/bench.json`

### Understanding Output Formats
- Study `examples/bench_artifacts/` to understand bench report structure
- Review `examples/telemetry/` to see telemetry event formats
- Use as templates for your own artifacts

### Integration Testing
- Copy example configs and modify for your use case
- Use example adapters as baseline for comparison
- Validate your pipeline against known-good artifacts

---

**Note**: These are minimal examples for testing and understanding. For production use, see the full documentation and real model examples.