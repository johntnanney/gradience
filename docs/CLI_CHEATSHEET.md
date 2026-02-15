# Gradience CLI Cheat Sheet

## Most Common Commands (Copy & Paste)

### Post-training analysis

```bash
# Quick summary of training dynamics
gradience monitor <output_dir>/run.jsonl

# Detailed analysis with diagnostic signals
gradience monitor <output_dir>/run.jsonl --verbose

# Spectral analysis of LoRA adapter (if using PEFT)
gradience audit --peft-dir <output_dir>/adapter --layers

# Complete analysis (audit + monitor)
gradience audit --peft-dir <output_dir>/adapter --append <output_dir>/run.jsonl
gradience monitor <output_dir>/run.jsonl --verbose
```

### Experiment validation

```bash
# Validate config before launching a run
gradience check --task <task_type> --peft adapter_config.json --training training_args.json

# From directories
gradience check --task <task_type> --peft-dir ./peft_out --training-dir ./trainer_out
```

## Task Types

- `text_generation` - GPT-style language modeling
- `seq_cls` / `sequence_classification` - Text classification
- `qa` / `question_answering` - Question answering
- `easy_classification` - Simple classification (few classes)
- `hard_classification` - Complex classification (many classes)

## Output Formats

```bash
# JSON output (for analysis scripts and plotting)
gradience monitor run.jsonl --json

# Pretty JSON
gradience monitor run.jsonl --json | python -m json.tool

# Verbose human-readable (detailed diagnostic evidence)
gradience monitor run.jsonl --verbose
```

## Audit Options

```bash
# Basic audit (summary only)
gradience audit --peft-dir adapter

# With per-layer spectral analysis
gradience audit --peft-dir adapter --layers

# With rank suggestions (identifies overparameterized layers)
gradience audit --peft-dir adapter --suggest-per-layer

# Top underutilized layers by rank
gradience audit --peft-dir adapter --top-wasteful 10

# Everything in JSON (for cross-run comparison)
gradience audit --peft-dir adapter --layers --suggest-per-layer --json

# Append to telemetry (joint dynamics + spectral analysis)
gradience audit --peft-dir adapter --append run.jsonl
```

## Rank Structure Analysis via SVD

```bash
# Decompose rank 16 adapter, retain top 8 singular components
gradience truncate --peft-dir adapter_r16 --out-dir adapter_r8 --rank 8

# With specific alpha scaling mode
gradience truncate --peft-dir adapter_r16 --out-dir adapter_r8 --rank 8 --alpha-mode keep_ratio

# Detailed per-layer energy retention report
gradience truncate --peft-dir adapter_r16 --out-dir adapter_r4 --rank 4 --verbose --report compression_report.json

# Different data types
gradience truncate --peft-dir adapter --out-dir adapter_bf16 --rank 8 --dtype bf16

# JSON output for downstream analysis
gradience truncate --peft-dir adapter --out-dir compressed --rank 6 --json
```

## Diagnostic Commands

```bash
# Find alerts in telemetry
gradience monitor run.jsonl --json | jq '.alerts'

# Check for Guard events
grep "GUARD_" run.jsonl

# Count training steps
grep '"event":"train_step"' run.jsonl | wc -l

# Extract final metrics
grep '"event":"run_end"' run.jsonl | jq '.'
```

## Research Workflows

### Multi-seed comparison

```bash
# Audit each seed and extract spectral signatures
for seed in 42 123 7; do
  gradience audit --peft-dir runs/seed_${seed}/peft --json > spectra_${seed}.json
  gradience monitor runs/seed_${seed}/run.jsonl --json > dynamics_${seed}.json
done

# Compare spectral summaries across seeds
diff <(jq '.summary' spectra_42.json) <(jq '.summary' spectra_123.json)
```

### Spectral trajectory extraction

```bash
# Extract per-layer spectral data for external plotting
gradience audit --peft-dir runs/experiment/peft --layers --json > layer_spectra.json

# Extract training dynamics for loss/lr trajectory analysis
gradience monitor runs/experiment/run.jsonl --json > dynamics.json
```

### Cross-adapter geometric comparison

```bash
# Compare rank utilization across different adapter configurations
gradience audit --peft-dir runs/r8_experiment/peft --suggest-per-layer --json > r8_structure.json
gradience audit --peft-dir runs/r16_experiment/peft --suggest-per-layer --json > r16_structure.json
gradience audit --peft-dir runs/r32_experiment/peft --suggest-per-layer --json > r32_structure.json

# Compare effective rank distributions
diff r8_structure.json r16_structure.json
```

### Merge compatibility investigation

```bash
# Audit adapters trained on different tasks to study structural compatibility
gradience audit --peft-dir runs/task_a/peft --layers --json > task_a_spectra.json
gradience audit --peft-dir runs/task_b/peft --layers --json > task_b_spectra.json

# Compare per-layer singular value distributions
# Layers with similar spectral profiles may merge more cleanly
```

## Real Examples

### After HuggingFace Training

```bash
# Typical output structure:
# ./results/
#   ├── run.jsonl           # Gradience telemetry
#   ├── adapter_model.bin   # PEFT adapter
#   └── adapter_config.json # PEFT config

# Analyze everything
gradience audit --peft-dir ./results --append ./results/run.jsonl
gradience monitor ./results/run.jsonl --verbose
```

### Studying rank evolution across configurations

```bash
# Sweep over ranks and audit each
for rank in 4 8 16 32; do
  gradience audit --peft-dir ./runs/r${rank}/peft --suggest-per-layer --json > rank_${rank}.json
done

# Compare how rank utilization scales — do higher-rank adapters use the extra capacity?
# Look at utilization_mean and spectral energy retention across the sweep
```

### Investigating spectral phase transitions

```bash
# Monitor telemetry from a long training run
gradience monitor runs/long_run/run.jsonl --json > dynamics.json

# Audit at multiple checkpoints to track spectral evolution
for step in 500 1000 2000 5000 10000; do
  gradience audit --peft-dir runs/long_run/checkpoint_${step}/peft --json > spectra_step_${step}.json
done

# Look for phase transitions: sudden changes in rank utilization or energy distribution
```

### Debugging Training Failure

```bash
# Find when it failed
gradience monitor failed/run.jsonl --verbose | grep -A10 "ERROR\|WARNING"

# Check last few steps
tail -5 failed/run.jsonl | jq '.'

# Look for numerical issues
grep -E "nan|inf" failed/run.jsonl
```

### Analyzing rank structure after training

```bash
# Your training produced ./results/adapter with r=16
# Probe how much rank is actually utilized
gradience truncate --peft-dir ./results/adapter --out-dir ./results/adapter_r8 --rank 8

# Examine the output:
# Input rank: 16
# Output rank: 8
# Mean retained energy: 87.3%
# LoRA parameter reduction: 589,824 → 294,912 (2.0x)
#
# High retained energy at half rank suggests the adapter is overparameterized.
# Low retained energy would indicate the full rank is structurally necessary.
```

## Pro Tips

1. **Always use `--verbose`** for exploratory analysis
2. **Always use `--json`** for reproducible pipelines and cross-run comparison
3. **Combine audit + monitor** to correlate spectral structure with training dynamics
4. **Use `--append`** to merge spectral data into the telemetry stream for joint analysis
5. **Check task type** matches your problem (`text_generation` vs `seq_cls`)
6. **Audit before truncating** -- understand rank utilization before testing compression hypotheses
7. **Save truncation reports** with `--report` for experiment reproducibility
8. **Compare across seeds** to distinguish learned structure from initialization artifacts
9. **Track spectra at checkpoints** to study how rank utilization evolves during training
10. **Use `--suggest-per-layer`** to identify which layers carry signal vs. noise

## Environment Variables

```bash
# Disable color output
NO_COLOR=1 gradience monitor run.jsonl

# Custom output directory
GRADIENCE_OUTPUT_DIR=/tmp gradience audit --peft-dir adapter
```
