# Gradience CLI Cheat Sheet

## Most Common Commands (Copy & Paste)

### After Training Completes

```bash
# Quick summary
gradience monitor <output_dir>/run.jsonl

# Detailed analysis with recommendations  
gradience monitor <output_dir>/run.jsonl --verbose

# Analyze LoRA adapter (if using PEFT)
gradience audit --peft-dir <output_dir>/adapter --layers

# Complete analysis (audit + monitor)
gradience audit --peft-dir <output_dir>/adapter --append <output_dir>/run.jsonl
gradience monitor <output_dir>/run.jsonl --verbose
```

### Before Training Starts

```bash
# Validate config
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
# JSON output (for scripts)
gradience monitor run.jsonl --json

# Pretty JSON
gradience monitor run.jsonl --json | python -m json.tool

# Verbose human-readable
gradience monitor run.jsonl --verbose
```

## Audit Options

```bash
# Basic audit (summary only)
gradience audit --peft-dir adapter

# With per-layer analysis
gradience audit --peft-dir adapter --layers

# With rank suggestions
gradience audit --peft-dir adapter --suggest-per-layer

# Top wasteful layers
gradience audit --peft-dir adapter --top-wasteful 10

# Everything in JSON
gradience audit --peft-dir adapter --layers --suggest-per-layer --json

# Append to telemetry
gradience audit --peft-dir adapter --append run.jsonl
```

## LoRA Compression (SVD Truncation)

```bash
# Basic truncation (rank 16 → 8)
gradience truncate --peft-dir adapter_r16 --out-dir adapter_r8 --rank 8

# With specific alpha scaling mode
gradience truncate --peft-dir adapter_r16 --out-dir adapter_r8 --rank 8 --alpha-mode keep_ratio

# High compression with detailed report
gradience truncate --peft-dir adapter_r16 --out-dir adapter_r4 --rank 4 --verbose --report compression_report.json

# Different data types
gradience truncate --peft-dir adapter --out-dir adapter_bf16 --rank 8 --dtype bf16

# JSON output for automation
gradience truncate --peft-dir adapter --out-dir compressed --rank 6 --json
```

## Debug Commands

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

### After LoRA Fine-tuning

```bash
# Check if rank 16 was too high
gradience audit --peft-dir ./lora_r16 --suggest-per-layer --json

# Compare two runs
gradience monitor ./run_r8/run.jsonl --json > r8.json
gradience monitor ./run_r16/run.jsonl --json > r16.json
diff r8.json r16.json
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

### Compressing Adapter After Training

```bash
# Your training produced ./results/adapter with r=16
# Compress to r=8 for faster inference
gradience truncate --peft-dir ./results/adapter --out-dir ./results/adapter_compressed --rank 8

# Check compression quality
# Input rank: 16
# Output rank: 8  
# Mean retained energy: 87.3%
# LoRA parameter reduction: 589,824 → 294,912 (2.0x)

# Use compressed adapter in inference
# (Same API as original - drop-in replacement)
```

## Pro Tips

1. **Always use `--verbose`** for human analysis
2. **Always use `--json`** for scripting
3. **Combine audit + monitor** for complete picture
4. **Use `--append`** to merge audit into telemetry
5. **Check task type** matches your problem (`text_generation` vs `seq_cls`)
6. **Audit before truncating** - understand rank utilization first
7. **Use `keep_ratio`** alpha mode for consistent scaling behavior
8. **Save truncation reports** with `--report` for reproducibility

## Environment Variables

```bash
# Disable color output
NO_COLOR=1 gradience monitor run.jsonl

# Custom output directory
GRADIENCE_OUTPUT_DIR=/tmp gradience audit --peft-dir adapter
```