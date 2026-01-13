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

## Pro Tips

1. **Always use `--verbose`** for human analysis
2. **Always use `--json`** for scripting
3. **Combine audit + monitor** for complete picture
4. **Use `--append`** to merge audit into telemetry
5. **Check task type** matches your problem (`text_generation` vs `seq_cls`)

## Environment Variables

```bash
# Disable color output
NO_COLOR=1 gradience monitor run.jsonl

# Custom output directory
GRADIENCE_OUTPUT_DIR=/tmp gradience audit --peft-dir adapter
```