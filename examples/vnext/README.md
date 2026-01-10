# vNext Toy LoRA Run

This folder contains a **tiny, cheap** LoRA fine-tune script whose only job is to validate that the canonical Gradience vNext tools work end-to-end.

It produces:

- a vNext telemetry JSONL (`run.jsonl`)
- a PEFT adapter directory (`peft/`) for `gradience audit`
- a minimal `training_args.json` (`training/`) for `gradience check`

## Run

From the repo root:

```bash
python examples/vnext/toy_lora_run.py --out runs/toy_run
```

The defaults are CPU-friendly and should finish quickly.

## Then validate the Gradience CLI

```bash
gradience check --task sst2 --peft-dir runs/toy_run/peft --training-dir runs/toy_run/training
gradience monitor runs/toy_run/run.jsonl
gradience audit --peft-dir runs/toy_run/peft --top-wasteful 10
```

## Make it even cheaper

- Fewer steps:
  ```bash
  python examples/vnext/toy_lora_run.py --max-steps 30
  ```

- Smaller slices:
  ```bash
  python examples/vnext/toy_lora_run.py --train-samples 64 --eval-samples 64
  ```

## Want to see compression recommendations?

Run with an intentionally oversized rank:

```bash
python examples/vnext/toy_lora_run.py --r 32 --alpha 32
gradience monitor runs/toy_run/run.jsonl
```

If the audit shows low utilization, monitor should emit a `compress_rank` recommendation.
