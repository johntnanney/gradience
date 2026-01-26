# Gradience v0.6.0

This release focuses on making Bench/Audit runs more reliable in cloud environments (especially RunPod): clearer preflight checks, faster GPU smoke testing, better defaults for disk hygiene, and more explicit audit configuration.

## Breaking change: UDR is opt‑in

UDR/SDI (base-model norm loading) is no longer enabled implicitly.

**Before (v0.5.x):**
```yaml
audit: {}
```

**Now (v0.6.0):**
```yaml
audit:
  compute_udr: true
  base_model: "mistralai/Mistral-7B-v0.1"
```

Rationale: avoids expensive base-model loading during audit by default, which can stall or fail on constrained pods. If you rely on UDR, explicitly enable it and specify the base model.

---

## What’s new

### Preflight validation
Adds a lightweight “fail fast” check before long runs (device availability, disk space, cache health, model accessibility). Intended to catch common cases like silently running on CPU, corrupted HF cache, or insufficient storage.

### GPU smoke tests
New GPU-focused smoke configs/scripts that exercise the full pipeline (load → short LoRA train → audit → compress → eval) in minutes, suitable for validating a fresh pod or CI environment.

### Artifact hygiene defaults
Improves default behavior around large artifacts (adapter weights/checkpoints) so repeated runs are less likely to exhaust disk. Reports and summary artifacts remain intact (bench.json/bench.md/audit.json/eval.json).

### “No tmux” friendly running
Adds/standardizes runner scripts that support disconnect-prone environments with simple state markers (`_pid.txt`, `STAGE.txt`, `_exit_code.txt`) and log-friendly execution.

---

## RunPod support

### New RunPod docs and setup script
Adds `docs/runpod.md` plus an environment setup script to standardize HF cache locations on persistent storage and reduce cache corruption / quota surprises.

### Hugging Face cache vars updated
Moves away from deprecated `TRANSFORMERS_CACHE` toward `HF_HOME`, `HF_HUB_CACHE`, and `HF_DATASETS_CACHE`.

---

## Testing & contributor hygiene

- Adds/expands CI coverage (unit tests + CPU smoke) to catch regressions early.
- Adds contributor guidance (what not to commit; what to include in bug reports).
- Ensures configs/scripts/docs are included in packaged installs where applicable.

---

## Migration notes (v0.5.x → v0.6.0)

1. If you want UDR/SDI, explicitly enable it:
   ```yaml
   audit:
     compute_udr: true
     base_model: "your-base-model-id"
   ```
2. On RunPod, use the provided environment script to set cache paths.
3. Validate with the GPU smoke test before launching full multi-seed runs.
