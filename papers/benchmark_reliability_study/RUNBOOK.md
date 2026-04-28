# Pod Runbook — Benchmark Reliability Study

**Companion to:** `SPEC_GPU_v0_1.md` §14
**Target:** RunPod RTX 4090 community-tier; estimated 25–40 GPU-hours, $10–16
**Pre-registration tag:** `v1_1_1_LOCKED` (see `LOCK_NOTES.md`)

This runbook is the operator-facing checklist. Every step is reproducible: if
the pod dies mid-run, you can spin up another pod, rerun setup, and
`gpu_inference.py` resumes from where it stopped (per-condition resume via
`runs/raw/{run_id}/run_metadata.json` markers).

---

## 0. Workstation pre-flight (before pod allocation)

- [ ] On `master`, confirm `git tag --list 'v1_1_1_*'` shows `v1_1_1_LOCKED`.
- [ ] Generate a Hugging Face read-only token at
  https://huggingface.co/settings/tokens. Profile must have access to
  `winogrande` (gated; click "Agree" on its dataset page first).
- [ ] Verify the token reads `winogrande`:
  ```bash
  HF_TOKEN=hf_xxx python3 -c "from datasets import load_dataset; \
    print(load_dataset('winogrande', 'winogrande_xl', split='validation', \
    revision='01e74176c63542e6b0bcb004dcdea22d94fb67b5')[0])"
  ```
- [ ] Decide cost cap. Default: $30 hard halt. Surface to user if exceeded.

## 1. Pod allocation

- [ ] RunPod web UI → Deploy → Community Cloud → RTX 4090 (24 GB VRAM).
- [ ] Volume: 50 GB persistent at `/workspace`.
- [ ] Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04` (or
  closest available — pin the tag in your run notes).
- [ ] Wait for SSH access. Save the `ssh root@<ip> -p <port> -i <key>` line.

Verify the pod once SSH is up:

```bash
ssh root@<ip> -p <port> -i <key> 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python3 --version; df -h /workspace'
```

Expected: `RTX 4090, 24564 MiB`, `Python 3.11.10`, `/workspace` mounted.

## 2. Pod setup

```bash
ssh root@<ip> -p <port> -i <key>
```

Inside the pod:

```bash
# HF cache → persistent volume (survives pod restart)
mkdir -p /workspace/hf_cache
ln -sfn /workspace/hf_cache /root/.cache/huggingface

# Set the HF token for gated datasets (Winogrande)
export HF_TOKEN=hf_xxx

# Install deps. The locked file is the contract; it pins concrete versions
# resolved against transformers 4.46.0.
pip install --upgrade pip
pip install -r /workspace/study/requirements.gpu.lock

# Verify GPU + bf16 path
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True, NVIDIA GeForce RTX 4090
```

## 3. Get the locked code onto the pod

Two options. Pick one.

### Option A: clone from git (preferred when the tag is pushed)

```bash
cd /workspace
git clone https://github.com/<user>/gradience.git
cd gradience
git checkout v1_1_1_LOCKED
ln -sfn /workspace/gradience/papers/benchmark_reliability_study /workspace/study
```

### Option B: tar-pipe from workstation (works without remote git)

On workstation:

```bash
cd /Users/john/code/gradience/papers/benchmark_reliability_study
COPYFILE_DISABLE=1 tar czf /tmp/study.tar.gz \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='runs/raw' --exclude='runs/normalized' \
  --exclude='._*' .
scp -P <port> -i <key> /tmp/study.tar.gz root@<ip>:/tmp/study.tar.gz
```

On pod:

```bash
mkdir -p /workspace/study
cd /workspace/study
tar --no-same-owner -xzf /tmp/study.tar.gz
```

## 4. Pre-download model weights (optional but saves ~10 minutes during the run)

```bash
python3 - <<'EOF'
from huggingface_hub import snapshot_download
models = [
  ("EleutherAI/pythia-410m",       "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"),
  ("EleutherAI/pythia-1.4b",       "fedc38a16eea3bd36a96b906d78d11d2ce18ed79"),
  ("Qwen/Qwen2.5-1.5B-Instruct",   "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
]
for name, rev in models:
    print(f"==> {name} @ {rev[:8]}")
    snapshot_download(name, revision=rev, allow_patterns=[
        "*.json", "*.txt", "*.safetensors", "*.model",
        "tokenizer*", "vocab*", "merges*",
    ])
print("done")
EOF
```

Warm cache size: ~6.5 GB.

## 5. Pre-flight check

```bash
cd /workspace/study
python3 scripts/gpu_inference.py --preflight-only --config configs/study_config.yaml
```

Expected output ends with `[preflight] OK` after these green lines:

- `[preflight] checking schemas readable…`
- `[preflight] checking prompt files + content hashes…`
- `[preflight] checking output dir writable…`
- `[preflight] checking fewshot lock entries cover all conditions…`
- `[preflight] checking models loadable (1 model probe)…`

If any check raises, fix before launching the main run. The most common
failure mode is HF auth missing for Winogrande — `export HF_TOKEN=…` and
re-run.

## 6. Smoke test (recommended)

Before committing to ~30 GPU-hours, run one condition end-to-end:

```bash
python3 scripts/gpu_inference.py \
  --config configs/study_config.yaml \
  --filter-model pythia_410m \
  --filter-benchmark arc_challenge \
  --max-conditions 1
```

Expected: ~1–2 minutes; `runs/raw/arc_challenge____pythia_410m__P1_original__s42__ll_norm/` (or similar) is created with `run_metadata.json` + `item_scores.jsonl` (1172 lines).

Sanity-check the output:

```bash
RUN=$(ls -1 runs/raw/ | head -1)
python3 -c "
import json, jsonschema
schemas = {n: json.load(open(f'schemas/{n}.schema.json')) for n in
           ['run_metadata', 'item_scores']}
m = json.load(open(f'runs/raw/{RUN}/run_metadata.json'))
jsonschema.validate(m, schemas['run_metadata'])
n = sum(1 for _ in open(f'runs/raw/{RUN}/item_scores.jsonl'))
assert n == m['num_items_completed'], f'jsonl line count {n} != metadata {m[\"num_items_completed\"]}'
print('smoke OK:', m['condition_id'], n, 'items')
" RUN=$RUN
```

## 7. Main run

```bash
cd /workspace/study
nohup python3 scripts/gpu_inference.py \
  --config configs/study_config.yaml \
  --conditions manifests/conditions_primary.csv \
  --conditions-secondary manifests/conditions_gsm8k.csv \
  --fewshot manifests/fewshot_manifest.csv \
  --out-dir runs/raw/ \
  > runs/inference.log 2>&1 &

echo "PID=$!"
```

Monitor in another shell:

```bash
tail -f /workspace/study/runs/inference.log
```

The script:

- Groups conditions by `model_id` (loads each of the 3 models exactly once).
- Resumes per-condition: re-running with completed conditions just skips them.
- Writes atomically: `runs/raw/.tmp/{condition_id}/` then `os.replace` to
  `runs/raw/{condition_id}/`. Partial directories never appear in the
  final layout.
- Per-condition failures land in `runs/failures.jsonl` (one JSONL row per
  failure, with traceback + timestamp). The main loop continues.

## 8. Periodic checks (every few hours)

```bash
# Progress
ls runs/raw/ | wc -l                    # completed
wc -l runs/failures.jsonl 2>/dev/null   # failures
du -sh runs/raw/                        # output size

# Cost projection (RunPod pricing × elapsed hours; check the dashboard)
```

If failures > 5% or cost projects over $30, halt:

```bash
kill $(pgrep -f gpu_inference.py)
```

## 8a. Mark conditions complete in the manifest

After the GPU run, the conditions manifests still show `condition_status="pending"` —
`gpu_inference.py` doesn't mutate them mid-run (would race with concurrent
reads). The CPU normalizer at script 04 only processes rows with
`condition_status="complete"`, so we need a post-run pass to flip the
status for rows whose `runs/raw/{condition_id}/run_metadata.json` exists
and reports `status="complete"`. See `IMPLEMENTATION_DEVIATIONS.md` D-17.

On the pod (or workstation, post-retrieval), run:

```bash
cd /workspace/study  # or local study dir
python3 - <<'PY'
import csv, json
from pathlib import Path
for manifest in [Path("manifests/conditions_primary.csv"),
                 Path("manifests/conditions_gsm8k.csv")]:
    rows = list(csv.DictReader(open(manifest)))
    fields = list(rows[0].keys()) if rows else []
    n_marked = 0
    for r in rows:
        meta = Path("runs/raw") / r["condition_id"] / "run_metadata.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            if m["status"] == "complete" and r["condition_status"] != "complete":
                r["condition_status"] = "complete"
                n_marked += 1
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{manifest}: marked {n_marked} rows complete")
PY
```

This is idempotent: running it twice is a no-op. Manifest changes are
intentional (lock-amendment scope) — commit them on the pod-results
commit so reviewers see the state transition.

## 9. Completion + retrieval

When the script exits with `[summary] completed=672/672`:

```bash
cd /workspace/study
tar -czf runs_final.tar.gz runs/raw/ runs/failures.jsonl runs/inference.log
ls -lh runs_final.tar.gz   # expect ~50–200 MB
```

Pull from workstation:

```bash
scp -P <port> -i <key> \
  root@<ip>:/workspace/study/runs_final.tar.gz \
  /Users/john/code/gradience/papers/benchmark_reliability_study/
```

Or, if scp is slow, upload to the user's preferred object store from the pod
and download at workstation.

## 10. Workstation verification

```bash
cd /Users/john/code/gradience/papers/benchmark_reliability_study
mkdir -p runs/raw && tar -xzf runs_final.tar.gz

# Normalizer must accept all 672 raw runs
python3 scripts/04_normalize_outputs.py \
  --conditions manifests/conditions_primary.csv \
  --raw-dir runs/raw/ \
  --schemas-dir schemas/ \
  --out runs/normalized/item_level_primary.parquet

python3 scripts/04_normalize_outputs.py \
  --conditions manifests/conditions_gsm8k.csv \
  --raw-dir runs/raw/ \
  --schemas-dir schemas/ \
  --out runs/normalized/item_level_gsm8k.parquet

# Reproducibility trace must report status=pass
python3 scripts/98_reproducibility_trace.py \
  --config configs/study_config.yaml \
  --manifests-dir manifests/ \
  --raw-dir runs/raw/ \
  --normalized-dir runs/normalized/ \
  --analysis-dir analysis/ \
  --sample-n 5 --seed 20260424 \
  --out reports/reproducibility_trace.md
```

Both must succeed (exit 0). Failures here block Phase 5 (analysis) — surface
the failure rather than continuing.

## 11. Pod teardown

After verification:

- [ ] RunPod web UI: stop pod.
- [ ] If `/workspace/hf_cache` will be reused for a future run, leave the
  persistent volume alone. Otherwise delete it to avoid storage charges.

Tag the inference state once verified:

```bash
git add runs/raw/ runs/failures.jsonl runs/inference.log
git commit -m "papers/benchmark_reliability_study: GPU inference complete (v1.1.1)"
git tag v1_1_1_INFERENCE_COMPLETE
```

(Or, if the raw-runs tarball is too large to commit, archive it on object
storage and record the URL + SHA-256 in `LOCK_NOTES.md`. A 50–200 MB tarball
is fine for git, so committing is the simplest option.)

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `[preflight] checking models loadable` hangs | Slow first model download | `du -sh /workspace/hf_cache` to see progress; verify HF Hub is reachable |
| `winogrande` 401 / 403 on dataset load | HF token missing or no Winogrande accept | `export HF_TOKEN=…`; click "Agree" on https://huggingface.co/datasets/winogrande |
| `torch.cuda.OutOfMemoryError` on Qwen | bf16 + KV cache pushed past 24 GB on a long-context item | This is rare on Qwen-1.5B; if it does occur, halve the per-condition batch size in code (currently 1; nothing to halve) and re-run; root-cause via `nvidia-smi` |
| Failure log accumulates fast | Per-item exceptions in scoring | `head -3 runs/failures.jsonl` to inspect; common cause is a dataset row with unexpected null fields |
| `Final dir already exists` error | Stale `runs/raw/.tmp/{cond}/` from a prior crash without cleanup | Manually `rm -rf runs/raw/.tmp/{cond}/`; the resume protocol handles the rest |

## Cost cap protocol

If running cost projects above $30 at any point:

1. `kill $(pgrep -f gpu_inference.py)` (preserves all completed conditions).
2. Surface to user with elapsed time, completed-condition count, and
   conditions-remaining estimate.
3. Decide: continue with explicit override, or stop and analyze partial data.

The pre-registration's H1 test does not require all 672 conditions —
surviving conditions still support the variance-components decomposition,
just with fewer DOF. Document any partial-data state in `deviations.md`.
