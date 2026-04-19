# N134 Scripts — Operational README

**Pre-registration:** `sidecar/notes/n134_spec.md` (v3.1)
**Incident log:** `sidecar/notes/n134_incident_log.md`

## Single-tenant pod rule

**During Phases 0, 1, 3, and 5, the pod runs one thing at a time.**

Training and merge-evaluation are compute-intensive and share CPU with
data-loading and BLAS-backed audit computations. Concurrent ad-hoc
Python processes during these phases can silently halve training
throughput and, in extreme cases, contend enough to distort timing-based
control paths (see `n134_incident_log.md`).

All verification, exploration, and monitoring happens from the
developer workstation, **not** from fresh Python sessions on the pod.
The one exception is non-intrusive monitoring tools that do not import
torch/numpy: `nvidia-smi`, `top`, `nvitop`.

**Post-phase verification** (e.g. `verify_v21_minimal.py` to check
audit U/V output after Phase 2) should run with CPU isolation:

    OMP_NUM_THREADS=1 taskset -c 0 python3 -u verify_v21_minimal.py

## Phase execution order (§9.2 of spec)

| Phase | Script                      | GPU? | Runtime | Notes |
|-------|-----------------------------|------|---------|-------|
| 0     | `00_pilot_train.py`         | yes  | ~10h    | 8 tasks × seed 42 |
| 0b    | `01_pilot_gate.py`          | no   | <1m     | emits retry commands on failure |
| 1     | `02_train_adapters.py`      | yes  | ~20h    | seeds 123, 456 (seed 42 copied from pilot) |
| 2     | `03_spectral_audit.py`      | mixed| ~2h     | v2.1 schema (U/V persistence) |
| 3a    | `04_sample_pairs.py`        | no   | <1s     | commits pair_sample.json (must be git-committed before 3b) |
| 3b    | `05_merge_eval.py`          | yes  | ~3.5h   | 69 merges (24 same + 45 cross) |
| 4     | `06_analysis_h1.py`         | no   | ~1m     | H1 decision + B-P replications |
| 4b    | `07_analysis_secondary.py`  | no   | ~1m     | exploratory, non-evidential |
| 5     | `08_compare_methods.py`     | no   | ~5m     | Gradience vs KnOTS vs TSV vs SVC |

## Committed tests (dry-run against synthetic data)

| Test file                  | What it verifies                             | # cases |
|----------------------------|----------------------------------------------|---------|
| `test_pilot_gate.py`       | accuracy band, retry ladder, dip test        | 14      |
| `test_analysis_h1.py`      | H1 decision rule (4 scenarios + ties)        | 5       |
| `verify_v21_minimal.py`    | v2.1 U/V orthonormality + reconstruction     | (smoke) |

Run all three tests before each phase. The H1 test should re-run after
any change to the decision-rule code, to catch regressions in the
pre-registered gate.

## Dependencies

Pod-side install for a fresh RunPod Secure Cloud pod:

```
pip install transformers==4.44.2 peft==0.13.2 datasets==3.0.1 \
            safetensors==0.4.5 accelerate==1.0.1 scipy==1.14.1 \
            pandas==2.2.3 matplotlib sentencepiece protobuf diptest
pip install -e /workspace/gradience
```

`diptest` is mandatory — §3.2 commits to Hartigan's dip test and the
pilot-gate script refuses to run without it (no silent substitutions).

## Resume / idempotence

Every phase is idempotent:
- Phase 0/1: skip tasks whose `adapter_model.safetensors` already exists
- Phase 2: skip adapters whose `_v2_1.json` audit file exists
- Phase 3b: skip pairs whose merge-eval JSON exists
- Phase 4/5: analysis scripts rebuild from upstream JSONs

To recover from a mid-phase crash: just re-run the same script; it
picks up where it left off.

## Incident handling

If training throughput degrades unexpectedly:

1. `ps aux | awk '$3 > 5'` — list CPU hogs
2. Any Python process that isn't the training PID gets `kill -9`'d
3. Document the event (PID, time range, training task affected) in
   `sidecar/notes/n134_incident_log.md`
4. Check the affected task's loss trajectory against
   `logging_steps=10` output for discontinuities
5. If training was purely step-based (no time-leaked control), the
   slowdown changed wall-clock but not training trajectory — continue
   with phase

Training in `00_pilot_train.py` and `02_train_adapters.py` is
verified step-based (cosine scheduler on optimizer steps,
`max_steps=1000`, no early-stopping, `eval_strategy="no"`).
