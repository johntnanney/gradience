# N134 Incident Log

Committed record of anomalies observed during data collection. Entries
are timestamped and include enough detail to rule out (or confirm) the
incident as a confound after H1 outcome is known.

---

## 2026-04-19 14:05–14:16 UTC — CPU-contention from rogue audit processes

**Phase:** 0 (pilot training)
**Training PID:** 2055 (alive throughout)
**Task in progress at time of incident:** hellaswag_s42 (steps ~100–150)
**Task already complete:** arc_s42 (finished at 14:00, before incident)

### What happened

During Phase 0 pilot training, I attempted to verify the v2.1 audit schema
on the freshly-completed arc_s42 adapter. Four separate SSH invocations
spawned `python3` audit processes in the background on the pod:

| PID  | Started | CPU%  | RSS     | Purpose                          |
|------|---------|-------|---------|----------------------------------|
| 3882 | 14:01   | 607%  | 9.3 GB  | full-adapter audit (SSH heredoc) |
| 4139 | 14:04   | 452%  | 5.3 GB  | full-adapter audit (retry)       |
| 4559 | 14:13   | 465%  | 0.9 GB  | verify_v21.py run #1             |
| 4710 | 14:16   | 341%  | 0.7 GB  | verify_v21.py run #2             |
| 4831 | 14:18   | (killed before fully running)    |

These processes were doing NumPy SVD over all 128 LoRA layers of the
arc_s42 adapter. Each SVD spawns BLAS threads (OpenMP default), which
contended with the training process's CPU-bound path (data loading,
gradient-checkpointing backward pass, cosine scheduler bookkeeping).

Training throughput degraded from the arc baseline of **~1.03 s/step** to
**18–21 s/step** during the contention window (hellaswag steps 120–160).

### Remediation

At 14:16 UTC, I issued `kill -9` on all four rogue audit PIDs (3882, 4139,
4559, 4710) plus 4831. Training PID 2055 survived. Training throughput
recovered to **~2.14 s/step** by step 165, which is the steady-state for
hellaswag (not residual contention — see "Throughput investigation" below).

### Training-trajectory verification

The primary concern was that a wall-clock-dependent training-control path
would have been affected by the slowdown, producing a non-comparable
arc_s42 or hellaswag_s42 adapter. I audited `00_pilot_train.py` for
time-based logic:

- `max_steps=1000` (hard cap, step-based)
- `lr_scheduler_type="cosine"` + `warmup_ratio=0.06` (step-based, not time)
- `eval_strategy="no"`, `save_strategy="no"` (no intermediate triggers)
- No early-stopping, no `patience`, no `load_best_model_at_end`
- `time.time()` usage is limited to wall-clock accounting in
  `training_meta.json`, never to control flow

**Training control is purely step-based. The slowdown changed wall-clock
duration but not the effective training trajectory.**

Loss-curve inspection on hellaswag_s42 (steps 10-620, logged at
`logging_steps=10`) confirmed no discontinuity at the contention window:

| Steps      | Loss trajectory       |
|------------|-----------------------|
| 10 – 60    | 2.55 → 1.98 (descent) |
| 70 – 120   | 1.94 → 1.87           |
| **130 – 160**  | **1.85 → 1.85 (contention window)** |
| 170 – 260  | 1.89 → 1.89           |
| 270 – 620  | 1.86 → 1.78           |

The contention-window section (130–160) shows typical LoRA plateau
behavior, not a spike or discontinuity. No anomaly attributable to the
slowdown.

**arc_s42 is CLEAN** (training completed at 14:00, before any rogue
process launched). **hellaswag_s42 is CLEAN** based on loss-curve
inspection.

### Throughput investigation (post-recovery)

At 14:45 UTC, after training had been stable for 30+ minutes:

- `top -b -n 1`: only PID 2055 consuming CPU, no zombies, no other
  Python processes
- `nvidia-smi`: **100% GPU utilization**, 31.4 GB / 48 GB VRAM, 281W / 300W
- Load avg 13.03 (consistent with training + 4 dataloader workers on
  a healthy system)

The "2x residual slowdown" observation from 14:20 was not actually
residual contention. It was task-dependent step cost: hellaswag has
longer sequences and denser token distributions than arc, producing a
~2x higher per-step cost. The arc steady-state (~1.03 s/step) was
atypically fast. 2.14 s/step is the honest steady-state for this
workload on the RTX 6000 Ada.

### Operational rules added

1. **No ad-hoc Python on the pod during training phases.** All
   verification runs happen after training completes, or on a separate
   throwaway pod. The one exception is `nvidia-smi` / `top` /
   `nvitop`-style monitoring tools that do not import torch / numpy.

2. **Audit CPU-isolation for post-Phase work.** When audits must run,
   they do so under `OMP_NUM_THREADS=1 taskset -c 0` to prevent BLAS
   thread explosion.

3. **Committed script `verify_v21_minimal.py`** (at
   `scripts/n134/verify_v21_minimal.py`) does SVD on a single layer
   only, takes <1s, and is safe to run post-Phase-0. It is NOT safe
   to run on all 128 layers during training.

### Impact on H1 outcome

No expected impact on H1 outcome, given:

- Training is purely step-based (no time-leaked control flow)
- Loss trajectories for the two tasks training during or after the
  incident (arc, hellaswag) show no discontinuity
- Eligibility band [0.70, 0.90] check at pilot gate will independently
  flag any adapter that landed outside the band

If the H1 gate produces an unusual result at Phase 4, this entry is
the committed reference to check against. Specifically, if hellaswag's
final_val_accuracy falls below 0.70 or shows unusual merge-degradation
behavior relative to the other 7 tasks, the first diagnostic step is
to compare its loss trajectory against the committed loss log here and
rule out contention as a cause.

---
