# Study 13: Windowed DFA Across the Grokking Phase Transition

## Motivation

Study 12 established that DFA exponents of spectral complexity are regime-diagnostic during smooth NanoGPT training (F=116.86, p≈10⁻²³). Applying DFA to existing grokking telemetry (Study 7, modular addition mod 97) revealed that during the memorisation plateau, Hessian curvature signals (λ₁, gᵀHg) exhibit white-noise dynamics (α ≈ 0.53) while weight norm is Brownian (α ≈ 2.0). However, Study 7 never completed the grokking transition (wd=0.001 was too low).

Study 13 tests whether windowed DFA exponents shift across the grokking phase transition, and whether this shift is detectable *before* the transition is visible in validation accuracy.

## Central Hypothesis

Windowed DFA exponents of Hessian curvature metrics (λ₁, trace_H, gHg) will shift from α ≈ 0.5 (white noise, memorisation plateau) toward α > 1.0 (persistent, productive learning) as the network undergoes grokking.

## Experimental Design

### Architecture & Task
- **Model:** MLP — Embedding(98, 128) → Flatten → Linear(384, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 97)
- **Task:** Modular addition (a + b) mod 97
- **Optimizer:** AdamW, lr=1e-3, weight_decay=0.1
- **Train fraction:** 30% of all pairs (2,822 train / 6,587 test)
- **Max steps:** 80,000 (no early stopping)

### Why wd=0.1
Study 7 used wd=0.001 and never grokked in 200k steps. Power et al. (2022) showed that weight decay ≥ 0.1 is necessary for grokking to occur within reasonable step budgets. The existing `run_grokking_experiment()` in the Gradience codebase already uses wd=0.1.

### Telemetry
Every 50 training steps:
- **Training metrics:** train_loss, train_acc, val_loss, val_acc
- **Norms:** weight_norm, grad_norm
- **Hessian curvature:** λ₁ (power iteration, 40 iters), trace_H (Hutchinson, R=4), gᵀHg

Yields ~1,600 measurements per run over 80k steps.

### Seeds
12 seeds (42–53), enabling paired within-seed statistical testing.

## Windowed DFA Protocol

- **Window size:** 150 points (7,500 training steps)
- **Stride:** 30 points (1,500 steps)
- **DFA scales:** log-spaced from 4 to 37 (0.25 × window), 20 values
- **Quality threshold:** R² > 0.80

### Phase Segmentation
- **T_grok:** first step where val_acc > 0.95
- **Pre-grok windows:** centre < T_grok − 5,000 steps
- **Post-grok windows:** centre > T_grok + 5,000 steps
- **Transition windows:** excluded from primary test

## Statistical Analysis

### Test 1 (Primary): Paired t-test on DFA shift
For each seed and curvature metric: Δα = mean(α_post) − mean(α_pre). Paired t-test across 12 seeds. Success: p < 0.05 for ≥ 2/3 curvature metrics.

### Test 2: Effect size
Cohen's d for each metric. Target: d > 0.7.

### Test 3: Metric consistency
Spearman ρ between Δα values across metrics. Target: ρ > 0.5.

### Test 4: Control metrics
Weight_norm and grad_norm DFA exponents should NOT shift (p > 0.05).

### Test 5: Early-warning lead time
T_detect = first window where α > mean(α_pre) + 2·std(α_pre). Report T_grok − T_detect. Success: ≥ 50% of runs detect shift ≥ 500 steps early.

## Files

| File | Purpose |
|------|---------|
| `study13_train_grokking.py` | Training script with Hessian telemetry |
| `run_study13.sh` | Multi-seed bash harness (resume-safe) |
| `analyze_study13.py` | Post-hoc analysis pipeline |
| `plot_study13.py` | Figure generation |

## Running

```bash
# Single seed (dry run)
python study13_train_grokking.py --seed 42 --out_dir results/study13/seed_42

# All 12 seeds (sequential)
bash run_study13.sh

# All 12 seeds (4-way parallel on multi-GPU)
N_PARALLEL=4 bash run_study13.sh
```

## Computational Requirements

- Per seed: ~45 min on A100
- 12 seeds sequential: ~9 hours
- 4-way parallel: ~2.5 hours
- Storage: ~600 MB total

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| No grokking within 80k steps | Dry-run seed 42 first; increase wd to 0.2 or reduce train_frac to 0.1 |
| Hessian too slow | Fall back to every 100 steps; adjust window to 100 points |
| Null result (α stays ~0.5) | Valid scientific outcome; report as such |
