# Study 12: Replication Experiment Protocol

*Gradience Research Programme — March 2026*

---

## Motivation

Study 11 established the regime classification paradigm (5 regimes × 3 seeds = 15 runs) and produced the central claims of the Gradience research programme. The March 2026 reanalysis (Post 4) found that the 100% accuracy claim does not reproduce on the available five-class data, and that the n=15 sample size prevents statistical resolution of the geometry-vs-loss comparison.

Study 12 addresses this by replicating the experiment with 10 seeds per regime (n=50), which will:

1. **Resolve the geometry-vs-loss question.** McNemar's test at n=50 has adequate power to detect a 20-percentage-point accuracy gap.
2. **Test whether spectral complexity improves multivariate classification.** At n=15, the 7-feature model suffered from curse-of-dimensionality. At n=50 the feature-to-sample ratio improves to 7:50.
3. **Enable per-regime DFA analysis.** With 10 trajectories per regime, we can compute DFA exponents within each regime and test (via ANOVA) whether long-range Hessian correlations differ between healthy and pathological training — the key experiment Post 5 calls for.

## Experimental Design

### Training Runs

| Parameter | Value |
|-----------|-------|
| Architecture | NanoGPT (gpt_small) |
| Task | Shakespeare character-level language modelling |
| Config file | `config/train_shakespeare_char.py` |
| Training steps | 2,000 |
| Telemetry mode | alwayson (spectral every 10 steps) |
| Spectral settings | max_matrices=8, power_iters=8, threshold=0 |
| Eval interval | 200 steps |
| Log interval | 10 steps |
| Checkpoint saving | Disabled (telemetry only) |

### Regimes (identical to Study 11)

| Regime | Learning Rate | Weight Decay | Tag |
|--------|---------------|--------------|-----|
| baseline | 1e-3 | 0.1 | lr1em03_wd0p1 |
| low_wd | 1e-3 | 0.0 | lr1em03_wd0 |
| high_wd | 1e-3 | 1.0 | lr1em03_wd1 |
| low_lr | 1e-4 | 0.1 | lr1em04_wd0p1 |
| high_lr | 1e-2 | 0.1 | lr1em02_wd0p1 |

### Seeds

Seeds 1337, 1338, 1339 (matching Study 11) plus 1340, 1341, 1342, 1343, 1344, 1345, 1346. Total: 10 seeds × 5 regimes = **50 runs**.

## Infrastructure

### RunPod Setup

1. Provision a GPU pod (any CUDA-capable GPU; A100 preferred for speed but not required — the model is gpt_small)
2. Clone the nanogpt_gradience workspace (the Study 11 codebase)
3. Verify that the Study 11 alwayson arm reproduces for seeds 1337–1339 before running new seeds
4. Run: `bash run_study12_replication.sh`

### Compute Estimate

- Per run: ~5–10 minutes on a single A100 (2,000 steps with spectral every 10 steps)
- Total: 50 runs × ~7.5 min = ~6.25 hours (single GPU, sequential)
- Parallelised: <1 hour with 8 slots

### Storage

- Per run: ~200KB (telemetry.jsonl + console.log + time.txt)
- Total: ~10MB for all 50 runs

## Verification Protocol

### Before launching new seeds (seeds 1340–1346):

1. Run seeds 1337–1339 for all 5 regimes (15 runs)
2. Compare telemetry.jsonl outputs to Study 11 rngfix_alwayson arm
3. Verify: early_spectral_complexity_mean values match within floating-point tolerance
4. If verification fails: investigate before proceeding (likely a code or environment difference)

### After all 50 runs complete:

1. Check that all 50 `telemetry.jsonl` files exist and have >100 records each
2. Run `analyze_study12.py` for feature extraction and classification
3. Verify that seeds 1337–1339 reproduce the Study 11 classification results

## Analysis Plan

### Analysis 1: Regime Classification (primary)

Replicate the Module A analysis from the March 2026 reanalysis with n=50:

- **LOSO cross-validation** for 4 feature sets: loss_only, geometry_6, spectral_only, geometry_7
- **Permutation test** (10,000 permutations) for each feature set
- **Bootstrap 95% CI** (5,000 resamples)
- **McNemar's test** for pairwise comparisons (geometry vs loss, spectral vs loss, geometry_7 vs geometry_6)
- **Feature ablation** (leave-one-out) for geometry_7

**Success criteria:** If spectral_only or geometry_7 accuracy significantly exceeds loss_only by McNemar's test (p < 0.05), the geometry-vs-loss thesis is supported at the required sample size. If not, the thesis requires revision.

### Analysis 2: Information Theory

- KSG mutual information for each feature set
- Conditional entropy reduction
- BIC/AIC model comparison (the parsimony analysis that favoured loss at n=15 may reverse at n=50)

### Analysis 3: Per-Regime DFA Exponents (the Post 5 follow-up)

For each of the 50 runs, compute DFA exponents for:
- train_loss
- weight_norm
- grad_norm
- spectral_complexity

Then test (one-way ANOVA across 5 regimes) whether DFA exponents differ by regime.

**Key hypothesis:** If α_spectral differs between regimes (e.g., healthy runs show α ≈ 0.7, high_lr runs show α ≈ 0.5), this would mean long-range Hessian correlations are diagnostic of training health — a genuinely novel finding. If α is regime-invariant, it's a generic property of SGD.

### Analysis 4: PR Dynamics Replication

For each run, fit the participation ratio trajectory to the expand-then-compress model and extract:
- Compression onset step
- Expansion rate
- Compression rate
- Final plateau value

Test whether these parameters differ by regime (Kruskal-Wallis).

## Deliverables

1. `early_features_study12.csv` — 50-row feature matrix
2. `study12_results.json` — full analysis output
3. Figures: classification confusion matrices, DFA exponent boxplots by regime, PR trajectories by regime
4. Updated FINDINGS.md sections 6 and 7 (if results change the picture)
5. Blog post update or new Post 6 (if DFA results are decisive)

## File Locations

```
Gradience II/
├── reanalysis/study12_replication/
│   ├── STUDY12_PROTOCOL.md          ← this document
│   ├── run_study12_replication.sh   ← training launcher
│   └── analyze_study12.py           ← analysis pipeline
└── results/study12_replication/     ← output (created by launcher)
    ├── baseline_lr1em03_wd0p1_seed1337/
    │   ├── telemetry.jsonl
    │   ├── console.log
    │   └── time.txt
    ├── ...
    └── high_lr_lr1em02_wd0p1_seed1346/
```

## Timeline

1. **Day 1:** Provision RunPod, verify Study 11 reproducibility for seeds 1337–1339
2. **Day 1–2:** Run all 50 training runs
3. **Day 2:** Download results, run `analyze_study12.py`
4. **Day 2–3:** Interpret results, update documentation, draft blog update

## Risk Register

| Risk | Mitigation |
|------|------------|
| RunPod environment differs from Study 11 | Verification protocol (seeds 1337–1339 must match) |
| Training failures for some seeds | Launcher logs failures but continues; re-run failed seeds |
| DFA exponents don't converge at 2,000 steps | Increase MAX_ITERS to 5,000 for a subset of runs as sensitivity check |
| Spectral complexity adds no value even at n=50 | Report as definitive negative result; revise thesis |
| Classification accuracy *decreases* with more seeds | Would indicate Study 11 was overfit; report honestly |
