# The Spectral Microscope Finds Structure in Time

*Gradience Series, Post 6*

---

Post 4 was a correction. We reported 100% regime classification from geometric features; the real number was closer to 67%, and the gap over loss wasn't statistically significant at n=15. Post 5 was a prospectus: here are the analyses you *could* run on Gradience telemetry, and here's what they seem to show in a single run. Both posts ended with the same prescription — run more seeds.

We ran more seeds. This post reports what happened.

---

The Replication Experiment

The design was simple. Five training regimes — baseline, high learning rate, high weight decay, low learning rate, low weight decay — each run with 10 random seeds instead of the original 3. Same model (NanoGPT, 6 layers, 6 heads, 384-dim embeddings), same task (Shakespeare character-level), same telemetry pipeline. Fifty runs total, each producing a `telemetry.jsonl` file with Hessian eigenvalues, spectral complexity, gradient norms, and loss at regular intervals.

The point was to answer two questions. First, do the classification results from Study 11 hold up with more data? Second — and this was the question Post 5 flagged as the important one — do the DFA exponents differ across regimes, or are long-range temporal correlations just a property of SGD?

We got 49 usable runs. One baseline seed produced an empty telemetry file. Everything else completed normally.

---

## The Classification Results: A Productive Deflation

Here's what happened to the classification numbers when we more than tripled the sample size:

| Feature set | Study 11 (n=15) | Study 12 (n=49) | Permutation p |
|---|---|---|---|
| Loss only | 66.7% | 73.5% | 0.0002 |
| 6 geometric features | 66.7% | 79.6% | 0.0002 |
| Spectral complexity only | 86.7% | 83.7% | 0.0002 |
| 7 geometric features | 66.7% | 79.6% | 0.0002 |

Every feature set is robustly significant against the permutation null. Early features — whether geometric, spectral, or loss-based — carry real information about which regime produced a run. That core claim survives and, if anything, strengthened: permutation p-values dropped from the 0.0004–0.0008 range in Study 11 to a uniform 0.0002 in Study 12 (the floor of our 5,000-permutation test).

The relative performance tells an interesting story. At n=15, loss and 6-geometry were tied at 66.7%. At n=49, both improved — loss to 73.5%, geometry to 79.6% — with geometry pulling slightly ahead. Spectral complexity alone held essentially steady (86.7% → 83.7%) and remains the best single-feature classifier. The 7-feature model that adds spectral to the 6 geometric features didn't outperform 6 features alone — the extra dimension adds noise at this sample-to-feature ratio, exactly as we found in Study 11.

The McNemar tests are uniformly non-significant: spectral vs. loss at p = 0.30, geometry vs. loss at p = 0.58. We cannot say any feature set is significantly better than any other at n=49. The gap between spectral-only (83.7%) and loss-only (73.5%) is 10 percentage points — real but not statistically demonstrable at this sample size.

If you're following this series for the classification story, the replication is modestly encouraging. The results are stable: nothing that held up at n=15 collapsed at n=49. Spectral complexity alone continues to be the most efficient single predictor, consistent with the thesis that Gradience's core quantity captures something important about training regimes. But proving its *superiority* over loss would require either a larger sample or a harder classification problem.

---

## The DFA Results: A Different Kind of Question

Classification asks: can you tell regimes apart? It's a question about our ability to distinguish things. DFA asks something else entirely: do regimes *have different dynamical structures?* It's a question about what training regimes are.

The distinction matters. A classification result tells you that some feature differs, on average, between groups. A DFA result tells you that the *temporal organisation* of that feature — its long-range memory, its correlation structure across timescales — differs between groups. You could have perfect classification with identical dynamics (if the means differ but the temporal structure is the same), and you could have identical classification with radically different dynamics (if the means converge but the correlation structure doesn't). They're orthogonal questions.

Post 5 reported DFA exponents from a single Mistral-7B run and flagged the key open question: are these exponents a property of the regime or a property of SGD? If all regimes produce the same DFA exponent, the long-range correlations are generic — interesting for understanding optimisers, but not useful for understanding training quality. If regimes produce different exponents, the temporal correlation structure is a *signature* of the training regime itself.

Study 12 answers this decisively.

---

## Spectral Complexity Has Regime-Specific Temporal Structure

We computed DFA exponents for the spectral complexity time series in each of the 49 runs and grouped them by regime. All exponents are well above 1.0, indicating strongly persistent (superdiffusive) dynamics — these are non-stationary series with pronounced trending behaviour. The key question is whether the *degree* of persistence differs.

It does.

| Regime | α (spectral complexity) | Std |
|---|---|---|
| Low learning rate | 2.073 | ± 0.025 |
| High weight decay | 1.917 | ± 0.026 |
| Baseline | 1.905 | ± 0.058 |
| Low weight decay | 1.896 | ± 0.048 |
| High learning rate | 1.574 | ± 0.077 |

The one-way ANOVA yields F = 116.86, p ≈ 7.7 × 10⁻²³. This isn't marginal. It's the strongest statistical result in the entire Gradience project by a wide margin.

The pattern is interpretable. Low learning rate produces the highest exponent (most persistent spectral dynamics — the Hessian spectrum evolves smoothly, each step highly correlated with the last). High learning rate produces the lowest exponent by a large gap — roughly 0.5 units below the cluster of other regimes. The spectral complexity time series under high learning rate is more stochastic, less persistent, more "noisy" in its temporal structure.

The within-regime standard deviations are small — 0.025 to 0.077 — while the between-regime gap from high_lr to low_lr is 0.499. The effect is clean. Individual runs within a regime produce nearly identical DFA exponents; different regimes produce clearly different ones.

This pattern is consistent across metrics, though the separation is sharpest for spectral complexity. Gradient norm DFA exponents show a related but distinct ordering: high_lr has the *highest* grad norm α (1.053), while high_wd has the lowest (0.687). The gradient norm and spectral complexity respond to hyperparameter perturbations in different, non-trivially related ways — they're not just echoing the same underlying signal.

---

## What This Means

The classification results tell you that early geometric features are informative about regimes. The DFA results tell you something stronger: training regimes don't just produce different average spectral profiles — they produce different *kinds* of spectral dynamics. The Hessian spectrum under low learning rate doesn't just have a different mean from the Hessian spectrum under high learning rate; it has a different temporal correlation structure. It evolves differently. The "type" of stochastic process is regime-dependent.

This is, to our knowledge, a novel observation. The DFA literature in deep learning is sparse — a handful of papers have applied it to loss curves or gradient norms, but we're not aware of prior work applying it to spectral complexity (the rank-utilisation signature of the Hessian spectrum) and finding regime-dependent exponents. We flag this with appropriate caution: our architecture is small, our task is simple, and our regime perturbations are basic. Whether this finding generalises to larger models and more nuanced hyperparameter variations is genuinely unknown.

But the result has a clear implication for the Gradience project. If you want to characterise training regimes from telemetry, you shouldn't just look at *where* the spectral features are (their values at a point in time or averaged over a window). You should look at *how they move* — their temporal correlation structure across the full training trajectory. The DFA exponent is a single number that compresses the entire temporal dynamics of spectral complexity into a regime-diagnostic quantity. That's what a spectral microscope with a time-lapse function gives you.

---

## The Photograph and the Film

Post 5 used the metaphor of a spectral microscope to describe what Gradience's telemetry can reveal. That metaphor was about static snapshots — pointing the microscope at a moment in training and seeing geometric structure that loss curves miss.

Study 12 extends the metaphor. A single photograph can tell you what a river looks like. But to know whether it's turbulent or laminar, fast or slow, you need film. The DFA exponent is the simplest summary of the film: it tells you, in a single number, how persistent or stochastic the flow is.

The classification story is the photograph. Different regimes produce different geometric snapshots, and you can tell them apart with modest accuracy. The DFA story is the film. Different regimes produce different *kinds of flow* in spectral space, and you can tell them apart with extraordinary reliability (F = 116.86, p < 10⁻²²).

Both are useful. But if you're trying to understand what training regimes *are* — not just label them, but characterise the dynamical process that produced them — the film is where the structure lives.

---

## What's Next

Three things follow from this.

**First**, we should test whether DFA exponents are directly useful for anomaly detection. If a training run's spectral complexity DFA exponent deviates from the expected value for its intended regime, that could be an early signal that something has gone wrong — before the loss curve shows any obvious pathology. This would convert the finding from a scientific observation into an engineering tool.

**Second**, the universality question. Our five regimes are simple perturbations of a single small model. The exponent ordering (low_lr > baseline ≈ high_wd ≈ low_wd > high_lr) needs to be checked in larger models, different tasks, and more aggressive hyperparameter variations. If the ordering is robust — if high learning rate *always* produces lower spectral complexity DFA exponents — that would be a genuine empirical law of training dynamics.

**Third**, the theoretical question. Why does high learning rate specifically disrupt the temporal persistence of spectral complexity? One hypothesis: higher learning rate produces larger per-step perturbations to the Hessian spectrum, breaking the smooth trending behaviour and introducing more stochastic variability. That's almost tautological. A deeper hypothesis: high learning rate pushes the optimiser into a genuinely different dynamical regime (in the physics sense — a different phase), characterised by weaker coupling between consecutive Hessian snapshots. DFA can't distinguish these explanations. Spectral analysis of the DFA residuals, or transfer entropy between the Hessian spectrum and the loss surface, might.

---

## Try It

The Study 12 analysis pipeline is at `reanalysis/study12_replication/analyze_study12.py`. It expects telemetry in `results/study12_replication/<run_id>/telemetry.jsonl`. The launcher script `run_study12_replication.sh` will reproduce the full experiment on a single GPU in under an hour.

```bash
# Run analysis on existing Study 12 data:
cd "Gradience II"
python reanalysis/study12_replication/analyze_study12.py

# Or reproduce from scratch:
bash reanalysis/study12_replication/run_study12_replication.sh
```

All data, scripts, and raw telemetry are in the repository.

---

**Code:** [github.com/johntnanney/gradience](https://github.com/johntnanney/gradience)
**Documentation:** THEORY.md and METRICS_GUIDE.md in the repo
**License:** Apache 2.0

*This is Post 6 in the Gradience series. Previous posts: [Post 1: A Flight Recorder for LoRA Fine-Tuning], [Post 2: Loss Tells You How Well You're Fitting. Geometry Tells You What Kind of Fitting You're Doing], [Post 3: Before You Merge: Subspace Overlap Predicts Adapter Dominance], [Post 4: What We Got Wrong About Geometry vs. Loss], [Post 5: What Can You See With a Spectral Microscope?].*
