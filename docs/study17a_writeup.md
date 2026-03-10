# Study 17A: Compress-Then-Merge with 95% Energy Retention

## Objective

Test whether per-layer SVD compression before merging improves the spectral quality of LoRA adapter merges. The hypothesis is that trimming low-energy singular components reduces noise in the merged subspace, yielding higher alignment (Q_min) and lower dominance asymmetry (D).

This study evaluates a fixed 95% energy-retention threshold as a candidate default compression rule.

## Method

Each adapter pair is merged under five conditions:

- **A** — Naive (equal 0.5/0.5 coefficients)
- **B** — Norm-equalized (Frobenius-proportional coefficients)
- **C** — Recommended (audit_aware strategy from `gradience merge-audit`)
- **D** — Compress at 95% energy, then norm-equalized merge
- **E** — Compress at 95% energy, then recommended merge

Compression uses per-layer truncated SVD: for each LoRA weight matrix, find the smallest rank k such that the cumulative squared singular-value energy reaches 95% of the total. Truncate to rank k, then proceed to merge.

The key comparisons are B vs D (does compression help norm-equalized merging?) and C vs E (does compression help recommended merging?). Metrics are Q_min (worst-case cosine alignment across merged sources) and D (dominance asymmetry between sources). Higher Q_min and lower D are better.

All measurements are Phase 1 (spectral, CPU-only) on Llama-2-7b-hf base weights. No perplexity evaluation in this phase.

## Pair Set

| Pair | Adapters | Ranks | Audit Verdict | Compression |
|------|----------|-------|---------------|-------------|
| 01 | metamath-r16 × openwebmath-r16 | 16 × 16 | imbalanced | 16 → 14, 16 → 16 |
| 02 | metamath-r16 × magicoder-r16 | 16 × 16 | safe | 16 → 14, 16 → 15 |
| 03 | magicoder-r16 × btgenbot-r8 | 16 × 8 | imbalanced | 16 → 15, 8 → 7 |
| 04 | openwebmath-r64 × btgenbot-r8 | 64 × 8 | imbalanced | 64 → 61, 8 → 7 |
| 06 | catsubcat-r16 × btgenbot-r8 | 16 × 8 | imbalanced | 16 → 15, 8 → 7 |

Pairs 01–04 are primary (used in aggregates). Pair 06 is a boundary appendix case. The set covers balanced ranks, rank mismatch (16 × 8, 64 × 8), safe verdicts, and imbalanced verdicts.

At 95% energy retention, compression is minimal: typically 1–2 ranks removed. One adapter (openwebmath-r16) compressed 16 → 16 — every singular value carried more than 5% of remaining energy, so no truncation occurred at all.

## Key Table

Compression deltas across all primary pairs (positive = compression helped):

| Pair | Strategy | ΔQ_min | ΔD |
|------|----------|--------|----|
| 01 — metamath × openwebmath | NormEq | −0.002 | −0.003 |
| | Recommended | +0.005 | +0.008 |
| 02 — metamath × magicoder | NormEq | −0.010 | −0.004 |
| | Recommended | −0.005 | −0.002 |
| 03 — magicoder × btgenbot | NormEq | −0.003 | +0.003 |
| | Recommended | −0.007 | −0.002 |
| 04 — openwebmath × btgenbot | NormEq | +0.000 | +0.009 |
| | Recommended | −0.011 | −0.007 |
| **Primary mean** | **NormEq** | **−0.004** | **+0.001** |
| | **Recommended** | **−0.004** | **−0.001** |

Win rate for Q_min improvement: 1/4 pairs under both strategies.

## Result

At 95% energy retention, compression has no meaningful effect on merge quality. The average Q_min shift is −0.004 under both strategies — compression makes things marginally *worse*, not better. The dominance metric D is flat (mean |ΔD| < 0.002). Only pair 01 under the recommended strategy shows a positive Q_min delta, and even that (+0.005) is negligible in practical terms.

The root cause is clear from the compression column: 95% retains too much. Removing 1–2 ranks from a rank-16 adapter or 3 ranks from a rank-64 adapter does not meaningfully change the subspace geometry. The "noise floor" that compression is supposed to trim sits below the 5%-per-component threshold at this setting.

## Conclusion

**95% energy-threshold compression should not be adopted as a default pre-merge step.** It removes too little to help and introduces a small negative drag on Q_min, likely from the slight rank mismatch it creates between the original and compressed adapters during coefficient computation.

## Implication

The question is not whether compression helps — it is whether there exists a threshold where it helps. Study 17A eliminates the conservative end of the range. The next step is a threshold sensitivity sweep (Study 17B) across 90%, 85%, and 80% energy retention on the same pair set, looking for the crossover point where rank reduction becomes large enough to materially reshape the merged subspace geometry.
