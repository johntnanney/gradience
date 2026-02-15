# Statistical Methodology for Spectral Compression Studies

This document defines the **statistical rigor requirements** for making defensible claims from spectral data in Gradience benchmark experiments.

## Validation Levels

Gradience classifies every benchmark run by its statistical power, so researchers can calibrate claims to the evidence that supports them.

### Exploratory Analysis (Screening)
- **Criteria**: Single seed, any training budget
- **Tolerance**: +/-2.5% accuracy delta
- **Purpose**: Hypothesis exploration -- rapidly test whether a compression configuration is worth deeper investigation
- **Statistical power**: None -- no variance estimation possible
- **Example**: 1 seed x 200 steps

### Preliminary Findings (Screening+)
- **Criteria**: Multi-seed but limited budget or seeds (< 3 seeds OR < 500 steps)
- **Tolerance**: +/-2.5% accuracy delta
- **Purpose**: Narrowing the hypothesis space -- identify promising spectral configurations for rigorous follow-up
- **Statistical power**: Limited -- partial variance estimation, insufficient for strong claims
- **Example**: 2 seeds x 200 steps, or 3 seeds x 100 steps

### Publication-Ready Evidence (Certifiable)
- **Criteria**: >=3 seeds AND >=500 steps
- **Tolerance**: +/-2.5% accuracy delta + statistical significance testing
- **Purpose**: Defensible claims in publications -- results that withstand peer review
- **Statistical power**: Sufficient -- full variance estimation with confidence intervals
- **Example**: 3+ seeds x 500+ steps

## Classification Logic

### Automatic Level Assignment
```python
def classify_validation_level(config):
    seeds = compression.get("seeds", [])
    max_steps = train.get("max_steps", 0)

    if len(seeds) >= 3 and max_steps >= 500:
        return "certifiable"
    elif len(seeds) > 1:
        return "screening_plus"
    else:
        return "screening"
```

### Where Levels Appear
- **bench.json**: `env.validation_classification.level`
- **bench.md**: Header shows validation level + rationale
- **Console output**: Verdict analysis includes validation level

### Interpreting Results by Level

| Level | Result Means | Supports | Statistical Power |
|-------|-------------|----------|-------------------|
| **Exploratory** | Single run within tolerance | Hypothesis generation | None |
| **Preliminary** | Limited multi-seed agreement | Narrowing candidates | Limited |
| **Publication-Ready** | Statistically defensible result | Peer-reviewed claims | Sufficient |

## What Each Level Ensures

1. **Exploratory** ensures fast iteration without over-interpreting single-seed outcomes -- researchers know the result is a starting point, not a conclusion
2. **Preliminary** ensures that multi-seed agreement is not mistaken for statistical significance when seed count or training budget is insufficient
3. **Publication-Ready** ensures that reported results include proper variance estimation, making claims robust to reviewer scrutiny

## Experimental Design Guidelines

### Research Progression
1. **Exploratory** (50-200 steps) -- Sweep broadly across rank configurations and policies to identify interesting spectral behavior
2. **Preliminary** (200+ steps, 2-3 seeds) -- Focus on promising configurations, begin estimating variance to assess stability
3. **Publication-Ready** (500+ steps, 3+ seeds) -- Full multi-seed protocol with confidence intervals for any result that will appear in a paper

### Publication and Reporting Standards
- **Always use Publication-Ready level** for claims in papers
- **Report confidence intervals** (mean +/- std across seeds, with 95% CI)
- **Report Cohen's d** for effect size when comparing configurations
- **Include pass rates** across seeds (e.g., "2/3 seeds passed tolerance") to convey result stability
- **Document the validation level** in your methods section so reviewers can assess rigor

### Applied Use
- For applied work (deployment, product integration), a minimum of Preliminary level is recommended before acting on results
- Publication-Ready level is strongly recommended before any high-stakes decision

## Reference Baselines for Controlled Comparison

### Baseline Policy
A **reference baseline** provides a controlled comparison point for spectral compression experiments.

**Reference baseline criterion**: ≥ 67% seeds PASS AND worst seed Δ ≥ -2.5%

This dual requirement -- majority pass rate plus bounded worst-case degradation -- ensures that the baseline itself is stable enough to serve as a meaningful comparator. A baseline that fails across seeds or shows large worst-case drops would introduce confounding variance into any comparison.

### Validated Reference Baselines (DistilBERT/SST-2)

Canonical reference results: `gradience/bench/results/distilbert_sst2_v0.1/`

- **uniform_median**: 61% compression, 100% pass rate, worst delta = -1.0% -- meets baseline criterion
- **Uniform r=20**: Primary reference baseline under the criterion above
- **Uniform r=16**: Does not meet baseline criterion (fails in multi-seed runs) -- not suitable as a stable comparator

### Selecting Reference Baselines
1. **Stability first**: Must meet pass rate AND worst-case delta thresholds to serve as a reliable comparison point
2. **Then optimize compression**: Among stable baselines, prefer higher compression to make the comparison more informative
3. **Fallback**: If no uniform baseline is stable, consider per-layer adaptive configurations as the reference

### Important Limitations
**Task/model dependent:** These baselines are calibrated for DistilBERT on SST-2 and should not be assumed to transfer. Always validate reference baselines on your specific task/model combination before using them as comparators in experiments.

### Current Baseline Status
- **Validation Level**: Preliminary (limited seeds, full training budget)
- **Next Steps**: Multi-seed validation recommended to reach Publication-Ready status
- **Additional validation** on your specific workload is always required

## Version History

- **v0.1**: Initial methodology defining exploratory, preliminary, and publication-ready evidence levels
- **Future**: Thresholds may be refined as the community accumulates empirical experience across diverse tasks and models

*This methodology ensures researchers understand the statistical power of their results and calibrate their claims accordingly.*
