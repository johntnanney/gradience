# PEFT rank_pattern Compatibility Notes

## Current Implementation (as of PEFT 0.18.1)

### The Problem
PEFT's `rank_pattern` feature has compatibility issues when:
1. Module names have wrapper prefixes (`base_model.model.`)
2. Default rank (`r`) is higher than some pattern values
3. Not all target modules are explicitly listed in the pattern

### Our Solution
We implemented a conservative but robust approach:

#### 1. Module Name Normalization
- Strip common prefixes: `base_model.model.`, `base_model.`, `model.`
- Implemented in: `gradience.peft_utils.normalize_peft_module_name()`

#### 2. Complete Pattern Specification
- Include ALL target modules in rank_pattern, not just overrides
- Modules not in the audit's suggestions get the default rank explicitly
- Implemented in: `gradience.peft_utils.create_complete_rank_pattern()`

#### 3. Conservative Default Rank
- Use `min(rank_pattern.values())` as default `r`
- Ensures all pattern values >= default rank (works around PEFT issue)
- Applied in: `gradience.bench.protocol` for per_layer variants

### Trade-offs

**Pros:**
- ✅ Works reliably with PEFT 0.18.1
- ✅ Heterogeneous ranks are correctly applied
- ✅ Regression check validates correct application

**Cons:**
- ❌ Creates larger rank_pattern dictionaries than necessary
- ❌ Not the cleanest long-term approach
- ❌ May need updates when PEFT improves

### Future Cleaner Approach

Once PEFT rank_pattern becomes more robust:

```python
# Ideal future approach
default_r = max(rank_pattern.values())  # or global p90 rank
peft_config = LoraConfig(
    r=default_r,
    rank_pattern={
        # Only modules that differ from default
        "module.layer.1.attn": 8,
        "module.layer.3.attn": 4,
    }
)
```

This would:
- Reduce pattern size (only overrides)
- Be more maintainable
- Follow the intended PEFT design

### Regression Testing

We added `check_heterogeneous_ranks()` to verify:
- `len(set(ranks)) >= 2` (heterogeneous requirement)
- `ranks ⊆ allowed_ranks` (configuration compliance)

This prevents silent fallbacks to uniform ranks and ensures per-layer compression is genuinely applied.

### References
- Fix implemented in: PR/commit [add commit hash]
- PEFT version tested: 0.18.1
- Related issue: Module name mismatch causing rank_pattern to be ignored

### When to Revisit
- When upgrading PEFT beyond 0.18.1
- If PEFT documentation indicates rank_pattern improvements
- When the regression check starts passing with the cleaner approach