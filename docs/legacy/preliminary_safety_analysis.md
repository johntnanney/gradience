# Preliminary Safe Uniform Baseline Analysis

## Findings So Far

### ❌ **Uniform r=16: UNSAFE** 
- **Evidence from 2 seeds**: Both failed safety criteria
- **Seed 42**: Both uniform_median and uniform_p90 **FAILED** with Δ=-0.025 (exactly at threshold)
- **Seed 123**: Both variants **FAILED** with Δ=-0.080 (much worse, 3x threshold)  
- **Conclusion**: r=16 is consistently too aggressive, fails safety across seeds

### 🔄 **Uniform r=20: TESTING** (Very Promising)
- **Seed 42**: Currently running, excellent probe quality (83% vs 75% required)
- **Training budget**: Full 500 steps + comprehensive evaluation
- **Expected completion**: Soon, this is our leading candidate

### 🔄 **Uniform r=24: TESTING** 
- **Seed 42**: Currently running in background
- **Conservative approach**: Likely safe but may sacrifice compression efficiency

## Next Steps (In Order)
1. ✅ **Wait for r=20 results** - Most promising candidate  
2. **Test r=20 multi-seed** if initial results are good
3. **Compare r=20 vs r=24** efficiency/safety tradeoff
4. **Test extended training r=16** (for completeness, but unlikely to overcome 8% accuracy drop)

## Empirical Insights
- **r=16 threshold confirmed**: Your intuition was correct - r=16 is not safe
- **Seed variation significant**: 2.5% vs 8.0% accuracy drop shows importance of multi-seed validation
- **r=20 range promising**: Sweet spot between aggressive compression and safety

## Validation Policy Context  
According to VALIDATION_POLICY.md:
- Current tests are **"Screening"** level (single seed, shorter training)
- For production: Need **"Screening+"** (multi-seed) or **"Certifiable"** (≥3 seeds + ≥500 steps)
- **r=20 test** is approaching Screening+ with 500 steps

## Expected Recommendations
Based on preliminary data, likely outcome:
1. **Primary recommendation**: Uniform r=20 (pending validation)
2. **Conservative fallback**: Uniform r=24 (if r=20 shows any issues)  
3. **Avoid**: Uniform r=16 (empirically unsafe)