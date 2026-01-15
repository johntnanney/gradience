# Gradience Bench Validation Policy v0.1

This document defines the **PASS criteria policy** for Gradience benchmark results, distinguishing between different levels of validation rigor.

## 🎯 **Validation Levels**

### **🔬 Screening**
- **Criteria**: Single seed, any training budget
- **PASS threshold**: ±2.5% accuracy tolerance  
- **Use case**: Rapid development iteration, initial exploration
- **Confidence**: Low - no variance estimation
- **Example**: 1 seed × 200 steps

### **🔬+ Screening Plus**  
- **Criteria**: Multi-seed but limited budget/seeds (< 3 seeds OR < 500 steps)
- **PASS threshold**: ±2.5% accuracy tolerance
- **Use case**: Enhanced development validation, promising candidate verification
- **Confidence**: Medium - limited statistical power
- **Example**: 2 seeds × 200 steps, or 3 seeds × 100 steps

### **✅ Certifiable**
- **Criteria**: ≥3 seeds AND ≥500 steps
- **PASS threshold**: ±2.5% accuracy tolerance + statistical significance
- **Use case**: Production decisions, academic publications, defensible claims  
- **Confidence**: High - statistical rigor with variance estimation
- **Example**: 3+ seeds × 500+ steps

## 📊 **Policy Implementation**

### **Automatic Classification**
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

### **Reporting Labels**
- **bench.json**: `env.validation_classification.level`
- **bench.md**: Header shows validation level + rationale
- **Verdict analysis**: Console output includes validation level

### **PASS Criteria Interpretation**

| Level | PASS Means | Suitable For | Statistical Power |
|-------|------------|--------------|-------------------|
| **Screening** | Single run within tolerance | Dev iteration | None |
| **Screening+** | Limited multi-seed validation | Dev validation | Limited |
| **Certifiable** | Statistically defensible | Production use | Sufficient |

## 🚫 **What This Policy Prevents**

1. **False confidence** from single-seed "validation"
2. **Inappropriate production deployment** based on screening results  
3. **Statistical claims** without proper variance estimation
4. **Unrealistic expectations** from mini-validation budgets

## ✅ **Best Practices**

### **Development Workflow**
1. **Screening** (50-200 steps) → Rapid experimentation
2. **Screening+** (200+ steps, 2-3 seeds) → Validate promising candidates
3. **Certifiable** (500+ steps, 3+ seeds) → Final production validation

### **Academic/Publication Standards**
- **Always use Certifiable level** for claims in papers
- **Report confidence intervals** (mean ± std across seeds)
- **Include pass rates** (e.g., "2/3 seeds passed tolerance")

### **Production Deployment**
- **Minimum Screening+ level** for any production use

## 🏛️ **Safe Uniform Baseline Policy**

### **Policy Definition**
**Safe uniform baseline** = ≥ 67% seeds PASS AND worst seed Δ ≥ -2.5%

### **Current Validated Baselines (DistilBERT/SST-2)**
- **Primary**: Uniform r=20 (25.0% compression, validated 500 steps, n=1 seed)
- **Conservative**: Uniform r=24 (16.6% compression, validated 500 steps, n=1 seed) 
- **Avoid**: Uniform r=16 (fails safety policy: 0% pass rate across 3 seeds)

### **Selection Criteria**
1. **Safety first**: Must meet pass rate AND worst-case delta thresholds
2. **Efficiency optimization**: Highest compression among safe candidates wins
3. **Fallback strategy**: If no uniform baseline is safe → recommend per-layer adaptive

### **Important Limitations**
**⚠️ TASK/MODEL DEPENDENT:** These baselines are calibrated defaults for DistilBERT on SST-2, not universal truth. **Always validate on your specific task/model combination before production deployment.**

### **Implementation Status**
- **Validation Level**: Screening+ (limited seeds but full training budget)
- **Next Steps**: Multi-seed validation recommended for Certifiable status
- **Certifiable level recommended** for critical applications
- **Additional validation** on real workload always required

## 🔄 **Version History**

- **v0.1**: Initial policy defining screening vs certifiable distinction
- **Future**: May adjust thresholds based on empirical validation experience

*This policy ensures users understand the statistical limitations of their validation results and make appropriate decisions based on the level of rigor achieved.*
## Safe Uniform Baseline Policy (Bench)

Policy definition (verbatim):
**≥ 67% seeds PASS AND worst seed Δ ≥ -2.5%**

### Reference Results

📁 **Canonical reference results:** `gradience/bench/results/distilbert_sst2_v0.1/`

Validated baselines (DistilBERT + SST-2, Bench v0.1):
- **uniform_median**: 61% compression, 100% pass rate, worst Δ = -1.0% ✅ POLICY COMPLIANT
- See frozen artifacts in `gradience/bench/results/distilbert_sst2_v0.1/` for complete results

Current calibrated baselines:
- **Uniform r=20** — primary safe uniform baseline under the policy above.
- **Uniform r=16** — observed unsafe/unstable in this benchmark setup (fails policy in multi-seed runs).

