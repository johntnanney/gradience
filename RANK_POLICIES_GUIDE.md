# Rank Selection Policies — Complete Guide

Gradience provides multiple **scientifically-grounded policies** for LoRA rank selection, moving beyond single heuristics to **evidence-based compression decisions**.

## 🎯 The Four Core Policies

### 1. **Energy@90% (Traditional)** — `energy@0.90`

**What it measures:** Cumulative energy retention in singular value spectrum  
**Formula:** Find minimal k where `Σ(σᵢ²) / Σ(σⱼ²) ≥ 0.90`

**What it assumes:**
- 90% energy retention preserves most important information
- Energy distribution correlates with importance
- Tail singular values contribute negligible signal

**Behavior:**
- **Conservative when:** Clear low-rank structure (captures most energy early)
- **Aggressive when:** Gradual energy decay (forces higher thresholds)

**Typical rank suggestions:** Medium (4-8 for most adapters)

---

### 2. **Knee Detection** — `knee`

**What it measures:** Elbow point in singular value scree plot  
**Method:** Kneedle algorithm - finds maximum distance from straight line connecting endpoints

**What it assumes:**
- Singular values have clear "elbow" structure 
- Point of maximum curvature indicates signal/noise boundary
- Human visual intuition about "obvious" cutoff points

**Behavior:**
- **Conservative when:** Gentle elbow (gradual transitions)
- **Aggressive when:** Sharp elbow early in spectrum

**Typical rank suggestions:** Low-Medium (2-4 for most adapters)

---

### 3. **Entropy Effective Rank** — `erank`

**What it measures:** Information-theoretic effective dimensionality  
**Formula:** `exp(-Σ pᵢ log pᵢ)` where `pᵢ = σᵢ / Σσⱼ` (Roy & Vetterli normalization)

**What it assumes:**
- Spectral entropy reflects true dimensionality
- Uniform distribution = maximum rank needed
- Concentrated distribution = low rank sufficient

**Behavior:**
- **Conservative when:** Nearly uniform singular values (high entropy)
- **Aggressive when:** Dominated by few large singular values (low entropy)  

**Typical rank suggestions:** Medium-High (5-8 for most adapters)

---

### 4. **Optimal Hard Threshold (Experimental)** — `oht`

**What it measures:** Signal vs noise separation using random matrix theory  
**Formula:** Gavish-Donoho threshold `τ = ω(β) × median(σ)` where `ω(β) ≈ 0.56β³ - 0.95β² + 1.82β + 1.43`

**What it assumes:**
- Noise follows random matrix theory predictions
- Signal singular values are "large" relative to noise floor
- Matrix aspect ratio affects optimal threshold

**Behavior:**
- **Conservative when:** High estimated noise floor
- **Aggressive when:** Clear signal/noise separation

**Typical rank suggestions:** Low (1-3 for most adapters)  
**⚠️ Experimental:** Adapted from full-rank matrix theory for LoRA use

## 📊 How to Interpret Policy Disagreement

When policies disagree significantly, it reveals **structural insights** about your adapter:

### **Example: energy@90=8, knee=3, erank=6, oht=2**

**Analysis:**
- **Large gap (energy vs oht):** Tail vs head disagreement
- **energy@90=8:** Gradual energy decay, needs higher rank for 90% retention
- **oht=2:** Sharp signal/noise boundary, minimal rank sufficient  
- **knee=3:** Visual elbow around rank 3
- **erank=6:** Moderate entropy, suggests medium dimensionality

**Interpretation:**
- Adapter has **gradual energy decay** with **some tail structure**
- **Head-heavy:** Most information in first few singular values
- **Tail-sensitive:** Remaining energy spread across many small values

**Recommendation:** 🎯 **Prime candidate for Bench validation!**  
Test r=2 (aggressive), r=4 (balanced), r=6 (conservative) empirically

### **Common Disagreement Patterns**

| Pattern | Energy@90 | Knee | eRank | OHT | Interpretation |
|---------|-----------|------|-------|-----|----------------|
| **Head-heavy** | High (6-8) | Low (2-3) | Medium (4-6) | Low (1-2) | Few dominant SVs, long tail |
| **Uniform** | Medium (4-6) | Medium (4-6) | High (7-8) | Medium (3-4) | Even energy distribution |
| **Clean low-rank** | Low (2-4) | Low (2-4) | Low (3-5) | Low (1-3) | Clear rank structure |
| **Noisy** | High (6-8) | High (5-8) | High (6-8) | High (4-6) | No clear structure |

## 🛠️ CLI Usage

### **Default Policies (Recommended)**
```bash
gradience audit --peft-dir ./adapter
# Uses: energy@0.90, knee, erank (balanced coverage)
```

### **Custom Policy Selection**
```bash
# Conservative analysis
gradience audit --peft-dir ./adapter --rank-policies energy@0.90,energy@0.95

# Aggressive analysis  
gradience audit --peft-dir ./adapter --rank-policies knee,oht

# Full comparison (experimental)
gradience audit --peft-dir ./adapter --rank-policies energy@0.90,knee,erank,oht
```

### **CLI Output Example**
```
Rank policy suggestions:
  Policy            Median   P90   Max   Don't Compress
  ----------------  ------  ----  ----  --------------
  energy@0.90            4     6     8         25%
  knee                   2     3     4          0%
  erank                  6     7     8         50%
  oht                    2     2     3          0%
```

**Reading the table:**
- **Median:** Typical rank suggestion across layers
- **P90:** Conservative rank suggestion (90th percentile)
- **Max:** Most conservative suggestion
- **Don't Compress:** % of layers where policy suggests keeping full rank

## ⚙️ Bench Integration

### **Automatic Policy Testing**
```yaml
# bench_config.yaml (policies auto-applied)
compression:
  allowed_ranks: [1, 2, 4, 8, 16]
  
# Creates variants: uniform_knee_p90, uniform_erank_p90, uniform_oht_p90
```

### **Custom Policy Targeting**
```yaml
compression:
  svd_variants:
    - name: conservative_choice
      rank_source: audit.rank_suggestions.energy_90.uniform_p90
      
    - name: aggressive_choice  
      rank_source: audit.rank_suggestions.oht.uniform_median
      
    - name: balanced_choice
      rank_source: audit.rank_suggestions.knee.uniform_p90
```

### **Policy Comparison Workflow**
1. **Audit:** Generate policy suggestions with `gradience audit`
2. **Hypothesis:** Each policy = compression hypothesis
3. **Test:** Bench evaluates all policies on real tasks  
4. **Evidence:** Performance metrics validate/refute policies
5. **Decision:** Choose policy based on empirical results

## 🧪 When Policies Fail

### **Energy@90% fails when:**
- ✅ Gradual singular value decay → overly conservative
- ✅ Energy threshold arbitrary (why 90% not 85%?)
- ✅ Ignores structural information beyond energy

### **Knee detection fails when:**  
- ✅ No clear elbow point (uniform decay)
- ✅ Multiple elbows (complex structure)
- ✅ Noisy singular values (spurious elbows)

### **Entropy effective rank fails when:**
- ✅ Roy & Vetterli normalization doesn't match task importance
- ✅ Information theory ≠ task-relevant dimensionality  
- ✅ Sensitive to small singular values

### **OHT fails when:**
- ⚠️  LoRA adaptation ≠ noisy full-rank matrices
- ⚠️  Noise characteristics differ from random matrix assumptions
- ⚠️  Signal/noise boundary unclear

## 📚 Mathematical Details & Citations

### **Energy Threshold**
```
Cumulative energy: E(k) = Σᵢ₌₁ᵏ σᵢ² / Σⱼ₌₁ʳ σⱼ²
Rank selection: k = argmin{k : E(k) ≥ threshold}
```

### **Optimal Hard Threshold** 
**Citation:** Gavish & Donoho (2014) "The Optimal Hard Threshold for Singular Values is 4/√3"

```
Threshold: τ = ω(β) × median(σ)
Aspect ratio: β = min(m,n) / max(m,n) 
Cubic approximation: ω(β) ≈ 0.56β³ - 0.95β² + 1.82β + 1.43
```

### **Entropy Effective Rank**
**Citation:** Roy & Vetterli (2007) "The effective rank: A measure of effective dimensionality"

```
Normalization: pᵢ = σᵢ / Σⱼ σⱼ
Shannon entropy: H = -Σᵢ pᵢ log pᵢ  
Effective rank: erank = exp(H)
```

### **Knee Detection**  
**Citation:** Satopaa et al. (2011) "Finding a 'Kneedle' in a Haystack"

```
Difference curve: D(i) = y(i) - x(i)
where y = normalized cumulative energy, x = normalized index
Knee point: k = argmax D(i)
```

## 🎯 Practical Recommendations

### **For Production Systems:**
- Start with **default policies** (`energy@0.90,knee,erank`)
- Use **Bench validation** when policies disagree >2x
- **Conservative choice:** Take p90 of policy suggestions
- **Aggressive choice:** Take median of policy suggestions  

### **For Research/Experimentation:**
- Include **OHT** for theoretical interest
- **Compare all policies** against task performance
- **Document policy effectiveness** for your domain
- **Contribute insights** back to Gradience community

### **Red Flags (When to be skeptical):**
- 🚨 **All policies agree perfectly:** May indicate trivial structure
- 🚨 **OHT suggests rank=1:** Check if adaptation actually happened
- 🚨 **eRank >> others:** May be entropy-sensitive outlier
- 🚨 **High disagreement + poor task performance:** Consider different compression approach

---

## 🔬 Bottom Line: Policies as Hypotheses

Each policy embodies a **hypothesis about what matters** in singular value spectra:

- **Energy@90:** "90% energy retention suffices"
- **Knee:** "Humans can visually identify the right cutoff"  
- **eRank:** "Information entropy reflects task dimensionality"
- **OHT:** "Signal detection theory applies to LoRA"

**No policy is universally correct.** The goal is **evidence-based decisions** through empirical validation, not perfect theoretical guarantees.

**Use Bench to test these hypotheses against your actual tasks.** 🧪