# Rank Selection Policies in Gradience

Gradience supports multiple **scientifically-grounded policies** for LoRA rank selection, enabling **evidence-based compression decisions** beyond single heuristics.

> 📖 **For detailed explanations, see [RANK_POLICIES_GUIDE.md](RANK_POLICIES_GUIDE.md)**

## Quick Policy Overview

| Policy | What it measures | Typical behavior | Best for |
|--------|-----------------|------------------|----------|
| **energy@0.90** | Cumulative energy retention | Conservative (medium ranks) | Backward compatibility |
| **knee** | Elbow point in scree plot | Aggressive (low ranks) | Visual intuition |
| **erank** | Information-theoretic dimensionality | Medium (varies with entropy) | Tail-sensitive analysis |
| **oht** | Signal vs noise separation | Aggressive (low ranks) | Theoretical foundation |

## Default Usage

```bash
# Sensible defaults (energy@0.90, knee, erank)
gradience audit --peft-dir ./adapter

# Output includes policy comparison table:
# Policy        Median  P90  Max  Don't Compress
# energy@0.90      4     6    8       25%
# knee             2     3    4        0%  
# erank            6     7    8       50%
```

## When Policies Disagree

**Large disagreements reveal structural insights:**

```bash
# Example: energy@90=8, knee=3, erank=6, oht=2
# → Head-heavy adapter with gradual tail decay
# → Prime candidate for Bench validation!
```

**Test multiple ranks empirically:**
```bash
gradience bench config.yaml  # Auto-tests policy suggestions
```

## Custom Policy Selection

```bash
# Conservative analysis
gradience audit --peft-dir ./adapter --rank-policies energy@0.90,energy@0.95

# Aggressive analysis  
gradience audit --peft-dir ./adapter --rank-policies knee,oht

# Full comparison 
gradience audit --peft-dir ./adapter --rank-policies energy@0.90,knee,erank,oht
```

## Bench Integration (Automatic)

Policies automatically create compression variants for empirical testing:

```bash
gradience bench config.yaml
# Auto-creates: uniform_knee_p90, uniform_erank_p90, uniform_oht_p90
```

Or specify custom policy targeting:

```yaml
compression:
  svd_variants:
    - name: aggressive_choice
      rank_source: audit.rank_suggestions.oht.uniform_median
    - name: balanced_choice  
      rank_source: audit.rank_suggestions.knee.uniform_p90
    - name: conservative_choice
      rank_source: audit.rank_suggestions.energy_90.uniform_p90
```

## Policy-Based Workflow

1. **Audit:** `gradience audit --peft-dir ./adapter` 
2. **Analyze disagreements:** Large gaps = structural insights
3. **Bench validation:** `gradience bench config.yaml`
4. **Evidence-based decisions:** Choose policy based on task performance

## Key Principle: Policies as Hypotheses

Each policy embodies a **hypothesis** about rank selection:
- **energy@90:** "90% energy retention suffices" 
- **knee:** "Humans can visually identify the right cutoff"
- **erank:** "Information entropy reflects task dimensionality"
- **oht:** "Signal detection theory applies to LoRA" (experimental)

**No policy is universally correct.** Use Bench to test hypotheses against your actual tasks. 🧪

---

> 📖 **For complete explanations, mathematical details, and troubleshooting, see [RANK_POLICIES_GUIDE.md](RANK_POLICIES_GUIDE.md)**