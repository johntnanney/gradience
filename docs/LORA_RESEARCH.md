# LoRA Research Agenda

## Why LoRA Matters

**90% of fine-tuning uses LoRA or QLoRA.** This is where the market is.

LoRA (Low-Rank Adaptation) has become the dominant fine-tuning method because:
- Memory efficient: Only train small adapter matrices
- Fast: Fewer parameters to update
- Composable: Can merge/swap adapters
- "Safe": Doesn't modify base weights directly

**But LoRA has failure modes that nobody monitors.**

---

## How LoRA Works

Standard fine-tuning: `W_new = W_base + ΔW`

LoRA constrains the update to be low-rank:
```
W_new = W_base + BA

Where:
  B ∈ ℝ^{d×r}  (down-projection)
  A ∈ ℝ^{r×k}  (up-projection)
  r << min(d, k)  (the "rank" of LoRA)
```

The product `BA` has rank at most `r`, so the update is constrained.

**Scaling factor:**
```
W_new = W_base + (α/r) × BA

Where α = lora_alpha (typically α = r or α = 2r)
```

---

## Research Questions

### RQ1: When does LoRA's effective rank collapse?

**Hypothesis:** Even with nominal rank-64, the effective rank of BA can collapse to near-1 under certain conditions.

**Experiments:**
1. Train LoRA adapters with various configurations
2. Measure effective_rank(BA) throughout training
3. Compare to nominal rank
4. Identify conditions that cause collapse

**Metrics:**
- `effective_rank(BA)` - entropy-based rank of the product
- `effective_rank(A)` - rank of A alone
- `effective_rank(B)` - rank of B alone
- `κ(A)`, `κ(B)` - condition numbers

**Expected findings:**
- High LR → faster collapse
- Small datasets → collapse (not enough signal)
- Narrow task → collapse (only needs few directions)

### RQ2: What's the relationship between α/r and stability?

**Hypothesis:** The ratio α/r controls the effective learning rate for the adapter. Suboptimal ratios cause instability.

**Experiments:**
1. Fix r, vary α
2. Fix α, vary r
3. Measure stability metrics (gradient variance, κ volatility)

**Key insight:** `(α/r) × BA` means:
- Higher α → larger effective update
- Higher r → more capacity but updates divided across more dims

### RQ3: Do different layers need different LoRA ranks?

**Hypothesis:** Attention layers vs. FFN layers may have different optimal ranks.

**Experiments:**
1. Apply different ranks to different layer types
2. Measure which layers use their capacity
3. Identify if rank can be reduced in certain layers

**Product opportunity:** "We recommend rank-32 for attention, rank-8 for FFN based on capacity utilization."

### RQ4: Can we predict optimal LoRA rank from task/data?

**Hypothesis:** The spectral structure of gradients early in training predicts how much rank is needed.

**Experiments:**
1. Run initial gradient accumulation (no weight updates)
2. Compute effective rank of accumulated gradient
3. Compare to optimal LoRA rank found empirically

**Product opportunity:** "Based on your data, we recommend LoRA r=16" (before training starts).

### RQ5: How does QLoRA quantization affect spectral properties?

**Hypothesis:** 4-bit quantization changes the spectral structure of base weights, which affects adapter dynamics.

**Experiments:**
1. Compare spectral metrics of fp16 vs. 4-bit base
2. Track adapter behavior on quantized vs. non-quantized
3. Identify if QLoRA needs different monitoring thresholds

---

## Key Metrics for LoRA Monitoring

### Adapter-Level Metrics

```python
@dataclass
class LoRAAdapterMetrics:
    # Nominal configuration
    nominal_rank: int           # Configured rank (r)
    alpha: float                # Scaling factor (α)
    
    # Effective capacity
    effective_rank_BA: float    # Effective rank of product BA
    effective_rank_A: float     # Effective rank of A
    effective_rank_B: float     # Effective rank of B
    rank_utilization: float     # effective_rank_BA / nominal_rank
    
    # Conditioning
    kappa_A: float              # Condition number of A
    kappa_B: float              # Condition number of B
    
    # Magnitude
    sigma_max_BA: float         # Spectral norm of BA
    frobenius_BA: float         # Frobenius norm of BA
    
    # Update scale
    effective_lr_scale: float   # (α/r) × ||BA|| relative to ||W_base||
```

### Structural Metrics (LoRA's Muon Ratio)

In pretraining, ρ = λ × σ_max measures expansion vs regularization.
In LoRA, the "gravity" is the frozen base model itself.

```python
@dataclass
class LoRAStructuralMetrics:
    # Adapter dominance (LoRA's equivalent of muon ratio)
    # ρ_lora = σ_max(BA) / σ_max(W_base)
    mean_dominance: float           # Mean across all adapters
    max_dominance: float            # Worst case
    scaled_mean_dominance: float    # Including (α/r) scaling
    
    # Per-layer breakdown
    per_layer_dominance: Dict[str, float]
    
    # Conditioning health
    adapter_health: float           # 1 / max(κ(A), κ(B))
    
    # Alerts
    is_dominating: bool             # Any layer with dominance > 0.3
    is_ill_conditioned: bool        # Any κ > 1000
```

**Interpretation of ρ_lora:**
```
ρ_lora < 0.05  →  Minimal modification (very conservative)
ρ_lora ~ 0.10  →  Healthy fine-tuning  
ρ_lora > 0.30  →  Significant modification (warning)
ρ_lora > 0.50  →  Adapter dominating base (critical)
```

**The key insight:** In LoRA, we're not worried about weights exploding (base is frozen). We're worried about the adapter *overwhelming* the base model's learned representations.

### Training Dynamics Metrics

```python
@dataclass  
class LoRADynamicsMetrics:
    step: int
    
    # Rank trajectory
    rank_utilization_trend: str    # "stable", "increasing", "collapsing"
    steps_to_saturation: int       # When did rank stop changing?
    
    # Conditioning trajectory
    kappa_A_volatility: float      # Is A becoming ill-conditioned?
    kappa_B_volatility: float      # Is B becoming ill-conditioned?
    
    # Gradient flow
    gradient_rank: float           # Effective rank of gradients through adapter
```

---

## Alert Conditions for LoRA

### 1. Effective Rank Too Low
```
Trigger: effective_rank_BA < 0.25 × nominal_rank
Message: "LoRA adapter using only 15% of configured rank-64"
Action: Suggest reducing rank or adjusting alpha
```

### 2. Adapter Ill-Conditioned
```
Trigger: κ(A) > 1000 or κ(B) > 1000
Message: "LoRA matrices becoming ill-conditioned"
Action: Suggest reducing learning rate
```

### 3. Rank Collapsed Early
```
Trigger: effective_rank_BA dropped >50% in first 100 steps
Message: "LoRA adapter collapsed during warmup"
Action: Suggest different initialization or alpha
```

### 4. Rank Saturated
```
Trigger: effective_rank_BA hasn't changed in N steps
Message: "LoRA adaptation has saturated"
Action: Consider stopping or the task may be learned
```

### 5. Scale Mismatch
```
Trigger: (α/r) × ||BA|| >> ||W_base|| or << ||W_base||
Message: "LoRA update scale may be suboptimal"
Action: Suggest adjusting alpha
```

---

## Implementation Plan

### Phase 1: Basic LoRA Analyzer (This Week)

```python
class LoRAAnalyzer:
    """Analyze LoRA adapter spectral properties."""
    
    def __init__(self, model):
        self.adapters = self._find_lora_adapters(model)
    
    def analyze(self) -> Dict[str, LoRAAdapterMetrics]:
        """Compute metrics for all adapters."""
        pass
    
    def get_rank_utilization(self) -> float:
        """Overall rank utilization across adapters."""
        pass
    
    def suggest_rank(self) -> int:
        """Suggest optimal rank based on observed utilization."""
        pass
```

### Phase 2: LoRA-Specific Alerts (Next Week)

- Integrate LoRAAnalyzer with FineTuneMonitor
- Add LoRA-specific alert conditions
- Test on real LoRA training runs

### Phase 3: LoRA Optimization Advisor (Week 3)

- Predict optimal rank from gradients
- Layer-wise rank recommendations
- Alpha/r ratio tuning suggestions

---

## Validation Experiments

### Experiment 1: Rank Collapse Demonstration

**Setup:**
- Model: LLaMA-7B or similar
- LoRA: rank-64
- Task: Simple classification (should need low rank)
- Monitor: effective_rank(BA) over training

**Expected result:** Effective rank collapses to ~5-10, proving rank-64 is wasteful.

### Experiment 2: Optimal Rank Search

**Setup:**
- Model: Same base
- Task: Same task
- LoRA ranks: 4, 8, 16, 32, 64
- Metric: Final accuracy + training cost

**Expected result:** Find that rank-16 matches rank-64 performance (validating our rank suggestion).

### Experiment 3: α/r Sensitivity

**Setup:**
- Fixed rank = 16
- Vary α = 8, 16, 32, 64
- Monitor stability + final performance

**Expected result:** Characterize the α/r tradeoff.

### Experiment 4: Early Rank Prediction

**Setup:**
- Compute gradient rank at step 0 (before updates)
- Compare to empirically optimal LoRA rank
- Many tasks/datasets

**Expected result:** Correlation between gradient rank and optimal LoRA rank.

---

## References

- Hu et al., 2021: "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al., 2023: "QLoRA: Efficient Finetuning of Quantized LLMs"
- Lialin et al., 2023: "ReLoRA: High-Rank Training Through Low-Rank Updates"

---

## Product Implications

### The Pitch

> "We ensure your LoRA adapter uses the rank you paid for."

### Key Features

1. **Rank utilization monitoring:** See if you're wasting VRAM
2. **Optimal rank suggestion:** "Based on this task, r=16 is sufficient"
3. **Conditioning alerts:** Catch instability early
4. **Layer-wise analysis:** Which layers need more/less rank

### Value Proposition

```
Without Gradience:
  - Use rank-64 everywhere "to be safe"
  - Waste VRAM on unused capacity
  - No visibility into adapter health

With Gradience:
  - Know exact rank needed
  - Detect collapse before it affects quality
  - Optimize memory/performance tradeoff
```
