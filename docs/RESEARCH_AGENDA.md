# Gradience Research Agenda

## Positioning

Gradience is a **research instrument** for studying the geometry and dynamics of neural network training. It provides dense telemetry on spectral properties, curvature estimates, and information-geometric quantities that are otherwise difficult to observe.

**Target audience**: ML researchers studying optimization, generalization, and training dynamics.

**Not a product**: We make no claims about "early warning" or "prediction" without rigorous validation. The goal is to generate empirical findings about training geometry.

---

## Core Research Questions

### 1. Weight Spectrum ↔ Hessian Spectrum Relationship

**Background**: The weight matrices W define the linear operators in each layer. The Hessian H = ∇²L defines local curvature of the loss landscape. These are related but distinct:

- W spectrum: Properties of the forward/backward pass operators
- H spectrum: How the loss changes with parameter perturbations

The Neural Tangent Kernel literature connects these at initialization, but the relationship during training is less understood.

**Key Questions**:
- Does the weight spectrum predict Hessian spectral properties?
- Do they co-evolve, or does one lead the other?
- At instability, which changes first?

**Measurements needed**:
- Full singular value spectrum of weight matrices (per layer)
- Top-k Hessian eigenvalues (via power iteration / Lanczos)
- Trace of Hessian (via Hutchinson estimator)
- Cross-correlation analysis over training

**Hypothesis**: Weight matrix ill-conditioning (high κ) may precede Hessian ill-conditioning because gradient propagation degrades before the loss surface "knows" about it.

---

### 2. Phase Transitions in Training

**Background**: Statistical mechanics perspective treats training as a dynamical system that may exhibit phase transitions:

- **First-order**: Discontinuous jumps (loss suddenly drops)
- **Second-order**: Continuous but with diverging susceptibility (grokking?)
- **Critical phenomena**: Correlation lengths diverge, fluctuations increase

"Grokking" (delayed generalization) looks like a phase transition—the model suddenly transitions from memorization to generalization.

**Key Questions**:
- Can we detect signatures of approaching phase transitions?
- Is there "critical slowing down" before transitions?
- Do spectral metrics show characteristic scaling near criticality?

**Measurements needed**:
- Autocorrelation time of various metrics (does it diverge?)
- Variance/fluctuation magnitude (does it spike?)
- Loss curvature (second derivative estimates)
- Generalization gap dynamics (train vs val loss)

**Hypothesis**: Near phase transitions, we should observe:
1. Increasing autocorrelation time (system "slows down")
2. Increasing variance in gradients and activations
3. Possible power-law distributions in fluctuations

---

### 3. Information Geometry of Training

**Background**: The Fisher Information Matrix F defines a Riemannian metric on parameter space:

    F_ij = E[∂log p(y|x,θ)/∂θ_i · ∂log p(y|x,θ)/∂θ_j]

This is related to the Hessian: for negative log-likelihood, F ≈ H under certain conditions.

The natural gradient ∇̃L = F⁻¹∇L accounts for the geometry—it's the steepest descent direction in the Fisher metric, not Euclidean.

High condition number κ(F) means parameter space is "stretched"—some directions carry much more information about the output distribution than others.

**Key Questions**:
- Does κ̃ (weight condition) correlate with κ(F) (Fisher condition)?
- Is SGD implicitly approximating natural gradient when training is stable?
- Does instability correspond to extreme anisotropy in the Fisher metric?

**Measurements needed**:
- Empirical Fisher: F̂ = (1/n) Σ ∇log p(y|x,θ) ∇log p(y|x,θ)ᵀ
- Top eigenvalues of F̂ (via power iteration)
- Trace(F̂) and Trace(F̂²) for effective dimensionality
- Comparison with weight matrix spectra

**Hypothesis**: Training instability may correspond to extreme anisotropy in the Fisher metric—the loss landscape becomes so stretched that uniform-LR SGD can't navigate it.

---

### 4. Implicit Regularization and Rank Dynamics

**Background**: Theory suggests SGD on overparameterized networks implicitly regularizes toward low-rank solutions. This is part of the "simplicity bias" story—why do NNs generalize despite having more parameters than data?

The singular value spectrum of weight matrices encodes effective rank. Spectral decay (how fast σ_i drops with i) measures complexity.

**Key Questions**:
- How does effective rank evolve during training?
- Does stable training correspond to smooth rank dynamics?
- Do instabilities correlate with rank spikes or sudden spectral changes?
- Is there a characteristic spectral signature of "healthy" vs "pathological" training?

**Measurements needed**:
- Full singular value spectrum per layer
- Effective rank: exp(entropy of normalized singular values)
- Stable rank: ||W||_F² / ||W||_2²
- Nuclear norm (sum of singular values)
- Spectral decay rate (fit power law σ_i ~ i^(-α))

**Hypothesis**: Stable training exhibits smooth, monotonically decreasing effective rank as the network finds a low-complexity solution. Instabilities may show rank inflation or oscillation.

---

## Experimental Protocols

### Protocol A: Hessian-Weight Co-evolution Study

1. Train small transformer (GPT-2 scale) on standard corpus
2. Every N steps, compute:
   - Full SVD of all weight matrices
   - Top-10 Hessian eigenvalues (Lanczos)
   - Hessian trace (Hutchinson, 100 samples)
3. Vary training regime:
   - Stable (conservative LR, gradient clipping)
   - Marginal (LR at edge of stability)
   - Unstable (LR too high, will eventually diverge)
4. Analyze cross-correlation structure

### Protocol B: Grokking Phase Transition Study

1. Use algorithmic dataset (modular arithmetic) known to exhibit grokking
2. Dense telemetry throughout training (every step if feasible)
3. Track:
   - Train/val loss and accuracy
   - All spectral metrics
   - Gradient statistics
   - Autocorrelation times
4. Identify transition point, analyze precursor signatures

### Protocol C: Information Geometry Tracking

1. Train with dense Fisher estimation (expensive—small model only)
2. Compare Fisher spectrum with weight spectra
3. Compute "natural gradient alignment": cos(∇L, F⁻¹∇L)
4. Hypothesize: stable training ≈ high alignment, instability ≈ low alignment

### Protocol D: Rank Evolution Survey

1. Train multiple architectures: transformer, CNN, MLP
2. Track effective rank per layer throughout training
3. Characterize "healthy" rank trajectories
4. Induce instability, observe rank dynamics

---

## What Success Looks Like

**Publishable findings might include**:

1. "The weight spectrum leads the Hessian spectrum by K steps during approach to instability"
2. "Grokking is preceded by diverging autocorrelation time in layer-wise spectral metrics"
3. "Stable training maintains Fisher-weight spectral alignment above threshold T"
4. "Effective rank follows characteristic trajectory: expansion → compression → plateau"

**What would falsify our hypotheses**:

1. No significant cross-correlation between weight and Hessian spectra
2. No detectable precursors to phase transitions
3. Fisher and weight spectra are unrelated in practice
4. Rank dynamics are noisy/random, not structured

---

## Relation to Prior Work

- **Hessian eigenspectrum**: Sagun et al., Ghorbani et al.—we extend to track co-evolution with weight spectra
- **Grokking**: Power et al.—we add spectral perspective on the transition
- **Fisher information**: Amari (natural gradient), Martens (K-FAC)—we study empirically during training
- **Implicit regularization**: Gunasekar, Arora, etc.—we provide empirical spectral evidence
- **Loss landscape geometry**: Li et al. (visualizations), Fort et al.—we add dynamical perspective

---

## Prioritization for GPU Time

Given limited compute, prioritize:

1. **Protocol D (Rank Evolution)**: Cheapest (no Hessian), high signal potential
2. **Protocol B (Grokking)**: Small model, known phenomenon, publishable angle
3. **Protocol A (Hessian Co-evolution)**: Moderate cost, novel contribution
4. **Protocol C (Fisher Geometry)**: Most expensive, do last if resources allow
