# Research Roadmap

Open questions and planned investigations, organized by time horizon.
This is a research agenda, not a product roadmap.

---

## Near-Term: Exploiting Current Capabilities

These investigations require no new instrumentation -- only systematic
application of Gradience's existing spectral audit, merge audit, and
multi-seed benchmarking.

### Spectral trajectory analysis during training

**Question:** How does the singular value spectrum of each layer evolve
over the course of fine-tuning? Is rank compression monotonic, or does
it exhibit non-monotonic phases (exploration followed by consolidation)?

**Method:** Use Gradience's telemetry hooks (`vnext.telemetry`) to record
per-layer SVD snapshots at regular training intervals. Plot spectral
trajectories: stable rank vs. step, energy rank vs. step, and full
singular value waterfall plots.

**Why it matters:** If spectral trajectories have a characteristic shape,
early checkpoints could predict the final rank structure, enabling
adaptive rank allocation during training.

### Cross-architecture spectral comparison

**Question:** Do Mistral, LLaMA, Phi, and Gemma produce qualitatively
different spectral structures when fine-tuned on the same task? Are there
architectural features (grouped query attention, gated MLPs, depth) that
predict spectral geometry?

**Method:** Run `gradience audit` on adapters from multiple architectures
fine-tuned on GSM8K with matched hyperparameter sweeps. Compare stable
rank distributions, energy concentration, and attention-vs-MLP patterns.

**Why it matters:** Architecture-dependent spectral signatures would
indicate that capacity allocation is partly determined by the model
structure, not just the task, which would inform rank allocation heuristics.

### Multi-task spectral fingerprinting

**Question:** Do different tasks produce distinguishable spectral
fingerprints on the same architecture? Can spectral audit predict which
adapters are functionally compatible for merging based on their spectral
similarity?

**Method:** Fine-tune the same model on multiple tasks (GSM8K, coding,
summarization, instruction-following). Compare per-layer spectral
profiles and test whether spectral similarity correlates with merge
compatibility as measured by `gradience merge-audit`.

### Rank policy validation study

**Question:** Which rank policy (`energy_threshold`, `knee_elbow`,
`optimal_hard_threshold`, `entropy_effective`, `stable_rank_ceil`)
best predicts the minimum rank that preserves downstream accuracy?

**Method:** For each layer, truncate to the rank suggested by each
policy and measure accuracy. Compute per-policy accuracy-vs-compression
Pareto fronts across layers and seeds.

**Why it matters:** Currently, policy selection is a user choice.
Systematic validation would identify which policy (or ensemble) is
most reliable, and under what spectral conditions.


---

## Medium-Term: New Instrumentation

These require extending Gradience with new measurement capabilities,
using the infrastructure already scaffolded in `gradience.research.*`.

### Hessian-spectrum co-evolution

**Question:** Do the top Hessian eigenvectors align with the top singular
vectors of $\Delta W$ during training? Does this alignment strengthen over
time?

**Method:** Extend `research.hessian` to compute directional Hessian
curvature along adapter singular vectors at training checkpoints.
Measure the cosine similarity between top-$k$ Hessian eigenvectors and
top-$k$ right singular vectors of $\Delta W$.

**Why it matters:** A strong alignment would provide theoretical
justification for spectral compression: truncated directions are not
just low-energy, they are low-curvature (flat directions in the loss
landscape that do not affect the loss).

### Phase transition detection in spectral observables

**Question:** Can spectral quantities (stable rank, energy concentration)
serve as order parameters for detecting training phase transitions
(grokking, sudden generalization, mode collapse)?

**Method:** Use `research.phase_transitions` to compute autocorrelation
time, variance ratios, and susceptibility of spectral observables during
training. Look for divergences that precede qualitative changes in the
loss curve or evaluation metrics.

**Why it matters:** Early detection of phase transitions would enable
adaptive training strategies (early stopping, learning rate adjustment,
rank reallocation) before performance degrades.

### Fisher-spectrum correspondence

**Question:** Does the entropy effective rank of the adapter correlate
with the effective dimensionality of the empirical Fisher information,
restricted to the adapter subspace?

**Method:** Use `research.fisher` to estimate the empirical Fisher
spectrum for the adapter parameters. Compare $\text{erank}(\Delta W)$
with $\text{erank}(F|_{\text{adapter}})$ across layers and training
stages.

**Why it matters:** A positive correlation would connect spectral audit
(cheap, post-hoc) to Fisher geometry (expensive, online), validating
the spectral audit as a tractable proxy for information-geometric
analysis.

### Spectral dynamics of catastrophic forgetting

**Question:** When fine-tuning erases previously learned capabilities
(catastrophic forgetting), what happens to the spectral structure? Does
forgetting correspond to rank inflation, subspace rotation, or energy
redistribution?

**Method:** Fine-tune sequentially on two tasks. After each task,
run `gradience audit` and compare spectral profiles. Use principal
angle analysis to measure how much the learned subspace rotates
between tasks.


---

## Long-Term: Open Theoretical Questions

These are fundamental research questions that Gradience can help
investigate but that likely require contributions from the broader
research community.

### Spectral scaling laws

**Question:** How does $\text{erank}(\Delta W)$ scale with model size $N$,
dataset size $D$, and training compute $C$? Do spectral quantities obey
power laws analogous to the Chinchilla scaling laws for loss?

**Expected form:** $\text{erank}(\Delta W) \sim N^\alpha D^\beta C^\gamma$
for some exponents $\alpha, \beta, \gamma$. If $\alpha < 0$, larger
models use proportionally fewer effective dimensions, suggesting that
overparameterization makes representations more compressible.

### Generalization bounds from spectral complexity

**Question:** Can stable rank or entropy effective rank of the adapter
provide tighter PAC-Bayes or compression-based generalization bounds than
parameter-count-based bounds?

**Potential approach:** Use stable rank as a data-dependent complexity
measure in PAC-Bayes bounds. The effective parameter count
$\sum_l \text{srank}(\Delta W_l) \cdot (d_{\text{in},l} + d_{\text{out},l})$
may be a tighter capacity measure than $\sum_l r_l \cdot (d_{\text{in},l} + d_{\text{out},l})$.

### Cross-architecture geometric universals

**Question:** Is there a universal spectral geometry shared by all
architectures trained on the same task, or does architecture determine
the spectral structure? Are there invariant spectral quantities under
architectural changes (analogous to universal critical exponents in
statistical mechanics)?

### The rank allocation problem

**Question:** Given a fixed parameter budget, what is the optimal
per-layer rank allocation? Can spectral audit of a cheap pilot run
(short training, small rank) predict the optimal allocation for a
full training run?

**Potential approach:** Train a pilot adapter at uniform low rank,
audit its spectral structure, use the spectral profile to allocate
higher rank to layers that need it, and validate with full training.


---

## Collaboration Opportunities

Gradience provides the measurement infrastructure. Several of the above
questions would benefit from external expertise:

- **Theoretical ML:** Spectral scaling laws, generalization bounds from
  spectral complexity, connections to PAC-Bayes or compression-based
  learning theory.
- **Statistical mechanics / phase transitions:** Rigorous application of
  critical phenomena theory to training dynamics, universality classes
  for spectral transitions.
- **Information geometry:** Fisher-spectrum correspondence, natural gradient
  connections, Riemannian optimization on adapter manifolds.
- **Empirical scaling:** Large-scale compute for cross-architecture and
  cross-scale spectral comparisons (requires many training runs on
  diverse hardware).

Researchers interested in using Gradience's spectral audit as a
measurement tool for their own investigations are encouraged to
contribute findings back to this document.
