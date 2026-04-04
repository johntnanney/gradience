# ANALYTICAL_SPECTRAL_GEOMETRY_OF_MERGE_OPERATIONS_SPEC
## Repo-Facing CPU-Only Research Plan

## Purpose

Define a CPU-only mathematical research line that derives closed-form or
semi-analytical results about how merge operations transform singular value
spectra:

> **Analytical Spectral Geometry of Merge Operations**

Central question:

> **Given two LoRA adapters with known SVD decompositions and known principal
> angles between their column spaces, what can be proved about the singular
> value spectrum of the merged output under each merge strategy?**

This is purely mathematical work. It requires no training compute, no
adapter downloads, no GPU access. The deliverables are theorems,
inequalities, and conditions expressed in quantities Gradience already
computes — which can then be validated numerically on synthetic matrices
using existing CPU infrastructure.

## Relationship to Existing Infrastructure

This study builds directly on and feeds back into three existing modules:

1. **`spectral_compat.py`**: computes the exact quantities (principal
   angles, directional agreement, stable rank, effective rank, Frobenius
   norms) that parameterize the analytical results. The theorems derived
   here will be stated in terms of `SubspaceMetrics` fields.

2. **`over_accumulation.py`**: the current heuristic diagnostic for
   shared-direction inflation under linear merge. The CPU consolidation
   memo classified this as "exploratory, structural signal exists, but
   policy-direction validity is weak/mixed." Analytical results should
   either provide a theoretical foundation for the heuristic's structure
   or identify where it fails.

3. **`strategies.py`**: implements the five merge strategies (linear,
   TIES, DARE-linear, DARE-TIES, norm-equalized). Each strategy defines
   a specific mathematical operation on ΔW matrices. The analytical work
   derives what each operation does to the spectrum as a function of the
   inter-adapter geometry.

4. **`verdicts.py`**: the 6-branch decision tree that recommends merge
   strategies based on spectral metrics. Analytical results may sharpen
   the thresholds or provide principled justification for the branch
   structure.

## Why This Study Now

The over-accumulation diagnostic was the one component of the CPU
consolidation that did not achieve a clean bounded-positive outcome. Its
status is "exploratory" — the structural signal exists, but the mapping
from spectral observables to merge-quality predictions lacks theoretical
grounding. The heuristic scoring formula
(`alignment × (0.7 × concentration + 0.3 × coefficient_exposure)`) was
designed by intuition and tuned on limited empirical data.

A mathematical analysis can:

- Replace heuristic weights with principled inequalities.
- Identify regimes where each merge strategy is provably better or worse.
- Provide exact conditions (not threshold-fitted approximations) for when
  over-accumulation occurs.
- Give the verdict tree's branch boundaries a theoretical interpretation.

This work has zero compute cost beyond a laptop with a Python environment.
It is the highest-leverage CPU-only investment for improving the merge
pipeline's theoretical foundations.

## Scope

### In scope

- Closed-form spectral analysis of linear merge under known SVD
  parameterization and principal angles
- Semi-analytical bounds for TIES, DARE, and norm-equalized merge
- Numerical validation on synthetic matrix pairs with controlled geometry
- Connection to existing `SubspaceMetrics` quantities
- Conditions/inequalities expressed as functions of quantities Gradience
  already computes
- Comparison of analytical predictions to empirical field trial data
  (where available)

### Out of scope

- Downstream task performance prediction (no behavioral evaluation)
- New merge strategy invention (analysis of existing strategies only)
- GPU compute or adapter training
- Changes to production merge pipeline from first-pass results
- General matrix perturbation theory not specific to LoRA structure

## Mathematical Framework

### Setup and notation

Two LoRA adapters share a base model and target the same module. Each
adapter contributes a low-rank weight update:

```
ΔW_a = (α_a / r_a) · B_a A_a     where B_a ∈ ℝ^{d_out × r_a}, A_a ∈ ℝ^{r_a × d_in}
ΔW_b = (α_b / r_b) · B_b A_b     where B_b ∈ ℝ^{d_out × r_b}, A_b ∈ ℝ^{r_b × d_in}
```

Each has a thin SVD (computed via the QR-based method in
`compute_layer_svd`):

```
ΔW_a = U_a Σ_a V_a^T     Σ_a = diag(σ_{a,1}, ..., σ_{a,r_a})
ΔW_b = U_b Σ_b V_b^T     Σ_b = diag(σ_{b,1}, ..., σ_{b,r_b})
```

The geometric relationship between the two adapters is captured by
principal angles. Let `k = min(r_a, r_b)`. The principal angle cosines
between the column spaces of U_a and U_b are:

```
cos(θ_1) ≥ cos(θ_2) ≥ ... ≥ cos(θ_k) ≥ 0
```

computed as the singular values of `U_a^T U_b` (the Bjorck-Golub
method, already implemented in `compute_subspace_metrics`).

Similarly, define principal angles `{φ_i}` between the row spaces
(column spaces of V_a and V_b).

### Key derived quantities (already in SubspaceMetrics)

- **Mean overlap**: `μ_θ = (1/k) Σ cos(θ_i)` — `SubspaceMetrics.mean_overlap`
- **Overlap score**: `Ω = (1/k) Σ cos²(θ_i)` — `overlap_score()` in `spectral_compat.py`
- **Directional agreement**: `δ` — `SubspaceMetrics.directional_agreement`
- **Frobenius norms**: `‖ΔW_a‖_F`, `‖ΔW_b‖_F` — `SubspaceMetrics.frobenius_norm_{a,b}`
- **Stable rank**: `sr = ‖ΔW‖_F² / σ_1²` — `SubspaceMetrics.stable_rank_{a,b}`
- **Frobenius ratio**: `ρ_F = max / min` — `SubspaceMetrics.frobenius_ratio`

## Program Questions

### Q1: Linear merge spectral structure

For the linear merge `ΔW_m = α · ΔW_a + β · ΔW_b`:

1. What are the singular values of ΔW_m as a function of
   `{σ_{a,i}}`, `{σ_{b,j}}`, `{θ_i}`, `{φ_i}`, `α`, `β`?
2. Under what conditions on the principal angles does
   `σ_1(ΔW_m) > max(α·σ_1(ΔW_a), β·σ_1(ΔW_b))`?
   (This is the over-accumulation condition.)
3. What is `sr(ΔW_m)` as a function of `sr(ΔW_a)`, `sr(ΔW_b)`, and
   the inter-adapter geometry?

### Q2: Over-accumulation characterization

1. Derive the exact condition under which linear merge inflates the
   leading singular value beyond what either adapter contributes alone.
2. Express this condition in terms of quantities already in
   `SubspaceMetrics`.
3. Compare the derived condition to the heuristic in
   `over_accumulation.py` — does the heuristic approximate the true
   condition? Where does it over- or under-estimate risk?

### Q3: Norm-equalized merge analysis

For the norm-equalized merge (scale both to geometric-mean Frobenius
norm, then linearly combine):

1. Does norm equalization reduce or eliminate over-accumulation risk?
2. What is the spectral distortion introduced by norm equalization
   when adapters have very different spectral shapes (same Frobenius
   norm but different rank structures)?
3. Under what geometric conditions is norm equalization provably
   better than naive linear merge?

### Q4: TIES spectral analysis

The TIES pipeline (trim → elect sign → disjoint mean) operates
element-wise, not spectrally. But it has spectral consequences:

1. What is the expected effect of magnitude trimming on the singular
   value spectrum? (Trimming removes small-magnitude entries, which
   preferentially affects directions with small singular values.)
2. How does sign election interact with subspace overlap? (When
   adapters share a direction with opposing signs, sign election
   resolves the conflict — what does this do to the spectrum?)
3. Can the spectral effect of TIES be bounded in terms of `Ω` and
   the individual spectra?

### Q5: DARE spectral analysis

DARE applies random dropout with rescaling before merging:

1. What is the expected spectrum of a DARE-sparsified adapter?
   (Random dropout is a random projection — its spectral effects
   should be characterizable in expectation.)
2. Does DARE reduce over-accumulation risk in expectation, and if so
   by how much as a function of `Ω`?
3. What is the variance of the merged spectrum under DARE? (DARE
   introduces stochasticity — spectral stability matters for
   reproducibility.)

### Q6: Strategy selection conditions

Given the analytical results from Q1-Q5:

1. For which regions of `(Ω, ρ_F, sr_a, sr_b)` space is each strategy
   provably best (in terms of spectral preservation)?
2. Do these regions align with the verdict tree's branch boundaries
   in `verdicts.py`?
3. Can the current threshold values (`low_overlap=0.2`,
   `high_overlap=0.5`, `imbalanced=5.0`) be derived or justified from
   the analytical conditions?

### Q7: Spectral partitioning and subspace convergence

**Motivation.** Tian, Ledent, and Sun (2026; ICLR 2026, arXiv:2603.01526)
observe empirically that in multi-task LoRA training, the high-SV
directions of adapter B-matrices show 89% inter-task alignment while
low-SV directions show only 3%. This partitioning, if it holds for
independently trained adapters, would provide the generative explanation
for why Gradience's energy-weighted interaction terms predict merge
compatibility — the high-energy directions that dominate the cross term
in Q1 are precisely the shared directions where same-task adapters agree.

**Questions:**

1. **Convergence bound.** Given the pre-trained weight matrix $W_0$ with
   spectral gap $g_k = \sigma_k(W_0) - \sigma_{k+1}(W_0)$, and two
   independently trained LoRA updates $\Delta W_a$, $\Delta W_b$ of
   rank $r \leq k$ satisfying $\|\Delta W\|_2 \leq \epsilon$, can we
   bound the principal angle between their dominant singular subspaces
   as a function of $g_k$ and $\epsilon$? The Davis-Kahan theorem
   provides the tool; the question is whether the bound is tight enough
   to be informative for typical LoRA parameters.

2. **Partitioning threshold.** Is the boundary between the shared and
   task-specific spectral bands predictable from $W_0$'s spectrum?
   Candidate: the Marchenko-Pastur bulk edge (already implemented in
   `optimal_hard_threshold`). If directions above the noise floor in
   $W_0$ constrain adaptation, and directions within the bulk are free
   to specialize, then the `optimal_hard_threshold` policy may
   approximate the natural partitioning point.

3. **Energy-weighted interaction consequence.** Given partitioning
   (high-SV aligned, low-SV orthogonal), does the cross-term bound
   from Q1 simplify? Specifically, if $\cos(\theta_i) \approx 1$ for
   $i \leq k^*$ (shared band) and $\cos(\theta_i) \approx 0$ for
   $i > k^*$ (task-specific band), the sum collapses to the top-$k^*$
   terms. This would formalize the empirical observation that
   energy-rank concentration predicts merge safety.

4. **Block-level vs. component-level.** Tian et al. find that
   block-level LoRA (whole attention block) reduces gradient conflict
   by 76% vs. component-level (individual Q/K/V/O). If block-level
   becomes standard, does the V-module-specific pathology identified
   by Gradience (Technical Report §3.2) still manifest? Can the
   analytical framework predict how spectral signatures redistribute
   when adapter granularity changes?

**Approach.** Q7.1 is purely mathematical (Davis-Kahan applied to the
LoRA setting). Q7.2 and Q7.3 are semi-analytical (bounds + numerical
validation on synthetic matrices with planted spectral gaps). Q7.4 is
conjectural and should wait for empirical evidence of block-level
adapters appearing in the audit corpus.

**Connection to existing work.** Q7.1 provides the theoretical
foundation that Q1's interaction term currently lacks — it explains
*why* the principal angles take the values they do, rather than merely
deriving *what happens given those values*. Together, Q1 + Q7 would
give the full causal chain: pre-trained spectrum constrains adaptation
→ adaptation produces predictable subspace geometry → subspace geometry
determines merge interaction → merge interaction predicts outcome.

## Analytical Approach

### Phase 1: Linear merge (exact results)

The linear merge is the most tractable case because it is a direct
matrix sum. The key mathematical tool is the relationship between the
SVD of a sum and the SVDs of the summands.

**Core derivation**: Express `ΔW_m = α U_a Σ_a V_a^T + β U_b Σ_b V_b^T`
and derive the Gram matrix:

```
ΔW_m^T ΔW_m = α² V_a Σ_a² V_a^T + β² V_b Σ_b² V_b^T
             + αβ (V_a Σ_a U_a^T U_b Σ_b V_b^T + V_b Σ_b U_b^T U_a Σ_a V_a^T)
```

The cross terms involve `U_a^T U_b` — whose singular values are exactly
the principal angle cosines `{cos(θ_i)}` already computed by Gradience.

**Special cases with exact solutions**:

- **Orthogonal subspaces** (`cos(θ_i) = 0 ∀i`): Cross terms vanish.
  `σ²(ΔW_m)` is the union of `α²σ²(ΔW_a)` and `β²σ²(ΔW_b)`. No
  over-accumulation possible. This justifies the SAFE (orthogonal)
  verdict branch.

- **Identical subspaces** (`cos(θ_i) = 1 ∀i`, shared U and V):
  `ΔW_m = (αΣ_a + βΣ_b)` in the shared basis. Singular values add
  directly when directional agreement is positive. Maximum
  over-accumulation. This justifies the REDUNDANT verdict branch.

- **Rank-1 case** (`r_a = r_b = 1`): Fully solvable. The merged
  singular values depend on the single principal angle `θ` between
  the two rank-1 column spaces and the single angle `φ` between
  row spaces. Closed form:

  ```
  σ₁²(ΔW_m) = α²σ_a² + β²σ_b² + 2αβ·σ_a·σ_b·cos(θ)·cos(φ)·sign(δ)
  ```

  where `δ` is the directional agreement. Over-accumulation occurs
  iff the cross term is positive and large enough, which requires
  `cos(θ)·cos(φ)·sign(δ) > 0` and sufficient magnitude.

- **Equal-rank, equal-spectrum case** (`r_a = r_b = r`,
  `Σ_a = Σ_b = Σ`): Simplifies the Gram matrix to a symmetric form
  parameterized only by `{cos(θ_i)}`, `{cos(φ_i)}`, and `α/β`.

**General case (bounds)**: For arbitrary rank and spectrum, use:

- Weyl's inequality: `σ_i(A+B) ≤ σ_j(A) + σ_{i-j+1}(B)` for tight
  upper bounds on merged singular values.
- Fan's inequality for singular value sums.
- Lidskii-Wielandt for finer interlacing when principal angles are known.

The goal is bounds tight enough to be decision-relevant, not
necessarily exact eigenvalue expressions for the general case.

### Phase 2: Over-accumulation theory

**Definition**: Over-accumulation occurs when the merged adapter's
leading singular value exceeds the contribution-weighted maximum:

```
σ_1(ΔW_m) > max(α·σ_1(ΔW_a), β·σ_1(ΔW_b))
```

From the rank-1 analysis, the necessary and sufficient condition
(in the rank-1 case) is:

```
2αβ·σ_a·σ_b·cos(θ)·cos(φ)·sign(δ) > 0
```

i.e., the adapters must have positive directional alignment along at
least one shared direction. This connects directly to the existing
`_alignment_component` in `over_accumulation.py`, which uses
`mean_overlap` and `directional_agreement` as proxies for exactly
this quantity.

**Analytical tasks**:

1. Generalize the rank-1 condition to rank-r with explicit dependence
   on the full set of principal angles.
2. Derive the over-accumulation magnitude (how much inflation) as a
   function of the alignment structure, not just a binary condition.
3. Express the result as an inequality on `SubspaceMetrics` fields:
   find `f(mean_overlap, directional_agreement, stable_rank_a,
   stable_rank_b, frobenius_ratio)` such that over-accumulation
   is bounded by `f`.
4. Compare `f` to the heuristic
   `alignment × (0.7·concentration + 0.3·coefficient_exposure)` and
   identify the approximation errors.

### Phase 3: Norm-equalized analysis

Norm equalization rescales both adapters to the geometric-mean
Frobenius norm `g = √(‖ΔW_a‖_F · ‖ΔW_b‖_F)` before combining:

```
ΔW_m^{neq} = α · (g/‖ΔW_a‖_F) · ΔW_a + β · (g/‖ΔW_b‖_F) · ΔW_b
```

This is equivalent to a linear merge of spectrally rescaled adapters.
The rescaling multiplies all singular values of adapter `a` by
`g/‖ΔW_a‖_F = √(‖ΔW_b‖_F / ‖ΔW_a‖_F)` and similarly for `b`.

**Key questions**:

- The Frobenius ratio `ρ_F` enters directly: rescaling factors are
  `ρ_F^{±1/2}`. When `ρ_F ≈ 1`, norm equalization ≈ linear merge.
  When `ρ_F ≫ 1`, norm equalization compresses the larger adapter
  and amplifies the smaller.

- Does compression of the dominant adapter reduce over-accumulation?
  Intuitively yes — it reduces the effective `α·σ_1(ΔW_a)` that
  enters the cross term. Derive the exact condition.

- When two adapters have the same Frobenius norm but very different
  stable ranks (one flat spectrum, one peaked), norm equalization
  changes nothing about scale but the linear merge still produces
  different spectral distortions. Characterize this case.

### Phase 4: TIES bounds (semi-analytical)

TIES operates element-wise, making exact spectral analysis harder
than for linear merge. The approach is to bound the spectral effect.

**Trimming**: Zeroing out the bottom `p` fraction of entries by
magnitude is a structured sparsification. In the worst case
(adversarial alignment of small entries with important spectral
directions), trimming can destroy a significant singular value. In
the expected case (small entries distributed roughly uniformly across
spectral directions), trimming reduces all singular values by
approximately the same fraction.

- Derive expected spectral effect under the assumption that the
  magnitude distribution of entries is independent of spectral
  direction (the "isotropic trim" assumption).
- Bound the worst-case spectral distortion from trimming.
- Identify when the isotropic assumption fails (highly structured
  adapters with concentrated entries).

**Sign election + disjoint mean**: When two adapters have high
overlap (`Ω ≈ 1`) and opposing directional agreement (`δ < 0`), TIES
resolves the sign conflict by majority vote. In the rank-1 case with
opposing signs, TIES keeps only the entries from the adapter whose
sign wins the vote, effectively discarding one adapter's contribution
per entry. The spectral effect is a stochastic mixture of the two
individual spectra.

- Bound the spectral distortion from sign election as a function
  of `Ω` and `δ`.
- Compare to the CONFLICTING verdict branch, which currently
  recommends TIES for high-overlap + negative-agreement layers.

### Phase 5: DARE bounds (probabilistic)

DARE applies Bernoulli dropout to each entry independently with
probability `p`, then rescales survivors by `1/(1-p)`. This is
a random diagonal projection.

**Expected spectrum**: For a matrix `M`, the DARE-processed matrix
`M̃ = (1/(1-p)) · D · M` where `D` is a diagonal Bernoulli mask has:

```
E[M̃^T M̃] = (1/(1-p)) · M^T M + (p/(1-p)²) · diag(M^T M)
```

The first term preserves the spectrum; the second adds a diagonal
perturbation proportional to the column norms. The spectral effect
is a rank-dependent inflation of singular values, stronger for
directions whose entries have unequal magnitude distribution across
rows.

- Derive the expected singular value perturbation as a function of
  `p` and the original spectrum.
- Derive concentration bounds (Bernstein or matrix Chernoff) on the
  deviation of the actual spectrum from the expected spectrum.
- Characterize when DARE reduces over-accumulation risk: the dropout
  decorrelates the two adapters' shared directions, reducing the
  cross term in the Gram matrix. Quantify this reduction as a
  function of `p` and `Ω`.

### Phase 6: Strategy selection map

Synthesize results from Phases 1-5 into a strategy selection map:
for each region of the observable space
`(Ω, δ, ρ_F, sr_a, sr_b, concentration)`, which strategy minimizes
spectral distortion?

**Deliverable**: A decision tree or partition of the observable space
with:

- For each region, the recommended strategy and the theorem/bound
  that justifies it.
- Comparison to the existing verdict tree in `verdicts.py`.
- Identification of regions where the current tree is provably
  suboptimal and proposed corrections.

## Numerical Validation Plan

Every analytical result is validated numerically on synthetic matrices
before being considered established. This uses only CPU computation.

### Synthetic matrix generation

Construct adapter pairs `(ΔW_a, ΔW_b)` with controlled geometry:

1. **Fix dimensions**: `d_out = 4096`, `d_in = 4096` (representative
   of 7B-class models). Rank `r ∈ {4, 8, 16, 32, 64}`.

2. **Control principal angles**: Generate `U_a` randomly, then
   construct `U_b` such that `U_a^T U_b` has prescribed singular
   values `{cos(θ_i)}`. (Rotate U_a by controlled angles.)

3. **Control spectra**: Prescribe `Σ_a` and `Σ_b` directly (flat,
   peaked, exponential decay, power-law decay profiles).

4. **Control directional agreement**: Prescribe sign relationships
   between corresponding spectral directions.

### Validation protocol

For each analytical result:

1. Generate 1000 random matrix pairs per parameter setting.
2. Compute the analytically predicted quantity.
3. Compute the empirically observed quantity.
4. Report: mean absolute error, max absolute error, fraction of
   cases where the bound holds (should be 100% for proved bounds,
   ~expected coverage for probabilistic bounds).

### Parameter sweep

Sweep over a grid of:

- `Ω ∈ {0.0, 0.1, 0.2, ..., 1.0}` (overlap)
- `δ ∈ {-1.0, -0.5, 0.0, 0.5, 1.0}` (directional agreement)
- `ρ_F ∈ {1.0, 2.0, 5.0, 10.0}` (Frobenius ratio)
- `sr ∈ {1.5, 3.0, 8.0, r}` (stable rank)
- `α/β ∈ {0.5/0.5, 0.7/0.3, 0.9/0.1}` (coefficient asymmetry)

This is ~5000 parameter settings × 1000 samples = 5M matrix pairs.
At ~1ms per pair (SVD of small matrices), total wall time is ~1-2
hours on a modern CPU.

### Empirical cross-check

Where field trial data exists (field_trials/inventory_01 through
inventory_05, checkpoint_inventory_t02), compute the analytical
predictions for the actual adapter pairs and compare to the observed
merge outcomes. This is not the primary validation (synthetic matrices
are more controlled), but it tests whether the analytical results are
useful on real adapters.

## Hypotheses

Pre-execution hypotheses (retained for continuity; see outcome status below):

- **H1**: The over-accumulation condition for linear merge can be
  expressed as a closed-form inequality on `(Ω, δ, ρ_F)` that is
  tighter than the current heuristic.

- **H2**: Norm equalization provably eliminates over-accumulation in
  the `ρ_F ≫ 1` regime, and this explains its empirical success as
  a merge baseline.

- **H3**: TIES spectral distortion is bounded by a decreasing
  function of `|δ|` — i.e., TIES does least damage when the sign
  conflict is strong and most damage when agreement is high. This
  would explain why the verdict tree recommends TIES for conflicting
  layers.

- **H4**: DARE reduces over-accumulation risk proportionally to the
  dropout fraction, with the reduction concentrated in the
  high-overlap regime (`Ω > 0.5`).

- **H5**: The existing verdict tree's branch boundaries approximately
  correspond to transitions between strategy-optimality regions in
  the analytical map, but with at least one provably suboptimal
  threshold.

### Hypothesis outcome status

- **H1: confirmed mathematically, invalidated empirically.** The
  closed-form condition was derived and is tighter than the heuristic
  (cross-term requires both left AND right principal angle products,
  which the heuristic reduces to `mean_overlap` alone). The analytical
  condition correctly predicts spectral inflation on synthetic matrices.
  However, the 30-pair field-trial cross-check showed that spectral
  inflation has the wrong sign for merge quality prediction: higher
  predicted inflation correlates with *better* merges (Spearman +0.38),
  not worse ones. The mathematical answer to H1 is correct; the
  engineering premise was wrong.

- **H2: confirmed analytically, trivially true empirically.** Norm
  equalization does reduce the cross-term when `ρ_F >> 1`. However,
  in the field-trial cohort most pairs are approximately balanced
  (`ρ_F ≈ 1`), so the intervention reduced inflation in only 3.3% of
  layers. The theoretical basis is established but the practical impact
  is marginal in populations without extreme scale imbalance.

- **H3: not pursued.** The empirical cross-check revealed that
  TIES/DARE spectral bounds would answer a question already shown to
  be the wrong one for quality prediction. Work paused before Phase 4.

- **H4: not pursued.** Same rationale as H3.

- **H5: invalidated.** The verdict tree thresholds cannot be derived
  from or justified by spectral inflation conditions, because spectral
  inflation is not the failure mode in the tested population. The branch
  structure is sound as a *geometric classification* tool, but the
  thresholds do not correspond to quality-prediction boundaries.

## Deliverables

### Mathematical deliverables (all written)

```
docs/theory/
├── README.md                              # Index
├── linear_merge_spectral_analysis.md      # Phase 1 results (substantive)
├── over_accumulation_theory.md            # Phase 2 results (substantive)
├── norm_equalized_analysis.md             # Phase 3 stub with derivation targets
├── ties_spectral_bounds.md                # Phase 4 scaffold (not pursued)
├── dare_spectral_bounds.md                # Phase 5 scaffold (not pursued)
├── strategy_selection_map.md              # Phase 6 scaffold (not pursued)
└── analytical_spectral_geometry_summary.md # Executive summary
```

### Code deliverables (all implemented)

```
gradience/vnext/merge/
├── spectral_theory.py              # 13 dataclasses, 11 functions:
│                                   #   rank1_linear_merge_result()
│                                   #   equal_rank_equal_spectrum_merge()
│                                   #   linear_merge_frobenius()
│                                   #   general_rank_gram_decomposition()
│                                   #   linear_merge_stable_rank_bounds()
│                                   #   linear_merge_sigma1_bounds()
│                                   #   norm_equalized_scaling()
│                                   #   expected_dare_frobenius_multiplier()
│                                   #   over_accumulation_exact_condition()
│                                   #   norm_equalized_over_accumulation_analysis()
│                                   #   compute_rankr_interaction_geometry()
│                                   #   estimate_over_accumulation_theory()
│                                   #   compare_over_accumulation_theory_to_heuristic()
│                                   #   estimate_linear_merge_bounds_from_metrics()
└── spectral_theory_test_utils.py   # Synthetic matrix generation:
                                    #   make_rankr_pair_with_geometry()
                                    #   make_equal_spectrum_pair()
                                    #   frobenius(), stable_rank(), singular_values()

tests/merge/
├── test_spectral_theory.py            # 22 unit tests
└── test_spectral_theory_validation.py # 4 numerical validation sweeps
```

### Infrastructure changes (v2 geometry fields)

```
gradience/vnext/merge/spectral_compat.py:
  SubspaceMetrics now carries:
  - right_principal_angle_cosines  (row-space cos(φ_i), descending)
  - effective_singular_values_a    (top effective-rank singular values)
  - effective_singular_values_b    (top effective-rank singular values)
  compute_subspace_metrics() now computes row-space principal angles.

gradience/vnext/merge/over_accumulation.py:
  New parallel estimator:
  - estimate_over_accumulation_v2()  (interaction-first, non-authoritative)
  - OverAccumulationEstimateV2, OverAccumulationFactorsV2 dataclasses
  OA-v1 remains authoritative production path.
```

### Field trial deliverables (all written)

```
field_trials/analytical_spectral_geometry/
├── manifest.json                                    # Study metadata (negative_completion)
├── synthetic_validation_results.json                # 6,912-case synthetic sweep
├── heuristic_comparison.json                        # OA-v1 vs OA-v2 on synthetic data
├── empirical_crosscheck_oa_v2_30_40_r1.json         # 30-pair v1/v2 cross-check
├── theory_crosscheck.json                           # 30-pair theory + v1 + v2 + outcomes
├── theory_crosscheck.md                             # Per-pair summary table
├── run_phase2_heuristic_comparison.py               # Higher-rank synthetic sweep runner
├── run_empirical_crosscheck.py                      # Strict-naive cross-check runner
├── run_oa_v2_failure_anatomy.py                     # Failure anatomy decomposition
├── run_theory_crosscheck.py                         # Theory cross-check runner
└── study_memo.md                                    # Full write-up (negative completion)
```

## Success / Failure Criteria

### Success condition

Closed-form over-accumulation condition for linear merge (Q2) is
derived, validated numerically, and expressed in `SubspaceMetrics`
terms. At least two additional strategies (from Q3-Q5) have useful
semi-analytical bounds. The strategy selection map (Q6) identifies
at least one region where the current verdict tree is provably
suboptimal.

### Partial success condition

Linear merge analysis (Q1-Q2) produces useful results, but TIES and
DARE resist clean spectral characterization (their element-wise
operations break spectral structure in ways that only probabilistic
bounds can capture). Over-accumulation theory improves on the
heuristic but does not fully replace it. Strategy selection map is
suggestive but not conclusive.

### Negative completion condition

The general-rank over-accumulation condition cannot be expressed in
terms of the quantities Gradience currently computes — it requires
additional geometric information (e.g., individual principal vector
alignments, not just angle cosines) that the pipeline does not
extract. The analytical work identifies what additional quantities
would be needed, which becomes an infrastructure requirement for
future work.

All outcomes are useful and should be documented as such.

### Actual outcome: negative completion (informative)

**Status: negative completion.** The study hit a variant of the
negative completion condition — not because the quantities are
insufficient to express the mathematical condition (they are
sufficient; the exact rank-r condition was derived and validated),
but because the mathematical condition answers the wrong engineering
question.

The 30-pair field-trial cross-check showed:

| Metric | Spearman vs merge quality | Direction |
|--------|---------------------------|-----------|
| OA-v1 max score | +0.186 | wrong |
| OA-v2 max score | +0.262 | wrong |
| Theory risk score | +0.382 | wrong |
| Theory inflation ratio | +0.386 | wrong |

All three metrics (heuristic v1, interaction-first v2, analytical
theory) have wrong-sign correlation with merge quality. The analytical
theory is the *best ranking metric* by Spearman magnitude, but it
selects for same-task pairs that merge well, not cross-task pairs that
merge poorly.

**Root cause**: Spectral overlap is confounded with task similarity.
Same-task adapters have high overlap and merge well (shared features
reinforce). Cross-task adapters have low overlap and merge poorly
(incompatible behavioral directions). The spectral inflation predicted
by the theory is real, but it is beneficial in the same-task case and
absent in the cross-task case.

**What this means for the pipeline**: Spectral metrics characterize
geometry; task relationship predicts quality. The merge pipeline should
use both, not substitute one for the other. The verdict tree's branch
structure is sound as a geometric classifier. The over-accumulation
diagnostic should remain informational/exploratory — it describes
spectral regime, not merge risk.

Phases 4-6 (TIES bounds, DARE bounds, strategy selection map) were
not pursued because the empirical cross-check showed that further
strategy-specific spectral bounds would answer a question already
demonstrated to be wrong for quality prediction. This is the correct
stopping decision.

Full analysis in `field_trials/analytical_spectral_geometry/study_memo.md`.

## Execution Sequence

### Phase 1 — Linear merge (complete)

- [x] Derive Gram matrix of linear merge in terms of SVDs and principal angles
- [x] Solve rank-1 case completely (closed-form singular values)
- [x] Solve equal-rank equal-spectrum case
- [x] Derive Weyl/Fan bounds for general case
- [x] Derive exact Frobenius norm of merge with cross-trace
- [x] Derive general rank-r Gram decomposition and cross-term bound
- [x] Derive stable rank bounds from Frobenius and σ_1 bounds
- [x] Implement `spectral_theory.py` (13 dataclasses, 11 functions)
- [x] Implement synthetic matrix generation utilities (`spectral_theory_test_utils.py`)
- [x] Run numerical validation on rank-1 and equal-rank cases
- [x] Run validation on general case (Weyl bounds: 100% containment, 24 settings)
- [x] Write `linear_merge_spectral_analysis.md`

### Phase 2 — Over-accumulation theory (complete)

- [x] Generalize rank-1 over-accumulation condition to rank-r
- [x] Express exact condition in SubspaceMetrics terms (including new v2 fields)
- [x] Compare to `over_accumulation.py` heuristic analytically
- [x] Identify three structural errors in heuristic (missing right-space angles,
      surrogate concentration metric, scalar reduction of full angle spectrum)
- [x] Run numerical comparison: analytical prediction vs heuristic score
- [x] Implement OA-v2 interaction-first estimator (parallel, non-authoritative)
- [x] Run higher-rank synthetic sweep (r ∈ {4,8,16,32})
- [x] Cross-check against 30 field trial merge outcomes
- [x] Run failure anatomy decomposition (task family, backbone, source gap, rank mismatch)
- [x] Write `over_accumulation_theory.md`

### Phase 3 — Norm-equalized analysis (complete, stub)

- [x] Derive spectral rescaling effect of norm equalization
- [x] Confirm over-accumulation reduction in ρ_F ≫ 1 regime analytically
- [x] Implement `norm_equalized_over_accumulation_analysis()` utility
- [ ] ~~Characterize distortion when stable ranks differ at equal Frobenius norm~~
      (not pursued; empirical cross-check showed spectral inflation is wrong target)
- [x] Write `norm_equalized_analysis.md` (stub with derivation targets)

### Phase 4 — TIES bounds (not pursued)

Paused after Phase 2 empirical cross-check showed spectral inflation
does not predict merge quality. Further element-wise spectral bounds
would answer a question already demonstrated to be wrong.

- [x] Write `ties_spectral_bounds.md` (scaffold only, documenting planned goals)

### Phase 5 — DARE bounds (not pursued)

Same rationale as Phase 4. Only the DARE Frobenius multiplier helper
(`expected_dare_frobenius_multiplier`) was implemented.

- [x] Write `dare_spectral_bounds.md` (scaffold only, documenting planned goals)

### Phase 6 — Strategy selection map (not pursued)

Cannot be built without Phases 4-5, and the engineering premise is
invalidated. The verdict tree's branch structure is sound as geometric
classification but its thresholds do not correspond to quality-prediction
boundaries.

- [x] Write `strategy_selection_map.md` (scaffold only, documenting planned comparison)

### Integration (complete — negative completion path)

- [x] Implement validated analytical predictions as utility functions
- [x] Add test suite (22 unit tests + 4 validation sweeps)
- [x] Cross-check predictions on 30 field trial adapter pairs
- [x] Write study memo documenting negative completion
- [x] Write `analytical_spectral_geometry_summary.md`
- [x] Update manifest to reflect negative completion status

**Actual timeline: ~1 day** (Phases 1-2 plus empirical cross-check;
Phases 4-6 correctly stopped after negative cross-check result).

## Estimated Resource Requirements

| Phase | CPU time | Disk | Dependencies |
|-------|----------|------|--------------|
| Mathematical derivation | 0 | 0 | Paper, pen, LaTeX |
| Synthetic validation | ~2-4 hours total | <1 GB | numpy, torch (CPU) |
| Field trial cross-check | ~10 minutes | Existing field trial data | gradience[dev] |

No GPU, no network access, no adapter downloads required.

## Guardrails

- Do not claim that analytical results on spectra imply downstream
  task performance predictions. Spectral preservation is a necessary
  condition for merge quality, not a sufficient one.
- Do not replace the heuristic in `over_accumulation.py` until
  analytical alternatives are validated both synthetically and on
  field trial data.
- Do not modify verdict thresholds in `verdicts.py` based on
  analytical results alone — empirical validation on real merges is
  required.
- Keep the analytical-vs-heuristic comparison honest: report cases
  where the heuristic outperforms the analytical prediction (possible
  if the heuristic implicitly captures structure the analysis misses).
- TIES and DARE bounds will be looser than linear merge results.
  Do not over-interpret loose bounds as strategy recommendations.
- All proofs should be self-contained and readable by someone with
  graduate-level linear algebra. Do not assume familiarity with
  Gradience internals in the mathematical write-ups.

## Connections to Other Research Lines

### Census study

The census study's ecological data provides a population-level
distribution of the spectral quantities (`Ω`, `δ`, `ρ_F`, `sr`) that
parameterize the analytical results. The census tells us which regions
of the parameter space are actually populated in practice. Analytical
results that cover well-populated regions are more valuable than
results that cover only extreme cases.

### GPU-return study

If and when GPU access returns, the decoder-only spectral
fingerprinting study will produce controlled adapter pairs with known
training conditions. These are ideal test cases for the analytical
predictions — the training conditions that produced the spectral
geometry are known, unlike in the census.

### Over-accumulation diagnostic

The analytical work directly addresses the CPU consolidation memo's
assessment that the over-accumulation diagnostic is "exploratory."
The outcome is nuanced: the theory provides a rigorous mathematical
foundation (the heuristic's structural errors are identified, the
exact cross-term condition is derived), but the empirical cross-check
shows that spectral inflation — correctly measured or not — is not
the failure mode in the tested population. The OA diagnostic's status
remains "exploratory/informational": it describes spectral regime
accurately, but spectral regime does not predict merge quality
without task relationship context.

## Bottom Line

This study derived what linear merge does to the singular value
spectrum as a function of the inter-adapter geometry already computed
by Gradience. The mathematical results are correct and numerically
validated. The empirical cross-check then revealed that spectral
inflation — the phenomenon the entire analytical line was designed to
characterize — is confounded with task similarity and does not predict
merge quality in isolation.

> **Spectral metrics characterize geometry; task relationship predicts
> quality. The merge pipeline should use both, not substitute one for
> the other.**

The study achieved negative completion: a clean result that closes the
analytical line with a clear finding and identifies the missing
ingredient (task relationship information, which the pipeline already
captures via `task_relationship_advisory`).

### Current status

**Study complete (negative completion).** Phases 1-2 produced correct
mathematical results implemented in `spectral_theory.py` and validated
on 6,912 synthetic cases. The 30-pair empirical cross-check showed all
spectral inflation metrics (heuristic and analytical) have wrong-sign
correlation with merge quality. Phases 4-6 were correctly not pursued.
Infrastructure changes (v2 geometry fields on `SubspaceMetrics`, OA-v2
parallel estimator) remain in the codebase as exploratory/parallel
tools. OA-v1 remains the authoritative production advisory path.

The key forward implication is that quality prediction requires
conditioning spectral diagnostics on task relationship — not replacing
spectral metrics, but contextualizing them. Within same-task pairs,
high overlap is beneficial. Across tasks, even low overlap can produce
harmful interference that no spectral metric can detect.
