# Ring 2 Stage A — Checkpoint Delta Representation Selection

Generated on March 30, 2026 from CPU-only Stage A run artifacts in:
- `experiments/ring2_checkpoint_delta/stage_a_representation_results.json`
- `experiments/ring2_checkpoint_delta/stage_a_representation_results.md`

## Stage A Question

What is the right CPU-feasible representation of a full fine-tune checkpoint delta for Gradience-style audit and comparison?

## Panel Used

Backbone:
- `distilbert-base-uncased`

Checkpoint panel (4 total):
- `sst2_s42` (SST-2, seed 42)
- `sst2_s123` (SST-2, seed 123)
- `mrpc_s42` (MRPC, seed 42)
- `qnli_s42` (QNLI, seed 42)

This panel includes one same-task pair (`sst2_s42` vs `sst2_s123`) plus cross-task pairs.

## Fixed Parameter Selection Rule

Stage A fixed the selected tensor classes to matrix weights matching:
- attention projections: `q_lin`, `k_lin`, `v_lin`, `out_lin`
- FFN dense layers: `lin1`, `lin2`
- classifier head: `pre_classifier`, `classifier`

Excluded by rule:
- embeddings
- layer norms
- biases
- non-matrix tensors

Selected matrix count per checkpoint in this run: 38.

## Representation Comparison (A/B/C)

## A: Raw Delta Matrices (Baseline)

Observed:
- Highest fidelity (exact delta, no approximation).
- Largest artifact footprint: ~657 MB across panel.
- Weak operational bridge to factor-based Gradience pairwise logic.

Stage A disposition:
- `stage_b_readiness`: `medium`
- `recommendation`: `hold_as_fallback`

## B: Truncated Low-Rank Delta Approximation (k=4/8/16)

Observed:
- Strong bridge to current substrate (synthetic LoRA-like factors).
- CPU/storage footprint much lower than raw.
- In this panel, approximation fidelity was low at tested ranks:
  - mean retention: k=4 `0.4863`, k=8 `0.5577`, k=16 `0.6380`
- Reconstruction error remained high at tested ranks.

Interpretation:
- At small ranks, the deltas appear insufficiently low-rank for stable Stage B use without expanding rank budget or redesigning representation assumptions.

Stage A disposition:
- `stage_b_readiness`: `low`
- `recommendation`: `reject`

## C: Layerwise Summary Representation

Observed:
- Best CPU/storage profile: ~35.6 KB across panel.
- Stable, interpretable layer metrics (effective rank, stable rank, energy concentrations, decay summaries).
- Directly usable for single-artifact audit-style diagnostics.
- Weaker direct reuse of existing factor-level pairwise logic than B, but materially more practical under current fidelity constraints.

Stage A disposition:
- `stage_b_readiness`: `high`
- `recommendation`: `advance`

## Stage A Decision

Selected outcome:
- **Outcome 2 — Representation C selected**

Stage B representation:
- **Layerwise summary representation (C)**

Why:
- Representation B did not retain enough delta energy at tested CPU-feasible ranks (`k=4,8,16`) to be considered stable/faithful for immediate advancement.
- Representation C preserved interpretable, compact structural signals while remaining fully CPU-native.
- Representation A remains a useful baseline/reference but is too heavy as a primary Stage B object.

## What Worked / What Did Not

Worked:
- End-to-end CPU extraction and representation generation on a real checkpoint panel.
- Clean fixed tensor-selection rule.
- Structured outputs across A/B/C with compute and size accounting.

Did not work:
- Low-rank truncation at `k <= 16` did not provide sufficient fidelity in this panel.

## Guardrails and Limits

This Stage A decision is scoped to:
- one backbone (`distilbert-base-uncased`)
- one small classification panel
- CPU-only operation
- tested low-rank set (`k=4,8,16`)

No claim is made yet for decoder models, generative tasks, or broader architecture support.

## Stage B Starting Point

Proceed to Stage B with Representation C as the primary checkpoint-delta object, while retaining:
- Representation A as a faithfulness baseline for spot checks.
- Representation B as a conditional future path if rank budget or approximation method changes materially.

Method note:
- Stage A extraction was regenerated with seed-aware base head initialization for sequence-classification checkpoints so classifier deltas reflect training updates rather than random re-initialization drift.
