# PEFT Generalization Audit -- Ring 1 Design Document

## 1. Purpose

Gradience was built on LoRA adapters. Every measurement, every pipeline
stage, and every test fixture assumes LoRA's (A, B) factorization as the
input substrate. This is fine as long as LoRA remains the dominant PEFT
method, but the PEFT ecosystem now includes LoHa, LoKr, IA3, AdaLoRA,
OFT, and others. If Gradience's spectral analysis is genuinely about
low-rank structure rather than about LoRA specifically, it should extend
to at least some of these methods with minimal changes.

The Ring 1 Generalization Audit tests this hypothesis. It asks: can a
single non-LoRA adapter class pass through the full Gradience workflow
(audit, pairwise comparison, inventory triage) using only thin extraction
shims, without modifying the core measurement layer?

Ring 1 is deliberately narrow. It targets one additional PEFT class (LoHa),
uses small encoder models on CPU, and limits scope to classification tasks.
The goal is not production support for LoHa. The goal is to learn whether
the measurement substrate is actually generic or whether it encodes
LoRA-specific assumptions that would require deeper refactoring.


## 2. Current Substrate Analysis

### 2.1 Generic Components

The following components operate on singular values, singular vectors, or
dense matrices. They have no knowledge of where those values came from.

**`low_rank_singular_values(A, B)`** (`vnext/audit/lora_audit.py`).
Computes SVD of B @ A. Despite the parameter names, there is nothing
LoRA-specific about the computation. Any two matrices whose product is the
quantity of interest will work.

**`_energy_rank(singular_values, threshold)`** (`vnext/audit/lora_audit.py`).
Returns the number of singular values needed to capture a given fraction of
total spectral energy. Operates on a 1-D array. Fully generic.

**`compute_subspace_metrics(U1, U2, S1, S2)`** (`vnext/merge/spectral_compat.py`).
Computes subspace overlap, principal angle statistics, and norm ratio between
two sets of singular vectors and values. No LoRA-specific logic.

**Update norm computation.** The Frobenius norm of B @ A (or any dense
delta) is computed via standard linear algebra. Generic.

**QA eligibility logic** (`vnext/audit/qa_artifact.py`). The
`AdapterQAArtifact` and `EligibilityStatus` machinery operates on measured
quantities (effective rank, energy rank, update norm) and user-reported
behavioral evidence. It does not know how those quantities were produced.

**Inventory summary and action plan** (`vnext/inventory/summary.py`).
Aggregates QA artifacts and merge reports. Operates entirely on the
structured output of earlier pipeline stages. Generic by construction.

### 2.2 LoRA-Specific Components

The following components contain explicit LoRA assumptions that would need
shims or modifications to support other PEFT classes.

**`_iter_lora_pairs(state_dict)`** (`vnext/audit/lora_audit.py`).
The primary extraction function. It scans a state dict for keys matching
the pattern `*.lora_A.*` / `*.lora_B.*` and yields `(layer_name, A, B)`
triples. This is the single most LoRA-specific function in the codebase.
Everything downstream depends on the triples it produces.

**LoRA config parsing.** The audit reads `adapter_config.json` to extract
`r` (nominal rank) and `lora_alpha` (scaling factor). These field names
are LoRA-specific. LoHa uses the same field names (`r`, `alpha`), but
other methods may not.

**Scaling factor: `alpha / r`.** LoRA applies `(alpha / r) * B @ A` as
the effective update. The audit uses this to compute the scaled update
norm. LoHa uses the same scaling convention. IA3 and OFT do not.

**`LoRALayerAudit` field names.** The per-layer audit result dataclass
uses field names like `rank_nominal` that reference LoRA concepts. These
are cosmetic but would be confusing in a multi-method context.

**`rank_nominal` in QA artifact.** The `AdapterQAArtifact` includes
`rank_nominal` as a top-level field. For LoRA and LoHa this is `r`. For
methods without a rank parameter, this field has no natural value.

### 2.3 Summary

The measurement layer (SVD, energy rank, subspace metrics) is generic. The
extraction layer (finding and pairing weight tensors from a state dict) is
LoRA-specific. The boundary between them is clean: `_iter_lora_pairs()` is
the only function that bridges extraction and measurement. A shim that
produces the same `(layer_name, A, B)` triples from a different PEFT
format would make the entire downstream pipeline available without changes.


## 3. Candidate Assessment

### 3.1 Why LoHa is the Best Ring 1 Candidate

LoHa (Low-Rank Hadamard Adaptation) stores two pairs of low-rank factors
per layer. The effective weight update is:

    delta_W = (alpha / r) * (w1_a @ w1_b) * (w2_a @ w2_b)

where `*` denotes element-wise (Hadamard) product and each `(w_a, w_b)`
pair has the same shape constraints as a LoRA `(A, B)` pair.

Properties that make LoHa suitable for Ring 1:

1. **Has `r` and `alpha`.** LoHa adapters carry the same rank and scaling
   parameters as LoRA. The QA artifact's `rank_nominal` field applies
   directly.

2. **Factor pairs are (A, B)-shaped.** Each of the two factor pairs can
   be fed into `low_rank_singular_values()` without modification. This
   gives a "factor-level" audit mode for free.

3. **Materialization is cheap.** Reconstructing the full delta (Hadamard
   product of the two reconstructed factor matrices) produces a dense
   matrix that can be SVD'd for a "materialized" audit. For the small
   encoder models in Ring 1 (distilbert, bert-base), this is
   computationally trivial on CPU.

4. **Two audit modes provide cross-validation.** Factor-level and
   materialized modes should produce related but different measurements.
   Comparing them tests whether the measurement layer captures meaningful
   structure or is sensitive to representation choices.

5. **PEFT support is mature.** The `peft` library has stable LoHa support.
   Adapters can be trained and saved with standard PEFT APIs.

6. **Subspace comparison transfers.** If two LoHa adapters are materialized
   into dense deltas, `compute_subspace_metrics()` applies directly. At the
   factor level, comparing corresponding factor pairs (w1 of adapter A vs
   w1 of adapter B) tests whether factor-level subspace overlap is
   informative.

### 3.2 Why LoKr is Deferred to Ring 2

LoKr uses Kronecker products, which mix spatial dimensions in ways that
complicate interpretation. The factored variant has an inner (A, B) pair
that is directly usable, but the outer Kronecker structure has no low-rank
analog. Materialization produces matrices much larger than parameter count.
Pairwise comparison is feasible but the geometric interpretation is less
clear than for LoHa.

LoKr is a reasonable Ring 2 target if Ring 1 succeeds. It tests a harder
case (structural mismatch with the SVD pipeline) while still having some
low-rank components.

### 3.3 Why IA3 is Deferred Indefinitely

IA3 learns diagonal scaling vectors, not matrix factorizations. There is no
rank parameter, no subspace to orient, and no spectral energy distribution
to analyze. SVD of a diagonal matrix returns the sorted absolute values of
the diagonal entries -- technically valid but carrying none of the geometric
information that Gradience's measurements are designed to extract.

Pairwise comparison of IA3 adapters reduces to element-wise vector
comparison (cosine similarity, L2 distance), which is a fundamentally
different measurement domain. Supporting IA3 would require new measurement
concepts, not shims.

IA3 is out of scope for the generalization program unless the project's
measurement philosophy changes.


## 4. Technical Approach

### 4.1 The LoHa Shim

The core technical deliverable is a shim function that reads a LoHa
adapter's state dict and produces output compatible with the existing
pipeline. The shim operates in two modes.

**Factor-level mode.** For each target layer, extract the two factor pairs
`(w1_a, w1_b)` and `(w2_a, w2_b)`. Yield each pair as a separate
`(layer_name, A, B)` triple, with the layer name suffixed to distinguish
the two factors (e.g., `model.encoder.layer.0.attention.self.query/factor1`
and `.../factor2`). This feeds directly into `low_rank_singular_values()`
and the rest of the per-layer audit pipeline.

Advantages: no materialization, preserves the low-rank structure, doubles
the number of per-layer measurements (which may be informative or noisy).

**Materialized mode.** For each target layer, reconstruct the full delta:

    delta = (alpha / r) * (w1_a @ w1_b) * (w2_a @ w2_b)

Compute the SVD of this dense matrix and yield the result as a single
per-layer measurement. This mode produces one measurement per layer, like
LoRA, but loses the factor-level structure.

Advantages: directly comparable to LoRA audit results, captures the
combined effect of both factor pairs, standard SVD interpretation applies.

### 4.2 Output Format

The shim writes its output as a LoRA-format adapter directory: an
`adapter_config.json` with `peft_type: "LORA"` and a `adapter_model.safetensors`
containing `lora_A` / `lora_B` weight keys. This means the existing audit
pipeline reads the output without any code changes.

For factor-level mode, the synthetic LoRA directory contains twice as many
layers as the original LoHa adapter (two factor pairs per original layer).

For materialized mode, the shim performs an SVD of the dense delta and
stores the top-r left and right singular vectors (scaled by singular values)
as synthetic A and B matrices. The synthetic LoRA has the same layer count
as the original.

### 4.3 Integration Points

The shim is a standalone script, not a modification to Gradience's core.
The workflow is:

    1. Train LoHa adapter (PEFT, standard training script)
    2. Run shim: loha_adapter_dir -> synthetic_lora_dir
    3. Run gradience audit on synthetic_lora_dir
    4. Run gradience merge-audit on two synthetic_lora_dirs
    5. Run gradience summarize-inventory on the QA and merge outputs

Steps 3-5 use existing CLI commands without modification. The shim is the
only new code.

### 4.4 Key Extraction Details

PEFT stores LoHa weights under keys following this pattern:

    base_model.model.{module_path}.hada_w1_a
    base_model.model.{module_path}.hada_w1_b
    base_model.model.{module_path}.hada_w2_a
    base_model.model.{module_path}.hada_w2_b

The shim must:

- Parse these keys to identify layer names and factor pairs.
- Handle the `base_model.model.` prefix that PEFT prepends.
- Read `r` and `alpha` from `adapter_config.json`.
- Apply the correct scaling convention (alpha/r, matching LoRA).
- Handle potential shape transpositions (PEFT may store factors in
  either orientation depending on the target module).


## 5. What This Program Will Test

### 5.1 Measurement Layer

- Do `low_rank_singular_values()`, `_energy_rank()`, and effective rank
  produce meaningful results on LoHa factor pairs?
- Does the materialized delta's SVD spectrum show the same qualitative
  patterns as LoRA (energy concentration, rank-energy relationship)?
- Do factor-level and materialized measurements agree in direction (if
  factor-level says "low effective rank," does materialized agree)?

### 5.2 Pairwise Comparison

- Does `compute_subspace_metrics()` produce interpretable overlap and
  angle statistics when comparing two LoHa adapters (materialized mode)?
- Does the merge verdict logic (SAFE/CONFLICTING/REDUNDANT/IMBALANCED)
  produce sensible classifications for LoHa pairs?
- Do factor-level pairwise comparisons (w1 of adapter A vs w1 of adapter B)
  carry information, or is materialized comparison the only useful mode?

### 5.3 Inventory Triage

- Do QA eligibility classifications make sense for LoHa adapters?
- Does the inventory action plan route LoHa pairs to reasonable buckets
  (same-task priority, cross-task caution, evaluate-first)?
- Do near-miss severity classifications behave sensibly?

### 5.4 Shim Fidelity

- Does the synthetic LoRA directory pass all Gradience validation checks?
- Are there hidden assumptions in `adapter_config.json` parsing that break
  on synthetic configs?
- Does round-tripping (LoHa -> shim -> audit -> QA artifact) preserve
  information, or does the translation introduce artifacts?


## 6. Success Criteria

The Ring 1 audit succeeds if:

1. **At least one non-LoRA adapter class completes the full Gradience
   workflow** (audit -> pairwise merge-audit -> inventory triage) using
   only thin extraction shims and no modifications to core measurement
   or pipeline code.

2. **Measurements are qualitatively sensible.** LoHa adapters trained on
   the same task with different hyperparameters should show similar spectral
   profiles. Adapters trained on different tasks should show measurably
   different profiles. This does not require quantitative benchmarks, only
   directional correctness.

3. **Pairwise comparisons produce actionable verdicts.** The merge verdict
   logic should classify LoHa adapter pairs into the same categories it
   uses for LoRA (SAFE, CONFLICTING, REDUNDANT, IMBALANCED) with results
   that pass a sanity check against known adapter relationships.

4. **The shim boundary is clean.** All LoHa-specific code lives in the
   shim script. No changes to files under `gradience/` are required. This
   demonstrates that the extraction/measurement boundary identified in
   Section 2 is real, not theoretical.

The Ring 1 audit produces a negative result (still valuable) if:

- The measurement layer produces degenerate or uninterpretable results on
  LoHa factors, suggesting that the SVD pipeline encodes LoRA-specific
  structure despite appearing generic.
- The pairwise comparison produces random or uniform verdicts, suggesting
  that subspace overlap is not meaningful for Hadamard-product adapters.
- The shim requires modifications to core code, revealing hidden coupling
  between extraction and measurement.


## 7. Limitations

### 7.1 Scope Constraints

- **CPU only.** All experiments run on CPU with small models (distilbert,
  bert-base). GPU-scale behavior is not tested.
- **Small models only.** Encoder models with < 200M parameters.
  Decoder models (LLaMA, Mistral) are not in scope for Ring 1.
- **Classification only.** SST-2 and similar binary/multi-class
  classification tasks. Generative tasks are not tested.
- **No core refactor.** Ring 1 deliberately avoids modifying any code
  under `gradience/`. If the shim approach fails, Ring 2 would need to
  consider core changes, but Ring 1's purpose is to test the boundary
  as it exists today.

### 7.2 What Ring 1 Does Not Prove

- That Gradience measurements are *optimal* for LoHa. The measurements
  may be valid but less informative than LoHa-specific alternatives.
- That the shim approach scales to many PEFT methods. LoHa is the
  easiest case. LoKr and other methods may require deeper integration.
- That materialized-mode measurements are equivalent to factor-level
  measurements. They test different things and may disagree.
- That pairwise merge recommendations are *correct* for LoHa. The
  verdicts may be sensible without being optimal. Validating merge
  quality would require downstream evaluation, which is out of scope.

### 7.3 Relationship to Future Work

If Ring 1 succeeds, the natural next steps are:

- **Ring 2 (LoKr).** Test a harder case where Kronecker structure
  creates a larger gap between the learned representation and the
  SVD pipeline's expectations.
- **Extraction interface refactor.** Replace `_iter_lora_pairs()` with
  a pluggable extraction interface that dispatches on `peft_type`. This
  would eliminate the shim step and allow native support for multiple
  PEFT methods.
- **Config generalization.** Abstract `rank_nominal`, `lora_alpha`, and
  related fields in QA artifacts to method-neutral equivalents.

If Ring 1 produces a negative result, the implication is that Gradience's
measurement layer, while technically generic in its linear algebra, depends
on LoRA's specific factorization structure to produce meaningful results.
This would redirect future work toward method-specific measurement
pipelines rather than a unified substrate.
