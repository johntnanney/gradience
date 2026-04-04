# PUBLIC_ECOSYSTEM_SPECTRAL_CENSUS_STUDY_SPEC
## Repo-Facing CPU-Only Execution Plan

## Purpose

Define a CPU-only proving-ground study that addresses the decoder-only
architecture-vs-task question without training compute:

> **Public Ecosystem Spectral Census**

Central question:

> **Do spectral fingerprints of publicly available decoder-only LoRA adapters
> separate architecture effects from task effects at population scale,
> using found artifacts rather than controlled training?**

This study runs entirely on CPU and reuses existing discovery/audit
infrastructure and field-trial manifest conventions.

## Relationship to GPU-Return Decoder Study

This census is the naturalistic complement to controlled decoder GPU-return
fingerprinting:

- controlled GPU study: matched training confounds, causal architecture-vs-task claims
- ecosystem census: found artifacts, larger n, ecological and confound-aware claims

Neither replaces the other.

## Why this study now

- GPU remains unavailable.
- Spectral LoRA audit is CPU-cheap once adapter weights are downloaded.
- HuggingFace Hub has broad decoder adapter availability.
- Existing code already covers most of the pipeline:
  - `scripts/broader_benchmarks.py`
  - `gradience.vnext.audit.lora_audit.audit_lora_peft_dir()`

## Scope

### In scope

- public decoder-only LoRA adapters from Hub
- architecture-family and task-category labeling from metadata
- CPU spectral audit and fingerprint extraction
- architecture-vs-task decomposition with explicit confound assessment

### Out of scope

- training any new adapters
- downstream task evaluation or behavioral benchmarking
- causal claims from observational data
- product policy changes from first pass

## Program questions

1. Architecture clustering at population scale
2. Task clustering after architecture conditioning
3. Confound contribution (rank/alpha/target-modules/popularity)
4. Module-type asymmetry replication (attn vs mlp)
5. Ecological baseline distributions for decoder adapter spectra

## Cohort design

### Minimum architecture families

- Llama
- Mistral
- Qwen

Optional: Phi, Gemma.

### Task categories

Derived from existing task heuristics and tags:

- Chat / Instruct
- Code
- Math / Reasoning
- Domain specialist (medical/legal/data)
- Classification
- General / Unknown

### Size targets

- Pilot: 30–50
- Core: 100–150
- Extended: 200+

## Inclusion / exclusion

### Include

- `peft_type == "LORA"`
- decoder-only base-model family
- downloadable adapter config + weights
- adapter size < 500MB
- at least one extractable LoRA layer

### Exclude

- encoder or encoder-decoder bases
- zero-layer extract failures
- structural audit failures (log with reason)
- duplicate weight copies

## Spectral fingerprint vector (v1)

Fixed 10D vector:

1. `stable_rank_mean`
2. `stable_rank_std`
3. `utilization_mean`
4. `energy_rank_90_p50`
5. `entropy_erank_mean`
6. `attn_stable_rank_mean`
7. `mlp_stable_rank_mean`
8. `attn_utilization_mean`
9. `mlp_utilization_mean`
10. `edge_gap_mean`

Secondary probes are computed but not promoted.

## Analysis phases

1. Distributional characterization
2. Confound assessment
3. Architecture-vs-task decomposition
4. Clustering / nearest-neighbor purity checks
5. Module-type replication tests

## Pilot gate (must check before core expansion)

1. ≥80% audit success on attempted adapters
2. Metric sanity (finite variance, plausible ranges)
3. Architecture coverage: at least 2 families with ≥8 each
4. Task coverage: at least 3 categories with ≥5 each
5. Non-degenerate visible signal in at least one core metric

## Infrastructure entrypoint

Script:

- `scripts/ecosystem_census.py`

Expected behavior:

- idempotent cache-aware download/audit
- adapter metadata + exclusion reasons persisted
- fingerprint table and pilot report generation

## Deliverables

Directory:

- `field_trials/public_ecosystem_census/`

Primary files:

- `manifest.json`
- `adapter_records.json`
- `fingerprint_table.json`
- `fingerprint_table.md`
- `confound_assessment.json`
- `architecture_task_decomposition.json`
- `architecture_task_decomposition.md`
- `clustering_results.json`
- `module_type_replication.json`
- `pilot_gate_report.md`
- `study_memo.md`
- `excluded_adapters.json`

## Success / partial / negative completion

### Success

Architecture-family signal survives confound controls in multiple metrics with
meaningful effect size, and module-type asymmetry replicates.

### Partial

Signal appears in subsets/metrics but is weak or confounded; supports bounded
extension language only.

### Negative

No stable architecture/task signal survives confound controls; controlled GPU
study remains strictly necessary for resolution.

All outcomes remain useful.

## Guardrails

- no causal claims from found-artifact census
- report confounds before interpretation
- keep secondary probes secondary
- do not convert first-pass census findings into product policy

## Bottom line

Run a CPU-only decoder adapter spectral census now as ecological evidence,
while keeping controlled decoder GPU study as the causal adjudication path.
