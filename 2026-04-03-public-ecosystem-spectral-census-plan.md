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

This study runs entirely on CPU. It reuses existing infrastructure
(`scripts/broader_benchmarks.py`, `gradience audit`, field trial
manifest conventions) and extends them to a systematic census of the
public HuggingFace Hub adapter ecosystem.

## Relationship to the GPU-Return Study

The decoder-only spectral fingerprinting GPU-return plan
([`2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`](2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md))
defines a controlled experiment with matched hyperparameters across
architectures and tasks. That study remains the gold standard for
causal claims about architecture-vs-task structure.

This census is the **naturalistic complement**:

- **GPU study**: controlled training, small n, matched confounds, causal claims.
- **Census study**: found artifacts, large n, uncontrolled confounds, ecological claims.

The two studies answer different questions. The census asks whether
spectral structure is visible *despite* uncontrolled variation. The GPU
study asks whether it holds *because of* controlled architecture/task
differences. Either can run first. If the census finds clear clustering,
the GPU study's role shifts from "does the signal exist?" to "is it
causal?" If the census finds no clustering, the GPU study's role becomes
more critical — the signal may exist but be obscured by confounds at
population scale.

Neither study's outcome preempts the other.

## Why This Study Now

GPU access is unavailable for the foreseeable future. The decoder-only
question is the highest-priority open item from CPU consolidation. The
key insight is that spectral audit of LoRA adapters requires only the
adapter weight matrices (B and A factors), not the base model or GPU
compute. SVD of low-rank adapter products is a small CPU operation even
for 7B-class models. HuggingFace Hub hosts thousands of published
adapters with downloadable `adapter_model.safetensors` and
`adapter_config.json` files.

Existing infrastructure covers most of the pipeline:

- `scripts/broader_benchmarks.py` (Study 14): discovery, download, audit, analysis
- `.cache/study17_adapters/`: prior Hub adapter cache with working download pattern
- `gradience.vnext.audit.lora_audit.audit_lora_peft_dir()`: CPU spectral audit with `compute_udr=False`
- `field_trials/` manifest and artifact conventions: proven organizational structure

## Scope

### In scope

- decoder-only LoRA adapters from HuggingFace Hub (public, downloadable)
- architecture-family and task-category labeling from adapter metadata
- CPU spectral audit using existing `audit_lora_peft_dir()` pipeline
- architecture-vs-task variance decomposition on spectral fingerprint metrics
- population-level distributional characterization of decoder adapter spectra

### Out of scope

- training new adapters (requires GPU)
- hyperparameter matching or training-condition control (not available from Hub metadata)
- causal claims about architecture effects (requires controlled study)
- evaluation or behavioral validation of adapters (no downstream task running)
- policy changes or product decisions from first pass
- encoder adapters (already covered by existing field trials)

## Program Questions

1. **Architecture clustering:**
   Do spectral profiles cluster by base model family when examined across
   a diverse task population?

2. **Task clustering:**
   Conditioning on architecture, do task categories produce stable
   spectral offsets (e.g., math adapters vs. chat adapters on the same
   base model)?

3. **Confound assessment:**
   How much of the observed spectral variation is attributable to
   nominal rank, training duration, or other adapter-config metadata
   vs. architecture-or-task signal?

4. **Module-type patterns:**
   Does the encoder-era finding (attention layers more spectrally
   concentrated than MLP layers; `v_proj` more universal than `q_proj`)
   replicate on decoder architectures at population scale?

5. **Ecological baselines:**
   What is the empirical distribution of stable rank, utilization, and
   energy concentration across the public decoder adapter ecosystem?

## Hypotheses

Pre-pilot hypotheses (retained for continuity; see pilot revision below):

- H1: Architecture-family is the strongest first-order predictor of
  spectral profile shape (stable rank distribution, energy concentration
  pattern).

- H2: Task category is a detectable second-order signal after
  conditioning on architecture, but weaker and noisier than in
  controlled settings.

- H3: Nominal rank and adapter configuration metadata (alpha, target
  modules) explain a substantial portion of variance and must be
  controlled for before architecture/task effects are interpretable.

- H4: Module-type spectral asymmetry (attention vs. MLP) replicates on
  decoders.

- H5: The outcome will be bounded — confounds will limit claim strength
  — and that is expected and acceptable for a census study.

### Pilot-revised hypothesis status

The pilot (n=26, Llama + Mistral) produced evidence against H1 and H4
and in favor of a richer decomposition than the original hypotheses
assumed. Updated status:

- **H1: weakened.** Task η² (0.26 mean) exceeded architecture η² (0.12
  mean). Architecture is *not* the dominant first-order predictor of
  spectral shape. However, architecture kNN purity is very high (0.90),
  indicating tight local clustering despite low global variance
  explained. Revised interpretation: architecture may determine the
  *precision* of the spectral fingerprint (tight clusters) while task
  determines its *location* (wider separation). This is a richer
  decomposition than simple dominance ordering.

- **H2: strengthened, but inverted.** Task is the stronger global
  signal, not the weaker one. The pilot suggests task category drives
  more spectral variation than architecture family in found artifacts.
  Whether this holds after confound residualization is the key
  pilot-plus question.

- **H3: confirmed.** Nominal rank R² = 0.66. Confound control is
  essential before any architecture/task interpretation. Rank-matched
  subset analysis is required, not optional.

- **H4: contradicted.** Attention < MLP utilization holds in only 25%
  of decoder adapters. The encoder-era module-type asymmetry does not
  replicate. This is a first-class finding, not a side result. Likely
  explanation: grouped query attention and SwiGLU MLP structures in
  decoder architectures alter the parameter geometry relative to
  encoder attention/FFN blocks.

- **H5: confirmed.** The outcome is bounded, and the confound structure
  is exactly what a census study needed to reveal.

## Cohort Design

### Architecture families

Minimum three families (matching GPU-return plan alignment):

| Family | Representative base models | Role |
|--------|---------------------------|------|
| Llama | `meta-llama/Llama-2-7b-hf`, `Meta-Llama-3-8B`, `Meta-Llama-3.1-8B` | Baseline anchor |
| Mistral | `mistralai/Mistral-7B-v0.1`, `Mistral-7B-Instruct-v0.1` | Attention/norm variation |
| Qwen | `Qwen/Qwen2-7B` | Tokenization/implementation diversity |

Optional expansion families (if cohort size permits):

| Family | Representative base models | Role |
|--------|---------------------------|------|
| Phi | `microsoft/phi-2` | Smaller architecture anchor |
| Gemma | `google/gemma-7b`, `google/gemma-2b` | Alternative design point |

### Task categories

Use the existing `_infer_task()` heuristic from `scripts/broader_benchmarks.py`
as the starting taxonomy, refined into census-level groupings:

| Category | Inferred-task values | Census role |
|----------|---------------------|-------------|
| Chat / Instruct | `chat`, `text-generation` | Broad adaptation signal |
| Code | `code` | Narrow domain signal |
| Math / Reasoning | `math` | Narrow domain signal |
| Domain specialist | `medical`, `legal`, `data` | Domain adaptation signal |
| Classification | `classification` | Supervised task signal |
| General / Unknown | `general` | Confound baseline |

### Cohort size targets

| Tier | Adapters | Architectures | Purpose |
|------|----------|---------------|---------|
| Pilot | 30-50 | 2-3 | Protocol validation, pipeline shakeout |
| Core | 100-150 | 3+ | Primary analysis cohort |
| Extended | 200+ | 4-5 | Power for interaction effects |

Start at pilot tier. Promote to core only after pilot validation passes.

### Inclusion criteria

- `peft_type == "LORA"` in `adapter_config.json`
- Decoder-only base model architecture
- Adapter weights downloadable as `adapter_model.safetensors` or `adapter_model.bin`
- Adapter file size < 500 MB
- At least 1 LoRA layer extractable by `audit_lora_peft_dir()`

### Exclusion criteria

- Encoder-only or encoder-decoder base models
- Adapters with no extractable LoRA layers (audit returns `n_layers == 0`)
- Adapters where audit fails with structural errors (logged, not silently dropped)
- Duplicate adapters (same weights under different repo IDs)

### Metadata to record per adapter

From `adapter_config.json`:
- `peft_type`, `r`, `lora_alpha`, `lora_dropout`
- `target_modules` (list)
- `base_model_name_or_path`
- `task_type`

From Hub API:
- `repo_id`, `downloads`, `tags`
- Inferred task category
- Inferred architecture family

From audit:
- `n_layers`, `total_lora_params`
- Download size (MB)
- Audit duration (seconds)
- Audit issues (if any)

## Spectral Fingerprint Metrics

### Core metrics (per adapter, per layer)

All currently produced by `audit_lora_peft_dir()`:

- `stable_rank`: effective dimensionality (Frobenius/spectral norm ratio)
- `utilization`: stable_rank / nominal_rank
- `energy_rank_90`: minimal rank capturing 90% Frobenius energy
- `entropy_effective_rank`: exponential of normalized spectral entropy
- Top singular values (first 4, for spectral shape characterization)

### Aggregate metrics (per adapter)

- `stable_rank_mean`, `stable_rank_std` across layers
- `utilization_mean`, `utilization_std` across layers
- `energy_rank_90_p50` (median across layers)
- Module-type breakdown: mean metrics for attention layers vs. MLP layers

### Secondary probes (compute but do not promote)

- `edge_gap`: σ₁/σ₂ ratio per layer (spectral concentration)
- `tail_energy_fraction`: 1 - (energy in top-k) for k = energy_rank_90 (tail weight)
- Per-policy rank suggestions: `energy@0.90`, `knee`, `erank`

### Fingerprint vector definition

For clustering and variance decomposition, define the per-adapter
fingerprint as a fixed-length vector:

```
fingerprint = [
    stable_rank_mean,
    stable_rank_std,
    utilization_mean,
    energy_rank_90_p50,
    entropy_erank_mean,
    attn_stable_rank_mean,    # attention layers only
    mlp_stable_rank_mean,     # MLP layers only
    attn_utilization_mean,
    mlp_utilization_mean,
    edge_gap_mean,
]
```

This is a 10-dimensional vector. Keep it fixed for first pass.

## Analysis Plan

### Phase 1: Distributional characterization

- Empirical distributions of each core metric across the full cohort
- Histograms by architecture family and task category
- Summary statistics table (mean, std, median, IQR by architecture and task)

### Phase 2: Confound assessment

Before testing architecture/task effects, quantify confound contribution:

- Regress core metrics on `nominal_rank`, `lora_alpha`, `len(target_modules)`,
  `download_count` (popularity proxy)
- Report R² for confound-only model
- If R² > 0.3, residualize metrics before architecture/task analysis

### Phase 2b: Analytic control subsets (added post-pilot)

The pilot confound R² (0.66 for nominal rank) demands that architecture
and task effects be tested not only via residualization but also on
subsets where the confound is eliminated by construction:

- **Rank-matched subset**: restrict to the most common nominal rank in
  the cohort (likely r=16 or r=8). Architecture/task effects that
  survive in a rank-matched subset are harder to dismiss than effects
  that survive only after statistical residualization. Lead with this
  result in all summaries.

- **Task-label confidence subset**: assign each adapter a label
  confidence tier — high (explicit dataset name in repo ID or tags),
  medium (tag-inferred), low (fallback heuristic / "general"). Run
  task analysis on the high-confidence subset only. Report whether
  task η² strengthens or collapses when noisy labels are excluded.

- **Variant-aware architecture robustness checks**: within each
  architecture family, test whether model variants (e.g., Llama-2-7b
  vs. Llama-3-8B, Mistral-v0.1 vs. Instruct-v0.1) produce internal
  spectral differences comparable to between-family differences. If
  within-family variant spread approaches between-family spread, the
  "architecture" label is too coarse.

### Phase 3: Architecture-vs-task decomposition

- Two-way ANOVA (or Kruskal-Wallis if normality fails) on core metrics:
  architecture family × task category
- Report effect sizes (η²) for architecture, task, and interaction terms
- Report both raw and residualized/rank-matched results side by side
- Visualization: fingerprint-space scatter colored by architecture, shaped by task

### Phase 4: Clustering validation

- k-nearest-neighbor purity: for each adapter, what fraction of its
  k=5 nearest neighbors (in fingerprint space) share the same
  architecture family? Same task category?
- Simple clustering (k-means or hierarchical) on fingerprint vectors:
  do emergent clusters align with architecture, task, or neither?

### Phase 5: Module-type asymmetry analysis (first-class output)

The pilot found that the encoder-era module-type asymmetry (attention
layers more spectrally concentrated than MLP layers) does **not**
replicate on decoder architectures (25% of adapters, vs. consistent
majority on encoders). This is a substantive divergence, not a null
result, and is treated as a first-class deliverable.

- Paired comparison: attention-layer mean stable rank vs. MLP-layer
  mean stable rank, within each adapter
- Sign test or Wilcoxon across the cohort
- Breakdown by architecture family: is the non-replication universal
  across decoder families, or family-specific?
- Breakdown by module subtype where possible: `q_proj` vs. `k_proj` vs.
  `v_proj` vs. `o_proj` (attention); `gate_proj` vs. `up_proj` vs.
  `down_proj` (MLP). Report whether the asymmetry appears at the
  subtype level even if it is absent at the aggregate attention-vs-MLP
  level.
- Architectural attribution: relate findings to known structural
  differences between decoder and encoder blocks (grouped query
  attention, SwiGLU activation, RMSNorm placement) that could explain
  the changed spectral geometry.

## Pilot Validation Gate

The pilot tier (30-50 adapters) must pass these criteria before
proceeding to core cohort:

1. **Pipeline viability**: ≥80% of attempted adapters successfully
   audit (download + spectral extraction completes without error).

2. **Metric sanity**: core metric distributions have finite variance and
   plausible ranges (stable rank > 1.0, utilization ∈ (0, 1]).

3. **Architecture coverage**: at least 2 architecture families with ≥8
   adapters each.

4. **Task coverage**: at least 3 task categories with ≥5 adapters each.

5. **Visible signal**: at least one core metric shows a visually
   detectable difference between architecture families in pilot
   distributions (does not need to be significant — just non-degenerate).

If criteria 1-4 pass but criterion 5 fails, document and proceed to
core cohort (larger n may reveal what pilot cannot). If criteria 1-3
fail, diagnose pipeline issues before expanding.

### Pilot outcome (2026-04-03)

**Status: partial success.** The pilot ran 50 discovered adapters with
26 successfully audited (Llama: 18, Mistral: 8, Qwen: 0).

| Criterion | Result | Detail |
|-----------|--------|--------|
| C1 (viability) | **FAIL** (52%) | Disk space + size limits caused 23 failures. Pipeline itself is functional. |
| C2 (metric sanity) | **PASS** | All metrics in expected ranges. |
| C3 (arch coverage) | **PASS** | 2 families with 8+ (Llama: 18, Mistral: 8). |
| C4 (task coverage) | **FAIL** | Only 2 task categories with 5+; need 3. |
| C5 (visible signal) | **PASS** | Llama SR=2.03 vs. Mistral SR=1.68 (18.8% relative diff). |

**Assessment:** Failures are operational (disk, size limits, cohort
composition), not methodological. The pipeline works end-to-end. The
spectral signal is visible. The original gate is not retroactively
amended — instead, a pilot-plus gate is defined below.

Key pilot findings that inform the pilot-plus design:

- Task η² (0.26) > architecture η² (0.12): task is the dominant factor,
  contradicting H1.
- Architecture kNN purity (0.90) is very high despite low η²: tight
  local clusters, low global variance explained.
- Nominal rank confound R² = 0.66: residualization alone is
  insufficient; rank-matched subset is required.
- Module-type asymmetry (attn < MLP) replicates in only 25% of decoder
  adapters: encoder pattern does not hold. First-class finding.

## Pilot-Plus Gate

The pilot-plus is a corrected rerun that addresses operational failures
and adds the analytic controls the pilot results demand. It is not a
full core rollout.

### Pilot-plus operational fixes

1. **Fix disk/size constraint.** Either raise the adapter size ceiling
   above 500 MB, implement stream-audit-then-clean (audit adapters and
   cache only the audit JSON, not the full weights), or provision
   additional disk. The 52% viability rate is unacceptable for a census;
   the target is ≥80%.

2. **Force the third architecture family.** Target Qwen explicitly in
   the discovery pass, prioritizing it before other families if needed.
   If Qwen remains operationally unavailable (download restrictions,
   adapter format issues), substitute Phi or Gemma with explicit
   rationale documented in the manifest.

### Pilot-plus analytic additions

3. **Include analytic controls from Phase 2b.** The pilot-plus pass
   must complete the residualized analysis, rank-matched subset, task-label
   confidence subset, and variant-aware architecture checks defined in
   Phase 2b before the core gate is evaluated. These are not optional
   for the pilot-plus.

4. **Module-type asymmetry as first-class output.** The non-replication
   of encoder module-type patterns is analyzed per Phase 5 with
   subtype-level breakdown and architectural attribution. This is a
   deliverable of the pilot-plus, not deferred to core.

### Pilot-plus success conditions (gate to core)

Proceed to core cohort only if the pilot-plus achieves **all** of:

1. ≥3 architecture families with usable density (≥5 adapters each).
2. ≥3 task categories with usable density (≥5 adapters each).
3. Residualized and rank-matched analyses are completed.
4. Non-random architecture or task structure survives at least one
   analytic control (residualized η² > 0.05 or rank-matched subset
   shows visible separation).

If conditions 1-3 pass but condition 4 fails, the census approach is
likely insufficient and the GPU-return controlled study becomes the
required path. Document and pause.

### Pilot-plus target cohort

| Family | Target adapters | Notes |
|--------|----------------|-------|
| Llama | 18 (carry forward from pilot) + up to 10 new | Already well-represented |
| Mistral | 8 (carry forward) + up to 10 new | Increase density |
| Qwen | 10-15 new | Must achieve ≥5 usable |
| **Total** | **50-70 audited** | Not a full core rollout |

## Infrastructure

### Script entrypoint

Extend `scripts/broader_benchmarks.py` or create a new
`scripts/ecosystem_census.py` that:

1. Reuses `discover_adapters()` with expanded `BASE_MODELS` list
   filtered to decoder-only families.
2. Reuses `download_adapter()` unchanged.
3. Reuses `audit_adapter()` unchanged.
4. Adds architecture-family and task-category labeling to `AdapterRecord`.
5. Adds fingerprint vector extraction from audit results.
6. Adds census-specific analysis functions (ANOVA, clustering, kNN purity).
7. Writes census manifest and fingerprint table as structured JSON.

### Idempotency

Follow existing `broader_benchmarks.py` pattern: re-running skips
already-audited adapters. Cache downloaded adapters in a persistent
directory. Store audit results per-adapter as JSON for incremental
reanalysis.

### Adapter cache location

```
.cache/census_adapters/{repo_id_with_dashes}/
├── adapter_config.json
├── adapter_model.safetensors
└── (optional) README.md
```

## Deliverables

Planned outputs under a new field trial directory:

```
field_trials/public_ecosystem_census/
├── manifest.json                           # cohort definition, inclusion/exclusion log
├── adapter_records.json                    # per-adapter metadata + audit status
├── fingerprint_table.json                  # per-adapter fingerprint vectors
├── fingerprint_table.md                    # human-readable fingerprint summary
├── confound_assessment.json                # confound regression results
├── rank_matched_analysis.json              # rank-matched subset results (Phase 2b)
├── task_label_confidence.json              # task-label confidence subset results (Phase 2b)
├── variant_robustness.json                 # within-family variant checks (Phase 2b)
├── architecture_task_decomposition.json    # ANOVA / effect size results (raw + residualized)
├── architecture_task_decomposition.md      # human-readable decomposition summary
├── clustering_results.json                 # kNN purity, cluster assignments
├── module_type_asymmetry.json              # attention-vs-MLP + subtype-level analysis
├── module_type_asymmetry.md                # first-class writeup with arch attribution
├── pilot_gate_report.md                    # pilot validation criteria results (complete)
├── pilot_plus_gate_report.md               # pilot-plus gate evaluation
├── study_memo.md                           # interpretive summary (success/partial/negative)
└── excluded_adapters.json                  # adapters that failed inclusion, with reasons
```

## Success / Failure Criteria

Revised post-pilot to reflect the actual signal structure observed.
The original criteria assumed architecture dominance (H1); the pilot
showed task dominance. Criteria are updated accordingly.

### Success condition

Architecture and task effects are both detectable in fingerprint space
after confound residualization and in rank-matched subsets, with η² >
0.05 for each in at least 2 core metrics. The pilot's geometric
interpretation (architecture = cluster precision, task = cluster
location) is testable with 3+ architecture families. Module-type
asymmetry non-replication is characterized at the subtype level with
architectural attribution.

### Partial success condition

One of architecture or task signal survives confound controls, but not
both. Or signals are present but η² drops below 0.05 after
residualization, leaving only the kNN-purity evidence for local
structure. Module-type analysis is descriptive but attribution is
speculative. Bounded extension language is supported with prominent
confound caveats.

### Negative completion condition

No architecture or task signal survives rank-matched subset analysis.
The pilot's visible separation (Llama SR=2.03 vs. Mistral SR=1.68)
collapses when nominal rank is controlled. Census approach does not
resolve the architecture-vs-task question; controlled GPU study
becomes strictly necessary.

All outcomes are useful and should be documented as such.

## Guardrails

- Do not claim causal architecture effects from observational data.
- Do not treat Hub download count or popularity as quality evidence.
- Report confound assessment before architecture/task effects in all summaries.
- Keep "found artifact" caveat explicit in every interpretive statement.
- Do not convert census findings into product policy without replication
  (either via the GPU-return controlled study or an independent census).
- Do not promote secondary probes to primary decision metrics in first pass.
- If a finding seems too clean, check for confound leakage (e.g., one
  architecture family having systematically different nominal ranks in
  the population).

## Execution Sequence

### Pre-execution (current, no compute required)

- [x] Finalize cohort table template and inclusion criteria
- [x] Confirm `audit_lora_peft_dir()` handles all target adapter formats
  on CPU (safetensors + bin, various target_module patterns)
- [x] Draft census script skeleton (`scripts/ecosystem_census.py`)
- [x] Create empty `field_trials/public_ecosystem_census/` directory
  with `manifest.json` stub
- [x] Draft analysis notebook/script skeleton for phases 1-5

### Pilot execution (complete — partial success)

- [x] Run discovery for decoder-only families, log candidate pool size
- [x] Download + audit first 30-50 adapters
- [x] Evaluate pilot validation gate criteria 1-5
- [x] Write `pilot_gate_report.md`
- [x] Decision: partial success; proceed to pilot-plus rerun

### Pilot-plus execution (current stage)

Operational fixes:
- [ ] Fix disk/size constraint (raise limit, stream-audit-then-clean, or both)
- [ ] Target Qwen explicitly in discovery pass; document substitute if needed

Cohort expansion:
- [ ] Carry forward 26 audited pilot adapters
- [ ] Add Qwen adapters (target: ≥5 usable)
- [ ] Backfill Mistral and Llama if needed for task coverage
- [ ] Achieve ≥50 total audited adapters across 3 architecture families

Analytic controls (required before core gate evaluation):
- [ ] Run Phase 2b: rank-matched subset analysis
- [ ] Run Phase 2b: task-label confidence subset analysis
- [ ] Run Phase 2b: variant-aware architecture robustness checks
- [ ] Run Phase 5: module-type asymmetry at subtype level with arch attribution
- [ ] Run Phases 1-4 on expanded cohort with raw + residualized results

Gate evaluation:
- [ ] Evaluate pilot-plus success conditions 1-4
- [ ] Write `pilot_plus_gate_report.md`
- [ ] Decision: proceed to core, or pause pending GPU study

### Core execution (gated on pilot-plus success)

- [ ] Expand to 100-150 adapters
- [ ] Run full analysis plan (phases 1-5, including Phase 2b controls)
- [ ] Write `architecture_task_decomposition.md`
- [ ] Write `module_type_asymmetry.md` (first-class output)
- [ ] Write `study_memo.md` with success/partial/negative assessment

### Optional extension

- [ ] Expand to 200+ adapters if core results are promising but underpowered
- [ ] Add additional architecture families (Phi, Gemma)
- [ ] Explore interaction effects with larger cohort

## Estimated Resource Requirements

| Phase | CPU time | Disk | Network |
|-------|----------|------|---------|
| Pilot (50 adapters) | ~1-2 hours | ~5-10 GB adapter cache | Hub downloads |
| Core (150 adapters) | ~4-8 hours | ~15-30 GB adapter cache | Hub downloads |
| Extended (200+) | ~8-16 hours | ~30-50 GB adapter cache | Hub downloads |

CPU time estimates assume ~1-2 minutes per adapter (download + audit).
Disk estimates assume average adapter size ~100-200 MB.

## Bottom Line

This study answers a bounded version of the decoder-only question using
resources available now:

> **Run a CPU-only spectral census of public decoder-only LoRA adapters
> that tests architecture-vs-task clustering in found artifacts, with
> explicit confound assessment and bounded claims.**

It does not replace the controlled GPU-return study. It provides
ecological evidence at scale that the GPU study cannot, and the GPU
study provides causal evidence that the census cannot. Together they
form a complementary pair.

### Current status

**Pilot complete, pilot-plus in progress.** The pilot (n=26) achieved
partial success: the pipeline works, spectral signal is visible, but
operational failures (disk, size limits) and missing architecture
coverage (no Qwen) prevented the original gate from passing. The pilot
also surfaced findings that are more interesting than a clean
confirmation would have been: task dominance over architecture in
variance explained, high architecture kNN purity despite low η², strong
nominal-rank confound, and non-replication of encoder module-type
asymmetry on decoders.

The pilot-plus corrects operational constraints, forces the third
architecture family, adds the analytic controls the pilot results
demand, and promotes the module-type asymmetry finding to a first-class
output. It targets 50-70 audited adapters across 3 families — enough to
evaluate the pilot-plus gate, not a full core rollout. Core expansion is
gated on the pilot-plus success conditions.
