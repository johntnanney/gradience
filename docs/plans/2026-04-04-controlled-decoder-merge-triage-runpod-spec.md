# Controlled Decoder Merge Triage — RunPod GPU Study Spec

**Date:** April 4, 2026
**Status:** proposed
**Depends on:** decoder ecosystem census (closed), Post 3 (Mistral-7B merge dominance), Study 16 (Llama-2-7B ablation), existing RunPod infrastructure
**Addresses:** Whether the spectral triage pipeline — not just the spectral measurements — transfers to the commercially relevant decoder regime
**See also:** [`2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`](2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md), [`../strategy/state-of-program-april-2026.md`](../strategy/state-of-program-april-2026.md) (§5b), [`../technical-report.md`](../technical-report.md) (§7)


## Motivation

The Gradience triage pipeline — adapter QA, eligibility gating, pairwise merge audit, task-boundary detection, near-miss identification, inventory action plan — is validated on small encoders (DistilBERT, BERT-base, RoBERTa-base) with 90–93% candidate elimination, zero false positives on task boundaries, and correct prioritization across 53+ pairs and 16 evaluated merges.

The entire commercially relevant regime (7B+ decoder models, instruction/generation tasks, rank 8–64) is outside this validated envelope. Two prior studies provide suggestive but insufficient evidence:

- **Post 3** (Mistral-7B): Subspace overlap predicts merge dominance at r = 0.846 across 27 cross-task pairs, with 2.4× same-task/cross-task separation. This is a correlation study on individual merge pairs, not an inventory-level triage validation.
- **Study 16** (Llama-2-7B): End-to-end ablation on 5 pairs confirms structural compatibility is necessary but not sufficient. This motivated eligibility gating but was not a triage pipeline run.

Neither study answers the question this spec addresses: **does the full inventory pipeline — the thing a practitioner would actually use — produce correct narrowing and prioritization on a decoder-scale adapter inventory?**

The decoder census (closed April 2026) established that decoder spectral structure is real and non-random, but also that encoder-derived intuitions do not transfer cleanly: module-type asymmetry does not replicate, confound pressure from nominal rank is stronger, and the attention/MLP utilization gap vanishes at scale. These non-replications mean the triage pipeline cannot be assumed to work at decoder scale — it must be tested.

### Strategic rationale

This study is proposed ahead of DeBERTa adjudication (§5a in state-of-program) despite DeBERTa's higher information-per-compute-hour, because the two experiments resolve different kinds of uncertainty:

- DeBERTa resolves **parametric uncertainty within a settled regime** (does the conjunctive model generalize to a third small encoder). The prior is strong; the expected information gain per experiment is moderate.
- This study resolves **structural uncertainty across regimes** (does the triage system work where practitioners actually work). The prior is genuinely split; the expected information gain per experiment is high.

The paper framing positions Gradience as a spectral triage system. A reviewer asking "does this work on models people actually use?" needs an answer from this study, not from DeBERTa.


## Scope

### In scope

- One decoder architecture (Mistral-7B-v0.1), deepening on existing merge evidence
- Two task families: classification-adjacent (for continuity with encoder validation) and instruction-following (for commercial relevance)
- LoRA adapters at rank 8 and rank 16 (continuity with encoder regime + modest rank expansion)
- Full inventory-level triage pipeline: adapter QA → eligibility gating → pairwise merge audit → task-boundary detection → near-miss identification → action plan
- Behavioral evaluation of retained, near-miss, and stratified control merges
- Direct comparison with encoder-regime pipeline behavior

### Out of scope

- Multiple decoder architectures (deferred to follow-on if this study succeeds)
- Generation-task evaluation (summarization, translation, open-ended text)
- High-rank adapters (r ≥ 32) as a primary variable
- Rank compression / bench pipeline validation (separate study)
- V-module conjunctive mechanism testing at decoder scale (requires multi-backbone, deferred)
- Spectral fingerprinting / architecture-vs-task variance decomposition (covered by the existing decoder fingerprinting GPU plan)
- Universal decoder generality claims


## Program Questions

1. **Pipeline transfer.** Does the full Gradience inventory pipeline (not just spectral measurements) produce a useful action plan on a Mistral-7B adapter inventory?
2. **Narrowing rate.** Does the 90%+ candidate elimination rate hold at decoder scale, or does it collapse (over-retaining) or over-exclude (eliminating viable merges)?
3. **Task-boundary detection.** Does the zero-false-positive record on task boundaries extend to decoder classification and instruction-following tasks?
4. **Near-miss at decoder scale.** Do near-miss pairs behave like retained pairs (as on encoders) or like cross-task controls?
5. **Verdict calibration.** Do the current verdict thresholds (calibrated on encoder profiles) produce sensible layer-level classifications on decoder adapter pairs, or do they need recalibration?
6. **Evidence gate at decoder scale.** Does the 500-sample CPU evidence bootstrap remain sufficient to calibrate eligibility on decoder adapters, or does the higher capacity of 7B models require larger evaluation samples?


## Hypotheses

- **H1:** The inventory pipeline will produce a non-degenerate action plan (neither empty nor full) on a mixed-task decoder inventory.
- **H2:** Retained pairs will outperform cross-task controls on merge evaluation, preserving the correct prioritization ordering.
- **H3:** Task-boundary detection will maintain zero false positives (same-task pairs never receive cross-task advisory).
- **H4:** Near-miss pairs will degrade less than cross-task controls (within 2× of retained-pair degradation, as on encoders).
- **H5:** Current verdict thresholds will classify at least 60% of layers with the same verdict that a human reviewer would assign. (Below 60% indicates recalibration is needed before the pipeline is usable at decoder scale.)
- **H6:** Norm imbalance will be less dominant as a structural issue than in field trials (because adapters are trained under controlled conditions with matched rank, unlike the public r=1 vs r=16 heterogeneity of the encoder field trials).


## Experiment Design

### Phase 0: Environment validation (30 min)

Validate the RunPod environment, cache configuration, and Gradience installation against the Mistral-7B base model before committing to the full training schedule.

**Steps:**
1. Provision a single A100 40GB pod on RunPod
2. Run environment setup (see Infrastructure section below)
3. Verify base model download and inference: load `mistralai/Mistral-7B-v0.1`, run 10 inference examples
4. Verify LoRA training: train a throwaway rank-8 adapter for 50 steps on a small slice of SST-2
5. Verify spectral audit: run `gradience audit` on the throwaway adapter
6. Verify merge pipeline: run `gradience merge-audit` on two throwaway adapters

**Gate:** All six verifications pass. If any fail, debug before proceeding.

### Phase 1: Adapter training (8–10 hours)

Train 16 LoRA adapters on Mistral-7B-v0.1 under controlled conditions.

**Cohort design:**

| Task family | Dataset | Task type | Seeds | Rank | Adapters |
|---|---|---|---|---|---|
| Classification | SST-2 (sentiment) | SEQ_CLS | 42, 123 | 8 | 2 |
| Classification | SST-2 (sentiment) | SEQ_CLS | 42, 123 | 16 | 2 |
| Classification | QNLI (entailment) | SEQ_CLS | 42, 123 | 8 | 2 |
| Classification | QNLI (entailment) | SEQ_CLS | 42, 123 | 16 | 2 |
| Instruction | Alpaca-cleaned | CAUSAL_LM | 42, 123 | 8 | 2 |
| Instruction | Alpaca-cleaned | CAUSAL_LM | 42, 123 | 16 | 2 |
| Instruction | OpenAssistant-1 | CAUSAL_LM | 42, 123 | 8 | 2 |
| Instruction | OpenAssistant-1 | CAUSAL_LM | 42, 123 | 16 | 2 |
| **Total** | | | | | **16** |

This produces 120 unique pairs (16 choose 2), of which:

- 12 same-task pairs (same dataset, different seed or rank)
- 24 same-family pairs (same task type, different dataset)
- 84 cross-task pairs (classification × instruction)

**Training protocol (classification adapters):**

```yaml
model:
  name: mistralai/Mistral-7B-v0.1
  load_in_4bit: false
  torch_dtype: bfloat16

task:
  type: SEQ_CLS
  dataset: glue
  subset: sst2  # or qnli
  metric: accuracy
  num_labels: 2  # or 2 for QNLI (entailment/not)

train:
  max_steps: 1000
  eval_steps: 200
  save_steps: 200
  save_total_limit: 3
  lr: 2e-5
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
  weight_decay: 0.01
  warmup_ratio: 0.06
  train_samples: 5000
  eval_samples: 1000

lora:
  r: 8  # or 16
  alpha: 16  # 2× rank
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  task_type: SEQ_CLS

runtime:
  device: cuda
  bf16: true
```

**Training protocol (instruction adapters):**

Same base configuration except:

```yaml
task:
  type: CAUSAL_LM
  dataset: tatsu-lab/alpaca  # or OpenAssistant/oasst1
  metric: eval_loss  # + downstream eval via lm-evaluation-harness or manual rubric
  max_seq_length: 512

train:
  max_steps: 2000
  lr: 1e-4
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8

lora:
  task_type: CAUSAL_LM
```

**Checkpoint strategy:** Save final checkpoint + checkpoint at 50% of training for optional progression analysis. Each adapter's final checkpoint is the unit of analysis.

**Estimated time:** ~30–45 min per adapter × 16 adapters = 8–12 hours. Classification adapters will be faster (shorter sequences, fewer steps); instruction adapters will be slower.

**Resumability:** The training script must save each adapter independently. If the pod dies after training 12 of 16 adapters, restart from adapter 13. Implement as a manifest-driven loop that checks for existing adapter directories and skips completed ones.

### Phase 2: Spectral audit and pipeline run (1–2 hours, partially CPU)

This phase can overlap with Phase 1 — audit completed adapters while later ones are still training.

**Steps:**
1. Run `gradience audit` on each completed adapter → per-adapter QA artifact
2. Run evidence bootstrap (1000-sample eval on held-out data) → eligibility classification
3. Run `gradience merge-audit` on all eligible pairs → per-pair risk reports
4. Run `gradience summarize-inventory` → inventory summary + action plan + preflight bundle

**Key observation points:**
- How many adapters are classified as eligible vs uncertain vs flagged_weak?
- What is the pair-level risk distribution (low/medium/high)?
- What dominant issues appear (norm_imbalance, subspace_conflict, high_redundancy, etc.)?
- Does the action plan contain a non-degenerate retained set?
- How many near-miss pairs are identified?
- Does task-boundary detection fire correctly on all cross-task pairs?

### Phase 3: Merge evaluation (3–5 hours)

Merge and evaluate a stratified sample of pairs. Do not evaluate all 120 — evaluate enough to answer the program questions.

**Evaluation cohort:**

| Category | Pairs to evaluate | Selection rule |
|---|---|---|
| Retained same-task | All (up to 6) | Pipeline's top recommendations |
| Retained same-family | Up to 4 | Pipeline's same-family recommendations |
| Near-miss | All (up to 6) | All near-miss pairs from action plan |
| Cross-task control | 4–6 | Stratified sample from cross-task skip list |
| **Total target** | 16–22 merges | |

**Merge protocol:**
- Use `norm_equalized` (pipeline's default for moderate-risk) and `linear` (baseline) strategies
- For each merge: load merged adapter onto base model, evaluate on the held-out evaluation split used for evidence bootstrap
- Record: accuracy / eval_loss, per-example predictions (for neither-source rate computation if feasible), merge time

**Evaluation metrics:**
- Primary: Δ vs best source (same metric used for evidence bootstrap)
- Secondary: neither-source rate (if per-example predictions are captured)
- Ordering: retained > near-miss > cross-task control (by average Δ)

### Phase 4: Analysis and decision (post-GPU, CPU only)

**Analyses:**
1. **Narrowing rate:** retained / total eligible pairs. Compare with encoder baseline (7–10% retention).
2. **Prioritization correctness:** Is the retained category's average Δ better than near-miss, which is better than cross-task control?
3. **Task-boundary accuracy:** Count false positives and false negatives on task-boundary detection.
4. **Near-miss behavior:** Average Δ for near-miss vs retained vs cross-task. Test whether near-miss is closer to retained (as on encoders) or to cross-task.
5. **Verdict distribution:** Tabulate layer-level verdicts. Compare distribution with encoder-regime profiles. Flag any verdict that appears at >80% frequency (the norm-imbalance saturation problem from field trials).
6. **Evidence gate calibration:** Were any adapters misclassified? Did any eligible adapters produce bad merges? Did any flagged adapters look like they should have passed?
7. **Pipeline-to-encoder comparison:** Structural comparison of the decoder action plan with encoder action plans from Pilots 2, 3, and Phase 2b.


## Infrastructure

### Hardware

- **Pod type:** RunPod 1× A100 40GB (sufficient for Mistral-7B LoRA training at bf16)
- **Disk:** 200GB workspace volume (base model ~14GB, 16 adapters ~2GB total, datasets ~5GB, working space)
- **Estimated rental:** 20–24 hours @ ~$1.50/hour = $30–36

### Environment setup

```bash
# 1. Cache configuration (MUST be first)
export HF_HOME=/workspace/hf_cache/hf_home
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
mkdir -p /workspace/hf_cache/{hf_home,hub,datasets}

# 2. Repository setup
cd /workspace
git clone https://github.com/johntnanney/gradience.git
cd gradience
git checkout <study-branch>  # create a study branch before starting

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[hf,dev]"

# 3. Additional dependencies for decoder training
pip install bitsandbytes  # optional, only if 4-bit quantization needed
pip install accelerate
pip install trl  # for SFTTrainer if using instruction fine-tuning

# 4. Pre-download base model (do this early, ~14GB)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-v0.1',
    torch_dtype='bfloat16',
    device_map='auto'
)
print('Model cached successfully')
"

# 5. Pre-download datasets
python -c "
from datasets import load_dataset
load_dataset('glue', 'sst2')
load_dataset('glue', 'qnli')
load_dataset('tatsu-lab/alpaca')
load_dataset('OpenAssistant/oasst1')
print('Datasets cached successfully')
"

# 6. Verify Gradience installation
make verify-version
```

### Directory structure

```
/workspace/
├── gradience/                          # Repository checkout
├── hf_cache/                           # HuggingFace model/dataset cache
│   ├── hf_home/
│   ├── hub/
│   └── datasets/
└── experiments/
    └── decoder_merge_triage/           # This study's working directory
        ├── adapters/                   # Trained adapter checkpoints
        │   ├── sst2_r8_s42/
        │   ├── sst2_r8_s123/
        │   ├── sst2_r16_s42/
        │   ├── ...
        │   └── oasst1_r16_s123/
        ├── training_logs/              # Per-adapter training logs
        ├── audits/                     # Per-adapter QA artifacts
        ├── evidence/                   # Evidence bootstrap results
        ├── merge_reports/              # Per-pair merge risk reports
        ├── merges/                     # Merged adapter weights + eval results
        ├── inventory/                  # Inventory summary + action plan + bundle
        ├── manifest.json               # Adapter training manifest (resumability)
        └── study_log.md                # Running notes during execution
```


## Scripts to Prepare Before GPU Day

The following scripts should be written and tested (on CPU with dummy data where possible) before renting the pod. Each script should be independently runnable and idempotent (safe to re-run if interrupted).

### 1. `scripts/decoder_triage_study/phase0_validate_env.py`

Runs the six-gate Phase 0 smoke validation exactly as defined above:
GPU/runtime, deps/CLI, base-model inference, throwaway LoRA training,
`gradience audit`, and `gradience merge-audit`.

**Interface:**
```bash
python scripts/decoder_triage_study/phase0_validate_env.py \
    --base-model mistralai/Mistral-7B-v0.1 \
    --output-dir /workspace/experiments/decoder_merge_triage/phase0_validation
```

### 2. `scripts/decoder_triage_study/train_cohort.py`

Trains all 16 adapters from the cohort table. Reads a manifest JSON specifying the adapter configurations. Skips adapters whose output directories already exist (resumability). Saves final checkpoint + midpoint checkpoint per adapter.

**Interface:**
```bash
python scripts/decoder_triage_study/train_cohort.py \
    --manifest scripts/decoder_triage_study/cohort_manifest.json \
    --output-dir /workspace/experiments/decoder_merge_triage/adapters \
    --log-dir /workspace/experiments/decoder_merge_triage/training_logs
```

**Manifest format:**
```json
[
    {
        "adapter_id": "sst2_r8_s42",
        "base_model": "mistralai/Mistral-7B-v0.1",
        "task_type": "SEQ_CLS",
        "dataset": "glue",
        "dataset_subset": "sst2",
        "lora_rank": 8,
        "lora_alpha": 16,
        "seed": 42,
        "max_steps": 1000,
        "lr": 2e-5,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
]
```

### 3. `scripts/decoder_triage_study/cohort_manifest.json`

The full manifest for all 16 adapters. Should be committed to the repo and version-controlled.

### 4. `scripts/decoder_triage_study/run_pipeline.py`

Runs the Gradience inventory pipeline on completed adapters: audit → evidence bootstrap → pairwise merge audit → inventory summary. Outputs the full preflight bundle.

**Interface:**
```bash
python scripts/decoder_triage_study/run_pipeline.py \
    --adapter-dir /workspace/experiments/decoder_merge_triage/adapters \
    --output-dir /workspace/experiments/decoder_merge_triage/inventory \
    --base-model mistralai/Mistral-7B-v0.1 \
    --eval-samples 1000
```

### 5. `scripts/decoder_triage_study/evaluate_merges.py`

Merges and evaluates the stratified pair sample selected from the inventory action plan. Takes a merge plan JSON (generated from the action plan output) specifying which pairs to merge and evaluate.

**Interface:**
```bash
python scripts/decoder_triage_study/evaluate_merges.py \
    --merge-plan /workspace/experiments/decoder_merge_triage/merge_plan.json \
    --adapter-dir /workspace/experiments/decoder_merge_triage/adapters \
    --output-dir /workspace/experiments/decoder_merge_triage/merges \
    --base-model mistralai/Mistral-7B-v0.1 \
    --eval-samples 1000
```

### 6. `scripts/decoder_triage_study/analyze_results.py`

Post-GPU analysis. Computes all metrics from Program Questions and generates the study memo. Can run on CPU.

**Interface:**
```bash
python scripts/decoder_triage_study/analyze_results.py \
    --study-dir /workspace/experiments/decoder_merge_triage \
    --output /workspace/experiments/decoder_merge_triage/study_memo.md
```


## Estimated Effort

| Phase | GPU hours | Wall time | Cost (A100 @ $1.50/hr) |
|---|---|---|---|
| Phase 0: Environment validation | 0.5 | 30 min | $0.75 |
| Phase 1: Adapter training | 8–12 | 8–12 hr | $12–18 |
| Phase 2: Pipeline run | 1–2 | 1–2 hr | $1.50–3.00 |
| Phase 3: Merge evaluation | 3–5 | 3–5 hr | $4.50–7.50 |
| **Total GPU** | **12.5–19.5** | **12.5–19.5 hr** | **$19–29** |
| Phase 4: Analysis | 0 (CPU) | 2–4 hr | $0 |

**Budget recommendation:** Rent for 24 hours ($36) to provide margin for debugging, re-runs, and unexpected issues. Total expected spend including margin: **$36–40**.


## Decision Criteria

### Success

The full inventory pipeline produces a non-degenerate action plan with correct prioritization: retained pairs outperform cross-task controls, near-miss pairs behave like retained pairs (not like controls), and task-boundary detection has zero false positives. Verdict distribution is interpretable without recalibration. The decoder triage pipeline is operationally validated.

**Consequence:** Extend the validated-claims envelope to include Mistral-7B classification and instruction-following. Write the decoder triage section of the paper with operational evidence, not just suggestive correlations.

### Partial success

The pipeline runs and produces a non-degenerate action plan, but one or more of: prioritization ordering is correct but margins are thin; near-miss pairs don't clearly separate from controls; verdict thresholds need recalibration (>40% of layers mis-classified); evidence gate requires larger sample sizes. The pipeline is functional but needs tuning.

**Consequence:** Document the specific recalibration needed. Defer strong decoder claims until recalibrated thresholds are validated. Still publishable as a transfer study with honest boundary reporting.

### Negative

The pipeline produces a degenerate action plan (either empty or retaining nearly everything), or prioritization ordering is wrong (cross-task controls outperform retained pairs), or task-boundary detection has false positives. The triage logic does not transfer to decoders in its current form.

**Consequence:** Document precisely where the pipeline breaks. This is a first-class result — it would mean the encoder-calibrated logic is genuinely architecture-specific and the triage pipeline needs decoder-specific threshold development before it can claim decoder applicability. Write up as a negative-result contribution showing the limits of encoder-to-decoder transfer.

All three outcomes are publishable. None are wasted compute.


## What This Does Not Address

- **Conjunctive mechanism at decoder scale.** Testing whether V-module dimensionality ratio separates catastrophic from safe merges on decoders requires multiple backbones (to distinguish mechanism from backbone effect). This study uses one backbone.
- **Generation-task triage.** The instruction-following adapters are trained on instruction data but evaluated on held-out instruction quality, not on open-ended generation benchmarks. True generation-task triage (summarization, translation) remains untested.
- **Architecture generality.** This study validates on Mistral-7B only. Whether the results transfer to Llama, Qwen, or other decoder families is a separate question addressed by the existing decoder fingerprinting GPU plan.
- **High-rank behavior.** Rank 16 is included but rank 32/64 are not. The Post 7 audit shows consistent spectral profiles at higher ranks, but merge triage at high rank is untested.
- **Scale beyond 7B.** Whether the pipeline works on 13B, 70B, or larger models is outside scope.


## Relationship to Existing Work

| Existing study | What it established | What this study adds |
|---|---|---|
| Post 3 (Mistral-7B merge) | Overlap predicts dominance (r=0.846, 27 pairs) | Full pipeline inventory run, not just pairwise correlation |
| Study 16 (Llama-2-7B ablation) | Structural compatibility necessary not sufficient | Evidence gate + eligibility + near-miss at decoder scale |
| Decoder census (n=36) | Decoder spectral structure exists, non-random | Controlled training replaces found-artifact confounds |
| Post 7 (n=86 audit) | Spectral metrics are architecture-agnostic | Merge triage pipeline (not just audit) at decoder scale |
| Encoder field trials (Pilots 1–3, Phase 2b) | Pipeline validated on small encoders | Direct pipeline-to-pipeline comparison across regimes |
| Decoder fingerprinting GPU plan | Architecture-vs-task variance decomposition | Merge triage validation (complementary, not overlapping) |


## Deliverables

All outputs under `field_trials/decoder_merge_triage/`:

```
field_trials/decoder_merge_triage/
├── cohort_definition.md              # Adapter cohort table and training protocol
├── cohort_manifest.json              # Machine-readable adapter configurations
├── adapter_manifest.json             # Completed adapters with training metadata
├── pipeline_output/                  # Full Gradience preflight bundle
│   ├── preflight_summary.json
│   ├── preflight_summary.md
│   ├── inventory_action_plan.md
│   └── run_manifest.json
├── merge_evaluation/                 # Merge results
│   ├── merge_plan.json               # Which pairs were merged and why
│   ├── merge_results.json            # Per-pair evaluation metrics
│   └── merge_results.md              # Human-readable results table
├── analysis/                         # Post-GPU analysis
│   ├── narrowing_rate.json           # Retention rate comparison with encoder baseline
│   ├── prioritization_test.json      # Retained vs near-miss vs control ordering
│   ├── verdict_distribution.json     # Layer-level verdict profile
│   └── encoder_comparison.json       # Structural comparison with encoder field trials
├── study_memo.md                     # Primary narrative deliverable
└── study_log.md                      # Running notes from execution
```


## Guardrails

- Do not claim decoder generality from one architecture
- Do not claim generation-task coverage from instruction-following evaluation
- Do not recalibrate thresholds during the study — run the existing pipeline, document where it breaks, propose recalibration after
- Do not treat instruction-following evaluation quality as equivalent to the classification evaluation quality that grounds the encoder validation
- Keep the honest distinction between "the pipeline runs" and "the pipeline produces correct recommendations" — the former is necessary but not sufficient
- Report verdict-threshold mismatches as transparently as verdict successes


## Bottom Line

This study answers one question: **does the Gradience triage pipeline work on a Mistral-7B adapter inventory?**

A positive answer extends the validated regime into the commercially relevant space and strengthens the system-paper framing. A negative answer identifies precisely where the encoder-calibrated logic breaks, which is equally valuable for the research program and equally publishable. Either outcome materially advances the program; neither is wasted compute.

Estimated cost: ~$36. Estimated time: 24 hours of pod rental (13–20 hours active compute + margin). Scripts should be written and tested on CPU before the pod is provisioned.
