# Design: M1 Controlled Interference Experiment

**Date:** 2026-02-15
**Author:** John T. Nanney
**Status:** Approved for implementation

---

## Purpose

Demonstrate that Gradience's spectral metrics predict LoRA adapter merge quality before merging. Train 4 specialized adapters on Mistral-7B, merge them pairwise using 4 methods, and correlate pre-merge spectral signatures with post-merge performance.

## Success Criteria

- Pre-merge spectral metrics explain >=50% of variance in post-merge quality, OR
- Correctly flag >=80% of "bad merges" (>5% degradation on either constituent task)

---

## Architecture

### File Layout

```
scripts/m1_experiment/
  m1_config.yaml          # Master experiment configuration
  task_configs.py         # Per-task training configs (datasets, prompts, formatting)
  phase1_train.py         # Train 12 adapters (4 tasks x 3 seeds)
  phase2_audit.py         # Pairwise merge-audit (6 pairs x 3 seeds = 18 audits)
  phase3_merge.py         # Execute merges (18 pairs x 4 methods = 72 merges)
  phase4_evaluate.py      # Evaluate all adapters via lm-evaluation-harness
  phase5_analyze.py       # Correlation analysis + report generation
  run_all.sh              # Master orchestrator
```

### Workspace Layout (RunPod)

```
/workspace/m1/
  adapters/               # Phase 1 output: 12 PEFT adapter directories
    sql/seed_42/
    sql/seed_123/
    sql/seed_456/
    chat/seed_42/  ...    # 4 tasks x 3 seeds
  audits/                 # Phase 2 output: 18 merge-audit results
    sql_chat/seed_42/merge_audit.json
    sql_math/seed_42/merge_audit.json  ...
  merges/                 # Phase 3 output: 72 merged adapters
    sql_chat/seed_42/linear/
    sql_chat/seed_42/ties/
    sql_chat/seed_42/dare_linear/
    sql_chat/seed_42/dare_ties/  ...
  evals/                  # Phase 4 output
    individual/           # 12 base adapter eval results
    merged/               # 72 merged adapter eval results
  analysis/               # Phase 5 output
    correlation_report.json
    correlation_report.md
```

Each phase checks for existing outputs and skips completed work (resumable).

---

## Core Library Changes

### DARE Merge Strategy

Add to `gradience/vnext/merge/strategies.py`:

**DARELinearMerge**: For each parameter in the delta weight matrices, randomly drop with probability `p` (from `trim_fraction`), rescale remaining by `1/(1-p)`, then linear average with coefficients.

**DARETIESMerge**: Apply DARE random dropout first, then run the existing TIES pipeline (trim by magnitude, elect sign, disjoint mean) on the sparsified deltas.

Both register in the existing `STRATEGIES` dict so `get_strategy("dare_linear")` and `get_strategy("dare_ties")` work with `gradience merge` CLI immediately.

Add corresponding entries to `PLAN_STRATEGIES` in `plan.py` for `gradience merge-plan` support.

### No Other Core Changes

The existing merge-audit API, merge execution engine, SVD refactoring, and reporting infrastructure handle everything else. The experiment scripts call the Python API directly.

---

## Experiment Configuration

Single `m1_config.yaml` defines the entire experiment:

```yaml
experiment:
  name: "m1_controlled_interference"
  version: "1.0"
  base_model: "mistralai/Mistral-7B-v0.1"
  seeds: [42, 123, 456]

adapters:
  sql:
    dataset: "b-mc2/sql-create-context"
    max_train_samples: 10000
    eval_task: "sql_generation"
  chat:
    dataset: "yahma/alpaca-cleaned"
    max_train_samples: 10000
    eval_task: "mmlu"
  math:
    dataset: "gsm8k"
    subset: "main"
    max_train_samples: 7473
    eval_task: "gsm8k"
  code:
    dataset: "sahil2801/CodeAlpaca-20k"
    max_train_samples: 10000
    eval_task: "humaneval"

training:
  rank: 32
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  learning_rate: 5e-5
  max_steps: 1200
  batch_size: 1
  gradient_accumulation: 16
  torch_dtype: "bfloat16"

merge:
  methods: ["linear", "ties", "dare_linear", "dare_ties"]
  linear_coefficients: [0.5, 0.5]
  ties_density: 0.5
  dare_linear_density: 0.7
  dare_ties_density: 0.5
  output_rank: 32

evaluation:
  framework: "lm-evaluation-harness"
  general_capability: "mmlu"
  general_capability_subjects: ["abstract_algebra", "college_mathematics", "formal_logic"]
  max_eval_samples: 500

runtime:
  device: "cuda"
  workspace: "/workspace/m1"
```

---

## Phase Details

### Phase 1: Train Adapters (~24 GPU-hours)

For each (task, seed) pair: load Mistral-7B, attach LoRA (r=32, alpha=32, q/k/v/o_proj), fine-tune with HF Trainer for 1200 steps. Each task has its own data formatting function in `task_configs.py`:

- **SQL**: text-to-SQL with schema context
- **Chat**: instruction-following (Alpaca format)
- **Math**: GSM8K chain-of-thought
- **Code**: code generation from docstrings

Skip if `adapter_config.json` already exists in output directory.

### Phase 2: Pairwise Merge-Audit (~1 GPU-hour)

Generate all C(4,2) = 6 unique pairs. For each pair and seed, call `gradience.vnext.merge.merge_audit()` Python API. Writes `merge_audit.json` per (pair, seed). This is CPU-bound SVD work.

### Phase 3: Execute Merges (~1 GPU-hour)

For each (pair, seed, method): generate a merge plan via `plan_from_audit()` using the Phase 2 audit results, then execute via `execute_merge()`. The 4 methods map to:

- `linear` -> `uniform_linear` plan strategy
- `ties` -> `overlap_ties` plan strategy
- `dare_linear` -> new DARE-linear plan strategy
- `dare_ties` -> new DARE-TIES plan strategy

### Phase 4: Evaluate (~7 GPU-hours)

Use `lm-evaluation-harness` for standardized evaluation:

- 12 individual adapters: each on its own task
- 72 merged adapters: each on both constituent tasks + MMLU subset
- Total: ~156 evaluation runs

Output: JSON results per adapter with task-specific metrics.

### Phase 5: Analyze (<1 min, CPU)

Load all audit JSONs + eval results. Compute:

1. **Correlation analysis**: Pearson/Spearman between pre-merge metrics (mean_overlap, directional_agreement, magnitude_ratio, stable_rank_ratio) and post-merge accuracy degradation
2. **Linear regression**: `merge_quality ~ overlap + rank_ratio + scale_ratio` with R-squared
3. **Binary classification**: predict "bad merge" (>5% degradation) from spectral metrics, report accuracy/precision/recall
4. **Per-module-type breakdown**: attention Q/K/V/O patterns
5. **Per-method comparison**: which merge method is most robust to spectral incompatibility

Output: `correlation_report.json` (structured data) + `correlation_report.md` (human-readable).

---

## RunPod Execution

`run_all.sh` runs phases sequentially. Each phase is independently runnable:

```bash
# Full experiment
bash scripts/m1_experiment/run_all.sh

# Or run phases independently
python scripts/m1_experiment/phase1_train.py --config scripts/m1_experiment/m1_config.yaml
python scripts/m1_experiment/phase2_audit.py --config scripts/m1_experiment/m1_config.yaml
# ...
```

Estimated total: ~33 GPU-hours on A100 (protocol budget: ~40 on A40).

---

## Testing

- **DARE strategy**: Unit tests with small random tensors in `tests/test_merge_strategies.py`
- **Phase scripts**: `--smoke` flag trains for 5 steps, evaluates on 10 samples
- **CI**: Smoke config runs full pipeline on CPU with tiny model (not Mistral)

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training approach | Standalone scripts | Heterogeneous tasks (SQL, chat, math, code) don't fit bench's single-task profile model |
| DARE implementation | Core library | Reusable via CLI, follows existing strategy pattern |
| Evaluation framework | lm-evaluation-harness | Industry standard, handles GSM8K/MMLU/HumanEval without custom eval code |
| Execution strategy | Phased with checkpoints | Resumable after failures, phases can run on different pod sizes |
