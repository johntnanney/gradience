# M1 Controlled Interference Experiment — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the DARE merge strategies in the core library and create all M1 experiment scripts (training, auditing, merging, evaluating, analyzing) that run on RunPod.

**Architecture:** Add two new merge strategies (DARELinearMerge, DARETIESMerge) to `gradience/vnext/merge/strategies.py` with corresponding plan strategies in `plan.py`. Create a standalone `scripts/m1_experiment/` directory with 5 phase scripts, a master config YAML, per-task data formatting, and a shell orchestrator. All phases are independently runnable and resumable.

**Tech Stack:** PyTorch, PEFT, HuggingFace Transformers/Datasets, lm-evaluation-harness, scipy (for correlation analysis), scikit-learn (for binary classification metrics).

---

## Task 1: DARE Merge Strategies — Tests

**Files:**
- Modify: `tests/merge/test_strategies.py`

**Step 1: Write failing tests for DARELinearMerge**

Add to the end of `tests/merge/test_strategies.py`:

```python
# ---------------------------------------------------------------------------
# DARELinearMerge
# ---------------------------------------------------------------------------


class TestDARELinearMerge:
    def test_output_shape_preserved(self, simple_dW_pair, make_config):
        """Output has the same shape as inputs."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_linear", trim_fraction=0.3)
        strategy = get_strategy("dare_linear")
        torch.manual_seed(0)
        result = strategy.merge(dW_a, dW_b, config)
        assert result.shape == dW_a.shape

    def test_no_dropout_equals_linear(self, simple_dW_pair, make_config):
        """With trim_fraction=0.0 (no dropout), DARE-Linear = Linear."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_linear", trim_fraction=0.0)
        strategy = get_strategy("dare_linear")
        result = strategy.merge(dW_a, dW_b, config)
        expected = 0.5 * dW_a + 0.5 * dW_b
        torch.testing.assert_close(result, expected)

    def test_full_dropout_gives_zeros(self, simple_dW_pair, make_config):
        """With trim_fraction=1.0, all params dropped → zeros."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_linear", trim_fraction=1.0)
        strategy = get_strategy("dare_linear")
        result = strategy.merge(dW_a, dW_b, config)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_rescaling_preserves_expected_value(self, make_config):
        """After drop+rescale, expected value matches original."""
        torch.manual_seed(42)
        # Large tensor for statistical convergence
        dW_a = torch.ones(1000, 1000)
        dW_b = torch.ones(1000, 1000)
        config = make_config(strategy="dare_linear", trim_fraction=0.3, coefficients=(0.5, 0.5))
        strategy = get_strategy("dare_linear")
        # Average over many trials
        results = []
        for seed in range(50):
            torch.manual_seed(seed)
            r = strategy.merge(dW_a, dW_b, config)
            results.append(r.mean().item())
        avg = sum(results) / len(results)
        # Expected: 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert abs(avg - 1.0) < 0.05, f"Expected ~1.0, got {avg}"

    def test_deterministic_with_same_seed(self, simple_dW_pair, make_config):
        """Same manual seed → same result."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_linear", trim_fraction=0.5)
        strategy = get_strategy("dare_linear")
        torch.manual_seed(99)
        r1 = strategy.merge(dW_a, dW_b, config)
        torch.manual_seed(99)
        r2 = strategy.merge(dW_a, dW_b, config)
        torch.testing.assert_close(r1, r2)


# ---------------------------------------------------------------------------
# DARETIESMerge
# ---------------------------------------------------------------------------


class TestDARETIESMerge:
    def test_output_shape_preserved(self, simple_dW_pair, make_config):
        """Output has the same shape as inputs."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_ties", trim_fraction=0.3)
        strategy = get_strategy("dare_ties")
        torch.manual_seed(0)
        result = strategy.merge(dW_a, dW_b, config)
        assert result.shape == dW_a.shape

    def test_no_dropout_equals_ties(self, simple_dW_pair, make_config):
        """With trim_fraction=0.0, DARE-TIES = TIES (no DARE dropout applied)."""
        dW_a, dW_b = simple_dW_pair
        # trim_fraction=0.0 means no DARE dropout.
        # But TIES itself also uses trim_fraction for its magnitude trim.
        # When trim_fraction=0, both DARE dropout and TIES trim are skipped.
        config_dare = make_config(strategy="dare_ties", trim_fraction=0.0)
        config_ties = make_config(strategy="ties", trim_fraction=0.0)
        dare_strategy = get_strategy("dare_ties")
        ties_strategy = get_strategy("ties")
        result_dare = dare_strategy.merge(dW_a, dW_b, config_dare)
        result_ties = ties_strategy.merge(dW_a, dW_b, config_ties)
        torch.testing.assert_close(result_dare, result_ties)

    def test_full_dropout_gives_zeros(self, simple_dW_pair, make_config):
        """With trim_fraction=1.0, all params dropped → zeros."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_ties", trim_fraction=1.0)
        strategy = get_strategy("dare_ties")
        result = strategy.merge(dW_a, dW_b, config)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_deterministic_with_same_seed(self, simple_dW_pair, make_config):
        """Same manual seed → same result."""
        dW_a, dW_b = simple_dW_pair
        config = make_config(strategy="dare_ties", trim_fraction=0.5)
        strategy = get_strategy("dare_ties")
        torch.manual_seed(99)
        r1 = strategy.merge(dW_a, dW_b, config)
        torch.manual_seed(99)
        r2 = strategy.merge(dW_a, dW_b, config)
        torch.testing.assert_close(r1, r2)


# ---------------------------------------------------------------------------
# Updated factory tests
# ---------------------------------------------------------------------------


class TestGetStrategyExtended:
    def test_dare_linear(self):
        s = get_strategy("dare_linear")
        assert isinstance(s, DARELinearMerge)

    def test_dare_ties(self):
        s = get_strategy("dare_ties")
        assert isinstance(s, DARETIESMerge)
```

Also update the imports at the top of the test file to include the new classes:

```python
from gradience.vnext.merge.strategies import (
    LayerMergeConfig,
    LinearMerge,
    TIESMerge,
    DARELinearMerge,
    DARETIESMerge,
    get_strategy,
)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/merge/test_strategies.py -v`
Expected: ImportError or multiple FAILs (DARELinearMerge/DARETIESMerge don't exist yet)

---

## Task 2: DARE Merge Strategies — Implementation

**Files:**
- Modify: `gradience/vnext/merge/strategies.py`
- Modify: `gradience/vnext/merge/__init__.py`

**Step 1: Implement DARELinearMerge and DARETIESMerge**

Add to `gradience/vnext/merge/strategies.py`, after the `TIESMerge` class and before the `_STRATEGIES` dict:

```python
# ---------------------------------------------------------------------------
# DARE helpers
# ---------------------------------------------------------------------------


def _dare_dropout(task_vector: Tensor, drop_fraction: float) -> Tensor:
    """Randomly drop parameters and rescale survivors by 1/(1-p).

    Implements the DARE (Drop And REscale) sparsification from
    Yu et al., 2023 ("Language Models are Super Mario").

    Parameters
    ----------
    task_vector : tensor of any shape
    drop_fraction : probability of dropping each parameter (0 = no drop, 1 = all dropped)

    Returns
    -------
    Sparsified and rescaled tensor with same shape.
    """
    if drop_fraction <= 0.0:
        return task_vector.clone()
    if drop_fraction >= 1.0:
        return torch.zeros_like(task_vector)

    # Bernoulli mask: 1 = keep, 0 = drop
    mask = torch.bernoulli(torch.full_like(task_vector, 1.0 - drop_fraction))
    # Rescale kept values so E[output] = input
    rescale = 1.0 / (1.0 - drop_fraction)
    return task_vector * mask * rescale


# ---------------------------------------------------------------------------
# DARE-Linear merge
# ---------------------------------------------------------------------------


class DARELinearMerge(MergeStrategy):
    """DARE + Linear: random dropout with rescaling, then weighted average.

    For each task vector:
    1. Randomly drop parameters with probability ``trim_fraction``
    2. Rescale surviving parameters by ``1 / (1 - trim_fraction)``
    3. Weighted linear combination of the sparsified task vectors

    This reduces parameter interference while preserving the expected
    value of the merged output (Yu et al., 2023).

    When ``trim_fraction=0.0``, this is identical to ``LinearMerge``.
    """

    def merge(
        self,
        dW_a: Tensor,
        dW_b: Tensor,
        config: LayerMergeConfig,
    ) -> Tensor:
        coeff_a, coeff_b = config.coefficients
        drop_fraction = config.trim_fraction

        # DARE sparsification
        sparse_a = _dare_dropout(dW_a, drop_fraction)
        sparse_b = _dare_dropout(dW_b, drop_fraction)

        # Linear combination
        return coeff_a * sparse_a + coeff_b * sparse_b


# ---------------------------------------------------------------------------
# DARE-TIES merge
# ---------------------------------------------------------------------------


class DARETIESMerge(MergeStrategy):
    """DARE + TIES: random dropout with rescaling, then TIES pipeline.

    For each task vector:
    1. DARE: randomly drop with probability ``trim_fraction``, rescale by ``1/(1-p)``
    2. TIES: elect majority sign across sparsified task vectors
    3. TIES: disjoint mean of values agreeing with elected sign

    Note: ``trim_fraction`` controls the DARE dropout probability.  The TIES
    magnitude trim step is *not* applied separately — DARE sparsification
    replaces it.  This follows the DARE-TIES formulation from Yu et al. (2023).

    When ``trim_fraction=0.0``, this is identical to ``TIESMerge`` with no trim.
    """

    def merge(
        self,
        dW_a: Tensor,
        dW_b: Tensor,
        config: LayerMergeConfig,
    ) -> Tensor:
        coeff_a, coeff_b = config.coefficients
        drop_fraction = config.trim_fraction

        # Scale by coefficients
        tv_a = coeff_a * dW_a
        tv_b = coeff_b * dW_b

        # DARE sparsification (replaces TIES magnitude trim)
        tv_a_sparse = _dare_dropout(tv_a, drop_fraction)
        tv_b_sparse = _dare_dropout(tv_b, drop_fraction)

        # TIES: elect sign + disjoint mean
        elected = _elect_sign([tv_a_sparse, tv_b_sparse])
        return _disjoint_mean([tv_a_sparse, tv_b_sparse], elected)
```

Update the `_STRATEGIES` dict:

```python
_STRATEGIES = {
    "linear": LinearMerge,
    "ties": TIESMerge,
    "dare_linear": DARELinearMerge,
    "dare_ties": DARETIESMerge,
}
```

Update `get_strategy` docstring to include new names:

```python
    name : ``"linear"``, ``"ties"``, ``"dare_linear"``, or ``"dare_ties"``
```

**Step 2: Update `__init__.py` exports**

In `gradience/vnext/merge/__init__.py`, add to the import from strategies:

```python
from gradience.vnext.merge.strategies import (
    LayerMergeConfig,
    MergeStrategy,
    LinearMerge,
    TIESMerge,
    DARELinearMerge,
    DARETIESMerge,
    get_strategy,
)
```

And add to `__all__`:

```python
    "DARELinearMerge",
    "DARETIESMerge",
```

**Step 3: Run tests to verify they pass**

Run: `python3 -m pytest tests/merge/test_strategies.py -v`
Expected: All tests PASS

**Step 4: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: 586+ tests pass, no regressions

**Step 5: Commit**

```bash
git add gradience/vnext/merge/strategies.py gradience/vnext/merge/__init__.py tests/merge/test_strategies.py
git commit -m "feat: add DARE-Linear and DARE-TIES merge strategies

Implements Drop And REscale (DARE) sparsification from Yu et al. 2023.
Two new strategies registered in the factory:
- dare_linear: DARE dropout → weighted linear combination
- dare_ties: DARE dropout → TIES sign election + disjoint mean

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: DARE Plan Strategies

**Files:**
- Modify: `gradience/vnext/merge/plan.py`
- Create: `tests/merge/test_dare_plan.py`

**Step 1: Write failing tests for DARE plan strategies**

Create `tests/merge/test_dare_plan.py`:

```python
"""Tests for DARE plan strategy generation."""

from __future__ import annotations

import pytest

from gradience.vnext.merge.plan import (
    PLAN_STRATEGIES,
    plan_from_audit,
)
from gradience.vnext.merge.report import MergeAuditReport


@pytest.fixture
def mock_report():
    """Minimal MergeAuditReport for plan generation tests."""
    return MergeAuditReport(
        adapter_a={
            "path": "/tmp/adapter_a",
            "rank": 32,
            "alpha": 32.0,
            "n_layers": 2,
        },
        adapter_b={
            "path": "/tmp/adapter_b",
            "rank": 32,
            "alpha": 32.0,
            "n_layers": 2,
        },
        matching={
            "n_shared": 2,
            "n_only_a": 0,
            "n_only_b": 0,
        },
        layer_verdicts=[
            {
                "layer_name": "model.layers.0.self_attn.q_proj",
                "verdict": "safe",
                "metrics": {"mean_overlap": 0.3},
                "suggested_coefficients": None,
            },
            {
                "layer_name": "model.layers.0.self_attn.v_proj",
                "verdict": "conflicting",
                "metrics": {"mean_overlap": 0.7},
                "suggested_coefficients": None,
            },
        ],
        aggregate={
            "overall_verdict": "safe",
            "compatibility_score": 0.65,
        },
    )


class TestDARELinearPlan:
    def test_registered(self):
        assert "dare_linear" in PLAN_STRATEGIES

    def test_generates_plan(self, mock_report):
        plan = plan_from_audit(
            "dare_linear",
            mock_report,
            "/tmp/adapter_a",
            "/tmp/adapter_b",
            output_rank=32,
            dare_drop_fraction=0.3,
        )
        assert plan.strategy_name == "dare_linear"
        assert len(plan.layer_configs) == 2
        for lc in plan.layer_configs:
            assert lc.strategy == "dare_linear"
            assert lc.trim_fraction == 0.3

    def test_default_drop_fraction(self, mock_report):
        plan = plan_from_audit(
            "dare_linear",
            mock_report,
            "/tmp/adapter_a",
            "/tmp/adapter_b",
        )
        for lc in plan.layer_configs:
            assert lc.trim_fraction == 0.3  # default


class TestDARETIESPlan:
    def test_registered(self):
        assert "dare_ties" in PLAN_STRATEGIES

    def test_generates_plan(self, mock_report):
        plan = plan_from_audit(
            "dare_ties",
            mock_report,
            "/tmp/adapter_a",
            "/tmp/adapter_b",
            output_rank=32,
            dare_drop_fraction=0.5,
        )
        assert plan.strategy_name == "dare_ties"
        assert len(plan.layer_configs) == 2
        for lc in plan.layer_configs:
            assert lc.strategy == "dare_ties"
            assert lc.trim_fraction == 0.5

    def test_default_drop_fraction(self, mock_report):
        plan = plan_from_audit(
            "dare_ties",
            mock_report,
            "/tmp/adapter_a",
            "/tmp/adapter_b",
        )
        for lc in plan.layer_configs:
            assert lc.trim_fraction == 0.5  # default
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/merge/test_dare_plan.py -v`
Expected: FAIL (dare_linear/dare_ties not in PLAN_STRATEGIES)

**Step 3: Implement DARE plan strategies**

Add to `gradience/vnext/merge/plan.py`, after `plan_overlap_ties` and before the `PLAN_STRATEGIES` dict:

```python
def plan_dare_linear(
    report: MergeAuditReport,
    adapter_a_dir: str,
    adapter_b_dir: str,
    coefficients: Tuple[float, float] = (0.5, 0.5),
    output_rank: int = 8,
    output_alpha: float = 16.0,
    dare_drop_fraction: float = 0.3,
) -> MergePlan:
    """DARE-Linear merge on all shared layers.

    Applies DARE random dropout (controlled by ``dare_drop_fraction``) before
    a weighted linear combination.  The ``trim_fraction`` field in each
    LayerMergeConfig carries the DARE drop probability.
    """
    layer_names = _shared_layer_names(report)

    layer_configs = tuple(
        LayerMergeConfig(
            module_prefix=name,
            strategy="dare_linear",
            coefficients=coefficients,
            target_rank=output_rank,
            trim_fraction=dare_drop_fraction,
        )
        for name in layer_names
    )

    return MergePlan(
        plan_id=str(uuid.uuid4()),
        strategy_name="dare_linear",
        adapter_a_dir=adapter_a_dir,
        adapter_b_dir=adapter_b_dir,
        output_rank=output_rank,
        output_alpha=output_alpha,
        layer_configs=layer_configs,
        metadata=_make_metadata(
            report, "dare_linear",
            {"coefficients": list(coefficients), "dare_drop_fraction": dare_drop_fraction},
        ),
    )


def plan_dare_ties(
    report: MergeAuditReport,
    adapter_a_dir: str,
    adapter_b_dir: str,
    coefficients: Tuple[float, float] = (0.5, 0.5),
    output_rank: int = 8,
    output_alpha: float = 16.0,
    dare_drop_fraction: float = 0.5,
) -> MergePlan:
    """DARE-TIES merge on all shared layers.

    Applies DARE random dropout (controlled by ``dare_drop_fraction``) before
    running the TIES pipeline (sign election + disjoint mean).
    """
    layer_names = _shared_layer_names(report)

    layer_configs = tuple(
        LayerMergeConfig(
            module_prefix=name,
            strategy="dare_ties",
            coefficients=coefficients,
            target_rank=output_rank,
            trim_fraction=dare_drop_fraction,
        )
        for name in layer_names
    )

    return MergePlan(
        plan_id=str(uuid.uuid4()),
        strategy_name="dare_ties",
        adapter_a_dir=adapter_a_dir,
        adapter_b_dir=adapter_b_dir,
        output_rank=output_rank,
        output_alpha=output_alpha,
        layer_configs=layer_configs,
        metadata=_make_metadata(
            report, "dare_ties",
            {"coefficients": list(coefficients), "dare_drop_fraction": dare_drop_fraction},
        ),
    )
```

Update `PLAN_STRATEGIES`:

```python
PLAN_STRATEGIES: Dict[str, Callable[..., MergePlan]] = {
    "uniform_linear": plan_uniform_linear,
    "audit_aware": plan_audit_aware,
    "overlap_ties": plan_overlap_ties,
    "dare_linear": plan_dare_linear,
    "dare_ties": plan_dare_ties,
}
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/merge/test_dare_plan.py -v`
Expected: All PASS

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass

**Step 5: Commit**

```bash
git add gradience/vnext/merge/plan.py tests/merge/test_dare_plan.py
git commit -m "feat: add dare_linear and dare_ties plan strategies

Register DARE plan strategies in PLAN_STRATEGIES so merge-plan CLI
and plan_from_audit() support dare_linear and dare_ties methods.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: M1 Config YAML

**Files:**
- Create: `scripts/m1_experiment/m1_config.yaml`

**Step 1: Write the master config**

Create `scripts/m1_experiment/m1_config.yaml`:

```yaml
# M1 Controlled Interference Experiment — Master Configuration
#
# Train 4 specialized LoRA adapters on Mistral-7B (3 seeds each),
# run pairwise merge-audit, merge with 4 methods, and evaluate.
#
# Usage:
#   python scripts/m1_experiment/phase1_train.py --config scripts/m1_experiment/m1_config.yaml
#   python scripts/m1_experiment/phase2_audit.py --config scripts/m1_experiment/m1_config.yaml
#   python scripts/m1_experiment/phase3_merge.py --config scripts/m1_experiment/m1_config.yaml
#   python scripts/m1_experiment/phase4_evaluate.py --config scripts/m1_experiment/m1_config.yaml
#   python scripts/m1_experiment/phase5_analyze.py --config scripts/m1_experiment/m1_config.yaml

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
  learning_rate: 5.0e-5
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

# Smoke test overrides (used with --smoke flag)
smoke:
  max_steps: 5
  max_train_samples: 50
  max_eval_samples: 10
  seeds: [42]
```

**Step 2: Commit**

```bash
git add scripts/m1_experiment/m1_config.yaml
git commit -m "config: add M1 experiment master configuration

Defines adapters (SQL, chat, math, code), training hyperparameters,
merge methods (linear, TIES, DARE-linear, DARE-TIES), and evaluation
settings for the controlled interference experiment.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Task Configs Module

**Files:**
- Create: `scripts/m1_experiment/task_configs.py`

**Step 1: Write task formatting functions**

Create `scripts/m1_experiment/task_configs.py`:

```python
#!/usr/bin/env python3
"""
Per-task training data formatting for M1 experiment.

Each task has a `format_<task>` function that takes a dataset example
and returns a formatted string for causal LM fine-tuning.

Supported tasks: sql, chat, math, code
"""

from __future__ import annotations


def format_sql(example: dict) -> str:
    """Format SQL generation example: schema + question → SQL query.

    Dataset: b-mc2/sql-create-context
    Fields: context (schema), question, answer (SQL)
    """
    return (
        f"### Schema:\n{example['context']}\n\n"
        f"### Question:\n{example['question']}\n\n"
        f"### SQL:\n{example['answer']}"
    )


def format_chat(example: dict) -> str:
    """Format instruction-following example in Alpaca format.

    Dataset: yahma/alpaca-cleaned
    Fields: instruction, input (optional), output
    """
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    if inp and inp.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def format_math(example: dict) -> str:
    """Format GSM8K chain-of-thought math example.

    Dataset: gsm8k (main)
    Fields: question, answer (includes reasoning + #### final_answer)
    """
    return (
        f"### Question:\n{example['question']}\n\n"
        f"### Answer:\n{example['answer']}"
    )


def format_code(example: dict) -> str:
    """Format code generation example from docstring.

    Dataset: sahil2801/CodeAlpaca-20k
    Fields: prompt, completion
    """
    return (
        f"### Instruction:\n{example['prompt']}\n\n"
        f"### Code:\n{example['completion']}"
    )


# Registry mapping task name → formatter
TASK_FORMATTERS = {
    "sql": format_sql,
    "chat": format_chat,
    "math": format_math,
    "code": format_code,
}


def get_formatter(task_name: str):
    """Get the formatting function for a task.

    Raises ValueError if task_name is not recognized.
    """
    fn = TASK_FORMATTERS.get(task_name)
    if fn is None:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {sorted(TASK_FORMATTERS.keys())}"
        )
    return fn
```

**Step 2: Commit**

```bash
git add scripts/m1_experiment/task_configs.py
git commit -m "feat(m1): add per-task data formatting for training

Formatting functions for SQL, chat/instruction, math/GSM8K, and code
generation. Each maps dataset fields to a prompt+completion string
for causal LM fine-tuning.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Phase 1 — Training Script

**Files:**
- Create: `scripts/m1_experiment/phase1_train.py`

**Step 1: Write the training orchestrator**

Create `scripts/m1_experiment/phase1_train.py`:

```python
#!/usr/bin/env python3
"""
Phase 1: Train 12 adapters (4 tasks x 3 seeds) on Mistral-7B.

For each (task, seed) pair:
  1. Load Mistral-7B base model
  2. Attach LoRA (r=32, alpha=32, q/k/v/o_proj)
  3. Fine-tune with HF Trainer for 1200 steps
  4. Save PEFT adapter

Skips if adapter_config.json already exists in the output directory.

Usage:
    python scripts/m1_experiment/phase1_train.py \\
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test (5 steps, 1 seed):
    python scripts/m1_experiment/phase1_train.py \\
        --config scripts/m1_experiment/m1_config.yaml --smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


def load_config(config_path: str, smoke: bool = False) -> dict:
    """Load and optionally apply smoke test overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["training"]["max_steps"] = smoke_cfg.get("max_steps", 5)
        for task in config["adapters"].values():
            task["max_train_samples"] = smoke_cfg.get("max_train_samples", 50)
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])

    return config


def train_single_adapter(
    base_model: str,
    task_name: str,
    task_config: dict,
    training_config: dict,
    seed: int,
    output_dir: Path,
    device: str = "cuda",
) -> Path:
    """Train one LoRA adapter for a (task, seed) pair."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # Import task formatter (sibling module)
    sys.path.insert(0, str(Path(__file__).parent))
    from task_configs import get_formatter

    adapter_dir = output_dir / task_name / f"seed_{seed}"

    # Skip if already trained
    if (adapter_dir / "adapter_config.json").exists():
        print(f"  [SKIP] {task_name}/seed_{seed} — already exists")
        return adapter_dir

    print(f"\n  Training {task_name}/seed_{seed}...")
    start = time.monotonic()

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(training_config.get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=training_config["rank"],
        lora_alpha=training_config["alpha"],
        lora_dropout=0.0,
        target_modules=training_config["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load dataset
    ds_name = task_config["dataset"]
    ds_subset = task_config.get("subset", None)
    if ds_subset:
        ds = load_dataset(ds_name, ds_subset, split="train")
    else:
        ds = load_dataset(ds_name, split="train")

    # Subsample
    max_samples = task_config.get("max_train_samples", 10000)
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    print(f"    Dataset: {ds_name}, {len(ds)} examples")

    # Format + tokenize
    formatter = get_formatter(task_name)

    def tokenize_fn(example):
        text = formatter(example)
        enc = tokenizer(text, truncation=True, max_length=512, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names)

    # Collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    train_dir = adapter_dir / "training_logs"
    training_args = TrainingArguments(
        output_dir=str(train_dir),
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation"],
        learning_rate=training_config["learning_rate"],
        max_steps=training_config["max_steps"],
        logging_steps=50,
        save_strategy="no",
        bf16=(training_config.get("torch_dtype") == "bfloat16"),
        fp16=(training_config.get("torch_dtype") == "float16"),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    # Save adapter
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))

    elapsed = time.monotonic() - start
    print(f"    Saved to: {adapter_dir} ({elapsed / 60:.1f} min)")

    # Cleanup GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return adapter_dir


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 1: Train adapters")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (5 steps, 1 seed)")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    base_model = config["experiment"]["base_model"]
    seeds = config["experiment"]["seeds"]
    training_config = config["training"]

    total_start = time.monotonic()
    n_total = len(config["adapters"]) * len(seeds)
    n_done = 0

    print(f"Phase 1: Training {n_total} adapters")
    print(f"  Base model: {base_model}")
    print(f"  Seeds: {seeds}")
    print(f"  Tasks: {list(config['adapters'].keys())}")

    for task_name, task_config in config["adapters"].items():
        for seed in seeds:
            n_done += 1
            print(f"\n[{n_done}/{n_total}] {task_name}/seed_{seed}")
            train_single_adapter(
                base_model=base_model,
                task_name=task_name,
                task_config=task_config,
                training_config=training_config,
                seed=seed,
                output_dir=adapters_dir,
                device=config["runtime"]["device"],
            )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 1 complete: {n_total} adapters in {elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/m1_experiment/phase1_train.py
git commit -m "feat(m1): add phase 1 training script

Trains 12 LoRA adapters (4 tasks x 3 seeds) on Mistral-7B.
Supports --smoke flag for quick testing. Skips existing adapters.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Phase 2 — Audit Script

**Files:**
- Create: `scripts/m1_experiment/phase2_audit.py`

**Step 1: Write the pairwise audit orchestrator**

Create `scripts/m1_experiment/phase2_audit.py`:

```python
#!/usr/bin/env python3
"""
Phase 2: Pairwise merge-audit (6 pairs x 3 seeds = 18 audits).

For each unique adapter pair and seed, runs gradience merge_audit()
to compute spectral compatibility metrics (principal angles, directional
agreement, magnitude balance).

Output: merge_audit.json per (pair, seed) in workspace/audits/.

Usage:
    python scripts/m1_experiment/phase2_audit.py \\
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test:
    python scripts/m1_experiment/phase2_audit.py \\
        --config scripts/m1_experiment/m1_config.yaml --smoke
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import yaml

from gradience.vnext.merge import merge_audit


def load_config(config_path: str, smoke: bool = False) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])
    return config


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 2: Pairwise merge-audit")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (1 seed)")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    audits_dir = workspace / "audits"

    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())

    # Generate all C(4,2) = 6 unique pairs
    pairs = list(itertools.combinations(task_names, 2))
    n_total = len(pairs) * len(seeds)

    total_start = time.monotonic()
    print(f"Phase 2: Running {n_total} pairwise merge-audits")
    print(f"  Pairs: {pairs}")
    print(f"  Seeds: {seeds}")

    n_done = 0
    for task_a, task_b in pairs:
        for seed in seeds:
            n_done += 1
            pair_name = f"{task_a}_{task_b}"
            audit_dir = audits_dir / pair_name / f"seed_{seed}"

            # Skip if already done
            if (audit_dir / "merge_audit.json").exists():
                print(f"  [{n_done}/{n_total}] [SKIP] {pair_name}/seed_{seed}")
                continue

            adapter_a = adapters_dir / task_a / f"seed_{seed}"
            adapter_b = adapters_dir / task_b / f"seed_{seed}"

            if not adapter_a.exists() or not adapter_b.exists():
                print(f"  [{n_done}/{n_total}] [MISSING] {pair_name}/seed_{seed}")
                continue

            print(f"  [{n_done}/{n_total}] Auditing {pair_name}/seed_{seed}...")
            start = time.monotonic()

            report = merge_audit(
                adapter_a_dir=adapter_a,
                adapter_b_dir=adapter_b,
                output_dir=audit_dir,
                verbose=False,
            )

            elapsed = time.monotonic() - start
            verdict = report.aggregate.get("overall_verdict", "unknown")
            score = report.aggregate.get("compatibility_score", 0.0)
            print(f"    Verdict: {verdict} (score={score:.3f}) [{elapsed:.1f}s]")

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 2 complete: {n_total} audits in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/m1_experiment/phase2_audit.py
git commit -m "feat(m1): add phase 2 pairwise merge-audit script

Runs merge_audit() on all C(4,2)=6 adapter pairs x 3 seeds.
Skips existing audit results for resumability.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Phase 3 — Merge Script

**Files:**
- Create: `scripts/m1_experiment/phase3_merge.py`

**Step 1: Write the merge execution orchestrator**

Create `scripts/m1_experiment/phase3_merge.py`:

```python
#!/usr/bin/env python3
"""
Phase 3: Execute merges (18 pairs x 4 methods = 72 merges).

For each (pair, seed, method): generate a merge plan from the Phase 2
audit report, then execute via the gradience merge engine.

Output: PEFT-compatible merged adapters in workspace/merges/.

Usage:
    python scripts/m1_experiment/phase3_merge.py \\
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test:
    python scripts/m1_experiment/phase3_merge.py \\
        --config scripts/m1_experiment/m1_config.yaml --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import yaml

from gradience.vnext.merge import execute_merge, plan_from_audit
from gradience.vnext.merge.report import MergeAuditReport


def load_config(config_path: str, smoke: bool = False) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])
    return config


def load_audit_report(audit_dir: Path) -> MergeAuditReport:
    """Load a merge audit report from JSON."""
    report_path = audit_dir / "merge_audit.json"
    with open(report_path) as f:
        data = json.load(f)
    return MergeAuditReport(
        adapter_a=data["adapter_a"],
        adapter_b=data["adapter_b"],
        matching=data["matching"],
        layer_verdicts=data["layer_verdicts"],
        aggregate=data["aggregate"],
    )


def get_plan_kwargs(method: str, merge_config: dict) -> dict:
    """Build plan_from_audit kwargs for a given method."""
    output_rank = merge_config["output_rank"]
    coefficients = tuple(merge_config.get("linear_coefficients", [0.5, 0.5]))

    base = {"output_rank": output_rank, "output_alpha": float(output_rank)}

    if method == "linear":
        return {**base, "coefficients": coefficients}
    elif method == "ties":
        return {**base, "trim_fraction": merge_config.get("ties_density", 0.5)}
    elif method == "dare_linear":
        return {**base, "coefficients": coefficients,
                "dare_drop_fraction": 1.0 - merge_config.get("dare_linear_density", 0.7)}
    elif method == "dare_ties":
        return {**base, "coefficients": coefficients,
                "dare_drop_fraction": 1.0 - merge_config.get("dare_ties_density", 0.5)}
    else:
        return base


# Map from m1_config method names to plan strategy names
METHOD_TO_PLAN = {
    "linear": "uniform_linear",
    "ties": "overlap_ties",
    "dare_linear": "dare_linear",
    "dare_ties": "dare_ties",
}


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 3: Execute merges")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (1 seed)")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    audits_dir = workspace / "audits"
    merges_dir = workspace / "merges"

    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    methods = config["merge"]["methods"]
    merge_config = config["merge"]

    pairs = list(itertools.combinations(task_names, 2))
    n_total = len(pairs) * len(seeds) * len(methods)

    total_start = time.monotonic()
    print(f"Phase 3: Executing {n_total} merges")
    print(f"  Pairs: {len(pairs)}, Seeds: {len(seeds)}, Methods: {methods}")

    n_done = 0
    for task_a, task_b in pairs:
        for seed in seeds:
            pair_name = f"{task_a}_{task_b}"
            audit_dir = audits_dir / pair_name / f"seed_{seed}"

            if not (audit_dir / "merge_audit.json").exists():
                print(f"  [MISSING AUDIT] {pair_name}/seed_{seed} — skipping all methods")
                n_done += len(methods)
                continue

            report = load_audit_report(audit_dir)

            adapter_a = str(adapters_dir / task_a / f"seed_{seed}")
            adapter_b = str(adapters_dir / task_b / f"seed_{seed}")

            for method in methods:
                n_done += 1
                merge_dir = merges_dir / pair_name / f"seed_{seed}" / method

                # Skip if already done
                if (merge_dir / "adapter_config.json").exists():
                    print(f"  [{n_done}/{n_total}] [SKIP] {pair_name}/seed_{seed}/{method}")
                    continue

                print(f"  [{n_done}/{n_total}] Merging {pair_name}/seed_{seed}/{method}...")
                start = time.monotonic()

                plan_strategy = METHOD_TO_PLAN[method]
                plan_kwargs = get_plan_kwargs(method, merge_config)

                plan = plan_from_audit(
                    plan_strategy,
                    report,
                    adapter_a,
                    adapter_b,
                    **plan_kwargs,
                )

                result = execute_merge(plan, merge_dir, verbose=False)

                elapsed = time.monotonic() - start
                print(
                    f"    recon_error={result.mean_reconstruction_error:.4f} "
                    f"[{elapsed:.1f}s]"
                )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 3 complete: {n_total} merges in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/m1_experiment/phase3_merge.py
git commit -m "feat(m1): add phase 3 merge execution script

Executes 72 merges (6 pairs x 3 seeds x 4 methods) using
gradience plan_from_audit + execute_merge. Skips completed merges.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 9: Phase 4 — Evaluation Script

**Files:**
- Create: `scripts/m1_experiment/phase4_evaluate.py`

**Step 1: Write the evaluation orchestrator**

Create `scripts/m1_experiment/phase4_evaluate.py`:

```python
#!/usr/bin/env python3
"""
Phase 4: Evaluate all adapters via lm-evaluation-harness.

Evaluates:
  - 12 individual adapters (each on its own task)
  - 72 merged adapters (each on both constituent tasks + MMLU subset)
  - Total: ~156 evaluation runs

Output: JSON results per adapter in workspace/evals/.

Usage:
    python scripts/m1_experiment/phase4_evaluate.py \\
        --config scripts/m1_experiment/m1_config.yaml

    # Smoke test:
    python scripts/m1_experiment/phase4_evaluate.py \\
        --config scripts/m1_experiment/m1_config.yaml --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config(config_path: str, smoke: bool = False) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if smoke:
        smoke_cfg = config.get("smoke", {})
        config["experiment"]["seeds"] = smoke_cfg.get("seeds", [42])
        config["evaluation"]["max_eval_samples"] = smoke_cfg.get("max_eval_samples", 10)
    return config


# Map M1 task eval_task names to lm-eval-harness task names
EVAL_TASK_MAP = {
    "sql_generation": "sql_generation",  # custom task or use exact match
    "mmlu": "mmlu",
    "gsm8k": "gsm8k",
    "humaneval": "humaneval",
}


def run_lm_eval(
    base_model: str,
    adapter_dir: str | None,
    task: str,
    output_path: Path,
    max_samples: int = 500,
    device: str = "cuda",
) -> dict | None:
    """Run lm-evaluation-harness for a single (adapter, task) combo.

    Returns parsed results dict, or None on failure.
    """
    if output_path.exists():
        print(f"      [SKIP] {output_path.name} exists")
        with open(output_path) as f:
            return json.load(f)

    # Build lm_eval command
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={base_model}",
        "--tasks", task,
        "--batch_size", "auto",
        "--device", device,
        "--output_path", str(output_path.parent),
        "--log_samples",
    ]

    if adapter_dir:
        # Add PEFT adapter
        cmd[cmd.index("--model_args") + 1] += f",peft={adapter_dir}"

    if max_samples > 0:
        cmd.extend(["--limit", str(max_samples)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per eval
        )

        if result.returncode != 0:
            print(f"      [ERROR] lm_eval failed: {result.stderr[:200]}")
            # Save error info
            error_data = {
                "error": True,
                "returncode": result.returncode,
                "stderr": result.stderr[:1000],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(error_data, f, indent=2)
            return error_data

        # Parse results from lm_eval output directory
        results_dir = output_path.parent
        results_files = list(results_dir.glob("results_*.json"))
        if results_files:
            with open(results_files[-1]) as f:
                results = json.load(f)
            # Save a normalized copy
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            return results

    except subprocess.TimeoutExpired:
        print(f"      [TIMEOUT] lm_eval timed out")
        return {"error": True, "reason": "timeout"}
    except Exception as e:
        print(f"      [ERROR] {e}")
        return {"error": True, "reason": str(e)}

    return None


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 4: Evaluate adapters")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Smoke test")
    args = parser.parse_args()

    config = load_config(args.config, smoke=args.smoke)
    workspace = Path(config["runtime"]["workspace"])
    adapters_dir = workspace / "adapters"
    merges_dir = workspace / "merges"
    evals_dir = workspace / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    base_model = config["experiment"]["base_model"]
    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    methods = config["merge"]["methods"]
    max_samples = config["evaluation"]["max_eval_samples"]
    device = config["runtime"]["device"]

    total_start = time.monotonic()

    # --- Part A: Evaluate individual adapters on their own task ---
    print("Phase 4a: Evaluating individual adapters")
    individual_dir = evals_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for task_name, task_cfg in config["adapters"].items():
        eval_task = task_cfg["eval_task"]
        for seed in seeds:
            adapter_dir = adapters_dir / task_name / f"seed_{seed}"
            if not adapter_dir.exists():
                print(f"  [MISSING] {task_name}/seed_{seed}")
                continue

            output_path = individual_dir / f"{task_name}_seed_{seed}_{eval_task}.json"
            print(f"  Evaluating {task_name}/seed_{seed} on {eval_task}...")
            run_lm_eval(
                base_model=base_model,
                adapter_dir=str(adapter_dir),
                task=eval_task,
                output_path=output_path,
                max_samples=max_samples,
                device=device,
            )

    # --- Part B: Evaluate merged adapters on both constituent tasks + MMLU ---
    print("\nPhase 4b: Evaluating merged adapters")
    merged_eval_dir = evals_dir / "merged"
    merged_eval_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(itertools.combinations(task_names, 2))
    general_task = config["evaluation"]["general_capability"]

    for task_a, task_b in pairs:
        pair_name = f"{task_a}_{task_b}"
        eval_tasks_for_pair = [
            config["adapters"][task_a]["eval_task"],
            config["adapters"][task_b]["eval_task"],
            general_task,
        ]
        # Deduplicate (e.g., if one task's eval_task is already mmlu)
        eval_tasks_for_pair = list(dict.fromkeys(eval_tasks_for_pair))

        for seed in seeds:
            for method in methods:
                merge_dir = merges_dir / pair_name / f"seed_{seed}" / method
                if not merge_dir.exists():
                    continue

                for eval_task in eval_tasks_for_pair:
                    output_path = (
                        merged_eval_dir
                        / f"{pair_name}_seed_{seed}_{method}_{eval_task}.json"
                    )
                    print(f"  Evaluating {pair_name}/seed_{seed}/{method} on {eval_task}...")
                    run_lm_eval(
                        base_model=base_model,
                        adapter_dir=str(merge_dir),
                        task=eval_task,
                        output_path=output_path,
                        max_samples=max_samples,
                        device=device,
                    )

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 4 complete in {elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/m1_experiment/phase4_evaluate.py
git commit -m "feat(m1): add phase 4 evaluation script

Evaluates individual and merged adapters using lm-evaluation-harness.
Each merged adapter tested on both constituent tasks plus MMLU.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 10: Phase 5 — Analysis Script

**Files:**
- Create: `scripts/m1_experiment/phase5_analyze.py`

**Step 1: Write the correlation analysis script**

Create `scripts/m1_experiment/phase5_analyze.py`:

```python
#!/usr/bin/env python3
"""
Phase 5: Correlation analysis + report generation.

Loads all audit JSONs + eval results and computes:
  1. Pearson/Spearman correlation between spectral metrics and merge quality
  2. Linear regression: merge_quality ~ overlap + rank_ratio + scale_ratio
  3. Binary classification: predict "bad merge" (>5% degradation)
  4. Per-module-type breakdown
  5. Per-method comparison

Output: correlation_report.json + correlation_report.md

Usage:
    python scripts/m1_experiment/phase5_analyze.py \\
        --config scripts/m1_experiment/m1_config.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_audit_metrics(audits_dir: Path, pair_name: str, seed: int) -> dict | None:
    """Load aggregate spectral metrics from a merge audit."""
    audit_path = audits_dir / pair_name / f"seed_{seed}" / "merge_audit.json"
    if not audit_path.exists():
        return None
    with open(audit_path) as f:
        data = json.load(f)

    # Extract key metrics from aggregate and layer-level data
    aggregate = data.get("aggregate", {})
    layer_verdicts = data.get("layer_verdicts", [])

    # Compute mean metrics across layers
    overlaps = [lv["metrics"]["mean_overlap"] for lv in layer_verdicts if "metrics" in lv]
    dir_agreements = [
        lv["metrics"].get("directional_agreement", 0.0)
        for lv in layer_verdicts if "metrics" in lv
    ]
    mag_ratios = [
        lv["metrics"].get("magnitude_ratio", 1.0)
        for lv in layer_verdicts if "metrics" in lv
    ]
    stable_rank_ratios = [
        lv["metrics"].get("stable_rank_ratio", 1.0)
        for lv in layer_verdicts if "metrics" in lv
    ]

    def safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "pair": pair_name,
        "seed": seed,
        "overall_verdict": aggregate.get("overall_verdict", "unknown"),
        "compatibility_score": aggregate.get("compatibility_score", 0.0),
        "mean_overlap": safe_mean(overlaps),
        "mean_directional_agreement": safe_mean(dir_agreements),
        "mean_magnitude_ratio": safe_mean(mag_ratios),
        "mean_stable_rank_ratio": safe_mean(stable_rank_ratios),
        "layer_verdicts": layer_verdicts,
    }


def load_eval_result(evals_dir: Path, filename: str) -> dict | None:
    """Load an evaluation result JSON."""
    path = evals_dir / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_accuracy(eval_result: dict | None) -> float | None:
    """Extract the primary accuracy metric from an eval result."""
    if eval_result is None:
        return None
    if eval_result.get("error"):
        return None

    # lm-eval-harness result format
    results = eval_result.get("results", {})
    for task_name, task_results in results.items():
        # Try common metric keys
        for key in ["acc,none", "acc_norm,none", "exact_match,strict-match", "pass@1"]:
            if key in task_results:
                return task_results[key]

    return None


def compute_degradation(
    merged_acc: float | None,
    baseline_a_acc: float | None,
    baseline_b_acc: float | None,
) -> float | None:
    """Compute worst-case degradation from the better baseline."""
    if any(x is None for x in [merged_acc, baseline_a_acc, baseline_b_acc]):
        return None
    best_baseline = max(baseline_a_acc, baseline_b_acc)
    if best_baseline == 0:
        return None
    return (best_baseline - merged_acc) / best_baseline


def main():
    parser = argparse.ArgumentParser(description="M1 Phase 5: Correlation analysis")
    parser.add_argument("--config", required=True, help="Path to m1_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = Path(config["runtime"]["workspace"])
    audits_dir = workspace / "audits"
    evals_dir = workspace / "evals"
    analysis_dir = workspace / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    seeds = config["experiment"]["seeds"]
    task_names = list(config["adapters"].keys())
    methods = config["merge"]["methods"]

    pairs = list(itertools.combinations(task_names, 2))

    total_start = time.monotonic()
    print("Phase 5: Correlation analysis")

    # --- Collect all data points ---
    data_points: list[dict[str, Any]] = []

    for task_a, task_b in pairs:
        pair_name = f"{task_a}_{task_b}"
        eval_task_a = config["adapters"][task_a]["eval_task"]
        eval_task_b = config["adapters"][task_b]["eval_task"]

        for seed in seeds:
            audit = load_audit_metrics(audits_dir, pair_name, seed)
            if audit is None:
                continue

            # Load baseline eval results
            baseline_a_result = load_eval_result(
                evals_dir / "individual",
                f"{task_a}_seed_{seed}_{eval_task_a}.json",
            )
            baseline_b_result = load_eval_result(
                evals_dir / "individual",
                f"{task_b}_seed_{seed}_{eval_task_b}.json",
            )
            baseline_a_acc = extract_accuracy(baseline_a_result)
            baseline_b_acc = extract_accuracy(baseline_b_result)

            for method in methods:
                # Load merged eval on task A
                merged_a_result = load_eval_result(
                    evals_dir / "merged",
                    f"{pair_name}_seed_{seed}_{method}_{eval_task_a}.json",
                )
                merged_b_result = load_eval_result(
                    evals_dir / "merged",
                    f"{pair_name}_seed_{seed}_{method}_{eval_task_b}.json",
                )

                merged_a_acc = extract_accuracy(merged_a_result)
                merged_b_acc = extract_accuracy(merged_b_result)

                degradation_a = compute_degradation(merged_a_acc, baseline_a_acc, baseline_a_acc)
                degradation_b = compute_degradation(merged_b_acc, baseline_b_acc, baseline_b_acc)

                # Worst degradation across both tasks
                degradations = [d for d in [degradation_a, degradation_b] if d is not None]
                worst_degradation = max(degradations) if degradations else None

                data_points.append({
                    "pair": pair_name,
                    "seed": seed,
                    "method": method,
                    "mean_overlap": audit["mean_overlap"],
                    "mean_directional_agreement": audit["mean_directional_agreement"],
                    "mean_magnitude_ratio": audit["mean_magnitude_ratio"],
                    "mean_stable_rank_ratio": audit["mean_stable_rank_ratio"],
                    "compatibility_score": audit["compatibility_score"],
                    "merged_acc_task_a": merged_a_acc,
                    "merged_acc_task_b": merged_b_acc,
                    "baseline_acc_a": baseline_a_acc,
                    "baseline_acc_b": baseline_b_acc,
                    "degradation_a": degradation_a,
                    "degradation_b": degradation_b,
                    "worst_degradation": worst_degradation,
                    "is_bad_merge": worst_degradation > 0.05 if worst_degradation is not None else None,
                })

    print(f"  Collected {len(data_points)} data points")

    # --- Statistical analysis ---
    # Filter to valid points
    valid = [p for p in data_points if p["worst_degradation"] is not None]
    print(f"  Valid data points (with eval results): {len(valid)}")

    report: dict[str, Any] = {
        "schema_version": "gradience.m1_analysis/v1",
        "n_total_points": len(data_points),
        "n_valid_points": len(valid),
    }

    if len(valid) >= 5:
        try:
            from scipy import stats
            import numpy as np

            # Extract arrays
            overlaps = np.array([p["mean_overlap"] for p in valid])
            dir_agree = np.array([p["mean_directional_agreement"] for p in valid])
            mag_ratio = np.array([p["mean_magnitude_ratio"] for p in valid])
            rank_ratio = np.array([p["mean_stable_rank_ratio"] for p in valid])
            degradation = np.array([p["worst_degradation"] for p in valid])

            # 1. Correlation analysis
            correlations = {}
            for name, values in [
                ("mean_overlap", overlaps),
                ("directional_agreement", dir_agree),
                ("magnitude_ratio", mag_ratio),
                ("stable_rank_ratio", rank_ratio),
            ]:
                pearson_r, pearson_p = stats.pearsonr(values, degradation)
                spearman_r, spearman_p = stats.spearmanr(values, degradation)
                correlations[name] = {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }
            report["correlations"] = correlations

            # 2. Linear regression
            from sklearn.linear_model import LinearRegression

            X = np.column_stack([overlaps, rank_ratio, mag_ratio])
            y = degradation
            reg = LinearRegression().fit(X, y)
            r_squared = reg.score(X, y)
            report["linear_regression"] = {
                "r_squared": float(r_squared),
                "coefficients": {
                    "mean_overlap": float(reg.coef_[0]),
                    "stable_rank_ratio": float(reg.coef_[1]),
                    "magnitude_ratio": float(reg.coef_[2]),
                },
                "intercept": float(reg.intercept_),
            }

            # 3. Binary classification: predict bad merge
            bad_labels = np.array([p["is_bad_merge"] for p in valid], dtype=float)
            if bad_labels.sum() > 0 and bad_labels.sum() < len(bad_labels):
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, precision_score, recall_score

                clf = LogisticRegression().fit(X, bad_labels)
                predictions = clf.predict(X)
                report["binary_classification"] = {
                    "accuracy": float(accuracy_score(bad_labels, predictions)),
                    "precision": float(precision_score(bad_labels, predictions, zero_division=0)),
                    "recall": float(recall_score(bad_labels, predictions, zero_division=0)),
                    "n_bad_merges": int(bad_labels.sum()),
                    "n_good_merges": int(len(bad_labels) - bad_labels.sum()),
                }
            else:
                report["binary_classification"] = {
                    "note": "All merges same class — cannot fit classifier",
                    "n_bad_merges": int(bad_labels.sum()),
                    "n_good_merges": int(len(bad_labels) - bad_labels.sum()),
                }

            # 4. Per-method comparison
            method_stats = {}
            for method in methods:
                method_points = [p for p in valid if p["method"] == method]
                if method_points:
                    degs = [p["worst_degradation"] for p in method_points]
                    n_bad = sum(1 for d in degs if d > 0.05)
                    method_stats[method] = {
                        "n_merges": len(method_points),
                        "mean_degradation": float(np.mean(degs)),
                        "std_degradation": float(np.std(degs, ddof=1)) if len(degs) > 1 else 0.0,
                        "max_degradation": float(max(degs)),
                        "n_bad_merges": n_bad,
                        "bad_merge_rate": n_bad / len(method_points),
                    }
            report["per_method"] = method_stats

        except ImportError as e:
            report["error"] = f"Missing dependency: {e}. Install scipy and scikit-learn."
    else:
        report["error"] = f"Insufficient valid data points ({len(valid)}) for analysis."

    # --- Success criteria evaluation ---
    report["success_criteria"] = {}
    if "linear_regression" in report:
        r2 = report["linear_regression"]["r_squared"]
        report["success_criteria"]["variance_explained"] = {
            "value": r2,
            "threshold": 0.50,
            "met": r2 >= 0.50,
        }
    if "binary_classification" in report and "recall" in report["binary_classification"]:
        recall = report["binary_classification"]["recall"]
        report["success_criteria"]["bad_merge_detection"] = {
            "value": recall,
            "threshold": 0.80,
            "met": recall >= 0.80,
        }

    # Save data points
    report["data_points"] = data_points

    # --- Write outputs ---
    report_json_path = analysis_dir / "correlation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_json_path}")

    # Generate markdown report
    md = _generate_markdown(report, config)
    report_md_path = analysis_dir / "correlation_report.md"
    with open(report_md_path, "w") as f:
        f.write(md)
    print(f"  Saved: {report_md_path}")

    elapsed = time.monotonic() - total_start
    print(f"\nPhase 5 complete in {elapsed:.1f}s")

    # Print summary
    if "success_criteria" in report:
        print("\n--- Success Criteria ---")
        for name, criterion in report["success_criteria"].items():
            status = "PASS" if criterion["met"] else "FAIL"
            print(f"  {name}: {criterion['value']:.3f} vs {criterion['threshold']:.2f} [{status}]")


def _generate_markdown(report: dict, config: dict) -> str:
    """Generate human-readable markdown report."""
    lines = [
        f"# M1 Controlled Interference Experiment — Results",
        "",
        f"**Experiment**: {config['experiment']['name']}",
        f"**Base Model**: {config['experiment']['base_model']}",
        f"**Data Points**: {report['n_valid_points']} valid / {report['n_total_points']} total",
        "",
    ]

    if "correlations" in report:
        lines.extend([
            "## Correlation Analysis",
            "",
            "| Metric | Pearson r | p-value | Spearman r | p-value |",
            "|--------|-----------|---------|------------|---------|",
        ])
        for name, corr in report["correlations"].items():
            lines.append(
                f"| {name} | {corr['pearson_r']:.3f} | {corr['pearson_p']:.4f} "
                f"| {corr['spearman_r']:.3f} | {corr['spearman_p']:.4f} |"
            )
        lines.append("")

    if "linear_regression" in report:
        reg = report["linear_regression"]
        lines.extend([
            "## Linear Regression",
            "",
            f"**R-squared**: {reg['r_squared']:.3f}",
            "",
            "| Feature | Coefficient |",
            "|---------|-------------|",
        ])
        for name, coef in reg["coefficients"].items():
            lines.append(f"| {name} | {coef:.4f} |")
        lines.append(f"| intercept | {reg['intercept']:.4f} |")
        lines.append("")

    if "binary_classification" in report:
        bc = report["binary_classification"]
        if "accuracy" in bc:
            lines.extend([
                "## Bad Merge Detection",
                "",
                f"- **Accuracy**: {bc['accuracy']:.3f}",
                f"- **Precision**: {bc['precision']:.3f}",
                f"- **Recall**: {bc['recall']:.3f}",
                f"- Bad merges: {bc['n_bad_merges']} / {bc['n_bad_merges'] + bc['n_good_merges']}",
                "",
            ])

    if "per_method" in report:
        lines.extend([
            "## Per-Method Comparison",
            "",
            "| Method | Mean Deg. | Std | Max Deg. | Bad Merges |",
            "|--------|-----------|-----|----------|------------|",
        ])
        for method, stats in report["per_method"].items():
            lines.append(
                f"| {method} | {stats['mean_degradation']:.3f} "
                f"| {stats['std_degradation']:.3f} "
                f"| {stats['max_degradation']:.3f} "
                f"| {stats['n_bad_merges']}/{stats['n_merges']} |"
            )
        lines.append("")

    if "success_criteria" in report:
        lines.extend([
            "## Success Criteria",
            "",
        ])
        for name, criterion in report["success_criteria"].items():
            status = "PASS" if criterion["met"] else "FAIL"
            lines.append(
                f"- **{name}**: {criterion['value']:.3f} "
                f"(threshold: {criterion['threshold']:.2f}) — **{status}**"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/m1_experiment/phase5_analyze.py
git commit -m "feat(m1): add phase 5 correlation analysis script

Computes Pearson/Spearman correlations, linear regression (R²),
and binary classification for bad merge detection. Generates
both JSON and markdown reports with success criteria evaluation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 11: Shell Orchestrator

**Files:**
- Create: `scripts/m1_experiment/run_all.sh`

**Step 1: Write the master orchestrator**

Create `scripts/m1_experiment/run_all.sh`:

```bash
#!/usr/bin/env bash
#
# M1 Controlled Interference Experiment — Master Orchestrator
#
# Runs all 5 phases sequentially. Each phase is independently runnable
# and resumable (skips completed work).
#
# Usage:
#   bash scripts/m1_experiment/run_all.sh [--smoke]
#
# Environment:
#   Expects to run on RunPod with CUDA available.
#   Install: pip install "gradience[bench]" lm-eval scipy scikit-learn

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/m1_config.yaml"

# Pass --smoke flag through
EXTRA_ARGS="${*}"

echo "=================================================================="
echo "  M1 Controlled Interference Experiment"
echo "  Config: ${CONFIG}"
echo "  Args: ${EXTRA_ARGS:-none}"
echo "  Started: $(date)"
echo "=================================================================="

echo ""
echo "--- Phase 1: Train Adapters ---"
python "${SCRIPT_DIR}/phase1_train.py" --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "--- Phase 2: Pairwise Merge-Audit ---"
python "${SCRIPT_DIR}/phase2_audit.py" --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "--- Phase 3: Execute Merges ---"
python "${SCRIPT_DIR}/phase3_merge.py" --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "--- Phase 4: Evaluate ---"
python "${SCRIPT_DIR}/phase4_evaluate.py" --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "--- Phase 5: Analyze ---"
python "${SCRIPT_DIR}/phase5_analyze.py" --config "${CONFIG}"

echo ""
echo "=================================================================="
echo "  M1 experiment complete!"
echo "  Results: /workspace/m1/analysis/"
echo "  Finished: $(date)"
echo "=================================================================="
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/m1_experiment/run_all.sh
git add scripts/m1_experiment/run_all.sh
git commit -m "feat(m1): add master shell orchestrator

Runs all 5 phases sequentially with --smoke passthrough.
Each phase is independently runnable and resumable.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 12: Final Verification

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (no regressions from DARE additions)

**Step 2: Verify DARE strategy roundtrip with merge engine**

Run: `python3 -c "from gradience.vnext.merge import get_strategy, plan_from_audit, PLAN_STRATEGIES; print('Strategies:', sorted(get_strategy.__module__)); print('Plan strategies:', sorted(PLAN_STRATEGIES.keys()))"`

Expected output includes `dare_linear` and `dare_ties` in both registries.

**Step 3: Verify all M1 scripts parse without errors**

```bash
python3 -c "import scripts.m1_experiment.task_configs" 2>/dev/null || python3 scripts/m1_experiment/task_configs.py --help 2>&1 | head -1 || echo "OK: module loaded"
python3 scripts/m1_experiment/phase1_train.py --help
python3 scripts/m1_experiment/phase2_audit.py --help
python3 scripts/m1_experiment/phase3_merge.py --help
python3 scripts/m1_experiment/phase4_evaluate.py --help
python3 scripts/m1_experiment/phase5_analyze.py --help
```

Expected: Each prints help text without import errors.

**Step 4: Verify file structure**

```bash
find scripts/m1_experiment/ -type f | sort
```

Expected:
```
scripts/m1_experiment/m1_config.yaml
scripts/m1_experiment/phase1_train.py
scripts/m1_experiment/phase2_audit.py
scripts/m1_experiment/phase3_merge.py
scripts/m1_experiment/phase4_evaluate.py
scripts/m1_experiment/phase5_analyze.py
scripts/m1_experiment/run_all.sh
scripts/m1_experiment/task_configs.py
```

**Step 5: Final commit (if any remaining changes)**

```bash
git status
# If clean, nothing to commit.
# If there are uncommitted changes from verification fixes, commit them.
```

---

## Summary

| Task | What | Files | Estimated Time |
|------|------|-------|---------------|
| 1 | DARE strategy tests | `tests/merge/test_strategies.py` | 5 min |
| 2 | DARE strategy implementation | `strategies.py`, `__init__.py` | 10 min |
| 3 | DARE plan strategies | `plan.py`, `test_dare_plan.py` | 10 min |
| 4 | M1 config YAML | `m1_config.yaml` | 3 min |
| 5 | Task configs module | `task_configs.py` | 5 min |
| 6 | Phase 1: Training | `phase1_train.py` | 10 min |
| 7 | Phase 2: Audit | `phase2_audit.py` | 5 min |
| 8 | Phase 3: Merge | `phase3_merge.py` | 10 min |
| 9 | Phase 4: Evaluate | `phase4_evaluate.py` | 10 min |
| 10 | Phase 5: Analyze | `phase5_analyze.py` | 10 min |
| 11 | Shell orchestrator | `run_all.sh` | 3 min |
| 12 | Final verification | — | 5 min |

**Total estimated time: ~85 minutes**
