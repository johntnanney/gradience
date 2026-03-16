# Merge Audit API

::: gradience.vnext.merge

Geometric compatibility analysis and execution for merging LoRA adapter pairs.

## Quick usage

```python
from gradience.vnext.merge import merge_audit, plan_from_audit, execute_merge

# Phase 1: Audit compatibility
report = merge_audit("./adapter_a", "./adapter_b")
print(f"Verdict: {report.verdict}")  # safe, redundant, or conflicting

# Phase 2: Plan and execute merge
plan = plan_from_audit("audit_aware", report, "./adapter_a", "./adapter_b")
result = execute_merge(plan, "./merged_adapter")
print(f"Reconstruction error: {result.mean_reconstruction_error:.4f}")
```

## Phase 1: Analysis

### `merge_audit()`

Main entry point for spectral compatibility analysis.

```python
def merge_audit(
    adapter_a: str | Path,
    adapter_b: str | Path,
    *,
    thresholds: VerdictThresholds | None = None,
    device: str = "cpu",
    output_dir: str | Path | None = None,
) -> MergeAuditReport
```

| Parameter | Description |
|-----------|-------------|
| `adapter_a` | Path to first PEFT adapter |
| `adapter_b` | Path to second PEFT adapter |
| `thresholds` | Custom verdict thresholds (default: conservative defaults) |
| `device` | Torch device for computation |
| `output_dir` | If set, writes `merge_audit.json` and `merge_audit.md` |

**Returns:** `MergeAuditReport`

## Phase 2: Execution

### `plan_from_audit()`

Generate a merge plan from an audit report.

```python
def plan_from_audit(
    strategy: str,
    report: MergeAuditReport,
    adapter_a: str | Path,
    adapter_b: str | Path,
    *,
    alpha: float = 0.5,
) -> MergePlan
```

**Available strategies:** `"linear"`, `"ties"`, `"dare_linear"`, `"dare_ties"`, `"norm_equalized"`, `"audit_aware"`

### `execute_merge()`

Execute a merge plan to produce a merged adapter.

```python
def execute_merge(
    plan: MergePlan,
    output_dir: str | Path,
) -> MergeResult
```

**Returns:** `MergeResult` with per-layer reconstruction errors.

## Data classes

### `MergeAuditReport`

| Attribute | Type | Description |
|-----------|------|-------------|
| `verdict` | `CompatibilityVerdict` | Overall compatibility verdict |
| `layer_verdicts` | `list[LayerVerdict]` | Per-layer assessments |
| `summary` | `dict` | Aggregate statistics |

### `CompatibilityVerdict`

Enum with values:

- `"safe"` — Subspaces are compatible, merge should preserve quality
- `"redundant"` — Subspaces overlap significantly (merging adds little)
- `"conflicting"` — Subspaces interfere (merging may degrade quality)

### `VerdictThresholds`

Customize the thresholds for verdict classification:

```python
thresholds = VerdictThresholds(
    safe_min_angle=30.0,        # Minimum principal angle for "safe"
    redundant_max_angle=10.0,   # Maximum principal angle for "redundant"
    conflict_min_disagreement=0.7,  # Minimum directional disagreement for "conflicting"
)
report = merge_audit(a, b, thresholds=thresholds)
```

### `LayerVerdict`

Per-layer compatibility assessment with metrics.

### Merge strategies

| Strategy | Description |
|----------|-------------|
| `LinearMerge` | Weighted linear combination: `α·A + (1-α)·B` |
| `TIESMerge` | TIES: trim, elect sign, disjoint merge |
| `DARELinearMerge` | DARE with linear combination |
| `DARETIESMerge` | DARE with TIES |
| `NormEqualizedMerge` | Norm-equalized merge preserving spectral properties |

### `MergeResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `output_dir` | `Path` | Directory containing merged adapter |
| `layer_results` | `list[LayerMergeResult]` | Per-layer merge details |
| `mean_reconstruction_error` | `float` | Average reconstruction error across layers |
