# Rank Suggestion API

::: gradience.vnext.rank_suggestion

Pure functions that convert audit results into conservative rank compression suggestions.

## Design

- **No side effects** — No file I/O, no logging, no state mutation
- **Conservative** — Rounds up to standard LoRA rank buckets, never suggests increasing rank
- **Minimal dependencies** — Works with plain dicts (no torch required)

## Constants

### `DEFAULT_ALLOWED_RANKS`

```python
DEFAULT_ALLOWED_RANKS = (1, 2, 4, 8, 16, 32, 64)
```

Standard LoRA rank buckets. Suggestions are rounded up to the nearest value in this tuple.

## Functions

### `suggest_global_ranks_from_audit()`

Derive conservative global rank suggestions from audit summary statistics.

```python
def suggest_global_ranks_from_audit(
    audit: dict[str, Any],
    *,
    allowed_ranks: Sequence[int] = DEFAULT_ALLOWED_RANKS,
) -> GlobalRankSuggestion
```

**Input dict keys** (from `gradience audit --json`):

| Key | Type | Required |
|-----|------|----------|
| `total_lora_params` | `int` | For savings math (defaults to 0) |
| `current_r` or `r` | `int` | Yes (or inferred from stable_rank_mean + utilization_mean) |
| `energy_rank_90_p50` | `float` | For median suggestion |
| `energy_rank_90_p90` | `float` | For conservative (p90) suggestion |

**Behavior:**

1. Prefers explicit suggested ranks if the audit already provides them
2. Otherwise derives from `energy_rank_90_p50`/`p90` and buckets to allowed ranks
3. Infers `current_r` from `stable_rank_mean / utilization_mean` if not explicit
4. Never suggests increasing rank above `current_r`
5. Estimates parameter savings using linear scaling

**Example:**

```python
from gradience.vnext.rank_suggestion import suggest_global_ranks_from_audit

audit = {
    "r": 16,
    "total_lora_params": 1_000_000,
    "energy_rank_90_p50": 3.2,
    "energy_rank_90_p90": 6.8,
    "stable_rank_mean": 4.1,
    "utilization_mean": 0.26,
}

suggestion = suggest_global_ranks_from_audit(audit)
print(f"Median suggestion: r={suggestion.suggested_r_median}")
print(f"Conservative (p90): r={suggestion.suggested_r_p90}")
print(f"Param reduction (median): {suggestion.reduction_ratio_median:.0%}")
```

---

### `suggest_per_layer_ranks()`

Derive per-layer rank suggestions from audit data with layer detail.

```python
def suggest_per_layer_ranks(
    audit: dict[str, Any],
    *,
    margin: float = 1.0,
    allowed_ranks: Sequence[int] = DEFAULT_ALLOWED_RANKS,
) -> PerLayerRankSuggestionReport
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audit` | — | Audit dict with `layers` list (from `gradience audit --layers --json`) |
| `margin` | `1.0` | Multiplicative headroom (1.0 = no extra margin) |
| `allowed_ranks` | `DEFAULT_ALLOWED_RANKS` | Rank buckets to round up to |

**Returns:** `PerLayerRankSuggestionReport` with suggested `default_r`, per-layer `rank_pattern` overrides, and module-type-level aggregation.

## Data classes

### `GlobalRankSuggestion`

```python
@dataclass(frozen=True)
class GlobalRankSuggestion:
    current_r: int                  # Current LoRA rank
    suggested_r_median: int         # Median-based suggestion
    suggested_r_p90: int            # Conservative (p90) suggestion
    total_lora_params: int          # Current total LoRA parameters
    params_at_r_median: int         # Estimated params at median rank
    params_at_r_p90: int            # Estimated params at p90 rank
    reduction_ratio_median: float   # Parameter reduction fraction (median)
    reduction_ratio_p90: float      # Parameter reduction fraction (p90)
    evidence: dict[str, Any]        # Input data used for the suggestion
```

### `PerLayerRankSuggestion`

```python
@dataclass(frozen=True)
class PerLayerRankSuggestion:
    name: str                       # Layer name
    current_r: int                  # Current rank for this layer
    energy_rank_90: float           # Energy rank at 90%
    suggested_r: int                # Suggested rank (bucketed)
    reduction_ratio: float          # Per-layer reduction fraction
    stable_rank: float | None       # Stable rank (if available)
    utilization: float | None       # Utilization ratio (if available)
    module_type: str | None         # Module type (q_proj, v_proj, etc.)
```

### `PerLayerRankSuggestionReport`

```python
@dataclass(frozen=True)
class PerLayerRankSuggestionReport:
    layers: tuple[PerLayerRankSuggestion, ...]
    default_r: int                     # Mode of suggested ranks
    rank_pattern: dict[str, int]       # Overrides where suggested != default_r
    by_module_type_p90: dict[str, int] # Per-module-type p90 ranks
    notes: str                         # Summary string
```
