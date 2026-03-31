# Routing Pilot Spec

**Status:** VALIDATED — full success (2026-03-29)
**Date:** 2026-03-29
**Validates:** Architecture assessment §5 (substrate generality thesis)
**Scope:** One non-merge scenario consuming existing Zone A + Zone B modules

---

## Goal

Write a thin routing-triage layer that consumes the same spectral analysis Gradience uses for merge preflight, but interprets it for an adapter-routing context. If this works without modifying any existing module, the architecture assessment's generality claim is confirmed.

---

## Inputs

1. **3–5 PEFT adapter directories.** Same format as merge audit inputs — each contains `adapter_config.json` + `adapter_model.safetensors`. Reuse adapters from existing field trials where possible (the same-family confirmation set has 4 suitable adapters: two SST-2, one IMDB, one AG News).

2. **QA artifacts (optional).** If available, `AdapterQAArtifact` JSON files from prior `gradience audit` runs. These provide eligibility status and behavioral evidence. The pilot should work with or without them.

---

## Reused Modules (no modifications)

| Module | What the pilot uses | Import path |
|--------|-------------------|-------------|
| `vnext/merge/io.py` | `load_adapter`, `match_layers`, `extract_factors` | `gradience.vnext.merge` |
| `vnext/merge/spectral_compat.py` | `compute_subspace_metrics` → `SubspaceMetrics` | `gradience.vnext.merge` |
| `vnext/merge/verdicts.py` | `assess_layer` → `LayerVerdict`, `VerdictThresholds` | `gradience.vnext.merge` |
| `vnext/audit/lora_audit.py` | `audit_lora_peft_dir` → `LoRAAuditResult` | `gradience.vnext.audit.lora_audit` |
| `vnext/audit/qa_artifact.py` | `build_qa_artifact` → `AdapterQAArtifact` | `gradience.vnext.audit.qa_artifact` |

The pilot imports these and calls them exactly as merge does. Zero changes to their source.

---

## New Thin Modules

### `routing_compat.py` (~100–150 lines)

Routing-specific interpretation of `SubspaceMetrics` and `LayerVerdict`.

```python
# Core type
@dataclass(frozen=True)
class RoutingLayerAssessment:
    layer_name: str
    confusability: str          # "high" | "moderate" | "low"
    confusability_score: float  # 0–1, higher = more confusable
    notes: list[str]

# Core type
@dataclass(frozen=True)
class RoutingPairAssessment:
    adapter_a: str
    adapter_b: str
    overall_confusability: str  # "high" | "moderate" | "low"
    confusability_score: float
    layer_assessments: tuple[RoutingLayerAssessment, ...]
    recommendation: str         # "easily_routed" | "needs_disambiguation" | "consider_dedup"

# Functions
def assess_routing_layer(
    layer_name: str,
    metrics: SubspaceMetrics,
) -> RoutingLayerAssessment:
    """Interpret SubspaceMetrics for routing.

    Key inversion from merge verdicts:
    - High overlap + aligned → "high confusability" (merge calls this "redundant")
    - High overlap + opposing → "moderate confusability" (merge calls this "conflicting")
    - Low overlap → "low confusability" (merge calls this "safe")

    Magnitude imbalance is less relevant for routing than for merge — a
    router doesn't need the adapters to be scale-balanced, only separable.
    """

def assess_routing_pair(
    adapter_a_name: str,
    adapter_b_name: str,
    layer_assessments: list[RoutingLayerAssessment],
) -> RoutingPairAssessment:
    """Aggregate layer assessments into a pair-level routing verdict."""
```

### `routing_report.py` (~80–120 lines)

Formats `RoutingPairAssessment` objects into a human-readable fleet-level report.

```python
@dataclass(frozen=True)
class RoutingFleetReport:
    adapters: tuple[str, ...]
    pair_assessments: tuple[RoutingPairAssessment, ...]
    confusable_pairs: tuple[str, ...]     # pairs with high confusability
    clean_pairs: tuple[str, ...]          # pairs with low confusability
    summary_line: str

def build_routing_report(
    assessments: list[RoutingPairAssessment],
) -> RoutingFleetReport:
    """Aggregate all pair assessments into a fleet-level routing triage."""

def format_routing_report(report: RoutingFleetReport) -> str:
    """Terminal-friendly formatted output."""
```

### `run_routing_pilot.py` (~100–150 lines)

End-to-end script. Loads adapters, computes all-pairs metrics, produces routing report.

```python
def main():
    adapter_dirs = [...]  # 3-5 PEFT dirs
    # For each pair:
    #   load_adapter → match_layers → extract_factors → compute_subspace_metrics
    #   assess_routing_layer per matched layer
    #   assess_routing_pair
    # build_routing_report from all pair assessments
    # format_routing_report → stdout + JSON
```

**Total new code: ~300–400 lines across three files.** No framework, no CLI integration, no schema versioning. This is a trial, not a product.

---

## Expected Outputs

1. **`routing_report.json`** — Machine-readable fleet report with per-pair confusability scores and recommendations.

2. **`routing_report.txt`** — Human-readable terminal output showing which adapter pairs are easily routed vs. confusable.

3. **`routing_pilot_field_note.md`** — Documents what happened: which adapters, what the pilot found, whether Zone B code was consumed unmodified, any friction points.

---

## Success Criteria

### Full success

All three conditions hold:

- Zero modifications to any existing Gradience module. The pilot imports and calls existing functions exactly as documented.
- The routing interpretation produces meaningfully different verdicts from merge interpretation on the same adapter pairs. (Specifically: at least one pair where merge says "redundant/safe" but routing says "high confusability," confirming the policy-layer inversion works.)
- The pilot script runs end-to-end on CPU with the existing field-trial adapters.

### Partial success

The pilot runs but requires workarounds:

- Existing functions need wrapper logic to handle input format mismatches (e.g., the pilot can't call `assess_layer` directly because the function signature assumes merge-specific context). This would indicate Zone B is *mostly* general but has implicit merge assumptions.
- The confusability scores are mathematically correct but don't produce useful routing guidance for the specific adapters tested (e.g., all pairs are either trivially separable or trivially confusable). This validates the substrate but doesn't validate the routing scenario as a real product.

### Failure

Any of these:

- The pilot requires modifying `spectral_compat.py`, `verdicts.py`, or `io.py` to produce correct results. This would mean Zone B has hard merge dependencies that the architecture assessment missed.
- The `SubspaceMetrics` fields don't carry enough information for routing-specific interpretation — the pilot needs to compute additional spectral quantities not captured by the existing data structure. This would mean the measurement layer is merge-shaped, not math-shaped.
- The pilot can't reuse `load_adapter` / `extract_factors` because these functions make merge-specific assumptions about adapter format or structure.

---

## What Will Not Be Refactored Beforehand

- **No import moves.** `EligibilityStatus` stays in `merge/eligibility.py`. The pilot imports it from there. Mechanical cleanup happens after the pilot, if warranted.
- **No exception renames.** `spectral_compat.py` continues to raise `MergeError`. The pilot catches it as-is.
- **No vocabulary changes.** `InventoryActionPlan` keeps its merge-specific bucket names. The pilot does not use the inventory layer — it builds its own report from pair assessments directly.
- **No abstract base classes.** No `TriagePolicy` protocol. The pilot writes concrete functions. Abstraction comes after the second concrete scenario, not before it.

---

## File Locations

```
experiments/routing_pilot/
├── routing_compat.py           # New: routing-specific interpretation
├── routing_report.py           # New: fleet-level report builder
├── run_routing_pilot.py        # New: end-to-end script
├── routing_report.json         # Output: machine-readable report
├── routing_report.txt          # Output: terminal-formatted report
└── routing_pilot_field_note.md # Output: what happened
```

All new files live in `experiments/`, not in the main package. This is a trial that earns its way into the package, not a feature that assumes it belongs there.

---

## Estimated Effort

~300–400 lines of new code. No dependencies beyond what Gradience already requires (torch, safetensors). CPU-only. The field-trial adapters from `targeted_confirmation_same_family` provide the test inputs — no new adapter downloads needed.

The pilot answers one question: does the diagnosis/policy boundary in the architecture assessment hold under a genuinely different use case? Everything else is commentary.
