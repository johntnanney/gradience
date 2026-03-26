# Merge Risk Report

## 1. What It Is

A Gradience merge risk report is the canonical record of a pairwise adapter comparison: structural compatibility, eligibility status of both sources, risk level, and recommended merge action. It is a decision-bearing object: downstream scripts and workflows consume it and change behavior based on its contents.

Schema identifier: `gradience.merge_qa_report/v1`

## 2. How to Produce It

### CLI

```bash
gradience merge-audit \
  --adapter-a ./adapters/adapter-a \
  --adapter-b ./adapters/adapter-b \
  --source-a-qa qa_a.json \
  --source-b-qa qa_b.json \
  --qa-report \
  --emit-report report.json
```

`--qa-report` prints the 4-section terminal format. `--emit-report <path>` writes the v1 JSON to a file. Both can be used together.

Source QA arguments are optional. Without them, `eligibility_status` will be `null` for the corresponding adapter.

Optional core-space diagnostic:

```bash
gradience merge-audit \
  --adapter-a ./adapters/adapter-a \
  --adapter-b ./adapters/adapter-b \
  --compute-core-space \
  --emit-report report_with_core_space.json
```

`--compute-core-space` adds an optional `core_space` block to the emitted `MergeQAReport`.

Current status: core-space is documented as an **advanced optional diagnostic**. It remains additive and does not change default merge recommendation behavior.

### Python API

```python
from gradience.api import compute_core_space_diagnostic, merge_risk_report

report = merge_risk_report(
    adapter_a="./adapters/adapter-a",
    adapter_b="./adapters/adapter-b",
    source_a_qa="qa_a.json",
    source_b_qa="qa_b.json",
)

# Optional advanced extension: include core_space in the report
report_with_core_space = merge_risk_report(
    adapter_a="./adapters/adapter-a",
    adapter_b="./adapters/adapter-b",
    compute_core_space=True,
)

# Optional advanced helper: return only the core_space diagnostic block
core_space = compute_core_space_diagnostic(
    adapter_a="./adapters/adapter-a",
    adapter_b="./adapters/adapter-b",
)

# Serialize
import json
with open("report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

`merge_risk_report()` is a thin wrapper over the CLI -- it runs `merge-audit --qa-report --emit-report` via subprocess, then loads the resulting JSON through `MergeQAReport.from_dict()`. It is not an alternate implementation path.

`compute_core_space_diagnostic()` is an advanced optional helper that enables core-space and returns the report's `core_space` block directly. It remains diagnostic-only.

## 3. How to Read It

A v1 report has these top-level sections:

- **`adapter_a`, `adapter_b`** -- identity and status of each adapter: path, rank, alpha, layer count, base model, and eligibility status.
- **`pair_risk`** -- structural risk level for the pair: `"low"`, `"medium"`, or `"high"`. Derived from structural analysis only; eligibility never affects it.
- **`dominant_issue`** -- machine-readable label identifying the single biggest structural concern. `dominant_issue_detail` provides a human-readable explanation.
- **`recommended_strategy`** -- the primary machine-readable merge strategy recommendation, derived from pair risk and compression needs.
- **`recommended_action`** -- explanatory prose describing what to do. Does not override `recommended_strategy`.
- **`confidence`** -- categorical confidence in the recommendation: `"high"`, `"medium"`, or `"low"`. `confidence_note` provides a prose companion.
- **`caveats`** -- list of things the user should know before proceeding (eligibility warnings, structural concerns, compression advice).
- **`verdict_distribution`** -- layer verdict counts: how many layers were classified as safe, redundant, conflicting, or imbalanced.
- **`compatibility_score`** -- numeric score from 0 to 1. Higher means more compatible. Derived from the layer verdict distribution.
- **`core_space`** *(optional)* -- shared-basis diagnostic summary (`shared_basis_score`, `basis_distortion`, `effective_shared_rank`, `status`) when `--compute-core-space` is enabled.
- **`task_relationship_advisory`** *(optional)* -- present when source QA artifacts indicate the adapters were evaluated on different tasks. Part of the stable interpretive layer. Across 132+ checked pairs on two backbones, the advisory has 0% false positive rate on same-task pairs and 100% correct fire rate on different-task pairs. Most valuable for inventory-level partitioning of mixed-task pools — in observation testing, it collapsed 11 medium-risk candidates to 2 actionable pairs. Does not alter structural risk classification or recommendation logic. Note: in same-task/different-domain regimes with high cross-domain transfer (e.g., sentiment across review domains), the advisory may overcall — flagging merges that are actually safe. Treat as "worth checking" rather than "likely degraded" in such cases.

## 4. How to Consume It

### Loading in Python

```python
import json
from gradience import MergeQAReport

with open("report.json") as f:
    report = MergeQAReport.from_dict(json.load(f))

if report.pair_risk == "high":
    print("High risk -- validate after merge")

if report.recommended_strategy == "linear":
    print("Simple linear merge is appropriate")
```

### `--strict-qa` behavior

`--strict-qa` requires behavioral evidence for both adapters. Without it, eligibility generates caveats but does not block.

| Adapter A | Adapter B | Behavior |
|-----------|-----------|----------|
| `eligible` | `eligible` | Allow |
| `eligible` | `uncertain` | Allow with warning |
| `eligible` | `null` (no QA) | Block |
| `eligible` | `flagged_weak` | Block |
| `null` | `null` | Block |
| any | `unknown_no_behavioral_eval` | Block |

`null` (no QA artifact provided) and `unknown_no_behavioral_eval` (QA artifact has no behavioral data) are both blocked under `--strict-qa`.

## 5. Schema Contract

### Required fields

| Path | Type | Notes |
|------|------|-------|
| `schema` | `str` | Must be `"gradience.merge_qa_report/v1"` |
| `adapter_a` | `dict` | Adapter summary (see below) |
| `adapter_b` | `dict` | Adapter summary (see below) |
| `adapter_*.path` | `str` | Path to adapter directory |
| `adapter_*.rank` | `int` | Nominal LoRA rank (accepts numeric, normalized to int) |
| `pair_risk` | `str` | One of: `"low"`, `"medium"`, `"high"` |
| `dominant_issue` | `str` | One of the frozen labels (see Decision Semantics) |
| `recommended_strategy` | `str` | Required but not restricted to frozen vocabulary (forward compatible) |
| `confidence` | `str` | One of: `"high"`, `"medium"`, `"low"` |
| `compatibility_score` | `float` | Range 0-1, higher = more compatible (accepts numeric, normalized to float) |

### Optional fields

| Path | Type | Default | Notes |
|------|------|---------|-------|
| `adapter_*.alpha` | `float` | `0.0` | Accepts numeric, normalized to float |
| `adapter_*.n_layers` | `int` | `0` | Number of adapter layers |
| `adapter_*.base_model` | `str` | `""` | Base model identifier |
| `adapter_*.eligibility_status` | `str\|null` | `null` | One of the four `EligibilityStatus` values, or `null` when no QA artifact was provided |
| `dominant_issue_detail` | `str` | `""` | Human-readable explanation of the dominant issue |
| `recommended_action` | `str` | `""` | Explanatory prose (does not override `recommended_strategy`) |
| `confidence_note` | `str` | `""` | Prose companion to `confidence` |
| `caveats` | `list[str]` | `[]` | Warnings and advisories |
| `verdict_distribution` | `dict[str, int]` | `{}` | Layer verdict counts (values must be integers) |
| `core_space` | `dict` | omitted | Optional shared-basis diagnostic block (present only when computed) |
| `task_relationship_advisory` | `str` | omitted | Advisory when adapters were evaluated on different tasks (present only when applicable) |

Extra keys at any level are silently ignored (forward compatible).

### Validation rules

- Missing or wrong `schema` raises `QASchemaError`.
- Unknown `eligibility_status` values (non-null, not a valid `EligibilityStatus`) raise `QASchemaError`.
- Unknown `pair_risk`, `dominant_issue`, or `confidence` values raise `QASchemaError`.
- `recommended_strategy` accepts any string (lenient for forward compatibility).
- `caveats` must be `list[str]` if present.
- `verdict_distribution` values must be integers if present.
- If `core_space` is present, it must include numeric `shared_basis_score`, numeric `basis_distortion`, integer `effective_shared_rank`, and status in `{compatible, marginal, incompatible, not_applicable}`.
- Numeric fields accept `int` or `float`, normalized to the declared type.
- If `task_relationship_advisory` is present, it must be a string.

## 6. Decision Semantics

### Eligibility status

| Value | Meaning |
|-------|---------|
| `eligible` | Adapter outperforms base on provided eval |
| `uncertain` | Performance within margin of base |
| `flagged_weak` | Adapter underperforms base on provided eval |
| `unknown_no_behavioral_eval` | QA artifact exists but contains no behavioral data |
| `null` | No QA artifact was provided for this adapter |

`null` and `unknown_no_behavioral_eval` are different states. `null` means absence of data; `unknown_no_behavioral_eval` means the data was provided but lacked behavioral evidence.

### Dominant issue labels

| Label | When |
|-------|------|
| `norm_imbalance` | Imbalanced layers with high magnitude ratio |
| `subspace_conflict` | Conflicting layers dominate |
| `high_redundancy` | Redundant layers outnumber safe layers |
| `partial_redundancy` | Some redundant layers, but safe layers dominate |
| `none` | Adapters are spectrally compatible |
| `unknown` | No layer data available |

### Recommended strategy derivation

`recommended_strategy` is the primary machine-readable recommendation. It is derived from `pair_risk` and whether compression is needed (any layer has `compress_first=True`).

| `pair_risk` | Compression needed | `recommended_strategy` |
|-------------|-------------------|----------------------|
| `low` | no | `"linear"` |
| `low` | yes | `"audit_aware"` |
| `medium` | no | `"norm_equalized"` |
| `medium` | yes | `"audit_aware"` |
| `high` | any | `"audit_aware"` |

### Confidence

| Level | When |
|-------|------|
| `high` | Both adapters eligible, low structural risk, compatibility score >= 0.8 |
| `medium` | Behavioral evidence exists but incomplete, or moderate structural risk |
| `low` | No behavioral evidence for either adapter, or high structural risk |

### Pair risk

| Value | Meaning |
|-------|---------|
| `low` | Adapters are structurally compatible for merging |
| `medium` | Some structural concerns; strategy matters |
| `high` | Significant structural risk; validate after merge |

`pair_risk` is derived from structural analysis (layer verdicts, magnitude ratios). Eligibility status never affects `pair_risk`. Eligibility affects `caveats`, `recommended_action`, and `--strict-qa` behavior.

## 7. Versioning Policy

The schema identifier `gradience.merge_qa_report/v1` is frozen.

- New fields may be added without a version bump.
- No existing field will be renamed, removed, or have its semantics changed.
- A future version that changes the contract must use a new schema identifier (e.g., `gradience.merge_qa_report/v2`).
