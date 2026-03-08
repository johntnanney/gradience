# Day 2 Plan: Make Single-Adapter QA Real

**Date:** 2026-03-08
**Goal:** A user can run `audit-adapter`, get a stable QA artifact, and feed it into `merge-audit` in a way that changes warnings and recommendation behavior.

**Success criteria** — three things must be true by end of day:

1. There is a stable, documented single-adapter QA schema (v1)
2. `audit-adapter` emits that schema cleanly and consistently
3. Docs show the actual workflow: audit adapter → inspect eligibility → merge-audit with QA inputs

---

## Current State (what exists today)

Before building, it's important to know exactly what's already in place:

| Component | Status | Location |
|-----------|--------|----------|
| `EligibilityStatus` enum | Done | `vnext/merge/eligibility.py:33-53` |
| `AdapterQAResult` dataclass | Done | `vnext/merge/eligibility.py:61-130` |
| `classify_eligibility()` | Done | `vnext/merge/eligibility.py:138-213` |
| `screen_adapters()` | Done | `vnext/merge/eligibility.py:221-274` |
| `MergeQAReport` + builder | Done | `vnext/merge/qa_report.py` |
| `--source-a-qa / --source-b-qa` CLI flags | Done | `cli.py:1992-2008` |
| `--strict-qa` gating | Done | `cli.py:2075-2088` |
| `--qa-report` output | Done | `cli.py:2152-2155` |
| `gradience audit` command | Done | `cli.py:1832-1985` (single-adapter structural audit) |
| Eligibility tests | Done | `tests/merge/test_eligibility.py` |
| QA report tests | Done | `tests/merge/test_qa_report.py` |
| **`audit-adapter` command** | **Does not exist** | — |
| **Single-adapter QA JSON schema** | **Not defined** | — |
| **Terminal QA summary from audit** | **Not implemented** | — |
| **Workflow docs** | **Not written** | — |

**Key insight:** The plumbing exists. `AdapterQAResult` can serialize/deserialize. `merge-audit` can consume QA files. What's missing is the *production side*: a command that creates QA artifacts from a real adapter audit, and docs that show how to use it.

---

## Block 1: Finalize the Single-Adapter QA Schema

**Timebox:** Morning
**Files touched:** New schema doc, possibly extend `AdapterQAResult`

### 1.1 Define the v1 QA artifact schema

The artifact JSON needs to answer four questions:
1. What adapter is this?
2. What did the structural audit observe?
3. What is the current eligibility judgment?
4. Why was that judgment made?

**Target schema shape:**

```json
{
  "schema": "gradience.adapter_qa/v1",
  "adapter": {
    "name": "<adapter directory name>",
    "path": "./adapters/catsubcat-r16",
    "base_model": "meta-llama/Llama-2-7b-hf",
    "rank_nominal": 16,
    "n_layers": 32
  },
  "structural_summary": {
    "utilization_mean": 0.08,
    "utilization_median": 0.05,
    "stable_rank_mean": 1.9,
    "energy_rank_90_p50": 2.3,
    "rank_waste_ratio": 0.82,
    "flags": ["low_utilization", "high_rank_waste"]
  },
  "behavioral_summary": {
    "eval_available": true,
    "eval_dataset": "oasst2",
    "metric_name": "perplexity",
    "adapter_score": 6.81,
    "base_score": 4.66,
    "lower_is_better": true,
    "beats_base": false
  },
  "eligibility": {
    "status": "flagged_weak",
    "confidence": "medium",
    "reasons": [
      "adapter underperforms base on perplexity (oasst2)",
      "low utilization across layers",
      "high rank waste"
    ]
  }
}
```

### 1.2 Implementation mapping to existing code

Almost every field in the schema maps to something we already compute:

| Schema field | Source |
|-------------|--------|
| `adapter.name` | `Path(peft_dir).name` |
| `adapter.path` | `peft_dir` argument |
| `adapter.base_model` | `--base-model` CLI arg |
| `adapter.rank_nominal` | `result.layers[0].r` (from `LoRALayerAudit`) |
| `adapter.n_layers` | `result.n_layers` |
| `structural_summary.utilization_mean` | `result.utilization_mean` |
| `structural_summary.stable_rank_mean` | `result.stable_rank_mean` |
| `structural_summary.energy_rank_90_p50` | `result.energy_rank_90_p50` |
| `structural_summary.rank_waste_ratio` | Derived: `1 - utilization_mean` |
| `structural_summary.flags` | New: derive from thresholds on existing metrics |
| `behavioral_summary.*` | From CLI args: `--eval-name`, `--adapter-score`, `--base-score` |
| `eligibility.status` | `classify_eligibility()` return value |
| `eligibility.confidence` | New: derive from available evidence |
| `eligibility.reasons` | New: combine structural flags + behavioral notes |

### 1.3 What needs to be built for schema

**New code:**

1. **`gradience/vnext/audit/qa_artifact.py`** — New module (~150 lines)
   - `AdapterQAArtifact` frozen dataclass matching the schema above
   - `build_qa_artifact(audit_result, behavioral_args, base_model)` → `AdapterQAArtifact`
   - `to_dict()` / `from_dict()` serialization
   - Structural flag derivation logic (thresholds on utilization, rank waste)
   - Confidence derivation logic

2. **`docs/schemas/adapter_qa_v1.md`** — Internal schema reference (~60 lines)
   - Field definitions, types, allowed values
   - Status enum values: `eligible`, `uncertain`, `flagged_weak`, `unknown`
   - Confidence values: `low`, `medium`, `high`

### 1.4 Structural flag thresholds (initial policy)

These are starting values, not gospel:

| Flag | Condition |
|------|-----------|
| `low_utilization` | `utilization_mean < 0.25` |
| `high_rank_waste` | `rank_waste_ratio > 0.75` (i.e. `utilization_mean < 0.25`) |
| `concentrated_spectrum` | `energy_rank_90_p50 <= 2.0` and `rank_nominal >= 8` |
| `underutilized_capacity` | `stable_rank_mean < rank_nominal * 0.2` |

Note: `low_utilization` and `high_rank_waste` are correlated by definition. That's fine — they express the same fact from different angles. Keep both for readability.

### 1.5 Confidence derivation logic

| Condition | Confidence |
|-----------|-----------|
| Behavioral eval available + clear result (delta > margin) | `high` |
| Behavioral eval available + marginal result | `medium` |
| No behavioral eval, structural flags present | `low` |
| No behavioral eval, no structural flags | `low` |

The key constraint: **never assign `high` confidence without behavioral evidence.** This keeps the system honest.

### 1.6 Eligibility reason generation

Reasons list is built by concatenation:

1. If behavioral eval available and beats base: `"adapter outperforms base on {metric} ({dataset})"`
2. If behavioral eval available and worse: `"adapter underperforms base on {metric} ({dataset})"`
3. If no behavioral eval: `"no behavioral evaluation available"`
4. For each structural flag: append human-readable version (e.g. `"low utilization across layers"`)

### 1.7 Relationship to existing `AdapterQAResult`

`AdapterQAResult` (in `eligibility.py`) is the **merge-facing** type — it's what `merge-audit` consumes.
`AdapterQAArtifact` is the **user-facing** type — it's what `audit-adapter` produces and writes to disk.

The bridge: `AdapterQAArtifact.to_qa_result() → AdapterQAResult`. This method extracts the fields that `merge-audit` needs. `AdapterQAResult.from_dict()` should also be able to load directly from the artifact JSON's `eligibility` + `behavioral_summary` sections.

**Decision:** Keep both types. `AdapterQAArtifact` is richer (has structural summary, flags, confidence). `AdapterQAResult` stays lean for merge consumption. The artifact file can be loaded by either path:
- `AdapterQAArtifact.from_dict(json.load(f))` — full round-trip
- `AdapterQAResult.from_dict(json.load(f))` — merge-audit's `_load_source_qa` already does this; needs minor update to handle the new schema shape

### 1.8 Block 1 deliverables checklist

- [ ] `gradience/vnext/audit/qa_artifact.py` with `AdapterQAArtifact` dataclass
- [ ] `build_qa_artifact()` function
- [ ] Structural flag derivation
- [ ] Confidence derivation
- [ ] Reason list generation
- [ ] `to_dict()` / `from_dict()` with schema version
- [ ] `to_qa_result()` bridge method
- [ ] `docs/schemas/adapter_qa_v1.md` field reference
- [ ] Two hand-written example JSONs validating the schema shape

---

## Block 2: Clean `audit-adapter` CLI Output

**Timebox:** Late morning → early afternoon
**Files touched:** `cli.py`, `vnext/audit/__init__.py`

### 2.1 Add `audit-adapter` subcommand to CLI

This is a **new subcommand**, not a modification of `audit`. The existing `audit` command focuses on raw structural metrics with JSON/text output modes. `audit-adapter` wraps the same audit engine but adds behavioral context and QA judgment.

**New CLI flags (beyond what `audit` already has):**

```
gradience audit-adapter \
    --peft-dir ./adapters/catsubcat-r16 \          # required, same as audit
    --base-model meta-llama/Llama-2-7b-hf \        # optional, for UDR + metadata
    --eval-dataset oasst2 \                         # optional, behavioral context
    --adapter-score 6.81 \                          # optional, adapter metric value
    --base-score 4.66 \                             # optional, base model metric value
    --metric-name perplexity \                      # optional, default ""
    --lower-is-better \                             # flag, default true
    --margin 0.0 \                                  # optional, eligibility margin
    --out catsubcat_qa.json \                       # optional, output path
    --json                                          # flag, emit JSON to stdout instead of terminal summary
```

**Implementation location in `cli.py`:**

Add `cmd_audit_adapter` function after `cmd_audit` (~line 1985). Register it in the subparser setup (find `add_subparsers` call).

### 2.2 Command implementation pseudocode

```python
def cmd_audit_adapter(args):
    # 1. Run the structural audit (reuse audit_lora_peft_dir)
    result = audit_lora_peft_dir(peft_dir, ...)

    # 2. Build behavioral summary from CLI args
    behavioral = {
        "eval_available": args.adapter_score is not None,
        "eval_dataset": args.eval_dataset,
        ...
    }

    # 3. Run classify_eligibility() for the status
    qa_result = classify_eligibility(
        adapter_path=peft_dir,
        adapter_metric=args.adapter_score,
        base_metric=args.base_score,
        metric_name=args.metric_name,
        lower_is_better=args.lower_is_better,
        eval_dataset=args.eval_dataset,
        margin=args.margin,
    )

    # 4. Build full QA artifact
    artifact = build_qa_artifact(result, qa_result, base_model=args.base_model)

    # 5. Print terminal summary
    _print_qa_summary(artifact)

    # 6. Write JSON if --out provided
    if args.out:
        write artifact to file
        print(f"Wrote QA artifact to: {args.out}")
```

### 2.3 Terminal output format

Machine-readable first, human-readable second. Both should exist.

```
ADAPTER QA SUMMARY
──────────────────────────────────────────────────
  Adapter:       catsubcat-r16
  Path:          ./adapters/catsubcat-r16
  Base model:    meta-llama/Llama-2-7b-hf
  Rank:          16
  Layers:        32

STRUCTURAL SUMMARY
──────────────────────────────────────────────────
  Utilization (mean):  0.080
  Stable rank (mean):  1.900
  Energy rank 90 p50:  2.300
  Rank waste ratio:    0.820
  Flags:               low_utilization, high_rank_waste

BEHAVIORAL SUMMARY
──────────────────────────────────────────────────
  Eval dataset:  oasst2
  Metric:        perplexity (lower is better)
  Adapter score: 6.810
  Base score:    4.660
  Beats base:    no

ELIGIBILITY
──────────────────────────────────────────────────
  Status:      FLAGGED_WEAK
  Confidence:  medium
  Reasons:
    - adapter underperforms base on perplexity (oasst2)
    - low utilization across layers
    - high rank waste

OUTPUT
──────────────────────────────────────────────────
  Wrote QA artifact to: catsubcat_qa.json
```

**When no behavioral eval is provided:**

```
BEHAVIORAL SUMMARY
──────────────────────────────────────────────────
  Eval available: no
  Eligibility determined from structural evidence only

ELIGIBILITY
──────────────────────────────────────────────────
  Status:      UNKNOWN
  Confidence:  low
  Reasons:
    - no behavioral evaluation available
    - low utilization across layers (structural flag)
```

### 2.4 `_load_source_qa` compatibility

The existing `_load_source_qa` in `cli.py` calls `AdapterQAResult.from_dict(data)`. The new QA artifact JSON has a different shape. Two options:

**Option A (recommended):** Update `_load_source_qa` to detect schema version and dispatch:
```python
def _load_source_qa(path):
    data = json.load(f)
    if data.get("schema") == "gradience.adapter_qa/v1":
        # New artifact format — extract what merge-audit needs
        return AdapterQAArtifact.from_dict(data).to_qa_result()
    else:
        # Legacy format — direct load
        return AdapterQAResult.from_dict(data)
```

**Option B:** Always write a separate `AdapterQAResult`-compatible JSON. Rejected — adds friction and file sprawl.

### 2.5 Block 2 deliverables checklist

- [ ] `cmd_audit_adapter()` function in `cli.py`
- [ ] Subparser registration for `audit-adapter`
- [ ] `_print_qa_summary()` terminal formatter
- [ ] `--out` writes valid `gradience.adapter_qa/v1` JSON
- [ ] `--json` mode prints artifact JSON to stdout
- [ ] Update `_load_source_qa` to handle new schema
- [ ] Generate 2 example QA artifacts:
  - `examples/qa/catsubcat_r16_qa.json` (flagged_weak, behavioral eval available)
  - `examples/qa/btgenbot_r8_qa.json` (flagged_weak, behavioral eval available)

---

## Block 3: Connect QA Output to `merge-audit` Docs

**Timebox:** Afternoon
**Files touched:** `README.md`, new `docs/source_qa_workflow.md`, `examples/`

### 3.1 New doc: `docs/source_qa_workflow.md`

Structure:

#### Section 1: Why source QA comes before merge (5-8 lines)

> Study 16 showed that structural merge recommendations can be locally correct
> while operating on globally weak source adapters. A merge can be spectrally
> balanced yet behaviorally worthless if neither adapter outperforms the base model.
>
> Therefore: assess source-adapter eligibility before requesting pairwise merge
> recommendations. `merge-audit` accepts QA artifacts from `audit-adapter` and
> adjusts its warnings and policy accordingly.

#### Section 2: Basic workflow example

Show the exact three-step sequence:

```bash
# Step 1: Audit each adapter
gradience audit-adapter \
    --peft-dir ./adapters/catsubcat-r16 \
    --base-model meta-llama/Llama-2-7b-hf \
    --eval-dataset oasst2 \
    --metric-name perplexity \
    --adapter-score 6.81 \
    --base-score 4.66 \
    --lower-is-better \
    --out catsubcat_qa.json

gradience audit-adapter \
    --peft-dir ./adapters/btgenbot-r8 \
    --base-model meta-llama/Llama-2-7b-hf \
    --eval-dataset oasst2 \
    --metric-name perplexity \
    --adapter-score 5.47 \
    --base-score 4.66 \
    --lower-is-better \
    --out btgenbot_qa.json

# Step 2: Inspect eligibility
cat catsubcat_qa.json | jq '.eligibility'

# Step 3: Run merge audit with QA context
gradience merge-audit \
    --adapter-a ./adapters/catsubcat-r16 \
    --adapter-b ./adapters/btgenbot-r8 \
    --source-a-qa catsubcat_qa.json \
    --source-b-qa btgenbot_qa.json \
    --emit-report pair06_report.json
```

#### Section 3: Example interpretation (weak-source case)

Show what the user should see and infer from the Pair 06 (both-weak) case:
- Both adapters flagged weak
- Structural audit still runs, still produces merge strategy
- But warnings now say: "both sources underperform base"
- Deployment interpretation: do not assume merged adapter is worth using

#### Section 4: Strict QA gating

Show `--strict-qa` behavior:
- Structural audit runs
- Final recommendation blocked because both sources are weak
- Exit code indicates gating failure

#### Section 5: Balanced pair (happy path)

Show the boring-good case:
- Both adapters eligible
- Low structural risk
- Clean merge recommendation
- No eligibility warnings

### 3.2 README updates

Add a short section after the existing "Example workflow" block. ~15 lines max:

```markdown
### Source QA Workflow

Before merging, assess each adapter's standalone quality:

\`\`\`bash
# Audit each adapter with behavioral context
gradience audit-adapter --peft-dir ./adapter_a --eval-dataset oasst2 \
    --adapter-score 3.21 --base-score 4.66 --metric-name perplexity \
    --lower-is-better --out adapter_a_qa.json

# Feed QA artifacts into merge audit
gradience merge-audit --adapter-a ./adapter_a --adapter-b ./adapter_b \
    --source-a-qa adapter_a_qa.json --source-b-qa adapter_b_qa.json

# Use --strict-qa to gate on source quality
gradience merge-audit ... --strict-qa
\`\`\`

See [Source QA Workflow](docs/source_qa_workflow.md) for full documentation.
```

### 3.3 Example files

Add to `examples/qa/`:

1. **`catsubcat_r16_qa.json`** — Pair 06 weak source (flagged_weak, behavioral available, perplexity worse than base)
2. **`btgenbot_r8_qa.json`** — Pair 06 weak source (flagged_weak, behavioral available)
3. **`eligible_adapter_qa.json`** — Happy path (eligible, beats base)
4. **`structural_only_qa.json`** — No behavioral eval (unknown status, low confidence)

### 3.4 Block 3 deliverables checklist

- [ ] `docs/source_qa_workflow.md` with 5 sections
- [ ] README.md "Source QA Workflow" short section
- [ ] 4 example QA JSON files in `examples/qa/`
- [ ] Update `docs/cli.md` with `audit-adapter` command reference

---

## Block 4: Tests

**Timebox:** Late afternoon
**Files touched:** New test file(s)

### 4.1 New test file: `tests/test_qa_artifact.py`

**Schema output tests:**
- `test_qa_artifact_schema_version` — output dict has `"schema": "gradience.adapter_qa/v1"`
- `test_qa_artifact_required_fields` — all top-level keys present: `schema`, `adapter`, `structural_summary`, `behavioral_summary`, `eligibility`
- `test_qa_artifact_adapter_fields` — `name`, `path`, `base_model`, `rank_nominal`, `n_layers` all present and typed correctly
- `test_qa_artifact_roundtrip` — `from_dict(artifact.to_dict()) == artifact`
- `test_qa_artifact_json_serializable` — `json.dumps(artifact.to_dict())` doesn't raise

**Eligibility status assignment tests:**
- `test_eligible_when_beats_base` — adapter score < base score (lower_is_better) → `eligible`
- `test_flagged_weak_when_worse_than_base` — adapter score > base score → `flagged_weak`
- `test_uncertain_within_margin` — delta within margin → `uncertain`
- `test_unknown_no_behavioral` — no scores provided → `unknown`

**Structural flag tests:**
- `test_low_utilization_flag` — utilization_mean < 0.25 triggers `low_utilization`
- `test_no_flags_when_healthy` — utilization_mean > 0.5 → empty flags list
- `test_concentrated_spectrum_flag` — energy_rank_90_p50 <= 2.0 with rank >= 8

**Confidence tests:**
- `test_high_confidence_requires_behavioral` — behavioral + clear delta → `high`
- `test_low_confidence_structural_only` — no behavioral → never `high`
- `test_medium_confidence_marginal_behavioral` — behavioral + small delta → `medium`

**Bridge tests:**
- `test_to_qa_result_preserves_status` — artifact.to_qa_result().status matches
- `test_load_source_qa_handles_new_schema` — `_load_source_qa` with v1 artifact returns valid `AdapterQAResult`
- `test_load_source_qa_handles_legacy` — old-format JSON still works

**Missing behavioral data:**
- `test_behavioral_summary_eval_not_available` — `eval_available` is `false`, scores are `null`
- `test_reasons_mention_no_eval` — reasons list includes "no behavioral evaluation available"

### 4.2 CLI integration tests (if time permits)

- `test_audit_adapter_produces_json_file` — run command, check file exists and is valid JSON
- `test_audit_adapter_json_stdout` — `--json` prints to stdout
- `test_audit_adapter_no_behavioral_args` — runs without error, status is `unknown`

These can use the existing `examples/adapters/tiny_lora/` fixture.

### 4.3 Block 4 deliverables checklist

- [ ] `tests/test_qa_artifact.py` with ~18 tests
- [ ] All tests pass with `pytest tests/test_qa_artifact.py -v`
- [ ] Existing tests still pass: `pytest tests/merge/test_eligibility.py tests/merge/test_qa_report.py -v`

---

## Execution Order and Dependencies

```
Block 1 (schema)
  ├── 1.1-1.2: Define schema, map to existing code
  ├── 1.3: Build qa_artifact.py module
  ├── 1.4-1.6: Flag/confidence/reason logic
  └── 1.7: Bridge to AdapterQAResult
         │
Block 2 (CLI) ← depends on Block 1
  ├── 2.1-2.2: cmd_audit_adapter implementation
  ├── 2.3: Terminal formatter
  ├── 2.4: _load_source_qa update
  └── 2.5: Example artifacts
         │
Block 4 (tests) ← depends on Blocks 1+2
  ├── 4.1: Unit tests for qa_artifact
  └── 4.2: CLI integration tests
         │
Block 3 (docs) ← can run partly in parallel with Block 2
  ├── 3.1: source_qa_workflow.md
  ├── 3.2: README update
  ├── 3.3: Example files
  └── 3.4: CLI docs update
```

---

## Files Created or Modified (Complete List)

### New files
| File | Purpose | Est. lines |
|------|---------|-----------|
| `gradience/vnext/audit/qa_artifact.py` | QA artifact dataclass + builder | ~200 |
| `docs/schemas/adapter_qa_v1.md` | Schema field reference | ~80 |
| `docs/source_qa_workflow.md` | User-facing workflow guide | ~150 |
| `examples/qa/catsubcat_r16_qa.json` | Weak-source example | ~35 |
| `examples/qa/btgenbot_r8_qa.json` | Weak-source example | ~35 |
| `examples/qa/eligible_adapter_qa.json` | Happy-path example | ~35 |
| `examples/qa/structural_only_qa.json` | No-behavioral example | ~30 |
| `tests/test_qa_artifact.py` | Schema + eligibility + bridge tests | ~250 |

### Modified files
| File | Changes |
|------|---------|
| `gradience/cli.py` | Add `cmd_audit_adapter`, register subparser, update `_load_source_qa` |
| `gradience/vnext/audit/__init__.py` | Export `build_qa_artifact`, `AdapterQAArtifact` |
| `README.md` | Add ~15-line "Source QA Workflow" section |
| `docs/cli.md` | Add `audit-adapter` command reference |

---

## What NOT to Do on Day 2

1. **Do not invent a structural oracle for adapter quality.** Structural flags are warnings, not proof. The system is honest about this.
2. **Do not expand the status enum beyond 4 values.** `eligible`, `uncertain`, `flagged_weak`, `unknown` is enough. More categories create more explanations.
3. **Do not merge `audit` and `audit-adapter` into one command.** They serve different audiences. `audit` is for structural inspection. `audit-adapter` is for QA artifact production.
4. **Do not dump every computed metric into the terminal.** Only show metrics that inform the eligibility judgment.
5. **Do not call anything `eligible` without behavioral evidence.** This is the most important policy decision of the day.

---

## End-of-Day Checkpoint

You should be able to say:

> "A user can now run `audit-adapter`, get a stable QA artifact, pass it into `merge-audit`, and receive a recommendation whose warnings and policy reflect source eligibility."

**Concrete verification:**

```bash
# 1. Audit an adapter
gradience audit-adapter \
    --peft-dir examples/adapters/tiny_lora \
    --eval-dataset test_eval \
    --metric-name perplexity \
    --adapter-score 6.81 \
    --base-score 4.66 \
    --lower-is-better \
    --out /tmp/test_qa.json

# 2. Verify artifact
python -c "import json; d=json.load(open('/tmp/test_qa.json')); assert d['schema']=='gradience.adapter_qa/v1'; print(d['eligibility']['status'])"

# 3. Tests pass
pytest tests/test_qa_artifact.py tests/merge/test_eligibility.py tests/merge/test_qa_report.py -v

# 4. Lint passes
ruff check gradience/vnext/audit/qa_artifact.py
mypy gradience/vnext/audit/qa_artifact.py
```

That's the whole point of the day. No wizard hats. Just plumbing.
