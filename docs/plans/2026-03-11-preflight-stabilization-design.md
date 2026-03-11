# Pre-GPU CPU-Side Stabilization Design

> Make the artifact spine (AdapterQAArtifact → MergeQAReport → InventorySummary) boringly reliable before GPU experiments begin.

## Goal

Close reliability gaps in the three-tier artifact pipeline: canonical workflow, cross-artifact policy consistency, CLI scripting reliability, and inventory usefulness. No new features, no GPU work — just stabilization.

## Approach

Two-tier fixture strategy:
1. **Real adapter fixture** (`examples/adapters/tiny_lora/`) for exercising `audit-adapter` through its true PEFT input path
2. **Pre-built JSON artifacts** (existing `examples/qa/`, `examples/reports/`, `examples/inventory/`) for fast tests, aggregation tests, and documentation

## Deliverables

### Priority 1: Canonical Workflow & Getting Started

**Real adapter fixture:** `examples/adapters/tiny_lora/` — minimal PEFT adapter (rank 4, single linear layer). Created once via `scripts/create_tiny_lora_fixture.py`, committed as binary fixtures (adapter_config.json + adapter_model.safetensors).

**Workflow smoke script:** `scripts/preflight_smoke.sh` — runs the full artifact spine end-to-end:
1. `gradience audit-adapter` on the real fixture → emits QA artifact JSON
2. `gradience merge-audit` on two copies → emits merge report JSON
3. `gradience summarize-inventory` on the output dirs → emits inventory summary JSON
4. Validates all three outputs are valid JSON with correct schema fields
5. Cleans up temp outputs

Exit 0 = all green, non-zero = something broke.

**Getting started guide:** `docs/getting-started-preflight.md` — walk-through mirroring the smoke script with explanations and expected output snippets. Includes two shell blocks: happy path (exit 0) and blocked path (strict-QA rejects weak adapter, non-zero exit).

### Priority 2: Cross-Artifact Policy Consistency

**Policy doc:** `docs/preflight-policy.md` — documents cross-artifact contracts:
- Eligibility status flow: QA artifact → merge report → inventory summary
- `--strict-qa` blocking semantics (consistent across merge-audit and summarize-inventory)
- Strategy/action alignment (low risk → `linear`, medium → `norm_equalized`, high → `audit_aware`)

**Cross-artifact regression tests:** `tests/test_cross_artifact_policy.py`
- Happy path: eligible adapters → low-risk merge → zero block candidates
- Strict-QA blocked: `flagged_weak` → blocked under strict → counted as block candidate
- Missing QA: `null` eligibility → noted in merge report → counted as block candidate
- Strategy/action alignment: each risk level maps to correct strategy string

**Strict reload invariant tests:** `from_dict(to_dict(obj))` round-trips for all three artifact types. Added to existing test files.

### Priority 3: CLI/Scripting Reliability

**Exit code tests:** `tests/test_cli_exit_codes.py`
- All three commands: exit 0 on success, non-zero on invalid input
- `--strict-input` / `--strict-qa`: non-zero on policy violation

**Overwrite behavior:** Document in `--help` text that `--emit-report` / `--emit-artifact` silently overwrites. No behavioral change.

### Priority 4: Inventory Usefulness

**Terminal formatter fix:** `format_inventory_summary()` labels — "Qa artifact:" → "QA artifacts:", "Merge report:" → "Merge reports:".

**Realistic inventory example:** `examples/inventory/realistic_inventory_summary.json` — mix of statuses, multiple flags, varied risk levels. Looks like a real 10-adapter inventory.

## File Placement

| Deliverable | Path |
|---|---|
| Real adapter fixture | `examples/adapters/tiny_lora/` |
| Fixture creation script | `scripts/create_tiny_lora_fixture.py` |
| Workflow smoke script | `scripts/preflight_smoke.sh` |
| Getting started guide | `docs/getting-started-preflight.md` |
| Policy doc | `docs/preflight-policy.md` |
| Cross-artifact regression tests | `tests/test_cross_artifact_policy.py` |
| CLI exit code tests | `tests/test_cli_exit_codes.py` |
| Realistic inventory example | `examples/inventory/realistic_inventory_summary.json` |
| Terminal formatter fix | `gradience/vnext/inventory/summary.py` |
| Overwrite docs | `gradience/cli.py` help text |
| Strict reload invariant tests | Existing test files per artifact type |

## Task Ordering

Fixture creation first (everything downstream depends on the real adapter). Then tests + docs can parallelize where independent.

## What Is NOT in Scope

- GPU experiments or model loading beyond the tiny fixture
- New features or new artifact types
- Dashboard, graphing, or visualization
- Performance optimization
- CI pipeline changes
