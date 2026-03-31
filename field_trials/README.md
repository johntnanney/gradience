# Field Trials — CPU-Only Workflow Validation

Plan: `sidecar/notes/n51_cpu_field_trial_plan.md`

## Status

**Phase 1 — Pilot** (complete)

| Inventory | Type | Adapters | Pairs | Retained | Status |
|-----------|------|----------|-------|----------|--------|
| `inventory_01_same_task_control` | Same-task control | 3 of 4 | 3 | 0 | complete |
| `inventory_02_mixed_task` | Standard mixed-task | 5 of 5 | 10 | 1 (90%) | complete |
| `inventory_03_large_mixed_task` | Large mixed-task | 8 of 9 | 28 | 2 (93%) | complete |

Cross-pilot comparison: `pilot_phase1_comparison.md`

**Phase 2 — Merge Evaluation** (complete)

| Pilot | Retained merges | Controls | Near-miss | Key finding |
|-------|----------------|----------|-----------|-------------|
| Pilot 2 | 1 (AG News: +0.006 vs best source) | 2 | — | Retained pair beats both sources |
| Pilot 3 | 2 (AG News: -0.018, SST-2: -0.066) | 2 | 1 (hate: +0.078) | Retained pairs degrade least; near-miss improves |

Evaluation report: `phase2_evaluation_report.md`

**Phase 2b — Near-Miss Confirmation** (complete)

| Inventory | Backbone | Near-miss pairs | Retained pairs | Controls |
|-----------|----------|----------------|----------------|----------|
| `inventory_04_distilbert_irony_cluster` | distilbert | 3 (irony) | 2 (irony) | — |
| `inventory_05_bert_hate_emotion` | bert-base-uncased | 3 (hate, emotion) | 2 (hate, emotion) | 1 (cross-task) |

**Result:** Near-miss confirmed. Avg Δ vs best source: retained -0.018, near-miss -0.006, control -0.096. Near-miss pairs behave like retained pairs, not like excluded controls. Current action-plan feature validated. No further product change required.

Validation memo: `near_miss_validation.md`
Full confirmation data: `phase2b_confirmation_memo.md`

**Route 2 — Checkpoint triage alpha** (bounded)

- Canonical trial: `checkpoint_inventory_t02/`
- Alpha bundle builder: `checkpoint_inventory_t02/build_alpha_bundle.py`
- Polished report bundle: `checkpoint_inventory_t02/preflight/alpha_bundle/`
- Workflow guide: `../docs/examples/checkpoint-triage-alpha-workflow.md`
- Scope contract: `../docs/strategy/checkpoint_triage_alpha_scope.md`

## Per-inventory artifacts

Each inventory directory should contain:

```
manifest.json          # adapter names, tasks, backbone, metadata
adapter_cache/         # downloaded adapter weights
evidence/              # behavioral evidence JSON per adapter
preflight/             # Gradience preflight outputs (no evidence)
preflight_ev_*/        # Gradience preflight outputs (with evidence)
field_note.md          # after-action note (v1 = no evidence)
field_note_v2.md       # after-action note (v2 = with evidence, if both exist)
eval_plan.md           # tiny evaluation plan (retained + controls)
eval_results.json      # merge/eval outcomes
```

## Cross-trial outputs (in summary/)

```
trial_comparison_table.md
trial_summary_memo.md
product_pain_points.md
strengths.md
```
