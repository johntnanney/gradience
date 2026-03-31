# Checkpoint Triage Alpha Workflow (Route 2)

This is the first polished broadened workflow candidate in Route 2.

Canonical example: `field_trials/checkpoint_inventory_t02/`.

## Scope (Alpha Contract)

- shared base only
- small encoder checkpoints only
- classification tasks only
- evidence bootstrap required before triage decisions

If any of these fail, treat the run as experimental and do not promote conclusions.

## Quickstart

1. Run the canonical trial workflow:

```bash
python3 field_trials/checkpoint_inventory_t02/run_trial.py
```

2. Build the polished alpha bundle (HTML + compact summary):

```bash
python3 field_trials/checkpoint_inventory_t02/build_alpha_bundle.py
```

3. Open the report:

- `field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html`

## Workflow shape

1. Evidence bootstrap (first-class gate)
2. QA artifact generation
3. Pairwise compatibility summaries
4. Inventory action plan
5. Tiny follow-through evaluation

The workflow is intentionally conservative: weak source evidence can block otherwise plausible pairwise structure.

## Why QA dominates in this workflow

Checkpoint triage alpha is designed as evidence-aware narrowing, not structure-only ranking:

- structural signals help separate plausible vs implausible relations,
- QA/evidence status determines whether those relations are operationally usable,
- mixed-evidence cases should default to review-first routing, not automatic retention.

This is the expected behavior, not a failure mode. In Route 2 language, QA-dominant aggregation is a distinct operational family.

## Reading the triage middle (review and optional)

Use this interpretation in alpha reports:

- `qa_clear`: evaluate-first / retain-candidate lane.
- `qa_review`: review/optional lane (often same-family or near-miss-like, evidence-mixed).
- `qa_blocked`: do-not-prioritize lane until evidence improves.

Important guardrail: review/optional states are generally closer to safe-like triage behavior than collapse-like behavior, but exact thresholds and fine-grained ordering inside `qa_review` should remain explicitly guarded.

See also:

- `docs/strategy/aggregation_stability_summary.md`
- `docs/strategy/aggregation_mixed_evidence_summary.md`

## Canonical outputs

- `field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html`
- `field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/alpha_summary.json`
- `field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/bundle_manifest.json`
- `field_trials/checkpoint_inventory_t02/trial_memo.md`

## Full documentation

For detailed usage, output reference, example walkthrough, and adaptation guide, see the mini-product README:

- [`field_trials/checkpoint_inventory_t02/README.md`](../../field_trials/checkpoint_inventory_t02/README.md)

## External pull test rule

Run one additional checkpoint-inventory deployment only when externally motivated by a real manual inventory problem. If no concrete workflow owner exists, stop at this alpha package.
