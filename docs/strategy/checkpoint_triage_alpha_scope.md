# Checkpoint Triage Alpha Scope

Date: 2026-03-31
Status: Route 2 alpha contract

Checkpoint triage alpha is bounded to:

1. shared base model inventories only
2. small encoder checkpoints only
3. classification tasks only
4. mandatory evidence bootstrap before triage routing

Current canonical alpha instance:

- `field_trials/checkpoint_inventory_t02/`
- `field_trials/checkpoint_inventory_t02/preflight/alpha_bundle/report.html`

Interpretation contract inside this scope:

- QA-dominant triage logic is first-class,
- review/optional same-family states should be treated as review-like by default (not collapse-like),
- exact review thresholds remain guarded and non-canonical.

Anything outside this contract remains experimental and should not be presented as stable Route 2 behavior.
