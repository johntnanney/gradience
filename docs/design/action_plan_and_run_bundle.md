# Design Note: Action Plan Completion & Run Bundle Integration

## Status

Both features have substantial implementations that are not yet fully wired or tested.

### Inventory Action Plan — current state

- `InventoryActionPlan` dataclass: **implemented** (`summary.py` L372–386)
- `build_action_plan()`: **implemented**, uses only existing stable signals (source QA status, pair-risk, task-relationship advisory)
- `format_action_plan()`: **implemented**, renders 5-section terminal block
- CLI integration: **done** — `summarize-inventory` calls it best-effort after the main summary
- Tests: **done** — 4 test classes covering same-task, mixed-task, messy, and guardrails
- Public API export: **missing** — `InventoryActionPlan` not in `gradience.__init__.__all__`

### Run Bundle — current state

- `run_bundle.py` module: **implemented** with `emit_run_bundle()`, `build_preflight_summary_json()`, `build_run_manifest()`, `build_preflight_summary_md()`, `build_action_plan_md()`, `build_comparison_md()`, `update_latest_pointer()`
- CLI integration: **missing** — no `--emit-bundle` flag, no `preflight` subcommand
- Tests: **none**
- Public API: **not exposed**

## What needs to happen

### Action Plan hardening

1. **Export to public API.** Add `InventoryActionPlan` to `gradience.__init__.__all__`. It's stable enough — the dataclass is frozen, the builder uses only existing signals, the tests cover the three inventory archetypes.

2. **Add per-pair risk and strategy to the action plan.** Currently the action plan partitions pairs into same-task/cross-task/excluded but doesn't surface the pair-level risk or recommended strategy. For retained same-task pairs, include the pair-risk level and recommended strategy from the merge report. This is still presentation-only (no new scoring), but gives the user a complete picture:

   ```
   Evaluate first:
     - sst2_a × sst2_b  (low risk, linear)
     - qnli_a × qnli_b  (medium risk, norm_equalized)
   ```

3. **Tighten the rendering.** The current `format_action_plan()` is clean but could:
   - Show the reduced-candidate ratio more prominently
   - Show cross-task pair count (not just regions)
   - Add a one-line provenance note (how many sources had behavioral evidence)

### Run Bundle CLI integration

1. **Add `--emit-bundle <DIR>` flag to `summarize-inventory`.** When present:
   - Call `emit_run_bundle()` to write all standard files
   - Print the path to `preflight_summary.md`
   - If a `--previous-run <DIR>` is also provided, generate `compare_to_previous.md`

2. **Add `--inventory-id` and `--run-id` flags.** These are required by `emit_run_bundle()`. Default `--run-id` to a timestamp. Default `--inventory-id` to the directory basename.

3. **Wire `update_latest_pointer()`.** After emitting a bundle, update the `latest/` symlink if the bundle is inside a recognized inventory root.

### Tests

1. **Run bundle tests.** Cover:
   - `build_preflight_summary_json()` — key presence and value types
   - `build_run_manifest()` — metadata accuracy
   - `build_preflight_summary_md()` — section presence
   - `build_action_plan_md()` — markdown well-formedness
   - `build_comparison_md()` — change detection (narrower, broader, unchanged)
   - `emit_run_bundle()` — file creation and content sanity
   - `update_latest_pointer()` — symlink creation and update

2. **Action plan integration test.** One test that goes from example inventory files → `summarize_inventory()` + `build_action_plan()` → formatted output, verifying the full round-trip.

### Completion signals

**Action plan done when:** A user running `gradience summarize-inventory` sees the action plan block after the summary, with per-pair risk labels and a clear candidate reduction ratio. The output is the obvious first artifact to read.

**Run bundle done when:** A user running `gradience summarize-inventory --emit-bundle ./run01` gets a directory containing `preflight_summary.md`, `preflight_summary.json`, `run_manifest.json`, `inventory_action_plan.md`, and optionally `compare_to_previous.md`. The summary is the single entry point; everything else is detail.
