# Neighborhood Evaluation Protocol (Internal)

## Goal

Evaluate `suggest-neighborhoods` on fixed inventory fixtures in a repeatable way and record whether behavior matches expected grouping/exclusion/boundary outcomes.

## Harness

Use:

```bash
python3 scripts/eval_neighborhoods.py
```

Default inputs:
- fixtures root: `examples/inventories`

Default outputs:
- run bundle: `results/neighborhood_eval/<timestamp>/`

## Fixture layout

Each fixture directory must contain:

- `qa/*.json` (valid `gradience.adapter_qa/v1` artifacts)
- `reports/*.json` (valid `gradience.merge_qa_report/v1` reports)
- optional `expected.json` (expected outcomes for comparison)
- optional `expected_notes.md` (hand-authored interpretation notes)

## `expected.json` shape

```json
{
  "expected_groups": [["adapter_a", "adapter_b"], ["adapter_c"]],
  "expected_excluded": ["adapter_x"],
  "expected_boundary_warnings": [
    [["adapter_a", "adapter_b"], ["adapter_c"]]
  ],
  "expected_characterizations": [
    {
      "members": ["adapter_a", "adapter_b"],
      "characterization": "likely-safe neighborhood"
    }
  ]
}
```

Notes:
- Group and boundary comparisons are order-insensitive.
- Boundary comparisons are done via member sets, not cluster IDs.
- Characterization checks apply only to groups listed in `expected_characterizations`.

## Per-fixture output bundle

For each fixture:

- `neighborhood_report.json` (emitted `gradience.merge_neighborhoods/v1`)
- `terminal_summary.txt` (captured formatter output)
- `comparison.json` (expected vs actual + mismatches)
- `expected_notes.md` (copied when present in fixture input)
- `verdict.txt` (`passed`, `partially passed`, `unexpected grouping`)

## Run-level outputs

- `summary.json` (machine-readable fixture results)
- `summary.md` (compact table)

## Verdict semantics

- `passed`: groups, exclusions, boundaries, and declared characterizations all match expectations.
- `partially passed`: grouping matches, but one or more non-group checks differ.
- `unexpected grouping`: expected/actual group sets differ.

## Fixed fixtures

Current baseline fixture set:

- `inventory_safe_small`
- `inventory_mixed_small`
- `inventory_fragmented_small`
- `inventory_with_weak_sources`
- `inventory_large_realistic`

The first four are small deterministic fixtures. The large fixture is used to confirm evaluation works on a broader inventory without manual cleanup.
