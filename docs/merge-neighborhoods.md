# Merge Neighborhoods (Rule-Based Inventory Aid)

## What this is

`suggest-neighborhoods` is an inventory-level, rule-based suggester that groups adapters into conservative merge neighborhoods.

Schema identifier: `gradience.merge_neighborhoods/v1`

Current classification: **advanced workflow extension** (practitioner-usable, optional, and conservative).

## What this is not

- Not a graph visualization feature.
- Not clustering with hidden scoring or ML embeddings.
- Not an objective truth layer; it is a conservative decision aid from existing artifacts.

## How to run it

```bash
gradience suggest-neighborhoods \
  --qa-dir examples/qa \
  --report-dir examples/reports \
  --emit-report neighborhoods.json
```

Optional flags:
- `--strict-qa` : exclude strict-QA ineligible adapters from grouping.
- `--strict-input` : fail on malformed input files instead of skipping.
- `--min-compatibility <float>` : force low-score edges to `incompatible`.
- `--exclude-unknown` : exclude unknown/missing QA adapters from grouping.

### Python API (advanced wrapper)

```python
from gradience.api import suggest_neighborhoods

report = suggest_neighborhoods(
    qa_dir="examples/qa",
    report_dir="examples/reports",
    strict_qa=False,
)

print(report.to_dict()["schema"])  # gradience.merge_neighborhoods/v1
```

## Output sections

- `groups`: suggested neighborhoods with characterization, common strategy, and dominant issue.
- `excluded`: adapters removed by policy (for example `flagged_weak`).
- `boundary_warnings`: risky cross-group boundaries.

## Characterization vocabulary

- `likely-safe neighborhood`
- `caution neighborhood`
- `audit-aware neighborhood`

## First-pass decision rules

1. Exclude weak adapters by default.
2. Build compatibility edges from `MergeQAReport` fields (`pair_risk`, `recommended_strategy`, `compatibility_score`, eligibility status).
3. Form groups from high-compatibility edges first.
4. Merge moderate neighbors only when no contradictory high-risk boundary is introduced.
5. Emit boundary warnings for incompatible or conditional cross-group relations.

## Evaluation harness

For repeatable fixture-based validation, use:

```bash
python3 scripts/eval_neighborhoods.py
```

Protocol and fixture format:
- `docs/internal/neighborhood-eval-protocol.md`
- `examples/inventories/`

## Interpretive guidance

- Treat group membership as **screening output**, not merge authorization.
- Use pair reports for final decisions and strict-QA gates where needed.
- Review `boundary_warnings` before planning any cross-group merge sequence.

See also: `docs/advanced-workflows.md`
