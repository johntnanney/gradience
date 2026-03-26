# Advanced Workflows (Optional Tier)

This page collects Gradience workflows that are **usable and documented**, but not part of the default preflight spine.

Core default artifacts remain:
- `gradience.adapter_qa/v1`
- `gradience.merge_qa_report/v1`
- `gradience.inventory_summary/v1`

Advanced workflows are additive. They do not change default recommendation behavior unless you explicitly opt in.

## 1) Core-Space Pair Diagnostic

Use this when pair risk is ambiguous, especially for pairs with genuinely unclear task relationships where ordinary pair-risk is not already decisive. Verified adjudication (2026-03) showed core-space is structurally informative but its behaviorally useful role is narrower and more regime-dependent than initially expected.

### CLI

```bash
gradience merge-audit \
  --adapter-a ./adapter_a \
  --adapter-b ./adapter_b \
  --compute-core-space \
  --emit-report reports/ab_report.json
```

### Python API (advanced wrapper)

```python
from gradience.api import compute_core_space_diagnostic

core_space = compute_core_space_diagnostic(
    adapter_a="./adapter_a",
    adapter_b="./adapter_b",
)

print(core_space.status)
print(core_space.shared_basis_score)
```

### Interpretation guardrail

- Treat `core_space` as diagnostic metadata.
- Keep final merge decisions grounded in the full `MergeQAReport` (risk, issues, strategy, caveats), not one metric block.

See also: `docs/core-space-audit.md`

## 1b) Task-Relationship Advisory

The `task_relationship_advisory` is part of the stable interpretive layer, addressing the most important current regime boundary: **task identity**. On small encoder models, same-task pairs are broadly safe (confirmed across 45 pairs, 3 blind-spot studies, 0 material degradations), while cross-task pairs are where meaningful failure modes appear. The advisory fires when source QA artifacts indicate different evaluation tasks. It does not alter structural risk classification or recommendation logic.

The advisory is generated automatically when both `--source-a-qa` and `--source-b-qa` are provided and their `eval_dataset` fields differ. No additional flags are needed.

**When it matters most:** The advisory is most valuable in larger mixed-task inventories where structural pair-risk alone cannot distinguish safe same-task pairs from unsafe cross-task pairs. In observation testing, it collapsed an 11-candidate medium-risk pair set to 2 actionable same-task pairs in a 6-adapter/15-pair inventory.

**Evidence base:** 132+ advisory checks across 3 adjudication studies (2 backbones), 5 validation inventories, and 5 observation inventories. 0 same-task false positives, 100% different-task correct fire rate. Caution-raising in 52% of advisory-bearing pairs (concentrated on medium-risk cross-task pairs where pair-risk is permissive). Redundant when pair-risk is already high.

**Known overcaution regime:** When adapters share the same broad task but differ in training domain (e.g., movie sentiment vs product sentiment), the advisory fires because `eval_dataset` differs — but merges may be safe if the task features transfer well across domains. In a 15-pair sentiment domain-shift study, all 12 cross-domain merges were safe despite the advisory flagging them. The advisory reports a metadata fact, not a behavioral prediction; practitioners should treat its caution as "worth checking" in high-transfer task families.

## 2) Inventory Neighborhood Suggestions

Use this to organize larger adapter pools into conservative merge neighborhoods and highlight risky boundaries.

### CLI

```bash
gradience suggest-neighborhoods \
  --qa-dir examples/qa \
  --report-dir examples/reports \
  --emit-report neighborhoods.json
```

### Python API (advanced wrapper)

```python
from gradience.api import suggest_neighborhoods

report = suggest_neighborhoods(
    qa_dir="examples/qa",
    report_dir="examples/reports",
    strict_qa=False,
)

print(len(report.groups), "groups")
```

### Interpretation guardrail

- Neighborhoods are conservative planning aids, not automatic authorization.
- Use pair-level reports to validate any cross-group merge plan.

See also: `docs/merge-neighborhoods.md`

## 3) Demo Bundle

Use these example assets to evaluate the advanced tier quickly:
- `examples/reports/core_space_compatible_report.json`
- `examples/neighborhoods/sample_merge_neighborhoods.json`
- `examples/inventories/` (named fixture inventories + expectations)

## 4) What This Tier Does Not Do

- No default workflow overrides
- No graph UI or clustering UI
- No automatic policy promotion based only on advanced outputs
