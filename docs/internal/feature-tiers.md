# Gradience Feature Tiers (Internal)

This table tracks near-term product tiering so documentation and implementation stay aligned.

## Tier Table

| Surface | Tier | Current Classification | Default Workflow Impact |
|---|---|---|---|
| `AdapterQAArtifact` (`gradience.adapter_qa/v1`) | Core | stable | core preflight artifact |
| `MergeQAReport` (`gradience.merge_qa_report/v1`) | Core | stable | core pairwise decision artifact |
| `InventorySummary` (`gradience.inventory_summary/v1`) | Core | stable | core inventory aggregate artifact |
| Core-space audit (`merge-audit --compute-core-space`) | Advanced | advanced optional diagnostic | no default recommendation change |
| Merge neighborhoods (`suggest-neighborhoods`) | Advanced | advanced workflow extension | optional inventory decision aid |
| Compression-related studies/paths | Experimental | secondary | not part of default preflight workflow |

## Notes

- “Advanced” means documented and practitioner-usable, but still optional.
- “Core” means default preflight spine and stable product center.
- “Experimental” means useful for research/exploration but not default workflow guidance.
