# Status and Verdicts

**Audience:** practitioner, maintainer, collaborator  
**Status:** stable (vocabulary boundary)  
**Purpose:** separate user-facing operational language from internal/research language  
**Canonical for:** status/verdict wording in product docs, workflow output, and demos  
**Supersedes:** mixed status vocabulary usage across docs and research notes  
**See also:** [`../product_surface.md`](../product_surface.md), [`../workflows/canonical_merge_triage_workflow.md`](../workflows/canonical_merge_triage_workflow.md), [`../adapter-qa-artifact.md`](../adapter-qa-artifact.md), [`../merge-risk-report.md`](../merge-risk-report.md), [`../inventory-summary.md`](../inventory-summary.md)

## User-Facing Operational Statuses

Use this vocabulary in product docs, demo paths, and operator guidance.

| Layer | Preferred User-Facing Terms | Meaning |
|---|---|---|
| Source quality | `eligible` / `not eligible` | Whether an adapter should proceed in the canonical workflow. |
| Inventory action | `retained` / `monitor` / `skip` | Action-plan triage: evaluate-first, watchlist, or deprioritize/exclude. |
| Task relationship | `same-task` / `same-family` / `cross-task` | Relationship used for caution boundaries and prioritization. |
| Pair recommendation | `safe` / `caution` / `not recommended` | High-level decision framing for what to evaluate now vs later vs not by default. |

### User-Facing Mapping Rules

| Artifact Field(s) | User-Facing Label |
|---|---|
| `eligibility.status == eligible` | `eligible` |
| `eligibility.status in {uncertain, flagged_weak, unknown_no_behavioral_eval}` | `not eligible` (with reason note) |
| Action-plan "evaluate first" | `retained` |
| Action-plan "near-miss" | `monitor` |
| Action-plan "exclude/deprioritize" | `skip` |
| Pair with low structural risk and no cross-task advisory | `safe` |
| Pair with medium risk, near-miss context, or cross-task advisory | `caution` |
| Pair with high risk, strict-QA block, or strong source weakness | `not recommended` |

## Internal / Research Statuses

Use these terms in engineering, diagnostics, and research writeups. Do not make them the primary language of the product front door.

| Internal Term Class | Examples |
|---|---|
| Low-level verdict internals | `branch-5`, per-layer verdict distributions |
| Confidence mechanics | `verdict confidence`, confidence strata |
| Spectral partition internals | `high_sv_alignment` thresholds and related slice cutoffs |
| Experimental signal classes | `bounded_keep`, `keep_exploratory`, exploratory probe classes |
| Research diagnostics | OA factor decompositions, HTSR fit-quality internals, direction-aware companion diagnostics |

## Translation Guide (Internal -> User-Facing)

When communicating with users, translate internal terms to operational labels:

| Internal / Research Phrase | User-Facing Translation |
|---|---|
| `branch-5 heterogeneous` | `caution` (needs closer review) |
| `bounded_keep companion` | `advanced diagnostic (optional)` |
| `keep_exploratory` | `research-only signal` |
| `high_sv_alignment threshold exceeded` | `structurally aligned` (context-dependent) |
| `strict_qa_block_candidates` | `not recommended until source evidence improves` |

## Style Rules

1. Product-facing docs should prefer operational labels first, with internals optionally in parentheses.
2. Internal/research labels are allowed in technical references, strategy memos, and sidecar-facing material.
3. If both are shown, present user-facing label first, then internal label.
4. Do not use `branch-*` labels in user-facing action plans.
5. Keep `bounded_keep` / `keep_exploratory` in internal status matrices, not front-door workflow copy.
