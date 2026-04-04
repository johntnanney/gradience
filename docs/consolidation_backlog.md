# Consolidation Backlog

**Audience:** maintainer, product owner, documentation lead  
**Status:** active  
**Purpose:** convert research inventory into a product-facing consolidation checklist  
**Canonical for:** structural consolidation tasks and archive boundaries  
**Source:** [`RESEARCH_INVENTORY.md`](RESEARCH_INVENTORY.md) (April 4, 2026)  
**See also:** [`claims.md`](claims.md), [`product_surface.md`](product_surface.md), [`product_shipping_surface.md`](product_shipping_surface.md), [`workflows/canonical_merge_triage_workflow.md`](workflows/canonical_merge_triage_workflow.md)

This checklist is the execution layer for repository consolidation. It is focused on making product-facing docs and workflows easy to find, while moving historical/research overflow into explicit archive lanes.

## Must Consolidate Now

### A) Canonical docs that belong in main repo docs

- [x] Add canonical claims boundary page (`docs/claims.md`).
- [x] Add canonical product surface page (`docs/product_surface.md`).
- [x] Add canonical shipping-surface page (`docs/product_shipping_surface.md`).
- [x] Add canonical workflow rationale translation page (`docs/explanations/why_the_workflow_looks_like_this.md`).
- [ ] Normalize one-page "where to start" links so all product-facing entry points route through:
  `claims -> product_surface -> canonical workflow -> standard output bundle`.

### B) Authoritative workflow docs

- [x] Create canonical happy path (`docs/workflows/canonical_merge_triage_workflow.md`).
- [x] Define standard output deliverable (`docs/reference/standard_output_bundle.md`).
- [ ] Ensure duplicate/legacy workflow docs either:
  1) become compatibility pointers, or
  2) are folded into canonical references.

### C) Product-facing summaries

- [x] Add authoritative status snapshot (`docs/strategy/state-of-program-april-2026.md`).
- [x] Add product-facing status/vocabulary boundary (`docs/glossary/status_and_verdicts.md`).
- [ ] Add one short "product capabilities now" summary table in product docs that mirrors claims + shipping surface.

### D) Key specs and manuscripts currently stranded outside canonical repo docs

These items were identified in the inventory as high priority to consolidate from external locations.

- [ ] `~/Downloads/GRADIENCE_PAPER_DRAFT_REVISED.md`
  -> target: `docs/research/external_drafts/paper_draft_revised.md`
- [ ] `~/Downloads/SERIES_POST_7_FINAL.md`
  -> target: `docs/research/external_drafts/blog_post_7_final.md`
- [ ] `~/Downloads/POST8_FINAL.md`
  -> target: `docs/research/external_drafts/blog_post_8_final.md`
- [ ] `~/Downloads/STUDY16B_DRAFT.md`
  -> target: `docs/research/external_drafts/study16b_draft.md`
- [ ] `~/Downloads/STUDY17A_INTERIM_RESULTS.md`
  -> target: `docs/research/external_drafts/study17a_interim_results.md`
- [ ] `~/Downloads/DFA_WORKSHOP_PAPER.md`
  -> target: `docs/research/external_drafts/dfa_workshop_paper.md`
- [ ] `~/Downloads/spec-phase-a.md`
  -> target: `docs/plans/v0_12_phase_a_spec.md`
- [ ] `~/Desktop/Gradience_LoRA_Decision_Infrastructure.pdf`
  -> target: `docs/research/external_drafts/Gradience_LoRA_Decision_Infrastructure.pdf`

## Archive or Separate

These should not crowd the product front door. Keep them available but clearly separated.

### A) Older drafts and superseded notes

- [ ] Move superseded one-off drafts to a clearly labeled archive subtree:
  `docs/archive/superseded/`
- [ ] Add an `ARCHIVED` header block to moved files with replacement pointers.

### B) Historical blog drafts

- [ ] Move non-canonical blog drafts to:
  `docs/archive/blog_drafts/`
- [ ] Keep only canonical published blog references in front-facing research/product indexes.

### C) Philosophy materials

- [ ] Keep philosophy materials separate from product/research-engineering path.
- [ ] Ensure all philosophy references point to separated location and are not listed as product dependencies.

### D) One-off experiment manuscripts

- [ ] Move one-off manuscripts not tied to active product claims into:
  `docs/archive/experiment_manuscripts/`
- [ ] Keep only claim-relevant, currently cited manuscripts in curated `docs/research/` entry points.

## Operational Rules

1. One topic, one canonical page.
2. Everything else must be marked as supporting, archived, or superseded.
3. Product entry docs should never require sidecar-note context to interpret status language.
4. Research-heavy artifacts can remain, but must not define default product workflow.

## Definition of Done

This backlog is complete when:

1. Product-facing entry points consistently route through canonical docs.
2. External high-priority drafts/specs are either imported into target locations or explicitly deferred with owner/date.
3. Archive lanes exist and are used for superseded/historical material.
4. The repository front door reads like a labeled toolkit, not an unlabeled workshop.
