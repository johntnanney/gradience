# Gradience v1.0.1 Release Notes

**Release Date:** April 4, 2026  
**Tag:** `v1.0.1`  
**Commit:** `3c51abb` (release tag target), with version-alignment follow-up in `f06c1bf`

## Summary

This release consolidates the CPU phase into a clearer product-facing shape. It does not introduce new merge-policy behavior; it sharpens documentation, workflow entry points, claim boundaries, and research-to-product translation so teams can run the validated triage flow with less ambiguity.

## Highlights

- Added a canonical internal product brief and claim-boundary docs.
- Canonicalized the merge triage "happy path" workflow.
- Split docs into clearer product zones: getting started, workflows, reference, explanations, research.
- Added shipping-surface definitions (`default`, `advanced`, `research-only`).
- Added standard output-bundle guidance and unified status/verdict glossary.
- Wired state-of-program and consolidation artifacts into top-level docs navigation.

## What Changed

### Product framing and boundaries

- `docs/product_brief.md`
- `docs/product_surface.md`
- `docs/product_shipping_surface.md`
- `docs/claims.md`

### Canonical workflow and reference docs

- `docs/workflows/canonical_merge_triage_workflow.md`
- `docs/reference/standard_output_bundle.md`
- `docs/glossary/status_and_verdicts.md`
- new `docs/getting_started/*`, `docs/reference/*`, `docs/explanations/*`

### Program consolidation artifacts

- `docs/strategy/state-of-program-april-2026.md`
- `docs/consolidation_backlog.md`
- index wiring updates in `docs/README.md` and `docs/product/README.md`

### Version alignment follow-up

- Package version updated to `1.0.1` in:
  - `pyproject.toml`
  - `gradience/__init__.py` fallback `__version__`
- `CHANGELOG.md` updated with a `[1.0.1] - 2026-04-04` entry

## Impact

- Easier onboarding and collaborator handoff
- Stronger guardrails against overclaiming
- Cleaner separation of validated core vs bounded/exploratory companions
- Better preparation for GPU-return proving-ground studies

## Known Boundaries

- Structural triage remains a prefilter, not a replacement for behavioral merge evaluation.
- Decoder conclusions remain bounded/observational until controlled GPU validation.
- Experimental probes remain companion signals, not default policy drivers.

## Upgrade Note

```bash
pip install --upgrade gradience==1.0.1
```

