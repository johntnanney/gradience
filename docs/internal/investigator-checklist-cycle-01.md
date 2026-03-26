# Investigator Checklist — Corpus Review Cycle 01

Use this as a quick execution checklist during cycle-01 collection and review.

Date: 2026-03-17

## 0) Before Starting

- [ ] Freeze behavior for one cycle
- [ ] No default logic changes
- [ ] No threshold changes
- [ ] No neighborhood logic edits
- [ ] No core-space formula edits
- [ ] No feature expansion

## 1) For Each Real Inventory

- [ ] Run adapter QA for all adapters
- [ ] Run pairwise merge reports
- [ ] Use core-space only on genuinely ambiguous pairs
- [ ] Run inventory summary
- [ ] Run neighborhood suggestion
- [ ] Append full run to corpus
- [ ] Confirm manifest validates
- [ ] Confirm no partial registration

## 2) Minimum Cycle Target

- [ ] At least 3 real inventories appended
- [ ] Prefer 4–5 if cleanly available

## 3) While Running Inventories (Field Notes)

- [ ] Note cases where strict QA feels obviously right
- [ ] Note cases where strict QA feels too harsh
- [ ] Note whether neighborhoods feel useful or trivial
- [ ] Note whether boundary warnings feel informative
- [ ] Note whether core-space changes judgment or only confirms it

## 4) After Corpus Collection

- [ ] Run corpus summary
- [ ] Review strategy distribution
- [ ] Review dominant issue distribution
- [ ] Review strict-block counts
- [ ] Review neighborhood exclusions and boundaries
- [ ] Review core-space usage frequency

## 5) Memo Questions

- [ ] What is the system mostly recommending?
- [ ] What issue is dominating real inventories?
- [ ] Are neighborhoods earning their keep?
- [ ] Is core-space being used meaningfully?
- [ ] Do any policies look obviously miscalibrated?

## 6) Final Cycle Decision (Choose One)

- [ ] `no_change`
- [ ] `targeted_calibration`
- [ ] `defer`

Decision guardrail:

- [ ] Prefer `no_change` unless evidence is strong and narrow
- [ ] If calibration is approved, allow only one small change
- [ ] Record rationale in a short addendum note

## 7) Definition of Done

- [ ] 3+ real inventories in corpus
- [ ] Corpus review memo written
- [ ] Explicit policy decision logged
- [ ] No feature-scope expansion occurred

## Quick References

- Runbook: `docs/internal/corpus-review-cycle-01.md`
- Review memo template: `docs/internal/templates/corpus-review-memo-template.md`
- Decision memo template: `docs/internal/templates/selective-calibration-decision-template.md`
