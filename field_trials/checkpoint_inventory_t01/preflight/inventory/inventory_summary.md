
  INVENTORY OVERVIEW
  ============================================================
  Mixed-quality inventory — 3 weak/unknown source(s) identified

  Merge reports:      6
  QA artifacts:       4

  SOURCE QA SNAPSHOT
  ----------------------------------------
  Eligibility:
    eligible:  1
    flagged_weak:  3

  Evidence tier:
    behavioral_reported:  1
    behavioral_weak:     3

  Note: behavioral scores are user-reported; Gradience does not
  independently verify claimed evaluation results.
  Strict-QA block candidates: 6

  STRUCTURAL DETAIL
  ----------------------------------------
  Flags: diffuse_delta_spectrum: 1, high_effective_rank: 1
  Pair risk: high: 3, medium: 3
  Strategies: audit_aware: 3, norm_equalized: 3
  Issues: high_redundancy: 1, partial_redundancy: 2, subspace_conflict: 3

  INVENTORY POLICY SUMMARY
  ----------------------------------------
  Type:        mixed_quality
  Driver:      source_qa
  Posture:     narrow
  Constraint:  Source QA is the binding constraint;
               resolve weak evidence before exploring merges.

  INTERPRETATION
  ----------------------------------------
  3 adapter(s) have weak or missing behavioral evidence.
  Source QA is likely the main narrowing step for this inventory.

