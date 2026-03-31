
  INVENTORY OVERVIEW
  ============================================================
  Mixed-quality inventory — 3 weak/unknown source(s) identified

  Merge reports:      10
  QA artifacts:       5

  SOURCE QA SNAPSHOT
  ----------------------------------------
  Eligibility:
    eligible:  1
    flagged_weak:  3
    uncertain:  1

  Evidence tier:
    behavioral_reported:  2
    behavioral_weak:     3

  Note: behavioral scores are user-reported; Gradience does not
  independently verify claimed evaluation results.
  Strict-QA block candidates: 9

  STRUCTURAL DETAIL
  ----------------------------------------
  Flags: diffuse_delta_spectrum: 1, high_effective_rank: 1, steep_core_singular_decay: 1
  Pair risk: high: 7, medium: 3
  Strategies: audit_aware: 7, norm_equalized: 3
  Issues: high_redundancy: 1, partial_redundancy: 2, subspace_conflict: 7

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

