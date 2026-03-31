# Stability Verdicts (Stage D)

| claim_id | original_verdict | perturbed_verdict | stability_verdict | short interpretation |
|---|---|---|---|---|
| A1 | strong | strong | stable | QA/evidence gating remains dominant across all classes. |
| A2 | strong | strong | stable | Conservative narrowing remains robust at workflow level. |
| B1 | moderate | moderate_with_caveat | moderately_stable | Same-task vs cross-task separation survives where testable, still coverage-limited. |
| B2 | moderate | mixed_weakened | panel_sensitive | Same-family strict intermediate ordering is not robust under local substitutions. |
| C1 | local_only | local_only | stable | Strongest structural metrics remain representation-local. |
| D1 | inconclusive | inconclusive | still_inconclusive | Near-miss portability remains unresolved outside LoRA. |
