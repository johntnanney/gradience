# Cross-Task Boundary — Public Summary

## The regime boundary

On small encoder models, the most meaningful boundary in Gradience preflight is task identity. Same-task pairs — even with varied training styles, different training domains, and substantial source-strength asymmetry — remain broadly safe. Cross-task pairs are where meaningful merge failure modes appear: weaker-task dilution, asymmetric degradation, and structural permissiveness that masks functional incompatibility.

## The advisory's role

The task-relationship advisory is part of the stable interpretive layer. It fires when source QA artifacts indicate different evaluation datasets and stays silent on same-task pairs. Its strongest value is inventory-level: in mixed-task pools, it cleanly partitions the pair matrix into same-task safe zones and cross-task caution zones. In observation testing, it collapsed 11 medium-risk candidates to 2 actionable pairs in a single inventory.

## What the same-task negative results mean

Three blind-spot studies tested whether same-task pairs hide meaningful merge risk. They do not — at least not on small encoder models with GLUE-family tasks. This is a positive result: it means the workflow can confidently treat same-task pairs as low-priority and focus its interpretive effort on cross-task boundaries where the advisory and structural analysis together provide the most value.
