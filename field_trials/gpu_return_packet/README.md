# GPU Return Packet Workspace

This directory is the execution bundle for the first post-CPU proving-ground sequence.

## Run Order

1. `deberta_adjudication/`
2. `decoder_fingerprinting_merge_triage/`
3. `verdict_confidence_validation/`

## Root Artifacts

- `packet_manifest.json` — packet state, owners, and study order
- `gpu_return_outcome_summary.md` — integrated closeout memo after all three studies run

## Principle

This packet is execution-oriented, not exploratory.  
Each proving ground has a locked runbook with required outputs and a clear stop condition.

