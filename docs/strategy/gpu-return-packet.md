# GPU-Return Packet

**Date:** April 4, 2026  
**Audience:** maintainers, research operators, collaborators with GPU access  
**Status:** execution-ready packet  
**Purpose:** make the next three proving grounds turn-key after CPU consolidation  
**Canonical for:** GPU-return execution order, gates, and required artifacts  
**See also:** [`current-release-state.md`](current-release-state.md), [`state-of-program-april-2026.md`](state-of-program-april-2026.md), [`../plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md`](../plans/2026-04-03-decoder-only-spectral-fingerprinting-gpu-return-plan.md)

## Packet Scope

This packet is the direct continuation of the CPU-complete boundary:

1. DeBERTa adjudication
2. Controlled decoder fingerprinting / merge triage
3. Verdict-confidence validation

Packet workspace:

- [`field_trials/gpu_return_packet/README.md`](../../field_trials/gpu_return_packet/README.md)
- [`field_trials/gpu_return_packet/packet_manifest.json`](../../field_trials/gpu_return_packet/packet_manifest.json)

## Execution Order

1. **PG1: DeBERTa adjudication** (~3 GPU hours)
2. **PG2: Controlled decoder fingerprinting / merge triage** (~8-12 GPU hours)
3. **PG3: Verdict-confidence validation** (~2-4 GPU hours + analysis)

Do not reorder. PG3 should consume outcomes from PG1 and PG2.

## Proving Ground Runbooks

1. [`field_trials/gpu_return_packet/deberta_adjudication/runbook.md`](../../field_trials/gpu_return_packet/deberta_adjudication/runbook.md)
2. [`field_trials/gpu_return_packet/decoder_fingerprinting_merge_triage/runbook.md`](../../field_trials/gpu_return_packet/decoder_fingerprinting_merge_triage/runbook.md)
3. [`field_trials/gpu_return_packet/verdict_confidence_validation/runbook.md`](../../field_trials/gpu_return_packet/verdict_confidence_validation/runbook.md)

## Day-0 Preflight (Before GPU Starts)

1. Confirm environment supports the intended model families and adapter training stack.
2. Confirm dataset access and evaluation scripts are available for all planned tasks.
3. Confirm output roots are writable and persistent.
4. Confirm reproducibility settings (seeds, config capture, command capture) are locked.
5. Confirm packet manifest owner/date/status are updated before first run.

## Decision Gate

No product-policy update is authorized from packet execution alone.

Packet completion requires:

1. All three runbooks executed with required output artifacts present.
2. One integrated outcome memo written:
   - `field_trials/gpu_return_packet/gpu_return_outcome_summary.md`
3. Explicit call on each line:
   - `promote`, `bounded_keep`, or `hold`.

