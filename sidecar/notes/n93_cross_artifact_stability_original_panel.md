# n93 — Cross-Artifact Stability Check: Original Panel Freeze

**Type:** substudy setup note  
**Date:** 2026-03-31  
**Program:** Route2 Substudy 1 — Cross-Artifact Portability Stability Check  
**Status:** Stage A complete

---

## Objective

Freeze the original cross-artifact panel and original claim verdicts before any perturbation.

This note is the fixed reference for Stages B-D.

---

## Frozen references

- `sidecar/results/cross_artifact_portability/panel_table.json`
- `sidecar/results/cross_artifact_portability/invariant_signal_matrix.json`
- `sidecar/results/cross_artifact_portability/local_signal_table.json`
- `sidecar/results/cross_artifact_portability/product_relevance_filter.json`

New Stage A snapshots:

- `sidecar/results/route2_stability/cross_artifact/original_panel_snapshot.json`
- `sidecar/results/route2_stability/cross_artifact/original_claims_snapshot.json`

---

## Original claims under test

- **A1:** QA / evidence gating is a strong cross-artifact invariant.
- **A2:** Conservative narrowing is a strong cross-artifact invariant.
- **B1:** Task-relation ordering portability is moderate.
- **B2:** Same-family intermediate status portability is moderate.
- **C1:** Strongest structural metrics are representation-local.
- **D1:** Near-miss / optional middle-state portability is inconclusive.

---

## Stage A result

Stage A succeeds: both panel and claims are now frozen in machine-readable snapshots, enabling direct original-vs-perturbed comparison in later stages.
