# Evidence Register

**Date:** March 2026

---

## Settled claims

| # | Claim | Key evidence | Source |
|---|-------|-------------|--------|
| 1 | Task-boundary detection is reliable | 0 false positives, 5 inventories, 53+ pairs | Field trials |
| 2 | Severity is not portable across backbones | Rankings reverse completely: QNLI×MRPC 41.7% on db, 1.7% on rb | S01 |
| 3 | Readout orthogonality is benign in isolation | 5/14 same-task pairs orthogonal, all safe (max Δ=2.2%) | n36 |
| 4 | Multi-attractor readout ≠ fragile | 10 families mapped, all safe regardless of attractor type | n41 |
| 5 | Catastrophe requires V-module pathology AND readout incompatibility | SC-QMRB falsifier + same-task seed evidence + V-module dim ratio + CA-01 seed contrast | n32, n36, n21 |
| 6 | Fragile and cross-task failure are qualitatively distinct | Double dissociation: confidence collapse (fragile) vs high-confidence wrong (cross-task) | n64 |
| 7 | Near-miss is behaviorally safe, not a fragile precursor | Neither-source <2%, zero confidence collapse, boundary noise only | n65 |
| 8 | Near-miss is a useful product category | 7 pairs, 3 backbones, avg Δ=-0.006 | Field trials |

## Confirmed but thin (two backbones only)

| # | Claim | Current evidence | What would strengthen it |
|---|-------|-----------------|------------------------|
| 9 | Instability is a portable descriptor | Consistent ranking on db + rb, clean gap 0.30–0.74 | DeBERTa adjudication |
| 10 | V-module dim ratio separates catastrophic from safe | d=3.36, zero range overlap (0.64–0.74 vs 0.79–0.89) | Third backbone |
| 11 | Head-level cancellation explains seed sensitivity | 7 heads with \|Δ_DR\|≥0.15 resolve CA-01's 29pp gap | Replication on DeBERTa |
| 12 | Two readout mechanisms (degeneracy vs switching) | All degeneracy on db, all switching on rb — perfectly confounded | Third backbone breaks or confirms confound |

## Open questions

| # | Question | Status | Blocked on |
|---|----------|--------|-----------|
| 13 | Does instability ranking survive on DeBERTa? | Pre-registered (n07, Predictions A–C) | GPU |
| 14 | Does V-module signal survive on disentangled attention? | Pre-registered (n07, Prediction D) | GPU |
| 15 | Does the backbone–mechanism confound dissolve? | Will be answered by DeBERTa attractor analysis | GPU |
| 16 | Where is the V-module pathology threshold in geometric terms? | Example-level program observes consequences, not coordinates | Per-example × per-layer correlation (CPU-feasible) |
| 17 | Does the cross-task failure mode replicate beyond CT-01? | One case only; double dissociation suggestive but unreplicated | Additional cross-task merge evaluations |

## Ruled-out hypotheses (summary)

| # | Hypothesis | Falsifier |
|---|-----------|-----------|
| R1 | Portable severity score | Ranking reversal across backbones |
| R2 | Task-pair catastrophe lookup | No pair catastrophic on both backbones |
| R3 | Aggregate within-layer threshold | Backbone confound; signal was in V-module, not concatenation |
| R4 | Readout orthogonality as risk marker | SC-QMRB falsifier + 5/14 same-task orthogonal |
| R5 | Readout-alone / amplifier model | SC-QMRB safe despite identical readout geometry to CA-01 |
| R6 | Feature plurality as universal attractor origin | Most multi-attractor families show rotational degeneracy, not switching |

---

*Full ruled-out packet: `03_ruled_out.md`. Full behavioral findings: `01_where_the_research_stands.md` §7.*
