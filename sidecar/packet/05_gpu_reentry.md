# GPU Re-Entry Note

**Status:** Pre-registered. Executable as-is once compute is available.

---

## What this tests

The DeBERTa-v3-base adjudication is the next decisive empirical step. It tests whether the sidecar's three core findings transfer to a third backbone with a qualitatively different attention mechanism (disentangled attention with separate content and position projections).

Three rungs of the mechanism ladder are independently testable:

**Rung 1 — V-module dimensionality ratio.** The strongest signal in the evidence base (d=3.36, zero range overlap on two backbones). Does it survive when the value projection plays a structurally different role in disentangled attention?

**Rung 2 — Head-level cancellation.** The mechanism that explains seed sensitivity (CA-01's 29pp gap localizes to 7 heads with opposite-sign deltas). Does it recur on DeBERTa, or is it specific to standard attention?

**Rung 3 — Readout gating / O-module escalation.** Not directly tested. The joint outcome of Rungs 1–2 determines whether O-module analysis is the confirmed next step.

---

## Protocol summary

**Training:** 8 LoRA adapters on DeBERTa-v3-base — 4 tasks (QNLI, RTE, MRPC, SST-2) × 2 seeds (42, 7). LoRA config: rank 16, alpha 16, dropout 0.1, all four attention modules (Q/K/V/O), 3 epochs.

**Merging:** All 28 pairs (24 cross-task + 4 same-task controls), linear merge at equal weights.

**Evaluation:** Both source tasks per merge. Severity classification using P01 thresholds (catastrophic >15%, severe 10–15%, broad 5–10%, mild ≤5%).

**Compute:** ~3 hours on a single consumer GPU. Training ~1.5h, merging ~0.5h, evaluation ~1h.

---

## Pre-registered predictions

| # | Prediction | PASS criterion | FAIL criterion |
|---|-----------|---------------|----------------|
| A | Instability ranking preserved | QNLI×MRPC and QNLI×SST-2 have the two highest DeBERTa seed ranges, each ≥2× median | Any other pair has higher seed range than both |
| B | Stable cluster preserved | All four stable-asymmetric pairs: seed range <10%, no severity class reversal | Any stable pair: seed range >15% or reversal |
| C | Gap preserved | No pair's three-backbone instability score falls in [0.30, 0.70] | Two or more pairs in that range |
| D | V-module dim ratio survives | Catastrophic collision: ratio <0.75; safe collision: ratio >0.78; zero overlap | Overlapping ranges between catastrophic and safe |
| E | Head-level cancellation recurs | ≥3 head×layer positions with \|Δ_DR\|≥0.10, opposite signs, module-level Δ<0.05 | No seed-sensitive case, or uniform small deltas |

---

## Decision tree

| Outcome | Interpretation | Next step |
|---------|---------------|-----------|
| **A–C + D + E all PASS** | Full mechanism ladder transfers. Architecturally generic. | Promote V-module ratio to computable warning signal. Design O-module analysis (Rung 3). Write up multiscale mechanism as standalone finding. |
| **D PASS, E FAIL/untestable** | Module-level risk transfers; head-level modulation may be backbone-specific. | Promote V-module signal. Investigate whether disentangled attention produces a different modulation pattern. |
| **D PASS, A–C mixed** | Structural signal more robust than retrospective descriptor. | Prioritize V-module over instability. Recalibrate instability formula. |
| **A–C PASS, D FAIL** | Descriptor portable; mechanism needs revision for disentangled attention. | Confirm instability. Search for replacement discriminator module on DeBERTa. |
| **A FAIL** | Instability rankings not portable. Framework needs fundamental revision. | Write closure note. Assess whether V-module signal (if D passed) supports per-backbone use. Do not abandon — revise. |

---

## Blockers and contingencies

**Blocker:** GPU compute. The protocol is CPU-infeasible — training 8 adapters requires GPU.

**Contingency (Prediction D):** If no DeBERTa pair is both catastrophic and collision-prone, fall back to testing whether V-module dim ratio correlates with severity within the collision subset (Spearman ρ, expecting ρ < −0.5).

**Contingency (Prediction E):** If no DeBERTa case has seed range >15pp, E is UNTESTABLE. Check whether any case with seed range >8pp shows qualitatively similar head-level structure as a weaker signal.

---

*Full protocol: `sidecar/notes/n07_deberta_adjudication_protocol.md`. Pre-registered predictions reference the frozen glossary (`sidecar/glossary.md`).*
