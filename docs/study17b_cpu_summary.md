# Study 17B CPU Summary (Structural Phase)

Date: March 10-11, 2026 (CDT)  
Scope: CPU-only structural phase for Study 17B (`full_normeq`, `comp90_normeq`, `comp80_normeq`)

## Data Sources

- `results/study17b/study17b_results.json` (full 2-pair CPU structural run)
- `results/study17b_pair04_80_isolated/study17b_results.json` (isolated rerun for pair_04 @ 0.80)
- `results/study17b_cpu_structural_freeze_20260310_192206/study17b_cpu_structural_freeze.json`

## Completed on CPU

- `pair_03` (`magicoder-r16 x btgenbot-r8`)
- `pair_04` (`openwebmath-r64 x btgenbot-r8`)
- Conditions run: `full_normeq`, `comp90_normeq`, `comp80_normeq`
- Structural endpoints populated: `Q_min`, `D`, compression intensity, distortion

## Not Yet Completed (GPU / Deferred)

- Behavioral evaluation (GSM8K, MBPP, OASST2; perplexity/loss endpoints)
- Secondary strategy family (`full_recommended`, `comp90_recommended`, `comp80_recommended`)
- Optional control pair (`pair_02`)

## Core Structural Results

| Pair | Condition | Q_min | D | dQ_min vs full | dD vs full |
|------|-----------|------:|--:|---------------:|-----------:|
| pair_03 | full_normeq | 0.561934 | 0.152360 | --- | --- |
| pair_03 | comp90_normeq | 0.557160 | 0.144496 | -0.004774 | +0.007864 |
| pair_03 | comp80_normeq | 0.546543 | 0.130607 | -0.015391 | +0.021753 |
| pair_04 | full_normeq | 0.659581 | 0.022654 | --- | --- |
| pair_04 | comp90_normeq | 0.659325 | 0.000171 | -0.000256 | +0.022483 |
| pair_04 | comp80_normeq | 0.624146 | 0.027612 | -0.035435 | -0.004958 |

Interpretation of deltas:
- `dQ_min > 0` is better.
- `dD > 0` means lower dominance gap (better balance).

## Compression Intensity and Distortion

| Pair | Threshold | A nom->eff | B nom->eff | A rank red. | B rank red. | A distortion | B distortion |
|------|-----------|-----------:|-----------:|------------:|------------:|-------------:|-------------:|
| pair_03 | 90% | 16->14 | 8->6 | 12.5% | 25.0% | 0.198155 | 0.139465 |
| pair_03 | 80% | 16->11 | 8->5 | 31.25% | 37.5% | 0.343374 | 0.185065 |
| pair_04 | 90% | 64->57 | 8->6 | 10.94% | 25.0% | 0.276853 | 0.139465 |
| pair_04 | 80% | 64->50 | 8->5 | 21.88% | 37.5% | 0.400794 | 0.185065 |

## Aggregate CPU Structural Readout

- `comp90_normeq` vs full: mean `dQ_min = -0.0025` (0/2 improved), mean `dD = +0.0152` (2/2 improved)
- `comp80_normeq` vs full: mean `dQ_min = -0.0254` (0/2 improved), mean `dD = +0.0084` (1/2 improved)

## What This Says So Far

- No threshold improved `Q_min` on CPU structural metrics.
- 90% showed a possible structural-balance effect (`D` improved in both pairs), but with flat/slightly worse `Q_min`.
- 80% appears over-aggressive for this setup (larger `Q_min` drops; pair_04 also worsened `D`).

## Isolated pair_04 @ 0.80 Check

Isolated rerun (`results/study17b_pair04_80_isolated/study17b_results.json`) reproduced the same direction:
- full: `Q_min=0.659277`, `D=0.023047`
- comp80: `Q_min=0.624260`, `D=0.027353`
- deltas: `dQ_min=-0.035017`, `dD=-0.004306`

This is directionally consistent with the full run.

## Protocol/Interpretation Caveats

- Study 17B protocol frames the final decision around structural + behavioral tradeoff; behavioral side is still pending GPU.
- Source-QA boundary: `btgenbot-r8` is currently `flagged_weak` in `examples/qa/btgenbot_r8_qa.json`, so pair_03 and pair_04 should be interpreted as boundary/workflow-limit cases rather than clean primary efficacy evidence until that is resolved.
