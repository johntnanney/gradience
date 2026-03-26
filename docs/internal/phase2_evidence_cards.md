# Phase 2 Evidence Cards — Side-by-Side

## Summary table

| | T5: Control | T1: Behavioral | T4: Core-space | T2: Neighborhood | T3: Messy |
|---|---|---|---|---|---|
| **Adapters** | 3 | 5 | 4 | 7 | 6 |
| **QA composition** | 3 elig | 4 elig + 1 weak | 4 elig | 5 elig + 2 unk | 3 elig + 2 unk + 1 weak |
| **Total pairs** | 3 | 10 | 6 | 21 | 15 |
| **Strict-QA survivors** | 3 | 6 | 6 | 10 | 3 |
| **Risk: low/med/high** | 2/1/0 | 6/4/0 | 5/1/0 | 18/3/0 | 9/2/4 |
| **Core-space used** | no | yes (3) | yes (6) | yes (2) | no |
| **Core-space changed** | — | 3/3 | 5/5 | 2/2 | — |
| **Neighborhood groups** | 1 | 1+1excl | 1 | 3+3bw | 3+1excl+3bw |
| **Action reduction** | 3→3 | 10→3 | 6→0 | 21→1 group | 15→1 group |

## Key observations across all 5

1. **Source QA dominance is regime-dependent.** In messy pools (T3: 80% of narrowing), QA dominates. In behaviorally complete pools (T1: 57%), it shares the work. In all-eligible pools (T5, T4), it does nothing.

2. **Core-space changed structural judgments consistently but verified adjudication narrowed its behavioral claim.** When used in the case series, it changed judgment 10/10 times across 3 inventories. But verified adjudication (2026-03) showed same-task merges were safe even when flagged as incompatible, and cross-task degradation was already captured by ordinary pair-risk. Its behaviorally useful role is narrower and more regime-dependent than this table alone suggests.

3. **Neighborhoods compress but reflect QA status.** At 7 adapters (T2), real compression (21→3 groups). But grouping follows QA status, not structural similarity.

4. **No high-risk pairs appeared outside the r=2 per-layer adapter.** All 4 high-risk pairs in T3 involved final_per_layer_ckpt50. The other 4 inventories had 0 high-risk pairs.

5. **The workflow adds least value when the pool is clean and same-task (T5).** It adds most value when QA helps AND core-space resolves cross-task ambiguity (T1).
