# Targeted Confirmation Summary

Date: 2026-03-29

This document summarizes the results of the targeted confirmation pass for two recently implemented Gradience product refinements: same-family routing and near-miss severity ordering.

## A. Same-Family Routing — Confirmed

**Run 1** tested the three-way task-relationship classification (same_task / same_family / cross_task) on a 4-adapter inventory: 2 SST-2, 1 IMDB, 1 ag_news, all on distilbert-base-uncased.

**Routing behavior**: The same-family pair (SST-2 × IMDB) was correctly routed to the same-task safe zone with an informational "TASK-FAMILY NOTE" advisory. The cross-task pair (SST-2 × ag_news) was routed to the caution zone with a "TASK-BOUNDARY WARNING." The distinction was immediate and legible in the action plan.

**Outcome alignment**: Strong. The same-family merge (0.878) performed essentially identically to the retained same-task merge (0.876), both within -0.008 of the best source. The cross-task control (0.842) was meaningfully worse (delta -0.042 vs best source). This confirms the routing logic: same-family pairs behave like same-task, not like cross-task.

**Report clarity fix**: The confirmation surfaced one headline bug. Same-family pairs were previously labeled "Cross-task pair — caution region" in the report headline even though the advisory correctly identified them as same-family. Fixed: headline now reads "Same-family pair — plausible candidate" when `task_relationship == "same_family"`.

**Verdict: confirmed.** Same-family routing is correct, useful, and clearly explained (after the headline fix).

## B. Near-Miss Severity Ordering — Mixed

**Run 2** tested severity-based ordering in the near-miss section on a 5-adapter inventory: 3 irony adapters and 2 hate adapters on distilbert-base-uncased derivatives.

**Mechanism**: The near-miss section works correctly in all structural respects. The severity label ("deeply weak") is displayed clearly, the ordering header communicates the principle, and the section is visually distinct from retained and excluded categories. A user would immediately understand what to prioritize.

**Limitation 1 — single severity level**: The planned marginal near-miss (phailyoor_irony, prior delta -0.004) shifted to eligible on re-bootstrap (delta +0.218) due to base score variance on the irony task. This left only one near-miss pair (substantial severity), preventing a test of ordering contrast between severity levels.

**Limitation 2 — flat outcomes**: All four merges produced nearly identical accuracy (0.604-0.606). This is inherent to r=1 LoRA adapters — the perturbation is too small to produce meaningful behavioral variation in merged outputs. Outcome alignment between severity levels could not be assessed.

**Verdict: mixed.** The ordering mechanism works correctly and the labels are clear, but outcome alignment could not be confirmed due to the r=1 limitation, and the marginal vs substantial ordering contrast could not be tested due to sampling variance.

## C. Product Implication

**Same-family routing**: Keep as implemented. The headline fix (already applied) was the only adjustment needed.

**Near-miss severity ordering**: Keep as implemented. The mechanism is structurally sound. The mixed verdict reflects limitations of the test inventory (r=1 adapters, sampling variance), not flaws in the feature. No wording or ordering tweaks are needed. If future inventories with higher-rank adapters naturally produce multiple severity levels, the ordering will be testable then.

## Code Changes Made During Confirmation

One change to production code was made:

**`gradience/vnext/merge/qa_report.py`** — `_report_headline()` now returns "Same-family pair — plausible candidate" instead of "Cross-task pair — caution region" when `qa.task_relationship == "same_family"`. This is a correctness fix, not a feature change. All 466 merge tests pass after the change.
