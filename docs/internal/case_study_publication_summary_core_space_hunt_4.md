# Publication Summary — core_space_hunt_4

Series: wave 2, Target 4
RQ: RQ3 — How often does core-space change a real decision when ordinary pair risk is low?
Date: 2026-03-22

## One-line summary

A full 6-pair core-space census on all-eligible adapters found every pair incompatible or marginal — raising the question of whether the diagnostic is correctly calibrated.

## Setup

- 4 distilbert-base-uncased adapters, all eligible
- 2 generic fine-tuning (r=16): final_uniform_median_r16, priority_probe_r16
- 2 QNLI (r=32): qnli_probe_elig, qnli_uniform_elig
- Core-space run on all 6 pairs, not just selected ambiguous ones

## What happened

1. Source QA: nothing (all eligible). Pair audit: 5 low-risk, 1 medium.
2. Core-space: 4 incompatible (0.807–0.860), 2 marginal (0.863–0.864). No pair compatible.
3. Neighborhoods: all 4 in one caution group.

## The surprise

**Same-task pairs are not safe at depth.** qprobe × quniform (both QNLI, both r=32) had the lowest shared_basis_score in the inventory (0.807). Different rank allocation policies (probe vs uniform) override task similarity.

**Same-group pairs are not safe either.** final × priority (both generic, both r=16) scored 0.824 — incompatible.

## Where the workflow was strong

Core-space was the only diagnostic that flagged structural problems in this inventory. Without it, 5 of 6 pairs look safe. Source QA, pair audit, and neighborhoods all say "proceed." However, verified adjudication (2026-03) later showed that core-space structural flags on same-task pairs do not predict behavioral harm — same-task merges were safe even when flagged as incompatible.

## Where the workflow starts to strain

When core-space flags everything, its discriminative value drops. The diagnostic is most useful when it distinguishes among pairs — here it flags all uniformly. Either the threshold needs investigation, or this adapter pool is genuinely structurally fractured despite surface compatibility.

## Inventory-level lesson

This inventory raised the calibration question that verified adjudication later answered. The downstream merge evaluation was performed (2026-03): same-task merges flagged as incompatible did NOT fail in practice. Cross-task merges did degrade, but ordinary pair-risk already separated them. Core-space finds real structural divergence in this adapter pool, but that divergence is not behaviorally decisive for same-task pairs. The diagnostic's strongest supported role is in genuinely ambiguous task relationships where pair-risk is permissive.
