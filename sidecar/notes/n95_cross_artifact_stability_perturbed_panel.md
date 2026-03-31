# n95 — Cross-Artifact Stability Check: Perturbed Panel

**Type:** panel definition
**Date:** 2026-03-31
**Program:** Route 2 Stability and Replication Check, Substudy 1
**Depends on:** n94 (original panel freeze), n76-n80 (cross-artifact program)
**Status:** Panel constructed. Ready for signal audit.

---

## Perturbation design

Four substitutions across two artifact classes. LoHa unperturbed (no alternatives exist).

### Substitution 1 — LoRA same-task: different adapter pair
**Original:** inventory_t01 SST-2 pair. **Perturbed:** inventory_a01 SST-2 pair (myselfmankar x rambodazimi).
**Rationale:** Tests whether the same-task LoRA slot depends on specific adapters or the task relation.

### Substitution 2 — LoRA same-family: different task family
**Original:** MNLI x QNLI (NLI family, structural_only). **Perturbed:** SST-2 x IMDB (sentiment_binary family, behavioral_reported).
**Rationale:** The strongest perturbation. Changes both task family and evidence regime. If same-family intermediate status survives, it is genuinely task-family-general.

### Substitution 3 — Checkpoint same-family: different seed
**Original:** SST-2 s42 x Yelp (compat=0.652). **Perturbed:** SST-2 s123 x Yelp (compat=0.641).
**Rationale:** Minimal structural change. Tests whether the same-family checkpoint slot is seed-sensitive.

### Substitution 4 — Checkpoint cross-task: different cross task
**Original:** SST-2 x QNLI (compat=0.626, risk=high). **Perturbed:** SST-2 x MRPC (compat=0.798, risk=medium).
**Rationale:** The hardest test for task-relation ordering. MRPC is structurally much closer to same-task (gap: 0.094) than QNLI was (gap: 0.266). If the ordering still holds, it is robust to structurally close cross-task pairs.

---

## Panel coverage after perturbation

| Relation | LoRA | LoHa | Ckpt delta |
|----------|------|------|-----------|
| same_task | 1 (changed) | 3 (retained) | 1 (retained) |
| same_family | 1 (changed) | 0 | 1 (changed) |
| cross_task | 1 (retained) | 0 | 1 (changed) |

Coverage structure is preserved. The same slots exist in the same positions.

---

## Data locations

- Perturbed panel: `results/route2_stability/cross_artifact/perturbed_panel_table.json`
- Panel diff: `results/route2_stability/cross_artifact/panel_diff_table.md`
