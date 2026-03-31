# n76 -- Cross-Artifact Portability Panel Definition

**Type:** panel definition
**Date:** 2026-03-31
**Program:** Cross-Artifact Compatibility Research (Route 2)
**Depends on:** Ring 1 (LoHa), Ring 2 (checkpoint deltas), field trials, routing pilot, decision-dependent compatibility (n70-n74), n75 (initial cross-artifact note)
**Status:** complete

---

## Question

Which compatibility signals recur across artifact classes (LoRA, LoHa, checkpoint delta), and which are representation-local?

This panel defines the fixed comparison surface for that question.

---

## Panel design

### Artifact classes

1. **LoRA** -- native low-rank factors (A/B matrices), direct extraction from adapter state dicts.
2. **LoHa** -- Low-Rank Hadamard Product adapters, shimmed to LoRA-format via materialized SVD. Extraction shim: `experiments/peft_ring1/loha_shim.py`.
3. **Checkpoint delta** -- full fine-tuned checkpoint deltas from a shared base model, represented via layerwise summary (Representation C from Ring 2 Stage A). Extraction: `experiments/ring2_checkpoint_delta/extract_checkpoint_delta.py`.

### Shared conditions

- Backbone: `distilbert-base-uncased` (all 9 cases).
- Classification tasks only (SST-2, AG News, MNLI, QNLI, Yelp Polarity).
- CPU-only extraction and analysis.
- No merge execution outside native LoRA.

### Task relation targets

Each class should include same-task, same-family, and cross-task where existing artifacts permit.

---

## Panel cases (9 total)

### LoRA (3 cases)

| Case | Tasks | Relation | Evidence | Scenarios |
|------|-------|----------|----------|-----------|
| lora_same_task_sst2_pair | SST-2 x SST-2 | same_task | behavioral (merge acc 0.876) | merge, triage |
| lora_same_family_mnli_qnli | MNLI x QNLI | same_family | structural + routing | merge, routing |
| lora_cross_task_sst2_agnews | SST-2 x AG News | cross_task | behavioral (merge acc 0.842) | merge, triage |

Source: field trials (targeted confirmation T01), routing pilot.

### LoHa (3 cases)

| Case | Tasks | Relation | Evidence | Scenarios |
|------|-------|----------|----------|-----------|
| loha_same_task_r4_r8 | SST-2 x SST-2 | same_task | structural only | triage |
| loha_same_task_r4_r16 | SST-2 x SST-2 | same_task | structural only | triage |
| loha_same_task_r8_r16 | SST-2 x SST-2 | same_task | structural only | triage |

Source: Ring 1 (`experiments/peft_ring1/`).

**Critical gap:** All LoHa cases are same-task. No same-family or cross-task LoHa pairs exist. This limits cross-artifact comparison to the same-task regime for LoHa.

### Checkpoint delta (3 cases)

| Case | Tasks | Relation | Evidence | Scenarios |
|------|-------|----------|----------|-----------|
| ckpt_same_task_sst2_seeds | SST-2 s42 x SST-2 s123 | same_task | structural only | triage |
| ckpt_same_family_sst2_yelp | SST-2 x Yelp Polarity | same_family | structural only | triage |
| ckpt_cross_task_sst2_qnli | SST-2 x QNLI | cross_task | structural only | triage |

Source: Ring 2 (`experiments/ring2_checkpoint_delta/`), checkpoint inventory T02 (`field_trials/checkpoint_inventory_t02/`).

---

## Coverage analysis

### What is covered

- All three artifact classes represented.
- Same-task relation appears in all three classes (strongest comparison axis).
- Same-family and cross-task appear in two classes (LoRA + checkpoint delta).
- All cases share the same backbone (`distilbert-base-uncased`).
- Structural (spectral/pairwise) outputs exist for all 9 cases.

### What is not covered

- **LoHa is same-task only.** The Ring 1 panel trained only SST-2 adapters at different ranks. Cross-task and same-family LoHa pairs do not exist.
- **Behavioral evaluation is sparse.** Only 2 of 9 cases (LoRA same-task and cross-task) have merge accuracy data. LoHa and checkpoint delta cases have no behavioral evaluation.
- **Compatibility scores are not directly comparable.** LoRA uses factor-based merge-audit scoring, LoHa uses shimmed factor-based scoring, checkpoint deltas use summary-based pairwise scoring. The numerical scales and semantics differ.
- **Scenario coverage is uneven.** LoRA cases have merge and/or routing data. LoHa and checkpoint delta cases have triage only.
- **Single backbone.** All cases use DistilBERT. No cross-backbone comparison within this panel.

### Comparison feasibility by task relation

| Relation | LoRA | LoHa | Ckpt delta | Cross-class comparison feasible? |
|----------|------|------|-----------|--------------------------------|
| same_task | Yes | Yes | Yes | **Yes** -- all three classes |
| same_family | Yes | No | Yes | **Partial** -- two classes only |
| cross_task | Yes | No | Yes | **Partial** -- two classes only |

---

## Relationship to prior panels

This panel draws cases from:
- Decision-dependent compatibility panel (n70): 3 LoRA cases overlap with merge/routing/triage groups.
- Ring 1 inventory pilot: 3 LoHa pairs directly reused.
- Ring 2 Stage B: 2 checkpoint delta pairs directly reused. 1 from checkpoint inventory T02.
- n75 (initial cross-artifact note): this panel formalizes and extends the n75 comparison surface.

---

## Success criteria assessment

| Criterion | Met? |
|-----------|------|
| All three artifact classes represented | Yes |
| Same-task relation in at least two classes | Yes (all three) |
| Same-family relation in at least two classes | Yes (LoRA + ckpt delta) |
| Cross-task relation in at least two classes | Yes (LoRA + ckpt delta) |
| Panel explicit enough for later comparison | Yes |

Stage A is **successful** by the spec's criteria.

---

## Output artifacts

- `sidecar/results/cross_artifact_portability/panel_table.json`
- `sidecar/results/cross_artifact_portability/panel_table.md`
- `sidecar/notes/n76_cross_artifact_panel_definition.md` (this note)
