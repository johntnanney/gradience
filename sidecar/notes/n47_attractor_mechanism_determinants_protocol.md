# n47 — Attractor Mechanism Determinants: Protocol

**Type:** protocol
**Date:** 2026-03-28
**Depends on:** n46 (mechanism classification), n44 (decision-axis findings), n41 (attractor mapping)
**Status:** Stage B protocol for the Attractor Mechanism Determinants program.

---

## Objective

Identify which factors best explain attractor mechanism choice. Stage A
(n46) established the classification. This stage asks **why** each
family falls into its mechanism class.

---

## 1. Candidate determinants

### Factor 1 — Task family

Does the task's intrinsic structure determine mechanism class? Some
tasks may admit multiple valid feature decompositions (favoring
feature-set switching or rotational degeneracy), while others may have
a single dominant discriminative direction (favoring single-attractor).

**Operationalization:** Compare the same task across backbones and
training conditions. If the task always has the same attractor class
regardless of backbone, task identity is the primary determinant.

### Factor 2 — Backbone architecture

Does backbone depth/width determine which mechanism appears? Deeper
backbones (more layers, more parameters) may provide richer
representations that support feature-set switching, while shallower
backbones may compress representations into low-rank subspaces where
rotational degeneracy is the only available multi-attractor mechanism.

**Operationalization:** Compare the same task across DistilBERT (6
layers) and RoBERTa (12 layers). Record whether mechanism class
changes.

**Proxy metric:** PC effective rank of the decision axis. Higher rank
indicates the decision axis engages more principal components — a
signature of richer representation access.

### Factor 3 — Training convergence / strength

Does training depth change attractor count or mechanism? Longer
training may concentrate the decision axis (lowering effective rank)
and open rotational degeneracy, or it may open access to secondary
attractor basins entirely.

**Operationalization:** Compare Strong QNLI (multi-attractor,
rotational degeneracy) vs Medium QNLI (single-attractor) vs Weak QNLI
(single-attractor). Same task, same backbone (DistilBERT), same seeds,
different training depth.

### Factor 4 — Domain / corpus structure

Does the training domain affect mechanism? Domain-shifted training
(SST-2 domain) might change the decision-axis geometry relative to
standard training (SST-2 core).

**Operationalization:** Compare SST-2 (single-attractor) vs SST-2
domain (multi-attractor, rotational degeneracy). Same backbone
(DistilBERT), different training distribution.

### Factor 5 — Representation richness proxy

Can a lightweight metric from existing outputs predict mechanism class?

**Candidate metrics:**
- Mean PC effective rank of seeds' decision axes
- Energy overlap between seeds
- Top-1 PC loading concentration

These are derived from n44 outputs. No new computation required.

---

## 2. Required contrasts

### Contrast 1 — Backbone contrast

**Comparison:** Same task across backbones.
**Cases:**
- QNLI: DistilBERT (rotational_degeneracy) vs RoBERTa (feature_set_switching)
- MRPC: DistilBERT (rotational_degeneracy) vs RoBERTa (single_attractor)
- RTE: DistilBERT (single_attractor) vs RoBERTa (single_attractor)
- SST-2: DistilBERT (single_attractor) vs RoBERTa (single_attractor)

**Question:** Does backbone change mechanism class?

### Contrast 2 — Convergence contrast

**Comparison:** Same task, same backbone, different training depth.
**Cases:**
- Strong QNLI (rotational_degeneracy) vs Medium QNLI (single_attractor) vs Weak QNLI (single_attractor)

**Question:** Does training depth change attractor count, mechanism, or both?

### Contrast 3 — Domain / label contrast

**Comparison:** Same broad task type, different training distribution.
**Cases:**
- SST-2 core (single_attractor) vs SST-2 domain (rotational_degeneracy)
- SST-2 core vs Yelp (single_attractor) vs Amazon (single_attractor)

**Question:** Does domain structure affect mechanism independent of task?

### Contrast 4 — Task-family contrast

**Comparison:** Different tasks on the same backbone.
**Cases:**
- QNLI vs RTE vs MRPC vs SST-2 on DistilBERT
- QNLI vs RTE vs MRPC vs SST-2 on RoBERTa

**Question:** Is task identity the dominant factor when backbone is held constant?

---

## 3. Required outputs

### A. Family factor table

For each family×backbone condition, record:
- Attractor class
- Mechanism class
- Backbone effect (Y/N/NA)
- Convergence effect (Y/N/NA)
- Domain effect (Y/N/NA)
- Representation richness proxy value

### B. Determinant matrix

Rows = families×backbone conditions
Columns = candidate determinants
Values = `present` / `absent` / `unclear`

### C. Determinant synthesis

Short narrative explaining:
- Which factors have the strongest support
- Which are only weakly supported
- What remains confounded or undertested

---

## 4. Required figures

### Figure 1 — Mechanism map by family and backbone

Visualization showing each family×backbone condition positioned by
mechanism class, with task identity as grouping and backbone as
color/shape.

### Figure 2 — Determinant matrix heatmap

Heatmap of the determinant matrix (families × factors).

### Figure 3 — Convergence contrast panel

QNLI convergence series (Weak → Medium → Strong) showing:
- Effective axis cosine (attractor status)
- Energy overlap (mechanism signature)
- Mean PC effective rank (axis concentration)

---

## 5. Interpretation framework

### Strong positive outcome

A structured determinant hierarchy emerges. For example: backbone
architecture (representation depth) is the primary mechanism
determinant, task identity determines attractor *count*, and training
depth modulates both.

### Mixed outcome

Some determinants explain some families but no clean hierarchy. For
example: backbone explains the QNLI mechanism split but convergence
and domain effects operate through different pathways.

### Negative outcome

Mechanism choice appears too panel-specific to support a generalizable
hierarchy. The backbone confound cannot be disentangled from task
effects with the current panel.

All three outcomes are scientifically useful. The negative outcome
rules out a tempting explanatory path; the mixed outcome identifies
which specific interactions matter; the positive outcome advances
commensurability theory.

---

## 6. Success criteria

Stage B succeeds if it produces:
1. A complete family factor table with no missing entries
2. A determinant matrix covering all 5 factors
3. An honest synthesis that distinguishes supported claims from
   speculation
4. At least one figure visualizing the mechanism landscape
