# What Would Make the Next Leap Possible

**Last updated:** 2026-03-31

The project is at a natural inflection point. The stable product works. The research program has a coherent theoretical account. The broadened Route 2 framework is validated in bounded scope. What follows describes the specific things that would move the project forward — and what each one would unlock.

---

## What GPU compute would unlock

### The DeBERTa adjudication (~3 hours on one consumer GPU)

This is the single highest-leverage thing that could happen to the project. Everything else is either CPU-feasible or waiting for this.

**What it tests:** Whether the core mechanism ladder — instability as a portable descriptor, V-module pathology as catastrophe discriminator, head-level cancellation as the seed-sensitivity mechanism — transfers to a third backbone with fundamentally different attention (DeBERTa's disentangled content/position projections).

**What it requires:** Train 8 adapters (4 GLUE tasks x 2 seeds), merge 28 pairs, evaluate 56 conditions. The protocol is pre-registered with five predictions (A–E), a decision tree for every outcome combination, and a pre-run checklist. It is executable as-is.

**What passes would unlock:**

| Prediction | What it tests | What passing unlocks |
|-----------|---------------|---------------------|
| A–C | Instability rankings preserved on 3rd backbone | Instability promoted from "consistent on 2 backbones" to "architecturally portable" |
| D | V-module dim ratio still separates catastrophic from safe | V-module ratio promoted to computable warning signal in the product |
| E | Head-level cancellation recurs | O-module escalation design (the next mechanistic rung) |

**What failure would mean:** Not the end of the project — the decision tree has productive paths for every outcome. D-pass/E-fail means the module-level signal transfers but head-level modulation may be backbone-specific. A-fail means instability needs recalibration. The worst case (A-fail, D-fail) triggers a formal reassessment, but even then, the product's evidence gating and task-boundary detection remain unaffected.

**Why it's blocked:** No GPU currently available. The protocol has been ready since it was written. This is pure compute constraint, not a design or data problem.

### What else GPU enables

- **GPU-accelerated evidence evaluation.** Currently CPU-only with 200–500 sample evaluations. GPU on full validation sets would tighten the evidence gate and reduce the marginal-adapter problem (adapters that barely beat base but pass the binary gate).
- **High-rank adapter validation.** Most field trial adapters are rank 1. The spectral layer's full diagnostic power appears at rank >= 8. GPU training of higher-rank adapters would test whether spectral analysis adds value beyond what the evidence gate already provides.
- **Decoder model validation.** All current evidence is on small encoders (DistilBERT, RoBERTa, BERT-class). Breaking out of the encoder/classification regime requires GPU training and evaluation.

---

## What an external workflow owner would unlock

### The checkpoint triage alpha needs a real user

The alpha workflow is functional. It has a canonical instance, an HTML report, a scope contract, and a README. What it does not have is a user who didn't build it.

**What a workflow owner would provide:**

- **A real inventory problem.** The canonical T02 trial uses 5 checkpoints on a base model we chose. A workflow owner brings their own checkpoints with their own base model and their own evaluation criteria. This tests whether the workflow is actually useful, not just technically correct.

- **The external pull test.** The project's standing rule: run one additional checkpoint-inventory deployment only when externally motivated by a real manual inventory problem. If no concrete workflow owner exists, the alpha stays where it is. A workflow owner converts "validated experiment" into "useful tool."

- **Feedback on the scope contract.** The alpha is bounded to shared-base, small encoders, classification. A workflow owner tells us which bound they hit first — and whether loosening it is worth the engineering.

### What this looks like concretely

Someone who has 5–20 fine-tuned checkpoints from the same base model and wants to know which pairs are worth exploring. They run the trial script, look at the action plan, and tell us: was this faster than what you were doing before? Did it get the triage right? What did it miss?

The minimal version is one conversation and one trial run. No code contribution required.

---

## What a collaborator could contribute

### GPU time (highest value, lowest coordination cost)

Run the DeBERTa adjudication. The protocol is fully specified. The output is a JSON file with merge results for 28 pairs. This is the single most valuable external contribution — it unblocks everything in the "what GPU would unlock" section above.

**Time:** ~3 hours. **Coordination:** Minimal — execute the script, return the output.

### A second checkpoint triage deployment

Run the alpha workflow on a different set of checkpoints. Ideally: different base model, different task mix, or larger inventory (10+ checkpoints). Report what worked and what didn't.

**Time:** 1–2 hours. **Coordination:** Low — follow the README, share the action plan output and your assessment.

### Broader adapter validation

Run the stable product on adapters outside the current validated envelope:

- **Higher-rank adapters** (r=8, r=16, r=32) on the same tasks — tests whether spectral analysis adds triage value beyond evidence gating
- **Different task types** (NER, extractive QA, summarization) — tests whether task-boundary detection works outside classification
- **Larger models** (BERT-large, DeBERTa-base without the mechanism-ladder protocol) — tests whether the triage workflow scales

**Time:** Variable. **Coordination:** Medium — need to agree on evaluation protocol.

### Routing-confusability behavioral experiment

The routing pilot validated that the spectral substrate works for routing. But the behavioral signature of routing-confusability is untested — confusable pairs look safe in merge but may look confused in actual routing. Someone with a routing setup could test this.

**Time:** Moderate (requires routing infrastructure). **Coordination:** High — need to define what "confusion" looks like in a routing context.

### Example-level spectral correlation (CPU-feasible)

The behavioral bridge is currently interpretive: neither-source examples are predicted to cluster at layers where V-module dim ratio is lowest, but this correlation has not been directly measured. The data exists (500 examples per case with per-example predictions, plus per-layer spectral profiles). This is a CPU analysis that could convert the bridge from interpretive to quantitative.

**Time:** A few hours. **Coordination:** Low — data is in `sidecar/results/`, analysis script needs to be written.

---

## What questions are blocked vs not blocked

### Blocked on GPU (~3 hours)

| Question | Why GPU | What it unlocks |
|----------|---------|----------------|
| Is the mechanism ladder architecture-general? | DeBERTa training + evaluation | Promotion of V-module signal to product |
| Is instability portable to 3 backbones? | DeBERTa training + evaluation | Instability as an architectural invariant |
| Does head-level cancellation recur? | DeBERTa per-head analysis | O-module escalation design |
| Is the backbone confound in attractor mechanisms real? | 3rd backbone breaks the confound | Clean mechanism taxonomy |

### Blocked on external use (no GPU needed)

| Question | What's needed | What it unlocks |
|----------|--------------|----------------|
| Is checkpoint triage useful to someone who didn't build it? | One external deployment | Alpha → stable promotion path |
| Does the workflow scale to 50+ pairs? | Larger inventory test | Scale ceiling validation |
| Does task-boundary detection work for non-classification? | Adapters from different task types | Broader product scope |

### Not blocked (CPU-feasible, data exists)

| Question | What's needed | Why it hasn't been done |
|----------|--------------|----------------------|
| Do neither-source examples cluster at low-ratio layers? | Analysis script (~2h) | Prioritized behind other programs |
| Can marginal adapters be graduated instead of binary-gated? | Threshold analysis on existing field trial data | Design question, not data question |
| Does norm-imbalance severity ranking add triage value? | Reanalysis of existing pair data | Engineering, not research |

### Not blocked but needs new data

| Question | What's needed | Why it hasn't been done |
|----------|--------------|----------------------|
| Does cross-task contamination generalize beyond one case? | Additional catastrophic cross-task pairs from field trials | Single-case evidence base is thin |
| Do behavioral tiers transfer across artifact classes? | Per-example data for LoHa or checkpoint delta merges | No non-LoRA behavioral data exists |
| Does routing-confusability have a behavioral signature? | Routing infrastructure + confusable adapter pairs | Requires routing setup |

---

## The minimal path forward

If you could do exactly three things:

1. **Run the DeBERTa adjudication** (~3 hours GPU). This unblocks mechanism-ladder promotion, V-module signal productization, and O-module design. It is the rate-limiting step for the entire research program.

2. **Find one checkpoint triage user.** One real inventory, one real action plan, one real assessment. This converts the alpha from "technically validated" to "useful to someone."

3. **Write the example-level spectral correlation script** (~2 hours CPU). This converts the behavioral bridge from interpretive to quantitative and is the strongest CPU-feasible research contribution remaining.

Everything else — decoder models, larger inventories, routing behavioral experiments — is valuable but secondary. These three would move the project further than any combination of alternatives.
