# Source QA Workflow

Assess each adapter's standalone quality before merging.

## Why source QA comes first

Study 16 showed that structural merge recommendations can be locally correct
while operating on globally weak source adapters. A merge can be spectrally
balanced yet behaviorally worthless if neither adapter outperforms the base model.

Therefore: assess source-adapter eligibility *before* requesting pairwise merge
recommendations. `merge-audit` accepts QA artifacts from `audit-adapter` and
adjusts its warnings and policy accordingly.

## Basic workflow

Three steps: audit each adapter, inspect eligibility, merge-audit with QA context.

### Step 1: Audit each adapter

```bash
gradience audit-adapter \
    --peft-dir ./adapters/catsubcat-r16 \
    --base-model meta-llama/Llama-2-7b-hf \
    --eval-dataset oasst2 \
    --metric-name perplexity \
    --adapter-score 6.81 \
    --base-score 4.66 \
    --lower-is-better \
    --out catsubcat_qa.json

gradience audit-adapter \
    --peft-dir ./adapters/btgenbot-r8 \
    --base-model meta-llama/Llama-2-7b-hf \
    --eval-dataset oasst2 \
    --metric-name perplexity \
    --adapter-score 5.47 \
    --base-score 4.66 \
    --lower-is-better \
    --out btgenbot_qa.json
```

Each command produces:
- A terminal summary showing structural metrics, behavioral comparison, and eligibility status
- A JSON artifact (`gradience.adapter_qa/v1`) written to the `--out` path

### Step 2: Inspect eligibility

```bash
# Quick check
cat catsubcat_qa.json | python -m json.tool | grep -A5 '"eligibility"'
```

Output:
```json
"eligibility": {
    "status": "flagged_weak",
    "confidence": "medium",
    "reasons": [
        "adapter underperforms base on perplexity (oasst2)",
        "low utilization across layers",
        "high rank waste"
    ]
}
```

### Step 3: Merge audit with QA context

```bash
gradience merge-audit \
    --adapter-a ./adapters/catsubcat-r16 \
    --adapter-b ./adapters/btgenbot-r8 \
    --source-a-qa catsubcat_qa.json \
    --source-b-qa btgenbot_qa.json \
    --emit-report pair06_report.json
```

The merge audit now sees that both sources are flagged weak and adjusts warnings accordingly.

## Interpreting a weak-source merge

When both adapters are flagged weak (Pair 06 archetype):

- **Structural audit still runs.** Per-layer verdicts (SAFE, REDUNDANT, CONFLICTING, IMBALANCED) are computed as usual.
- **Merge strategy is still recommended.** Structural rebalancing may be available.
- **But warnings now reflect source quality:**
  - "Source adapter A is flagged as weaker than the base model on perplexity (oasst2)"
  - "Source adapter B is flagged as weaker than the base model on perplexity (oasst2)"
  - "Both source adapters are flagged as weaker than the base model. The merge problem may be ill-posed."
- **Deployment interpretation:** Do not assume the merged adapter is worth deploying. A structurally clean merge of two weak adapters is still a weak adapter.

The merge recommendation tells you *how* to merge. Source QA tells you *whether* to merge.

## Strict QA gating

Use `--strict-qa` to prevent merge recommendations when sources are weak:

```bash
gradience merge-audit \
    --adapter-a ./adapters/catsubcat-r16 \
    --adapter-b ./adapters/btgenbot-r8 \
    --source-a-qa catsubcat_qa.json \
    --source-b-qa btgenbot_qa.json \
    --strict-qa
```

Behavior:
- The structural audit still runs (you still get per-layer diagnostics).
- But if any source adapter is `flagged_weak`, the command exits with an error:

```
Error: --strict-qa gate failed. Adapter(s) A, B flagged as weak.
  Recommendations withheld. Review source adapter quality before merging.
```

`--strict-qa` also requires that *both* adapters have QA data. If you omit `--source-a-qa` or `--source-b-qa`, it will tell you:

```
Error: --strict-qa requires source QA data for both adapters.
  Provide --source-a-qa and --source-b-qa, or remove --strict-qa.
```

This is intentional. Strict mode means: "I want guarantees before proceeding." No QA data means no guarantees.

## Balanced pair (happy path)

When both adapters beat the base model (eligible status):

```bash
gradience audit-adapter \
    --peft-dir ./adapters/good-adapter-a \
    --eval-dataset hellaswag \
    --metric-name accuracy \
    --adapter-score 0.78 \
    --base-score 0.72 \
    --higher-is-better \
    --out good_a_qa.json

gradience audit-adapter \
    --peft-dir ./adapters/good-adapter-b \
    --eval-dataset hellaswag \
    --metric-name accuracy \
    --adapter-score 0.81 \
    --base-score 0.72 \
    --higher-is-better \
    --out good_b_qa.json

gradience merge-audit \
    --adapter-a ./adapters/good-adapter-a \
    --adapter-b ./adapters/good-adapter-b \
    --source-a-qa good_a_qa.json \
    --source-b-qa good_b_qa.json \
    --strict-qa
```

In this case:
- Both adapters are `eligible` with `high` confidence
- No eligibility warnings are emitted
- `--strict-qa` passes without error
- The merge recommendation is based purely on structural analysis
- This is the boring, good outcome

## Without behavioral evaluation

If you don't have evaluation scores, `audit-adapter` still works:

```bash
gradience audit-adapter \
    --peft-dir ./adapters/unknown-adapter \
    --out unknown_qa.json
```

The artifact will show:
- `status: "unknown"` — no behavioral evaluation available
- `confidence: "low"` — structural evidence only
- Structural flags are still computed and reported

When fed into `merge-audit`, this produces a warning:

```
No behavioral evaluation available for adapter A. Merge recommendation is based
on structural analysis only.
```

This is the honest default. The system does not pretend to know more than it does.

## Schema reference

See [Adapter QA v1 Schema](schemas/adapter_qa_v1.md) for the complete field reference.

## Example artifacts

- [`examples/qa/catsubcat_r16_qa.json`](../examples/qa/catsubcat_r16_qa.json) — Flagged weak, behavioral eval available
- [`examples/qa/btgenbot_r8_qa.json`](../examples/qa/btgenbot_r8_qa.json) — Flagged weak, concentrated spectrum
- [`examples/qa/eligible_adapter_qa.json`](../examples/qa/eligible_adapter_qa.json) — Eligible, beats base on accuracy
- [`examples/qa/structural_only_qa.json`](../examples/qa/structural_only_qa.json) — No behavioral eval, structural flags only
