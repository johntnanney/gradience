# Admissibility sources (DRAFT — pending v1.1 lock)

This appendix discharges the §4.1 admissibility requirement of the
benchmark-reliability pre-registration: every prompt template used in the study
is traceable to a community source — benchmark authors, lm-evaluation-harness,
HELM, a published peer-reviewed variant, or an author-constructed minimal
template declared here.

The four prompts per benchmark are:

- **P1_original** — original benchmark authors' prompt (paper or repo).
- **P2_lm_eval** — `EleutherAI/lm-evaluation-harness` default.
- **P3_helm_or_published** — HELM reference; if HELM does not cover the
  benchmark, a published peer-reviewed variant is used instead.
- **P4_minimal_sourced** — author-constructed minimal variant declared in this
  appendix; the shortest admissible template that preserves task semantics.

## Status

This file is the **draft** version produced before the v1.1 admissibility lock.
Content hashes for all 24 prompt files are recorded as `TODO` in
`configs/prompts.yaml` and will be filled in (and `admissibility_status`
promoted from `draft` to `locked`) at lock time, after the user has reviewed
the drafts.

## Source-URL pinning decisions

The pre-registration requires that source provenance be pinned to specific
commits where possible. The following commits were resolved at sourcing time
(2026-04-24):

- **lm-evaluation-harness** — `c1c4bea3777f73e188395264083adcf454913344`
  (`main` branch HEAD on 2026-04-08).
- **stanford-crfm/helm** — `11937097bd9534e538eaaa31b21197086fc1a113`
  (`main` branch HEAD on 2026-04-23).
- Benchmark-author repositories (`allenai/ai2_arc`, `rowanz/hellaswag`,
  `sylinrl/TruthfulQA`, `hendrycks/test`, `allenai/winogrande`,
  `openai/grade-school-math`) host data, not single canonical prompt
  template files. The original prompt formats are described in the
  benchmarks' papers; we cite the paper and repo together but do not pin a
  single template-file commit. P1 entries therefore have
  `source_commit: null` in `prompts.yaml` and inherit canonical structure
  from the paper text.

## Conventions

All prompts use the placeholder syntax established for this study:

- `{{fewshot_examples}}` — rendered few-shot exemplars block (empty for 0-shot).
- `{{question}}` — the item's question / context / sentence stem.
- `{{choices}}` — a multi-line lettered choice block (`A) ...` / `B) ...` / etc.).
- `{{answer_instruction}}` — optional preamble or instruction text.

LL/G&P compatibility (study convention) requires every constrained-choice
prompt to display the choice letters visibly so log-likelihood scoring can
score per letter and generate-and-parse can extract the model's letter answer.
Where the canonical lm-eval-harness template scores by completion-likelihood
without rendering choices (ARC, HellaSwag, TruthfulQA-MC, Winogrande), our P2
adds the lettered choice block; the divergence is recorded in the `notes`
field of the corresponding `prompts.yaml` entry.

Per study convention, P4 minimal includes `{{fewshot_examples}}` — minimal
means stripped of decoration, not stripped of few-shot infrastructure.

## Per-benchmark P4 admissibility justifications

### arc_challenge

Path: `prompts/arc_challenge/P4_minimal_sourced.txt`

```
{{fewshot_examples}}{{question}}

{{choices}}
Answer:
```

**Justification.** ARC items are four-way multiple-choice science questions
with a stem and four labelled choices (Clark et al. 2018). The minimal
template preserves:

- **Task identity.** Item rendering retains the question stem and the four
  candidate answers in lettered form; the model is asked to select among them
  via the `Answer:` cue.
- **Answer space.** The lettered choice block exposes A/B/C/D explicitly so
  both LL scoring (per-letter likelihood) and G&P scoring (letter extraction)
  can operate.
- **Instruction semantics.** No HELM-style preamble or `Question:` framing;
  the bare stem is sufficient because the lettered choice block plus `Answer:`
  cue carries the instruction implicitly.

The minimal template strips only decoration (`Question:` framing, `The
following are…` preamble) — none of these affect the task identity, answer
space, or the LL/G&P scoring contract.

### hellaswag

Path: `prompts/hellaswag/P4_minimal_sourced.txt`

```
{{fewshot_examples}}{{question}}

{{choices}}
Answer:
```

**Justification.** HellaSwag items are sentence-continuation problems with an
activity-label-prefixed context and four candidate completions (Zellers et al.
2019). In our rendering:

- **Task identity.** `{{question}}` is the activity-label-plus-context string
  (matching the harness's `process_docs` and HELM's `json_to_instance`); the
  four endings are rendered as the lettered choice block. The model selects an
  ending.
- **Answer space.** Lettered A/B/C/D exposed for LL/G&P compatibility.
- **Instruction semantics.** Preserved by the lettered-choice + `Answer:`
  scaffold; the explicit HELM preamble (`The following are multiple choice
  questions (with answers) about common sense.`) is omitted as decoration.

The minimal template intentionally drops the HELM commonsense preamble and
the `Question:` framing while keeping the activity-label-plus-context as the
stem. This matches the original benchmark task definition.

### truthfulqa_mc

Path: `prompts/truthfulqa_mc/P4_minimal_sourced.txt`

```
{{fewshot_examples}}{{question}}

{{choices}}
Answer:
```

**Justification.** TruthfulQA-MC1/MC2 items are multiple-choice items with one
correct factual answer (MC1) or a calibrated set of correct answers (MC2),
selected from a candidate set (Lin et al. 2022). Minimal template:

- **Task identity.** Bare question + lettered candidate set + `Answer:` cue.
  The MC1/MC2 candidate set is the canonical answer space; lettering it is
  faithful to the underlying task.
- **Answer space.** A/B/C/… letters expose the candidate set for LL/G&P
  scoring.
- **Instruction semantics.** The six-shot QA preamble in the harness's native
  prompt is decoration aimed at calibrating non-MC accuracy; the MC task is
  evaluable without it. The HELM TruthfulQA adapter also uses an empty
  instruction string — corroborating that no preamble is required for task
  identity.

### mmlu_panel

Path: `prompts/mmlu_panel/P4_minimal_sourced.txt`

```
{{fewshot_examples}}{{question}}

{{choices}}
Answer:
```

**Justification.** MMLU items are four-way MCQs across 57 subjects (Hendrycks
et al. 2021). Minimal template:

- **Task identity.** Bare stem + four-way lettered choices + `Answer:`. The
  subject-aware preamble (`The following are multiple choice questions (with
  answers) about <subject>.`) is decoration; MMLU items are evaluable
  per-subject without it (lm-evaluation-harness's MMLU default also omits the
  preamble — corroborating evidence that the preamble is not load-bearing for
  task identity).
- **Answer space.** A/B/C/D exposed.
- **Instruction semantics.** Preserved by the four-way lettered choices and
  `Answer:` cue; the subject preamble is the variable-content piece that
  varies across subjects, but our subject-panel evaluation is per-subject and
  does not require subject identification at prompt time.

### winogrande

Path: `prompts/winogrande/P4_minimal_sourced.txt`

```
{{fewshot_examples}}{{question}}

{{choices}}
Answer:
```

**Justification.** Winogrande items are fill-in-the-blank coreference items
with a sentence containing a literal `_` and two candidate fillers
(Sakaguchi et al. 2020). Minimal template:

- **Task identity.** The sentence stem (with `_` preserved) plus an A/B
  lettered enumeration of the two options + `Answer:`. The blank carries the
  task identity implicitly: the model must choose the option that fills the
  blank coherently.
- **Answer space.** Exactly two options (A, B) — the smallest constrained
  choice set; LL and G&P both operate on the binary letter choice.
- **Instruction semantics.** No `Sentence:` framing, no preamble. The blank
  itself is the cue.

Note: this minimal template differs from the lm-evaluation-harness native
scoring, which compares the LL of two sentence-completion strings (sentence
prefix concatenated with each option). The minimal template translates
Winogrande into the LL/G&P-compatible MCQ shape used uniformly across this
study; we record the divergence in the `notes` field of `prompts.yaml`.

### gsm8k

Path: `prompts/gsm8k/P4_minimal_sourced.txt`

```
{{fewshot_examples}}{{question}}
Answer:
```

**Justification.** GSM8K items are grade-school math word problems with a
free-text numeric answer (Cobbe et al. 2021). Open generation; no
`{{choices}}`. Minimal template:

- **Task identity.** Bare problem stem + `Answer:` cue. The model produces
  an open generation that contains a numeric answer (extracted by the
  scorer).
- **Answer space.** Numeric (open). The `Answer:` cue is the standard signal
  for the model to begin its solution; both lm-eval-harness and HELM use the
  same single-token cue (the harness uses `Answer:`, HELM uses `A:` — both
  are admissible single-token prompts).
- **Instruction semantics.** No `Question:` / `Q:` framing. Stop-sequence
  handling (e.g. `\n\n` per HELM, or `Question:` per lm-eval) is the
  responsibility of the runner, not the template; the template carries no
  load there.

The few-shot block is essential for GSM8K performance (the standard 8-shot
chain-of-thought prompt from Wei et al. 2022 is the de facto baseline) and is
retained in P4.

## Audit notes

- All P4 prompts share the same skeletal structure
  (`{{fewshot_examples}}{{question}}\n\n{{choices}}\nAnswer:` for choice tasks,
  with `{{choices}}` removed for `gsm8k`). The shared structure is intentional:
  it provides the cleanest contrast against the more decorated P1/P2/P3
  prompts.
- Where the canonical lm-eval-harness template omits visible choice
  letters (ARC, HellaSwag, TruthfulQA-MC, Winogrande), P2 in this study
  re-adds them so all four prompts can be scored under both LL and G&P.
  This study-convention divergence is documented in the `notes` field of
  the corresponding `prompts.yaml` entry.
- HELM does not provide stock English scenarios for ARC or Winogrande
  (only `arabic_mmlu_scenario.py`-style or Afrikaans variants). For those
  two benchmarks, P3 falls back to a published peer-reviewed variant
  (Brown et al. 2020 GPT-3 evaluation) — `source_type:
  published_variant` for Winogrande, `source_type: helm_reference` for
  ARC where the HELM common adapter pattern (used by other MCQ
  scenarios) is the natural transferred prompt.
- P1 ARC: Clark et al. (2018) do not specify a single canonical
  natural-language wrapper. We adopt the `Question:`/`Answer:` framing
  used by the paper's reported baselines; this is the same framing used
  by the lm-evaluation-harness ARC default, so P1 and P2 differ here
  only in the (study-imposed) addition of the lettered choice block to
  P2.

## Open questions for review

1. Should P3 Winogrande's `answer_instruction` use the literal Brown et al.
   (2020) GPT-3 paper wording (`The following are sentence completion
   problems.`), or should it match HELM's empty-instruction adapter pattern
   for parallelism with HELM-MMLU and HELM-HellaSwag? (Currently we use the
   GPT-3 wording.)
2. Should P3 GSM8K render `{{answer_instruction}}` as the empty string (HELM
   default) or as a short HELM-style preamble (e.g. `Solve the following math
   word problem.`)? (Currently empty, matching HELM's `instructions=''`.)
3. ARC P1 follows the lm-eval template byte-for-byte. Should P1 introduce a
   subtle benchmark-author-only marker (e.g. an explicit `Choices:` label) to
   distinguish it from P2 in the empirical contrast? Tradeoff: faithfulness
   to the (under-specified) author intent vs. measurability of P1↔P2 effect.
