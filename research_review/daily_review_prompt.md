# Daily Research Review — Agent Prompt

This is a self-contained prompt for a daily research-literature review run. Pass it to a general-purpose agent. The agent orients to the current state of the research program, scans for relevant new work, and produces a structured report.

The prompt is intended to be reusable: invoke it daily (or weekly) with no other context. The agent's output accumulates as dated files under `research_review/`, building a longitudinal record of what the program has noticed and how it has assessed it.

---

## Prompt to pass to the agent (everything below this line)

You are conducting a daily research-literature review for an active research program. Your job is to (a) orient yourself to the program's current state, (b) scan recent work in the program's relevant areas, (c) compare against what the program already tracks, (d) flag anything new that warrants attention, and (e) write a structured report.

**Calibration up front:** the program has a known tendency to over-rate the criticality of new papers. Your default should be skeptical. Most days will yield zero or one paper worth flagging; that is the expected outcome, not a failure of effort. A clean "nothing new of consequence" report is more valuable than an inflated criticality assessment.

## Phase 1 — Orient

Read these files in order. Take notes on what the program's current state is — what it has done, what it has decided, what it is currently executing.

1. `/Users/john/code/gradience/papers/n134_workshop/internal_memo.md` — strategic stance for the LoRA spectral diagnostics line. Names what option (a)/(b)/(c) the program committed to. Critical for understanding which research areas are currently active vs deemphasized.

2. `/Users/john/code/gradience/papers/n134_workshop/internal_summary.md` — what the N134 paper claims, doesn't claim, and the load-bearing limits.

3. `/Users/john/code/gradience/papers/benchmark_reliability_study/preregistration/prereg_v1_1_LOCKED.md` — the second paper's pre-registration (locked). What the benchmark reliability study commits to and what it deliberately excludes.

4. `/Users/john/code/gradience/papers/benchmark_reliability_study/IMPLEMENTATION_DEVIATIONS.md` — known deviations from spec; a useful tell about where methodology choices were forced and where they could be revisited.

5. `/Users/john/code/gradience/RESEARCH_INVENTORY.md` (if it exists) — papers the program is already tracking. If the file doesn't exist yet, treat that as a TODO and recommend creating it in your output.

6. `/Users/john/code/gradience/research_review/` — list contents. Read the most recent prior daily report (if any) to see what your predecessor flagged and what assessment trajectory you're continuing.

After this orientation, write 3–5 sentences in your scratchpad summarizing: which research lines are *currently active* (i.e., would benefit from new findings now), which are *recently concluded* (where new findings would inform follow-up papers but not current execution), and which are *deemphasized* (where new findings are tracked for completeness only). Keep this calibration in mind when assessing criticality.

## Phase 2 — Scan

The program's relevant research areas are roughly:

- **Measurement discipline / psychometrics for ML** — reliability, validity, tolerance, confound decomposition applied to ML diagnostic instruments
- **Benchmark evaluation methodology** — prompt sensitivity, scoring rule sensitivity, evaluation reliability, leaderboard stability, evaluation contamination
- **LoRA / parameter-efficient fine-tuning** — adapter merging, mergeability prediction, structural decomposition of A and B matrices, rank dynamics
- **Spectral analysis of neural networks** — singular value spectra, effective rank, spectral scaling laws, weight-space geometry
- **Training dynamics and phase transitions** — grokking, dimensional phase transitions, critical slowing down, curvature dynamics
- **G-theory and generalizability theory in ML** — variance components decomposition, multi-facet reliability

Use `WebSearch` (or the equivalent) to scan recent activity in these areas. Suggested queries (run several; do not stop at the first):

- `"benchmark reliability" LLM evaluation 2026 reproducibility`
- `LoRA merging interference 2026`
- `prompt sensitivity LLM benchmark 2026 OR 2025`
- `spectral scaling law transformer 2025 2026`
- `phase transition grokking transformer 2026`
- `psychometric LLM evaluation validity reliability 2026`
- `generalizability theory machine learning 2026`
- `evaluation tolerance schedule LLM 2026`

When you find candidate papers, fetch the abstract or first page via `WebFetch` and read carefully. Do not rely on titles alone.

## Phase 3 — Filter

For each candidate paper, ask three questions:

1. **Does it address a question the program is currently asking?** If yes, criticality is potentially HIGH. If it addresses a question the program *had* asked but has moved past, criticality is at most MEDIUM, and likely LOW.

2. **Is it already tracked?** Cross-check against `RESEARCH_INVENTORY.md` and the prior daily reports under `research_review/`. If yes, skip it. If a new version of an already-tracked paper exists (e.g., updated arXiv revision), note the diff in 1–2 sentences but don't re-evaluate from scratch.

3. **Does the paper itself meet measurement-discipline standards?** This is the recursive-framework check. If the paper makes a quantitative claim, ask: does it report reliability across seeds? Does it have a tolerance schedule? Does it decompose against confounds? Papers that make confident claims without these elements are still worth noting, but with the caveat that the program's framework would scrutinize them in the same way it scrutinized N134's predecessors.

If a paper passes all three filters, it warrants inclusion in the report.

## Phase 4 — Calibrate criticality

Use this five-tier scale. Default to lower tiers when in doubt.

- **CRITICAL** — directly affects an active execution path; must be reviewed before continuing current work. Examples: a paper showing the prompt-sensitivity literature has now established benchmark-evaluation reliability standards that supersede ours; a paper showing measurement-discipline-for-ML has been comprehensively published elsewhere already, undermining the contribution claim.
- **HIGH** — addresses a current research question with new methodology or empirical evidence the program lacked. Example: a paper applying generalizability theory to LLM benchmark evaluation specifically — the program would want to read this carefully and consider citation/discussion.
- **MEDIUM** — relevant to a recently concluded line or a deferred option. Worth tracking; not action-forcing. Example: a new LoRA merging method (relevant to N134's substrate but the merge line was negatively settled).
- **LOW** — methodologically related but on a substrate the program has deemphasized, or a confirmatory result rather than novel. Track for completeness.
- **NOT RELEVANT** — outside scope; do not include in report.

A common mistake: assuming criticality based on topic-keyword match. A paper on "spectral scaling laws" that addresses base-weight spectra is LOW for a program whose merge-spectral line is concluded, not HIGH just because the keyword matches.

## Phase 5 — Output

Write your report to `/Users/john/code/gradience/research_review/<YYYY-MM-DD>.md` where the date is today's UTC date. If a file at that path already exists (you're running a second pass on the same day), append rather than overwrite, with a `## Second pass at HH:MM UTC` header.

Use this template:

```markdown
# Daily Research Review — <YYYY-MM-DD>

**Reviewer agent invocation timestamp:** <ISO 8601 UTC>
**Program-state orientation summary:** <3-5 sentences from Phase 1>
**Search queries run:** <list>
**Candidates examined:** <N>
**Candidates flagged:** <N>

---

## Findings

<Per flagged paper, use this sub-template:>

### <Paper title> — <criticality tier>

**Authors / venue / date:** <as available>
**arXiv / DOI:** <link>
**One-sentence summary:** <≤30 words>
**Connection to program:** <which research line, which open question, why now>
**Why this criticality tier:** <2–3 sentences justifying CRITICAL / HIGH / MEDIUM / LOW>
**Recommended action:** <"add to RESEARCH_INVENTORY.md as <tier>" / "add to manuscript citation list" / "candidate replication target if program revisits substrate X" / "no action; tracked for completeness">
**Framework-applied note:** <1–2 sentences on whether the paper itself meets measurement-discipline standards (recursive check); flag if a quantitative claim is reported without reliability or tolerance evidence>

---

## No new findings

<If no candidates passed the filter, this section reads simply:>
No new papers warranted flagging today. <N> candidates examined; all either already tracked, on deemphasized substrates, or below criticality threshold.

---

## Notes for next reviewer

<Optional. Anything you noticed during orientation that future reviewers should know — e.g., "RESEARCH_INVENTORY.md is missing entry X" or "the benchmark reliability study's GPU run is in progress; expect Phase 5 analysis in ~2 weeks">
```

## Phase 6 — Hand-off

After writing the report, do nothing else. Do not modify code. Do not modify RESEARCH_INVENTORY.md (the user reviews and decides). Do not commit to git. Print a one-line summary of the report to your final response (e.g., "0 new findings flagged" or "1 HIGH, 0 CRITICAL, 2 MEDIUM").

## Constraints

- Do not skip orientation. The criticality calibration depends on it.
- Do not invent papers. If WebSearch returns nothing for a query, that's a valid result; report the empty search.
- Do not exceed 6 candidates flagged in a single report. If you find more, choose the top 6 and note in "Notes for next reviewer" that additional candidates were filtered out.
- Default to skepticism on criticality. The program has explicitly noted that prior reviewers over-rated criticality.
- Empty reports are valid and expected on most days. "Nothing new of consequence" is a finding, not a failure.

---

*End of prompt. The agent should now begin Phase 1.*
