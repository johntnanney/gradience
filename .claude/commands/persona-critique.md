---
description: Single-persona methodological critique per research_review/rotating_persona_prompt.md
allowed-tools: Read, Agent, Bash
---

Run a single-persona methodological critique using the protocol defined in
`/Users/john/code/gradience/research_review/rotating_persona_prompt.md`.

This is the rotation-discipline counterpart to `/tension-finder`: where
tension-finder catches cross-document inconsistency, persona-critique
forces variance by adopting an outside-frame methodological voice. Lower
frequency than daily review; weekly is the default cadence; on-demand is
fine for moments of significant program-state change.

## Steps

1. **Read the prompt file** at `/Users/john/code/gradience/research_review/rotating_persona_prompt.md`. The agent-prompt body is everything from the line `## Prompt to pass to the agent (everything below this line)` onward. The Phase 2 roster lists the six personas and their snake-case names: `bayesian_skeptic`, `nist_policy_reader`, `philosopher_of_measurement`, `frequentist_hostile_to_glmm`, `ml_systems_practitioner`, `applied_benchmarking_ranking`.

2. **Resolve the persona** for this invocation:
   - If `$ARGUMENTS` is non-empty and matches one of the six snake-case persona names, use that persona.
   - If `$ARGUMENTS` is non-empty but doesn't match (typo, abbreviation, etc.), surface to the user: "Persona '$ARGUMENTS' not in roster. Available: bayesian_skeptic, nist_policy_reader, philosopher_of_measurement, frequentist_hostile_to_glmm, ml_systems_practitioner, applied_benchmarking_ranking." Then stop — don't guess what they meant.
   - If `$ARGUMENTS` is empty, run `ls /Users/john/code/gradience/research_review/persona_*_*.md 2>/dev/null | head -20` to inspect rotation history. Identify which personas have been used in the past four weeks (date suffix within ~28 days of today). Pick a persona that has *not* been used in that window. If multiple are eligible, pick the one with the longest gap since last use (or any, if none have been used yet). Surface the choice to the user before spawning the agent: "Selected persona: <name> (last used: <date or 'never'>; rotation: <reason>)." Wait for user confirmation before continuing — the persona choice is the constraint, and they should sign off on it explicitly.

3. **Compose the spawned-agent prompt** by composing in this order:
   - Prepend a persona-selection note exactly:
     ```
     Persona for this invocation: <persona-snake-case-name>

     Read Phase 2 of the prompt below; locate the matching persona's framing paragraph and adopt that frame. Do not deviate from the persona's commitments, characteristic pressure, or boundary statement.

     ```
   - Append the agent-prompt body from step 1.
   - Append a date note: a single line `Date: <today's date in YYYY-MM-DD form>.`

4. **Spawn a fresh general-purpose Agent** using the Agent tool:
   - `subagent_type`: `general-purpose`
   - `description`: `Persona critique: <persona name>`
   - `prompt`: the composed prompt from step 3

   Run in foreground unless the user explicitly requests background. Persona critiques are short (500–800 words) and the user typically wants to read the result immediately.

5. **After the agent returns**, do not paraphrase or expand the critique. Verify the agent wrote the report file by checking `/Users/john/code/gradience/research_review/persona_<persona-name>_<YYYY-MM-DD>.md` exists. Then confirm to the user with:
   - The report path.
   - The persona that was adopted.
   - The "Focus of this critique" line extracted from the file (the one-sentence framing that names the angle taken).
   - A one-line note that the report is theirs to read and engage with.

   Do not editorialize on the critique. The persona's pressure is the deliverable; commenting on whether the persona "made a good point" would re-collapse the variance the rotation is designed to surface.

## Constraints

- Never edit the rotating-persona prompt itself based on a single critique's content. The prompt is the framework; individual critiques accumulate as longitudinal data, not as edits to the framework.
- Never write a critique inline in the conversation. The critique lives in the dated `persona_*_*.md` file; pointing at it is the deliverable.
- If the agent reports back with "this persona has nothing to push on this week" (the prompt explicitly allows this), accept it as a valid output and pass through to the user. Don't pressure the agent to produce a critique anyway.
- If the same persona is requested twice within four weeks, surface that to the user before spawning: "Persona <name> was last used on <date> (within four-week rotation window). Continue anyway, or rotate?" Wait for explicit confirmation.
