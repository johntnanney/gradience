---
description: Cross-document tension audit per research_review/tension_finder_prompt.md
allowed-tools: Read, Agent, Bash
---

Run a cross-document tension audit using the protocol defined in
`/Users/john/code/gradience/research_review/tension_finder_prompt.md`.

The audit is on-demand only — never run on a schedule. Trigger when the
user senses drift between commitments and current state. Typical triggers:
before a lock amendment, before manuscript submission, after absorbing a
significant external input.

## Steps

1. **Read the prompt file** at `/Users/john/code/gradience/research_review/tension_finder_prompt.md`. The agent-prompt body is everything from the line `## Prompt to pass to the agent (everything below this line)` onward (skip the header preamble — it's documentation for the user, not for the spawned agent).

2. **Construct the spawned-agent prompt** by composing in this order:
   - If `$ARGUMENTS` is non-empty, prepend exactly:
     ```
     Scope override (user-supplied at invocation): $ARGUMENTS

     Apply this scope override to Phase 1; otherwise follow the prompt as written.

     ```
   - Append the agent-prompt body from step 1.
   - Append a trigger note: a single line `Trigger note: on-demand audit, $(date +%Y-%m-%d).` (substitute today's date from the env block; if `$ARGUMENTS` named a trigger reason, use that reason instead of "on-demand audit").

3. **Spawn a fresh general-purpose Agent** using the Agent tool:
   - `subagent_type`: `general-purpose`
   - `description`: `Tension audit`
   - `prompt`: the composed prompt from step 2

   Do not run any other agents in parallel for this command — the audit must complete in a fresh context with no other concurrent work.

4. **After the agent returns**, do not paraphrase or expand its findings. Verify the agent wrote the report file by checking `/Users/john/code/gradience/research_review/tension_audit_YYYY-MM-DD.md` exists. Then confirm to the user with:
   - The report path.
   - The findings count (extracted from the file's "Findings count:" header line).
   - A one-line note that the report is theirs to read and act on.

   Do not editorialize on the findings. The prompt's anti-patterns explicitly forbid the audit agent from proposing resolutions; the same discipline applies to your post-completion summary.

## Constraints

- Never edit program documents based on audit findings — that's the user's role.
- Never re-summarize the agent's report contents in this conversation. The report is the artifact; pointing at it is the deliverable.
- If the report file already exists for today's date when the agent finishes, alert the user — the agent may have overwritten a prior run. The convention is one audit per day; if a second is genuinely needed, the user will rename or move the prior one before re-running.
