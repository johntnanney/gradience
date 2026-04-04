# Task-Balanced Extension Design

- Snapshot date (UTC): 2026-04-03T22:11:34.600114+00:00
- Families preserved: llama, mistral, qwen
- Prioritize: code, math_reasoning, domain_specialist, classification
- Deprioritize: chat_instruct (unless needed for architecture parity)

## Task Targets

| Category | Current | Target | Needed |
|----------|---------|--------|--------|
| code | 0 | 8 | 8 |
| math_reasoning | 0 | 8 | 8 |
| domain_specialist | 0 | 6 | 6 |
| classification | 10 | 5 | 0 |

## Architecture Coverage Goals

| Family | Current | Target Min | Needed |
|--------|---------|------------|--------|
| llama | 18 | 5 | 0 |
| mistral | 8 | 5 | 0 |
| qwen | 0 | 5 | 5 |
