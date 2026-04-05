# Cohort Definition

Canonical cohort manifest: [`scripts/decoder_triage_study/cohort_manifest.json`](../../scripts/decoder_triage_study/cohort_manifest.json)

Current design:

- Base model: `mistralai/Mistral-7B-v0.1`
- Total adapters: 16
- Classification family:
  - `glue/sst2` (r=8,16; seeds 42,123)
  - `glue/qnli` (r=8,16; seeds 42,123)
- Instruction family:
  - `yahma/alpaca-cleaned` (r=8,16; seeds 42,123)
  - `OpenAssistant/oasst1` (r=8,16; seeds 42,123)

Total pair count from cohort: 120 (`16 choose 2`).

