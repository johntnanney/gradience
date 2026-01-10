# Telemetry & Privacy


TelemetryWriter redacts strings longer than 256 characters by default.

Gradience vNext telemetry (`gradience.vnext.telemetry/v1`) is designed to log:
- numeric metrics (loss, ppl, accuracy, counts)
- run metadata (model name, dataset name, task profile, timestamps/steps)
- derived diagnostics (gap, stable rank/utilization summaries)
- recommendations/alerts

Gradience does **not** log raw training examples, prompts, completions, or labels by default.

Treat telemetry files as shareable diagnostics, not as a data dump.
