# Sample Merge QA Reports

Canonical examples of `gradience merge-audit` QA report output, based on Study 16 archetypes.

## Files

| File | Archetype | Risk | Key Feature |
|------|-----------|------|-------------|
| `pair06_weak_source_high_risk.json` | Pair 06 | HIGH | One weak adapter + 5 conflicting layers. Shows eligibility warnings, asymmetric coefficients, DARE-TIES recommendation. |
| `pair02_balanced_low_risk.json` | Pair 02 | LOW | Both adapters eligible, all 32 layers safe. The happy-path no-op: linear merge, equal coefficients, no warnings. |

## Schema

Both files follow `gradience.merge_qa_report/v1`. The core fields match `MergeQAReport.to_dict()` output from `gradience.vnext.merge.qa_report`. Additional fields (`description`, `source_qa`, `recommendation`, `formatted_report`) are included for documentation purposes.

## Usage

These files can be used as:

- **Documentation** — show users what to expect from `gradience merge-audit --emit-report`
- **Test fixtures** — load via `MergeQAReport.from_dict()` for roundtrip validation
- **Communication anchors** — reference specific scenarios by name ("Pair 06 pattern", "Pair 02 pattern")
