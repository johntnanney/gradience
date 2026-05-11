# ANON Audit Checklist — Supplementary Bundle

**Status:** internal (do NOT include in the OpenReview upload). This checklist is the executable procedure for anonymizing the supplementary bundle before attaching it to the TMLR submission. Retain in the repo for camera-ready restoration and as a persistent audit trail.

**Scope.** Every file that will be attached as supplementary material at OpenReview. Unattached originals in `sidecar/` are out of scope — leave those alone. Work against copies in `papers/n134_workshop/supplementary/` only.

**Convention.** Same as the main manuscript: each edited region gets an adjacent marker comment naming what changed and why. Use `# ANON:` for Markdown / Python / plaintext; use a top-level `"_anon_note"` field at the JSON root for JSON files (JSON does not support line comments). `grep -rn "ANON:\|_anon_note" supplementary/` at the end of the pass should enumerate every edit site.

---

## Phase 0 — Before you start

Copy originals into the supplementary directory as a clean starting state. Work on the copies; never edit the originals.

```
cd papers/n134_workshop/supplementary/
mkdir -p pre_registration raw_analytical_artifacts/per_adapter_audits analysis_scripts

cp ../../../sidecar/notes/n134_spec.md                    pre_registration/preregistration_v3_1.md
cp ../../../sidecar/notes/n134_icc_spec.md                pre_registration/icc_analysis_preregistration.md

cp ../../../sidecar/results/n134/analysis_h1.json         raw_analytical_artifacts/analysis_h1.json
cp ../../../sidecar/results/n134/analysis_icc.json        raw_analytical_artifacts/analysis_icc.json
cp ../../../sidecar/results/n134/analysis_secondary.json  raw_analytical_artifacts/analysis_secondary.json
cp ../../../sidecar/results/n134/method_comparison.json   raw_analytical_artifacts/method_comparison.json
cp ../../../sidecar/results/n134/audit/pair_alignment_full.json    raw_analytical_artifacts/pair_alignment_full.json
cp ../../../sidecar/results/n134/audit/pair_alignment_summary.json raw_analytical_artifacts/pair_alignment_summary.json
cp ../../../sidecar/results/n134/audit/adapter_profiles.json       raw_analytical_artifacts/adapter_profiles.json
cp ../../../sidecar/results/n134/audit/w0_properties.json          raw_analytical_artifacts/w0_properties.json
cp ../../../sidecar/results/n134/audit/*_s*_summary.json           raw_analytical_artifacts/per_adapter_audits/

cp ../../../scripts/n134/06_analysis_h1.py       analysis_scripts/compute_s_h1.py
cp ../../../scripts/n134/09_analysis_icc.py      analysis_scripts/compute_icc.py
cp ../../../scripts/n134/08_compare_methods.py   analysis_scripts/compare_methods.py
```

Commit this copy-in state to a branch before starting the ANON edits. The branch gives you a clean diff base for camera-ready restoration.

---

## Phase 1 — Universal grep sweep

These patterns must NOT appear anywhere in the supplementary bundle at submission time. Run each grep before starting per-file edits to establish the leak inventory, and again at the end of the pass to verify zero hits.

```
cd papers/n134_workshop/supplementary/

# Project-internal identifiers in any file
grep -rn -E '\bN(07|127|130|132|133|134|135)(\b|-alt\b)' .
# Expected initial hits: dozens. Expected after pass: 0.

# Brand / project names
grep -rn -iE '\bgradience\b|\bcocchieri\b' .
# Expected initial hits: many in the pre-reg docs and scripts. Expected after pass: 0.

# Repository path prefixes
grep -rn -E '\bsidecar/|/gradience/|gradience\.|"sidecar"' .
# Expected initial hits: many. Expected after pass: 0.

# Filenames carrying the project number
grep -rn -E 'n13[0-5]_[a-z_]+\.md|n13[0-5]_[a-z_]+\.py|n13[0-5]_[a-z_]+\.json' .
# Expected after pass: 0.

# Cloud provider / host names (RunPod was explicitly noted in memory)
grep -rn -iE 'runpod|lambda labs|vast\.ai|coreweave' .
# Generic "commercial cloud" / "single RTX 6000 Ada 48 GB" is fine; provider name is not.

# User and machine identifiers
grep -rn -E '/Users/[a-zA-Z]+|/home/[a-zA-Z]+|johntnanney|nanney|john@' .
# Expected after pass: 0.

# URLs pointing to the canonical repo
grep -rn -E 'github\.com/[a-zA-Z0-9_-]+/gradience|anthropics/[a-zA-Z-]*gradience' .
# Expected after pass: 0.

# Git metadata — commit hashes look like [a-f0-9]{7,40}
grep -rn -E '\b[a-f0-9]{7}\b|\b[a-f0-9]{40}\b' . | grep -vE '\.json:' | head
# Review hits individually; JSON files with legitimate hex content (SHA256 of a checkpoint etc.) are fine if unrelated to commits. Paper/commit-hash leaks are not.

# Tag pointers
grep -rn -E '\bn134-[a-z-]+|v1-[a-z-]+\b' .
# Expected after pass: 0.

# Paper-private working-session labels (e.g., "RN-018", "A2 pass")
grep -rn -E '\bRN-0[0-9]{2}\b|\bA[1-4] (pass|workstream)\b' .
# Expected after pass: 0.
```

---

## Phase 2 — Per-file remediation

### 2.1 — `pre_registration/preregistration_v3_1.md`

**Original:** `sidecar/notes/n134_spec.md`, 385 lines. Heaviest-leakage file in the bundle. Budget 60–90 minutes for this file alone.

Leak classes and remediation:

1. **Project number in title.** Rename the document title from `N134 — Decoder-Scale Controlled Merge Triage (Pre-Registration, v3)` to `Decoder-Scale Controlled Merge Triage: Pre-Registration Document (v3.1)`.
2. **Version-history block (lines ~3–8).** Rewrite references to "N134 spec v1/v2/v3/v3.1" as "this document, v1/v2/v3/v3.1". Preserve all substantive amendments — only the project-number identifier changes.
3. **Supersession clause (line ~10).** The sentence `Complements N133 (sidecar/notes/n133_bp5_diagnostic.md) as its confirmatory follow-up` must be rewritten to name the precursor descriptively: `Complements the precursor confound-diagnostic study on the same adapter-pair substrate (see paper §5.3 for that study's task-boundary detection results)`.
4. **§1 "What N134 Is For".** Rename heading to `What This Study Is For`. Replace every `N133`, `N134`, `N134 design`, `Gradience` occurrence with either `the precursor study`, `the present study`, or the descriptive phrase already used in the paper at the matching section (`the measurement-discipline framework`, `the spectral-triage hypothesis`).
5. **§1 positioning paragraphs.** Reference to `Gradience's measurements` → `the spectral measurement approach under test`. External-citation paragraphs (OSRM, TSV, SVC, TARA, KnOTS, OSM) are fine as-is; they cite public literature.
6. **§2 confound block.** Each confound's `N134 constraint:` label → `Constraint:`. The reference to N133 results inside each confound's rationale → `the precursor study's results`.
7. **§3 experimental design.** Model-path references (probably `mistralai/Mistral-7B-v0.3`) are fine — public model. Any path strings like `sidecar/...` or `gradience/...` must be stripped or replaced with generic "working-directory" language.
8. **§9 resource block (the v3.1 amendment).** `single RTX 6000 Ada 48GB (RunPod Secure Cloud)` → `single RTX 6000 Ada 48 GB on commercial cloud`. Drop the provider name.
9. **End-of-document appendix references.** Any `sidecar/results/n134/...` output-path references → drop the `sidecar/` prefix and name the artifact by its bundle-relative path (`raw_analytical_artifacts/analysis_h1.json` etc.).

Verification for this file:
```
grep -n -E '\bN13[0-5]|\bGradience\b|\bsidecar/|RunPod' pre_registration/preregistration_v3_1.md
# Target: 0 hits.
grep -c "^# ANON:" pre_registration/preregistration_v3_1.md
# Target: ≥ 15 (substantial edit footprint, one marker per edit region).
```

### 2.2 — `pre_registration/icc_analysis_preregistration.md`

**Original:** `sidecar/notes/n134_icc_spec.md`, 288 lines.

Same class of edits as 2.1 but less dense — this document is scoped to the ICC analysis so it has less background prose. Specific watch-items:
- Every `N134 ICC` heading → `ICC reliability analysis` (or equivalent).
- References to `analysis_h1.py` as the instrument source → `the S_H1 compute script (see analysis_scripts/compute_s_h1.py in this bundle)`.
- Any mention of `sidecar/results/n134/` output paths → bundle-relative replacements.
- The `scripts/n134/06_analysis_h1.py:compute_s_h1` instrument-source reference → `analysis_scripts/compute_s_h1.py::compute_s_h1`.

Verification:
```
grep -n -E '\bN13[0-5]|\bGradience\b|\bsidecar/|scripts/n134/' pre_registration/icc_analysis_preregistration.md
# Target: 0 hits.
```

### 2.3 — JSON files in `raw_analytical_artifacts/` (top-level four)

**Files:** `analysis_h1.json`, `analysis_icc.json`, `analysis_secondary.json`, `method_comparison.json`.

JSON does not support inline comments, so use a top-level `"_anon_note"` field at the root of each file to record what was stripped, and a matching entry in this checklist.

Common leak patterns and remediation:

1. **`"experiment"` field.** Values like `"N134 four-method scheduled comparison"` → rewrite without the project number: `"Four-method scheduled comparison (decoder-scale controlled triage)"`.
2. **`"instrument_source"` field in `analysis_icc.json`.** Value `"scripts/n134/06_analysis_h1.py:compute_s_h1"` → `"analysis_scripts/compute_s_h1.py:compute_s_h1"`.
3. **`"spec"` field.** Value `"sidecar/notes/n134_icc_spec.md"` → `"pre_registration/icc_analysis_preregistration.md"`.
4. **Any path-valued fields** — hunt for `sidecar/`, `gradience/`, `scripts/n134/` prefixes and replace with bundle-relative paths.
5. **Method naming in `method_comparison.json`.** The `"gradience (S_H1)"` entry → `"S_H1 (this paper)"`. All other method names (KnOTS, TSV, SVC) are fine — they're public literature.
6. **Environment blocks.** Check each top-level JSON for any embedded `environment`, `paths`, `host`, `working_dir`, or similar metadata fields. Strip to generic or remove entirely.
7. **Timestamps.** If any `generated_at` / `run_timestamp` field is present and uses a fine-grained ISO timestamp, round to the day (e.g., `"2026-04-19T14:32:11Z"` → `"2026-04-19"`). Timestamp-based identity inference is low-risk but cheap to defeat.

Verification per file:
```
for f in raw_analytical_artifacts/analysis_*.json raw_analytical_artifacts/method_comparison.json; do
  echo "=== $f ==="
  grep -E '"experiment"|"instrument_source"|"spec"|"_anon_note"' "$f"
  grep -iE 'gradience|sidecar|scripts/n134|N13[0-5]|runpod' "$f" && echo "LEAKS in $f" || echo "clean"
done
```

### 2.4 — Pair alignment, adapter profiles, base-model reference

**Files:** `pair_alignment_full.json`, `pair_alignment_summary.json`, `adapter_profiles.json`, `w0_properties.json`.

Lower leakage surface — these are mostly numerical content (principal angles, singular values, per-adapter accuracies). Primary watch-items:

1. **Path fields.** Each per-adapter entry in `adapter_profiles.json` probably has a `"path"` or `"checkpoint"` field naming where the adapter was loaded from. Strip these entirely (reviewers don't need them to run the compute scripts — they work off task + seed as the identifier) or replace with `"checkpoint": "local filesystem path redacted"`.
2. **Base-model reference.** The `w0_properties.json` probably names `"mistralai/Mistral-7B-v0.3"` — this is fine, public model. Any local-path field naming where the base model was cached on disk must be stripped.
3. **Environment blocks.** Same as 2.3 — any embedded environment metadata gets the same treatment.

Verification:
```
python3 -c "
import json
for f in ['raw_analytical_artifacts/pair_alignment_full.json',
         'raw_analytical_artifacts/pair_alignment_summary.json',
         'raw_analytical_artifacts/adapter_profiles.json',
         'raw_analytical_artifacts/w0_properties.json']:
    d = json.load(open(f))
    s = json.dumps(d)
    for pat in ['gradience', 'sidecar', 'N134', 'N133', 'runpod', '/Users/', '/home/', 'johntnanney']:
        if pat.lower() in s.lower():
            print(f'LEAK in {f}: pattern {pat!r}')
"
```

### 2.5 — `per_adapter_audits/` (24 files)

Per-task-per-seed summaries like `arc_challenge_s42_summary.json`. 24 files. Each is small (spectral-measurement content only). The primary leak risk:

1. **`source_adapter_path`** or similar field naming where the adapter was loaded from.
2. **`base_model_path`** if it resolves to a local cache rather than the HuggingFace ID.
3. **Any `run_metadata` block** naming the user, host, or cloud provider.

The fastest way to handle 24 files uniformly is a small Python script:

```python
import json, os
from pathlib import Path

PATHS_TO_STRIP = ['source_adapter_path', 'checkpoint_path', 'base_model_path',
                  'working_directory', 'host', 'cloud_provider', 'user']

for f in Path('raw_analytical_artifacts/per_adapter_audits').glob('*_summary.json'):
    d = json.load(f.open())
    # Track edits for the audit note
    edits = []
    for key in PATHS_TO_STRIP:
        if key in d:
            edits.append(key)
            del d[key]
    # Also scrub any string field containing 'sidecar' or 'gradience' or user identifiers
    def scrub(obj):
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str) and any(p in v.lower() for p in
                        ['sidecar', 'gradience', '/users/', '/home/', 'runpod', 'johntnanney']):
                    obj[k] = '[path redacted for review]'
                    edits.append(f'scrubbed field {k}')
                else:
                    scrub(v)
        elif isinstance(obj, list):
            for x in obj: scrub(x)
    scrub(d)
    if edits:
        d['_anon_note'] = f'Stripped fields: {sorted(set(edits))}. Restore at camera-ready.'
    json.dump(d, f.open('w'), indent=2)
    print(f'{f.name}: {len(edits)} edits')
```

Run the script once, then verify:

```
grep -rl -iE 'gradience|sidecar|/Users/|/home/|runpod|johntnanney' raw_analytical_artifacts/per_adapter_audits/
# Target: no output (i.e., zero files with leaks).
```

### 2.6 — Analysis scripts

**Files:** `analysis_scripts/compute_s_h1.py`, `compute_icc.py`, `compare_methods.py`.

Scripts are the most subtle leakage surface — they carry both string content and structural metadata (module names, docstrings, logger names).

Leak classes and remediation:

1. **Module docstrings.** The `"""N134 H1 analysis — ..."""` header at the top of each file → rewrite without the project number.
2. **Author headers.** If any `# Author: ...` or `# Created by ...` comment exists, strip it.
3. **Hardcoded paths.** Any `DEFAULT_AUDIT_DIR = 'sidecar/results/n134/audit'` → rewrite to a bundle-relative default or force the path to be supplied via `--audit-dir` argument only.
4. **Logger names.** `logging.getLogger('gradience.n134.analysis_h1')` → `logging.getLogger('analysis_h1')`.
5. **Import statements.** Any `from gradience.sidecar.utils import ...` → either inline the needed utilities into the script, or restructure so the script is self-contained. Reviewers will not have the `gradience` package installable; the script must run against a plain numpy/scipy/sklearn environment.
6. **References to sibling scripts.** Comments like `# See also scripts/n134/07_analysis_secondary.py` → drop or generalize.
7. **Pre-registration document references.** `# Pre-registered in sidecar/notes/n134_spec.md §4.4` → `# Pre-registered in pre_registration/preregistration_v3_1.md §4.4`.
8. **Repo-tag references.** Anything like `# Corresponds to commit abc1234 on branch n134-submission` → strip.

Verification per script:
```
for f in analysis_scripts/*.py; do
  echo "=== $f ==="
  grep -nE 'from gradience|import gradience|sidecar/|scripts/n134|N13[0-5]|johntnanney|runpod' "$f"
done
# Target: 0 hits per file.

# Also verify each script is runnable against a clean environment:
python -c "import ast; ast.parse(open('analysis_scripts/compute_s_h1.py').read()); print('compute_s_h1.py: syntax OK')"
```

The self-containment requirement (item 5) is the only remediation step that requires real rework rather than find-replace. Budget 30–60 minutes per script for the inline-the-utilities pass, less if the utilities turn out to be thin wrappers around numpy/scipy.

---

## Phase 3 — Final verification sweep

Run every grep from Phase 1 again against the finished supplementary directory. Expected result: zero hits on every pattern.

```
cd papers/n134_workshop/supplementary/

# Master sweep — every forbidden pattern, one pass
grep -rnE '\bN(07|127|130|132|133|134|135)(\b|-alt\b)|\bgradience\b|\bcocchieri\b|\bsidecar/|/gradience/|runpod|/Users/|/home/|johntnanney|nanney|john@|\bn13[0-5]-[a-z-]+\b|\bRN-0[0-9]{2}\b' . \
  --include='*.md' --include='*.py' --include='*.json' \
  | grep -v ANON_AUDIT_CHECKLIST.md
# Expected: no output.

# Script runnability sanity check — syntax only, no execution
for f in analysis_scripts/*.py; do
  python -c "import ast; ast.parse(open('$f').read())" && echo "$f: syntactic OK" || echo "$f: FAILED"
done

# JSON validity sanity check
for f in raw_analytical_artifacts/*.json raw_analytical_artifacts/per_adapter_audits/*.json; do
  python -c "import json; json.load(open('$f'))" 2>/dev/null && : || echo "$f: INVALID JSON"
done

# ANON marker footprint — one line per edited file, count of markers
echo "=== edit-site marker footprint ==="
grep -rcE '^# ANON:|"_anon_note"' . --include='*.md' --include='*.py' --include='*.json'
# Every file that received edits should show ≥ 1 marker; pre_registration files should show many.
```

---

## Phase 4 — Bundle the archive

```
cd papers/n134_workshop/
# Create a clean tar.gz, excluding this checklist (internal) and any .pyc / __pycache__ / .DS_Store
tar --exclude='ANON_AUDIT_CHECKLIST.md' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
    -czvf supplementary_bundle.tar.gz supplementary/
ls -lh supplementary_bundle.tar.gz
```

Inspect the tarball's contents before uploading:

```
tar -tzvf supplementary_bundle.tar.gz | head -40
# Confirm: README.md present at root of supplementary/; no ANON_AUDIT_CHECKLIST.md in archive; no stray dotfiles.
```

---

## Camera-ready restoration

At acceptance, the restoration procedure mirrors the main manuscript's (RN-018 / RN-019 in `revision_notes.md`):

1. Check out the `v2-anonymized-supplementary` tag (or whatever tag was used to mark the pre-submission state).
2. For each `# ANON:` comment, consult the git diff at the anonymization commit to see the pre-anon text and restore it.
3. For each JSON file with a `"_anon_note"` field, the full pre-anon content is available at the same path in the original `sidecar/` working directory — copy that content back.
4. Re-bundle and upload to the camera-ready form.

---

## Time budget

Rough estimates, based on the file-by-file scope analysis:

- Phase 0 (copy-in + branch): 10 minutes.
- Phase 1 (universal sweep, leak inventory): 15 minutes.
- Phase 2.1 (preregistration_v3_1.md): 60–90 minutes. This is the long file.
- Phase 2.2 (icc_analysis_preregistration.md): 30 minutes.
- Phase 2.3 (top-four JSONs): 30 minutes.
- Phase 2.4 (four meta-JSONs): 15 minutes.
- Phase 2.5 (24 per-adapter audits, via script): 30 minutes including script authorship + verification.
- Phase 2.6 (three analysis scripts): 90–180 minutes. The self-containment pass (inline utilities) is the variable; if the scripts turn out to import heavily from the `gradience` package, budget the upper end.
- Phase 3 (final verification sweep): 15 minutes.
- Phase 4 (bundle + sanity check): 10 minutes.

**Total: 4–7 hours.** Compatible with the April 28 decision gate provided the pass starts within the next 48 hours.

---

## Phase 2.3 re-audit (2026-05-11)

Re-audit triggered by `CODING_AGENT_HANDOFF_2026-05-08.md` Phase 2.3.
Scope: check the manuscript-side post-Wang (commit 6c84e3a) +
post-Voudouris (commit 8854e2f) state for residual identifying content
not caught at the prior audit (3be686d).

### Grep coverage

| Category | Pattern | Hits |
|---|---|---:|
| Author / affiliation | `Cocchieri\|Nanney\|johntnanney\|gradience` | 0 |
| Project numbers | `N13[0-9]\b\|Thesis [AB]\b\|tier_1_5\|prereg_v1_1` | 1 → 0 after strip |
| Email patterns | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` | 0 |
| EDIT/ANON markers | `^% EDIT:\|^% ANON:` | 0 |
| AppleDouble / DS_Store in `supplementary/` | `\._*\|DS_Store` | 0 |

### Single hit + disposition

| Hit | Location | Disposition |
|---|---|---|
| `% Foundational references for the Thesis B measurement-discipline framework.` | `references.bib` line 176 (pre-strip) | **STRIPPED 2026-05-11.** Rewrote to `% Foundational references for the measurement-discipline framework.` Drops "Thesis B" project identifier. The line is a section divider comment, not a bibtex entry; the underlying psychometric-tradition entries below the divider are unchanged. |

### PDF metadata audit (Phase 2.4 inline)

| Field | Value | Disposition |
|---|---|---|
| `/Author`, `/Title`, `/Subject`, `/Keywords` | empty | OK. |
| `/Creator` | `LaTeX with hyperref` | OK. |
| `/Producer` | `pdfTeX-1.40.29` | OK. |
| PDF-stream identifying-string count | 0 | OK. |

### Post-strip recompile

21 pages, 0 citation undefined, 0 reference undefined, 0 errors.
Pre-existing 2 "empty journal" bibtex warnings on `jo2025evalinference`
and `karmakar2026singleprompt` unchanged; same warnings appear on
bench-reliability (pre-existing across both papers).
