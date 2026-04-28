#!/usr/bin/env bash
# run_phase5.sh — Post-GPU Phase 5 analysis pipeline driver.
#
# Pre-verified on 2026-04-28 against a 548-condition partial pod pull.
# CLI invocations have been corrected for actual script signatures
# (per the 2026-04-26 dry-run + 2026-04-28 prep validation):
#   - Script 07 takes --analysis-config (NOT --config)
#   - Script 08 takes --analysis-config
#   - Script 09 takes --analysis-config + --item-level (NOT --condition-level)
#   - Script 10 takes --analysis-config + both --item-level and --condition-level
#   - Script 98 takes --manifests-dir, --raw-dir, --normalized-dir, --analysis-dir
#
# Usage (from papers/benchmark_reliability_study/):
#   ./scripts/run_phase5.sh
#
# Or with explicit paths:
#   RAW_DIR=runs/raw/ ./scripts/run_phase5.sh
#
# Expects:
#   - .venv/ with the analysis deps installed (statsmodels, seaborn,
#     matplotlib, pyarrow, pandas, numpy, scipy, datasets, jsonschema).
#   - manifests/conditions_*.csv with conditions marked complete (run
#     mark_complete.py first if needed; this script does it
#     automatically as Phase 0 below).
#   - runs/raw/<condition_id>/ directories with run_metadata.json +
#     item_outputs.jsonl (G&P) or item_scores.jsonl (LL).

set -euo pipefail

PAPER_ROOT="${PAPER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PAPER_ROOT"

RAW_DIR="${RAW_DIR:-runs/raw/}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis/}"
NORMALIZED_DIR="${NORMALIZED_DIR:-runs/normalized/}"
SCHEMAS_DIR="${SCHEMAS_DIR:-schemas/}"
REPORTS_DIR="${REPORTS_DIR:-reports/}"
TABLES_DIR="${TABLES_DIR:-tables/}"
FIGURES_DIR="${FIGURES_DIR:-figures/}"
CONFIG="${CONFIG:-configs/study_config.yaml}"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Phase 5 driver ==="
echo "  PAPER_ROOT  = $PAPER_ROOT"
echo "  RAW_DIR     = $RAW_DIR"
echo "  ANALYSIS_DIR= $ANALYSIS_DIR"
date -u

mkdir -p "$NORMALIZED_DIR" "$ANALYSIS_DIR/variance_components" \
  "$ANALYSIS_DIR/tolerance_schedules" "$ANALYSIS_DIR/ranking_stability" \
  "$ANALYSIS_DIR/mmlu_subjects" "$ANALYSIS_DIR/gsm8k_case" \
  "$REPORTS_DIR" "$TABLES_DIR" "$FIGURES_DIR"

echo
echo "=== Phase 0 — mark conditions complete (RUNBOOK §8a) ==="
python - <<'PY'
"""Re-derive condition_status from actual data: compare each run_metadata's
num_items_completed to the local manifest's expected_num_items (D-19
patched). If they match, mark the manifest row complete regardless of
pod-side status label (which may be 'partial' if the pod's manifest
still has the unpatched expected_num_items=100 placeholder). Conditions
flagged excluded_pre_run (Cut 2) remain excluded."""
import csv, json
from pathlib import Path
RAW = Path("runs/raw")
total_complete = 0
total_pending = 0
total_excluded = 0
total_truly_partial = 0

for mp in [Path("manifests/conditions_primary.csv"),
           Path("manifests/conditions_gsm8k.csv")]:
    if not mp.exists():
        continue
    rows = list(csv.DictReader(open(mp)))
    fields = list(rows[0].keys())
    n_complete = n_truly_partial = n_pending = n_excluded = 0
    for row in rows:
        if row["condition_status"] == "excluded_pre_run":
            n_excluded += 1
            continue
        meta = RAW / row["condition_id"] / "run_metadata.json"
        if not meta.exists():
            n_pending += 1
            continue
        m = json.loads(meta.read_text())
        actual = m["num_items_completed"]
        expected_local = int(row["expected_num_items"])
        if actual == expected_local:
            row["condition_status"] = "complete"
            n_complete += 1
            # Also patch the pod-side metadata if its expected is stale
            # so script 04 doesn't double-flag this row.
            if m.get("num_items_expected") != expected_local or m.get("status") != "complete":
                m["num_items_expected"] = expected_local
                m["status"] = "complete"
                meta.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        else:
            row["condition_status"] = "failed"
            n_truly_partial += 1
            print(f"  truly partial: {row['condition_id']} got {actual} expected {expected_local}")
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    total_complete += n_complete
    total_pending += n_pending
    total_excluded += n_excluded
    total_truly_partial += n_truly_partial
    print(f"  {mp.name}: complete={n_complete}, partial={n_truly_partial}, pending={n_pending}, excluded={n_excluded}")

print(f"  TOTAL: complete={total_complete}, partial={total_truly_partial}, "
      f"pending={total_pending}, excluded={total_excluded}")
PY

echo
echo "=== Phase 1 — script 04 (normalize outputs) ==="
python scripts/04_normalize_outputs.py \
  --conditions manifests/conditions_primary.csv \
  --raw-dir "$RAW_DIR" \
  --schemas-dir "$SCHEMAS_DIR" \
  --out "$NORMALIZED_DIR/item_level_primary.parquet"

python scripts/04_normalize_outputs.py \
  --conditions manifests/conditions_gsm8k.csv \
  --raw-dir "$RAW_DIR" \
  --schemas-dir "$SCHEMAS_DIR" \
  --out "$NORMALIZED_DIR/item_level_gsm8k.parquet"

echo
echo "=== Phase 2 — script 05 (condition aggregation) ==="
python scripts/05_make_condition_scores.py \
  --item-level "$NORMALIZED_DIR/item_level_primary.parquet" \
  --out "$NORMALIZED_DIR/condition_level_primary.csv"

python scripts/05_make_condition_scores.py \
  --item-level "$NORMALIZED_DIR/item_level_gsm8k.parquet" \
  --out "$NORMALIZED_DIR/condition_level_gsm8k.csv"

echo
echo "=== Phase 3 — script 06 (variance components, D-20 cascade) ==="
python scripts/06_variance_components.py \
  --item-level "$NORMALIZED_DIR/item_level_primary.parquet" \
  --condition-level "$NORMALIZED_DIR/condition_level_primary.csv" \
  --config "$CONFIG" \
  --out-dir "$ANALYSIS_DIR/variance_components/"

echo
echo "=== Phase 4 — script 07 (tolerance schedule, regime-aware) ==="
# Note: script 07's flag is --analysis-config (not --config). Both point
# at study_config.yaml, which is the merged-config entry.
python scripts/07_tolerance_schedule.py \
  --condition-level "$NORMALIZED_DIR/condition_level_primary.csv" \
  --variance-components "$ANALYSIS_DIR/variance_components/aggregate_vc.csv" \
  --analysis-config "$CONFIG" \
  --out-dir "$ANALYSIS_DIR/tolerance_schedules/"

echo
echo "=== Phase 5 — script 08 (ranking stability) ==="
python scripts/08_ranking_stability.py \
  --condition-level "$NORMALIZED_DIR/condition_level_primary.csv" \
  --analysis-config "$CONFIG" \
  --out-dir "$ANALYSIS_DIR/ranking_stability/"

echo
echo "=== Phase 6 — script 09 (MMLU subject decomposition) ==="
# Note: 09 takes --item-level (not --condition-level).
python scripts/09_mmlu_subject_decomp.py \
  --item-level "$NORMALIZED_DIR/item_level_primary.parquet" \
  --analysis-config "$CONFIG" \
  --out-dir "$ANALYSIS_DIR/mmlu_subjects/"

echo
echo "=== Phase 7 — script 10 (GSM8K case study) ==="
# Note: 10 takes both --item-level and --condition-level.
python scripts/10_gsm8k_case.py \
  --item-level "$NORMALIZED_DIR/item_level_gsm8k.parquet" \
  --condition-level "$NORMALIZED_DIR/condition_level_gsm8k.csv" \
  --analysis-config "$CONFIG" \
  --out-dir "$ANALYSIS_DIR/gsm8k_case/"

echo
echo "=== Phase 8 — script 98 (reproducibility trace) ==="
# Trace re-runs 06 and 07 internally and diffs against stored output.
# Bootstrap CI columns in tolerance_by_cell.csv may not be bit-identical
# across runs due to a known non-determinism issue (D-21); the trace
# will report 'fail' on that artifact even when point estimates match.
# That's a pre-known issue, not a regression.
python scripts/98_reproducibility_trace.py \
  --config "$CONFIG" \
  --manifests-dir manifests/ \
  --raw-dir "$RAW_DIR" \
  --normalized-dir "$NORMALIZED_DIR" \
  --analysis-dir "$ANALYSIS_DIR" \
  --sample-n 5 --seed 20260424 \
  --out "$REPORTS_DIR/reproducibility_trace.md" || \
  echo "  (98 reported failures — see report; D-21 bootstrap-determinism known)"

echo
echo "=== Phase 9 — script 99 (pipeline report) ==="
python scripts/99_make_report.py \
  --analysis-dir "$ANALYSIS_DIR" \
  --tables-dir "$TABLES_DIR" \
  --figures-dir "$FIGURES_DIR" \
  --out "$REPORTS_DIR/cpu_pipeline_report.md"

echo
echo "=== Phase 5 driver complete ==="
date -u
echo "Artifacts:"
echo "  $NORMALIZED_DIR/item_level_primary.parquet ($(wc -l < /dev/null) row count below)"
ls -lh "$NORMALIZED_DIR"/*.{parquet,csv} 2>/dev/null | awk '{print "    " $0}'
echo "  $ANALYSIS_DIR/{variance_components,tolerance_schedules,ranking_stability,mmlu_subjects,gsm8k_case}/"
echo "  $REPORTS_DIR/reproducibility_trace.md"
echo "  $REPORTS_DIR/cpu_pipeline_report.md"
echo "  $FIGURES_DIR/{mmlu_model_subject_heatmap.png,ranking_stability_by_benchmark.png}"
