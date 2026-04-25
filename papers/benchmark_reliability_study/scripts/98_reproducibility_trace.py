"""98_reproducibility_trace.py — SPEC_CPU_v0_2 §8.12, §13

Generate the reproducibility-trace report. Verifies that the pipeline's
outputs can be re-derived from the committed inputs (the reviewer's
first sanity check).

Usage:
    python scripts/98_reproducibility_trace.py \\
      --config configs/study_config.yaml \\
      --manifests-dir manifests/ \\
      --raw-dir runs/raw/ \\
      --normalized-dir runs/normalized/ \\
      --analysis-dir analysis/ \\
      --sample-n 5 \\
      --seed 20260424 \\
      --out reports/reproducibility_trace.md \\
      [--compare-env <dir>]

Trace structure (SPEC §13.1):
    1. Config state.
    2. Manifest state.
    3. Raw-run coverage.
    4. Per-condition recompute sample.
    5. Analysis re-derivation (variance components, tolerance schedule).
    6. Cross-environment check (optional, requires --compare-env).
    7. Summary.

Graceful degradation: an artifact's absence is not a failure. Failure
is defined as a present artifact failing to reproduce. A brand-new
pipeline (configs only) yields trace status "pass" with most sections
reporting "not yet runnable".

Exit codes per SPEC §11.1:
    0 — trace status "pass"
    5 — trace status "fail" (any reproducibility-critical artifact
        present that fails to reproduce)
    1 — generic / implementation error
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from gradience_study.config import (  # noqa: E402
    ConfigValidationError,
    compute_config_hash,
    load_config,
)


# Files we expect to exist in the manifests dir (subset of SPEC §8.2-§8.5).
MANIFEST_FILES: tuple[str, ...] = (
    "conditions_primary.csv",
    "conditions_gsm8k.csv",
    "prompt_manifest.csv",
    "fewshot_manifest.csv",
)

# Six YAML files that constitute the merged config, per SPEC §4.
CONFIG_YAML_FILES: tuple[str, ...] = (
    "study_config.yaml",
    "models.yaml",
    "benchmarks.yaml",
    "prompts.yaml",
    "scoring_rules.yaml",
    "analysis_config.yaml",
)

# Tolerance for floating-point recompute diffs (SPEC §13.1 item 4).
RECOMPUTE_TOLERANCE: float = 1e-6


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [98_reproducibility_trace] {msg}", file=sys.stderr)


# -- Helpers -----------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_csv_rows(path: Path) -> int:
    """Count data rows (excluding header) in a CSV. Returns 0 for empty/headerless."""
    n = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # header
        except StopIteration:
            return 0
        for _ in reader:
            n += 1
    return n


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# -- Section 1: Config state -------------------------------------------------


def _section_config_state(config_path: Path) -> dict[str, Any]:
    section: dict[str, Any] = {
        "title": "1. Config state",
        "config_hash_full": None,
        "config_hash": None,
        "yaml_hashes": {},
        "errors": [],
        "warnings": [],
        "failures": 0,
        "artifacts_checked": 0,
    }

    # Per-file SHA-256 of the six YAML files (regardless of merge result).
    configs_dir = config_path.parent
    for fname in CONFIG_YAML_FILES:
        p = configs_dir / fname
        if p.exists():
            section["yaml_hashes"][fname] = _sha256_file(p)
            section["artifacts_checked"] += 1
        else:
            section["yaml_hashes"][fname] = None
            section["warnings"].append(f"Config YAML missing: {fname}")

    # Compute merged config hash. Failure to load is a config-state error
    # but does NOT count as a reproducibility failure (it's a missing
    # artifact, not a present-but-non-reproducing one).
    try:
        merged = load_config(config_path)
        full, short = compute_config_hash(merged.raw_dict)
        section["config_hash_full"] = full
        section["config_hash"] = short
    except (FileNotFoundError, ConfigValidationError, KeyError) as e:
        section["errors"].append(
            f"Could not load merged config ({type(e).__name__}): {e}"
        )

    return section


def _render_config_state(section: dict[str, Any]) -> list[str]:
    lines = ["## " + section["title"], ""]
    if section["config_hash_full"]:
        lines.append(f"- `config_hash_full`: `{section['config_hash_full']}`")
        lines.append(f"- `config_hash` (short): `{section['config_hash']}`")
    else:
        lines.append("- `config_hash_full`: not computed (config did not load)")
        lines.append("- `config_hash` (short): not computed")
    lines.append("")
    lines.append("**Per-file SHA-256:**")
    lines.append("")
    lines.append("| File | SHA-256 |")
    lines.append("|---|---|")
    for fname in CONFIG_YAML_FILES:
        h = section["yaml_hashes"].get(fname)
        lines.append(f"| `{fname}` | `{h}` |" if h else f"| `{fname}` | missing |")
    lines.append("")
    if section["errors"]:
        lines.append("**Errors:**")
        for e in section["errors"]:
            lines.append(f"- {e}")
        lines.append("")
    return lines


# -- Section 2: Manifest state -----------------------------------------------


def _section_manifest_state(manifests_dir: Path) -> dict[str, Any]:
    section: dict[str, Any] = {
        "title": "2. Manifest state",
        "manifests": {},  # name -> {hash, row_count} or {hash: None, row_count: None}
        "failures": 0,
        "artifacts_checked": 0,
    }
    for name in MANIFEST_FILES:
        p = manifests_dir / name
        if p.exists():
            section["manifests"][name] = {
                "hash": _sha256_file(p),
                "row_count": _count_csv_rows(p),
                "present": True,
            }
            section["artifacts_checked"] += 1
        else:
            section["manifests"][name] = {
                "hash": None,
                "row_count": None,
                "present": False,
            }
    return section


def _render_manifest_state(section: dict[str, Any]) -> list[str]:
    lines = ["## " + section["title"], ""]
    lines.append("| Manifest | SHA-256 | Row count |")
    lines.append("|---|---|---|")
    for name, info in section["manifests"].items():
        if info["present"]:
            lines.append(f"| `{name}` | `{info['hash']}` | {info['row_count']} |")
        else:
            lines.append(f"| `{name}` | missing | — |")
    lines.append("")
    return lines


# -- Section 3: Raw-run coverage ---------------------------------------------


def _section_raw_coverage(manifests_dir: Path, raw_dir: Path) -> dict[str, Any]:
    section: dict[str, Any] = {
        "title": "3. Raw-run coverage",
        "manifest_present": False,
        "raw_dir_exists": False,
        "complete_in_manifest": 0,
        "raw_subdir_count": 0,
        "complete_missing_raw": [],     # condition_ids
        "raw_without_manifest": [],     # raw_dir names
        "note": "",
        "failures": 0,
        "artifacts_checked": 0,
        "complete_condition_ids": [],
    }

    primary = manifests_dir / "conditions_primary.csv"
    if not primary.exists():
        section["note"] = (
            "conditions_primary.csv not present; coverage check not yet runnable."
        )
        return section
    section["manifest_present"] = True

    # Read complete condition_ids from the manifest.
    complete_ids: list[str] = []
    with open(primary, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("condition_status") == "complete":
                complete_ids.append(row["condition_id"])
    section["complete_in_manifest"] = len(complete_ids)
    section["complete_condition_ids"] = complete_ids

    if not raw_dir.exists():
        section["note"] = (
            "raw-run directory not present; no inference outputs yet."
        )
        return section
    section["raw_dir_exists"] = True

    # Enumerate immediate subdirectories of raw_dir.
    raw_subdirs = {p.name for p in raw_dir.iterdir() if p.is_dir()}
    section["raw_subdir_count"] = len(raw_subdirs)

    if not raw_subdirs:
        section["note"] = "no inference outputs yet present"
        return section

    section["artifacts_checked"] = 1  # raw-coverage check itself

    complete_set = set(complete_ids)
    missing = sorted(complete_set - raw_subdirs)
    extra = sorted(raw_subdirs - complete_set)
    section["complete_missing_raw"] = missing
    section["raw_without_manifest"] = extra
    if missing or extra:
        section["failures"] += 1

    return section


def _render_raw_coverage(section: dict[str, Any]) -> list[str]:
    lines = ["## " + section["title"], ""]
    if not section["manifest_present"]:
        lines.append("- " + section["note"])
        lines.append("")
        return lines
    lines.append(
        f"- Conditions marked `complete` in manifest: "
        f"{section['complete_in_manifest']}"
    )
    if not section["raw_dir_exists"]:
        lines.append("- Raw-run directory: missing")
        lines.append("- " + section["note"])
        lines.append("")
        return lines
    lines.append(
        f"- Raw-run subdirectories on disk: {section['raw_subdir_count']}"
    )
    if section["raw_subdir_count"] == 0:
        lines.append("- " + section["note"])
        lines.append("")
        return lines
    if section["complete_missing_raw"]:
        lines.append("- **Manifest says complete, but raw_dir missing:**")
        for cid in section["complete_missing_raw"]:
            lines.append(f"    - `{cid}`")
    else:
        lines.append("- All complete conditions have raw directories.")
    if section["raw_without_manifest"]:
        lines.append("- **Raw directories without a matching manifest row:**")
        for cid in section["raw_without_manifest"]:
            lines.append(f"    - `{cid}`")
    lines.append("")
    return lines


# -- Section 4: Per-condition recompute sample ------------------------------


def _scoring_rule_from_condition_id(cid: str) -> str | None:
    """Extract scoring_rule_id from canonical condition_id form.

    Canonical form (SPEC §5.1) is
        {benchmark}__{model}__{prompt}__{seed}__{scoring_rule_id}
    with the scoring rule as the trailing component.
    """
    parts = cid.split("__")
    if not parts:
        return None
    return parts[-1]


def _recompute_condition_accuracy(run_dir: Path, scoring_rule: str) -> tuple[int, float] | None:
    """Recompute (num_items, accuracy) from raw item-level output.

    For G&P (`generate_parse`): use `item_outputs.jsonl`'s `is_correct`.
    For LL (`ll_norm`): use `item_scores.jsonl`'s `is_correct`.
    Returns None if the relevant file is missing or empty.
    """
    if scoring_rule == "generate_parse":
        path = run_dir / "item_outputs.jsonl"
    elif scoring_rule == "ll_norm":
        path = run_dir / "item_scores.jsonl"
    else:
        # Try either available file.
        if (run_dir / "item_outputs.jsonl").exists():
            path = run_dir / "item_outputs.jsonl"
        elif (run_dir / "item_scores.jsonl").exists():
            path = run_dir / "item_scores.jsonl"
        else:
            return None

    if not path.exists():
        return None
    rows = _read_jsonl(path)
    if not rows:
        return None
    n = len(rows)
    correct = sum(1 for r in rows if bool(r.get("is_correct")))
    return n, correct / n if n else 0.0


def _load_stored_condition_accuracy(
    normalized_dir: Path,
) -> dict[str, float]:
    """Load condition_id -> accuracy from any condition_level_*.csv present."""
    stored: dict[str, float] = {}
    if not normalized_dir.exists():
        return stored
    for csv_path in sorted(normalized_dir.glob("condition_level_*.csv")):
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("condition_id")
                    if not cid:
                        continue
                    raw = row.get("accuracy", "")
                    if raw == "" or raw is None:
                        continue
                    try:
                        stored[cid] = float(raw)
                    except ValueError:
                        continue
        except OSError:
            continue
    return stored


def _section_recompute_sample(
    raw_dir: Path,
    normalized_dir: Path,
    complete_condition_ids: list[str],
    sample_n: int,
    seed: int,
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "title": "4. Per-condition recompute sample",
        "sampled": [],     # list of dicts: condition_id, n_items, recomputed, stored, delta, status
        "note": "",
        "sample_n": sample_n,
        "seed": seed,
        "failures": 0,
        "artifacts_checked": 0,
    }

    if not complete_condition_ids:
        section["note"] = (
            "no complete conditions in manifest; recompute sample not yet runnable."
        )
        return section

    if not raw_dir.exists() or not any(raw_dir.iterdir() if raw_dir.exists() else []):
        section["note"] = "raw data absent; recompute sample not yet runnable."
        return section

    # Sample N condition_ids deterministically.
    rng = np.random.default_rng(seed)
    available = [c for c in complete_condition_ids if (raw_dir / c).exists()]
    if not available:
        section["note"] = (
            "no complete-condition raw_dirs present; recompute sample not yet runnable."
        )
        return section

    n = min(sample_n, len(available))
    indices = rng.choice(len(available), size=n, replace=False)
    sampled_ids = sorted([available[int(i)] for i in indices])

    stored_accuracies = _load_stored_condition_accuracy(normalized_dir)

    for cid in sampled_ids:
        run_dir = raw_dir / cid
        scoring_rule = _scoring_rule_from_condition_id(cid) or ""
        rec = _recompute_condition_accuracy(run_dir, scoring_rule)
        record: dict[str, Any] = {
            "condition_id": cid,
            "scoring_rule_id": scoring_rule,
            "num_items": None,
            "recomputed_accuracy": None,
            "stored_accuracy": stored_accuracies.get(cid),
            "delta": None,
            "status": "skipped",
            "reason": "",
        }
        if rec is None:
            record["status"] = "skipped"
            record["reason"] = "raw item-level file absent or empty"
            section["sampled"].append(record)
            continue
        n_items, recomputed = rec
        record["num_items"] = n_items
        record["recomputed_accuracy"] = recomputed
        section["artifacts_checked"] += 1
        if record["stored_accuracy"] is None:
            record["status"] = "skipped"
            record["reason"] = "no stored accuracy in normalized condition_level CSV"
        else:
            delta = abs(recomputed - record["stored_accuracy"])
            record["delta"] = delta
            if delta > RECOMPUTE_TOLERANCE:
                record["status"] = "fail"
                section["failures"] += 1
            else:
                record["status"] = "pass"
        section["sampled"].append(record)

    return section


def _render_recompute_sample(section: dict[str, Any]) -> list[str]:
    lines = ["## " + section["title"], ""]
    lines.append(
        f"- Sample size requested: N = {section['sample_n']}; "
        f"seed = {section['seed']}"
    )
    if section["note"]:
        lines.append(f"- {section['note']}")
        lines.append("")
        return lines
    if not section["sampled"]:
        lines.append("- No conditions sampled.")
        lines.append("")
        return lines
    lines.append("")
    lines.append(
        "| condition_id | scoring_rule | n_items | recomputed | stored | delta | status |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in section["sampled"]:
        rec = (f"{r['recomputed_accuracy']:.6f}"
               if isinstance(r["recomputed_accuracy"], float) else "—")
        sto = (f"{r['stored_accuracy']:.6f}"
               if isinstance(r["stored_accuracy"], float) else "—")
        delta = (f"{r['delta']:.2e}"
                 if isinstance(r["delta"], float) else "—")
        n_items = r["num_items"] if r["num_items"] is not None else "—"
        lines.append(
            f"| `{r['condition_id']}` | `{r['scoring_rule_id']}` | "
            f"{n_items} | {rec} | {sto} | {delta} | {r['status']} |"
        )
    skipped = [r for r in section["sampled"] if r["status"] == "skipped"]
    if skipped:
        lines.append("")
        lines.append("**Skipped reasons:**")
        for r in skipped:
            lines.append(f"- `{r['condition_id']}`: {r['reason']}")
    lines.append("")
    return lines


# -- Section 5: Analysis re-derivation ---------------------------------------


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _diff_csv(a: Path, b: Path) -> bool:
    """Return True iff CSVs differ (byte-level)."""
    return not filecmp.cmp(str(a), str(b), shallow=False)


def _diff_json(a: Path, b: Path, float_tolerance: float = 1e-9) -> list[str]:
    """Return list of differing JSON paths/keys, with float tolerance."""
    with open(a, "r", encoding="utf-8") as fa:
        ja = json.load(fa)
    with open(b, "r", encoding="utf-8") as fb:
        jb = json.load(fb)
    diffs: list[str] = []
    _walk_json_diff(ja, jb, "", diffs, float_tolerance)
    return diffs


def _walk_json_diff(a: Any, b: Any, path: str, diffs: list[str], tol: float) -> None:
    if type(a) is not type(b):
        # Permit int<->float compare if both numeric.
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > tol:
                diffs.append(f"{path}: {a} != {b}")
            return
        diffs.append(f"{path}: type mismatch ({type(a).__name__} vs {type(b).__name__})")
        return
    if isinstance(a, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            if k not in a:
                diffs.append(f"{path}/{k}: missing in left")
            elif k not in b:
                diffs.append(f"{path}/{k}: missing in right")
            else:
                _walk_json_diff(a[k], b[k], f"{path}/{k}", diffs, tol)
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length mismatch ({len(a)} vs {len(b)})")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk_json_diff(x, y, f"{path}[{i}]", diffs, tol)
    elif isinstance(a, float):
        if abs(a - b) > tol:
            diffs.append(f"{path}: {a} != {b}")
    else:
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")


def _section_analysis_rederivation(
    normalized_dir: Path,
    analysis_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "title": "5. Analysis re-derivation",
        "variance_components": {
            "stored_path": str(analysis_dir / "variance_components" / "aggregate_vc.csv"),
            "status": "not yet available",
            "diff_summary": "",
        },
        "tolerance_schedule": {
            "stored_path": str(analysis_dir / "tolerance_schedules" / "tolerance_by_cell.csv"),
            "status": "not yet available",
            "diff_summary": "",
        },
        "failures": 0,
        "artifacts_checked": 0,
    }

    vc_path = analysis_dir / "variance_components" / "aggregate_vc.csv"
    if vc_path.exists():
        section["artifacts_checked"] += 1
        ok, summary = _attempt_rederive_variance_components(
            stored_path=vc_path,
            normalized_dir=normalized_dir,
            config_path=config_path,
        )
        section["variance_components"]["status"] = (
            "pass" if ok else ("skipped" if "inputs unavailable" in summary else "fail")
        )
        section["variance_components"]["diff_summary"] = summary
        if not ok and "inputs unavailable" not in summary:
            section["failures"] += 1

    tol_path = analysis_dir / "tolerance_schedules" / "tolerance_by_cell.csv"
    if tol_path.exists():
        section["artifacts_checked"] += 1
        ok, summary = _attempt_rederive_tolerance_schedule(
            stored_path=tol_path,
            normalized_dir=normalized_dir,
            analysis_dir=analysis_dir,
            config_path=config_path,
        )
        section["tolerance_schedule"]["status"] = (
            "pass" if ok else ("skipped" if "inputs unavailable" in summary else "fail")
        )
        section["tolerance_schedule"]["diff_summary"] = summary
        if not ok and "inputs unavailable" not in summary:
            section["failures"] += 1

    return section


def _attempt_rederive_variance_components(
    stored_path: Path,
    normalized_dir: Path,
    config_path: Path,
) -> tuple[bool, str]:
    """Invoke 06_variance_components.py to re-derive aggregate_vc.csv.

    Returns (ok, summary). If the script doesn't exist or required
    inputs are missing, returns (True, "<reason>; re-derivation skipped")
    — graceful degradation.
    """
    script_path = _scripts_dir() / "06_variance_components.py"
    if not script_path.exists():
        return True, "06_variance_components.py not yet implemented; diff skipped"

    item_path = normalized_dir / "item_level_primary.parquet"
    cond_path = normalized_dir / "condition_level_primary.csv"
    missing = [p.name for p in (item_path, cond_path, config_path) if not p.exists()]
    if missing:
        return True, f"inputs unavailable ({', '.join(missing)}); re-derivation skipped"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "variance_components"
        try:
            result = subprocess.run(
                [
                    sys.executable, str(script_path),
                    "--item-level", str(item_path),
                    "--condition-level", str(cond_path),
                    "--config", str(config_path),
                    "--out-dir", str(out_dir),
                ],
                capture_output=True, text=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, f"06_variance_components.py invocation error: {e}"
        # Exit codes 0 (clean) and 4 (Level-4 fallback used) are both
        # non-failure per SPEC §8.7.
        if result.returncode not in (0, 4):
            return False, (
                f"06_variance_components.py exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:200]}"
            )
        rederived = out_dir / "aggregate_vc.csv"
        if not rederived.exists():
            return False, "06_variance_components.py did not produce aggregate_vc.csv"
        if _diff_csv(stored_path, rederived):
            return False, "aggregate_vc.csv differs from stored output"
        return True, "aggregate_vc.csv re-derived identically"


def _attempt_rederive_tolerance_schedule(
    stored_path: Path,
    normalized_dir: Path,
    analysis_dir: Path,
    config_path: Path,
) -> tuple[bool, str]:
    """Invoke 07_tolerance_schedule.py to re-derive tolerance_by_cell.csv."""
    script_path = _scripts_dir() / "07_tolerance_schedule.py"
    if not script_path.exists():
        return True, "07_tolerance_schedule.py not yet implemented; diff skipped"

    cond_path = normalized_dir / "condition_level_primary.csv"
    vc_path = analysis_dir / "variance_components" / "aggregate_vc.csv"
    missing = [p.name for p in (cond_path, vc_path, config_path) if not p.exists()]
    if missing:
        return True, f"inputs unavailable ({', '.join(missing)}); re-derivation skipped"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "tolerance_schedules"
        try:
            result = subprocess.run(
                [
                    sys.executable, str(script_path),
                    "--condition-level", str(cond_path),
                    "--variance-components", str(vc_path),
                    "--analysis-config", str(config_path),
                    "--out-dir", str(out_dir),
                ],
                capture_output=True, text=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, f"07_tolerance_schedule.py invocation error: {e}"
        if result.returncode != 0:
            return False, (
                f"07_tolerance_schedule.py exited {result.returncode}: "
                f"{(result.stderr or '').strip()[:200]}"
            )
        rederived = out_dir / "tolerance_by_cell.csv"
        if not rederived.exists():
            return False, "07_tolerance_schedule.py did not produce tolerance_by_cell.csv"
        if _diff_csv(stored_path, rederived):
            return False, "tolerance_by_cell.csv differs from stored output"
        return True, "tolerance_by_cell.csv re-derived identically"
        return True, f"{out_filename} matches stored output"


def _render_analysis_rederivation(section: dict[str, Any]) -> list[str]:
    lines = ["## " + section["title"], ""]
    vc = section["variance_components"]
    lines.append(f"- **Variance components** (`{vc['stored_path']}`)")
    lines.append(f"    - status: {vc['status']}")
    if vc["diff_summary"]:
        lines.append(f"    - {vc['diff_summary']}")
    tol = section["tolerance_schedule"]
    lines.append(f"- **Tolerance schedule** (`{tol['stored_path']}`)")
    lines.append(f"    - status: {tol['status']}")
    if tol["diff_summary"]:
        lines.append(f"    - {tol['diff_summary']}")
    lines.append("")
    return lines


# -- Section 6: Cross-environment check --------------------------------------


def _section_cross_env(
    analysis_dir: Path,
    compare_env: Path | None,
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "title": "6. Cross-environment check",
        "performed": False,
        "compare_env_path": str(compare_env) if compare_env else None,
        "diffs": [],     # list of {path, status, detail}
        "failures": 0,
        "artifacts_checked": 0,
    }
    if compare_env is None:
        return section
    if not compare_env.exists():
        section["diffs"].append({
            "path": str(compare_env),
            "status": "missing",
            "detail": "compare-env directory does not exist",
        })
        return section
    section["performed"] = True

    # Walk every file in the local analysis dir and try to match in compare_env.
    if not analysis_dir.exists():
        return section

    for local_file in sorted(analysis_dir.rglob("*")):
        if not local_file.is_file():
            continue
        rel = local_file.relative_to(analysis_dir)
        peer = compare_env / rel
        section["artifacts_checked"] += 1
        if not peer.exists():
            section["diffs"].append({
                "path": str(rel), "status": "fail",
                "detail": "missing in compare-env",
            })
            section["failures"] += 1
            continue
        if local_file.suffix == ".json":
            try:
                jdiffs = _diff_json(local_file, peer)
            except json.JSONDecodeError as e:
                section["diffs"].append({
                    "path": str(rel), "status": "fail",
                    "detail": f"JSON decode error: {e}",
                })
                section["failures"] += 1
                continue
            if jdiffs:
                section["diffs"].append({
                    "path": str(rel), "status": "fail",
                    "detail": "; ".join(jdiffs[:5])
                              + (" ..." if len(jdiffs) > 5 else ""),
                })
                section["failures"] += 1
            else:
                section["diffs"].append({
                    "path": str(rel), "status": "pass", "detail": "json equal",
                })
        else:
            if filecmp.cmp(str(local_file), str(peer), shallow=False):
                section["diffs"].append({
                    "path": str(rel), "status": "pass", "detail": "byte-equal",
                })
            else:
                section["diffs"].append({
                    "path": str(rel), "status": "fail",
                    "detail": "byte-level differ",
                })
                section["failures"] += 1
    return section


def _render_cross_env(section: dict[str, Any]) -> list[str]:
    lines = ["## " + section["title"], ""]
    if not section["performed"]:
        if section["compare_env_path"]:
            lines.append(
                f"- compare-env path provided (`{section['compare_env_path']}`) "
                f"but does not exist; check skipped."
            )
        else:
            lines.append("- not performed (no `--compare-env` provided).")
        lines.append("")
        return lines
    lines.append(f"- compare-env: `{section['compare_env_path']}`")
    lines.append(f"- artifacts compared: {section['artifacts_checked']}")
    lines.append(f"- failures: {section['failures']}")
    if section["diffs"]:
        lines.append("")
        lines.append("| Path | Status | Detail |")
        lines.append("|---|---|---|")
        for d in section["diffs"]:
            lines.append(f"| `{d['path']}` | {d['status']} | {d['detail']} |")
    lines.append("")
    return lines


# -- Section 7: Summary ------------------------------------------------------


def _section_summary(sections: list[dict[str, Any]]) -> dict[str, Any]:
    total_artifacts = sum(s.get("artifacts_checked", 0) for s in sections)
    total_failures = sum(s.get("failures", 0) for s in sections)
    return {
        "title": "7. Summary",
        "total_artifacts_checked": total_artifacts,
        "total_failures": total_failures,
        "trace_status": "pass" if total_failures == 0 else "fail",
    }


def _render_summary(summary: dict[str, Any]) -> list[str]:
    lines = ["## " + summary["title"], ""]
    lines.append(
        f"- Total reproducibility-critical artifacts checked: "
        f"{summary['total_artifacts_checked']}"
    )
    lines.append(f"- Total reproducibility failures: {summary['total_failures']}")
    lines.append(f"- **Trace status: `{summary['trace_status']}`**")
    lines.append("")
    return lines


# -- Trace generation --------------------------------------------------------


def generate_trace(
    config_path: Path,
    manifests_dir: Path,
    raw_dir: Path,
    normalized_dir: Path,
    analysis_dir: Path,
    sample_n: int,
    seed: int,
    compare_env: Path | None,
) -> tuple[str, dict[str, Any]]:
    """Build the trace report; return (markdown, summary_dict)."""
    sec1 = _section_config_state(config_path)
    sec2 = _section_manifest_state(manifests_dir)
    sec3 = _section_raw_coverage(manifests_dir, raw_dir)
    sec4 = _section_recompute_sample(
        raw_dir, normalized_dir, sec3.get("complete_condition_ids", []),
        sample_n, seed,
    )
    sec5 = _section_analysis_rederivation(normalized_dir, analysis_dir, config_path)
    sec6 = _section_cross_env(analysis_dir, compare_env)
    summary = _section_summary([sec1, sec2, sec3, sec4, sec5, sec6])

    timestamp = datetime.now(timezone.utc).isoformat()

    out_lines: list[str] = []
    out_lines.append("# Reproducibility Trace")
    out_lines.append("")
    out_lines.append(f"_Generated: {timestamp}_")
    out_lines.append("")
    out_lines.append("Per SPEC_CPU_v0_2 §13. Verifies that the pipeline's outputs")
    out_lines.append("can be re-derived from the committed inputs.")
    out_lines.append("")
    out_lines.extend(_render_config_state(sec1))
    out_lines.extend(_render_manifest_state(sec2))
    out_lines.extend(_render_raw_coverage(sec3))
    out_lines.extend(_render_recompute_sample(sec4))
    out_lines.extend(_render_analysis_rederivation(sec5))
    out_lines.extend(_render_cross_env(sec6))
    out_lines.extend(_render_summary(summary))
    md = "\n".join(out_lines)

    summary_payload = {
        "trace_status": summary["trace_status"],
        "total_artifacts_checked": summary["total_artifacts_checked"],
        "total_failures": summary["total_failures"],
        "config_hash_full": sec1["config_hash_full"],
        "config_hash": sec1["config_hash"],
        "timestamp": timestamp,
        "section_failures": {
            "config_state": sec1.get("failures", 0),
            "manifest_state": sec2.get("failures", 0),
            "raw_coverage": sec3.get("failures", 0),
            "recompute_sample": sec4.get("failures", 0),
            "analysis_rederivation": sec5.get("failures", 0),
            "cross_env": sec6.get("failures", 0),
        },
    }
    return md, summary_payload


# -- Main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the reproducibility-trace report "
                    "(SPEC_CPU_v0_2 §8.12, §13)."
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to study_config.yaml.")
    parser.add_argument("--manifests-dir", required=True, type=Path,
                        help="Directory containing the manifest CSVs.")
    parser.add_argument("--raw-dir", required=True, type=Path,
                        help="Directory containing per-condition raw run dirs.")
    parser.add_argument("--normalized-dir", required=True, type=Path,
                        help="Directory containing normalized item-level "
                             "and condition-level outputs.")
    parser.add_argument("--analysis-dir", required=True, type=Path,
                        help="Directory containing analysis outputs "
                             "(variance_components/, tolerance_schedules/, ...).")
    parser.add_argument("--sample-n", type=int, default=5,
                        help="Number of conditions to sample for the "
                             "recompute check (default: 5).")
    parser.add_argument("--seed", type=int, default=20260424,
                        help="Seed for the recompute-sample RNG "
                             "(default: 20260424).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output markdown path "
                             "(typically reports/reproducibility_trace.md).")
    parser.add_argument("--compare-env", type=Path, default=None,
                        help="Optional path to a second environment's "
                             "analysis outputs for cross-environment diff.")
    args = parser.parse_args()

    try:
        md, summary = generate_trace(
            config_path=args.config,
            manifests_dir=args.manifests_dir,
            raw_dir=args.raw_dir,
            normalized_dir=args.normalized_dir,
            analysis_dir=args.analysis_dir,
            sample_n=args.sample_n,
            seed=args.seed,
            compare_env=args.compare_env,
        )
    except Exception as e:  # noqa: BLE001
        _log("CRITICAL", f"Unhandled exception: {type(e).__name__}: {e}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
        if not md.endswith("\n"):
            f.write("\n")

    if summary["trace_status"] == "pass":
        _log("INFO",
             f"Reproducibility trace: PASS "
             f"({summary['total_artifacts_checked']} artifacts checked, "
             f"0 failures).")
        return 0
    else:
        _log("ERROR",
             f"Reproducibility trace: FAIL "
             f"({summary['total_failures']} failures across "
             f"{summary['total_artifacts_checked']} artifacts).")
        return 5


if __name__ == "__main__":
    sys.exit(main())
