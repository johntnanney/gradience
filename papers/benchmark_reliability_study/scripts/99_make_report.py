"""99_make_report.py — SPEC_CPU_v0_2 §8.13

Assemble the top-level CPU pipeline report. Links every analysis output,
embeds the deviations log, and surfaces the reproducibility-trace
status. This is the single document a reviewer reads to navigate the
pipeline's outputs.

Usage:
    python scripts/99_make_report.py \\
      --analysis-dir analysis/ \\
      --tables-dir tables/ \\
      --figures-dir figures/ \\
      --out reports/cpu_pipeline_report.md \\
      [--config configs/study_config.yaml] \\
      [--manifests-dir manifests/] \\
      [--reports-dir reports/]

Sections:
    1. Pipeline execution summary.
    2. Headline numbers (H1, tolerances, ranking stability, H4).
    3. Artifact links (analysis, tables, figures, reports).
    4. Deviations.
    5. Reproducibility trace status.

Graceful degradation: missing artifacts produce "not yet available"
annotations rather than errors.

Exit codes per SPEC §11.1:
    0 — success
    1 — generic failure (implementation error)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the paper-root package importable without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradience_study.config import (  # noqa: E402
    ConfigValidationError,
    compute_config_hash,
    load_config,
)


MANIFEST_FILES: tuple[str, ...] = (
    "conditions_primary.csv",
    "conditions_gsm8k.csv",
    "prompt_manifest.csv",
    "fewshot_manifest.csv",
)


# -- Logging -----------------------------------------------------------------


def _log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] [{level}] [99_make_report] {msg}", file=sys.stderr)


# -- Helpers -----------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_csv_rows(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        for _ in reader:
            n += 1
    return n


def _rel_link(target: Path, anchored_at: Path) -> str:
    """Render a markdown-friendly relative path from anchored_at to target."""
    try:
        return str(target.resolve().relative_to(anchored_at.resolve()))
    except ValueError:
        return str(target)


# -- Section 1: Pipeline execution summary ----------------------------------


def _summary_pipeline_execution(
    config_path: Path | None,
    manifests_dir: Path | None,
    normalized_dir: Path | None,
    raw_dir: Path | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": "1. Pipeline execution summary",
        "config_hash_full": None,
        "config_hash": None,
        "study_id": None,
        "prereg_version": None,
        "manifests": {},
        "raw_subdir_count": None,
        "normalized_files": [],
    }
    if config_path is not None and config_path.exists():
        try:
            merged = load_config(config_path)
            full, short = compute_config_hash(merged.raw_dict)
            out["config_hash_full"] = full
            out["config_hash"] = short
            out["study_id"] = merged.study_config.study_id
            out["prereg_version"] = merged.study_config.prereg_version
        except (FileNotFoundError, ConfigValidationError, KeyError) as e:
            out["config_load_error"] = f"{type(e).__name__}: {e}"

    if manifests_dir is not None and manifests_dir.exists():
        for name in MANIFEST_FILES:
            p = manifests_dir / name
            if p.exists():
                out["manifests"][name] = {
                    "hash": _sha256_file(p),
                    "row_count": _count_csv_rows(p),
                    "present": True,
                }
            else:
                out["manifests"][name] = {
                    "hash": None, "row_count": None, "present": False,
                }

    if raw_dir is not None and raw_dir.exists():
        out["raw_subdir_count"] = sum(
            1 for p in raw_dir.iterdir() if p.is_dir()
        )

    if normalized_dir is not None and normalized_dir.exists():
        for f in sorted(normalized_dir.iterdir()):
            if f.is_file():
                row_count = None
                if f.suffix == ".csv":
                    try:
                        row_count = _count_csv_rows(f)
                    except OSError:
                        row_count = None
                out["normalized_files"].append({
                    "name": f.name,
                    "row_count": row_count,
                })
    return out


def _render_pipeline_summary(s: dict[str, Any]) -> list[str]:
    lines = ["## " + s["title"], ""]
    lines.append("**Study identity**")
    lines.append("")
    lines.append(f"- Study ID: `{s['study_id'] or 'n/a'}`")
    lines.append(f"- Prereg version: `{s['prereg_version'] or 'n/a'}`")
    lines.append(f"- Config hash (short): `{s['config_hash'] or 'n/a'}`")
    if s.get("config_hash_full"):
        lines.append(f"- Config hash (full): `{s['config_hash_full']}`")
    if s.get("config_load_error"):
        lines.append(f"- Config load error: {s['config_load_error']}")
    lines.append("")

    lines.append("**Manifests**")
    lines.append("")
    if s["manifests"]:
        lines.append("| Manifest | SHA-256 | Rows |")
        lines.append("|---|---|---|")
        for name, info in s["manifests"].items():
            if info["present"]:
                lines.append(f"| `{name}` | `{info['hash']}` | {info['row_count']} |")
            else:
                lines.append(f"| `{name}` | missing | — |")
    else:
        lines.append("- not yet available (no manifests directory provided or empty)")
    lines.append("")

    lines.append("**Inference outputs**")
    lines.append("")
    if s["raw_subdir_count"] is None:
        lines.append("- raw-run directory: not yet available")
    else:
        lines.append(f"- raw-run subdirectories: {s['raw_subdir_count']}")
    if s["normalized_files"]:
        lines.append("- normalized files:")
        for f in s["normalized_files"]:
            rc = f"{f['row_count']} rows" if f["row_count"] is not None else "—"
            lines.append(f"    - `{f['name']}` ({rc})")
    else:
        lines.append("- normalized files: not yet available")
    lines.append("")
    return lines


# -- Section 2: Headline numbers ---------------------------------------------


def _headline_numbers(analysis_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": "2. Headline numbers",
        "h1": None,
        "tolerance_by_benchmark": [],
        "ranking_stability_files": [],
        "h4": None,
    }
    if not analysis_dir.exists():
        return out

    h1_path = analysis_dir / "tolerance_schedules" / "h1_test.json"
    if h1_path.exists():
        try:
            with open(h1_path, "r", encoding="utf-8") as f:
                out["h1"] = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            out["h1_error"] = str(e)

    summary_csv = analysis_dir / "tolerance_schedules" / "tolerance_by_benchmark_summary.csv"
    if summary_csv.exists():
        try:
            with open(summary_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    out["tolerance_by_benchmark"].append(row)
        except OSError:
            pass

    rs_dir = analysis_dir / "ranking_stability"
    if rs_dir.exists():
        for f in sorted(rs_dir.iterdir()):
            if f.is_file():
                out["ranking_stability_files"].append(f.name)

    h4_path = analysis_dir / "mmlu_subjects" / "h4_test.json"
    if h4_path.exists():
        try:
            with open(h4_path, "r", encoding="utf-8") as f:
                out["h4"] = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            out["h4_error"] = str(e)

    return out


def _render_headline(h: dict[str, Any]) -> list[str]:
    lines = ["## " + h["title"], ""]
    lines.append("**H1 (single-occasion tolerance dominates the precision frontier)**")
    lines.append("")
    if h["h1"] is None:
        lines.append("- not yet available")
    else:
        h1 = h["h1"]
        confirmed = h1.get("h1_confirmed")
        decision = "CONFIRMED" if confirmed else "NOT CONFIRMED" if confirmed is not None else "?"
        lines.append(f"- Decision: **{decision}**")
        if "n_exceeding" in h1 and "n_required" in h1:
            lines.append(
                f"- Benchmarks exceeding threshold: {h1['n_exceeding']} "
                f"(required: {h1['n_required']})"
            )
        if h1.get("benchmarks_exceeding"):
            joined = ", ".join(f"`{b}`" for b in h1["benchmarks_exceeding"])
            lines.append(f"- Exceeding: {joined}")
        if "threshold" in h1:
            lines.append(f"- Threshold: {h1['threshold']}")
    lines.append("")

    lines.append("**Per-benchmark cross-model median tolerance**")
    lines.append("")
    if not h["tolerance_by_benchmark"]:
        lines.append("- not yet available")
    else:
        cols = list(h["tolerance_by_benchmark"][0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for row in h["tolerance_by_benchmark"]:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    lines.append("")

    lines.append("**Ranking stability**")
    lines.append("")
    if not h["ranking_stability_files"]:
        lines.append("- not yet available")
    else:
        for name in h["ranking_stability_files"]:
            lines.append(f"- `{name}`")
    lines.append("")

    lines.append("**H4 (MMLU model x subject interaction)**")
    lines.append("")
    if h["h4"] is None:
        lines.append("- not yet available")
    else:
        h4 = h["h4"]
        confirmed = h4.get("h4_confirmed")
        decision = "CONFIRMED" if confirmed else "NOT CONFIRMED" if confirmed is not None else "?"
        lines.append(f"- Decision: **{decision}**")
        if "model_x_subject_proportion" in h4:
            lines.append(
                f"- Model x subject interaction proportion: "
                f"{h4['model_x_subject_proportion']}"
            )
        if "threshold" in h4:
            lines.append(f"- Threshold: {h4['threshold']}")
    lines.append("")
    return lines


# -- Section 3: Artifact links -----------------------------------------------


def _artifact_links(
    analysis_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    reports_dir: Path | None,
    out_path: Path,
) -> dict[str, Any]:
    out_anchor = out_path.resolve().parent
    out: dict[str, Any] = {
        "title": "3. Artifact links",
        "analysis": [],
        "tables": [],
        "figures": [],
        "reports": [],
    }
    for d, key in (
        (analysis_dir, "analysis"),
        (tables_dir, "tables"),
        (figures_dir, "figures"),
    ):
        if d.exists():
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    out[key].append(_rel_link(f, out_anchor))
    if reports_dir is not None and reports_dir.exists():
        for f in sorted(reports_dir.rglob("*")):
            if f.is_file() and f.resolve() != out_path.resolve():
                out["reports"].append(_rel_link(f, out_anchor))
    return out


def _render_artifact_links(a: dict[str, Any]) -> list[str]:
    lines = ["## " + a["title"], ""]
    for key, label in (
        ("analysis", "Analysis"),
        ("tables", "Tables"),
        ("figures", "Figures"),
        ("reports", "Reports"),
    ):
        lines.append(f"**{label}**")
        lines.append("")
        if not a[key]:
            lines.append(f"- not yet available")
        else:
            for path_str in a[key]:
                lines.append(f"- [`{path_str}`]({path_str})")
        lines.append("")
    return lines


# -- Section 4: Deviations ---------------------------------------------------


def _section_deviations(reports_dir: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": "4. Deviations",
        "content": None,
        "path": None,
    }
    if reports_dir is None:
        return out
    p = reports_dir / "deviations.md"
    if p.exists():
        out["path"] = str(p)
        try:
            out["content"] = p.read_text(encoding="utf-8")
        except OSError as e:
            out["content"] = f"<error reading deviations.md: {e}>"
    return out


def _render_deviations(d: dict[str, Any]) -> list[str]:
    lines = ["## " + d["title"], ""]
    if d["content"] is None:
        lines.append("No deviations recorded.")
        lines.append("")
        return lines
    lines.append(f"_Embedded from `{d['path']}`._")
    lines.append("")
    lines.append(d["content"].rstrip("\n"))
    lines.append("")
    return lines


# -- Section 5: Reproducibility-trace status ---------------------------------


def _section_repro_status(reports_dir: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": "5. Reproducibility trace status",
        "trace_present": False,
        "trace_status": None,
        "trace_path": None,
        "summary_excerpt": None,
    }
    if reports_dir is None:
        return out
    p = reports_dir / "reproducibility_trace.md"
    if not p.exists():
        return out
    out["trace_present"] = True
    out["trace_path"] = str(p)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        out["summary_excerpt"] = f"<error reading trace: {e}>"
        return out

    # Heuristic parse: find a "Trace status: `pass`" or `fail` line, and
    # excerpt the Summary section if present.
    status: str | None = None
    for line in text.splitlines():
        lower = line.lower()
        if "trace status" in lower:
            if "pass" in lower:
                status = "pass"
            elif "fail" in lower:
                status = "fail"
            if status:
                break
    out["trace_status"] = status

    # Excerpt: from "## 7. Summary" or "## Summary" through end (or next ##).
    excerpt_lines: list[str] = []
    in_summary = False
    for line in text.splitlines():
        if line.startswith("## ") and "summary" in line.lower():
            in_summary = True
            excerpt_lines.append(line)
            continue
        if in_summary:
            if line.startswith("## "):
                break
            excerpt_lines.append(line)
    if excerpt_lines:
        out["summary_excerpt"] = "\n".join(excerpt_lines).rstrip("\n")
    return out


def _render_repro_status(r: dict[str, Any]) -> list[str]:
    lines = ["## " + r["title"], ""]
    if not r["trace_present"]:
        lines.append("Reproducibility trace not yet generated.")
        lines.append("")
        return lines
    lines.append(f"- Trace document: `{r['trace_path']}`")
    if r["trace_status"]:
        lines.append(f"- Trace status: **`{r['trace_status']}`**")
    else:
        lines.append("- Trace status: could not parse status from trace document.")
    if r["summary_excerpt"]:
        lines.append("")
        lines.append("**Trace summary excerpt:**")
        lines.append("")
        lines.append(r["summary_excerpt"])
    lines.append("")
    return lines


# -- Report assembly ---------------------------------------------------------


def build_report(
    analysis_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    out_path: Path,
    config_path: Path | None,
    manifests_dir: Path | None,
    reports_dir: Path | None,
    normalized_dir: Path | None,
    raw_dir: Path | None,
) -> str:
    s1 = _summary_pipeline_execution(
        config_path, manifests_dir, normalized_dir, raw_dir
    )
    s2 = _headline_numbers(analysis_dir)
    s3 = _artifact_links(
        analysis_dir, tables_dir, figures_dir, reports_dir, out_path
    )
    s4 = _section_deviations(reports_dir)
    s5 = _section_repro_status(reports_dir)

    timestamp = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append("# CPU Pipeline Report")
    lines.append("")
    lines.append(f"_Generated: {timestamp}_")
    lines.append("")
    lines.append(
        "Per SPEC_CPU_v0_2 §8.13. Top-level navigation document for the "
        "benchmark-reliability-study CPU pipeline outputs."
    )
    lines.append("")
    lines.extend(_render_pipeline_summary(s1))
    lines.extend(_render_headline(s2))
    lines.extend(_render_artifact_links(s3))
    lines.extend(_render_deviations(s4))
    lines.extend(_render_repro_status(s5))
    return "\n".join(lines)


# -- Main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the CPU pipeline report (SPEC_CPU_v0_2 §8.13)."
    )
    parser.add_argument("--analysis-dir", required=True, type=Path,
                        help="Directory containing analysis outputs.")
    parser.add_argument("--tables-dir", required=True, type=Path,
                        help="Directory containing manuscript tables.")
    parser.add_argument("--figures-dir", required=True, type=Path,
                        help="Directory containing manuscript figures.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output markdown path "
                             "(typically reports/cpu_pipeline_report.md).")
    # Optional contextual inputs.
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to study_config.yaml (for config hash).")
    parser.add_argument("--manifests-dir", type=Path, default=None,
                        help="Manifests directory (for manifest hashes/counts).")
    parser.add_argument("--reports-dir", type=Path, default=None,
                        help="Reports directory (for deviations.md, "
                             "reproducibility_trace.md links).")
    parser.add_argument("--normalized-dir", type=Path, default=None,
                        help="Normalized outputs directory.")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Raw runs directory.")
    args = parser.parse_args()

    try:
        md = build_report(
            analysis_dir=args.analysis_dir,
            tables_dir=args.tables_dir,
            figures_dir=args.figures_dir,
            out_path=args.out,
            config_path=args.config,
            manifests_dir=args.manifests_dir,
            reports_dir=args.reports_dir,
            normalized_dir=args.normalized_dir,
            raw_dir=args.raw_dir,
        )
    except Exception as e:  # noqa: BLE001
        _log("CRITICAL", f"Unhandled exception: {type(e).__name__}: {e}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
        if not md.endswith("\n"):
            f.write("\n")

    _log("INFO", f"Pipeline report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
