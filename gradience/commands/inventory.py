"""Inventory commands — summarize, neighborhoods, batch, portfolio, preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gradience.exceptions import ConfigError, DependencyError, GradienceError, QASchemaError


def cmd_portfolio(args: argparse.Namespace) -> None:
    """Display a cross-inventory portfolio table."""
    directory = getattr(args, "directory", None)
    html_path = getattr(args, "html", None)
    strict_input = bool(getattr(args, "strict_input", False))

    try:
        from gradience.vnext.inventory.portfolio import (
            format_portfolio_html,
            format_portfolio_table,
            scan_portfolio,
        )
    except ImportError as e:
        raise DependencyError(f"Failed to import portfolio module: {e}") from e

    try:
        if directory is None:
            directory = "."
        view = scan_portfolio(directory, strict_input=strict_input)
    except GradienceError:
        raise
    except (ValueError, Exception) as e:
        raise ConfigError(f"Portfolio scan failed: {e}") from e

    if html_path:
        html = format_portfolio_html(view)
        out = Path(html_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"Portfolio HTML written to {html_path}")
        print(f"  {len(view.rows)} inventor{'y' if len(view.rows) == 1 else 'ies'} included")
    else:
        print(format_portfolio_table(view))


def cmd_summarize_inventory(args: argparse.Namespace) -> None:
    """Summarize an inventory of adapter QA artifacts and merge risk reports."""
    qa_dir = getattr(args, "qa_dir", None)
    report_dir = getattr(args, "report_dir", None)
    strict_input = bool(getattr(args, "strict_input", False))
    emit_path = getattr(args, "emit_report", None)

    if not qa_dir and not report_dir:
        raise ConfigError("At least one of --qa-dir or --report-dir is required")

    try:
        from gradience.api import summarize_inventory
        from gradience.vnext.inventory.summary import (
            build_action_plan,
            format_action_plan,
            format_inventory_summary,
        )
    except ImportError as e:
        raise DependencyError(f"Failed to import inventory modules: {e}") from e

    try:
        summary = summarize_inventory(qa_dir=qa_dir, report_dir=report_dir, strict_input=strict_input)
    except GradienceError:
        raise
    except Exception as e:
        raise GradienceError(f"Inventory summary failed: {e}") from e

    print(format_inventory_summary(summary))

    # --- Action plan (derived from raw artifacts) ---
    try:
        from gradience.api import _load_schema_artifacts  # noqa: PLC0415
        from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
        from gradience.vnext.merge.qa_report import MergeQAReport

        qa_files = sorted(Path(qa_dir).glob("*.json")) if qa_dir else []
        report_files = sorted(Path(report_dir).glob("*.json")) if report_dir else []

        qa_artifacts = _load_schema_artifacts(
            files=qa_files,
            schema_prefix="gradience.adapter_qa/",
            from_dict=AdapterQAArtifact.from_dict,
            strict_input=strict_input,
            malformed_label="QA artifact",
        )
        merge_reports = _load_schema_artifacts(
            files=report_files,
            schema_prefix="gradience.merge_qa_report/",
            from_dict=MergeQAReport.from_dict,
            strict_input=strict_input,
            malformed_label="merge report",
        )

        action_plan = build_action_plan(qa_artifacts, merge_reports)
        print(format_action_plan(action_plan))
    except Exception:
        action_plan = None
        qa_artifacts = []
        merge_reports = []

    if emit_path:
        try:
            summary.to_json(emit_path)
            print(f"Inventory summary written to {emit_path}")
        except Exception as e:
            raise GradienceError(f"Failed to write inventory report: {e}") from e

    # --- Run bundle emission ---
    emit_bundle = getattr(args, "emit_bundle", None)
    if emit_bundle and action_plan is not None:
        from datetime import datetime, timezone

        try:
            from gradience.vnext.inventory.run_bundle import emit_run_bundle, update_latest_pointer

            bundle_dir = Path(emit_bundle)
            inventory_id = getattr(args, "inventory_id", None) or bundle_dir.parent.name or "inventory"
            run_id = getattr(args, "run_id", None) or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

            previous_run = getattr(args, "previous_run", None)
            if previous_run:
                previous_run_dir: Path | None = Path(previous_run)
            else:
                # Auto-discover previous run from sibling directories
                from gradience.vnext.inventory.batch import discover_previous_run

                previous_run_dir = discover_previous_run(bundle_dir.parent)
                if previous_run_dir and previous_run_dir.resolve() == bundle_dir.resolve():
                    previous_run_dir = None  # Don't compare to self

            emit_run_bundle(
                inventory_id=inventory_id,
                run_id=run_id,
                run_dir=bundle_dir,
                qa_artifacts=qa_artifacts,
                merge_reports=merge_reports,
                action_plan=action_plan,
                previous_run_dir=previous_run_dir,
            )

            # Update latest pointer if bundle is inside a recognized root
            import contextlib

            if bundle_dir.parent.exists():
                with contextlib.suppress(Exception):
                    update_latest_pointer(bundle_dir.parent, bundle_dir)

            print(f"\nPreflight bundle written to {bundle_dir}/")
            print(f"  Start here: {bundle_dir / 'preflight_summary.md'}")
        except Exception as e:
            print(f"Warning: Bundle emission failed: {e}", file=sys.stderr)


def cmd_suggest_neighborhoods(args: argparse.Namespace) -> None:
    """Suggest conservative merge neighborhoods from preflight artifacts."""
    import sys as _sys

    qa_dir = getattr(args, "qa_dir", None)
    report_dir = getattr(args, "report_dir", None)
    strict_input = bool(getattr(args, "strict_input", False))
    strict_qa = bool(getattr(args, "strict_qa", False))
    min_compatibility = float(getattr(args, "min_compatibility", 0.0))
    exclude_unknown = bool(getattr(args, "exclude_unknown", False))
    emit_path = getattr(args, "emit_report", None)

    if not report_dir:
        raise ConfigError("--report-dir is required")

    try:
        from gradience.vnext.audit.qa_artifact import AdapterQAArtifact
        from gradience.vnext.inventory.merge_neighborhood_report import format_merge_neighborhood_report
        from gradience.vnext.inventory.neighborhoods import suggest_merge_neighborhoods
        from gradience.vnext.merge.qa_report import MergeQAReport
    except ImportError as e:
        raise DependencyError(f"Failed to import neighborhoods modules: {e}") from e

    qa_artifacts: list[Any] = []
    merge_reports: list[Any] = []

    qa_files = sorted(Path(qa_dir).glob("*.json")) if qa_dir else []
    report_files = sorted(Path(report_dir).glob("*.json"))

    for fp in qa_files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception as e:
            if strict_input:
                raise QASchemaError(f"Unreadable file under --strict-input: {fp} ({e})") from e
            print(f"WARNING: skipping unreadable file: {fp}", file=_sys.stderr)
            continue
        schema = data.get("schema", "")
        if not isinstance(schema, str) or not schema.startswith("gradience.adapter_qa/"):
            continue
        try:
            qa_artifacts.append(AdapterQAArtifact.from_dict(data))
        except Exception as e:
            if strict_input:
                raise QASchemaError(f"Malformed QA artifact under --strict-input: {fp} ({e})") from e
            print(f"WARNING: skipping malformed QA artifact: {fp}", file=_sys.stderr)

    for fp in report_files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception as e:
            if strict_input:
                raise QASchemaError(f"Unreadable file under --strict-input: {fp} ({e})") from e
            print(f"WARNING: skipping unreadable file: {fp}", file=_sys.stderr)
            continue
        schema = data.get("schema", "")
        if not isinstance(schema, str) or not schema.startswith("gradience.merge_qa_report/"):
            continue
        try:
            merge_reports.append(MergeQAReport.from_dict(data))
        except Exception as e:
            if strict_input:
                raise QASchemaError(f"Malformed merge report under --strict-input: {fp} ({e})") from e
            print(f"WARNING: skipping malformed merge report: {fp}", file=_sys.stderr)

    if not merge_reports:
        raise ConfigError("No valid merge reports were found in --report-dir")

    report = suggest_merge_neighborhoods(
        qa_artifacts,
        merge_reports,
        strict_qa=strict_qa,
        min_compatibility=min_compatibility,
        exclude_unknown=exclude_unknown,
    )
    print(format_merge_neighborhood_report(report))

    if emit_path:
        try:
            report.to_json(emit_path)
            print(f"Merge neighborhoods report written to {emit_path}")
        except Exception as e:
            raise GradienceError(f"Failed to write neighborhoods report: {e}") from e


# ---------------------------------------------------------------------------
# batch-summary
# ---------------------------------------------------------------------------


def cmd_batch_summary(args: argparse.Namespace) -> None:
    """Produce a cross-run comparison table from multiple preflight bundles."""
    run_dir = Path(getattr(args, "run_dir", "."))
    emit_report = getattr(args, "emit_report", None)

    if not run_dir.is_dir():
        raise ConfigError(f"{run_dir} is not a directory")

    from gradience.vnext.inventory.batch import (
        build_batch_summary,
        collect_run_summaries,
        emit_batch_summary,
        format_batch_summary,
    )

    summaries = collect_run_summaries(run_dir)
    if not summaries:
        print(f"No preflight runs found in {run_dir}")
        sys.exit(0)

    batch = build_batch_summary(summaries)
    print(format_batch_summary(batch))

    if emit_report:
        output_dir = Path(emit_report)
        emit_batch_summary(batch=batch, output_dir=output_dir)
        print(f"Batch summary written to {output_dir}/")
        print("  batch_summary.md")
        print("  batch_summary.json")


# ---------------------------------------------------------------------------
# preflight-report (static HTML)
# ---------------------------------------------------------------------------


def cmd_preflight_report(args: argparse.Namespace) -> None:
    """Generate a static HTML preflight report."""
    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.is_dir():
        raise ConfigError(f"{bundle_dir} is not a directory")

    summary_path = bundle_dir / "preflight_summary.json"
    if not summary_path.exists():
        raise ConfigError(
            f"{bundle_dir} does not contain preflight_summary.json.\n"
            "Is this a valid preflight run bundle?"
        )

    from gradience.vnext.inventory.html_report import render_html_report

    html = render_html_report(bundle_dir, title=args.title)

    output = Path(args.output) if args.output else bundle_dir / "preflight_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"Preflight report written to {output}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------



def _setup_portfolio_command(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "portfolio",
        help="[ADVANCED] Scan inventory artifacts and display a cross-inventory portfolio table",
    )
    p.add_argument(
        "directory",
        type=str,
        help="Directory to scan for gradience.inventory_summary/v1 JSON files",
    )
    p.add_argument(
        "--html",
        type=str,
        default=None,
        metavar="PATH",
        help="Write a standalone HTML portfolio page to this path instead of printing a table",
    )
    p.add_argument("--strict-input", action="store_true", help="Fail on first malformed file")
    p.set_defaults(func=cmd_portfolio)


def _setup_suggest_neighborhoods_command(subparsers):
    p = subparsers.add_parser(
        "suggest-neighborhoods",
        help="[ADVANCED] Suggest conservative merge neighborhoods (optional companion, non-default workflow)",
    )
    p.add_argument("--qa-dir", type=str, default=None, help="Directory to scan for QA artifact JSON files")
    p.add_argument("--report-dir", type=str, required=True, help="Directory to scan for merge report JSON files")
    p.add_argument(
        "--emit-report",
        type=str,
        default=None,
        help="Write merge neighborhoods v1 JSON to this path (overwrites existing file)",
    )
    p.add_argument("--strict-qa", action="store_true", help="Exclude adapters that fail strict QA eligibility")
    p.add_argument("--strict-input", action="store_true", help="Fail on first malformed input file")
    p.add_argument(
        "--min-compatibility",
        type=float,
        default=0.0,
        help="Minimum compatibility score required for non-incompatible edges (default: 0.0)",
    )
    p.add_argument(
        "--exclude-unknown",
        action="store_true",
        help="Exclude adapters with unknown or missing QA status from neighborhood construction",
    )
    p.set_defaults(func=cmd_suggest_neighborhoods)


def _setup_summarize_inventory_command(subparsers):
    p = subparsers.add_parser(
        "summarize-inventory",
        help="[RECOMMENDED] Analyze inventory QA + merge reports and produce action-plan summary",
    )
    p.add_argument("--qa-dir", type=str, default=None, help="Directory to scan for QA artifact JSON files")
    p.add_argument("--report-dir", type=str, default=None, help="Directory to scan for merge report JSON files")
    p.add_argument(
        "--emit-report",
        type=str,
        default=None,
        help="Write inventory summary v1 JSON to this path (overwrites existing file)",
    )
    p.add_argument("--strict-input", action="store_true", help="Fail on first malformed file")
    p.add_argument(
        "--emit-bundle",
        type=str,
        default=None,
        help="Emit a preflight run bundle to this directory (summary, manifest, action plan, comparison)",
    )
    p.add_argument(
        "--previous-run",
        type=str,
        default=None,
        help="Path to a previous run bundle for comparison (requires --emit-bundle)",
    )
    p.add_argument(
        "--inventory-id",
        type=str,
        default=None,
        help="Inventory identifier for bundle metadata (default: directory basename)",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier for bundle metadata (default: timestamp)",
    )
    p.set_defaults(func=cmd_summarize_inventory)


def _setup_batch_summary_command(subparsers):
    p = subparsers.add_parser(
        "batch-summary",
        help="[ADVANCED] Summarize multiple preflight runs into a cross-run comparison table",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing preflight run bundle subdirectories",
    )
    p.add_argument(
        "--emit-report",
        type=str,
        default=None,
        help="Write batch summary (JSON + markdown) to this directory",
    )
    p.set_defaults(func=cmd_batch_summary)


def _setup_preflight_report_command(subparsers):
    p = subparsers.add_parser(
        "preflight-report",
        help="[RECOMMENDED] Generate a static HTML preflight report from a run bundle",
    )
    p.add_argument(
        "--bundle-dir",
        type=str,
        required=True,
        help="Path to a preflight run bundle directory (from --emit-bundle)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: <bundle-dir>/preflight_report.html)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional page title override",
    )
    p.set_defaults(func=cmd_preflight_report)



def setup_inventory_commands(subparsers) -> None:
    """Register inventory commands with the argument parser."""
    _setup_summarize_inventory_command(subparsers)
    _setup_portfolio_command(subparsers)
    _setup_suggest_neighborhoods_command(subparsers)
    _setup_batch_summary_command(subparsers)
    _setup_preflight_report_command(subparsers)
