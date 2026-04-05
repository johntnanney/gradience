"""Monitor and report commands — telemetry analysis."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

from gradience.cli_utils import (
    _extract_guard_activity,
    _print_monitor_result,
)
from gradience.exceptions import ConfigError, DependencyError, TelemetryError


def cmd_report(args: argparse.Namespace) -> None:
    """Generate report from telemetry file."""
    import statistics

    telemetry_path = Path(args.file)
    if not telemetry_path.exists():
        raise TelemetryError(f"File not found: {telemetry_path}")

    # Load telemetry
    events: list[dict[str, Any]] = []
    with open(telemetry_path) as f:
        for line in f:
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line))

    if not events:
        raise TelemetryError("No events found in telemetry file")

    # Basic report
    print("=" * 60)
    print("GRADIENCE TELEMETRY REPORT")
    print("=" * 60)
    print(f"\nFile: {telemetry_path}")
    print(f"Total events: {len(events)}")

    # Event breakdown
    event_types: dict[str, int] = {}
    for e in events:
        t = e.get("event", "unknown")
        event_types[t] = event_types.get(t, 0) + 1

    print("\nEvent breakdown:")
    for t, count in sorted(event_types.items()):
        print(f"  {t}: {count}")

    # Spectral summary
    spectral = [e for e in events if e.get("event") == "spectral"]
    if spectral:
        kappas: list[float] = []
        for e in spectral:
            # Support both per-matrix and aggregate spectral events.
            k = e.get("kappa_mean")
            if k is None:
                k = e.get("kappa")
            if isinstance(k, (int, float)):
                kappas.append(float(k))

        if kappas:
            print("\nSpectral metrics:")
            print(f"  κ mean: {statistics.mean(kappas):.1f}")
            print(f"  κ std: {statistics.stdev(kappas) if len(kappas) > 1 else 0:.1f}")
            print(f"  κ range: {min(kappas):.1f} - {max(kappas):.1f}")

    # Eval summary
    evals = [e for e in events if e.get("event") == "eval"]
    if evals:
        accs: list[float] = []
        for e in evals:
            a = e.get("accuracy")
            if a is None:
                a = e.get("acc")
            if isinstance(a, (int, float)):
                accs.append(float(a))

        if accs:
            print("\nEvaluation metrics:")
            print(f"  Final accuracy: {accs[-1]:.1%}")
            print(f"  Best accuracy: {max(accs):.1%}")

    # Guard events
    guard_events = [
        e
        for e in events
        if e.get("event") in ("corruption_detected", "rollback_started", "rollback_succeeded", "would_rollback")
    ]
    if guard_events:
        print("\nGuard events:")
        for e in guard_events:
            print(f"  Step {e.get('step', '?')}: {e.get('event')}")

    print()


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def cmd_monitor(args: argparse.Namespace) -> None:
    """Monitor a vNext telemetry run and emit alerts/recommendations."""

    telemetry_path = Path(args.file)
    if not telemetry_path.exists():
        raise TelemetryError(f"File not found: {telemetry_path}")

    try:
        from gradience.vnext import TelemetryReader
        from gradience.vnext.policy import check_run
    except ImportError as e:
        raise DependencyError(f"Failed to import Gradience vNext components: {e}") from e

    try:
        reader = TelemetryReader(
            telemetry_path,
            strict_schema=bool(getattr(args, "strict_schema", False)),
            normalize=True,
        )
    except (OSError, ValueError) as e:
        raise TelemetryError(f"Failed to open telemetry file: {e}") from e

    # Pull typed summaries
    config = None
    try:
        config = reader.latest_config()
    except (AttributeError, KeyError, ValueError):
        config = None

    try:
        signals = reader.summarize()
    except (AttributeError, KeyError, ValueError) as e:
        raise TelemetryError(f"Failed to summarize telemetry: {e}") from e

    # Emit alerts (simple, gap-first)
    alerts: list[dict[str, Any]] = []
    gap = getattr(signals, "gap", None)
    if gap is not None:
        try:
            gap_f = float(gap)
            if gap_f >= float(args.gap_threshold):
                alerts.append(
                    {
                        "severity": "warning",
                        "code": "memorization_gap",
                        "message": f"Train/test PPL gap {gap_f:.2f}x >= {float(args.gap_threshold):.2f}x (memorization risk)",
                        "context": {"gap": gap_f, "threshold": float(args.gap_threshold)},
                    }
                )
        except (ValueError, TypeError):
            pass
    else:
        alerts.append(
            {
                "severity": "info",
                "code": "gap_unavailable",
                "message": "Could not compute train/test gap (missing train/test PPL).",
                "context": {},
            }
        )

    # Policy-driven recommendations (config + signals)
    try:
        recs = check_run(config, signals, gap_threshold=float(args.gap_threshold))
    except (ValueError, RuntimeError) as e:
        raise ConfigError(f"Policy evaluation failed: {e}") from e

    issues = [str(i) for i in getattr(reader, "issues", []) or []]

    # Extract Guard activity from telemetry
    guard_activity = _extract_guard_activity(reader)

    # Guard alerts (conservative triage advice)
    if guard_activity and guard_activity.get("present"):
        # Abort takes precedence over rollback as it's the more serious condition
        if guard_activity.get("aborted"):
            alerts.append(
                {
                    "severity": "error",
                    "code": "guard_abort",
                    "message": "⚠️ EXPERIMENTAL Guard stopped training due to repeated instability. Guard CANNOT fix root causes (data bugs, bad objectives). Investigate underlying issues before re-running. ALWAYS validate with eval.",
                    "context": {
                        "last_action": guard_activity.get("last_action"),
                        "snapshot_count": guard_activity.get("snapshot_count", 0),
                        "rollback_count": guard_activity.get("rollback_count", 0),
                        "note": "Guard is experimental and can stop training",
                    },
                }
            )
        elif guard_activity.get("rollback_occurred"):
            rollback_count = guard_activity.get("rollback_count", 1)
            alerts.append(
                {
                    "severity": "warning",
                    "code": "guard_intervention",
                    "message": f"⚠️ EXPERIMENTAL Guard rolled back adapter weights {rollback_count} time(s). This does NOT fix data bugs or bad objectives. Investigate triggers (grad explosion/NaN), check data pipeline, consider lowering LR. ALWAYS validate with eval.",
                    "context": {
                        "rollback_count": rollback_count,
                        "last_action": guard_activity.get("last_action"),
                        "snapshot_count": guard_activity.get("snapshot_count", 0),
                        "note": "Guard is experimental and rolls back weights",
                    },
                }
            )

    if args.json:
        payload = {
            "file": str(telemetry_path),
            "config": config.to_dict() if config is not None else None,
            "signals": signals.to_dict() if hasattr(signals, "to_dict") else {},
            "alerts": alerts,
            "recommendations": [r.to_dict() for r in recs],
            "telemetry_issues": issues,
        }
        print(json.dumps(payload, indent=2))
        return

    _print_monitor_result(
        telemetry_path=telemetry_path,
        config=config,
        signals=signals,
        alerts=alerts,
        recs=recs,
        issues=issues,
        verbose=args.verbose,
        guard_activity=guard_activity,
    )


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------



def _setup_report_command(subparsers):
    report_parser = subparsers.add_parser("report", help="[ADVANCED] Generate report from telemetry")
    report_parser.add_argument("file", help="Path to telemetry JSONL file")
    report_parser.set_defaults(func=cmd_report)


def _setup_monitor_command(subparsers):
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="[EXPERIMENTAL] Analyze telemetry JSONL and emit research-side alerts/recommendations",
    )
    monitor_parser.add_argument("file", help="Path to vNext telemetry JSONL file")
    monitor_parser.add_argument(
        "--gap-threshold",
        type=float,
        default=1.5,
        help="Train/test PPL ratio threshold above which we warn about memorization (default: 1.5)",
    )
    monitor_parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Fail fast on schema/envelope validation issues instead of skipping bad lines",
    )
    monitor_parser.add_argument("--verbose", action="store_true", help="Print rationale/evidence and telemetry issues")
    monitor_parser.add_argument("--json", action="store_true", help="Output JSON instead of pretty text")
    monitor_parser.set_defaults(func=cmd_monitor)



def setup_monitor_commands(subparsers) -> None:
    """Register monitor and report commands with the argument parser."""
    _setup_report_command(subparsers)
    _setup_monitor_command(subparsers)
